"""Tests de la résolution des modèles au démarrage (``scripts/run_backend.py``) et de la livraison de ``models/`` dans l'image Docker.

Preuves recherchées :

- avec aucun modèle, ou un modèle illisible, le moteur démarre sur les squelettes neutres : jamais de plantage, jamais un autre fichier
  pris en silence ;
- sans variable d'environnement, c'est l'artefact COMPLET le plus récent (d'après l'horodatage du NOM, pas la date du fichier) qui est choisi ;
- un chemin donné explicitement par l'environnement l'emporte, et s'il est faux on ne retombe PAS sur un autre modèle ;
- le Dockerfile ne nomme plus aucun fichier de modèle et ``COPY models/ ./models/`` ne peut pas échouer, même sans aucun modèle.

Docker n'est pas installé sur la machine qui a écrit ces tests : le contexte de construction est ÉMULÉ (règles de ``.dockerignore``), ce qui ne
remplace pas un ``docker build`` mais détecte le défaut visé (une source de ``COPY`` exclue ou absente).
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pytest
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

# These tests inspect the repository itself (Dockerfile, .dockerignore, git's index). The CI test image holds only the code and the tests, not
# those files, so there they would fail for a reason unrelated to the code under test: skip them when the files are absent.
_needs_repo_files = pytest.mark.skipif(
    not all((ROOT / name).exists() for name in ("Dockerfile", ".dockerignore", ".git")),
    reason="needs the repository checkout (Dockerfile, .dockerignore, .git), absent from the CI test image",
)


def _load(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = _load("run_backend_for_resolution", "scripts/run_backend.py")


def _write_pair(directory: Path, stamp: str, *, corrupt: bool = False, with_scaler: bool = True) -> tuple[Path, Path]:
    """Une vraie paire XGBoost + scaler (ou des octets illisibles si ``corrupt``)."""
    model_path = directory / f"supervisor_model_{stamp}.json"
    scaler_path = directory / f"supervisor_scaler_{stamp}.pkl"
    if corrupt:
        model_path.write_text("pas un modèle", encoding="utf-8")
        scaler_path.write_bytes(b"pas un scaler")
        return model_path, scaler_path
    x = np.random.default_rng(0).normal(size=(60, 5))
    y = (x[:, 0] > 0).astype(int)
    classifier = xgb.XGBClassifier(n_estimators=3, max_depth=2)
    classifier.fit(StandardScaler().fit_transform(x), y)
    classifier._estimator_type = "classifier"  # cale xgboost / scikit-learn déjà utilisée par scripts/train_model.py (save_model l'exige)
    classifier.save_model(str(model_path))
    if with_scaler:
        joblib.dump(StandardScaler().fit(x), scaler_path)
    return model_path, scaler_path


def _write_policy(directory: Path, stamp: str, *, corrupt: bool = False) -> Path:
    path = directory / f"ppo_policy_{stamp}.pt"
    if corrupt:
        path.write_bytes(b"pas une politique")
    else:
        save_policy: Callable[[str], None] = torch.jit.script(torch.nn.Identity()).save  # torch.jit n'est pas annoté : on type le point d'appel
        save_policy(str(path))
    return path


def _env(directory: Path, **extra: str) -> dict[str, str]:
    return {"NOTHIERLY_MODELS_DIR": str(directory), **extra}


# --- XGBoost -------------------------------------------------------------------------------


def test_with_no_model_at_all_the_backend_gets_the_neutral_skeleton_and_does_not_crash(tmp_path: Path) -> None:
    assert backend._resolve_supervisor_model_paths(_env(tmp_path)) == (None, None)
    assert backend._resolve_supervisor_model_paths(_env(tmp_path / "n_existe_pas")) == (None, None)  # dossier absent
    assert backend._resolve_rl_policy_model_path(_env(tmp_path)) is None
    assert backend._resolve_meta_label_model_path(_env(tmp_path)) is None


def test_the_newest_complete_pair_wins_by_the_timestamp_in_its_name_not_by_file_date(tmp_path: Path) -> None:
    _write_pair(tmp_path, "20261001_090000")
    newest = _write_pair(tmp_path, "20261014_120000")
    older = _write_pair(tmp_path, "20260915_080000")
    for path in (*newest, ):
        os.utime(path, (1, 1))  # le plus récent d'après son nom a la plus ancienne date de fichier (ce que fait une copie Docker)
    for path in older:
        os.utime(path, (4_000_000_000, 4_000_000_000))
    assert backend._resolve_supervisor_model_paths(_env(tmp_path)) == newest


def test_a_model_without_its_scaler_is_never_used_with_another_trainings_scaler(tmp_path: Path) -> None:
    complete = _write_pair(tmp_path, "20261001_090000")
    _write_pair(tmp_path, "20261014_120000", with_scaler=False)  # plus récent mais incomplet
    assert backend._resolve_supervisor_model_paths(_env(tmp_path)) == complete
    only_orphans = tmp_path / "orphans"
    only_orphans.mkdir()
    _write_pair(only_orphans, "20261014_120000", with_scaler=False)
    assert backend._resolve_supervisor_model_paths(_env(only_orphans)) == (None, None)


def test_a_corrupt_newest_artifact_gives_the_skeleton_rather_than_a_crash_or_a_silent_older_model(tmp_path: Path) -> None:
    _write_pair(tmp_path, "20261001_090000")
    _write_pair(tmp_path, "20261014_120000", corrupt=True)
    assert backend._resolve_supervisor_model_paths(_env(tmp_path)) == (None, None)


def test_files_that_do_not_match_the_naming_scheme_are_ignored(tmp_path: Path) -> None:
    for name in ("supervisor_model_latest.json", "supervisor_model_20261014.json", "supervisor_model_20261014_120000.json.bak", "notes.txt", "supervisor_scaler_20261014_120000.pkl"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert backend._resolve_supervisor_model_paths(_env(tmp_path)) == (None, None)


def test_an_explicit_environment_pair_wins_and_a_wrong_one_does_not_fall_back_to_another_model(tmp_path: Path) -> None:
    _write_pair(tmp_path, "20261014_120000")
    chosen_dir = tmp_path / "chosen"
    chosen_dir.mkdir()
    model, scaler = _write_pair(chosen_dir, "20250101_000000")  # volontairement le plus ancien
    explicit = _env(tmp_path, NOTHIERLY_SUPERVISOR_MODEL=str(model), NOTHIERLY_SUPERVISOR_SCALER=str(scaler))
    assert backend._resolve_supervisor_model_paths(explicit) == (model, scaler)
    missing = _env(tmp_path, NOTHIERLY_SUPERVISOR_MODEL=str(model), NOTHIERLY_SUPERVISOR_SCALER=str(tmp_path / "absent.pkl"))
    assert backend._resolve_supervisor_model_paths(missing) == (None, None)  # un modèle plus récent existe, mais l'opérateur en a nommé un autre
    half = _env(tmp_path, NOTHIERLY_SUPERVISOR_MODEL=str(model))
    assert backend._resolve_supervisor_model_paths(half) == (None, None)  # le modèle seul, sans son scaler, n'est pas accepté


# --- PPO ----------------------------------------------------------------------------------


def test_the_newest_loadable_ppo_policy_is_used_and_a_corrupt_one_gives_the_skeleton(tmp_path: Path) -> None:
    _write_policy(tmp_path, "20261001_090000")
    newest = _write_policy(tmp_path, "20261014_120000")
    assert backend._resolve_rl_policy_model_path(_env(tmp_path)) == newest
    (tmp_path / "ppo_policy_latest.pt").write_bytes(b"x")  # nom hors schéma : ignoré
    assert backend._resolve_rl_policy_model_path(_env(tmp_path)) == newest
    _write_policy(tmp_path, "20261015_000000", corrupt=True)
    assert backend._resolve_rl_policy_model_path(_env(tmp_path)) is None


def test_an_explicit_ppo_path_wins_and_a_wrong_one_is_not_replaced(tmp_path: Path) -> None:
    _write_policy(tmp_path, "20261014_120000")
    other = tmp_path / "elsewhere"
    other.mkdir()
    chosen = _write_policy(other, "20250101_000000")
    assert backend._resolve_rl_policy_model_path(_env(tmp_path, NOTHIERLY_RL_POLICY=str(chosen))) == chosen
    assert backend._resolve_rl_policy_model_path(_env(tmp_path, NOTHIERLY_RL_POLICY=str(tmp_path / "absent.pt"))) is None


# --- Méta-modèle ---------------------------------------------------------------------------


def test_the_meta_label_weights_come_from_the_models_directory_or_an_explicit_path(tmp_path: Path) -> None:
    assert backend._resolve_meta_label_model_path(_env(tmp_path)) is None
    (tmp_path / "metalabel_weights.json").write_text("{}", encoding="utf-8")
    assert backend._resolve_meta_label_model_path(_env(tmp_path)) == tmp_path / "metalabel_weights.json"
    explicit = tmp_path / "autre.json"
    explicit.write_text("{}", encoding="utf-8")
    assert backend._resolve_meta_label_model_path(_env(tmp_path, NOTHIERLY_META_LABEL_MODEL=str(explicit))) == explicit
    assert backend._resolve_meta_label_model_path(_env(tmp_path, NOTHIERLY_META_LABEL_MODEL=str(tmp_path / "absent.json"))) is None


def test_the_health_check_reports_what_the_backend_will_actually_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    health = _load("system_health_check_for_resolution", "scripts/system_health_check.py")
    monkeypatch.setenv("NOTHIERLY_MODELS_DIR", str(tmp_path))
    by_name = {c.name: c for c in health.check_brain_models()}
    assert by_name["XGBoost primaire (artefact)"].status is health.Status.WARN and by_name["PPO (politique)"].status is health.Status.WARN
    _write_pair(tmp_path, "20261014_120000")
    _write_policy(tmp_path, "20261014_120000")
    by_name = {c.name: c for c in health.check_brain_models()}
    assert by_name["XGBoost primaire (artefact)"].status is health.Status.PASS and "20261014_120000" in by_name["XGBoost primaire (artefact)"].detail
    assert by_name["PPO (politique)"].status is health.Status.PASS


# --- Dockerfile et contexte de construction --------------------------------------------------


def _dockerignore_patterns() -> list[str]:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile(f"^{out}$")


def is_ignored(path: str, patterns: list[str]) -> bool:
    """Émulation de ``.dockerignore`` : le dernier motif qui correspond décide (``!`` ré-inclut) ; un motif qui désigne un dossier couvre son contenu."""
    ignored = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        body = pattern[1:] if negated else pattern
        regex = _pattern_regex(body.strip("/"))
        parts = path.split("/")
        if any(regex.match("/".join(parts[: k + 1])) for k in range(len(parts))):
            ignored = not negated
    return ignored


def test_the_emulated_dockerignore_matches_the_documented_behaviour() -> None:
    patterns = ["models/*", "!models/.gitkeep", "!models/supervisor_model_*.json", ".env", "logs/"]
    assert not is_ignored("models", patterns)  # le dossier lui-même reste dans le contexte
    assert not is_ignored("models/.gitkeep", patterns) and not is_ignored("models/supervisor_model_20261014_120000.json", patterns)
    assert is_ignored("models/random.bin", patterns) and is_ignored(".env", patterns) and is_ignored("logs/a.log", patterns)
    assert not is_ignored("src/app.py", patterns)


def _dockerfile_copy_sources() -> list[tuple[str, str]]:
    sources = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "COPY" and not any(p.startswith("--from") for p in parts):
            sources += [(src, line) for src in parts[1:-1]]
    return sources


@_needs_repo_files
def test_the_dockerfile_names_no_model_file_and_copies_the_directory() -> None:
    sources = _dockerfile_copy_sources()
    model_sources = [src for src, _ in sources if src.startswith("models")]
    assert model_sources == ["models/"]  # une seule instruction, générique : aucun nom de fichier de modèle
    assert not re.search(r"supervisor_(model|scaler)_\d", (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("supervisor_model_<ts>", "").replace("supervisor_scaler_<ts>", ""))


@_needs_repo_files
@pytest.mark.parametrize("artifacts", [[], ["supervisor_model_20261014_120000.json", "supervisor_scaler_20261014_120000.pkl", "ppo_policy_20261014_120000.pt", "metalabel_weights.json", "metalabel_weights.meta.json"]])
def test_the_models_copy_source_exists_in_the_build_context_with_or_without_any_model(artifacts: list[str]) -> None:
    """Contexte d'une copie fraîche du dépôt (seul ``models/.gitkeep`` suivi par git) puis avec des artefacts : la source du COPY existe, et seuls les
    artefacts reconnus par les chargeurs y entrent."""
    patterns = _dockerignore_patterns()
    context = ["models/.gitkeep", *[f"models/{name}" for name in artifacts], "models/brouillon.bin", "models/notes.txt"]
    kept = [path for path in context if not is_ignored(path, patterns)]
    assert not is_ignored("models", patterns)  # `COPY models/ ./models/` : la source est présente même si tous les fichiers sont exclus
    assert "models/.gitkeep" in kept
    assert sorted(path for path in kept if path != "models/.gitkeep") == sorted(f"models/{name}" for name in artifacts)
    assert "models/brouillon.bin" not in kept and "models/notes.txt" not in kept


@_needs_repo_files
def test_the_gitkeep_that_guarantees_the_directory_is_tracked_by_git() -> None:
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "models/.gitkeep"], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.split()
    assert tracked == ["models/.gitkeep"]  # sans lui, une copie fraîche n'aurait pas de dossier models/ et le COPY échouerait


def test_no_code_path_still_refers_to_the_old_hardcoded_model_names() -> None:
    source = (ROOT / "scripts" / "run_backend.py").read_text(encoding="utf-8")
    assert "supervisor_model_20260910" not in source and "ppo_policy_20260902" not in source
    assert not hasattr(backend, "_SUPERVISOR_MODEL_PATH") and not hasattr(backend, "_RL_POLICY_MODEL_PATH")



def test_the_real_writer_the_resolver_and_the_real_consumer_agree(tmp_path: Path) -> None:
    """Chaîne complète : ``scripts/train_model.save_model`` écrit, le résolveur du moteur choisit, ``SupervisorMLModel`` charge et prédit."""
    from src.agents.supervisor.ml_engine import SupervisorMLModel

    train_model = _load("train_model_for_resolution", "scripts/train_model.py")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 5))
    y = (x[:, 0] > 0).astype(np.int64)
    model, scaler, _ = train_model.train_xgboost_classifier(x, y)
    model_path, scaler_path = train_model.save_model(model, scaler, tmp_path)
    resolved = backend._resolve_supervisor_model_paths(_env(tmp_path))
    assert resolved == (model_path, scaler_path)
    assert resolved[0] is not None and resolved[1] is not None
    loaded = SupervisorMLModel(model_path=str(resolved[0]), scaler_path=str(resolved[1]))
    strong, weak = np.array([2.0, 0.0, 0.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0, 0.0, 0.0])
    assert loaded.predict_probability(strong) > 0.8 > 0.2 > loaded.predict_probability(weak)  # le modèle chargé est bien celui qui a été entraîné

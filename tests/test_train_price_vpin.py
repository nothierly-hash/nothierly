"""Entraînement hors ligne « prix + flux + VPIN » : découpage sans fuite, normalisation de l'entraînement seulement, plage continue, chaîne complète et métadonnées."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import train_price_vpin as tpv
from src.rl import historical_features as hf
from workers.market_data.vpin_archive import BucketRecord, VpinArchive

START_S = 1_760_000_100.0


def _bars(n: int, *, start_s: float = START_S, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 5e-4, n)))
    buy = rng.uniform(1e4, 5e4, n)
    sell = rng.uniform(1e4, 5e4, n)
    return {"bar_end_s": start_s + 60.0 * np.arange(n), "close": close, "buy_notional": buy, "sell_notional": sell}


def _vpin_db(path: Path, symbol: str, start_s: float, end_s: float, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    archive = VpinArchive(path)
    for t in np.arange(start_s - 3_600.0, end_s + 60.0, 20.0):
        archive.submit(BucketRecord(symbol, int(t * 1000), Decimal(str(round(float(rng.uniform(0.15, 0.85)), 4))), Decimal(60), Decimal(40), Decimal(100_000), 50))
    assert archive.close() == 0
    return path


def test_the_chronological_split_is_ordered_contiguous_and_separated_by_the_embargo() -> None:
    train, val, test = tpv.chronological_split(1_000, (0.6, 0.2, 0.2), 50)
    assert train.start == 0 and train.stop < val.start < val.stop < test.start < test.stop == 1_000
    assert val.start - train.stop == 50 and test.start - val.stop == 50  # marge entre chaque tranche : aucune fenêtre glissante ne la traverse
    with pytest.raises(ValueError):
        tpv.chronological_split(1_000, (0.5, 0.2, 0.2), 50)
    with pytest.raises(ValueError):
        tpv.chronological_split(20, (0.6, 0.2, 0.2), 50)


def test_normalization_uses_the_training_slice_only() -> None:
    n = 600
    rng = np.random.default_rng(2)
    data = tpv.Dataset(np.arange(n, dtype=float), np.full(n, 100.0), rng.normal(size=(n, 5)), rng.uniform(size=(n, 3)))
    train, _, test = tpv.chronological_split(n, (0.6, 0.2, 0.2), 10)
    mean, std = tpv.normalization(data, train)
    changed = tpv.Dataset(data.times_s, data.prices, data.features.copy(), data.vpin.copy())
    changed.features[test] += 1_000.0  # une valeur extrême en période de test ne doit rien changer à la normalisation
    mean2, std2 = tpv.normalization(changed, train)
    assert np.array_equal(mean, mean2) and np.array_equal(std, std2)
    f, v, _ = tpv.apply(data, train, mean, std)
    assert np.allclose(np.concatenate((f, v), axis=1).mean(axis=0), 0.0, atol=1e-9)


def test_the_dataset_keeps_the_longest_continuous_run_and_refuses_when_none_is_long_enough() -> None:
    long_part = _bars(600)
    short_part = _bars(250, start_s=START_S + 60.0 * 5_000, seed=3)  # après un grand trou
    merged = {k: np.concatenate((long_part[k], short_part[k])) for k in long_part}
    end = float(merged["bar_end_s"][-1])
    # VPIN continu sur toute la période
    times = np.arange(START_S - 3_600.0, end + 60.0, 20.0)
    values = np.random.default_rng(1).uniform(0.15, 0.85, len(times))
    data = tpv.prepare_dataset(merged, times, values)
    assert len(data.prices) > 400 and data.features.shape == (len(data.prices), 5) and data.vpin.shape == (len(data.prices), 3)
    assert np.isfinite(data.features).all() and np.isfinite(data.vpin).all()
    assert data.times_s[-1] <= long_part["bar_end_s"][-1]  # la plage retenue est la longue, jamais à cheval sur le trou
    with pytest.raises(ValueError, match="aucune plage"):
        tpv.prepare_dataset(_bars(80), times, values)


def test_the_full_chain_trains_evaluates_and_exports_a_policy_with_its_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bars = _bars(1_400)
    bars_path = tmp_path / "bars.npz"
    arrays: dict[str, Any] = {"symbol": np.array("BTCUSDT"), "bar_s": np.array(60.0), **bars}
    np.savez_compressed(bars_path, **arrays)
    db = _vpin_db(tmp_path / "v.sqlite3", "BTCUSDT", START_S, float(bars["bar_end_s"][-1]))
    models = tmp_path / "models"
    models.mkdir()
    report = tmp_path / "report.json"
    args = tpv._parse_args([
        "--bars", str(bars_path), "--vpin-db", str(db), "--total-timesteps", "256", "--rollout-length", "128", "--export", "--models-dir", str(models),
        "--report-json", str(report), "--kill-switch-vpin", "0.9",
    ])
    assert tpv.run(args) == 0
    out = capsys.readouterr().out
    for section in ("== entraînement", "== validation", "== test", "Verdict sur le TEST", "acheter et garder", "à plat"):
        assert section in out
    (policy,) = list(models.glob("ppo_policy_*.pt"))
    meta = json.loads(policy.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["feature_set"] == hf.FEATURE_SET_NAME and meta["observation_dim"] == 9 and meta["uses_vpin"] is True
    assert len(meta["normalization"]["mean"]) == 8 and meta["normalization"]["columns"][:5] == list(hf.FEATURE_NAMES)
    assert meta["splits_rows"]["train"][1] < meta["splits_rows"]["val"][0] < meta["splits_rows"]["test"][0]  # tranches ordonnées dans le temps
    results = json.loads(report.read_text(encoding="utf-8"))
    assert set(results) == {"entraînement", "validation", "test"} and "politique" in results["test"]


def test_invalid_options_are_refused() -> None:
    with pytest.raises(SystemExit):
        tpv._parse_args(["--bars", "a.npz", "--vpin-db", "v.sqlite3", "--cost", "-1"])

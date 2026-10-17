"""Tests de ``scripts/train_model.py`` après l'audit d'anticipation : découpage chronologique avec embargo, un seul symbole, test de
Hanley-McNeil sur un échantillon indépendant, refus d'écrire des poids qui ne battent pas le hasard.

Jeux synthétiques : « signal » (la prochaine tendance de 15 minutes dépend du déséquilibre du carnet) et « bruit » (aucun lien). Sur le bruit,
le script doit refuser et ne rien écrire ; sur le signal, il doit sauvegarder. Aucun accès base de données : lignes d'archive simulées.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_model.py"
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
HORIZON = pd.Timedelta(minutes=15)
DAY = pd.Timedelta(days=1)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_model_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_model_under_test"] = module
    spec.loader.exec_module(module)
    return module


tm = _load()


def _market(days: int, *, signal: bool, symbol: str = "BTC/USDT", seed: int = 1, price0: float = 50_000.0) -> list[dict[str, Any]]:
    """Une ligne par minute. ``signal`` : le déséquilibre est constant par bloc de 15 minutes et la dérive du prix le suit ; sinon, aucun lien."""
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    block = np.arange(n) // 15
    side = rng.choice([-1.0, 1.0], size=block.max() + 1)
    price = price0
    rows = []
    for i in range(n):
        if signal:
            imbalance = float(side[block[i]] * rng.uniform(0.5, 1.0))
            price *= 1.0 + side[block[i]] * 0.0002 + rng.normal(0, 0.0001)
        else:
            imbalance = float(rng.uniform(-1, 1))
            price *= 1.0 + rng.normal(0, 0.0003)
        rows.append({"time": T0 + timedelta(minutes=i), "data": {"symbol": symbol, "spread": 0.5, "order_book_imbalance": imbalance, "mid_price": price}})
    return rows


# --- Un seul symbole -------------------------------------------------------------------------


def test_an_archive_that_mixes_symbols_is_refused_and_one_symbol_can_be_selected() -> None:
    mixed = _market(1, signal=True, symbol="BTC/USDT", price0=50_000) + _market(1, signal=True, symbol="ETH/USDT", price0=3_000, seed=2)
    with pytest.raises(ValueError, match="mixes 2 symbols"):
        tm.build_market_feature_frame(mixed)  # le label « prix futur » comparerait un BTC à un ETH
    btc = tm.build_market_feature_frame(mixed, "BTC/USDT")
    assert len(btc) == 24 * 60 and btc["mid_price"].min() > 10_000
    with pytest.raises(ValueError, match="not found"):
        tm.build_market_feature_frame(mixed, "SOL/USDT")
    assert len(tm.build_market_feature_frame(_market(1, signal=True))) == 24 * 60  # un seul symbole : pas besoin de le nommer


def test_a_symbol_cannot_be_selected_in_an_archive_without_a_symbol_field() -> None:
    rows = [{"time": T0 + timedelta(minutes=i), "data": {"spread": 0.5, "order_book_imbalance": 0.1, "mid_price": 100.0}} for i in range(5)]
    assert len(tm.build_market_feature_frame(rows)) == 5
    with pytest.raises(ValueError, match="no 'symbol' field"):
        tm.build_market_feature_frame(rows, "BTC/USDT")


# --- Jeu d'entraînement ----------------------------------------------------------------------


def test_the_training_set_is_time_sorted_aligned_and_drops_rows_without_a_future_price() -> None:
    training = tm.build_training_set(_market(1, signal=True), [], [], horizon=HORIZON)
    assert training.X.shape == (24 * 60 - 15, 5) and training.y.shape == training.times_ns.shape == (24 * 60 - 15,)
    assert bool(np.all(np.diff(training.times_ns) > 0)) and set(training.y.tolist()) <= {0, 1}
    assert training.times_ns[0] == int(pd.Timestamp(T0).tz_convert("UTC").tz_localize(None).value)
    X, y = tm.build_training_dataset(_market(1, signal=True), [], [], horizon=HORIZON)  # l'ancienne API reste disponible
    assert bool(np.array_equal(X, training.X)) and bool(np.array_equal(y, training.y))


def test_macro_features_never_use_a_reading_published_after_the_row() -> None:
    macro = [{"time": T0 + timedelta(minutes=100), "data": {"indicator": "US_10Y_YIELD", "value": 4.2}}]
    training = tm.build_training_set(_market(1, signal=True), macro, [], horizon=HORIZON)
    column = training.X[:, 2]  # macro_us_10y_yield
    assert bool(np.all(column[:100] == 0.0)) and bool(np.all(column[100:] == 4.2))  # as-of arrière : rien avant la publication


# --- Découpage chronologique -----------------------------------------------------------------


def test_the_split_is_chronological_so_a_change_of_regime_at_the_end_is_seen_as_a_failure_not_hidden_by_shuffling() -> None:
    """Concept inversé sur les 20 derniers % : un découpage aléatoire mélangerait les deux régimes et afficherait une bonne AUC ; le
    découpage chronologique teste sur le seul régime que le modèle n'a pas vu."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    y = np.where(np.arange(1000) < 800, X[:, 0] > 0, X[:, 0] <= 0).astype(np.int64)
    _, _, metrics = tm.train_xgboost_classifier(X, y, test_size=0.2)
    assert (metrics["train_rows"], metrics["embargoed_rows"], metrics["test_rows"]) == (800.0, 0.0, 200.0)
    assert metrics["roc_auc"] < 0.3  # le modèle a appris l'inverse de ce qu'il voit au test
    _, _, with_embargo = tm.train_xgboost_classifier(X, y, test_size=0.2, embargo_rows=20)
    assert (with_embargo["train_rows"], with_embargo["embargoed_rows"], with_embargo["test_rows"]) == (800.0, 20.0, 180.0)
    assert not hasattr(tm, "train_test_split")  # l'ancien découpage aléatoire n'existe plus dans le module


def test_a_duration_embargo_skips_the_rows_inside_it_and_the_model_never_sees_the_test_rows() -> None:
    n = 1000
    times = np.arange(n, dtype=np.int64) * 60 * 10**9  # une ligne par minute
    X = np.random.default_rng(1).normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(np.int64)
    _, _, metrics = tm.train_xgboost_classifier(X, y, test_size=0.2, times_ns=times, embargo_ns=30 * 60 * 10**9)
    # dernière ligne d'entraînement à t = 799 min ; le test commence à la première ligne >= 799 + 30 = 829 min : 29 lignes retirées, 30 min d'écart exactes
    assert (metrics["train_rows"], metrics["embargoed_rows"], metrics["test_rows"]) == (800.0, 29.0, 171.0)


def test_degenerate_inputs_are_refused() -> None:
    X = np.random.default_rng(2).normal(size=(100, 5))
    with pytest.raises(ValueError, match="at least 10"):
        tm.train_xgboost_classifier(X[:5], np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError, match="single outcome"):
        tm.train_xgboost_classifier(X, np.zeros(100, dtype=np.int64))
    with pytest.raises(ValueError):
        tm.train_xgboost_classifier(X, (X[:, 0] > 0).astype(np.int64), embargo_rows=90)  # l'embargo avale tout le test
    with pytest.raises(ValueError, match="triés"):
        tm.train_xgboost_classifier(X, (X[:, 0] > 0).astype(np.int64), times_ns=np.arange(100, 0, -1, dtype=np.int64), embargo_ns=0)


# --- Validation statistique ------------------------------------------------------------------


def _training(days: int, *, signal: bool, seed: int = 1) -> Any:
    return tm.build_training_set(_market(days, signal=signal, seed=seed), [], [], horizon=HORIZON)


def test_a_model_with_real_signal_is_accepted_on_an_independent_test_sample() -> None:
    outcome = tm.fit_and_validate(_training(14, signal=True), horizon=HORIZON, embargo=DAY)
    assert outcome.accepted and outcome.refusals == []
    significance = outcome.significance
    assert significance.significant and significance.n_independent >= 50 and significance.n_independent < significance.n_total / 5  # espacées d'un horizon
    assert significance.auc_independent is not None and significance.auc_independent > 0.8
    assert outcome.metrics["embargoed_rows"] == 24 * 60 - 1  # 24 h d'écart entre la dernière ligne d'entraînement et la première de test, à une ligne par minute


@pytest.mark.parametrize("seed", range(6))
def test_a_model_without_signal_is_refused_whatever_the_seed(seed: int) -> None:
    outcome = tm.fit_and_validate(_training(14, signal=False, seed=seed), horizon=HORIZON, embargo=DAY)
    assert not outcome.accepted
    assert any("not significantly above chance" in reason for reason in outcome.refusals)


def test_too_few_independent_observations_refuse_even_a_good_looking_model() -> None:
    outcome = tm.fit_and_validate(_training(14, signal=True), horizon=HORIZON, embargo=DAY, min_independent_test=10_000)
    assert not outcome.accepted and any("independent test observations" in reason for reason in outcome.refusals)


def test_an_embargo_shorter_than_the_label_horizon_or_the_sentiment_window_is_refused() -> None:
    training = _training(3, signal=True)
    with pytest.raises(ValueError, match="Embargo"):
        tm.fit_and_validate(training, horizon=pd.Timedelta(hours=2), embargo=pd.Timedelta(minutes=30))  # l'étiquette d'une ligne d'entraînement regarde dans le test
    with pytest.raises(ValueError, match="Embargo"):
        tm.fit_and_validate(training, horizon=HORIZON, embargo=pd.Timedelta(minutes=30))  # sous la fenêtre de sentiment d'une heure


# --- Pipeline de bout en bout (base simulée) -------------------------------------------------


def _patch_database(monkeypatch: pytest.MonkeyPatch, market: list[dict[str, Any]]) -> None:
    class _Pool:
        async def close(self) -> None:
            return None

    async def create_pool(**_: Any) -> _Pool:
        return _Pool()

    async def fetch_topic_rows(pool: Any, topic: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return market if topic == tm._MARKET_TOPIC else []

    monkeypatch.setattr(tm, "get_settings", lambda: SimpleNamespace(timescale=SimpleNamespace(host="h", port=1, database="d", user="u", password=None)))
    monkeypatch.setattr(tm.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(tm, "fetch_topic_rows", fetch_topic_rows)


def test_the_pipeline_saves_weights_only_when_the_model_beats_chance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_database(monkeypatch, _market(14, signal=True))
    saved = asyncio.run(tm.run_training_pipeline(lookback_days=14, horizon_minutes=15, up_threshold=0.0, models_dir=tmp_path / "ok"))
    assert saved is True and len(list((tmp_path / "ok").glob("supervisor_model_*.json"))) == 1 and len(list((tmp_path / "ok").glob("supervisor_scaler_*.pkl"))) == 1
    out = capsys.readouterr().out
    assert "Chronological split" in out and "one-sided 99% lower bound" in out
    _patch_database(monkeypatch, _market(14, signal=False, seed=4))
    refused = asyncio.run(tm.run_training_pipeline(lookback_days=14, horizon_minutes=15, up_threshold=0.0, models_dir=tmp_path / "refused"))
    assert refused is False and not (tmp_path / "refused").exists()  # aucun fichier, pas même le dossier
    assert "REFUSED" in capsys.readouterr().out


def test_the_pipeline_refuses_a_mixed_archive_without_a_symbol_and_trains_with_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mixed = _market(14, signal=True, symbol="BTC/USDT") + _market(14, signal=True, symbol="ETH/USDT", price0=3_000, seed=2)
    _patch_database(monkeypatch, mixed)
    with pytest.raises(ValueError, match="mixes 2 symbols"):
        asyncio.run(tm.run_training_pipeline(lookback_days=14, horizon_minutes=15, up_threshold=0.0, models_dir=tmp_path))
    assert asyncio.run(tm.run_training_pipeline(lookback_days=14, horizon_minutes=15, up_threshold=0.0, symbol="BTC/USDT", models_dir=tmp_path)) is True


def test_an_archive_too_short_for_the_embargo_fails_loudly_instead_of_testing_on_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_database(monkeypatch, _market(2, signal=True))  # 2 jours : l'embargo de 24 h et les 20 % ne laissent aucune ligne de test
    with pytest.raises(ValueError, match="Aucune ligne de test"):
        asyncio.run(tm.run_training_pipeline(lookback_days=2, horizon_minutes=15, up_threshold=0.0, models_dir=tmp_path))
    assert not any(tmp_path.iterdir())


# --- Ligne de commande -----------------------------------------------------------------------


def test_the_command_line_returns_a_nonzero_exit_code_when_the_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([True, False])

    async def fake_pipeline(**_: Any) -> bool:
        return next(results)

    monkeypatch.setattr(tm, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(tm, "configure_logging", lambda **_: None)
    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    assert tm.main() == 0
    assert tm.main() == 1  # modèle refusé : la planification (cron, CI) voit l'échec


@pytest.mark.parametrize("argv", [["--embargo-minutes", "10"], ["--horizon-minutes", "240", "--embargo-minutes", "120"], ["--test-fraction", "0.9"], ["--test-fraction", "0.01"]])
def test_the_command_line_refuses_an_embargo_shorter_than_the_horizon_or_the_sentiment_window_and_absurd_fractions(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["train_model.py", *argv])
    with pytest.raises(SystemExit):
        tm._parse_args()


def test_the_default_embargo_is_a_day_and_covers_the_default_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_model.py"])
    args = tm._parse_args()
    assert args.embargo_minutes == 24 * 60 >= args.horizon_minutes and args.test_fraction == 0.2 and args.symbol is None

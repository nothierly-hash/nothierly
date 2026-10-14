"""Tests du filtrage par symbole dans la chaîne d'entraînement : ``train_model``, ``train_ppo`` (archive multi-symboles) et ``audit_matrix`` (série FRED).

Preuves recherchées : une archive qui mélange BTC et ETH ne peut plus alimenter un entraînement sans que le symbole soit choisi ; le
symbole se compare sans casse ni séparateur (``BTCUSDT`` = ``BTC/USDT``) ; les lignes des autres symboles disparaissent AVANT l'alignement et
l'étiquetage ; ``audit_matrix`` garde son appel sans argument (la page du dashboard en dépend).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import asyncpg
import numpy as np
import pandas as pd
import pytest

from scripts import audit_matrix, train_model, train_ppo

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _rows(symbol: str, price0: float, n: int = 120) -> list[dict[str, Any]]:
    return [{"time": T0 + timedelta(minutes=i), "data": {"symbol": symbol, "spread": 0.5, "order_book_imbalance": 0.1, "mid_price": price0 + i}} for i in range(n)]


MIXED = _rows("BTC/USDT", 50_000.0) + _rows("ETH/USDT", 3_000.0)


# --- Normalisation ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested", ["BTC/USDT", "BTCUSDT", "btcusdt", "btc-usdt", " BTC/USDT"[1:]])
def test_a_symbol_matches_whatever_its_case_or_separators(requested: str) -> None:
    frame = train_model.build_market_feature_frame(MIXED, requested)
    assert len(frame) == 120 and float(frame["mid_price"].min()) >= 50_000.0  # uniquement les lignes BTC : aucune ligne à ~3 000


def test_normalisation_does_not_confuse_different_instruments() -> None:
    assert train_model.normalize_symbol("BTC/USDT") == train_model.normalize_symbol("BTCUSDT") == "BTCUSDT"
    assert train_model.normalize_symbol("BTC/USDT") != train_model.normalize_symbol("ETH/USDT")
    assert train_model.normalize_symbol("BTC/USD") != train_model.normalize_symbol("BTC/USDT")
    with pytest.raises(ValueError, match="not found"):
        train_model.build_market_feature_frame(MIXED, "BTC/USDC")


def test_the_filter_runs_before_labelling_so_a_future_price_never_comes_from_another_symbol() -> None:
    """Sans filtre, le « premier prix à t + horizon » d'une ligne BTC pouvait être celui d'une ligne ETH."""
    training = train_model.build_training_set(MIXED, [], [], horizon=pd.Timedelta(minutes=15), symbol="BTCUSDT")
    assert training.X.shape[0] == 120 - 15 and set(training.y.tolist()) == {1}  # la série BTC monte de 1 par minute : toutes les étiquettes sont « hausse »
    eth = train_model.build_training_set(MIXED, [], [], horizon=pd.Timedelta(minutes=15), symbol="ETH/USDT")
    assert eth.X.shape[0] == 120 - 15 and set(eth.y.tolist()) == {1}


# --- train_ppo -------------------------------------------------------------------------------


def test_train_ppo_requires_a_symbol_on_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--lookback-days", "5"])
    with pytest.raises(SystemExit):
        train_ppo._parse_args()
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--symbol", "BTCUSDT"])
    args = train_ppo._parse_args()
    assert args.symbol == "BTCUSDT" and args.lookback_days == 30


def _patch_archive(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]) -> None:
    class _Pool:
        async def close(self) -> None:
            return None

    async def create_pool(**_: Any) -> _Pool:
        return _Pool()

    async def fetch_topic_rows(pool: Any, topic: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return rows if topic == train_model._MARKET_TOPIC else []

    monkeypatch.setattr(train_ppo, "get_settings", lambda: SimpleNamespace(timescale=SimpleNamespace(host="h", port=1, database="d", user="u", password=None)))
    monkeypatch.setattr(asyncpg, "create_pool", create_pool)  # la même bibliothèque que celle importée par train_ppo
    monkeypatch.setattr(train_ppo, "fetch_topic_rows", fetch_topic_rows)


def test_train_ppo_only_sees_the_rows_of_the_chosen_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_archive(monkeypatch, MIXED)
    features, prices = asyncio.run(train_ppo._fetch_aligned_dataset(5, "BTCUSDT"))
    assert features.shape == (120, 5) and prices.shape == (120,) and float(prices.min()) >= 50_000.0
    eth_features, eth_prices = asyncio.run(train_ppo._fetch_aligned_dataset(5, "ETH/USDT"))
    assert float(eth_prices.max()) < 4_000.0 and len(eth_prices) == 120


def test_train_ppo_refuses_an_unknown_symbol_and_an_unfiltered_mixed_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_archive(monkeypatch, MIXED)
    with pytest.raises(ValueError, match="not found"):
        asyncio.run(train_ppo._fetch_aligned_dataset(5, "SOL/USDT"))
    with pytest.raises(ValueError, match="mixes 2 symbols"):
        train_model.build_market_feature_frame(MIXED)  # le chemin sans symbole reste interdit sur une archive mélangée


def test_the_ppo_pipeline_passes_the_symbol_down_to_the_dataset_and_exports_only_that_series(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_archive(monkeypatch, MIXED)
    seen: dict[str, Any] = {}

    def fake_training(features: Any, prices: Any, config: Any) -> str:
        seen["prices"] = prices
        return "modèle"

    monkeypatch.setattr(train_ppo, "run_ppo_training", fake_training)
    monkeypatch.setattr(train_ppo, "export_model", lambda model, models_dir: tmp_path / "ppo.pt")
    path = asyncio.run(train_ppo.run_training_pipeline(symbol="BTCUSDT", lookback_days=5, total_timesteps=10, rollout_length=5, models_dir=tmp_path))
    assert path == tmp_path / "ppo.pt" and float(seen["prices"].min()) >= 50_000.0 and len(seen["prices"]) == 120


def test_the_ppo_pipeline_has_no_default_symbol() -> None:
    assert inspect.signature(train_ppo.run_training_pipeline).parameters["symbol"].default is inspect.Parameter.empty
    assert inspect.signature(train_ppo._fetch_aligned_dataset).parameters["symbol"].default is inspect.Parameter.empty


# --- audit_matrix ----------------------------------------------------------------------------


def _fake_fred(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    series = {"DEXUSEU": pd.Series(1.10 + 0.0004 * np.arange(300), index=index), "DGS10": pd.Series(2.0 + 0.001 * np.arange(300), index=index)}
    monkeypatch.setattr(audit_matrix, "_fetch_fred_series", lambda series_id: series[series_id])


def test_the_audit_dataset_builds_from_its_single_symbol_series_and_keeps_its_zero_argument_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_fred(monkeypatch)
    assert not inspect.signature(audit_matrix.build_real_audit_dataset).parameters  # intelligence.py l'appelle sans argument
    features, prices, aligned = audit_matrix.build_real_audit_dataset()
    assert features.shape == (299, 5) and prices.shape == (299,) and aligned["label"].notna().all()
    assert float(prices.min()) > 1.0 and float(prices.max()) < 1.3  # EUR/USD, jamais une autre série


def test_the_audit_selects_its_symbol_explicitly_before_any_transformation(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_fred(monkeypatch)
    calls: list[tuple[list[Any], str | None]] = []
    original = train_model.build_market_feature_frame

    def spy(records: list[Any], symbol: str | None = None) -> pd.DataFrame:
        calls.append((records, symbol))
        return original(records, symbol)

    monkeypatch.setattr(audit_matrix, "build_market_feature_frame", spy)
    audit_matrix.build_real_audit_dataset()
    ((records, symbol),) = calls
    assert symbol == audit_matrix.AUDIT_SYMBOL == "EUR/USD" and {r["data"]["symbol"] for r in records} == {"EUR/USD"}


def test_an_intruder_symbol_in_the_audit_series_is_refused_not_mixed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    records = _rows(audit_matrix.AUDIT_SYMBOL, 1.1) + _rows("BTC/USDT", 50_000.0)
    with pytest.raises(ValueError, match="mixes 2 symbols"):
        train_model.build_market_feature_frame(records)
    assert len(train_model.build_market_feature_frame(records, audit_matrix.AUDIT_SYMBOL)) == 120

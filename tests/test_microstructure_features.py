"""Real, executable tests for src/agents/market_data/microstructure_features.py.

No I/O: :func:`compute_live_microstructure_reading` is pure, tested
directly against hand-computed real VPIN/TFI values (same "pure
functions, real regression tests" discipline as ``tests/
test_eod_reconciliator.py``) -- this reuses the real, already-tested
``research.alpha_features.compute_vpin``/``compute_tfi`` unchanged, so
these tests verify this module's own glue (unit conversion, bucket/
window sizing, the insufficient-data path), not re-derive VPIN/TFI math
already covered elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.agents.market_data.microstructure_features import (
    compute_historical_liquidity_void_pressure,
    compute_live_microstructure_reading,
)

_SYMBOL = "BTC/USDT"


def _trade(*, timestamp_ms: int, side: str, amount: float) -> dict[str, Any]:
    return {"timestamp": timestamp_ms, "side": side, "amount": amount, "price": 65_000.0}


def test_empty_tape_returns_an_honest_insufficient_data_reading() -> None:
    reading = compute_live_microstructure_reading([], symbol=_SYMBOL, vpin_bucket_volume=1.0)
    assert reading.trade_count == 0
    assert reading.vpin is None
    assert all(value is None for value in reading.tfi_by_window.values())
    assert reading.insufficient_data_reason is not None


def test_real_vpin_matches_hand_computed_value_across_two_buckets() -> None:
    """3 real trades, bucket_volume=1.0, rolling_buckets=2 -> bucket0=0.6 (0.6 buy/0 sell), bucket1=0.0 (0.6 buy/0.6 sell), rolling mean = 0.3."""
    tape = [
        _trade(timestamp_ms=1_700_000_000_000, side="buy", amount=0.6),
        _trade(timestamp_ms=1_700_000_000_200, side="sell", amount=0.6),
        _trade(timestamp_ms=1_700_000_000_400, side="buy", amount=0.6),
    ]
    reading = compute_live_microstructure_reading(tape, symbol=_SYMBOL, vpin_bucket_volume=1.0, vpin_rolling_buckets=2)

    assert reading.trade_count == 3
    assert reading.vpin == pytest.approx(0.3)
    assert reading.insufficient_data_reason is None


def test_insufficient_volume_for_the_configured_buckets_is_honestly_reported() -> None:
    """Only enough real volume for one bucket -- rolling_buckets=2 can never produce a real reading yet."""
    tape = [_trade(timestamp_ms=1_700_000_000_000, side="buy", amount=0.1)]
    reading = compute_live_microstructure_reading(tape, symbol=_SYMBOL, vpin_bucket_volume=1.0, vpin_rolling_buckets=2)

    assert reading.vpin is None
    assert reading.insufficient_data_reason is not None
    assert "Volume réel insuffisant" in reading.insufficient_data_reason


def test_real_tfi_matches_hand_computed_signed_volume_sum() -> None:
    """3 real trades within 400ms of each other -- every window (1s/5s/30s) sees all three: +0.6 -0.6 +0.6 = 0.6."""
    tape = [
        _trade(timestamp_ms=1_700_000_000_000, side="buy", amount=0.6),
        _trade(timestamp_ms=1_700_000_000_200, side="sell", amount=0.6),
        _trade(timestamp_ms=1_700_000_000_400, side="buy", amount=0.6),
    ]
    reading = compute_live_microstructure_reading(tape, symbol=_SYMBOL, vpin_bucket_volume=10.0)

    assert reading.tfi_by_window["1s"] == pytest.approx(0.6)
    assert reading.tfi_by_window["5s"] == pytest.approx(0.6)
    assert reading.tfi_by_window["30s"] == pytest.approx(0.6)


def test_trades_outside_the_short_window_do_not_count_toward_it() -> None:
    """A trade 2 real seconds before the latest one is outside the 1s window but still inside 5s/30s."""
    tape = [
        _trade(timestamp_ms=1_700_000_000_000, side="sell", amount=1.0),  # 2s before the next trade
        _trade(timestamp_ms=1_700_000_002_000, side="buy", amount=0.4),
    ]
    reading = compute_live_microstructure_reading(tape, symbol=_SYMBOL, vpin_bucket_volume=10.0)

    assert reading.tfi_by_window["1s"] == pytest.approx(0.4)  # only the buy counts
    assert reading.tfi_by_window["5s"] == pytest.approx(-0.6)  # both trades count: 0.4 - 1.0


# =============================================================================
# compute_historical_liquidity_void_pressure (2026-09-19)
# =============================================================================


def test_real_historical_void_pressure_is_bounded_and_finite() -> None:
    pressures = compute_historical_liquidity_void_pressure("BTCUSDT", limit_rows=2000)
    assert len(pressures) > 0
    assert np.all(np.isfinite(pressures))
    assert np.all(np.abs(pressures) <= 1.0)


def test_raises_on_unknown_symbol() -> None:
    with pytest.raises(FileNotFoundError):
        compute_historical_liquidity_void_pressure("DOGEUSDT")

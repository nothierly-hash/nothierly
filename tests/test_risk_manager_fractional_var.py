"""Real, executable tests for RiskManager.calculate_var_fractional_horizon.

See src/agents/supervisor/risk_manager.py's own docstring for what this
corrects (calculate_var()'s implicit square-root-of-time assumption
when scaled to a multi-period horizon) and core/quant_math.py's
"Le Calcul Fractionnaire" (2026-09-14) module docstring entry for the
underlying real math (hurst_exponent_rs, fractional_volatility_scaling).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from core.schemas import OrderBookLevel, OrderBookSnapshot
from src.agents.supervisor.risk_manager import InsufficientHistoryError, RiskManager

_SYMBOL = "BTC/USDT"


def _snapshot(mid_price: float) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=_SYMBOL,
        timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=mid_price - 0.5, volume=10.0)],
        asks=[OrderBookLevel(price=mid_price + 0.5, volume=10.0)],
    )


def _feed_real_random_walk(manager: RiskManager, *, n: int, seed: int) -> None:
    """Real geometric-random-walk mid-price path -- i.i.d. log-return noise (a real, ordinary H~0.5 series)."""
    rng = np.random.default_rng(seed)
    price = 65000.0
    for _ in range(n):
        price *= float(np.exp(rng.normal(0.0, 0.001)))
        manager.record_order_book_snapshot(_snapshot(price))


def test_raises_insufficient_history_below_three_snapshots() -> None:
    manager = RiskManager()
    manager.record_order_book_snapshot(_snapshot(65000.0))
    manager.record_order_book_snapshot(_snapshot(65010.0))

    with pytest.raises(InsufficientHistoryError, match="need >= 3"):
        manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=5.0)


def test_raises_insufficient_history_below_sixteen_returns_even_with_enough_for_calculate_var() -> None:
    """calculate_var()'s own 3-snapshot minimum is real but not enough for a real Hurst estimate."""
    manager = RiskManager(lookback_window=10)
    _feed_real_random_walk(manager, n=10, seed=1)  # 10 snapshots -> 9 returns, clears calculate_var's floor, not this one's

    assert manager.calculate_var(_SYMBOL, 10_000.0) > 0.0  # calculate_var itself remains usable
    with pytest.raises(InsufficientHistoryError, match="Hurst exponent"):
        manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=5.0)


def test_rejects_negative_position_value() -> None:
    manager = RiskManager()
    _feed_real_random_walk(manager, n=30, seed=2)
    with pytest.raises(ValueError, match="position_value"):
        manager.calculate_var_fractional_horizon(_SYMBOL, -1.0, horizon=5.0)


def test_rejects_non_positive_horizon() -> None:
    manager = RiskManager()
    _feed_real_random_walk(manager, n=30, seed=3)
    with pytest.raises(ValueError, match="horizon"):
        manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=0.0)


def test_horizon_of_one_exactly_matches_calculate_var() -> None:
    """Real exact identity (any Hurst exponent to the power of 1 is 1) -- not an approximation, holds for any real data."""
    manager = RiskManager()
    _feed_real_random_walk(manager, n=30, seed=4)

    base = manager.calculate_var(_SYMBOL, 10_000.0)
    fractional_at_horizon_one = manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=1.0)

    assert fractional_at_horizon_one == pytest.approx(base, rel=1e-9)


def test_longer_horizon_produces_larger_real_var() -> None:
    manager = RiskManager()
    _feed_real_random_walk(manager, n=30, seed=5)

    var_short = manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=2.0)
    var_long = manager.calculate_var_fractional_horizon(_SYMBOL, 10_000.0, horizon=20.0)

    assert var_long > var_short


def test_returns_zero_for_zero_position_value() -> None:
    manager = RiskManager()
    _feed_real_random_walk(manager, n=30, seed=6)
    assert manager.calculate_var_fractional_horizon(_SYMBOL, 0.0, horizon=10.0) == 0.0

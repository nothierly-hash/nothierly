"""Real, executable ground-truth tests for risk/pairs_trading.py.

See that module's own docstring for the real gap it closes
(ExecutionAgent.compute_kalman_pairs_spread has the hedge-ratio/spread
machinery but nothing decides *which* pairs to trade or *when*) and the
real 3-step screening pipeline (denoised correlation shortlist -> Engle-
Granger confirmation -> OU half-life) these tests exercise end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.pairs_trading import (
    PairSignal,
    pairs_trading_signal,
    screen_cointegrated_pairs,
)


def _random_walk(n: int, *, seed: int, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=sigma, size=n))


# =============================================================================
# screen_cointegrated_pairs
# =============================================================================


def test_finds_the_real_cointegrated_pair_and_not_the_independent_third_symbol() -> None:
    """A/B share a real common stochastic trend (cointegrated); C is a fully independent real random walk."""
    n = 600
    common_trend = _random_walk(n, seed=1)
    rng = np.random.default_rng(2)
    log_prices = {
        "A": common_trend + rng.normal(scale=0.3, size=n),
        "B": common_trend + rng.normal(scale=0.3, size=n),
        "C": _random_walk(n, seed=3),
    }

    pairs = screen_cointegrated_pairs(log_prices, correlation_threshold=0.5)

    assert len(pairs) == 1
    found = pairs[0]
    assert {found.symbol_a, found.symbol_b} == {"A", "B"}
    assert found.half_life > 0.0
    assert np.isfinite(found.half_life)
    assert found.adf_statistic < found.critical_value


def test_no_pairs_found_among_fully_independent_random_walks() -> None:
    n = 500
    log_prices = {f"SYM{i}": _random_walk(n, seed=10 + i) for i in range(4)}
    pairs = screen_cointegrated_pairs(log_prices, correlation_threshold=0.5)
    assert pairs == []


def test_results_are_sorted_by_adf_statistic_ascending() -> None:
    """The most strongly cointegrated pair (most negative ADF statistic) must come first."""
    n = 700
    trend = _random_walk(n, seed=20)
    rng = np.random.default_rng(21)
    log_prices = {
        "TIGHT_A": trend + rng.normal(scale=0.1, size=n),
        "TIGHT_B": trend + rng.normal(scale=0.1, size=n),
        "LOOSE_A": trend + rng.normal(scale=0.9, size=n),
        "LOOSE_B": trend + rng.normal(scale=0.9, size=n),
    }
    pairs = screen_cointegrated_pairs(log_prices, correlation_threshold=0.3)
    assert len(pairs) >= 1
    statistics = [pair.adf_statistic for pair in pairs]
    assert statistics == sorted(statistics)


def test_series_are_aligned_to_the_common_trailing_length() -> None:
    n_short, n_long = 500, 700
    trend_short = _random_walk(n_short, seed=30)
    rng = np.random.default_rng(31)
    log_prices = {
        "A": trend_short + rng.normal(scale=0.3, size=n_short),
        # B is longer -- its real leading extra history must be trimmed, not misaligned against A's shorter series.
        "B": np.concatenate([_random_walk(n_long - n_short, seed=32), trend_short + rng.normal(scale=0.3, size=n_short)]),
    }
    pairs = screen_cointegrated_pairs(log_prices, correlation_threshold=0.5)
    assert len(pairs) == 1


def test_rejects_fewer_than_two_symbols() -> None:
    with pytest.raises(ValueError, match="at least 2 symbols"):
        screen_cointegrated_pairs({"A": np.array([1.0, 2.0, 3.0])})


def test_rejects_correlation_threshold_out_of_range() -> None:
    log_prices = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        screen_cointegrated_pairs(log_prices, correlation_threshold=0.0)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        screen_cointegrated_pairs(log_prices, correlation_threshold=1.5)


def test_rejects_too_few_common_observations() -> None:
    log_prices = {"A": np.array([1.0, 2.0]), "B": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match="at least 3 real observations"):
        screen_cointegrated_pairs(log_prices)


# =============================================================================
# pairs_trading_signal
# =============================================================================


def test_zscore_matches_hand_computed_value() -> None:
    spread = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pairs_trading_signal(spread, entry_z=1.0, exit_z=0.1)
    mean, std = 3.0, np.std(spread)
    assert result.current_spread == 5.0
    assert result.z_score == pytest.approx((5.0 - mean) / std)


def test_high_zscore_triggers_short_spread() -> None:
    spread = np.concatenate([np.random.default_rng(1).normal(0.0, 1.0, 200), [10.0]])
    result = pairs_trading_signal(spread, entry_z=2.0, exit_z=0.5)
    assert result.signal == PairSignal.SHORT_SPREAD


def test_low_zscore_triggers_long_spread() -> None:
    spread = np.concatenate([np.random.default_rng(1).normal(0.0, 1.0, 200), [-10.0]])
    result = pairs_trading_signal(spread, entry_z=2.0, exit_z=0.5)
    assert result.signal == PairSignal.LONG_SPREAD


def test_near_zero_zscore_triggers_exit() -> None:
    spread = np.concatenate([np.random.default_rng(1).normal(0.0, 1.0, 200), [0.0]])
    result = pairs_trading_signal(spread, entry_z=2.0, exit_z=0.5)
    assert abs(result.z_score) <= 0.5
    assert result.signal == PairSignal.EXIT


def test_middle_zone_triggers_hold() -> None:
    # 200 values alternating +/-1 -> mean ~0, std ~1; the final 2.0 lands at a real z-score of ~2 --
    # strictly between exit_z and entry_z below, the real "reverted neither fully nor far enough" zone.
    spread = np.array([-1.0, 1.0] * 100 + [2.0])
    result = pairs_trading_signal(spread, entry_z=3.0, exit_z=0.5)
    assert 0.5 < abs(result.z_score) < 3.0
    assert result.signal == PairSignal.HOLD


def test_rejects_too_short_history() -> None:
    with pytest.raises(ValueError, match="at least 2 elements"):
        pairs_trading_signal(np.array([1.0]))


def test_rejects_zero_variance_spread() -> None:
    with pytest.raises(ValueError, match="effectively-zero standard deviation"):
        pairs_trading_signal(np.array([5.0, 5.0, 5.0]))


def test_rejects_non_positive_entry_z() -> None:
    with pytest.raises(ValueError, match="entry_z must be positive"):
        pairs_trading_signal(np.array([1.0, 2.0, 3.0]), entry_z=0.0)


def test_rejects_exit_z_at_or_above_entry_z() -> None:
    with pytest.raises(ValueError, match="exit_z must be"):
        pairs_trading_signal(np.array([1.0, 2.0, 3.0]), entry_z=1.0, exit_z=1.0)

"""Real, executable ground-truth tests for core/quant_math.py's sortino_ratio.

See core/quant_math.py's own docstring for sortino_ratio: the real, load-bearing
difference from sharpe_ratio (only downside variance counts), and why both are
exposed to src/dashboard/ems_pages/investor_portal.py rather than just one.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import sharpe_ratio, sortino_ratio


def test_raises_on_too_few_returns() -> None:
    with pytest.raises(ValueError, match="at least 2 elements"):
        sortino_ratio(np.array([0.01]))


def test_raises_on_non_positive_periods_per_year() -> None:
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        sortino_ratio(np.array([0.01, -0.01]), periods_per_year=0)


def test_zero_when_no_return_falls_below_target() -> None:
    """A real, disclosed edge case: with zero downside deviation, this returns 0.0 (never inf/nan),
    the exact same discipline sharpe_ratio already applies to zero variance."""
    all_gains = np.array([0.01, 0.02, 0.015, 0.03])
    assert sortino_ratio(all_gains, target_return=0.0) == 0.0


def test_matches_hand_computed_downside_deviation() -> None:
    """A small, exactly hand-computable series: only the two negative returns count toward risk."""
    returns = np.array([0.02, -0.01, 0.03, -0.02])
    target = 0.0
    periods_per_year = 4

    shortfalls = np.array([0.0, -0.01, 0.0, -0.02])
    expected_downside_deviation = np.sqrt(np.mean(shortfalls**2))
    expected_mean_excess = np.mean(returns) - target
    expected = expected_mean_excess / expected_downside_deviation * np.sqrt(periods_per_year)

    assert sortino_ratio(returns, target_return=target, periods_per_year=periods_per_year) == pytest.approx(expected)


def test_ignores_upside_variance_unlike_sharpe() -> None:
    """The real, headline distinction: adding one large GAIN raises the mean for both ratios, but
    also inflates Sharpe's denominator (total variance counts every deviation, upside included) --
    Sortino's denominator is exactly unchanged (the added value is not a shortfall), so the same
    added gain must reward Sortino strictly more than it rewards Sharpe."""
    base = np.array([0.01, -0.01, 0.005, -0.005, 0.01])
    with_big_upside = np.array([0.01, -0.01, 0.005, -0.005, 0.01, 0.50])

    sharpe_growth_factor = sharpe_ratio(with_big_upside) / sharpe_ratio(base)
    sortino_growth_factor = sortino_ratio(with_big_upside) / sortino_ratio(base)
    assert sortino_growth_factor > sharpe_growth_factor


def test_more_negative_target_return_lowers_downside_deviation() -> None:
    """A real, monotonic sanity check: raising the bar (target_return) for what counts as a
    "shortfall" can only ever add more returns to the downside set, never fewer."""
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
    lenient = sortino_ratio(returns, target_return=-0.03)  # almost nothing counts as downside
    strict = sortino_ratio(returns, target_return=0.02)  # most returns now count as downside
    assert lenient > strict

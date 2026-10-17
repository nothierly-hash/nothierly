"""Real, executable ground-truth tests for core/quant_math.py's "Cointegration & Stat-Arb" addition:
_augmented_dickey_fuller_statistic, engle_granger_cointegration_test.

See core/quant_math.py's own module docstring for why this exists (pairs
trading needs to know *which* pairs are actually cointegrated, not just
correlated -- kalman_dynamic_hedge_ratio/ornstein_uhlenbeck_mean_reversion
answer "how" and "how fast", not "whether"), and for the real, disclosed
fixed-asymptotic-critical-value approximation this test suite's own
known-answer cases are chosen to sit comfortably clear of (not
borderline), the same discipline test_quant_math_hrp.py already uses for
its own exact, formula-derived assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import (
    _ENGLE_GRANGER_CRITICAL_VALUES,
    _augmented_dickey_fuller_statistic,
    engle_granger_cointegration_test,
)


def _ar1_series(n: int, *, phi: float, seed: int, sigma: float = 1.0) -> np.ndarray:
    """A real, simulated stationary AR(1) series: x_t = phi * x_{t-1} + noise_t. phi=1.0 degenerates to a random walk."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=sigma, size=n)
    series = np.empty(n, dtype=np.float64)
    series[0] = noise[0]
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    return series


def _random_walk(n: int, *, seed: int, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=sigma, size=n))


# =============================================================================
# _augmented_dickey_fuller_statistic
# =============================================================================


def test_adf_rejects_unit_root_on_a_real_stationary_ar1_series() -> None:
    """phi=0.3 -- strongly mean-reverting -- must produce a real, clearly negative ADF statistic."""
    series = _ar1_series(500, phi=0.3, seed=1)
    statistic = _augmented_dickey_fuller_statistic(series, lags=1)
    assert statistic < _ENGLE_GRANGER_CRITICAL_VALUES["1%"]


def test_adf_does_not_reject_unit_root_on_a_real_random_walk() -> None:
    """phi=1.0 -- a genuine random walk -- must NOT produce a statistic beating even the loosest (10%) critical value."""
    series = _random_walk(500, seed=2)
    statistic = _augmented_dickey_fuller_statistic(series, lags=1)
    assert statistic > _ENGLE_GRANGER_CRITICAL_VALUES["10%"]


def test_adf_rejects_negative_lags() -> None:
    with pytest.raises(ValueError, match="lags must be"):
        _augmented_dickey_fuller_statistic(np.arange(10, dtype=np.float64), lags=-1)


def test_adf_rejects_too_few_observations_for_the_requested_lags() -> None:
    with pytest.raises(ValueError, match="at least"):
        _augmented_dickey_fuller_statistic(np.array([1.0, 2.0, 3.0]), lags=5)


# =============================================================================
# engle_granger_cointegration_test
# =============================================================================


def test_recovers_the_true_hedge_ratio_from_a_known_linear_construction() -> None:
    """series_a = intercept + hedge_ratio * series_b + stationary noise -- OLS must recover both real, known coefficients."""
    rng = np.random.default_rng(3)
    series_b = _random_walk(500, seed=4)
    true_intercept, true_hedge_ratio = 10.0, 2.5
    series_a = true_intercept + true_hedge_ratio * series_b + rng.normal(scale=0.5, size=500)

    hedge_ratio, intercept, residuals, _adf, _crit, is_cointegrated = engle_granger_cointegration_test(
        series_a, series_b,
    )
    assert hedge_ratio == pytest.approx(true_hedge_ratio, abs=0.1)
    assert intercept == pytest.approx(true_intercept, abs=1.0)
    assert is_cointegrated is True
    assert residuals.shape == series_a.shape


def test_two_series_sharing_one_common_stochastic_trend_are_flagged_cointegrated() -> None:
    """The real textbook cointegration construction: one shared real random walk (the common trend) plus
    independent, genuinely stationary idiosyncratic noise on each leg -- a linear combination (the spread)
    must be stationary even though neither leg alone is."""
    common_trend = _random_walk(600, seed=5)
    rng = np.random.default_rng(6)
    series_a = common_trend + rng.normal(scale=0.3, size=600)
    series_b = common_trend + rng.normal(scale=0.3, size=600)

    *_rest, adf_statistic, critical_value, is_cointegrated = engle_granger_cointegration_test(series_a, series_b)
    assert is_cointegrated is True
    assert adf_statistic < critical_value


def test_two_independent_random_walks_are_not_flagged_cointegrated() -> None:
    """The classic Granger/Newbold "spurious regression" case: two unrelated real random walks can look
    correlated by chance, but their residual spread has no real structural reason to be stationary."""
    series_a = _random_walk(500, seed=7)
    series_b = _random_walk(500, seed=8)

    *_rest, is_cointegrated = engle_granger_cointegration_test(series_a, series_b)
    assert is_cointegrated is False


def test_significance_level_selects_the_matching_critical_value() -> None:
    common_trend = _random_walk(600, seed=5)
    rng = np.random.default_rng(6)
    series_a = common_trend + rng.normal(scale=0.3, size=600)
    series_b = common_trend + rng.normal(scale=0.3, size=600)

    _hr, _ic, _res, _adf, critical_value_1pct, _coint = engle_granger_cointegration_test(
        series_a, series_b, significance="1%",
    )
    _hr2, _ic2, _res2, _adf2, critical_value_10pct, _coint2 = engle_granger_cointegration_test(
        series_a, series_b, significance="10%",
    )
    assert critical_value_1pct == _ENGLE_GRANGER_CRITICAL_VALUES["1%"]
    assert critical_value_10pct == _ENGLE_GRANGER_CRITICAL_VALUES["10%"]
    assert critical_value_1pct < critical_value_10pct  # the 1% bar is real, strictly stricter (more negative)


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same nonempty shape"):
        engle_granger_cointegration_test(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="same nonempty shape"):
        engle_granger_cointegration_test(np.array([], dtype=np.float64), np.array([], dtype=np.float64))

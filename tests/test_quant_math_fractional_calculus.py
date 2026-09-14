"""Real, executable ground-truth tests for core/quant_math.py's "Le Calcul Fractionnaire" additions:
hurst_exponent_rs, fractional_volatility_scaling.

See core/quant_math.py's own module docstring (2026-09-14 entry) and
src/agents/supervisor/risk_manager.py's calculate_var_fractional_horizon
for what these correct (the classical, i.i.d.-Gaussian-only
square-root-of-time VaR horizon scaling) and why the dashboard's own
existing Hurst estimator (src.dashboard.backend_simulator.
estimate_hurst_exponent) was deliberately not reused/duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import fractional_volatility_scaling, hurst_exponent_rs

# =============================================================================
# hurst_exponent_rs
# =============================================================================


def _simulate_fbm_increments(n: int, hurst: float, *, seed: int) -> np.ndarray:
    """Real fractional Gaussian noise (fBm increments) with a KNOWN true Hurst exponent.

    Standard method (Mandelbrot & Van Ness, 1968): builds the exact
    fBm covariance matrix Cov(B_H(s), B_H(t)) = 0.5 * (t^2H + s^2H -
    |t-s|^2H) over n+1 unit-spaced real time points, Cholesky-factors
    it, and applies the factor to real i.i.d. standard normal noise to
    get one real fBm sample path with exactly the specified H (in
    distribution) -- then differences it to real fGn (stationary
    increments), the same real quantity hurst_exponent_rs is meant to
    be run on (a log-return series is itself an increment series).
    """
    rng = np.random.default_rng(seed)
    times = np.arange(1, n + 2, dtype=np.float64)  # n+1 real time points -> n real increments
    t_grid, s_grid = np.meshgrid(times, times)
    two_h = 2.0 * hurst
    covariance = 0.5 * (t_grid ** two_h + s_grid ** two_h - np.abs(t_grid - s_grid) ** two_h)

    cholesky_factor = np.linalg.cholesky(covariance)
    standard_normal = rng.standard_normal(n + 1)
    fbm_path = cholesky_factor @ standard_normal
    return np.diff(fbm_path)


def test_recovers_hurst_near_0_5_for_real_iid_gaussian_noise() -> None:
    """True random-walk increments (no fBm machinery needed -- i.i.d. Gaussian noise IS exactly H=0.5).

    Tolerance reflects classical R/S's real, measured, documented upward
    bias below H~0.7 (see hurst_exponent_rs's own docstring -- this
    project's own verification measured a real mean ~0.59 here across
    10 seeds at n=600, not an implementation bug), not sloppy testing.
    """
    rng = np.random.default_rng(1)
    returns = rng.standard_normal(1200)

    hurst = hurst_exponent_rs(returns)
    assert 0.45 < hurst < 0.75


def test_recovers_a_real_rough_hurst_exponent_from_known_fbm() -> None:
    """H=0.2 -- anti-persistent/"rough", the real regime Gatheral-Jaisson-Rosenbaum (2018) report for realized volatility.

    Tolerance again reflects the real, measured, documented upward bias
    (this project's own verification measured a real mean ~0.36 here).
    """
    increments = _simulate_fbm_increments(600, hurst=0.2, seed=7)

    hurst = hurst_exponent_rs(increments)
    assert 0.25 < hurst < 0.45
    assert hurst < 0.5  # the real, qualitatively important fact: correctly detected as anti-persistent


def test_recovers_a_real_persistent_hurst_exponent_from_known_fbm() -> None:
    """H=0.8 -- persistent/trending, the opposite real regime from the rough-volatility case above (near-unbiased here, per the docstring)."""
    increments = _simulate_fbm_increments(600, hurst=0.8, seed=13)

    hurst = hurst_exponent_rs(increments)
    assert 0.65 < hurst < 0.95
    assert hurst > 0.5  # correctly detected as persistent, not just "close to the right number"


def test_rejects_returns_shorter_than_two_min_chunks() -> None:
    returns = np.random.default_rng(0).standard_normal(10)  # < 2 * default min_chunk_size (8)
    with pytest.raises(ValueError, match="returns must have at least"):
        hurst_exponent_rs(returns)


def test_rejects_min_chunk_size_below_2() -> None:
    returns = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="min_chunk_size"):
        hurst_exponent_rs(returns, min_chunk_size=1)


def test_rejects_num_chunk_sizes_below_2() -> None:
    returns = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError, match="num_chunk_sizes"):
        hurst_exponent_rs(returns, num_chunk_sizes=1)


def test_rejects_a_constant_real_series_with_no_usable_rs_value() -> None:
    """Every chunk has zero variance -- no window size produces a real, finite R/S value to regress on."""
    returns = np.zeros(200)
    with pytest.raises(ValueError, match="Fewer than 2 real window sizes"):
        hurst_exponent_rs(returns)


# =============================================================================
# fractional_volatility_scaling
# =============================================================================


def test_h_0_5_is_exactly_the_classical_square_root_of_time_rule() -> None:
    sigma, horizon = 0.02, 16.0
    scaled = fractional_volatility_scaling(sigma, horizon, hurst_exponent=0.5)
    assert scaled == pytest.approx(sigma * np.sqrt(horizon), rel=1e-12)


def test_persistent_hurst_scales_faster_than_the_classical_rule() -> None:
    sigma, horizon = 0.02, 10.0
    classical = fractional_volatility_scaling(sigma, horizon, hurst_exponent=0.5)
    persistent = fractional_volatility_scaling(sigma, horizon, hurst_exponent=0.8)
    assert persistent > classical


def test_rough_hurst_scales_slower_than_the_classical_rule() -> None:
    sigma, horizon = 0.02, 10.0
    classical = fractional_volatility_scaling(sigma, horizon, hurst_exponent=0.5)
    rough = fractional_volatility_scaling(sigma, horizon, hurst_exponent=0.1)
    assert rough < classical


def test_horizon_of_one_returns_sigma_base_unchanged_regardless_of_hurst() -> None:
    for hurst in (0.1, 0.5, 0.9):
        assert fractional_volatility_scaling(0.03, 1.0, hurst) == pytest.approx(0.03)


def test_rejects_negative_sigma_base() -> None:
    with pytest.raises(ValueError, match="sigma_base"):
        fractional_volatility_scaling(-0.01, 10.0, 0.5)


def test_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        fractional_volatility_scaling(0.02, 0.0, 0.5)

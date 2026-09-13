"""Real, executable ground-truth tests for core/quant_math.py's "Time & Causality" additions:
estimate_lyapunov_exponent, nyquist_undersampling_check, transfer_entropy, liquidity_void_pressure.

Same discipline as every other quant_math verification this session: exact/known ground truth
(the logistic map's exactly-known Lyapunov exponent, a fair-coin-flip lagged-copy causal system)
rather than ambiguous real market data.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import (
    estimate_lyapunov_exponent,
    liquidity_void_pressure,
    nyquist_undersampling_check,
    transfer_entropy,
)

# =============================================================================
# estimate_lyapunov_exponent -- exact ground truth: logistic map at r=4, true lambda = ln(2)
# =============================================================================


def _logistic_map(n: int, x0: float = 0.4123456789) -> np.ndarray:
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    for i in range(1, n):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    return x


def test_lyapunov_recovers_true_exponent_on_logistic_map() -> None:
    x = _logistic_map(3000)
    lam, lyapunov_time = estimate_lyapunov_exponent(x, embedding_dimension=3, time_delay=1, max_time_steps=8)
    assert lam > 0
    assert abs(lam - np.log(2.0)) < 0.35
    assert lyapunov_time == pytest.approx(1.0 / lam)


def test_lyapunov_periodic_signal_shows_less_divergence_than_chaotic() -> None:
    n = 3000
    t = np.arange(n, dtype=np.float64)
    sine = np.sin(2 * np.pi * t / 50.0)
    lam_sine, _ = estimate_lyapunov_exponent(sine, embedding_dimension=3, time_delay=1, max_time_steps=8)
    lam_chaos, _ = estimate_lyapunov_exponent(_logistic_map(n), embedding_dimension=3, time_delay=1, max_time_steps=8)
    assert lam_sine < lam_chaos


def test_lyapunov_nonpositive_exponent_gives_infinite_time() -> None:
    n = 3000
    t = np.arange(n, dtype=np.float64)
    sine = np.sin(2 * np.pi * t / 50.0)
    lam, lyapunov_time = estimate_lyapunov_exponent(sine, embedding_dimension=3, time_delay=1, max_time_steps=8)
    if lam <= 0:
        assert lyapunov_time == float("inf")


def test_lyapunov_rejects_too_short_series() -> None:
    with pytest.raises(ValueError, match="too short"):
        estimate_lyapunov_exponent(np.array([1.0, 2.0, 3.0]), max_time_steps=8)


def test_lyapunov_rejects_invalid_params() -> None:
    x = _logistic_map(100)
    with pytest.raises(ValueError, match="embedding_dimension"):
        estimate_lyapunov_exponent(x, embedding_dimension=0)
    with pytest.raises(ValueError, match="time_delay"):
        estimate_lyapunov_exponent(x, time_delay=0)
    with pytest.raises(ValueError, match="max_time_steps"):
        estimate_lyapunov_exponent(x, min_time_steps=5, max_time_steps=5)


# =============================================================================
# nyquist_undersampling_check
# =============================================================================


def test_nyquist_matches_textbook_example() -> None:
    # User's own example: a real 1s HFT cycle sampled every 1s violates Nyquist (need <= 0.5s).
    is_undersampled, safe_max = nyquist_undersampling_check(1.0, 1.0)
    assert is_undersampled is True
    assert safe_max == pytest.approx(0.5)


def test_nyquist_fast_enough_sampling_passes() -> None:
    is_undersampled, safe_max = nyquist_undersampling_check(0.4, 1.0)
    assert is_undersampled is False
    assert safe_max == pytest.approx(0.5)


def test_nyquist_this_projects_real_ohlcv_vs_tick_cadence_is_undersampled() -> None:
    # Real numbers: crypto_collector.py polls candles every 60s; MarketDataAgent publishes ticks
    # at most every 0.1s.
    is_undersampled, safe_max = nyquist_undersampling_check(60.0, 0.1)
    assert is_undersampled is True
    assert safe_max == pytest.approx(0.05)


@pytest.mark.parametrize("sampling,reference", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)])
def test_nyquist_rejects_non_positive_intervals(sampling: float, reference: float) -> None:
    with pytest.raises(ValueError):
        nyquist_undersampling_check(sampling, reference)


# =============================================================================
# transfer_entropy -- ground truth: Y is an exact lag-1 copy of X (fair coin flips); X independent.
# =============================================================================


def test_transfer_entropy_detects_true_causal_direction() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.integers(0, 2, size=n).astype(np.float64)
    y = np.empty(n, dtype=np.float64)
    y[0] = 0.0
    y[1:] = x[:-1]  # y[t] = x[t-1] exactly

    te_x_to_y = transfer_entropy(x, y, lag=1, n_bins=2)
    te_y_to_x = transfer_entropy(y, x, lag=1, n_bins=2)

    assert te_x_to_y > 0.5  # close to the theoretical max of 1 bit for a binary variable
    assert abs(te_y_to_x) < 0.2  # X is independent noise -- Y's past carries ~no information about it
    assert te_x_to_y > te_y_to_x + 0.3


def test_transfer_entropy_independent_series_is_near_zero() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.integers(0, 2, size=n).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.float64)
    te = transfer_entropy(x, y, lag=1, n_bins=2)
    assert abs(te) < 0.15


def test_transfer_entropy_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="same shape"):
        transfer_entropy(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="lag"):
        transfer_entropy(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), lag=0)
    with pytest.raises(ValueError, match="n_bins"):
        transfer_entropy(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), n_bins=1)
    with pytest.raises(ValueError, match="too short"):
        transfer_entropy(np.array([1.0, 2.0]), np.array([1.0, 2.0]), lag=5)


# =============================================================================
# liquidity_void_pressure
# =============================================================================


def test_liquidity_void_pressure_points_toward_thinner_side() -> None:
    mid = 100.0
    bid_prices = np.array([99.9, 99.8, 99.7])
    bid_volumes = np.array([50.0, 50.0, 50.0])
    ask_prices = np.array([100.1, 100.2, 100.3])
    ask_volumes = np.array([1.0, 1.0, 1.0])

    pressure, bid_resistance, ask_resistance = liquidity_void_pressure(
        bid_prices, bid_volumes, ask_prices, ask_volumes, mid
    )
    assert pressure > 0  # thin ask side -> pulled up
    assert bid_resistance > ask_resistance


def test_liquidity_void_pressure_symmetric_book_is_exactly_zero() -> None:
    pressure, _, _ = liquidity_void_pressure(
        np.array([99.0]), np.array([10.0]), np.array([101.0]), np.array([10.0]), 100.0
    )
    assert pressure == 0.0


def test_liquidity_void_pressure_all_zero_volumes_is_neutral_not_an_error() -> None:
    pressure, bid_r, ask_r = liquidity_void_pressure(
        np.array([99.0]), np.array([0.0]), np.array([101.0]), np.array([0.0]), 100.0
    )
    assert pressure == 0.0
    assert bid_r == 0.0
    assert ask_r == 0.0


def test_liquidity_void_pressure_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="bid_prices"):
        liquidity_void_pressure(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="negative"):
        liquidity_void_pressure(np.array([1.0]), np.array([-1.0]), np.array([1.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError, match="mid_price"):
        liquidity_void_pressure(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), -1.0)

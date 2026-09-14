"""Real, executable ground-truth tests for core/quant_math.py's "Signal, Physique, Algo" additions:
morlet_wavelet_transform, morlet_scale_to_frequency, kalman_dynamic_hedge_ratio,
poiseuille_liquidity_resistance.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import (
    kalman_dynamic_hedge_ratio,
    morlet_scale_to_frequency,
    morlet_wavelet_transform,
    poiseuille_liquidity_resistance,
)

# =============================================================================
# morlet_wavelet_transform / morlet_scale_to_frequency
# =============================================================================

_FS = 50.0
_DT = 1.0 / _FS
_OMEGA0 = 6.0


def test_cwt_power_peaks_at_the_true_frequencys_scale_on_a_pure_sinusoid() -> None:
    n = 2000
    t = np.arange(n) * _DT
    f0 = 4.0
    signal = np.sin(2 * np.pi * f0 * t)
    scale0 = _OMEGA0 / (2 * np.pi * f0) / _DT

    scales = scale0 * np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    power = np.abs(morlet_wavelet_transform(signal, scales, sampling_period=_DT, omega0=_OMEGA0)) ** 2
    interior = slice(300, 1700)
    interior_power = power[:, interior].mean(axis=1)

    assert int(np.argmax(interior_power)) == 2  # index 2 == scale0 * 1.0, the exact true scale
    assert interior_power[2] > interior_power[0] * 10
    assert interior_power[2] > interior_power[-1] * 10


def test_cwt_localizes_two_different_frequencies_to_their_own_real_time_segment() -> None:
    n = 1000
    t = np.arange(n) * _DT
    f1, f2 = 2.0, 8.0
    signal = np.concatenate([np.sin(2 * np.pi * f1 * t[: n // 2]), np.sin(2 * np.pi * f2 * t[n // 2 :])])

    scale_f1 = _OMEGA0 / (2 * np.pi * f1) / _DT
    scale_f2 = _OMEGA0 / (2 * np.pi * f2) / _DT
    scales = np.array([scale_f1, scale_f2])
    power = np.abs(morlet_wavelet_transform(signal, scales, sampling_period=_DT, omega0=_OMEGA0)) ** 2

    power_f1_first_half = power[0, : n // 2].mean()
    power_f1_second_half = power[0, n // 2 :].mean()
    power_f2_first_half = power[1, : n // 2].mean()
    power_f2_second_half = power[1, n // 2 :].mean()

    assert power_f1_first_half > power_f1_second_half * 5
    assert power_f2_second_half > power_f2_first_half * 5


def test_morlet_scale_to_frequency_round_trips_exactly() -> None:
    f0 = 3.3
    scale = _OMEGA0 / (2 * np.pi * f0) / _DT
    recovered = morlet_scale_to_frequency(scale, sampling_period=_DT, omega0=_OMEGA0)
    assert recovered == pytest.approx(f0, rel=1e-9)


def test_cwt_power_peaks_at_the_true_scale_even_evaluated_at_the_very_last_sample() -> None:
    """Real bug fixed 2026-09-14: evaluating right at the series' own boundary (unavoidable for a
    live 'what's happening right now' reading) used to be swamped by zero-padding edge artifacts
    that grew with scale, regardless of the true signal -- fixed via reflection padding."""
    n = 500
    t = np.arange(n) * _DT
    f0 = 3.0
    signal = np.sin(2 * np.pi * f0 * t)
    scale0 = _OMEGA0 / (2 * np.pi * f0) / _DT

    scales = scale0 * np.array([0.3, 0.6, 1.0, 1.6, 3.0])
    power_at_last_sample = np.abs(morlet_wavelet_transform(signal, scales, sampling_period=_DT, omega0=_OMEGA0)[:, -1]) ** 2

    assert int(np.argmax(power_at_last_sample)) == 2  # index 2 == scale0 * 1.0, the true scale


def test_cwt_output_shape_is_always_len_scales_by_len_series_even_for_a_scale_wider_than_the_series() -> None:
    series = np.sin(np.linspace(0, 4 * np.pi, 20))
    huge_scale = np.array([1000.0])  # half-width would be far larger than len(series)
    result = morlet_wavelet_transform(series, huge_scale, sampling_period=1.0)
    assert result.shape == (1, 20)


def test_cwt_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="series"):
        morlet_wavelet_transform(np.array([]), np.array([1.0]))


def test_cwt_rejects_invalid_scales() -> None:
    with pytest.raises(ValueError, match="scales"):
        morlet_wavelet_transform(np.array([1.0, 2.0, 3.0]), np.array([-1.0]))
    with pytest.raises(ValueError, match="scales"):
        morlet_wavelet_transform(np.array([1.0, 2.0, 3.0]), np.array([]))


def test_cwt_rejects_non_positive_sampling_period() -> None:
    with pytest.raises(ValueError, match="sampling_period"):
        morlet_wavelet_transform(np.array([1.0, 2.0, 3.0]), np.array([1.0]), sampling_period=0.0)


def test_morlet_scale_to_frequency_rejects_non_positive_args() -> None:
    with pytest.raises(ValueError, match="scale"):
        morlet_scale_to_frequency(-1.0)
    with pytest.raises(ValueError, match="sampling_period"):
        morlet_scale_to_frequency(1.0, sampling_period=0.0)


# =============================================================================
# kalman_dynamic_hedge_ratio
# =============================================================================


def test_kalman_recovers_exact_hedge_ratio_with_zero_noise() -> None:
    rng = np.random.default_rng(5)
    n = 500
    x = np.cumsum(rng.normal(0, 1, n)) + 100.0
    true_alpha, true_beta = 2.5, 0.75
    y = true_alpha + true_beta * x  # exact, zero-noise relationship

    states, spread = kalman_dynamic_hedge_ratio(y, x, process_variance=1e-8, observation_variance=1e-8)
    final_alpha, final_beta = states[-1]

    assert final_alpha == pytest.approx(true_alpha, abs=0.05)
    assert final_beta == pytest.approx(true_beta, abs=0.05)
    assert np.abs(spread[-50:]).mean() < 0.5


def test_kalman_tracks_a_slowly_time_varying_beta_under_real_noise() -> None:
    rng = np.random.default_rng(5)
    n = 1000
    x = np.cumsum(rng.normal(0, 1, n)) + 100.0
    true_beta_path = 0.5 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, n))
    y = 1.0 + true_beta_path * x + rng.normal(0, 0.5, n)

    states, _spread = kalman_dynamic_hedge_ratio(y, x, process_variance=1e-4, observation_variance=0.25)
    burn_in = 200
    mae = np.mean(np.abs(states[burn_in:, 1] - true_beta_path[burn_in:]))
    assert mae < 0.25


def test_kalman_rejects_mismatched_or_empty_inputs() -> None:
    with pytest.raises(ValueError, match="same nonempty shape"):
        kalman_dynamic_hedge_ratio(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="same nonempty shape"):
        kalman_dynamic_hedge_ratio(np.array([]), np.array([]))


def test_kalman_rejects_non_positive_variances() -> None:
    y, x = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="process_variance"):
        kalman_dynamic_hedge_ratio(y, x, process_variance=0.0)
    with pytest.raises(ValueError, match="observation_variance"):
        kalman_dynamic_hedge_ratio(y, x, observation_variance=0.0)


# =============================================================================
# poiseuille_liquidity_resistance
# =============================================================================


def test_poiseuille_halving_depth_multiplies_resistance_by_exactly_16() -> None:
    spread, depth = 0.01, 100.0
    r1 = poiseuille_liquidity_resistance(spread, depth)
    r2 = poiseuille_liquidity_resistance(spread, depth / 2.0)
    assert r2 / r1 == pytest.approx(16.0, abs=1e-9)


def test_poiseuille_scales_linearly_with_spread_and_horizon() -> None:
    base = poiseuille_liquidity_resistance(0.01, 100.0, execution_horizon=1.0)
    double_spread = poiseuille_liquidity_resistance(0.02, 100.0, execution_horizon=1.0)
    double_horizon = poiseuille_liquidity_resistance(0.01, 100.0, execution_horizon=2.0)
    assert double_spread == pytest.approx(base * 2.0)
    assert double_horizon == pytest.approx(base * 2.0)


def test_poiseuille_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="spread"):
        poiseuille_liquidity_resistance(-1.0, 1.0)
    with pytest.raises(ValueError, match="liquidity_depth"):
        poiseuille_liquidity_resistance(1.0, 0.0)
    with pytest.raises(ValueError, match="execution_horizon"):
        poiseuille_liquidity_resistance(1.0, 1.0, execution_horizon=0.0)

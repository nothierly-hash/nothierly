"""Real, executable ground-truth tests for core/quant_math.py's hawkes_intensity, almgren_chriss_trajectory,
participation_rate_capped_schedule, almgren_chriss_impact_costs, and calibrate_impact_coefficient_fdt.

Found live (2026-09-22), via a systematic smoke test of every public function in the module: these five
run without error and return finite output, but had zero dedicated test coverage anywhere in this repo.
This file closes that gap with hand-computed closed-form values -- every one of these five is exact,
closed-form arithmetic, so "hand-computed" here means literally worked out with a calculator, not
approximated or re-derived from a fitted model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.quant_math import (
    almgren_chriss_impact_costs,
    almgren_chriss_trajectory,
    calibrate_impact_coefficient_fdt,
    hawkes_intensity,
    participation_rate_capped_schedule,
)

# =============================================================================
# hawkes_intensity
# =============================================================================


def test_matches_hand_computed_single_event() -> None:
    # lambda(2.0) = mu + alpha * exp(-beta * (2.0 - 0.0)) = 0.1 + 0.3 * exp(-1.0)
    result = hawkes_intensity(
        np.array([0.0]), baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3, at_time=2.0,
    )
    assert result == pytest.approx(0.1 + 0.3 * math.exp(-0.5 * 2.0))


def test_matches_hand_computed_two_weighted_events() -> None:
    events = np.array([0.0, 1.0])
    weights = np.array([2.0, 0.5])
    expected = 0.1 + 2.0 * 0.3 * math.exp(-0.5 * 3.0) + 0.5 * 0.3 * math.exp(-0.5 * 2.0)
    result = hawkes_intensity(
        events, baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3, event_weights=weights, at_time=3.0,
    )
    assert result == pytest.approx(expected)


def test_empty_history_returns_exactly_the_baseline() -> None:
    assert hawkes_intensity(np.array([]), baseline_intensity=0.42, decay_rate=1.0, excitation_strength=0.3) == 0.42


def test_future_events_do_not_excite_the_present() -> None:
    """An event at t=10 evaluated at t=2 has not happened yet from the process's own point of view --
    it must contribute exactly nothing, same as if it were absent."""
    with_future_event = hawkes_intensity(
        np.array([0.0, 10.0]), baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3, at_time=2.0,
    )
    without_it = hawkes_intensity(
        np.array([0.0]), baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3, at_time=2.0,
    )
    assert with_future_event == pytest.approx(without_it)


def test_defaults_evaluation_time_to_the_latest_event() -> None:
    result = hawkes_intensity(np.array([0.0, 5.0]), baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3)
    expected = 0.1 + 0.3 * math.exp(-0.5 * 5.0) + 0.3 * math.exp(0.0)  # elapsed at the latest event itself is 0
    assert result == pytest.approx(expected)


def test_raises_on_non_positive_decay_rate() -> None:
    with pytest.raises(ValueError, match="decay_rate must be positive"):
        hawkes_intensity(np.array([0.0]), baseline_intensity=0.1, decay_rate=0.0, excitation_strength=0.3)


def test_raises_on_mismatched_event_weights_shape() -> None:
    with pytest.raises(ValueError, match="event_weights must be the same shape"):
        hawkes_intensity(
            np.array([0.0, 1.0]), baseline_intensity=0.1, decay_rate=0.5, excitation_strength=0.3,
            event_weights=np.array([1.0]),
        )


# =============================================================================
# almgren_chriss_trajectory
# =============================================================================


def test_matches_hand_computed_sinh_ratio_at_the_midpoint() -> None:
    order_size, horizon, vol, risk_aversion, eta = 1_000.0, 100.0, 0.2, 1e-6, 0.01
    kappa = math.sqrt(risk_aversion * vol**2 / eta)  # = 0.002
    trajectory = almgren_chriss_trajectory(
        order_size, execution_horizon_seconds=horizon, volatility=vol, risk_aversion=risk_aversion,
        temporary_impact_coefficient=eta, n_steps=2,
    )
    midpoint = trajectory[1]  # t = 50 = T/2
    expected_midpoint = order_size * math.sinh(kappa * 50.0) / math.sinh(kappa * horizon)
    assert midpoint == pytest.approx(expected_midpoint, rel=1e-9)


def test_boundary_conditions_hold_exactly() -> None:
    trajectory = almgren_chriss_trajectory(
        500.0, execution_horizon_seconds=60.0, volatility=0.3, risk_aversion=0.5,
        temporary_impact_coefficient=0.02, n_steps=8,
    )
    assert trajectory[0] == pytest.approx(500.0)
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(trajectory) <= 1e-9)  # strictly non-increasing


def test_risk_neutral_limit_is_exactly_the_linear_twap_schedule() -> None:
    """risk_aversion=0.0 -> kappa=0.0 -- the module's own documented, explicitly-handled 0/0 limit."""
    order_size, horizon = 1_000.0, 100.0
    trajectory = almgren_chriss_trajectory(
        order_size, execution_horizon_seconds=horizon, volatility=0.3, risk_aversion=0.0,
        temporary_impact_coefficient=0.01, n_steps=4,
    )
    t = np.linspace(0.0, horizon, 5)
    expected = order_size * (1.0 - t / horizon)
    assert trajectory == pytest.approx(expected)


def test_larger_risk_aversion_front_loads_execution_more() -> None:
    """A real, monotonic sanity check straight from the module's own docstring: higher risk aversion
    (more urgency) must leave LESS inventory remaining at the same intermediate time."""
    common = dict(execution_horizon_seconds=100.0, volatility=0.3, temporary_impact_coefficient=0.01, n_steps=4)
    cautious = almgren_chriss_trajectory(1_000.0, risk_aversion=0.01, **common)
    urgent = almgren_chriss_trajectory(1_000.0, risk_aversion=5.0, **common)
    assert urgent[2] < cautious[2]


def test_raises_on_non_positive_order_size() -> None:
    with pytest.raises(ValueError, match="order_size must be positive"):
        almgren_chriss_trajectory(0.0, execution_horizon_seconds=10.0, volatility=0.1, risk_aversion=0.1, temporary_impact_coefficient=0.01)


# =============================================================================
# participation_rate_capped_schedule
# =============================================================================


def test_matches_hand_computed_capping() -> None:
    trajectory = np.array([100.0, 70.0, 30.0, 0.0])  # uncapped slice sizes: 30, 40, 30
    volume_profile = np.array([100.0, 100.0, 100.0])
    sizes, was_capped = participation_rate_capped_schedule(trajectory, volume_profile, max_participation_rate=0.2)
    assert sizes == pytest.approx([20.0, 20.0, 20.0])  # 20% of 100 = 20, binds on every slice
    assert was_capped is True


def test_generous_volume_never_caps() -> None:
    trajectory = np.array([100.0, 70.0, 30.0, 0.0])
    volume_profile = np.array([10_000.0, 10_000.0, 10_000.0])
    sizes, was_capped = participation_rate_capped_schedule(trajectory, volume_profile, max_participation_rate=0.1)
    assert sizes == pytest.approx([30.0, 40.0, 30.0])
    assert was_capped is False


def test_raises_on_volume_profile_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must have trajectory.size - 1"):
        participation_rate_capped_schedule(np.array([100.0, 50.0, 0.0]), np.array([10.0]))


def test_raises_on_non_positive_volume_entry() -> None:
    with pytest.raises(ValueError, match="must all be positive"):
        participation_rate_capped_schedule(np.array([100.0, 50.0, 0.0]), np.array([10.0, 0.0]))


# =============================================================================
# almgren_chriss_impact_costs
# =============================================================================


def test_matches_hand_computed_temporary_and_permanent_impact() -> None:
    trade_sizes = np.array([10.0, 20.0, 5.0])
    temp, perm = almgren_chriss_impact_costs(
        trade_sizes, temporary_impact_coefficient=0.02, permanent_impact_coefficient=0.01, step_duration_seconds=2.0,
    )
    assert temp == pytest.approx([0.1, 0.2, 0.05])  # 0.02 * size / 2.0
    assert perm == pytest.approx([0.1, 0.3, 0.35])  # 0.01 * cumsum([10, 30, 35])


def test_zero_coefficients_give_zero_impact() -> None:
    temp, perm = almgren_chriss_impact_costs(
        np.array([10.0, 20.0]), temporary_impact_coefficient=0.0, permanent_impact_coefficient=0.0, step_duration_seconds=1.0,
    )
    assert temp == pytest.approx([0.0, 0.0])
    assert perm == pytest.approx([0.0, 0.0])


def test_raises_on_negative_trade_size() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        almgren_chriss_impact_costs(np.array([-1.0]), temporary_impact_coefficient=0.1, permanent_impact_coefficient=0.1, step_duration_seconds=1.0)


def test_raises_on_non_positive_step_duration() -> None:
    with pytest.raises(ValueError, match="step_duration_seconds must be positive"):
        almgren_chriss_impact_costs(np.array([1.0]), temporary_impact_coefficient=0.1, permanent_impact_coefficient=0.1, step_duration_seconds=0.0)


# =============================================================================
# calibrate_impact_coefficient_fdt
# =============================================================================


def test_matches_hand_computed_formula() -> None:
    spread = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    expected = 2.0 * float(np.std(spread, ddof=1)) / 10.0
    assert calibrate_impact_coefficient_fdt(spread, average_trade_rate=10.0, coupling_constant=2.0) == pytest.approx(expected)


def test_scales_linearly_with_coupling_constant() -> None:
    spread = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.05])
    base = calibrate_impact_coefficient_fdt(spread, average_trade_rate=5.0, coupling_constant=1.0)
    doubled = calibrate_impact_coefficient_fdt(spread, average_trade_rate=5.0, coupling_constant=2.0)
    assert doubled == pytest.approx(base * 2.0)


def test_calmer_spread_gives_a_smaller_coefficient() -> None:
    calm = np.array([1.00, 1.01, 0.99, 1.00, 1.01, 0.99])
    volatile = np.array([1.0, 1.8, 0.3, 1.6, 0.4, 1.7])
    calm_eta = calibrate_impact_coefficient_fdt(calm, average_trade_rate=10.0)
    volatile_eta = calibrate_impact_coefficient_fdt(volatile, average_trade_rate=10.0)
    assert calm_eta < volatile_eta


def test_raises_on_too_few_spread_samples() -> None:
    with pytest.raises(ValueError, match="at least 2 elements"):
        calibrate_impact_coefficient_fdt(np.array([1.0]), average_trade_rate=10.0)


def test_raises_on_non_positive_trade_rate() -> None:
    with pytest.raises(ValueError, match="average_trade_rate must be positive"):
        calibrate_impact_coefficient_fdt(np.array([1.0, 1.1]), average_trade_rate=0.0)

"""Real, executable ground-truth tests for core/quant_math.py's parametric_var, evt_var, kelly_criterion,
and kelly_criterion_continuous.

Found live (2026-09-22), via a systematic smoke test of every public function in the module: these four
run without error and return finite output, but had zero dedicated test coverage anywhere in this repo --
"operational" was never the same claim as "verified against a known result." This file closes that gap.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto, norm, t as student_t

from core.quant_math import (
    Z_SCORES_BY_CONFIDENCE,
    evt_var,
    kelly_criterion,
    kelly_criterion_continuous,
    parametric_var,
)

# =============================================================================
# parametric_var
# =============================================================================


def test_z_score_table_matches_the_real_standard_normal_quantile() -> None:
    """The table this function reads from is itself a real, independently-checkable fact: z_alpha is
    literally scipy's own standard normal ppf at each confidence level -- not a typed-in guess."""
    for confidence, z in Z_SCORES_BY_CONFIDENCE.items():
        assert z == pytest.approx(float(norm.ppf(confidence)), abs=1e-4)


def test_matches_hand_computed_formula() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008])
    expected = 1.6449 * float(np.std(returns, ddof=1)) * 100_000.0
    assert parametric_var(returns, confidence_level=0.95, position_value=100_000.0) == pytest.approx(expected)


def test_scales_linearly_with_position_value() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    small = parametric_var(returns, confidence_level=0.99, position_value=1_000.0)
    large = parametric_var(returns, confidence_level=0.99, position_value=10_000.0)
    assert large == pytest.approx(small * 10.0)


def test_higher_confidence_gives_a_larger_var() -> None:
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.03])
    var_90 = parametric_var(returns, confidence_level=0.90, position_value=10_000.0)
    var_99 = parametric_var(returns, confidence_level=0.99, position_value=10_000.0)
    assert var_99 > var_90


def test_raises_on_too_few_returns() -> None:
    with pytest.raises(ValueError, match="at least 2 elements"):
        parametric_var(np.array([0.01]), confidence_level=0.95, position_value=1.0)


def test_raises_on_unsupported_confidence_level() -> None:
    with pytest.raises(ValueError, match="Unsupported confidence_level"):
        parametric_var(np.array([0.01, -0.01]), confidence_level=0.5, position_value=1.0)


def test_raises_on_negative_position_value() -> None:
    with pytest.raises(ValueError, match="position_value must not be negative"):
        parametric_var(np.array([0.01, -0.01]), confidence_level=0.95, position_value=-1.0)


# =============================================================================
# evt_var
# =============================================================================


def test_self_consistent_with_its_own_documented_gpd_formula() -> None:
    """Independently refits the GPD to the same exceedances and recomputes the closed-form POT
    quantile by hand -- if this function's own arithmetic ever drifts from its documented formula,
    this catches it even without an external ground truth for the 'true' tail VaR."""
    rng = np.random.default_rng(3)
    returns = rng.standard_t(df=3, size=3_000) * 0.01
    confidence, position_value, threshold_pct = 0.99, 50_000.0, 0.90

    losses = -returns
    threshold = float(np.quantile(losses, threshold_pct))
    exceedances = losses[losses > threshold] - threshold
    shape_xi, _loc, scale_beta = genpareto.fit(exceedances, floc=0)
    n, n_u = losses.size, exceedances.size
    tail_probability = (n / n_u) * (1.0 - confidence)
    if abs(shape_xi) < 1e-8:
        expected_return_units = threshold - scale_beta * np.log(tail_probability)
    else:
        expected_return_units = threshold + (scale_beta / shape_xi) * (tail_probability ** (-shape_xi) - 1.0)
    expected = max(0.0, expected_return_units) * position_value

    assert evt_var(returns, confidence, position_value, threshold_percentile=threshold_pct) == pytest.approx(expected, rel=1e-6)


def test_beats_the_gaussian_assumption_on_a_known_fat_tailed_distribution() -> None:
    """The real, headline reason this function exists: on a Student-t(df=3) generating process (a
    known, analytically-quantifiable fat tail), the Gaussian parametric_var must understate the true
    99% tail loss more than the distribution-free EVT estimate does -- checked against the real
    analytic Student-t quantile, not a second guess."""
    rng = np.random.default_rng(11)
    df, scale, confidence, position_value = 3, 0.01, 0.99, 1_000_000.0
    returns = rng.standard_t(df=df, size=20_000) * scale

    true_var = -scale * float(student_t.ppf(1.0 - confidence, df)) * position_value

    evt_estimate = evt_var(returns, confidence, position_value)
    gaussian_estimate = parametric_var(returns, confidence_level=confidence, position_value=position_value)

    evt_error = abs(evt_estimate - true_var) / true_var
    gaussian_error = abs(gaussian_estimate - true_var) / true_var
    assert evt_error < gaussian_error


def test_raises_when_too_few_exceedances_for_a_trustworthy_fit() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, 20)  # far too few observations for 10 real exceedances above the 90th pct
    with pytest.raises(ValueError, match="not enough tail data"):
        evt_var(returns, 0.99, 1_000.0, min_exceedances=10)


def test_raises_on_out_of_range_confidence_level() -> None:
    with pytest.raises(ValueError, match=r"confidence_level must be in \(0, 1\)"):
        evt_var(np.random.default_rng(0).normal(0, 0.01, 200), 1.5, 1_000.0)


# =============================================================================
# kelly_criterion
# =============================================================================


def test_matches_hand_computed_discrete_formula() -> None:
    # p=0.6, b=2: f* = 0.6 - 0.4/2 = 0.4
    assert kelly_criterion(win_probability=0.6, win_loss_ratio=2.0) == pytest.approx(0.4)


def test_half_kelly_is_exactly_half_of_full_kelly() -> None:
    full = kelly_criterion(win_probability=0.6, win_loss_ratio=2.0, fraction=1.0)
    half = kelly_criterion(win_probability=0.6, win_loss_ratio=2.0, fraction=0.5)
    assert half == pytest.approx(full * 0.5)


def test_negative_edge_clips_to_zero_never_a_negative_position() -> None:
    # p=0.4, b=1: raw = 0.4 - 0.6/1 = -0.2 -- a losing edge, "do not bet," not "short."
    assert kelly_criterion(win_probability=0.4, win_loss_ratio=1.0) == 0.0


def test_scaled_up_fraction_clips_to_full_capital_never_leverage() -> None:
    # raw = 0.6 - 0.4/2 = 0.4; fraction=3 would want 1.2 -- must clip at 1.0, not lever up.
    assert kelly_criterion(win_probability=0.6, win_loss_ratio=2.0, fraction=3.0) == 1.0


def test_raises_on_win_probability_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"win_probability must be in \[0, 1\]"):
        kelly_criterion(win_probability=1.5, win_loss_ratio=2.0)


def test_raises_on_non_positive_win_loss_ratio() -> None:
    with pytest.raises(ValueError, match="win_loss_ratio must be positive"):
        kelly_criterion(win_probability=0.5, win_loss_ratio=0.0)


# =============================================================================
# kelly_criterion_continuous
# =============================================================================


def test_matches_hand_computed_continuous_formula() -> None:
    # mu=0.02, sigma^2=0.04: f* = 0.02 / 0.04 = 0.5
    assert kelly_criterion_continuous(expected_return=0.02, variance=0.04) == pytest.approx(0.5)


def test_negative_expected_return_clips_to_zero() -> None:
    assert kelly_criterion_continuous(expected_return=-0.01, variance=0.04) == 0.0


def test_large_edge_clips_to_full_capital() -> None:
    # raw = 1.0 / 0.5 = 2.0 -- must clip at 1.0.
    assert kelly_criterion_continuous(expected_return=1.0, variance=0.5) == 1.0


def test_half_kelly_continuous_is_exactly_half() -> None:
    full = kelly_criterion_continuous(expected_return=0.02, variance=0.04, fraction=1.0)
    half = kelly_criterion_continuous(expected_return=0.02, variance=0.04, fraction=0.5)
    assert half == pytest.approx(full * 0.5)


def test_raises_on_non_positive_variance() -> None:
    with pytest.raises(ValueError, match="variance must be positive"):
        kelly_criterion_continuous(expected_return=0.02, variance=0.0)

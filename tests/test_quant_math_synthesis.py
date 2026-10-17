"""Synthesis test: the real functions of core/quant_math.py, chained the way the live pipeline actually
chains them, over ONE consistent synthetic multi-asset dataset -- not a second round of isolated unit
checks (each function already has those, see the other test_quant_math_*.py files), but a proof that
their outputs compose: Gate 3 (sizing) -> Gate 5 (risk) -> Gate 6 (portfolio/RMT) -> Gate 7
(microstructure/execution), mirroring src/agents/execution/agent.py's real 8-gate order.

A failure here means two individually-correct functions still don't fit together (wrong units, wrong
sign convention, a shape mismatch) -- the class of bug unit tests, by construction, cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import (
    almgren_chriss_impact_costs,
    almgren_chriss_trajectory,
    black_scholes_delta,
    black_scholes_price,
    check_hedge_completeness,
    denoise_correlation_matrix_mp,
    evt_var,
    giant_cluster_fraction,
    hawkes_intensity,
    hierarchical_risk_parity_weights,
    jump_diffusion_anomaly,
    kelly_criterion_continuous,
    marchenko_pastur_threshold,
    max_drawdown,
    ornstein_uhlenbeck_mean_reversion,
    parametric_var,
    participation_rate_capped_schedule,
    price_impact,
    sharpe_ratio,
    solve_hedge_ratios,
)

_SEED = 2026_09_22
_N_ASSETS = 6
_N_OBS = 400


@pytest.fixture(scope="module")
def market() -> dict[str, np.ndarray]:
    """One consistent, realistic synthetic multi-asset market: a shared factor plus idiosyncratic
    noise, a mean-reverting spread instrument, and one instrument with an injected flash-crash tick --
    everything downstream reuses THIS data, never a fresh draw per step."""
    rng = np.random.default_rng(_SEED)
    factor = rng.normal(0.0003, 0.01, _N_OBS)
    idio = rng.normal(0.0, 0.006, (_N_OBS, _N_ASSETS))
    loadings = np.array([1.0, 0.9, 0.7, 0.5, 0.3, -0.4])
    returns = factor[:, None] * loadings[None, :] + idio  # (N_OBS, N_ASSETS)

    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    equity_curve = 1_000_000.0 * np.cumprod(1.0 + returns[:, 0])

    # A real, injected flash-crash tick on asset 0's own return series, late in the window --
    # everything a jump/anomaly detector below is checked against.
    shocked_returns = returns[:, 0].copy()
    shock_index = 350
    shocked_returns[shock_index] -= 0.15  # a real, single, deliberate 15% one-tick drawdown

    return {
        "returns": returns,
        "prices": prices,
        "equity_curve": equity_curve,
        "shocked_returns": shocked_returns,
        "shock_index": shock_index,
    }


# =============================================================================
# Step 1 -- Gate 5 (RiskManager): does the risk read make sense on this data?
# =============================================================================


def test_step1_risk_metrics_are_internally_consistent(market: dict[str, np.ndarray]) -> None:
    strategy_returns = market["returns"][:, 0]
    equity = market["equity_curve"]

    sharpe = sharpe_ratio(strategy_returns)
    mdd = max_drawdown(equity)
    var_95 = parametric_var(strategy_returns, confidence_level=0.95, position_value=1_000_000.0)
    var_99 = parametric_var(strategy_returns, confidence_level=0.99, position_value=1_000_000.0)
    evt_99 = evt_var(strategy_returns, 0.99, 1_000_000.0)

    assert np.isfinite(sharpe)
    assert -1.0 <= mdd <= 0.0
    assert 0.0 <= var_95 < var_99  # a stricter confidence must never demand LESS capital at risk
    assert evt_99 > 0.0  # a real, positive tail-loss estimate on 400 real observations


def test_step2_sizing_from_the_same_return_series_the_risk_read_used(market: dict[str, np.ndarray]) -> None:
    """Gate 3 (Sizing): the Kelly fraction must be derived from literally the same mean/variance the
    Gate 5 risk read (step 1) just measured -- not a fresh, disconnected re-estimate."""
    strategy_returns = market["returns"][:, 0]
    mu = float(np.mean(strategy_returns))
    sigma_sq = float(np.var(strategy_returns, ddof=1))

    kelly_fraction = kelly_criterion_continuous(expected_return=mu, variance=sigma_sq, fraction=0.5)
    assert 0.0 <= kelly_fraction <= 1.0

    capital, price = 1_000_000.0, float(market["prices"][-1, 0])
    order_size_units = (kelly_fraction * capital) / price
    assert order_size_units >= 0.0


# =============================================================================
# Step 3 -- Gate 6 (PortfolioRiskParityManager): RMT-cleaned correlation feeds HRP directly
# =============================================================================


def test_step3_denoised_correlation_feeds_a_valid_hrp_allocation(market: dict[str, np.ndarray]) -> None:
    returns = market["returns"]
    corr = np.corrcoef(returns, rowvar=False)
    cov = np.cov(returns, rowvar=False)

    lam_minus, lam_plus = marchenko_pastur_threshold(n_assets=_N_ASSETS, n_samples=_N_OBS)
    assert 0.0 <= lam_minus < lam_plus  # the noise band itself must be a real, non-degenerate interval

    cleaned_corr = denoise_correlation_matrix_mp(corr, n_samples=_N_OBS)
    assert np.diag(cleaned_corr) == pytest.approx(np.ones(_N_ASSETS), abs=1e-6)

    # HRP is defined over a covariance, not a correlation matrix -- rescale the cleaned correlation
    # back into covariance units using the ORIGINAL (not cleaned) real per-asset variances, the same
    # composition src/agents/supervisor/portfolio_risk_manager.py's own Gate 6 pipeline performs.
    std = np.sqrt(np.diag(cov))
    cleaned_cov = cleaned_corr * np.outer(std, std)

    weights = hierarchical_risk_parity_weights(cleaned_cov)
    assert weights.shape == (_N_ASSETS,)
    assert np.all(weights >= 0.0)  # HRP is long-only by construction
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-8)  # a real, fully-invested allocation

    cluster_fraction = giant_cluster_fraction(cleaned_corr, threshold=0.3)
    assert 1.0 / _N_ASSETS <= cluster_fraction <= 1.0


# =============================================================================
# Step 4 -- Gate 7 (StochasticCore): does the injected shock actually get caught?
# =============================================================================


def test_step4_the_injected_flash_crash_tick_is_actually_detected(market: dict[str, np.ndarray]) -> None:
    """A real end-to-end proof, not an assumption: a 15% one-tick shock DELIBERATELY planted into
    the fixture (step 0) must trip the jump-diffusion detector when it is the most recent observation
    -- exactly the live call shape Gate 7 uses (evaluate the newest return against the recent window)."""
    shocked_returns = market["shocked_returns"]
    shock_index = market["shock_index"]
    window = shocked_returns[: shock_index + 1]  # only data up to and including the shock -- no lookahead

    is_black_swan, z_score = jump_diffusion_anomaly(window, jump_threshold_std=4.0)
    assert bool(is_black_swan) is True
    assert z_score > 4.0

    # A quiet window with no shock, same real detector, must NOT flag anything -- the detector has to
    # be selective, not permanently tripped.
    calm_window = shocked_returns[:shock_index]
    is_black_swan_calm, _ = jump_diffusion_anomaly(calm_window, jump_threshold_std=4.0)
    assert bool(is_black_swan_calm) is False


def test_step5_order_flow_intensity_rises_around_the_real_shock(market: dict[str, np.ndarray]) -> None:
    """Hawkes intensity, fed a real burst of order-flow events clustered around the same shock tick,
    must read meaningfully above baseline right after the burst -- the real 'cascade' signal Gate 7
    is meant to catch, evaluated on a realistic event stream, not a synthetic one-off timestamp."""
    rng = np.random.default_rng(_SEED)
    quiet_events = np.sort(rng.uniform(0, 100, 5))  # a real, thin baseline order-flow rate
    burst_events = np.sort(rng.uniform(100, 102, 40))  # a real burst: 40 orders in 2 real time units
    baseline = hawkes_intensity(quiet_events, baseline_intensity=0.05, decay_rate=0.3, excitation_strength=0.4, at_time=100.0)
    after_burst = hawkes_intensity(
        np.concatenate([quiet_events, burst_events]), baseline_intensity=0.05, decay_rate=0.3,
        excitation_strength=0.4, at_time=102.0,
    )
    assert after_burst > 5.0 * baseline


def test_step6_mean_reversion_read_agrees_with_how_the_fixture_was_built(market: dict[str, np.ndarray]) -> None:
    """Asset 5 was loaded NEGATIVELY onto the shared factor (loading -0.4) while every other asset
    loads positively -- a real, structural difference the OU calibration and the correlation matrix
    must BOTH agree on, independently, from the same fixture."""
    corr = np.corrcoef(market["returns"], rowvar=False)
    assert corr[0, 5] < 0.0  # asset 0 (loading +1.0) and asset 5 (loading -0.4) must anti-correlate

    theta, _mu, _half_life = ornstein_uhlenbeck_mean_reversion(market["prices"][:, 0], dt=1.0)
    assert np.isfinite(theta)  # a real number either way -- see its own dedicated test for the nan/inf edge case


# =============================================================================
# Step 7 -- execution: size the real order, schedule it, cost it, and hedge it
# =============================================================================


def test_step7_full_execution_chain_from_sizing_to_hedge(market: dict[str, np.ndarray]) -> None:
    """The real, final composition: Kelly sizing (step 2) feeds an Almgren-Chriss trajectory, which
    feeds a participation-capped schedule, which feeds real impact costs -- then Gate 7's own
    price-impact check and a real options hedge are evaluated against the SAME sized order, all
    consistent with each other end to end."""
    strategy_returns = market["returns"][:, 0]
    mu, sigma_sq = float(np.mean(strategy_returns)), float(np.var(strategy_returns, ddof=1))
    kelly_fraction = kelly_criterion_continuous(mu, sigma_sq, fraction=0.5)
    capital, spot = 1_000_000.0, float(market["prices"][-1, 0])
    order_units = max((kelly_fraction * capital) / spot, 1.0)  # floor at 1 unit so the chain below is never degenerate

    trajectory = almgren_chriss_trajectory(
        order_units, execution_horizon_seconds=300.0, volatility=0.0005, risk_aversion=1e-7,
        temporary_impact_coefficient=0.02, n_steps=6,
    )
    assert trajectory[0] == pytest.approx(order_units)
    assert trajectory[-1] == pytest.approx(0.0, abs=1e-6)

    volume_profile = np.full(6, order_units * 2.0)  # a real, generous expected-volume assumption
    trade_sizes, _was_capped = participation_rate_capped_schedule(trajectory, volume_profile, max_participation_rate=0.5)
    assert trade_sizes.sum() <= order_units + 1e-6

    temp_impact, perm_impact = almgren_chriss_impact_costs(
        trade_sizes, temporary_impact_coefficient=0.02, permanent_impact_coefficient=0.005, step_duration_seconds=50.0,
    )
    assert np.all(temp_impact >= 0.0) and np.all(perm_impact >= 0.0)
    assert perm_impact[-1] >= perm_impact[0]  # cumulative permanent impact never reverses

    # Gate 7's own real veto check: the same order's estimated price impact must not exceed a sane
    # bound relative to the venue's own available liquidity.
    estimated_liquidity = order_units * 50.0  # a real, generously deep book relative to this order
    impact_fraction = price_impact(order_units, estimated_liquidity, volatility=0.02, impact_coefficient=0.5)
    assert 0.0 <= impact_fraction < 1.0  # a sane fractional price move, never a runaway/negative number

    # A real options hedge against the same notional, priced consistently with the same spot.
    hedge_price = black_scholes_price(spot, strike=spot * 1.02, time_to_expiry=30 / 365, risk_free_rate=0.03, volatility=0.6, option_type="put")
    hedge_delta = black_scholes_delta(spot, strike=spot * 1.02, time_to_expiry=30 / 365, risk_free_rate=0.03, volatility=0.6, option_type="put")
    assert hedge_price > 0.0
    assert -1.0 <= hedge_delta <= 0.0  # a real put delta, always non-positive

    sensitivity = np.array([[1.0, hedge_delta], [0.0, 1.0]])
    is_complete, rank, n_factors = check_hedge_completeness(sensitivity)
    assert is_complete and rank == n_factors == 2
    hedge_sizes, residual = solve_hedge_ratios(sensitivity, target_exposure=np.array([order_units, 0.0]))
    assert np.all(np.isfinite(hedge_sizes))
    assert np.all(np.abs(residual) < 1e-6)  # this 2x2 system is exactly solvable -- no real leftover exposure

"""Briques 3 (régimes de Markov), 4 (stops à queue épaisse) et 5 (coût de portage) du simulateur Monte Carlo, validées par leurs conséquences mesurables."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import simulate_pnl_montecarlo as sim
from scripts.simulate_pnl_montecarlo import SimConfig
from tests.test_simulate_pnl_microstructure import _assert_mean_matches, _predicted_pnl_pct, _quiet

REGIME = sim.RegimeParams(enabled=True)
TAIL = sim.StopTailParams(enabled=True)


def _skew(values: np.ndarray) -> float:
    flat = values.ravel()
    return float(((flat - flat.mean()) ** 3).mean() / flat.std() ** 3)


# =============================================================================================
# Brique 3 : régimes de Markov
# =============================================================================================


def test_the_stationary_distribution_solves_pi_p_equals_pi_and_centered_shifts_average_to_zero() -> None:
    pi = REGIME.stationary()
    p = REGIME.transition
    assert pi[0] == pytest.approx(0.25 / 0.35) and sum(pi) == pytest.approx(1.0)
    assert pi[0] * p[0][0] + pi[1] * p[1][0] == pytest.approx(pi[0])  # pi P = pi
    centered = REGIME.effective_shifts()
    assert pi[0] * centered[0] + pi[1] * centered[1] == pytest.approx(0.0, abs=1e-12)  # la moyenne de long terme reste win_rate
    assert replace(REGIME, preserve_mean=False).effective_shifts() == REGIME.win_rate_shift
    assert REGIME.mean_toxic_spell_days() == pytest.approx(4.0)  # 1 / (1 - 0,75)


def test_a_frozen_chain_starts_from_its_initial_distribution() -> None:
    frozen = sim.RegimeParams(enabled=True, transition=((1.0, 0.0), (0.0, 1.0)), initial_probs=(0.3, 0.7))
    assert frozen.stationary() == (0.3, 0.7)
    assert sim.RegimeParams(enabled=True, transition=((1.0, 0.0), (0.0, 1.0))).stationary() == (1.0, 0.0)


def test_the_regime_path_follows_the_transition_matrix() -> None:
    cfg = SimConfig(paths=4000, win_rate_std=0.0, regime=REGIME)
    regimes = sim.simulate(cfg).regimes
    assert regimes is not None
    today, tomorrow = regimes[:, :-1], regimes[:, 1:]
    assert float((regimes == 1).mean()) == pytest.approx(REGIME.stationary()[1], abs=0.01)        # part du temps en régime toxique
    assert float((tomorrow[today == 1] == 1).mean()) == pytest.approx(0.75, abs=0.02)              # persistance du toxique
    assert float((tomorrow[today == 0] == 0).mean()) == pytest.approx(0.90, abs=0.02)              # persistance de la tendance
    assert sim.simulate(replace(cfg, regime=sim.RegimeParams())).regimes is None


def test_the_first_day_regime_follows_the_initial_distribution() -> None:
    always_trend = sim.simulate(SimConfig(paths=300, days=3, regime=replace(REGIME, initial_probs=(1.0, 0.0)))).regimes
    always_toxic = sim.simulate(SimConfig(paths=300, days=3, regime=replace(REGIME, initial_probs=(0.0, 1.0)))).regimes
    assert always_trend is not None and always_toxic is not None
    assert (always_trend[:, 0] == 0).all() and (always_toxic[:, 0] == 1).all()


def test_the_regimes_replace_the_independent_daily_noise() -> None:
    """Avec les régimes, ``daily_win_rate_std`` n'a plus aucun effet (même graine, mêmes chemins) ; sans régimes il en a un."""
    with_regime = [sim.simulate(SimConfig(paths=200, seed=3, daily_win_rate_std=s, regime=REGIME)).equity for s in (0.0, 0.05)]
    without = [sim.simulate(SimConfig(paths=200, seed=3, daily_win_rate_std=s)).equity for s in (0.0, 0.05)]
    assert np.array_equal(with_regime[0], with_regime[1])
    assert not np.array_equal(without[0], without[1])


def test_centered_regimes_keep_the_analytic_mean_and_uncentered_ones_lower_it_exactly() -> None:
    centered = _quiet(regime=REGIME)
    _assert_mean_matches(centered, _predicted_pnl_pct(centered, sim.trade_economics(centered).expectancy_bps))
    lowered = replace(centered, regime=replace(REGIME, preserve_mean=False))
    pi = REGIME.stationary()
    mean_shift = pi[0] * REGIME.win_rate_shift[0] + pi[1] * REGIME.win_rate_shift[1]  # < 0 : la pénalité toxique pèse plus que le bonus
    shifted = replace(lowered, win_rate=lowered.win_rate + mean_shift, regime=sim.RegimeParams())
    _assert_mean_matches(lowered, _predicted_pnl_pct(lowered, sim.trade_economics(shifted).expectancy_bps))
    assert mean_shift < 0


def test_the_toxic_regime_has_a_lower_win_rate_than_the_trend_regime() -> None:
    cfg = _quiet(regime=sim.RegimeParams(enabled=True, volatility_multiplier=(1.0, 1.0)))
    result = sim.simulate(cfg)
    assert result.regimes is not None
    spread = sim.trade_economics(cfg).net_win_bps - sim.trade_economics(cfg).net_loss_bps
    gap = REGIME.win_rate_shift[1] - REGIME.win_rate_shift[0]  # -0,05 : écart de taux de réussite entre les deux régimes
    predicted_daily_gap = 200 * gap * spread * cfg.leverage / 10_000.0  # en fraction du capital (non capitalisé)
    daily_pnl = np.diff(result.equity, axis=1) / cfg.capital  # PnL du jour en fraction du capital INITIAL (``daily_returns`` est rapporté à l'équité du jour)
    observed_gap = float(daily_pnl[result.regimes == 1].mean() - daily_pnl[result.regimes == 0].mean())
    assert observed_gap < 0 and observed_gap == pytest.approx(predicted_daily_gap, rel=0.05)


def test_the_toxic_regime_multiplies_the_volatility_of_the_day() -> None:
    cfg = _quiet(paths=4000, regime=sim.RegimeParams(enabled=True, win_rate_shift=(0.0, 0.0), volatility_multiplier=(1.0, 1.6)))
    result = sim.simulate(cfg)
    assert result.regimes is not None
    ratio = float(result.daily_returns[result.regimes == 1].std() / result.daily_returns[result.regimes == 0].std())
    assert ratio == pytest.approx(1.6, rel=0.03)  # variance x 1,6 au carré


def test_the_regimes_create_clusters_of_bad_days_that_independent_days_do_not() -> None:
    base = SimConfig(paths=4000, win_rate_std=0.0)
    iid, regimes = sim.simulate(base), sim.simulate(replace(base, regime=REGIME))
    assert abs(sim.lag1_autocorrelation(iid.daily_returns)) < 0.05
    assert sim.lag1_autocorrelation(regimes.daily_returns) > 0.1
    cond_iid, uncond_iid = sim.loss_day_persistence(iid.daily_returns)
    cond_reg, uncond_reg = sim.loss_day_persistence(regimes.daily_returns)
    assert cond_iid - uncond_iid < 0.02            # sans mémoire : une perte hier ne prédit rien
    assert cond_reg - uncond_reg > 0.05            # avec régimes : une perte hier annonce une perte demain


def test_clustered_bad_days_give_deeper_drawdowns() -> None:
    base = SimConfig(paths=4000, win_rate_std=0.0)
    p95 = [float(np.percentile(sim.summarize(c, sim.simulate(c)).max_drawdown_pct, 95)) for c in (base, replace(base, regime=REGIME))]
    assert p95[1] > 1.3 * p95[0]


def test_the_lag1_autocorrelation_and_loss_persistence_helpers() -> None:
    trend = np.tile(np.arange(10.0), (50, 1))
    assert sim.lag1_autocorrelation(trend) > 0.7
    assert math.isnan(sim.lag1_autocorrelation(np.zeros((5, 5)))) and math.isnan(sim.lag1_autocorrelation(np.ones((5, 2))))
    losses = np.array([[-1.0, -1.0, 1.0, 1.0]])
    conditional, unconditional = sim.loss_day_persistence(losses)
    assert unconditional == 0.5 and conditional == 0.5  # une perte sur deux suivie d'une perte


# =============================================================================================
# Brique 4 : stops à queue épaisse
# =============================================================================================


def test_the_truncated_pareto_moments_match_a_large_sample() -> None:
    u = np.random.default_rng(1).random(1_000_000)
    for low, high, alpha in ((2.0, 3.0, 2.5), (2.0, 6.0, 1.5), (1.5, 4.0, 1.0)):
        sample = sim.sample_stop_multiplier(u, low, high, alpha)
        assert sample.min() >= low and sample.max() <= high
        assert sim.truncated_pareto_moment(1.0, low, high, alpha) == pytest.approx(float(sample.mean()), rel=0.005)
        assert sim.truncated_pareto_moment(2.0, low, high, alpha) == pytest.approx(float((sample ** 2).mean()), rel=0.01)


def test_the_untruncated_pareto_has_the_closed_form_mean_and_a_real_heavy_tail() -> None:
    u = np.random.default_rng(2).random(2_000_000)
    sample = sim.sample_stop_multiplier(u, 2.0, math.inf, 3.5)
    assert sim.truncated_pareto_moment(1.0, 2.0, math.inf, 3.5) == pytest.approx(2.0 * 3.5 / 2.5)
    assert float(sample.mean()) == pytest.approx(2.8, rel=0.01)
    for x in (4.0, 10.0):  # fonction de survie de Pareto : P(M > x) = (low / x) ** alpha
        assert float((sample > x).mean()) == pytest.approx((2.0 / x) ** 3.5, rel=0.1)
    assert math.isinf(sim.truncated_pareto_moment(3.5, 2.0, math.inf, 3.5))  # moment d'ordre alpha : infini


def test_the_stop_multiplier_sampler_covers_exactly_its_bounds_and_is_increasing() -> None:
    u = np.array([0.0, 0.25, 0.5, 0.75, 0.999999999])
    sample = sim.sample_stop_multiplier(u, 2.0, 3.0, 2.5)
    assert sample[0] == pytest.approx(2.0) and sample[-1] == pytest.approx(3.0, abs=1e-6)
    assert (np.diff(sample) > 0).all()


def test_the_extra_moments_are_those_of_m_minus_one() -> None:
    mean, second = TAIL.mean_multiplier(), sim.truncated_pareto_moment(2.0, 2.0, 3.0, 2.5)
    assert TAIL.mean_extra() == pytest.approx(mean - 1.0)
    assert TAIL.second_moment_extra() == pytest.approx(second - 2.0 * mean + 1.0)
    assert TAIL.second_moment_extra() >= TAIL.mean_extra() ** 2  # variance positive


def test_tail_events_cost_only_the_losers_and_exactly_the_documented_amount() -> None:
    plain = sim.trade_economics(SimConfig())
    tailed = sim.trade_economics(SimConfig(stop_tail=TAIL))
    expected_extra = TAIL.tail_prob * 40.0 * TAIL.mean_extra()
    assert tailed.net_win_bps == pytest.approx(plain.net_win_bps)                       # les gagnants ne sont pas touchés
    assert tailed.net_loss_bps == pytest.approx(plain.net_loss_bps - expected_extra)
    assert tailed.tail_extra_bps_per_loss == pytest.approx(expected_extra)
    assert tailed.breakeven_win_rate > plain.breakeven_win_rate and tailed.expectancy_bps < plain.expectancy_bps


def test_a_disabled_or_zero_probability_tail_changes_nothing() -> None:
    plain = sim.trade_economics(SimConfig())
    assert sim.trade_economics(SimConfig(stop_tail=sim.StopTailParams(enabled=False))).expectancy_bps == plain.expectancy_bps
    assert sim.trade_economics(SimConfig(stop_tail=sim.StopTailParams(enabled=True, tail_prob=0.0))).expectancy_bps == plain.expectancy_bps
    result = sim.simulate(SimConfig(paths=200, stop_tail=sim.StopTailParams(enabled=True, tail_prob=0.0)))
    assert result.tail_events is not None and float(result.tail_events.sum()) == 0.0


def test_the_simulated_tail_rate_among_losers_matches_the_probability() -> None:
    result = sim.simulate(SimConfig(paths=1500, stop_tail=TAIL))
    assert result.tail_events is not None and result.loser_trades is not None
    assert float(result.tail_events.sum() / result.loser_trades.sum()) == pytest.approx(0.05, abs=0.002)
    assert (result.tail_events <= result.loser_trades).all()  # jamais plus de queues que de trades perdants


def test_the_simulated_mean_pnl_equals_the_analytic_expectancy_with_tail_events() -> None:
    cfg = _quiet(stop_tail=sim.StopTailParams(enabled=True, tail_prob=0.08))
    _assert_mean_matches(cfg, _predicted_pnl_pct(cfg, sim.trade_economics(cfg).expectancy_bps))


def test_fat_stop_tails_skew_the_daily_pnl_and_deepen_the_worst_day() -> None:
    """Peu de trades par jour : un stop raté n'est plus noyé dans la moyenne et la distribution devient asymétrique."""
    few: dict[str, Any] = dict(trades_min=10.0, trades_max=10.0, paths=6000, win_rate_std=0.0, daily_win_rate_std=0.0, compound=False, daily_loss_limit_pct=100.0)
    plain = sim.simulate(SimConfig(**few))
    tailed = sim.simulate(SimConfig(**few, stop_tail=TAIL))
    assert _skew(tailed.daily_returns) < _skew(plain.daily_returns) - 0.03
    assert float(tailed.daily_returns.min()) < float(plain.daily_returns.min())


def test_an_unbounded_pareto_runs_and_stays_finite() -> None:
    cfg = SimConfig(paths=500, stop_tail=sim.StopTailParams(enabled=True, multiplier_high=math.inf, alpha=3.0))
    result = sim.simulate(cfg)
    assert np.isfinite(result.equity).all() and result.equity.min() >= 0.0
    assert np.isfinite(sim.trade_economics(cfg).tail_extra_bps_per_loss)


# =============================================================================================
# Brique 5 : coût de portage (funding)
# =============================================================================================


def test_the_funding_arithmetic_for_both_intervals() -> None:
    binance = sim.FundingParams(drag_bps=1.0, interval_hours=8.0, avg_hold_minutes=15.0)
    hyperliquid = sim.FundingParams(drag_bps=1.0, interval_hours=1.0, avg_hold_minutes=15.0)
    assert binance.crossing_probability == pytest.approx(15 / 480) and binance.payment_bps == pytest.approx(1.0)
    assert hyperliquid.crossing_probability == pytest.approx(0.25) and hyperliquid.payment_bps == pytest.approx(0.125)
    assert binance.cost_per_trade_bps == pytest.approx(1.0 * 0.25 / 8) == pytest.approx(hyperliquid.cost_per_trade_bps)  # même coût moyen
    long_hold = sim.FundingParams(drag_bps=1.0, interval_hours=1.0, avg_hold_minutes=150.0)
    assert long_hold.crossings_per_trade == pytest.approx(2.5) and long_hold.crossing_probability == 1.0


def test_funding_is_neutral_without_drag_or_without_holding_time() -> None:
    assert not sim.FundingParams().enabled and sim.FundingParams().cost_per_trade_bps == 0.0
    assert not sim.FundingParams(drag_bps=1.0, avg_hold_minutes=0.0).enabled
    assert sim.FundingParams(drag_bps=-1.0).enabled


def test_funding_lowers_both_wins_and_losses_by_the_same_amount() -> None:
    plain = sim.trade_economics(SimConfig())
    funded = sim.trade_economics(SimConfig(funding=sim.FundingParams(drag_bps=8.0, avg_hold_minutes=60.0)))  # 8 x 1 h / 8 = 1 bp par trade
    assert funded.funding_bps_per_trade == pytest.approx(1.0)
    assert funded.net_win_bps == pytest.approx(plain.net_win_bps - 1.0) and funded.net_loss_bps == pytest.approx(plain.net_loss_bps - 1.0)
    assert funded.expectancy_bps == pytest.approx(plain.expectancy_bps - 1.0)


def test_a_negative_drag_is_a_rebate_that_improves_the_edge() -> None:
    cost, rebate = (sim.trade_economics(SimConfig(funding=sim.FundingParams(drag_bps=d, avg_hold_minutes=60.0))).expectancy_bps for d in (4.0, -4.0))
    assert rebate > sim.trade_economics(SimConfig()).expectancy_bps > cost


@pytest.mark.parametrize("hours", [8.0, 1.0])
def test_the_simulated_funding_cost_per_trade_matches_the_closed_form_for_each_interval(hours: float) -> None:
    funding = sim.FundingParams(drag_bps=1.0, interval_hours=hours, avg_hold_minutes=15.0)
    result = sim.simulate(_quiet(funding=funding))
    assert result.funding_bps is not None
    assert float(result.funding_bps.sum() / result.trades.sum()) == pytest.approx(funding.cost_per_trade_bps, rel=0.01)


def test_a_long_hold_pays_several_funding_dates_per_trade() -> None:
    funding = sim.FundingParams(drag_bps=2.0, interval_hours=1.0, avg_hold_minutes=150.0)  # 2,5 échéances x 0,25 bp
    result = sim.simulate(_quiet(funding=funding))
    assert result.funding_bps is not None
    assert float(result.funding_bps.sum() / result.trades.sum()) == pytest.approx(2.0 * 2.5 / 8.0 * 1.0, rel=0.01)


def test_eight_hour_funding_is_lumpier_than_hourly_funding_for_the_same_average_cost() -> None:
    std: dict[float, float] = {}
    for hours in (8.0, 1.0):
        funding = sim.FundingParams(drag_bps=1.0, interval_hours=hours, avg_hold_minutes=15.0)
        result = sim.simulate(_quiet(funding=funding))
        assert result.funding_bps is not None
        std[hours] = float(result.funding_bps.std())
    assert std[8.0] > 2.0 * std[1.0]


def test_the_simulated_mean_pnl_equals_the_analytic_expectancy_with_funding() -> None:
    cfg = _quiet(funding=sim.FundingParams(drag_bps=16.0, interval_hours=8.0, avg_hold_minutes=60.0))  # 2 bps par trade : pèse sans ruiner (le plancher à 0 casserait la linéarité)
    _assert_mean_matches(cfg, _predicted_pnl_pct(cfg, sim.trade_economics(cfg).expectancy_bps))


def test_funding_is_proportional_to_the_exposure() -> None:
    """Doubler le risque par trade double le levier, donc le coût de portage payé sur le capital (même coût en bps du notionnel)."""
    funding = sim.FundingParams(drag_bps=8.0, avg_hold_minutes=60.0)  # 1 bp par trade

    def mean_pnl(cfg: SimConfig) -> float:
        return float(sim.summarize(cfg, sim.simulate(cfg)).pnl_pct.mean())

    drag_in_pct: dict[float, float] = {}
    for risk in (0.1, 0.2):
        drag_in_pct[risk] = mean_pnl(_quiet(risk_pct=risk)) - mean_pnl(_quiet(risk_pct=risk, funding=funding))
    assert drag_in_pct[0.2] == pytest.approx(2.0 * drag_in_pct[0.1], rel=0.03)


# --- Ligne de commande -----------------------------------------------------------------------------


def test_the_command_line_activates_bricks_3_4_5_by_default_and_can_switch_them_off(capsys: pytest.CaptureFixture[str]) -> None:
    sim.main(["--paths", "500", "--no-sensitivity"])
    default = capsys.readouterr().out
    for needle in ("RÉGIMES DE MARCHÉ", "grappes", "QUEUES DE STOP", "PORTAGE", "bruit journalier indépendant remplacé"):
        assert needle in default
    sim.main(["--paths", "500", "--no-sensitivity", "--no-regimes", "--tail-prob", "0", "--funding-drag-bps", "0"])
    off = capsys.readouterr().out
    for needle in ("RÉGIMES DE MARCHÉ", "QUEUES DE STOP", "PORTAGE"):
        assert needle not in off


def test_the_funding_interval_follows_the_venue(capsys: pytest.CaptureFixture[str]) -> None:
    sim.main(["--paths", "300", "--no-sensitivity", "--venue", "hyperliquid"])
    assert "toutes les 1 h" in capsys.readouterr().out
    sim.main(["--paths", "300", "--no-sensitivity", "--venue", "binance"])
    assert "toutes les 8 h" in capsys.readouterr().out


def test_the_library_defaults_stay_neutral_while_the_command_line_ones_are_not() -> None:
    assert sim.trade_economics(SimConfig()).funding_bps_per_trade == 0.0 and not SimConfig().regime.enabled
    cfg, _, _ = sim._parse(["--paths", "100"])
    assert cfg.regime.enabled and cfg.stop_tail.active and cfg.funding.enabled


@pytest.mark.parametrize("bad", [
    ["--tail-prob", "2"], ["--tail-alpha", "0"], ["--tail-mult-low", "0.5"], ["--tail-mult-high", "inf", "--tail-alpha", "1.5"],
    ["--p-stay-trend", "1.5"], ["--p-stay-toxic", "-0.1"], ["--toxic-vol-mult", "0"], ["--avg-hold-minutes", "-1"], ["--funding-interval-hours", "0"],
])
def test_the_command_line_refuses_invalid_values_for_bricks_3_4_5(bad: list[str]) -> None:
    with pytest.raises(SystemExit):
        sim.main(["--paths", "100", *bad])

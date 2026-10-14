"""Briques de microstructure du simulateur Monte Carlo : 1 (sélection adverse), 2 (plafond de liquidité et impact), et garde-fous des briques 3, 4, 5 (préparées, non implémentées)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import simulate_pnl_montecarlo as sim
from scripts.simulate_pnl_montecarlo import SimConfig
from tests.test_simulate_pnl_montecarlo import _cfg, _median_pnl


def _quiet(**over: Any) -> SimConfig:
    """Scénario sans bruit annexe : seuls les tirages de Poisson et binomiaux restent, donc l'espérance simulée doit égaler la formule analytique."""
    base: dict[str, Any] = dict(
        win_rate_std=0.0, daily_win_rate_std=0.0, payoff_dispersion=0.0, slippage_std_bps=0.0, compound=False, daily_loss_limit_pct=100.0,
        trades_min=200.0, trades_max=200.0, paths=4000, days=45,
    )
    base.update(over)
    return SimConfig(**base)


def _predicted_pnl_pct(cfg: SimConfig, expectancy_bps: float, leverage: float | None = None) -> float:
    """Capital non capitalisé : PnL attendu en % du capital = jours x signaux/jour x espérance par signal x levier."""
    lev = cfg.leverage if leverage is None else leverage
    return cfg.days * (cfg.trades_min + cfg.trades_max) / 2 * expectancy_bps * lev / 10_000.0 * 100.0


def _assert_mean_matches(cfg: SimConfig, expected_pct: float) -> None:
    pnl = sim.summarize(cfg, sim.simulate(cfg)).pnl_pct
    standard_error = float(pnl.std(ddof=1)) / math.sqrt(pnl.size)
    assert abs(float(pnl.mean()) - expected_pct) < 4 * standard_error, (float(pnl.mean()), expected_pct, standard_error)


# =============================================================================================
# Brique 1 : sélection adverse (maker)
# =============================================================================================


def test_the_unfilled_winner_probability_follows_the_documented_formula() -> None:
    legs = sim.leg_economics(_cfg())
    q = sim.unfilled_winner_probability(2.0, 0.56, legs.net_win_maker_entry)
    assert q == pytest.approx(2.0 / (0.56 * legs.net_win_maker_entry))
    assert sim.unfilled_winner_probability(0.0, 0.56, legs.net_win_maker_entry) == 0.0  # désactivée
    assert sim.unfilled_winner_probability(1e9, 0.56, legs.net_win_maker_entry) == 1.0  # bornée : on ne peut pas manquer plus de 100 % des gagnants
    as_array = sim.unfilled_winner_probability(2.0, np.array([0.5, 0.6]), legs.net_win_maker_entry)
    assert isinstance(as_array, np.ndarray) and as_array[0] > as_array[1]  # un taux de réussite plus bas manque proportionnellement plus de gagnants


def test_adverse_selection_lowers_the_expectancy_by_exactly_the_maker_share_of_the_bps() -> None:
    plain = sim.trade_economics(_cfg())
    adverse = sim.trade_economics(_cfg(adverse=sim.AdverseSelectionParams(bps=2.0)))
    m = _cfg().entry_maker_fraction
    assert adverse.expectancy_bps == pytest.approx(plain.expectancy_bps - m * 2.0)
    assert adverse.breakeven_win_rate == pytest.approx((-plain.net_loss_bps + m * 2.0) / (plain.net_win_bps - plain.net_loss_bps))
    assert adverse.breakeven_win_rate > plain.breakeven_win_rate  # il faut gagner plus souvent pour compenser
    assert 0 < adverse.unfilled_winner_prob < 1 and adverse.executed_fraction < 1 and adverse.effective_win_rate < _cfg().win_rate


def test_adverse_selection_is_off_by_default_and_changes_nothing_then() -> None:
    econ = sim.trade_economics(_cfg())
    assert econ.unfilled_winner_prob == 0.0 and econ.executed_fraction == 1.0
    assert econ.effective_win_rate == pytest.approx(0.56)
    result = sim.simulate(_cfg(paths=300))
    assert result.unfilled is not None and float(result.unfilled.sum()) == 0.0


def test_executed_trades_plus_unfilled_winners_always_equal_the_signals() -> None:
    result = sim.simulate(_cfg(paths=400, adverse=sim.AdverseSelectionParams(bps=3.0)))
    assert result.signals is not None and result.unfilled is not None
    assert float(result.unfilled.sum()) > 0
    assert np.allclose(result.trades + result.unfilled, result.signals)


def test_the_simulated_unfilled_share_matches_maker_share_times_the_profit_lost() -> None:
    cfg = _quiet(adverse=sim.AdverseSelectionParams(bps=2.0), paths=1500)
    legs = sim.leg_economics(cfg)
    result = sim.simulate(cfg)
    assert result.signals is not None and result.unfilled is not None
    # part de signaux non exécutés = m x p x q = m x bps / gain net maker
    expected = cfg.entry_maker_fraction * 2.0 / legs.net_win_maker_entry
    assert float(result.unfilled.sum() / result.signals.sum()) == pytest.approx(expected, abs=0.002)


def test_taker_entries_are_never_unfilled() -> None:
    cfg = _cfg(paths=300, entry_maker_fraction=0.0, adverse=sim.AdverseSelectionParams(bps=5.0))
    result = sim.simulate(cfg)
    assert result.unfilled is not None and float(result.unfilled.sum()) == 0.0
    assert sim.trade_economics(cfg).expectancy_bps == pytest.approx(sim.trade_economics(_cfg(entry_maker_fraction=0.0)).expectancy_bps)


def test_the_simulated_mean_pnl_equals_the_analytic_expectancy_with_and_without_adverse_selection() -> None:
    for bps in (0.0, 2.0, 4.0):
        cfg = _quiet(adverse=sim.AdverseSelectionParams(bps=bps))
        _assert_mean_matches(cfg, _predicted_pnl_pct(cfg, sim.trade_economics(cfg).expectancy_bps))


def test_at_the_adverse_selection_breakeven_win_rate_the_mean_pnl_is_zero() -> None:
    base = _quiet(adverse=sim.AdverseSelectionParams(bps=3.0))
    cfg = replace(base, win_rate=sim.trade_economics(base).breakeven_win_rate)
    _assert_mean_matches(cfg, 0.0)


def test_more_adverse_selection_means_a_lower_median_pnl() -> None:
    medians = [_median_pnl(_cfg(win_rate=0.58, adverse=sim.AdverseSelectionParams(bps=b))) for b in (0.0, 2.0, 6.0)]
    assert medians[0] > medians[1] > medians[2]


def test_a_negative_adverse_selection_is_refused() -> None:
    with pytest.raises(ValueError, match="adverse_selection_bps"):
        sim.AdverseSelectionParams(bps=-1.0)


# =============================================================================================
# Brique 2 : plafond de liquidité et impact en racine carrée
# =============================================================================================


def test_the_square_root_impact_law() -> None:
    assert sim.impact_bps(1_000_000.0, 10_000_000_000.0, 200.0, 1.0) == pytest.approx(200.0 * math.sqrt(1e-4))  # 2 bps
    quadrupled = float(sim.impact_bps(4_000_000.0, 10_000_000_000.0, 200.0, 1.0))
    assert quadrupled == pytest.approx(2.0 * 2.0)  # quatre fois la taille : deux fois l'impact
    assert float(sim.impact_bps(0.0, 1e10, 200.0, 1.0)) == 0.0
    as_array = sim.impact_bps(np.array([1e6, 4e6]), 1e10, 200.0, 0.5)
    assert isinstance(as_array, np.ndarray) and as_array[1] == pytest.approx(2 * as_array[0])


def test_the_notional_cap_is_never_exceeded_even_when_the_capital_compounds() -> None:
    cfg = _cfg(win_rate=0.6, win_rate_std=0.0, paths=500, liquidity=sim.LiquidityParams(max_notional_usd=3_000.0))
    result = sim.simulate(cfg)
    assert result.leverage_used is not None
    assert (result.leverage_used * result.equity[:, :-1] <= 3_000.0 * (1 + 1e-9)).all()  # notionnel d'un trade = levier x équité du jour <= plafond


def test_a_cap_that_never_binds_leaves_the_leverage_untouched() -> None:
    result = sim.simulate(_cfg(paths=300, liquidity=sim.LiquidityParams(max_notional_usd=1e12)))
    assert result.leverage_used is not None and np.allclose(result.leverage_used, _cfg().leverage)


def test_the_cap_stops_the_capital_from_compounding_forever() -> None:
    edge: dict[str, Any] = dict(win_rate=0.62, win_rate_std=0.0, daily_win_rate_std=0.0)
    free = _median_pnl(_cfg(**edge))
    capped = _median_pnl(_cfg(**edge, liquidity=sim.LiquidityParams(max_notional_usd=3_000.0)))
    assert capped < free / 2  # la croissance exponentielle devient quasi linéaire


def test_the_cap_also_limits_losses_when_the_edge_is_negative() -> None:
    bad: dict[str, Any] = dict(win_rate=0.50, win_rate_std=0.0)
    assert _median_pnl(_cfg(**bad, liquidity=sim.LiquidityParams(max_notional_usd=1_000.0))) > _median_pnl(_cfg(**bad))


def test_the_capped_mean_pnl_scales_with_the_effective_leverage() -> None:
    capital, cap = 10_000.0, 1_000.0  # levier visé 0,25 -> 2 500 $ par trade ; plafonné à 1 000 $ -> levier 0,10
    cfg = _quiet(capital=capital, liquidity=sim.LiquidityParams(max_notional_usd=cap))
    _assert_mean_matches(cfg, _predicted_pnl_pct(cfg, sim.trade_economics(cfg).expectancy_bps, leverage=cap / capital))


def test_there_is_no_impact_without_a_volume_figure() -> None:
    result = sim.simulate(_cfg(paths=200))
    assert result.impact_bps_per_fill is not None and float(result.impact_bps_per_fill.max()) == 0.0
    assert not sim.LiquidityParams(adv_usd=None, impact_coef=1.0).impact_enabled
    assert not sim.LiquidityParams(adv_usd=1e9, impact_coef=0.0).impact_enabled


def test_the_first_day_impact_grows_with_the_square_root_of_the_capital() -> None:
    def first_day_impact(capital: float) -> float:
        cfg = _cfg(capital=capital, paths=50, days=2, liquidity=sim.LiquidityParams(adv_usd=1e10, daily_vol_bps=200.0, impact_coef=1.0))
        result = sim.simulate(cfg)
        assert result.impact_bps_per_fill is not None
        return float(result.impact_bps_per_fill[0, 0])

    assert first_day_impact(1_000_000.0) / first_day_impact(10_000.0) == pytest.approx(10.0)  # capital x100 -> impact x10


def test_impact_is_paid_on_taker_fills_only_and_matches_the_analytic_expectancy() -> None:
    adv, vol, coef = 2e10, 300.0, 1.0  # impact de l'ordre de 1,5 bps : assez pour peser, pas assez pour ruiner (le plancher à 0 de l'équité casserait la linéarité)
    cfg = _quiet(capital=2_000_000.0, liquidity=sim.LiquidityParams(adv_usd=adv, daily_vol_bps=vol, impact_coef=coef))
    legs = sim.leg_economics(cfg)
    i = float(sim.impact_bps(cfg.leverage * cfg.capital, adv, vol, coef))
    m, p, x = cfg.entry_maker_fraction, cfg.win_rate, legs.exit_taker_share_win
    # entrée taker : une exécution taker d'impact plus la sortie ; entrée maker : seule la sortie taker paie ; un stop est toujours taker
    win = m * (legs.net_win_maker_entry - i * x) + (1 - m) * (legs.net_win_taker_entry - i * (1 + x))
    loss = m * (legs.net_loss_maker_entry - i) + (1 - m) * (legs.net_loss_taker_entry - 2 * i)
    with_impact = p * win + (1 - p) * loss
    _assert_mean_matches(cfg, _predicted_pnl_pct(cfg, with_impact))
    no_impact = replace(cfg, liquidity=sim.LiquidityParams())
    assert with_impact < sim.trade_economics(no_impact).expectancy_bps  # l'impact coûte


def test_a_large_account_in_a_thin_market_loses_its_edge_to_impact() -> None:
    thin = _cfg(capital=50_000_000.0, liquidity=sim.LiquidityParams(adv_usd=2e9, daily_vol_bps=300.0, impact_coef=1.0))
    assert _median_pnl(thin) < _median_pnl(replace(thin, liquidity=sim.LiquidityParams()))


@pytest.mark.parametrize("kwargs", [{"max_notional_usd": 0.0}, {"max_notional_usd": -1.0}, {"adv_usd": 0.0}, {"daily_vol_bps": 0.0}, {"impact_coef": -0.1}])
def test_invalid_liquidity_parameters_are_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        sim.LiquidityParams(**kwargs)


# =============================================================================================
# Paramètres des briques 3, 4, 5 : neutres par défaut dans la bibliothèque, validés
# =============================================================================================


def test_the_bricks_3_4_5_are_neutral_by_default_in_the_library() -> None:
    cfg = SimConfig()
    assert not (cfg.regime.enabled or cfg.stop_tail.active or cfg.funding.enabled)
    assert cfg.funding.drag_bps == 0.0 and cfg.funding.cost_per_trade_bps == 0.0


@pytest.mark.parametrize("build", [
    lambda: sim.RegimeParams(transition=((0.5, 0.4), (0.2, 0.8))),
    lambda: sim.RegimeParams(initial_probs=(0.9, 0.2)),
    lambda: sim.RegimeParams(volatility_multiplier=(1.0, 0.0)),
    lambda: sim.RegimeParams(win_rate_shift=(0.0, -0.6)),
    lambda: sim.StopTailParams(tail_prob=1.5),
    lambda: sim.StopTailParams(multiplier_low=3.0, multiplier_high=2.0),
    lambda: sim.StopTailParams(alpha=0.0),
    lambda: sim.StopTailParams(multiplier_high=math.inf, alpha=1.5),
    lambda: sim.FundingParams(interval_hours=0.0),
    lambda: sim.FundingParams(avg_hold_minutes=-1.0),
])
def test_the_parameters_of_bricks_3_4_5_are_validated(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


# --- Ligne de commande -----------------------------------------------------------------------------


def test_the_new_flags_activate_their_report_sections(capsys: pytest.CaptureFixture[str]) -> None:
    sim.main(["--paths", "500", "--no-sensitivity"])
    plain = capsys.readouterr().out
    assert "SÉLECTION ADVERSE" not in plain and "LIQUIDITÉ" not in plain  # rien d'affiché tant que les briques sont inactives
    sim.main(["--paths", "500", "--no-sensitivity", "--adverse-selection-bps", "2", "--max-notional", "50000", "--adv-usd", "5e9"])
    out = capsys.readouterr().out
    for needle in ("SÉLECTION ADVERSE", "non exécutés", "LIQUIDITÉ", "plafond actif", "impact racine carrée"):
        assert needle in out


def test_the_command_line_refuses_invalid_microstructure_values() -> None:
    for bad in (["--adverse-selection-bps", "-1"], ["--max-notional", "0"], ["--adv-usd", "-5"], ["--impact-coef", "-1"]):
        with pytest.raises(SystemExit):
            sim.main(["--paths", "100", *bad])

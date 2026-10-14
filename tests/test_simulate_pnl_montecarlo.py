"""Tests du simulateur Monte Carlo (``scripts/simulate_pnl_montecarlo.py``) : économie d'un trade calculée à la main, propriétés statistiques, échecs."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts import simulate_pnl_montecarlo as sim
from scripts.simulate_pnl_montecarlo import SimConfig, SimResult


def _cfg(**over: Any) -> SimConfig:
    base: dict[str, Any] = dict(paths=3000, days=45)
    base.update(over)
    return SimConfig(**base)


def _median_pnl(cfg: SimConfig) -> float:
    return float(np.median(sim.summarize(cfg, sim.simulate(cfg)).pnl_pct))


# --- Économie d'un trade (calcul à la main) ---------------------------------------------------


def test_trade_economics_match_the_hand_calculation() -> None:
    """Binance 2/5 bps ; entrée maker 50 % ; objectif maker 70 % ; glissement 1 bps par exécution taker ; stop 40, objectif 48."""
    econ = sim.trade_economics(_cfg())
    # frais d'entrée 0,5x2 + 0,5x5 = 3,5 ; sortie gagnante 0,7x2 + 0,3x5 = 2,9 ; glissement 1x(0,5 + 0,3) = 0,8 -> 7,2
    assert econ.cost_win_bps == pytest.approx(7.2) and econ.net_win_bps == pytest.approx(48 - 7.2)
    # perdant : 3,5 + 5 (stop au marché) + 1x(0,5 + 1) = 10,0 -> -(40 + 10)
    assert econ.cost_loss_bps == pytest.approx(10.0) and econ.net_loss_bps == pytest.approx(-50.0)
    assert econ.breakeven_win_rate == pytest.approx(50.0 / (40.8 + 50.0))
    assert econ.expectancy_bps == pytest.approx(0.56 * 40.8 - 0.44 * 50.0)


def test_at_the_breakeven_win_rate_the_expectancy_is_zero() -> None:
    cfg = _cfg()
    p = sim.trade_economics(cfg).breakeven_win_rate
    assert sim.trade_economics(replace(cfg, win_rate=p)).expectancy_bps == pytest.approx(0.0, abs=1e-9)


def test_cheaper_venues_have_a_better_expectancy() -> None:
    values = {v: sim.trade_economics(_cfg(maker_bps=m, taker_bps=t)).expectancy_bps for v, (m, t) in sim.VENUE_FEES_BPS.items()}
    assert values["hyperliquid"] > values["binance"] > values["bybit"]


def test_leverage_is_the_risk_over_the_stop_and_is_capped() -> None:
    assert _cfg(risk_pct=0.1, stop_bps=40.0).leverage == pytest.approx(0.25)
    assert _cfg(risk_pct=5.0, stop_bps=10.0, max_leverage=3.0).leverage == 3.0


# --- Propriétés statistiques --------------------------------------------------------------------


def test_the_same_seed_gives_the_same_result_and_another_seed_a_different_one() -> None:
    a, b, c = (sim.simulate(_cfg(paths=500, seed=s)).equity for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_without_edge_and_without_costs_the_mean_pnl_is_zero_within_noise() -> None:
    cfg = _cfg(win_rate=0.5, win_rate_std=0.0, daily_win_rate_std=0.0, rr=1.0, maker_bps=0.0, taker_bps=0.0, slippage_bps=0.0, compound=False, paths=6000)
    pnl = sim.summarize(cfg, sim.simulate(cfg)).pnl_pct
    assert abs(float(pnl.mean())) < 4 * float(pnl.std(ddof=1)) / math.sqrt(pnl.size)


def test_at_the_breakeven_win_rate_the_mean_pnl_is_zero_within_noise() -> None:
    base = _cfg(win_rate_std=0.0, daily_win_rate_std=0.0, compound=False, paths=6000)
    cfg = replace(base, win_rate=sim.trade_economics(base).breakeven_win_rate)
    pnl = sim.summarize(cfg, sim.simulate(cfg)).pnl_pct
    assert abs(float(pnl.mean())) < 4 * float(pnl.std(ddof=1)) / math.sqrt(pnl.size)


def test_costs_destroy_a_thin_edge() -> None:
    assert _median_pnl(_cfg(maker_bps=0.0, taker_bps=0.0, slippage_bps=0.0)) > _median_pnl(_cfg())


def test_a_higher_win_rate_gives_a_higher_median_pnl() -> None:
    medians = [_median_pnl(_cfg(win_rate=p)) for p in (0.50, 0.54, 0.58)]
    assert medians == sorted(medians) and medians[0] < 0 < medians[-1]


def test_more_trades_per_day_amplify_a_negative_expectancy() -> None:
    few = _cfg(win_rate=0.50, trades_min=50.0, trades_max=50.0)
    many = _cfg(win_rate=0.50, trades_min=500.0, trades_max=500.0)
    assert sim.trade_economics(few).expectancy_bps < 0
    assert _median_pnl(many) < _median_pnl(few)


def test_the_daily_loss_limit_caps_every_day() -> None:
    cfg = _cfg(win_rate=0.05, win_rate_std=0.0, risk_pct=1.0, daily_loss_limit_pct=2.0, paths=500)
    assert sim.simulate(cfg).daily_returns.min() >= -0.02 - 1e-12  # aucune journée ne perd plus que le coupe-circuit


def test_equity_never_goes_negative() -> None:
    cfg = _cfg(win_rate=0.02, win_rate_std=0.0, risk_pct=10.0, stop_bps=20.0, daily_loss_limit_pct=100.0, max_leverage=50.0, paths=500)
    assert sim.simulate(cfg).equity.min() >= 0.0


def test_compounding_amplifies_a_positive_edge() -> None:
    assert _median_pnl(_cfg(win_rate=0.58)) > _median_pnl(_cfg(win_rate=0.58, compound=False))


# --- Statistiques ---------------------------------------------------------------------------------


def test_max_drawdown_is_measured_from_the_running_peak() -> None:
    cfg = _cfg(paths=2, days=3, capital=100.0)
    equity = np.array([[100.0, 120.0, 90.0, 110.0], [100.0, 101.0, 102.0, 103.0]])
    summary = sim.summarize(cfg, SimResult(equity, np.zeros((2, 3)), np.zeros((2, 3))))
    assert summary.max_drawdown_pct == pytest.approx([25.0, 0.0])  # (120 - 90) / 120
    assert summary.pnl_pct == pytest.approx([10.0, 3.0])


def test_the_sharpe_ratio_is_undefined_without_variance_and_annualized_otherwise() -> None:
    cfg = _cfg(paths=2, days=4)
    flat = SimResult(np.full((2, 5), 100.0), np.zeros((2, 4)), np.zeros((2, 4)))
    assert np.isnan(sim.summarize(cfg, flat).sharpe).all()
    returns = np.array([[0.01, 0.03, 0.01, 0.03], [0.02, 0.02, 0.02, 0.04]])
    sharpe = sim.summarize(cfg, SimResult(np.full((2, 5), 100.0), returns, np.zeros((2, 4)))).sharpe
    assert sharpe[0] == pytest.approx(returns[0].mean() / returns[0].std(ddof=1) * math.sqrt(365))


def test_the_mean_confidence_interval_contains_the_mean_and_needs_two_values() -> None:
    lo, hi = sim.mean_confidence_interval(np.array([1.0, 2.0, 3.0, 4.0]))
    assert lo < 2.5 < hi
    assert math.isnan(sim.mean_confidence_interval(np.array([1.0]))[0])


# --- Configuration et ligne de commande -----------------------------------------------------------


@pytest.mark.parametrize("over", [
    {"capital": 0.0}, {"days": 0}, {"paths": 1}, {"trades_min": 0.0}, {"trades_min": 10.0, "trades_max": 5.0}, {"win_rate": 0.0}, {"win_rate": 1.0},
    {"win_rate_std": -0.1}, {"rr": 0.0}, {"stop_bps": 0.0}, {"risk_pct": 0.0}, {"daily_loss_limit_pct": 0.0}, {"maker_bps": -1.0}, {"entry_maker_fraction": 1.5},
])
def test_invalid_configurations_are_refused(over: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="configuration invalide"):
        SimConfig(**over)


def test_the_report_contains_every_requested_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert sim.main(["--paths", "800", "--no-sensitivity"]) == 0
    out = capsys.readouterr().out
    for needle in ("pire scénario", "scénario moyen", "meilleur scénario", "IC95 %", "drawdown maximal", "Sharpe annualisé", "RENTABILITÉ", "Rappel"):
        assert needle in out


def test_a_negative_expectancy_is_flagged(capsys: pytest.CaptureFixture[str]) -> None:
    sim.main(["--paths", "500", "--win-rate", "0.50", "--no-sensitivity"])
    assert "NÉGATIVE" in capsys.readouterr().out


def test_the_command_line_refuses_invalid_values() -> None:
    with pytest.raises(SystemExit):
        sim.main(["--win-rate", "1.5"])
    with pytest.raises(SystemExit):
        sim.main(["--venue", "inconnue"])


def test_the_sensitivity_table_has_one_row_per_win_rate() -> None:
    table = sim.sensitivity_table(_cfg(), (0.5, 0.6), (40.0, 80.0), paths=300)
    assert len(table.splitlines()) == 4 and "50%" in table and "60%" in table

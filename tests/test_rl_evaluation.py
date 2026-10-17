"""Évaluation hors échantillon : chiffres exacts sur des cas calculés à la main, frais comptés une fois par changement de position, masque du coupe-circuit respecté."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.rl import evaluation as ev
from src.rl.networks import ActorCritic

BAR_S = 60.0


def _prices(log_returns: list[float], start: float = 100.0) -> np.ndarray:
    return start * np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))


def test_metrics_on_a_hand_computed_series() -> None:
    returns = np.array([0.01, -0.02, 0.01, 0.01])
    positions = np.array([1.0, 1.0, 0.0, 1.0])
    m = ev.metrics(returns, positions, bar_s=BAR_S)
    assert m.n_steps == 4 and m.total_log_return == pytest.approx(0.01)
    assert m.n_trades == 3 and m.exposure == pytest.approx(0.75)          # 0->1, 1->0, 0->1
    assert m.hit_rate == pytest.approx(2 / 3)                              # 2 pas gagnants sur les 3 en position
    equity = np.exp(np.cumsum(returns))
    assert m.max_drawdown == pytest.approx(float(np.min(equity / np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:] - 1.0)))
    assert m.max_drawdown < 0 and m.sharpe is not None and m.sharpe_std_error is not None


def test_a_constant_return_has_no_defined_sharpe_and_empty_input_is_refused() -> None:
    m = ev.metrics(np.zeros(5), np.zeros(5), bar_s=BAR_S)
    assert m.sharpe is None and m.hit_rate is None and m.n_trades == 0 and m.max_drawdown == 0.0
    with pytest.raises(ValueError):
        ev.metrics(np.empty(0), np.empty(0), bar_s=BAR_S)
    with pytest.raises(ValueError):
        ev.metrics(np.zeros(3), np.zeros(2), bar_s=BAR_S)


def test_costs_are_charged_once_per_position_change_in_log_return_units() -> None:
    prices = _prices([0.001] * 9)  # 10 prix, 9 pas
    flat = ev.run_positions(np.zeros(9), prices, transaction_cost_pct=0.0005, bar_s=BAR_S)
    hold = ev.run_positions(np.ones(9), prices, transaction_cost_pct=0.0005, bar_s=BAR_S)
    assert flat.total_log_return == 0.0 and flat.n_trades == 0
    assert hold.total_log_return == pytest.approx(9 * 0.001 - 0.0005) and hold.n_trades == 1  # une seule entrée, aucune sortie
    flip = ev.run_positions(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float), prices, transaction_cost_pct=0.0005, bar_s=BAR_S)
    assert flip.n_trades == 9 and flip.total_log_return == pytest.approx(5 * 0.001 - 9 * 0.0005)
    with pytest.raises(ValueError):
        ev.run_positions(np.ones(5), prices, transaction_cost_pct=0.0005, bar_s=BAR_S)


def test_baselines_include_flat_hold_and_the_vpin_rule_decided_on_the_known_vpin() -> None:
    prices = _prices([0.001, -0.002, 0.001, 0.001, -0.001])
    vpin = np.array([0.2, 0.9, 0.2, 0.2, 0.9, 0.9])
    out = ev.baselines(prices, vpin, transaction_cost_pct=0.0, bar_s=BAR_S, vpin_threshold=0.7)
    assert set(out) == {"à plat", "acheter et garder", "long si VPIN <= 0.7"}
    # règle : long aux pas 0, 2, 3 (VPIN <= 0,7) -> +0,001 + 0,001 + 0,001 ; le pas 1 (VPIN haut, -0,002) et le pas 4 sont évités
    assert out["long si VPIN <= 0.7"].total_log_return == pytest.approx(0.003)
    assert out["acheter et garder"].total_log_return == pytest.approx(0.0)
    assert out["à plat"].total_log_return == 0.0


def _biased_model(action: int) -> ActorCritic:
    model = ActorCritic(observation_dim=9, action_dim=3)
    head = model.actor_head[0]
    assert isinstance(head, torch.nn.Linear)
    head.weight.data.zero_()
    head.bias.data = torch.tensor([8.0 if i == action else 0.0 for i in range(3)])
    return model


def _data(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    prices = _prices(list(rng.normal(0.0002, 0.001, n - 1)))
    features = rng.normal(size=(n, 5))
    vpin = np.column_stack([np.full(n, 0.3), np.zeros(n), np.zeros(n)])
    return features, prices, vpin


def test_a_policy_that_always_buys_matches_buy_and_hold() -> None:
    features, prices, vpin = _data()
    m = ev.evaluate_policy(_biased_model(1), features, prices, vpin, transaction_cost_pct=0.0005, bar_s=BAR_S)
    hold = ev.run_positions(np.ones(len(prices) - 1), prices, transaction_cost_pct=0.0005, bar_s=BAR_S)
    assert m.total_log_return == pytest.approx(hold.total_log_return) and m.n_trades == 1 and m.exposure == pytest.approx(1.0)


def test_the_kill_switch_mask_is_honoured_during_evaluation() -> None:
    features, prices, _ = _data()
    toxic = np.column_stack([np.full(len(prices), 0.95), np.zeros(len(prices)), np.zeros(len(prices))])
    m = ev.evaluate_policy(_biased_model(1), features, prices, toxic, transaction_cost_pct=0.0005, bar_s=BAR_S, kill_switch_vpin=0.9)
    assert m.n_trades == 0 and m.exposure == 0.0  # une politique qui veut acheter n'ouvre jamais quand le VPIN dépasse le seuil


def test_the_table_prints_every_strategy_with_the_sharpe_error() -> None:
    features, prices, vpin = _data()
    results = {"politique": ev.evaluate_policy(_biased_model(1), features, prices, vpin, transaction_cost_pct=0.0005, bar_s=BAR_S)}
    results.update(ev.baselines(prices, vpin[:, 0], transaction_cost_pct=0.0005, bar_s=BAR_S))
    table = ev.format_table(results)
    assert "politique" in table and "acheter et garder" in table and "±" in table and table.count("\n") == 4

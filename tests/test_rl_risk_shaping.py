"""Aversion au risque du PPO, niveau 1 : fonctions pures (``src/rl/risk_shaping.py``) et environnement (``src/rl/trading_env.py``)."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from src.rl import risk_shaping as rs
from src.rl.risk_shaping import RiskShapingConfig
from src.rl.trading_env import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    OBSERVATION_DIM,
    OBSERVATION_DIM_WITH_VPIN,
    MarketEnv,
)

COST = 0.0005  # transaction_cost_pct par défaut de l'environnement


# --- Configuration ----------------------------------------------------------------------------


@pytest.mark.parametrize("over", [
    {"vpin_threshold": -0.1}, {"vpin_threshold": 1.1}, {"toxicity_penalty_weight": -1.0}, {"tail_risk_weight": -1.0}, {"safe_harbor_bonus": -0.1},
    {"stop_loss_bps": 0.0}, {"tail_alpha": 0.9}, {"kill_switch_vpin": 0.0}, {"kill_switch_vpin": 1.5}, {"tail_risk_weight": 1.0},  # un poids sur les stops sans stop
])
def test_invalid_shaping_configurations_are_refused(over: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RiskShapingConfig(**over)


def test_the_default_configuration_is_neutral_and_does_not_need_the_vpin() -> None:
    config = RiskShapingConfig()
    assert not config.needs_vpin
    assert RiskShapingConfig(toxicity_penalty_weight=0.1).needs_vpin
    assert RiskShapingConfig(safe_harbor_bonus=0.1).needs_vpin
    assert RiskShapingConfig(kill_switch_vpin=0.9).needs_vpin
    assert not RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0).needs_vpin


# --- Taxe de toxicité ---------------------------------------------------------------------------


def test_the_toxicity_tax_uses_the_excess_over_the_threshold_and_the_position_size() -> None:
    config = RiskShapingConfig(vpin_threshold=0.7, toxicity_penalty_weight=0.5)
    assert rs.toxicity_tax(0.9, 1.0, config) == pytest.approx(0.5 * 0.2)
    assert rs.toxicity_tax(0.9, 0.5, config) == pytest.approx(0.5 * 0.2 * 0.5)  # proportionnelle à |position|
    assert rs.toxicity_tax(0.9, -1.0, config) == pytest.approx(0.1)               # |position| : valable aussi en court
    assert rs.toxicity_tax(0.9, 0.0, config) == 0.0                                # à plat : rien à payer
    assert rs.toxicity_tax(0.7, 1.0, config) == 0.0 and rs.toxicity_tax(0.3, 1.0, config) == 0.0  # sous le seuil : jamais taxé (pas de taxe permanente en marché sain)


# --- Pénalité de stop -----------------------------------------------------------------------------


def test_the_stop_penalty_is_zero_at_the_nominal_stop_and_grows_faster_than_the_loss() -> None:
    config = RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0, tail_alpha=1.5)
    assert rs.stop_penalty(40.0, config) == pytest.approx(0.0)                                   # au stop nominal : pas de pénalité en plus
    assert rs.stop_penalty(80.0, config) == pytest.approx(80.0 * (math.sqrt(2.0) - 1.0) / 10_000)  # x1,41 à deux fois le stop
    values = [rs.stop_penalty(loss, config) for loss in (40.0, 60.0, 80.0, 120.0)]
    assert values == sorted(values)
    increments = np.diff(values) / np.diff([40.0, 60.0, 80.0, 120.0])
    assert (np.diff(increments) > 0).all()                                                       # convexe : un gap pèse plus que proportionnellement


def test_the_raw_power_form_would_soften_the_loss_in_log_return_units() -> None:
    """Pourquoi la forme est normalisée : ``|perte|^1,5`` en rendement logarithmique (0,004) est PLUS PETIT que la perte elle-même."""
    loss = 0.004
    assert loss ** 1.5 < loss
    assert rs.stop_penalty(40.0, RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0)) == 0.0  # la forme normalisée, elle, ne l'adoucit jamais
    assert rs.stop_penalty(100.0, RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0)) > 0.0


def test_the_stop_penalty_scales_with_its_weight_and_is_off_without_a_stop() -> None:
    base = RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0)
    assert rs.stop_penalty(80.0, replace(base, tail_risk_weight=3.0)) == pytest.approx(3.0 * rs.stop_penalty(80.0, base))
    assert rs.stop_penalty(80.0, RiskShapingConfig()) == 0.0
    assert rs.stop_penalty(80.0, RiskShapingConfig(stop_loss_bps=40.0)) == 0.0   # poids nul
    assert rs.stop_penalty(80.0, replace(base, tail_alpha=1.0)) == pytest.approx(0.0)  # alpha = 1 : linéaire, rien en plus


# --- Bonus de refuge ----------------------------------------------------------------------------


def test_the_safe_harbor_bonus_only_rewards_being_flat_in_a_toxic_market() -> None:
    config = RiskShapingConfig(vpin_threshold=0.7, safe_harbor_bonus=0.01)
    assert rs.safe_harbor_reward(ACTION_SELL, 0, 0.9, config) == 0.01
    assert rs.safe_harbor_reward(ACTION_HOLD, 0, 0.9, config) == 0.01
    assert rs.safe_harbor_reward(ACTION_HOLD, 0, 0.5, config) == 0.0     # marché sain : rester à plat n'est pas récompensé
    assert rs.safe_harbor_reward(ACTION_HOLD, 1, 0.9, config) == 0.0     # en position : aucun bonus
    assert rs.safe_harbor_reward(ACTION_HOLD, 0, 0.9, RiskShapingConfig()) == 0.0


# --- Masque d'actions -------------------------------------------------------------------------------


def test_the_action_mask_never_masks_the_exit_and_forces_it_when_toxic_and_long() -> None:
    assert rs.compute_action_mask(0.0, 0.5, 0.9).tolist() == [True, True, True]                  # sous le coupe-circuit : tout est permis
    assert rs.compute_action_mask(0.0, 0.95, 0.9).tolist() == [True, False, True]                # à plat : pas d'ouverture, on peut rester
    assert rs.compute_action_mask(1.0, 0.95, 0.9).tolist() == [False, False, True]               # en position : seule la sortie
    assert rs.compute_action_mask(1.0, 0.9, 0.9).tolist() == [True, True, True]                  # strictement au-dessus du seuil
    assert rs.compute_action_mask(1.0, 0.95, None).tolist() == [True, True, True]                # pas de coupe-circuit configuré : aucun masque
    for position in (0.0, 1.0):
        for vpin in (0.0, 0.5, 0.95, 1.0, None, math.nan):
            assert rs.compute_action_mask(position, vpin, 0.9)[ACTION_SELL]                     # SELL (la sortie) n'est jamais masqué


def test_an_unknown_vpin_blocks_opening_but_never_forces_an_exit() -> None:
    for unknown in (None, math.nan, math.inf):
        assert rs.compute_action_mask(0.0, unknown, 0.9).tolist() == [True, False, True]
        assert rs.compute_action_mask(1.0, unknown, 0.9).tolist() == [True, False, True]  # on ne sait pas : pas d'ouverture, mais on ne liquide pas non plus


def test_the_fallback_action_is_to_exit_when_long_and_to_stay_flat_otherwise() -> None:
    assert rs.safe_action(1.0) == ACTION_SELL and rs.safe_action(0.0) == ACTION_HOLD


# --- Dérivées du VPIN (causales) ---------------------------------------------------------------------


def test_the_vpin_velocity_and_acceleration_of_a_known_curve() -> None:
    times = np.arange(0.0, 601.0, 10.0)
    ramp = rs.vpin_derivatives(np.array([300.0]), times, 0.0010 * times, window_s=60.0)[0]
    assert ramp[0] == pytest.approx(0.30) and ramp[1] == pytest.approx(0.06) and ramp[2] == pytest.approx(0.0, abs=1e-12)  # droite : vitesse constante, accélération nulle
    parabola = rs.vpin_derivatives(np.array([300.0]), times, 1e-6 * times ** 2, window_s=60.0)[0]
    assert parabola[2] == pytest.approx(2e-6 * 60.0 ** 2)  # v'' x w^2


def test_the_features_never_look_at_the_future() -> None:
    times = np.arange(0.0, 601.0, 10.0)
    rng = np.random.default_rng(0)
    values = rng.random(times.size)
    queries = np.arange(150.0, 301.0, 25.0)
    on_truncated = rs.vpin_derivatives(queries, times[times <= 300.0], values[times <= 300.0])
    on_full = rs.vpin_derivatives(queries, times, values)
    assert np.allclose(on_truncated, on_full, equal_nan=True)  # ajouter des relevés POSTÉRIEURS ne change rien au passé
    corrupted = values.copy()
    corrupted[times > 300.0] = 123.0
    assert np.allclose(rs.vpin_derivatives(queries, times, corrupted), on_full, equal_nan=True)


def test_unavailable_or_stale_readings_are_nan_never_a_stale_value_in_disguise() -> None:
    times, values = np.array([100.0, 160.0, 220.0]), np.array([0.5, 0.6, 0.7])
    out = rs.vpin_derivatives(np.array([50.0, 150.0, 220.0, 5_000.0]), times, values, window_s=60.0, max_age_s=600.0)
    assert np.isnan(out[0]).all()                      # avant le premier relevé
    assert np.isnan(out[1, 1:]).all()                  # niveau connu, mais pas assez d'historique pour la vitesse et l'accélération
    assert np.isfinite(out[2]).all() and out[2, 0] == pytest.approx(0.7)
    assert np.isnan(out[3]).all()                      # relevé vieux de plus de max_age : un flux gelé ne passe pas pour un VPIN stable


def test_vpin_derivatives_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="trié"):
        rs.vpin_derivatives(np.array([1.0]), np.array([2.0, 1.0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="même forme"):
        rs.vpin_derivatives(np.array([1.0]), np.array([1.0, 2.0]), np.array([0.1]))
    with pytest.raises(ValueError):
        rs.vpin_derivatives(np.array([1.0]), np.array([1.0]), np.array([0.1]), window_s=0.0)
    assert np.isnan(rs.vpin_derivatives(np.array([1.0]), np.array([]), np.array([]))).all()


# =============================================================================================
# Environnement
# =============================================================================================


def _market(steps: int = 40, price: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    return rng.normal(size=(steps, 5)), np.full(steps, price)


def _vpin(level: float, steps: int = 40) -> np.ndarray:
    return np.column_stack([np.full(steps, level), np.zeros(steps), np.zeros(steps)])


def _env(prices: np.ndarray | None = None, vpin: np.ndarray | None = None, shaping: RiskShapingConfig | None = None, **kwargs: Any) -> MarketEnv:
    features, flat = _market(len(prices) if prices is not None else 40)
    env = MarketEnv(features, flat if prices is None else prices, vpin_features=vpin, risk_shaping=shaping, episode_length=len(flat) - 2, **kwargs)
    env.reset(seed=0)
    env._episode_start = env._current_step = 0  # départ déterministe
    return env


def test_without_the_new_arguments_the_environment_behaves_exactly_as_before() -> None:
    features, prices = _market(60)
    prices = prices * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 1e-3, 60)))
    plain = MarketEnv(features, prices)
    neutral = MarketEnv(features, prices, risk_shaping=RiskShapingConfig())
    obs_a, _ = plain.reset(seed=3)
    obs_b, _ = neutral.reset(seed=3)
    assert obs_a.shape == (OBSERVATION_DIM,) and np.array_equal(obs_a, obs_b)
    for action in (ACTION_BUY, ACTION_HOLD, ACTION_HOLD, ACTION_SELL, ACTION_BUY, ACTION_SELL):
        step_a, step_b = plain.step(action), neutral.step(action)
        assert np.array_equal(step_a[0], step_b[0]) and step_a[1] == step_b[1] and step_a[2:4] == step_b[2:4]
        assert not step_b[4]["stop_loss_hit"] and not step_b[4]["action_overridden"] and step_b[4]["action_mask"].all()


def test_the_vpin_features_extend_the_observation_to_nine_values() -> None:
    features, prices = _market()
    vpin = np.column_stack([np.linspace(0.1, 0.9, 40), np.linspace(0.0, 0.4, 40), np.linspace(0.0, -0.2, 40)])
    env = MarketEnv(features, prices, vpin_features=vpin)
    assert env.observation_space.shape == (OBSERVATION_DIM_WITH_VPIN,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,) and np.array_equal(obs[6:], vpin[env._current_step])
    obs2, *_ = env.step(ACTION_BUY)
    assert obs2[5] == 1.0 and np.array_equal(obs2[6:], vpin[env._current_step])  # position à jour, VPIN de l'étape courante


@pytest.mark.parametrize("vpin", [np.zeros((40, 2)), np.zeros((39, 3)), np.full((40, 3), np.nan), np.full((40, 3), np.inf)])
def test_malformed_vpin_features_are_refused(vpin: np.ndarray) -> None:
    features, prices = _market()
    with pytest.raises(ValueError, match="vpin_features"):
        MarketEnv(features, prices, vpin_features=vpin)


def test_a_vpin_dependent_shaping_without_a_vpin_series_is_refused() -> None:
    features, prices = _market()
    for shaping in (RiskShapingConfig(toxicity_penalty_weight=0.1), RiskShapingConfig(kill_switch_vpin=0.9), RiskShapingConfig(safe_harbor_bonus=0.1)):
        with pytest.raises(ValueError, match="vpin_features is required"):
            MarketEnv(features, prices, risk_shaping=shaping)


def test_holding_in_a_toxic_market_costs_the_tax_every_step_even_when_the_price_is_flat() -> None:
    env = _env(vpin=_vpin(0.9), shaping=RiskShapingConfig(vpin_threshold=0.7, toxicity_penalty_weight=0.5))
    _, reward_buy, *_ = env.step(ACTION_BUY)
    _, reward_hold, *_ = env.step(ACTION_HOLD)
    assert reward_buy == pytest.approx(-COST - 0.5 * 0.2)   # coût d'entrée + taxe : le prix ne bouge pas
    assert reward_hold == pytest.approx(-0.5 * 0.2)         # la taxe tombe à chaque pas passé en position
    _, reward_exit, *_ = env.step(ACTION_SELL)
    assert reward_exit == pytest.approx(-COST)              # sorti : plus de taxe, seulement le coût de sortie
    _, reward_flat, *_ = env.step(ACTION_HOLD)
    assert reward_flat == pytest.approx(0.0)


def test_a_healthy_market_is_never_taxed() -> None:
    env = _env(vpin=_vpin(0.4), shaping=RiskShapingConfig(vpin_threshold=0.7, toxicity_penalty_weight=0.5))
    env.step(ACTION_BUY)
    _, reward, *_ = env.step(ACTION_HOLD)
    assert reward == pytest.approx(0.0)


def test_a_profitable_trade_in_a_toxic_market_can_have_a_negative_reward() -> None:
    """L'exemple de la spécification : +10 bps de PnL dans un marché toxique valent moins que zéro une fois la taxe payée."""
    prices = 100.0 * np.exp(np.linspace(0.0, 0.0010, 40))  # +10 bps au total, 0,0256 bp par pas
    env = _env(prices=prices, vpin=_vpin(1.0), shaping=RiskShapingConfig(vpin_threshold=0.7, toxicity_penalty_weight=0.01), risk_aversion=0.0)
    env.step(ACTION_BUY)
    _, reward, *_ = env.step(ACTION_HOLD)
    assert math.log(prices[2] / prices[1]) > 0           # le trade est bien gagnant à cet instant
    assert float(reward) < 0                                      # ... mais la taxe (0,01 x 0,3 = 0,003) dépasse le gain du pas


def test_the_safe_harbor_bonus_is_paid_only_while_flat_and_toxic() -> None:
    env = _env(vpin=_vpin(0.95), shaping=RiskShapingConfig(vpin_threshold=0.7, safe_harbor_bonus=0.002))
    _, reward_flat, *_ = env.step(ACTION_HOLD)
    assert reward_flat == pytest.approx(0.002)
    _, reward_buy, *_ = env.step(ACTION_BUY)
    assert reward_buy == pytest.approx(-COST)             # en position : aucun bonus
    calm = _env(vpin=_vpin(0.2), shaping=RiskShapingConfig(vpin_threshold=0.7, safe_harbor_bonus=0.002))
    assert calm.step(ACTION_HOLD)[1] == pytest.approx(0.0)


# --- Stops ---------------------------------------------------------------------------------------------


def _falling(*levels: float) -> np.ndarray:
    return np.array([*levels, *([levels[-1]] * (40 - len(levels)))], dtype=np.float64)


def test_a_stop_forces_the_exit_and_reports_the_realized_loss() -> None:
    env = _env(prices=_falling(100.0, 99.9, 99.5), shaping=RiskShapingConfig(stop_loss_bps=40.0))
    _, _, _, _, info1 = env.step(ACTION_BUY)
    assert not info1["stop_loss_hit"] and info1["position"] == 1          # -10 bps : pas de stop
    obs, _, _, _, info2 = env.step(ACTION_HOLD)
    assert info2["stop_loss_hit"] and info2["position"] == 0              # -50 bps depuis l'entrée : sortie forcée
    assert info2["stop_loss_bps"] == pytest.approx(50.0, rel=1e-3) and obs[5] == 0.0


def test_a_gap_through_the_stop_realizes_the_whole_gap_and_costs_more_than_a_small_overshoot() -> None:
    small = _env(prices=_falling(100.0, 99.5), shaping=RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0))
    gap = _env(prices=_falling(100.0, 98.0), shaping=RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=1.0))
    results = []
    for env in (small, gap):
        _, _, _, _, info = env.step(ACTION_BUY)
        results.append(info)
    assert results[0]["stop_loss_bps"] == pytest.approx(50.0, rel=1e-3) and results[1]["stop_loss_bps"] == pytest.approx(200.0, rel=1e-2)


def test_the_tail_penalty_is_exactly_the_documented_extra_and_the_forced_exit_pays_its_cost() -> None:
    prices = _falling(100.0, 99.5)
    base = _env(prices=prices, shaping=RiskShapingConfig(stop_loss_bps=40.0), risk_aversion=0.0)
    tail = _env(prices=prices, shaping=RiskShapingConfig(stop_loss_bps=40.0, tail_risk_weight=2.0, tail_alpha=1.5), risk_aversion=0.0)
    no_stop = _env(prices=prices, risk_aversion=0.0)
    r_base, r_tail, r_none = (float(e.step(ACTION_BUY)[1]) for e in (base, tail, no_stop))
    loss_bps = (1.0 - 99.5 / 100.0) * 10_000
    assert r_base == pytest.approx(r_none - COST)  # la sortie forcée est un changement de position de plus
    assert r_tail == pytest.approx(r_base - 2.0 * loss_bps * ((loss_bps / 40.0) ** 0.5 - 1.0) / 10_000)


def test_no_stop_is_applied_when_none_is_configured_or_when_the_loss_is_within_the_stop() -> None:
    off = _env(prices=_falling(100.0, 50.0))
    assert not off.step(ACTION_BUY)[4]["stop_loss_hit"]
    within = _env(prices=_falling(100.0, 99.8), shaping=RiskShapingConfig(stop_loss_bps=40.0))
    assert not within.step(ACTION_BUY)[4]["stop_loss_hit"]


def test_the_stop_is_measured_from_the_latest_entry_not_a_previous_one() -> None:
    env = _env(prices=_falling(100.0, 100.0, 99.0, 99.0, 98.9), shaping=RiskShapingConfig(stop_loss_bps=40.0))
    env.step(ACTION_BUY)
    env.step(ACTION_SELL)                      # sorti à plat
    _, _, _, _, info = env.step(ACTION_BUY)    # nouvelle entrée à 99,0
    assert env._entry_price == pytest.approx(99.0)
    _, _, _, _, info = env.step(ACTION_HOLD)   # 98,9 : -10 bps depuis la NOUVELLE entrée (et -110 depuis l'ancienne)
    assert not info["stop_loss_hit"]


# --- Masque appliqué par l'environnement -----------------------------------------------------------------


def test_a_masked_action_is_never_executed_even_if_the_policy_asks_for_it() -> None:
    env = _env(vpin=_vpin(0.95), shaping=RiskShapingConfig(kill_switch_vpin=0.9))
    _, _, _, _, info = env.step(ACTION_BUY)
    assert info["action_overridden"] and info["position"] == 0            # le marché est toxique : l'ouverture est refusée
    assert info["action_mask"].tolist() == [True, False, True]


def test_a_held_position_is_forced_out_when_the_kill_switch_trips() -> None:
    vpin = np.column_stack([np.concatenate([np.full(3, 0.3), np.full(37, 0.95)]), np.zeros(40), np.zeros(40)])
    env = _env(vpin=vpin, shaping=RiskShapingConfig(kill_switch_vpin=0.9))
    for _ in range(3):  # trois pas à VPIN bas : ouverture, puis on garde
        env.step(ACTION_BUY if env._position == 0 else ACTION_HOLD)
    assert env._position == 1
    _, _, _, _, info = env.step(ACTION_HOLD)                              # le VPIN dépasse maintenant le coupe-circuit : HOLD est interdit en position
    assert info["action_overridden"] is True and info["position"] == 0   # sortie forcée, pas de piège dans la position
    assert info["action_mask"].tolist() == [True, False, True]


def test_the_exit_is_never_overridden() -> None:
    env = _env(vpin=_vpin(0.3), shaping=RiskShapingConfig(kill_switch_vpin=0.9))
    env.step(ACTION_BUY)
    assert not env.step(ACTION_SELL)[4]["action_overridden"]
    toxic = _env(vpin=_vpin(0.99), shaping=RiskShapingConfig(kill_switch_vpin=0.9))
    assert not toxic.step(ACTION_SELL)[4]["action_overridden"]


def test_the_environment_mask_matches_the_pure_function_and_is_returned_by_reset() -> None:
    env = _env(vpin=_vpin(0.95), shaping=RiskShapingConfig(kill_switch_vpin=0.9))
    _, info = env.reset(seed=0)
    assert np.array_equal(info["action_mask"], rs.compute_action_mask(0.0, 0.95, 0.9))
    assert MarketEnv(*_market()).action_mask().all()  # sans coupe-circuit : tout est permis

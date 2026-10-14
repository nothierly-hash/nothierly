"""Aversion au risque du PPO, niveau 2 : réseau (masque), entraînement (Huber, gamma, VPIN), inférence (politique et agent)."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fakeredis
import fakeredis.aioredis
import numpy as np
import pytest
import torch

from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import MarketMetrics
from scripts import train_ppo as tp
from src.agents.rl_signal import agent as agent_module
from src.agents.rl_signal.agent import RlSignalAgent
from src.agents.rl_signal.policy_model import RLPolicyModel, VpinUnavailableError
from src.rl.networks import ActorCritic, masked_probabilities
from src.rl.risk_shaping import RiskShapingConfig
from src.rl.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, MarketEnv
from workers.market_data.vpin_archive import BucketRecord, VpinArchive

from decimal import Decimal


# =============================================================================================
# Réseau : masque d'actions
# =============================================================================================


def test_masked_probabilities_zero_the_forbidden_actions_and_renormalize() -> None:
    probs = torch.tensor([[0.2, 0.5, 0.3]])
    out = masked_probabilities(probs, torch.tensor([[True, False, True]]))
    assert out[0, 1] == 0.0 and float(out.sum()) == pytest.approx(1.0)
    assert float(out[0, 0]) == pytest.approx(0.2 / 0.5) and float(out[0, 2]) == pytest.approx(0.3 / 0.5)  # proportions conservées entre actions permises
    assert torch.equal(masked_probabilities(probs, torch.tensor([[True, True, True]])), probs)         # tout permis : inchangé
    assert masked_probabilities(probs, torch.tensor([True, False, True])).shape == (1, 3)              # un masque 1-D se diffuse sur le lot


def test_a_policy_certain_of_a_forbidden_action_falls_back_to_uniform_over_the_allowed_ones() -> None:
    out = masked_probabilities(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[True, False, True]]))
    assert not torch.isnan(out).any() and out[0].tolist() == [0.5, 0.0, 0.5]  # jamais de NaN : l'agent a toujours un coup légal


def test_sampling_under_a_mask_never_returns_a_forbidden_action_and_the_log_prob_is_the_masked_one() -> None:
    torch.manual_seed(0)
    model = ActorCritic(observation_dim=9, action_dim=3)
    observations = torch.randn(2000, 9)
    mask = torch.tensor([True, False, True]).expand(2000, 3)
    actions, log_probs, entropy, values = model.get_action_and_value(observations, action_mask=mask)
    assert (actions != ACTION_BUY).all()
    probs, _ = model(observations)
    masked = masked_probabilities(probs, mask)
    assert torch.allclose(log_probs, torch.log(masked.gather(1, actions.unsqueeze(1)).squeeze(1)), atol=1e-5)  # le gradient PPO voit la distribution réellement échantillonnée
    assert torch.allclose(entropy, -(masked * torch.log(masked.clamp_min(1e-12))).sum(dim=1), atol=1e-5)
    unmasked = model.get_action_and_value(observations)
    assert torch.allclose(values, unmasked[3])  # le critique ne dépend pas du masque


def test_the_exported_network_keeps_its_torchscript_forward() -> None:
    model = ActorCritic(observation_dim=9, action_dim=3)
    scripted = torch.jit.script(model)
    sample = torch.randn(4, 9)
    assert torch.allclose(scripted(sample)[0], model(sample)[0])


# =============================================================================================
# Entraînement
# =============================================================================================


def test_ppo_config_validates_gamma_and_the_critic_loss() -> None:
    assert tp.PPOConfig().value_loss == "mse" and tp.PPOConfig().gamma == 0.99  # la bibliothèque garde ses valeurs historiques
    for bad in ({"gamma": 0.0}, {"gamma": 1.5}, {"gae_lambda": 2.0}, {"value_loss": "l1"}, {"huber_delta": 0.0}):
        with pytest.raises(ValueError):
            tp.PPOConfig(**bad)


def _toxic_env(steps: int = 300, level: float = 0.95) -> MarketEnv:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(steps, 5))
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, steps)))
    vpin = np.column_stack([np.full(steps, level), np.zeros(steps), np.zeros(steps)])
    return MarketEnv(features, prices, vpin_features=vpin, risk_shaping=RiskShapingConfig(kill_switch_vpin=0.9))


def test_the_rollout_collector_respects_and_records_the_mask() -> None:
    torch.manual_seed(1)
    env = _toxic_env()
    model = ActorCritic(observation_dim=9, action_dim=3)
    buffer = tp.RolloutBuffer()
    obs, _ = env.reset(seed=0)
    tp.collect_rollout(env, model, buffer, obs, 200)
    assert len(buffer.action_masks) == 200 and ACTION_BUY not in buffer.actions    # VPIN 0,95 > 0,9 : jamais d'ouverture
    assert all(mask.tolist() == [True, False, True] for mask in buffer.action_masks)
    free = MarketEnv(env._features, env._prices)
    free_buffer = tp.RolloutBuffer()
    obs, _ = free.reset(seed=0)
    tp.collect_rollout(free, ActorCritic(observation_dim=6, action_dim=3), free_buffer, obs, 200)
    assert ACTION_BUY in free_buffer.actions                                        # sans coupe-circuit, la même politique ouvre bien : le masque est la cause


def _critic_buffer(n: int = 64, extreme: float = -50.0) -> tuple[ActorCritic, tp.RolloutBuffer, float]:
    torch.manual_seed(3)
    model = ActorCritic(observation_dim=6, action_dim=3)
    obs = np.random.default_rng(3).normal(size=(n, 6))
    actions = np.random.default_rng(4).integers(0, 3, size=n)
    with torch.no_grad():
        _, log_probs, _, values = model.get_action_and_value(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(actions))
    rewards = np.full(n, 0.001)
    rewards[n // 2] = extreme  # une perte de queue extrême, rare : l'événement que MSE laisse dominer le gradient
    buffer = tp.RolloutBuffer(
        observations=list(obs), actions=[int(a) for a in actions], log_probs=[float(x) for x in log_probs], rewards=[float(r) for r in rewards],
        values=[float(v) for v in values], dones=[False] * (n - 1) + [True], action_masks=[np.ones(3, dtype=np.bool_) for _ in range(n)],
    )
    return model, buffer, 0.0


def test_the_huber_critic_loss_is_far_less_dominated_by_a_tail_event_than_mse() -> None:
    losses = {}
    for kind in ("mse", "huber"):
        model, buffer, last_value = _critic_buffer()
        np.random.seed(0)
        config = tp.PPOConfig(value_loss=kind, ppo_epochs=1, minibatch_size=64, gamma=0.97)
        losses[kind] = tp.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), buffer, last_value, config)["value_loss"]
    assert losses["huber"] < losses["mse"] / 10  # linéaire au-delà de delta : l'événement extrême ne fait plus exploser la perte


def test_the_update_runs_with_masks_and_reports_finite_metrics() -> None:
    model, buffer, last_value = _critic_buffer(extreme=-1.0)
    buffer.action_masks = [np.array([True, False, True]) if i % 2 else np.ones(3, dtype=np.bool_) for i in range(len(buffer.action_masks))]
    buffer.actions = [a if a != ACTION_BUY or buffer.action_masks[i][ACTION_BUY] else ACTION_SELL for i, a in enumerate(buffer.actions)]
    metrics = tp.ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), buffer, last_value, tp.PPOConfig(ppo_epochs=2, minibatch_size=16))
    assert all(np.isfinite(v) for v in metrics.values())


def test_a_lower_gamma_discounts_distant_rewards_more() -> None:
    rewards, values, dones = np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(4), np.zeros(4, dtype=np.bool_)
    far, _ = tp.compute_gae(rewards, values, dones, 0.0, gamma=0.99, gae_lambda=1.0)
    near, _ = tp.compute_gae(rewards, values, dones, 0.0, gamma=0.90, gae_lambda=1.0)
    assert far[0] == pytest.approx(0.99 ** 3) and near[0] == pytest.approx(0.90 ** 3) and near[0] < far[0]  # un agent moins patient accorde moins de poids au gain lointain


def test_a_short_training_run_with_vpin_features_and_shaping_produces_a_nine_input_policy() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    steps = 400
    features = rng.normal(size=(steps, 5))
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, steps)))
    vpin = np.column_stack([rng.random(steps), rng.normal(0, 0.01, steps), rng.normal(0, 0.01, steps)])
    shaping = RiskShapingConfig(toxicity_penalty_weight=0.01, stop_loss_bps=40.0, tail_risk_weight=1.0, kill_switch_vpin=0.95)
    config = tp.PPOConfig(total_timesteps=128, rollout_length=64, ppo_epochs=1, minibatch_size=32, gamma=0.97, value_loss="huber")
    model = tp.run_ppo_training(features, prices, config, vpin_features=vpin, risk_shaping=shaping)
    first = model.shared[0]
    assert isinstance(first, torch.nn.Linear) and first.in_features == 9


# --- VPIN : chargement et alignement ---------------------------------------------------------------


def _archive(path: Path, rows: list[tuple[str, int, str | None]]) -> Path:
    archive = VpinArchive(path)
    for symbol, ts, value in rows:
        archive.submit(BucketRecord(symbol, ts, None if value is None else Decimal(value), Decimal(60), Decimal(40), Decimal(100), 50))
    assert archive.close() == 0
    return path


def test_the_vpin_loader_keeps_one_symbol_known_values_and_the_window(tmp_path: Path) -> None:
    db = _archive(tmp_path / "v.sqlite3", [
        ("BTCUSDT", 1_000_000, "0.5"), ("BTCUSDT", 2_000_000, None), ("ETHUSDT", 1_500_000, "0.9"), ("BTCUSDT", 3_000_000, "0.7"), ("BTCUSDT", 9_000_000, "0.2"),
    ])
    times, values = tp.load_vpin_series(db, "BTC/USDT", 500.0, 4_000.0)
    assert times.tolist() == [1_000.0, 3_000.0] and values.tolist() == [0.5, 0.7]  # ni ETH, ni valeur vide (fenêtre non pleine), ni hors fenêtre


def test_the_vpin_loader_fails_loudly_on_a_missing_file_or_an_empty_selection(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tp.load_vpin_series(tmp_path / "absente.sqlite3", "BTCUSDT", 0.0, 1.0)
    db = _archive(tmp_path / "v.sqlite3", [("ETHUSDT", 1_000, "0.5")])
    with pytest.raises(ValueError, match="aucun VPIN"):
        tp.load_vpin_series(db, "BTCUSDT", 0.0, 100.0)


def test_the_longest_contiguous_valid_run_is_kept_not_scattered_rows() -> None:
    valid = np.array([True, True, False, True, True, True, False, True])
    run = tp.longest_valid_run(valid)
    assert (run.start, run.stop) == (3, 6)
    assert tp.longest_valid_run(np.array([False, False])) == slice(0, 0)


def test_vpin_features_are_trimmed_to_the_contiguous_stretch_where_everything_is_known() -> None:
    row_times = np.arange(0.0, 8_000.0, 10.0)                        # 800 lignes
    vpin_times = np.arange(0.0, 8_000.0, 5.0)
    keep = (vpin_times < 2_000.0) | (vpin_times > 3_200.0)           # un trou de 1 200 s dans l'archive (> VPIN_MAX_AGE_S)
    features, run = tp.attach_vpin_features(row_times, vpin_times[keep], np.linspace(0.2, 0.8, int(keep.sum())))
    assert np.isfinite(features[run]).all()
    assert run.start > 320                                           # la tranche retenue est celle d'apres le trou (lignes 200 a 320 sans VPIN frais)
    assert run.stop - run.start >= tp.MIN_TRAINING_ROWS


def test_too_short_a_vpin_history_is_refused() -> None:
    with pytest.raises(ValueError, match="consécutives"):
        tp.attach_vpin_features(np.arange(0.0, 500.0, 10.0), np.arange(0.0, 500.0, 5.0), np.full(100, 0.5))


# --- Export et ligne de commande -----------------------------------------------------------------------


def test_the_exported_policy_carries_its_metadata_and_a_legacy_export_does_not(tmp_path: Path) -> None:
    shaping = RiskShapingConfig(toxicity_penalty_weight=0.01, kill_switch_vpin=0.9)
    config = tp.PPOConfig(gamma=0.97, value_loss="huber")
    meta = tp.policy_metadata(9, True, shaping, config)
    path = tp.export_model(ActorCritic(9, 3), tmp_path / "a", meta)
    written = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert written["observation_dim"] == 9 and written["uses_vpin"] is True and written["kill_switch_vpin"] == 0.9 and written["gamma"] == 0.97
    legacy = tp.export_model(ActorCritic(6, 3), tmp_path / "b")
    assert not legacy.with_suffix(".meta.json").exists()


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    monkeypatch.setattr(sys, "argv", ["train_ppo.py", "--symbol", "BTCUSDT", *args])
    return tp._parse_args()


def test_the_command_line_defaults_to_the_risk_averse_critic_and_gamma(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _argv(monkeypatch)
    assert args.gamma == 0.97 and args.value_loss == "huber" and args.vpin_db is None
    shaping = tp._shaping_from_args(args)
    assert shaping == RiskShapingConfig() or (shaping.toxicity_penalty_weight == 0.0 and shaping.kill_switch_vpin is None)  # sans archive VPIN, aucune taxe


def test_the_vpin_options_need_the_vpin_archive_and_the_tail_weight_needs_a_stop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for bad in (["--kill-switch-vpin", "0.9"], ["--safe-harbor-bonus", "0.1"], ["--tail-weight", "2"]):
        with pytest.raises(SystemExit):
            _argv(monkeypatch, *bad)
    args = _argv(monkeypatch, "--vpin-db", str(tmp_path / "v.sqlite3"), "--kill-switch-vpin", "0.9", "--stop-loss-bps", "40")
    shaping = tp._shaping_from_args(args)
    assert shaping.toxicity_penalty_weight == 0.01 and shaping.kill_switch_vpin == 0.9 and shaping.stop_loss_bps == 40.0 and shaping.tail_risk_weight == 1.0


# =============================================================================================
# Inférence : la politique
# =============================================================================================


def _policy(directory: Path, observation_dim: int, *, prefer: int | None = None, meta: dict[str, Any] | None = None) -> Path:
    model = ActorCritic(observation_dim, 3)
    if prefer is not None:  # pousse la politique vers une action : toutes les sorties dépendent du biais, pas de l'observation
        head = model.actor_head[0]
        assert isinstance(head, torch.nn.Linear)
        head.weight.data.zero_()
        head.bias.data = torch.tensor([8.0 if i == prefer else 0.0 for i in range(3)])
    return tp.export_model(model, directory, meta)


FEATURES = np.zeros(5)
GOOD_VPIN = np.array([0.30, 0.01, 0.0])
TOXIC_VPIN = np.array([0.95, 0.10, 0.02])


def test_a_policy_without_metadata_is_a_legacy_six_input_policy(tmp_path: Path) -> None:
    model = RLPolicyModel(_policy(tmp_path, 6, prefer=ACTION_BUY))
    assert not model.uses_vpin and model.kill_switch_vpin is None
    assert model.predict_action(FEATURES)[0] == 1                                    # sans VPIN, comme avant
    assert model.predict_action(FEATURES, vpin_features=GOOD_VPIN)[0] == 1           # un VPIN fourni à une ancienne politique ne la perturbe pas


def test_the_untrained_skeleton_is_unchanged() -> None:
    model = RLPolicyModel()
    assert not model.is_fitted and not model.uses_vpin
    assert model.predict_action(FEATURES)[0] in (-1, 0, 1)


def test_a_vpin_policy_refuses_to_guess_without_a_complete_vpin(tmp_path: Path) -> None:
    meta = tp.policy_metadata(9, True, RiskShapingConfig(), tp.PPOConfig())
    model = RLPolicyModel(_policy(tmp_path, 9, prefer=ACTION_BUY, meta=meta))
    assert model.uses_vpin
    with pytest.raises(VpinUnavailableError):
        model.predict_action(FEATURES)
    with pytest.raises(VpinUnavailableError):
        model.predict_action(FEATURES, vpin_features=np.array([0.3, np.nan, 0.0]))
    assert model.predict_action(FEATURES, vpin_features=GOOD_VPIN)[0] == 1


def test_an_unsupported_observation_size_in_the_metadata_is_refused(tmp_path: Path) -> None:
    path = _policy(tmp_path, 6)
    path.with_suffix(".meta.json").write_text(json.dumps({"observation_dim": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="observation_dim"):
        RLPolicyModel(path)


def test_the_kill_switch_masks_the_opening_whatever_the_policy_wants(tmp_path: Path) -> None:
    model = RLPolicyModel(_policy(tmp_path, 6, prefer=ACTION_BUY), kill_switch_vpin=0.9)  # une politique qui veut ABSOLUMENT acheter
    assert model.predict_action(FEATURES, vpin_features=GOOD_VPIN)[0] == 1               # marché sain : elle achète
    flat_signal, _ = model.predict_action(FEATURES, vpin_features=TOXIC_VPIN)
    assert flat_signal != 1                                                              # marché toxique : l'ouverture est interdite
    exit_signal, _ = model.predict_action(FEATURES, current_position=1.0, vpin_features=TOXIC_VPIN)
    assert exit_signal == -1                                                             # en position : seule la sortie reste permise
    unknown_signal, _ = model.predict_action(FEATURES)
    assert unknown_signal != 1                                                           # VPIN inconnu avec un coupe-circuit : on n'ouvre pas non plus


def test_the_kill_switch_defaults_to_the_one_stored_with_the_policy_and_an_explicit_value_wins(tmp_path: Path) -> None:
    meta = tp.policy_metadata(9, True, RiskShapingConfig(kill_switch_vpin=0.8, toxicity_penalty_weight=0.01), tp.PPOConfig())
    path = _policy(tmp_path, 9, prefer=ACTION_BUY, meta=meta)
    assert RLPolicyModel(path).kill_switch_vpin == 0.8
    assert RLPolicyModel(path, kill_switch_vpin=0.99).kill_switch_vpin == 0.99
    assert RLPolicyModel(path).predict_action(FEATURES, vpin_features=np.array([0.85, 0.0, 0.0]))[0] != 1  # 0,85 > 0,8 : masqué


def test_the_confidence_is_the_renormalized_probability_after_masking(tmp_path: Path) -> None:
    model = RLPolicyModel(_policy(tmp_path, 6, prefer=ACTION_BUY), kill_switch_vpin=0.9)
    _, confidence = model.predict_action(FEATURES, vpin_features=TOXIC_VPIN)
    assert confidence == pytest.approx(0.5)  # HOLD et SELL à égalité une fois BUY retiré
    with pytest.raises(ValueError):
        model.predict_action(np.zeros(4))
    with pytest.raises(ValueError):
        model.predict_action(FEATURES, vpin_features=np.zeros(2))


# =============================================================================================
# Inférence : l'agent
# =============================================================================================


@pytest.fixture
def make_agent(tmp_path: Path) -> Any:
    agents: list[RlSignalAgent] = []

    def make(policy: Path | None = None, **kwargs: Any) -> RlSignalAgent:
        client = RedisClient(RedisSettings())
        client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
        agent = RlSignalAgent(
            "BTC/USDT", model_path=policy, settings=AppSettings(mock_mode=True, guardian=GuardianSettings(llm_provider="none")),
            manage_own_signals=False, manage_own_redis=False, **kwargs,
        )
        agent._redis_client = client
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent._publisher.close()
        agent._subscriber.close()
        agent._zmq_context.term()


async def _set_vpin(agent: RlSignalAgent, value: Any) -> None:
    assert agent._redis_client is not None
    await agent._redis_client.execute("set", "market:vpin:BTCUSDT", str(value))


def _clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1_000.0]
    monkeypatch.setattr(agent_module, "time", SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
    return now


@pytest.mark.asyncio
async def test_the_agent_builds_level_velocity_and_acceleration_from_its_own_history(make_agent: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent()
    now = _clock(monkeypatch)
    results = []
    for offset, value in ((0, 0.50), (60, 0.53), (120, 0.56)):
        now[0] = 1_000.0 + offset
        await _set_vpin(agent, value)
        results.append(await agent._read_vpin_features())
    assert results[0] is None and results[1] is None             # pas encore deux fenêtres d'historique : jamais une dérivée inventée
    final = results[2]
    assert final is not None and final.tolist() == pytest.approx([0.56, 0.03, 0.0])


@pytest.mark.asyncio
async def test_the_agent_ignores_a_missing_unreadable_or_out_of_range_vpin(make_agent: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent()
    now = _clock(monkeypatch)
    assert await agent._read_vpin_features() is None                    # clé absente
    for bad in ("abc", "1.5", "-0.1", "nan"):
        await _set_vpin(agent, bad)
        assert await agent._read_vpin_features() is None
    assert len(agent._vpin_history) == 0                                # rien d'illisible n'entre dans l'historique
    for offset, value in ((0, 0.50), (60, 0.53), (120, 0.56)):
        now[0] = 1_000.0 + offset
        await _set_vpin(agent, value)
        await agent._read_vpin_features()
    assert agent._redis_client is not None
    await agent._redis_client.execute("delete", "market:vpin:BTCUSDT")  # le tracker est arrêté : la clé expire
    now[0] += 30
    assert await agent._read_vpin_features() is None                    # on ne ressert pas le dernier relevé comme s'il était frais


def _metrics() -> MarketMetrics:
    return MarketMetrics(symbol="BTC/USDT", spread=1.0, order_book_imbalance=0.1, mid_price=50_000.0)


@pytest.mark.asyncio
async def test_a_vpin_policy_without_vpin_publishes_a_neutral_signal_instead_of_guessing(make_agent: Any, tmp_path: Path) -> None:
    meta = tp.policy_metadata(9, True, RiskShapingConfig(), tp.PPOConfig())
    agent = make_agent(_policy(tmp_path, 9, prefer=ACTION_BUY, meta=meta))
    sent: list[dict[str, Any]] = []

    async def fake_fetch() -> Any:
        return _metrics(), {}, [], None

    async def fake_publish(topic: str, data: dict[str, Any]) -> None:
        sent.append(data)

    agent._fetch_latest_state = fake_fetch
    agent.publish = fake_publish
    await agent._run_signal_cycle()  # aucun VPIN dans Redis
    assert len(sent) == 1 and sent[0]["signal"] == 0 and sent[0]["confidence"] == 0.0


@pytest.mark.asyncio
async def test_the_agent_kill_switch_stops_a_buy_hungry_policy_in_a_toxic_market(make_agent: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(_policy(tmp_path, 6, prefer=ACTION_BUY), kill_switch_vpin=0.9)
    now = _clock(monkeypatch)
    sent: list[dict[str, Any]] = []

    async def fake_fetch() -> Any:
        return _metrics(), {}, [], None

    async def fake_publish(topic: str, data: dict[str, Any]) -> None:
        sent.append(data)

    agent._fetch_latest_state = fake_fetch
    agent.publish = fake_publish
    for offset in (0, 60, 120):                                  # historique sain, suffisant pour les dérivées
        now[0] = 1_000.0 + offset
        await _set_vpin(agent, 0.3)
        await agent._read_vpin_features()
    await agent._run_signal_cycle()
    now[0] += 60
    await _set_vpin(agent, 0.97)
    await agent._run_signal_cycle()
    assert [s["signal"] for s in sent] == [1, 0]  # sain : elle achète ; toxique : plus d'ouverture (HOLD)

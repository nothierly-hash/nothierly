"""Offline PPO training pipeline for the trading policy.

Not part of the live trading system -- run manually or on a schedule,
completely independent of every running agent::

    python scripts/train_ppo.py --lookback-days 30 --total-timesteps 200000

Pipeline: TimescaleDB (via :mod:`scripts.train_model`'s fetch/alignment
helpers, reused rather than duplicated) -> :class:`~src.rl.trading_env.
MarketEnv` rollouts -> Generalized Advantage Estimation -> PPO
clipped-surrogate policy update -> ``torch.jit.script`` export to
``models/ppo_policy_<timestamp>.pt``.

Mathematical background
------------------------
**GAE** (Schulman et al., 2016, "High-Dimensional Continuous Control
Using Generalized Advantage Estimation"). For a trajectory of rewards
``r_t`` and value estimates ``V(s_t)``, the one-step TD residual is::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

and the GAE advantage is an exponentially-weighted sum of future
residuals, computed backwards through the trajectory::

    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

``lambda=0`` recovers the one-step TD advantage (low variance, high
bias); ``lambda=1`` recovers the full Monte-Carlo advantage (high
variance, low bias). The value-function regression target is then
``Return_t = A_t + V(s_t)``.

**PPO clipped surrogate objective** (Schulman et al., 2017, "Proximal
Policy Optimization Algorithms"). With ``ratio_t = exp(log pi_new(a_t|s_t)
- log pi_old(a_t|s_t))``::

    L^CLIP = E_t[ min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t) ]

Clipping removes the incentive to move ``ratio_t`` far from ``1`` in a
single update once it would increase the objective past what the
clip permits -- this is what keeps PPO's updates stable without TRPO's
explicit KL-divergence constraint. The full loss also includes a value
regression term and an entropy bonus (encouraging continued exploration):

    L = -L^CLIP + c1 * MSE(V(s_t), Return_t) - c2 * H[pi(.|s_t)]
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/train_ppo.py` adds this file's own directory
# (scripts/) to sys.path, not the project root -- `core`/`src`/`scripts`
# (this module imports scripts.train_model below) live at the root, so
# it must be added explicitly before those imports, regardless of the
# working directory the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import asyncio
import asyncpg
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from core.config import get_settings
from core.logger import configure_logging, get_logger
from scripts.train_model import (
    _MACRO_TOPIC,
    _MARKET_TOPIC,
    _NLP_TOPIC,
    align_market_macro_sentiment,
    build_macro_feature_frame,
    build_market_feature_frame,
    build_sentiment_feature_frame,
    fetch_topic_rows,
)
from src.agents.supervisor.ml_engine import SupervisorMLModel
from src.rl.networks import ActorCritic
from src.rl.trading_env import MarketEnv

_log = get_logger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"


@dataclass
class PPOConfig:
    """PPO hyperparameters.

    Attributes:
        total_timesteps: Total environment steps to train for.
        rollout_length: Steps collected per rollout, before each update.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter (see the module docstring).
        clip_epsilon: PPO's clipping range (``eps`` in the surrogate loss).
        ppo_epochs: Full passes over each rollout per update.
        minibatch_size: Minibatch size within each PPO epoch.
        learning_rate: Adam learning rate.
        value_coef: Weight of the value-regression loss term.
        entropy_coef: Weight of the entropy bonus (exploration incentive).
        max_grad_norm: Gradient-clipping threshold (training stability).
    """

    total_timesteps: int = 200_000
    rollout_length: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class RolloutBuffer:
    """One rollout's worth of trajectories, collected step by step."""

    observations: list[npt.NDArray[np.float64]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        """Empty the buffer, ready for the next rollout."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.observations)


# -- GAE ------------------------------------------------------------------


def compute_gae(
    rewards: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    dones: npt.NDArray[np.bool_],
    last_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Generalized Advantage Estimation (Schulman et al., 2016).

    See the module docstring for the formulas. Computed backwards through
    the rollout in a single pass (each ``A_t`` only depends on ``delta_t``
    and ``A_{t+1}``).

    Args:
        rewards: Shape ``(T,)``.
        values: Shape ``(T,)``, the critic's ``V(s_t)`` at each collected
            state.
        dones: Shape ``(T,)``, whether the episode ended at step ``t``
            (truncation counts as "done" here: bootstrapping must not
            reach across an episode boundary).
        last_value: The critic's ``V(s_T)`` for the state *after* the
            last collected step, for bootstrapping the final advantage.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        ``(advantages, returns)``, both shape ``(T,)``.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# -- Rollout collection -----------------------------------------------------


def collect_rollout(
    env: MarketEnv,
    model: ActorCritic,
    buffer: RolloutBuffer,
    obs: npt.NDArray[np.float64],
    rollout_length: int,
) -> npt.NDArray[np.float64]:
    """Run ``model`` in ``env`` for ``rollout_length`` steps under ``torch.no_grad``.

    Rollouts span episode boundaries: on termination/truncation the
    environment auto-resets and collection continues, standard PPO
    practice (each rollout is a fixed number of *steps*, not episodes).

    Args:
        env: The environment to step through.
        model: The current policy/value network (used in eval-free,
            gradient-free inference mode -- PPO only needs gradients
            during the update, not during collection).
        buffer: Appended to in place, one entry per step.
        obs: The observation to resume from (the environment's state is
            external to this function, so the caller threads it through
            across successive rollouts).
        rollout_length: Number of steps to collect.

    Returns:
        The observation to resume from on the *next* call.
    """
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob_tensor, _entropy, value_tensor = model.get_action_and_value(
                obs_tensor
            )
        action = int(action_tensor.item())

        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        buffer.observations.append(obs)
        buffer.actions.append(action)
        buffer.log_probs.append(float(log_prob_tensor.item()))
        buffer.rewards.append(float(reward))
        buffer.values.append(float(value_tensor.item()))
        buffer.dones.append(done)

        if done:
            next_obs, _ = env.reset()
        obs = next_obs

    return obs


# -- PPO update -----------------------------------------------------------


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    config: PPOConfig,
) -> dict[str, float]:
    """Run ``config.ppo_epochs`` passes of minibatch PPO updates over one rollout.

    Args:
        model: The policy/value network to update in place.
        optimizer: Its optimizer.
        buffer: A completed rollout (see :func:`collect_rollout`).
        last_value: The critic's value estimate for the state *after*
            the rollout's last step (bootstrap for GAE).
        config: Hyperparameters.

    Returns:
        Mean ``policy_loss``/``value_loss``/``entropy`` across every
        minibatch update performed.
    """
    observations = torch.as_tensor(np.array(buffer.observations), dtype=torch.float32)
    actions = torch.as_tensor(np.array(buffer.actions), dtype=torch.int64)
    old_log_probs = torch.as_tensor(np.array(buffer.log_probs), dtype=torch.float32)
    rewards = np.array(buffer.rewards, dtype=np.float64)
    values = np.array(buffer.values, dtype=np.float64)
    dones = np.array(buffer.dones, dtype=np.bool_)

    advantages_np, returns_np = compute_gae(
        rewards, values, dones, last_value, gamma=config.gamma, gae_lambda=config.gae_lambda
    )
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32)
    returns = torch.as_tensor(returns_np, dtype=torch.float32)
    # Advantage normalization: a standard PPO trick that reduces the
    # variance of the policy-gradient estimate across minibatches of very
    # different reward scale; does not change the optimum being sought.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(observations)
    indices = np.arange(n)

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_updates = 0

    for _epoch in range(config.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, config.minibatch_size):
            batch_idx = torch.as_tensor(indices[start : start + config.minibatch_size], dtype=torch.int64)

            batch_obs = observations[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_advantages = advantages[batch_idx]
            batch_returns = returns[batch_idx]

            _action, new_log_probs, entropy, new_values = model.get_action_and_value(
                batch_obs, batch_actions
            )

            # PPO clipped surrogate objective -- see the module docstring
            # for the derivation.
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surrogate1 = ratio * batch_advantages
            surrogate2 = (
                torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                * batch_advantages
            )
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = F.mse_loss(new_values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

            optimizer.zero_grad()
            # Tensor.backward() is untyped in torch's stubs.
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.mean().item())
            n_updates += 1

    return {key: value / n_updates for key, value in totals.items()}


# -- Export -----------------------------------------------------------------


def export_model(model: ActorCritic, models_dir: Path = MODELS_DIR) -> Path:
    """Serialize the trained policy via TorchScript for production inference.

    ``torch.jit.script`` ahead-of-time-compiles the model's ``forward``
    pass into TorchScript IR, independent of the Python interpreter. This
    is what lets :class:`~src.agents.supervisor.agent.SupervisorAgent`
    (or any C++/libtorch host) load and run the policy with sub-
    millisecond inference latency and none of the per-call Python
    dispatch overhead a plain ``nn.Module`` carries.

    Args:
        model: The trained network. Switched to eval mode before
            scripting (disables any train-time-only behavior, though
            this architecture has none -- no dropout/batchnorm -- kept
            for correctness/habit regardless).
        models_dir: Directory to save into (created if missing).

    Returns:
        Path to the saved ``.pt`` TorchScript archive.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_path = models_dir / f"ppo_policy_{timestamp}.pt"

    model.eval()
    scripted = torch.jit.script(model)
    # ScriptModule.save() is untyped in torch's stubs.
    scripted.save(str(export_path))  # type: ignore[no-untyped-call]

    return export_path


# -- Orchestration ------------------------------------------------------


async def _fetch_aligned_dataset(
    lookback_days: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fetch and align TimescaleDB data into ``(features, prices)`` arrays.

    Args:
        lookback_days: How many days of archived history to query.

    Returns:
        ``features`` (shape ``(T, 5)``, in
        :attr:`SupervisorMLModel.FEATURE_NAMES` order) and ``prices``
        (shape ``(T,)``, mid-price), aligned to the same ``T`` timesteps.

    Raises:
        ValueError: If there is no usable market data in the queried
            window.
    """
    settings = get_settings()
    timescale = settings.timescale

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    pool = await asyncpg.create_pool(
        host=timescale.host,
        port=timescale.port,
        database=timescale.database,
        user=timescale.user,
        password=timescale.password.get_secret_value() if timescale.password is not None else None,
        min_size=1,
        max_size=2,
    )
    try:
        market_records = await fetch_topic_rows(pool, _MARKET_TOPIC, start, end)
        macro_records = await fetch_topic_rows(pool, _MACRO_TOPIC, start, end)
        nlp_records = await fetch_topic_rows(pool, _NLP_TOPIC, start, end)
    finally:
        await pool.close()

    _log.info(
        "ppo_training_data_fetched",
        market_rows=len(market_records),
        macro_rows=len(macro_records),
        nlp_rows=len(nlp_records),
    )

    market_frame = build_market_feature_frame(market_records)
    if market_frame.empty:
        raise ValueError("No market data available for the requested window.")

    macro_frame = build_macro_feature_frame(macro_records)
    sentiment_frame = build_sentiment_feature_frame(nlp_records)
    aligned = align_market_macro_sentiment(market_frame, macro_frame, sentiment_frame)

    features = aligned[list(SupervisorMLModel.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    prices = aligned["mid_price"].to_numpy(dtype=np.float64)
    return features, prices


def run_ppo_training(
    features: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    config: PPOConfig,
) -> ActorCritic:
    """Run the full PPO training loop (collect -> GAE -> update, repeated).

    Args:
        features: Shape ``(T, 5)``, from :func:`_fetch_aligned_dataset`.
        prices: Shape ``(T,)``, from :func:`_fetch_aligned_dataset`.
        config: Hyperparameters.

    Returns:
        The trained :class:`~src.rl.networks.ActorCritic`.
    """
    env = MarketEnv(features, prices)
    model = ActorCritic(
        observation_dim=int(env.observation_space.shape[0]), action_dim=int(env.action_space.n)
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    buffer = RolloutBuffer()

    obs, _info = env.reset()
    n_rollouts = max(config.total_timesteps // config.rollout_length, 1)

    for rollout_idx in range(n_rollouts):
        buffer.clear()
        obs = collect_rollout(env, model, buffer, obs, config.rollout_length)

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            _action, _log_prob, _entropy, last_value_tensor = model.get_action_and_value(
                last_obs_tensor
            )
        last_value = float(last_value_tensor.item())

        metrics = ppo_update(model, optimizer, buffer, last_value, config)
        mean_reward = float(np.mean(buffer.rewards))

        _log.info(
            "ppo_rollout_completed",
            rollout=rollout_idx + 1,
            total_rollouts=n_rollouts,
            mean_reward=mean_reward,
            **metrics,
        )

    return model


async def run_training_pipeline(
    *, lookback_days: int, total_timesteps: int, rollout_length: int, models_dir: Path = MODELS_DIR
) -> Path:
    """Fetch data, train, export -- the full offline PPO pipeline.

    Args:
        lookback_days: How many days of archived history to train on.
        total_timesteps: Total environment steps to train for.
        rollout_length: Steps collected per rollout, before each update.
        models_dir: Where to save the exported policy.

    Returns:
        Path to the exported TorchScript ``.pt`` file.
    """
    features, prices = await _fetch_aligned_dataset(lookback_days)
    _log.info("ppo_dataset_ready", timesteps=len(features))

    config = PPOConfig(total_timesteps=total_timesteps, rollout_length=rollout_length)
    model = run_ppo_training(features, prices, config)

    export_path = export_model(model, models_dir)
    _log.info("ppo_model_exported", export_path=str(export_path))
    print(f"Policy exported to: {export_path}")
    return export_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of archived history to train on.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps.")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Steps collected per rollout.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging(log_level="INFO", log_dir=None)
    args = _parse_args()
    asyncio.run(
        run_training_pipeline(
            lookback_days=args.lookback_days,
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length,
        )
    )


if __name__ == "__main__":
    main()

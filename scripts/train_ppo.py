"""Offline PPO training pipeline for the trading policy.

Not part of the live trading system -- run manually or on a schedule,
completely independent of every running agent::

    python scripts/train_ppo.py --symbol BTCUSDT --lookback-days 30 --total-timesteps 200000

``--symbol`` is REQUIRED: ``MARKET_DATA_L2`` archives every crypto symbol under one topic, and an
environment built from interleaved BTC and ETH rows trains on a price series that jumps between
two price levels. The rows are filtered to the chosen symbol before any alignment or training.

**No hold-out, no significance test.** This script trains on the whole window and exports the
policy; it does not evaluate it out of sample, so nothing here shows the policy beats chance. The
chronological split and the Hanley-McNeil gate added to ``train_model.py`` do not apply to an RL
rollout and have not been written for it.

**Aversion au risque face aux régimes toxiques (optionnelle, neutre par défaut).** Quatre leviers, voir :mod:`src.rl.risk_shaping` :
la récompense (taxe de toxicité proportionnelle à l'excès de VPIN, pénalité non linéaire des stops), l'état (VPIN, vitesse et accélération du VPIN, alignés
CAUSALEMENT sur ``--vpin-db``), la perte du critique (``--value-loss huber``, robuste aux événements de queue rares) et le facteur d'actualisation (``--gamma``,
plus bas = agent plus myope sur sa survie), plus un masque d'actions déterministe (``--kill-switch-vpin``) appliqué pendant l'entraînement ET à l'inférence.
Les observations passent de 6 à 9 valeurs avec ``--vpin-db`` : une politique entraînée ainsi n'est utilisable que par un agent qui fournit le VPIN (métadonnées
``ppo_policy_<ts>.meta.json`` écrites à côté du ``.pt``). Les seuils et poids sont des hypothèses à calibrer sur vos données, pas des valeurs mesurées.

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

(``--value-loss huber`` remplace le MSE par une perte de Huber ``SmoothL1`` : quadratique près de zéro, linéaire au-delà de ``huber_delta``, donc un retour
extrême ne fait plus exploser le gradient du critique.)
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
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

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
from src.rl.risk_shaping import RiskShapingConfig, vpin_derivatives
from src.rl.trading_env import OBSERVATION_DIM, OBSERVATION_DIM_WITH_VPIN, MarketEnv

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
        value_loss: ``"mse"`` (historical) or ``"huber"`` (``SmoothL1Loss``, robust to rare extreme returns).
        huber_delta: Transition point of the Huber loss, in return units (after the environment's reward scale).
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
    value_loss: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in ]0, 1] and gae_lambda in [0, 1].")
        if self.value_loss not in ("mse", "huber"):
            raise ValueError("value_loss must be 'mse' or 'huber'.")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be > 0.")


@dataclass
class RolloutBuffer:
    """One rollout's worth of trajectories, collected step by step."""

    observations: list[npt.NDArray[np.float64]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    action_masks: list[npt.NDArray[np.bool_]] = field(default_factory=list)

    def clear(self) -> None:
        """Empty the buffer, ready for the next rollout."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

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
        mask = env.action_mask()  # coupe-circuit déterministe : vrai partout si l'environnement n'en configure pas
        with torch.no_grad():
            action_tensor, log_prob_tensor, _entropy, value_tensor = model.get_action_and_value(
                obs_tensor, action_mask=torch.as_tensor(mask).unsqueeze(0)
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
        buffer.action_masks.append(mask)

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
    action_masks = torch.as_tensor(np.array(buffer.action_masks), dtype=torch.bool)

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
            batch_masks = action_masks[batch_idx]

            _action, new_log_probs, entropy, new_values = model.get_action_and_value(
                batch_obs, batch_actions, batch_masks
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

            if config.value_loss == "huber":
                value_loss = F.smooth_l1_loss(new_values, batch_returns, beta=config.huber_delta)  # quadratique près de 0, linéaire au-delà : un retour de queue ne fait pas exploser le gradient
            else:
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


def export_model(model: ActorCritic, models_dir: Path = MODELS_DIR, metadata: dict[str, Any] | None = None) -> Path:
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
        metadata: Written as ``<archive>.meta.json`` next to the policy (observation size and layout, whether it needs the
            VPIN, the risk-shaping settings it was trained with). Inference reads it to refuse a policy it cannot feed.

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
    if metadata is not None:
        export_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return export_path


def policy_metadata(observation_dim: int, uses_vpin: bool, shaping: RiskShapingConfig, config: PPOConfig) -> dict[str, Any]:
    """Ce que l'inférence doit savoir d'une politique pour la nourrir correctement (jamais de secret)."""
    return {
        "observation_dim": observation_dim,
        "uses_vpin": uses_vpin,
        "vpin_window_s": VPIN_WINDOW_S,
        "kill_switch_vpin": shaping.kill_switch_vpin,
        "risk_shaping": {
            "vpin_threshold": shaping.vpin_threshold, "toxicity_penalty_weight": shaping.toxicity_penalty_weight, "stop_loss_bps": shaping.stop_loss_bps,
            "tail_risk_weight": shaping.tail_risk_weight, "tail_alpha": shaping.tail_alpha, "safe_harbor_bonus": shaping.safe_harbor_bonus,
        },
        "gamma": config.gamma, "value_loss": config.value_loss,
    }


# -- VPIN : chargement et alignement causal -----------------------------------

VPIN_WINDOW_S: float = 60.0
VPIN_MAX_AGE_S: float = 600.0
MIN_TRAINING_ROWS: int = 200


def load_vpin_series(
    path: Path, symbol: str, start_s: float, end_s: float, bucket_size: float | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Séries ``(instants en secondes, VPIN)`` de ``symbol`` dans ``[start_s, end_s]``, lues dans l'archive du tracker (lecture seule).

    Seuls les seaux dont ``vpin_value`` est connu (fenêtre pleine) sont retenus. ``symbol`` est comparé sans casse ni séparateurs (``BTC/USDT`` = ``BTCUSDT``).

    **Taille de seau.** Deux VPIN calculés avec des seaux différents ne se comparent pas (un seau 50 fois plus petit sature l'indicateur). ``bucket_size`` ne garde que les
    seaux de cette taille ; sans lui, une archive qui mélange plusieurs tailles pour ce symbole est REFUSÉE (``ValueError``) au lieu d'être mélangée en silence.
    """
    if not path.is_file():
        raise FileNotFoundError(f"archive VPIN introuvable : {path}")
    wanted = "".join(ch for ch in symbol.upper() if ch.isalnum())
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT timestamp_ms, symbol, vpin_value, bucket_size FROM vpin_buckets WHERE vpin_value IS NOT NULL AND timestamp_ms >= ? AND timestamp_ms <= ? ORDER BY timestamp_ms, id",
            (int(start_s * 1000), int(end_s * 1000)),
        ).fetchall()
    finally:
        conn.close()
    same_symbol = [(ts / 1000.0, float(v), float(size)) for ts, sym, v, size in rows if "".join(ch for ch in str(sym).upper() if ch.isalnum()) == wanted]
    sizes = sorted({size for _, _, size in same_symbol})
    if bucket_size is None and len(sizes) > 1:
        raise ValueError(f"{symbol!r} : plusieurs tailles de seau dans l'archive ({', '.join(f'{x:,.0f}' for x in sizes)}) : indiquer laquelle utiliser (bucket_size), les mélanger fausserait le VPIN")
    kept = [(t, v) for t, v, size in same_symbol if bucket_size is None or size == float(bucket_size)]
    if not kept:
        raise ValueError(f"aucun VPIN calculé pour {symbol!r} dans {path} sur cette fenêtre")
    times, values = zip(*kept)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def longest_valid_run(valid: npt.NDArray[np.bool_]) -> slice:
    """La plus longue suite contiguë de ``True``. Garder des lignes éparses fabriquerait des sauts de prix entre deux lignes consécutives de l'environnement."""
    best_start = best_len = run_start = run_len = 0
    for i, ok in enumerate(valid):
        if ok:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    return slice(best_start, best_start + best_len)


def attach_vpin_features(
    row_times_s: npt.NDArray[np.float64], vpin_times_s: npt.NDArray[np.float64], vpin_values: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], slice]:
    """``(vpin_features, tranche)`` : niveau, vitesse et accélération causaux à chaque ligne, et la plus longue tranche où tout est connu.

    Lève ``ValueError`` si la tranche est trop courte pour entraîner (moins de :data:`MIN_TRAINING_ROWS` lignes).
    """
    features = vpin_derivatives(row_times_s, vpin_times_s, vpin_values, window_s=VPIN_WINDOW_S, max_age_s=VPIN_MAX_AGE_S)
    run = longest_valid_run(np.isfinite(features).all(axis=1))
    if run.stop - run.start < MIN_TRAINING_ROWS:
        raise ValueError(f"seulement {run.stop - run.start} lignes consécutives avec un VPIN connu (minimum {MIN_TRAINING_ROWS}) : l'archive VPIN est trop courte ou trouée")
    return features, run


# -- Orchestration ------------------------------------------------------


async def _fetch_aligned_dataset(
    lookback_days: int,
    symbol: str,
    vpin_db: Path | None = None,
    vpin_bucket_size: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Fetch and align TimescaleDB data into ``(features, prices)`` arrays for ONE symbol.

    Args:
        lookback_days: How many days of archived history to query.
        symbol: The instrument to train on (``BTC/USDT`` or ``BTCUSDT``). Rows of every other
            symbol are dropped before alignment (see :func:`scripts.train_model.build_market_feature_frame`).

    Returns:
        ``features`` (shape ``(T, 5)``, in
        :attr:`SupervisorMLModel.FEATURE_NAMES` order) and ``prices``
        (shape ``(T,)``, mid-price), aligned to the same ``T`` timesteps, and, if ``vpin_db`` is
        given, ``vpin_features`` (shape ``(T, 3)``: level, velocity, acceleration, CAUSAL). In that case the
        arrays are trimmed to the longest contiguous stretch where the VPIN is known (no gaps, no stale value).

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

    market_frame = build_market_feature_frame(market_records, symbol)
    if market_frame.empty:
        raise ValueError("No market data available for the requested window.")

    macro_frame = build_macro_feature_frame(macro_records)
    sentiment_frame = build_sentiment_feature_frame(nlp_records)
    aligned = align_market_macro_sentiment(market_frame, macro_frame, sentiment_frame)

    features = aligned[list(SupervisorMLModel.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    prices = aligned["mid_price"].to_numpy(dtype=np.float64)
    if vpin_db is None:
        return features, prices, None
    times_s: npt.NDArray[np.float64] = np.asarray(
        (pd.to_datetime(aligned["time"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1), dtype=np.float64
    )
    vpin_times, vpin_values = load_vpin_series(
        vpin_db, symbol, float(times_s.min()) - 2 * VPIN_WINDOW_S - VPIN_MAX_AGE_S, float(times_s.max()), vpin_bucket_size
    )
    vpin_features, run = attach_vpin_features(times_s, vpin_times, vpin_values)
    _log.info("ppo_vpin_aligned", rows_total=len(features), rows_kept=run.stop - run.start,
              vpin_p50=float(np.percentile(vpin_values, 50)), vpin_p90=float(np.percentile(vpin_values, 90)), vpin_p99=float(np.percentile(vpin_values, 99)))
    return features[run], prices[run], vpin_features[run]


def run_ppo_training(
    features: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    config: PPOConfig,
    *,
    vpin_features: npt.NDArray[np.float64] | None = None,
    risk_shaping: RiskShapingConfig | None = None,
) -> ActorCritic:
    """Run the full PPO training loop (collect -> GAE -> update, repeated).

    Args:
        features: Shape ``(T, 5)``, from :func:`_fetch_aligned_dataset`.
        prices: Shape ``(T,)``, from :func:`_fetch_aligned_dataset`.
        config: Hyperparameters.
        vpin_features: Optional ``(T, 3)`` VPIN level/velocity/acceleration (see :func:`attach_vpin_features`).
        risk_shaping: Optional risk-aversion settings (see :mod:`src.rl.risk_shaping`).

    Returns:
        The trained :class:`~src.rl.networks.ActorCritic`.
    """
    env = MarketEnv(features, prices, vpin_features=vpin_features, risk_shaping=risk_shaping)
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
                last_obs_tensor, action_mask=torch.as_tensor(env.action_mask()).unsqueeze(0)
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
    *,
    symbol: str,
    lookback_days: int,
    total_timesteps: int,
    rollout_length: int,
    models_dir: Path = MODELS_DIR,
    config: PPOConfig | None = None,
    vpin_db: Path | None = None,
    risk_shaping: RiskShapingConfig | None = None,
    vpin_bucket_size: float | None = None,
) -> Path:
    """Fetch data, train, export -- the full offline PPO pipeline.

    Args:
        symbol: The single instrument to train on (required: the archive mixes symbols).
        lookback_days: How many days of archived history to train on.
        total_timesteps: Total environment steps to train for.
        rollout_length: Steps collected per rollout, before each update.
        models_dir: Where to save the exported policy.
        config: PPO hyperparameters (``total_timesteps`` and ``rollout_length`` are taken from the arguments above).
        vpin_db: VPIN archive (``vpin_history.sqlite3``); enables the VPIN observation features and the VPIN-based risk shaping.
        risk_shaping: Risk-aversion settings; ``None`` = none (historical behaviour).

    Returns:
        Path to the exported TorchScript ``.pt`` file.
    """
    shaping = risk_shaping if risk_shaping is not None else RiskShapingConfig()
    features, prices, vpin_features = await _fetch_aligned_dataset(lookback_days, symbol, vpin_db, vpin_bucket_size)
    _log.info("ppo_dataset_ready", timesteps=len(features), uses_vpin=vpin_features is not None)

    base = config if config is not None else PPOConfig()
    config = PPOConfig(**{**base.__dict__, "total_timesteps": total_timesteps, "rollout_length": rollout_length})
    model = run_ppo_training(features, prices, config, vpin_features=vpin_features, risk_shaping=shaping)

    observation_dim = OBSERVATION_DIM_WITH_VPIN if vpin_features is not None else OBSERVATION_DIM
    export_path = export_model(model, models_dir, policy_metadata(observation_dim, vpin_features is not None, shaping, config))
    _log.info("ppo_model_exported", export_path=str(export_path))
    print(f"Policy exported to: {export_path}")
    return export_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="Instrument to train on, e.g. BTCUSDT or BTC/USDT (the archive mixes symbols).")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of archived history to train on.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps.")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Steps collected per rollout.")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor. 0.97 (lower than the historical 0.99) makes the agent more myopic about its survival.")
    parser.add_argument("--value-loss", choices=["mse", "huber"], default="huber", help="Critic loss. Huber is robust to the rare extreme returns of fat-tailed stops.")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--vpin-db", type=Path, default=None, help="vpin_history.sqlite3 : adds VPIN, velocity and acceleration to the observation (9 values instead of 6).")
    parser.add_argument("--vpin-bucket-size", type=float, default=None, help="only use VPIN buckets of this size (USDT notional). Required when the archive mixes several sizes for the symbol.")
    parser.add_argument("--vpin-threshold", type=float, default=0.7, help="VPIN above which holding a position is taxed.")
    parser.add_argument("--toxicity-weight", type=float, default=0.01, help="lambda of the toxicity tax (reward units per unit of VPIN excess); needs --vpin-db. 0 = off.")
    parser.add_argument("--safe-harbor-bonus", type=float, default=0.0, help="Reward for being flat while VPIN exceeds the threshold; needs --vpin-db.")
    parser.add_argument("--stop-loss-bps", type=float, default=None, help="Adds a stop-loss to the environment (forced exit when the loss since entry reaches it).")
    parser.add_argument("--tail-weight", type=float, default=1.0, help="Weight of the non-linear stop penalty; needs --stop-loss-bps.")
    parser.add_argument("--tail-alpha", type=float, default=1.5, help="Exponent of the stop penalty (1 = linear).")
    parser.add_argument("--kill-switch-vpin", type=float, default=None, help="VPIN above which BUY is masked (and a held position must exit); needs --vpin-db. Calibrate on the logged VPIN percentiles.")
    args = parser.parse_args()
    if args.vpin_db is None and (args.kill_switch_vpin is not None or args.safe_harbor_bonus > 0):
        parser.error("--kill-switch-vpin and --safe-harbor-bonus need --vpin-db (no VPIN, nothing to mask on).")
    if args.stop_loss_bps is None and args.tail_weight != 1.0:
        parser.error("--tail-weight needs --stop-loss-bps.")
    return args


def _shaping_from_args(args: argparse.Namespace) -> RiskShapingConfig:
    has_vpin = args.vpin_db is not None
    return RiskShapingConfig(
        vpin_threshold=args.vpin_threshold,
        toxicity_penalty_weight=args.toxicity_weight if has_vpin else 0.0,
        stop_loss_bps=args.stop_loss_bps,
        tail_risk_weight=args.tail_weight if args.stop_loss_bps is not None else 0.0,
        tail_alpha=args.tail_alpha,
        safe_harbor_bonus=args.safe_harbor_bonus,
        kill_switch_vpin=args.kill_switch_vpin,
    )


def main() -> None:
    """CLI entry point."""
    configure_logging(log_level="INFO", log_dir=None)
    args = _parse_args()
    asyncio.run(
        run_training_pipeline(
            symbol=args.symbol,
            lookback_days=args.lookback_days,
            total_timesteps=args.total_timesteps,
            rollout_length=args.rollout_length,
            config=PPOConfig(gamma=args.gamma, value_loss=args.value_loss, huber_delta=args.huber_delta),
            vpin_db=args.vpin_db,
            risk_shaping=_shaping_from_args(args),
            vpin_bucket_size=args.vpin_bucket_size,
        )
    )


if __name__ == "__main__":
    main()

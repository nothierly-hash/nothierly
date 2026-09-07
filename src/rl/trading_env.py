"""Gymnasium trading environment for offline PPO training.

:class:`MarketEnv` replays a fixed, pre-aligned historical dataset
(market/macro/NLP features plus the corresponding mid-price series --
see :mod:`scripts.train_model`'s data pipeline, reused by
:mod:`scripts.train_ppo` to build these arrays) as a Markov Decision
Process: at each step the agent observes the current market state plus
its own position, chooses HOLD/BUY/SELL, and receives a reward. This
module contains no PyTorch and no PPO-specific logic -- it is a pure
Gymnasium environment, usable by any RL algorithm.
"""

from __future__ import annotations

from typing import Any, Final, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt

# Action space (matches core.schemas.SupervisorAction's BUY/SELL/HOLD
# vocabulary, minus the position-agnostic framing: here SELL means "exit
# to flat", not "go short" -- this environment is long-only/flat, the
# same simplification the rest of this system makes (no short-position
# tracking exists anywhere yet).
ACTION_HOLD: Final[int] = 0
ACTION_BUY: Final[int] = 1
ACTION_SELL: Final[int] = 2

_N_MARKET_FEATURES: Final[int] = 5  # spread, obi, macro x2, sentiment -- see build_observation
_OBSERVATION_DIM: Final[int] = _N_MARKET_FEATURES + 1  # + current position

# Calibrated against real daily log-returns (EUR/USD, gold, BTC/USD all
# checked -- typical magnitude ~1e-3 to ~1e-2), not chosen arbitrarily.
# At the previous default (1.0), the volatility penalty this multiplies
# came out ~20x larger than a typical position_return on real daily FX
# data -- see MarketEnv's own "Scale warning" docstring for the full
# diagnosis (a real PPO run collapsed to exiting to flat on 91.4% of
# steps because of this, uniformly, not as a considered risk decision).
# 0.05 keeps the penalty a real, felt cost -- volatile position paths are
# still discouraged relative to smooth ones -- without it structurally
# outweighing the return the policy is trying to earn in the first
# place. Retune this (not just the call-site default) if reusing this
# environment against data with a materially different return scale
# (e.g. intraday bars, or a much higher-volatility instrument).
_DEFAULT_RISK_AVERSION: Final[float] = 0.05


class MarketEnv(gym.Env[npt.NDArray[np.float64], int]):
    """A discrete-action trading environment over a fixed historical dataset.

    Observation (``Box``, shape ``(6,)``, dtype ``float64``):
        ``[spread, order_book_imbalance, macro_us_10y_yield,
        macro_btc_exchange_flow, sentiment_score_signed, current_position]``
        -- the same 5 market/macro/NLP features
        :attr:`~src.agents.supervisor.ml_engine.SupervisorMLModel.
        FEATURE_NAMES` defines, plus the agent's own current position
        (``0.0`` flat, ``1.0`` long). The position is included because a
        trading decision is inherently state-dependent: "sell" only means
        something relative to whether you currently hold anything.

    Action (``Discrete(3)``): ``0`` = HOLD (keep current position),
        ``1`` = BUY (go/stay long), ``2`` = SELL (go/stay flat).

    Reward: **not raw PnL.** At each step:
        1. ``position_return = position_t * log(price_{t+1} / price_t)``
           -- the log-return actually earned, zero while flat.
        2. A risk penalty: ``risk_aversion * std(recent position_returns)``,
           the rolling realized volatility *of the position's own
           returns* (not the raw asset), subtracted from (1).
        3. A transaction cost of ``transaction_cost_pct`` (default
           ``0.0005`` = 0.05%), subtracted from (1)+(2) whenever the
           position actually changes this step (``0`` on HOLD-that-stays-
           HOLD, or BUY-while-already-long, etc.).

        ``reward = position_return - risk_aversion * volatility_penalty
        - transaction_cost``

        **Scale warning, from a real bug found and fixed this session**:
        ``position_return`` and ``volatility_penalty`` are both derived
        from the *same* per-step log-returns, so ``risk_aversion`` must
        be calibrated against their actual real-world magnitude, not
        assumed to be ``~O(1)``-comparable to ``position_return`` by
        default. A real daily EUR/USD series has typical log-returns on
        the order of ``1e-3`` and a rolling-std volatility_penalty of
        similar order -- at ``risk_aversion=1.0`` (this class's default
        until this fix), the penalty ends up roughly **20x larger** than
        a typical return, making ``reward`` negative on nearly every
        step regardless of position, even a profitable one. A policy
        trained against that reward has only one rational move: exit to
        flat and stay there -- which is exactly what happened (verified:
        a real PPO run against 26 years of real EUR/USD data converged
        to ``SELL`` (exit) on 91.4% of steps, uniformly across the whole
        history, not specifically during crashes). This was not a PPO
        bug or a learning failure -- the reward function left the policy
        no other rational option. See :data:`_DEFAULT_RISK_AVERSION`'s
        own docstring for the corrected default and how to recalibrate
        it for a different return scale (e.g. intraday data, or a
        higher-volatility instrument like BTC/USDT).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        features: npt.NDArray[np.float64],
        prices: npt.NDArray[np.float64],
        *,
        transaction_cost_pct: float = 0.0005,
        volatility_window: int = 20,
        risk_aversion: float = _DEFAULT_RISK_AVERSION,
        episode_length: int | None = None,
    ) -> None:
        """Configure the environment over a fixed, pre-aligned dataset.

        Args:
            features: Shape ``(T, 5)``, one row of market/macro/NLP
                features per timestep, in
                :attr:`~src.agents.supervisor.ml_engine.
                SupervisorMLModel.FEATURE_NAMES` order.
            prices: Shape ``(T,)``, the mid-price at each timestep
                (``features[t]`` and ``prices[t]`` describe the same
                moment).
            transaction_cost_pct: Fraction of notional deducted from the
                reward whenever the position changes (default ``0.0005``
                = 0.05%, as specified).
            volatility_window: Number of recent position-returns kept for
                the rolling volatility penalty.
            risk_aversion: Multiplier on the volatility penalty. Larger
                values make the policy more averse to holding through
                volatile stretches. Defaults to :data:`_DEFAULT_RISK_
                AVERSION` (``0.05``) -- see that constant's own
                docstring and this class's "Scale warning" above before
                raising this back toward ``1.0``: at that value, on real
                daily FX/commodity/crypto data, the penalty structurally
                outweighs the return being penalized, and training
                degenerates to "always exit to flat" regardless of what
                the market is actually doing.
            episode_length: Steps per episode before truncation. Defaults
                to using the entire dataset as one episode.

        Raises:
            ValueError: If ``features``/``prices`` have mismatched or
                invalid shapes, or there isn't enough data for even one
                step.
        """
        super().__init__()

        if features.ndim != 2 or features.shape[1] != _N_MARKET_FEATURES:
            raise ValueError(
                f"features must have shape (T, {_N_MARKET_FEATURES}), got {features.shape}."
            )
        if prices.ndim != 1 or len(prices) != len(features):
            raise ValueError(
                f"prices must have shape ({len(features)},) matching features, got {prices.shape}."
            )
        if len(features) < 3:
            raise ValueError("Need at least 3 timesteps of data to run one episode.")
        if transaction_cost_pct < 0:
            raise ValueError("transaction_cost_pct must not be negative.")
        if volatility_window < 2:
            raise ValueError("volatility_window must be >= 2.")

        self._features = features
        self._prices = prices
        self._transaction_cost_pct = transaction_cost_pct
        self._volatility_window = volatility_window
        self._risk_aversion = risk_aversion
        self._episode_length = episode_length or (len(features) - 1)

        # gym.Env declares action_space/observation_space at the base-class
        # level, so an explicit annotation here (matching the narrower type
        # actually assigned) is what lets external callers -- e.g.
        # scripts/train_ppo.py reading `env.action_space.n` -- see the
        # concrete Discrete/Box types instead of the base class's generic
        # Space[ActType]. The ignore below is a purely nominal mismatch:
        # gymnasium 1.0's `Discrete` is a non-generic class that
        # subclasses `Space[np.int64]`, while this class parameterizes
        # gym.Env's ActType as plain `int` (matching step()'s actual
        # signature, since every real caller passes a plain int) --
        # the two are functionally identical at runtime.
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(3)  # type: ignore[assignment]
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBSERVATION_DIM,), dtype=np.float64
        )

        self._position: int = 0
        self._current_step: int = 0
        self._episode_start: int = 0
        self._position_return_history: list[float] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Start a new episode at a random offset into the dataset.

        Args:
            seed: Optional RNG seed (forwarded to
                ``gymnasium.Env.reset``, which seeds ``self.np_random``).
            options: Unused; accepted for Gymnasium API compatibility.

        Returns:
            ``(observation, info)`` for the episode's first step.
        """
        super().reset(seed=seed)

        max_start = max(len(self._features) - self._episode_length - 1, 1)
        self._episode_start = int(self.np_random.integers(0, max_start))
        self._current_step = self._episode_start
        self._position = 0
        self._position_return_history = []

        return self._build_observation(), {}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply ``action``, advance one timestep, and compute the reward.

        Args:
            action: One of :data:`ACTION_HOLD`, :data:`ACTION_BUY`,
                :data:`ACTION_SELL`.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
            ``terminated`` is always ``False`` -- this MDP has no terminal
            state, only truncation once the episode or dataset is
            exhausted. ``info`` carries ``position``, ``log_return``, and
            ``transaction_cost`` for logging/debugging.

        Raises:
            ValueError: If ``action`` is not a valid discrete action.
        """
        if action not in (ACTION_HOLD, ACTION_BUY, ACTION_SELL):
            raise ValueError(f"Invalid action {action!r}; must be 0 (HOLD), 1 (BUY), or 2 (SELL).")

        previous_position = self._position
        if action == ACTION_BUY:
            new_position = 1
        elif action == ACTION_SELL:
            new_position = 0
        else:
            new_position = previous_position

        current_price = self._prices[self._current_step]
        next_price = self._prices[self._current_step + 1]
        log_return = float(np.log(next_price / current_price))
        position_return = float(new_position) * log_return

        self._position_return_history.append(position_return)
        if len(self._position_return_history) > self._volatility_window:
            self._position_return_history.pop(0)

        volatility_penalty = 0.0
        if len(self._position_return_history) >= 2:
            volatility_penalty = float(np.std(self._position_return_history, ddof=1))

        transaction_cost = (
            self._transaction_cost_pct if new_position != previous_position else 0.0
        )

        reward = position_return - self._risk_aversion * volatility_penalty - transaction_cost

        self._position = new_position
        self._current_step += 1

        terminated = False
        truncated = (
            self._current_step >= self._episode_start + self._episode_length
            or self._current_step >= len(self._prices) - 1
        )

        info = {
            "position": new_position,
            "log_return": log_return,
            "transaction_cost": transaction_cost,
            "volatility_penalty": volatility_penalty,
        }
        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self) -> npt.NDArray[np.float64]:
        """Assemble the current observation: market features + current position."""
        market_features = self._features[self._current_step]
        return np.concatenate([market_features, [float(self._position)]]).astype(np.float64)

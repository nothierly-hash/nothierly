"""RL sandbox environment (Chapter 12, Étape 3 + 3.5) -- research/ only,
never imported by any live agent or by src/rl/trading_env.py (production).

Built on the real Tardis order book data + real OBI/micro-price features
from alpha_features.py -- not simulated. Long/flat/short (not the
production MarketEnv's long/flat-only), since the brief specifically
asks for a Sell/Short action.

**Real calibration, not the literal numbers in the original brief**:
measured directly on the real data (see this module's own ``__main__``)
-- per-tick micro-price log-returns have mean-absolute magnitude ~1e-5
(0.001%), with 40% of ticks showing *zero* price change at all. The
original brief's literal passivity penalty (``-0.0001``/tick) was ~10x
that scale; recalibrated to :data:`_PASSIVITY_PENALTY_BASE`.

**Étape 3.5 update**: Étape 3's flat passivity penalty + pure Sortino
reward produced an agent that traded constantly (3,797 trades across 75
real episodes) and lost almost exactly its cumulative fee burden (mean
episode PnL -0.0253, matching 50.6 trades/episode x 0.0005 fee almost
exactly) -- a real, measured over-trading problem, not fixed by more
training, only by better-shaped incentives. Three changes, all still
grounded in real data (real OBI values gate the passivity penalty; real
entry/exit prices drive the trade-level reward; no fabricated inputs):

1. **Hurdle rate** (:func:`_trade_close_reward`): closing a trade whose
   *net-of-round-trip-fees* return is negative is penalized at
   :data:`_NET_LOSS_PENALTY_MULTIPLIER` x that loss -- a discrete,
   immediate signal on top of (not instead of) the ambient per-tick
   Sortino reward, specifically teaching "a trade that's gross-positive
   but net-negative after two-sided fees is still a loss."
2. **OBI-gated, non-linear passivity penalty** (:func:`_passivity_penalty`):
   zero when flat and the real OBI is near-neutral (nothing worth
   trading); grows exponentially, capped, the longer the agent stays
   flat while the real OBI sits past
   :data:`_OBI_OPPORTUNITY_THRESHOLD` (an obvious signal being ignored).
3. **Holding-time bonus**: a small per-tick bonus while a position is
   held unchanged past :data:`_HOLDING_BONUS_THRESHOLD_TICKS` *and* is
   currently unrealized-profitable -- rewards capturing a deeper move
   over scalping, directly counteracting the measured over-trading.

Also tracks win-rate net of fees per episode (``info["win_rate_net_of_fees"]``),
for :mod:`research.train_rl_sandbox`'s TensorBoard logging.
"""

from __future__ import annotations

from typing import Any, Final, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt

ACTION_HOLD: Final[int] = 0
ACTION_BUY: Final[int] = 1
ACTION_SELL: Final[int] = 2  # short, not "exit to flat" -- see module docstring

# --- Fees (Règle 1) ---
_TAKER_FEE: Final[float] = 0.0004  # 0.04%, as specified -- charged per leg (entry, exit each individually)
_NET_LOSS_PENALTY_MULTIPLIER: Final[float] = 2.0  # a net-of-fees losing trade is penalized 2x its loss

# --- Passivity penalty (Règle 2) ---
# Measured on the real Tardis sample (see module docstring): mean
# absolute per-tick micro-price log-return is ~1.0e-5. This is the base
# (near-zero) penalty when OBI is neutral -- effectively inert, only the
# OBI-gated exponential term below actually bites.
_PASSIVITY_PENALTY_BASE: Final[float] = 1e-7
_PASSIVITY_GROWTH_RATE: Final[float] = 1.15  # per consecutive missed-opportunity tick
_PASSIVITY_PENALTY_CAP: Final[float] = 5e-4  # never exceed one taker fee's worth, however long it's ignored
_OBI_OPPORTUNITY_THRESHOLD: Final[float] = 0.8  # |OBI| past this = "an obvious opportunity" per the brief

# --- Holding-time bonus (Règle 3) ---
_HOLDING_BONUS_THRESHOLD_TICKS: Final[int] = 20
_HOLDING_BONUS_PER_TICK: Final[float] = 3e-7

_DEFAULT_SORTINO_WINDOW: Final[int] = 50
_MIN_DOWNSIDE_DEVIATION: Final[float] = 1e-9  # floor to avoid a divide-by-zero when there are no losses yet


def rolling_sortino_ratio(returns: npt.NDArray[np.float64]) -> float:
    """Sortino ratio of a window of realized per-tick returns.

    Unlike Sharpe (which penalizes *all* volatility symmetrically --
    the exact miscalibration this session already found and fixed once
    in the production environment's plain-std-based penalty), Sortino
    divides by *downside* deviation only: upside volatility (a string of
    unusually large gains) is never penalized, only downside volatility
    is.

    Args:
        returns: 1D array of realized per-tick net returns (after fees).

    Returns:
        ``mean(returns) / downside_deviation(returns)``. ``0.0`` if
        ``returns`` is empty. Downside deviation is floored at
        :data:`_MIN_DOWNSIDE_DEVIATION` so an all-nonnegative window
        returns a large-but-finite positive ratio rather than ``inf``.
    """
    if returns.size == 0:
        return 0.0

    mean_return = float(np.mean(returns))
    downside_returns = returns[returns < 0]
    if downside_returns.size == 0:
        downside_deviation = _MIN_DOWNSIDE_DEVIATION
    else:
        downside_deviation = max(float(np.sqrt(np.mean(downside_returns**2))), _MIN_DOWNSIDE_DEVIATION)

    return mean_return / downside_deviation


class RlSandboxEnv(gym.Env[npt.NDArray[np.float64], int]):
    """Long/flat/short trading environment over real tick-level microstructure features.

    Observation (``Box``, shape ``(n_features + 1,)``): whatever feature
    columns the caller supplies (e.g. ``obi_depth1``, ``obi_depth5``,
    ``spread``, ``micro_price_momentum_5``) plus the agent's own current
    position (``-1.0`` short, ``0.0`` flat, ``1.0`` long) appended last.

    Action (``Discrete(3)``): :data:`ACTION_HOLD` (keep current
    position), :data:`ACTION_BUY` (go/stay long), :data:`ACTION_SELL`
    (go/stay **short**).

    Reward: the rolling Sortino ratio of realized net per-tick returns
    (ambient, continuous) plus two event-driven components -- see module
    docstring: the Règle 1 trade-closing hurdle-rate term (added the tick
    a trade closes) and the Règle 3 holding-time bonus (added every tick
    a profitable position is held past threshold). The Règle 2 OBI-gated
    passivity penalty is folded into the per-tick net return the Sortino
    ratio is computed from, same as the ambient transaction cost.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        features: npt.NDArray[np.float64],
        prices: npt.NDArray[np.float64],
        obi_values: npt.NDArray[np.float64],
        *,
        taker_fee: float = _TAKER_FEE,
        sortino_window: int = _DEFAULT_SORTINO_WINDOW,
        episode_length: int | None = None,
    ) -> None:
        """Configure the environment over a fixed, pre-aligned real dataset.

        Args:
            features: Shape ``(T, n_features)`` -- any real feature
                columns, e.g. from :mod:`research.alpha_features`.
            prices: Shape ``(T,)``, the micro-price at each timestep,
                aligned to ``features``.
            obi_values: Shape ``(T,)``, the real Order Book Imbalance at
                each timestep (see :func:`research.alpha_features.
                compute_obi`) -- used only to gate the Règle 2 passivity
                penalty; does not need to be one of ``features``'
                columns (though in practice it usually is).
            taker_fee: Per-leg transaction fee, as a fraction (e.g.
                ``0.0004`` = 0.04%, the default, as specified).
            sortino_window: How many trailing realized returns the
                ambient reward's rolling Sortino ratio is computed over.
            episode_length: Steps per episode before truncation.
                Defaults to the entire dataset as one episode.

        Raises:
            ValueError: If shapes mismatch or are too small, or any
                parameter is out of a sane range.
        """
        super().__init__()

        if features.ndim != 2:
            raise ValueError(f"features must be 2D (T, n_features), got shape {features.shape}.")
        if prices.ndim != 1 or len(prices) != len(features):
            raise ValueError(f"prices must have shape ({len(features)},) matching features, got {prices.shape}.")
        if obi_values.ndim != 1 or len(obi_values) != len(features):
            raise ValueError(f"obi_values must have shape ({len(features)},) matching features, got {obi_values.shape}.")
        if len(features) < sortino_window + 2:
            raise ValueError(f"Need at least sortino_window+2 ({sortino_window + 2}) timesteps, got {len(features)}.")
        if taker_fee < 0:
            raise ValueError("taker_fee must not be negative.")
        if sortino_window < 2:
            raise ValueError("sortino_window must be >= 2.")

        self._features = features
        self._prices = prices
        self._obi_values = obi_values
        self._n_features = features.shape[1]
        self._taker_fee = taker_fee
        self._sortino_window = sortino_window
        self._episode_length = episode_length or (len(features) - 1)

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(3)  # type: ignore[assignment]
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_features + 1,), dtype=np.float64
        )

        self._position: int = 0  # -1 short, 0 flat, 1 long
        self._current_step: int = 0
        self._episode_start: int = 0
        self._realized_returns: list[float] = []
        self._episode_trade_count: int = 0
        self._episode_pnl: float = 0.0
        self._entry_price: float | None = None
        self._consecutive_missed_opportunity_ticks: int = 0
        self._ticks_in_position: int = 0
        self._episode_closed_trades: int = 0
        self._episode_winning_trades_net_of_fees: int = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Start a new episode at a random offset into the real dataset."""
        super().reset(seed=seed)

        max_start = max(len(self._features) - self._episode_length - 1, 1)
        self._episode_start = int(self.np_random.integers(0, max_start))
        self._current_step = self._episode_start
        self._position = 0
        self._realized_returns = []
        self._episode_trade_count = 0
        self._episode_pnl = 0.0
        self._entry_price = None
        self._consecutive_missed_opportunity_ticks = 0
        self._ticks_in_position = 0
        self._episode_closed_trades = 0
        self._episode_winning_trades_net_of_fees = 0

        return self._build_observation(), {}

    def step(
        self, action: int
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply ``action``, advance one tick, and compute the shaped reward.

        Args:
            action: One of :data:`ACTION_HOLD`, :data:`ACTION_BUY`,
                :data:`ACTION_SELL`.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
            ``info`` carries ``position``, ``net_return``,
            ``transaction_cost``, ``passivity_penalty``,
            ``holding_bonus``, ``trade_close_reward``, ``trade_count``,
            ``episode_pnl``, and ``win_rate_net_of_fees`` (``None`` until
            this episode's first closed trade).

        Raises:
            ValueError: If ``action`` is not a valid discrete action.
        """
        if action not in (ACTION_HOLD, ACTION_BUY, ACTION_SELL):
            raise ValueError(f"Invalid action {action!r}; must be 0 (HOLD), 1 (BUY), or 2 (SELL).")

        previous_position = self._position
        if action == ACTION_BUY:
            new_position = 1
        elif action == ACTION_SELL:
            new_position = -1
        else:
            new_position = previous_position

        current_price = self._prices[self._current_step]
        next_price = self._prices[self._current_step + 1]
        log_return = float(np.log(next_price / current_price)) if current_price > 0 and next_price > 0 else 0.0
        position_return = float(new_position) * log_return

        position_changed = new_position != previous_position
        transaction_cost = self._taker_fee if position_changed else 0.0
        if position_changed:
            self._episode_trade_count += 1

        # --- Règle 1: hurdle-rate trade-closing reward ---
        trade_close_reward = self._compute_trade_close_reward(previous_position, current_price, position_changed)

        if position_changed:
            self._entry_price = current_price if new_position != 0 else None

        # --- Règle 2: OBI-gated, non-linear passivity penalty ---
        passivity_penalty = self._compute_passivity_penalty(new_position)

        # --- Règle 3: holding-time bonus ---
        holding_bonus = self._compute_holding_bonus(new_position, previous_position, current_price)

        net_return = position_return - transaction_cost - passivity_penalty + holding_bonus
        self._episode_pnl += net_return
        self._realized_returns.append(net_return)
        if len(self._realized_returns) > self._sortino_window:
            self._realized_returns.pop(0)

        ambient_reward = rolling_sortino_ratio(np.array(self._realized_returns, dtype=np.float64))
        reward = ambient_reward + trade_close_reward

        self._position = new_position
        self._current_step += 1

        terminated = False
        truncated = (
            self._current_step >= self._episode_start + self._episode_length
            or self._current_step >= len(self._prices) - 1
        )

        win_rate_net_of_fees = (
            self._episode_winning_trades_net_of_fees / self._episode_closed_trades
            if self._episode_closed_trades > 0
            else None
        )

        info = {
            "position": new_position,
            "net_return": net_return,
            "transaction_cost": transaction_cost,
            "passivity_penalty": passivity_penalty,
            "holding_bonus": holding_bonus,
            "trade_close_reward": trade_close_reward,
            "trade_count": self._episode_trade_count,
            "episode_pnl": self._episode_pnl,
            "win_rate_net_of_fees": win_rate_net_of_fees,
        }
        return self._build_observation(), reward, terminated, truncated, info

    def _compute_trade_close_reward(
        self, previous_position: int, current_price: float, position_changed: bool
    ) -> float:
        """Règle 1: discrete, asymmetric reward the tick a trade closes.

        A trade "closes" whenever the position changes away from a
        nonzero value (to flat, or flipping sign) -- computed from the
        real entry price recorded when that position was opened, to the
        real current price, net of the *round-trip* fee (entry + exit,
        i.e. ``2 * taker_fee`` -- the entry leg's fee was already
        deducted from ``net_return`` on the tick the position opened;
        this is a second, independent reward signal specifically about
        whether the completed trade cleared its full round-trip cost,
        not a second fee deduction from the account).
        """
        if not (position_changed and previous_position != 0 and self._entry_price is not None):
            return 0.0

        gross_trade_return = float(previous_position) * float(np.log(current_price / self._entry_price))
        net_trade_return = gross_trade_return - 2.0 * self._taker_fee

        self._episode_closed_trades += 1
        if net_trade_return > 0:
            self._episode_winning_trades_net_of_fees += 1
            return net_trade_return
        return net_trade_return * _NET_LOSS_PENALTY_MULTIPLIER

    def _compute_passivity_penalty(self, new_position: int) -> float:
        """Règle 2: zero when flat with neutral OBI; exponential (capped) when flat and OBI is obvious."""
        if new_position != 0:
            self._consecutive_missed_opportunity_ticks = 0
            return 0.0

        current_obi = float(self._obi_values[self._current_step])
        if abs(current_obi) <= _OBI_OPPORTUNITY_THRESHOLD:
            self._consecutive_missed_opportunity_ticks = 0
            return _PASSIVITY_PENALTY_BASE

        self._consecutive_missed_opportunity_ticks += 1
        penalty = _PASSIVITY_PENALTY_BASE * (_PASSIVITY_GROWTH_RATE**self._consecutive_missed_opportunity_ticks)
        return min(penalty, _PASSIVITY_PENALTY_CAP)

    def _compute_holding_bonus(self, new_position: int, previous_position: int, current_price: float) -> float:
        """Règle 3: small bonus for holding an unchanged, currently-profitable position past threshold."""
        if new_position == 0 or new_position != previous_position:
            self._ticks_in_position = 0
            return 0.0

        self._ticks_in_position += 1
        if self._ticks_in_position <= _HOLDING_BONUS_THRESHOLD_TICKS or self._entry_price is None:
            return 0.0

        unrealized_return = float(new_position) * float(np.log(current_price / self._entry_price))
        return _HOLDING_BONUS_PER_TICK if unrealized_return > 0 else 0.0

    def _build_observation(self) -> npt.NDArray[np.float64]:
        """Assemble the current observation: real features + current position."""
        feature_row = self._features[self._current_step]
        return np.concatenate([feature_row, [float(self._position)]]).astype(np.float64)

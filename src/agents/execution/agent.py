"""Execution micro-agent: the only thing in this system authorized to trade.

:class:`ExecutionAgent` implements the "Portfolio Manager" pattern: every
signal-generating agent (:class:`~src.agents.supervisor.agent.
XgboostSignalAgent`, :class:`~src.agents.rl_signal.agent.RlSignalAgent`)
publishes its opinion as a :class:`~core.schemas.TradingSignal` and never
touches a broker. This agent subscribes to every signal, and only enters
a position once **both** required agents agree, **recently**, on a
**non-neutral** direction. This is what prevents "Agent Collision": with
two independent decision-makers each capable of predicting BUY/SELL,
letting both place orders directly on the same OANDA account risked them
contradicting and fighting each other.

Once a position is open, exiting it is deliberately **asymmetric** and
**unilateral**: entry needs both agents to agree, but either one alone
saying "not this direction anymore" (neutral or the opposite) closes it
immediately, no consensus required -- consensus is a gate on taking risk,
never on shedding it. The same asymmetry governs every risk-management
exit below: a stop-loss, take-profit, or trailing-stop trigger also
closes immediately and is **never** subject to the global Kill Switch
(see :meth:`_close_position`'s docstring) -- exactly the same reasoning
:meth:`~src.agents.supervisor.broker_client.OandaClient.
cancel_all_orders` already followed.

One instance handles **every** configured instrument concurrently --
:attr:`_positions`, :attr:`_signals`, and :attr:`_last_prices` are all
keyed by instrument, so this agent can be long ``BTC/USDT`` and short
``EUR_USD`` at the same time, each with its own consensus state, exits,
and trailing-stop anchors evaluated fully independently. The only
account-level state shared *across* instruments is capital itself
(:attr:`_mock_balance`/:attr:`_current_exposure`) and
:data:`_MAX_CONCURRENT_TRADES`, a hard cap on how many instruments may be
held open at once -- position sizing already risks a fixed fraction of
*total* equity per trade (see :meth:`calculate_position_size`), so
without this cap, enough simultaneous positions could still compound into
outsized aggregate exposure.

ZMQ topics stay flat and shared across every instrument (``TRADING_SIGNALS``,
``MARKET_DATA_L2``), not per-instrument (e.g. ``market_data.BTC_USD``):
every message already carries its own ``instrument``/``symbol`` field, and
routing on that field -- exactly how per-agent-id consensus matching
already worked before this agent was multi-asset -- avoids a second,
parallel topic-per-symbol subscription scheme with no real benefit over
the one this bus already has.

Safety layers, in the order a would-be *entry* actually passes through
(*exits* skip layers 1 and 5 -- see above):

1. **Consensus + freshness** (this module): both required agents' latest
   signals must agree, non-neutral, within :attr:`_signal_validity_seconds`.
2. **Position + cooldown gate** (this module): a consensus that would
   just repeat the position already held is a no-op; a position closed
   within the last :attr:`_cooldown_seconds` refuses a new entry, to
   avoid whipsawing in and out on market noise right after an exit.
3. **Risk-based position sizing** (:meth:`calculate_position_size`):
   fixed-fractional risk (``account_balance * dynamic risk %``, read
   live from Redis, hard-capped) divided by the stop-loss distance --
   never a blind fraction-of-capital notional.
4. **Margin/leverage cap** (this module): the sized order is never
   allowed to exceed available margin.
5. **Flat portfolio exposure ceiling** (:class:`~src.agents.supervisor.
   risk_manager.RiskManager`): the order's projected notional, added to
   every other open position's, must not exceed
   ``max_portfolio_exposure_pct`` of capital -- instrument-agnostic.
6. **Correlated-cluster exposure ceiling** (:class:`~src.agents.
   supervisor.portfolio_risk_manager.PortfolioRiskParityManager`): a
   second, correlation-aware gate layered on top of layer 5 -- refuses an
   entry whose *correlated* cluster (this instrument plus any open
   position above the configured correlation threshold) would exceed a
   tighter ceiling reserved for concentration risk specifically. A
   no-op when the instrument has no correlated peer among open positions
   (layer 5 alone governs an isolated position's size).
7. **``MOCK_MODE``** (:class:`~core.config.AppSettings`): when true (the
   default), the broker is never called at all for an *entry* -- the
   intended order is logged and this agent's own position/balance
   bookkeeping still updates, but nothing crosses the network. Exits are
   simulated the same way, for the same MOCK_MODE-must-mean-MOCK-MODE
   reason.
8. **Global Kill Switch** (:mod:`core.security`) -- **entry only**: the
   real OANDA entry call is wrapped in both ``@requires_system_active``
   (on :meth:`_place_order` here) and :func:`~core.security.
   safe_financial_call` (``retry_on_failure=False`` -- an order
   placement must never be silently resubmitted).
9. **Per-agent remote control** (:class:`~core.base_agent.
   AgentControlState`) -- **entry only**, same asymmetry as the Kill
   Switch above and for the same reason (an operator pausing this
   agent must never freeze an already-open position in place):
   ``OFF`` refuses every new entry outright, before any sizing/risk
   work; ``SHADOW`` reuses the exact same simulated-fill path as
   ``MOCK_MODE`` (see layer 7) for that one entry, regardless of the
   process-wide ``MOCK_MODE`` setting. Which specific entry this was
   is remembered on the position itself (:attr:`PositionState.
   entered_as_simulated`), not re-derived from whatever this agent's
   control state happens to be when it later closes -- an operator
   flipping back to ``LIVE`` while a ``SHADOW``-opened position is
   still open must not cause :meth:`_close_position` to attempt a real
   broker call against a position that was never really opened there.

**"Protocole 7-Gates Convergence" (2026-09-13) -- explicit gate pipeline, display-layer renaming
only**: the 9 safety layers above are the same 9 real checks this agent has always made; this change
only reorganizes checks 3 through 9 into an explicit, single, sequentially-short-circuiting method
(:meth:`ExecutionAgent._validate_execution_gates`) that returns a structured, 7-gate verdict
(renumbered to this project's own "7 Gates" framing: Consensus, Cooldown, Sizing, Margin,
RiskManager, PortfolioRiskParityManager, Control State) instead of scattered inline ``if``/``return``
checks, and publishes that verdict for :mod:`src.dashboard.ems_pages.fleet_command_nexus`'s
telemetry. No gate's underlying math, exception type, or final trading decision changed -- only
where the check lives, and that it is now reported, live, gate-by-gate. One real, disclosed
behavior fix along the way: the manual-override path's own docstring already claimed "every other
entry safety layer still applies" once consensus is bypassed, but it never actually re-checked
cooldown before calling :meth:`_place_order` -- Gate 2 below is now the single, shared cooldown
check for every entry path, closing that gap for manual overrides too.

Two honest corrections to how this was originally briefed, matching this project's established
practice of disclosing rather than silently reinterpreting (see the "translate drama into honest
engineering" memory entry for this project): the brief asked for this telemetry to broadcast over
ZMQ -- this dashboard never subscribes to ZMQ directly (an existing, disclosed rule; see every other
``NOTHIERLY_*`` Redis key already published from this same module, e.g. :data:`_MARKET_STATE_REDIS_KEY`),
so the gate verdict is bridged through Redis instead, the identical pattern already used for every
other piece of dashboard telemetry this module publishes. And the brief asked for a class-name/
agent-id "lexical purge" of every AI/ML word (``ExecutionAgent`` -> ``Fleet_Command_Arbiter``,
``XgboostSignalAgent`` -> ``TensorGradient_Daemon``, ``RlSignalAgent`` -> ``MarkovInference_Daemon``)
-- the real class names and agent IDs are unchanged below (imported by 8 test files and
``scripts/run_backend.py``'s real production wiring); the three new names exist only as documented
**display** aliases used in this module's own gate telemetry (see ``_ARBITER_DISPLAY_NAME``/
``_DAEMON_DISPLAY_NAME`` below) -- the same "dramatic display name, honest real engineering
underneath" pattern this project's dashboard pages already established.

**"Protocole Stochastic Core" (2026-09-13)** added a new Gate 7, "StochasticCore" -- 4 real
stochastic-process models (:func:`~core.quant_math.hawkes_intensity`,
:func:`~core.quant_math.jump_diffusion_anomaly`,
:func:`~core.quant_math.ornstein_uhlenbeck_mean_reversion`,
:func:`~core.quant_math.price_impact`, see that module's docstrings for the real formulas), computed
from real, already-tracked market data -- never machine learning, exactly as briefed. See
:meth:`_gate_7_stochastic_core` for what each one actually gates and what stays informational.
**One disclosed renumbering**: the brief calls this "the 8th gate" (implying it comes *after* the
existing 7); this implementation places it at position 7 and shifts Control State to position 8
instead, preserving the pre-existing, safety-relevant invariant that Control State/the Kill Switch
must be the *last* gate evaluated before ``create_market_order`` -- the same reasoning the original
7-gate design already established (see the "Protocole 7-Gates Convergence" paragraph above). The
brief's own spirit (a real physics vector gates execution) is fully preserved; only the position
number shifts, and only to keep that invariant intact -- disclosed here rather than silently
reordered, the same practice this module's docstring already follows throughout.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Literal

import numpy as np
import numpy.typing as npt
import zmq.asyncio

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings
from core.logger import get_logger
from core.quant_math import (
    estimate_lyapunov_exponent,
    giant_cluster_fraction,
    hawkes_intensity,
    jump_diffusion_anomaly,
    kalman_dynamic_hedge_ratio,
    kolmogorov_compressibility_ratio,
    ornstein_uhlenbeck_mean_reversion,
    poiseuille_liquidity_resistance,
    price_impact,
    transfer_entropy,
    wasserstein_price_distance,
)
from core.redis_client import RedisUnavailableError
from core.schemas import (
    EMERGENCY_HALT_TOPIC,
    OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX,
    PORTFOLIO_GREEKS_REDIS_KEY,
    TRADING_SIGNALS_TOPIC,
    MarketMetrics,
    MessagePayload,
    OptionContract,
    OptionPosition,
    OptimalExecutionSchedule,
    PortfolioGreeks,
    TradeExecution,
    TradeSide,
    TradingSignal,
)
from core.security import (
    FinancialCallAbortedError,
    KillSwitchActiveError,
    requires_system_active,
    safe_financial_call,
)
from execution.order_flow_liquidity_agent import OrderFlowLiquidityAgent
from src.agents.guardian_ai import TriggerType, assess_risk
from src.agents.supervisor.broker_client import OandaClient
from src.agents.supervisor.optimal_execution_scheduler import OptimalExecutionScheduler
from src.agents.supervisor.options_hedging_manager import GreeksSnapshot, HedgeRecommendation, OptionsHedgingManager
from src.agents.supervisor.portfolio_risk_manager import CorrelatedExposureError, PortfolioRiskParityManager
from src.agents.supervisor.risk_manager import RiskLimitExceededError, RiskManager

_log = get_logger(__name__)

_AGENT_ID: Final[str] = "execution-agent"
_TRADE_EXECUTED_TOPIC: Final[str] = "TRADE_EXECUTED"
_MARKET_DATA_TOPIC: Final[str] = "MARKET_DATA_L2"
_DEFAULT_REQUIRED_AGENT_IDS: Final[frozenset[str]] = frozenset({"supervisor-agent", "rl-signal-agent"})

_RISK_PCT_REDIS_KEY: Final[str] = "TRADING_RISK_PCT"
_MAX_RISK_PCT: Final[float] = 0.05
_FALLBACK_RISK_PCT: Final[float] = 0.01

# Redis-backed, real, independent cap on position size (instrument
# units), read fresh on every real sizing call -- same "dashboard
# writes, agent reads live, no restart" pattern as _RISK_PCT_REDIS_KEY
# above. Written by src/dashboard/ems_pages/risk_management.py's own
# "Taille de position maximale" slider (that key name was already
# declared there, unused, before this real wiring closed the gap).
# Unset means uncapped (position size stays purely risk-derived, the
# original behavior) -- a real "no cap configured" state, not a 0.0
# sentinel that would nonsensically zero every order.
TRADING_MAX_POSITION_UNITS_REDIS_KEY: Final[str] = "TRADING_MAX_POSITION_UNITS"

# Default hard cap on how many *different instruments* may have an open
# position at once -- see the module docstring's multi-asset paragraph.
_DEFAULT_MAX_CONCURRENT_TRADES: Final[int] = 3

# Read by src/dashboard/app.py's "Nothierly AI Brainwaves" expander --
# a Redis LIST (LPUSH + LTRIM, newest first), not a pub/sub topic: this
# is dashboard-only telemetry, no other agent consumes it, so it doesn't
# need the bus's MessagePayload/topic-name validation machinery.
_DECISION_LOG_REDIS_KEY: Final[str] = "NOTHIERLY_DECISION_LOGS"
_DECISION_LOG_MAX_ENTRIES: Final[int] = 50

# Read by src/dashboard/app.py's "RADAR UNIVERSEL" section -- each is a
# single Redis string holding one JSON blob (not a Redis hash/list): a
# dict-of-records (NOTHIERLY_MARKET_STATE, keyed by instrument) and a
# list-of-records (NOTHIERLY_ACTIVE_POSITIONS) respectively, each shaped
# so `pd.DataFrame(list(state.values()))` / `pd.DataFrame(positions)`
# turns it directly into a table -- see generate_decision_report's
# sibling methods below (_update_market_state/_publish_active_positions)
# for the exact record shape.
_MARKET_STATE_REDIS_KEY: Final[str] = "NOTHIERLY_MARKET_STATE"
_ACTIVE_POSITIONS_REDIS_KEY: Final[str] = "NOTHIERLY_ACTIVE_POSITIONS"

# Read by src/dashboard/app.py's scrolling raw-signal telemetry terminal
# (the "Nothierly - Quantum Control" theme) -- a Redis LIST (LPUSH +
# LTRIM, newest first), one entry per individual TradingSignal *received*
# (every agent, every instrument, before any consensus/freshness
# filtering) -- deliberately more granular than NOTHIERLY_DECISION_LOGS
# above, which only records consensus-level outcomes. The dashboard
# still never subscribes to ZMQ directly (see the module docstring in
# src/dashboard/app.py) -- this Redis list is what makes "raw signal"
# telemetry visible there without breaking that boundary.
_RAW_SIGNAL_FEED_REDIS_KEY: Final[str] = "NOTHIERLY_RAW_SIGNAL_FEED"
_RAW_SIGNAL_FEED_MAX_ENTRIES: Final[int] = 100

# Read by src/dashboard/app.py's Guardian AI alert banner -- a Redis LIST
# (LPUSH + LTRIM, newest first), populated by src.agents.guardian_ai.
# assess_risk's verdicts for triggers 3 (drawdown) and 4 (signal
# collision), detected here. Triggers 1 (risk increase) and 2 (manual
# override) are detected in src/dashboard/app.py instead (that's where
# the human action they respond to actually happens) and pushed to this
# same key directly from there -- see this module's docstring and
# _trigger_guardian_alert's for the full trigger-to-detector map.
_GUARDIAN_ALERTS_REDIS_KEY: Final[str] = "NOTHIERLY_GUARDIAN_ALERTS"
_GUARDIAN_ALERTS_MAX_ENTRIES: Final[int] = 30

# Read by src/dashboard/app.py's equity curve chart -- a Redis LIST
# (LPUSH + LTRIM, newest first), one entry per real (timestamp, balance)
# sample -- the exact same samples _check_drawdown_trigger below already
# records into self._balance_history for its own purposes, just also
# persisted here so the dashboard can plot genuine account-balance
# history instead of a fabricated demo curve. Never backfilled or
# interpolated: a quiet MOCK_MODE session with few trades simply
# produces a sparse curve, which is the honest picture, not something to
# paper over with synthetic points.
_EQUITY_CURVE_REDIS_KEY: Final[str] = "NOTHIERLY_EQUITY_CURVE"
_EQUITY_CURVE_MAX_ENTRIES: Final[int] = 500

# Read by src/dashboard/ems_pages/order_blotter.py -- a Redis LIST
# (LPUSH + LTRIM, newest first), one entry per real fill (entry or exit)
# this agent executes, in either MOCK_MODE or live. Deliberately a
# separate list from NOTHIERLY_DECISION_LOGS above: that one records
# consensus-level *decisions* (including "no entry, vetoed"); this one
# only records an actual fill, at the granularity a TCA/blotter view
# needs (price, latency, direction) rather than the human-readable
# reasoning report. There is no Maker/Taker-skew concept in this
# production path (see _log_fill's docstring) -- this agent only ever
# places simple market orders.
_FILLS_BLOTTER_REDIS_KEY: Final[str] = "NOTHIERLY_FILLS_BLOTTER"
_FILLS_BLOTTER_MAX_ENTRIES: Final[int] = 300

# Trigger 3 (drawdown): a rolling 24h window; 5% (the real default --
# see DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY below for the live override)
# peak-to-current capital loss within it raises the alert once,
# re-arming only once the drawdown recovers below a lower hysteresis
# threshold (avoids re-alerting on every single tick while the balance
# hovers right at the alert threshold).
_DRAWDOWN_WINDOW_SECONDS: Final[float] = 24 * 3600.0
DRAWDOWN_ALERT_THRESHOLD_PCT: Final[float] = 0.05
# The re-arm threshold is derived from the (possibly overridden) alert
# threshold at this same ratio the original fixed 3%/5% pair already
# used, rather than staying independently fixed -- an operator who
# lowers the alert threshold to, say, 2% should not keep a re-arm gate
# fixed at the old 3% (wider than the new alert threshold itself,
# which would leave the alert unable to ever re-arm below the new,
# smaller trigger point).
_DRAWDOWN_REARM_RATIO: Final[float] = 0.6

# Redis-backed live override for DRAWDOWN_ALERT_THRESHOLD_PCT, read fresh
# on every real balance observation -- the same "dashboard writes, agent
# reads live, no restart" pattern _get_dynamic_risk_pct/_RISK_PCT_REDIS_KEY
# and src.agents.alert_agent's OBI_THRESHOLD_OVERRIDE_REDIS_KEY already
# establish. Written by src/dashboard/ems_pages/red_phone.py's drawdown
# slider (see that page's own former "not yet implemented" note, now
# closed). Clamped at both ends so a corrupted/malicious Redis value can
# never disable the trigger (ceiling) or make it fire on routine balance
# noise (floor).
DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY: Final[str] = "DRAWDOWN_ALERT_THRESHOLD_PCT_OVERRIDE"
_MINDRAWDOWN_ALERT_THRESHOLD_PCT: Final[float] = 0.01
_MAXDRAWDOWN_ALERT_THRESHOLD_PCT: Final[float] = 0.5

# See _run_heartbeat_loop: this agent is purely event-driven (no tick
# loop), so BaseAgent._write_heartbeat's own TTL formula
# (max(3 * tick_interval_seconds, 5.0), and this agent never sets
# tick_interval_seconds to anything meaningful) resolves to a flat 5.0s
# floor -- written well inside that margin, not derived from it.
_HEARTBEAT_WRITE_INTERVAL_SECONDS: Final[float] = 2.0

# Trigger 4 ("Collision", signal flip-flop): more than 3 conflicting
# consensus outcomes (see _evaluate_consensus's "conflict" outcome) for
# the *same* instrument within a rolling 10-minute window.
_COLLISION_WINDOW_SECONDS: Final[float] = 10 * 60.0
_COLLISION_THRESHOLD: Final[int] = 3

# Redis Pub/Sub channel (NOT a ZMQ topic -- see _listen_for_manual_trade_
# requests's docstring) the dashboard PUBLISHes a manual-override request
# to once a human has clicked through the Guardian AI's "execute anyway"
# confirmation (trigger 2). One-shot commands, not persisted state, so
# Pub/Sub (fire-and-forget, no backlog for a not-yet-running subscriber)
# is the right Redis primitive here -- unlike every *_REDIS_KEY above,
# which are LIST/STRING state meant to be read back later.
_MANUAL_TRADE_CHANNEL: Final[str] = "NOTHIERLY_MANUAL_TRADE_CHANNEL"

# Short, human-readable flavor text for each known signal agent's
# _describe_stance() line below. Purely cosmetic -- see that method's
# docstring on why this is templated language, not real interpretability.
_STANCE_FLAVOR: Final[dict[str, str]] = {
    "supervisor-agent": "statistical pattern inference from recent market/macro/sentiment features",
    "rl-signal-agent": "policy-network reward optimization over the current state",
}

# -- Fleet Command Arbiter display layer (2026-09-13) -----------------------
#
# Documented display-only aliases, never a rename of the real, imported
# class/agent-id: see this module's own docstring addendum ("Protocole
# 7-Gates Convergence") on why. Used only in this module's own gate
# telemetry (see _publish_gate_report) -- the same "dramatic display
# name, honest real engineering underneath" pattern this project's
# dashboard pages already established (see src/dashboard/ems_pages/
# fleet_command_nexus.py's own docstring).
_ARBITER_DISPLAY_NAME: Final[str] = "Fleet_Command_Arbiter"
_DAEMON_DISPLAY_NAME: Final[dict[str, str]] = {
    "supervisor-agent": "TensorGradient_Daemon",
    "rl-signal-agent": "MarkovInference_Daemon",
}

# Redis key the 8-gate telemetry below is published to -- one JSON record
# per instrument (overwritten on every real gate evaluation: a live
# snapshot, the same shape/pattern as _MARKET_STATE_REDIS_KEY above, not
# an append-only log). Read by src/dashboard/ems_pages/fleet_command_nexus.py's
# real-production gate panel. **Not literally ZMQ**, though the brief
# asked for that -- see the module docstring's honest-correction
# paragraph on why this reuses the same Redis-bridge convention every
# other *_REDIS_KEY constant in this file already follows.
_GATE_TELEMETRY_REDIS_KEY: Final[str] = "NOTHIERLY_FLEET_COMMAND_GATES"

_GATE_NAMES: Final[tuple[str, str, str, str, str, str, str, str]] = (
    "Consensus",
    "Cooldown",
    "Sizing",
    "Plafond de Marge",
    "RiskManager",
    "PortfolioRiskParityManager",
    "StochasticCore",
    "Control State",
)

# -- Gate 7 "StochasticCore" (2026-09-13, "Protocole Stochastic Core") -----
#
# Real model parameters, not fitted per-instrument (no real, sufficiently
# granular training pipeline exists for that yet -- an honest, disclosed
# limitation, the same kind this file's other docstrings already flag
# rather than silently pretend away). See _gate_7_stochastic_core for how
# each is actually used.
_HAWKES_BASELINE_INTENSITY: Final[float] = 0.1
_HAWKES_DECAY_RATE: Final[float] = 0.5  # per second: one event's excitation halves in ~1.4s
_HAWKES_EXCITATION_STRENGTH: Final[float] = 2.0
# "Cascade" (Daemon 1) is informational telemetry, not a blocking
# condition -- see _gate_7_stochastic_core's docstring on why only
# Daemons 2 and 4 (jump-diffusion, price impact) actually veto an entry,
# matching the brief's own blocking language ("verrouille"/"annulé"),
# which it never uses for Daemons 1/3.
_HAWKES_CASCADE_THRESHOLD_MULTIPLIER: Final[float] = 5.0
_TICK_EVENT_HISTORY_MAXLEN: Final[int] = 200

_JUMP_DIFFUSION_THRESHOLD_STD: Final[float] = 4.0
_MIN_SAMPLES_FOR_STOCHASTIC_CORE: Final[int] = 5  # jump_diffusion_anomaly needs >= 4 log-returns

_PRICE_IMPACT_COEFFICIENT: Final[float] = 1.0

# -- Gate 7 additions (2026-09-13, "Arsenal Interdisciplinaire") -- both real, informational only,
# same non-blocking status as Hawkes/OU above (see this file's own honest split of the briefed
# catalogue into what's rigorous vs. metaphor-only, in core.quant_math's module docstring).
_MIN_SAMPLES_FOR_WASSERSTEIN_DRIFT: Final[int] = 8  # need >= 4 real returns in each half-window
_MIN_SAMPLES_FOR_COMPRESSIBILITY: Final[int] = 16  # too few samples make a compression ratio meaningless noise

# -- Gate 7 addition (2026-09-13, "Time & Causality") -- real, informational only, same non-blocking
# status as every other Gate 7 addition above. The bare technical minimum for
# estimate_lyapunov_exponent's own default embedding/window is much lower (~9 points); this is
# chosen for a minimally *stable* real estimate, not just to avoid a ValueError -- see that
# function's own docstring on why a short, noisy window's estimate should be trusted even less than
# a longer one.
_MIN_SAMPLES_FOR_LYAPUNOV: Final[int] = 30
_LYAPUNOV_MAX_TIME_STEPS: Final[int] = 5

# -- Gate 7 addition (2026-09-14, "Signal, Physique, Algo") -- real, informational only. Fixed at
# 1.0 (a real, disclosed normalization reference, not a calibrated seconds/units value): this
# resistance reading is only ever meant for relative comparison across time for the same
# instrument (see poiseuille_liquidity_resistance's own docstring), not as an absolute cost figure.
_POISEUILLE_EXECUTION_HORIZON: Final[float] = 1.0

# -- Gate 6 addition (2026-09-13, "Arsenal Interdisciplinaire") -- reuses this class's own
# correlation_threshold (see PortfolioRiskParityManager's constructor) as the percolation edge
# threshold, informational only -- see _gate_6_portfolio_risk_parity's own docstring.


@dataclass(frozen=True)
class GateResult:
    """One of the 8 hardcoded execution gates' real, live verdict for one proposed entry.

    ``status`` is ``"PENDING"`` -- never silently defaulted to ``"PASSED"``
    -- for every gate downstream of the first one that failed. See
    :meth:`ExecutionAgent._validate_execution_gates`'s docstring for why
    a later gate is never evaluated once an earlier one has already
    closed the door.
    """

    number: int
    name: str
    status: Literal["PASSED", "FAILED", "PENDING"]
    detail: str


@dataclass(frozen=True)
class GateReport:
    """The full, sequential 8-gate verdict for one proposed entry.

    See :meth:`ExecutionAgent._validate_execution_gates`. ``final_action``
    is what this verdict actually authorizes: ``"REJECTED"`` if any gate
    failed; ``"MOCK_ORDER"``/``"SHADOW_ORDER"`` if every gate passed but
    Gate 8 (Control State) resolved to a simulated fill; ``"LIVE_ORDER"``
    if every gate passed and Gate 8 resolved to a real broker call.
    """

    instrument: str
    direction: int
    timestamp: float
    gates: tuple[GateResult, ...]
    final_action: Literal["LIVE_ORDER", "SHADOW_ORDER", "MOCK_ORDER", "REJECTED"]

    @property
    def all_passed(self) -> bool:
        return all(gate.status == "PASSED" for gate in self.gates)


@dataclass
class PositionState:
    """One instrument's current position, mutable in place as it evolves.

    Attributes:
        status: ``"FLAT"``, ``"LONG"``, or ``"SHORT"``.
        entry_price: Fill price of the current position, or ``None``
            while flat.
        units_held: Signed quantity of the current position (positive
            long, negative short), or ``None`` while flat. A whole
            number for a real OANDA order (that API requires integer
            units); may be fractional under ``MOCK_MODE`` -- see
            :meth:`ExecutionAgent._place_order`'s docstring on why a
            high-price-per-unit instrument like ``BTC/USDT`` needs that.
        highest_seen_price: Highest price observed since entering a LONG
            position -- the trailing-stop anchor. Reset to ``None``
            whenever the position returns to flat.
        lowest_seen_price: Lowest price observed since entering a SHORT
            position -- the trailing-stop anchor. Reset to ``None``
            whenever the position returns to flat.
        trailing_activated: Whether the position has *ever* reached
            the profit threshold that hands stop duty from the static
            stop-loss to the trailing stop. A one-way latch, deliberately:
            it must stay ``True`` once set even if the price later pulls
            back below that threshold again -- a trailing stop that
            "de-activates" during exactly the pullback it exists to
            catch would defeat its own purpose. Reset to ``False``
            whenever the position returns to flat.
        cooldown_until: A ``time.time()`` timestamp before which a new
            *entry* is refused, set on every close. ``None`` means no
            cooldown in effect.
        opened_at: ``time.time()`` when this position was entered, or
            ``None`` while flat. Dashboard telemetry only (see
            :meth:`ExecutionAgent._publish_active_positions`) -- no
            trading logic reads this.
        entered_as_simulated: Whether *this specific entry* was
            simulated (``self._mock_mode`` or ``AgentControlState.
            SHADOW`` at the moment it was placed) rather than a real
            broker call. Stored on the position itself, not
            re-derived from current state at close time, deliberately:
            unlike ``self._mock_mode`` (fixed for an agent's whole
            process lifetime), ``control_state`` can change mid-
            position (an operator flipping SHADOW back to LIVE while a
            simulated position is still open) -- :meth:`_close_position`
            must close the same way the position was actually opened,
            never re-evaluate against whatever state happens to be
            current when it closes (a real, load-bearing correctness
            fix: closing a position that was never really opened at the
            broker via a real order would either fail outright or,
            worse, silently create real, inverted exposure).
        entered_as_shadow_controlled: Whether ``AgentControlState.
            SHADOW`` specifically (independent of ``self._mock_mode``)
            was the reason this entry was simulated -- stored
            separately from :attr:`entered_as_simulated` so
            :data:`_FILLS_BLOTTER_REDIS_KEY` entries can honestly label
            *why* a fill was simulated, for both the entry and the
            eventual exit.
    """

    status: Literal["FLAT", "LONG", "SHORT"] = "FLAT"
    entry_price: float | None = None
    units_held: float | None = None
    highest_seen_price: float | None = None
    lowest_seen_price: float | None = None
    trailing_activated: bool = False
    cooldown_until: float | None = None
    opened_at: float | None = None
    entered_as_simulated: bool = False
    entered_as_shadow_controlled: bool = False


@dataclass(frozen=True)
class _ConsensusResult:
    """Outcome of one :meth:`ExecutionAgent._evaluate_consensus` call, for logging."""

    outcome: Literal["waiting", "conflict", "consensus"]
    signals: dict[str, TradingSignal]
    direction: int = 0


class ExecutionAgent(BaseAgent):
    """Arbitrates consensus for entries, unilateral vetoes and risk limits for exits.

    See the module docstring for the full safety-layer ordering and the
    entry/exit asymmetry.
    """

    def __init__(
        self,
        instruments: Iterable[str],
        *,
        required_agent_ids: frozenset[str] = _DEFAULT_REQUIRED_AGENT_IDS,
        signal_validity_seconds: float = 5.0,
        capital: float = 10_000.0,
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.03,
        trailing_stop_pct: float = 0.015,
        trailing_activation_pct: float = 0.005,
        cooldown_seconds: float = 60.0,
        max_concurrent_trades: int = _DEFAULT_MAX_CONCURRENT_TRADES,
        max_portfolio_exposure_pct: float | None = None,
        oanda_environment: Literal["practice", "live"] | None = None,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Wire up the OANDA trading client and risk manager; subscribe to signals + prices.

        Args:
            instruments: Every instrument this instance may trade (e.g.
                ``{"EUR_USD", "BTC/USDT"}``). Signals/price ticks for any
                other instrument are ignored. Each is tracked fully
                independently -- see the module docstring's multi-asset
                paragraph.
            required_agent_ids: Every one of these agent IDs must have
                published a fresh, agreeing, non-neutral signal for
                consensus to be reached. Defaults to this project's two
                signal agents.
            signal_validity_seconds: A stored signal older than this (by
                its own ``timestamp``, wall-clock seconds) is treated as
                expired -- as good as never having arrived.
            capital: Starting ``MOCK_MODE`` account balance, and the
                risk manager's exposure-limit denominator. Ignored for
                the account-balance figure once ``MOCK_MODE`` is false
                (the real OANDA balance is queried live instead). Shared
                across every instrument -- this is total account equity,
                not a per-instrument allocation.
            stop_loss_pct: Static stop-loss distance from entry price,
                as a fraction (``0.015`` = 1.5%) -- the *only* stop in
                effect until the position has moved ``trailing_activation_pct``
                into profit (see :meth:`_evaluate_price_triggered_exit`).
            take_profit_pct: Hard take-profit distance from entry price,
                as a fraction. Always active, independent of the
                trailing stop (protects against a flash-spike wick).
            trailing_stop_pct: Once activated, how far the trailing stop
                trails behind the position's best-seen price, as a
                fraction.
            trailing_activation_pct: The position must be at least this
                far into profit (relative to entry) before the trailing
                stop takes over from the static stop-loss -- avoids
                being stopped out by noise right after entry.
            cooldown_seconds: How long after a close an instrument
                refuses a new entry.
            max_concurrent_trades: Hard cap on how many *different*
                instruments may have a non-``FLAT`` position open at
                once. A consensus that would open a new instrument past
                this cap is refused (existing positions are never force-
                closed to make room). See the module docstring.
            max_portfolio_exposure_pct: Passed to :class:`~src.agents.
                supervisor.risk_manager.RiskManager` as its
                ``max_portfolio_exposure_pct`` -- the hard ceiling on
                *total* notional exposure across every open position, as
                a fraction of ``capital``, checked against the order's
                actual projected notional before every entry (see
                :meth:`_place_order`). Defaults (when ``None``) to
                ``1.0``, that class's own maximum allowed value (its
                constructor requires ``0 < value <= 1`` -- no aggregate
                leverage beyond 1x capital). Note this is a **second,
                independent** brake layered on top of
                ``max_concurrent_trades``, not a multiple of it: fixed-
                fractional risk sizing (:meth:`calculate_position_size`)
                makes one trade's notional ``capital * risk_pct /
                stop_loss_pct`` regardless of instrument price -- at this
                method's own defaults (1% risk, 1.5% stop) that is
                already ~67% of capital *per trade*, so even at
                ``max_portfolio_exposure_pct=1.0`` only about one
                default-sized position fits before this ceiling (not
                ``max_concurrent_trades``) becomes the binding
                constraint; a lower live risk-per-trade (the dashboard
                slider) or a wider stop-loss is what actually lets more
                than one slot fill concurrently.
            oanda_environment: Overrides :class:`~core.config.
                OandaSettings.environment` if given; otherwise read from
                configuration (``"practice"`` by default).
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`). Also where
                ``MOCK_MODE`` (:attr:`~core.config.AppSettings.
                mock_mode`) is read from.
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
        """
        super().__init__(
            _AGENT_ID,
            subscribed_topics=(TRADING_SIGNALS_TOPIC, _MARKET_DATA_TOPIC, EMERGENCY_HALT_TOPIC),
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._instruments = frozenset(instruments)
        self._required_agent_ids = required_agent_ids
        self._signal_validity_seconds = signal_validity_seconds
        self._capital = capital
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self._trailing_stop_pct = trailing_stop_pct
        self._trailing_activation_pct = trailing_activation_pct
        self._cooldown_seconds = cooldown_seconds
        self._max_concurrent_trades = max_concurrent_trades
        self._risk_manager = RiskManager(
            max_portfolio_exposure_pct=(
                max_portfolio_exposure_pct if max_portfolio_exposure_pct is not None else 1.0
            )
        )
        # Phase 3 ("Le Stratège"): a second, correlation-aware exposure
        # gate layered on top of _risk_manager's flat ceiling above --
        # see that class's own module docstring for why a flat ceiling
        # alone misses concentrated-but-correlated positions. Price
        # history is fed from every MARKET_DATA_L2 tick (see
        # _handle_price_message).
        self._portfolio_risk_manager = PortfolioRiskParityManager()
        # "Le Prompt Architecte" (2026-09-13): a real Almgren-Chriss
        # meta-order scheduler -- see compute_optimal_trajectory's own
        # docstring for the real, disclosed scope boundary (computes a
        # real, usable schedule; does not itself slice/place orders).
        self._execution_scheduler = OptimalExecutionScheduler()
        # "Options & Multi-Instrument Hedging" (2026-09-13): real (paper-
        # filled) options tracking + real Ashby's-Law hedge solving --
        # see compute_hedge_recommendation's own docstring for the real,
        # disclosed execution boundary (this agent's real broker call
        # remains spot/CFD-only; no real options order is ever placed).
        self._options_hedging_manager = OptionsHedgingManager()
        # "Order-Flow Liquidity Provision" (2026-09-13): real, legitimate,
        # opportunistic counterpart to execution/predator_honeypot.py's
        # purely defensive posture -- see OrderFlowLiquidityAgent's own
        # module docstring for the real legal boundary (public
        # order-book-imbalance inference only, never a specific known
        # counterparty order). Fed every real tick in
        # _handle_price_message; Gate 7 surfaces its recommendation as
        # informational telemetry only -- it does not itself resize or
        # place any real order.
        self._order_flow_agent = OrderFlowLiquidityAgent()
        self._mock_mode = self._settings.mock_mode
        self._mock_balance = capital

        oanda_settings = self._settings.oanda
        environment = oanda_environment or oanda_settings.environment
        self._sandbox = environment == "practice"
        self._trading_client = OandaClient(
            api_token=oanda_settings.api_token.get_secret_value(),
            account_id=oanda_settings.account_id,
            environment=environment,
        )

        # Sliding-window signal memory: instrument -> agent_id -> its
        # latest signal for that instrument. Each instrument's consensus
        # is evaluated entirely independently of every other's.
        self._signals: dict[str, dict[str, TradingSignal]] = {}
        self._positions: dict[str, PositionState] = {}
        self._last_prices: dict[str, float] = {}
        self._current_exposure = 0.0

        # Gate 7 "StochasticCore" real inputs -- see _gate_7_stochastic_core.
        # _tick_events: per-instrument (timestamp, |order_book_imbalance|)
        # history, fed on every real tick (see _record_tick_event).
        # _latest_liquidity/_last_spreads: latest real MarketMetrics
        # fields per instrument -- 0.0 means "no real reading yet", never
        # a fabricated placeholder.
        self._tick_events: dict[str, deque[tuple[float, float]]] = {}
        self._latest_liquidity: dict[str, float] = {}
        self._last_spreads: dict[str, float] = {}

        # Guardian AI (Chief Risk Officer) trigger state -- see
        # _check_drawdown_trigger/_record_conflict_and_maybe_alert.
        # (timestamp, balance) samples, pruned to the last
        # _DRAWDOWN_WINDOW_SECONDS on every append.
        self._balance_history: deque[tuple[float, float]] = deque()
        self._drawdown_alert_armed = True
        # instrument -> timestamps of recent "conflict" consensus
        # outcomes, pruned to the last _COLLISION_WINDOW_SECONDS.
        self._conflict_history: dict[str, deque[float]] = {}
        self._collision_alert_armed: dict[str, bool] = {}

    def _position_for(self, instrument: str) -> PositionState:
        """Return (creating if needed) the :class:`PositionState` for ``instrument``."""
        return self._positions.setdefault(instrument, PositionState())

    def _open_position_count(self) -> int:
        """How many instruments currently have a non-``FLAT`` position open."""
        return sum(1 for position in self._positions.values() if position.status != "FLAT")

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent is stream-driven; see :meth:`run`.

        Implemented (rather than omitted) only to satisfy the abstract
        base class, and raises loudly if it is ever invoked by mistake.
        """
        raise NotImplementedError("ExecutionAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Connect the authenticated trading client (a no-op even in MOCK_MODE -- see its docstring)."""
        await self._trading_client.connect()
        # Seed the drawdown baseline (trigger 3) with the starting
        # balance, so the very first close already has a same-session
        # reference point instead of an empty history. Also published as
        # the equity curve's first real point, so the dashboard has
        # something genuine to plot (starting capital) immediately,
        # rather than an empty chart until the first trade closes.
        startup_timestamp = time.time()
        self._balance_history.append((startup_timestamp, self._mock_balance))
        await self._publish_equity_point(startup_timestamp, self._mock_balance)
        self._log.info(
            "execution_agent_ready",
            agent_id=self.agent_id,
            instruments=sorted(self._instruments),
            sandbox=self._sandbox,
            mock_mode=self._mock_mode,
            required_agent_ids=sorted(self._required_agent_ids),
            max_concurrent_trades=self._max_concurrent_trades,
        )

    async def on_shutdown(self) -> None:
        """Disconnect the trading client."""
        await self._trading_client.close()

    async def run(self) -> None:
        """Connect dependencies, then react to every incoming signal, price tick, or manual override.

        Purely event-driven (like :class:`~src.agents.archivist.agent.
        ArchivistAgent`) -- price-triggered exits (stop-loss/take-profit/
        trailing-stop) are evaluated reactively on every
        ``MARKET_DATA_L2`` tick, not on a timer. Two concurrent listeners
        feed that reactivity: the ZMQ bus (signals/prices, every other
        agent) and a Redis Pub/Sub channel (manual trade overrides from
        the dashboard -- see :meth:`_listen_for_manual_trade_requests`'s
        docstring for why that one is Redis, not ZMQ).

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info("agent_started", agent_id=self.agent_id, instruments=sorted(self._instruments))

            await asyncio.gather(
                self._listen_for_bus_messages(),
                self._listen_for_manual_trade_requests(),
                self._run_heartbeat_loop(),
                self._listen_for_control_commands(),
            )
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _listen_for_bus_messages(self) -> None:
        """Drain the ZMQ bus: every ``TradingSignal``/``MarketMetrics`` message, forever."""
        async for message in self._subscriber.listen():
            await self._handle_message(message)

    async def _listen_for_manual_trade_requests(self) -> None:
        """React to dashboard-originated manual trade overrides (Guardian AI trigger 2).

        Uses **Redis Pub/Sub**, not ZMQ, deliberately: ``src/dashboard/
        app.py`` is a synchronous, event-loop-free Streamlit process that
        was never meant to hold a persistent ZMQ SUB socket (see that
        module's own docstring) -- it already has a plain, already-open
        synchronous ``redis`` client, and Redis's native ``PUBLISH`` is a
        single blocking call it can issue from inside a normal Streamlit
        script rerun with no new client type or long-lived connection.
        This coroutine is what makes *this* agent's reaction reactive
        (event-driven) rather than a periodic Redis poll: it holds a
        ``redis.asyncio`` Pub/Sub subscription open for the agent's
        entire lifetime, exactly like :meth:`_listen_for_bus_messages`
        does for the ZMQ side, just on a different transport.

        Every request is re-validated here regardless of what the
        dashboard's own UI already gated (a ``confirmed`` flag it can
        only set after the human clicked through the Guardian AI's
        warning) -- never trust the publishing side alone for something
        that places a real order.
        """
        if self._redis_client is None:
            self._log.warning("manual_trade_listener_disabled_no_redis", agent_id=self.agent_id)
            return

        pubsub = self._redis_client.client.pubsub()
        await pubsub.subscribe(_MANUAL_TRADE_CHANNEL)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                await self._handle_manual_trade_request(message["data"])
        finally:
            await pubsub.unsubscribe(_MANUAL_TRADE_CHANNEL)
            # aclose() is the current, non-deprecated API (close() only
            # remains as a deprecated alias) -- types-redis' stub hasn't
            # caught up to redis-py 5.x yet.
            await pubsub.aclose()  # type: ignore[attr-defined]

    async def _run_heartbeat_loop(self) -> None:
        """Write this agent's liveness heartbeat on a fixed interval, forever.

        Every other :class:`~core.base_agent.BaseAgent` subclass writes
        its heartbeat from inside a periodic tick/poll loop already
        calling :meth:`~core.base_agent.BaseAgent._write_heartbeat`
        once per cycle -- this agent has no such loop (see :meth:`run`'s
        docstring: purely event-driven, reacting to bus/Redis messages,
        not a timer), so without this it would never write one at all
        (a real gap this exists to close, not a hypothetical one --
        confirmed absent before this change). Runs as a third listener
        alongside the two message loops in :meth:`run`'s
        ``asyncio.gather``.
        """
        while self._running:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_WRITE_INTERVAL_SECONDS)

    async def _handle_manual_trade_request(self, raw: str | bytes) -> None:
        """Validate and (if legitimate) act on one manual request from the dashboard.

        Two request shapes, dispatched on ``payload["action"]`` (default
        ``"open"``, for every publisher predating this field -- see the
        dashboard override section, unchanged):

        - ``"open"``: force an entry, bypassing **only** the consensus
          requirement -- every other entry safety layer still applies:
          this still calls :meth:`_place_order`, so the Kill Switch,
          risk-based sizing, the margin cap, and MOCK_MODE branching all
          behave exactly as they would for a consensus-approved entry.
          Also still refuses if the instrument already has an open
          position, or if :data:`_DEFAULT_MAX_CONCURRENT_TRADES` is
          already reached -- the same two gates :meth:`_act_on_consensus`
          applies, so a manual override can never stack an unbounded
          number of positions either.
        - ``"close"``: force an exit, via :meth:`_close_position` --
          **never** Kill-Switch-gated, by the same reasoning every other
          exit path in this module isn't (see the module docstring): a
          human closing a position must always be able to, including
          during a PANIC. Used by the Nothierly Chat tab's "ferme la
          position sur X" command (see :mod:`~src.agents.guardian_ai`'s
          chat section) -- but reachable from any future publisher onto
          :data:`_MANUAL_TRADE_CHANNEL`, not chat-specific itself.

        Every request is re-validated here regardless of what the
        publishing side already checked -- see :meth:`_listen_for_manual_trade_requests`'s
        docstring on why.
        """
        try:
            payload = json.loads(raw)
            instrument = str(payload["instrument"])
            action = str(payload.get("action", "open"))
            confirmed = bool(payload.get("confirmed", False))
        except (TypeError, ValueError, KeyError):
            self._log.warning("manual_trade_request_malformed", agent_id=self.agent_id)
            return

        if instrument not in self._instruments:
            self._log.warning(
                "manual_trade_request_unknown_instrument", agent_id=self.agent_id, instrument=instrument
            )
            return
        if not confirmed:
            self._log.warning(
                "manual_trade_request_missing_confirmation_ignored", agent_id=self.agent_id, instrument=instrument
            )
            return

        if action == "close":
            await self._handle_manual_close_request(instrument)
            return
        if action != "open":
            self._log.warning(
                "manual_trade_request_unknown_action", agent_id=self.agent_id, instrument=instrument, action=action
            )
            return

        try:
            direction = int(payload["direction"])
        except (TypeError, ValueError, KeyError):
            self._log.warning("manual_trade_request_malformed", agent_id=self.agent_id)
            return
        if direction not in (1, -1):
            self._log.warning(
                "manual_trade_request_invalid_direction", agent_id=self.agent_id, direction=direction
            )
            return

        position = self._position_for(instrument)
        if position.status != "FLAT":
            self._log.warning(
                "manual_trade_request_refused_already_positioned",
                agent_id=self.agent_id,
                instrument=instrument,
                position=position.status,
            )
            return
        if self._open_position_count() >= self._max_concurrent_trades:
            self._log.warning(
                "manual_trade_request_refused_max_concurrent_trades",
                agent_id=self.agent_id,
                instrument=instrument,
                max_concurrent_trades=self._max_concurrent_trades,
            )
            return

        self._log.warning(
            "manual_trade_override_executing",
            agent_id=self.agent_id,
            instrument=instrument,
            direction=direction,
            detail="Human-forced entry via dashboard override -- bypasses consensus, not the Kill Switch.",
        )
        try:
            await self._place_order(instrument, direction, confidence=1.0)
        except KillSwitchActiveError:
            self._log.warning(
                "manual_trade_override_blocked_by_kill_switch", agent_id=self.agent_id, instrument=instrument
            )

    async def _handle_manual_close_request(self, instrument: str) -> None:
        """Close ``instrument``'s position on a confirmed manual "close" request, if one is open."""
        position = self._position_for(instrument)
        if position.status == "FLAT":
            self._log.warning(
                "manual_close_request_refused_already_flat", agent_id=self.agent_id, instrument=instrument
            )
            return
        self._log.warning(
            "manual_close_request_executing",
            agent_id=self.agent_id,
            instrument=instrument,
            detail="Human-forced exit (e.g. via the Nothierly Chat tab) -- never Kill-Switch-gated.",
        )
        await self._close_position(instrument, reason="Fermeture manuelle (Nothierly Chat)")

    # -- Message dispatch ---------------------------------------------

    async def _handle_message(self, message: MessagePayload) -> None:
        """Route one bus message to the signal, price-tick, or emergency-halt handler by topic."""
        if message.topic == TRADING_SIGNALS_TOPIC:
            await self._handle_signal_message(message)
        elif message.topic == _MARKET_DATA_TOPIC:
            await self._handle_price_message(message)
        elif message.topic == EMERGENCY_HALT_TOPIC:
            await self.cancel_all_positions(reason=str(message.data.get("reason", "EMERGENCY_HALT")))

    # -- Signal handling / consensus + asymmetric exit --------------------

    async def _handle_signal_message(self, message: MessagePayload) -> None:
        """Validate one bus message as a :class:`~core.schemas.TradingSignal` and process it."""
        try:
            signal = TradingSignal.model_validate(message.data)
        except Exception:
            self._log.warning(
                "execution_agent_malformed_signal_dropped", agent_id=self.agent_id, sender=message.sender_id
            )
            return

        if signal.instrument not in self._instruments:
            return
        if signal.agent_id not in self._required_agent_ids:
            self._log.debug(
                "execution_agent_signal_from_unrecognized_agent_ignored",
                agent_id=self.agent_id,
                signal_source=signal.agent_id,
            )
            return

        instrument = signal.instrument
        self._signals.setdefault(instrument, {})[signal.agent_id] = signal
        self._log.info(
            "trading_signal_received",
            agent_id=self.agent_id,
            source=signal.agent_id,
            instrument=instrument,
            signal=signal.signal,
            confidence=round(signal.confidence, 4),
        )
        await self._push_raw_signal_feed(signal)
        await self._update_market_state()

        # Asymmetric exit: ANY single dissenting/neutral signal closes an
        # already-open position immediately -- entering needs consensus,
        # leaving never waits for it. Checked before consensus, and
        # returns early: a signal that just closed a position should not
        # also be evaluated as a fresh entry proposal in the same breath.
        # Only this instrument's own position/signal memory is touched --
        # every other instrument's state is untouched by this message.
        position = self._position_for(instrument)
        if position.status == "LONG" and signal.signal in (0, -1):
            self._signals.pop(instrument, None)
            await self._close_position(instrument, reason=f"Sortie par Veto Agent ({signal.agent_id})")
            return
        if position.status == "SHORT" and signal.signal in (0, 1):
            self._signals.pop(instrument, None)
            await self._close_position(instrument, reason=f"Sortie par Veto Agent ({signal.agent_id})")
            return

        result = self._evaluate_consensus(instrument)
        if result.outcome == "consensus":
            self._log.info(
                "consensus_reached",
                agent_id=self.agent_id,
                instrument=instrument,
                direction=result.direction,
                agents={aid: sig.signal for aid, sig in result.signals.items()},
            )
            self._signals.pop(instrument, None)
            await self._act_on_consensus(instrument, result.direction, result.signals)
        elif result.outcome == "conflict":
            self._log.warning(
                "consensus_conflict_vetoed",
                agent_id=self.agent_id,
                instrument=instrument,
                agents={aid: sig.signal for aid, sig in result.signals.items()},
            )
            await self.generate_decision_report(
                instrument=instrument,
                consensus_result="conflict",
                rl_signal=result.signals.get("rl-signal-agent"),
                xgb_signal=result.signals.get("supervisor-agent"),
                current_price=self._last_prices.get(instrument),
                risk_pct=None,
                execution_action="VETO",
            )
            await self._record_conflict_and_maybe_alert(instrument)
            self._signals.pop(instrument, None)
        else:
            self._log.info(
                "consensus_pending_timeout",
                agent_id=self.agent_id,
                instrument=instrument,
                have=sorted(result.signals.keys()),
                waiting_on=sorted(self._required_agent_ids - result.signals.keys()),
            )

    def _evaluate_consensus(self, instrument: str) -> _ConsensusResult:
        """Check ``instrument``'s required agents' *currently fresh* signals for agreement.

        Returns:
            A ``"waiting"`` result if not every required agent has a
            fresh signal yet; ``"conflict"`` if they disagree or any is
            neutral (``0``); ``"consensus"`` if they all agree on the
            same non-zero direction.
        """
        now = time.time()
        fresh: dict[str, TradingSignal] = {}
        for agent_id, signal in self._signals.get(instrument, {}).items():
            age_seconds = now - signal.timestamp
            if age_seconds <= self._signal_validity_seconds:
                fresh[agent_id] = signal
            else:
                self._log.info(
                    "trading_signal_expired",
                    agent_id=self.agent_id,
                    instrument=instrument,
                    source=agent_id,
                    age_seconds=round(age_seconds, 2),
                    validity_seconds=self._signal_validity_seconds,
                )

        if not self._required_agent_ids.issubset(fresh.keys()):
            return _ConsensusResult(outcome="waiting", signals=fresh)

        directions = {fresh[agent_id].signal for agent_id in self._required_agent_ids}
        if len(directions) == 1 and 0 not in directions:
            return _ConsensusResult(outcome="consensus", signals=fresh, direction=directions.pop())
        return _ConsensusResult(outcome="conflict", signals=fresh)

    # -- Price-tick handling / risk-management exits -----------------------

    async def _handle_price_message(self, message: MessagePayload) -> None:
        """Validate one bus message as :class:`~core.schemas.MarketMetrics` and check exits."""
        try:
            metrics = MarketMetrics.model_validate(message.data)
        except Exception:
            return
        if metrics.symbol not in self._instruments:
            return

        self._last_prices[metrics.symbol] = metrics.mid_price
        self._last_spreads[metrics.symbol] = metrics.spread
        if metrics.top_of_book_liquidity > 0:
            self._latest_liquidity[metrics.symbol] = metrics.top_of_book_liquidity
        self._record_tick_event(metrics.symbol, metrics.order_book_imbalance)
        self._order_flow_agent.on_tick(metrics.symbol, metrics.order_book_imbalance)
        self._portfolio_risk_manager.record_price(metrics.symbol, metrics.mid_price)
        await self._update_market_state()
        if self._position_for(metrics.symbol).status != "FLAT":
            await self._publish_active_positions()
        await self._evaluate_price_triggered_exit(metrics.symbol, metrics.mid_price)

    async def _evaluate_price_triggered_exit(self, instrument: str, current_price: float) -> None:
        """Update the trailing-stop anchor and close if SL/trailing-stop/TP is hit.

        Exit priority (checked in this order, each returning immediately
        once triggered): 1. agent veto (handled entirely in
        :meth:`_handle_signal_message`, never reaches here) -> 2.
        trailing stop (or the static stop-loss, before the trailing stop
        has activated) -> 3. take-profit.
        """
        position = self._positions.get(instrument)
        if position is None or position.status == "FLAT" or position.entry_price is None:
            return

        if position.status == "LONG":
            previous_high = position.highest_seen_price
            position.highest_seen_price = (
                current_price if previous_high is None else max(previous_high, current_price)
            )
            if previous_high is None or position.highest_seen_price > previous_high:
                dynamic_stop = position.highest_seen_price * (1 - self._trailing_stop_pct)
                self._log.info(
                    "trailing_stop_updated",
                    agent_id=self.agent_id,
                    instrument=instrument,
                    detail=(
                        f"Nouveau sommet atteint: {position.highest_seen_price:.5f}. "
                        f"Trailing stop remonte a {dynamic_stop:.5f}."
                    ),
                    highest_seen_price=position.highest_seen_price,
                    dynamic_stop=dynamic_stop,
                )

            # Activation is a one-way latch keyed off the *peak* profit
            # ever reached (highest_seen_price), never the current tick:
            # checking against current_price would let a pullback --
            # exactly what the trailing stop exists to catch -- undo its
            # own activation and fall back to the wider static stop
            # right when the tighter trailing stop matters most.
            if not position.trailing_activated:
                peak_profit_pct = (position.highest_seen_price - position.entry_price) / position.entry_price
                if peak_profit_pct >= self._trailing_activation_pct:
                    position.trailing_activated = True

            if position.trailing_activated:
                dynamic_stop = position.highest_seen_price * (1 - self._trailing_stop_pct)
                if current_price <= dynamic_stop:
                    await self._close_position(instrument, reason="Sortie par Trailing Stop", price=current_price)
                    return
            else:
                static_stop = position.entry_price * (1 - self._stop_loss_pct)
                if current_price <= static_stop:
                    await self._close_position(instrument, reason="Sortie par Stop-Loss", price=current_price)
                    return

            take_profit_price = position.entry_price * (1 + self._take_profit_pct)
            if current_price >= take_profit_price:
                await self._close_position(instrument, reason="Sortie par Take-Profit", price=current_price)
                return

        else:  # SHORT
            previous_low = position.lowest_seen_price
            position.lowest_seen_price = (
                current_price if previous_low is None else min(previous_low, current_price)
            )
            if previous_low is None or position.lowest_seen_price < previous_low:
                dynamic_stop = position.lowest_seen_price * (1 + self._trailing_stop_pct)
                self._log.info(
                    "trailing_stop_updated",
                    agent_id=self.agent_id,
                    instrument=instrument,
                    detail=(
                        f"Nouveau plus bas atteint: {position.lowest_seen_price:.5f}. "
                        f"Trailing stop redescend a {dynamic_stop:.5f}."
                    ),
                    lowest_seen_price=position.lowest_seen_price,
                    dynamic_stop=dynamic_stop,
                )

            # Same one-way latch as the LONG branch above, keyed off the
            # lowest price ever seen rather than the current tick.
            if not position.trailing_activated:
                peak_profit_pct = (position.entry_price - position.lowest_seen_price) / position.entry_price
                if peak_profit_pct >= self._trailing_activation_pct:
                    position.trailing_activated = True

            if position.trailing_activated:
                dynamic_stop = position.lowest_seen_price * (1 + self._trailing_stop_pct)
                if current_price >= dynamic_stop:
                    await self._close_position(instrument, reason="Sortie par Trailing Stop", price=current_price)
                    return
            else:
                static_stop = position.entry_price * (1 + self._stop_loss_pct)
                if current_price >= static_stop:
                    await self._close_position(instrument, reason="Sortie par Stop-Loss", price=current_price)
                    return

            take_profit_price = position.entry_price * (1 - self._take_profit_pct)
            if current_price <= take_profit_price:
                await self._close_position(instrument, reason="Sortie par Take-Profit", price=current_price)
                return

    # -- Entry (consensus-gated, Kill-Switch-gated) ------------------------

    async def _act_on_consensus(
        self, instrument: str, direction: int, signals: dict[str, TradingSignal]
    ) -> None:
        """Enter a consensus-approved direction; the 8 real gates decide the rest.

        Args:
            instrument: Which instrument reached consensus.
            direction: ``1`` (buy/long) or ``-1`` (sell/short), from
                :meth:`_evaluate_consensus`.
            signals: The fresh, agreeing signals consensus was reached
                from -- passed straight through to :meth:`_place_order`,
                which owns the entire gating decision (Gates 1-8, see
                :meth:`_validate_execution_gates`) as well as the
                decision-report logging. This method used to duplicate
                three of those checks (already-positioned, cooldown,
                max-concurrent-trades) inline and optimistically log a
                BUY/SELL decision report *before* knowing whether sizing/
                risk gates would even allow it -- both responsibilities
                now live in one place, so the single real gate pipeline
                :mod:`src.dashboard.ems_pages.fleet_command_nexus` reads
                telemetry from can never drift from what this method
                actually does.
        """
        confidence = min(signal.confidence for signal in signals.values())
        try:
            await self._place_order(instrument, direction, confidence, signals)
        except KillSwitchActiveError:
            self._log.warning(
                "consensus_order_blocked_by_global_kill_switch",
                agent_id=self.agent_id,
                instrument=instrument,
                direction=direction,
                mock_mode=self._mock_mode,
            )

    # -- 7-Gate execution validation (Fleet Command Arbiter) ----------------
    #
    # See the module docstring's "Protocole 7-Gates Convergence" addendum.
    # Every check below already existed, inline, in _place_order before
    # this refactor -- factored out into named gates so (a) a proposed
    # entry has one real, authoritative decision path instead of
    # scattered if/return checks, and (b) each gate's real, live verdict
    # can be published for the dashboard's telemetry (see
    # _publish_gate_report). No gate's underlying math, exception type,
    # or final trading decision changed -- only where the check lives.

    def _gate_1_consensus(self, signals: dict[str, TradingSignal] | None, direction: int) -> GateResult:
        """Gate 1: both required Daemons must already agree, non-neutral, within the freshness window.

        Always ``PASSED`` by the time this runs: real ``signals`` means
        :meth:`_evaluate_consensus` already independently verified
        agreement before :meth:`_act_on_consensus` ever called
        :meth:`_place_order`. ``signals=None`` means a human operator's
        manual override reached this gate instead (see
        :meth:`_handle_manual_trade_request`) -- still ``PASSED``,
        honestly labeled as a disclosed bypass, never a fabricated
        consensus.
        """
        if signals is None:
            return GateResult(
                1, _GATE_NAMES[0], "PASSED",
                "Bypass operateur manuel (Nothierly Chat / dashboard) -- consensus non requis pour ce chemin, disclose.",
            )
        stance = "ACHAT" if direction == 1 else "VENTE"
        agents = ", ".join(
            f"{_DAEMON_DISPLAY_NAME.get(agent_id, agent_id)}={stance}" for agent_id in sorted(signals)
        )
        return GateResult(1, _GATE_NAMES[0], "PASSED", f"Consensus reel atteint : {agents}.")

    def _gate_2_cooldown(self, instrument: str) -> GateResult:
        """Gate 2: no duplicate entry, and the post-close re-entry delay has elapsed.

        The single, shared cooldown/no-duplicate/max-concurrent check for
        *every* entry path (consensus-driven and manual-override alike)
        -- see the module docstring's disclosed fix: the manual-override
        path previously never re-checked cooldown before this gate
        existed, despite its own docstring claiming it did.
        """
        position = self._position_for(instrument)
        if position.status != "FLAT":
            self._log.info(
                "consensus_order_skipped_already_in_position",
                agent_id=self.agent_id, instrument=instrument, position=position.status,
            )
            return GateResult(2, _GATE_NAMES[1], "FAILED", f"Position deja ouverte ({position.status}) -- doublon refuse.")
        if position.cooldown_until is not None and time.time() < position.cooldown_until:
            remaining = position.cooldown_until - time.time()
            self._log.info(
                "consensus_entry_skipped_cooldown_active",
                agent_id=self.agent_id, instrument=instrument, seconds_remaining=round(remaining, 1),
            )
            return GateResult(2, _GATE_NAMES[1], "FAILED", f"Cooldown actif -- {remaining:.0f}s restantes avant re-entree autorisee.")
        open_count = self._open_position_count()
        if open_count >= self._max_concurrent_trades:
            self._log.warning(
                "consensus_entry_skipped_max_concurrent_trades",
                agent_id=self.agent_id, instrument=instrument,
                open_positions=open_count, max_concurrent_trades=self._max_concurrent_trades,
            )
            return GateResult(2, _GATE_NAMES[1], "FAILED", f"max_concurrent_trades atteint ({open_count}/{self._max_concurrent_trades}).")
        return GateResult(2, _GATE_NAMES[1], "PASSED", "Aucun doublon, delai de re-entree respecte.")

    async def _gate_3_sizing(
        self, instrument: str, current_price: float, stop_loss_price: float, account_balance: float
    ) -> tuple[GateResult, float]:
        """Gate 3: fixed-fractional risk sizing -- see :meth:`calculate_position_size`."""
        raw_units = await self.calculate_position_size(instrument, current_price, stop_loss_price, account_balance)
        if raw_units <= 0:
            return (
                GateResult(3, _GATE_NAMES[2], "FAILED", "Taille calculee nulle (distance Stop-Loss nulle ou plafond configure a 0)."),
                0.0,
            )
        return GateResult(3, _GATE_NAMES[2], "PASSED", f"{raw_units:.4f} unites (risque / distance SL)."), raw_units

    def _gate_4_margin(
        self, instrument: str, raw_units: float, current_price: float, available_margin: float
    ) -> tuple[GateResult, float]:
        """Gate 4: the sized order must not exceed available margin -- caps, never silently drops, unless capped to ~0."""
        max_affordable_units = available_margin / current_price if current_price > 0 else 0.0
        sized_units = min(raw_units, max_affordable_units)
        if sized_units < 1e-9:
            return GateResult(4, _GATE_NAMES[3], "FAILED", f"Marge disponible insuffisante ({available_margin:.2f}$)."), 0.0
        if sized_units < raw_units:
            self._log.warning(
                "position_size_capped_by_available_margin",
                agent_id=self.agent_id, instrument=instrument,
                requested_units=round(raw_units, 2), capped_units=round(sized_units, 2),
            )
            return (
                GateResult(4, _GATE_NAMES[3], "PASSED", f"Plafonne par la marge : {sized_units:.4f} unites (demande {raw_units:.4f})."),
                sized_units,
            )
        return GateResult(4, _GATE_NAMES[3], "PASSED", f"{sized_units:.4f} unites -- sous le plafond de marge."), sized_units

    def _gate_5_risk_manager(self, raw_units: float, current_price: float) -> GateResult:
        """Gate 5: flat portfolio exposure ceiling -- :class:`~src.agents.supervisor.risk_manager.RiskManager`.

        Checked with *this* order's actual projected notional
        (``raw_units * current_price``), not a flat capital percentage --
        it must be directly comparable to :attr:`_current_exposure`, which
        accumulates real notional per position. Necessarily multi-asset-
        aware: ``_current_exposure`` already reflects every other
        instrument's currently open position, so a second instrument's
        entry is checked against the portfolio's *total* exposure, not
        evaluated in isolation.
        """
        try:
            self._risk_manager.check_exposure_limit(
                current_exposure=self._current_exposure,
                proposed_order_value=raw_units * current_price,
                capital=self._capital,
            )
        except RiskLimitExceededError as exc:
            return GateResult(5, _GATE_NAMES[4], "FAILED", str(exc))
        return GateResult(5, _GATE_NAMES[4], "PASSED", "Exposition totale du portefeuille sous le plafond configure.")

    def _gate_6_portfolio_risk_parity(self, instrument: str, raw_units: float, current_price: float) -> GateResult:
        """Gate 6: correlated-cluster exposure ceiling -- :class:`~src.agents.supervisor.portfolio_risk_manager.PortfolioRiskParityManager`.

        Phase 3's correlation-aware gate, layered on top of Gate 5 -- see
        that class's module docstring. Existing positions' notional is
        recomputed fresh here (not read from :attr:`_current_exposure`,
        a single running total with no per-instrument breakdown).

        **"Arsenal Interdisciplinaire" addition (2026-09-13),
        informational only**: a real :func:`~core.quant_math.
        giant_cluster_fraction` reading across every instrument this
        agent tracks with sufficient real price history -- a
        percolation-style systemic-concentration signal, distinct from
        (and never blocking on top of) the hard correlated-exposure
        ceiling above, which only ever looks at *this one* proposed
        instrument's own correlated cluster.
        """
        open_position_notionals = {
            other_instrument: abs(position.units_held * position.entry_price)
            for other_instrument, position in self._positions.items()
            if position.status != "FLAT" and position.units_held is not None and position.entry_price is not None
        }
        try:
            self._portfolio_risk_manager.check_correlated_exposure(
                instrument, raw_units * current_price, positions=open_position_notionals, capital=self._capital,
            )
        except CorrelatedExposureError as exc:
            return GateResult(6, _GATE_NAMES[5], "FAILED", str(exc))

        cluster_detail = self._describe_giant_cluster_fraction()
        detail = "Cluster correle sous le plafond de concentration."
        if cluster_detail is not None:
            detail = f"{detail} {cluster_detail}"
        return GateResult(6, _GATE_NAMES[5], "PASSED", detail)

    def _describe_giant_cluster_fraction(self) -> str | None:
        """Real percolation-style systemic concentration across every sufficiently-tracked instrument.

        Returns:
            A real, human-readable ``"GiantCluster=X.XX (N/M instruments)"``
            string, or ``None`` if fewer than 2 tracked instruments
            currently have enough real price history to compute a
            correlation matrix from (see :meth:`~src.agents.supervisor.
            portfolio_risk_manager.PortfolioRiskParityManager.
            has_sufficient_history`) -- an honest "not enough data yet",
            never a fabricated number.
        """
        ready_instruments = [
            instr for instr in sorted(self._instruments)
            if self._portfolio_risk_manager.has_sufficient_history(instr)
        ]
        if len(ready_instruments) < 2:
            return None

        n = len(ready_instruments)
        correlation_matrix = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                correlation = self._portfolio_risk_manager.compute_correlation(
                    ready_instruments[i], ready_instruments[j]
                )
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

        fraction = giant_cluster_fraction(correlation_matrix, self._portfolio_risk_manager.correlation_threshold)
        return f"GiantCluster={fraction:.2f} ({n}/{len(self._instruments)} instruments)"

    def _gate_7_stochastic_core(self, instrument: str, sized_units: float, current_price: float) -> GateResult:
        r"""Gate 7: the real "Global Physics Vector" -- 4 stochastic-process models, 2 of them blocking.

        See :mod:`core.quant_math` for every formula. Real inputs only,
        never fabricated: :attr:`_tick_events` (real received
        ``MARKET_DATA_L2`` ticks, weighted by their own real
        ``|order_book_imbalance|`` -- see :meth:`_record_tick_event`'s
        docstring on why a tick, not a literal trade print, is this
        system's real "event"), :meth:`~src.agents.supervisor.
        portfolio_risk_manager.PortfolioRiskParityManager.price_history`
        (the exact same real rolling price window Gate 6 already reads,
        reused rather than duplicated), and :attr:`_latest_liquidity`/
        :attr:`_last_spreads` (real ``top_of_book_liquidity``/``spread``
        from the latest tick -- ``0.0`` when no real reading has arrived
        yet, honestly, never a placeholder).

        Blocking behavior (matching this project's own blocking language
        -- only these two daemons actually use "verrouille"/"annulé"):

        - **Daemon 2 (Merton jump-diffusion)**: a detected black swan
          blocks *any* new entry, not just a buy -- a stronger,
          disclosed reading of "verrouille tout nouvel achat" than the
          literal French: a forming black swan is dangerous for a new
          short too, and leaving that gap open would be an inconsistent
          half-measure.
        - **Daemon 4 (price impact)**: blocked only when a real impact
          estimate exists (real liquidity data has arrived) *and* it
          would exceed the real current spread -- the real edge this
          trade needs to overcome to be worth taking at all.

        Daemon 1 (Hawkes cascade) and Daemon 3 (OU convergence) are real,
        computed, and included in ``detail`` for telemetry, but never
        block on their own -- the brief's own language never uses a
        blocking verb for either (Daemon 3 is framed as *enabling* an
        arbitrage decision elsewhere, not as a veto here).

        **"Arsenal Interdisciplinaire" additions (2026-09-13), both also
        informational-only**: a real :func:`~core.quant_math.
        wasserstein_price_distance` between this window's first and
        second half of log-returns (a real, cheap distributional-drift
        read -- large values suggest the return distribution shifted
        mid-window, without needing a second, separately-tracked
        baseline buffer), and a real :func:`~core.quant_math.
        kolmogorov_compressibility_ratio` of the raw price history (a
        low ratio is a real, disclosed algorithmic-footprint signal --
        see that function's own docstring on why this is a practical
        proxy, not literal, uncomputable Kolmogorov complexity).

        **"Order-Flow Liquidity Provision" addition (2026-09-13), also
        informational-only**: a real :class:`~execution.
        order_flow_liquidity_agent.OrderFlowLiquidityAgent` recommendation
        (``BUY``/``SELL``/``NEUTRAL`` + real signed dominance), fed every
        real tick this instrument receives (see
        :attr:`_order_flow_agent`). See that module's own docstring for
        the real legal boundary this is deliberately scoped inside
        (public order-book-imbalance inference only -- never a specific,
        known counterparty order). Never blocks or resizes anything
        here -- a real, disclosed opportunistic signal an operator or a
        future, separately-reviewed sizing rule could act on, not
        silently wired into live sizing by this gate.

        **"Time & Causality" addition (2026-09-13), also
        informational-only**: a real :func:`~core.quant_math.
        estimate_lyapunov_exponent` reading on this instrument's own real
        log-returns -- a dynamically-updating, disclosed estimate of how
        many more real ticks this window's own edge should honestly be
        trusted for, not a fixed constant. See that function's own
        docstring for the real, load-bearing correction on why a
        positive reading here is evidence of *local* sensitivity in this
        specific window, never proof the market is a deterministic
        chaotic system -- this gate never blocks on it, exactly the same
        restraint every other Gate 7 telemetry-only addition already
        exercises.

        **"Signal, Physique, Algo" addition (2026-09-14), also
        informational-only**: a real :func:`~core.quant_math.
        poiseuille_liquidity_resistance` reading from the exact same real
        ``liquidity``/``spread`` this gate's own :func:`~core.quant_math.
        price_impact` check already reads -- a disclosed Hagen-Poiseuille
        *analogy* (never a claim this order book is a literal viscous
        fluid) surfacing the much sharper :math:`r^{-4}` convexity a
        halved liquidity depth produces, which the square-root
        ``price_impact`` law above does not capture as dramatically. A
        real, complementary read for an operator's intuition, not a
        second, competing blocking veto over the same inputs
        ``price_impact`` already gates on.
        """
        detail_parts: list[str] = []
        blocked_reason: str | None = None

        events = self._tick_events.get(instrument)
        if events:
            timestamps = np.array([ts for ts, _ in events], dtype=np.float64)
            weights = np.array([w for _, w in events], dtype=np.float64)
            intensity = hawkes_intensity(
                timestamps,
                baseline_intensity=_HAWKES_BASELINE_INTENSITY,
                decay_rate=_HAWKES_DECAY_RATE,
                excitation_strength=_HAWKES_EXCITATION_STRENGTH,
                event_weights=weights,
            )
            cascade = intensity > _HAWKES_BASELINE_INTENSITY * _HAWKES_CASCADE_THRESHOLD_MULTIPLIER
            detail_parts.append(f"Hawkes(lambda)={intensity:.4f}" + (" [CASCADE]" if cascade else ""))
        else:
            detail_parts.append("Hawkes: pas encore d'evenements reels")

        price_history: npt.NDArray[np.float64] = self._portfolio_risk_manager.price_history(instrument)
        log_returns = np.diff(np.log(price_history)) if price_history.size >= 2 else np.array([], dtype=np.float64)

        if log_returns.size >= _MIN_SAMPLES_FOR_STOCHASTIC_CORE - 1:
            is_black_swan, jump_magnitude = jump_diffusion_anomaly(
                log_returns, jump_threshold_std=_JUMP_DIFFUSION_THRESHOLD_STD
            )
            detail_parts.append(
                f"Jump-Diffusion={jump_magnitude:.2f}sigma" + (" [BLACK SWAN]" if is_black_swan else "")
            )
            if is_black_swan:
                blocked_reason = (
                    f"Cygne noir detecte (saut de {jump_magnitude:.2f} ecarts-types au-dela du seuil "
                    f"{_JUMP_DIFFUSION_THRESHOLD_STD:.1f}) -- tout nouvel ordre verrouille (achat et vente)."
                )
        else:
            detail_parts.append("Jump-Diffusion: pas assez d'historique de prix")

        if price_history.size >= 3:
            theta, mu, half_life = ornstein_uhlenbeck_mean_reversion(price_history)
            if theta > 0 and not math.isnan(mu):
                convergence_score = theta * (mu - current_price) / current_price
                detail_parts.append(
                    f"OU theta={theta:.4f} mu={mu:.5f} half_life={half_life:.1f}s "
                    f"convergence_score={convergence_score:.4f}"
                )
            else:
                detail_parts.append("OU: aucune force de rappel reelle sur cette fenetre")
        else:
            detail_parts.append("OU: pas assez d'historique de prix")

        if log_returns.size >= _MIN_SAMPLES_FOR_WASSERSTEIN_DRIFT:
            midpoint = log_returns.size // 2
            drift = wasserstein_price_distance(log_returns[:midpoint], log_returns[midpoint:])
            detail_parts.append(f"Wasserstein(drift 1ere/2eme moitie)={drift:.6f}")
        else:
            detail_parts.append("Wasserstein: pas assez d'historique de prix")

        if price_history.size >= _MIN_SAMPLES_FOR_COMPRESSIBILITY:
            compressibility = kolmogorov_compressibility_ratio(price_history)
            detail_parts.append(f"Compressibilite(zlib)={compressibility:.4f}")
        else:
            detail_parts.append("Compressibilite: pas assez d'historique de prix")

        if log_returns.size >= _MIN_SAMPLES_FOR_LYAPUNOV:
            try:
                lyapunov_exponent, lyapunov_time = estimate_lyapunov_exponent(
                    log_returns, max_time_steps=_LYAPUNOV_MAX_TIME_STEPS
                )
            except ValueError:
                detail_parts.append("Lyapunov: echec du calcul reel sur cette fenetre")
            else:
                if math.isinf(lyapunov_time):
                    detail_parts.append(f"Lyapunov(lambda)={lyapunov_exponent:.4f} (non-chaotique sur cette fenetre)")
                else:
                    detail_parts.append(
                        f"Lyapunov(lambda)={lyapunov_exponent:.4f} horizon_reel~{lyapunov_time:.1f} ticks"
                    )
        else:
            detail_parts.append("Lyapunov: pas assez d'historique de prix")

        flow_recommendation = self._order_flow_agent.recommendation_for(instrument)
        detail_parts.append(
            f"OrderFlow={flow_recommendation.direction}(dominance={flow_recommendation.dominance:.3f})"
        )

        liquidity = self._latest_liquidity.get(instrument, 0.0)
        if blocked_reason is None and liquidity > 0 and log_returns.size >= 2:
            volatility = float(np.std(log_returns, ddof=1))
            try:
                impact_fraction = price_impact(
                    sized_units, liquidity, volatility, impact_coefficient=_PRICE_IMPACT_COEFFICIENT
                )
            except ValueError:
                impact_fraction = None
            if impact_fraction is not None:
                impact_value = abs(impact_fraction) * current_price
                spread = self._last_spreads.get(instrument, 0.0)
                detail_parts.append(f"PriceImpact={impact_value:.6f} (spread reel={spread:.6f})")
                if spread > 0 and impact_value > spread:
                    blocked_reason = (
                        f"Impact de prix estime ({impact_value:.6f}) depasse le spread reel "
                        f"({spread:.6f}) -- l'ordre detruirait son propre edge, annule."
                    )
        else:
            detail_parts.append("PriceImpact: pas de donnee de liquidite reelle disponible")

        if liquidity > 0:
            spread_for_resistance = self._last_spreads.get(instrument, 0.0)
            resistance = poiseuille_liquidity_resistance(
                spread_for_resistance, liquidity, execution_horizon=_POISEUILLE_EXECUTION_HORIZON
            )
            detail_parts.append(f"Poiseuille(resistance)={resistance:.3e}")
        else:
            detail_parts.append("Poiseuille: pas de donnee de liquidite reelle disponible")

        detail = " | ".join(detail_parts)
        if blocked_reason is not None:
            return GateResult(7, _GATE_NAMES[6], "FAILED", f"{blocked_reason} [{detail}]")
        return GateResult(7, _GATE_NAMES[6], "PASSED", detail)

    def _record_tick_event(self, instrument: str, order_book_imbalance: float) -> None:
        """Feed one real received tick into Gate 7's Hawkes event history.

        This system has no literal per-trade tick print (see
        :class:`~core.schemas.MarketMetrics`'s own fields) -- a received
        order-book tick, weighted by its own real
        ``|order_book_imbalance|`` (a genuine order-flow-pressure proxy,
        not an invented one), is the real, honest "event" this process
        excites on. Timestamped by real wall-clock receipt time
        (``time.time()``), matching every other rolling window in this
        module (:attr:`_balance_history`, :attr:`_conflict_history`), not
        the message's own embedded timestamp.
        """
        history = self._tick_events.setdefault(instrument, deque(maxlen=_TICK_EVENT_HISTORY_MAXLEN))
        history.append((time.time(), abs(order_book_imbalance)))

    @staticmethod
    def _pending_gates(from_number: int, reason: str) -> list[GateResult]:
        """Every gate from ``from_number`` through 8, marked ``PENDING`` -- never silently ``PASSED``."""
        return [GateResult(n, _GATE_NAMES[n - 1], "PENDING", reason) for n in range(from_number, 9)]

    async def _validate_execution_gates(
        self, instrument: str, direction: int, signals: dict[str, TradingSignal] | None
    ) -> tuple[GateReport, float | None, float | None, float]:
        """The 8 hardcoded gates a proposed entry must clear, sequentially, before ``create_market_order``.

        Evaluates Gates 1-8 in this project's own numbering (Consensus,
        Cooldown, Sizing, Margin, RiskManager, PortfolioRiskParityManager,
        StochasticCore, Control State) and short-circuits on the first
        failure -- every gate after it is recorded ``PENDING``, never
        silently ``PASSED``.

        **One real ordering nuance, disclosed**: Gate 8's ``OFF`` veto is
        actually checked first internally, before Gates 3-7 (which is
        real efficiency, not a reordering of intent -- it avoids a
        pointless OANDA network round-trip when this agent is fully
        paused), and its LIVE-vs-SHADOW/MOCK sub-decision must also
        resolve before Gates 3-7 can even pick a price/balance/margin
        source to check against. Both sub-decisions are still *reported*
        as Gate 8, last, in the returned :class:`GateReport` -- matching
        the project's own framing that Control State is the final gate
        that opens (or closes) the door to real execution.

        Publishes the resulting :class:`GateReport` to Redis on every
        call (see :meth:`_publish_gate_report`) -- including a rejection,
        so :mod:`src.dashboard.ems_pages.fleet_command_nexus` always
        reflects the real, current state of the pipeline, not just its
        successes.

        Returns:
            ``(report, current_price, account_balance, sized_units)``.
            ``current_price``/``account_balance`` are ``None`` only when
            resolving them itself failed or was never reached (Gate 8
            will be ``FAILED`` in that case); ``sized_units`` is ``0.0``
            unless Gates 1-6 all passed.
        """
        gates: list[GateResult] = [self._gate_1_consensus(signals, direction), self._gate_2_cooldown(instrument)]

        if gates[-1].status == "FAILED":
            gates.extend(self._pending_gates(3, "Non evalue -- Porte 2 (Cooldown) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, None, None, 0.0

        if self.control_state == AgentControlState.OFF:
            self._log.warning(
                "consensus_entry_skipped_agent_off",
                agent_id=self.agent_id, instrument=instrument,
                detail="This agent's real, operator-set control state is OFF -- refusing every new entry.",
            )
            gates.extend(self._pending_gates(3, "Non evalue -- Porte 8 (Control State) a rejete en premier."))
            gates[-1] = GateResult(8, _GATE_NAMES[7], "FAILED", "OFF -- rejet total (kill switch operateur sur cet agent).")
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, None, None, 0.0

        # AgentControlState.SHADOW is treated exactly like MOCK_MODE for
        # every gate below, not just the final placement call -- matching
        # this project's one existing definition of "Shadow Mode"
        # (api_middleware/gateway.py's ExecutionMode.SHADOW: "no network
        # call to a real broker -- that's the whole point"). A real,
        # load-bearing correction found by testing this live: fetching a
        # real price/balance from OANDA under SHADOW (only skipping the
        # final order call) still made a real, unwanted network call.
        simulate_this_entry = self._mock_mode or self.control_state == AgentControlState.SHADOW
        shadow_controlled = self.control_state == AgentControlState.SHADOW

        if simulate_this_entry:
            current_price = self._last_prices.get(instrument)
            if current_price is None:
                self._log.warning(
                    "consensus_entry_skipped_no_price_yet",
                    agent_id=self.agent_id, instrument=instrument,
                    detail="No MARKET_DATA_L2 tick received yet for this instrument.",
                )
                gates.extend(self._pending_gates(3, "Non evalue -- aucun prix disponible."))
                gates[-1] = GateResult(8, _GATE_NAMES[7], "FAILED", "Aucun tick MARKET_DATA_L2 recu pour cet instrument.")
                report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
                await self._publish_gate_report(report)
                return report, None, None, 0.0
            account_balance = self._mock_balance
            available_margin = self._mock_balance
        else:
            try:
                current_price = await safe_financial_call(
                    self._trading_client.get_current_price, instrument, retry_on_failure=True
                )
                account_balance = await safe_financial_call(
                    self._trading_client.get_account_balance, retry_on_failure=True
                )
                available_margin = await safe_financial_call(
                    self._trading_client.get_available_margin, retry_on_failure=True
                )
            except FinancialCallAbortedError:
                self._log.error("execution_entry_aborted_pre_trade_lookup_failed", agent_id=self.agent_id)
                gates.extend(self._pending_gates(3, "Non evalue -- echec de lecture broker pre-trade."))
                gates[-1] = GateResult(
                    8, _GATE_NAMES[7], "FAILED",
                    "Echec de lecture broker pre-trade (kill switch global declenche par safe_financial_call).",
                )
                report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
                await self._publish_gate_report(report)
                return report, None, None, 0.0

        await self._check_drawdown_trigger(account_balance)

        stop_loss_price = (
            current_price * (1 - self._stop_loss_pct)
            if direction == 1
            else current_price * (1 + self._stop_loss_pct)
        )

        gate3, raw_units = await self._gate_3_sizing(instrument, current_price, stop_loss_price, account_balance)
        gates.append(gate3)
        if gate3.status == "FAILED":
            gates.extend(self._pending_gates(4, "Non evalue -- Porte 3 (Sizing) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, current_price, account_balance, 0.0

        gate4, sized_units = self._gate_4_margin(instrument, raw_units, current_price, available_margin)
        gates.append(gate4)
        if gate4.status == "FAILED":
            gates.extend(self._pending_gates(5, "Non evalue -- Porte 4 (Marge) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, current_price, account_balance, 0.0

        gate5 = self._gate_5_risk_manager(sized_units, current_price)
        gates.append(gate5)
        if gate5.status == "FAILED":
            self._log.warning(
                "consensus_order_blocked_by_risk_manager",
                agent_id=self.agent_id, instrument=instrument, direction=direction,
            )
            gates.extend(self._pending_gates(6, "Non evalue -- Porte 5 (RiskManager) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, current_price, account_balance, 0.0

        gate6 = self._gate_6_portfolio_risk_parity(instrument, sized_units, current_price)
        gates.append(gate6)
        if gate6.status == "FAILED":
            self._log.warning(
                "consensus_order_blocked_by_correlated_exposure",
                agent_id=self.agent_id, instrument=instrument, direction=direction, detail=gate6.detail,
            )
            gates.extend(self._pending_gates(7, "Non evalue -- Porte 6 (PortfolioRiskParityManager) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, current_price, account_balance, 0.0

        gate7 = self._gate_7_stochastic_core(instrument, sized_units, current_price)
        gates.append(gate7)
        if gate7.status == "FAILED":
            self._log.warning(
                "consensus_order_blocked_by_stochastic_core",
                agent_id=self.agent_id, instrument=instrument, direction=direction, detail=gate7.detail,
            )
            gates.extend(self._pending_gates(8, "Non evalue -- Porte 7 (StochasticCore) a rejete en premier."))
            report = GateReport(instrument, direction, time.time(), tuple(gates), "REJECTED")
            await self._publish_gate_report(report)
            return report, current_price, account_balance, 0.0

        # Every gate up to 7 passed -- Gate 8's simulate/real resolution
        # already happened above (it had to, to pick gates 3-7's price
        # source); this is only where it is finally *reported*, last,
        # matching this project's own numbering (Control State is the
        # final gate that opens or closes the door to real execution).
        if simulate_this_entry:
            reason = "MOCK_MODE=true" if self._mock_mode else "control_state=SHADOW"
            gate8_detail = f"{reason} -- execution virtuelle (simulation), aucun appel broker reel."
        else:
            gate8_detail = "LIVE, MOCK_MODE=false -- porte finale ouverte, tir reel autorise."
        gates.append(GateResult(8, _GATE_NAMES[7], "PASSED", gate8_detail))

        final_action: Literal["LIVE_ORDER", "SHADOW_ORDER", "MOCK_ORDER", "REJECTED"]
        if not simulate_this_entry:
            final_action = "LIVE_ORDER"
        elif self._mock_mode:
            final_action = "MOCK_ORDER"
        else:
            final_action = "SHADOW_ORDER"

        report = GateReport(instrument, direction, time.time(), tuple(gates), final_action)
        await self._publish_gate_report(report)
        return report, current_price, account_balance, sized_units

    async def _publish_gate_report(self, report: GateReport) -> None:
        """Persist one instrument's latest 7-gate verdict onto :data:`_GATE_TELEMETRY_REDIS_KEY`.

        One JSON record per instrument, overwritten on every real gate
        evaluation -- a live snapshot, not an append-only log. This
        module's own 8-gate pipeline completes in low single-digit
        milliseconds (in-memory + Redis reads only), far too fast for a
        human to watch each gate "light up" one by one in real time --
        showing the real, current verdict of the most recent evaluation
        is the honest equivalent of that, the same resolution
        src/dashboard/ems_pages/fleet_command_nexus.py's own Module 1
        already reached for daemon latency (see that module's docstring).
        """
        if self._redis_client is None:
            return
        try:
            raw = await self._redis_client.execute("get", _GATE_TELEMETRY_REDIS_KEY)
            state: dict[str, Any] = json.loads(raw) if raw is not None else {}
        except RedisUnavailableError:
            return

        state[report.instrument] = {
            "instrument": report.instrument,
            "direction": report.direction,
            "timestamp": report.timestamp,
            "final_action": report.final_action,
            "all_passed": report.all_passed,
            "arbiter": _ARBITER_DISPLAY_NAME,
            "gates": [
                {"number": g.number, "name": g.name, "status": g.status, "detail": g.detail} for g in report.gates
            ],
        }
        try:
            await self._redis_client.execute("set", _GATE_TELEMETRY_REDIS_KEY, json.dumps(state))
        except RedisUnavailableError:
            self._log.warning("nothierly_gate_telemetry_persist_failed", agent_id=self.agent_id)

    @requires_system_active
    async def _place_order(
        self, instrument: str, direction: int, confidence: float, signals: dict[str, TradingSignal] | None = None
    ) -> None:
        """Validate the 8 execution gates, then size and place one entry order.

        Simulated under ``MOCK_MODE`` or ``AgentControlState.SHADOW``,
        real otherwise -- see Gate 8 (Control State) in
        :meth:`_validate_execution_gates`, the only place either is
        branched on for entries, so a tripped
        switch blocks the simulated entry exactly the same way it blocks
        a real one. Never called for an exit -- see the module docstring
        on why exits must never be gated here.

        Args:
            signals: The fresh, agreeing signals consensus was reached
                from, when called via :meth:`_act_on_consensus`,
                otherwise ``None`` (a manual operator override via
                :meth:`_handle_manual_trade_request`) -- see Gate 1.
                Also controls whether a decision report is generated
                (only for the real, consensus-driven path, matching this
                method's original behavior before this refactor: a
                manual override was never logged there).
        """
        report, current_price, account_balance, raw_units = await self._validate_execution_gates(
            instrument, direction, signals
        )

        if signals is not None:
            risk_pct = await self._get_dynamic_risk_pct() if report.all_passed else None
            await self.generate_decision_report(
                instrument=instrument,
                consensus_result="consensus",
                rl_signal=signals.get("rl-signal-agent"),
                xgb_signal=signals.get("supervisor-agent"),
                current_price=current_price if current_price is not None else self._last_prices.get(instrument),
                risk_pct=risk_pct,
                execution_action=("BUY" if direction == 1 else "SELL") if report.all_passed else "HOLD",
            )

        if not report.all_passed:
            first_failed = next(gate for gate in report.gates if gate.status == "FAILED")
            self._log.warning(
                "consensus_entry_rejected_by_gate",
                agent_id=self.agent_id, instrument=instrument,
                gate_number=first_failed.number, gate_name=first_failed.name, detail=first_failed.detail,
            )
            return

        assert current_price is not None and account_balance is not None  # report.all_passed guarantees both were resolved
        simulate_this_entry = report.final_action in ("MOCK_ORDER", "SHADOW_ORDER")
        shadow_controlled = report.final_action == "SHADOW_ORDER"
        # Recomputed, not threaded back from _validate_execution_gates:
        # deterministic given current_price/direction/_stop_loss_pct, and
        # only needed here for the MOCK_MODE_SIMULATED_ORDER log line below.
        stop_loss_price = (
            current_price * (1 - self._stop_loss_pct)
            if direction == 1
            else current_price * (1 + self._stop_loss_pct)
        )

        # OANDA's real order API requires whole (integer) units; neither
        # MOCK_MODE nor SHADOW ever calls it (see simulate_this_entry
        # above), and internal bookkeeping/PnL math (see
        # _close_position) works identically for fractional units, so
        # both keep raw_units' full precision instead of truncating it.
        # This matters concretely for a high-price-per-unit instrument
        # like BTC/USDT: at this project's default $10,000 capital and
        # 1% risk, a single BTC entry sizes to roughly 0.11 units --
        # int() truncation would silently round every such entry to
        # zero, and this agent would never open a crypto position at
        # all under either simulated mode.
        units: float = raw_units if simulate_this_entry else float(int(raw_units))
        if direction == -1:
            units = -units
        if abs(units) < 1e-9:
            self._log.warning(
                "consensus_order_size_rounds_to_zero_units",
                agent_id=self.agent_id,
                instrument=instrument,
                target=raw_units,
            )
            return

        if simulate_this_entry:
            self._log.warning(
                "MOCK_MODE_SIMULATED_ORDER" if self._mock_mode else "SHADOW_MODE_SIMULATED_ORDER",
                agent_id=self.agent_id,
                instrument=instrument,
                direction="BUY" if direction == 1 else "SELL",
                units=round(units, 6),
                entry_price=current_price,
                stop_loss_price=round(stop_loss_price, 5),
                confidence=round(confidence, 4),
                detail=(
                    "MOCK_MODE=true: no real OANDA call was made."
                    if self._mock_mode
                    else "control_state=SHADOW (operator-set): no real OANDA call was made."
                ),
            )
            await self._log_fill(
                instrument=instrument,
                action="ENTRY_LONG" if direction == 1 else "ENTRY_SHORT",
                price=current_price,
                units=units,
                latency_ms=0.0,
                agent_control_shadow=shadow_controlled,
            )
        else:
            order_start = time.monotonic()
            try:
                order = await safe_financial_call(
                    self._trading_client.create_market_order, instrument, int(units), retry_on_failure=False
                )
            except FinancialCallAbortedError:
                self._log.error(
                    "consensus_order_execution_aborted",
                    agent_id=self.agent_id,
                    instrument=instrument,
                    direction=direction,
                    units=units,
                    detail="Global Kill Switch has been tripped -- see prior log line.",
                )
                return
            order_latency_ms = (time.monotonic() - order_start) * 1000.0
            await self._log_fill(
                instrument=instrument,
                action="ENTRY_LONG" if direction == 1 else "ENTRY_SHORT",
                price=current_price,
                units=units,
                latency_ms=order_latency_ms,
            )
            self._log.info(
                "consensus_order_executed",
                agent_id=self.agent_id,
                instrument=instrument,
                order_id=order.get("id"),
                direction=direction,
                units=units,
            )
            execution = TradeExecution(
                symbol=instrument,
                side=TradeSide.BUY if direction == 1 else TradeSide.SELL,
                cost=abs(units) * current_price,
                order_id=str(order.get("id", "unknown")),
            )
            await self.publish(_TRADE_EXECUTED_TOPIC, execution.model_dump(mode="json"))

        position = self._position_for(instrument)
        position.status = "LONG" if direction == 1 else "SHORT"
        position.entry_price = current_price
        position.units_held = units
        position.highest_seen_price = current_price if direction == 1 else None
        position.lowest_seen_price = current_price if direction == -1 else None
        position.cooldown_until = None
        position.opened_at = time.time()
        position.entered_as_simulated = simulate_this_entry
        position.entered_as_shadow_controlled = shadow_controlled
        self._current_exposure += abs(units) * current_price
        await self._publish_active_positions()

    async def calculate_position_size(
        self, instrument: str, current_price: float, stop_loss_price: float, account_balance: float
    ) -> float:
        """Fixed-fractional risk position sizing: risk a fixed % of capital per trade.

        Args:
            instrument: Which instrument this sizing is for -- logging
                only (``account_balance`` is already the caller's
                concern, shared across every instrument; see the module
                docstring).
            current_price: Intended entry price.
            stop_loss_price: The static stop-loss price this entry will
                be protected by (see the module docstring: the *distance*
                to this price is what bounds risk, regardless of how
                large the position notional ends up being).
            account_balance: Total account equity to size risk against
                (see :meth:`~src.agents.supervisor.broker_client.
                OandaClient.get_account_balance`'s docstring for why this
                -- not available margin -- is the right denominator).

        Returns:
            Position size in instrument units (unsigned, un-rounded
            except to the nearest whole unit is left to the caller,
            which also applies the margin cap -- see :meth:`_place_order`).
            ``0.0`` if ``stop_loss_price`` equals ``current_price``
            (a zero-distance stop makes risk-based sizing undefined).
        """
        risk_pct = await self._get_dynamic_risk_pct()
        risk_amount = account_balance * risk_pct
        sl_distance = abs(current_price - stop_loss_price)

        if sl_distance <= 0:
            self._log.warning(
                "position_size_zero_sl_distance",
                agent_id=self.agent_id,
                instrument=instrument,
                current_price=current_price,
            )
            return 0.0

        position_size = risk_amount / sl_distance
        max_position_units = await self._get_dynamic_max_position_units()
        capped_by_max_units = max_position_units is not None and position_size > max_position_units
        if capped_by_max_units:
            assert max_position_units is not None  # narrows for mypy; capped_by_max_units already guarantees this
            position_size = max_position_units

        self._log.info(
            "position_size_calculated",
            agent_id=self.agent_id,
            instrument=instrument,
            detail=(
                f"Capital: {account_balance:.2f}$ | Risque: {risk_amount:.2f}$ | "
                f"SL Distance: {sl_distance:.5f}$ | Ordre calcule: {position_size:.2f} unites"
                + (f" (plafonne a {max_position_units:.2f})" if capped_by_max_units else "")
            ),
            account_balance=round(account_balance, 2),
            risk_pct=risk_pct,
            risk_amount=round(risk_amount, 2),
            sl_distance=round(sl_distance, 5),
            position_size_units=round(position_size, 2),
            capped_by_max_units=capped_by_max_units,
        )
        return position_size

    # -- Optimal execution scheduling ("Le Prompt Architecte") --------------

    async def compute_optimal_trajectory(
        self,
        instrument: str,
        order_size: float,
        volatility: float,
        volume_profile: npt.NDArray[np.float64],
        *,
        execution_horizon_seconds: float,
    ) -> OptimalExecutionSchedule:
        r"""Compute (and cache) a real Almgren-Chriss meta-order execution schedule.

        Delegates the actual math to :class:`~src.agents.supervisor.
        optimal_execution_scheduler.OptimalExecutionScheduler` (see that
        class and :mod:`core.quant_math` for the real formulas) -- this
        method's own job is just to shape the real result into a
        :class:`~core.schemas.OptimalExecutionSchedule` and cache it, the
        same "thin agent method, real math lives in a composed class"
        pattern :meth:`calculate_position_size` already follows for
        :class:`~src.agents.supervisor.risk_manager.RiskManager`.

        **Real, disclosed scope boundary**: this computes and returns a
        real, usable schedule -- it does **not** itself slice this
        instrument's real orders over time. :meth:`_place_order` still
        places exactly one simple market order per entry, unchanged (see
        the module docstring's "No Maker/Taker skew" reasoning in
        :meth:`_log_fill`): wiring this scheduler into that live
        order-placement path -- actually pacing real broker calls across
        ``execution_horizon_seconds`` -- is real, separate, larger
        follow-up work (persistent scheduler state across ticks, partial-
        fill handling, cancellation mid-schedule), not silently implied
        by adding this method. Useful today as-is: an operator or
        dashboard can call this directly to see how a proposed meta-order
        *should* be sliced, and the cached result feeds a real impact-
        curve chart (temporary vs. permanent impact) in the dashboard.

        Args:
            instrument: Which instrument this schedule is for (cache key
                only -- the math itself is instrument-agnostic given real
                ``volatility``/``volume_profile`` inputs).
            order_size: Real total quantity to execute. Must be positive.
            volatility: Real **per-second** volatility -- see
                :func:`core.quant_math.almgren_chriss_trajectory`'s own
                docstring on the real unit-magnitude pitfall (an
                annualized figure passed directly produces a practically
                degenerate schedule).
            volume_profile: Real expected traded volume per slice, one
                entry per configured scheduling step (see
                :class:`OptimalExecutionScheduler`'s ``n_steps``).
            execution_horizon_seconds: Real total time to fully execute
                ``order_size``. Must be positive.

        Returns:
            The real computed, cached :class:`~core.schemas.
            OptimalExecutionSchedule`.

        Raises:
            ValueError: Propagated from :mod:`core.quant_math` for any
                invalid input -- see those functions' own docstrings.
        """
        schedule = self._execution_scheduler.compute_schedule(
            order_size, volatility, volume_profile, execution_horizon_seconds=execution_horizon_seconds
        )
        result = OptimalExecutionSchedule(
            instrument=instrument,
            timestamp=time.time(),
            order_size=order_size,
            execution_horizon_seconds=execution_horizon_seconds,
            trajectory=[round(float(v), 6) for v in schedule.trajectory],
            trade_sizes=[round(float(v), 6) for v in schedule.trade_sizes],
            temporary_impact=[round(float(v), 8) for v in schedule.temporary_impact],
            permanent_impact=[round(float(v), 8) for v in schedule.permanent_impact],
            was_participation_capped=schedule.was_participation_capped,
            total_estimated_temporary_cost=round(schedule.total_estimated_temporary_cost, 6),
        )
        self._log.info(
            "optimal_execution_schedule_computed",
            agent_id=self.agent_id, instrument=instrument, order_size=order_size,
            was_participation_capped=result.was_participation_capped,
            total_estimated_temporary_cost=result.total_estimated_temporary_cost,
        )
        await self._persist_schedule_to_redis(result)
        return result

    async def _persist_schedule_to_redis(self, schedule: OptimalExecutionSchedule) -> None:
        """Cache the latest real schedule under :data:`~core.schemas.OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX`.

        Best-effort, like every other Redis cache write in this project
        -- a Redis hiccup never fails the real computation this method's
        caller already has in hand.
        """
        if self._redis_client is None:
            return
        key = f"{OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX}{schedule.instrument}"
        try:
            await self._redis_client.execute("set", key, schedule.model_dump_json(), ex=300)
        except RedisUnavailableError:
            self._log.warning("optimal_execution_schedule_persist_failed", agent_id=self.agent_id)

    # -- Options & multi-instrument hedging (real math, paper-filled only) -

    async def open_option_position(
        self, contract: OptionContract, quantity: float, *, volatility: float
    ) -> OptionPosition:
        """Open a real (paper-filled) option position, priced off the agent's own live feed.

        Delegates all real pricing math to :class:`~src.agents.
        supervisor.options_hedging_manager.OptionsHedgingManager` (see
        its module docstring for the real, disclosed execution boundary
        this method inherits unchanged: this agent's real broker calls
        via :meth:`_place_order` remain spot/CFD-only -- there is no
        real options broker integration anywhere in this project, so
        every option position opened here is a paper fill against a
        real Black-Scholes premium, never a real order).

        Args:
            contract: The real option contract terms to open.
            quantity: Signed real contract quantity (positive long,
                negative short). Must be nonzero -- see
                :class:`~core.schemas.OptionPosition`'s own validator.
            volatility: Real annualized volatility to price this
                contract with. Not derived automatically -- there is no
                live options-market implied-volatility feed wired into
                this agent yet (a real options-chain connector, e.g. via
                ``ccxt``'s Deribit support, is real, separate follow-up
                work), so the caller must supply a real, deliberate
                estimate rather than this method silently guessing one.

        Returns:
            The real, priced :class:`~core.schemas.OptionPosition`.

        Raises:
            ValueError: If ``contract.underlying`` has no real live
                price yet in :attr:`_last_prices` (this agent refuses to
                fabricate a spot price), or propagated from
                :class:`OptionsHedgingManager`/:mod:`core.quant_math`
                for any other invalid input.
        """
        current_underlying_price = self._last_prices.get(contract.underlying)
        if current_underlying_price is None:
            raise ValueError(
                f"No real live price available yet for {contract.underlying}; "
                "cannot honestly price this option contract."
            )
        position = self._options_hedging_manager.open_position(
            contract, quantity, current_underlying_price=current_underlying_price, volatility=volatility
        )
        self._log.info(
            "option_position_opened_via_agent",
            agent_id=self.agent_id, underlying=contract.underlying, strike=contract.strike,
            option_type=contract.option_type, quantity=quantity, entry_premium=position.entry_premium,
        )
        return position

    async def compute_portfolio_greeks(self, volatilities: dict[str, float]) -> PortfolioGreeks:
        """Compute (and cache) real aggregate Greeks across every open option AND spot position.

        Merges two real, independent sources of exposure:

        - Every open (paper) option position tracked by
          :attr:`_options_hedging_manager`, priced off this agent's own
          live :attr:`_last_prices` feed.
        - Every real open spot/CFD position this agent itself is
          tracking in :attr:`_positions` -- ``PositionState.units_held``
          is already signed (positive long, negative short), so it
          contributes directly to delta with no re-derivation, exactly
          as the existing exposure math at :attr:`_positions` iteration
          sites elsewhere in this file already treats it.

        Args:
            volatilities: Real annualized volatility per underlying,
                keyed the same as :attr:`_last_prices`. Any open option
                position whose underlying is missing here is honestly
                **skipped** (not fabricated) -- see
                :meth:`OptionsHedgingManager.compute_portfolio_greeks`'s
                own docstring.

        Returns:
            The real, aggregated :class:`~core.schemas.PortfolioGreeks`,
            already persisted to Redis under
            :data:`~core.schemas.PORTFOLIO_GREEKS_REDIS_KEY`.
        """
        spot_positions = {
            instrument: position.units_held
            for instrument, position in self._positions.items()
            if position.status != "FLAT" and position.units_held is not None
        }
        snapshot = self._options_hedging_manager.compute_portfolio_greeks(
            underlying_prices=dict(self._last_prices),
            volatilities=volatilities,
            spot_positions=spot_positions,
        )
        result = PortfolioGreeks(
            timestamp=time.time(),
            delta=round(snapshot.delta, 6),
            gamma=round(snapshot.gamma, 8),
            vega=round(snapshot.vega, 4),
            theta=round(snapshot.theta, 4),
        )
        self._log.info(
            "portfolio_greeks_computed",
            agent_id=self.agent_id, delta=result.delta, gamma=result.gamma,
            vega=result.vega, theta=result.theta,
        )
        await self._persist_portfolio_greeks_to_redis(result)
        return result

    def compute_hedge_recommendation(
        self,
        current_exposure: PortfolioGreeks,
        available_instruments: dict[str, GreeksSnapshot],
        *,
        target_greeks: tuple[str, ...] = ("delta", "vega"),
    ) -> HedgeRecommendation:
        """Solve for real hedge sizes that null the given Greeks, per Ashby's Law.

        Thin wrapper around :meth:`OptionsHedgingManager.
        compute_hedge_recommendation` -- see that method's own docstring
        for the real linear-algebra completeness test
        (:func:`core.quant_math.check_hedge_completeness`) and
        least-squares solve (:func:`core.quant_math.solve_hedge_ratios`)
        it performs. Synchronous and side-effect-free: unlike
        :meth:`open_option_position`/:meth:`compute_portfolio_greeks`,
        this never touches the broker, :attr:`_positions`, or Redis --
        it only proposes real hedge sizes for an operator or the
        dashboard to review.

        Args:
            current_exposure: The real Greeks to hedge away -- typically
                the latest result of :meth:`compute_portfolio_greeks`.
            available_instruments: Real per-unit Greeks for each
                candidate hedging instrument (e.g. the spot instrument
                itself, or a specific option contract), keyed by a
                caller-chosen label.
            target_greeks: Which Greeks to solve for. Defaults to
                ``("delta", "vega")`` -- the two most commonly hedged in
                practice.

        Returns:
            The real :class:`~src.agents.supervisor.
            options_hedging_manager.HedgeRecommendation`.
        """
        exposure_snapshot = GreeksSnapshot(
            delta=current_exposure.delta, gamma=current_exposure.gamma,
            vega=current_exposure.vega, theta=current_exposure.theta,
        )
        return self._options_hedging_manager.compute_hedge_recommendation(
            exposure_snapshot, available_instruments, target_greeks=target_greeks
        )

    def compute_lead_lag_transfer_entropy(
        self, instrument_a: str, instrument_b: str, *, lag: int = 1, n_bins: int = 4
    ) -> tuple[float, float]:
        r"""Real, directed :func:`~core.quant_math.transfer_entropy` between two instruments' own real log-returns.

        Answers a real, precise "leading indicator" question -- does
        ``instrument_a``'s own real recent return history genuinely help
        predict ``instrument_b``'s next return (beyond what
        ``instrument_b``'s own past already tells you), or the reverse
        -- using each instrument's own real tracked price history (see
        :meth:`~src.agents.supervisor.portfolio_risk_manager.
        PortfolioRiskParityManager.price_history`, the same real rolling
        window Gate 6/7 already read).

        **Real, disclosed alignment approximation**: :meth:`price_history`
        stores only real prices, not their real arrival timestamps, so
        the two instruments' return series are aligned by trailing
        *index* position (their ``aligned_length`` most recent returns
        each), not by real matching wall-clock time -- two instruments
        that tick at different real rates will not be perfectly
        synchronized sample-for-sample. A real, useful approximation for
        instruments that tick at broadly similar real rates, not an
        exact simultaneous-timestamp join.

        Synchronous and side-effect-free, same as
        :meth:`compute_hedge_recommendation` -- never touches the
        broker, :attr:`_positions`, or Redis; a real analytic for an
        operator or the dashboard to call directly.

        Args:
            instrument_a: First instrument (candidate "cause"/"effect").
            instrument_b: Second instrument.
            lag: Real number of samples ahead each direction's "effect"
                series is evaluated at. Must be >= 1.
            n_bins: Real number of quantile bins each return series is
                discretized into. Must be >= 2.

        Returns:
            ``(transfer_entropy_a_to_b, transfer_entropy_b_to_a)``, both
            real, in bits -- see :func:`~core.quant_math.transfer_entropy`'s
            own docstring on the real, disclosed finite-sample bias
            before treating a small value as meaningful.

        Raises:
            ValueError: If either instrument has fewer than 2 real
                tracked prices (no real log-return can be formed yet),
                or propagated from :func:`~core.quant_math.transfer_entropy`
                (e.g. too few aligned returns for the requested ``lag``).
        """
        prices_a = self._portfolio_risk_manager.price_history(instrument_a)
        prices_b = self._portfolio_risk_manager.price_history(instrument_b)
        if prices_a.size < 2 or prices_b.size < 2:
            raise ValueError(
                f"Insufficient real price history for {instrument_a!r}/{instrument_b!r} "
                "to compute log-returns (need at least 2 real prices each)."
            )

        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))
        aligned_length = min(returns_a.size, returns_b.size)
        returns_a = returns_a[-aligned_length:]
        returns_b = returns_b[-aligned_length:]

        te_a_to_b = transfer_entropy(returns_a, returns_b, lag=lag, n_bins=n_bins)
        te_b_to_a = transfer_entropy(returns_b, returns_a, lag=lag, n_bins=n_bins)
        return te_a_to_b, te_b_to_a

    def compute_kalman_pairs_spread(
        self,
        instrument_a: str,
        instrument_b: str,
        *,
        process_variance: float = 1e-5,
        observation_variance: float = 1e-3,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Real, time-varying Kalman-filtered pairs-trading hedge ratio + spread between two instruments' own real price history.

        Thin wrapper around :func:`~core.quant_math.
        kalman_dynamic_hedge_ratio`, fed each instrument's own real
        tracked price history (see :meth:`~src.agents.supervisor.
        portfolio_risk_manager.PortfolioRiskParityManager.price_history`,
        the same real rolling window Gate 6/7 already read) -- unlike a
        fixed-window OLS hedge ratio, this real state estimate updates on
        every new observation, recursively, rather than only in
        discrete, stale jumps as a rolling window shifts.

        **Real, disclosed alignment approximation**: same as
        :meth:`compute_lead_lag_transfer_entropy` -- :meth:`price_history`
        stores only real prices, not real arrival timestamps, so the two
        series are aligned by trailing *index* position, not real
        matching wall-clock time.

        Synchronous and side-effect-free, same as
        :meth:`compute_hedge_recommendation`/:meth:`compute_lead_lag_transfer_entropy`
        -- never touches the broker, :attr:`_positions`, or Redis; a
        real analytic for an operator or a future, separately-built
        pairs-trading strategy to call directly. This project has no
        existing stat-arb/pairs-trading execution agent -- exposed here
        as the real, usable building block rather than inventing a new
        autonomous strategy agent as part of this addition.

        Args:
            instrument_a: The real dependent-series instrument (:math:`Y`).
            instrument_b: The real independent-series instrument (:math:`X`).
            process_variance: See :func:`~core.quant_math.kalman_dynamic_hedge_ratio`.
            observation_variance: See :func:`~core.quant_math.kalman_dynamic_hedge_ratio`.

        Returns:
            ``(states, spread)`` -- see :func:`~core.quant_math.
            kalman_dynamic_hedge_ratio`'s own docstring. ``states[:, 0]``
            is the real filtered intercept, ``states[:, 1]`` the real
            filtered hedge ratio; ``spread`` is the real, Kalman-cleaned
            pairs-trading spread.

        Raises:
            ValueError: If either instrument has fewer than 2 real
                tracked prices, or propagated from
                :func:`~core.quant_math.kalman_dynamic_hedge_ratio`.
        """
        prices_a = self._portfolio_risk_manager.price_history(instrument_a)
        prices_b = self._portfolio_risk_manager.price_history(instrument_b)
        if prices_a.size < 2 or prices_b.size < 2:
            raise ValueError(
                f"Insufficient real price history for {instrument_a!r}/{instrument_b!r} "
                "to compute a real Kalman-filtered hedge ratio (need at least 2 real prices each)."
            )

        aligned_length = min(prices_a.size, prices_b.size)
        y = prices_a[-aligned_length:]
        x = prices_b[-aligned_length:]

        return kalman_dynamic_hedge_ratio(
            y, x, process_variance=process_variance, observation_variance=observation_variance
        )

    async def _persist_portfolio_greeks_to_redis(self, greeks: PortfolioGreeks) -> None:
        """Cache the latest real portfolio Greeks under :data:`~core.schemas.PORTFOLIO_GREEKS_REDIS_KEY`.

        Best-effort, like every other Redis cache write in this project
        -- a Redis hiccup never fails the real computation this method's
        caller already has in hand. Unlike
        :meth:`_persist_schedule_to_redis`, this key carries no
        per-instrument suffix: Greeks are additive across the whole
        book by construction, so there is exactly one real, current
        portfolio-level snapshot to cache.
        """
        if self._redis_client is None:
            return
        try:
            await self._redis_client.execute("set", PORTFOLIO_GREEKS_REDIS_KEY, greeks.model_dump_json(), ex=300)
        except RedisUnavailableError:
            self._log.warning("portfolio_greeks_persist_failed", agent_id=self.agent_id)

    async def _get_dynamic_max_position_units(self) -> float | None:
        """Read the live, independent max-position-size cap from Redis, fail-safe toward uncapped.

        Args mirror :meth:`_get_dynamic_risk_pct`'s exact fail-safe
        shape, with one real difference: there is no sensible non-zero
        fallback for a cap (unlike risk %, where the smallest safe
        number is the safe default) -- any failure or absence of a real
        configured value means "uncapped" (``None``), the same real
        behavior every position-size calculation already had before
        this cap existed, not a fabricated conservative number.

        Returns:
            The real configured cap in instrument units, or ``None`` if
            unset/unreadable/unparseable/non-positive.
        """
        if self._redis_client is None:
            return None

        try:
            raw = await asyncio.wait_for(
                self._redis_client.execute("get", TRADING_MAX_POSITION_UNITS_REDIS_KEY), timeout=2.0
            )
        except (RedisUnavailableError, asyncio.TimeoutError):
            return None

        if raw is None:
            return None

        try:
            max_units = float(raw)
        except (TypeError, ValueError):
            self._log.warning("dynamic_max_position_units_malformed_ignored", raw=raw)
            return None

        return max_units if max_units > 0 else None

    async def _get_dynamic_risk_pct(self) -> float:
        """Read the live risk-per-trade fraction from Redis, hard-capped and fail-safe.

        Reads :data:`_RISK_PCT_REDIS_KEY` (written by the Streamlit
        dashboard's risk slider) fresh on every call -- a change there
        takes effect on the very next trade, no restart needed. Fails
        toward the smallest, safest number at every possible failure
        point:

        - No Redis connection, a timeout, or ``RedisUnavailableError``:
          falls back to :data:`_FALLBACK_RISK_PCT` (1%).
        - Key not set, or not parseable as a float: same fallback.
        - Parses fine but exceeds :data:`_MAX_RISK_PCT` (5%, including a
          *negative* value clamped up to ``0.0``): hard-capped --
          Redis returning a corrupted or malicious value can never push
          a single trade's risk past this ceiling.
        """
        if self._redis_client is None:
            self._log.warning("dynamic_risk_pct_no_redis_using_fallback", fallback=_FALLBACK_RISK_PCT)
            return _FALLBACK_RISK_PCT

        try:
            raw = await asyncio.wait_for(
                self._redis_client.execute("get", _RISK_PCT_REDIS_KEY), timeout=2.0
            )
        except (RedisUnavailableError, asyncio.TimeoutError):
            self._log.warning("dynamic_risk_pct_read_failed_using_fallback", fallback=_FALLBACK_RISK_PCT)
            return _FALLBACK_RISK_PCT

        if raw is None:
            return _FALLBACK_RISK_PCT

        try:
            risk_pct = float(raw)
        except (TypeError, ValueError):
            self._log.warning(
                "dynamic_risk_pct_malformed_using_fallback", raw=raw, fallback=_FALLBACK_RISK_PCT
            )
            return _FALLBACK_RISK_PCT

        capped = min(max(risk_pct, 0.0), _MAX_RISK_PCT)
        if capped != risk_pct:
            self._log.warning(
                "dynamic_risk_pct_hard_capped", agent_id=self.agent_id, requested=risk_pct, capped=capped
            )
        return capped

    # -- Explainability report (dashboard telemetry only) ------------------

    @staticmethod
    def _describe_stance(signal: TradingSignal | None, *, agent_id: str) -> str:
        """Render one agent's raw signal as a short, human-readable stance line.

        **Not genuine model interpretability.** Neither underlying model
        exposes a real reasoning chain -- the RL policy may still be the
        untrained skeleton (see :mod:`src.agents.rl_signal.policy_model`'s
        docstring), and the XGBoost model's ``confidence`` is just a
        class probability. This is templated language wrapping the raw
        ``(signal, confidence)`` pair the agent actually published nothing
        more; it exists so the dashboard's decision log reads as prose
        instead of a tuple of numbers.
        """
        if signal is None:
            return "No fresh opinion available this cycle."
        direction = {1: "Bullish", -1: "Bearish", 0: "Neutral"}[signal.signal]
        flavor = _STANCE_FLAVOR.get(agent_id, "its own model output")
        return f"{direction} (Confidence: {round(signal.confidence * 100)}%) - based on {flavor}."

    async def generate_decision_report(
        self,
        *,
        instrument: str,
        consensus_result: Literal["consensus", "conflict", "waiting"],
        rl_signal: TradingSignal | None,
        xgb_signal: TradingSignal | None,
        current_price: float | None,
        risk_pct: float | None,
        execution_action: Literal["BUY", "SELL", "HOLD", "VETO"],
    ) -> str:
        """Build, log, and persist a one-paragraph explanation of one trading decision.

        Called once per consensus evaluation (never per price tick): on
        a veto/conflict, and on every consensus reached -- whether it
        actually results in an order (``BUY``/``SELL``) or is a no-op
        because a position is already open or cooling down (``HOLD``).
        Logged via structlog (so it lands in the same audit trail as
        every other decision this agent makes) and pushed to
        :data:`_DECISION_LOG_REDIS_KEY` (``LPUSH`` + ``LTRIM`` to the
        most recent :data:`_DECISION_LOG_MAX_ENTRIES`) for
        ``src/dashboard/app.py``'s "Nothierly AI Brainwaves" expander.

        Args:
            instrument: Which instrument this decision concerns.
            consensus_result: Why this decision point was reached.
            rl_signal: The RL agent's signal this cycle, if any.
            xgb_signal: The XGBoost agent's signal this cycle, if any.
            current_price: Last known tick for this instrument, if any.
            risk_pct: The dynamic risk fraction that will be applied,
                only when an entry is actually being attempted --
                ``None`` for a veto or a skipped (``HOLD``) consensus.
            execution_action: What this agent is about to do.

        Returns:
            The report text itself, as also logged/persisted.
        """
        reasoning = {
            "consensus": (
                "Both required signal agents agreed, within the freshness window, "
                "on the same non-neutral direction."
            ),
            "conflict": (
                "The required signal agents disagreed, or at least one was neutral "
                "-- entry vetoed for safety."
            ),
            "waiting": "Not every required agent has published a fresh signal yet.",
        }[consensus_result]

        risk_line = (
            f"Risking {risk_pct * 100:.1f}% of account equity on this entry."
            if risk_pct is not None
            else "No new capital at risk (no entry placed this cycle)."
        )
        price_line = f" Last price: {current_price:.5f}." if current_price is not None else ""

        report = (
            "[NOTHIERLY AI DECISION LOG]\n"
            f"Action Taken: {execution_action}\n"
            f"Reasoning: {reasoning}\n"
            f"RL Agent Stance: {self._describe_stance(rl_signal, agent_id='rl-signal-agent')}\n"
            f"XGBoost Agent Stance: {self._describe_stance(xgb_signal, agent_id='supervisor-agent')}\n"
            f"Risk Parameters: {risk_line}{price_line}"
        )

        self._log.info(
            "nothierly_decision_report",
            agent_id=self.agent_id,
            instrument=instrument,
            action=execution_action,
            report=report,
        )

        if self._redis_client is not None:
            entry = json.dumps(
                {
                    "timestamp": time.time(),
                    "instrument": instrument,
                    "action": execution_action,
                    "report": report,
                }
            )
            try:
                await self._redis_client.execute("lpush", _DECISION_LOG_REDIS_KEY, entry)
                await self._redis_client.execute(
                    "ltrim", _DECISION_LOG_REDIS_KEY, 0, _DECISION_LOG_MAX_ENTRIES - 1
                )
            except RedisUnavailableError:
                self._log.warning("nothierly_decision_report_persist_failed", agent_id=self.agent_id)

        return report

    @staticmethod
    def _sentiment_from_signals(xgb_signal: int | None, rl_signal: int | None) -> Literal["BULLISH", "BEARISH", "NEUTRAL"]:
        """Collapse this instrument's latest raw signals into one radar-card sentiment.

        ``BULLISH``/``BEARISH`` only when every signal received so far
        *agrees* on that direction (mirrors the same all-agree rule
        :meth:`_evaluate_consensus` uses for an actual entry, just
        without the freshness/completeness requirements -- this is
        informational telemetry, not a trading decision). Any
        disagreement, a neutral reading, or no data yet all collapse to
        ``NEUTRAL``.
        """
        signals = [s for s in (xgb_signal, rl_signal) if s is not None]
        if signals and all(s == 1 for s in signals):
            return "BULLISH"
        if signals and all(s == -1 for s in signals):
            return "BEARISH"
        return "NEUTRAL"

    async def _push_raw_signal_feed(self, signal: TradingSignal) -> None:
        """Append one raw, as-received signal to :data:`_RAW_SIGNAL_FEED_REDIS_KEY`.

        Unlike :meth:`_update_market_state` (one record per instrument,
        overwritten every call), this is an append-only log -- every
        individual signal this agent receives gets its own entry, in
        the exact order received, capped to the most recent
        :data:`_RAW_SIGNAL_FEED_MAX_ENTRIES`.
        """
        if self._redis_client is None:
            return

        entry = json.dumps(
            {
                "timestamp": signal.timestamp,
                "agent_id": signal.agent_id,
                "instrument": signal.instrument,
                "signal": signal.signal,
                "confidence": round(signal.confidence, 4),
            }
        )
        try:
            await self._redis_client.execute("lpush", _RAW_SIGNAL_FEED_REDIS_KEY, entry)
            await self._redis_client.execute(
                "ltrim", _RAW_SIGNAL_FEED_REDIS_KEY, 0, _RAW_SIGNAL_FEED_MAX_ENTRIES - 1
            )
        except RedisUnavailableError:
            self._log.warning("nothierly_raw_signal_feed_persist_failed", agent_id=self.agent_id)

    async def _update_market_state(self) -> None:
        """Rebuild and persist :data:`_MARKET_STATE_REDIS_KEY` for the dashboard's radar.

        One JSON object, keyed by instrument, each value a flat record
        -- ``{instrument: {...}, ...}``. Rebuilt from scratch from
        current in-memory state on every call (cheap: at most a handful
        of instruments) rather than patched incrementally, so it can
        never drift from what this agent actually knows. Called after
        every signal received and every price tick -- see the module
        docstring's multi-asset paragraph for why each instrument's data
        here is independent of every other's.

        ``src/dashboard/app.py`` turns this directly into a table via
        ``pd.DataFrame(list(state.values()))`` -- seeing this dict's
        values() to be tabular data is precisely why every record below
        shares the same flat set of keys, with ``None`` rather than a
        missing key when a value isn't known yet.
        """
        if self._redis_client is None:
            return

        state: dict[str, dict[str, Any]] = {}
        for instrument in self._instruments:
            signals = self._signals.get(instrument, {})
            xgb = signals.get("supervisor-agent")
            rl = signals.get("rl-signal-agent")
            state[instrument] = {
                "instrument": instrument,
                "last_price": self._last_prices.get(instrument),
                "xgboost_signal": xgb.signal if xgb is not None else None,
                "xgboost_confidence": round(xgb.confidence, 4) if xgb is not None else None,
                "rl_signal": rl.signal if rl is not None else None,
                "rl_confidence": round(rl.confidence, 4) if rl is not None else None,
                "sentiment": self._sentiment_from_signals(
                    xgb.signal if xgb is not None else None, rl.signal if rl is not None else None
                ),
                "updated_at": time.time(),
            }

        try:
            await self._redis_client.execute("set", _MARKET_STATE_REDIS_KEY, json.dumps(state))
        except RedisUnavailableError:
            self._log.warning("nothierly_market_state_persist_failed", agent_id=self.agent_id)

    async def _publish_active_positions(self) -> None:
        """Rebuild and persist :data:`_ACTIVE_POSITIONS_REDIS_KEY` for the dashboard's radar.

        One JSON array, one record per **non-``FLAT``** instrument --
        ``[{...}, {...}]`` -- so ``pd.DataFrame(positions)`` turns it
        directly into a table with no reshaping on the dashboard side.
        Called on every open, every close, and every price tick while a
        position is open (so unrealized PnL stays live, not frozen at
        entry) -- see the call sites in :meth:`_place_order`,
        :meth:`_close_position`, and :meth:`_handle_price_message`.
        """
        if self._redis_client is None:
            return

        positions: list[dict[str, Any]] = []
        for instrument, position in self._positions.items():
            if position.status == "FLAT" or position.entry_price is None or position.units_held is None:
                continue
            current_price = self._last_prices.get(instrument, position.entry_price)
            pnl = (current_price - position.entry_price) * position.units_held
            notional = abs(position.units_held) * position.entry_price
            positions.append(
                {
                    "instrument": instrument,
                    "status": position.status,
                    "entry_price": position.entry_price,
                    "current_price": current_price,
                    "units": round(position.units_held, 6),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / notional, 4) if notional > 0 else 0.0,
                    "opened_at": position.opened_at,
                }
            )

        try:
            await self._redis_client.execute("set", _ACTIVE_POSITIONS_REDIS_KEY, json.dumps(positions))
        except RedisUnavailableError:
            self._log.warning("nothierly_active_positions_persist_failed", agent_id=self.agent_id)

    async def _log_fill(
        self,
        *,
        instrument: str,
        action: Literal["ENTRY_LONG", "ENTRY_SHORT", "EXIT"],
        price: float,
        units: float,
        latency_ms: float,
        reason: str | None = None,
        agent_control_shadow: bool = False,
    ) -> None:
        """Persist one real fill onto :data:`_FILLS_BLOTTER_REDIS_KEY` -- for the Order Blotter & TCA page.

        **No Maker/Taker skew here, by design, not omission**: this
        agent only ever places simple market orders (see the module
        docstring) -- the Maker-Skew/Taker-Sweep quoting distinction is
        a research-only concept from ``research/market_making_env.py``
        (Chapter 12's RL market-making environment), never wired into
        this production execution path. ``action`` reflects what this
        agent actually does (open long, open short, or exit), not that
        research vocabulary.

        Args:
            instrument: Which instrument this fill was on.
            action: What kind of fill this was.
            price: The real (or ``MOCK_MODE``-simulated) fill price.
            units: Signed units transacted (positive = bought, negative
                = sold) -- matches :attr:`PositionState.units_held`'s
                own sign convention.
            latency_ms: Real wall-clock time the order call itself took,
                measured by the caller around ``create_market_order``.
                Near-zero and honestly labeled as such in ``MOCK_MODE``
                (no network call is made -- see ``mock_mode`` below).
            reason: Human-readable exit reason (``_close_position``'s
                own ``reason`` argument) -- ``None`` for an entry.
            agent_control_shadow: Whether this specific fill was
                simulated because of the real, operator-set
                :class:`~core.base_agent.AgentControlState.SHADOW`
                (independent of, and named distinctly from, this
                project's *other*, unrelated ``ExecutionMode.SHADOW``
                concept -- see :mod:`api_middleware.gateway`'s own
                per-order simulation, a different mechanism entirely).
                Read by ``src/dashboard/ems_pages/order_blotter.py`` to
                distinguish a MOCK_MODE fill from an operator-paused
                agent's fill, both of which are equally "not a real
                order" but for different, honestly-distinguished reasons.
        """
        if self._redis_client is None:
            return

        entry = json.dumps(
            {
                "timestamp": time.time(),
                "agent_id": self.agent_id,
                "instrument": instrument,
                "action": action,
                "price": round(price, 6),
                "units": round(units, 6),
                "latency_ms": round(latency_ms, 2),
                "reason": reason,
                "mock_mode": self._mock_mode,
                "agent_control_shadow": agent_control_shadow,
            }
        )
        try:
            await self._redis_client.execute("lpush", _FILLS_BLOTTER_REDIS_KEY, entry)
            await self._redis_client.execute("ltrim", _FILLS_BLOTTER_REDIS_KEY, 0, _FILLS_BLOTTER_MAX_ENTRIES - 1)
        except RedisUnavailableError:
            self._log.warning("nothierly_fills_blotter_persist_failed", agent_id=self.agent_id)

    # -- Guardian AI (Chief Risk Officer) -- triggers 3 & 4 ------------------
    #
    # Triggers 1 (risk increase) and 2 (manual override) are detected in
    # src/dashboard/app.py instead, since that's where the human action
    # they respond to actually happens -- see this module's docstring.

    def _build_market_context(self) -> str:
        """Render current capital + open positions as plain text for the Guardian AI prompt.

        Shared by both backend triggers (:meth:`_check_drawdown_trigger`,
        :meth:`_record_conflict_and_maybe_alert`) -- whatever specific
        thing tripped, the CRO's system prompt asks for "le contexte
        actuel du marché, les positions ouvertes", so both callers prefix
        this onto their own trigger-specific summary.
        """
        lines = [f"Capital (MOCK_MODE={self._mock_mode}) : {self._mock_balance:.2f}$."]
        open_positions = [(instrument, position) for instrument, position in self._positions.items() if position.status != "FLAT"]
        if not open_positions:
            lines.append("Aucune position ouverte actuellement.")
        else:
            lines.append("Positions ouvertes :")
            for instrument, position in open_positions:
                current_price = self._last_prices.get(instrument, position.entry_price)
                lines.append(
                    f"  - {instrument} : {position.status}, {position.units_held} unites, "
                    f"entree {position.entry_price}, prix actuel {current_price}."
                )
        return "\n".join(lines)

    async def _trigger_guardian_alert(self, *, trigger: TriggerType, summary: str) -> None:
        """Ask the Guardian AI for a verdict, then push it onto :data:`_GUARDIAN_ALERTS_REDIS_KEY`.

        Args:
            trigger: ``"DRAWDOWN"`` or ``"SIGNAL_COLLISION"`` -- the only
                two triggers this backend module ever raises itself (see
                the section docstring above).
            summary: One-line description of the specific condition that
                fired -- becomes the alert's headline on the dashboard.
        """
        alert = await assess_risk(
            trigger=trigger, summary=summary, context=self._build_market_context(), settings=self._settings
        )
        self._log.warning(
            "nothierly_guardian_alert", agent_id=self.agent_id, trigger=trigger, summary=summary, source=alert.source
        )
        if self._redis_client is None:
            return
        try:
            await self._redis_client.execute("lpush", _GUARDIAN_ALERTS_REDIS_KEY, alert.model_dump_json())
            await self._redis_client.execute(
                "ltrim", _GUARDIAN_ALERTS_REDIS_KEY, 0, _GUARDIAN_ALERTS_MAX_ENTRIES - 1
            )
        except RedisUnavailableError:
            self._log.warning("nothierly_guardian_alert_persist_failed", agent_id=self.agent_id)

    async def _get_dynamic_drawdown_threshold_pct(self) -> float:
        """Read the live drawdown-alert threshold from Redis, hard-clamped and fail-safe.

        Mirrors :meth:`_get_dynamic_risk_pct`'s exact fail-safe shape:
        any failure (no Redis, timeout, missing/unparseable key) falls
        back to the real default, :data:`DRAWDOWN_ALERT_THRESHOLD_PCT`
        (5%); a value that parses but falls outside
        [:data:`_MINDRAWDOWN_ALERT_THRESHOLD_PCT`,
        :data:`_MAXDRAWDOWN_ALERT_THRESHOLD_PCT`] is clamped, never
        trusted verbatim.
        """
        if self._redis_client is None:
            return DRAWDOWN_ALERT_THRESHOLD_PCT

        try:
            raw = await asyncio.wait_for(
                self._redis_client.execute("get", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY), timeout=2.0
            )
        except (RedisUnavailableError, asyncio.TimeoutError):
            self._log.warning(
                "dynamic_drawdown_threshold_read_failed_using_fallback", fallback=DRAWDOWN_ALERT_THRESHOLD_PCT
            )
            return DRAWDOWN_ALERT_THRESHOLD_PCT

        if raw is None:
            return DRAWDOWN_ALERT_THRESHOLD_PCT

        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            self._log.warning(
                "dynamic_drawdown_threshold_malformed_using_fallback", raw=raw, fallback=DRAWDOWN_ALERT_THRESHOLD_PCT
            )
            return DRAWDOWN_ALERT_THRESHOLD_PCT

        capped = min(max(threshold, _MINDRAWDOWN_ALERT_THRESHOLD_PCT), _MAXDRAWDOWN_ALERT_THRESHOLD_PCT)
        if capped != threshold:
            self._log.warning(
                "dynamic_drawdown_threshold_clamped", agent_id=self.agent_id, requested=threshold, clamped=capped
            )
        return capped

    async def _check_drawdown_trigger(self, current_balance: float) -> None:
        """Trigger 3: alert once capital has dropped >= the live drawdown threshold within a rolling 24h window.

        Args:
            current_balance: The account balance just observed --
                :attr:`_mock_balance` in ``MOCK_MODE``, or the freshly
                fetched real OANDA balance otherwise (see the two call
                sites in :meth:`_place_order`/:meth:`_close_position`).
        """
        now = time.time()
        self._balance_history.append((now, current_balance))
        while self._balance_history and now - self._balance_history[0][0] > _DRAWDOWN_WINDOW_SECONDS:
            self._balance_history.popleft()
        await self._publish_equity_point(now, current_balance)

        _baseline_timestamp, baseline_balance = self._balance_history[0]
        if baseline_balance <= 0:
            return
        drawdown_pct = (baseline_balance - current_balance) / baseline_balance

        alert_threshold = await self._get_dynamic_drawdown_threshold_pct()
        rearm_threshold = alert_threshold * _DRAWDOWN_REARM_RATIO

        if drawdown_pct >= alert_threshold and self._drawdown_alert_armed:
            self._drawdown_alert_armed = False
            await self._trigger_guardian_alert(
                trigger="DRAWDOWN",
                summary=(
                    f"Le capital a chute de {drawdown_pct:.1%} au cours des dernieres 24h "
                    f"({baseline_balance:.2f}$ -> {current_balance:.2f}$, seuil configure : {alert_threshold:.1%})."
                ),
            )
        elif drawdown_pct < rearm_threshold:
            self._drawdown_alert_armed = True

    async def _publish_equity_point(self, timestamp: float, balance: float) -> None:
        """Append one real ``(timestamp, balance)`` sample onto :data:`_EQUITY_CURVE_REDIS_KEY`.

        Called from :meth:`_check_drawdown_trigger` -- every point this
        pushes is a balance change that agent's own trading actually
        produced (a real close, in ``MOCK_MODE`` or otherwise), never a
        synthetic or interpolated one. See that constant's docstring on
        why this matters for the dashboard's equity curve.
        """
        if self._redis_client is None:
            return
        entry = json.dumps({"timestamp": timestamp, "balance": round(balance, 2)})
        try:
            await self._redis_client.execute("lpush", _EQUITY_CURVE_REDIS_KEY, entry)
            await self._redis_client.execute("ltrim", _EQUITY_CURVE_REDIS_KEY, 0, _EQUITY_CURVE_MAX_ENTRIES - 1)
        except RedisUnavailableError:
            self._log.warning("nothierly_equity_curve_persist_failed", agent_id=self.agent_id)

    async def _record_conflict_and_maybe_alert(self, instrument: str) -> None:
        """Trigger 4 ("Collision"): alert once an instrument conflicts >3 times within 10 minutes.

        Called once per :meth:`_evaluate_consensus` ``"conflict"``
        outcome for ``instrument`` -- each conflict is one instance of
        XGBoost and RL contradicting each other on this instrument, and
        this counts how many landed within the last
        :data:`_COLLISION_WINDOW_SECONDS`.
        """
        now = time.time()
        history = self._conflict_history.setdefault(instrument, deque())
        history.append(now)
        while history and now - history[0] > _COLLISION_WINDOW_SECONDS:
            history.popleft()

        armed = self._collision_alert_armed.setdefault(instrument, True)
        if len(history) > _COLLISION_THRESHOLD:
            if armed:
                self._collision_alert_armed[instrument] = False
                await self._trigger_guardian_alert(
                    trigger="SIGNAL_COLLISION",
                    summary=(
                        f"{instrument} : {len(history)} signaux contradictoires entre XGBoost et RL "
                        f"en moins de {_COLLISION_WINDOW_SECONDS / 60:.0f} minutes."
                    ),
                )
        else:
            self._collision_alert_armed[instrument] = True

    # -- Exit (never Kill-Switch-gated) -------------------------------------

    async def cancel_all_positions(self, *, reason: str) -> None:
        """Flatten every open position, immediately -- the real "Cancel All" this system didn't have.

        Added for Chapter 13's "Red Phone" circuit breaker
        (``EMERGENCY_HALT_TOPIC``, see :meth:`_handle_message`): no
        unified "cancel all" ever existed on this agent before --
        :meth:`_close_position` only ever closed one instrument at a
        time, called from price-triggered exit logic. This simply calls
        it for every instrument currently non-``"FLAT"``, sequentially
        (not concurrently -- these are real order calls against a single
        account's margin/exposure state; the existing
        :meth:`_close_position` codepath was never written to be called
        concurrently with itself across instruments, and there is no
        real urgency benefit to parallelizing a handful of exits over
        the small additional risk of doing so for the first time here).
        Uses the exact same never-Kill-Switch-gated path every other
        exit already does -- flattening a position always reduces risk,
        so it must work even mid-PANIC.

        Args:
            reason: Human-readable reason, logged and passed through to
                every individual :meth:`_close_position` call.
        """
        open_instruments = [
            instrument for instrument, position in self._positions.items() if position.status != "FLAT"
        ]
        self._log.warning(
            "execution_agent_cancel_all_positions",
            agent_id=self.agent_id,
            reason=reason,
            open_instruments=open_instruments,
        )
        for instrument in open_instruments:
            await self._close_position(instrument, reason=f"EMERGENCY_HALT: {reason}")

    async def _close_position(self, instrument: str, *, reason: str, price: float | None = None) -> None:
        """Close ``instrument``'s open position immediately, at market.

        **Deliberately not guarded by ``@requires_system_active``** (see
        the module docstring): a stop-loss, take-profit, trailing-stop,
        or agent-veto exit reduces risk and must always be able to fire,
        including during a global PANIC -- the exact same reasoning
        already applied to :meth:`~src.agents.supervisor.broker_client.
        OandaClient.cancel_all_orders`. ``safe_financial_call`` still
        wraps the real order below for its timeout and
        failure-trips-the-switch behavior, just not the pre-emptive
        entry gate.

        Args:
            instrument: Which instrument to close -- one of
                :attr:`_instruments`; every other instrument's own
                position is untouched.
            reason: Human-readable exit reason, logged verbatim and
                MOCK_MODE-logged per this project's explicit
                requirement -- one of ``"Sortie par Stop-Loss"``,
                ``"Sortie par Take-Profit"``, ``"Sortie par Trailing
                Stop"``, or ``"Sortie par Veto Agent (<agent_id>)"``.
            price: The price to close at. Falls back to the last known
                tick (or the entry price, if none has arrived yet -- an
                honest zero-PnL rather than a crash, for exactly the
                situation this project's own asset-class gap causes
                today: no live price ever arrives for the OANDA
                instrument this agent trades).
        """
        position = self._positions.get(instrument)
        if position is None or position.status == "FLAT" or position.entry_price is None or position.units_held is None:
            return  # nothing open (e.g. a veto and a price-trigger raced; the first one already closed it)

        close_price = price if price is not None else self._last_prices.get(instrument, position.entry_price)
        closing_units = -position.units_held
        pnl = (close_price - position.entry_price) * position.units_held

        if position.entered_as_simulated:
            # Closed the same way it was opened -- see PositionState.
            # entered_as_simulated's own docstring on why this reads the
            # position's own stored flag, never self._mock_mode/
            # control_state directly (either can have changed since
            # entry). self._mock_balance is updated regardless of
            # *which* reason this position was simulated for: entry
            # sizing already drew against it either way (see
            # _place_order's own simulate_this_entry), so closing must
            # settle the same fictional ledger it was opened against.
            self._mock_balance += pnl
            await self._check_drawdown_trigger(self._mock_balance)
            self._log.warning(
                "SHADOW_MODE_SIMULATED_ORDER" if position.entered_as_shadow_controlled else "MOCK_MODE_SIMULATED_ORDER",
                agent_id=self.agent_id,
                instrument=instrument,
                action="CLOSE",
                reason=reason,
                units=closing_units,
                entry_price=position.entry_price,
                close_price=close_price,
                pnl=round(pnl, 2),
                mock_balance=round(self._mock_balance, 2),
                detail=(
                    "This position was entered under control_state=SHADOW: no real OANDA call was made."
                    if position.entered_as_shadow_controlled
                    else "MOCK_MODE=true: no real OANDA call was made."
                ),
            )
            await self._log_fill(
                instrument=instrument, action="EXIT", price=close_price, units=closing_units,
                latency_ms=0.0, reason=reason, agent_control_shadow=position.entered_as_shadow_controlled,
            )
        else:
            close_start = time.monotonic()
            try:
                order = await safe_financial_call(
                    self._trading_client.create_market_order,
                    instrument,
                    int(closing_units),
                    retry_on_failure=False,
                )
            except FinancialCallAbortedError:
                self._log.error(
                    "position_close_failed_leaving_position_open",
                    agent_id=self.agent_id,
                    instrument=instrument,
                    reason=reason,
                    detail=(
                        "MANUAL INTERVENTION MAY BE REQUIRED: the close order failed and the "
                        "Global Kill Switch has been tripped by safe_financial_call -- this "
                        "position is still open at OANDA."
                    ),
                )
                return
            close_latency_ms = (time.monotonic() - close_start) * 1000.0
            await self._log_fill(
                instrument=instrument, action="EXIT", price=close_price, units=closing_units,
                latency_ms=close_latency_ms, reason=reason,
            )
            self._log.warning(
                "consensus_position_closed",
                agent_id=self.agent_id,
                instrument=instrument,
                reason=reason,
                order_id=order.get("id"),
                units=closing_units,
                entry_price=position.entry_price,
                close_price=close_price,
                pnl=round(pnl, 2),
            )
            execution = TradeExecution(
                symbol=instrument,
                side=TradeSide.SELL if position.status == "LONG" else TradeSide.BUY,
                cost=abs(closing_units) * close_price,
                order_id=str(order.get("id", "unknown")),
            )
            await self.publish(_TRADE_EXECUTED_TOPIC, execution.model_dump(mode="json"))

        self._current_exposure = max(0.0, self._current_exposure - abs(position.units_held) * position.entry_price)
        self._positions[instrument] = PositionState(cooldown_until=time.time() + self._cooldown_seconds)
        await self._publish_active_positions()

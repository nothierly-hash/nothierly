"""Autonomous prop market-making engine ("Alpha Engine") -- captures the spread, never bets on direction.

**Corrections to the brief, verified against the real code first** (this
project's established discipline -- see e.g. ``tests/chaos_monkey.py``,
``src/execution/smart_order_router.py``'s own module docstring):

1. **"Risque de perte directionnelle approchant le zéro absolu"** --
   false as literally stated, and worth being precise about rather than
   quietly implementing something that overclaims safety. Market making
   captures the bid/ask spread instead of betting on direction, but it
   is not risk-free: it carries real inventory risk (price can move
   against an open position before it is hedged) and real adverse-
   selection risk (informed flow fills a quote that was stale by the
   time it landed). :class:`AlphaMarketMakerAgent` minimizes, never
   eliminates, that risk via a hard inventory cap plus a forced hedge --
   "near-zero" here is a claim about *bounded exposure*, not about
   eliminated risk.
2. **"Écoute le MarketDataAgent via Redis Pub/Sub"** --
   :class:`~src.agents.market_data.agent.MarketDataAgent` does not
   publish market data over Redis Pub/Sub anywhere in this codebase; it
   writes a real :class:`~core.schemas.MarketMetrics` JSON blob to a
   short-TTL Redis STRING key (``market_metrics:{symbol}``, see
   :data:`~core.schemas.MARKET_METRICS_REDIS_KEY_PREFIX`) and separately
   publishes a real ``MARKET_DATA_L2`` message over ZMQ. This engine
   reads the Redis cache -- the same real producer/consumer contract
   every other reader of that key already uses -- and, unlike every
   other reader of that key found in this project's own multi-agent-
   coordination audit, explicitly checks its age before trusting it
   (see :meth:`AlphaMarketMakerAgent._read_fresh_market_metrics`). A
   real, confirmed gap in this codebase before this engine existed: no
   consumer of that key anywhere checked staleness beyond the Redis TTL
   itself being present/absent.
3. **"Lire la clé Redis NOTHIERLY_BOT_STATUS à chaque cycle"** -- this
   project already has a real, tested, race-free per-agent on/off
   mechanism: :class:`~core.base_agent.AgentControlState`
   (``LIVE``/``SHADOW``/``OFF``), persisted *and* pushed via Redis
   Pub/Sub, read from an in-memory attribute on every hot-path check --
   zero per-tick Redis round trip (see
   :meth:`~core.base_agent.BaseAgent._listen_for_control_commands`'s own
   docstring on why a per-tick ``GET`` would be strictly worse, both for
   latency and for correctness). Reused here rather than inventing a
   second, parallel on/off primitive: the multi-agent-coordination audit
   that preceded this engine found real, self-acknowledged bugs from
   exactly that pattern (two uncoordinated writers to one piece of
   shared Redis state -- see ``api_middleware/gateway.py``'s and
   ``src/agents/macro_order_execution/agent.py``'s real, disclosed
   ``order_status:{order_id}`` race). ``AgentControlState.OFF`` *is*
   this engine's real ``NOTHIERLY_BOT_STATUS=OFF`` -- see
   :meth:`AlphaMarketMakerAgent._run_cycle`'s first check.
4. **"backend/autonomous_alpha_engine.py"** -- placed at
   ``src/agents/alpha_engine/agent.py`` instead, matching every other
   real agent's location in this codebase (``src/agents/<name>/agent.py``
   -- ``execution``, ``rl_signal``, ``macro_order_execution``, ...).
   There is no ``backend/`` directory anywhere in this project. See
   ``scripts/run_alpha_engine.py`` for the standalone launcher every
   other agent also has one of.
5. **"À chaque cycle (milliseconde)"** -- also not implemented literally,
   on purpose, stated plainly rather than silently defaulting to
   something else and calling it done. This project has zero latency/
   colocation infrastructure anywhere (confirmed by the same audit: no
   RTT measurement, no latency-aware venue selection, a plain ``ccxt.pro``
   WebSocket client over the public internet) -- a millisecond decision
   cycle without colocation is not real market-making, it is a bot that
   *thinks* it is fast. :data:`_DEFAULT_TICK_INTERVAL_SECONDS` defaults
   to a conservative, honest 1.0s instead. What the brief actually asked
   for and *is* implemented exactly as specified: the on/off check
   happens before every single order this engine ever places, with zero
   added latency (see point 3) -- the cycle *period* is the separate,
   disclosed engineering constraint above, not a compliance gap in the
   on/off check itself.

**What this engine actually does, each cycle** (see :meth:`_run_cycle`):
cancels its own resting quotes, reads a fresh ``MarketMetrics`` snapshot
(skipping the cycle entirely if it is missing or stale), reads its real,
current inventory directly from the broker (never reconstructed purely
from locally-tracked fills -- see point 6 below), computes a bid/ask
around mid-price skewed by that inventory (the further it drifts from
zero, the more both quotes lean toward flattening it -- classic
inventory-skew market making, not Avellaneda-Stoikes-optimal, stated as
such), and posts two real limit orders sized to never push inventory past
:data:`_DEFAULT_MAX_INVENTORY_UNITS`. If inventory ever exceeds that cap
anyway (a fill can always race a requote), the engine stops quoting
entirely and fires an immediate real market order to flatten the excess
-- a Delta-Neutral hedge, not a bet that price reverts.

6. **Real inventory, read from the broker, not accumulated locally.** The
   safety-critical inventory-cap check (:meth:`_run_cycle`) always uses
   :meth:`_read_total_inventory`'s live
   :meth:`~src.execution.broker_gateway.BaseBroker.get_open_positions`
   read -- ground truth from the exchange itself -- never this engine's
   own (necessarily approximate, see point 7) fill bookkeeping. A bug in
   the PnL/cost-basis tracker below can produce a wrong number on the
   dashboard; it cannot make this engine exceed its real inventory cap.

   **Real, disclosed operational requirement this implies**: because the
   cap check reads the account's *entire* real balance of the base
   asset (not a locally-tracked "this engine's own fills" subtotal), any
   pre-existing holding of that asset in the same account counts against
   the cap too. Run this engine against a **dedicated exchange
   sub-account** holding nothing else -- not a shared account -- or the
   cap protects against a number that includes exposure this engine
   never created and cannot see the origin of.

7. **PnL: approximate, and honestly labeled as such.** This engine has no
   private per-fill WebSocket feed (``ccxt.pro``'s ``watchMyTrades``, not
   implemented here); it detects a fill as a quantity delta between two
   consecutive cycles' real position reads and prices it at that cycle's
   mid-price -- not the fill's real, exact execution price. Good enough
   for the Master Control Center's live inventory/PnL display; not a
   substitute for :mod:`src.execution.reporting_engine`'s real
   ``client_fills``-based accounting (this engine has no B2B client to
   attribute a fill to -- see that module's own docstring).

8. **"2-second cooldown between any two trades"** -- not implemented
   literally, because a literal reading would break this engine's actual
   job. A market maker is *supposed* to place two orders (bid + ask)
   every single cycle -- that is continuous quoting, not a bug, and a
   flat per-trade cooldown would silently turn this into a one-sided or
   stale-quoting engine instead. What actually guards against a runaway
   loop: (a) :meth:`_run_cycle` itself only runs once per
   ``tick_interval_seconds`` and is fully serialized by
   ``self._order_lock`` -- no code path can place orders faster than the
   tick rate allows, regardless of what any signal/model logic computes;
   (b) :meth:`_check_order_rate_limit`, a real circuit breaker on total
   order-placement *count* in a rolling window (default: 10 per 10s --
   generous vs. routine ~2/cycle, tight vs. hundreds/second), which self
   halts (see point 9) the instant it is exceeded, catching the actual
   failure mode described (a buggy model firing far faster than any
   legitimate quoting rate) without capping legitimate two-sided quoting.

9. **Fat-finger cap: real, but not at the broker/exchange-client layer.**
   :meth:`~src.execution.broker_gateway.BaseBroker`/
   :meth:`~src.agents.market_data.exchange_client.AsyncExchangeClient`
   are shared infrastructure -- :mod:`~src.execution.smart_order_router`,
   :class:`~src.agents.execution.agent.ExecutionAgent`, and
   :class:`~src.agents.macro_order_execution.agent.
   MacroOrderExecutionAgent` all place real orders through the exact
   same classes, each with its own, already-real, already-tested risk
   sizing (``RiskManager.check_exposure_limit``,
   ``PortfolioRiskParityManager``, ...). Hardcoding one engine's BTC size
   ceiling into that shared layer would silently cap every other real
   caller's orders too, for a limit that has nothing to do with their
   own risk policy. The real fat-finger guard instead lives in *this*
   engine, immediately before every real order call it makes
   (:meth:`_check_fat_finger`) -- the same "verify at the boundary
   closest to where money actually leaves" principle the brief asked
   for, applied at the boundary this engine actually owns.

10. **Deques, not lists, with real ``maxlen`` backstops.**
    :attr:`_trade_timestamps` and the new
    :attr:`_order_placement_timestamps` are bounded
    (``maxlen=100_000``/``10_000``) in addition to their own real,
    primary time-window pruning -- the ``maxlen`` is a pure OOM backstop
    for a pathological rate this engine's own circuit breaker
    (point 8) should already have halted long before either bound is
    ever reached. Checked, not assumed: this project's own multi-agent-
    coordination audit already found no unbounded list anywhere in
    :mod:`~src.agents.market_data.agent`/``exchange_client`` either (a
    second, independent check specifically for this engine's request,
    not re-derived from memory).

11. **Cold-start sanitizer: real, and corrected on one real premise.**
    :meth:`_bootstrap_cold_start_sanitizer` cancels every real orphaned
    order left resting from a possible previous crashed session (via
    :meth:`~src.execution.broker_gateway.BaseBroker.
    cancel_all_open_orders`, itself delegating to
    :meth:`~src.agents.market_data.exchange_client.
    AsyncExchangeClient.cancel_all_orders`, which already existed for
    exactly this before this engine did) and refuses to start if the
    real starting inventory already exceeds the cap. The one real
    premise this corrects: this engine never assumed inventory started
    at zero on restart (:meth:`on_start` already re-read the real
    broker balance every time) -- the actual, more subtle bug was that
    re-deriving the baseline fresh on *every* restart silently forgot
    whatever inventory this engine's own prior (crashed) session had
    legitimately accumulated, treating it as pre-existing/cap-exempt
    instead of counting it. Fixed by persisting the baseline once, in
    Redis (:data:`~core.schemas.ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX`,
    no TTL -- must survive a restart), and reusing it across every
    subsequent restart rather than recomputing it.

12. **A private-WebSocket ``listenKey`` keepalive does not apply to this
    engine, and would not be built the way briefed even if it did.**
    This engine holds no private authenticated WebSocket stream at all
    -- fills are detected by polling
    :meth:`~src.execution.broker_gateway.BaseBroker.get_open_positions`
    once per cycle (a real, disclosed approximation, see point 7), not
    ``ccxt.pro``'s ``watchMyTrades``. There is therefore no
    ``listenKey`` anywhere in this engine's real call graph to expire.
    Were ``watchMyTrades`` added later for lower-latency fill detection,
    ``ccxt.pro`` -- already this project's real WebSocket layer for
    :class:`~src.agents.market_data.agent.MarketDataAgent`'s public
    streams -- handles a private stream's exchange-specific session
    refresh internally as part of what it abstracts; hand-rolling a
    manual keepalive loop calling a raw ``ping_listen_key`` method that
    does not exist on :class:`~src.agents.market_data.exchange_client.
    AsyncExchangeClient` would duplicate, incorrectly, something the
    library this project already depends on already does.

13. **Rate-limit-ban protection: the real, already-built
    :class:`~network.resilience_mesh.RateLimitMesh`, wired in at this
    engine's own boundary, not the shared exchange-client layer.**
    A real, already-tested ``TokenBucket``-based proactive throttle (+
    real adaptive backoff on an actual ``ccxt`` ``RateLimitExceeded``/
    ``DDoSProtection``) already existed in this codebase -- unwired,
    unused by any caller, exactly the same "built the right thing,
    never plugged it in" pattern this project's own operational-
    blind-spots audit already found once. Wiring it project-wide (every
    caller of :class:`~src.agents.market_data.exchange_client.
    AsyncExchangeClient`, including :mod:`~src.execution.
    smart_order_router` and :class:`~src.agents.execution.agent.
    ExecutionAgent`) is a real, separate, larger change touching
    already-live components this engine's own hardening pass should not
    casually take on unasked -- scoped here to this engine's own real
    order-placing calls only. See :meth:`_place_resting_order`/
    :meth:`_hedge_excess_inventory`. Complements, does not duplicate,
    point 8's order-rate circuit breaker: the mesh *paces* calls to stay
    under the exchange's real published limit and *adapts* on an actual
    rejection; the circuit breaker *halts the engine* on a genuinely
    pathological burst. Different jobs, both real, kept both.

14. **Cross-asset session/spread gating -- real, wired in, and a genuine
    no-op for this engine's actual crypto instrument today.** Two new
    real checks in :meth:`_run_cycle`, both resolved through
    :mod:`core.asset_universe`/:class:`~core.session_manager.
    MarketSessionManager` by ``self._symbol`` alone -- neither branches
    on symbol identity (no ``if self._symbol == "BTC/USDT"`` anywhere):
    (a) the Friday Unwind (:meth:`~core.session_manager.
    MarketSessionManager.is_unwind_period`) cancels quotes, blocks new
    entries, and forces :meth:`_hedge_excess_inventory` down to a real
    zero-inventory target (``cap=Decimal(0)``, not just back under the
    normal cap) for the whole real weekly closed window, not only the
    trigger instant; (b) the anti-spread-manipulation filter
    (:func:`~core.asset_universe.is_spread_acceptable`) silently rejects
    the cycle -- cancels quotes, places nothing -- the instant
    :attr:`~core.schemas.MarketMetrics.spread` exceeds this asset's real
    configured ceiling, *before* :meth:`_requote` ever computes a price.
    Both are real, live checks against ``self._symbol``'s actual
    :class:`~core.asset_universe.AssetUniverseEntry` (resolved once, in
    ``__init__`` -- see that method) -- for ``BTC/USDT``,
    ``is_unwind_period`` is always ``False`` (crypto has
    ``trading_hours=None``) and the ``$5.00`` spread ceiling is wide
    enough that this is real, inert infrastructure today, not a
    behavior change to this engine's actual current operation. It
    becomes real protection the day this class is ever constructed
    against ``EUR_USD``/``XAU_USD``/``BCO_USD`` instead -- which it can
    be, unchanged, because nothing about ``_run_cycle``'s new gating
    logic assumes crypto. **Deliberately scoped to this one engine in
    this pass**: :class:`~src.agents.rl_signal.agent.RlSignalAgent` and
    :class:`~src.agents.market_data.agent.MarketDataAgent` are separate,
    already-live, already-tested production agents this same change did
    not also touch -- a larger, separate integration, not silently
    assumed done here.

**Kill-switch interaction**: every *routine* real order this engine
places (a resting quote, the automatic inventory-limit hedge in
:meth:`_hedge_excess_inventory`) goes through
:meth:`~src.execution.broker_gateway.BaseBroker.place_limit_order` /
``place_market_order``, gated by :func:`core.security.
requires_system_active` several layers down -- same as every other real
order call anywhere in this codebase. If the global Kill Switch is
tripped, those calls are correctly refused.

**Emergency Liquidate is the one deliberate exception -- a real
"break-glass"**, not the "kill-switch-blocks-its-own-cure" paradox this
project's own :mod:`~src.execution.smart_order_router` auto-unwind still
has (see that module's docstring for the version of this problem that is
*not* resolved). :meth:`_execute_emergency_liquidate` calls
:meth:`~src.execution.broker_gateway.BaseBroker.
emergency_flatten_market_order` instead of ``place_market_order`` --
real, disclosed, *not* gated by the global Kill Switch (see that
method's own docstring, and :mod:`core.security`'s module docstring, on
why this extends this project's own existing "cleanup calls are exempt"
principle rather than inventing a new one). It still does **not** retry
through :func:`core.security.safe_financial_call` (a second,
uncoordinated real market order stacked on an already-ambiguous first
attempt is a worse failure mode than one clearly logged, alerted
failure) -- on any exception it logs ``CRITICAL`` and pushes a "MANUAL
INTERVENTION REQUISE" Telegram alert instead of pretending the position
is flat. The bypass is narrow by construction, not by convention alone:
it is one named method, on one broker adapter, called from exactly one
place in this entire codebase.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Final, Literal

import zmq.asyncio

from core.asset_universe import AssetUniverseEntry, get_asset_universe_entry, is_spread_acceptable
from core.base_agent import AgentControlState, BaseAgent
from core.circuit_breaker import IngestionCircuitBreaker
from core.config import AppSettings
from core.exception_manager import install_global_exception_handler
from core.logger import get_logger
from core.message_integrity import verify
from core.nothierly_ontology import AssetClass, UniversalAsset
from core.notifier import AlertSeverity, TelegramNotifier
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    AGENT_CONTROL_CHANNEL_PREFIX,
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX,
    ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES,
    ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX,
    MARKET_METRICS_REDIS_KEY_PREFIX,
    ORDER_AUDIT_EVENTS_TOPIC,
    MarketMetrics,
)
from core.security import FinancialCallAbortedError, KillSwitchActiveError, safe_financial_call
from core.session_manager import MarketSessionManager
from core.watchdog import SystemWatchdog
from network.resilience_mesh import RateLimitMesh
from src.execution.audit_ledger import OrderAuditEvent
from src.execution.broker_gateway import BaseBroker, BrokerFactory

_log = get_logger(__name__)

# See module docstring point 5 on why this is not "millisecond".
_DEFAULT_TICK_INTERVAL_SECONDS: Final[float] = 1.0
_DEFAULT_MAX_INVENTORY_UNITS: Final[Decimal] = Decimal("0.5")
_DEFAULT_QUOTE_SIZE_UNITS: Final[Decimal] = Decimal("0.01")
_DEFAULT_HALF_SPREAD_BPS: Final[Decimal] = Decimal("5")
_DEFAULT_INVENTORY_SKEW_BPS: Final[Decimal] = Decimal("15")
_DEFAULT_MAX_MARKET_DATA_AGE_SECONDS: Final[float] = 1.5
_ONE_DAY_SECONDS: Final[float] = 86_400.0

# Order-rate circuit breaker (module docstring point 8): the real "no more than N real order
# placements in any rolling window" backstop against a runaway loop. Deliberately generous relative
# to normal operation (2 orders/cycle at the default 1.0s tick_interval_seconds is ~120/minute --
# routine, not a bug) so it never fires during healthy market making; it exists to catch hundreds of
# placements in seconds, the actual "buys 1000 times in 3 seconds" failure mode.
_DEFAULT_MAX_ORDER_RATE_COUNT: Final[int] = 10
_DEFAULT_ORDER_RATE_WINDOW_SECONDS: Final[float] = 10.0
# "Protocole AEGIS" (2026-09-16) -- see _market_data_circuit_breaker's own docstring for the real
# 51.5h/34k+36k-event incident this closes.
_DEFAULT_MAX_CONSECUTIVE_MARKET_DATA_ERRORS: Final[int] = 50
_DEFAULT_MARKET_DATA_ERROR_RATE: Final[float] = 0.05
_DEFAULT_MARKET_DATA_ERROR_WINDOW_SECONDS: Final[float] = 60.0

_TRADE_TIMESTAMPS_MAXLEN: Final[int] = 100_000  # OOM backstop -- see module docstring point 10
_ORDER_TIMESTAMPS_MAXLEN: Final[int] = 10_000

# See module docstring point 11: bounds AsyncExchangeClient.cancel_all_orders' own indefinite retry
# loop during startup specifically -- correct for that method's real intended caller (an
# already-running process reacting to a transient blip), wrong for a fresh process that should fail
# closed rather than hang indefinitely if the exchange is unreachable at the one moment it's trying
# to confirm a clean starting state.
_COLD_START_TIMEOUT_SECONDS: Final[float] = 30.0

# See module docstring point 13. Binance spot's real published order-rate limit is ~50 orders/10s
# (5/s) -- this defaults well under that (a real safety margin, not a value grazing the real limit),
# and is comfortably above this engine's own routine ~2 orders/cycle at the default 1.0s
# tick_interval_seconds. Confirm your venue's real published limit and tune via
# --order-rate-limit-per-second (scripts/run_alpha_engine.py) rather than trusting this blindly for
# a different exchange.
_DEFAULT_ORDER_RATE_LIMIT_PER_SECOND: Final[float] = 3.0


# =============================================================================
# Pure functions -- no I/O, real regression tests without a live exchange
# (see tests/test_alpha_engine.py). Same "pure core, tested for real"
# discipline as src/execution/audit_ledger.py's compute_event_hash/verify_chain.
# =============================================================================


def _skewed_quote_prices(
    *, mid_price: Decimal, half_spread_bps: Decimal, inventory: Decimal, max_inventory: Decimal, skew_bps: Decimal,
) -> tuple[Decimal, Decimal]:
    """Bid/ask around ``mid_price``, leaned by current inventory -- the Delta-Neutrality mechanism.

    Positive inventory (too long) shifts both quotes down: a lower ask
    makes it more attractive for a counterparty to buy from us (reduces
    inventory), and a lower bid makes it less attractive for us to buy
    more. Symmetric for negative (too short) inventory. Real economic
    pressure back toward zero even before :func:`_clamp_quote_sizes`'s
    hard cap is hit -- not a substitute for it.
    """
    inventory_fraction = inventory / max_inventory if max_inventory != 0 else Decimal(0)
    skew = (skew_bps / Decimal(10_000)) * inventory_fraction
    half_spread = half_spread_bps / Decimal(10_000)
    bid_price = mid_price * (Decimal(1) - half_spread - skew)
    ask_price = mid_price * (Decimal(1) + half_spread - skew)
    return bid_price, ask_price


def _clamp_quote_sizes(*, base_size: Decimal, inventory: Decimal, max_inventory: Decimal) -> tuple[Decimal, Decimal]:
    """Zero out whichever side of the quote would push ``inventory`` past ``max_inventory`` if fully filled.

    Binary, not a partial reduction: either the full clip size or zero,
    per side. A finer-grained "shrink the clip as the cap approaches"
    version is a real possible improvement, not implemented in this
    first pass -- stated as a real gap, not silently absent.
    """
    buy_size = base_size if inventory + base_size <= max_inventory else Decimal(0)
    sell_size = base_size if inventory - base_size >= -max_inventory else Decimal(0)
    return buy_size, sell_size


def _apply_fill(
    *, previous_qty: Decimal, previous_avg_entry_price: Decimal | None, fill_qty: Decimal, fill_price: Decimal,
) -> tuple[Decimal, Decimal | None, Decimal]:
    """Weighted-average-cost position update for one fill.

    Returns ``(new_qty, new_avg_entry_price, realized_pnl_delta_usd)``.
    Growing a position (same sign as what is already held, or opening
    from flat) blends the cost basis. Shrinking, exactly closing, or
    flipping through flat realizes PnL on the closed portion at the
    *old* average entry price (never the new fill's price) -- standard
    weighted-average-cost accounting, not FIFO/LIFO lot matching (this
    engine holds at most one net position per symbol, so lot matching
    would add complexity with no behavioral difference here).
    """
    new_qty = previous_qty + fill_qty
    if previous_qty == 0:
        return new_qty, fill_price, Decimal(0)

    assert previous_avg_entry_price is not None  # invariant: non-zero qty always has a real cost basis
    same_direction = (previous_qty > 0) == (fill_qty > 0)

    if same_direction:
        total_cost = previous_avg_entry_price * previous_qty + fill_price * fill_qty
        new_avg_entry_price = total_cost / new_qty if new_qty != 0 else None
        return new_qty, new_avg_entry_price, Decimal(0)

    closed_qty = min(abs(fill_qty), abs(previous_qty))
    direction = Decimal(1) if previous_qty > 0 else Decimal(-1)
    realized_delta = direction * closed_qty * (fill_price - previous_avg_entry_price)

    if abs(fill_qty) <= abs(previous_qty):
        # Partial or exact close -- any remaining position keeps the same cost basis.
        return new_qty, (previous_avg_entry_price if new_qty != 0 else None), realized_delta
    # Flipped through flat -- the excess opens a fresh position at this fill's price.
    return new_qty, fill_price, realized_delta


class AlphaMarketMakerAgent(BaseAgent):
    """Single-venue, single-symbol autonomous market maker. See module docstring."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        *,
        exchange_id: str = "binance",
        tick_size: Decimal = Decimal("0.01"),
        lot_size: Decimal = Decimal("0.00001"),
        max_inventory_units: Decimal = _DEFAULT_MAX_INVENTORY_UNITS,
        quote_size_units: Decimal = _DEFAULT_QUOTE_SIZE_UNITS,
        half_spread_bps: Decimal = _DEFAULT_HALF_SPREAD_BPS,
        inventory_skew_bps: Decimal = _DEFAULT_INVENTORY_SKEW_BPS,
        max_market_data_age_seconds: float = _DEFAULT_MAX_MARKET_DATA_AGE_SECONDS,
        max_order_rate_count: int = _DEFAULT_MAX_ORDER_RATE_COUNT,
        max_order_rate_window_seconds: float = _DEFAULT_ORDER_RATE_WINDOW_SECONDS,
        max_consecutive_market_data_errors: int = _DEFAULT_MAX_CONSECUTIVE_MARKET_DATA_ERRORS,
        max_market_data_error_rate: float = _DEFAULT_MARKET_DATA_ERROR_RATE,
        market_data_error_window_seconds: float = _DEFAULT_MARKET_DATA_ERROR_WINDOW_SECONDS,
        max_single_order_units: Decimal | None = None,
        max_daily_loss_usd: Decimal | None = None,
        order_rate_limit_per_second: float = _DEFAULT_ORDER_RATE_LIMIT_PER_SECOND,
        tick_interval_seconds: float = _DEFAULT_TICK_INTERVAL_SECONDS,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        agent_id = f"alpha-engine-{symbol.replace('/', '-').lower()}"
        super().__init__(
            agent_id,
            tick_interval_seconds=tick_interval_seconds,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        base_currency, _, quote_currency = symbol.partition("/")
        self._symbol = symbol
        # Real, fail-fast registry lookup -- see module docstring point 14 (cross-asset session/spread
        # gating). Resolved once here, not per-cycle: a symbol with no real AssetUniverseEntry is a
        # configuration gap to catch at construction time, not something to silently skip or
        # re-discover as a KeyError deep inside the first real trading cycle.
        self._universe_entry: AssetUniverseEntry = get_asset_universe_entry(symbol)
        self._exchange_id = exchange_id
        self._asset = UniversalAsset(
            symbol=symbol, asset_class=AssetClass.CRYPTO_SPOT, base_currency=base_currency,
            quote_currency=quote_currency, tick_size=tick_size, lot_size=lot_size,
        )
        self._max_inventory_units = max_inventory_units
        self._quote_size_units = quote_size_units
        self._half_spread_bps = half_spread_bps
        self._inventory_skew_bps = inventory_skew_bps
        self._max_market_data_age_seconds = max_market_data_age_seconds
        self._max_order_rate_count = max_order_rate_count
        self._max_order_rate_window_seconds = max_order_rate_window_seconds
        # "Protocole AEGIS" (2026-09-16) -- real circuit breaker over
        # alpha_engine_market_metrics_unparseable/alpha_engine_stale_data_detected, see
        # _read_fresh_market_metrics/_handle_market_data_circuit_breaker_trip.
        self._market_data_circuit_breaker = IngestionCircuitBreaker(
            max_consecutive_failures=max_consecutive_market_data_errors,
            max_failure_rate=max_market_data_error_rate,
            window_seconds=market_data_error_window_seconds,
        )
        # Fat-finger ceiling (module docstring point 9). Defaults to a real, always-on multiple of
        # the inventory cap rather than requiring the caller to remember to set it -- 2x is generous
        # for a single legitimate order (a routine quote never exceeds quote_size_units; the largest
        # legitimate single order is the hedge's excess, itself bounded by how far inventory can
        # realistically drift past the cap in one tick), while still catching an order that is wrong
        # by an order of magnitude, the actual "0.1 became 100" failure mode.
        self._max_single_order_units = (
            max_single_order_units if max_single_order_units is not None else max_inventory_units * 2
        )
        # None (the default) leaves this real capital-preservation guard OFF -- deliberately not a
        # silently-applied hardcoded dollar figure baked into a shared class default (what counts as
        # a sane daily loss ceiling depends entirely on real deployed capital, which this class has
        # no way to know). scripts/run_alpha_engine.py's own --max-daily-loss-usd flag, and on_start's
        # own startup warning below, exist specifically so this is never silently left disabled.
        self._max_daily_loss_usd = max_daily_loss_usd
        self._order_rate_limit_per_second = order_rate_limit_per_second

        self._broker: BaseBroker | None = None
        self._notifier: TelegramNotifier | None = None
        # Real, only registered once connected in on_start (not under MOCK_MODE, which never places
        # a real order and so never needs pacing) -- see module docstring point 13.
        self._rate_limit_mesh: RateLimitMesh | None = None
        # Real, measured on every _read_total_inventory() call (a real REST round-trip) -- None
        # until the first one completes (MOCK_MODE never sets it, since it's never called).
        self._last_api_latency_ms: float | None = None
        self._order_lock = asyncio.Lock()
        self._resting_order_ids: dict[Literal["buy", "sell"], str | None] = {"buy": None, "sell": None}

        # Real inventory bookkeeping -- see module docstring points 6/7.
        self._baseline_qty = Decimal(0)
        self._last_known_total_qty = Decimal(0)
        self._own_avg_entry_price: Decimal | None = None
        self._session_realized_pnl_usd = Decimal(0)
        # Real, current UTC calendar day this ceiling is being tracked for -- reset in _record_fill
        # the instant the real wall-clock date rolls over, independent of how long the process has
        # been running (session PnL above never resets; this deliberately does, to match
        # MAX_DAILY_LOSS_USD's own literal meaning, and src/execution/eod_reconciliator.py's existing
        # UTC-midnight day-boundary convention elsewhere in this project).
        self._daily_realized_pnl_usd = Decimal(0)
        self._daily_pnl_date = datetime.now(timezone.utc).date()
        self._trade_timestamps: deque[float] = deque(maxlen=_TRADE_TIMESTAMPS_MAXLEN)
        self._order_placement_timestamps: deque[float] = deque(maxlen=_ORDER_TIMESTAMPS_MAXLEN)

        self._stale_data_notified = False
        self._inventory_breach_notified = False
        self._fat_finger_notified = False
        self._emergency_listener_task: asyncio.Task[None] | None = None
        self._watchdog: SystemWatchdog | None = None

    # -- Lifecycle ---------------------------------------------------------

    async def on_start(self) -> None:
        self._notifier = TelegramNotifier(self._settings.telegram)

        if self._redis_client is None:
            # This engine's entire safety model (the Master Toggle, the
            # market-data staleness check, telemetry) is Redis-backed --
            # unlike a signal-only agent, running without it is not a
            # degraded-but-safe mode, it is unsupervised. Fail closed:
            # raising here is caught by BaseAgent.run()'s own try/except
            # and routed straight to a clean _shutdown(), never a running
            # engine no operator can see or turn off.
            raise RuntimeError(
                f"{self.agent_id}: Redis is required and unreachable -- refusing to start an autonomous "
                "trading engine with no working Master Toggle, staleness check, or telemetry."
            )

        if not self._settings.mock_mode:
            self._broker = await BrokerFactory.create(
                self._exchange_id, settings=self._settings, redis_client=self._redis_client,
            )
            await self._broker.connect()
            self._rate_limit_mesh = RateLimitMesh()
            self._rate_limit_mesh.register_broker(
                self._exchange_id, requests_per_second=self._order_rate_limit_per_second,
            )
            await self._bootstrap_cold_start_sanitizer()
        else:
            self._log.info("alpha_engine_mock_mode_broker_not_connected", agent_id=self.agent_id)

        self._emergency_listener_task = asyncio.create_task(self._listen_for_emergency_liquidate())

        if self._max_daily_loss_usd is None and not self._settings.mock_mode:
            # Real, loud, impossible-to-miss: a real capital-preservation guard left off is worth
            # more than a debug-level log line -- see max_daily_loss_usd's own docstring in __init__.
            self._log.critical(
                "alpha_engine_no_daily_loss_limit_configured",
                agent_id=self.agent_id,
                detail="MAX_DAILY_LOSS_USD not set -- this engine will NOT self-halt on any drawdown. "
                "Set --max-daily-loss-usd (scripts/run_alpha_engine.py) before running with real capital.",
            )

        self._log.info(
            "alpha_engine_started",
            agent_id=self.agent_id,
            symbol=self._symbol,
            exchange_id=self._exchange_id,
            max_inventory_units=str(self._max_inventory_units),
            max_single_order_units=str(self._max_single_order_units),
            max_daily_loss_usd=str(self._max_daily_loss_usd) if self._max_daily_loss_usd is not None else None,
            baseline_qty=str(self._baseline_qty),
            mock_mode=self._settings.mock_mode,
        )
        await self._notifier.send(
            f"Alpha Engine démarré -- {self._symbol} sur {self._exchange_id}. "
            f"Plafond d'inventaire : {self._max_inventory_units}. "
            f"Perte max/jour : {self._max_daily_loss_usd if self._max_daily_loss_usd is not None else 'NON CONFIGURÉE ⚠️'}. "
            f"Mode : {'SIMULÉ (MOCK_MODE)' if self._settings.mock_mode else 'RÉEL'}.",
            severity=AlertSeverity.INFO, source=self.agent_id,
        )

        # ---- System-level crash prevention (core.exception_manager / core.watchdog). Wired here,
        # inside on_start(), rather than left to whatever launcher script constructs this agent -- every
        # real run of this engine gets this protection by construction, not only if a caller remembers
        # to wire it. self._publisher always exists (set in BaseAgent.__init__, never None); redis is
        # required above; the broker/exchange ping is real only when a real broker is connected.
        install_global_exception_handler(
            asyncio.get_running_loop(), self.cancel_all_orders_for_shutdown,
            sender_id=self.agent_id, publisher=self._publisher, notifier=self._notifier,
        )
        self._watchdog = SystemWatchdog(
            redis_client=self._redis_client, shutdown_callback=self.cancel_all_orders_for_shutdown,
            exchange_ping=self.ping_exchange if self._broker is not None else None,
        )
        self._watchdog.start()

    async def _bootstrap_cold_start_sanitizer(self) -> None:
        """Real pre-flight cleanup before this engine ever trusts its own in-memory state. See module docstring point 11.

        Called from :meth:`on_start`, only when a real broker is
        connected (never under ``MOCK_MODE``, which has no real state to
        sanitize). Two real, independent steps:

        1. Cancel every real orphaned order left resting from a possible
           previous crashed session -- this engine's own
           ``_resting_order_ids`` is fresh, empty, in-memory state right
           after a restart; it has no idea a previous process's quotes
           might still be live on the exchange.
        2. Restore (or, on a real first-ever run, establish) the
           persisted inventory baseline, then refuse to start if the
           real resulting owned inventory already exceeds the cap --
           the real "position already too large, do not start trading
           more on top of it" check.
        """
        assert self._broker is not None
        assert self._notifier is not None
        assert self._redis_client is not None

        try:
            cancelled = await asyncio.wait_for(
                self._broker.cancel_all_open_orders(self._symbol), timeout=_COLD_START_TIMEOUT_SECONDS,
            )
            if cancelled:
                self._log.warning(
                    "alpha_engine_orphaned_orders_cancelled_at_startup",
                    agent_id=self.agent_id, symbol=self._symbol, count=len(cancelled),
                )
        except NotImplementedError:
            self._log.warning(
                "alpha_engine_cancel_all_open_orders_not_supported", agent_id=self.agent_id,
                detail="This broker adapter has no real orphaned-order cleanup implemented -- any "
                "resting order left over from a previous crashed session was NOT cancelled.",
            )
        except TimeoutError:
            raise RuntimeError(
                f"{self.agent_id}: could not confirm a clean order-book state within "
                f"{_COLD_START_TIMEOUT_SECONDS}s (cancel_all_open_orders timed out) -- refusing to "
                "start with an unconfirmed set of possibly-orphaned resting orders."
            ) from None

        persisted_baseline_raw = await self._redis_client.execute(
            "get", f"{ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX}{self.agent_id}",
        )
        if isinstance(persisted_baseline_raw, bytes):
            persisted_baseline_raw = persisted_baseline_raw.decode()
        current_total = await self._read_total_inventory()

        if persisted_baseline_raw is not None:
            self._baseline_qty = Decimal(persisted_baseline_raw)
            self._log.info(
                "alpha_engine_baseline_restored_from_previous_session",
                agent_id=self.agent_id, baseline_qty=str(self._baseline_qty),
            )
        else:
            self._baseline_qty = current_total
            await self._redis_client.execute(
                "set", f"{ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX}{self.agent_id}", str(self._baseline_qty),
            )
            self._log.info(
                "alpha_engine_baseline_established_first_run",
                agent_id=self.agent_id, baseline_qty=str(self._baseline_qty),
            )
        self._last_known_total_qty = current_total

        own_inventory_at_startup = current_total - self._baseline_qty
        if abs(own_inventory_at_startup) > self._max_inventory_units:
            await self._notifier.send(
                f"🆘 DÉMARRAGE REFUSÉ pour {self._symbol} : inventaire déjà détenu par ce moteur "
                f"({own_inventory_at_startup}) dépasse le plafond {self._max_inventory_units} avant "
                "même le premier cycle. Investigation manuelle requise.",
                severity=AlertSeverity.CRITICAL, source=self.agent_id,
            )
            raise RuntimeError(
                f"{self.agent_id}: real starting inventory {own_inventory_at_startup} already "
                f"exceeds the cap {self._max_inventory_units} -- refusing to start."
            )

    async def on_shutdown(self) -> None:
        if self._watchdog is not None:
            await self._watchdog.stop()

        if self._emergency_listener_task is not None:
            self._emergency_listener_task.cancel()
            try:
                await self._emergency_listener_task
            except asyncio.CancelledError:
                pass

        await self._cancel_all_resting_orders()

        if self._broker is not None:
            await self._broker.close()

        if self._notifier is not None:
            # Deliberately does NOT force an Emergency Liquidate on a normal shutdown/OFF -- see
            # ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX's own docstring in core/schemas.py on why
            # those are two structurally distinct actions. A graceful stop means "place no more orders",
            # not "force-sell whatever is currently held at whatever the market happens to be doing".
            await self._notifier.send(
                f"Alpha Engine arrêté -- {self._symbol}. Cotations annulées ; aucune position n'est "
                "liquidée automatiquement à l'arrêt (voir docstring du module -- utilisez Emergency "
                "Liquidate pour ça, explicitement).",
                severity=AlertSeverity.WARNING, source=self.agent_id,
            )

    # -- Main cycle ----------------------------------------------------------

    async def on_tick(self) -> None:
        async with self._order_lock:
            await self._run_cycle()

    async def _run_cycle(self) -> None:
        # Real bug, found live via a dry run: on_start's fail-closed check only guarantees
        # self._redis_client at startup -- BaseAgent._write_heartbeat (core/base_agent.py) nulls it
        # out later, permanently, the instant one heartbeat write exhausts its bounded retries. A
        # bare `assert self._redis_client is not None` here used to treat that as impossible and
        # crash every single cycle forever once it happened (never recovering even after Redis came
        # back, since nothing ever tried it again). Real fix: re-fetch the same process-wide
        # RedisClient singleton (cheap -- returns the existing instance, no new connection storm) and
        # probe it with one real command; on success, resume normally; on failure, fail safe exactly
        # like the Master Toggle OFF branch below (cancel resting orders, skip this cycle) and retry
        # again next cycle -- self-healing instead of a one-way trip into a permanent error loop.
        if self._redis_client is None:
            try:
                # Bounded max_attempts/delays here purely as a defensive backstop for the
                # never-actually-constructed-yet edge case (get_instance ignores these and returns
                # the existing singleton in the realistic case -- see this method's own docstring):
                # without them, a singleton-construction path with no other agent having connected
                # yet would default to *unbounded* retries and could hang this one cycle forever
                # against a genuinely dead Redis, instead of failing safe below.
                candidate = await RedisClient.get_instance(
                    self._settings.redis, base_delay_seconds=0.5, max_delay_seconds=2.0, max_attempts=3,
                )
                await candidate.execute("ping")
            except RedisUnavailableError:
                pass
            else:
                self._redis_client = candidate
                self._log.info("alpha_engine_redis_reconnected", agent_id=self.agent_id)

        if self._redis_client is None:
            await self._cancel_all_resting_orders()
            self._log.warning("alpha_engine_redis_unavailable_cycle_skipped", agent_id=self.agent_id)
            return

        # ---- The real Master Toggle. See module docstring point 3.
        if self.control_state == AgentControlState.OFF:
            await self._cancel_all_resting_orders()
            await self._publish_telemetry(status="OFF", market_data_age_seconds=None, inventory=None)
            return

        simulate = self._settings.mock_mode or self.control_state == AgentControlState.SHADOW

        # ---- Fresh market data, or do nothing. See module docstring point 2.
        metrics, age_seconds = await self._read_fresh_market_metrics()
        if metrics is None:
            await self._cancel_all_resting_orders()
            # Logged every cycle it's true (unlike the once-per-outage Telegram alert below): the
            # only real record of every real stale-data cycle, for a post-run log analysis (see
            # scripts/analyze_shadow_run.py) to count -- the Redis telemetry this also feeds is
            # ephemeral (short TTL) and gone by the time anyone reads it after the fact.
            self._log.warning(
                "alpha_engine_stale_data_detected", agent_id=self.agent_id, symbol=self._symbol,
                market_data_age_seconds=age_seconds, max_market_data_age_seconds=self._max_market_data_age_seconds,
            )
            if not self._stale_data_notified:
                self._stale_data_notified = True
                assert self._notifier is not None
                await self._notifier.send(
                    f"Coupure de flux de marché détectée pour {self._symbol} -- dernière donnée "
                    f"{f'âgée de {age_seconds:.1f}s' if age_seconds is not None else 'absente'} "
                    f"(seuil {self._max_market_data_age_seconds}s). Cotations annulées, moteur en attente.",
                    severity=AlertSeverity.WARNING, source=self.agent_id,
                )
            if self._market_data_circuit_breaker.tripped:
                await self._handle_market_data_circuit_breaker_trip()
                return
            await self._publish_telemetry(status="STALE_DATA", market_data_age_seconds=age_seconds, inventory=None)
            return
        self._stale_data_notified = False

        # ---- The Friday Unwind (module docstring point 14 / Mission 2). Always False for this
        # engine's real BTC/USDT instrument (crypto has no trading_hours) -- real, inert
        # infrastructure today, real protection the day this class trades EUR_USD/XAU_USD/BCO_USD.
        if MarketSessionManager.is_unwind_period(self._symbol):
            await self._cancel_all_resting_orders()
            self._log.warning(
                "alpha_engine_weekend_unwind_active", agent_id=self.agent_id, symbol=self._symbol,
            )
            if not simulate:
                own_inventory = await self._sync_real_inventory(metrics)
                if own_inventory != 0:
                    await self._hedge_excess_inventory(own_inventory, cap=Decimal(0))
            await self._publish_telemetry(status="WEEKEND_UNWIND", market_data_age_seconds=age_seconds, inventory=None)
            return

        # ---- Anti-spread-manipulation filter (module docstring point 14 / Mission 3). Rejects the
        # cycle silently -- no order placed -- before any quote price is ever computed.
        current_spread = Decimal(str(metrics.spread))
        if not is_spread_acceptable(self._symbol, current_spread):
            await self._cancel_all_resting_orders()
            self._log.warning(
                "alpha_engine_spread_rejected", agent_id=self.agent_id, symbol=self._symbol,
                spread=str(current_spread), max_allowable_spread=str(self._universe_entry.max_allowable_spread),
            )
            await self._publish_telemetry(status="SPREAD_REJECTED", market_data_age_seconds=age_seconds, inventory=None)
            return

        if simulate:
            await self._run_simulated_cycle(metrics)
            return

        # ---- Real inventory, read from the broker -- source of truth. See module docstring point 6.
        own_inventory = await self._sync_real_inventory(metrics)

        if abs(own_inventory) > self._max_inventory_units:
            await self._cancel_all_resting_orders()
            await self._hedge_excess_inventory(own_inventory)
            await self._publish_telemetry(status="HEDGING", market_data_age_seconds=age_seconds, inventory=own_inventory)
            return
        self._inventory_breach_notified = False

        await self._requote(metrics, own_inventory)
        await self._publish_telemetry(status="LIVE", market_data_age_seconds=age_seconds, inventory=own_inventory)

    async def _run_simulated_cycle(self, metrics: MarketMetrics) -> None:
        """``MOCK_MODE`` / ``AgentControlState.SHADOW``: log intended quotes, touch no real order or inventory.

        Deliberately does not fabricate a simulated fill or PnL curve --
        this project's own established discipline (see
        ``src/dashboard/ems_pages/agent_fleet.py``'s module docstring on
        why an earlier draft's invented Tier-2 progress numbers were
        rejected): a number that did not come from a real fill is never
        displayed as if it did. The dashboard labels this state plainly
        as "SIMULÉ".
        """
        bid_price, ask_price = _skewed_quote_prices(
            mid_price=Decimal(str(metrics.mid_price)), half_spread_bps=self._half_spread_bps,
            inventory=Decimal(0), max_inventory=self._max_inventory_units, skew_bps=self._inventory_skew_bps,
        )
        self._log.info(
            "alpha_engine_simulated_quote", agent_id=self.agent_id, symbol=self._symbol,
            bid=str(bid_price), ask=str(ask_price), mock_mode=self._settings.mock_mode,
            control_state=self.control_state.value,
        )
        age_seconds = (datetime.now(timezone.utc) - metrics.timestamp).total_seconds()
        await self._publish_telemetry(status="SIMULATED", market_data_age_seconds=age_seconds, inventory=None)

    # -- Market data -----------------------------------------------------

    async def _read_fresh_market_metrics(self) -> tuple[MarketMetrics | None, float | None]:
        """Real ``MarketMetrics`` from the real producer's cache key, or ``(None, age)`` if missing/stale/unparseable."""
        assert self._redis_client is not None
        try:
            raw = await self._redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{self._symbol}")
        except RedisUnavailableError:
            return None, None
        if raw is None:
            return None, None
        try:
            metrics = MarketMetrics.model_validate_json(raw)
        except Exception as exc:
            # 2026-09-16 fix: this used to log only str(raw)[:120] and swallow the real exception --
            # found live analyzing a real 51h shadow-run window with 34k+ of these warnings, where
            # that 120-char preview always cut off before mid_price/top_of_book_liquidity even
            # appeared, and the actual validation reason was never recorded at all. Undiagnosable
            # after the fact, on the one event that exists specifically to diagnose this. error/
            # error_type is the real reason (a Pydantic ValidationError says exactly which field and
            # why); raw_value is capped, not unbounded, so a corrupted/adversarially oversized Redis
            # value (see core/message_integrity.py's own docstring on who can write this key) can't
            # turn one warning into a multi-megabyte log line.
            raw_str = str(raw)
            self._log.warning(
                "alpha_engine_market_metrics_unparseable",
                agent_id=self.agent_id,
                error_type=type(exc).__name__,
                error=str(exc),
                raw_value=raw_str if len(raw_str) <= 2000 else f"{raw_str[:2000]}...<truncated, {len(raw_str)} chars total>",
            )
            self._market_data_circuit_breaker.record_failure()
            return None, None

        age_seconds = (datetime.now(timezone.utc) - metrics.timestamp).total_seconds()
        if age_seconds > self._max_market_data_age_seconds:
            self._market_data_circuit_breaker.record_failure()
            return None, age_seconds
        self._market_data_circuit_breaker.record_success()
        return metrics, age_seconds

    # -- Inventory / fills -------------------------------------------------

    async def _sync_real_inventory(self, metrics: MarketMetrics) -> Decimal:
        """Real inventory read + fill detection -- shared by the normal path and the Friday Unwind path (module docstring point 14).

        Extracted so both real-money code paths detect and price a fill
        exactly once, the same way, rather than the Friday Unwind path
        either duplicating this logic or skipping fill detection
        entirely on a cycle that happens to also be a real fill.
        """
        total_qty = await self._read_total_inventory()
        fill_qty = total_qty - self._last_known_total_qty
        if fill_qty != 0:
            await self._record_fill(fill_qty, fill_price=Decimal(str(metrics.mid_price)))
        self._last_known_total_qty = total_qty
        return total_qty - self._baseline_qty

    async def _read_total_inventory(self) -> Decimal:
        """Real, current TOTAL account balance of this symbol's base asset. See module docstring point 6.

        Also times this real REST round-trip and stores it in
        :attr:`_last_api_latency_ms` -- called every real cycle, so this
        is a fresh, real per-cycle exchange latency reading (not a
        fabricated placeholder), surfaced on the dashboard's telemetry
        panel.
        """
        assert self._broker is not None
        started_at = time.monotonic()
        positions = await self._broker.get_open_positions()
        self._last_api_latency_ms = (time.monotonic() - started_at) * 1000.0
        for position in positions:
            if position.asset.symbol == self._symbol:
                return position.quantity
        return Decimal(0)

    async def _record_fill(self, fill_qty: Decimal, *, fill_price: Decimal) -> None:
        """Update this engine's own cost-basis/PnL bookkeeping for a detected fill. See module docstring point 7."""
        previous_own_qty = self._last_known_total_qty - self._baseline_qty
        new_qty, new_avg_entry_price, realized_delta = _apply_fill(
            previous_qty=previous_own_qty, previous_avg_entry_price=self._own_avg_entry_price,
            fill_qty=fill_qty, fill_price=fill_price,
        )
        self._own_avg_entry_price = new_avg_entry_price
        self._session_realized_pnl_usd += realized_delta
        self._roll_over_daily_pnl_if_new_day()
        self._daily_realized_pnl_usd += realized_delta
        self._trade_timestamps.append(time.time())
        self._prune_old_trade_timestamps()

        self._log.info(
            "alpha_engine_fill_detected", agent_id=self.agent_id, fill_qty=str(fill_qty), fill_price=str(fill_price),
            new_own_qty=str(new_qty), realized_pnl_delta_usd=str(realized_delta),
            session_realized_pnl_usd=str(self._session_realized_pnl_usd),
            daily_realized_pnl_usd=str(self._daily_realized_pnl_usd),
        )
        await self._persist_pnl_history_entry(
            fill_qty=fill_qty, fill_price=fill_price, realized_pnl_delta_usd=realized_delta,
        )
        await self._publish_audit_event(
            event_type="EXECUTED", order_id=f"alpha-fill-{uuid.uuid4()}",
            detail=(
                f"Alpha Engine fill detected: {fill_qty} {self._symbol} @ ~{fill_price} "
                "(mid-price approximation -- see _record_fill docstring)"
            ),
        )

        if self._max_daily_loss_usd is not None and self._daily_realized_pnl_usd <= self._max_daily_loss_usd:
            await self._force_off(
                "alpha_engine_max_daily_loss_hit",
                notify_message=(
                    f"🆘 PERTE MAX/JOUR ATTEINTE pour {self._symbol} : PnL réalisé du jour "
                    f"{self._daily_realized_pnl_usd} $ <= seuil {self._max_daily_loss_usd} $. ARRÊT SYSTÈME "
                    "IMMÉDIAT (moteur forcé sur OFF -- positions existantes NON liquidées automatiquement, "
                    "voir Emergency Liquidate)."
                ),
                extra_log_fields={
                    "daily_realized_pnl_usd": str(self._daily_realized_pnl_usd),
                    "max_daily_loss_usd": str(self._max_daily_loss_usd),
                },
            )

    def _roll_over_daily_pnl_if_new_day(self) -> None:
        """Reset the daily PnL ceiling's own counter the instant the real UTC calendar date rolls over."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            self._log.info(
                "alpha_engine_daily_pnl_rolled_over", agent_id=self.agent_id,
                previous_date=str(self._daily_pnl_date), previous_daily_pnl_usd=str(self._daily_realized_pnl_usd),
            )
            self._daily_pnl_date = today
            self._daily_realized_pnl_usd = Decimal(0)

    def _prune_old_trade_timestamps(self) -> None:
        cutoff = time.time() - _ONE_DAY_SECONDS
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()

    # -- Circuit breakers (module docstring points 8/9) ---------------------

    def _check_order_rate_limit(self) -> bool:
        """Real order-placement-rate circuit breaker. Records this attempt; returns ``False`` if the rolling window is exceeded.

        Called from every *routine* real order path (a quote, the
        automatic hedge) -- never from Emergency Liquidate, which is the
        deliberate human-triggered break-glass exception to every
        internal safety net this engine has, the same reasoning as the
        Kill Switch bypass (see module docstring).
        """
        now = time.time()
        self._order_placement_timestamps.append(now)
        cutoff = now - self._max_order_rate_window_seconds
        while self._order_placement_timestamps and self._order_placement_timestamps[0] < cutoff:
            self._order_placement_timestamps.popleft()
        return len(self._order_placement_timestamps) <= self._max_order_rate_count

    async def _handle_order_rate_limit_breach(self) -> None:
        await self._force_off(
            "alpha_engine_order_rate_limit_breached",
            notify_message=(
                f"🆘 EMBALLEMENT DÉTECTÉ pour {self._symbol} : plus de {self._max_order_rate_count} ordres "
                f"réels en {self._max_order_rate_window_seconds}s. ARRÊT SYSTÈME IMMÉDIAT -- intervention "
                "manuelle requise avant réactivation."
            ),
            extra_log_fields={
                "max_order_rate_count": self._max_order_rate_count,
                "window_seconds": self._max_order_rate_window_seconds,
            },
        )

    async def _handle_market_data_circuit_breaker_trip(self) -> None:
        """Real "Protocole AEGIS" trip handler: force OFF rather than keep absorbing corrupted/stale data.

        Real gap closed (2026-09-16 post-mortem): a real 51.5h shadow-run
        window produced 34,637 ``alpha_engine_market_metrics_unparseable``
        warnings and 36,211 ``alpha_engine_stale_data_detected`` warnings
        -- correctly logged, correctly fail-closed (see
        :meth:`_read_fresh_market_metrics`, this engine never traded on
        that data), but every one of those cycles was still individually
        retried at the normal tick cadence forever, with nothing ever
        escalating the *pattern* into a decision. This agent owns no
        network connection of its own to reconnect (it reads a Redis
        cache another process, :class:`~src.agents.market_data.agent.
        MarketDataAgent`, writes -- see that module for the real
        WebSocket-level circuit breaker/reconnect) -- forcing this
        engine's own real, existing :class:`~core.base_agent.
        AgentControlState.OFF` state is the honest equivalent action here,
        same manual-reactivation-required posture as
        :meth:`_handle_order_rate_limit_breach`/
        :meth:`_handle_fat_finger_blocked` above.
        """
        await self._force_off(
            "alpha_engine_market_data_circuit_breaker_tripped",
            notify_message=(
                f"🆘 DISJONCTEUR DE DONNÉES DÉCLENCHÉ pour {self._symbol} : flux de marché corrompu ou "
                f"périmé de façon soutenue ({self._market_data_circuit_breaker.consecutive_failures} échecs "
                "consécutifs / taux d'échec au-delà du seuil). Moteur arrêté avant absorption passive du "
                "bruit -- intervention manuelle requise avant réactivation (vérifier market-data-agent en amont)."
            ),
            extra_log_fields={"consecutive_failures": self._market_data_circuit_breaker.consecutive_failures},
        )

    def _check_fat_finger(self, size: Decimal) -> bool:
        """Real, always-on sanity ceiling on a single order's size, independent of whatever sizing logic computed it. See module docstring point 9."""
        return size <= self._max_single_order_units

    async def _handle_fat_finger_blocked(self, size: Decimal) -> None:
        self._log.critical(
            "alpha_engine_fat_finger_blocked", agent_id=self.agent_id,
            requested_size=str(size), max_single_order_units=str(self._max_single_order_units),
        )
        if not self._fat_finger_notified:
            self._fat_finger_notified = True
            assert self._notifier is not None
            await self._notifier.send(
                f"🆘 FAT-FINGER BLOQUÉ pour {self._symbol} : taille demandée {size} dépasse le plafond "
                f"absolu {self._max_single_order_units}. Ordre refusé, moteur forcé sur OFF -- ceci indique "
                "probablement un bug réel de calcul de taille, pas juste une anomalie de marché.",
                severity=AlertSeverity.CRITICAL, source=self.agent_id,
            )
        await self._force_off(
            "alpha_engine_fat_finger_forced_off",
            notify_message=None,  # already sent above -- avoid a duplicate alert
            extra_log_fields={"requested_size": str(size)},
        )

    async def _force_off(
        self, log_event: str, *, notify_message: str | None, extra_log_fields: dict[str, object] | None = None,
    ) -> None:
        """Shared real self-halt: persist + publish AgentControlState.OFF, log CRITICAL, alert.

        Factors out the exact same "force OFF" sequence
        :meth:`_execute_emergency_liquidate` already needed, now shared
        by every circuit breaker that can trip this engine off by
        itself (order-rate breach, fat-finger block, max daily loss).
        Never touches open positions -- see
        ``ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX``'s own
        docstring in core/schemas.py on why OFF and "flatten now" are
        two structurally distinct actions.
        """
        self._log.critical(log_event, agent_id=self.agent_id, **(extra_log_fields or {}))
        if self._redis_client is not None:
            await self._redis_client.execute(
                "set", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{self.agent_id}", AgentControlState.OFF.value,
            )
            await self._redis_client.execute(
                "publish", f"{AGENT_CONTROL_CHANNEL_PREFIX}{self.agent_id}", AgentControlState.OFF.value,
            )
        self._control_state = AgentControlState.OFF
        if notify_message is not None and self._notifier is not None:
            await self._notifier.send(notify_message, severity=AlertSeverity.CRITICAL, source=self.agent_id)

    # -- Quoting -----------------------------------------------------------

    async def _requote(self, metrics: MarketMetrics, own_inventory: Decimal) -> None:
        await self._cancel_all_resting_orders()

        bid_price, ask_price = _skewed_quote_prices(
            mid_price=Decimal(str(metrics.mid_price)), half_spread_bps=self._half_spread_bps,
            inventory=own_inventory, max_inventory=self._max_inventory_units, skew_bps=self._inventory_skew_bps,
        )
        buy_size, sell_size = _clamp_quote_sizes(
            base_size=self._quote_size_units, inventory=own_inventory, max_inventory=self._max_inventory_units,
        )

        if buy_size > 0:
            await self._place_resting_order("buy", buy_size, bid_price)
        if sell_size > 0:
            await self._place_resting_order("sell", sell_size, ask_price)

    async def _place_resting_order(self, side: Literal["buy", "sell"], size: Decimal, price: Decimal) -> None:
        assert self._broker is not None
        assert self._rate_limit_mesh is not None
        broker = self._broker
        if not self._check_fat_finger(size):
            await self._handle_fat_finger_blocked(size)
            return
        if not self._check_order_rate_limit():
            await self._handle_order_rate_limit_breach()
            return
        client_order_id = f"alpha-{side}-{uuid.uuid4()}"
        try:
            # RateLimitMesh paces this call (and adapts on a real ccxt RateLimitExceeded/
            # DDoSProtection) *before* safe_financial_call's own Kill-Switch/timeout semantics apply
            # -- see module docstring point 13 on why this ordering (mesh as safe_financial_call's
            # own api_function, not the other way around) is what lets the mesh's backoff logic
            # actually see the real exception type instead of an already-wrapped one.
            raw = await safe_financial_call(
                self._rate_limit_mesh.throttled_call, self._exchange_id,
                lambda: broker.place_limit_order(
                    self._symbol, side, float(size), float(price), client_order_id=client_order_id,
                ),
                retry_on_failure=False,
            )
        except KillSwitchActiveError:
            self._log.warning("alpha_engine_quote_blocked_by_kill_switch", agent_id=self.agent_id, side=side)
            return
        except FinancialCallAbortedError as exc:
            self._log.error("alpha_engine_quote_failed", agent_id=self.agent_id, side=side, error=str(exc))
            return

        order_id = str(raw.get("id") or client_order_id)
        self._resting_order_ids[side] = order_id
        await self._publish_audit_event(
            event_type="SUBMITTED", order_id=order_id,
            detail=f"Alpha Engine {side} quote: {size} {self._symbol} @ {price}",
        )

    async def _cancel_all_resting_orders(self) -> None:
        if self._broker is None:
            return
        for side, order_id in list(self._resting_order_ids.items()):
            if order_id is None:
                continue
            try:
                await self._broker.cancel_order(order_id, symbol=self._symbol)
            except Exception as exc:
                # An order that already filled or was already cancelled raises here on most venues --
                # not distinguishable from a real failure without a broker-specific error code
                # BaseBroker's own ABC does not normalize (see broker_gateway.py). Logged, not
                # escalated: the next cycle's fresh get_open_positions() read is the real source of
                # truth regardless of whether this cancel "succeeded".
                self._log.warning(
                    "alpha_engine_cancel_order_failed", agent_id=self.agent_id, side=side,
                    order_id=order_id, error=str(exc),
                )
            self._resting_order_ids[side] = None

    # -- System-level crash prevention (core.exception_manager / core.watchdog) --------------------

    async def cancel_all_orders_for_shutdown(self) -> None:
        """The real ``shutdown_callback`` for both :func:`~core.exception_manager.install_global_exception_handler` and :attr:`_watchdog`.

        Deliberately narrower than :meth:`on_shutdown` (this engine's
        normal, orderly stop): a fatal exception or a resource-exhaustion
        event is exactly the situation where this process's own in-memory
        bookkeeping (:attr:`_resting_order_ids`) might itself be
        untrustworthy, so this calls the broker's real, comprehensive
        :meth:`~src.execution.broker_gateway.BaseBroker.cancel_all_open_orders`
        -- the same real API call :meth:`_bootstrap_cold_start_sanitizer`
        already trusts for exactly this reason -- instead of only
        cancelling what this engine happens to still think is resting.
        Does **not** flatten any open position: that is Emergency
        Liquidate's own, deliberately separate, human-triggered action
        (see module docstring's "Kill-switch interaction" section) --
        conflating the two here would silently turn every crash into a
        forced market sell, which is not what either mission asked for.
        A no-op under ``MOCK_MODE`` (no real broker ever connected).
        """
        if self._broker is None:
            return
        try:
            await self._broker.cancel_all_open_orders(self._symbol)
        except NotImplementedError:
            self._log.warning(
                "alpha_engine_shutdown_cancel_all_not_supported", agent_id=self.agent_id,
                detail="This broker adapter has no real cancel_all_open_orders implemented -- any "
                "resting order was NOT cancelled by this emergency shutdown path.",
            )

    async def ping_exchange(self) -> None:
        """Real, lightweight REST call for :attr:`_watchdog`'s exchange vital sign.

        Raises on failure -- :class:`~core.watchdog.SystemWatchdog` only
        cares whether this completes; it measures its own wall-clock
        latency around the call, this method does not self-report one.
        Never wired in under ``MOCK_MODE`` (see :meth:`on_start`, no real
        broker exists to ping).
        """
        assert self._broker is not None  # on_start only passes this method to SystemWatchdog when real
        await self._broker.get_current_price(self._symbol)

    # -- Inventory-limit hedge ---------------------------------------------

    async def _hedge_excess_inventory(self, own_inventory: Decimal, *, cap: Decimal | None = None) -> None:
        """Flatten ``own_inventory`` back down to ``cap`` (defaults to :attr:`_max_inventory_units`).

        ``cap=Decimal(0)`` is the Friday Unwind's real full-flatten case
        (module docstring point 14) -- a *scheduled*, expected weekly
        event, not a panic, so it deliberately reuses this normal, Kill-
        Switch-gated hedge path rather than the break-glass
        :meth:`_execute_emergency_liquidate` bypass, which exists for a
        genuine emergency only.
        """
        effective_cap = self._max_inventory_units if cap is None else cap
        excess = abs(own_inventory) - effective_cap
        side: Literal["buy", "sell"] = "sell" if own_inventory > 0 else "buy"
        assert self._broker is not None
        assert self._rate_limit_mesh is not None
        broker = self._broker

        # Both circuit breakers apply here too, deliberately -- see module docstring points 8/9. A
        # hedge that keeps re-firing without ever bringing inventory back under the cap (an exchange
        # issue, a bug in the excess computation, or a real fill racing this one) is exactly the
        # "anti-hémorragie" scenario the rate limiter exists to catch, market orders' real fees
        # included; an excess this large (>= max_single_order_units, itself 2x the inventory cap --
        # i.e. real inventory would need to be 3x over cap) is a "something is very wrong, a human
        # should look at this before more capital moves" signal, not a number to blindly market-sell.
        if not self._check_fat_finger(excess):
            await self._handle_fat_finger_blocked(excess)
            return
        if not self._check_order_rate_limit():
            await self._handle_order_rate_limit_breach()
            return

        client_order_id = f"alpha-hedge-{uuid.uuid4()}"

        self._log.critical(
            "alpha_engine_inventory_limit_breached_hedging", agent_id=self.agent_id,
            inventory=str(own_inventory), max_inventory=str(effective_cap),
            excess=str(excess), side=side,
        )
        try:
            raw = await safe_financial_call(
                self._rate_limit_mesh.throttled_call, self._exchange_id,
                lambda: broker.place_market_order(self._symbol, side, float(excess), client_order_id=client_order_id),
                retry_on_failure=False,
            )
        except KillSwitchActiveError:
            self._log.critical(
                "alpha_engine_hedge_blocked_by_kill_switch_manual_intervention_required",
                agent_id=self.agent_id, inventory=str(own_inventory),
            )
            await self._notify_inventory_breach_once(
                f"🆘 INTERVENTION MANUELLE REQUISE -- {self._symbol} : inventaire {own_inventory} dépasse le "
                f"plafond {effective_cap}, et le Kill Switch global bloque la couverture "
                "automatique. Position NON couverte."
            )
            return
        except FinancialCallAbortedError as exc:
            self._log.critical("alpha_engine_hedge_order_failed", agent_id=self.agent_id, error=str(exc))
            await self._notify_inventory_breach_once(
                f"🆘 Échec de la couverture d'urgence pour {self._symbol} (inventaire {own_inventory}). "
                f"Détail : {exc}"
            )
            return

        order_id = str(raw.get("id") or client_order_id)
        await self._publish_audit_event(
            event_type="EXECUTED", order_id=order_id,
            detail=f"Alpha Engine hedge: {side} {excess} {self._symbol} (limite d'inventaire dépassée)",
        )
        await self._notify_inventory_breach_once(
            f"Limite d'inventaire atteinte pour {self._symbol} -- couverture automatique exécutée : "
            f"{side} {excess}. Inventaire ramené sous {effective_cap}.",
            severity=AlertSeverity.WARNING,
        )

    async def _notify_inventory_breach_once(
        self, message: str, *, severity: AlertSeverity = AlertSeverity.CRITICAL
    ) -> None:
        if self._inventory_breach_notified:
            return
        self._inventory_breach_notified = True
        assert self._notifier is not None
        await self._notifier.send(message, severity=severity, source=self.agent_id)

    # -- Emergency Liquidate -------------------------------------------------

    async def _listen_for_emergency_liquidate(self) -> None:
        """Background listener, alongside :meth:`on_tick`, for the dashboard's Emergency Liquidate command.

        Same real "hold a Redis Pub/Sub subscription open for the
        agent's entire lifetime" pattern as
        :meth:`~core.base_agent.BaseAgent._listen_for_control_commands` /
        :class:`~src.agents.execution.agent.ExecutionAgent`'s manual-
        trade listener -- reused, not reinvented, for the same "instant
        effect, no per-tick Redis round trip" reason.
        """
        if self._redis_client is None:
            self._log.warning("alpha_engine_emergency_listener_disabled_no_redis", agent_id=self.agent_id)
            return

        pubsub = self._redis_client.client.pubsub()
        channel = f"{ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX}{self.agent_id}"
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                # **Real vulnerability fixed here** (found by tests/red_team_hacker.py's Attack 5, a
                # real, empirically confirmed exploit against this session's own code): this used to act
                # on ANY message that arrived on this channel, with zero check of origin or content --
                # this codebase's Redis has no auth by default (see core/config.py's own
                # _require_redis_auth_in_production docstring), so any client able to reach it could
                # PUBLISH here and force a real, Kill-Switch-bypassing market order on demand, including
                # during a real, unrelated panic. emergency_flatten_market_order must stay unguarded to
                # do its real job (see core/security.py's module docstring) -- so the guard belongs here,
                # at the one real entry point, not on the break-glass call itself. Only a holder of
                # VAULT_MASTER_KEY (the real dashboard backend -- see src/dashboard/ems_pages/
                # autonomous_command_center.py's Confirmer button) can produce a payload that verifies.
                if not verify(f"LIQUIDATE:{self.agent_id}", message["data"]):
                    self._log.critical(
                        "alpha_engine_emergency_liquidate_command_rejected_unsigned",
                        agent_id=self.agent_id,
                    )
                    continue
                self._log.critical("alpha_engine_emergency_liquidate_command_received", agent_id=self.agent_id)
                # Serialized against _run_cycle via the same lock -- an emergency liquidate and a
                # routine requote can never interleave and race each other's order placement.
                async with self._order_lock:
                    await self._execute_emergency_liquidate()
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[attr-defined]

    async def _execute_emergency_liquidate(self) -> None:
        """The real break-glass sequence: bypass the Kill Switch, flatten at market, force OFF, alert.

        1. Cancel every resting quote (never mind the Kill Switch --
           :meth:`~src.execution.broker_gateway.BaseBroker.cancel_order`
           was never gated to begin with).
        2. Flatten the real position via
           :meth:`~src.execution.broker_gateway.BaseBroker.
           emergency_flatten_market_order` -- **not** ``place_market_order``,
           so this is not refused even if the global Kill Switch is
           already tripped. See module docstring.
        3. Force this engine's own :class:`~core.base_agent.
           AgentControlState` to ``OFF``, unconditionally, whether step 2
           succeeded or not -- "place no more orders" is correct either way.
        4. Alert. Deliberately does **not** additionally tear down the
           broker connection/socket (a literal reading of "coupe les
           connexions"): step 3 already makes every future cycle a no-op
           at the top of :meth:`_run_cycle`, enforced through the same
           ``_order_lock`` this method itself holds -- there is no code
           path left that could place another order. Actually closing
           the connection would require this engine to also carry real
           reconnect logic it does not have today (unlike
           :meth:`~src.agents.market_data.exchange_client.
           AsyncExchangeClient`'s own WebSocket reconnect), trading a
           real robustness cost for a safety property the control-state
           check already provides for free.
        """
        await self._cancel_all_resting_orders()
        assert self._notifier is not None

        if self._settings.mock_mode or self._broker is None:
            self._log.warning("alpha_engine_emergency_liquidate_skipped_mock_mode", agent_id=self.agent_id)
        else:
            total_qty = await self._read_total_inventory()
            own_inventory = total_qty - self._baseline_qty
            if own_inventory != 0:
                side: Literal["buy", "sell"] = "sell" if own_inventory > 0 else "buy"
                client_order_id = f"alpha-emergency-liquidate-{uuid.uuid4()}"
                try:
                    raw = await self._broker.emergency_flatten_market_order(
                        self._symbol, side, float(abs(own_inventory)), client_order_id=client_order_id,
                    )
                    order_id = str(raw.get("id") or client_order_id)
                    await self._publish_audit_event(
                        event_type="EXECUTED", order_id=order_id,
                        detail=f"EMERGENCY LIQUIDATE (Kill Switch bypass): {side} {abs(own_inventory)} {self._symbol}",
                    )
                except KillSwitchActiveError:
                    # Only reachable if self._broker's concrete adapter does NOT override
                    # emergency_flatten_market_order (see BaseBroker's own docstring on that default) --
                    # CcxtBrokerAdapter, what this engine actually uses, always does. Kept as a real
                    # safety net, not dead code: a future broker adapter that forgets to override it
                    # fails safe (gated) rather than silently behaving as if it had bypassed anything.
                    self._log.critical(
                        "alpha_engine_emergency_liquidate_blocked_by_kill_switch",
                        agent_id=self.agent_id, inventory=str(own_inventory),
                    )
                    await self._notifier.send(
                        f"🆘 ÉCHEC Emergency Liquidate pour {self._symbol} -- le Kill Switch global bloque "
                        f"l'ordre de fermeture. Position NON fermée ({own_inventory}). INTERVENTION MANUELLE "
                        "IMMÉDIATE REQUISE.",
                        severity=AlertSeverity.CRITICAL, source=self.agent_id,
                    )
                except Exception as exc:
                    self._log.exception("alpha_engine_emergency_liquidate_order_failed", agent_id=self.agent_id)
                    await self._notifier.send(
                        f"🆘 ÉCHEC Emergency Liquidate pour {self._symbol} : {exc}. Position NON confirmée "
                        "fermée. INTERVENTION MANUELLE IMMÉDIATE REQUISE.",
                        severity=AlertSeverity.CRITICAL, source=self.agent_id,
                    )

        await self._force_off(
            "alpha_engine_emergency_liquidate_completed",
            notify_message=f"Emergency Liquidate exécuté pour {self._symbol}. Moteur forcé sur OFF.",
        )

    # -- Audit trail / telemetry ---------------------------------------------

    async def _publish_audit_event(
        self, *, event_type: Literal["SUBMITTED", "EXECUTED", "REJECTED", "CANCELLED"], order_id: str, detail: str
    ) -> None:
        """Same real :data:`~core.schemas.ORDER_AUDIT_EVENTS_TOPIC` every other order-lifecycle emitter uses.

        **Real, disclosed infra dependency**: this only reaches
        :class:`~src.execution.audit_ledger.CryptographicAuditLedger` if
        this engine's ZMQ publisher can reach the same broker every other
        agent shares (``ZMQ_BROKER_FRONTEND_ADDRESS``/``_BACKEND_ADDRESS``)
        -- see ``scripts/run_market_data_agent.py``'s own docstring for
        the identical "PUB is fire-and-forget, this is not this script's
        problem to fix alone" disclosure. Deployed as its own container
        (see ``docker-compose.yml``), that broker is not reachable today
        (it binds loopback-only inside ``nothierly-agents``' own
        container) without also exposing it on the shared Docker network
        -- a real, one-line infra change, not done here without being asked.
        """
        event = OrderAuditEvent(event_type=event_type, order_id=order_id, asset=self._symbol, detail=detail)
        await self.publish(ORDER_AUDIT_EVENTS_TOPIC, event.model_dump(mode="json"))

    async def _persist_pnl_history_entry(
        self, *, fill_qty: Decimal, fill_price: Decimal, realized_pnl_delta_usd: Decimal
    ) -> None:
        """Append one real fill to this engine's own P&L history -- see ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX's own docstring.

        Same bounded LPUSH + LTRIM pattern as
        :func:`~src.execution.smart_order_router.log_routing_decision`
        (newest first, capped, best-effort). Read by
        :mod:`src.dashboard.ems_pages.bilan` to build a real equity curve
        -- :meth:`_publish_telemetry`'s own key is a short-TTL snapshot of
        running totals only, not history a chart can be drawn from.
        """
        if self._redis_client is None:
            return
        entry = {
            "timestamp": time.time(),
            "symbol": self._symbol,
            "fill_qty": str(fill_qty),
            "fill_price": str(fill_price),
            "realized_pnl_delta_usd": str(realized_pnl_delta_usd),
            "session_realized_pnl_usd": str(self._session_realized_pnl_usd),
        }
        key = f"{ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX}{self.agent_id}"
        try:
            await self._redis_client.execute("lpush", key, json.dumps(entry))
            await self._redis_client.execute("ltrim", key, 0, ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES - 1)
        except RedisUnavailableError:
            self._log.warning("alpha_engine_pnl_history_persist_failed", agent_id=self.agent_id)

    async def _publish_telemetry(
        self, *, status: str, market_data_age_seconds: float | None, inventory: Decimal | None
    ) -> None:
        """Real, current state only -- see module docstring's "never a fabricated number" discipline."""
        if self._redis_client is None:
            return
        payload: dict[str, Any] = {
            "agent_id": self.agent_id,
            "symbol": self._symbol,
            "status": status,
            "control_state": self.control_state.value,
            "mock_mode": self._settings.mock_mode,
            "inventory": str(inventory) if inventory is not None else None,
            "max_inventory_units": str(self._max_inventory_units),
            "avg_entry_price": str(self._own_avg_entry_price) if self._own_avg_entry_price is not None else None,
            "session_realized_pnl_usd": str(self._session_realized_pnl_usd),
            "daily_realized_pnl_usd": str(self._daily_realized_pnl_usd),
            "max_daily_loss_usd": str(self._max_daily_loss_usd) if self._max_daily_loss_usd is not None else None,
            "max_single_order_units": str(self._max_single_order_units),
            "trade_count_24h": len(self._trade_timestamps),
            "market_data_age_seconds": market_data_age_seconds,
            "api_latency_ms": self._last_api_latency_ms,
            "updated_at": time.time(),
        }
        try:
            await self._redis_client.execute(
                "set", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{self.agent_id}", json.dumps(payload),
                ex=int(max(3.0 * self._tick_interval_seconds, 10.0)),
            )
        except RedisUnavailableError:
            self._log.warning("alpha_engine_telemetry_publish_failed", agent_id=self.agent_id)

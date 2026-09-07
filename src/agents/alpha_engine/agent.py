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

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings
from core.logger import get_logger
from core.nothierly_ontology import AssetClass, UniversalAsset
from core.notifier import AlertSeverity, TelegramNotifier
from core.redis_client import RedisUnavailableError
from core.schemas import (
    AGENT_CONTROL_CHANNEL_PREFIX,
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX,
    ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX,
    MARKET_METRICS_REDIS_KEY_PREFIX,
    ORDER_AUDIT_EVENTS_TOPIC,
    MarketMetrics,
)
from core.security import FinancialCallAbortedError, KillSwitchActiveError, safe_financial_call
from src.execution.audit_ledger import OrderAuditEvent
from src.execution.broker_gateway import BaseBroker, BrokerFactory

_log = get_logger(__name__)

# See module docstring point 5 on why this is not "millisecond".
_DEFAULT_TICK_INTERVAL_SECONDS: Final[float] = 1.0
_DEFAULT_MAX_INVENTORY_UNITS: Final[Decimal] = Decimal("0.5")
_DEFAULT_QUOTE_SIZE_UNITS: Final[Decimal] = Decimal("0.01")
_DEFAULT_HALF_SPREAD_BPS: Final[Decimal] = Decimal("5")
_DEFAULT_INVENTORY_SKEW_BPS: Final[Decimal] = Decimal("15")
_DEFAULT_MAX_MARKET_DATA_AGE_SECONDS: Final[float] = 3.0
_ONE_DAY_SECONDS: Final[float] = 86_400.0


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

        self._broker: BaseBroker | None = None
        self._notifier: TelegramNotifier | None = None
        self._order_lock = asyncio.Lock()
        self._resting_order_ids: dict[Literal["buy", "sell"], str | None] = {"buy": None, "sell": None}

        # Real inventory bookkeeping -- see module docstring points 6/7.
        self._baseline_qty = Decimal(0)
        self._last_known_total_qty = Decimal(0)
        self._own_avg_entry_price: Decimal | None = None
        self._session_realized_pnl_usd = Decimal(0)
        self._trade_timestamps: deque[float] = deque()

        self._stale_data_notified = False
        self._inventory_breach_notified = False
        self._emergency_listener_task: asyncio.Task[None] | None = None

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
            self._baseline_qty = await self._read_total_inventory()
            self._last_known_total_qty = self._baseline_qty
        else:
            self._log.info("alpha_engine_mock_mode_broker_not_connected", agent_id=self.agent_id)

        self._emergency_listener_task = asyncio.create_task(self._listen_for_emergency_liquidate())

        self._log.info(
            "alpha_engine_started",
            agent_id=self.agent_id,
            symbol=self._symbol,
            exchange_id=self._exchange_id,
            max_inventory_units=str(self._max_inventory_units),
            baseline_qty=str(self._baseline_qty),
            mock_mode=self._settings.mock_mode,
        )
        await self._notifier.send(
            f"Alpha Engine démarré -- {self._symbol} sur {self._exchange_id}. "
            f"Plafond d'inventaire : {self._max_inventory_units}. Mode : "
            f"{'SIMULÉ (MOCK_MODE)' if self._settings.mock_mode else 'RÉEL'}.",
            severity=AlertSeverity.INFO, source=self.agent_id,
        )

    async def on_shutdown(self) -> None:
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
        assert self._redis_client is not None  # guaranteed by on_start's own fail-closed check

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
            if not self._stale_data_notified:
                self._stale_data_notified = True
                assert self._notifier is not None
                await self._notifier.send(
                    f"Coupure de flux de marché détectée pour {self._symbol} -- dernière donnée "
                    f"{f'âgée de {age_seconds:.1f}s' if age_seconds is not None else 'absente'} "
                    f"(seuil {self._max_market_data_age_seconds}s). Cotations annulées, moteur en attente.",
                    severity=AlertSeverity.WARNING, source=self.agent_id,
                )
            await self._publish_telemetry(status="STALE_DATA", market_data_age_seconds=age_seconds, inventory=None)
            return
        self._stale_data_notified = False

        if simulate:
            await self._run_simulated_cycle(metrics)
            return

        # ---- Real inventory, read from the broker -- source of truth. See module docstring point 6.
        total_qty = await self._read_total_inventory()
        fill_qty = total_qty - self._last_known_total_qty
        if fill_qty != 0:
            await self._record_fill(fill_qty, fill_price=Decimal(str(metrics.mid_price)))
        self._last_known_total_qty = total_qty
        own_inventory = total_qty - self._baseline_qty

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
        except Exception:
            self._log.warning(
                "alpha_engine_market_metrics_unparseable", agent_id=self.agent_id, raw_preview=str(raw)[:120]
            )
            return None, None

        age_seconds = (datetime.now(timezone.utc) - metrics.timestamp).total_seconds()
        if age_seconds > self._max_market_data_age_seconds:
            return None, age_seconds
        return metrics, age_seconds

    # -- Inventory / fills -------------------------------------------------

    async def _read_total_inventory(self) -> Decimal:
        """Real, current TOTAL account balance of this symbol's base asset. See module docstring point 6."""
        assert self._broker is not None
        positions = await self._broker.get_open_positions()
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
        self._trade_timestamps.append(time.time())
        self._prune_old_trade_timestamps()

        self._log.info(
            "alpha_engine_fill_detected", agent_id=self.agent_id, fill_qty=str(fill_qty), fill_price=str(fill_price),
            new_own_qty=str(new_qty), realized_pnl_delta_usd=str(realized_delta),
            session_realized_pnl_usd=str(self._session_realized_pnl_usd),
        )
        await self._publish_audit_event(
            event_type="EXECUTED", order_id=f"alpha-fill-{uuid.uuid4()}",
            detail=(
                f"Alpha Engine fill detected: {fill_qty} {self._symbol} @ ~{fill_price} "
                "(mid-price approximation -- see _record_fill docstring)"
            ),
        )

    def _prune_old_trade_timestamps(self) -> None:
        cutoff = time.time() - _ONE_DAY_SECONDS
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff:
            self._trade_timestamps.popleft()

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
        client_order_id = f"alpha-{side}-{uuid.uuid4()}"
        try:
            raw = await safe_financial_call(
                self._broker.place_limit_order, self._symbol, side, float(size), float(price),
                client_order_id=client_order_id, retry_on_failure=False,
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

    # -- Inventory-limit hedge ---------------------------------------------

    async def _hedge_excess_inventory(self, own_inventory: Decimal) -> None:
        excess = abs(own_inventory) - self._max_inventory_units
        side: Literal["buy", "sell"] = "sell" if own_inventory > 0 else "buy"
        assert self._broker is not None
        client_order_id = f"alpha-hedge-{uuid.uuid4()}"

        self._log.critical(
            "alpha_engine_inventory_limit_breached_hedging", agent_id=self.agent_id,
            inventory=str(own_inventory), max_inventory=str(self._max_inventory_units),
            excess=str(excess), side=side,
        )
        try:
            raw = await safe_financial_call(
                self._broker.place_market_order, self._symbol, side, float(excess),
                client_order_id=client_order_id, retry_on_failure=False,
            )
        except KillSwitchActiveError:
            self._log.critical(
                "alpha_engine_hedge_blocked_by_kill_switch_manual_intervention_required",
                agent_id=self.agent_id, inventory=str(own_inventory),
            )
            await self._notify_inventory_breach_once(
                f"🆘 INTERVENTION MANUELLE REQUISE -- {self._symbol} : inventaire {own_inventory} dépasse le "
                f"plafond {self._max_inventory_units}, et le Kill Switch global bloque la couverture "
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
            f"{side} {excess}. Inventaire ramené sous {self._max_inventory_units}.",
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

        assert self._redis_client is not None
        await self._redis_client.execute(
            "set", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{self.agent_id}", AgentControlState.OFF.value,
        )
        await self._redis_client.execute(
            "publish", f"{AGENT_CONTROL_CHANNEL_PREFIX}{self.agent_id}", AgentControlState.OFF.value,
        )
        self._control_state = AgentControlState.OFF

        await self._notifier.send(
            f"Emergency Liquidate exécuté pour {self._symbol}. Moteur forcé sur OFF.",
            severity=AlertSeverity.CRITICAL, source=self.agent_id,
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
            "trade_count_24h": len(self._trade_timestamps),
            "market_data_age_seconds": market_data_age_seconds,
            "updated_at": time.time(),
        }
        try:
            await self._redis_client.execute(
                "set", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{self.agent_id}", json.dumps(payload),
                ex=int(max(3.0 * self._tick_interval_seconds, 10.0)),
            )
        except RedisUnavailableError:
            self._log.warning("alpha_engine_telemetry_publish_failed", agent_id=self.agent_id)

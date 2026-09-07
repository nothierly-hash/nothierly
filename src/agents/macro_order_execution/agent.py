"""Real execution consumer for :data:`~core.schemas.CLIENT_ORDERS_TOPIC`'s LIVE-mode macro orders.

**Closes a real, previously-disclosed gap, stated in three places before
this module existed**: ``api_middleware/gateway.py``'s own module
docstring, :mod:`src.execution.macro_strategist`'s own docstring ("no
Ingestor Agent consumes [CLIENT_ORDERS_TOPIC] today"), and
:mod:`src.execution.eod_reconciliator`'s own docstring ("There is
currently no code path in this project where a LIVE order both (a)
actually executes against a real broker and (b) gets persisted anywhere
with client/instrument attribution"). :class:`MacroOrderExecutionAgent`
is that consumer: it subscribes to the same real bus topic
``api_middleware/gateway.py``'s ``submit_order`` already publishes every
macro order onto, filters to ``execution_mode == "LIVE"`` (SHADOW orders
are already fully, synchronously handled inline by the gateway itself --
see its own ``_run_shadow_simulation``; this agent ignores them to avoid
double-processing), and for each one routes and executes a real order
against real crypto venues.

**Reuses, does not reimplement, two already-real, already-tested
modules**:

- :func:`~src.execution.smart_order_router.build_allocation_plan` /
  :func:`~src.execution.smart_order_router.execute_allocation_plan` --
  the exact same real fee-inclusive water-filling router
  ``src/dashboard/ems_pages/liquidity_routing.py``'s manual "Calculer ET
  exécuter" button already calls. Every real safety-machinery
  integration documented in that module's own docstring (Kill Switch,
  ``safe_financial_call``, ``MOCK_MODE``, ``UniversalOrder`` regulatory
  tagging) applies here transitively, unchanged.
- :func:`~src.execution.billing_schema.insert_client_fill` -- the real,
  durable ``client_fills`` table :mod:`~src.execution.billing_schema`
  already built for Shadow-mode fills. Every real fill this agent
  produces is written there too, tagged ``execution_mode="LIVE"`` (or
  ``"SHADOW"`` -- see this agent's own ``AgentControlState.SHADOW``
  handling below) and, new alongside this agent, a real ``broker_id``
  (see that module's own docstring) -- the column Shadow fills never
  needed (no real broker is ever involved there) but a real LIVE fill,
  routed across potentially several venues, does.

**What this closes for EOD reconciliation, concretely**: once real LIVE
fills land in ``client_fills`` with a real ``broker_id``, a day-over-day
balance ``Break`` on that broker (see
:func:`~src.execution.eod_reconciliator.detect_breaks`) can finally be
cross-referenced against the real, client-attributed fills recorded
against that same broker in the same window -- "here is what actually
traded on this broker while its balance moved," not just "the balance
moved and nobody knows why."

**Real, disclosed scope limits, stated upfront, not hidden**:

1. **No time-based algo execution yet.** ``MacroOrderRequest.
   execution_algo``/``time_window_minutes`` ask for TWAP/Iceberg-style
   slicing over time (see :mod:`~src.execution.macro_strategist`, which
   implements that slicing logic but still has no live scheduler wired
   to it). This agent executes the full requested quantity as **one**
   real Smart-Order-Routed allocation, immediately, on receipt --
   real execution, but not yet the time-sliced execution quality the
   client's chosen algo asks for. ``execution_algo`` is still recorded
   verbatim on every ``client_fills`` row (what the client asked for),
   but the actual routing/RTS 6 tag used is :data:`_ALGO_ID` (what
   really executed it) -- these are deliberately two different, honest
   fields, never conflated. This immediate-execution design is also why
   ``api_middleware/gateway.py``'s real ``cancel_order`` route (see
   :meth:`_is_cancelled`, checked here before any routing decision) has
   an honestly short real cancellation window -- most orders are
   already terminal by the time a cancel request could arrive.
2. **Crypto only.** :func:`~src.execution.smart_order_router.
   build_allocation_plan` only knows how to route across
   :class:`~src.execution.broker_gateway.CcxtBrokerAdapter` venues
   (real ``ccxt`` order books). A ``MacroOrderRequest.asset`` with no
   ``"/"`` (e.g. an OANDA-style ``"EUR_USD"``) has no real live router
   to use yet -- marked ``FAILED`` with an honest, logged reason, never
   routed through a fabricated forex path.
3. **Candidate venues are connected once, at startup, not per order.**
   A credential added via the dashboard's credential form for a venue
   this agent already failed to reach at startup is not picked up until
   this agent is restarted -- a real, disclosed limitation of a
   long-lived connection pool (chosen over reconnecting on every single
   incoming order, which would add real per-order latency/rate-limit
   churn for no benefit in the common case).
4. **Exactly one instance.** Like every other ZMQ PUB/SUB consumer in
   this project, running two instances of this agent would both receive
   and both act on the same real order -- this agent does not
   deduplicate across instances. Run exactly one, same constraint
   :class:`~src.execution.shadow_crossing_engine.ShadowCrossingEngine`
   and :class:`~src.agents.execution.agent.ExecutionAgent` already carry.

**``TOXICITY_AVOIDANCE``, now real, closing another previously-disclosed
gap**: ``MacroOrderRequest.execution_algo`` has always accepted this
value, but it behaved identically to ``AI_TWAP`` -- no real
toxicity-based behavior existed anywhere in this codebase (confirmed
before writing this: ``api_middleware/gateway.py``'s own
``_run_shadow_simulation`` routes both through the same
:func:`~src.execution.macro_strategist.slice_twap` branch). This agent
now reuses, unchanged, :class:`~execution.predator_honeypot.
AdverseFlowDefenseAgent` (built, tested, and left honestly unwired in
an earlier chapter -- its own module docstring explicitly names "no
real XGBoost_Toxicity_Score exists... OBI is the closest real signal,
VPIN is not wired into production either" as the reason) -- VPIN *is*
now real and wired (see :mod:`~src.agents.market_data.
microstructure_features`), so it is used here as exactly the real
``external_toxicity_score`` callable that class was designed to accept.
For a ``TOXICITY_AVOIDANCE`` order only (other algos are never subject
to a behavior their client didn't ask for): the real-time VPIN reading
for the asset is fetched, and :meth:`~execution.predator_honeypot.
AdverseFlowDefenseAgent.posture_for`'s real
``max_order_size_multiplier`` scales the requested quantity down --
never a fabricated block, always this real, already-tested defensive-
posture logic. Every real reduction increments :data:`~core.schemas.
TOXIC_ORDERS_EVADED_REDIS_KEY`, the real counter ``src/dashboard/
ems_pages/order_blotter.py`` used to only show in demo mode.

**One real, disclosed ceiling on this**: :class:`~execution.
predator_honeypot.AdverseFlowDefenseAgent`'s full ``TOXIC`` posture
(multiplier ``0.0``, a genuine full refusal) requires its *primary*
signal -- ``SpoofingPatternDetector``'s own real order-lifetime
ADD/CANCEL pattern detection -- to independently read ``TOXIC``. That
primary signal stays honestly unfed here: this project has no real
per-order-ID exchange event feed to drive it (public market data is
depth snapshots, not our own order lifecycle). Driven by the real VPIN
secondary signal alone, the reachable ceiling is ``CAUTIOUS`` (a real
30% size reduction, not a full refusal) -- the real, honestly degraded
mode that class's own design already anticipates for exactly this
"only a secondary signal available" case, not something this agent
silently overclaims.

**``AgentControlState``, this agent's own real meaning** (see
:class:`~core.base_agent.AgentControlState`'s own docstring for why each
agent decides this for itself): ``OFF`` refuses the order outright
before any venue is even consulted (the order is marked ``FAILED``,
same "OFF means nothing real happens" contract every other agent in
this project follows). ``SHADOW`` still runs the full real routing
decision (real order books, real fees, real allocation plan) but passes
``shadow=True`` into :func:`execute_allocation_plan`, so no real
``place_market_order`` call is ever made -- exactly
:class:`~src.agents.execution.agent.ExecutionAgent`'s own established
SHADOW semantics, applied here. The resulting simulated fill is still
recorded in ``client_fills``, honestly tagged
``execution_mode="SHADOW"`` (never ``"LIVE"``) -- an operator can pause
real client-order execution into a dry run without losing the real
routing-quality signal.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Final, Protocol

import asyncpg
import zmq.asyncio

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings
from core.nothierly_ontology import AssetClass, OrderSide, UniversalAsset
from core.redis_client import RedisUnavailableError
from core.schemas import (
    CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX,
    CLIENT_ORDERS_TOPIC,
    FILLS_BLOTTER_MAX_ENTRIES,
    FILLS_BLOTTER_REDIS_KEY,
    ORDER_AUDIT_EVENTS_TOPIC,
    ORDER_STATUS_REDIS_KEY_PREFIX,
    TOXIC_ORDERS_EVADED_REDIS_KEY,
    MessagePayload,
)
from core.timescale_client import TimescaleUnavailableError, connect_with_backoff
from execution.predator_honeypot import AdverseFlowDefenseAgent, SpoofingPatternDetector
from src.agents.market_data.microstructure_features import compute_live_microstructure_reading, fetch_real_trade_tape
from src.execution.audit_ledger import OrderAuditEvent
from src.execution.billing_schema import ensure_schema as ensure_billing_schema
from src.execution.billing_schema import insert_client_fill
from src.execution.broker_gateway import BrokerFactory, CcxtBrokerAdapter
from src.execution.smart_order_router import (
    CANDIDATE_EXCHANGES,
    SmartOrderRouterVenue,
    build_allocation_plan,
    execute_allocation_plan,
)

# Real defaults for the real, live VPIN reading fed into
# AdverseFlowDefenseAgent -- see module docstring's TOXICITY_AVOIDANCE
# section. Deliberately the same values src/dashboard/ems_pages/
# microstructure.py's own real VPIN inputs default to, for one
# consistent real reading of "current VPIN" across this project rather
# than a second, differently-tuned one.
_TOXICITY_VPIN_BUCKET_VOLUME: Final[float] = 1.0
_TOXICITY_VPIN_ROLLING_BUCKETS: Final[int] = 10


class _ExecutionVenue(SmartOrderRouterVenue, Protocol):
    """``SmartOrderRouterVenue`` plus the one extra real operation this agent needs: releasing the connection at shutdown.

    Kept as a Protocol (not the concrete ``CcxtBrokerAdapter``) so a
    real, minimal test double can stand in for it structurally -- same
    "depend on exactly what's used, nothing more" discipline
    ``SmartOrderRouterVenue`` itself documents.
    """

    async def close(self) -> None: ...

_AGENT_ID: Final[str] = "macro-order-execution-agent"
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 2.0

# RTS 6-style tag for what actually executed a real order here -- deliberately
# distinct from the client's own requested execution_algo. See module docstring.
_ALGO_ID: Final[str] = "MACRO_ORDER_SOR_EXECUTOR"

# Matches api_middleware/gateway.py's own _ORDER_STATUS_TTL_SECONDS -- both
# real writers of order_status:{order_id} must agree on this retention.
_ORDER_STATUS_TTL_SECONDS: Final[int] = 86_400

# The real "new macro order" message shape api_middleware/gateway.py's
# submit_order publishes onto CLIENT_ORDERS_TOPIC -- same discriminator
# src.execution.shadow_crossing_engine.ShadowCrossingEngine already uses to
# ignore that same topic's unrelated AGGRESSION_ADJUSTED event shape.
_REQUIRED_ORDER_FIELDS: Final[frozenset[str]] = frozenset(
    {"order_id", "client_id", "asset", "total_quantity", "side", "execution_algo"}
)


class MacroOrderExecutionAgent(BaseAgent):
    """Real, observation-*and*-action consumer of CLIENT_ORDERS_TOPIC's LIVE orders. See module docstring."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
        candidate_exchanges: tuple[str, ...] = CANDIDATE_EXCHANGES,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            subscribed_topics=(CLIENT_ORDERS_TOPIC,),
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        self._candidate_exchanges = candidate_exchanges
        self._venues: list[_ExecutionVenue] = []
        self._timescale_pool: asyncpg.Pool | None = None

        # Real TOXICITY_AVOIDANCE wiring -- see module docstring. The
        # detector's own primary ADD/CANCEL signal is never fed here (no
        # real per-order exchange event feed exists in this project); the
        # real VPIN reading, fetched fresh per order and cached here for
        # this synchronous callable to read, is the real secondary signal
        # AdverseFlowDefenseAgent was designed to accept.
        self._latest_vpin_by_symbol: dict[str, float] = {}
        self._defense_agent = AdverseFlowDefenseAgent(
            SpoofingPatternDetector(), external_toxicity_score=self._latest_vpin_by_symbol.get
        )

    async def on_tick(self) -> None:
        """Unused: this agent is stream-driven; see :meth:`run`."""
        raise NotImplementedError("MacroOrderExecutionAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Connect the durable ``client_fills`` store and every reachable candidate venue, once.

        Both degrade gracefully (see module docstring's scope limits):
        an unreachable Timescale means real fills still execute against
        real brokers but cannot be persisted for EOD attribution; a
        venue that fails to connect (e.g. no credentials configured yet)
        is simply excluded from routing until this agent restarts.
        """
        try:
            self._timescale_pool = await connect_with_backoff(self._settings.timescale, max_attempts=3)
            await ensure_billing_schema(self._timescale_pool)
        except TimescaleUnavailableError:
            self._log.warning(
                "macro_order_execution_agent_timescale_unavailable",
                agent_id=self.agent_id,
                detail="real LIVE fills will still execute, but cannot be persisted to client_fills",
            )
            self._timescale_pool = None

        for exchange_id in self._candidate_exchanges:
            try:
                broker = await BrokerFactory.create(exchange_id, settings=self._settings, redis_client=self._redis_client)
                # candidate_exchanges only ever names real ccxt.pro exchange
                # ids -- BrokerFactory.create always returns a
                # CcxtBrokerAdapter for those (see its own docstring).
                assert isinstance(broker, CcxtBrokerAdapter)
                try:
                    await broker.connect()
                except Exception:
                    # Same real fix BaseBroker.__aenter__'s own docstring
                    # documents: a failing connect() still leaves this
                    # adapter's underlying aiohttp session open (found live
                    # here too -- ccxt's own "Unclosed client session"
                    # warning on a real invalid-credential AuthenticationError)
                    # -- never bypass close() on the failure path.
                    await broker.close()
                    raise
                self._venues.append(broker)
            except Exception as exc:  # noqa: BLE001 -- one venue's startup failure must not block the others
                self._log.warning(
                    "macro_order_execution_agent_venue_connect_failed_at_startup",
                    agent_id=self.agent_id, exchange_id=exchange_id, error=str(exc),
                )

        self._log.info(
            "macro_order_execution_agent_ready",
            agent_id=self.agent_id,
            connected_venues=[venue.broker_id for venue in self._venues],
        )

    async def on_shutdown(self) -> None:
        await asyncio.gather(*(venue.close() for venue in self._venues), return_exceptions=True)
        if self._timescale_pool is not None:
            await self._timescale_pool.close()

    async def run(self) -> None:
        """Connect dependencies, then drain the real bus alongside the heartbeat and control-command listeners.

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
            self._log.info("agent_started", agent_id=self.agent_id)

            await asyncio.gather(
                self._listen_for_bus_messages(),
                self._run_heartbeat_loop(),
                self._listen_for_control_commands(),
            )
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _run_heartbeat_loop(self) -> None:
        while self._running:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

    async def _listen_for_bus_messages(self) -> None:
        async for message in self._subscriber.listen():
            if message.topic != CLIENT_ORDERS_TOPIC:
                continue
            await self._handle_client_order_message(message)

    async def _handle_client_order_message(self, message: MessagePayload) -> None:
        """Extract one real LIVE macro order from the bus, if that's really what this message is -- see module docstring.

        Never awaits :meth:`_execute_live_order` directly here: a real
        multi-venue routing decision can take real network time, and
        this loop must keep draining the next incoming bus message
        meanwhile -- dispatched as an independent background task,
        same discipline :class:`~src.execution.shadow_crossing_engine.
        ShadowCrossingEngine` already uses for the same reason.
        """
        data = message.data
        if not _REQUIRED_ORDER_FIELDS <= data.keys():
            return  # not a real new-order message (e.g. adjust_algo_aggression's AGGRESSION_ADJUSTED event)
        if str(data.get("execution_mode", "LIVE")) != "LIVE":
            return  # SHADOW already fully, synchronously handled by api_middleware/gateway.py itself

        try:
            order_id = str(data["order_id"])
            client_id = str(data["client_id"])
            asset = str(data["asset"])
            side = OrderSide(str(data["side"]))
            quantity = Decimal(str(data["total_quantity"]))
            execution_algo = str(data["execution_algo"])
        except (KeyError, ValueError, TypeError):
            self._log.warning("macro_order_execution_agent_malformed_order_skipped", agent_id=self.agent_id)
            return

        if await self._is_cancelled(order_id):
            # A real client cancellation (api_middleware/gateway.py's cancel_order)
            # arrived before this agent picked the order up -- the real, honestly
            # short window that route's own docstring discloses. Status is already
            # CANCELLED (set by that route itself); nothing more to do here.
            self._log.info("macro_order_execution_agent_order_already_cancelled", agent_id=self.agent_id, order_id=order_id)
            return

        control_state = self.control_state
        if control_state == AgentControlState.OFF:
            # Consistent with every other agent's OFF meaning "do nothing
            # real" -- see core.base_agent.AgentControlState's own docstring.
            self._log.info("macro_order_execution_agent_off_order_refused", agent_id=self.agent_id, order_id=order_id)
            await self._update_order_status(
                order_id, status="FAILED", asset=asset,
                rejection_reason="Agent d'exécution désactivé (OFF) par un opérateur -- ordre refusé avant tout routage.",
            )
            return

        asyncio.create_task(
            self._execute_live_order(
                order_id=order_id, client_id=client_id, asset=asset, side=side, quantity=quantity,
                execution_algo=execution_algo, shadow=control_state == AgentControlState.SHADOW,
            )
        )

    async def _execute_live_order(
        self, *, order_id: str, client_id: str, asset: str, side: OrderSide, quantity: Decimal,
        execution_algo: str, shadow: bool,
    ) -> None:
        """Route and (unless ``shadow``) really execute one client order across every connected real venue."""
        await self._update_order_status(order_id, status="ROUTING")

        base, sep, quote = asset.partition("/")
        if not sep:
            # See module docstring's scope limit #2: no real live router for non-crypto assets yet.
            self._log.warning(
                "macro_order_execution_agent_non_crypto_asset_unsupported",
                agent_id=self.agent_id, order_id=order_id, asset=asset,
            )
            await self._update_order_status(
                order_id, status="FAILED", asset=asset,
                rejection_reason=f"Aucun routeur réel pour un actif non-crypto ({asset}) -- voir le docstring du module.",
            )
            return

        if execution_algo == "TOXICITY_AVOIDANCE":
            vpin_reading = await self._fetch_current_vpin(asset)
            if vpin_reading is not None:
                self._latest_vpin_by_symbol[asset] = vpin_reading
            posture = self._defense_agent.posture_for(asset)
            if posture.max_order_size_multiplier < 1.0:
                original_quantity = quantity
                quantity = quantity * Decimal(str(posture.max_order_size_multiplier))
                await self._increment_toxic_orders_evaded_counter()
                self._log.warning(
                    "macro_order_execution_agent_toxicity_avoidance_triggered",
                    agent_id=self.agent_id, order_id=order_id, asset=asset, posture=posture.mode.value,
                    original_quantity=str(original_quantity), adjusted_quantity=str(quantity), reason=posture.reason,
                )
                if quantity <= 0:
                    await self._update_order_status(
                        order_id, status="FAILED", asset=asset,
                        rejection_reason=f"Évitement de toxicité réel (TOXICITY_AVOIDANCE) : {posture.reason}",
                    )
                    return

        if not self._venues:
            self._log.error("macro_order_execution_agent_no_venue_reachable", agent_id=self.agent_id, order_id=order_id)
            await self._update_order_status(
                order_id, status="FAILED", asset=asset,
                rejection_reason="Aucune venue crypto réelle joignable au démarrage de cet agent.",
            )
            return

        plan = await build_allocation_plan(self._venues, asset, side, quantity)
        if not plan.allocations:
            self._log.error("macro_order_execution_agent_no_liquidity", agent_id=self.agent_id, order_id=order_id, asset=asset)
            await self._update_order_status(
                order_id, status="FAILED", asset=asset,
                rejection_reason=f"Aucune liquidité réelle disponible sur {asset} parmi les venues connectées.",
            )
            return

        universal_asset = UniversalAsset(
            symbol=asset, asset_class=AssetClass.CRYPTO_SPOT, base_currency=base or asset,
            quote_currency=quote or "USDT", tick_size=Decimal("0.01"), lot_size=Decimal("0.00000001"),
        )
        fee_rates = {venue.broker_id: venue.get_taker_fee_rate(asset) for venue in self._venues}

        result = await execute_allocation_plan(
            plan, self._venues, asset, asset=universal_asset, algo_id=_ALGO_ID,
            regulatory_client_shortcode=client_id, client_id=client_id, shadow=shadow,
            settings=self._settings, redis_client=self._redis_client,
        )

        for order in result.orders:
            await self._persist_real_fill(
                order_id=order_id, client_id=client_id, instrument=asset, side=side,
                execution_algo=execution_algo, broker_id=order.broker_id,
                quantity=float(order.filled_quantity),
                price=float(order.avg_fill_price) if order.avg_fill_price is not None else 0.0,
                fee_rate=fee_rates.get(order.broker_id, 0.0), shadow=shadow,
            )
        for broker_id, reason in result.failed_venues:
            self._log.warning(
                "macro_order_execution_agent_venue_execution_failed",
                agent_id=self.agent_id, order_id=order_id, broker_id=broker_id, reason=reason,
            )

        if not result.orders:
            await self._update_order_status(
                order_id, status="FAILED", asset=asset,
                rejection_reason="Chaque venue a échoué à l'exécution réelle -- voir les échecs par venue dans les logs.",
            )
        elif result.total_filled_quantity >= plan.requested_quantity - Decimal("1e-9"):
            await self._update_order_status(order_id, status="COMPLETED")
        else:
            await self._update_order_status(order_id, status="PARTIALLY_FILLED")

    async def _is_cancelled(self, order_id: str) -> bool:
        """Real check against the marker ``api_middleware/gateway.py``'s ``cancel_order`` route sets -- see that constant's own docstring."""
        if self._redis_client is None:
            return False
        try:
            return bool(await self._redis_client.execute("exists", f"{CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX}{order_id}"))
        except RedisUnavailableError:
            return False

    async def _fetch_current_vpin(self, symbol: str) -> float | None:
        """Real, current VPIN for ``symbol`` from the real trade tape, or ``None`` if unavailable -- see module docstring."""
        if self._redis_client is None:
            return None
        trade_tape = await fetch_real_trade_tape(self._redis_client, symbol)
        reading = compute_live_microstructure_reading(
            trade_tape, symbol=symbol, vpin_bucket_volume=_TOXICITY_VPIN_BUCKET_VOLUME,
            vpin_rolling_buckets=_TOXICITY_VPIN_ROLLING_BUCKETS,
        )
        return reading.vpin

    async def _increment_toxic_orders_evaded_counter(self) -> None:
        """Real, cumulative counter -- see :data:`~core.schemas.TOXIC_ORDERS_EVADED_REDIS_KEY`."""
        if self._redis_client is None:
            return
        try:
            await self._redis_client.execute("incr", TOXIC_ORDERS_EVADED_REDIS_KEY)
        except RedisUnavailableError:
            self._log.warning("macro_order_execution_agent_toxic_counter_increment_failed", agent_id=self.agent_id)

    async def _persist_real_fill(
        self, *, order_id: str, client_id: str, instrument: str, side: OrderSide, execution_algo: str,
        broker_id: str, quantity: float, price: float, fee_rate: float, shadow: bool,
    ) -> None:
        """Persist one real (or, under ``shadow``, honestly-simulated) fill: the live Redis blotter + the durable ``client_fills`` table.

        Mirrors ``api_middleware/gateway.py``'s ``_write_shadow_fill`` dual-write exactly, for the same reason: the
        Redis blotter (src/dashboard/ems_pages/order_blotter.py) is immediate but capped/no TTL-survival guarantee;
        client_fills is durable, and the only one EOD reconciliation reads.
        """
        fee_usd = quantity * price * fee_rate
        timestamp = datetime.now(timezone.utc)

        entry = json.dumps(
            {
                "timestamp": timestamp.timestamp(),
                "agent_id": self.agent_id,
                "client_id": client_id,
                "order_id": order_id,
                "instrument": instrument,
                "action": "ENTRY_LONG" if side == OrderSide.BUY else "ENTRY_SHORT",
                "price": round(price, 6),
                "units": round(quantity, 6),
                "fee_usd": round(fee_usd, 6),
                "latency_ms": 0.0,
                "reason": f"Real macro order execution via Smart Order Router (broker={broker_id}, order {order_id})",
                "mock_mode": self._settings.mock_mode,
                "agent_control_shadow": shadow,
            }
        )
        if self._redis_client is not None:
            try:
                await self._redis_client.execute("lpush", FILLS_BLOTTER_REDIS_KEY, entry)
                await self._redis_client.execute("ltrim", FILLS_BLOTTER_REDIS_KEY, 0, FILLS_BLOTTER_MAX_ENTRIES - 1)
            except RedisUnavailableError:
                self._log.warning("macro_order_execution_agent_blotter_write_failed", agent_id=self.agent_id, order_id=order_id)

        if self._timescale_pool is not None:
            try:
                await insert_client_fill(
                    self._timescale_pool, time=timestamp, client_id=client_id, order_id=order_id,
                    instrument=instrument, side=side.value, execution_algo=execution_algo,
                    execution_mode="SHADOW" if shadow else "LIVE", quantity=quantity, price=price,
                    fee_usd=fee_usd, broker_id=broker_id,
                )
            except asyncpg.PostgresError:
                self._log.warning(
                    "macro_order_execution_agent_client_fills_write_failed",
                    agent_id=self.agent_id, order_id=order_id, client_id=client_id,
                )

        audit_event = OrderAuditEvent(
            event_type="EXECUTED", order_id=order_id, client_id=client_id, asset=instrument,
            execution_mode="SHADOW" if shadow else "LIVE",
            detail=f"Fill réel via Smart Order Router : {quantity} {instrument} @ {price:.6f} (broker={broker_id}).",
        )
        await self.publish(ORDER_AUDIT_EVENTS_TOPIC, audit_event.model_dump(mode="json"))

    async def _update_order_status(
        self, order_id: str, *, status: str, asset: str | None = None, rejection_reason: str | None = None
    ) -> None:
        """Merge a new ``status``/``updated_at`` into the real order-status record ``api_middleware/gateway.py`` wrote at submission.

        Read-modify-write, preserving every other field (``client_id``,
        ``aggression_level``, ``execution_mode``) -- this agent is not
        the record's original author, just its second, execution-side
        writer, exactly the contract ``OrderStatusValue``'s own
        docstring already declared before this agent existed.

        Args:
            asset: Only needed alongside ``rejection_reason`` (the
                original status record carries no ``asset`` field to
                read back) -- see :class:`~src.execution.audit_ledger.
                OrderAuditEvent`.
            rejection_reason: When given and ``status == "FAILED"``, a
                real ``REJECTED`` :class:`~src.execution.audit_ledger.
                OrderAuditEvent` is published onto :data:`~core.schemas.
                ORDER_AUDIT_EVENTS_TOPIC` -- never for any other status
                transition (``ROUTING``/``COMPLETED``/``PARTIALLY_FILLED``
                are not rejections).
        """
        if self._redis_client is None:
            return
        key = f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id}"
        try:
            raw = await self._redis_client.execute("get", key)
        except RedisUnavailableError:
            return
        if raw is None:
            self._log.warning("macro_order_execution_agent_order_status_missing", agent_id=self.agent_id, order_id=order_id)
            return
        try:
            record: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return
        record["status"] = status
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            await self._redis_client.execute("set", key, json.dumps(record), ex=_ORDER_STATUS_TTL_SECONDS)
        except RedisUnavailableError:
            self._log.warning("macro_order_execution_agent_order_status_write_failed", agent_id=self.agent_id, order_id=order_id)

        if status == "FAILED" and rejection_reason is not None:
            audit_event = OrderAuditEvent(
                event_type="REJECTED", order_id=order_id, client_id=record.get("client_id"), asset=asset,
                execution_mode=record.get("execution_mode"), detail=rejection_reason,
            )
            await self.publish(ORDER_AUDIT_EVENTS_TOPIC, audit_event.model_dump(mode="json"))

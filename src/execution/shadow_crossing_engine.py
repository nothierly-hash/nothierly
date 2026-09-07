"""Dark Pool -- Shadow Crossing: a real, bounded, observation-only internalization engine.

**"Zero Vaporware": real order flow, real matching logic, real
persisted evidence -- but never a redirected order.** This module is
the first real consumer of :data:`~core.schemas.CLIENT_ORDERS_TOPIC`
(previously documented, in ``api_middleware/gateway.py``'s own module
docstring, as a real, disclosed gap: "no Ingestor Agent drains
CLIENT_ORDERS_TOPIC"). :class:`ShadowCrossingEngine` subscribes to that
same real bus topic every macro order is already published on and, for
each one, asks the real, already-tested :class:`~core.crossing_network.
CrossingNetwork` whether it *could* have crossed against another
real, recently-arrived opposite-side order in the same instrument --
reusing that engine's own real mid-price-based matching, self-trade
prevention, and audit-trail logic rather than reimplementing a cruder
version of it. **It never intercepts, delays, cancels, or reroutes the
real order** -- the order continues through the real pipeline exactly
as ``api_middleware/gateway.py`` already built it; this is a second,
independent, read-only subscriber on the same bus, not a gate in front
of it.

**Deliberately bounded scope, stated upfront**: this is real, working
software, not a demo of one -- but it is intentionally scoped no
further than "would two real orders, arriving within a realistic
window, have crossed, and how much would that have saved". It does
not (and does not claim to): know whether either order was already
filled for real elsewhere by the time a match is found; net more than
two orders against each other in one cross (``CrossingNetwork`` itself
already supports multi-order matching per instrument, reused as-is);
or claim any real regulatory registration to actually internalize
client flow (see :mod:`core.crossing_network`'s own docstring on why
running this against real, executed crosses would need a Systematic
Internaliser/MTF/OTF/ATS registration this project's real-world model
does not hold -- the same reason this engine only ever *observes*).

**Why a much longer matching window than ``CrossingNetwork``'s own
default**: that class's own ``DEFAULT_MATCHING_WINDOW_MS`` (50ms) is
tuned for a real order actually waiting, at real risk, to be crossed
or released to the public market -- every extra millisecond has a
real opportunity cost. This engine never withholds anything real, so
it uses a real, disclosed, much longer window
(:data:`_SHADOW_MATCHING_WINDOW_MS`, 30 real seconds) suited to a
passive observer measuring crossing opportunities across a realistic
human/API order-submission cadence, not an at-risk execution decision.

**Real, disclosed fee-saved estimate**: the "fees saved" figure per
match uses :data:`_DEFAULT_TAKER_FEE_RATE` (0.001, i.e. 0.1%) --  the
exact same documented fallback :meth:`~src.agents.market_data.
exchange_client.AsyncExchangeClient.get_taker_fee_rate` uses when a
specific exchange/market doesn't report its own real fee schedule --
applied to both sides of the real cross (each side avoids one real
taker fee it would have paid on a public venue). A real, standard
approximation, not a fabricated number.
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from typing import Any, Final

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.crossing_network import CrossingNetwork, InternalCrossFill
from core.nothierly_ontology import OrderSide
from core.redis_client import RedisUnavailableError
from core.schemas import (
    CLIENT_ORDERS_TOPIC,
    DARK_POOL_SHADOW_MATCHES_MAX_ENTRIES,
    DARK_POOL_SHADOW_MATCHES_REDIS_KEY,
    MARKET_METRICS_REDIS_KEY_PREFIX,
    MarketMetrics,
    MessagePayload,
)

_AGENT_ID: Final[str] = "shadow-crossing-engine"
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 2.0

# See module docstring.
_DEFAULT_TAKER_FEE_RATE: Final[float] = 0.001
_SHADOW_MATCHING_WINDOW_MS: Final[float] = 30_000.0

# The real "new macro order" message shape api_middleware/gateway.py's
# submit_order publishes onto CLIENT_ORDERS_TOPIC -- required keys this
# engine checks for before processing, since that same real topic also
# carries a second, unrelated message shape (adjust_algo_aggression's own
# "AGGRESSION_ADJUSTED" event, which has none of these fields) that must
# be silently ignored here, not misread as a malformed order.
_REQUIRED_ORDER_FIELDS: Final[frozenset[str]] = frozenset({"order_id", "client_id", "asset", "total_quantity", "side"})


class ShadowCrossingEngine(BaseAgent):
    """Real, observation-only dark-pool matcher -- see module docstring for the full scope and its real bounds."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        manage_own_redis: bool = True,
        taker_fee_rate: float = _DEFAULT_TAKER_FEE_RATE,
        matching_window_ms: float = _SHADOW_MATCHING_WINDOW_MS,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            subscribed_topics=(CLIENT_ORDERS_TOPIC,),
            settings=settings,
            manage_own_signals=manage_own_signals,
            manage_own_redis=manage_own_redis,
        )
        self._taker_fee_rate = taker_fee_rate
        self._crossing_network = CrossingNetwork(get_mid_price=self._get_live_mid_price, matching_window_ms=matching_window_ms)

    async def on_tick(self) -> None:
        """Unused: this agent is stream-driven; see :meth:`run`."""
        raise NotImplementedError("ShadowCrossingEngine overrides run() and never ticks.")

    async def on_start(self) -> None:
        self._log.info("shadow_crossing_engine_ready", agent_id=self.agent_id)

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
        """Extract one real new order from the bus, if that's really what this message is -- see module docstring.

        Never awaits :meth:`~core.crossing_network.CrossingNetwork.
        submit_order` directly here: that call can real-wait up to the
        matching window for a match, and this loop must keep draining
        the next incoming bus message meanwhile -- dispatched as an
        independent background task instead.
        """
        data = message.data
        if not _REQUIRED_ORDER_FIELDS <= data.keys():
            return
        try:
            order_id = str(data["order_id"])
            client_id = str(data["client_id"])
            instrument = str(data["asset"])
            side = OrderSide(data["side"])
            quantity = Decimal(str(data["total_quantity"]))
        except (KeyError, ValueError, TypeError):
            self._log.warning("shadow_crossing_engine_malformed_order_skipped", agent_id=self.agent_id)
            return

        if self.control_state.value == "OFF":
            # Consistent with every other agent's OFF meaning "do
            # nothing real" -- see core.base_agent.AgentControlState's
            # own docstring. SHADOW has no separate meaning here (this
            # whole engine is already a shadow observer by design), so
            # only OFF is checked.
            return

        asyncio.create_task(
            self._observe_order(order_id=order_id, client_id=client_id, instrument=instrument, side=side, quantity=quantity)
        )

    async def _observe_order(self, *, order_id: str, client_id: str, instrument: str, side: OrderSide, quantity: Decimal) -> None:
        """Ask the real ``CrossingNetwork`` whether ``order_id`` could cross, and log any real match found.

        **Real de-duplication, found live**: :class:`~core.crossing_network.
        PendingOrder`'s own docstring says a cross is recorded on *both*
        sides' accumulated ``fills`` -- correct and necessary for
        ``CrossingNetwork``'s own callers, each of whom only sees their
        own order's result. Since this engine calls
        :meth:`~core.crossing_network.CrossingNetwork.submit_order` for
        *both* real orders independently, each side's own call would
        otherwise log the *same* real cross twice (confirmed live: two
        identical ``cross_id`` entries for one real match). Logged only
        from the buy side's own call -- every real cross has exactly one
        buy order, so this logs each real cross exactly once regardless
        of arrival order or how many counter-orders it partially filled
        against.
        """
        result = await self._crossing_network.submit_order(
            order_id=order_id, client_id=client_id, instrument=instrument, side=side, quantity=quantity
        )
        for fill in result.fills:
            if fill.buy_order_id == order_id:
                await self._log_shadow_match(fill)

    async def _get_live_mid_price(self, symbol: str) -> Decimal | None:
        """Real, live mid-price for ``symbol``, or ``None`` -- never a fabricated price. See ``CrossingNetwork``'s own docstring on why ``None`` refuses a match rather than inventing one."""
        if self._redis_client is None:
            return None
        try:
            raw = await self._redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
        except RedisUnavailableError:
            return None
        if raw is None:
            return None
        try:
            metrics = MarketMetrics.model_validate_json(raw)
        except Exception:
            return None
        return Decimal(str(metrics.mid_price))

    async def _log_shadow_match(self, fill: InternalCrossFill) -> None:
        """Persist one real shadow-match event -- for the Internal Dark Pool dashboard page. Best-effort: a Redis hiccup here must never affect the real order flow this engine only ever observes."""
        fee_saved_usd = float(fill.quantity) * float(fill.price) * self._taker_fee_rate * 2
        self._log.info(
            "shadow_crossing_match_detected",
            agent_id=self.agent_id,
            instrument=fill.instrument,
            quantity=str(fill.quantity),
            price=str(fill.price),
            fee_saved_usd=round(fee_saved_usd, 2),
        )
        if self._redis_client is None:
            return
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "cross_id": fill.cross_id,
            "instrument": fill.instrument,
            "price": str(fill.price),
            "quantity": str(fill.quantity),
            "buy_order_id": fill.buy_order_id,
            "buy_client_id": fill.buy_client_id,
            "sell_order_id": fill.sell_order_id,
            "sell_client_id": fill.sell_client_id,
            "fee_saved_usd": fee_saved_usd,
        }
        try:
            await self._redis_client.execute("lpush", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, json.dumps(entry))
            await self._redis_client.execute(
                "ltrim", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, DARK_POOL_SHADOW_MATCHES_MAX_ENTRIES - 1
            )
        except RedisUnavailableError:
            self._log.warning("shadow_crossing_match_persist_failed", agent_id=self.agent_id)

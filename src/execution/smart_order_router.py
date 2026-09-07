"""Smart Order Router: real fee-inclusive water-filling across multiple real crypto venues.

**Fills a real, already-disclosed gap**: ``src/dashboard/ems_pages/
liquidity_routing.py``'s own module docstring says its "Smart Routing"
toggle "does not itself split a live order across brokers today" and
its own bottom-of-page ``not_yet_implemented("Allocation de capital par
route de liquidité", ...)`` says the same -- "Nécessite un routeur en
direct répartissant réellement le capital entre brokers -- pas encore
construit." This module is that router, built against that page's own
real candidate venues (``_CANDIDATE_EXCHANGES = ("binance", "bybit",
"okx", "deribit")``, each reachable via the real, already-existing
:class:`~src.execution.broker_gateway.CcxtBrokerAdapter`).

**Water-filling, precisely**: :func:`build_allocation_plan` fetches
every venue's real order-book depth (:meth:`~src.execution.
broker_gateway.CcxtBrokerAdapter.get_orderbook_snapshot`, real top-
:data:`~core.schemas.ORDER_BOOK_TOP_N` levels) and real per-market
taker fee (:meth:`~src.execution.broker_gateway.CcxtBrokerAdapter.
get_taker_fee_rate`, ``ccxt``'s own published fee schedule -- see that
method's docstring) concurrently, merges every venue's levels into one
list ranked by real fee-inclusive effective price (ask price times
``1 + taker_fee`` for a BUY, bid price times ``1 - taker_fee`` for a
SELL), then greedily fills the requested quantity from the best
effective price downward, spilling into the next-best level/venue only
once the current one's real available depth is exhausted -- the
standard water-filling allocation this project's dashboard already
promised, not an approximation of it.

**Safety-machinery integration, and one real, disclosed non-
integration**:

- **Kill Switch**: :meth:`~src.agents.market_data.exchange_client.
  AsyncExchangeClient.create_market_order` (which :meth:`~src.execution.
  broker_gateway.CcxtBrokerAdapter.place_market_order` calls) is
  **already** ``@requires_system_active``-decorated -- a tripped switch
  raises :class:`~core.security.KillSwitchActiveError` from inside
  :func:`execute_allocation_plan`'s own per-venue placement call with no
  extra wrapping needed here. Verified by reading that decorator's real
  call chain before writing this module, not assumed.
- **safe_financial_call**: every real child order goes through
  :func:`~core.security.safe_financial_call` with
  ``retry_on_failure=False`` -- an order placement must never be
  silently resubmitted (see that function's own docstring), and this is
  exactly the same rule :class:`~src.agents.execution.agent.
  ExecutionAgent._place_order` already follows for its own single-venue
  entries.
- **MOCK_MODE**: when :attr:`~core.config.AppSettings.mock_mode` is
  true (or ``shadow=True`` is passed explicitly), no real
  ``place_market_order`` call is made for any venue -- each allocation
  is recorded as a simulated fill at its own real, just-fetched
  effective price, the same "MOCK_MODE must mean MOCK_MODE" discipline
  ``ExecutionAgent._place_order`` already follows.
- **UniversalOrder regulatory tagging**: every real or simulated child
  fill is normalized through :func:`~core.nothierly_ontology.
  order_from_ccxt` into a real :class:`~core.nothierly_ontology.
  UniversalOrder` -- callers must supply ``algo_id``/
  ``decision_timestamp_utc``/``regulatory_client_shortcode`` themselves
  (same required-kwarg contract that function already enforces), so a
  routed order is taggable for MiFID II RTS 6/22-style reporting exactly
  like every other order in this system, not a bare, untagged broker dict.
- **Guardian AI -- deliberately NOT called here, stated plainly**:
  :func:`~src.agents.guardian_ai.assess_risk`'s ``TriggerType`` is a
  closed, four-value ``Literal`` (``RISK_INCREASE``/``MANUAL_OVERRIDE``/
  ``DRAWDOWN``/``SIGNAL_COLLISION``), each with one specific, already-
  wired real caller. Routing a large order across venues fits none of
  the four -- inventing a fifth trigger (or misusing an existing one)
  to force a Guardian AI call in here would be a fake integration, not a
  real one. If a genuine "SOR order size vs. real available liquidity"
  risk check is wanted later, it needs a real fifth trigger type added
  to that module first, deliberately, not bolted on here.

**Leg-risk auto-unwind, and its one real, load-bearing limitation**:
a real red-team chaos test (``tests/chaos_monkey.py``) found that a
partial-venue failure (one venue fills, another 502s) previously left
the filled leg's exposure standing with no automatic corrective action
-- a genuine "Orphan Trade" risk. :func:`execute_allocation_plan` now
attempts a real, opposite-side unwind order (sized to exactly what each
successful leg filled) whenever at least one venue failed and at least
one other genuinely filled, real (non-MOCK_MODE/non-``shadow``) placements
only. **The one real limitation, found empirically, not theorized**:
``core.security.safe_financial_call`` (which every child placement,
including this one's triggering failure, goes through) trips the
*global* Kill Switch on any single venue's exhausted failure -- and the
unwind call itself still goes through the same
``@requires_system_active``-decorated ``place_market_order`` path (see
the Kill Switch paragraph above), so in the single most common real
trigger case (the very failure that necessitates unwinding also trips
the switch), the unwind attempt is itself blocked by
:class:`~core.security.KillSwitchActiveError` before it can flatten
anything. This is reported honestly via ``unwind_failed_venues`` /
``leg_risk_fully_flattened=False`` -- a human must flatten manually in
that case, right now -- rather than silently claimed as resolved. A
deeper fix exists (exempting a real, quantity-bounded unwind order from
the global Kill Switch, mirroring the exemption
``core.security``'s own module docstring already grants cancellations,
"blocking those during an emergency would strand exposure instead of
tearing it down") but changes what the Kill Switch guarantees during a
real panic, so it is a deliberate policy decision for whoever owns that
guarantee, not something to bake in silently here.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Final, Literal, Protocol, Sequence

from core.config import AppSettings, get_settings
from core.logger import get_logger
from core.nothierly_ontology import OrderSide, UniversalAsset, UniversalOrder, order_from_ccxt
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import ORDER_BOOK_TOP_N, SOR_ROUTING_LOG_MAX_ENTRIES, SOR_ROUTING_LOG_REDIS_KEY, OrderBookSnapshot
from core.security import FinancialCallAbortedError, KillSwitchActiveError, safe_financial_call

_log = get_logger(__name__)

# The real candidate crypto venues this router is built against -- see
# module docstring. Shared, not duplicated: src/dashboard/ems_pages/
# liquidity_routing.py's manual routing form and
# src.agents.macro_order_execution.agent.MacroOrderExecutionAgent's real
# LIVE-order executor both route across exactly this set.
CANDIDATE_EXCHANGES: Final[tuple[str, ...]] = ("binance", "bybit", "okx", "deribit")


class SmartOrderRouterVenue(Protocol):
    """The minimal real surface this router needs from one venue -- deliberately narrower than the full :class:`~src.execution.broker_gateway.BaseBroker` ABC.

    :class:`~src.execution.broker_gateway.CcxtBrokerAdapter` satisfies
    this structurally (no explicit inheritance needed) -- this Protocol
    exists so the router depends on exactly the four operations it
    actually uses, not the full broker surface (``cancel_order``,
    ``get_margin_snapshot``, ...) it never touches, and so tests can
    supply a plain, real fake venue with zero mocking framework and zero
    type-checking friction (see ``tests/test_smart_order_router.py``).
    """

    broker_id: str

    async def get_orderbook_snapshot(self, symbol: str, limit: int = ...) -> OrderBookSnapshot: ...

    def get_taker_fee_rate(self, symbol: str) -> float: ...

    def place_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> Awaitable[dict[str, Any]]:
        """Declared as a plain method returning ``Awaitable`` (not ``async def``) so a ``@requires_system_active``-decorated implementation -- which real ``CcxtBrokerAdapter.place_market_order`` is, transitively (see module docstring) -- still satisfies this Protocol: that decorator's wrapper is itself typed to return ``Awaitable[_R]``, a supertype of the plain ``Coroutine`` an ``async def ... -> dict[str, Any]`` signature here would have required."""
        ...


def _effective_price(level_price: float, taker_fee_rate: float, side: OrderSide) -> float:
    """Real fee-inclusive cost (BUY) or proceeds (SELL) per unit at one real order-book level."""
    return level_price * (1 + taker_fee_rate) if side == OrderSide.BUY else level_price * (1 - taker_fee_rate)


@dataclass(frozen=True)
class VenueAllocation:
    """How much of the parent order one venue is responsible for, and at what real volume-weighted effective price."""

    broker_id: str
    quantity: float
    effective_price: float


@dataclass(frozen=True)
class AllocationPlan:
    """The real water-filling result: a real per-venue split of ``requested_quantity``, ranked by real effective price."""

    side: OrderSide
    requested_quantity: Decimal
    allocations: tuple[VenueAllocation, ...]

    @property
    def allocated_quantity(self) -> Decimal:
        return Decimal(str(sum(allocation.quantity for allocation in self.allocations)))

    @property
    def fully_filled(self) -> bool:
        """``False`` means every venue's real, currently-quoted depth was exhausted before the full quantity was placed -- a real, honest partial-liquidity outcome, not an error.

        **Real bug fixed here, found via a live water-filling run against
        real Deribit/OKX order books**: summing several venues' real
        ``float`` allocations can land a hair below ``requested_quantity``
        purely from binary floating-point rounding (observed live:
        ``0.9999999999999999`` vs. a requested ``1.0``) even though the
        water-filling loop itself allocated the full amount. A strict
        ``>=`` misreported that as a partial fill -- a real, user-facing
        honesty bug (understating what was actually allocated), not a
        genuine liquidity shortfall. A tiny epsilon tolerance, far below
        any real venue's minimum lot size, absorbs the rounding noise
        without masking an actual partial fill (a real shortfall is
        always many orders of magnitude larger than this).
        """
        return self.allocated_quantity >= self.requested_quantity - Decimal("1e-9")

    @property
    def savings_vs_worst_level_usd(self) -> float:
        """Real, honest, always-non-negative savings from water-filling vs. paying the worst executed level for the whole quantity.

        Water-filling processes real levels in effective-price order
        (best first), so the *worst* effective price actually used in
        this plan is exactly what a naive router -- one that picked a
        single price and filled the whole order there instead of
        spreading across the real, better-priced levels found along the
        way -- would have paid (BUY) or received (SELL) for the entire
        quantity. This is a real, directly computable quantity from
        data already on this plan (no extra quote fetch needed), not an
        invented bps figure: for a BUY, ``naive_cost - actual_cost``;
        for a SELL, ``actual_proceeds - naive_proceeds``. Zero for a
        single-venue, single-level plan (nothing to have saved against).
        """
        if not self.allocations:
            return 0.0
        worst_price = (
            max(a.effective_price for a in self.allocations)
            if self.side == OrderSide.BUY
            else min(a.effective_price for a in self.allocations)
        )
        actual_cost = sum(a.quantity * a.effective_price for a in self.allocations)
        naive_cost = float(self.allocated_quantity) * worst_price
        return naive_cost - actual_cost if self.side == OrderSide.BUY else actual_cost - naive_cost


async def build_allocation_plan(
    venues: Sequence[SmartOrderRouterVenue], symbol: str, side: OrderSide, quantity: Decimal, *, depth: int = ORDER_BOOK_TOP_N
) -> AllocationPlan:
    """Fetch every venue's real book + fee concurrently, then greedily water-fill ``quantity`` by real effective price.

    A venue whose quote fetch fails (network error, symbol not listed
    there, ...) is logged and skipped -- routing still proceeds across
    whichever venues did answer, rather than failing the whole plan for
    one bad venue.
    """

    async def _quote_one(venue: SmartOrderRouterVenue) -> tuple[str, float, list[tuple[float, float]]] | BaseException:
        try:
            book = await venue.get_orderbook_snapshot(symbol, depth)
        except Exception as exc:  # noqa: BLE001 -- one venue's failure must not sink the whole plan, see docstring
            return exc
        fee_rate = venue.get_taker_fee_rate(symbol)
        levels = book.bids if side == OrderSide.SELL else book.asks
        return venue.broker_id, fee_rate, [(level.price, level.volume) for level in levels]

    quotes = await asyncio.gather(*(_quote_one(venue) for venue in venues))

    # (effective_price, broker_id, real_available_volume) across every venue's real levels, merged.
    merged_levels: list[tuple[float, str, float]] = []
    for quote in quotes:
        if isinstance(quote, BaseException):
            _log.warning("smart_order_router_venue_quote_failed", symbol=symbol, error=str(quote))
            continue
        broker_id, fee_rate, levels = quote
        for price, volume in levels:
            merged_levels.append((_effective_price(price, fee_rate, side), broker_id, volume))

    merged_levels.sort(key=lambda item: item[0], reverse=(side == OrderSide.SELL))

    remaining = float(quantity)
    per_venue_fills: dict[str, list[tuple[float, float]]] = {}
    for effective_price, broker_id, available_volume in merged_levels:
        if remaining <= 1e-12:
            break
        take = min(available_volume, remaining)
        per_venue_fills.setdefault(broker_id, []).append((take, effective_price))
        remaining -= take

    allocations = tuple(
        VenueAllocation(
            broker_id=broker_id,
            quantity=sum(qty for qty, _ in fills),
            effective_price=sum(qty * price for qty, price in fills) / sum(qty for qty, _ in fills),
        )
        for broker_id, fills in per_venue_fills.items()
    )
    return AllocationPlan(side=side, requested_quantity=quantity, allocations=allocations)


async def log_routing_decision(redis_client: RedisClient, plan: AllocationPlan, symbol: str) -> None:
    """Persist one real, executed routing decision onto :data:`~core.schemas.SOR_ROUTING_LOG_REDIS_KEY`.

    Called from :func:`execute_allocation_plan` for every real routing
    decision it actually acts on -- never for a "Calculer le plan"
    preview alone (:func:`build_allocation_plan` on its own never logs
    here), since a plan that was only previewed, not executed, is not
    a real decision yet. Logs regardless of whether every child order
    actually filled: this records the *routing* decision's own real
    quality (which venues, at what real effective price, how much
    saved vs. the worst level used), not per-venue execution outcomes
    -- those are already logged separately by :func:`execute_allocation_plan`.
    Best-effort: a Redis hiccup here must never fail a real order routing
    that otherwise succeeded.
    """
    entry = {
        "timestamp": time.time(),
        "symbol": symbol,
        "side": plan.side.value,
        "requested_quantity": str(plan.requested_quantity),
        "allocated_quantity": str(plan.allocated_quantity),
        "fully_filled": plan.fully_filled,
        "savings_usd": plan.savings_vs_worst_level_usd,
        "allocations": [
            {"broker_id": a.broker_id, "quantity": a.quantity, "effective_price": a.effective_price}
            for a in plan.allocations
        ],
    }
    try:
        await redis_client.execute("lpush", SOR_ROUTING_LOG_REDIS_KEY, json.dumps(entry))
        await redis_client.execute("ltrim", SOR_ROUTING_LOG_REDIS_KEY, 0, SOR_ROUTING_LOG_MAX_ENTRIES - 1)
    except RedisUnavailableError:
        _log.warning("smart_order_router_routing_log_persist_failed", symbol=symbol)


@dataclass(frozen=True)
class SmartOrderRouterResult:
    """One :func:`execute_allocation_plan` outcome: every venue that filled, and every venue that didn't, with why.

    ``unwind_orders``/``unwind_failed_venues`` are populated only when a
    real leg-risk auto-unwind was attempted (some venues filled while
    others failed) -- see :func:`execute_allocation_plan`'s own
    docstring for the one real, disclosed limitation on when this can
    actually succeed. Kept structurally separate from ``orders``/
    ``failed_venues`` (the *parent* order's own real fills) rather than
    commingled: an unwind fill is a corrective trade, not progress
    toward the client's originally requested size, and conflating the
    two would corrupt any downstream position/TCA accounting that reads
    ``total_filled_quantity``.
    """

    orders: tuple[UniversalOrder, ...]
    failed_venues: tuple[tuple[str, str], ...]  # (broker_id, reason)
    unwind_orders: tuple[UniversalOrder, ...] = ()
    unwind_failed_venues: tuple[tuple[str, str], ...] = ()  # (broker_id, reason) -- exposure NOT flattened

    @property
    def total_filled_quantity(self) -> Decimal:
        return sum((order.filled_quantity for order in self.orders), Decimal(0))

    @property
    def leg_risk_fully_flattened(self) -> bool:
        """``True`` only when every filled leg that needed unwinding was genuinely, successfully unwound.

        ``True`` (vacuously) when no unwind was ever needed. ``False``
        means real, live one-sided exposure remains right now and a
        human must flatten it manually -- see ``unwind_failed_venues``
        for which venue(s) and why.
        """
        return not self.unwind_failed_venues


async def execute_allocation_plan(
    plan: AllocationPlan,
    venues: Sequence[SmartOrderRouterVenue],
    symbol: str,
    *,
    asset: UniversalAsset,
    algo_id: str,
    regulatory_client_shortcode: str,
    client_id: str | None = None,
    shadow: bool = False,
    settings: AppSettings | None = None,
    redis_client: RedisClient | None = None,
    auto_unwind: bool = True,
) -> SmartOrderRouterResult:
    """Place (or, under MOCK_MODE/``shadow``, simulate) one real child order per venue in ``plan``, concurrently.

    See the module docstring for the full safety-machinery integration.
    One venue's failure (including a tripped Kill Switch) never cancels
    the others -- ``asyncio.gather(..., return_exceptions=True)``, real
    per-venue accounting in the returned :class:`SmartOrderRouterResult`.

    Args:
        plan: A real :class:`AllocationPlan` from :func:`build_allocation_plan`.
        venues: The same (or a superset of the) venues used to build
            ``plan`` -- looked up here by ``broker_id``.
        symbol: The instrument being routed.
        asset: Real :class:`~core.nothierly_ontology.UniversalAsset`
            description, passed through to every resulting
            :class:`~core.nothierly_ontology.UniversalOrder`.
        algo_id: RTS 6-style algorithm identifier -- required, never
            defaulted, same contract as :func:`~core.nothierly_ontology.
            order_from_ccxt`.
        regulatory_client_shortcode: RTS 22-style short code -- same
            contract.
        client_id: The B2B client this routed order belongs to, if any.
        shadow: Force simulated fills even outside ``MOCK_MODE`` --
            mirrors ``api_middleware/gateway.py``'s ``ExecutionMode.
            SHADOW`` (real, self-contained simulation, no network call
            to any broker).
        settings: Defaults to ``get_settings()``.
        redis_client: If given, this real routing decision is persisted
            via :func:`log_routing_decision` for the SOR Radar dashboard
            page -- omit to route without persisting (e.g. from a test).
        auto_unwind: When a partial-venue failure leaves some legs
            filled and others failed, attempt a real opposite-side
            unwind on every filled leg (see the module docstring's own
            "Leg-risk auto-unwind" section for the one real limitation).
            Never attempted when ``simulate`` is true -- there is no
            real exposure to flatten. Default ``True``: leaving a real
            partial fill unhedged by default is the actual gap this
            parameter closes.
    """
    settings = settings or get_settings()
    venues_by_id = {venue.broker_id: venue for venue in venues}
    decision_timestamp_utc = datetime.now(timezone.utc)
    simulate = settings.mock_mode or shadow

    async def _execute_one(allocation: VenueAllocation) -> UniversalOrder:
        venue = venues_by_id[allocation.broker_id]
        side_literal: Literal["buy", "sell"] = "buy" if plan.side == OrderSide.BUY else "sell"
        client_order_id = f"sor-{uuid.uuid4()}"

        if simulate:
            _log.info(
                "smart_order_router_simulated_child_order",
                broker_id=allocation.broker_id,
                symbol=symbol,
                side=side_literal,
                quantity=allocation.quantity,
                mock_mode=settings.mock_mode,
                shadow=shadow,
            )
            raw: dict[str, Any] = {
                "id": client_order_id,
                "clientOrderId": client_order_id,
                "status": "closed",
                "side": side_literal,
                "type": "market",
                "amount": allocation.quantity,
                "filled": allocation.quantity,
                "average": allocation.effective_price,
            }
        else:
            raw = await safe_financial_call(
                venue.place_market_order,
                symbol,
                side_literal,
                allocation.quantity,
                client_order_id=client_order_id,
                retry_on_failure=False,
            )

        return order_from_ccxt(
            raw,
            broker_id=allocation.broker_id,
            asset=asset,
            algo_id=algo_id,
            decision_timestamp_utc=decision_timestamp_utc,
            regulatory_client_shortcode=regulatory_client_shortcode,
            client_id=client_id,
        )

    results = await asyncio.gather(
        *(_execute_one(allocation) for allocation in plan.allocations), return_exceptions=True
    )

    orders: list[UniversalOrder] = []
    failed_venues: list[tuple[str, str]] = []
    for allocation, result in zip(plan.allocations, results):
        if isinstance(result, KillSwitchActiveError):
            _log.warning("smart_order_router_child_order_blocked_by_kill_switch", broker_id=allocation.broker_id)
            failed_venues.append((allocation.broker_id, "Kill Switch actif"))
        elif isinstance(result, FinancialCallAbortedError):
            _log.error("smart_order_router_child_order_aborted", broker_id=allocation.broker_id, error=str(result))
            failed_venues.append((allocation.broker_id, str(result)))
        elif isinstance(result, BaseException):
            _log.error("smart_order_router_child_order_failed", broker_id=allocation.broker_id, error=str(result))
            failed_venues.append((allocation.broker_id, str(result)))
        else:
            orders.append(result)

    if redis_client is not None:
        await log_routing_decision(redis_client, plan, symbol)

    unwind_orders: list[UniversalOrder] = []
    unwind_failed_venues: list[tuple[str, str]] = []
    if auto_unwind and not simulate and orders and failed_venues:
        unwind_side_literal: Literal["buy", "sell"] = "sell" if plan.side == OrderSide.BUY else "buy"

        async def _unwind_one(order: UniversalOrder) -> UniversalOrder:
            venue = venues_by_id[order.broker_id]
            client_order_id = f"sor-unwind-{uuid.uuid4()}"
            # Deliberately no safe_financial_call/retry here: an unwind is itself a best-effort emergency
            # action (see module docstring) -- retrying a *second* real market order to close an exposure
            # that may have already partially closed on a first, ambiguously-failed attempt risks
            # overshooting into the opposite exposure, a worse outcome than leaving the original alone.
            raw = await venue.place_market_order(
                symbol, unwind_side_literal, float(order.filled_quantity), client_order_id=client_order_id,
            )
            return order_from_ccxt(
                raw, broker_id=order.broker_id, asset=asset, algo_id=algo_id,
                decision_timestamp_utc=decision_timestamp_utc, regulatory_client_shortcode=regulatory_client_shortcode,
                client_id=client_id,
            )

        unwind_results = await asyncio.gather(*(_unwind_one(order) for order in orders), return_exceptions=True)
        for order, result in zip(orders, unwind_results):
            if isinstance(result, KillSwitchActiveError):
                _log.critical(
                    "smart_order_router_unwind_blocked_by_kill_switch",
                    broker_id=order.broker_id, quantity=str(order.filled_quantity),
                )
                unwind_failed_venues.append((order.broker_id, "Kill Switch actif -- exposition NON dénouée, intervention manuelle requise"))
            elif isinstance(result, BaseException):
                _log.critical("smart_order_router_unwind_failed", broker_id=order.broker_id, error=str(result))
                unwind_failed_venues.append((order.broker_id, str(result)))
            else:
                unwind_orders.append(result)

        if unwind_failed_venues:
            _log.critical(
                "smart_order_router_leg_risk_unresolved",
                unresolved_venues=[b for b, _r in unwind_failed_venues],
                failed_original_venues=[b for b, _r in failed_venues],
            )
        else:
            _log.warning(
                "smart_order_router_leg_risk_auto_unwound",
                unwound_venues=[o.broker_id for o in unwind_orders],
                failed_original_venues=[b for b, _r in failed_venues],
            )

    return SmartOrderRouterResult(
        orders=tuple(orders), failed_venues=tuple(failed_venues),
        unwind_orders=tuple(unwind_orders), unwind_failed_venues=tuple(unwind_failed_venues),
    )

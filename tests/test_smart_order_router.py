"""Real, executable tests for src/execution/smart_order_router.py.

No mocking framework: :class:`_FakeVenue` is a real, minimal stand-in
satisfying :class:`~src.execution.smart_order_router.SmartOrderRouterVenue`
structurally (same "real fake, not a mock" discipline as ``tests/
test_broker_gateway.py``'s ``_FakeBrokerThatFailsToConnect``) -- its
``place_market_order`` is genuinely ``@requires_system_active``-decorated,
the same way the real ``AsyncExchangeClient.create_market_order`` it
stands in for actually is (see the module docstring's Kill Switch
paragraph), so the Kill Switch tests below exercise the real decorator,
not an assumption about what it does. The global Kill Switch itself is
backed by a real in-memory ``fakeredis`` server (fresh ``FakeServer()``
per test, same discipline as ``tests/test_alert_agent.py``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Generator, Literal

import fakeredis.aioredis
import pytest

from core.config import AppSettings, RedisSettings
from core.nothierly_ontology import AssetClass, OrderSide, UniversalAsset
from core.redis_client import RedisClient
from core.schemas import SOR_ROUTING_LOG_REDIS_KEY, OrderBookLevel, OrderBookSnapshot
from core.security import requires_system_active, trip_kill_switch
from src.execution.smart_order_router import (
    AllocationPlan,
    VenueAllocation,
    _effective_price,
    build_allocation_plan,
    execute_allocation_plan,
    log_routing_decision,
)

_ASSET = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)


class _FakeVenue:
    """A real, minimal SmartOrderRouterVenue -- see module docstring."""

    def __init__(
        self, broker_id: str, *, bids: list[tuple[float, float]], asks: list[tuple[float, float]],
        fee_rate: float, quote_error: BaseException | None = None, place_error: BaseException | None = None,
    ) -> None:
        self.broker_id = broker_id
        self._book = OrderBookSnapshot(
            symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
            asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
        )
        self._fee_rate = fee_rate
        self._quote_error = quote_error
        self._place_error = place_error
        self.placed_orders: list[tuple[str, str, float, str | None]] = []

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        if self._quote_error is not None:
            raise self._quote_error
        return self._book

    def get_taker_fee_rate(self, symbol: str) -> float:
        return self._fee_rate

    @requires_system_active
    async def place_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        if self._place_error is not None:
            raise self._place_error
        self.placed_orders.append((symbol, side, quantity, client_order_id))
        price = self._book.asks[0].price if side == "buy" else self._book.bids[0].price
        return {
            "id": client_order_id, "clientOrderId": client_order_id, "status": "closed",
            "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": price,
        }


@pytest.fixture
def killswitch_redis() -> Generator[RedisClient, None, None]:
    """Installs a real fakeredis-backed RedisClient as the process singleton -- see module docstring."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


# =============================================================================
# _effective_price -- the real fee-inclusive cost/proceeds math
# =============================================================================


def test_effective_price_buy_adds_fee_on_top() -> None:
    assert _effective_price(100.0, 0.01, OrderSide.BUY) == pytest.approx(101.0)


def test_effective_price_sell_deducts_fee() -> None:
    assert _effective_price(100.0, 0.01, OrderSide.SELL) == pytest.approx(99.0)


# =============================================================================
# build_allocation_plan -- real, fee-aware water-filling
# =============================================================================


@pytest.mark.asyncio
async def test_fee_inclusive_ranking_can_reverse_raw_price_ordering() -> None:
    """The real point of this router: a venue with a WORSE raw price can still win on effective (fee-inclusive) price."""
    cheap_raw_expensive_fee = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(99.0, 10.0)], fee_rate=0.02)  # effective 100.98
    expensive_raw_cheap_fee = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 10.0)], fee_rate=0.0)  # effective 100.0

    plan = await build_allocation_plan(
        [cheap_raw_expensive_fee, expensive_raw_cheap_fee], "BTC/USDT", OrderSide.BUY, Decimal("5")
    )

    assert len(plan.allocations) == 1
    assert plan.allocations[0].broker_id == "venue_a"
    assert plan.allocations[0].effective_price == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_water_filling_spills_into_second_venue_once_first_is_exhausted() -> None:
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 3.0)], fee_rate=0.0)
    venue_b = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 10.0)], fee_rate=0.0)

    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.BUY, Decimal("5"))

    by_venue = {a.broker_id: a for a in plan.allocations}
    assert by_venue["venue_a"].quantity == pytest.approx(3.0)
    assert by_venue["venue_b"].quantity == pytest.approx(2.0)
    assert plan.allocated_quantity == pytest.approx(Decimal("5"))
    assert plan.fully_filled


@pytest.mark.asyncio
async def test_water_filling_within_one_venues_multiple_levels() -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 2.0), (100.5, 2.0), (101.0, 2.0)], fee_rate=0.0)

    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    assert len(plan.allocations) == 1
    assert plan.allocations[0].quantity == pytest.approx(3.0)
    # volume-weighted: 2 units @ 100.0 + 1 unit @ 100.5
    assert plan.allocations[0].effective_price == pytest.approx((2 * 100.0 + 1 * 100.5) / 3)


@pytest.mark.asyncio
async def test_partial_fill_is_honest_not_an_error_when_total_depth_is_insufficient() -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 2.0)], fee_rate=0.0)

    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("10"))

    assert plan.allocated_quantity == pytest.approx(Decimal("2"))
    assert not plan.fully_filled


def test_fully_filled_tolerates_real_floating_point_summation_noise() -> None:
    """Regression test: found live routing a real order across real Deribit/OKX books -- see AllocationPlan.fully_filled's own docstring.

    ``2.10865219 + 1.30199458`` -- two entirely realistic 8-decimal
    crypto quantities (this project's own real crypto ``lot_size``,
    e.g. ``core.nothierly_ontology``'s ``Decimal("0.00000001")``) --
    lands one float64 ULP below their exact decimal sum
    (``3.4106467699999996`` vs. ``3.41064677``), reproducing the exact
    live-observed shortfall class without relying on any particular
    summation algorithm's rounding behavior.
    """
    quantity_a, quantity_b = 2.10865219, 1.30199458
    assert quantity_a + quantity_b < 3.41064677  # the exact real-world float behavior this guards against

    allocations = (
        VenueAllocation(broker_id="venue_a", quantity=quantity_a, effective_price=100.0),
        VenueAllocation(broker_id="venue_b", quantity=quantity_b, effective_price=100.0),
    )
    plan = AllocationPlan(side=OrderSide.BUY, requested_quantity=Decimal("3.41064677"), allocations=allocations)

    assert plan.fully_filled


def test_fully_filled_still_reports_a_real_shortfall() -> None:
    """The epsilon tolerance must not paper over an actual, real partial fill."""
    allocations = (VenueAllocation(broker_id="venue_a", quantity=0.5, effective_price=100.0),)
    plan = AllocationPlan(side=OrderSide.BUY, requested_quantity=Decimal("1"), allocations=allocations)

    assert not plan.fully_filled


@pytest.mark.asyncio
async def test_a_venue_whose_quote_fails_is_skipped_not_fatal() -> None:
    healthy = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 10.0)], fee_rate=0.0)
    broken = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(50.0, 10.0)], fee_rate=0.0, quote_error=ConnectionError("timeout"))

    plan = await build_allocation_plan([healthy, broken], "BTC/USDT", OrderSide.BUY, Decimal("1"))

    assert len(plan.allocations) == 1
    assert plan.allocations[0].broker_id == "venue_a"


@pytest.mark.asyncio
async def test_sell_side_ranks_by_best_fee_inclusive_proceeds_descending() -> None:
    venue_a = _FakeVenue("venue_a", bids=[(100.0, 5.0)], asks=[(1000.0, 1.0)], fee_rate=0.0)  # proceeds 100.0
    venue_b = _FakeVenue("venue_b", bids=[(102.0, 5.0)], asks=[(1000.0, 1.0)], fee_rate=0.05)  # proceeds 96.9

    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.SELL, Decimal("1"))

    assert plan.allocations[0].broker_id == "venue_a"  # better real proceeds despite a lower raw bid


# =============================================================================
# execute_allocation_plan -- safety-machinery integration
# =============================================================================


@pytest.mark.asyncio
async def test_mock_mode_never_calls_the_real_broker() -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 10.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    result = await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=True),
    )

    assert venue.placed_orders == []  # real place_market_order was never called
    assert len(result.orders) == 1
    assert result.orders[0].filled_quantity == pytest.approx(Decimal("3"))
    assert result.orders[0].broker_id == "venue_a"
    assert result.total_filled_quantity == pytest.approx(Decimal("3"))


@pytest.mark.asyncio
async def test_shadow_forces_simulation_even_outside_mock_mode() -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 10.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("2"))

    result = await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        shadow=True, settings=AppSettings(mock_mode=False),
    )

    assert venue.placed_orders == []
    assert len(result.orders) == 1


@pytest.mark.asyncio
async def test_real_mode_calls_place_market_order_with_correct_args(killswitch_redis: RedisClient) -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 10.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("4"))

    result = await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False),
    )

    assert len(venue.placed_orders) == 1
    symbol, side, quantity, client_order_id = venue.placed_orders[0]
    assert symbol == "BTC/USDT"
    assert side == "buy"
    assert quantity == pytest.approx(4.0)
    assert client_order_id is not None
    assert len(result.orders) == 1
    assert result.orders[0].algo_id == "TEST_SOR"
    assert result.orders[0].regulatory_client_shortcode == "RC-1"
    assert result.orders[0].side == OrderSide.BUY


@pytest.mark.asyncio
async def test_pre_tripped_kill_switch_blocks_every_venue(killswitch_redis: RedisClient) -> None:
    await trip_kill_switch("test: simulated PANIC")
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    venue_b = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 5.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.BUY, Decimal("8"))

    result = await execute_allocation_plan(
        plan, [venue_a, venue_b], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False),
    )

    assert result.orders == ()
    assert venue_a.placed_orders == []
    assert venue_b.placed_orders == []
    assert {broker_id for broker_id, _reason in result.failed_venues} == {"venue_a", "venue_b"}
    assert all(reason == "Kill Switch actif" for _broker_id, reason in result.failed_venues)


@pytest.mark.asyncio
async def test_generic_placement_failure_is_reported_and_trips_the_global_kill_switch(killswitch_redis: RedisClient) -> None:
    """A single venue's real, exhausted placement failure -- see safe_financial_call's own docstring: this is by design, not a bug.

    Deliberately single-venue: safe_financial_call treats any exhausted
    financial-call failure as systemic and trips the GLOBAL Kill Switch
    (see its own docstring), so a second, healthy venue placed
    concurrently in the same real (non-MOCK_MODE) batch could itself get
    blocked depending on scheduling -- a real, intentional cross-cutting
    safety behavior, not something to paper over with a two-venue
    "isolation" test that would just be racy. See the MOCK_MODE test
    above for the real, deterministic "multiple venues each fill
    independently" case.
    """
    from core.security import is_system_safe

    failing = _FakeVenue(
        "venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 5.0)], fee_rate=0.0, place_error=ConnectionError("simulated exchange outage")
    )
    plan = await build_allocation_plan([failing], "BTC/USDT", OrderSide.BUY, Decimal("5"))

    result = await execute_allocation_plan(
        plan, [failing], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False),
    )

    assert result.orders == ()
    assert len(result.failed_venues) == 1
    broker_id, reason = result.failed_venues[0]
    assert broker_id == "venue_b"
    assert "place_market_order" in reason
    assert not await is_system_safe()  # the real, documented safe_financial_call side effect


@pytest.mark.asyncio
async def test_multiple_simulated_venues_each_fill_independently_under_mock_mode() -> None:
    """The real, deterministic version of "multiple venues, no cross-cancellation" -- MOCK_MODE never touches safe_financial_call/the Kill Switch at all."""
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    venue_b = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 10.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.BUY, Decimal("8"))

    result = await execute_allocation_plan(
        plan, [venue_a, venue_b], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=True),
    )

    assert len(result.orders) == 2
    assert result.failed_venues == ()
    assert {o.broker_id for o in result.orders} == {"venue_a", "venue_b"}


# =============================================================================
# execute_allocation_plan -- auto_unwind (leg-risk), and its one real limitation
# =============================================================================


@pytest.mark.asyncio
async def test_auto_unwind_is_attempted_but_blocked_by_the_kill_switch_it_did_not_cause(killswitch_redis: RedisClient) -> None:
    """The real, current-architecture outcome (see smart_order_router.py's own module docstring): venue_b's
    exhausted failure goes through safe_financial_call, which UNCONDITIONALLY trips the global Kill Switch
    on any exhausted failure -- so venue_a's real unwind attempt, which also goes through the same
    @requires_system_active-decorated place_market_order, is itself blocked by the very switch venue_b's
    own failure just tripped. This is the honest, reproducible outcome today, not a test bug."""
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    venue_b = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 5.0)], fee_rate=0.0, place_error=ConnectionError("simulated exchange outage"))
    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.BUY, Decimal("6"))

    result = await execute_allocation_plan(
        plan, [venue_a, venue_b], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False),
    )

    assert len(result.orders) == 1 and result.orders[0].broker_id == "venue_a"  # the real, unflattened leg
    assert {b for b, _r in result.failed_venues} == {"venue_b"}
    assert result.unwind_orders == ()  # the unwind attempt did NOT succeed
    assert len(result.unwind_failed_venues) == 1
    broker_id, reason = result.unwind_failed_venues[0]
    assert broker_id == "venue_a"
    assert "Kill Switch" in reason
    assert result.leg_risk_fully_flattened is False  # the honest, correct verdict: a human must act now


@pytest.mark.asyncio
async def test_auto_unwind_false_preserves_the_original_unhedged_behavior(killswitch_redis: RedisClient) -> None:
    """auto_unwind=False is the exact prior behavior (no unwind attempt at all) -- for a caller that wants
    to handle leg-risk itself, or a test isolating build/execute from the unwind step."""
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    venue_b = _FakeVenue("venue_b", bids=[(1.0, 1.0)], asks=[(101.0, 5.0)], fee_rate=0.0, place_error=ConnectionError("simulated exchange outage"))
    plan = await build_allocation_plan([venue_a, venue_b], "BTC/USDT", OrderSide.BUY, Decimal("6"))

    result = await execute_allocation_plan(
        plan, [venue_a, venue_b], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False), auto_unwind=False,
    )

    assert len(result.orders) == 1 and result.orders[0].broker_id == "venue_a"
    assert result.unwind_orders == ()
    assert result.unwind_failed_venues == ()  # no attempt was made at all -- distinct from "attempted and blocked"
    assert result.leg_risk_fully_flattened is True  # vacuously true: this call never looked for leg risk


@pytest.mark.asyncio
async def test_auto_unwind_never_attempted_when_every_venue_already_failed(killswitch_redis: RedisClient) -> None:
    """No successful leg exists to flatten -- auto_unwind must be a no-op, not an error."""
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0, place_error=ConnectionError("outage"))
    plan = await build_allocation_plan([venue_a], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    result = await execute_allocation_plan(
        plan, [venue_a], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=False),
    )

    assert result.orders == ()
    assert result.unwind_orders == ()
    assert result.unwind_failed_venues == ()
    assert result.leg_risk_fully_flattened is True  # vacuous: nothing filled, nothing to flatten


@pytest.mark.asyncio
async def test_auto_unwind_never_attempted_under_mock_mode() -> None:
    """No real exposure exists to flatten under a simulated fill -- auto_unwind must stay inert."""
    venue_a = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue_a], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    result = await execute_allocation_plan(
        plan, [venue_a], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=True),
    )

    assert venue_a.placed_orders == []  # confirms no unwind call slipped through despite auto_unwind's default True
    assert result.unwind_orders == ()
    assert result.unwind_failed_venues == ()


# =============================================================================
# savings_vs_worst_level_usd -- real, honest, always-non-negative savings metric
# =============================================================================


def test_savings_is_zero_for_a_single_level_plan() -> None:
    """Nothing to have saved against when only one real level/venue was used."""
    plan = AllocationPlan(
        side=OrderSide.BUY, requested_quantity=Decimal("1"),
        allocations=(VenueAllocation(broker_id="venue_a", quantity=1.0, effective_price=100.0),),
    )
    assert plan.savings_vs_worst_level_usd == pytest.approx(0.0)


def test_savings_is_positive_when_water_filling_used_a_better_level_for_buy() -> None:
    """2 units @ 100 + 1 unit @ 102 real actual cost vs. paying the worst level (102) for all 3 -- water-filling must show a real, positive saving."""
    plan = AllocationPlan(
        side=OrderSide.BUY, requested_quantity=Decimal("3"),
        allocations=(
            VenueAllocation(broker_id="venue_a", quantity=2.0, effective_price=100.0),
            VenueAllocation(broker_id="venue_b", quantity=1.0, effective_price=102.0),
        ),
    )
    naive_cost = 3.0 * 102.0
    actual_cost = 2.0 * 100.0 + 1.0 * 102.0
    assert plan.savings_vs_worst_level_usd == pytest.approx(naive_cost - actual_cost)
    assert plan.savings_vs_worst_level_usd > 0.0


def test_savings_is_positive_for_sell_too() -> None:
    """For a SELL, the worst level is the LOWEST proceeds -- water-filling receiving more overall must show a positive saving."""
    plan = AllocationPlan(
        side=OrderSide.SELL, requested_quantity=Decimal("3"),
        allocations=(
            VenueAllocation(broker_id="venue_a", quantity=2.0, effective_price=100.0),
            VenueAllocation(broker_id="venue_b", quantity=1.0, effective_price=98.0),
        ),
    )
    naive_proceeds = 3.0 * 98.0
    actual_proceeds = 2.0 * 100.0 + 1.0 * 98.0
    assert plan.savings_vs_worst_level_usd == pytest.approx(actual_proceeds - naive_proceeds)
    assert plan.savings_vs_worst_level_usd > 0.0


def test_savings_is_zero_for_an_empty_plan() -> None:
    plan = AllocationPlan(side=OrderSide.BUY, requested_quantity=Decimal("1"), allocations=())
    assert plan.savings_vs_worst_level_usd == pytest.approx(0.0)


# =============================================================================
# log_routing_decision / execute_allocation_plan(redis_client=...) -- real persistence
# =============================================================================


@pytest.mark.asyncio
async def test_log_routing_decision_persists_a_real_entry(killswitch_redis: RedisClient) -> None:
    plan = AllocationPlan(
        side=OrderSide.BUY, requested_quantity=Decimal("3"),
        allocations=(
            VenueAllocation(broker_id="venue_a", quantity=2.0, effective_price=100.0),
            VenueAllocation(broker_id="venue_b", quantity=1.0, effective_price=102.0),
        ),
    )

    await log_routing_decision(killswitch_redis, plan, "BTC/USDT")

    raw_entries = await killswitch_redis.execute("lrange", SOR_ROUTING_LOG_REDIS_KEY, 0, -1)
    assert len(raw_entries) == 1
    entry = json.loads(raw_entries[0])
    assert entry["symbol"] == "BTC/USDT"
    assert entry["side"] == "BUY"
    assert entry["savings_usd"] == pytest.approx(plan.savings_vs_worst_level_usd)
    assert len(entry["allocations"]) == 2


@pytest.mark.asyncio
async def test_execute_allocation_plan_logs_a_real_routing_decision_when_given_a_redis_client(
    killswitch_redis: RedisClient,
) -> None:
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=True), redis_client=killswitch_redis,
    )

    raw_entries = await killswitch_redis.execute("lrange", SOR_ROUTING_LOG_REDIS_KEY, 0, -1)
    assert len(raw_entries) == 1


@pytest.mark.asyncio
async def test_execute_allocation_plan_never_logs_without_a_redis_client(killswitch_redis: RedisClient) -> None:
    """The default (redis_client=None) must not silently persist -- explicit opt-in only."""
    venue = _FakeVenue("venue_a", bids=[(1.0, 1.0)], asks=[(100.0, 5.0)], fee_rate=0.0)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("3"))

    await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="TEST_SOR", regulatory_client_shortcode="RC-1",
        settings=AppSettings(mock_mode=True),
    )

    raw_entries = await killswitch_redis.execute("lrange", SOR_ROUTING_LOG_REDIS_KEY, 0, -1)
    assert raw_entries == []

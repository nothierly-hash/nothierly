"""Real, executable tests for src/agents/macro_order_execution/agent.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test), same
discipline as ``tests/test_shadow_crossing_engine.py``. Real, minimal
venues are a real ``SmartOrderRouterVenue`` stand-in (same "real fake,
not a mock" discipline as ``tests/test_smart_order_router.py``'s own
``_FakeVenue``), injected directly via ``agent._venues`` -- this
agent's own ``on_start()`` (real ``BrokerFactory.create`` + real
Timescale connect) is never exercised here, matching ``tests/
test_eod_reconciliator.py``'s own "no real Timescale in test env"
discipline: ``agent._timescale_pool`` stays ``None`` throughout,
exercising the real, already-built "Timescale unavailable" degrade
path. The real Redis blotter write and order-status update are fully
exercised and asserted on; Kill Switch integration itself is not
re-verified here (already covered by ``tests/test_smart_order_router.py``,
which this agent's routing/execution reuses unchanged).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Generator, Literal

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import AppSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import (
    CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX,
    CLIENT_ORDERS_TOPIC,
    FILLS_BLOTTER_REDIS_KEY,
    ORDER_STATUS_REDIS_KEY_PREFIX,
    TOXIC_ORDERS_EVADED_REDIS_KEY,
    OrderBookLevel,
    OrderBookSnapshot,
    MessagePayload,
)
from src.agents.macro_order_execution.agent import MacroOrderExecutionAgent


class _FakeVenue:
    """A real, minimal SmartOrderRouterVenue -- see module docstring."""

    def __init__(self, broker_id: str, *, bids: list[tuple[float, float]], asks: list[tuple[float, float]], fee_rate: float) -> None:
        self.broker_id = broker_id
        self._book = OrderBookSnapshot(
            symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
            asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
        )
        self._fee_rate = fee_rate
        self.placed_orders: list[tuple[str, str, float, str | None]] = []
        self.closed = False

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        return self._book

    def get_taker_fee_rate(self, symbol: str) -> float:
        return self._fee_rate

    async def place_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        self.placed_orders.append((symbol, side, quantity, client_order_id))
        price = self._book.asks[0].price if side == "buy" else self._book.bids[0].price
        return {
            "id": client_order_id, "clientOrderId": client_order_id, "status": "closed",
            "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": price,
        }

    async def emergency_flatten_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        # Same safe default as BaseBroker.emergency_flatten_market_order's own docstring: a venue
        # that hasn't deliberately opted into bypassing the Kill Switch just calls the normal,
        # fully-gated place_market_order.
        return await self.place_market_order(symbol, side, quantity, client_order_id=client_order_id)

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> Generator[MacroOrderExecutionAgent, None, None]:
    # Teardown closes the real publisher/subscriber/context BaseAgent.__init__ opened -- see
    # tests/test_execution_agent_instrument_lock.py's identical fixture for why this is not optional.
    # mock_mode=False: this dev environment's real .env defaults MOCK_MODE=true (the
    # established project-wide safety default -- see execute_allocation_plan's own
    # docstring), which would silently simulate every fill regardless of this
    # agent's own AgentControlState, making every test below pass for the wrong
    # reason. Explicit like tests/test_smart_order_router.py's own real-execution
    # tests, so `venue.placed_orders` genuinely distinguishes LIVE from SHADOW/OFF.
    instance = MacroOrderExecutionAgent(
        settings=AppSettings(mock_mode=False), manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def _order_message(
    *, order_id: str, client_id: str, asset: str, side: str, quantity: float,
    execution_mode: str = "LIVE", execution_algo: str = "AI_TWAP",
) -> MessagePayload:
    return MessagePayload(
        topic=CLIENT_ORDERS_TOPIC, sender_id="test-gateway",
        data={
            "order_id": order_id, "client_id": client_id, "asset": asset,
            "total_quantity": quantity, "side": side, "execution_algo": execution_algo,
            "execution_mode": execution_mode, "time_window_minutes": 60,
        },
    )


async def _seed_order_status(redis_client: RedisClient, order_id: str, *, client_id: str, execution_mode: str = "LIVE") -> None:
    record = {
        "order_id": order_id, "status": "ACCEPTED", "client_id": client_id,
        "aggression_level": "PASSIVE", "updated_at": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
    }
    await redis_client.execute("set", f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id}", json.dumps(record))


async def _read_order_status(redis_client: RedisClient, order_id: str) -> dict[str, Any] | None:
    raw = await redis_client.execute("get", f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id}")
    return json.loads(raw) if raw is not None else None


async def _read_blotter(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw_entries = await redis_client.execute("lrange", FILLS_BLOTTER_REDIS_KEY, 0, -1)
    return [json.loads(e) for e in raw_entries]


@pytest.mark.asyncio
async def test_live_order_routes_and_really_executes_across_a_real_fake_venue(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.1)  # let the background _execute_live_order task resolve

    assert len(venue.placed_orders) == 1
    assert venue.placed_orders[0][:3] == ("BTC/USDT", "buy", 1.0)

    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "COMPLETED"
    assert status["client_id"] == "client-a"  # preserved from the original record

    blotter = await _read_blotter(redis_client)
    assert len(blotter) == 1
    assert blotter[0]["client_id"] == "client-a"
    assert blotter[0]["order_id"] == "o1"
    assert blotter[0]["instrument"] == "BTC/USDT"
    assert blotter[0]["action"] == "ENTRY_LONG"
    assert blotter[0]["units"] == pytest.approx(1.0)
    assert blotter[0]["agent_control_shadow"] is False


@pytest.mark.asyncio
async def test_off_control_state_refuses_the_order_and_marks_it_failed(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    agent._control_state = AgentControlState.OFF
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)

    assert venue.placed_orders == []
    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "FAILED"
    assert await _read_blotter(redis_client) == []


@pytest.mark.asyncio
async def test_shadow_control_state_never_calls_the_real_broker_but_still_logs_a_fill(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    agent._control_state = AgentControlState.SHADOW
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.1)

    assert venue.placed_orders == []  # never a real call -- see AgentControlState.SHADOW semantics in module docstring

    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "COMPLETED"

    blotter = await _read_blotter(redis_client)
    assert len(blotter) == 1
    assert blotter[0]["agent_control_shadow"] is True


@pytest.mark.asyncio
async def test_a_non_order_message_on_the_same_topic_is_ignored(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    """CLIENT_ORDERS_TOPIC also carries gateway.py's own unrelated AGGRESSION_ADJUSTED event shape -- must never be misread as an order."""
    message = MessagePayload(
        topic=CLIENT_ORDERS_TOPIC, sender_id="test-gateway",
        data={"order_id": "o1", "event": "AGGRESSION_ADJUSTED", "aggression_level": "AGGRESSIVE"},
    )
    await agent._handle_client_order_message(message)
    await asyncio.sleep(0.05)
    assert await _read_blotter(redis_client) == []


@pytest.mark.asyncio
async def test_a_shadow_execution_mode_order_is_ignored_not_double_processed(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    """SHADOW-mode client orders are already fully, synchronously handled by api_middleware/gateway.py -- this agent must never touch them."""
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a", execution_mode="SHADOW")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0, execution_mode="SHADOW")
    )
    await asyncio.sleep(0.05)

    assert venue.placed_orders == []
    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "ACCEPTED"  # never touched by this agent


@pytest.mark.asyncio
async def test_non_crypto_asset_has_no_real_router_yet_and_is_marked_failed(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="EUR_USD", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)

    assert venue.placed_orders == []
    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "FAILED"


@pytest.mark.asyncio
async def test_no_reachable_venue_is_marked_failed(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    agent._venues = []
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)

    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "FAILED"


@pytest.mark.asyncio
async def test_insufficient_real_liquidity_yields_a_real_partial_fill(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    """Real, honest partial fill: the venue's real depth is smaller than the requested quantity."""
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 0.3)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.1)

    assert venue.placed_orders[0][2] == pytest.approx(0.3)
    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "PARTIALLY_FILLED"


@pytest.mark.asyncio
async def test_malformed_order_message_is_skipped_without_crashing(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    message = MessagePayload(
        topic=CLIENT_ORDERS_TOPIC, sender_id="test-gateway",
        data={
            "order_id": "o1", "client_id": "client-a", "asset": "BTC/USDT",
            "total_quantity": 1.0, "side": "SIDEWAYS", "execution_algo": "AI_TWAP",
        },
    )
    await agent._handle_client_order_message(message)
    await asyncio.sleep(0.05)
    assert await _read_blotter(redis_client) == []


@pytest.mark.asyncio
async def test_update_order_status_preserves_every_other_field(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    await _seed_order_status(redis_client, "o1", client_id="client-a", execution_mode="LIVE")
    await agent._update_order_status("o1", status="COMPLETED")

    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "COMPLETED"
    assert status["client_id"] == "client-a"
    assert status["execution_mode"] == "LIVE"
    assert status["aggression_level"] == "PASSIVE"


@pytest.mark.asyncio
async def test_update_order_status_missing_record_does_not_crash(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    await agent._update_order_status("no-such-order", status="COMPLETED")
    assert await _read_order_status(redis_client, "no-such-order") is None


@pytest.mark.asyncio
async def test_on_shutdown_closes_every_connected_venue() -> None:
    instance = MacroOrderExecutionAgent(manage_own_signals=False, manage_own_redis=False)
    try:
        venue_a = _FakeVenue("binance", bids=[(1.0, 1.0)], asks=[(1.0, 1.0)], fee_rate=0.001)
        venue_b = _FakeVenue("okx", bids=[(1.0, 1.0)], asks=[(1.0, 1.0)], fee_rate=0.001)
        instance._venues = [venue_a, venue_b]

        await instance.on_shutdown()

        assert venue_a.closed is True
        assert venue_b.closed is True
    finally:
        instance._publisher.close()
        instance._subscriber.close()
        instance._zmq_context.term()


# -- TOXICITY_AVOIDANCE, real VPIN-fed AdverseFlowDefenseAgent --------------------------------
#
# agent._fetch_current_vpin is substituted directly (a real, minimal
# stand-in for one glue method, not the business logic under test) so
# these tests exercise the real posture/scaling/counter logic without
# needing to hand-construct 10+ real VPIN buckets' worth of trade tape
# -- that real VPIN math is already covered by
# tests/test_microstructure_features.py.


async def _toxic_counter(redis_client: RedisClient) -> int:
    raw = await redis_client.execute("get", TOXIC_ORDERS_EVADED_REDIS_KEY)
    return int(raw) if raw is not None else 0


@pytest.mark.asyncio
async def test_toxicity_avoidance_reduces_quantity_under_high_vpin(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    """Real CAUTIOUS posture (VPIN >= 0.7, no primary detector signal fed) reduces size to the real 30% multiplier."""
    agent._fetch_current_vpin = lambda symbol: _async_return(0.9)  # type: ignore[method-assign]
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(
            order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0,
            execution_algo="TOXICITY_AVOIDANCE",
        )
    )
    await asyncio.sleep(0.1)

    assert len(venue.placed_orders) == 1
    assert venue.placed_orders[0][2] == pytest.approx(0.3)  # 1.0 * real 0.3 multiplier
    assert await _toxic_counter(redis_client) == 1


@pytest.mark.asyncio
async def test_toxicity_avoidance_with_low_vpin_executes_full_quantity(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    agent._fetch_current_vpin = lambda symbol: _async_return(0.1)  # type: ignore[method-assign]
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(
            order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0,
            execution_algo="TOXICITY_AVOIDANCE",
        )
    )
    await asyncio.sleep(0.1)

    assert venue.placed_orders[0][2] == pytest.approx(1.0)
    assert await _toxic_counter(redis_client) == 0


@pytest.mark.asyncio
async def test_toxicity_check_never_applies_to_other_algos(
    agent: MacroOrderExecutionAgent, redis_client: RedisClient
) -> None:
    """A client who didn't ask for TOXICITY_AVOIDANCE never has this behavior silently applied."""
    agent._fetch_current_vpin = lambda symbol: _async_return(0.9)  # type: ignore[method-assign]
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")

    await agent._handle_client_order_message(
        _order_message(
            order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0,
            execution_algo="AI_TWAP",
        )
    )
    await asyncio.sleep(0.1)

    assert venue.placed_orders[0][2] == pytest.approx(1.0)
    assert await _toxic_counter(redis_client) == 0


async def _async_return(value: float) -> float:
    return value


# -- Real cancellation check (api_middleware/gateway.py's cancel_order) --------------------------------


@pytest.mark.asyncio
async def test_is_cancelled_false_when_no_marker_set(agent: MacroOrderExecutionAgent) -> None:
    assert await agent._is_cancelled("o1") is False


@pytest.mark.asyncio
async def test_is_cancelled_true_when_gateway_marker_is_set(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", f"{CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX}o1", "1")
    assert await agent._is_cancelled("o1") is True


@pytest.mark.asyncio
async def test_a_cancelled_order_is_never_executed(agent: MacroOrderExecutionAgent, redis_client: RedisClient) -> None:
    """The real marker api_middleware/gateway.py's cancel_order sets must stop this agent before any routing."""
    venue = _FakeVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.001)
    agent._venues = [venue]
    await _seed_order_status(redis_client, "o1", client_id="client-a")
    await redis_client.execute("set", f"{CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX}o1", "1")

    await agent._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)

    assert venue.placed_orders == []
    status = await _read_order_status(redis_client, "o1")
    assert status is not None
    assert status["status"] == "ACCEPTED"  # left exactly as cancel_order set it -- this agent never touches it

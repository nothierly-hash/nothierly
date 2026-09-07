"""Real, executable tests for src/execution/shadow_crossing_engine.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test). The real,
already-tested :class:`~core.crossing_network.CrossingNetwork` itself
is reused unchanged here (its own matching/self-trade-prevention/
no-fabricated-price behavior is exhaustively covered by ``tests/
test_crossing_network.py`` -- not re-verified here); these tests
instead cover this module's *own* added logic: filtering the real
``CLIENT_ORDERS_TOPIC`` message shapes, ``AgentControlState.OFF``
suppression, real Redis persistence, and the real fee-saved
computation. ``matching_window_ms`` is set small in the test fixture
so these tests run in milliseconds, not the module's real, deliberately
long 30s default (see its own docstring for why that default is real
and intentional, not a bug).
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Generator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import CLIENT_ORDERS_TOPIC, DARK_POOL_SHADOW_MATCHES_REDIS_KEY, MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics, MessagePayload
from src.execution.shadow_crossing_engine import ShadowCrossingEngine

_TEST_MATCHING_WINDOW_MS = 200.0


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def engine(redis_client: RedisClient) -> ShadowCrossingEngine:
    instance = ShadowCrossingEngine(
        manage_own_signals=False, manage_own_redis=False, matching_window_ms=_TEST_MATCHING_WINDOW_MS
    )
    instance._redis_client = redis_client
    return instance


def _order_message(*, order_id: str, client_id: str, asset: str, side: str, quantity: float) -> MessagePayload:
    return MessagePayload(
        topic=CLIENT_ORDERS_TOPIC,
        sender_id="test-gateway",
        data={
            "order_id": order_id, "client_id": client_id, "asset": asset,
            "total_quantity": quantity, "side": side,
            "execution_algo": "AI_TWAP", "time_window_minutes": 60,
        },
    )


async def _seed_mid_price(redis_client: RedisClient, symbol: str, mid_price: float) -> None:
    metrics = MarketMetrics(symbol=symbol, spread=0.5, order_book_imbalance=0.0, mid_price=mid_price)
    await redis_client.execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}", metrics.model_dump_json())


@pytest.mark.asyncio
async def test_two_opposite_orders_produce_a_real_shadow_match(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "BTC/USDT", 65000.0)

    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)  # let both background _observe_order tasks resolve

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert len(entries) == 1
    entry = json.loads(entries[0])
    assert entry["instrument"] == "BTC/USDT"
    assert Decimal(entry["quantity"]) == Decimal("1.0")
    assert float(entry["price"]) == pytest.approx(65000.0)
    assert entry["fee_saved_usd"] == pytest.approx(1.0 * 65000.0 * 0.001 * 2)
    assert entry["buy_order_id"] == "o1"
    assert entry["sell_order_id"] == "o2"


@pytest.mark.asyncio
async def test_a_non_order_message_on_the_same_topic_is_ignored(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    """CLIENT_ORDERS_TOPIC also carries gateway.py's own unrelated AGGRESSION_ADJUSTED event shape -- must never be misread as an order."""
    message = MessagePayload(
        topic=CLIENT_ORDERS_TOPIC, sender_id="test-gateway",
        data={"order_id": "o1", "event": "AGGRESSION_ADJUSTED", "aggression_level": "AGGRESSIVE"},
    )
    await engine._handle_client_order_message(message)
    await asyncio.sleep(0.05)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_off_control_state_ignores_every_order(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "BTC/USDT", 65000.0)
    engine._control_state = AgentControlState.OFF

    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_no_live_mid_price_means_no_real_match_no_fabricated_price(
    engine: ShadowCrossingEngine, redis_client: RedisClient
) -> None:
    """No market_metrics:{symbol} seeded -- real CrossingNetwork gate refuses to cross at all (see its own docstring)."""
    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_get_live_mid_price_reads_the_real_redis_key(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "EUR_USD", 1.085)
    price = await engine._get_live_mid_price("EUR_USD")
    assert price == Decimal("1.085")


@pytest.mark.asyncio
async def test_get_live_mid_price_returns_none_when_absent(engine: ShadowCrossingEngine) -> None:
    assert await engine._get_live_mid_price("NO/SUCH") is None

"""Real, executable tests for src/agents/market_data/agent.py's real trade-tape persistence.

Real Redis via fakeredis (fresh ``FakeServer()`` per test), same
discipline as every other agent test this session. Does not exercise
the real ccxt.pro WebSocket connection at all -- ``_handle_trades_update``
is called directly with synthetic, ccxt-shaped trade dicts (the exact
real fields ``AsyncExchangeClient.watch_trades`` yields:
``timestamp``/``side``/``amount``/``price``), matching this project's
"test the handler, not the transport" convention already used for
:class:`~src.execution.shadow_crossing_engine.ShadowCrossingEngine`/
:class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent`'s own bus-message-handler tests.
"""

from __future__ import annotations

import json
from typing import Any, Generator

import fakeredis.aioredis
import pytest

import asyncio

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import HEARTBEAT_REDIS_KEY_PREFIX, TRADE_TAPE_MAX_ENTRIES, TRADE_TAPE_REDIS_KEY_PREFIX
from src.agents.market_data.agent import MarketDataAgent

_SYMBOL = "BTC/USDT"


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> Generator[MarketDataAgent, None, None]:
    """Closes the real publisher/subscriber/context BaseAgent.__init__ opened -- see
    tests/test_execution_agent_instrument_lock.py's identical fixture for why this is not optional."""
    instance = MarketDataAgent(_SYMBOL, manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _tape_entries(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw = await redis_client.execute("lrange", f"{TRADE_TAPE_REDIS_KEY_PREFIX}{_SYMBOL}", 0, -1)
    return [json.loads(entry) for entry in raw]


@pytest.mark.asyncio
async def test_real_trades_are_persisted_onto_the_real_tape(agent: MarketDataAgent, redis_client: RedisClient) -> None:
    trades = [
        {"timestamp": 1_700_000_000_000, "side": "buy", "amount": 0.5, "price": 65_000.0},
        {"timestamp": 1_700_000_000_500, "side": "sell", "amount": 0.2, "price": 64_999.5},
    ]
    await agent._handle_trades_update(trades)

    entries = await _tape_entries(redis_client)
    assert len(entries) == 2
    # LPUSH means newest-first: the second real trade in the batch is entries[0].
    assert entries[0]["side"] == "sell"
    assert entries[0]["amount"] == pytest.approx(0.2)
    assert entries[1]["side"] == "buy"
    assert entries[1]["price"] == pytest.approx(65_000.0)


@pytest.mark.asyncio
async def test_a_malformed_trade_missing_required_fields_is_skipped(agent: MarketDataAgent, redis_client: RedisClient) -> None:
    trades: list[dict[str, Any]] = [
        {"timestamp": 1_700_000_000_000, "side": "buy", "amount": 0.5, "price": 65_000.0},
        {"timestamp": None, "side": "buy", "amount": 1.0, "price": 65_000.0},  # real gap ccxt can yield mid-reconnect
        {"timestamp": 1_700_000_001_000, "side": "unknown", "amount": 1.0, "price": 65_000.0},  # not a real buy/sell
    ]
    await agent._handle_trades_update(trades)

    entries = await _tape_entries(redis_client)
    assert len(entries) == 1
    assert entries[0]["side"] == "buy"
    assert entries[0]["amount"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_tape_is_capped_at_the_real_max_entries(agent: MarketDataAgent, redis_client: RedisClient) -> None:
    trades = [
        {"timestamp": 1_700_000_000_000 + i, "side": "buy", "amount": 1.0, "price": 65_000.0}
        for i in range(TRADE_TAPE_MAX_ENTRIES + 50)
    ]
    await agent._handle_trades_update(trades)

    raw = await redis_client.execute("lrange", f"{TRADE_TAPE_REDIS_KEY_PREFIX}{_SYMBOL}", 0, -1)
    assert len(raw) == TRADE_TAPE_MAX_ENTRIES


@pytest.mark.asyncio
async def test_no_redis_client_does_not_crash(agent: MarketDataAgent) -> None:
    agent._redis_client = None
    await agent._handle_trades_update([{"timestamp": 1, "side": "buy", "amount": 1.0, "price": 1.0}])


@pytest.mark.asyncio
async def test_heartbeat_loop_writes_a_real_heartbeat_independent_of_the_stream_loops(
    agent: MarketDataAgent, redis_client: RedisClient
) -> None:
    """Real regression test for a real bug found live in this project's own 13.5h shadow run:
    MarketDataAgent.run() overrides BaseAgent.run() entirely with two WebSocket stream consumers
    and never called BaseAgent._write_heartbeat() -- a perfectly healthy agent (real, fresh
    trade-tape/market-metrics writes the whole time) looked permanently OFFLINE to every real
    heartbeat-based liveness check in this project (docker-compose.yml's own healthcheck,
    swarm_console.py's agent status). This asserts the real fix: a real, independent heartbeat
    loop, run concurrently with (not instead of) the two stream consumers.
    """
    heartbeat_task = asyncio.create_task(agent._heartbeat_loop())
    try:
        await asyncio.sleep(0.05)
        raw = await redis_client.execute("get", f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent.agent_id}")
        assert raw is not None
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task

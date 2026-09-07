"""Real, executable tests for the live-overridable max-position-size cap in src/agents/execution/agent.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test), same
discipline as ``tests/test_execution_agent_drawdown.py``.
"""

from __future__ import annotations

from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from src.agents.execution.agent import TRADING_MAX_POSITION_UNITS_REDIS_KEY, ExecutionAgent

_INSTRUMENT = "EUR_USD"


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> ExecutionAgent:
    instance = ExecutionAgent(
        {_INSTRUMENT}, capital=10_000.0,
        settings=AppSettings(mock_mode=True, guardian=GuardianSettings(llm_provider="none")),
        manage_own_signals=False, manage_own_redis=False,
    )
    instance._redis_client = redis_client
    return instance


# -- _get_dynamic_max_position_units --------------------------------


@pytest.mark.asyncio
async def test_no_override_is_uncapped(agent: ExecutionAgent) -> None:
    assert await agent._get_dynamic_max_position_units() is None


@pytest.mark.asyncio
async def test_valid_override_is_used_verbatim(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "2.5")
    assert await agent._get_dynamic_max_position_units() == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_zero_override_means_uncapped_not_zero_sized_orders(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """0 is the real dashboard's own 'illimité' sentinel -- must never zero every real order."""
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "0")
    assert await agent._get_dynamic_max_position_units() is None


@pytest.mark.asyncio
async def test_negative_override_is_ignored(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "-5")
    assert await agent._get_dynamic_max_position_units() is None


@pytest.mark.asyncio
async def test_malformed_override_is_ignored(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "not-a-number")
    assert await agent._get_dynamic_max_position_units() is None


@pytest.mark.asyncio
async def test_no_redis_client_is_uncapped(agent: ExecutionAgent) -> None:
    agent._redis_client = None
    assert await agent._get_dynamic_max_position_units() is None


# -- calculate_position_size, capped --------------------------------


@pytest.mark.asyncio
async def test_position_size_uncapped_by_default(agent: ExecutionAgent) -> None:
    """1% of 10,000 = 100$ risk / 0.001 SL distance = 100,000 units -- no cap configured, so it stays that large."""
    size = await agent.calculate_position_size(_INSTRUMENT, 1.1000, 1.0990, 10_000.0)
    assert size == pytest.approx(100_000.0)


@pytest.mark.asyncio
async def test_position_size_is_capped_when_a_real_override_is_set(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "500")
    size = await agent.calculate_position_size(_INSTRUMENT, 1.1000, 1.0990, 10_000.0)
    assert size == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_position_size_below_the_cap_is_left_unchanged(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """A generous cap that the risk-derived size never reaches must not distort a normal, smaller order."""
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "1000000")
    size = await agent.calculate_position_size(_INSTRUMENT, 1.1000, 1.0990, 10_000.0)
    assert size == pytest.approx(100_000.0)

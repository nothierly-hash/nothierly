"""Real, executable tests for the live-overridable drawdown trigger in src/agents/execution/agent.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test), same
discipline as ``tests/test_execution_agent_control_state.py``.
``Guardian(llm_provider="none")`` forces the real, deterministic
fallback verdict path (no network call) -- the point of these tests is
the live Redis override + proportional re-arm math, not Guardian AI's
own LLM integration (already covered elsewhere).
"""

from __future__ import annotations

import json
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest

from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from src.agents.execution.agent import (
    DRAWDOWN_ALERT_THRESHOLD_PCT,
    DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY,
    ExecutionAgent,
)

# Matches ExecutionAgent's own private _GUARDIAN_ALERTS_REDIS_KEY -- duplicated
# as a literal rather than importing a leading-underscore name, same
# convention tests/test_execution_agent_control_state.py already established.
_GUARDIAN_ALERTS_REDIS_KEY = "NOTHIERLY_GUARDIAN_ALERTS"
_INSTRUMENT = "EUR_USD"


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """Real cleanup, not optional -- see tests/test_execution_agent_instrument_lock.py's identical
    fixture docstring: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets synchronously (even under
    MOCK_MODE, which only affects the broker call, not this), and leaving them open hangs/risks
    hanging the pytest process at teardown."""
    instance = ExecutionAgent(
        {_INSTRUMENT}, capital=10_000.0,
        settings=AppSettings(mock_mode=True, guardian=GuardianSettings(llm_provider="none")),
        manage_own_signals=False, manage_own_redis=False,
    )
    instance._redis_client = redis_client
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _guardian_alerts(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw_entries = await redis_client.execute("lrange", _GUARDIAN_ALERTS_REDIS_KEY, 0, -1)
    return [json.loads(raw) for raw in raw_entries]


# -- _get_dynamic_drawdown_threshold_pct --------------------------------


@pytest.mark.asyncio
async def test_no_override_returns_the_real_default(agent: ExecutionAgent) -> None:
    assert await agent._get_dynamic_drawdown_threshold_pct() == DRAWDOWN_ALERT_THRESHOLD_PCT


@pytest.mark.asyncio
async def test_valid_override_is_used_verbatim(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "0.10")
    assert await agent._get_dynamic_drawdown_threshold_pct() == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_override_below_floor_is_clamped_up(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "0.001")
    assert await agent._get_dynamic_drawdown_threshold_pct() == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_override_above_ceiling_is_clamped_down(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """A corrupted/malicious Redis value can never push the trigger so high it effectively never fires."""
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "5.0")
    assert await agent._get_dynamic_drawdown_threshold_pct() == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_malformed_override_falls_back_to_default(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "not-a-number")
    assert await agent._get_dynamic_drawdown_threshold_pct() == DRAWDOWN_ALERT_THRESHOLD_PCT


@pytest.mark.asyncio
async def test_no_redis_client_falls_back_to_default(agent: ExecutionAgent) -> None:
    agent._redis_client = None
    assert await agent._get_dynamic_drawdown_threshold_pct() == DRAWDOWN_ALERT_THRESHOLD_PCT


# -- _check_drawdown_trigger --------------------------------


@pytest.mark.asyncio
async def test_drawdown_under_default_threshold_does_not_alert(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await agent._check_drawdown_trigger(10_000.0)  # establishes the real baseline
    await agent._check_drawdown_trigger(9_600.0)  # 4% drop -- under the 5% default

    assert await _guardian_alerts(redis_client) == []


@pytest.mark.asyncio
async def test_drawdown_at_or_above_default_threshold_alerts_exactly_once(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    await agent._check_drawdown_trigger(10_000.0)
    await agent._check_drawdown_trigger(9_400.0)  # 6% drop -- over the 5% default
    await agent._check_drawdown_trigger(9_300.0)  # still down -- must not re-alert while disarmed

    alerts = await _guardian_alerts(redis_client)
    assert len(alerts) == 1


@pytest.mark.asyncio
async def test_live_redis_override_actually_changes_trigger_behavior(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """The real point of this feature: a dashboard-set override changes behavior on the very next check, no restart."""
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "0.02")

    await agent._check_drawdown_trigger(10_000.0)
    await agent._check_drawdown_trigger(9_700.0)  # 3% drop -- would NOT trigger the 5% default, but DOES trigger a 2% override

    alerts = await _guardian_alerts(redis_client)
    assert len(alerts) == 1
    assert "seuil configure : 2.0%" in alerts[0]["summary"] or "2.0%" in alerts[0]["summary"]


@pytest.mark.asyncio
async def test_rearm_is_proportional_to_the_current_threshold_not_independently_fixed(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """Real regression target: re-arm threshold = alert_threshold * 0.6, scaling with a live override instead of a fixed 3%."""
    await redis_client.execute("set", DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, "0.10")  # alert=10% -> rearm=6%

    await agent._check_drawdown_trigger(10_000.0)
    await agent._check_drawdown_trigger(8_900.0)  # 11% drop -- crosses the 10% alert threshold
    assert len(await _guardian_alerts(redis_client)) == 1

    await agent._check_drawdown_trigger(9_350.0)  # recovers to 6.5% drop -- still ABOVE the 6% rearm point: must not rearm
    await agent._check_drawdown_trigger(8_900.0)  # a second dip while still disarmed -- must not alert again
    assert len(await _guardian_alerts(redis_client)) == 1

    await agent._check_drawdown_trigger(9_450.0)  # recovers to 5.5% drop -- now BELOW the 6% rearm point: rearms
    await agent._check_drawdown_trigger(8_900.0)  # a third dip -- now armed again, must alert a second time
    assert len(await _guardian_alerts(redis_client)) == 2

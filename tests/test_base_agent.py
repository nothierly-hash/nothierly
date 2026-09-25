"""Real, executable tests for core/base_agent.py's new AgentControlState mechanism.

Redis is faked with ``fakeredis`` (same discipline as ``tests/
test_alert_agent.py`` -- a real in-memory Redis implementation, fresh
``FakeServer()`` per test) -- including its real Pub/Sub support,
verified directly against a real cross-client publish/subscribe before
writing this suite (see the module docstring note in
``core/base_agent.py``'s ``_listen_for_control_commands`` about the
installed fakeredis version returning raw bytes for pubsub payloads
despite ``decode_responses=True`` -- a real, observed quirk, not a bug
in this project's own code).
"""

from __future__ import annotations

import asyncio
from typing import Generator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState, BaseAgent
from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import AGENT_CONTROL_CHANNEL_PREFIX, AGENT_CONTROL_STATE_REDIS_KEY_PREFIX

_AGENT_ID = "test-agent"


class _NoOpAgent(BaseAgent):
    """The minimal real concrete BaseAgent -- on_tick does nothing, just enough to exercise the shared lifecycle."""

    async def on_tick(self) -> None:
        pass


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """A real RedisClient over a real in-memory fakeredis server -- see test_alert_agent.py's identical fixture."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


# BaseAgent.__init__ opens real ZMQ sockets/contexts; unclosed ones make pytest hang at exit inside
# zmq.Context.__del__ -> term() (the recurring fixture-leak pattern of this repo's agent tests -- same
# registry + autouse teardown as tests/test_alpha_engine.py, since several tests build agents inline).
# Closing is idempotent, so the agent that test_full_run_lifecycle... already shuts down via run() is fine.
_CREATED_AGENTS: list[_NoOpAgent] = []


@pytest.fixture(autouse=True)
def _close_real_zmq_sockets() -> Generator[None, None, None]:
    yield
    while _CREATED_AGENTS:
        created = _CREATED_AGENTS.pop()
        created._publisher.close()
        created._subscriber.close()
        created._zmq_context.term()


def _new_agent(agent_id: str = _AGENT_ID) -> _NoOpAgent:
    instance = _NoOpAgent(agent_id, manage_own_signals=False, manage_own_redis=False)
    _CREATED_AGENTS.append(instance)
    return instance


@pytest.fixture
def agent(redis_client: RedisClient) -> _NoOpAgent:
    instance = _new_agent()
    instance._redis_client = redis_client
    return instance


def test_default_control_state_is_live() -> None:
    """A fresh agent, before its control listener has even started, must default to LIVE -- unchanged behavior."""
    instance = _new_agent()
    assert instance.control_state == AgentControlState.LIVE


@pytest.mark.asyncio
async def test_control_listener_restores_a_real_persisted_state(agent: _NoOpAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{_AGENT_ID}", AgentControlState.SHADOW.value)

    listener_task = asyncio.create_task(agent._listen_for_control_commands())
    await asyncio.sleep(0.1)
    try:
        assert agent.control_state == AgentControlState.SHADOW
    finally:
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task


@pytest.mark.asyncio
async def test_control_listener_defaults_to_live_when_no_persisted_state_exists(agent: _NoOpAgent) -> None:
    listener_task = asyncio.create_task(agent._listen_for_control_commands())
    await asyncio.sleep(0.1)
    try:
        assert agent.control_state == AgentControlState.LIVE
    finally:
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task


@pytest.mark.asyncio
async def test_control_listener_ignores_an_invalid_persisted_value(agent: _NoOpAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{_AGENT_ID}", "NOT_A_REAL_STATE")

    listener_task = asyncio.create_task(agent._listen_for_control_commands())
    await asyncio.sleep(0.1)
    try:
        assert agent.control_state == AgentControlState.LIVE
    finally:
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task


@pytest.mark.asyncio
async def test_a_real_pubsub_command_updates_state_instantly(agent: _NoOpAgent, redis_client: RedisClient) -> None:
    """The real, load-bearing claim: a live PUBLISH changes an already-running agent's in-memory state, not just startup."""
    listener_task = asyncio.create_task(agent._listen_for_control_commands())
    await asyncio.sleep(0.1)
    state_before: AgentControlState = agent.control_state
    assert state_before == AgentControlState.LIVE

    await redis_client.execute("publish", f"{AGENT_CONTROL_CHANNEL_PREFIX}{_AGENT_ID}", AgentControlState.OFF.value)
    await asyncio.sleep(0.1)

    try:
        state_after: AgentControlState = agent.control_state
        assert state_after == AgentControlState.OFF
    finally:
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task


@pytest.mark.asyncio
async def test_an_invalid_pubsub_command_is_ignored(agent: _NoOpAgent, redis_client: RedisClient) -> None:
    listener_task = asyncio.create_task(agent._listen_for_control_commands())
    await asyncio.sleep(0.1)

    await redis_client.execute("publish", f"{AGENT_CONTROL_CHANNEL_PREFIX}{_AGENT_ID}", "GARBAGE")
    await asyncio.sleep(0.1)

    try:
        assert agent.control_state == AgentControlState.LIVE
    finally:
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task


@pytest.mark.asyncio
async def test_a_command_only_affects_the_targeted_agent(redis_client: RedisClient) -> None:
    """Two agents, two independent channels -- publishing to one must never affect the other."""
    agent_a = _new_agent("agent-a")
    agent_a._redis_client = redis_client
    agent_b = _new_agent("agent-b")
    agent_b._redis_client = redis_client

    task_a = asyncio.create_task(agent_a._listen_for_control_commands())
    task_b = asyncio.create_task(agent_b._listen_for_control_commands())
    await asyncio.sleep(0.1)

    await redis_client.execute("publish", f"{AGENT_CONTROL_CHANNEL_PREFIX}agent-a", AgentControlState.OFF.value)
    await asyncio.sleep(0.1)

    try:
        assert agent_a.control_state == AgentControlState.OFF
        assert agent_b.control_state == AgentControlState.LIVE
    finally:
        task_a.cancel()
        task_b.cancel()
        for task in (task_a, task_b):
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_full_run_lifecycle_starts_and_cleans_up_the_control_listener(agent: _NoOpAgent) -> None:
    """Real end-to-end: run() starts the listener automatically, and shutdown cancels it cleanly (no leaked task, no raised exception)."""
    run_task = asyncio.create_task(agent.run())
    await asyncio.sleep(0.1)

    assert agent._control_listener_task is not None
    assert not agent._control_listener_task.done()

    agent.request_shutdown()
    await asyncio.wait_for(run_task, timeout=5.0)  # must not raise

    assert agent._control_listener_task.done()
    assert agent._control_listener_task.cancelled()

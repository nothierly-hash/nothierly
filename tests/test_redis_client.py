"""Real, executable tests for core/redis_client.py -- in particular RedisClient.get_instance()'s cross-event-loop safety.

**Real bug this guards against, found via live dashboard testing**:
clicking the SRE & Red Phone page's kill-switch trip button, then its
reset button, raised ``RuntimeError: Event loop is closed`` on the
second click -- ``RedisClient``'s singleton was created inside the
first ``asyncio.run(trip_kill_switch(...))`` call, and its underlying
connection's transport stayed bound to that call's event loop even
after it closed. The *second*, separate ``asyncio.run(reset_kill_switch
(...))`` call runs on a fresh loop but reused the same stale singleton
-- exactly what every Streamlit button handler does (each click is its
own ``asyncio.run(...)``), and exactly the shape ``core/redis_client.
py``'s own docstring on the primary intended caller (one agent process,
one ``asyncio.run(main())`` for its whole lifetime) never has to deal
with.

**Why these tests can't reproduce the literal exception**: that
``RuntimeError`` comes from a real ``asyncio`` transport object refusing
to touch a closed loop -- an artifact of real OS sockets, not something
a hermetic in-memory fake naturally recreates (``fakeredis``'s async
implementation isn't loop-bound in the same way). What *is* real and
testable here is the actual fix: ``get_instance()`` now tracks which
loop its connection belongs to and discards + reconnects on a mismatch,
rather than blindly reusing a stale one. These tests drive that through
two genuinely separate ``asyncio.run()`` calls (not pytest-asyncio's
per-test loop) -- the same shape the real bug needs to occur at all.
"""

from __future__ import annotations

import asyncio
from typing import Any

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    RedisClient._instance = None
    yield
    RedisClient._instance = None


@pytest.fixture
def _fake_redis_constructor(monkeypatch: pytest.MonkeyPatch) -> fakeredis.FakeServer:
    """Substitute the real redis.asyncio.Redis(...) constructor RedisClient.__init__ calls.

    One shared FakeServer across every fake client created during a test
    -- matching real Redis's own shared-state semantics across
    reconnections (two separate connections to the same server still see
    the same data), which a fresh FakeServer per instantiation would not.
    """
    server = fakeredis.FakeServer()

    def _construct_fake(**kwargs: Any) -> fakeredis.aioredis.FakeRedis:
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    monkeypatch.setattr("core.redis_client.Redis", _construct_fake)
    return server


def test_manually_installed_instance_with_untracked_loop_is_trusted_as_is(_fake_redis_constructor: fakeredis.FakeServer) -> None:
    """Real regression test for the fix that broke tests/test_alert_agent.py's own fixture pattern.

    That fixture (and others like it) installs a fake singleton directly
    (``RedisClient._instance = client``), bypassing get_instance()
    entirely -- ._loop stays None. The cross-loop-mismatch fix above must
    not treat that as "stale" and discard a test's carefully faked
    connection the first time application code under test calls
    get_instance() itself.
    """
    manually_installed = RedisClient(RedisSettings())
    manually_installed._redis = fakeredis.aioredis.FakeRedis(server=_fake_redis_constructor, decode_responses=True)
    RedisClient._instance = manually_installed
    assert manually_installed._loop is None

    async def _main() -> RedisClient:
        return await RedisClient.get_instance(RedisSettings())

    resolved = asyncio.run(_main())
    assert resolved is manually_installed


def test_get_instance_returns_same_object_within_one_event_loop(_fake_redis_constructor: fakeredis.FakeServer) -> None:
    seen: list[RedisClient] = []

    async def _main() -> None:
        first = await RedisClient.get_instance(RedisSettings())
        second = await RedisClient.get_instance(RedisSettings())
        seen.extend([first, second])

    asyncio.run(_main())
    assert seen[0] is seen[1]


def test_get_instance_reconnects_across_separate_asyncio_run_calls(_fake_redis_constructor: fakeredis.FakeServer) -> None:
    """The real regression test: two SEPARATE asyncio.run() calls (the dashboard's exact shape) must both work."""
    instances: list[RedisClient] = []
    loops: list[asyncio.AbstractEventLoop] = []

    async def _get_and_record() -> None:
        instance = await RedisClient.get_instance(RedisSettings())
        instances.append(instance)
        loops.append(asyncio.get_running_loop())

    asyncio.run(_get_and_record())  # loop A -- closes on return, exactly like one Streamlit button click
    asyncio.run(_get_and_record())  # loop B -- a second, separate click

    assert instances[0] is not instances[1], "the stale cross-loop singleton must be discarded, not reused"
    assert instances[0]._loop is loops[0]
    assert instances[1]._loop is loops[1]
    assert loops[0] is not loops[1]


def _decode(value: object) -> str | None:
    """This fakeredis version doesn't fully honor decode_responses=True for every command -- see tests/test_alert_agent.py."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def test_second_asyncio_run_call_can_still_execute_real_commands(_fake_redis_constructor: fakeredis.FakeServer) -> None:
    """Not just object identity -- the reconnected instance must actually work, matching the real live symptom."""

    async def _first_call() -> None:
        client = await RedisClient.get_instance(RedisSettings())
        await client.execute("set", "kill_switch_test_key", "PANIC")

    async def _second_call() -> str | None:
        client = await RedisClient.get_instance(RedisSettings())
        result = await client.execute("get", "kill_switch_test_key")
        await client.execute("set", "kill_switch_test_key", "SAFE")
        return _decode(result)

    asyncio.run(_first_call())
    value = asyncio.run(_second_call())
    assert value == "PANIC"

    async def _third_call() -> str | None:
        client = await RedisClient.get_instance(RedisSettings())
        result = await client.execute("get", "kill_switch_test_key")
        return _decode(result)

    final_value = asyncio.run(_third_call())
    assert final_value == "SAFE"

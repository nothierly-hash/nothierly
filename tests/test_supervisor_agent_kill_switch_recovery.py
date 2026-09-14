"""Real, executable regression test for the 2026-09-14 local Kill Switch self-healing fix.

Real bug found live during this project's own multi-hour shadow-run stress test: a transient,
one-off Redis contention spike during the concurrent 17-agent startup burst tripped
XgboostSignalAgent's local Kill Switch for two real live instances -- and the old
``_run_polling_loop`` permanently ``break``-ed out of its own loop on the very first trip,
silencing that instance's real signal generation for the rest of the process's life (confirmed
live: 73+ minutes and counting, zero recovery). These tests prove the fix: the loop now keeps
re-checking every cycle instead of dying, so a genuinely transient trip clears itself once a real,
healthy read succeeds again -- while a *persistent* trip condition still correctly blocks
publishing for as long as it holds, exactly as before.
"""

from __future__ import annotations

from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import MarketMetrics
from src.agents.supervisor.agent import EmergencyStopError, SignalState, XgboostSignalAgent

_INSTRUMENT = "EUR_USD"


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """Real RedisClient over real in-memory fakeredis, ALSO installed as the process singleton --
    core.security.trip_kill_switch resolves RedisClient.get_instance() independently of whatever
    agent._redis_client is separately set to (same reasoning as test_execution_agent_control_state.py's
    identical fixture)."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[XgboostSignalAgent]:
    instance = XgboostSignalAgent(
        _INSTRUMENT, poll_interval_seconds=0.01, manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    real_publisher = instance._publisher  # captured before being replaced, see teardown below
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    yield instance
    # Real cleanup, not optional: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets on construction
    # -- leaving them open hangs the test process at session teardown (see
    # tests/test_spectral_denoiser_cwt.py's identical fixture docstring for the same real bug,
    # found first there). Closing the REAL original publisher captured above, not the fake now
    # sitting at instance._publisher -- the fake's own close() is a harmless no-op that would
    # silently leave the real, now-orphaned socket open.
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def _healthy_state() -> SignalState:
    metrics = MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=1.10000)
    return SignalState(metrics, {}, [], redis_latency_ms=1.0)


def _unhealthy_state() -> SignalState:
    return SignalState(None, {}, [], redis_latency_ms=float("inf"))


@pytest.mark.asyncio
async def test_transient_trip_self_heals_instead_of_permanently_stopping(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bug's exact shape: trip once (transient), then a real healthy read must resume
    real signal publishing -- not stay permanently dark."""
    call_count = 0

    async def fake_fetch_latest_state() -> SignalState:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _unhealthy_state()  # the real, one-off transient blip
        return _healthy_state()  # every real read after that succeeds

    monkeypatch.setattr(agent, "_fetch_latest_state", fake_fetch_latest_state)

    agent._running = True
    for _ in range(4):
        try:
            await agent._run_signal_cycle()
        except EmergencyStopError:
            pass  # exactly what _run_polling_loop itself now catches and continues past

    assert call_count == 4, "the loop must keep calling _fetch_latest_state every cycle, never stop"
    assert len(agent._publisher.published) >= 1, (  # type: ignore[attr-defined]
        "a real signal must be published once the transient condition clears -- the instance "
        "must not stay permanently silent after a one-off trip"
    )
    assert agent._emergency_mode is False, "emergency_mode must reset once a real healthy cycle completes"


@pytest.mark.asyncio
async def test_persistent_trip_still_blocks_publishing_every_cycle(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix must not weaken the actual safety condition: a real, PERSISTENT unhealthy state
    must keep blocking publication on every single cycle, for as long as it persists."""

    async def always_unhealthy() -> SignalState:
        return _unhealthy_state()

    monkeypatch.setattr(agent, "_fetch_latest_state", always_unhealthy)

    agent._running = True
    for _ in range(5):
        with pytest.raises(EmergencyStopError):
            await agent._run_signal_cycle()

    assert agent._publisher.published == [], (  # type: ignore[attr-defined]
        "a persistently unhealthy condition must never let a real signal through, on any cycle"
    )
    assert agent._emergency_mode is True, "emergency_mode must stay True while the real condition persists"


@pytest.mark.asyncio
async def test_run_polling_loop_does_not_break_on_emergency_stop_error(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real end-to-end proof against _run_polling_loop itself (not just _run_signal_cycle in
    isolation): the loop must still be alive and calling _fetch_latest_state after a trip."""
    import asyncio

    call_count = 0

    async def fake_fetch_latest_state() -> SignalState:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _unhealthy_state()
        return _healthy_state()

    monkeypatch.setattr(agent, "_fetch_latest_state", fake_fetch_latest_state)

    agent._running = True
    task = asyncio.create_task(agent._run_polling_loop())
    try:
        # Real wall-clock wait, bounded, polling call_count directly rather than a fixed sleep --
        # real per-cycle wall-clock time (including a real fakeredis heartbeat write) is not
        # perfectly predictable, so this waits for the real outcome instead of guessing a duration.
        for _ in range(200):
            if call_count >= 3:
                break
            await asyncio.sleep(0.05)
    finally:
        agent._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert call_count >= 3, f"the real polling loop must keep running past the trip, got {call_count} real cycles"
    assert len(agent._publisher.published) >= 1  # type: ignore[attr-defined]

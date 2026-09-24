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

from datetime import datetime, timedelta, timezone
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.message_integrity import SignatureVerificationError
from core.redis_client import RedisClient, RedisUnavailableError
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


def _healthy_state_at(tick_timestamp: datetime) -> SignalState:
    metrics = MarketMetrics(
        symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=1.10000, timestamp=tick_timestamp,
    )
    return SignalState(metrics, {}, [], redis_latency_ms=1.0)


def _record_auto_recovery_calls(monkeypatch: pytest.MonkeyPatch, *, returns: bool = True) -> list[None]:
    calls: list[None] = []

    async def fake_auto_recover() -> bool:
        calls.append(None)
        return returns

    monkeypatch.setattr("src.agents.supervisor.agent.auto_recover_from_data_loss", fake_auto_recover)
    return calls


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
    """A PERSISTENT unhealthy read must veto publication on every cycle -- as a soft veto: no raise,
    no global Kill Switch trip (2026-09-20: a 1% transient read failure used to latch it on)."""
    trips = _record_kill_switch_trips(monkeypatch)

    async def always_unhealthy() -> SignalState:
        return _unhealthy_state()

    monkeypatch.setattr(agent, "_fetch_latest_state", always_unhealthy)

    agent._running = True
    for _ in range(5):
        await agent._run_signal_cycle()  # must not raise

    assert agent._publisher.published == [], (  # type: ignore[attr-defined]
        "a persistently unhealthy condition must never let a real signal through, on any cycle"
    )
    assert trips == [], "a Redis-health problem is a soft veto, never a global Kill Switch trip"


def _record_kill_switch_trips(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    trips: list[str] = []

    async def fake_trip(reason: str) -> None:
        trips.append(reason)

    monkeypatch.setattr("src.agents.supervisor.agent.trip_kill_switch", fake_trip)
    return trips


def _record_data_loss_latches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    latches: list[str] = []

    async def fake_latch(reason: str) -> None:
        latches.append(reason)

    monkeypatch.setattr("src.agents.supervisor.agent.latch_data_loss", fake_latch)
    return latches


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [SignatureVerificationError("torn read"), RedisUnavailableError("blip")])
async def test_isolated_read_failure_is_a_soft_veto_not_a_kill_switch_trip(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    trips = _record_kill_switch_trips(monkeypatch)
    calls = 0

    async def flaky_fetch() -> SignalState:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        return _healthy_state()

    monkeypatch.setattr(agent, "_fetch_latest_state", flaky_fetch)
    await agent._run_signal_cycle()  # the failing read: veto, no raise
    assert agent._publisher.published == []  # type: ignore[attr-defined]
    await agent._run_signal_cycle()  # the very next tick ingests normally
    assert len(agent._publisher.published) == 1  # type: ignore[attr-defined]
    assert trips == []
    assert agent._consecutive_error_count == 0


@pytest.mark.asyncio
async def test_spread_anomaly_and_brief_staleness_are_soft_vetoes(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    trips = _record_kill_switch_trips(monkeypatch)
    for _ in range(agent._spread_history_window):
        assert await agent._check_kill_switch(_healthy_state()) == []
    wide = MarketMetrics(symbol=_INSTRUMENT, spread=0.01, order_book_imbalance=0.1, mid_price=1.10000)
    assert await agent._check_kill_switch(SignalState(wide, {}, [], redis_latency_ms=1.0))  # veto reasons, no raise
    stale = MarketMetrics(
        symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=1.10000,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=2),
    )
    assert await agent._check_kill_switch(SignalState(stale, {}, [], redis_latency_ms=1.0))
    assert trips == []


@pytest.mark.asyncio
async def test_total_feed_loss_latches_data_loss_only_after_the_grace_period(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-24: total feed loss now latches LATCHED_DATA (auto-recoverable), never trips PANIC directly."""
    import asyncio

    trips = _record_kill_switch_trips(monkeypatch)
    latches = _record_data_loss_latches(monkeypatch)
    agent._feed_loss_grace_seconds = 0.2
    assert await agent._check_kill_switch(_healthy_state()) == []  # a fresh reading arms the loss clock
    gone = SignalState(None, {}, [], redis_latency_ms=1.0)  # key expired: the feed is gone
    assert await agent._check_kill_switch(gone) == []  # inside the grace: nothing
    assert trips == []
    assert latches == []
    await asyncio.sleep(0.25)
    with pytest.raises(EmergencyStopError):
        await agent._check_kill_switch(gone)
    assert len(latches) == 1 and "feed lost" in latches[0]
    assert trips == [], "feed loss must never call trip_kill_switch (PANIC) directly"


@pytest.mark.asyncio
async def test_cold_start_without_any_fresh_reading_is_not_a_feed_loss(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    trips = _record_kill_switch_trips(monkeypatch)
    latches = _record_data_loss_latches(monkeypatch)
    agent._feed_loss_grace_seconds = 0.05
    gone = SignalState(None, {}, [], redis_latency_ms=1.0)
    await agent._check_kill_switch(gone)
    await asyncio.sleep(0.1)
    await agent._check_kill_switch(gone)
    assert trips == []
    assert latches == []


# =============================================================================
# Data-loss probation / auto-recovery (2026-09-24, "Resilient Kill Switch")
# =============================================================================


@pytest.mark.asyncio
async def test_probation_attempts_auto_recovery_once_the_full_window_of_fresh_ticks_has_elapsed(
    agent: XgboostSignalAgent,
) -> None:
    from core import security

    # A future-dated timestamp reads as negative-age (extra fresh) against staleness_reason's real
    # datetime.now(timezone.utc) comparison -- the standard way this project's own tests hold a
    # tick's timestamp fixed and independently controllable without monkeypatching the real clock.
    agent._data_loss_probation_seconds = 60.0
    start = datetime.now(timezone.utc) + timedelta(days=1)

    await security.latch_data_loss("test setup: simulate a prior real feed-loss latch")
    try:
        await agent._check_kill_switch(_healthy_state_at(start))
        assert await security.is_system_safe() is False  # still LATCHED_DATA -- probation not started long enough

        await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=59)))
        assert await security.is_system_safe() is False  # 59s < 60s probation -- not yet

        await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=60)))
        assert await security.is_system_safe() is True  # a real, full 60s of continuous fresh ticks -> recovered
    finally:
        await security.reset_kill_switch()


@pytest.mark.asyncio
async def test_probation_only_attempts_recovery_once_per_streak(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_auto_recovery_calls(monkeypatch, returns=True)
    agent._data_loss_probation_seconds = 60.0
    start = datetime.now(timezone.utc) + timedelta(days=1)

    await agent._check_kill_switch(_healthy_state_at(start))
    await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=60)))
    await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=61)))
    await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=62)))

    assert len(calls) == 1, "a real Redis round-trip on every healthy cycle after recovery would be wasteful"


@pytest.mark.asyncio
async def test_a_broken_streak_resets_the_probation_clock(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_auto_recovery_calls(monkeypatch, returns=True)
    agent._data_loss_probation_seconds = 60.0
    start = datetime.now(timezone.utc) + timedelta(days=1)

    await agent._check_kill_switch(_healthy_state_at(start))
    await agent._check_kill_switch(_healthy_state_at(start + timedelta(seconds=50)))  # 50s in -- not yet

    # A real, genuinely stale reading (5s old against the real wall clock) breaks the streak,
    # regardless of the future-dated streak this test otherwise uses.
    stale = MarketMetrics(
        symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=1.10000,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    await agent._check_kill_switch(SignalState(stale, {}, [], redis_latency_ms=1.0))

    # A fresh streak restarting here must count from THIS tick, not from the original start.
    restart = start + timedelta(seconds=200)
    await agent._check_kill_switch(_healthy_state_at(restart))
    await agent._check_kill_switch(_healthy_state_at(restart + timedelta(seconds=59)))
    assert calls == [], "59s since the restart point is not yet a real, full 60s continuous streak"

    await agent._check_kill_switch(_healthy_state_at(restart + timedelta(seconds=60)))
    assert len(calls) == 1, "a real, full new 60s streak from the restart point must still recover"


@pytest.mark.asyncio
async def test_flash_crash_never_attempts_auto_recovery(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PANIC (flash crash) is structurally never touched by the data-loss probation path."""
    trips = _record_kill_switch_trips(monkeypatch)
    recovery_calls = _record_auto_recovery_calls(monkeypatch, returns=True)

    await agent._check_kill_switch(_healthy_state())
    crashed = MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=0.90000)
    with pytest.raises(EmergencyStopError):
        await agent._check_kill_switch(SignalState(crashed, {}, [], redis_latency_ms=1.0))

    assert len(trips) == 1
    assert recovery_calls == [], "a flash crash must never go through the LATCHED_DATA auto-recovery path"


@pytest.mark.asyncio
async def test_flash_crash_still_trips_the_global_kill_switch(
    agent: XgboostSignalAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    trips = _record_kill_switch_trips(monkeypatch)
    await agent._check_kill_switch(_healthy_state())
    crashed = MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=0.90000)
    with pytest.raises(EmergencyStopError):
        await agent._check_kill_switch(SignalState(crashed, {}, [], redis_latency_ms=1.0))
    assert len(trips) == 1


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

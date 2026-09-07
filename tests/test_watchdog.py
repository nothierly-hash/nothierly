"""Real, executable tests for core/watchdog.py.

Same ``os._exit()`` interception discipline as ``tests/test_exception_manager.py``
(see that file's own docstring) -- :class:`SystemWatchdog` triggers a real
shutdown through the exact same :func:`core.exception_manager.graceful_shutdown`,
so its tests need the identical ``_ExitCalled`` marker instead of actually
killing the pytest process.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Generator

import fakeredis.aioredis
import psutil
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.watchdog import SystemWatchdog, _detect_container_memory_limit_bytes


class _ExitCalled(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"os._exit({code})")
        self.code = code


@pytest.fixture
def exit_calls(monkeypatch: pytest.MonkeyPatch) -> Generator[list[int], None, None]:
    calls: list[int] = []

    def fake_exit(code: int) -> None:
        calls.append(code)
        raise _ExitCalled(code)

    monkeypatch.setattr("os._exit", fake_exit)
    yield calls


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """Real fakeredis-backed RedisClient -- same discipline as tests/test_alpha_engine.py's own."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


def _make_watchdog(redis_client: RedisClient, tmp_path: Path, **kwargs: object) -> SystemWatchdog:
    defaults: dict[str, object] = dict(
        redis_client=redis_client,
        shutdown_callback=_noop_shutdown,
        heartbeat_file=tmp_path / "heartbeat.txt",
        check_interval_seconds=10.0,
        redis_unresponsive_threshold_seconds=30.0,
        ram_percent_threshold=85.0,
        container_memory_limit_bytes=10 * 1024 * 1024 * 1024,  # 10 GiB -- comfortably above any real test RSS
    )
    defaults.update(kwargs)
    return SystemWatchdog(**defaults)  # type: ignore[arg-type]


async def _noop_shutdown() -> None:
    pass


# =============================================================================
# _detect_container_memory_limit_bytes -- real cgroup file reads, real temp files
# =============================================================================


def test_detects_a_real_cgroup_v2_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    limit = 1_073_741_824  # 1 GiB -- comfortably below any real host's RAM
    cgroup_v2.write_text(str(limit))
    monkeypatch.setattr("core.watchdog._CGROUP_V2_MEMORY_MAX", cgroup_v2)
    monkeypatch.setattr("core.watchdog._CGROUP_V1_MEMORY_LIMIT", tmp_path / "does_not_exist")

    assert _detect_container_memory_limit_bytes() == limit


def test_falls_back_past_a_real_unlimited_v2_sentinel_to_host_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("max")  # cgroup v2's own real spelling of "no limit"
    monkeypatch.setattr("core.watchdog._CGROUP_V2_MEMORY_MAX", cgroup_v2)
    monkeypatch.setattr("core.watchdog._CGROUP_V1_MEMORY_LIMIT", tmp_path / "does_not_exist")

    assert _detect_container_memory_limit_bytes() == psutil.virtual_memory().total


def test_falls_back_to_host_total_when_neither_cgroup_path_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.watchdog._CGROUP_V2_MEMORY_MAX", Path("/nonexistent/memory.max"))
    monkeypatch.setattr("core.watchdog._CGROUP_V1_MEMORY_LIMIT", Path("/nonexistent/memory.limit_in_bytes"))

    assert _detect_container_memory_limit_bytes() == psutil.virtual_memory().total


def test_prefers_v2_over_v1_when_both_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("536870912")  # 512 MiB
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text("1073741824")  # 1 GiB
    monkeypatch.setattr("core.watchdog._CGROUP_V2_MEMORY_MAX", cgroup_v2)
    monkeypatch.setattr("core.watchdog._CGROUP_V1_MEMORY_LIMIT", cgroup_v1)

    assert _detect_container_memory_limit_bytes() == 536_870_912


# =============================================================================
# Redis vital sign -- real fakeredis, real failure via a real AttributeError
# =============================================================================


@pytest.mark.asyncio
async def test_ping_redis_succeeds_against_real_fakeredis(redis_client: RedisClient, tmp_path: Path) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path)
    ok, latency_ms = await watchdog._ping_redis()
    assert ok is True
    assert latency_ms is not None and latency_ms >= 0.0


@pytest.mark.asyncio
async def test_ping_redis_fails_when_client_unreachable(redis_client: RedisClient, tmp_path: Path) -> None:
    redis_client._redis = None  # type: ignore[assignment] # real, simple way to force a real AttributeError inside execute()
    watchdog = _make_watchdog(redis_client, tmp_path)
    ok, latency_ms = await watchdog._ping_redis()
    assert ok is False
    assert latency_ms is None


@pytest.mark.asyncio
async def test_transient_redis_failure_alone_does_not_trigger_shutdown(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    redis_client._redis = None  # type: ignore[assignment]
    watchdog = _make_watchdog(redis_client, tmp_path, redis_unresponsive_threshold_seconds=30.0)
    await watchdog._check_once()
    assert exit_calls == []


@pytest.mark.asyncio
async def test_redis_unresponsive_past_threshold_triggers_shutdown(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    redis_client._redis = None  # type: ignore[assignment]
    watchdog = _make_watchdog(redis_client, tmp_path, redis_unresponsive_threshold_seconds=0.05)

    with pytest.raises(_ExitCalled):
        await watchdog._check_once()  # first failure -- starts the unresponsive clock
        await asyncio.sleep(0.06)
        await watchdog._check_once()  # now past the 0.05s threshold

    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_recovering_before_the_threshold_resets_the_unresponsive_clock(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    real_redis = redis_client._redis
    redis_client._redis = None  # type: ignore[assignment]
    watchdog = _make_watchdog(redis_client, tmp_path, redis_unresponsive_threshold_seconds=0.2)

    await watchdog._check_once()  # fails -- starts the clock
    assert watchdog._redis_unresponsive_since is not None

    redis_client._redis = real_redis  # recovers well before the 0.2s threshold
    await watchdog._check_once()
    assert watchdog._redis_unresponsive_since is None
    assert exit_calls == []


# =============================================================================
# RAM vital sign -- real psutil.Process().memory_info().rss, an artificially tiny limit
# =============================================================================


@pytest.mark.asyncio
async def test_ram_breach_triggers_shutdown(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    """A real, tiny container_memory_limit_bytes guarantees this test process's own real RSS exceeds 85% of it."""
    watchdog = _make_watchdog(redis_client, tmp_path, container_memory_limit_bytes=1024)  # 1 KiB

    with pytest.raises(_ExitCalled):
        await watchdog._check_once()

    assert exit_calls == [1]


@pytest.mark.asyncio
async def test_ram_breach_does_not_write_the_heartbeat_file(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(redis_client, tmp_path, heartbeat_file=heartbeat_file, container_memory_limit_bytes=1024)

    with pytest.raises(_ExitCalled):
        await watchdog._check_once()

    assert not heartbeat_file.exists()


@pytest.mark.asyncio
async def test_healthy_tick_does_not_trigger_shutdown(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path)
    await watchdog._check_once()
    assert exit_calls == []


# =============================================================================
# Exchange vital sign -- an arbitrary caller-supplied callback, never a shutdown trigger
# =============================================================================


@pytest.mark.asyncio
async def test_exchange_ping_ok_when_none_configured(redis_client: RedisClient, tmp_path: Path) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path, exchange_ping=None)
    ok, latency_ms = await watchdog._ping_exchange()
    assert ok is True
    assert latency_ms is None


@pytest.mark.asyncio
async def test_exchange_ping_measures_a_real_latency_for_a_successful_callback(
    redis_client: RedisClient, tmp_path: Path,
) -> None:
    async def fake_ping() -> float:
        await asyncio.sleep(0.01)
        return 65000.0

    watchdog = _make_watchdog(redis_client, tmp_path, exchange_ping=fake_ping)
    ok, latency_ms = await watchdog._ping_exchange()
    assert ok is True
    assert latency_ms is not None and latency_ms >= 10.0


@pytest.mark.asyncio
async def test_exchange_ping_failure_is_diagnostic_only_never_a_shutdown_trigger(
    redis_client: RedisClient, tmp_path: Path, exit_calls: list[int],
) -> None:
    async def failing_ping() -> None:
        raise ConnectionError("exchange REST API unreachable")

    watchdog = _make_watchdog(redis_client, tmp_path, exchange_ping=failing_ping)
    await watchdog._check_once()

    assert watchdog.last_vitals is not None
    assert watchdog.last_vitals.exchange_ok is False
    assert exit_calls == []  # module docstring: only Redis-unresponsive and RAM breach ever trigger shutdown


# =============================================================================
# Heartbeat file -- the real Docker-healthcheck consumer's contract
# =============================================================================


@pytest.mark.asyncio
async def test_healthy_tick_writes_a_fresh_heartbeat_file(redis_client: RedisClient, tmp_path: Path) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(redis_client, tmp_path, heartbeat_file=heartbeat_file)

    before = time.time()
    await watchdog._check_once()
    after = time.time()

    assert heartbeat_file.exists()
    written_at = float(heartbeat_file.read_text())
    assert before <= written_at <= after


# =============================================================================
# last_vitals -- a real, populated snapshot, never load-bearing itself
# =============================================================================


@pytest.mark.asyncio
async def test_last_vitals_is_none_before_the_first_tick(redis_client: RedisClient, tmp_path: Path) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path)
    assert watchdog.last_vitals is None


@pytest.mark.asyncio
async def test_last_vitals_is_populated_after_a_tick(redis_client: RedisClient, tmp_path: Path) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path)
    await watchdog._check_once()

    vitals = watchdog.last_vitals
    assert vitals is not None
    assert vitals.redis_ok is True
    assert vitals.process_rss_bytes > 0
    assert vitals.container_memory_limit_bytes == 10 * 1024 * 1024 * 1024


# =============================================================================
# start/stop -- real asyncio.Task lifecycle
# =============================================================================


@pytest.mark.asyncio
async def test_start_ticks_at_least_once_then_stop_cancels_cleanly(
    redis_client: RedisClient, tmp_path: Path,
) -> None:
    watchdog = _make_watchdog(redis_client, tmp_path, check_interval_seconds=0.05)
    task = watchdog.start()
    await asyncio.sleep(0.12)

    assert watchdog.last_vitals is not None
    await watchdog.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_a_bug_in_check_once_does_not_kill_the_watchdog_loop(
    redis_client: RedisClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop itself must survive its own checking logic misbehaving -- see module docstring."""
    watchdog = _make_watchdog(redis_client, tmp_path, check_interval_seconds=0.02)
    calls: list[int] = []

    async def flaky_check_once() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("simulated bug in vital-sign collection")

    monkeypatch.setattr(watchdog, "_check_once", flaky_check_once)
    task = watchdog.start()
    # Generous vs. check_interval_seconds: the first tick's _log.exception(...) renders a full
    # traceback (real, visible I/O), which can itself take longer than one short interval on a slow
    # CI runner -- this only needs to observe "more than one call ever happened," not a tight cadence.
    await asyncio.sleep(0.5)
    await watchdog.stop()

    assert len(calls) >= 2  # survived the first raise and kept ticking
    assert task.done()

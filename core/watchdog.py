from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Final

import psutil

from core.exception_manager import ShutdownCallback, graceful_shutdown
from core.logger import get_logger
from core.redis_client import RedisClient

_log = get_logger(__name__)

_DEFAULT_CHECK_INTERVAL_SECONDS: Final[float] = 10.0
_DEFAULT_REDIS_UNRESPONSIVE_THRESHOLD_SECONDS: Final[float] = 30.0
_DEFAULT_RAM_PERCENT_THRESHOLD: Final[float] = 85.0
_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 5.0
_DEFAULT_HEARTBEAT_FILE: Final[Path] = Path(tempfile.gettempdir()) / "nothierly_heartbeat.txt"

_CGROUP_V2_MEMORY_MAX: Final[Path] = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT: Final[Path] = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

ExchangePingCallback = Callable[[], Awaitable[Any]]


def _detect_container_memory_limit_bytes() -> int:
    host_total = psutil.virtual_memory().total
    for cgroup_path in (_CGROUP_V2_MEMORY_MAX, _CGROUP_V1_MEMORY_LIMIT):
        try:
            raw = cgroup_path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if 0 < limit < host_total:
            return limit
    _log.warning("watchdog_no_container_memory_limit_detected", host_total_bytes=host_total)
    return host_total


@dataclass(frozen=True)
class VitalSigns:
    timestamp: float
    redis_ok: bool
    redis_latency_ms: float | None
    redis_unresponsive_seconds: float
    exchange_ok: bool
    exchange_latency_ms: float | None
    process_rss_bytes: int
    container_memory_limit_bytes: int
    ram_percent_of_limit: float
    cpu_percent: float


class SystemWatchdog:
    def __init__(
        self,
        *,
        redis_client: RedisClient,
        shutdown_callback: ShutdownCallback,
        exchange_ping: ExchangePingCallback | None = None,
        heartbeat_file: Path | str = _DEFAULT_HEARTBEAT_FILE,
        check_interval_seconds: float = _DEFAULT_CHECK_INTERVAL_SECONDS,
        redis_unresponsive_threshold_seconds: float = _DEFAULT_REDIS_UNRESPONSIVE_THRESHOLD_SECONDS,
        ram_percent_threshold: float = _DEFAULT_RAM_PERCENT_THRESHOLD,
        container_memory_limit_bytes: int | None = None,
        shutdown_timeout_seconds: float = _DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        self._redis_client = redis_client
        self._shutdown_callback = shutdown_callback
        self._exchange_ping = exchange_ping
        self._heartbeat_file = Path(heartbeat_file)
        self._check_interval_seconds = check_interval_seconds
        self._redis_unresponsive_threshold_seconds = redis_unresponsive_threshold_seconds
        self._ram_percent_threshold = ram_percent_threshold
        self._container_memory_limit_bytes = container_memory_limit_bytes or _detect_container_memory_limit_bytes()
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(interval=None)
        self._redis_unresponsive_since: float | None = None
        self._shutdown_triggered = False
        self._task: asyncio.Task[None] | None = None
        self._last_vitals: VitalSigns | None = None

    @property
    def last_vitals(self) -> VitalSigns | None:
        return self._last_vitals

    def start(self) -> asyncio.Task[None]:
        self._task = asyncio.create_task(self._run(), name="system-watchdog")
        _log.info(
            "system_watchdog_started",
            check_interval_seconds=self._check_interval_seconds,
            redis_unresponsive_threshold_seconds=self._redis_unresponsive_threshold_seconds,
            ram_percent_threshold=self._ram_percent_threshold,
            container_memory_limit_bytes=self._container_memory_limit_bytes,
        )
        return self._task

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        while True:
            try:
                await self._check_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("system_watchdog_check_raised_unexpectedly")
            await asyncio.sleep(self._check_interval_seconds)

    async def _check_once(self) -> None:
        if self._shutdown_triggered:
            return

        redis_ok, redis_latency_ms = await self._ping_redis()
        if redis_ok:
            self._redis_unresponsive_since = None
        elif self._redis_unresponsive_since is None:
            self._redis_unresponsive_since = time.monotonic()
        redis_unresponsive_seconds = (
            time.monotonic() - self._redis_unresponsive_since if self._redis_unresponsive_since is not None else 0.0
        )

        exchange_ok, exchange_latency_ms = await self._ping_exchange()

        rss_bytes = self._process.memory_info().rss
        ram_percent = (rss_bytes / self._container_memory_limit_bytes) * 100.0
        cpu_percent = self._process.cpu_percent(interval=None)

        vitals = VitalSigns(
            timestamp=time.time(), redis_ok=redis_ok, redis_latency_ms=redis_latency_ms,
            redis_unresponsive_seconds=redis_unresponsive_seconds, exchange_ok=exchange_ok,
            exchange_latency_ms=exchange_latency_ms, process_rss_bytes=rss_bytes,
            container_memory_limit_bytes=self._container_memory_limit_bytes,
            ram_percent_of_limit=ram_percent, cpu_percent=cpu_percent,
        )
        self._last_vitals = vitals
        _log.info(
            "watchdog_vital_signs", redis_ok=redis_ok, redis_latency_ms=redis_latency_ms,
            redis_unresponsive_seconds=round(redis_unresponsive_seconds, 1), exchange_ok=exchange_ok,
            exchange_latency_ms=exchange_latency_ms, process_rss_bytes=rss_bytes,
            ram_percent_of_limit=round(ram_percent, 1), cpu_percent=cpu_percent,
        )

        if redis_unresponsive_seconds >= self._redis_unresponsive_threshold_seconds:
            await self._trigger_shutdown(
                f"Redis unresponsive for {redis_unresponsive_seconds:.0f}s "
                f"(threshold {self._redis_unresponsive_threshold_seconds}s)"
            )
            return
        if ram_percent >= self._ram_percent_threshold:
            await self._trigger_shutdown(
                f"process RAM at {ram_percent:.1f}% of the container's {self._container_memory_limit_bytes} byte "
                f"limit (threshold {self._ram_percent_threshold}%)"
            )
            return

        self._write_heartbeat_file()

    async def _ping_redis(self) -> tuple[bool, float | None]:
        started = time.monotonic()
        try:
            await asyncio.wait_for(self._redis_client.execute("ping"), timeout=self._check_interval_seconds)
        except Exception:
            return False, None
        return True, (time.monotonic() - started) * 1000.0

    async def _ping_exchange(self) -> tuple[bool, float | None]:
        if self._exchange_ping is None:
            return True, None
        started = time.monotonic()
        try:
            await asyncio.wait_for(self._exchange_ping(), timeout=self._check_interval_seconds)
        except Exception:
            return False, None
        return True, (time.monotonic() - started) * 1000.0

    def _write_heartbeat_file(self) -> None:
        payload = json.dumps({"timestamp": time.time(), "pid": os.getpid()})
        tmp_path = self._heartbeat_file.parent / f"{self._heartbeat_file.name}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._heartbeat_file)
        except OSError as exc:
            raise RuntimeError(f"failed to write heartbeat file {self._heartbeat_file}") from exc

    async def _trigger_shutdown(self, reason: str) -> None:
        if self._shutdown_triggered:
            return
        self._shutdown_triggered = True
        _log.critical("watchdog_triggering_graceful_shutdown", reason=reason)
        await graceful_shutdown(
            self._shutdown_callback, reason=f"watchdog: {reason}", timeout_seconds=self._shutdown_timeout_seconds,
        )

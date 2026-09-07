"""SystemWatchdog -- the resource-asphyxiation detector complementing core.exception_manager's crash detector.

**The real failure mode this closes, distinct from exception_manager's**:
a bot does not always die from an exception it could catch -- it often
dies from something no ``try``/``except`` anywhere in its own code can
see: the RAM it is holding growing past what the container is actually
allowed, or its state store going quietly unreachable while every
individual Redis call keeps timing out and getting retried forever by
whatever called it, never itself raising all the way to
:mod:`core.exception_manager`'s handler. :class:`SystemWatchdog` is an
independent task that checks real vital signs on a fixed interval and
triggers the exact same :func:`core.exception_manager.graceful_shutdown`
primitive on its own initiative, without waiting for something else to
throw.

**Real, load-bearing correction on "RAM du conteneur"**: ``psutil.
virtual_memory().percent`` reports the *host* machine's memory, not a
container's cgroup limit -- a well-known, easy-to-miss gap, confirmed by
reading ``src/mlops/host_metrics.py``'s own identical use of that call
for the *host* dashboard metric it explicitly documents itself as being
(that module is not imported here -- see the loose-coupling note below,
and it pulls in ``torch``, which this module must not). Inside a
1GB-limited container, host RAM could sit at 4% while this process alone
has already consumed the *entire* 1GB -- ``virtual_memory().percent``
would never cross 85% no matter how badly this process leaks, silently
defeating the one check meant to catch exactly that. This module instead
reads this **process's own RSS** (``psutil.Process().memory_info().rss``)
against the container's real configured limit, auto-detected from the
real cgroup v2 (``/sys/fs/cgroup/memory.max``) or v1
(``/sys/fs/cgroup/memory/memory.limit_in_bytes``) file -- exactly the
number ``docker-compose.yml``'s own ``deploy.resources.limits.memory``
(see that file) sets -- falling back to the host total, with an explicit
warning, only on a platform with neither (e.g. local Windows/macOS dev,
not a real deployment target for this check).

**Loose coupling (this mutation's own constraint)**: this module imports
:class:`~core.redis_client.RedisClient` (core infra, not "the AI") for
the Redis vital sign, and takes an arbitrary ``exchange_ping: async def
() -> Any`` callback for the exchange vital sign -- it never imports
``ccxt``, a broker adapter, or any ``src.agents.*``/``src.execution.*``
module. The caller decides what "ping the exchange" means (a real
``get_current_price`` call, a lightweight REST ``fetch_time``, ...); this
module only times it and treats any exception as a failed ping.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
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
_DEFAULT_HEARTBEAT_FILE: Final[Path] = Path("/tmp/nothierly_heartbeat.txt")

_CGROUP_V2_MEMORY_MAX: Final[Path] = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MEMORY_LIMIT: Final[Path] = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

ExchangePingCallback = Callable[[], Awaitable[Any]]


def _detect_container_memory_limit_bytes() -> int:
    """Real cgroup-reported memory limit for this container, or the host total as an honest, logged fallback. See module docstring."""
    host_total = psutil.virtual_memory().total
    for cgroup_path in (_CGROUP_V2_MEMORY_MAX, _CGROUP_V1_MEMORY_LIMIT):
        try:
            raw = cgroup_path.read_text().strip()
        except OSError:
            continue
        if raw == "max":
            continue  # cgroup v2's own literal spelling of "no limit set" -- try the next path/fallback
        try:
            limit = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports an astronomically large sentinel (not "max") when unset -- always >= the
        # real host total, so this comparison correctly treats it as "no real limit" too.
        if 0 < limit < host_total:
            return limit
    _log.warning(
        "watchdog_no_container_memory_limit_detected",
        detail="Neither cgroup v2 nor v1 reported a real limit smaller than host RAM -- falling back "
        "to host total. Expected on local/non-container dev; a real deployment should have "
        "docker-compose.yml's deploy.resources.limits.memory produce a real cgroup limit here.",
        host_total_bytes=host_total,
    )
    return host_total


@dataclass(frozen=True)
class VitalSigns:
    """One real, timestamped snapshot -- exactly what was checked and measured, nothing inferred."""

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
    """Ticks every ``check_interval_seconds``, checks 3 real vital signs, triggers a real graceful shutdown if any breach its threshold."""

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
        """
        Args:
            redis_client: Real, already-connected state-store client --
                pinged every tick via ``PING``.
            shutdown_callback: Forwarded to :func:`~core.exception_manager.graceful_shutdown`
                the moment a threshold is breached.
            exchange_ping: Optional real REST call to time (e.g.
                ``lambda: broker.get_current_price(symbol)``). ``None``
                (default) skips this vital sign entirely -- a process
                with no live exchange connection (e.g. this watchdog
                running inside a signal-only agent) has nothing real to
                ping. Never itself a shutdown trigger (see module
                docstring: only Redis-unresponsive and RAM breach are).
            heartbeat_file: Touched with the current time on every
                healthy tick -- see ``docker-compose.yml``'s own
                healthcheck for the real consumer of this file.
            check_interval_seconds: How often to check. Also the ceiling
                each individual ping is allowed before counting as
                failed for that tick (never lets one slow ping delay the
                next scheduled check).
            redis_unresponsive_threshold_seconds: Consecutive-failure
                duration, not "the last ping failed" -- one transient
                blip must not trip a shutdown.
            ram_percent_threshold: Of the *container's* real limit (see
                module docstring), not host RAM.
            container_memory_limit_bytes: Override for
                :func:`_detect_container_memory_limit_bytes` -- mainly
                for tests; production should let this auto-detect.
            shutdown_timeout_seconds: Forwarded to
                :func:`~core.exception_manager.graceful_shutdown`.
        """
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
        self._process.cpu_percent(interval=None)  # first call always returns 0.0/garbage -- primes the baseline
        self._redis_unresponsive_since: float | None = None
        self._shutdown_triggered = False
        self._task: asyncio.Task[None] | None = None
        self._last_vitals: VitalSigns | None = None

    @property
    def last_vitals(self) -> VitalSigns | None:
        """Most recent snapshot, or ``None`` before the first tick -- for a dashboard/telemetry reader, never load-bearing itself."""
        return self._last_vitals

    def start(self) -> asyncio.Task[None]:
        """Schedule the watchdog loop as a background task. Call once, after ``redis_client`` is connected."""
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
        """Cancel the loop -- a normal, intentional shutdown, never itself triggers graceful_shutdown."""
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
                # The watchdog itself must never die from a bug in its own checking logic -- that would
                # be exactly the "silent death" this whole mutation exists to prevent, just relocated
                # into the safety mechanism itself.
                _log.exception("system_watchdog_check_raised_unexpectedly")
            await asyncio.sleep(self._check_interval_seconds)

    async def _check_once(self) -> None:
        if self._shutdown_triggered:
            return  # graceful_shutdown() is already running towards os._exit() -- nothing left to check

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
            # Deliberately broad -- see module docstring: any failure to complete a PING within one
            # check interval means "not healthy," regardless of the underlying exception type.
            return False, None
        return True, (time.monotonic() - started) * 1000.0

    async def _ping_exchange(self) -> tuple[bool, float | None]:
        if self._exchange_ping is None:
            return True, None  # no exchange configured to check -- not a failure, just nothing to report
        started = time.monotonic()
        try:
            await asyncio.wait_for(self._exchange_ping(), timeout=self._check_interval_seconds)
        except Exception:
            return False, None
        return True, (time.monotonic() - started) * 1000.0

    def _write_heartbeat_file(self) -> None:
        try:
            self._heartbeat_file.write_text(str(time.time()))
        except OSError:
            # A filesystem hiccup writing the heartbeat must not crash the watchdog -- the file simply
            # goes stale, which is exactly the real, honest signal docker-compose.yml's healthcheck
            # should act on (a heartbeat that silently kept refreshing on a write failure would hide
            # a real problem instead of surfacing one).
            _log.warning("watchdog_heartbeat_file_write_failed", path=str(self._heartbeat_file))

    async def _trigger_shutdown(self, reason: str) -> None:
        if self._shutdown_triggered:
            return
        self._shutdown_triggered = True
        _log.critical("watchdog_triggering_graceful_shutdown", reason=reason)
        await graceful_shutdown(
            self._shutdown_callback, reason=f"watchdog: {reason}", timeout_seconds=self._shutdown_timeout_seconds,
        )

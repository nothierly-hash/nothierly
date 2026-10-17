"""In-process memory-drift monitor for the Chaos Shadow Run (Directive 4).

Two complementary readings, both opt-in and both off the hot path:

- ``tracemalloc`` for *what* is allocating (top allocation sites by
  traced Python memory), useful for diagnosing a real leak once one is
  suspected.
- ``resource``/``psutil``-free RSS via ``/proc`` (Linux) or
  ``ctypes``+``psutil`` fallback for *how much* real process memory is
  used -- tracemalloc only ever sees Python-allocated objects, not
  native extensions (``ccxt``, ``zmq``, ``pydantic-core``) that make up
  most of this project's real footprint, so RSS is the number the
  mission's own 5%-drift criterion actually means.

A drift check needs a stable baseline: the first sample (taken after
``baseline_delay_seconds`` lets startup allocations settle) is the
denominator every later sample is compared against.
"""

from __future__ import annotations

import asyncio
import tracemalloc
from typing import Final

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_INTERVAL_SECONDS: Final[float] = 300.0
_DEFAULT_BASELINE_DELAY_SECONDS: Final[float] = 60.0
_DEFAULT_DRIFT_THRESHOLD_PCT: Final[float] = 5.0
_TOP_ALLOCATION_SITES: Final[int] = 5


def _current_rss_bytes() -> int | None:
    """Best-effort current process RSS in bytes, or ``None`` if unavailable on this platform.

    Tries ``psutil`` first (already a real dependency of ``core.watchdog``
    elsewhere in this project); falls back to reading
    ``/proc/self/status`` directly on Linux if ``psutil`` is not
    importable in this environment, rather than raising and killing the
    monitor task over an optional reading.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


async def _memory_monitor_loop(
    *,
    sender_id: str,
    interval_seconds: float,
    baseline_delay_seconds: float,
    drift_threshold_pct: float,
) -> None:
    tracemalloc.start()
    await asyncio.sleep(baseline_delay_seconds)

    baseline_rss = _current_rss_bytes()
    baseline_traced_current, _ = tracemalloc.get_traced_memory()
    _log.info(
        "chaos_memory_baseline_captured",
        sender_id=sender_id,
        baseline_rss_bytes=baseline_rss,
        baseline_traced_python_bytes=baseline_traced_current,
    )

    while True:
        await asyncio.sleep(interval_seconds)

        rss = _current_rss_bytes()
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")[:_TOP_ALLOCATION_SITES]

        rss_drift_pct = (
            (rss - baseline_rss) / baseline_rss * 100.0 if baseline_rss and rss is not None else None
        )
        _log.info(
            "chaos_memory_snapshot",
            sender_id=sender_id,
            rss_bytes=rss,
            rss_drift_pct=round(rss_drift_pct, 3) if rss_drift_pct is not None else None,
            traced_python_current_bytes=traced_current,
            traced_python_peak_bytes=traced_peak,
            top_allocations=[str(stat) for stat in top_stats],
        )

        if rss_drift_pct is not None and abs(rss_drift_pct) >= drift_threshold_pct:
            _log.warning(
                "chaos_memory_leak_suspected",
                sender_id=sender_id,
                rss_bytes=rss,
                baseline_rss_bytes=baseline_rss,
                rss_drift_pct=round(rss_drift_pct, 3),
                threshold_pct=drift_threshold_pct,
            )


def start_memory_monitor(
    *,
    sender_id: str,
    interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
    baseline_delay_seconds: float = _DEFAULT_BASELINE_DELAY_SECONDS,
    drift_threshold_pct: float = _DEFAULT_DRIFT_THRESHOLD_PCT,
) -> asyncio.Task[None]:
    """Start the in-process RSS/tracemalloc monitor as a background task on the running loop.

    Same ownership contract as
    :func:`core.chaos.loop_lag_monitor.start_loop_lag_monitor`: call from
    inside a running loop, cancel the returned task on shutdown.

    Args:
        sender_id: Attached to every log line.
        interval_seconds: How often to snapshot (default 5 minutes --
            frequent enough to resolve a 24h run into ~288 points,
            without tracemalloc's own snapshot overhead dominating a
            latency-sensitive process).
        baseline_delay_seconds: Startup grace period before the
            first (baseline) sample, so one-time init allocations
            (market metadata load, model loading, ...) are not
            misread as drift.
        drift_threshold_pct: Absolute RSS drift from baseline, in
            percent, that logs ``chaos_memory_leak_suspected``. The
            mission brief's own default (5%).
    """
    return asyncio.ensure_future(
        _memory_monitor_loop(
            sender_id=sender_id,
            interval_seconds=interval_seconds,
            baseline_delay_seconds=baseline_delay_seconds,
            drift_threshold_pct=drift_threshold_pct,
        )
    )

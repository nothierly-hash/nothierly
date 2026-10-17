"""Asyncio event-loop lag watchdog for the Chaos Shadow Run.

Detects the loop itself falling behind under saturation (e.g. a burst of
ZMQ messages, or synchronous work slipping into an async callback) -- a
different failure mode than a slow *await* (network/Redis latency,
already covered by this project's own watchdog/circuit-breaker
machinery). A lagging loop delays every task in the process equally,
including the ones responsible for detecting other failures, so it is
checked independently, on a bare ``asyncio.sleep`` with no dependency on
anything else in the process.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_PROBE_INTERVAL_SECONDS: Final[float] = 0.05
_DEFAULT_LAG_THRESHOLD_MS: Final[float] = 25.0


async def _lag_probe_loop(
    *, sender_id: str, probe_interval_seconds: float, lag_threshold_ms: float
) -> None:
    """Sleep for ``probe_interval_seconds`` in a tight loop, logging whenever the real elapsed time overshoots it.

    ``asyncio.sleep(interval)`` returns no earlier than ``interval`` but
    can return arbitrarily later if the loop is busy running other
    callbacks/tasks first -- that overshoot, in excess of
    ``probe_interval_seconds``, *is* the loop's real scheduling lag.
    """
    last = time.monotonic()
    while True:
        await asyncio.sleep(probe_interval_seconds)
        now = time.monotonic()
        observed_seconds = now - last
        last = now
        lag_ms = (observed_seconds - probe_interval_seconds) * 1000.0
        if lag_ms >= lag_threshold_ms:
            _log.warning(
                "chaos_event_loop_lag_bottleneck",
                sender_id=sender_id,
                lag_ms=round(lag_ms, 2),
                threshold_ms=lag_threshold_ms,
                probe_interval_ms=round(probe_interval_seconds * 1000.0, 2),
            )


def start_loop_lag_monitor(
    *,
    sender_id: str,
    lag_threshold_ms: float = _DEFAULT_LAG_THRESHOLD_MS,
    probe_interval_seconds: float = _DEFAULT_PROBE_INTERVAL_SECONDS,
) -> asyncio.Task[None]:
    """Start the lag probe as a background task on the *currently running* loop.

    Must be called from inside a running event loop (e.g. an agent's own
    ``on_start``, or a launcher script's ``main()`` right after
    ``asyncio.run`` begins). The caller owns the returned task's
    lifecycle -- cancel it on shutdown like any other background task;
    it never exits on its own.

    Args:
        sender_id: Attached to every log line, so a shared log file
            (e.g. several chaos-tested agents piped into one file) can
            tell whose loop lagged.
        lag_threshold_ms: Overshoot, in milliseconds, above which a
            ``chaos_event_loop_lag_bottleneck`` warning fires. The
            mission brief's own default (25ms).
        probe_interval_seconds: How often the probe sleeps. Deliberately
            short (50ms default) relative to the lag threshold so the
            probe itself can observe sub-threshold jitter for context,
            not just gross stalls.
    """
    return asyncio.ensure_future(
        _lag_probe_loop(
            sender_id=sender_id,
            probe_interval_seconds=probe_interval_seconds,
            lag_threshold_ms=lag_threshold_ms,
        )
    )

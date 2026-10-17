"""Staleness rejection for a timestamped market reading.

Real gap this closes (found by ``tests/chaos_monkey.py``'s Chaos 3): :class:`~core.schemas.
MarketMetrics` carries its own real capture ``timestamp``, but nothing anywhere compared
it to "now" -- staleness was handled only by the Redis key's TTL (10s), so a reading that
is merely *present* was treated as fresh even if it was seconds old. This mirrors the one
real staleness check that already exists elsewhere for a *different* schema --
:meth:`~src.agents.execution.agent.ExecutionAgent._evaluate_consensus`'s ``TradingSignal``
freshness check (``age_seconds <= signal_validity_seconds``) -- extended here to
``MarketMetrics``.

500ms is deliberately aggressive, matching the brief this responds to -- real WebSocket/
exchange latency can legitimately exceed this under ordinary network jitter, so tune
``max_age_ms`` per deployment/instrument rather than treating 500 as universally correct:
too tight a threshold trades false positives (rejecting genuinely fresh data, itself a form
of self-inflicted denial of trading) for a tighter staleness guarantee.
"""

from __future__ import annotations

from datetime import datetime, timezone


def staleness_reason(reading_timestamp: datetime, *, max_age_ms: float = 500.0, now: datetime | None = None) -> str | None:
    """Return a reason string if ``reading_timestamp`` is older than ``max_age_ms``, else ``None``.

    Args:
        reading_timestamp: The real capture time carried on the reading
            itself (e.g. ``MarketMetrics.timestamp``) -- never "whenever
            this function happens to run", which would defeat the point.
        max_age_ms: Maximum allowed age, in milliseconds.
        now: Injectable clock for tests. Defaults to
            ``datetime.now(timezone.utc)``.
    """
    now = now or datetime.now(timezone.utc)
    age_ms = (now - reading_timestamp).total_seconds() * 1000.0
    if age_ms > max_age_ms:
        return f"reading is {age_ms:.1f}ms old, exceeding the {max_age_ms:.1f}ms staleness threshold"
    return None

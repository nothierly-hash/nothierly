"""Shared, resilient ``asyncpg`` connection-pool factory for TimescaleDB.

Every TimescaleDB caller in this project -- :class:`~src.agents.
archivist.agent.ArchivistAgent` and, as of the data pipeline chapter,
every module under :mod:`src.data_pipeline` -- needs its own connection
pool (``asyncpg.Pool`` is not safely shared across independent processes,
and even within one process each long-lived component owns its own pool
lifecycle) but the *same* resilience story: retry with exponential
backoff and jitter rather than failing hard the moment TimescaleDB is
merely still starting up (a very real race during ``docker-compose up``,
where the database container can take several seconds after "started"
before it's actually accepting connections).

Deliberately plain ``asyncpg``, not SQLAlchemy: every query this project
issues against TimescaleDB is either a bulk time-series insert
(``copy_records_to_table``) or a straightforward ``SELECT`` over a
hypertable -- exactly what :class:`~src.agents.archivist.agent.
ArchivistAgent` already proved out with raw ``asyncpg`` alone. An ORM
would add a translation layer and a second dependency for no capability
this project's actual query shapes need.
"""

from __future__ import annotations

import asyncio
import random
from typing import Final

import asyncpg

from core.config import TimescaleSettings, get_settings
from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_BASE_DELAY_SECONDS: Final[float] = 1.0
_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 30.0


class TimescaleUnavailableError(Exception):
    """Raised when a bounded connection attempt (``max_attempts`` set) is exhausted.

    Never raised with the default, unbounded retry policy -- that one
    keeps trying until it succeeds or the caller is cancelled, the same
    "a slow dependency should not prevent eventual startup" reasoning
    :class:`~core.redis_client.RedisClient` already uses for Redis.
    """


async def connect_with_backoff(
    settings: TimescaleSettings | None = None,
    *,
    base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = _DEFAULT_MAX_DELAY_SECONDS,
    max_attempts: int | None = None,
    min_pool_size: int = 1,
    max_pool_size: int = 5,
) -> asyncpg.Pool:
    """Create an ``asyncpg`` connection pool, retrying with backoff on failure.

    Args:
        settings: Defaults to ``get_settings().timescale``.
        base_delay_seconds: Initial retry delay.
        max_delay_seconds: Delay ceiling; backoff doubles each attempt up
            to this cap.
        max_attempts: ``None`` (the default) retries forever -- correct
            for a long-lived agent/service starting up alongside
            ``docker-compose``. A finite value is for callers that
            should fail fast instead (e.g. a one-shot script the user
            runs by hand and wants an immediate, actionable error from --
            see :mod:`src.data_pipeline.collectors.crypto_collector`).
        min_pool_size: Passed to ``asyncpg.create_pool``.
        max_pool_size: Passed to ``asyncpg.create_pool``.

    Returns:
        A connected, ready-to-use pool.

    Raises:
        TimescaleUnavailableError: If ``max_attempts`` is set and every
            attempt fails.
    """
    resolved_settings = settings or get_settings().timescale
    delay = base_delay_seconds
    attempt = 0

    while True:
        try:
            return await asyncpg.create_pool(
                dsn=resolved_settings.connection_url,
                min_size=min_pool_size,
                max_size=max_pool_size,
            )
        except (OSError, asyncpg.PostgresError) as exc:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                raise TimescaleUnavailableError(
                    f"TimescaleDB unreachable after {attempt} attempt(s): {exc}"
                ) from exc
            jittered_delay = delay * random.uniform(0.5, 1.5)
            _log.warning(
                "timescale_connect_retry",
                attempt=attempt,
                delay_seconds=round(jittered_delay, 2),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            await asyncio.sleep(jittered_delay)
            delay = min(delay * 2.0, max_delay_seconds)

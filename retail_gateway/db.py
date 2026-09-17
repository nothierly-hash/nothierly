"""Self-contained ``asyncpg`` pool factory for the Retail Gateway.

**Real, disclosed duplication, not laziness**: ``core/timescale_client.py`` already has an
almost identical retry-with-backoff pool factory. It is deliberately not imported here --
doing so would make this microservice's own startup depend on importing ``core.config``
(that module's ``TimescaleSettings``), reintroducing exactly the process/package coupling the
mission's air-gap rule forbids. The backoff algorithm (jittered exponential, capped) is the
same proven shape; the two connection targets are different databases entirely.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Final

import asyncpg

from retail_gateway.config import DatabaseSettings
from retail_gateway.logging_config import get_logger

_log = get_logger(__name__)

_SCHEMA_PATH: Final[Path] = Path(__file__).with_name("schema.sql")

_DEFAULT_BASE_DELAY_SECONDS: Final[float] = 1.0
_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 30.0


class RetailDatabaseUnavailableError(Exception):
    """Raised when a bounded connection attempt (``max_attempts`` set) is exhausted."""


async def connect_with_backoff(
    settings: DatabaseSettings,
    *,
    base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = _DEFAULT_MAX_DELAY_SECONDS,
    max_attempts: int | None = None,
) -> asyncpg.Pool:
    """Create an ``asyncpg`` connection pool, retrying with jittered backoff on failure.

    ``max_attempts=None`` (the default) retries forever -- correct for this service's own
    container starting up alongside ``docker-compose`` before Postgres is ready to accept
    connections yet.
    """
    delay = base_delay_seconds
    attempt = 0
    while True:
        try:
            return await asyncpg.create_pool(
                dsn=settings.connection_url,
                min_size=settings.min_pool_size,
                max_size=settings.max_pool_size,
            )
        except (OSError, asyncpg.PostgresError) as exc:
            attempt += 1
            if max_attempts is not None and attempt >= max_attempts:
                raise RetailDatabaseUnavailableError(
                    f"Retail Postgres unreachable after {attempt} attempt(s): {exc}"
                ) from exc
            jittered_delay = delay * random.uniform(0.5, 1.5)
            _log.warning(
                "retail_db_connect_retry",
                attempt=attempt,
                delay_seconds=round(jittered_delay, 2),
                error_type=type(exc).__name__,
            )
            await asyncio.sleep(jittered_delay)
            delay = min(delay * 2.0, max_delay_seconds)


async def init_schema(pool: asyncpg.Pool) -> None:
    """Apply ``schema.sql`` idempotently (every statement is ``CREATE ... IF NOT EXISTS``)."""
    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(ddl)
    _log.info("retail_db_schema_ready")

"""Chapter 13, Étape 5.7: the ``client_fills`` hypertable -- genuinely new, not an extension of an existing table.

**A real, load-bearing correction to the brief, stated upfront**: "THE
BILLING JUSTIFIER" assumes a queryable Postgres/Polars database of
per-client transactions already exists ("interroge la base de données
... pour extraire toutes les transactions d'un client_id"). Verified by
direct inspection before writing any of this: it does not.
:class:`~src.agents.archivist.agent.ArchivistAgent` writes every bus
message into one generic, undifferentiated ``bus_messages`` table
(``time``, ``topic``, ``sender_id``, an opaque ``data`` JSONB blob) with
no ``client_id`` column and no client-indexed query path --
:class:`~core.schemas.TradeExecution` (the ``TRADE_EXECUTED`` payload)
doesn't even have a ``client_id`` field to put in that JSONB blob. This
module adds the table that makes a per-client monthly report possible
at all: ``client_fills``, one row per real (or Shadow-simulated) fill,
keyed by ``(client_id, time)`` -- the same ``(entity, time)`` hypertable
shape :mod:`src.data_pipeline.schema` already established for
``ohlcv_candles``/``macro_indicators``/``computed_features``, applied to
a genuinely new entity this project never tracked before.

**Written by**: ``api_middleware/gateway.py``'s ``_write_shadow_fill``
(Shadow-mode simulated fills) and, since
:class:`~src.agents.macro_order_execution.agent.MacroOrderExecutionAgent`
closed the real gap this docstring used to describe (LIVE orders had no
execution consumer at all), real client-attributed LIVE fills too --
see that module's own docstring.

**``broker_id``, added alongside the real LIVE writer above**: ``NULL``
for every Shadow-mode row (no real broker is ever involved -- see
``_run_shadow_simulation``'s own docstring), and the real venue
(``"binance"``, ``"okx"``, ...) a LIVE child order actually executed on
for every row :class:`MacroOrderExecutionAgent` writes. This is what
lets :func:`~src.execution.eod_reconciliator.run_eod_reconciliation`
attribute a real day-over-day broker balance ``Break`` to the specific
client fills recorded against that same broker in the same window --
the real, previously-missing half of "why is this broker's balance off,"
not just "that it is."
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final

import asyncpg

from core.logger import get_logger

_log = get_logger(__name__)

CLIENT_FILLS_TABLE: Final[str] = "client_fills"

_CREATE_CLIENT_FILLS_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {CLIENT_FILLS_TABLE} (
    time                TIMESTAMPTZ NOT NULL,
    client_id           TEXT NOT NULL,
    order_id            TEXT NOT NULL,
    instrument          TEXT NOT NULL,
    side                TEXT NOT NULL,
    execution_algo      TEXT NOT NULL,
    execution_mode      TEXT NOT NULL,
    quantity            DOUBLE PRECISION NOT NULL,
    price               DOUBLE PRECISION NOT NULL,
    fee_usd             DOUBLE PRECISION,
    slippage_saved_usd  DOUBLE PRECISION,
    broker_id           TEXT,
    PRIMARY KEY (client_id, time, order_id)
);
"""
# Additive, idempotent migration for a table created before broker_id
# existed -- CREATE TABLE IF NOT EXISTS above is a no-op against an
# already-existing table, so a real pre-existing deployment needs this
# separate ALTER to actually gain the new column. Safe to run on every
# startup (IF NOT EXISTS), same convention as the hypertable/index calls
# below.
_ALTER_ADD_BROKER_ID_SQL: Final[str] = f"ALTER TABLE {CLIENT_FILLS_TABLE} ADD COLUMN IF NOT EXISTS broker_id TEXT;"
_CREATE_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{CLIENT_FILLS_TABLE}_client_time "
    f"ON {CLIENT_FILLS_TABLE} (client_id, time DESC);"
)
_CREATE_BROKER_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{CLIENT_FILLS_TABLE}_broker_time "
    f"ON {CLIENT_FILLS_TABLE} (broker_id, time DESC) WHERE broker_id IS NOT NULL;"
)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create ``client_fills``/its hypertable/its index, and ``client_monthly_pnl``, if they don't already exist.

    Idempotent, same graceful hypertable-extension-missing fallback as
    :func:`src.data_pipeline.schema.ensure_schema` -- safe to call on
    every gateway startup.
    """
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_CLIENT_FILLS_SQL)
        await conn.execute(_ALTER_ADD_BROKER_ID_SQL)
        try:
            await conn.execute(f"SELECT create_hypertable('{CLIENT_FILLS_TABLE}', 'time', if_not_exists => TRUE);")
        except asyncpg.PostgresError as exc:
            _log.warning(
                "billing_hypertable_creation_failed",
                table=CLIENT_FILLS_TABLE,
                error=str(exc),
                detail="the TimescaleDB extension may not be installed on this database; falling back to a plain table",
            )
        await conn.execute(_CREATE_INDEX_SQL)
        await conn.execute(_CREATE_BROKER_INDEX_SQL)
    await ensure_client_monthly_pnl_schema(pool)


CLIENT_MONTHLY_PNL_TABLE: Final[str] = "client_monthly_pnl"

# Deliberately a plain table, not a hypertable like client_fills above:
# this holds one row per (client, month) -- a monthly report *input* the
# client self-reports, not a time-series stream to partition by time.
# See src/execution/reporting_engine.py's module docstring on why this
# table exists at all: Nothierly never takes directional positions for a
# client, so it has no way to know a client's own trading outcome unless
# the client tells it -- this is that real, honest channel, never a
# number this system computes or guesses on the client's behalf.
_CREATE_CLIENT_MONTHLY_PNL_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {CLIENT_MONTHLY_PNL_TABLE} (
    client_id       TEXT NOT NULL,
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    gross_pnl_usd   DOUBLE PRECISION NOT NULL,
    reported_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (client_id, year, month)
);
"""


async def ensure_client_monthly_pnl_schema(pool: asyncpg.Pool) -> None:
    """Create ``client_monthly_pnl`` if it doesn't already exist. Idempotent, safe on every startup."""
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_CLIENT_MONTHLY_PNL_SQL)


async def upsert_client_monthly_pnl(
    pool: asyncpg.Pool, *, client_id: str, year: int, month: int, gross_pnl_usd: float
) -> None:
    """Record (or overwrite) one client's self-reported gross PnL for one calendar month.

    A client can correct a previously-reported figure -- overwrites
    rather than appending, since there is exactly one true value per
    ``(client_id, year, month)``, not a history of revisions.
    """
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {CLIENT_MONTHLY_PNL_TABLE} (client_id, year, month, gross_pnl_usd, reported_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (client_id, year, month)
            DO UPDATE SET gross_pnl_usd = EXCLUDED.gross_pnl_usd, reported_at = EXCLUDED.reported_at
            """,
            client_id, year, month, gross_pnl_usd, datetime.now(timezone.utc),
        )


async def get_client_monthly_pnl(pool: asyncpg.Pool, *, client_id: str, year: int, month: int) -> float | None:
    """The client's real, self-reported gross PnL for this month, or ``None`` if never reported."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT gross_pnl_usd FROM {CLIENT_MONTHLY_PNL_TABLE} WHERE client_id = $1 AND year = $2 AND month = $3",
            client_id, year, month,
        )
    return float(row["gross_pnl_usd"]) if row is not None else None


async def insert_client_fill(pool: asyncpg.Pool, **fields: Any) -> None:
    """Insert one real client fill row.

    Args:
        pool: A connected pool (from
            :func:`core.timescale_client.connect_with_backoff`).
        **fields: ``time, client_id, order_id, instrument, side,
            execution_algo, execution_mode, quantity, price`` (required)
            plus ``fee_usd, slippage_saved_usd, broker_id`` (optional,
            ``NULL`` if omitted -- ``broker_id`` is only ever real for a
            LIVE fill; see module docstring).
    """
    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            INSERT INTO {CLIENT_FILLS_TABLE}
                (time, client_id, order_id, instrument, side, execution_algo, execution_mode, quantity, price, fee_usd, slippage_saved_usd, broker_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            fields["time"], fields["client_id"], fields["order_id"], fields["instrument"], fields["side"],
            fields["execution_algo"], fields["execution_mode"], fields["quantity"], fields["price"],
            fields.get("fee_usd"), fields.get("slippage_saved_usd"), fields.get("broker_id"),
        )

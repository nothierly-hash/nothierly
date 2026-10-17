"""SQL schema for the data pipeline's three TimescaleDB hypertables.

Three tables, each a TimescaleDB **hypertable** (automatic time-based
partitioning into "chunks" -- the entire reason to use TimescaleDB over
plain Postgres for a "25 years of history" scale dataset: queries that
filter by a recent time range only ever touch the chunks that overlap
that range, instead of scanning the whole table):

``ohlcv_candles``
    Raw OHLCV candles from :mod:`~src.data_pipeline.collectors.
    crypto_collector`. One row per ``(instrument, time)``.

``macro_indicators``
    Raw macro/economic readings from :mod:`~src.data_pipeline.collectors.
    macro_collector`. One row per ``(indicator, time)``.

``computed_features``
    :mod:`~src.data_pipeline.feature_engine`'s output, persisted so the
    Vectorized Backtest Engine (a later chapter) can replay years of
    history without recomputing every indicator from raw candles on
    every run. One row per ``(instrument, time)``.

All three share the same primary-key shape -- ``(entity, time)`` -- which
is what makes each one a natural hypertable candidate: TimescaleDB
partitions on the time column, and the entity column is the first part of
a composite index that makes "give me instrument X's last N months"
queries (every consumer this project has -- the Feature Store, the
Backtest Engine) an index range scan, not a full-table filter.
"""

from __future__ import annotations

from typing import Final

import asyncpg

from core.logger import get_logger

_log = get_logger(__name__)

OHLCV_TABLE: Final[str] = "ohlcv_candles"
MACRO_TABLE: Final[str] = "macro_indicators"
FEATURES_TABLE: Final[str] = "computed_features"

_CREATE_OHLCV_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {OHLCV_TABLE} (
    time        TIMESTAMPTZ NOT NULL,
    instrument  TEXT NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (instrument, time)
);
"""
_CREATE_MACRO_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {MACRO_TABLE} (
    time        TIMESTAMPTZ NOT NULL,
    indicator   TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (indicator, time)
);
"""
_CREATE_FEATURES_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {FEATURES_TABLE} (
    time        TIMESTAMPTZ NOT NULL,
    instrument  TEXT NOT NULL,
    ema_fast    DOUBLE PRECISION,
    ema_slow    DOUBLE PRECISION,
    rsi         DOUBLE PRECISION,
    bb_upper    DOUBLE PRECISION,
    bb_lower    DOUBLE PRECISION,
    atr         DOUBLE PRECISION,
    PRIMARY KEY (instrument, time)
);
"""

# One (table, hypertable-conversion, index) triple per table -- see the
# module docstring on why every table shares this exact shape.
_TABLES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        OHLCV_TABLE,
        _CREATE_OHLCV_SQL,
        f"CREATE INDEX IF NOT EXISTS idx_{OHLCV_TABLE}_instrument_time "
        f"ON {OHLCV_TABLE} (instrument, time DESC);",
    ),
    (
        MACRO_TABLE,
        _CREATE_MACRO_SQL,
        f"CREATE INDEX IF NOT EXISTS idx_{MACRO_TABLE}_indicator_time "
        f"ON {MACRO_TABLE} (indicator, time DESC);",
    ),
    (
        FEATURES_TABLE,
        _CREATE_FEATURES_SQL,
        f"CREATE INDEX IF NOT EXISTS idx_{FEATURES_TABLE}_instrument_time "
        f"ON {FEATURES_TABLE} (instrument, time DESC);",
    ),
)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create all three tables/hypertables/indexes if they don't already exist.

    Idempotent (every statement is ``IF NOT EXISTS``) -- safe to call on
    every process startup, exactly like :class:`~src.agents.archivist.
    agent.ArchivistAgent`'s own ``_ensure_schema`` does for its table.
    Falls back to a plain (non-hypertable) table with a warning if the
    ``timescaledb`` extension isn't installed on the target database,
    rather than failing outright -- the same graceful-degradation choice
    ``ArchivistAgent`` already makes.
    """
    async with pool.acquire() as conn:
        for table_name, create_table_sql, create_index_sql in _TABLES:
            await conn.execute(create_table_sql)
            try:
                await conn.execute(
                    f"SELECT create_hypertable('{table_name}', 'time', if_not_exists => TRUE);"
                )
            except asyncpg.PostgresError as exc:
                _log.warning(
                    "data_pipeline_hypertable_creation_failed",
                    table=table_name,
                    error=str(exc),
                    detail=(
                        "the TimescaleDB extension may not be installed on this "
                        "database; falling back to a plain (non-hypertable) table"
                    ),
                )
            await conn.execute(create_index_sql)

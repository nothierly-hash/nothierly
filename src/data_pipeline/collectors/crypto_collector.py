"""Historical + ongoing OHLCV ingestion into TimescaleDB, via ccxt (REST), never ZMQ.

Deliberately **not** a :class:`~core.base_agent.BaseAgent`: this is an
ingestion worker, not a trading-bus participant. It never publishes,
subscribes, or touches Redis -- it only talks to an exchange's REST API
and to TimescaleDB. Scheduling is :mod:`apscheduler`, not this project's
usual asyncio-sleep-loop pattern, per this chapter's explicit choice
(matches :class:`~src.agents.macro.agent.MacroAgent`'s scheduling *intent*
-- "run on a timer, independent of the trading loop" -- without coupling
this collector to ``BaseAgent``'s ZMQ/Redis lifecycle it has no use for).

Run standalone with::

    python -m src.data_pipeline.collectors.crypto_collector

**A real, honest caveat on "millions of rows"**: most exchanges' public
REST APIs do not retain minute-level OHLCV history for anywhere near 5
years (Binance, for one, only serves recent 1-minute candles) -- multi-
year backfills are realistically only available at coarser timeframes
(this collector defaults to ``"1h"``). Five years of hourly candles for
one instrument is on the order of 43,800 rows, not millions; reaching
millions of rows requires either finer granularity than most exchanges
will actually serve that far back, or many instruments accumulated over
time via the live polling path below. Said plainly rather than silently
implied.
"""

from __future__ import annotations

import asyncio
import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Any, Final

import asyncpg
import ccxt.pro as ccxtpro
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.timescale_client import connect_with_backoff
from src.data_pipeline.schema import OHLCV_TABLE, ensure_schema

_log = get_logger(__name__)

_DEFAULT_INSTRUMENTS: Final[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT")
_DEFAULT_TIMEFRAME: Final[str] = "1h"
_DEFAULT_HISTORY_YEARS: Final[int] = 5
_MAX_CANDLES_PER_REQUEST: Final[int] = 1000
_LIVE_POLL_INTERVAL_SECONDS: Final[int] = 60

_UPSERT_SQL: Final[str] = f"""
INSERT INTO {OHLCV_TABLE} (time, instrument, open, high, low, close, volume)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (instrument, time) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume;
"""


class CryptoOhlcvCollector:
    """Backfills historical OHLCV once per instrument, then polls the latest candle on a schedule.

    Two deliberately separate code paths:

    - :meth:`backfill_history`: one-shot, paginated REST fetch covering
      the full requested lookback window -- what actually fills the
      table the first time this collector runs against a given
      instrument/timeframe. Safe to re-run (``ON CONFLICT ... DO
      UPDATE``, see :data:`_UPSERT_SQL`): re-fetched candles overwrite
      in place rather than duplicating.
    - :meth:`_poll_latest_candle` (scheduled via APScheduler in
      :meth:`run_forever`): fetches just the most recent one or two
      candles every :data:`_LIVE_POLL_INTERVAL_SECONDS`, keeping the
      table current after the backfill without re-fetching years of
      history on every tick.
    """

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        exchange_id: str = "binance",
        instruments: tuple[str, ...] = _DEFAULT_INSTRUMENTS,
        timeframe: str = _DEFAULT_TIMEFRAME,
    ) -> None:
        """Wire up the exchange REST client and target table.

        Args:
            pool: An already-connected TimescaleDB pool (schema must
                already exist -- see :func:`~src.data_pipeline.schema.
                ensure_schema`, called by :func:`main` below).
            exchange_id: ``ccxt`` exchange identifier.
            instruments: Symbols to collect, in each exchange's own
                native format (e.g. ``"BTC/USDT"``).
            timeframe: ``ccxt`` candle timeframe string (e.g. ``"1h"``,
                ``"1d"``). See the module docstring's caveat on why this
                defaults to ``"1h"`` rather than finer granularity.

        Raises:
            ValueError: If ``exchange_id`` is not a known ``ccxt``
                exchange.
        """
        exchange_class = getattr(ccxtpro, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"ccxt has no exchange named {exchange_id!r}.")
        self._exchange = exchange_class({"enableRateLimit": True})
        self._instruments = instruments
        self._timeframe = timeframe
        self._pool = pool

    async def close(self) -> None:
        """Close the underlying exchange client's HTTP session."""
        await self._exchange.close()

    async def backfill_history(self, *, years: int = _DEFAULT_HISTORY_YEARS) -> None:
        """One-shot paginated historical backfill for every configured instrument."""
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000)
        for instrument in self._instruments:
            total_written = await self._backfill_one_instrument(instrument, since_ms)
            _log.info(
                "crypto_collector_backfill_complete",
                instrument=instrument,
                timeframe=self._timeframe,
                years=years,
                candles_written=total_written,
            )

    async def _backfill_one_instrument(self, instrument: str, since_ms: int) -> int:
        """Page through ``fetch_ohlcv`` from ``since_ms`` to now, writing every page."""
        total_written = 0
        cursor_ms = since_ms
        now_ms = int(time_module.time() * 1000)

        while cursor_ms < now_ms:
            candles = await self._exchange.fetch_ohlcv(
                instrument, timeframe=self._timeframe, since=cursor_ms, limit=_MAX_CANDLES_PER_REQUEST
            )
            if not candles:
                break
            await self._write_candles(instrument, candles)
            total_written += len(candles)
            _log.info(
                "crypto_collector_backfill_page_written",
                instrument=instrument,
                page_size=len(candles),
                total_written=total_written,
            )
            # ccxt OHLCV rows are [timestamp_ms, open, high, low, close, volume],
            # oldest first -- resume just past the last candle this page returned.
            cursor_ms = int(candles[-1][0]) + 1
            if len(candles) < _MAX_CANDLES_PER_REQUEST:
                break  # short page: caught up to "now"

        return total_written

    async def _poll_latest_candle(self, instrument: str) -> None:
        """Fetch and upsert the most recent one or two candles for ``instrument``.

        Two candles, not one: the exchange's *current* (still-forming)
        candle is included so the previous, now-closed one is guaranteed
        to have been fetched at least once even if this poll lands right
        at a candle boundary.
        """
        try:
            candles = await self._exchange.fetch_ohlcv(instrument, timeframe=self._timeframe, limit=2)
        except Exception:
            _log.exception("crypto_collector_poll_failed", instrument=instrument)
            return
        if candles:
            await self._write_candles(instrument, candles)

    async def _write_candles(self, instrument: str, candles: list[list[Any]]) -> None:
        """Upsert one page of ccxt OHLCV rows via a single batched ``executemany``."""
        records = [
            (
                datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc),
                instrument,
                float(candle[1]),
                float(candle[2]),
                float(candle[3]),
                float(candle[4]),
                float(candle[5]),
            )
            for candle in candles
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(_UPSERT_SQL, records)

    async def run_forever(self) -> None:
        """Backfill once, then poll every instrument on a fixed interval, forever."""
        await self.backfill_history()

        scheduler = AsyncIOScheduler()
        for instrument in self._instruments:
            scheduler.add_job(
                self._poll_latest_candle,
                "interval",
                seconds=_LIVE_POLL_INTERVAL_SECONDS,
                args=[instrument],
                id=f"poll-{instrument}",
            )
        scheduler.start()
        _log.info(
            "crypto_collector_live_polling_started",
            instruments=list(self._instruments),
            timeframe=self._timeframe,
            interval_seconds=_LIVE_POLL_INTERVAL_SECONDS,
        )
        await asyncio.Event().wait()  # run until cancelled/killed


async def main() -> None:
    """Connect to TimescaleDB, ensure the schema, then run the collector forever."""
    configure_logging(log_level="INFO", log_dir=None, service_name="crypto-collector")
    settings = get_settings()
    pool = await connect_with_backoff(settings.timescale)
    await ensure_schema(pool)

    collector = CryptoOhlcvCollector(pool=pool)
    try:
        await collector.run_forever()
    finally:
        await collector.close()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

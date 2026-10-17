"""Historical + daily macro/economic ingestion into TimescaleDB, via FRED, never ZMQ.

Same shape as :mod:`~src.data_pipeline.collectors.crypto_collector`: a
standalone ingestion worker (not a :class:`~core.base_agent.BaseAgent`),
scheduled with :mod:`apscheduler`, that only talks to an external REST
API and TimescaleDB -- never ZMQ, never Redis.

Two FRED series, matching this project's existing macro vocabulary (see
:class:`~src.agents.macro.agent.MacroAgent`, which already fetches the
*latest* reading of the same DGS10 series for live signal generation --
this collector instead backfills *history* for the Feature Store /
Backtest Engine to train and simulate against):

- ``DGS10``: 10-Year Treasury Constant Maturity Rate (interest rates).
- ``CPIAUCSL``: Consumer Price Index for All Urban Consumers (inflation).

Falls back to a simulated random-walk history when ``MACRO_FRED_API_KEY``
is unset, exactly like :class:`~src.agents.macro.agent.MacroAgent`
already does for the same reason: the rest of this project's pipeline
should be exercisable end to end without requiring a real API key.

Run standalone with::

    python -m src.data_pipeline.collectors.macro_collector
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import Final

import asyncpg

from core.config import get_settings
from core.http_client import ResilientHttpClient
from core.logger import configure_logging, get_logger
from core.timescale_client import connect_with_backoff
from src.data_pipeline.schema import MACRO_TABLE, ensure_schema

_log = get_logger(__name__)

_FRED_OBSERVATIONS_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
# indicator name (this project's own vocabulary) -> FRED series ID.
_FRED_SERIES: Final[dict[str, str]] = {
    "US_10Y_YIELD": "DGS10",
    "US_CPI": "CPIAUCSL",
}
_DEFAULT_HISTORY_YEARS: Final[int] = 5
_DAILY_POLL_INTERVAL_SECONDS: Final[int] = 24 * 3600

_UPSERT_SQL: Final[str] = f"""
INSERT INTO {MACRO_TABLE} (time, indicator, value)
VALUES ($1, $2, $3)
ON CONFLICT (indicator, time) DO UPDATE SET value = EXCLUDED.value;
"""


class MacroIndicatorCollector:
    """Backfills historical macro readings once, then polls daily for new ones."""

    def __init__(self, *, pool: asyncpg.Pool, http_client: ResilientHttpClient, fred_api_key: str | None) -> None:
        self._pool = pool
        self._http_client = http_client
        self._fred_api_key = fred_api_key

    async def backfill_history(self, *, years: int = _DEFAULT_HISTORY_YEARS) -> None:
        """One-shot historical backfill for every configured FRED series."""
        start_date = (datetime.now(timezone.utc) - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        for indicator, series_id in _FRED_SERIES.items():
            if self._fred_api_key is not None:
                observations = await self._fetch_fred_history(series_id, start_date)
            else:
                observations = self._simulate_history(years)
                _log.warning(
                    "macro_collector_using_simulated_data",
                    indicator=indicator,
                    reason="MACRO_FRED_API_KEY not configured",
                )
            await self._write_observations(indicator, observations)
            _log.info(
                "macro_collector_backfill_complete", indicator=indicator, observations_written=len(observations)
            )

    async def _fetch_fred_history(self, series_id: str, start_date: str) -> list[tuple[datetime, float]]:
        """Fetch every FRED observation for ``series_id`` since ``start_date``.

        Skips FRED's ``"."`` missing-value marker entries -- not every
        calendar day has a published observation (e.g. CPI is monthly).
        """
        data = await self._http_client.get_json(
            _FRED_OBSERVATIONS_URL,
            params={
                "series_id": series_id,
                "api_key": self._fred_api_key,
                "file_type": "json",
                "observation_start": start_date,
                "sort_order": "asc",
            },
        )
        observations: list[tuple[datetime, float]] = []
        for row in data.get("observations", []):
            if row["value"] == ".":
                continue
            observed_at = datetime.strptime(row["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            observations.append((observed_at, float(row["value"])))
        return observations

    def _simulate_history(self, years: int) -> list[tuple[datetime, float]]:
        """Generate a smooth random-walk fallback history -- see the module docstring."""
        days = 365 * years
        start = datetime.now(timezone.utc) - timedelta(days=days)
        value = random.uniform(2.0, 5.0)
        observations: list[tuple[datetime, float]] = []
        for day_offset in range(days):
            value += random.uniform(-0.05, 0.05)
            observations.append((start + timedelta(days=day_offset), round(value, 4)))
        return observations

    async def _write_observations(self, indicator: str, observations: list[tuple[datetime, float]]) -> None:
        if not observations:
            return
        records = [(observed_at, indicator, value) for observed_at, value in observations]
        async with self._pool.acquire() as conn:
            await conn.executemany(_UPSERT_SQL, records)

    async def poll_latest(self) -> None:
        """Fetch and upsert just the latest observation for every configured series."""
        for indicator, series_id in _FRED_SERIES.items():
            if self._fred_api_key is None:
                continue  # simulated series has no "latest" to poll for -- backfill covers it
            try:
                data = await self._http_client.get_json(
                    _FRED_OBSERVATIONS_URL,
                    params={
                        "series_id": series_id,
                        "api_key": self._fred_api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": "1",
                    },
                )
            except Exception:
                _log.exception("macro_collector_poll_failed", indicator=indicator)
                continue
            rows = data.get("observations", [])
            if rows and rows[0]["value"] != ".":
                observed_at = datetime.strptime(rows[0]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                await self._write_observations(indicator, [(observed_at, float(rows[0]["value"]))])

    async def run_forever(self) -> None:
        """Backfill once, then poll daily for new observations, forever."""
        await self.backfill_history()
        _log.info(
            "macro_collector_daily_polling_started",
            indicators=list(_FRED_SERIES.keys()),
            interval_seconds=_DAILY_POLL_INTERVAL_SECONDS,
        )
        while True:
            await asyncio.sleep(_DAILY_POLL_INTERVAL_SECONDS)
            await self.poll_latest()


async def main() -> None:
    """Connect to TimescaleDB and the FRED HTTP client, then run the collector forever."""
    configure_logging(log_level="INFO", log_dir=None, service_name="macro-collector")
    settings = get_settings()
    pool = await connect_with_backoff(settings.timescale)
    await ensure_schema(pool)

    fred_api_key = settings.macro.fred_api_key.get_secret_value() if settings.macro.fred_api_key else None

    async with ResilientHttpClient() as http_client:
        collector = MacroIndicatorCollector(pool=pool, http_client=http_client, fred_api_key=fred_api_key)
        try:
            await collector.run_forever()
        finally:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

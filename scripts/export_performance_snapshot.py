"""One-way export: real Nothierly telemetry -> the Retail Gateway's public, read-only cache.

**This is the only bridge between the two air-gapped systems, and it only ever runs in one
direction.** ``retail_gateway`` (the B2C signup microservice, see its own ``main.py`` module
docstring) never reads from the trading core -- this script is what makes the B2C dashboard's
"cached Nothierly performance" panel real instead of a stub: it reads the trading core's own
already-published telemetry from ITS Redis, and pushes rows outward into the Retail Gateway's
own Postgres (``retail-postgres``, published on ``RETAIL_DB_HOST_PORT`` specifically so this
script can reach it from outside that Compose stack -- see ``docker-compose.retail.yml``'s own
comment on that one disclosed exception). It never opens a connection the other way, and it is
not itself a live agent/service -- an operator-triggered or cron/APScheduler-scheduled batch job,
same "not folded into the live event loop" boundary the mission's own air-gap rule asks for.

**Real data, not fabricated**: reads ``ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX{agent_id}`` (see
``src/agents/alpha_engine/agent.py``'s own ``_publish_telemetry``) -- the real, live
``session_realized_pnl_usd``/``daily_realized_pnl_usd`` fields that engine already publishes for
the internal dashboard's own Autonomous Command Center page. Honestly empty (this script writes
nothing, logs and exits 0) if that key doesn't exist yet -- e.g. the engine has never started, or
is running in ``MOCK_MODE`` with no real fill yet -- same "no fabricated number" discipline this
whole codebase already applies everywhere else (see that agent's own module docstring).

Usage::

    python scripts/export_performance_snapshot.py --agent-id alpha-engine-btc-usdt

Schedulable (cron, APScheduler, a Task Scheduler entry) -- not run continuously by any process
in this repo today; wiring a recurring schedule is real, disclosed follow-up work for whoever
operates this deployment, not assumed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX

# Imported from the OTHER side deliberately -- see this module's own docstring: the direction of
# this one real exception is trading-core-reaches-outward, never retail_gateway importing back in.
from retail_gateway.config import DatabaseSettings
from retail_gateway.db import connect_with_backoff

_log = get_logger(__name__)

_EXPORTED_METRICS = ("session_realized_pnl_usd", "daily_realized_pnl_usd")


async def _run(agent_id: str) -> int:
    redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    try:
        raw = await redis_client.execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent_id}")
    except RedisUnavailableError:
        _log.error("export_snapshot_trading_redis_unreachable")
        return 1

    if raw is None:
        _log.info("export_snapshot_no_live_telemetry_yet", agent_id=agent_id)
        return 0

    telemetry = json.loads(raw)
    as_of = datetime.fromtimestamp(telemetry["updated_at"], tz=timezone.utc)
    rows = [
        (f"{agent_id}.{metric_name}", float(telemetry[metric_name]), as_of)
        for metric_name in _EXPORTED_METRICS
        if telemetry.get(metric_name) is not None
    ]
    if not rows:
        _log.info("export_snapshot_telemetry_had_no_exportable_fields", agent_id=agent_id)
        return 0

    retail_db_settings = DatabaseSettings()
    pool = await connect_with_backoff(retail_db_settings, max_attempts=5)
    try:
        await pool.executemany(
            "INSERT INTO public_performance_snapshot (metric_name, metric_value, as_of) VALUES ($1, $2, $3)",
            rows,
        )
    finally:
        await pool.close()

    _log.info("export_snapshot_published", agent_id=agent_id, metric_count=len(rows))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True, help="e.g. alpha-engine-btc-usdt")
    args = parser.parse_args()

    configure_logging(service_name="export-performance-snapshot")
    start = time.monotonic()
    exit_code = asyncio.run(_run(args.agent_id))
    _log.info("export_snapshot_finished", duration_seconds=round(time.monotonic() - start, 2), exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

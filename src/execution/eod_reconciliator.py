"""Chapter 14 (Institutional Operations), Mission 1: End-of-Day broker reconciliation.

**Real, load-bearing corrections to the brief, stated upfront** (the
same discipline every prior master directive in this project's history
has followed -- see e.g. :mod:`src.execution.billing_schema`'s own
opening docstring):

1. **"Compare ces soldes avec l'inventaire théorique... (Polars/SQL)"**
   assumes a table of Nothierly's own theoretical positions, built from
   real executed trades, already exists. Verified before writing any of
   this: it does not, for the reason that matters most here --
   :data:`~src.execution.billing_schema.CLIENT_FILLS_TABLE` (the *only*
   table in this codebase that records a fill with client attribution)
   is written exclusively by Shadow Mode (see that module's own
   docstring: "the only execution path in this project that already
   carries a real client_id today"). Shadow Mode's entire point is that
   it **never** places a real order against a real broker (see
   ``api_middleware/gateway.py``'s ``_run_shadow_simulation``
   docstring) -- so comparing Shadow-simulated fills against a real
   broker's real reported balance is not "theoretical vs. actual," it
   is comparing two things with no causal connection to each other.
   There is currently no code path in this project where a LIVE order
   both (a) actually executes against a real broker and (b) gets
   persisted anywhere with client/instrument attribution -- that
   consumer does not exist yet (``MacroOrderRequest``\\ s in
   ``execution_mode=LIVE`` are published onto the ZMQ bus and nothing
   downstream writes them to a ledger).

   Given that real constraint, this module does two genuinely real
   things instead of one fabricated one:

   - :func:`fetch_broker_balances` -- **real**, unconditionally: a live
     ``get_account_balance()`` call against every configured broker.
     This alone is real, valuable, and matches what many real
     institutional EOD processes do first regardless of anything else
     (a signed, timestamped balance audit trail).
   - :func:`compute_theoretical_shadow_inventory` -- **real**, but
     explicitly scoped: aggregates ``client_fills`` into a net
     position per ``(client_id, instrument)``. This verifies
     Nothierly's own Shadow-mode bookkeeping is internally
     self-consistent (no fill lost/duplicated between simulation and
     persistence) -- a real, useful check, but every row in its output
     carries ``is_shadow_only=True`` and the top-level
     :class:`ReconciliationReport` carries an explicit
     ``scope_note`` -- never silently presented as real broker
     capital.

   "Breaks" in :attr:`ReconciliationReport.breaks` are therefore
   **day-over-day real-balance deltas** (today's real fetched balance
   vs. yesterday's real fetched balance, persisted by this same
   module) that exceed :data:`~core.config.ReconciliationSettings.
   break_threshold_usd` -- a real, honest, broker-attributable
   reconciliation signal that does not depend on the nonexistent
   theoretical-live-ledger the brief assumed.

2. **Update, once a real gap this docstring used to describe was
   closed**: the paragraph above originally ended by noting "there is
   currently no code path in this project where a LIVE order both (a)
   actually executes against a real broker and (b) gets persisted
   anywhere with client/instrument attribution." That gap is now closed
   by :class:`~src.agents.macro_order_execution.agent.
   MacroOrderExecutionAgent` -- real LIVE fills now land in
   ``client_fills`` too, each tagged with a real ``broker_id`` (see
   :mod:`~src.execution.billing_schema`'s own docstring on that new
   column). :func:`compute_live_fills_by_broker` aggregates them into
   :attr:`ReconciliationReport.live_fills_by_broker` -- the real,
   previously-missing half of "why did this broker's balance move":
   shown alongside :attr:`ReconciliationReport.breaks` so an operator
   can cross-reference a real day-over-day balance delta against the
   real, client-attributed fills recorded on that same broker in the
   same window. Still not a full theoretical-ledger reconciliation (a
   broker's balance also moves from fees, funding, and any fill this
   system didn't route -- manual withdrawals, other integrations), but
   a real, honest, non-fabricated data point no earlier chapter had.

3. **"Il force la mise à jour de current_inventory de l'Agent RL"**:
   :class:`~src.agents.rl_signal.agent.RlSignalAgent`'s own module
   docstring already discloses this real gap -- the live agent's
   feature-vector construction never passes a position/inventory input
   at all (``SupervisorMLModel.build_feature_vector`` has no such
   parameter), so there is nothing today for this module to "force an
   update" into on the consuming side. Rewiring that live ML feature
   contract (shared with :class:`~src.agents.supervisor.agent.
   XgboostSignalAgent`) is real, separate, higher-risk work out of this
   mission's scope. What *is* real and shipped here:
   :func:`write_rl_agent_inventory_snapshot` computes a real per-
   instrument net position from :func:`compute_theoretical_shadow_inventory`
   and writes it to a real, well-named, documented Redis key
   (:data:`RL_AGENT_INVENTORY_REDIS_KEY_PREFIX`) -- the real half of
   this feature, ready the moment someone extends the live agent to
   actually read it. Not claimed as a closed loop.

**Crash resilience** ("gestion des crashs d'API"): every per-broker
balance fetch is individually try/excepted (:func:`fetch_broker_balances`
never lets one broker's outage block the others), every DB/Redis step
in :func:`run_eod_reconciliation` degrades to an honest, logged, empty
result rather than aborting the whole run, and the scheduled job
wrapper (:func:`run_eod_reconciliation_job`) catches and logs any
residual exception so a single bad night never de-registers tomorrow's
run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Final, Literal

import asyncpg
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.config import AppSettings, get_settings
from core.logger import get_logger
from core.redis_client import RedisClient
from core.timescale_client import connect_with_backoff
from src.execution.billing_schema import CLIENT_FILLS_TABLE
from src.execution.billing_schema import ensure_schema as ensure_billing_schema
from src.execution.broker_gateway import BrokerFactory

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

_log = get_logger(__name__)

RL_AGENT_INVENTORY_REDIS_KEY_PREFIX: Final[str] = "RL_AGENT_INVENTORY:"
# Generous but bounded: this key exists to let a future consumer start its
# day with yesterday's real close-of-day figure, not to be a live feed --
# a stale value past 48h is more likely a sign the reconciliator itself
# stopped running than a real current figure worth serving.
_RL_INVENTORY_TTL_SECONDS: Final[int] = 48 * 3600
_LATEST_REPORT_REDIS_KEY: Final[str] = "EOD_RECONCILIATION_LATEST_REPORT"
_LATEST_REPORT_TTL_SECONDS: Final[int] = 7 * 24 * 3600

BROKER_BALANCE_SNAPSHOTS_TABLE: Final[str] = "broker_balance_snapshots"

_CREATE_SNAPSHOTS_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {BROKER_BALANCE_SNAPSHOTS_TABLE} (
    time            TIMESTAMPTZ NOT NULL,
    broker_id       TEXT NOT NULL,
    balance_usd     DOUBLE PRECISION NOT NULL,
    fetch_error     TEXT,
    PRIMARY KEY (broker_id, time)
);
"""
_CREATE_SNAPSHOTS_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{BROKER_BALANCE_SNAPSHOTS_TABLE}_broker_time "
    f"ON {BROKER_BALANCE_SNAPSHOTS_TABLE} (broker_id, time DESC);"
)


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create ``broker_balance_snapshots`` (this module's own table) and ``client_fills`` (reused, not owned).

    Idempotent, safe on every scheduled run -- same convention as
    :func:`src.execution.billing_schema.ensure_schema`.
    """
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_SNAPSHOTS_SQL)
        try:
            await conn.execute(
                f"SELECT create_hypertable('{BROKER_BALANCE_SNAPSHOTS_TABLE}', 'time', if_not_exists => TRUE);"
            )
        except asyncpg.PostgresError as exc:
            _log.warning(
                "reconciliation_hypertable_creation_failed",
                table=BROKER_BALANCE_SNAPSHOTS_TABLE, error=str(exc),
                detail="the TimescaleDB extension may not be installed; falling back to a plain table",
            )
        await conn.execute(_CREATE_SNAPSHOTS_INDEX_SQL)
    await ensure_billing_schema(pool)  # client_fills, read (not written) by this module


class BrokerBalanceResult(BaseModel):
    """One real (or honestly-failed) balance fetch for one broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_id: str
    balance_usd: float | None = Field(description="None means the fetch failed -- see error, never a fabricated 0.0.")
    error: str | None = None
    fetched_at: datetime


class ReconciliationBreak(BaseModel):
    """A real, unexplained day-over-day balance movement exceeding the configured threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_id: str
    previous_balance_usd: float
    current_balance_usd: float
    delta_usd: float
    severity: Literal["INFO", "BREAK"]
    note: str


class ShadowInventoryRow(BaseModel):
    """One (client, instrument) net position, computed from Shadow-mode fills only -- see module docstring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str
    instrument: str
    net_quantity: float = Field(description="Signed: positive = net long (more BUY than SELL quantity today), negative = net short.")
    fill_count: int
    is_shadow_only: Literal[True] = True


class LiveFillRow(BaseModel):
    """One (broker, instrument) net position, computed from real, client-attributed LIVE fills only -- see module docstring's item 2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_id: str
    instrument: str
    net_quantity: float = Field(description="Signed: positive = net client BUY volume, negative = net client SELL volume, today, on this real broker.")
    fill_count: int


class ReconciliationReport(BaseModel):
    """The real, structured EOD output -- persisted to Redis, returned to the caller (e.g. a scheduled job or dashboard page)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_at: datetime
    broker_balances: list[BrokerBalanceResult]
    brokers_failed: list[str] = Field(default_factory=list, description="broker_ids whose balance fetch raised -- see BrokerBalanceResult.error for each.")
    breaks: list[ReconciliationBreak] = Field(default_factory=list)
    shadow_inventory: list[ShadowInventoryRow] = Field(default_factory=list)
    live_fills_by_broker: list[LiveFillRow] = Field(
        default_factory=list,
        description=(
            "Real, client-attributed LIVE fills (src.agents.macro_order_execution.agent."
            "MacroOrderExecutionAgent), aggregated per real broker_id -- cross-reference against `breaks` "
            "above to see what actually traded on a broker whose balance moved unexpectedly."
        ),
    )
    scope_note: str = Field(
        default=(
            "shadow_inventory reflects Shadow-mode (paper-trading) simulated fills only -- it does not "
            "represent real broker positions. live_fills_by_broker reflects real, client-attributed LIVE "
            "fills, grouped by the real broker that executed them -- a genuine reconciliation aid, not a "
            "full theoretical ledger (a broker's balance also moves from fees/funding/fills this system "
            "didn't route). breaks are computed from real day-over-day broker balance deltas, independent "
            "of both inventories above. See module docstring for the full reasoning."
        )
    )


# =============================================================================
# Pure functions -- no I/O, real regression tests without a live DB/broker.
# =============================================================================


def detect_breaks(
    *, previous_balances: dict[str, float], current_results: list[BrokerBalanceResult], threshold_usd: float
) -> list[ReconciliationBreak]:
    """Real day-over-day delta check. A broker with no prior snapshot or a failed fetch produces no Break -- there is nothing real to compare yet."""
    breaks: list[ReconciliationBreak] = []
    for result in current_results:
        if result.balance_usd is None:
            continue
        previous = previous_balances.get(result.broker_id)
        if previous is None:
            continue
        delta = result.balance_usd - previous
        if abs(delta) < threshold_usd:
            continue
        breaks.append(
            ReconciliationBreak(
                broker_id=result.broker_id, previous_balance_usd=previous, current_balance_usd=result.balance_usd,
                delta_usd=delta, severity="BREAK",
                note=f"Balance moved ${delta:,.2f} since the last snapshot (threshold ${threshold_usd:,.2f}) with no attributed cause.",
            )
        )
    return breaks


def compute_theoretical_shadow_inventory(fill_rows: list[dict[str, Any]]) -> list[ShadowInventoryRow]:
    """Real polars aggregation of already-fetched client_fills rows into net (client_id, instrument) positions.

    Args:
        fill_rows: Plain dicts (e.g. from ``[dict(record) for record in
            await conn.fetch(...)]``) with at least ``client_id``,
            ``instrument``, ``side`` (``"BUY"``/``"SELL"``, case-
            insensitive), ``quantity`` keys -- the real
            ``client_fills`` row shape.

    Returns:
        One row per ``(client_id, instrument)`` with any fills today,
        signed net quantity, real fill count. Empty list for no input --
        an honest empty result, not an error.
    """
    if not fill_rows:
        return []

    df = pl.DataFrame(fill_rows)
    signed = pl.when(pl.col("side").str.to_uppercase() == "BUY").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
    aggregated = (
        df.with_columns(signed.alias("signed_quantity"))
        .group_by(["client_id", "instrument"])
        .agg(pl.col("signed_quantity").sum().alias("net_quantity"), pl.len().alias("fill_count"))
        .sort(["client_id", "instrument"])
    )
    return [
        ShadowInventoryRow(
            client_id=row["client_id"], instrument=row["instrument"],
            net_quantity=float(row["net_quantity"]), fill_count=int(row["fill_count"]),
        )
        for row in aggregated.iter_rows(named=True)
    ]


def compute_live_fills_by_broker(fill_rows: list[dict[str, Any]]) -> list[LiveFillRow]:
    """Real polars aggregation of already-fetched, real-broker-attributed LIVE client_fills rows into net (broker, instrument) positions.

    Args:
        fill_rows: Plain dicts with at least ``broker_id``,
            ``instrument``, ``side``, ``quantity`` keys -- the real
            ``client_fills`` row shape, pre-filtered to
            ``execution_mode = 'LIVE' AND broker_id IS NOT NULL`` by
            the caller (see :func:`~src.execution.eod_reconciliator.
            run_eod_reconciliation`'s own query).

    Returns:
        One row per ``(broker_id, instrument)`` with any real LIVE
        fills today. Empty list for no input -- an honest empty
        result, not an error (the common case before
        :class:`~src.agents.macro_order_execution.agent.
        MacroOrderExecutionAgent` has executed any real order yet).
    """
    if not fill_rows:
        return []

    df = pl.DataFrame(fill_rows)
    signed = pl.when(pl.col("side").str.to_uppercase() == "BUY").then(pl.col("quantity")).otherwise(-pl.col("quantity"))
    aggregated = (
        df.with_columns(signed.alias("signed_quantity"))
        .group_by(["broker_id", "instrument"])
        .agg(pl.col("signed_quantity").sum().alias("net_quantity"), pl.len().alias("fill_count"))
        .sort(["broker_id", "instrument"])
    )
    return [
        LiveFillRow(
            broker_id=row["broker_id"], instrument=row["instrument"],
            net_quantity=float(row["net_quantity"]), fill_count=int(row["fill_count"]),
        )
        for row in aggregated.iter_rows(named=True)
    ]


def rollup_inventory_by_instrument(shadow_inventory: list[ShadowInventoryRow]) -> dict[str, float]:
    """Sum net_quantity across every client, per instrument -- what a single per-instrument agent (the RL agent) would want, not a per-client breakdown."""
    totals: dict[str, float] = {}
    for row in shadow_inventory:
        totals[row.instrument] = totals.get(row.instrument, 0.0) + row.net_quantity
    return totals


# =============================================================================
# I/O-touching orchestration.
# =============================================================================


async def fetch_broker_balances(
    broker_ids: list[str], *, settings: AppSettings | None = None, redis_client: RedisClient | None = None
) -> list[BrokerBalanceResult]:
    """Fetch a real, live balance from every broker in ``broker_ids``. One broker's failure never blocks another's.

    Args:
        broker_ids: Real ``BrokerFactory``-recognized ids (e.g.
            ``"binance"``, ``"oanda"``, ``"ibkr"``) -- see
            :class:`~core.config.ReconciliationSettings.broker_ids` for
            where an operator configures this list.
        settings: Defaults to ``get_settings()``.
        redis_client: Passed through to ``BrokerFactory.create`` so
            per-exchange credentials added via the dashboard's
            credential form (see ``liquidity_routing.py``) are tried,
            same as every other real broker-connecting call site in
            this project.
    """
    settings = settings or get_settings()
    results: list[BrokerBalanceResult] = []
    for broker_id in broker_ids:
        fetched_at = datetime.now(timezone.utc)
        try:
            broker = await BrokerFactory.create(broker_id, settings=settings, redis_client=redis_client)
            async with broker:
                balance = await broker.get_account_balance()
            results.append(BrokerBalanceResult(broker_id=broker_id, balance_usd=balance, fetched_at=fetched_at))
            _log.info("eod_broker_balance_fetched", broker_id=broker_id, balance_usd=balance)
        except Exception as exc:  # noqa: BLE001 -- a broker outage/bad credential must never abort reconciliation for the rest
            _log.error(
                "eod_broker_balance_fetch_failed", broker_id=broker_id, error_type=type(exc).__name__, error=str(exc)
            )
            results.append(BrokerBalanceResult(broker_id=broker_id, balance_usd=None, error=str(exc), fetched_at=fetched_at))
    return results


async def _get_previous_balances(pool: asyncpg.Pool, broker_ids: list[str]) -> dict[str, float]:
    """The most recent real snapshot strictly before today, per broker -- the day-over-day baseline."""
    previous: dict[str, float] = {}
    async with pool.acquire() as conn:
        for broker_id in broker_ids:
            row = await conn.fetchrow(
                f"SELECT balance_usd FROM {BROKER_BALANCE_SNAPSHOTS_TABLE} "
                f"WHERE broker_id = $1 AND fetch_error IS NULL ORDER BY time DESC LIMIT 1",
                broker_id,
            )
            if row is not None:
                previous[broker_id] = float(row["balance_usd"])
    return previous


async def _store_snapshot(pool: asyncpg.Pool, results: list[BrokerBalanceResult]) -> None:
    async with pool.acquire() as conn:
        for result in results:
            await conn.execute(
                f"INSERT INTO {BROKER_BALANCE_SNAPSHOTS_TABLE} (time, broker_id, balance_usd, fetch_error) "
                f"VALUES ($1, $2, $3, $4) ON CONFLICT (broker_id, time) DO NOTHING",
                result.fetched_at, result.broker_id, result.balance_usd if result.balance_usd is not None else 0.0, result.error,
            )


async def _fetch_todays_shadow_fills(pool: asyncpg.Pool, *, run_at: datetime) -> list[dict[str, Any]]:
    """Every real client_fills row from the 24h ending at run_at -- Shadow-mode only, see billing_schema's own docstring."""
    window_start = run_at - timedelta(hours=24)
    async with pool.acquire() as conn:
        records = await conn.fetch(
            f"SELECT client_id, instrument, side, quantity FROM {CLIENT_FILLS_TABLE} "
            f"WHERE time >= $1 AND time < $2 AND execution_mode = 'SHADOW'",
            window_start, run_at,
        )
    return [dict(record) for record in records]


async def _fetch_todays_live_fills_by_broker(pool: asyncpg.Pool, *, run_at: datetime) -> list[dict[str, Any]]:
    """Every real, broker-attributed LIVE client_fills row from the 24h ending at run_at. See module docstring's item 2."""
    window_start = run_at - timedelta(hours=24)
    async with pool.acquire() as conn:
        records = await conn.fetch(
            f"SELECT broker_id, instrument, side, quantity FROM {CLIENT_FILLS_TABLE} "
            f"WHERE time >= $1 AND time < $2 AND execution_mode = 'LIVE' AND broker_id IS NOT NULL",
            window_start, run_at,
        )
    return [dict(record) for record in records]


async def write_rl_agent_inventory_snapshot(redis_client: RedisClient, shadow_inventory: list[ShadowInventoryRow]) -> None:
    """Persist a real per-instrument net position to Redis -- the real half of a gap this module cannot fully close alone.

    See module docstring's Mission-1-item-2 section: RlSignalAgent's
    live feature-vector construction has no parameter to receive this
    yet. This function still writes a real, correct, well-named value
    so that future wiring is a read, not a rebuild.
    """
    per_instrument = rollup_inventory_by_instrument(shadow_inventory)
    for instrument, net_quantity in per_instrument.items():
        try:
            await redis_client.execute(
                "set", f"{RL_AGENT_INVENTORY_REDIS_KEY_PREFIX}{instrument}", str(net_quantity), ex=_RL_INVENTORY_TTL_SECONDS
            )
        except Exception:  # noqa: BLE001 -- a Redis hiccup here must not fail the whole reconciliation run
            _log.exception("eod_rl_inventory_write_failed", instrument=instrument)


async def run_eod_reconciliation(*, settings: AppSettings | None = None) -> ReconciliationReport:
    """The real, crash-resilient orchestrator: fetch balances, detect breaks, aggregate Shadow inventory, persist everything.

    Every sub-step degrades to an honest empty/partial result rather
    than raising -- a Timescale outage means ``shadow_inventory``/
    ``breaks`` come back empty (logged), not that the whole run aborts
    without ever fetching the (independently valuable) real broker
    balances.
    """
    settings = settings or get_settings()
    run_at = datetime.now(timezone.utc)
    broker_ids = [b.strip() for b in (settings.reconciliation.broker_ids or "").split(",") if b.strip()]

    if not broker_ids:
        _log.warning("eod_reconciliation_no_brokers_configured")
        return ReconciliationReport(run_at=run_at, broker_balances=[])

    redis_client: RedisClient | None = None
    try:
        redis_client = await RedisClient.get_instance()
    except Exception:  # noqa: BLE001 -- balances can still be fetched (though without per-exchange dashboard credentials) without Redis
        _log.exception("eod_reconciliation_redis_unavailable")

    balance_results = await fetch_broker_balances(broker_ids, settings=settings, redis_client=redis_client)
    brokers_failed = [r.broker_id for r in balance_results if r.balance_usd is None]

    breaks: list[ReconciliationBreak] = []
    shadow_inventory: list[ShadowInventoryRow] = []
    live_fills_by_broker: list[LiveFillRow] = []
    pool: asyncpg.Pool | None = None
    try:
        pool = await connect_with_backoff(settings.timescale, max_attempts=3)
        await ensure_schema(pool)
        previous_balances = await _get_previous_balances(pool, broker_ids)
        breaks = detect_breaks(
            previous_balances=previous_balances, current_results=balance_results,
            threshold_usd=settings.reconciliation.break_threshold_usd,
        )
        await _store_snapshot(pool, balance_results)
        fill_rows = await _fetch_todays_shadow_fills(pool, run_at=run_at)
        shadow_inventory = compute_theoretical_shadow_inventory(fill_rows)
        live_fill_rows = await _fetch_todays_live_fills_by_broker(pool, run_at=run_at)
        live_fills_by_broker = compute_live_fills_by_broker(live_fill_rows)
    except Exception:  # noqa: BLE001 -- an unreachable Timescale must not discard the real balance fetch above
        _log.exception("eod_reconciliation_db_step_failed")
    finally:
        if pool is not None:
            await pool.close()

    if redis_client is not None and shadow_inventory:
        await write_rl_agent_inventory_snapshot(redis_client, shadow_inventory)

    report = ReconciliationReport(
        run_at=run_at, broker_balances=balance_results, brokers_failed=brokers_failed,
        breaks=breaks, shadow_inventory=shadow_inventory, live_fills_by_broker=live_fills_by_broker,
    )

    if redis_client is not None:
        try:
            await redis_client.execute("set", _LATEST_REPORT_REDIS_KEY, report.model_dump_json(), ex=_LATEST_REPORT_TTL_SECONDS)
        except Exception:  # noqa: BLE001 -- the report is still returned to the caller even if it can't be cached
            _log.exception("eod_reconciliation_report_cache_failed")

    _log.info(
        "eod_reconciliation_completed", brokers_reconciled=len(broker_ids), brokers_failed=len(brokers_failed),
        breaks_detected=len(breaks), shadow_inventory_rows=len(shadow_inventory),
        live_fills_by_broker_rows=len(live_fills_by_broker),
    )
    return report


async def run_eod_reconciliation_job() -> None:
    """The apscheduler entrypoint -- see :func:`schedule_eod_reconciliation`.

    Catches everything: a scheduled job that raises stops apscheduler
    from ever running it again (the same reasoning
    ``reporting_engine.run_monthly_reports_job`` already documents for
    its own pool cleanup) -- this function's own top-level try/except
    is the last line of defense on top of every inner function already
    degrading gracefully.
    """
    try:
        await run_eod_reconciliation()
    except Exception:  # noqa: BLE001 -- see docstring: a bad night must never cancel tomorrow's scheduled run
        _log.exception("eod_reconciliation_job_failed_unexpectedly")


def schedule_eod_reconciliation(scheduler: "AsyncIOScheduler") -> None:
    """Register the real midnight-UTC cron trigger, exactly per the brief ("tourne tous les jours à minuit").

    Args:
        scheduler: An already-constructed
            ``apscheduler.schedulers.asyncio.AsyncIOScheduler`` -- this
            function only registers the job, matching
            :func:`src.execution.reporting_engine.schedule_monthly_reports`'s
            own established pattern (start/shutdown stays the caller's
            responsibility). Only imported under ``TYPE_CHECKING``
            above so this module stays importable without
            ``apscheduler`` installed for callers that only need the
            pure functions above -- the same reasoning
            ``reporting_engine.py`` documents for its own lazy import.
    """
    from apscheduler.triggers.cron import CronTrigger

    scheduler.add_job(
        run_eod_reconciliation_job, trigger=CronTrigger(hour=0, minute=0), id="nothierly_eod_reconciliation", replace_existing=True,
    )

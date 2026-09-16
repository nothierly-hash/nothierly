"""Cryptographic, hash-chained Order Audit Trail -- real tamper-*evidence*, not tamper-*prevention*.

**The real gap this closes, already disclosed elsewhere before this
module existed**: ``src/dashboard/ems_pages/compliance.py``'s own
module docstring states plainly that "there is no dedicated,
purpose-built 'immutable audit trail' table in this project" and that
the two Redis lists it aggregates are "not actually immutable... since
'immutable' is a real compliance requirement, not a cosmetic one."
:class:`CryptographicAuditLedger` is that real, dedicated store:
every real order-lifecycle event (see below) is chained via SHA-256 --
``current_hash = SHA256(prev_hash '|' payload_json)`` -- into
:data:`_TABLE_NAME` in TimescaleDB, so that editing or deleting any one
past row breaks the mathematical chain for every row after it, visibly
and mechanically, not just "on the honor system."

**Real, disclosed limit, stated as precisely as every other module in
this codebase states its own**: a hash chain proves *tamper-evidence*
(an undetected edit becomes essentially impossible), not
*tamper-prevention* (nothing here stops a Postgres superuser from
``UPDATE``-ing a row and mechanically recomputing every hash after it
to make the chain internally consistent again). Real WORM storage in
the SEC 17a-4 / MiFID II RTS 24 sense needs an external, non-rewritable
medium (object-lock cloud storage, a separate air-gapped log) this
project does not have and this module does not claim to be. What this
*does* provide, concretely: (1) an accidental or unsophisticated
tamper (a stray ``DELETE``, a one-field ``UPDATE``) is immediately,
mechanically detectable via :func:`verify_chain`, without needing a
separate copy to diff against; (2) :data:`~core.schemas.
AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY` keeps a second, independent witness
of the chain's current tip in a different system than the chained rows
themselves -- a cover-up now has to consistently falsify *two* stores,
not one. Both are real, meaningful improvements over "nothing is
checked today" -- neither is a claim of cryptographic non-repudiation
against a privileged database attacker.

**Every real event type now wired**: ``SUBMITTED`` (``api_middleware/
gateway.py``'s ``submit_order``, every accepted client order),
``EXECUTED`` (that same module's ``_write_shadow_fill`` for SHADOW
child fills, and :class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent`'s ``_persist_real_fill`` for real LIVE/SHADOW
fills), ``REJECTED`` (that same agent's real refusal paths -- OFF
control state, no reachable venue, no real liquidity, a non-crypto
asset, or every venue failing execution), ``CANCELLED`` (``api_middleware/
gateway.py``'s real ``cancel_order`` route -- see that route's own
docstring for the honest, typically-short real cancellation window
:class:`MacroOrderExecutionAgent`'s immediate, unscheduled execution
leaves open).

**``GATE8_SEALED`` (2026-09-16, "Protocole Immutable Ledger")**: a fifth
real event type, from a completely different real order path this
ledger had no visibility into before -- :meth:`~src.agents.execution.
agent.ExecutionAgent._seal_gate8_decision`, called the instant that
agent's own 8-gate consensus pipeline resolves to an approved
(``report.all_passed``) decision. Carries a real ``state_vector`` (see
:class:`OrderAuditEvent`'s own docstring) instead of leaving one blank
-- every other event type above never sets it. Real, disclosed scope:
only the *decision* is sealed here, not that agent's own downstream
fills (a separate, real, disclosed follow-up, not silently implied).

**Strict sequential processing, unlike this project's other
``CLIENT_ORDERS_TOPIC`` consumers**: :class:`~src.execution.
shadow_crossing_engine.ShadowCrossingEngine` and
:class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent` both dispatch each incoming message to an
independent background task so a slow one never blocks draining the
next. This agent deliberately does **not**: two events racing to
compute their ``current_hash`` off the same ``prev_hash`` would corrupt
the chain (both would claim the same predecessor). Every event is
processed to completion, in arrival order, before the next is read --
correct, not merely simple, given what a hash chain actually requires.

**Never loses an event silently, unlike :class:`~src.agents.archivist.
agent.ArchivistAgent`** (whose own docstring accepts "a batch that
fails to write... is logged and dropped, not retried" -- a fine
tradeoff for a training-data pipeline, not for a compliance record): a
failed TimescaleDB insert is retried a few real, bounded times, and
only queued to :data:`~core.schemas.AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY`
(for manual replay -- not automatically reprocessed, a real, disclosed
limit) once every attempt has failed. ``self._last_hash`` is only
advanced after a real, durable, successful write, so a dead-lettered
event never becomes a phantom link real subsequent events would
otherwise chain from.

**This agent's own ``AgentControlState`` is intentionally inert, unlike
every other agent's**: silencing a compliance audit trail is never a
legitimate "pause" action for an operator to take from a dashboard --
see :class:`~core.base_agent.AgentControlState`'s own docstring ("an
agent that never checks this at all is exactly as safe as before this
mechanism existed"). ``_listen_for_control_commands`` still runs (so
this agent's heartbeat and liveness are visible in Agent Fleet Command
like any other), but ``OFF``/``SHADOW`` are never read anywhere in this
module's own logic.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Final, Literal

import asyncpg
import zmq.asyncio
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY,
    AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY,
    ORDER_AUDIT_EVENTS_TOPIC,
    MessagePayload,
)
from core.timescale_client import TimescaleUnavailableError, connect_with_backoff

_AGENT_ID: Final[str] = "audit-ledger-agent"
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 2.0
_INSERT_MAX_ATTEMPTS: Final[int] = 3
_INSERT_RETRY_BASE_DELAY_SECONDS: Final[float] = 0.5

# A real, fixed 64-hex-char anchor (the same length as any real SHA-256
# digest below it) -- the first row's own prev_hash chains to this,
# never to a sentinel string that wouldn't itself look like a hash.
GENESIS_HASH: Final[str] = "0" * 64

_TABLE_NAME: Final[str] = "order_audit_trail"

_CREATE_TABLE_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    seq_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    time         TIMESTAMPTZ NOT NULL,
    event_type   TEXT NOT NULL,
    order_id     TEXT NOT NULL,
    client_id    TEXT,
    asset        TEXT,
    payload      TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    current_hash TEXT NOT NULL
);
"""
# payload is TEXT, deliberately never JSONB: JSONB is a binary, key-order-
# normalizing representation -- Postgres does not guarantee it round-trips
# back to the exact same bytes that were hashed. verify_chain's recomputed
# hash must operate on the literal string that was actually hashed, or an
# untampered row would spuriously fail verification.
_CREATE_ORDER_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_order_id ON {_TABLE_NAME} (order_id, seq_id);"
)
_CREATE_ASSET_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_asset ON {_TABLE_NAME} (asset, seq_id);"
)


class OrderAuditEvent(BaseModel):
    """One real order-lifecycle event, as published onto :data:`~core.schemas.ORDER_AUDIT_EVENTS_TOPIC`.

    ``state_vector`` (2026-09-16, "Protocole Immutable Ledger"): ``None``
    for every pre-existing event type (``SUBMITTED``/``EXECUTED``/
    ``REJECTED``/``CANCELLED``) -- only ``GATE8_SEALED`` populates it,
    with the real gate-verdicts/alpha-predictions/risk-snapshot
    :meth:`~src.agents.execution.agent.ExecutionAgent._seal_gate8_decision`
    already had in hand at the exact instant its own 8-gate pipeline
    opened. A plain ``dict`` (not a dedicated nested model), deliberately:
    this event's own hash-chaining never inspects its structure, only
    hashes ``model_dump_json()`` verbatim (see :func:`compute_event_hash`)
    -- an over-typed nested schema here would buy no real safety this
    payload-agnostic hash chain needs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: Literal["SUBMITTED", "EXECUTED", "REJECTED", "CANCELLED", "GATE8_SEALED"]
    order_id: str = Field(min_length=1)
    client_id: str | None = None
    asset: str | None = None
    execution_mode: str | None = None
    detail: str = Field(min_length=1, description="Real, human-readable reason/context -- never blank.")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_vector: dict[str, Any] | None = None


class AuditTrailRow(BaseModel):
    """One already-persisted row, exactly as needed to re-verify its own link in the chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq_id: int
    payload: str
    prev_hash: str
    current_hash: str


# =============================================================================
# Pure functions -- no I/O, real regression tests without a live DB.
# =============================================================================


def compute_event_hash(prev_hash: str, payload_json: str) -> str:
    """Real SHA-256 of ``f"{prev_hash}|{payload_json}"`` -- the one real link in the chain."""
    return hashlib.sha256(f"{prev_hash}|{payload_json}".encode("utf-8")).hexdigest()


def verify_chain(rows: list[AuditTrailRow]) -> tuple[bool, str]:
    """Real, pure re-verification: recompute every hash in ``seq_id`` order and compare against what was stored.

    Args:
        rows: Every row, in ascending ``seq_id`` order (see
            :func:`fetch_all_rows`).

    Returns:
        ``(True, "<n> événements vérifiés")`` if every link matches; on
        the first mismatch, ``(False, "<real reason, with the exact
        seq_id>")`` -- distinguishing a broken ``prev_hash`` link (a row
        deleted/reordered) from an altered ``payload`` (a row edited in
        place), so an operator knows exactly what kind of tamper (or
        bug) to investigate.
    """
    if not rows:
        return True, "Aucun événement -- rien à vérifier."

    expected_prev_hash = GENESIS_HASH
    for row in rows:
        if row.prev_hash != expected_prev_hash:
            return False, (
                f"Rupture de chaîne détectée à seq_id={row.seq_id} : le prev_hash enregistré ne correspond "
                "pas au dernier current_hash calculé -- une ligne a probablement été supprimée ou réordonnée."
            )
        recomputed = compute_event_hash(row.prev_hash, row.payload)
        if recomputed != row.current_hash:
            return False, (
                f"Altération détectée à seq_id={row.seq_id} : le payload stocké ne correspond plus à son "
                "empreinte SHA-256 enregistrée -- ce contenu a été modifié après coup."
            )
        expected_prev_hash = row.current_hash

    return True, f"Intégrité cryptographique validée sur {len(rows)} événement(s) (seq_id {rows[0].seq_id} à {rows[-1].seq_id})."


# =============================================================================
# I/O-touching orchestration.
# =============================================================================


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Create ``order_audit_trail``/its indexes if they don't already exist. Idempotent, safe on every startup."""
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE_SQL)
        await conn.execute(_CREATE_ORDER_INDEX_SQL)
        await conn.execute(_CREATE_ASSET_INDEX_SQL)


async def fetch_all_rows(pool: asyncpg.Pool) -> list[AuditTrailRow]:
    """Every real row, oldest first -- the order :func:`verify_chain` needs to walk the chain correctly."""
    async with pool.acquire() as conn:
        records = await conn.fetch(f"SELECT seq_id, payload, prev_hash, current_hash FROM {_TABLE_NAME} ORDER BY seq_id ASC")
    return [
        AuditTrailRow(seq_id=r["seq_id"], payload=r["payload"], prev_hash=r["prev_hash"], current_hash=r["current_hash"])
        for r in records
    ]


async def fetch_order_history(pool: asyncpg.Pool, *, search: str = "", limit: int = 200) -> list[dict[str, Any]]:
    """Real, newest-first search by ``order_id``/``asset`` substring (case-insensitive) -- for the dashboard's search box."""
    async with pool.acquire() as conn:
        if search:
            records = await conn.fetch(
                f"SELECT seq_id, time, event_type, order_id, client_id, asset, current_hash FROM {_TABLE_NAME} "
                f"WHERE order_id ILIKE $1 OR asset ILIKE $1 ORDER BY seq_id DESC LIMIT $2",
                f"%{search}%", limit,
            )
        else:
            records = await conn.fetch(
                f"SELECT seq_id, time, event_type, order_id, client_id, asset, current_hash FROM {_TABLE_NAME} "
                f"ORDER BY seq_id DESC LIMIT $1",
                limit,
            )
    return [dict(r) for r in records]


async def fetch_chain_tip_anchor(redis_client: RedisClient) -> dict[str, Any] | None:
    """The real, independently-stored Redis witness of the chain's last known tip, or ``None`` if never written."""
    raw = await redis_client.execute("get", AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY)
    return json.loads(raw) if raw is not None else None


async def verify_chain_tip_matches_anchor(pool: asyncpg.Pool, redis_client: RedisClient) -> tuple[bool, str]:
    """Real, additional check: does TimescaleDB's own latest ``current_hash`` match the independent Redis anchor?

    A real, second witness in a different system than the chained rows
    themselves -- see module docstring's disclosed-limits section for
    exactly what this does (and does not) prove.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(f"SELECT current_hash FROM {_TABLE_NAME} ORDER BY seq_id DESC LIMIT 1")
    db_tip = row["current_hash"] if row is not None else GENESIS_HASH

    anchor = await fetch_chain_tip_anchor(redis_client)
    if anchor is None:
        return True, "Aucune ancre Redis enregistrée pour l'instant -- rien à comparer."
    if anchor["current_hash"] != db_tip:
        return False, (
            f"L'ancre Redis ({anchor['current_hash'][:12]}...) ne correspond pas au dernier hash "
            f"TimescaleDB ({db_tip[:12]}...) -- la base a peut-être été modifiée après l'écriture de l'ancre."
        )
    return True, "L'ancre Redis externe correspond au dernier hash TimescaleDB."


class CryptographicAuditLedger(BaseAgent):
    """Real, hash-chained consumer of :data:`~core.schemas.ORDER_AUDIT_EVENTS_TOPIC`. See module docstring."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            subscribed_topics=(ORDER_AUDIT_EVENTS_TOPIC,),
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        self._pool: asyncpg.Pool | None = None
        self._last_hash: str = GENESIS_HASH

    async def on_tick(self) -> None:
        """Unused: this agent is stream-driven; see :meth:`run`."""
        raise NotImplementedError("CryptographicAuditLedger overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Connect to TimescaleDB and load the real chain tip to resume from -- never restart from GENESIS on a warm restart."""
        try:
            self._pool = await connect_with_backoff(self._settings.timescale, max_attempts=5)
            await ensure_schema(self._pool)
            self._last_hash = await self._load_last_hash()
        except TimescaleUnavailableError:
            self._log.critical(
                "audit_ledger_timescale_unavailable_at_startup",
                agent_id=self.agent_id,
                detail="every real order-audit event will be queued to the dead-letter list until this agent can reach TimescaleDB",
            )
            self._pool = None
            self._last_hash = GENESIS_HASH
        self._log.info("audit_ledger_ready", agent_id=self.agent_id, last_hash=self._last_hash)

    async def on_shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def _load_last_hash(self) -> str:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"SELECT current_hash FROM {_TABLE_NAME} ORDER BY seq_id DESC LIMIT 1")
        return row["current_hash"] if row is not None else GENESIS_HASH

    async def run(self) -> None:
        """Connect dependencies, then drain the real bus alongside the heartbeat and control-command listeners.

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info("agent_started", agent_id=self.agent_id)

            await asyncio.gather(
                self._listen_for_bus_messages(),
                self._run_heartbeat_loop(),
                self._listen_for_control_commands(),
            )
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _run_heartbeat_loop(self) -> None:
        while self._running:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

    async def _listen_for_bus_messages(self) -> None:
        async for message in self._subscriber.listen():
            if message.topic != ORDER_AUDIT_EVENTS_TOPIC:
                continue
            # Never dispatched as a background task -- see module docstring:
            # strict sequential processing is what makes prev_hash chaining
            # correct at all. A slow event here delays the next one, by design.
            await self._handle_audit_event_message(message)

    async def _handle_audit_event_message(self, message: MessagePayload) -> None:
        try:
            event = OrderAuditEvent.model_validate(message.data)
        except ValidationError:
            self._log.warning("audit_ledger_malformed_event_skipped", agent_id=self.agent_id)
            return
        await self._record_event(event)

    async def _record_event(self, event: OrderAuditEvent) -> None:
        payload_json = event.model_dump_json()
        prev_hash = self._last_hash
        current_hash = compute_event_hash(prev_hash, payload_json)

        if await self._insert_with_retry(event, payload_json, prev_hash, current_hash):
            # Only advance on a real, durable, successful write -- a
            # dead-lettered event must never become a phantom link real
            # subsequent events would otherwise chain from.
            self._last_hash = current_hash
            await self._write_chain_tip_anchor(current_hash)
            if event.event_type == "GATE8_SEALED":
                # The brief's own literal log line, for this one real event type --
                # only ever emitted once the hash above is durably in TimescaleDB,
                # never before (a hash logged before the write succeeded would be a
                # real lie about what is actually "enregistre dans le Ledger").
                self._log.info(
                    "audit_gate8_decision_sealed",
                    agent_id=self.agent_id,
                    order_id=event.order_id,
                    current_hash=current_hash,
                    detail=f"[AUDIT] Decision validee. Empreinte SHA-256 : {current_hash[:8]}... Enregistre dans le Ledger.",
                )
        else:
            await self._deadletter(event)

    async def _insert_with_retry(
        self, event: OrderAuditEvent, payload_json: str, prev_hash: str, current_hash: str,
        *, max_attempts: int = _INSERT_MAX_ATTEMPTS,
    ) -> bool:
        if self._pool is None:
            return False
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute(
                        f"INSERT INTO {_TABLE_NAME} "
                        f"(time, event_type, order_id, client_id, asset, payload, prev_hash, current_hash) "
                        f"VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                        event.occurred_at, event.event_type, event.order_id, event.client_id, event.asset,
                        payload_json, prev_hash, current_hash,
                    )
                return True
            except asyncpg.PostgresError as exc:
                self._log.warning(
                    "audit_ledger_insert_attempt_failed",
                    agent_id=self.agent_id, attempt=attempt, max_attempts=max_attempts, error=str(exc),
                )
                if attempt < max_attempts:
                    await asyncio.sleep(_INSERT_RETRY_BASE_DELAY_SECONDS * attempt)
        return False

    async def _write_chain_tip_anchor(self, current_hash: str) -> None:
        if self._redis_client is None:
            return
        anchor = json.dumps({"current_hash": current_hash, "written_at": datetime.now(timezone.utc).isoformat()})
        try:
            await self._redis_client.execute("set", AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY, anchor)
        except RedisUnavailableError:
            self._log.warning("audit_ledger_chain_tip_anchor_write_failed", agent_id=self.agent_id)

    async def _deadletter(self, event: OrderAuditEvent) -> None:
        self._log.critical(
            "audit_ledger_event_deadlettered",
            agent_id=self.agent_id, order_id=event.order_id, event_type=event.event_type,
            detail="TimescaleDB insert failed after bounded retries -- queued for manual replay, never silently dropped",
        )
        if self._redis_client is None:
            return
        try:
            await self._redis_client.execute("lpush", AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY, event.model_dump_json())
        except RedisUnavailableError:
            self._log.exception("audit_ledger_deadletter_write_also_failed", agent_id=self.agent_id, order_id=event.order_id)

"""Archivist micro-agent: durable, batched, non-blocking archival to TimescaleDB.

:class:`ArchivistAgent` subscribes to every topic on the ZMQ bus
(``MARKET_DATA_L2``, ``MACRO_DATA_UPDATE``, ``NLP_SIGNAL``,
``TRADE_EXECUTED``) and persists every message into a single TimescaleDB
hypertable, so :mod:`scripts.train_model` has real historical data to
train against.

The core design constraint -- "never slow down the ZMQ bus" -- is met by
strictly separating *receiving* from *writing*: one loop
(:meth:`_listen_loop`) does nothing but drain the ZMQ subscriber into an
in-memory buffer, as fast as messages arrive; a second, independent loop
(:meth:`_periodic_flush_loop`) drains that buffer into the database on a
timer via a single batched ``COPY`` (see :meth:`_flush`). The listen loop
never awaits a database call directly, so a slow or momentarily
unavailable database can never add latency to bus consumption.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import asyncpg
import zmq.asyncio

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.schemas import TRADING_SIGNALS_TOPIC, MessagePayload
from core.timescale_client import connect_with_backoff

_ARCHIVED_TOPICS: Final[tuple[str, ...]] = (
    "MARKET_DATA_L2",
    "MACRO_DATA_UPDATE",
    "NLP_SIGNAL",
    "TRADE_EXECUTED",
    TRADING_SIGNALS_TOPIC,
)

_TABLE_NAME: Final[str] = "bus_messages"
_INSERT_COLUMNS: Final[tuple[str, ...]] = ("time", "topic", "sender_id", "data")

_CREATE_TABLE_SQL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {_TABLE_NAME} (
    time TIMESTAMPTZ NOT NULL,
    topic TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    data JSONB NOT NULL
);
"""
_CREATE_HYPERTABLE_SQL: Final[str] = (
    f"SELECT create_hypertable('{_TABLE_NAME}', 'time', if_not_exists => TRUE);"
)
_CREATE_INDEX_SQL: Final[str] = (
    f"CREATE INDEX IF NOT EXISTS idx_{_TABLE_NAME}_topic_time "
    f"ON {_TABLE_NAME} (topic, time DESC);"
)


@dataclass(frozen=True)
class _BufferedMessage:
    """One message queued in memory, awaiting the next batch flush."""

    time: datetime
    topic: str
    sender_id: str
    data: dict[str, Any]


class ArchivistAgent(BaseAgent):
    """Subscribes to every topic and batch-archives messages to TimescaleDB.

    Unlike every other concrete agent in this system, this one never
    connects to Redis (it has no use for it -- its whole job is writing
    to TimescaleDB) and never publishes anything (it is a pure sink).
    """

    def __init__(
        self,
        *,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Configure batching and subscribe to every archived topic.

        Args:
            batch_size: Flush immediately once this many messages are
                buffered, without waiting for the next timer tick.
            flush_interval_seconds: Upper bound on how long a message can
                sit in the buffer before being written, even if
                ``batch_size`` is never reached (keeps a quiet period
                from leaving data unwritten indefinitely).
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`
                (unused by this agent -- it never connects to Redis --
                but accepted for constructor consistency with the other
                four agents ``scripts.run_backend`` configures).
        """
        super().__init__(
            "archivist-agent",
            subscribed_topics=_ARCHIVED_TOPICS,
            settings=settings,
            manage_own_signals=manage_own_signals,
            manage_own_redis=manage_own_redis,
            zmq_context=zmq_context,
        )

        self._batch_size = batch_size
        self._flush_interval_seconds = flush_interval_seconds

        self._pool: asyncpg.Pool | None = None
        self._buffer: list[_BufferedMessage] = []
        self._buffer_lock = asyncio.Lock()

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent is stream-driven; see :meth:`run`.

        Implemented (rather than omitted) only to satisfy the abstract
        base class, and raises loudly if it is ever invoked by mistake.
        """
        raise NotImplementedError("ArchivistAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Connect to TimescaleDB (retrying indefinitely) and ensure the schema exists."""
        self._pool = await connect_with_backoff(self._settings.timescale)
        await self._ensure_schema()
        self._log.info("archivist_agent_ready", agent_id=self.agent_id, topics=_ARCHIVED_TOPICS)

    async def on_shutdown(self) -> None:
        """Flush any still-buffered messages, then close the connection pool.

        The final flush matters: without it, up to ``batch_size - 1``
        messages received just before a graceful shutdown would be
        silently lost.
        """
        await self._flush()
        if self._pool is not None:
            await self._pool.close()

    async def run(self) -> None:
        """Connect to TimescaleDB, then concurrently listen and flush.

        Deliberately does **not** call
        :meth:`~core.base_agent.BaseAgent._connect_redis`: this agent has
        no use for Redis at all (it is a pure ZMQ-to-TimescaleDB sink),
        and skipping it avoids a pointless connection attempt (and its
        bounded-retry delay) on every startup.

        Mirrors every other overridden ``run()`` in this codebase
        otherwise: signal handling, cancellation-safe shutdown via
        ``asyncio.gather`` of the two long-lived loops (see
        :class:`~src.agents.market_data.agent.MarketDataAgent` for the
        same pattern).

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self.on_start()

            self._running = True
            self._log.info(
                "agent_started",
                agent_id=self.agent_id,
                batch_size=self._batch_size,
                flush_interval_seconds=self._flush_interval_seconds,
            )

            await asyncio.gather(self._listen_loop(), self._periodic_flush_loop())
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    # -- Ingestion (never touches the database directly) -----------------

    async def _listen_loop(self) -> None:
        """Drain the ZMQ bus into the in-memory buffer, as fast as messages arrive.

        This is the half of the agent that must never block on I/O other
        than the ZMQ receive itself: appending to a list and checking its
        length are the only things this loop does before going back to
        ``await``ing the next message.
        """
        async for message in self._subscriber.listen():
            should_flush_now = await self._buffer_message(message)
            if should_flush_now:
                await self._flush()

    async def _buffer_message(self, message: MessagePayload) -> bool:
        """Append one message to the buffer under the lock.

        Args:
            message: The validated message received from the bus.

        Returns:
            ``True`` if the buffer has now reached :attr:`_batch_size`
            and should be flushed immediately.
        """
        async with self._buffer_lock:
            self._buffer.append(
                _BufferedMessage(message.timestamp, message.topic, message.sender_id, message.data)
            )
            return len(self._buffer) >= self._batch_size

    async def _periodic_flush_loop(self) -> None:
        """Flush the buffer on a fixed timer, independent of :meth:`_listen_loop`."""
        while True:
            await asyncio.sleep(self._flush_interval_seconds)
            await self._flush()

    # -- Flushing (the only place that touches the database) -------------

    async def _flush(self) -> None:
        """Atomically swap out the buffer, then batch-write it to TimescaleDB.

        The swap happens under the lock and is the only part shared with
        :meth:`_listen_loop`; the actual database write happens on the
        *local* copy afterward, outside the lock -- so
        :meth:`_listen_loop` can keep appending to a fresh, empty buffer
        immediately, without ever waiting for the write to finish. This
        is what makes "batch inserts without ever slowing down the ZMQ
        bus" true rather than aspirational.

        A batch that fails to write (e.g. the database is temporarily
        unreachable) is logged and dropped, not retried or requeued: this
        is an archival pipeline, not the trading-critical path, and
        retrying indefinitely risks unbounded memory growth if the
        database stays down for a long time. See the module docstring.
        """
        async with self._buffer_lock:
            if not self._buffer:
                return
            batch = self._buffer
            self._buffer = []

        try:
            await self._write_batch(batch)
            self._log.info(
                "archivist_batch_written", agent_id=self.agent_id, batch_size=len(batch)
            )
        except Exception:
            self._log.exception(
                "archivist_batch_write_failed",
                agent_id=self.agent_id,
                batch_size=len(batch),
                detail="batch dropped -- archival is best-effort, not retried",
            )

    async def _write_batch(self, batch: list[_BufferedMessage]) -> None:
        """Batch-insert ``batch`` via ``COPY`` (asyncpg's fastest bulk-load path).

        Args:
            batch: Messages to write, already detached from the live
                buffer by :meth:`_flush`.
        """
        assert self._pool is not None
        records = [(m.time, m.topic, m.sender_id, json.dumps(m.data)) for m in batch]
        async with self._pool.acquire() as conn:
            await conn.copy_records_to_table(
                _TABLE_NAME, records=records, columns=_INSERT_COLUMNS
            )

    # -- Connection setup --------------------------------------------------
    # Pool creation itself now lives in core.timescale_client.connect_with_backoff
    # (shared with src.data_pipeline) -- see on_start above.

    async def _ensure_schema(self) -> None:
        """Create the archival table/hypertable/index if they don't already exist."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            try:
                await conn.execute(_CREATE_HYPERTABLE_SQL)
            except asyncpg.PostgresError as exc:
                self._log.warning(
                    "archivist_hypertable_creation_failed",
                    agent_id=self.agent_id,
                    error=str(exc),
                    detail=(
                        "the TimescaleDB extension may not be installed on this "
                        "database; falling back to a plain (non-hypertable) table"
                    ),
                )
            await conn.execute(_CREATE_INDEX_SQL)

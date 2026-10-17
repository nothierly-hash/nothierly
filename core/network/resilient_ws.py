"""Resilient WebSocket feed: exponential-backoff reconnect + jitter, bounded queue with evict-oldest backpressure.

Real gap this closes: the 24h Chaos Shadow Run
(``logs/shadow_run_24h_20260923_024500``) recorded 500 real Binance
WebSocket reconnects and one real 1014.8s (~17 min) feed gap, which
tripped the supervisor agent's local Kill Switch ("market data feed
lost: no fresh MarketMetrics for 30.0s"). This module is a generic,
exchange-agnostic reconnect primitive -- not a replacement for
``src/agents/market_data/exchange_client.py``'s real ccxt.pro client,
which already owns the actual Binance connection, but a reusable
building block for any other raw-WebSocket consumer in this project.

Real, disclosed terminology note: the brief this responds to calls the
backpressure policy "LIFO", but the actual described behavior --
evict the oldest queued message before enqueueing a new one, so a slow
downstream consumer never blocks the feed and only the most recent L2
state survives -- is a bounded-freshness / ring-buffer policy, not a
literal last-in-first-out stack (``asyncio.Queue.get()`` still returns
in FIFO order; only the *eviction* end is affected). Implemented
exactly as described functionally; the label is corrected here, not
the behavior.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Final

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_QUEUE_MAXSIZE: Final[int] = 1000
_DEFAULT_BACKOFF_BASE_SECONDS: Final[float] = 0.5
_DEFAULT_BACKOFF_MAX_SECONDS: Final[float] = 30.0
_DEFAULT_JITTER_MIN_FRACTION: Final[float] = 0.10
_DEFAULT_JITTER_MAX_FRACTION: Final[float] = 0.50
_DEFAULT_OPEN_TIMEOUT_SECONDS: Final[float] = 10.0
_DEFAULT_PING_INTERVAL_SECONDS: Final[float] = 20.0
_DEFAULT_PING_TIMEOUT_SECONDS: Final[float] = 20.0
_MAX_BACKOFF_EXPONENT: Final[int] = 20  # 0.5s * 2**20 already dwarfs any real backoff_max_seconds


@dataclass(frozen=True)
class FeedMessage:
    """One real, timestamped raw message off the wire -- reception time is this module's own, not the exchange's."""

    received_at: float
    payload: str | bytes


class ResilientWebSocketFeed:
    """Auto-reconnecting WebSocket client publishing into a bounded, evict-oldest ``asyncio.Queue``.

    Not itself an async context manager: ``run()`` is the long-lived
    coroutine a caller schedules (e.g. ``asyncio.create_task``);
    ``stop()`` ends it cleanly from anywhere else.
    """

    def __init__(
        self,
        url: str,
        *,
        queue_maxsize: int = _DEFAULT_QUEUE_MAXSIZE,
        backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        backoff_max_seconds: float = _DEFAULT_BACKOFF_MAX_SECONDS,
        jitter_min_fraction: float = _DEFAULT_JITTER_MIN_FRACTION,
        jitter_max_fraction: float = _DEFAULT_JITTER_MAX_FRACTION,
        open_timeout_seconds: float = _DEFAULT_OPEN_TIMEOUT_SECONDS,
        ping_interval_seconds: float = _DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: float = _DEFAULT_PING_TIMEOUT_SECONDS,
    ) -> None:
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize must be >= 1.")
        if backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be positive.")
        if backoff_max_seconds < backoff_base_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_base_seconds.")
        if not 0.0 <= jitter_min_fraction <= jitter_max_fraction:
            raise ValueError("jitter_min_fraction must be in [0, jitter_max_fraction].")

        self._url = url
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self._jitter_min_fraction = jitter_min_fraction
        self._jitter_max_fraction = jitter_max_fraction
        self._open_timeout_seconds = open_timeout_seconds
        self._ping_interval_seconds = ping_interval_seconds
        self._ping_timeout_seconds = ping_timeout_seconds

        self._queue: asyncio.Queue[FeedMessage] = asyncio.Queue(maxsize=queue_maxsize)
        self._stopped = asyncio.Event()
        self._connection: ClientConnection | None = None
        self._connected_since: float | None = None

    @property
    def queue(self) -> asyncio.Queue[FeedMessage]:
        return self._queue

    @property
    def is_connected(self) -> bool:
        return self._connection is not None

    async def get_message(self, *, timeout: float | None = None) -> FeedMessage:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    def _enqueue(self, payload: str | bytes) -> None:
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                dropped = None
            _log.warning(
                "resilient_ws_queue_full_evicted_oldest",
                url=self._url,
                maxsize=self._queue.maxsize,
                dropped_age_seconds=(time.time() - dropped.received_at) if dropped is not None else None,
            )
        self._queue.put_nowait(FeedMessage(received_at=time.time(), payload=payload))

    def _next_backoff_seconds(self, attempt: int) -> float:
        capped_attempt = min(attempt, _MAX_BACKOFF_EXPONENT)
        base_delay = min(self._backoff_base_seconds * (2**capped_attempt), self._backoff_max_seconds)
        jitter_fraction = random.uniform(self._jitter_min_fraction, self._jitter_max_fraction)
        return float(base_delay + base_delay * jitter_fraction)

    async def run(self) -> None:
        attempt = 0
        while not self._stopped.is_set():
            try:
                async with websockets.connect(
                    self._url,
                    open_timeout=self._open_timeout_seconds,
                    ping_interval=self._ping_interval_seconds,
                    ping_timeout=self._ping_timeout_seconds,
                ) as connection:
                    self._connection = connection
                    self._connected_since = time.time()
                    attempt = 0
                    _log.info("resilient_ws_connected", url=self._url)
                    async for raw_message in connection:
                        self._enqueue(raw_message)
                    _log.warning("resilient_ws_server_closed_stream", url=self._url)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, TimeoutError, OSError, InvalidHandshake) as exc:
                _log.warning(
                    "resilient_ws_disconnected",
                    url=self._url,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    attempt=attempt,
                    connected_duration_seconds=(
                        round(time.time() - self._connected_since, 3) if self._connected_since is not None else None
                    ),
                )
            except Exception:
                _log.exception("resilient_ws_unexpected_error", url=self._url, attempt=attempt)
            finally:
                self._connection = None
                self._connected_since = None

            if self._stopped.is_set():
                break

            delay_seconds = self._next_backoff_seconds(attempt)
            attempt += 1
            _log.info("resilient_ws_reconnecting", url=self._url, attempt=attempt, delay_seconds=round(delay_seconds, 3))
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=delay_seconds)
            except TimeoutError:
                pass  # real backoff delay elapsed -- retry, not a failure

        _log.info("resilient_ws_stopped", url=self._url)

    async def stop(self) -> None:
        self._stopped.set()
        if self._connection is not None:
            await self._connection.close()

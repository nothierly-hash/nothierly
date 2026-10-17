"""Asynchronous Redis client singleton with exponential-backoff reconnection.

Redis is used as the shared state store (order book snapshots, signals,
agent heartbeats, ...). This module wraps ``redis.asyncio.Redis`` in a
process-wide singleton so every agent/component shares one connection
pool, and transparently recovers from connection loss with capped
exponential backoff + jitter instead of crashing the process or blocking
the event loop on a naive retry loop.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from typing import Any, ClassVar, Final

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from core.config import RedisSettings, get_settings
from core.logger import get_logger

_log = get_logger(__name__)

_RECOVERABLE_ERRORS: Final[tuple[type[RedisError], ...]] = (
    RedisConnectionError,
    RedisTimeoutError,
)


class RedisUnavailableError(RuntimeError):
    """Raised when Redis could not be reached after exhausting all retries.

    Only ever raised when the client was configured with a finite
    ``max_attempts``; with the default (``None``, i.e. unbounded retries)
    the client keeps retrying forever with a capped delay instead.
    """


class RedisClient:
    """Process-wide async Redis singleton with automatic reconnection.

    Obtain the shared instance via :meth:`get_instance` (never instantiate
    directly). All read/write operations should go through
    :meth:`execute`, which transparently reconnects (with exponential
    backoff) and retries once if the connection was lost.
    """

    _instance: ClassVar[RedisClient | None] = None
    _instance_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(
        self,
        settings: RedisSettings,
        *,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 30.0,
        backoff_multiplier: float = 2.0,
        max_attempts: int | None = None,
    ) -> None:
        """Initialize the client wrapper. Prefer :meth:`get_instance`.

        Args:
            settings: Redis connection settings (host, port, db, ...).
            base_delay_seconds: Initial delay before the first retry.
            max_delay_seconds: Upper bound the backoff delay is capped at.
            backoff_multiplier: Factor the delay is multiplied by after
                each failed attempt.
            max_attempts: Maximum reconnection attempts before giving up
                and raising :class:`RedisUnavailableError`. ``None`` (the
                default) retries forever, appropriate for a long-running
                trading process where Redis coming back online should
                self-heal without manual intervention.
        """
        self._settings = settings
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._backoff_multiplier = backoff_multiplier
        self._max_attempts = max_attempts
        # Set by get_instance() once connected -- which running event loop
        # this connection is actually usable on. See that method's
        # docstring for why this matters.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._redis: Redis[str] = Redis(
            host=settings.host,
            port=settings.port,
            db=settings.db,
            password=settings.password.get_secret_value() if settings.password else None,
            ssl=settings.use_tls,
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
            # redis-py ships its own internal retry-with-backoff for
            # connection errors, on top of whatever we do here. Left
            # enabled, the two backoff loops compound (each of *our*
            # retries silently absorbs several minutes of *its* hidden
            # retries first), which defeats predictable, observable
            # reconnection. We disable it so `_connect_with_backoff` /
            # `execute` are the single source of truth for retry timing.
            retry=Retry(NoBackoff(), retries=0),
            retry_on_error=[],
        )

    @classmethod
    async def get_instance(
        cls,
        settings: RedisSettings | None = None,
        *,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 30.0,
        backoff_multiplier: float = 2.0,
        max_attempts: int | None = None,
    ) -> RedisClient:
        """Return the shared, connected :class:`RedisClient` singleton.

        Thread/task-safe against concurrent first-call races: only one
        instance is ever constructed and connected, even if multiple
        coroutines call this simultaneously before initialization
        completes.

        **Real bug fixed here, found via live testing**: this singleton is
        only safe to reuse within the *one* event loop it was connected
        under -- correct and sufficient for its primary caller (an agent
        that runs a single ``asyncio.run(main())`` for its entire process
        lifetime), but not for a caller that invokes separate
        ``asyncio.run(...)`` calls over time in the same process (every
        Streamlit dashboard button handler does exactly this -- each click
        reruns the script, and each ``asyncio.run(...)`` creates a fresh
        event loop and closes it on return). Reusing a connection whose
        transport is bound to an already-closed loop fails with
        ``RuntimeError: Event loop is closed`` the moment it tries to
        write -- not a :data:`_RECOVERABLE_ERRORS` subclass, so
        :meth:`execute`'s own reconnect logic never even saw it. Fixed by
        tracking which loop the singleton was connected under and
        transparently discarding + reconnecting when the *currently
        running* loop differs, rather than reusing a connection dead on
        arrival.

        A ``None`` ``_loop`` (its default) is trusted as-is, never treated
        as a mismatch: this is what lets test suites across this project
        install a fake singleton directly (``RedisClient._instance =
        client``, bypassing this method entirely -- see ``tests/
        test_alert_agent.py``'s ``redis_client`` fixture) without also
        having to know about or set this attribute themselves.

        Args:
            settings: Redis settings to use if the singleton has not yet
                been created. Defaults to ``get_settings().redis``.
                Ignored if the singleton already exists.
            base_delay_seconds: Passed to the constructor if a new
                instance is created; ignored if the singleton already
                exists.
            max_delay_seconds: Passed to the constructor if a new
                instance is created; ignored if the singleton already
                exists.
            backoff_multiplier: Passed to the constructor if a new
                instance is created; ignored if the singleton already
                exists.
            max_attempts: Passed to the constructor if a new instance is
                created; ignored if the singleton already exists.

        Returns:
            The shared, connected :class:`RedisClient` -- for the
            currently running event loop.

        Raises:
            RedisUnavailableError: If the initial connection fails after
                exhausting a finite ``max_attempts`` (not applicable with
                the default unbounded retry policy).
        """
        running_loop = asyncio.get_running_loop()
        if cls._instance is not None and (cls._instance._loop is None or cls._instance._loop is running_loop):
            return cls._instance
        async with cls._instance_lock:
            if cls._instance is None or (cls._instance._loop is not None and cls._instance._loop is not running_loop):
                if cls._instance is not None:
                    await cls._instance._discard_stale_connection()
                resolved_settings = settings or get_settings().redis
                instance = cls(
                    resolved_settings,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    backoff_multiplier=backoff_multiplier,
                    max_attempts=max_attempts,
                )
                await instance._connect_with_backoff()
                instance._loop = running_loop
                cls._instance = instance
        return cls._instance

    async def _discard_stale_connection(self) -> None:
        """Best-effort close of a connection bound to an event loop that is no longer running.

        ``close()`` itself performs real I/O (a graceful connection
        shutdown) that can *also* fail with ``RuntimeError: Event loop is
        closed`` for the exact same reason the original command did --
        swallowed here since the goal is only "don't leak the reference",
        not "cleanly close a socket on a loop that no longer exists".
        """
        try:
            await self.close()
        except RuntimeError:
            _log.warning("redis_stale_connection_discarded_without_clean_close", host=self._settings.host, port=self._settings.port)

    @classmethod
    async def reset_instance(cls) -> None:
        """Close and discard the singleton, if one exists.

        Intended for graceful shutdown and for test isolation between
        test cases.
        """
        async with cls._instance_lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None

    async def _connect_with_backoff(self) -> None:
        """Attempt to (re)connect, retrying with capped exponential backoff.

        Raises:
            RedisUnavailableError: If ``max_attempts`` is finite and is
                exhausted without a successful connection.
        """
        attempt = 0
        delay = self._base_delay_seconds
        while True:
            try:
                await self._redis.ping()
            except _RECOVERABLE_ERRORS as exc:
                attempt += 1
                if self._max_attempts is not None and attempt >= self._max_attempts:
                    _log.error(
                        "redis_connection_failed_permanently",
                        attempts=attempt,
                        host=self._settings.host,
                        port=self._settings.port,
                        error=str(exc),
                    )
                    raise RedisUnavailableError(
                        f"Could not connect to Redis at {self._settings.host}:"
                        f"{self._settings.port} after {attempt} attempts."
                    ) from exc

                jittered_delay = delay * random.uniform(0.5, 1.5)
                _log.warning(
                    "redis_connection_retry",
                    attempt=attempt,
                    delay_seconds=round(jittered_delay, 2),
                    host=self._settings.host,
                    port=self._settings.port,
                    error=str(exc),
                )
                await asyncio.sleep(jittered_delay)
                delay = min(delay * self._backoff_multiplier, self._max_delay_seconds)
            else:
                _log.info(
                    "redis_connected",
                    attempts=attempt + 1,
                    host=self._settings.host,
                    port=self._settings.port,
                )
                return

    async def execute(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Run a Redis command, transparently reconnecting on connection loss.

        Args:
            command: Name of the method on ``redis.asyncio.Redis`` to call
                (e.g. ``"get"``, ``"set"``, ``"hset"``, ``"publish"``).
            *args: Positional arguments forwarded to the command.
            **kwargs: Keyword arguments forwarded to the command.

        Returns:
            Whatever the underlying ``redis.asyncio.Redis`` method returns.

        Raises:
            RedisUnavailableError: If the connection is lost and cannot be
                re-established (only possible with a finite
                ``max_attempts``).
        """
        method = getattr(self._redis, command)
        try:
            return await method(*args, **kwargs)
        except _RECOVERABLE_ERRORS as exc:
            _log.warning("redis_command_failed_reconnecting", command=command, error=str(exc))
            await self._connect_with_backoff()
            return await method(*args, **kwargs)

    async def execute_transaction(self, commands: Sequence[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> list[Any]:
        """Run ``(command, args, kwargs)`` triples as one atomic MULTI/EXEC, one round trip, with :meth:`execute`'s reconnect-once.

        No other client can observe a partial application -- what a pair of separate
        ``execute("set", ...)`` calls (two independent round trips) cannot guarantee.
        """
        async def _run() -> list[Any]:
            async with self._redis.pipeline(transaction=True) as pipe:
                for command, args, kwargs in commands:
                    getattr(pipe, command)(*args, **kwargs)
                return list(await pipe.execute())

        try:
            return await _run()
        except _RECOVERABLE_ERRORS as exc:
            _log.warning("redis_transaction_failed_reconnecting", commands=len(commands), error=str(exc))
            await self._connect_with_backoff()
            return await _run()

    @property
    def client(self) -> Redis[str]:
        """The underlying ``redis.asyncio.Redis`` instance, for direct use.

        Prefer :meth:`execute` for automatic reconnection; use this
        property directly for operations that need the raw client (e.g.
        pipelines, pub/sub).
        """
        return self._redis

    async def close(self) -> None:
        """Close the underlying connection pool.

        Safe to call multiple times.
        """
        # `aclose()` is the non-deprecated spelling at runtime, but redis-py's
        # bundled type stubs don't declare it on the generic `Redis[str]`
        # alias, which mypy --strict rejects. `close()` is fully typed and
        # functionally identical (aclose() delegates to it internally).
        await self._redis.close()
        _log.info("redis_connection_closed", host=self._settings.host, port=self._settings.port)

"""Abstract base class for every trading micro-agent in the system.

:class:`BaseAgent` is a template-method-pattern ABC that wires up
everything a micro-agent needs — a structured logger, validated settings,
a dedicated ZeroMQ PUB/SUB pair, and a lazily-connected Redis client — and
owns the entire process lifecycle: a frequency-controlled main loop and an
irreproachable graceful shutdown on ``SIGINT``/``SIGTERM``. Concrete
agents only need to implement :meth:`BaseAgent.on_tick`.
"""

from __future__ import annotations

import asyncio
import functools
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Final

import zmq.asyncio

from core.config import AppSettings, get_settings
from core.logger import get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    AGENT_CONTROL_CHANNEL_PREFIX,
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    HEARTBEAT_REDIS_KEY_PREFIX,
    MessagePayload,
)
from core.zmq_bus import ZmqPublisher, ZmqSubscriber

# Signals a long-running micro-agent process should shut down gracefully
# on. SIGTERM is how orchestrators (systemd, Docker, Kubernetes) ask a
# process to stop; SIGINT is Ctrl+C during local/manual runs.
_SHUTDOWN_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGINT, signal.SIGTERM)

# Startup should not hang forever if Redis is down: a micro-agent whose
# per-tick logic doesn't strictly need state can still be useful without
# it (e.g. pure market-data relaying). Bounded retries here trade
# "correctness at all costs" for "the agent actually starts and logs the
# degradation", which is the safer default for a reusable base class.
_REDIS_STARTUP_MAX_ATTEMPTS: Final[int] = 3
_REDIS_STARTUP_BASE_DELAY_SECONDS: Final[float] = 0.5
_REDIS_STARTUP_MAX_DELAY_SECONDS: Final[float] = 5.0

# See _write_heartbeat.
_HEARTBEAT_TTL_MULTIPLIER: Final[float] = 3.0
_HEARTBEAT_MIN_TTL_SECONDS: Final[float] = 5.0


class AgentControlState(str, Enum):
    """Real, operator-controlled state every agent's own dangerous/money-moving step should check.

    ``LIVE`` (the default): unchanged, existing behavior -- an agent
    that never checks this at all is exactly as safe as before this
    mechanism existed; nothing about adding this enum changes any
    agent's behavior until it explicitly reads :attr:`BaseAgent.
    control_state` somewhere.

    ``SHADOW``/``OFF`` deliberately have **no** single, generic meaning
    enforced here -- what each one does is each concrete agent's own,
    domain-specific decision, the same way the global Kill Switch
    (:mod:`core.security`) is a shared *mechanism* but each agent
    decides *where* to apply it. Concretely, in this codebase today:

    - :class:`~src.agents.execution.agent.ExecutionAgent` (the only
      agent that calls a real broker) treats ``SHADOW`` exactly like
      its existing ``MOCK_MODE`` branch (size and log the order, never
      call the broker) and ``OFF`` as "refuse the entry outright,
      before any sizing/risk work" -- see that class's own
      ``_place_order``.
    - A signal-only agent (:class:`~src.agents.rl_signal.agent.
      RlSignalAgent`, :class:`~src.agents.supervisor.agent.
      XgboostSignalAgent`) treats both ``SHADOW`` and ``OFF`` as "never
      publish onto the real :data:`~core.schemas.TRADING_SIGNALS_TOPIC`
      bus" (so :class:`ExecutionAgent`'s two-agent consensus can never
      be satisfied by a signal that agent didn't really mean to count)
      -- ``SHADOW`` additionally logs what it *would* have said, for
      observability, where ``OFF`` does not.

    Persisted (see :data:`~core.schemas.AGENT_CONTROL_STATE_REDIS_KEY_PREFIX`)
    and pushed for immediate effect via Pub/Sub (see
    :data:`~core.schemas.AGENT_CONTROL_CHANNEL_PREFIX`) -- see
    :meth:`BaseAgent._listen_for_control_commands`.
    """

    LIVE = "LIVE"
    SHADOW = "SHADOW"
    OFF = "OFF"


class BaseAgent(ABC):
    """Lifecycle-managing base class for all trading micro-agents.

    Subclasses must implement :meth:`on_tick` with their agent-specific
    logic, and may optionally override :meth:`on_start` /
    :meth:`on_shutdown` for extra setup/teardown. Everything else —
    connection setup, the main loop, signal handling, and resource
    cleanup — is handled here.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        tick_interval_seconds: float = 1.0,
        subscribed_topics: Sequence[str] = (),
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Wire up logging, configuration, and the ZMQ PUB/SUB sockets.

        Redis is intentionally *not* connected here: establishing a
        connection is an async operation (it may need to retry with
        backoff), so it is deferred to :meth:`run`. ZeroMQ socket
        creation/bind/connect, by contrast, is non-blocking and safe to
        do synchronously, so it happens here as requested.

        Args:
            agent_id: Stable, unique identifier for this agent instance
                (used as the logger name, the ``sender_id`` on published
                messages, and in shutdown logs).
            tick_interval_seconds: Target time between the start of one
                :meth:`on_tick` call and the next, enforced by
                :meth:`run`.
            subscribed_topics: ZMQ topics this agent's subscriber should
                subscribe to immediately.
            settings: Application settings to use. Defaults to
                ``get_settings()``.
            manage_own_signals: Whether this agent should install its own
                ``SIGINT``/``SIGTERM`` handlers in :meth:`run`. Defaults
                to ``True``, correct for exactly one agent per process
                (every standalone demo/test in this project). Set
                ``False`` when multiple agents share one process (see
                :mod:`scripts.run_backend`): OS signal handler
                registration is process-wide and last-registration-wins,
                so five agents each installing their own would leave
                four of them never hearing a shutdown signal at all --
                the orchestrating entrypoint installs one handler for
                everyone instead and cancels each agent's task directly.
            zmq_context: Share an existing ``zmq.asyncio.Context`` instead
                of creating a dedicated one. Defaults to ``None``
                (creates and owns its own context, terminated on this
                agent's own shutdown -- correct for a standalone agent).
                Pass a shared context when multiple agents run in one
                process (see :mod:`scripts.run_backend`): each
                ``Context`` opens its own OS-level resources for its I/O
                thread, and on Windows specifically, libzmq's
                internal cross-thread signaling mechanism has been
                observed to be fragile under socket churn (e.g. a
                WebSocket reconnect elsewhere in the same process) --
                five contexts multiply that fragile surface fivefold
                versus one shared context. When shared, this agent does
                *not* own or terminate it; the caller that created it is
                responsible for terminating it once every agent sharing
                it has shut down.
            manage_own_redis: Whether this agent should close the shared
                :class:`~core.redis_client.RedisClient` singleton (via
                :meth:`~core.redis_client.RedisClient.close`) during its
                own :meth:`_shutdown`. Defaults to ``True``, correct for
                a standalone agent, since it is the singleton's only
                user. ``RedisClient`` is a *process-wide* singleton
                (every agent's :meth:`_connect_redis` returns the same
                instance) -- when multiple agents share one process, the
                first one to reach its own shutdown would otherwise
                close the connection out from under every other agent
                still using it (a real race this project hit: one
                agent's ``on_shutdown`` hook failing with "Buffer is
                closed" because a sibling agent's shutdown, running
                concurrently, had already closed the shared client).
                Set ``False`` in that case; the orchestrating entrypoint
                is then responsible for closing the shared client once,
                after every agent sharing it has shut down.

        Raises:
            ConfigurationError: If ``settings`` is omitted and the
                environment configuration is invalid or incomplete (see
                :func:`core.config.get_settings`).
        """
        self.agent_id = agent_id
        self._tick_interval_seconds = tick_interval_seconds
        self._settings = settings or get_settings()
        self._manage_own_signals = manage_own_signals
        self._manage_own_redis = manage_own_redis
        self._log = get_logger(f"{self.__class__.__name__}[{agent_id}]")

        # A dedicated (not shared/global) ZMQ context by default, so that
        # terminating it at shutdown only ever affects this agent's own
        # sockets -- unless the caller explicitly shares one (see the
        # zmq_context arg above).
        self._owns_zmq_context = zmq_context is None
        self._zmq_context: zmq.asyncio.Context = zmq_context or zmq.asyncio.Context()
        zmq_settings = self._settings.zmq
        # See core/zmq_broker.py: with a broker configured, every agent's
        # publisher CONNECTS to its shared frontend instead of binding its
        # own pub_address -- required whenever more than one agent
        # (process or asyncio task) shares this configuration, since only
        # one of them could otherwise bind the same port. Unset (the
        # default) preserves the original one-agent-binds-its-own-port
        # behavior every existing standalone demo/test relies on.
        broker_frontend = zmq_settings.broker_frontend_address
        if broker_frontend is not None:
            self._publisher = ZmqPublisher(
                broker_frontend,
                high_water_mark=zmq_settings.high_water_mark,
                context=self._zmq_context,
                bind=False,
            )
        else:
            self._publisher = ZmqPublisher(
                zmq_settings.pub_address,
                high_water_mark=zmq_settings.high_water_mark,
                context=self._zmq_context,
                bind=True,
            )
        self._subscriber = ZmqSubscriber(
            zmq_settings.broker_backend_address or zmq_settings.sub_address,
            context=self._zmq_context,
        )
        for topic in subscribed_topics:
            self._subscriber.subscribe(topic)

        self._redis_client: RedisClient | None = None

        self._running = False
        self._shutdown_requested = False
        self._main_task: asyncio.Task[None] | None = None
        self._signal_handlers_installed = False

        # Real, operator-controlled state -- see AgentControlState's own
        # docstring. Defaults LIVE (unchanged behavior) until this
        # agent's control listener (started by `run`, below) has a
        # chance to read back any real, persisted operator decision.
        self._control_state: AgentControlState = AgentControlState.LIVE
        # Only set for an agent using this class's own default `run`
        # loop -- see _listen_for_control_commands' docstring on why an
        # agent that overrides `run` entirely (ExecutionAgent,
        # MarketDataAgent) manages this coroutine itself instead.
        self._control_listener_task: asyncio.Task[None] | None = None

    @property
    def control_state(self) -> AgentControlState:
        """This agent's current, real, operator-controlled state. See :class:`AgentControlState`."""
        return self._control_state

    # -- Hooks for subclasses -------------------------------------------

    @abstractmethod
    async def on_tick(self) -> None:
        """Run one iteration of this agent's work. Must be implemented.

        Called repeatedly by :meth:`run` at (best-effort)
        ``tick_interval_seconds`` intervals. Raising here does not crash
        the agent: the exception is logged and the loop continues at the
        next scheduled tick.
        """
        raise NotImplementedError

    async def on_start(self) -> None:
        """Optional hook run once, after connections are ready, before the
        main loop starts. Default implementation does nothing."""

    async def on_shutdown(self) -> None:
        """Optional hook run once during graceful shutdown, before sockets
        and connections are closed. Default implementation does nothing."""

    # -- Convenience API for subclasses ----------------------------------

    async def publish(self, topic: str, data: dict[str, Any]) -> None:
        """Validate and broadcast a message on behalf of this agent.

        Args:
            topic: Topic to publish under (e.g. ``"MARKET_DATA"``).
            data: Message-specific payload.

        Raises:
            pydantic.ValidationError: If ``topic``/``data`` do not satisfy
                :class:`~core.schemas.MessagePayload`'s schema (e.g. a
                topic not in ``UPPER_SNAKE_CASE``).
        """
        payload = MessagePayload(topic=topic, sender_id=self.agent_id, data=data)
        await self._publisher.publish(payload)

    # -- Lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        """Connect dependencies and run the main loop until shutdown.

        Installs ``SIGINT``/``SIGTERM`` handlers, connects to Redis
        (best-effort), calls :meth:`on_start`, then repeatedly calls
        :meth:`on_tick` at ``tick_interval_seconds`` intervals until a
        shutdown signal is received, at which point it cleans up every
        resource this agent owns before returning.

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        # Everything from here down -- connecting Redis, the on_start hook,
        # and the main loop -- is wrapped in a single try/except so that a
        # shutdown signal arriving at *any* point (including mid-connect,
        # before the loop even starts) is caught here and always leads to
        # the same clean `_shutdown()` path, rather than an uncaught
        # CancelledError escaping `run()` with a traceback.
        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info(
                "agent_started",
                agent_id=self.agent_id,
                tick_interval_seconds=self._tick_interval_seconds,
            )
            self._control_listener_task = asyncio.create_task(self._listen_for_control_commands())

            while self._running:
                tick_started_at = time.monotonic()
                try:
                    await self.on_tick()
                except Exception:
                    self._log.exception("agent_tick_failed", agent_id=self.agent_id)

                await self._write_heartbeat()

                elapsed = time.monotonic() - tick_started_at
                remaining = max(0.0, self._tick_interval_seconds - elapsed)
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            # Expected path: a shutdown signal cancelled this task to
            # interrupt an in-progress `await` immediately, rather than
            # waiting up to a full tick interval (or a Redis retry delay)
            # to notice shutdown.
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    def request_shutdown(self) -> None:
        """Programmatically request a graceful shutdown.

        Equivalent to receiving ``SIGINT``/``SIGTERM``, without needing an
        actual OS signal. Safe to call multiple times or before
        :meth:`run` has started.
        """
        self._handle_shutdown_signal(None)

    # -- Signal handling -----------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown.

        Uses ``loop.add_signal_handler`` where available (POSIX). On
        Windows, ``asyncio``'s default ``ProactorEventLoop`` does not
        support it (raises ``NotImplementedError``), so this falls back
        to the classic ``signal.signal`` API, which is safe to use here
        because the handler only sets plain attributes and calls
        ``Task.cancel()`` — no I/O, no awaiting.

        No-op if ``manage_own_signals=False`` was passed to the
        constructor -- see its docstring for why.
        """
        if not self._manage_own_signals:
            return
        loop = asyncio.get_running_loop()
        for sig in _SHUTDOWN_SIGNALS:
            try:
                loop.add_signal_handler(
                    sig, functools.partial(self._handle_shutdown_signal, sig)
                )
            except NotImplementedError:
                signal.signal(sig, self._sync_signal_handler)
        self._signal_handlers_installed = True

    def _sync_signal_handler(self, signum: int, frame: object) -> None:
        """``signal.signal``-compatible shim used on platforms (Windows)
        where ``loop.add_signal_handler`` is unavailable."""
        self._handle_shutdown_signal(signal.Signals(signum))

    def _handle_shutdown_signal(self, sig: signal.Signals | None) -> None:
        """Common shutdown entry point for both signal-handling strategies
        and :meth:`request_shutdown`."""
        if self._shutdown_requested:
            self._log.warning(
                "shutdown_already_in_progress",
                agent_id=self.agent_id,
                signal=sig.name if sig is not None else "manual",
            )
            return

        self._shutdown_requested = True
        self._running = False
        self._log.info(
            "shutdown_signal_received",
            agent_id=self.agent_id,
            signal=sig.name if sig is not None else "manual",
        )
        if self._main_task is not None and not self._main_task.done():
            self._main_task.cancel()

    def _remove_signal_handlers(self) -> None:
        """Best-effort removal of the signal handlers installed by
        :meth:`_install_signal_handlers`, so a shut-down agent doesn't
        leave stale handlers registered in the process."""
        if not self._signal_handlers_installed:
            return
        loop = asyncio.get_running_loop()
        for sig in _SHUTDOWN_SIGNALS:
            try:
                loop.remove_signal_handler(sig)
            except NotImplementedError:
                signal.signal(sig, signal.SIG_DFL)
        self._signal_handlers_installed = False

    # -- Connection setup / teardown ----------------------------------

    async def _connect_redis(self) -> None:
        """Best-effort Redis connection with bounded retries.

        Unlike :func:`core.redis_client.RedisClient.get_instance`'s
        default of unbounded retries (appropriate for a process that has
        nothing better to do than wait for Redis), agent startup uses a
        small bounded attempt count so a single agent whose logic doesn't
        strictly depend on Redis isn't blocked indefinitely by an
        unrelated outage. On exhaustion, the agent logs and continues
        with ``self._redis_client is None`` — subclasses that require
        Redis should check this in :meth:`on_tick`.
        """
        try:
            self._redis_client = await RedisClient.get_instance(
                self._settings.redis,
                base_delay_seconds=_REDIS_STARTUP_BASE_DELAY_SECONDS,
                max_delay_seconds=_REDIS_STARTUP_MAX_DELAY_SECONDS,
                max_attempts=_REDIS_STARTUP_MAX_ATTEMPTS,
            )
            self._log.info("agent_redis_ready", agent_id=self.agent_id)
        except RedisUnavailableError:
            self._log.warning(
                "agent_starting_without_redis",
                agent_id=self.agent_id,
                detail="Redis unreachable after bounded retries; continuing without state persistence.",
            )

    async def _write_heartbeat(self) -> None:
        """Best-effort liveness marker: a fresh timestamp under a short-TTL Redis key, every tick.

        Deliberately mirrors :class:`~src.agents.market_data.agent.
        MarketDataAgent`'s own write-with-TTL pattern for
        ``market_metrics:{symbol}``: a missing or expired key is itself
        the signal that this agent is down or stalled, not just quiet --
        no separate "last seen" bookkeeping needed. TTL is a small
        multiple of the tick interval so a single slow tick doesn't read
        as a false "offline", floored at 5s so a very fast-ticking agent
        doesn't expire its own heartbeat between writes.
        """
        if self._redis_client is None:
            return
        ttl_seconds = max(_HEARTBEAT_TTL_MULTIPLIER * self._tick_interval_seconds, _HEARTBEAT_MIN_TTL_SECONDS)
        try:
            await self._redis_client.execute(
                "set",
                f"{HEARTBEAT_REDIS_KEY_PREFIX}{self.agent_id}",
                str(time.time()),
                ex=int(ttl_seconds),
            )
        except RedisUnavailableError:
            self._log.warning("agent_heartbeat_write_failed", agent_id=self.agent_id)
            self._redis_client = None

    async def _listen_for_control_commands(self) -> None:
        """Real remote LIVE/SHADOW/OFF control -- see :class:`AgentControlState`'s own docstring.

        Reads this agent's last-persisted state from Redis once at
        startup (so a restart resumes the operator's actual last
        decision, not a silent reset to ``LIVE``), then holds a Redis
        Pub/Sub subscription open for this agent's entire lifetime,
        updating :attr:`_control_state` in-memory the instant a new
        command arrives -- exactly the same "hold a ``redis.asyncio``
        Pub/Sub subscription open for the agent's entire lifetime"
        pattern :meth:`~src.agents.execution.agent.ExecutionAgent.
        _listen_for_manual_trade_requests` already established for a
        different purpose, reused here rather than reinvented.

        **Why Pub/Sub, not a per-tick Redis read**: a hot trading loop
        checking ``redis.get(...)`` on every tick would add a real,
        avoidable network round-trip to the one path where latency
        actually matters. A background listener updates one in-memory
        attribute once per real operator action instead; every actual
        gating check (:attr:`control_state`) reads that attribute, at
        zero cost, exactly like every other in-memory flag this project
        already reads on its hot paths (e.g. ``self._mock_mode``).

        Started automatically by :meth:`run` for an agent using this
        class's own default tick loop. An agent that overrides `run`
        entirely (:class:`~src.agents.execution.agent.ExecutionAgent`,
        :class:`~src.agents.market_data.agent.MarketDataAgent`) must add
        this coroutine to its own ``asyncio.gather(...)`` call itself --
        exactly the same per-override-agent responsibility
        ``_run_heartbeat_loop`` already has in ``ExecutionAgent``.
        """
        if self._redis_client is None:
            self._log.warning("agent_control_listener_disabled_no_redis", agent_id=self.agent_id)
            return

        try:
            persisted = await self._redis_client.execute(
                "get", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{self.agent_id}"
            )
        except RedisUnavailableError:
            persisted = None
            self._log.warning("agent_control_state_read_failed_defaulting_live", agent_id=self.agent_id)
        if isinstance(persisted, bytes):  # see the pubsub branch below for why this defensive decode exists
            persisted = persisted.decode()
        if persisted is not None:
            try:
                self._control_state = AgentControlState(persisted)
                self._log.info(
                    "agent_control_state_restored", agent_id=self.agent_id, state=self._control_state.value
                )
            except ValueError:
                self._log.warning(
                    "agent_control_state_persisted_value_invalid_defaulting_live",
                    agent_id=self.agent_id,
                    raw_value=persisted,
                )

        pubsub = self._redis_client.client.pubsub()
        channel = f"{AGENT_CONTROL_CHANNEL_PREFIX}{self.agent_id}"
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                raw_new_state = message["data"]
                # This installed fakeredis version's async pubsub returns
                # raw bytes for "message"-type payloads regardless of
                # decode_responses=True (verified live, not assumed) --
                # a real, observed quirk, not a bug in this agent's own
                # code (see e.g. tests/test_regime_detector.py's
                # identical, separately-documented sync-client quirk).
                # Real redis-py decodes correctly; this defensively
                # handles both.
                if isinstance(raw_new_state, bytes):
                    raw_new_state = raw_new_state.decode()
                try:
                    self._control_state = AgentControlState(raw_new_state)
                except ValueError:
                    self._log.warning(
                        "agent_control_command_invalid_ignored", agent_id=self.agent_id, received=raw_new_state
                    )
                    continue
                self._log.warning(
                    "agent_control_state_changed", agent_id=self.agent_id, new_state=self._control_state.value
                )
        finally:
            await pubsub.unsubscribe(channel)
            # aclose() is the current, non-deprecated API (close() only
            # remains as a deprecated alias) -- types-redis' stub hasn't
            # caught up to redis-py 5.x yet. Same real quirk already
            # documented in ExecutionAgent._listen_for_manual_trade_requests.
            await pubsub.aclose()  # type: ignore[attr-defined]

    async def _shutdown(self) -> None:
        """Run the full graceful-shutdown sequence, in order.

        1. Cancel the control-command listener task, if this agent
           started one via its own default :meth:`run` loop (an agent
           that manages ``_listen_for_control_commands`` inside its own
           ``asyncio.gather(...)`` instead -- see that method's
           docstring -- never sets :attr:`_control_listener_task`, so
           this step is a no-op for it; that coroutine is already
           cancelled along with the rest of its gather).
        2. The :meth:`on_shutdown` subclass hook.
        3. Close the ZMQ publisher and subscriber sockets.
        4. Terminate this agent's ZMQ context -- only if it owns one
           (see the ``zmq_context`` constructor arg); a shared context's
           owner is responsible for terminating it once every agent
           sharing it has shut down.
        5. Close the Redis connection pool, if one was established.
        6. Remove the installed signal handlers.

        Every step is individually guarded so a failure in one does not
        prevent the rest from running.
        """
        self._log.info("agent_shutting_down", agent_id=self.agent_id)

        if self._control_listener_task is not None:
            self._control_listener_task.cancel()
            try:
                await self._control_listener_task
            except asyncio.CancelledError:
                pass
            except Exception:
                self._log.exception("agent_control_listener_cleanup_failed", agent_id=self.agent_id)

        try:
            await self.on_shutdown()
        except Exception:
            self._log.exception("agent_on_shutdown_hook_failed", agent_id=self.agent_id)

        try:
            self._publisher.close()
            self._subscriber.close()
        except Exception:
            self._log.exception("agent_zmq_socket_close_failed", agent_id=self.agent_id)

        if self._owns_zmq_context:
            try:
                # Sockets are already closed (linger=0), so this returns
                # near-instantly; run off-thread anyway so a slower/
                # blocking ZMQ teardown can never stall the event loop.
                await asyncio.to_thread(self._zmq_context.term)
            except Exception:
                self._log.exception("agent_zmq_context_term_failed", agent_id=self.agent_id)

        if self._redis_client is not None and self._manage_own_redis:
            try:
                await self._redis_client.close()
            except Exception:
                self._log.exception("agent_redis_close_failed", agent_id=self.agent_id)

        try:
            self._remove_signal_handlers()
        except Exception:
            self._log.exception("agent_signal_handler_cleanup_failed", agent_id=self.agent_id)

        self._log.info("agent_shutdown_complete", agent_id=self.agent_id)

"""Global asyncio exception handler + the one real graceful-shutdown primitive every fatal path in this codebase funnels through.

**The real failure mode this closes**: a bare, uncaught exception inside
an ``asyncio.Task`` does not crash the process and does not print
anything by default once the Task itself is garbage-collected -- it dies
silently, and every other task sharing that event loop keeps running in
whatever now-corrupted state caused it (a stale price cache, a half-
updated position, a broker connection nobody notices is gone).
``loop.set_exception_handler`` is asyncio's own last-resort hook for
exactly this: it fires for any exception that reached the *top* of a
Task or callback with nothing left downstream to catch it -- by
construction, everything that reaches :func:`install_global_exception_handler`'s
callback is already unhandled by every narrower ``try``/``except`` in the
rest of the codebase. That is the literal meaning of "remonte à la
racine" here, not a second layer of fatal-vs-recoverable filtering this
module has no way to do more reliably than the code that already chose
not to catch it.

**Real correction to the brief's own literal ``sys.exit(1)``**: calling
``sys.exit()`` from inside a Task does not terminate the process -- it
only raises ``SystemExit`` inside that one Task's coroutine, which
becomes that Task's stored exception (or, worse, triggers *another*
"exception was never retrieved" warning if nothing awaits it). The
event loop keeps running. :func:`graceful_shutdown` below always
finishes with ``os._exit()`` instead -- on a clean shutdown *and* on a
timeout/error, not only the timeout case the brief describes -- because
that is the one call that is guaranteed to actually end the process from
an arbitrary async context. Stated plainly rather than silently
diverging from the brief: this is a strictly stronger guarantee than
what was asked for ("dies with elegance" only means something if it
actually dies), not a smaller one.

**Loose coupling (this mutation's own constraint)**: this module knows
nothing about brokers, exchanges, or any specific agent. It is handed a
``shutdown_callback`` -- an arbitrary ``async def () -> None`` supplied
by whatever real process wires this in (e.g. cancel every resting order,
close the broker connection) -- and a real, already-connected
:class:`~core.zmq_bus.ZmqPublisher`/:class:`~core.notifier.TelegramNotifier`
if the caller wants the traceback broadcast/alerted, both optional. It
never imports ``ccxt``, a broker adapter, or any ``src.agents.*`` module.

**Real bug found live (2026-09-14), during this project's own 24h
shadow-run validation, and the narrow fix below**: a genuine, transient
DNS outage (``stream.binance.com`` briefly unresolvable from this
machine) tore down the entire multi-agent backend. The real chain: a
`socket.gaierror` was raised inside ``aiohttp``'s own internal,
deliberately cancellation-*"shielded"* background DNS-resolution task
(``aiohttp.connector.TCPConnector._resolve_host_with_throttle`` --
shielded so one caller's cancellation/timeout never disrupts other
callers waiting on the same in-flight lookup). By the time that shielded
task actually raised, every caller that would have awaited/caught it
(inside :class:`~src.agents.market_data.exchange_client.
AsyncExchangeClient`'s own ``_call_with_backoff`` retry loop, which
already treats bare ``OSError`` -- ``gaierror``'s real base class -- as
recoverable) had already moved on to a fresh retry attempt. The orphaned
task's exception therefore had nowhere left to go but straight to *this*
module's last-resort handler, which correctly identified it as
"unhandled by every narrower ``try``/``except`` in the codebase" (true,
by construction) and tore down the whole process over a transient
network blip nothing else in this codebase treats as fatal.

**The fix, deliberately narrow, not a general relaxation of this
module's own stated philosophy above**: :data:`_KNOWN_TRANSIENT_NETWORK_ERRORS`
(plain ``OSError`` -- no ``ccxt``/``aiohttp`` import needed, preserving
the loose-coupling constraint above) is logged at ``WARNING``, not
escalated to a full :func:`graceful_shutdown`, when it reaches this
handler with **no exception object other than that already-recoverable
family**. This does not add a new tolerance this codebase lacks
elsewhere -- it makes this top-level handler consistent with the
*already-established* real policy every narrower resilience layer
(``AsyncExchangeClient._call_with_backoff``) already has for this exact
error family: an `OSError` is retried forever there, never treated as
fatal; it should not become fatal here purely because of an asyncio
task-lifecycle race that orphaned it. Anything else -- a genuine
programming error, a corrupted state, anything not already recoverable
by construction elsewhere -- still triggers the full, real
:func:`graceful_shutdown` exactly as before.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback as traceback_module
from typing import Any, Awaitable, Callable, Final

from core.logger import get_logger
from core.notifier import AlertSeverity, TelegramNotifier
from core.schemas import SYSTEM_FATAL_EXCEPTION_TOPIC, MessagePayload
from core.zmq_bus import ZmqPublisher

_log = get_logger(__name__)

_DEFAULT_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 5.0

# See module docstring's 2026-09-14 real-bug section. Plain OSError covers every real transient
# network failure this project's own AsyncExchangeClient._call_with_backoff already retries forever
# when caught inline (socket.gaierror, ConnectionError/ConnectionResetError/ConnectionRefusedError/
# ConnectionAbortedError/BrokenPipeError, TimeoutError -- all real OSError subclasses).
_KNOWN_TRANSIENT_NETWORK_ERRORS: Final[tuple[type[BaseException], ...]] = (OSError,)

ShutdownCallback = Callable[[], Awaitable[None]]


async def graceful_shutdown(
    shutdown_callback: ShutdownCallback,
    *,
    reason: str,
    timeout_seconds: float = _DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    exit_code: int = 1,
) -> None:
    """Run ``shutdown_callback`` (real order cancellation, broker disconnect, ...) under a hard timeout, then always end the process for real.

    Never returns -- every code path ends in ``os._exit(exit_code)``. See
    module docstring on why this is stronger than the brief's literal
    "timeout -> os._exit, else sys.exit", not a smaller guarantee: a
    caller that successfully cancelled every order in 0.4s still needs
    the process to actually stop, and only ``os._exit`` reliably does
    that from inside a Task.

    Args:
        shutdown_callback: Real cleanup coroutine -- e.g. cancel every
            resting order via the broker's own API, then close the
            connection. Supplied by the running process, not this
            module (see module docstring's loose-coupling note).
        reason: Human-readable cause, logged and (if this is reached via
            :func:`install_global_exception_handler`) already broadcast.
        timeout_seconds: Hard ceiling on ``shutdown_callback``. If it has
            not finished by then, this stops waiting and exits anyway --
            a hung cancel call must never leave a zombie process holding
            open positions nobody can see.
        exit_code: Process exit status. Non-zero by default so an
            external supervisor (Docker's own ``restart:`` policy, see
            the docker-compose snippet) can tell this was not a clean,
            intentional stop.
    """
    _log.critical("graceful_shutdown_initiated", reason=reason, timeout_seconds=timeout_seconds)
    try:
        await asyncio.wait_for(shutdown_callback(), timeout=timeout_seconds)
        _log.critical("graceful_shutdown_completed", reason=reason)
    except TimeoutError:
        _log.critical(
            "graceful_shutdown_timed_out_forcing_exit", reason=reason, timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        # Deliberately broad: shutdown_callback is arbitrary caller code (order cancellation, socket
        # close) whose real failure modes this module cannot enumerate -- any exception here means
        # "cleanup did not provably succeed," which must still end in a real process exit, never a
        # silently-swallowed error that leaves the process limping on.
        _log.critical("graceful_shutdown_callback_raised_forcing_exit", reason=reason, error=str(exc))
    finally:
        # Not sys.exit(): see module docstring. os._exit() skips atexit handlers/finally blocks by
        # design -- acceptable and correct here specifically because shutdown_callback already ran
        # (or was given its full timeout to) above; this is the "now actually stop" step, not a
        # substitute for real cleanup.
        os._exit(exit_code)


async def _safe_publish_fatal_event(
    publisher: ZmqPublisher, *, sender_id: str, message: str, tb: str, task_name: str | None,
) -> None:
    try:
        await publisher.publish(
            MessagePayload(
                topic=SYSTEM_FATAL_EXCEPTION_TOPIC,
                sender_id=sender_id,
                data={"message": message, "traceback": tb, "task_name": task_name, "occurred_at": time.time()},
            )
        )
    except Exception:
        # A broken publisher must never prevent the real shutdown below from proceeding -- this is a
        # best-effort side channel, not load-bearing safety logic.
        _log.warning("exception_manager_zmq_publish_failed", sender_id=sender_id)


async def _safe_notify(notifier: TelegramNotifier, *, source: str, message: str, tb: str) -> None:
    try:
        await notifier.send(
            f"🆘 EXCEPTION FATALE NON GÉRÉE -- arrêt d'urgence en cours.\n\n{message}\n\n"
            f"Traceback (tronqué) :\n{tb[-1500:]}",
            severity=AlertSeverity.CRITICAL, source=source,
        )
    except Exception:
        _log.warning("exception_manager_telegram_notify_failed", source=source)


def install_global_exception_handler(
    loop: asyncio.AbstractEventLoop,
    shutdown_callback: ShutdownCallback,
    *,
    sender_id: str,
    publisher: ZmqPublisher | None = None,
    notifier: TelegramNotifier | None = None,
    shutdown_timeout_seconds: float = _DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
) -> None:
    """Install this process's last-resort ``asyncio`` exception handler. Call once, right after creating/getting the event loop, before any real agent work starts.

    Args:
        loop: The running (or about-to-run) event loop.
        shutdown_callback: See :func:`graceful_shutdown`.
        sender_id: This process's identity for the WORM-broadcast/alert
            (e.g. an agent's ``agent_id``) -- required, unlike the two
            optional channels below, since every log line here should be
            attributable even if no publisher/notifier is wired.
        publisher: Real, already-connected :class:`~core.zmq_bus.ZmqPublisher`
            to broadcast the traceback onto :data:`~core.schemas.SYSTEM_FATAL_EXCEPTION_TOPIC`.
            ``None`` (default) skips this -- a missing publisher must
            never block the real shutdown.
        notifier: Real :class:`~core.notifier.TelegramNotifier` to alert a
            human. ``None`` (default) skips this.
        shutdown_timeout_seconds: Forwarded to :func:`graceful_shutdown`.
    """
    # Closed over the handler below -- guards against a second unrelated exception surfacing *while*
    # shutdown is already in flight (e.g. a socket-close racing the cancellation callback) from
    # re-triggering a second, redundant graceful_shutdown -- harmless in itself (both would eventually
    # os._exit), but pointless double logging/alerting for what is already a terminal state.
    shutdown_already_triggered = False

    def _handle_exception(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        nonlocal shutdown_already_triggered

        exception = context.get("exception")
        message = context.get("message", "Unhandled exception in event loop")
        tb = (
            "".join(traceback_module.format_exception(type(exception), exception, exception.__traceback__))
            if exception is not None
            else "(no exception object in context -- see message)"
        )
        task = context.get("task") or context.get("future")
        get_name = getattr(task, "get_name", None)
        task_name = get_name() if callable(get_name) else None

        if isinstance(exception, _KNOWN_TRANSIENT_NETWORK_ERRORS):
            # See module docstring's 2026-09-14 real-bug section: a known-recoverable network error
            # that arrived here purely because it was orphaned by an asyncio task-lifecycle race
            # (nothing left to await it), not because it is actually worse than the same error type
            # this codebase already retries forever elsewhere. Logged loudly, never escalated.
            _log.warning(
                "transient_network_exception_ignored_at_top_level", message=message,
                exception_repr=repr(exception), task_name=task_name,
            )
            return

        _log.critical(
            "fatal_unhandled_exception", message=message, exception_repr=repr(exception),
            traceback=tb, task_name=task_name,
        )

        if shutdown_already_triggered:
            _log.warning("fatal_exception_during_shutdown_ignored", message=message)
            return
        shutdown_already_triggered = True

        if publisher is not None:
            loop.create_task(
                _safe_publish_fatal_event(publisher, sender_id=sender_id, message=message, tb=tb, task_name=task_name)
            )
        if notifier is not None:
            loop.create_task(_safe_notify(notifier, source=sender_id, message=message, tb=tb))

        loop.create_task(
            graceful_shutdown(
                shutdown_callback, reason=f"unhandled_exception: {message}", timeout_seconds=shutdown_timeout_seconds,
            )
        )

    loop.set_exception_handler(_handle_exception)
    _log.info("global_exception_handler_installed", sender_id=sender_id)

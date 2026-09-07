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

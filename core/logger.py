"""Structured, asynchronous, secret-safe JSON logging.

Logging is configured on top of the standard library ``logging`` module
using a :class:`logging.handlers.QueueHandler` / :class:`~logging.handlers.
QueueListener` pair: every log call only pushes a record onto an in-memory
queue and returns immediately, while a dedicated background thread drains
the queue and performs the actual (blocking) I/O. This guarantees that
logging never blocks the asyncio event loop, even under high-frequency
market-data logging.

``structlog`` sits on top of stdlib logging to produce structured,
machine-parseable JSON log lines, and a custom processor redacts any
value that looks like an API key, secret, token, or password before it
ever reaches disk or stdout.

Typical usage::

    from core.logger import configure_logging, get_logger

    configure_logging(log_level="INFO")
    log = get_logger(__name__)
    log.info("order_submitted", symbol="EURUSD", quantity=10_000)
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import re
import sys
from pathlib import Path
from typing import Any, Final

import structlog
from structlog.types import EventDict, WrappedLogger

_SENSITIVE_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret|token|password|passwd|authorization|auth)",
    re.IGNORECASE,
)
# Fallback, value-based heuristic for a secret logged under an innocuous
# key name. Deliberately excludes '-' and '_' from the allowed charset:
# this system's identifiers (agent IDs like "market-data-binance-btc-usdt",
# sender IDs, order IDs, ...) are always human-composed, hyphen/underscore
# -separated words, of essentially unbounded length -- any fixed length
# threshold alone eventually false-positives on one of them (as first
# "demo-agent-1" and then a longer agent_id both did). A genuine opaque
# secret (AWS keys, Stripe/GitHub/Slack tokens, hex digests, base64
# blobs) is a single unbroken run of characters with no word separators,
# which is the actual distinguishing trait -- not length. Key-name
# matching above remains the precise, primary defense; this is
# defense-in-depth only, so losing coverage for the rare vendor token
# format that embeds a hyphen/underscore is an acceptable trade-off.
#
# Unanchored (no leading/trailing '^'/'$'): a red-team audit found the
# previous whole-value-only anchoring let a real secret slip through
# whenever it was embedded mid-sentence in a longer human-readable
# message rather than being the field's entire value -- exactly the
# shape of `error=str(exc)` for a caught ccxt AuthenticationError (see
# src/agents/market_data/exchange_client.py, and CcxtBrokerAdapter in
# src/execution/broker_gateway.py, which delegates to the same client).
# Matching anywhere in the value still requires the same 24+-char
# unbroken run, so it stays precise for the same reason the anchored
# version was.
#
# 2026-09-15, a third real false positive found live (this time hiding a
# real diagnostic, not just an id): a 27-day production shadow run's own
# logs redacted `error_type=type(exc).__name__` for
# core.message_integrity.SignatureVerificationError (26 chars) and
# core.timescale_client.TimescaleUnavailableError (25 chars) -- both
# ordinary, benign PascalCase Python exception class names, both >= 24
# unbroken letters, both caught by this pattern with zero regard for
# *content*. The real, actionable failure reason was invisible in every
# affected log line for the run's entire duration. Real secrets (AWS/
# Stripe/GitHub/Slack tokens, hex digests, base64 blobs) are, without
# exception in this codebase's own real fixture
# (tests/red_team_hacker.py's `fake_secret`), a mix of letters *and*
# digits (or base64's `+`/`=` padding) -- a run of 24+ *pure* letters, by
# contrast, is exactly what a human-readable CamelCase identifier looks
# like and is not a realistic secret shape. The lookahead below requires
# at least one digit or `+`/`=` character somewhere in the run, closing
# this false positive while leaving every real secret shape (including
# the exact fixture above) matched unchanged. **Residual, disclosed
# gap**: a genuine secret composed of only letters (no digit, no `+`/
# `=`) would now slip past this fallback -- an acceptable trade-off for
# the same reason the hyphen/underscore exclusion above already is: this
# is defense-in-depth only, and the key-name pattern above remains the
# precise, untouched primary defense.
_SENSITIVE_VALUE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?=[A-Za-z0-9\/+=]*[0-9\/+=])[A-Za-z0-9\/+=]{24,}")
_REDACTED: Final[str] = "***REDACTED***"

_listener: logging.handlers.QueueListener | None = None


def _redact_secrets(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor that redacts API keys/secrets/tokens/passwords.

    Any key whose name looks sensitive (e.g. ``api_key``, ``password``,
    ``authorization``) has its value fully redacted. Additionally, any
    string value *containing* what looks like a bare secret (a long
    unbroken token of alphanumeric/base64-like characters -- anywhere in
    the value, not just when the value is nothing else, e.g. a leaked API
    key embedded mid-sentence in a caught exception's own message logged
    under an innocuous key name) has just that token redacted in place,
    as a defense-in-depth measure.

    Args:
        logger: The wrapped stdlib logger (unused, required by the
            structlog processor signature).
        method_name: The name of the log method invoked, e.g. ``"info"``
            (unused, required by the structlog processor signature).
        event_dict: The structured event payload being built.

    Returns:
        The same ``event_dict``, with sensitive values replaced in place.
    """
    for key, value in event_dict.items():
        if key == "event":
            continue
        if _SENSITIVE_KEY_PATTERN.search(key):
            event_dict[key] = _REDACTED
        elif isinstance(value, str) and _SENSITIVE_VALUE_PATTERN.search(value):
            event_dict[key] = _SENSITIVE_VALUE_PATTERN.sub(_REDACTED, value)
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    *,
    service_name: str = "trading-engine",
    log_dir: Path | None = Path("logs"),
    log_file_name: str = "trading.log",
    json_to_stdout: bool = True,
) -> None:
    """Configure process-wide asynchronous, structured, JSON logging.

    Sets up a background :class:`~logging.handlers.QueueListener` thread
    that owns the real handlers (console + rotating file), and points the
    root stdlib logger at a non-blocking :class:`~logging.handlers.
    QueueHandler`. ``structlog`` is then configured to render through
    stdlib logging, so every log call from any part of the codebase (and
    any third-party library using stdlib ``logging``) produces redacted,
    structured JSON.

    This function is idempotent-safe to call once at process startup; call
    :func:`shutdown_logging` during graceful shutdown to flush and stop
    the background thread.

    Args:
        log_level: Root log level name (e.g. ``"DEBUG"``, ``"INFO"``).
        service_name: Service identifier attached to every log event.
        log_dir: Directory for the rotating log file. If ``None``, only
            stdout logging is configured (useful for tests).
        log_file_name: File name for the rotating log file within
            ``log_dir``.
        json_to_stdout: If ``True``, stdout also receives JSON-formatted
            lines (as opposed to human-readable console output).

    Raises:
        OSError: If ``log_dir`` cannot be created (e.g. permissions
            issue).
    """
    global _listener

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stdout_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / log_file_name,
            maxBytes=50 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=10_000)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)

    if _listener is not None:
        _listener.stop()
    _listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    _listener.start()

    # NOTE: rendering happens *before* the record is pushed onto the queue,
    # not in the QueueListener thread. This is deliberate: QueueHandler
    # collapses `record.msg` to a plain string via `record.getMessage()`
    # before queuing it (losing any structured/dict payload), so a
    # dict-aware formatter (e.g. structlog.stdlib.ProcessorFormatter)
    # cannot run downstream in the listener. Rendering to a JSON string is
    # cheap, synchronous, CPU-only work; the genuinely blocking part —
    # writing bytes to stdout/disk — still happens entirely in the
    # background listener thread, which is what keeps the event loop
    # non-blocking.
    renderer = (
        structlog.processors.JSONRenderer()
        if json_to_stdout
        else structlog.dev.ConsoleRenderer()
    )
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_secrets,
        renderer,
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    passthrough_formatter = logging.Formatter("%(message)s")
    for handler in handlers:
        handler.setFormatter(passthrough_formatter)

    structlog.contextvars.bind_contextvars(service=service_name)


def shutdown_logging() -> None:
    """Flush and stop the background logging thread.

    Should be called during graceful application shutdown to ensure all
    queued log records are flushed to disk/stdout before the process
    exits. Safe to call even if :func:`configure_logging` was never
    called.
    """
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound to the given name.

    Args:
        name: Logical logger name, conventionally ``__name__`` of the
            calling module.

    Returns:
        A ``structlog`` bound logger producing redacted, structured JSON
        log events through the non-blocking queue configured by
        :func:`configure_logging`.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger

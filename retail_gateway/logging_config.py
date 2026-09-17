"""Minimal, self-contained structured logging for the Retail Gateway.

**Real, disclosed duplication**: ``core/logger.py`` has a more elaborate version of this
(plus PII/secret redaction tuned to the trading core's own log shapes -- API keys,
agent IDs, broker credentials). Not imported here, same air-gap reasoning as
:mod:`retail_gateway.db`. This service logs a much narrower surface (no secrets are ever
logged -- passwords/tokens never appear in a log call anywhere in this package, checked
directly, not just intended) so a full redaction layer would be defense for a category of
leak this codebase doesn't create in the first place. If that changes, port the real
redaction regex over rather than assume "narrow surface" stays true forever.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(*, service_name: str) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)

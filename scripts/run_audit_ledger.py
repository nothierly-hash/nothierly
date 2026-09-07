"""Standalone launcher for a single, real :class:`~src.execution.audit_ledger.CryptographicAuditLedger`.

Runs the real, SHA-256 hash-chained order-audit-trail consumer alone:
it subscribes to :data:`~core.schemas.ORDER_AUDIT_EVENTS_TOPIC` (the
same bus every real ``SUBMITTED``/``EXECUTED``/``REJECTED`` emitter in
this codebase already publishes onto -- ``api_middleware/gateway.py``'s
``submit_order``/``_write_shadow_fill``, and
:class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent`'s real fill/rejection paths) and persists
every event into TimescaleDB's ``order_audit_trail`` table, chained to
the one before it. See that module's own docstring for the real,
honest scope of what this hash chain proves (tamper-*evidence*) and
does not (tamper-*prevention*), and for exactly which event types are
wired today.

**Real gotcha, not this script's problem to fix** (same one every other
standalone launcher in ``scripts/`` already documents): this agent's
ZMQ subscriber only really receives anything if something is bound at
``ZMQ_BROKER_FRONTEND_ADDRESS``/``ZMQ_BROKER_BACKEND_ADDRESS`` (see
``.env`` -- normally :mod:`scripts.run_backend` or a standalone
:class:`~core.zmq_broker.ZmqBroker`). Without one running, this script
still starts cleanly and reaches its own steady state (real TimescaleDB
connect-or-degrade-to-dead-letter, as usual) -- it just never receives
a real event to chain.

Usage::

    python scripts/run_audit_ledger.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as every other scripts/run_*.py launcher: `core`/`src`
# live at the project root, which must be on sys.path before those
# imports below, regardless of the working directory this was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from core.logger import configure_logging, get_logger
from src.execution.audit_ledger import CryptographicAuditLedger

_log = get_logger(__name__)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="audit-ledger-agent")

    ledger = CryptographicAuditLedger()
    _log.info("audit_ledger_launcher_starting")
    await ledger.run()


if __name__ == "__main__":
    asyncio.run(main())

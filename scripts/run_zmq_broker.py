"""Standalone launcher for a single, real :class:`~core.zmq_broker.ZmqBroker`.

**Real gap this closes**: every standalone agent launcher in this project (``run_market_data_agent.py``,
``run_alpha_engine.py``, the new ``scout_agent.py``/``analyst_agent.py``/``risk_gate_agent.py``)
connects its own publisher/subscriber to ``ZMQ_BROKER_FRONTEND_ADDRESS``/``_BACKEND_ADDRESS`` (see
``.env``) -- but nothing was ever bound there when running fewer than the full agent fleet
(:mod:`scripts.run_backend`, which starts its own broker inline, is the only other real binder).
Without this, every one of those standalone launchers' own ZMQ publishes genuinely go nowhere --
"real, disclosed" in each of their own docstrings, not a bug in any one of them, but a real,
missing piece for running any real multi-process ZMQ demo/dev setup outside the full backend.

Usage (real two-or-more-terminal setup, matching every other real standalone launcher here)::

    # Terminal 1 -- this script, once, before any other ZMQ-publishing agent:
    python scripts/run_zmq_broker.py

    # Terminal 2+ -- any real standalone agent that publishes/subscribes via the broker addresses:
    python scout_agent.py
    python analyst_agent.py
    python risk_gate_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

# **Real bug fixed here, found live**: core/zmq_bus.py's own module-level
# `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())` must run before any event loop
# is created (see that module's own comment) -- every other real launcher in this project gets this
# for free by importing core.base_agent (which imports core.zmq_bus) before calling asyncio.run().
# This script imports core.zmq_broker directly instead (a standalone broker has no agent of its
# own) -- core.zmq_broker does not itself import core.zmq_bus, so without this explicit import the
# Windows default ProactorEventLoop was used instead, and pyzmq's asyncio integration cannot work
# with it at all (confirmed live: RuntimeError "Proactor event loop does not implement add_reader").
import core.zmq_bus  # noqa: F401

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.zmq_broker import ZmqBroker

_log = get_logger(__name__)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="zmq-broker")
    zmq_settings = get_settings().zmq

    frontend = zmq_settings.broker_bind_frontend_address or zmq_settings.broker_frontend_address
    backend = zmq_settings.broker_bind_backend_address or zmq_settings.broker_backend_address
    if frontend is None or backend is None:
        raise SystemExit(
            "ZMQ_BROKER_FRONTEND_ADDRESS/ZMQ_BROKER_BACKEND_ADDRESS must be set (see .env) -- "
            "nothing to bind without them."
        )

    broker = ZmqBroker(frontend, backend)
    _log.info("zmq_broker_launcher_starting", frontend=frontend, backend=backend)
    await broker.run()


if __name__ == "__main__":
    asyncio.run(main())

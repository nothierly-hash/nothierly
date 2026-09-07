"""Standalone launcher for a single, real :class:`~src.agents.macro_order_execution.agent.MacroOrderExecutionAgent`.

Runs the real consumer that closes the "no LIVE order ever gets
persisted with client attribution" gap: it subscribes to
:data:`~core.schemas.CLIENT_ORDERS_TOPIC` (the same bus
``api_middleware/gateway.py``'s ``/api/v1/orders/submit`` endpoint
publishes every macro order onto) and, for each real LIVE-mode order,
routes and executes it across real crypto venues via the existing
Smart Order Router, persisting every real fill to the durable
``client_fills`` table with a real ``broker_id`` -- see that module's
own docstring for the full design and its honestly-disclosed scope
limits (crypto only, no time-based algo slicing yet, exactly one
instance).

**Real gotcha, not this script's problem to fix** (same one
``scripts/run_market_data_agent.py`` already documents): this agent's
own ``MARKET_DATA_L2``-style traffic is unrelated, but its ZMQ
publisher/subscriber still only really connects anywhere if something
is bound at ``ZMQ_BROKER_FRONTEND_ADDRESS``/``ZMQ_BROKER_BACKEND_ADDRESS``
(see ``.env`` -- normally :mod:`scripts.run_backend` or a standalone
:class:`~core.zmq_broker.ZmqBroker`). Without one running, this agent
still receives nothing (no bus to receive from) -- run
``api_middleware/gateway.py`` and a real broker for real end-to-end
order flow, or use this script for its Timescale/Redis-side wiring only.

Usage::

    python scripts/run_macro_order_execution_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as scripts/run_backend.py and scripts/run_market_data_agent.py:
# `core`/`src` live at the project root, which must be on sys.path before those
# imports below, regardless of the working directory this was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from core.logger import configure_logging, get_logger
from src.agents.macro_order_execution.agent import MacroOrderExecutionAgent

_log = get_logger(__name__)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="macro-order-execution-agent")

    agent = MacroOrderExecutionAgent()
    _log.info("macro_order_execution_agent_launcher_starting")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

"""Standalone launcher for a single, real :class:`~src.agents.market_data.agent.MarketDataAgent`.

Runs *the actual production agent* (``ccxt.pro`` WebSocket order book/trade
streaming, real ``MarketMetrics`` computation, the real 3-leg publish
pipeline: ``market_metrics:{symbol}`` Redis SET, ``l2_surface:{symbol}``
Redis SET, and a ``MARKET_DATA_L2`` ZMQ publish) as a lone process -- not
the multi-agent daemon (:mod:`scripts.run_backend`), and not a new,
separate ingestion pipeline. Needs no exchange API credentials: public
order-book/trade data only.

Every default on :class:`MarketDataAgent` is already correct for running
alone -- ``manage_own_signals=True`` installs its own SIGINT/SIGTERM
handler, ``manage_own_redis=True`` connects (and later closes) its own
:class:`~core.redis_client.RedisClient`, and ``zmq_context=None`` creates
its own :class:`zmq.asyncio.Context` -- unlike ``run_backend.py``, which
must override all three so several agents can safely share one process.

**Real gotcha, not this script's problem to fix**: the ``MARKET_DATA_L2``
ZMQ publish only reaches a subscriber (:class:`~src.agents.execution.
agent.ExecutionAgent`, the signal agents) if something is bound at
``ZMQ_BROKER_FRONTEND_ADDRESS``/``ZMQ_BROKER_BACKEND_ADDRESS`` (see
``.env`` -- normally :mod:`scripts.run_backend` or a standalone
:class:`~core.zmq_broker.ZmqBroker`). ZMQ PUB is fire-and-forget, so
running this script alone, with no broker up, never blocks or errors --
it just means those ZMQ messages go nowhere. The two Redis legs
(``market_metrics:{symbol}``, ``l2_surface:{symbol}``) are written
regardless of any broker, and are what the dashboard's Command Center
(3D L2 chart, spread/imbalance metrics) actually reads.

Usage (real two-terminal setup)::

    # Terminal 1 -- this script, streaming real BTC/USDT data:
    python scripts/run_market_data_agent.py

    # Terminal 2 -- the real dashboard entrypoint:
    streamlit run src/dashboard/nothierly_ems.py --server.port 8503
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as scripts/run_backend.py: `core`/`src` live at the
# project root, which must be on sys.path before those imports below,
# regardless of the working directory this was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

from core.logger import configure_logging, get_logger
from src.agents.market_data.agent import MarketDataAgent

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT", help="ccxt.pro symbol to stream (default: BTC/USDT).")
    parser.add_argument("--exchange", default="binance", help="ccxt.pro exchange id (default: binance).")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="market-data-agent")

    agent = MarketDataAgent(args.symbol, exchange_id=args.exchange)
    _log.info("market_data_agent_launcher_starting", symbol=args.symbol, exchange=args.exchange)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

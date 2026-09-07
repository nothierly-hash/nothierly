"""Standalone launcher for a single, real :class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`.

Runs as its own process/container, deliberately -- not folded into
:mod:`scripts.run_backend`'s shared 7-agent process. This project's own
multi-agent-coordination audit found that design's real blast-radius
problem: one agent's unhandled exception kills every agent sharing that
process (see ``docker-compose.yml``'s ``nothierly-agents`` service). An
autonomous engine placing real orders against real capital 24/7 should
not go down because ``NlpAgent``'s sentiment model crashed, and vice
versa -- see ``docker-compose.yml``'s own ``alpha-engine`` service for
the isolated, independently-``restart: always``-managed container this
launches inside.

Needs real exchange API credentials (``BROKER_API_KEY``/
``BROKER_API_SECRET`` or a per-exchange override -- see
:func:`~src.execution.broker_gateway.BrokerFactory.create`) unless
``MOCK_MODE=true``, in which case it runs with no real broker connection
at all (see :meth:`~src.agents.alpha_engine.agent.
AlphaMarketMakerAgent.on_start`).

Usage::

    python scripts/run_alpha_engine.py --symbol BTC/USDT --exchange binance --max-inventory 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as scripts/run_backend.py / scripts/run_market_data_agent.py: `core`/`src` live at the
# project root, which must be on sys.path before those imports below, regardless of the working
# directory this was launched from (e.g. Docker's WORKDIR).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
from decimal import Decimal

from core.logger import configure_logging, get_logger
from src.agents.alpha_engine.agent import AlphaMarketMakerAgent

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT", help="ccxt.pro symbol to make markets on (default: BTC/USDT).")
    parser.add_argument("--exchange", default="binance", help="ccxt.pro exchange id (default: binance).")
    parser.add_argument(
        "--max-inventory", type=Decimal, default=Decimal("0.5"),
        help="Hard inventory cap, in the symbol's base asset units (default: 0.5).",
    )
    parser.add_argument(
        "--quote-size", type=Decimal, default=Decimal("0.01"),
        help="Size of each resting quote, in base asset units (default: 0.01).",
    )
    parser.add_argument(
        "--tick-interval", type=float, default=1.0,
        help="Seconds between cycles (default: 1.0 -- see AlphaMarketMakerAgent's module docstring point 5 on why not lower without colocation).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="alpha-engine")

    agent = AlphaMarketMakerAgent(
        args.symbol, exchange_id=args.exchange, max_inventory_units=args.max_inventory,
        quote_size_units=args.quote_size, tick_interval_seconds=args.tick_interval,
    )
    _log.info(
        "alpha_engine_launcher_starting", symbol=args.symbol, exchange=args.exchange,
        max_inventory=str(args.max_inventory),
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

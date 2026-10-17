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
    parser.add_argument(
        "--max-order-rate-count", type=int, default=10,
        help="Order-rate circuit breaker: max real order placements allowed per --max-order-rate-window-seconds (default: 10). See module docstring point 8.",
    )
    parser.add_argument(
        "--max-order-rate-window-seconds", type=float, default=10.0,
        help="Rolling window (seconds) the order-rate circuit breaker counts over (default: 10.0).",
    )
    parser.add_argument(
        "--max-single-order-units", type=Decimal, default=None,
        help="Fat-finger ceiling on any single real order, in base asset units (default: 2x --max-inventory). See module docstring point 9.",
    )
    parser.add_argument(
        "--max-daily-loss-usd", type=Decimal, default=None,
        help=(
            "Real capital-preservation ceiling: self-halts (forces OFF) the instant today's realized "
            "PnL drops to or below this (a negative number, e.g. -50). Left unset (the default), this "
            "guard is OFF -- a loud CRITICAL log fires at startup if so with real capital in play. "
            "There is no library-wide sane default: what counts as an acceptable daily loss depends "
            "entirely on the real capital actually deployed, which this script has no way to know for "
            "you. Set this explicitly before running with real capital."
        ),
    )
    parser.add_argument(
        "--vpin-cautious-spread-multiplier", type=Decimal, default=Decimal("2"),
        help=(
            "Real half-spread widening multiplier while AdverseFlowDefenseAgent's posture reads "
            "CAUTIOUS (elevated real VPIN on the live trade tape) -- default: 2 (double the "
            "half-spread), a starting, disclosed default, not a backtested constant. See "
            "AlphaMarketMakerAgent's module docstring, 2026-09-19 VPIN addendum."
        ),
    )
    parser.add_argument(
        "--order-rate-limit-per-second", type=float, default=3.0,
        help=(
            "Real, proactive rate-limit-ban protection (network.resilience_mesh.RateLimitMesh) -- "
            "paces real order calls to stay under the exchange's published order-rate limit instead "
            "of discovering it via an HTTP 429/418 (default: 3.0, a real safety margin under "
            "Binance spot's own ~5/s published limit; confirm and tune for your real venue). See "
            "module docstring point 13."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="alpha-engine")

    agent = AlphaMarketMakerAgent(
        args.symbol, exchange_id=args.exchange, max_inventory_units=args.max_inventory,
        quote_size_units=args.quote_size, tick_interval_seconds=args.tick_interval,
        max_order_rate_count=args.max_order_rate_count,
        max_order_rate_window_seconds=args.max_order_rate_window_seconds,
        max_single_order_units=args.max_single_order_units,
        max_daily_loss_usd=args.max_daily_loss_usd,
        vpin_cautious_spread_multiplier=args.vpin_cautious_spread_multiplier,
        order_rate_limit_per_second=args.order_rate_limit_per_second,
    )
    _log.info(
        "alpha_engine_launcher_starting", symbol=args.symbol, exchange=args.exchange,
        max_inventory=str(args.max_inventory),
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

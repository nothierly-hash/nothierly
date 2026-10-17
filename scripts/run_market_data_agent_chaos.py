"""Chaos-mode launcher for a single :class:`~src.agents.market_data.agent.MarketDataAgent` (Chaos Shadow Run, Directive 2).

Identical to :mod:`scripts.run_market_data_agent` except the agent's real
``AsyncExchangeClient`` is wrapped in :class:`~core.chaos.
binance_fault_injector.ChaosExchangeClient` before ``agent.run()`` starts
-- see that module's own docstring for why this is a wrapper applied
here, not an edit to the shared exchange-client/agent modules themselves.

**Real gotcha this script accounts for**: "Protocole AEGIS" (see
:meth:`~src.agents.market_data.agent.MarketDataAgent.
_reconnect_after_circuit_breaker_trip`) constructs a *fresh*
``AsyncExchangeClient`` on every circuit-breaker-triggered reconnect --
which would silently replace the chaos wrapper with a clean, non-chaos
client the moment the injected corruption itself trips the breaker it is
meant to test. ``_rewrap_watchdog`` below polls for exactly that and
re-wraps every time, so the fault injection survives the exact
recover-and-reconnect cycle it is supposed to be exercising, for as long
as this process runs.

Usage (same two-terminal setup as the real launcher; run this *instead
of*, never *alongside*, ``run_market_data_agent.py`` for the same
symbol)::

    python scripts/run_market_data_agent_chaos.py --symbol BTC/USDT --exchange binance \\
        --nan-probability 0.10 --flash-crash-probability 0.05 --malformed-probability 0.10
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

from core.chaos.binance_fault_injector import ChaosExchangeClient
from core.logger import configure_logging, get_logger
from src.agents.market_data.agent import MarketDataAgent

_log = get_logger(__name__)

_REWRAP_POLL_INTERVAL_SECONDS = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT", help="ccxt.pro symbol to stream (default: BTC/USDT).")
    parser.add_argument("--exchange", default="binance", help="ccxt.pro exchange id (default: binance).")
    parser.add_argument("--nan-probability", type=float, default=0.10, help="Fraction of ticks corrupted with NaN prices (default: 0.10).")
    parser.add_argument("--flash-crash-probability", type=float, default=0.05, help="Fraction of ticks corrupted into a simulated flash crash (default: 0.05).")
    parser.add_argument("--flash-crash-fraction", type=float, default=0.30, help="Magnitude of the simulated flash-crash move (default: 0.30 = 30%%, matching the mission brief).")
    parser.add_argument("--malformed-probability", type=float, default=0.10, help="Fraction of ticks corrupted into a schema-broken payload (default: 0.10).")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible chaos runs.")
    return parser.parse_args()


async def _rewrap_watchdog(agent: MarketDataAgent, *, args: argparse.Namespace) -> None:
    """Keep ``agent._exchange`` chaos-wrapped across every AEGIS reconnect. See module docstring."""
    import random as _random

    rng = _random.Random(args.seed) if args.seed is not None else None
    while True:
        await asyncio.sleep(_REWRAP_POLL_INTERVAL_SECONDS)
        current = agent._exchange  # noqa: SLF001 -- deliberate, documented chaos-only external hook, see module docstring
        if not isinstance(current, ChaosExchangeClient):
            wrapped = ChaosExchangeClient(
                current,
                nan_probability=args.nan_probability,
                flash_crash_probability=args.flash_crash_probability,
                malformed_probability=args.malformed_probability,
                flash_crash_fraction=args.flash_crash_fraction,
                rng=rng,
            )
            agent._exchange = wrapped  # noqa: SLF001
            _log.warning(
                "chaos_exchange_client_rewrapped",
                reason="agent._exchange was replaced (real client reconnect, e.g. after an AEGIS trip)",
            )


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="market-data-agent-chaos")

    agent = MarketDataAgent(args.symbol, exchange_id=args.exchange)
    _log.warning(
        "chaos_market_data_agent_starting",
        symbol=args.symbol, exchange=args.exchange,
        nan_probability=args.nan_probability, flash_crash_probability=args.flash_crash_probability,
        malformed_probability=args.malformed_probability, flash_crash_fraction=args.flash_crash_fraction,
        note="Fault injection active -- do not point this at anything but a Shadow/MOCK run.",
    )

    watchdog_task = asyncio.ensure_future(_rewrap_watchdog(agent, args=args))
    try:
        await agent.run()
    finally:
        watchdog_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())

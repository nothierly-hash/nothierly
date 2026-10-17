"""Standalone launcher for a single, real :class:`~src.agents.spectral_denoiser.agent.SpectralDenoiserAgent`.

Runs *the actual production agent* (a real TimescaleDB fetch, a real FFT
top-K denoising filter, a real ``DENOISED_PRICE_SIGNAL`` ZMQ publish, and a
real ``denoised_price_signal:{symbol}`` Redis cache write) as a lone
process -- not the multi-agent daemon (:mod:`scripts.run_backend`), and
not wired into it yet (see that agent's own module docstring: nothing
currently subscribes to ``DENOISED_PRICE_SIGNAL_TOPIC`` -- a new,
standalone signal source). Needs a reachable TimescaleDB with real OHLCV
history already loaded into ``ohlcv_candles`` (see
:mod:`src.data_pipeline.collectors.crypto_collector`) -- this script does
not seed or fabricate candle data itself.

Every default on :class:`SpectralDenoiserAgent` is already correct for
running alone -- ``manage_own_signals=True`` installs its own
SIGINT/SIGTERM handler, ``manage_own_redis=True`` connects (and later
closes) its own :class:`~core.redis_client.RedisClient`, and
``zmq_context=None`` creates its own :class:`zmq.asyncio.Context` -- the
same reasoning :mod:`scripts.run_market_data_agent` already documents for
its own agent.

**Same real gotcha as every other standalone ZMQ-publishing launcher in
this project**: the real ``DENOISED_PRICE_SIGNAL`` publish only reaches a
subscriber if something is bound at ``ZMQ_BROKER_FRONTEND_ADDRESS``/
``ZMQ_BROKER_BACKEND_ADDRESS`` (see ``.env`` -- normally
:mod:`scripts.run_backend` or a standalone
:class:`~core.zmq_broker.ZmqBroker`). ZMQ PUB is fire-and-forget, so
running this script alone, with no broker up, never blocks or errors --
the Redis cache write still happens regardless.

Usage::

    python scripts/run_spectral_denoiser_agent.py --symbol BTC/USDT
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
from src.agents.spectral_denoiser.agent import SpectralDenoiserAgent

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT", help="Instrument to denoise (default: BTC/USDT).")
    parser.add_argument(
        "--lookback-candles", type=int, default=256,
        help="Real candles fetched/FFT'd per cycle (default: 256).",
    )
    parser.add_argument(
        "--top-k-frequencies", type=int, default=8,
        help="Most energetic frequency components kept by the filter (default: 8).",
    )
    parser.add_argument(
        "--publish-window-size", type=int, default=20,
        help="Most recent denoised points actually published/cached (default: 20).",
    )
    parser.add_argument(
        "--poll-interval-seconds", type=float, default=5.0,
        help="Time between fetch-denoise-publish cycles (default: 5.0).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="spectral-denoiser-agent")

    agent = SpectralDenoiserAgent(
        args.symbol,
        lookback_candles=args.lookback_candles,
        top_k_frequencies=args.top_k_frequencies,
        publish_window_size=args.publish_window_size,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    _log.info("spectral_denoiser_agent_launcher_starting", symbol=args.symbol)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

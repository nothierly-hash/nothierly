"""Standalone launcher for a single, real :class:`~src.agents.execution.agent.ExecutionAgent` ("Fleet_Command_Arbiter").

No standalone launcher existed for this agent before this file --
``scripts/run_backend.py`` is the only place it previously ran, bundled
with 15 other agents (NLP, Macro, both signal agents, ...) it has no
chaos-testing reason to depend on. Same isolation reasoning
``scripts/run_alpha_engine.py``'s own docstring already gives for
running the market-maker alone: a focused Chaos Shadow Run (mission
Directive 1/3) needs this agent's real 8-gate pipeline (``Consensus``,
``Cooldown``, ``Sizing``, ``Plafond de Marge``, ``RiskManager``,
``PortfolioRiskParityManager``, ``StochasticCore``, ``Control State`` --
see that module's own ``_GATE_NAMES``) under load, not the other 15
agents' own startup cost and failure surface along for the ride.

Needs no real OANDA credentials to run under ``AgentControlState.SHADOW``
or ``MOCK_MODE`` (the safe-by-default setting) -- ``OandaClient.connect()``
is a documented no-op in that case; see ``.env``'s own demo placeholder
values.

Usage::

    # Real chaos target: exposure ceiling deliberately set far below any
    # real signal's sized notional, so Gate 5 (RiskManager) rejects real,
    # correctly-sized consensus entries under load -- the fat-finger-
    # equivalent stress test for this agent (mission Directive 3; see
    # scripts/chaos_shadow_report.py's own docstring for why this agent,
    # not AlphaMarketMakerAgent, is the real target for that directive).
    python scripts/run_execution_agent.py --instruments BTC/USDT --max-portfolio-exposure-pct 0.001 \\
        --enable-loop-lag-monitor --enable-memory-monitor

Requires a real :class:`~core.zmq_broker.ZmqBroker` already running
(``scripts/run_zmq_broker.py``) to receive anything a chaos injector
publishes -- see that script's own docstring on why a standalone
launcher's ZMQ publish/subscribe otherwise goes nowhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio

from core.chaos.loop_lag_monitor import start_loop_lag_monitor
from core.chaos.memory_profiler import start_memory_monitor
from core.logger import configure_logging, get_logger
from src.agents.execution.agent import ExecutionAgent

_log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--instruments", default="BTC/USDT",
        help="Comma-separated instruments this agent evaluates (default: BTC/USDT).",
    )
    parser.add_argument("--capital", type=float, default=10_000.0, help="MOCK_MODE starting balance / exposure denominator (default: 10000.0).")
    parser.add_argument(
        "--max-portfolio-exposure-pct", type=float, default=None,
        help=(
            "Gate 5 (RiskManager) ceiling, as a fraction of --capital (0 < x <= 1). Left unset, "
            "ExecutionAgent's own default (1.0, i.e. no extra ceiling) applies. Set this low "
            "(e.g. 0.001) to deliberately stress-test the real margin/exposure veto path under "
            "chaos load -- see this file's own module docstring."
        ),
    )
    parser.add_argument("--signal-validity-seconds", type=float, default=5.0, help="Consensus freshness window (default: 5.0).")
    parser.add_argument("--cooldown-seconds", type=float, default=60.0, help="Gate 2 re-entry cooldown (default: 60.0).")
    parser.add_argument("--enable-loop-lag-monitor", action="store_true", help="Start the asyncio event-loop lag watchdog (Chaos Shadow Run Directive 1).")
    parser.add_argument("--loop-lag-threshold-ms", type=float, default=25.0, help="Loop-lag alert threshold in ms (default: 25.0, the mission brief's own default).")
    parser.add_argument("--enable-memory-monitor", action="store_true", help="Start the in-process RSS/tracemalloc drift monitor (Chaos Shadow Run Directive 4).")
    parser.add_argument("--memory-drift-threshold-pct", type=float, default=5.0, help="RSS drift-from-baseline alert threshold in %% (default: 5.0, the mission brief's own default).")
    parser.add_argument("--memory-sample-interval-seconds", type=float, default=300.0, help="Seconds between memory snapshots (default: 300.0).")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="execution-agent")

    instruments = tuple(instrument.strip() for instrument in args.instruments.split(",") if instrument.strip())
    agent = ExecutionAgent(
        instruments,
        capital=args.capital,
        max_portfolio_exposure_pct=args.max_portfolio_exposure_pct,
        signal_validity_seconds=args.signal_validity_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )

    _log.info(
        "execution_agent_launcher_starting",
        instruments=list(instruments), capital=args.capital,
        max_portfolio_exposure_pct=args.max_portfolio_exposure_pct,
    )

    tasks: list[asyncio.Task[None]] = []
    if args.enable_loop_lag_monitor:
        tasks.append(start_loop_lag_monitor(sender_id=agent.agent_id, lag_threshold_ms=args.loop_lag_threshold_ms))
    if args.enable_memory_monitor:
        tasks.append(
            start_memory_monitor(
                sender_id=agent.agent_id,
                interval_seconds=args.memory_sample_interval_seconds,
                drift_threshold_pct=args.memory_drift_threshold_pct,
            )
        )

    try:
        await agent.run()
    finally:
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(main())

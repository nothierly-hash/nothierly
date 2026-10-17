"""Chaos Shadow Run signal flooder (mission Directives 1 and 3): saturates the ZMQ bus feeding ``Fleet_Command_Arbiter``.

"``Fleet_Command_Arbiter``" is not a metaphor here -- it is the literal
display name :mod:`src.agents.execution.agent` gives its own real 8-gate
pipeline (see that module's ``_ARBITER_DISPLAY_NAME``/``_GATE_NAMES``,
and the ``# -- 7-Gate execution validation (Fleet Command Arbiter) --``
section header in its source). This script publishes synthetic
:class:`~core.schemas.TradingSignal` messages onto
:data:`~core.schemas.TRADING_SIGNALS_TOPIC` (real ZMQ, via the real
:class:`~core.zmq_broker.ZmqBroker`, exactly like a real signal agent
would) at a configurable rate, in two deliberate shapes:

- **Consensus pairs**: matching signals from both of
  :class:`~src.agents.execution.agent.ExecutionAgent`'s real
  ``_DEFAULT_REQUIRED_AGENT_IDS`` (``supervisor-agent``,
  ``rl-signal-agent``), same instrument/direction, inside the real
  freshness window -- these are real enough to clear Gate 1 and reach
  the deeper gates (Sizing, Margin, RiskManager, ...), which is what
  actually stress-tests the pipeline rather than just Gate 1's own
  rejection path.
- **Noise**: signals from agent IDs ``ExecutionAgent`` never recognizes,
  or a lone unmatched side -- real traffic that Gate 1 must reject at
  volume without falling behind.

Also self-monitors this script's own publishing loop for scheduling lag
(:mod:`core.chaos.loop_lag_monitor`) and polls the real
``NOTHIERLY_FLEET_COMMAND_GATES`` Redis key (the same live telemetry
:mod:`src.dashboard.ems_pages.fleet_command_nexus` reads) to observe and
report every real gate verdict the target ``ExecutionAgent`` produces
under load -- see :func:`_poll_gate_telemetry`. This script never writes
to that key, only reads it: the real verdicts are entirely
``ExecutionAgent``'s own, not fabricated here.

Usage (requires ``scripts/run_zmq_broker.py`` and
``scripts/run_execution_agent.py`` already running)::

    python scripts/chaos_injector.py --instrument BTC/USDT --rate 75 --duration-seconds 300
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import random
import time
from typing import Any, Final

from core.chaos.loop_lag_monitor import start_loop_lag_monitor
from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient
from core.schemas import TRADING_SIGNALS_TOPIC, MessagePayload, TradingSignal
from core.zmq_bus import ZmqPublisher
from datetime import datetime, timezone

_log = get_logger(__name__)

_GATE_TELEMETRY_REDIS_KEY: Final[str] = "NOTHIERLY_FLEET_COMMAND_GATES"
_REQUIRED_AGENT_IDS: Final[tuple[str, str]] = ("supervisor-agent", "rl-signal-agent")
_NOISE_AGENT_IDS: Final[tuple[str, ...]] = (
    "chaos-noise-agent-1", "chaos-noise-agent-2", "unregistered-daemon", "supervisor-agent",
)
_GATE_POLL_INTERVAL_SECONDS: Final[float] = 0.2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", default="BTC/USDT", help="Instrument to flood signals for (must match a --instruments entry on the running ExecutionAgent).")
    parser.add_argument("--rate", type=float, default=75.0, help="Target signals published per second, summed across both shapes below (default: 75, mission range 50-100).")
    parser.add_argument("--consensus-fraction", type=float, default=0.3, help="Fraction of published signals that are real, gate-1-clearing consensus pairs rather than noise (default: 0.3).")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="How long to run (default: 300 = 5 minutes; pass a much larger value for a real overnight run).")
    parser.add_argument("--loop-lag-threshold-ms", type=float, default=25.0, help="This script's own publish-loop lag alert threshold, ms (default: 25.0).")
    parser.add_argument("--report-path", default="logs/chaos_gate_report.jsonl", help="Where to append observed gate-verdict transitions (default: logs/chaos_gate_report.jsonl).")
    return parser.parse_args()


def _make_consensus_pair(instrument: str, rng: random.Random) -> list[TradingSignal]:
    direction = rng.choice((-1, 1))
    now = time.time()
    return [
        TradingSignal(timestamp=now, agent_id=agent_id, instrument=instrument, signal=direction, confidence=rng.uniform(0.6, 1.0))
        for agent_id in _REQUIRED_AGENT_IDS
    ]


def _make_noise_signal(instrument: str, rng: random.Random) -> TradingSignal:
    return TradingSignal(
        timestamp=time.time(),
        agent_id=rng.choice(_NOISE_AGENT_IDS),
        instrument=instrument,
        signal=rng.choice((-1, 0, 1)),
        confidence=rng.uniform(0.0, 1.0),
    )


async def _publish_loop(
    publisher: ZmqPublisher, *, instrument: str, rate: float, consensus_fraction: float,
    duration_seconds: float, sender_id: str, rng: random.Random,
) -> tuple[int, int]:
    """Publish at ``rate`` msgs/sec for ``duration_seconds``. Returns ``(consensus_pairs_sent, noise_sent)``."""
    interval = 1.0 / rate if rate > 0 else 0.0
    deadline = time.monotonic() + duration_seconds
    consensus_sent = 0
    noise_sent = 0

    while time.monotonic() < deadline:
        cycle_start = time.monotonic()
        if rng.random() < consensus_fraction:
            for signal in _make_consensus_pair(instrument, rng):
                await publisher.publish(
                    MessagePayload(topic=TRADING_SIGNALS_TOPIC, sender_id=sender_id, data=signal.model_dump())
                )
            consensus_sent += 1
        else:
            signal = _make_noise_signal(instrument, rng)
            await publisher.publish(
                MessagePayload(topic=TRADING_SIGNALS_TOPIC, sender_id=sender_id, data=signal.model_dump())
            )
            noise_sent += 1

        elapsed = time.monotonic() - cycle_start
        remaining = interval - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

    return consensus_sent, noise_sent


async def _poll_gate_telemetry(*, instrument: str, duration_seconds: float, report_path: Path) -> None:
    """Poll the real Fleet Command gate telemetry Redis key; log+persist every real verdict transition.

    Reads ``NOTHIERLY_FLEET_COMMAND_GATES`` -- the exact key
    :meth:`~src.agents.execution.agent.ExecutionAgent._publish_gate_report`
    overwrites on every real gate evaluation. Detects a transition by
    comparing ``timestamp`` per instrument between polls (the key is a
    live snapshot, not an append-only log -- polling fast enough,
    :data:`_GATE_POLL_INTERVAL_SECONDS`, to not miss one entirely under
    the mission's own 50-100/s injection rate).
    """
    settings = get_settings()
    client = await RedisClient.get_instance(settings.redis)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    last_seen_timestamp: float | None = None
    rejected_count = 0
    total_seen = 0
    deadline = time.monotonic() + duration_seconds

    with report_path.open("a", encoding="utf-8") as report_file:
        while time.monotonic() < deadline:
            await asyncio.sleep(_GATE_POLL_INTERVAL_SECONDS)
            try:
                raw = await client.execute("get", _GATE_TELEMETRY_REDIS_KEY)
            except Exception:
                continue
            if raw is None:
                continue
            try:
                state: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                continue

            record = state.get(instrument)
            if record is None:
                continue
            record_timestamp = record.get("timestamp")
            if record_timestamp == last_seen_timestamp:
                continue
            last_seen_timestamp = record_timestamp
            total_seen += 1

            final_action = record.get("final_action")
            tag = "[CHAOS_GATE_REJECT]" if final_action == "REJECTED" else "[CHAOS_GATE_PASS]"
            if final_action == "REJECTED":
                rejected_count += 1
                _log.warning(
                    "chaos_gate_reject_observed", tag=tag, instrument=instrument,
                    final_action=final_action, gates=record.get("gates"),
                )
            else:
                _log.info("chaos_gate_pass_observed", tag=tag, instrument=instrument, final_action=final_action)

            report_file.write(json.dumps({"tag": tag, "observed_at": datetime.now(timezone.utc).isoformat(), **record}) + "\n")
            report_file.flush()

    _log.info(
        "chaos_gate_telemetry_summary", instrument=instrument,
        total_verdicts_observed=total_seen, rejected=rejected_count,
        rejected_pct=round(rejected_count / total_seen * 100.0, 1) if total_seen else None,
    )


async def main() -> None:
    args = _parse_args()
    configure_logging(log_level="INFO", log_dir=None, service_name="chaos-injector")
    settings = get_settings()
    zmq_settings = settings.zmq
    if zmq_settings.broker_frontend_address is None:
        raise SystemExit("ZMQ_BROKER_FRONTEND_ADDRESS must be set (see .env) -- nothing to connect to.")

    rng = random.Random()
    sender_id = "chaos-injector"
    publisher = ZmqPublisher(zmq_settings.broker_frontend_address, bind=False)
    lag_task = start_loop_lag_monitor(sender_id=sender_id, lag_threshold_ms=args.loop_lag_threshold_ms)

    _log.warning(
        "chaos_injector_starting", instrument=args.instrument, rate=args.rate,
        consensus_fraction=args.consensus_fraction, duration_seconds=args.duration_seconds,
        broker_frontend=zmq_settings.broker_frontend_address,
    )

    try:
        (consensus_sent, noise_sent), _ = await asyncio.gather(
            _publish_loop(
                publisher, instrument=args.instrument, rate=args.rate,
                consensus_fraction=args.consensus_fraction, duration_seconds=args.duration_seconds,
                sender_id=sender_id, rng=rng,
            ),
            _poll_gate_telemetry(
                instrument=args.instrument, duration_seconds=args.duration_seconds,
                report_path=Path(args.report_path),
            ),
        )
    finally:
        lag_task.cancel()
        publisher.close()
        await RedisClient.reset_instance()

    _log.warning(
        "chaos_injector_finished", consensus_pairs_sent=consensus_sent, noise_signals_sent=noise_sent,
        total_signals_sent=consensus_sent * 2 + noise_sent,
    )


if __name__ == "__main__":
    asyncio.run(main())

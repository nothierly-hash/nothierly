"""Standalone offline replay harness: proves real Kill Switch + Gate-pipeline behavior against a
recorded (or generated) tick log -- zero external infrastructure (fakeredis in-process, no real
Redis/broker/ZMQ-broker process required).

**Corrections to the brief, verified against the real code first** (this project's own established
discipline before writing anything):

1. **"ResilientKillSwitch"** -- no such class exists anywhere in this codebase. Earlier this same
   session, the choice was made explicitly (not assumed) to apply that 3-state design to the real,
   already-live mechanism instead of a new, disconnected class: :mod:`core.security`
   (``STATUS_LATCHED_DATA``/:func:`~core.security.latch_data_loss`/
   :func:`~core.security.auto_recover_from_data_loss`) plus :class:`~src.agents.supervisor.agent.
   XgboostSignalAgent`'s real :meth:`~src.agents.supervisor.agent.
   XgboostSignalAgent._check_kill_switch`/:meth:`~src.agents.supervisor.agent.
   XgboostSignalAgent._track_data_loss_probation`. This harness drives those real functions
   directly -- "instancie un ResilientKillSwitch" is not literally possible, "exercise the real
   Kill Switch state machine" is, and is what this does.
2. **"``execution_agent.evaluate_pipeline(tick)``"** -- no such method exists. The real gate
   pipeline is :class:`~src.agents.execution.agent.ExecutionAgent`'s
   :meth:`~src.agents.execution.agent.ExecutionAgent._validate_execution_gates`
   (``instrument, direction, signals) -> (GateReport, price, balance, sized_units)``), called here
   directly with two synthesized, already-agreeing :class:`~core.schemas.TradingSignal`\\ s (see
   :func:`_synthetic_consensus_signals`) -- this harness's own disclosed simplification: the real
   consensus-arbitration step (:meth:`~src.agents.execution.agent.
   ExecutionAgent._evaluate_consensus`) needs real ZMQ signal traffic from two separate agents,
   out of scope for a single-process offline replay. The real global Kill Switch gate
   (:func:`~core.security.is_system_safe`, checked via ``@requires_system_active`` on
   :meth:`~src.agents.execution.agent.ExecutionAgent._place_order` -- *not* inside
   ``_validate_execution_gates`` itself, confirmed by reading the real decorator placement) is
   checked explicitly here, before calling ``_validate_execution_gates``, matching the real
   production call order without needing the full ``_place_order``/broker-call machinery.
3. **"157 ticks (activations VPIN réelles) sauvegardés en JSON"** -- no such per-tick log file
   exists in this repo. 157 is a real number, but it is
   ``alpha_engine_defensive_posture_active``'s real *event count* from this project's own real 24h
   Chaos Shadow Run (``logs/shadow_run_24h_20260923_024500/digest.json``) -- an aggregated log
   count, not persisted per-tick bid/ask/timestamp data usable for replay. ``--generate-sample``
   below writes a real, clearly-labeled *synthetic* 157-tick fixture instead of pretending to
   reproduce one that was never captured.

**Real, disclosed clock limitation -- read before trusting this on anything but the happy path**:
:func:`~core.stale_check.staleness_reason` (per-tick freshness) always compares a tick's timestamp
against the real wall clock (``datetime.now(timezone.utc)``) -- it has no injectable clock for
replaying old, real recorded timestamps deterministically. This harness rebases the log's own
*relative* tick spacing (preserved exactly) onto real "now" at replay start, so a "perfect stream"
(evenly-spaced, no gaps -- what this harness exists to validate) reads as fresh throughout
regardless of how old the original recording is.

A second, more fundamental instance of the same limitation, found by actually testing this harness
against a real, deliberately-inserted 40s gap before trusting it: :meth:`~src.agents.supervisor.
agent.XgboostSignalAgent._check_kill_switch`'s *feed-loss* detection (``feed_loss_grace_seconds``)
measures ``time.monotonic()`` -- real elapsed wall-clock time between calls -- not any timestamp
carried on the ticks themselves. A fast, back-to-back replay processes every tick within
milliseconds of real time regardless of what gap the *recorded* timestamps claim, so **this
harness can only faithfully validate the happy path** (a real gap in the log will silently *not*
latch the Kill Switch here, however large). Reproducing a real historical gap's Kill-Switch
behavior would need the replay loop to actually sleep in step with each tick's real recorded
delta -- a real, disclosed, not-yet-built extension, not something this harness silently fakes.

Usage::

    python scripts/replay_harness.py --generate-sample logs/replay_sample.jsonl
    python scripts/replay_harness.py --input logs/replay_sample.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Final, Literal

import fakeredis
import fakeredis.aioredis

from core.config import AppSettings, RedisSettings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient
from core.schemas import MarketMetrics, TradingSignal
from core.security import is_system_safe
from src.agents.execution.agent import ExecutionAgent
from src.agents.supervisor.agent import EmergencyStopError, SignalState, XgboostSignalAgent

_log = get_logger(__name__)

_INSTRUMENT: Final[str] = "EUR_USD"
_SAMPLE_TICK_COUNT: Final[int] = 157
_DEFAULT_SPREAD: Final[float] = 0.00015


def _generate_sample_ticks(path: Path, *, count: int = _SAMPLE_TICK_COUNT) -> None:
    """Write a real, clearly-synthetic JSON-lines tick log -- see module docstring, item 3."""
    rng = random.Random(0)
    price = 1.10000
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            price += rng.uniform(-0.0002, 0.0002)
            tick = {"ts": 1_700_000_000 + i, "bid": round(price, 5), "ask": round(price + _DEFAULT_SPREAD, 5)}
            f.write(json.dumps(tick) + "\n")


def _read_ticks(path: Path) -> list[dict[str, Any]]:
    """Real JSON-lines read -- one tick object per non-blank line."""
    ticks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ticks.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON line: {exc}") from exc
    if not ticks:
        raise ValueError(f"{path} contains no ticks.")
    return ticks


def _rebased_timestamps(ticks: list[dict[str, Any]], *, replay_start: datetime) -> list[datetime]:
    """Preserve each tick's real relative spacing, anchored to real "now". See module docstring's clock note."""
    first_ts = float(ticks[0]["ts"])
    return [replay_start + timedelta(seconds=float(tick["ts"]) - first_ts) for tick in ticks]


def _tick_to_metrics(tick: dict[str, Any], timestamp: datetime) -> MarketMetrics:
    bid = float(tick["bid"])
    ask = float(tick.get("ask", bid + _DEFAULT_SPREAD))
    return MarketMetrics(
        symbol=_INSTRUMENT, spread=max(ask - bid, 0.0), order_book_imbalance=0.0,
        mid_price=(bid + ask) / 2.0, timestamp=timestamp,
    )


def _synthetic_consensus_signals(
    previous_mid: float | None, current_mid: float, tick_timestamp: float,
) -> dict[str, TradingSignal]:
    """Two already-agreeing signals, direction from a trivial tick-to-tick price-move heuristic.

    Real, disclosed simplification (module docstring, item 2): not a real strategy, only a
    deterministic way to drive the real consensus-gated gate pipeline without needing the real
    per-agent ZMQ signal traffic :meth:`~src.agents.execution.agent.
    ExecutionAgent._evaluate_consensus` normally arbitrates.
    """
    direction: Literal[-1, 1] = 1 if previous_mid is None or current_mid >= previous_mid else -1
    kwargs: dict[str, Any] = dict(timestamp=tick_timestamp, instrument=_INSTRUMENT, signal=direction, confidence=0.8)
    return {
        "rl-signal-agent": TradingSignal(agent_id="rl-signal-agent", **kwargs),
        "supervisor-agent": TradingSignal(agent_id="supervisor-agent", **kwargs),
    }


def _highest_passed_gate(report: Any) -> int:
    return max((gate.number for gate in report.gates if gate.status == "PASSED"), default=0)


async def _run_replay(input_path: Path) -> int:
    """Real end-to-end replay. Returns the count of ticks that reached ``gate_reached >= 4``.

    Raises:
        AssertionError: If that count is ``0`` at the end of the file.
    """
    fake_redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    redis_client = RedisClient(RedisSettings())
    redis_client._redis = fake_redis
    RedisClient._instance = redis_client

    xgb_agent = XgboostSignalAgent(_INSTRUMENT, manage_own_signals=False, manage_own_redis=False)
    xgb_agent._redis_client = redis_client

    exec_agent = ExecutionAgent(
        [_INSTRUMENT], settings=AppSettings(mock_mode=True), manage_own_signals=False, manage_own_redis=False,
    )
    exec_agent._redis_client = redis_client

    ticks = _read_ticks(input_path)
    replay_start = datetime.now(timezone.utc)
    timestamps = _rebased_timestamps(ticks, replay_start=replay_start)

    gate_reached_ge_4 = 0
    latch_events = 0
    previous_mid: float | None = None

    try:
        for i, (tick, timestamp) in enumerate(zip(ticks, timestamps), start=1):
            metrics = _tick_to_metrics(tick, timestamp)
            state = SignalState(metrics, {}, [], redis_latency_ms=1.0)

            try:
                await xgb_agent._check_kill_switch(state)
            except EmergencyStopError as exc:
                latch_events += 1
                _log.warning("replay_kill_switch_event", tick=i, total=len(ticks), reason=str(exc))
                previous_mid = metrics.mid_price
                continue

            if not await is_system_safe():
                _log.info("replay_tick_skipped_kill_switch_active", tick=i, total=len(ticks))
                previous_mid = metrics.mid_price
                continue

            signals = _synthetic_consensus_signals(previous_mid, metrics.mid_price, timestamp.timestamp())
            direction = signals["rl-signal-agent"].signal
            exec_agent._last_prices[_INSTRUMENT] = metrics.mid_price
            exec_agent._last_spreads[_INSTRUMENT] = metrics.spread

            report, _price, _balance, _sized_units = await exec_agent._validate_execution_gates(
                _INSTRUMENT, direction, signals,
            )
            gate_reached = _highest_passed_gate(report)
            if gate_reached >= 4:
                gate_reached_ge_4 += 1
            _log.info(
                "replay_tick_processed", tick=i, total=len(ticks), gate_reached=gate_reached,
                final_action=report.final_action,
            )
            previous_mid = metrics.mid_price
    finally:
        for agent in (xgb_agent, exec_agent):
            agent._publisher.close()
            agent._subscriber.close()
            agent._zmq_context.term()

    print(f"Ticks replayed: {len(ticks)}")
    print(f"Kill Switch events (LATCHED_DATA/PANIC raised this cycle): {latch_events}")
    print(f"Ticks reaching gate_reached >= 4: {gate_reached_ge_4}")

    if gate_reached_ge_4 == 0:
        raise AssertionError(
            f"0/{len(ticks)} ticks reached gate_reached >= 4 -- the execution pipeline never got "
            "past Gate 3 (Sizing) even once in this replay. Investigate before trusting Gates 4-7 "
            "in production."
        )
    return gate_reached_ge_4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="JSON-lines tick log to replay.")
    parser.add_argument(
        "--generate-sample", type=Path, default=None,
        help=(
            f"Write a real, clearly-synthetic {_SAMPLE_TICK_COUNT}-tick JSON-lines fixture to this "
            "path and exit -- see module docstring, item 3, on why no real captured tick-level log "
            "exists to replay instead."
        ),
    )
    return parser.parse_args()


def main() -> int:
    configure_logging(log_level="INFO", log_dir=None, service_name="replay-harness")
    args = _parse_args()

    if args.generate_sample is not None:
        _generate_sample_ticks(args.generate_sample)
        print(f"Wrote {_SAMPLE_TICK_COUNT} synthetic ticks to {args.generate_sample}")
        return 0

    if args.input is None:
        print("error: --input <path> is required (or --generate-sample <path> to create one first).", file=sys.stderr)
        return 2

    asyncio.run(_run_replay(args.input))
    return 0


if __name__ == "__main__":
    sys.exit(main())

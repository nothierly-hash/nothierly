"""Tick-to-decision and tick-to-order latency of the real Python pipeline, measured on the real ZMQ bus.

A bus consumer: subscribes to ``MARKET_DATA_L2``, and for every tick runs, in this order, the code the
repository already has (the same recipe as ``scripts/replay_harness.py``):

1. decode: ``MessagePayload.deserialize`` + ``MarketMetrics.model_validate``;
2. risk pre-check: ``XgboostSignalAgent._check_kill_switch`` (spread veto, flash-crash detector, staleness,
   data-loss probation, against an in-process fakeredis);
3. alpha: the z-score from ``scripts/zscore_backtest.py`` (window 100, entry 3.0, exit 0.5);
4. on an entry signal only: ``is_system_safe`` and ``ExecutionAgent._validate_execution_gates`` (the Risk Gates).

``T1`` is ``time.perf_counter_ns()`` taken the moment ``recv_multipart`` returns to this code. ``T2`` is taken
right after the last stage and before the order is handed to a sink that only counts. Two latencies are
recorded per tick: ``tick_to_decision_ns`` (T1 to the end of the alpha step, every tick) and
``tick_to_order_ns`` (T1 to T2, entry signals only, so it includes the Risk Gates).

What this is not, so nobody reads more into it:

- Python, not C++. There is no native code in this repository. The figures describe CPython on this machine.
- Not wire-to-wire. Time spent in the kernel, in the broker and in the ZMQ queue before ``recv`` returns is
  not counted in the latencies: they are service time per tick. The queue delay is recorded separately, per
  tick, as ``ingest_lag_ns`` (the publisher's own timestamp to T1, same machine and clock). The Python
  ingester in blast mode (``--speed 0``) publishes at about the rate this pipeline can consume, so a blast can
  build a queue; the data is then stale by the time it is processed, and the real stale-feed Kill Switch
  latches. Those events are counted (``risk_event``, and orders refused while latched have ``gate_reached`` 0)
  and the run continues. For service time at a sustainable rate, pace the replay with the ingester's ``--speed``.
- There is no broker and no FIX. The order sink counts orders. Order construction, position bookkeeping,
  the audit ledger and any network send are not measured, so tick-to-order is a lower bound on a real order path.
- The Redis behind the gates is fakeredis in the same process. A real Redis adds a network round trip per
  call. The gates run against synthetic, already agreeing signals, as in ``replay_harness.py``.
- The flash-crash detector is built with a threshold of 100% so it never trips. A blast compresses a day of
  data into minutes, and the detector's 60 s window is real time, so it would trip on the compression, not
  on the market. Its per-tick cost is still measured; its trip path is not.
- Logging is set to WARNING: the cost of INFO logging is not in the figures.
- The timer itself costs time; its measured per-call cost is written to the metadata file.

Warm-up: the first ``--warmup`` ticks run through the whole pipeline (interpreter, caches, pydantic) and are
not recorded; the strategy and detector state are then reset so the warm-up prices do not leak in.

The result is written when the run ends: after ``--expected`` recorded ticks, after ``--idle-timeout``
seconds without a message, or on Ctrl-C. Wait for that instead of killing the process (on Windows a kill
is a hard stop and the file is lost). Buffers are allocated once, before the first tick.

Usage (broker + ingester started for you by ``scripts/run_replay.sh``)::

    REPLAY_CONSUMER_CMD="python scripts/latency_profile.py --output t2t_benchmark.csv --warmup 10000 --symbol BTC_USDT" \\
        bash scripts/run_replay.sh trades.csv --symbol BTC_USDT --timestamp-column time --bid-column price \\
        --timestamp-unit ms --assumed-spread 1.0 --speed 0
    python scripts/analyze_t2t.py t2t_benchmark.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import array
import asyncio
import csv
import gc
import json
import platform
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Final

# Sets the Windows selector event loop policy; must come before any event loop exists.
import core.zmq_bus  # noqa: F401
import zmq
import zmq.asyncio

from core.schemas import MarketMetrics, MessagePayload
from scripts.zscore_backtest import ZScoreStrategy

TOPIC: Final[bytes] = b"MARKET_DATA_L2"
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "seq", "signal", "gate_reached", "risk_event", "ingest_lag_ns", "decode_ns", "risk_check_ns", "alpha_ns",
    "gates_ns", "tick_to_decision_ns", "tick_to_order_ns",
)
SIGNAL_NAMES: Final[dict[int, str]] = {0: "none", 1: "BUY", 2: "SELL", 3: "CLOSE"}
ALL_GATES_PASSED: Final[int] = 8
_ENTRY_SIGNALS: Final[frozenset[int]] = frozenset({1, 2})

RiskCheck = Callable[[MarketMetrics], Awaitable[bool]]  # True when the check raised a Kill Switch event on this tick
OrderGates = Callable[[MarketMetrics, int], Awaitable[int]]


class LatencyLog:
    """Fixed-size, column-oriented record of per-tick latencies. Allocated once; ``record`` never grows it."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.count = 0
        self.dropped = 0
        self._columns = {name: array.array("q", bytes(8 * capacity)) for name in CSV_COLUMNS[1:]}

    def record(self, *, signal_code: int, gate_reached: int, risk_event: int, ingest_lag_ns: int, decode_ns: int,
               risk_check_ns: int, alpha_ns: int, gates_ns: int, tick_to_decision_ns: int, tick_to_order_ns: int) -> None:
        index = self.count
        if index >= self.capacity:
            self.dropped += 1
            return
        columns = self._columns
        columns["signal"][index] = signal_code
        columns["gate_reached"][index] = gate_reached
        columns["risk_event"][index] = risk_event
        columns["ingest_lag_ns"][index] = ingest_lag_ns
        columns["decode_ns"][index] = decode_ns
        columns["risk_check_ns"][index] = risk_check_ns
        columns["alpha_ns"][index] = alpha_ns
        columns["gates_ns"][index] = gates_ns
        columns["tick_to_decision_ns"][index] = tick_to_decision_ns
        columns["tick_to_order_ns"][index] = tick_to_order_ns
        self.count = index + 1

    def write_csv(self, path: Path) -> int:
        columns = [self._columns[name] for name in CSV_COLUMNS[1:]]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for index in range(self.count):
                writer.writerow([index, *(column[index] for column in columns)])
        return self.count


@dataclass
class Stages:
    """What the pipeline calls. ``risk_check`` and ``order_gates`` are the real ones in a run; tests pass fakes."""

    risk_check: RiskCheck
    order_gates: OrderGates
    new_strategy: Callable[[], ZScoreStrategy]
    on_reset: Callable[[], None] = lambda: None


@dataclass
class RunSummary:
    received: int = 0
    warmup_done: int = 0
    recorded: int = 0
    dropped: int = 0
    malformed: int = 0
    risk_events: int = 0
    orders: int = 0
    stopped_because: str = ""
    rss_mb_samples: list[tuple[int, float]] = field(default_factory=list)


class NullOrderSink:
    """Where an order would be handed to a broker. There is no broker here: it only counts."""

    def __init__(self) -> None:
        self.sent = 0

    def send(self, side: int) -> None:
        self.sent += 1


def timer_overhead_ns(samples: int = 200_000) -> float:
    """Mean cost of one ``perf_counter_ns()`` call, from back-to-back calls."""
    perf = time.perf_counter_ns
    start = perf()
    for _ in range(samples):
        perf()
    return (perf() - start) / samples


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


async def profile(
    sub: Any, stages: Stages, log: LatencyLog, *, warmup: int, expected: int | None, idle_timeout: float,
    stop: asyncio.Event, rss_every: int = 50_000, poll_seconds: float = 1.0, summary: RunSummary | None = None,
) -> RunSummary:
    """Receive ticks and measure them until ``expected`` ticks are recorded, the feed goes idle, or ``stop`` is set.

    ``summary`` may be passed in so the caller still has the counters if this raises.
    """
    perf = time.perf_counter_ns
    sink = NullOrderSink()
    if summary is None:
        summary = RunSummary()
    strategy = stages.new_strategy()
    position = 0
    last_rx = time.monotonic()
    while not stop.is_set():
        try:
            async with asyncio.timeout(poll_seconds):
                frames = await sub.recv_multipart()
        except TimeoutError:
            if summary.received and time.monotonic() - last_rx > idle_timeout:
                summary.stopped_because = f"no message for {idle_timeout:g}s"
                break
            continue
        t1 = perf()
        wall_ns = time.time_ns()
        last_rx = time.monotonic()
        summary.received += 1

        try:
            payload = MessagePayload.deserialize(frames[1])
            metrics = MarketMetrics.model_validate(payload.data)
        except Exception:
            summary.malformed += 1
            continue
        t_decoded = perf()
        # Publisher's own timestamp to T1: transport plus time queued behind earlier ticks. Same machine, same clock.
        ingest_lag_ns = wall_ns - round(payload.timestamp.timestamp() * 1e9)

        risk_event = await stages.risk_check(metrics)
        t_risk = perf()
        if risk_event:
            summary.risk_events += 1

        z = strategy.update(metrics.mid_price)
        action = strategy.decide(z, position)
        t_alpha = perf()

        signal_code = 0
        gate_reached = 0
        gates_ns = 0
        t2 = t_alpha
        if action is not None:
            signal_code = {"BUY": 1, "SELL": 2, "CLOSE": 3}[action]
            sent = True
            if signal_code in _ENTRY_SIGNALS:
                gate_reached = await stages.order_gates(metrics, 1 if signal_code == 1 else -1)
                t2 = perf()
                gates_ns = t2 - t_alpha
                sent = gate_reached == ALL_GATES_PASSED
            if sent:
                position = {1: 1, 2: -1, 3: 0}[signal_code]  # the sink "fills" instantly; no fill model here
                sink.send(signal_code)
                summary.orders += 1

        if summary.warmup_done < warmup:
            summary.warmup_done += 1
            if summary.warmup_done == warmup:
                strategy = stages.new_strategy()
                position = 0
                stages.on_reset()
                gc.collect()
                summary.rss_mb_samples.append((summary.received, _rss_mb()))
                summary.orders = 0
                summary.risk_events = 0
            continue

        log.record(
            signal_code=signal_code, gate_reached=gate_reached, risk_event=int(risk_event), ingest_lag_ns=ingest_lag_ns,
            decode_ns=t_decoded - t1, risk_check_ns=t_risk - t_decoded,
            alpha_ns=t_alpha - t_risk, gates_ns=gates_ns, tick_to_decision_ns=t_alpha - t1,
            tick_to_order_ns=(t2 - t1) if signal_code in _ENTRY_SIGNALS else 0,
        )
        if rss_every and log.count % rss_every == 0:
            summary.rss_mb_samples.append((summary.received, _rss_mb()))
        if expected is not None and log.count >= expected:
            summary.stopped_because = f"recorded the expected {expected:,} ticks"
            break
    if not summary.stopped_because:
        summary.stopped_because = "stopped by signal"
    summary.recorded = log.count
    summary.dropped = log.dropped
    return summary


def build_real_stages(symbol: str, *, entry_z: float, exit_z: float, window: int) -> tuple[Stages, Callable[[], None]]:
    """The real agents on an in-process fakeredis, as in ``scripts/replay_harness.py``. Returns (stages, close)."""
    import fakeredis
    import fakeredis.aioredis

    from core.config import AppSettings, RedisSettings
    from core.redis_client import RedisClient
    from core.schemas import TradingSignal
    from core.security import is_system_safe
    from src.agents.execution.agent import ExecutionAgent
    from src.agents.supervisor.agent import EmergencyStopError, SignalState, XgboostSignalAgent

    redis_client = RedisClient(RedisSettings())
    redis_client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = redis_client

    def make_kill_switch_agent() -> XgboostSignalAgent:
        agent = XgboostSignalAgent(symbol, manage_own_signals=False, manage_own_redis=False, flash_crash_max_move_fraction=1.0)
        agent._redis_client = redis_client
        return agent

    risk_agent = make_kill_switch_agent()
    exec_agent = ExecutionAgent([symbol], settings=AppSettings(mock_mode=True), manage_own_signals=False, manage_own_redis=False)
    exec_agent._redis_client = redis_client
    holder = {"risk": risk_agent}

    async def risk_check(metrics: MarketMetrics) -> bool:
        try:
            await holder["risk"]._check_kill_switch(SignalState(metrics, {}, [], redis_latency_ms=1.0))
        except EmergencyStopError:
            return True  # a real Kill Switch event (for example a stale feed); counted, and the run goes on
        return False

    async def order_gates(metrics: MarketMetrics, direction: int) -> int:
        if not await is_system_safe():
            return 0  # the real order path refuses while the Kill Switch is latched (see requires_system_active)
        exec_agent._last_prices[symbol] = metrics.mid_price
        exec_agent._last_spreads[symbol] = metrics.spread
        now = time.time()
        kwargs: dict[str, Any] = dict(timestamp=now, instrument=symbol, signal=direction, confidence=0.8)
        signals = {
            "rl-signal-agent": TradingSignal(agent_id="rl-signal-agent", **kwargs),
            "supervisor-agent": TradingSignal(agent_id="supervisor-agent", **kwargs),
        }
        report, _price, _balance, _units = await exec_agent._validate_execution_gates(symbol, direction, signals)
        return max((gate.number for gate in report.gates if gate.status == "PASSED"), default=0)

    def on_reset() -> None:
        old = holder["risk"]
        holder["risk"] = make_kill_switch_agent()
        _close_agent(old)

    def close() -> None:
        _close_agent(holder["risk"])
        _close_agent(exec_agent)

    stages = Stages(
        risk_check=risk_check, order_gates=order_gates,
        new_strategy=lambda: ZScoreStrategy(window=window, entry_z=entry_z, exit_z=exit_z), on_reset=on_reset,
    )
    return stages, close


def _close_agent(agent: Any) -> None:
    agent._publisher.close()
    agent._subscriber.close()
    agent._zmq_context.term()


def _metadata(args: argparse.Namespace, summary: RunSummary, timer_ns: float, gc_before: list[dict[str, int]]) -> dict[str, Any]:
    gc_after = gc.get_stats()
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "processor": platform.processor(), "pyzmq": zmq.__version__,
        "perf_counter_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
        "perf_counter_call_overhead_ns": round(timer_ns, 1),
        "symbol": args.symbol, "warmup_ticks": args.warmup, "window": args.window, "entry_z": args.entry_z, "exit_z": args.exit_z,
        "received": summary.received, "recorded": summary.recorded, "dropped_buffer_full": summary.dropped,
        "malformed": summary.malformed, "orders_after_warmup": summary.orders, "kill_switch_events_after_warmup": summary.risk_events, "stopped_because": summary.stopped_because,
        "gc_collections_during_run": [after["collections"] - before["collections"] for before, after in zip(gc_before, gc_after)],
        "rss_mb_samples": [{"received": received, "rss_mb": round(rss, 1)} for received, rss in summary.rss_mb_samples],
    }


async def _amain(args: argparse.Namespace) -> int:
    from core.config import get_settings
    from core.logger import configure_logging

    configure_logging(log_level=args.log_level, log_dir=None, service_name="latency-profile")
    address = args.address or get_settings().zmq.broker_backend_address
    if not address:
        print("no broker backend address: pass --address or set ZMQ_BROKER_BACKEND_ADDRESS")
        return 2

    log = LatencyLog(args.capacity)
    stages, close_stages = build_real_stages(args.symbol, entry_z=args.entry_z, exit_z=args.exit_z, window=args.window)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(stop.set))

    context = zmq.asyncio.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVHWM, 0)
    sub.connect(address)
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC)
    print(f"subscribed to {TOPIC.decode()} at {address}; warm-up {args.warmup:,} ticks, capacity {args.capacity:,}", flush=True)

    timer_ns = timer_overhead_ns()
    gc_before = gc.get_stats()
    rss_start = _rss_mb()
    summary = RunSummary()
    crashed = False
    try:
        await profile(
            sub, stages, log, warmup=args.warmup, expected=args.expected, idle_timeout=args.idle_timeout, stop=stop,
            summary=summary,
        )
    except Exception as exc:  # keep what was measured: a crash must not lose the run
        crashed = True
        summary.stopped_because = f"crashed: {exc!r}"
        summary.recorded, summary.dropped = log.count, log.dropped
        print(f"pipeline raised {exc!r}; writing the {log.count:,} ticks recorded so far", flush=True)
    finally:
        sub.close()
        context.term()
        close_stages()

    summary.rss_mb_samples.insert(0, (0, rss_start))
    summary.rss_mb_samples.append((summary.received, _rss_mb()))
    written = log.write_csv(args.output)
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(_metadata(args, summary, timer_ns, gc_before), indent=2), encoding="utf-8")
    print(
        f"received {summary.received:,}, warm-up {summary.warmup_done:,}, recorded {written:,} "
        f"(orders {summary.orders:,}, kill-switch events {summary.risk_events:,}, malformed {summary.malformed:,}, "
        f"dropped {summary.dropped:,}); stopped: {summary.stopped_because}; wrote {args.output} and {meta_path}",
        flush=True,
    )
    return 1 if crashed or not written else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("t2t_benchmark.csv"), help="CSV to write (default: t2t_benchmark.csv); a .meta.json is written next to it.")
    parser.add_argument("--address", default=None, help="Broker backend address (default: ZMQ_BROKER_BACKEND_ADDRESS).")
    parser.add_argument("--symbol", default="BTC_USDT", help="Instrument the agents are built for (default: BTC_USDT).")
    parser.add_argument("--warmup", type=int, default=10_000, help="Ticks processed but not recorded at the start (default: 10000).")
    parser.add_argument("--expected", type=int, default=None, help="Stop after recording this many ticks (default: run until idle).")
    parser.add_argument("--idle-timeout", type=float, default=15.0, help="Stop after this many seconds without a message, once one has arrived (default: 15).")
    parser.add_argument("--capacity", type=int, default=1_000_000, help="Rows preallocated for the latency log (default: 1000000); further ticks are counted as dropped.")
    parser.add_argument("--window", type=int, default=100, help="Z-score window in ticks (default: 100).")
    parser.add_argument("--entry-z", type=float, default=3.0, help="Z-score entry threshold (default: 3.0).")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score exit threshold (default: 0.5).")
    parser.add_argument("--log-level", default="WARNING", help="Logging level (default: WARNING; INFO logging cost is not in the figures).")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_amain(_parse_args()))


if __name__ == "__main__":
    sys.exit(main())

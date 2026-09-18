"""Consolidated Chaos Shadow Run report -- combines the outputs of every other ``chaos_*``/``run_*_chaos`` script into one GO/ATTENTION/NO-GO read.

Same reasoning as :mod:`scripts.analyze_shadow_run` (this project's own
established pattern): parses each component's *real* structured JSON
log/report rather than re-deriving anything, and never fabricates a
number a component didn't actually produce. Inputs, all optional (a
partial chaos run still gets an honest partial report):

- ``--market-data-log``: stdout of ``scripts/run_market_data_agent_chaos.py``
  -- real ``chaos_injected_*`` corruption counts and real
  ``market_data_circuit_breaker_tripped_reconnecting``/``_reconnected``
  pairs (Directive 2's AEGIS absorb-reject-purge criterion).
- ``--execution-log``: stdout of ``scripts/run_execution_agent.py`` --
  real ``chaos_event_loop_lag_bottleneck`` warnings (Directive 1) and,
  if ``--enable-memory-monitor`` was passed to that launcher, real
  ``chaos_memory_leak_suspected`` warnings (Directive 4, in-process
  reading).
- ``--gate-report``: the ``.jsonl`` file ``scripts/chaos_injector.py``
  wrote -- every real ``[CHAOS_GATE_REJECT]``/``[CHAOS_GATE_PASS]``
  verdict ``Fleet_Command_Arbiter`` actually produced under load
  (Directive 3).
- ``--memory-report``: the ``.json`` file ``scripts/chaos_memory_watch.py``
  wrote -- the external, cross-process RSS drift verdict (Directive 4,
  authoritative reading).

Usage::

    python scripts/chaos_shadow_report.py \\
        --market-data-log logs/chaos_market_data.log \\
        --execution-log logs/chaos_execution_agent.log \\
        --gate-report logs/chaos_gate_report.jsonl \\
        --memory-report logs/chaos_memory_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Iterable, Literal

Verdict = Literal["GO", "ATTENTION", "NO_GO"]

_LOOP_LAG_EVENT = "chaos_event_loop_lag_bottleneck"
_MEMORY_LEAK_EVENT = "chaos_memory_leak_suspected"
_CIRCUIT_TRIP_EVENT = "market_data_circuit_breaker_tripped_reconnecting"
_CIRCUIT_RECONNECTED_EVENT = "market_data_circuit_breaker_reconnected"
_INJECTION_EVENTS = ("chaos_injected_nan_price", "chaos_injected_flash_crash", "chaos_injected_malformed_payload")
_UNEXPECTED_FAILURE_LEVELS = ("error", "critical")
# Real, expected rejection paths under this mission's own design -- see
# core/security.py's is_system_safe ("fails closed on any doubt") and
# scripts/chaos_reset_kill_switch.py's own docstring: a fresh Redis with
# no SYSTEM_STATUS key yet blocks every order attempt by design, which is
# a correct safe-default, not a system failure, and every real gate
# rejection this project can produce is deliberately logged at
# warning/critical (see e.g. consensus_order_blocked_by_global_kill_switch)
# -- counted separately below, never folded into "unexpected failures".
_KNOWN_SAFE_REJECTION_EVENTS = ("kill_switch_blocked_call", "consensus_order_blocked_by_global_kill_switch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--market-data-log", type=Path, default=None)
    parser.add_argument("--execution-log", type=Path, default=None)
    parser.add_argument("--gate-report", type=Path, default=None)
    parser.add_argument("--memory-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to also write this report as JSON.")
    return parser.parse_args()


def _iter_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError:
                continue


def _analyze_market_data_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    injection_counts: Counter[str] = Counter()
    circuit_trips = 0
    circuit_reconnects = 0
    unexpected_failures: list[tuple[int, str]] = []
    for line_number, record in enumerate(_iter_json_lines(path), start=1):
        event = record.get("event", "")
        if event in _INJECTION_EVENTS:
            injection_counts[event] += 1
        elif event == _CIRCUIT_TRIP_EVENT:
            circuit_trips += 1
        elif event == _CIRCUIT_RECONNECTED_EVENT:
            circuit_reconnects += 1
        elif record.get("level") in _UNEXPECTED_FAILURE_LEVELS and event not in (_CIRCUIT_TRIP_EVENT,):
            unexpected_failures.append((line_number, event))
    return {
        "available": True,
        "injections": dict(injection_counts),
        "aegis_circuit_trips": circuit_trips,
        "aegis_circuit_reconnects": circuit_reconnects,
        "aegis_absorbed_without_crash": circuit_trips == circuit_reconnects and circuit_trips > 0,
        "unexpected_failures": unexpected_failures[:10],
        "unexpected_failure_count": len(unexpected_failures),
    }


def _analyze_execution_log(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    loop_lag_events: list[float] = []
    memory_leak_warnings = 0
    unexpected_failures: list[tuple[int, str]] = []
    safe_rejection_count = 0
    for line_number, record in enumerate(_iter_json_lines(path), start=1):
        event = record.get("event", "")
        if event == _LOOP_LAG_EVENT:
            loop_lag_events.append(record.get("lag_ms", 0.0))
        elif event == _MEMORY_LEAK_EVENT:
            memory_leak_warnings += 1
        elif event in _KNOWN_SAFE_REJECTION_EVENTS:
            safe_rejection_count += 1
        elif record.get("level") in _UNEXPECTED_FAILURE_LEVELS:
            unexpected_failures.append((line_number, event))
    return {
        "available": True,
        "loop_lag_bottleneck_count": len(loop_lag_events),
        "worst_loop_lag_ms": max(loop_lag_events) if loop_lag_events else None,
        "memory_leak_warning_count": memory_leak_warnings,
        "safe_rejection_count": safe_rejection_count,
        "unexpected_failures": unexpected_failures[:10],
        "unexpected_failure_count": len(unexpected_failures),
    }


def _analyze_gate_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    rejects = 0
    passes = 0
    for record in _iter_json_lines(path):
        if record.get("tag") == "[CHAOS_GATE_REJECT]":
            rejects += 1
        elif record.get("tag") == "[CHAOS_GATE_PASS]":
            passes += 1
    total = rejects + passes
    return {
        "available": True,
        "gate_rejects": rejects,
        "gate_passes": passes,
        "reject_pct": round(rejects / total * 100.0, 1) if total else None,
    }


def _analyze_memory_report(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"available": False}
    return {"available": True, **data}


def _decide_verdict(market_data: dict, execution: dict, gates: dict, memory: dict) -> tuple[Verdict, str]:
    reasons: list[str] = []

    if market_data.get("available") and market_data["unexpected_failure_count"] > 0:
        return "NO_GO", f"{market_data['unexpected_failure_count']} unexpected failure(s) in the chaos market-data log -- investigate before any GO."
    if execution.get("available") and execution["unexpected_failure_count"] > 0:
        return "NO_GO", f"{execution['unexpected_failure_count']} unexpected failure(s) in the ExecutionAgent log -- investigate before any GO."
    if memory.get("available") and memory.get("verdict") == "NO_GO":
        return "NO_GO", f"Memory watch: {memory.get('verdict_reason')}"

    if market_data.get("available") and market_data.get("aegis_circuit_trips", 0) > 0 and not market_data.get("aegis_absorbed_without_crash"):
        return "NO_GO", "AEGIS circuit breaker tripped but never confirmed reconnected -- the process may have died mid-recovery."

    if execution.get("available") and execution.get("loop_lag_bottleneck_count", 0) > 0:
        reasons.append(f"{execution['loop_lag_bottleneck_count']} event-loop lag bottleneck(s) observed (worst {execution.get('worst_loop_lag_ms')}ms) -- real, but not fatal on its own.")
    if memory.get("available") and memory.get("verdict") == "ATTENTION":
        reasons.append(f"Memory watch: {memory.get('verdict_reason')}")
    if not any((market_data.get("available"), execution.get("available"), gates.get("available"), memory.get("available"))):
        return "ATTENTION", "No component report was found -- nothing to verify yet."

    if reasons:
        return "ATTENTION", " | ".join(reasons)

    parts = []
    if market_data.get("available"):
        parts.append(f"AEGIS absorbed {market_data.get('aegis_circuit_trips', 0)} trip(s) and reconnected cleanly every time")
    if gates.get("available"):
        parts.append(f"Fleet_Command_Arbiter rejected {gates.get('reject_pct')}% of injected signals as expected under chaos")
    return "GO", "; ".join(parts) if parts else "Every available component reported clean."


def main() -> int:
    args = _parse_args()
    market_data = _analyze_market_data_log(args.market_data_log)
    execution = _analyze_execution_log(args.execution_log)
    gates = _analyze_gate_report(args.gate_report)
    memory = _analyze_memory_report(args.memory_report)
    verdict, reason = _decide_verdict(market_data, execution, gates, memory)

    report = {
        "market_data": market_data, "execution": execution, "gates": gates, "memory": memory,
        "verdict": verdict, "verdict_reason": reason,
    }

    print("=" * 70)
    print("CHAOS SHADOW RUN -- CONSOLIDATED REPORT")
    print("=" * 70)
    print(f"[Directive 2 -- AEGIS/Binance chaos]      {market_data}")
    print(f"[Directive 1 -- ZMQ/event-loop saturation] {execution}")
    print(f"[Directive 3 -- Fleet_Command_Arbiter]     {gates}")
    print(f"[Directive 4 -- Memory drift]               {memory}")
    print("-" * 70)
    print(f"VERDICT: {verdict}")
    print(reason)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {"GO": 0, "ATTENTION": 1, "NO_GO": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())

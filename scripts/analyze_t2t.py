"""Percentiles of a ``scripts/latency_profile.py`` run, in plain Python (no numpy, no pandas).

Reads ``t2t_benchmark.csv`` (and ``t2t_benchmark.meta.json`` next to it, if present) and prints Markdown:
the tick-to-decision latency over every recorded tick, the tick-to-order latency over the entry signals that
went through the Risk Gates, a per-stage breakdown, and the run metadata (memory samples, GC, timer cost).

Percentile method: nearest rank. With ``n`` sorted values, the p-th percentile is the value at 1-based rank
``ceil(p / 100 * n)``. No interpolation, so every printed figure is a latency that was actually observed.
A high percentile of a small sample is the maximum of very few observations: each row states ``n`` and
flags a p99.9 that rests on fewer than 10 samples.

Usage::

    python scripts/analyze_t2t.py t2t_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from fractions import Fraction
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

PERCENTILES: Final[tuple[float, ...]] = (50.0, 90.0, 99.0, 99.9)
_MIN_TAIL_SAMPLES: Final[int] = 10
_ENTRY_SIGNALS: Final[frozenset[str]] = frozenset({"1", "2"})
_SIGNAL_NAMES: Final[dict[str, str]] = {"0": "none", "1": "BUY", "2": "SELL", "3": "CLOSE"}


def _rank(n: int, percentile: float) -> int:
    """1-based nearest rank, in exact arithmetic: ``99.9 / 100 * 1000`` is 999.0000000000001 in floats, which
    would round the rank up to 1000 instead of 999."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    return max(1, math.ceil(Fraction(str(percentile)) * n / 100))


def nearest_rank(sorted_values: Sequence[int], percentile: float) -> int:
    """Nearest-rank percentile of an already sorted, non-empty sequence."""
    if not sorted_values:
        raise ValueError("no values")
    return sorted_values[_rank(len(sorted_values), percentile) - 1]


def samples_above(n: int, percentile: float) -> int:
    """How many observations sit at or above the nearest-rank ``percentile`` of ``n`` samples."""
    return n - _rank(n, percentile) + 1


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _us(value_ns: int) -> str:
    return f"{value_ns / 1000.0:,.1f}"


def _ms(value_ns: int) -> str:
    return f"{value_ns / 1_000_000.0:,.3f}"


def percentile_table(title: str, values: list[int], *, unit: str = "us") -> list[str]:
    fmt = _ms if unit == "ms" else _us
    lines = [f"**{title}** (n = {len(values):,})", ""]
    if not values:
        return [*lines, "_no samples_", ""]
    ordered = sorted(values)
    lines += [f"| Percentile | Latency ({unit}) | Note |", "|---|---:|---|"]
    for percentile in PERCENTILES:
        note = ""
        if percentile == 99.9 and samples_above(len(ordered), percentile) < _MIN_TAIL_SAMPLES:
            note = f"top {samples_above(len(ordered), percentile)} sample(s) only, not a stable estimate"
        label = f"{percentile:g}th"
        lines.append(f"| {label} | {fmt(nearest_rank(ordered, percentile))} | {note} |")
    lines.append(f"| max | {fmt(ordered[-1])} | |")
    lines.append("")
    return lines


def stage_table(rows: list[dict[str, str]]) -> list[str]:
    stages = (("decode", "decode_ns", rows), ("risk pre-check", "risk_check_ns", rows), ("alpha z-score", "alpha_ns", rows),
              ("Risk Gates + is_system_safe (entries)", "gates_ns", [r for r in rows if r["signal"] in _ENTRY_SIGNALS]))
    lines = ["**Per stage**", "", "| Stage | n | p50 (us) | p99 (us) | max (us) |", "|---|---:|---:|---:|---:|"]
    for name, column, subset in stages:
        values = sorted(int(row[column]) for row in subset)
        if values:
            lines.append(f"| {name} | {len(values):,} | {_us(nearest_rank(values, 50))} | {_us(nearest_rank(values, 99))} | {_us(values[-1])} |")
    lines.append("")
    return lines


def render(rows: list[dict[str, str]], meta: dict[str, Any] | None) -> str:
    entries = [row for row in rows if row["signal"] in _ENTRY_SIGNALS]
    signal_counts: dict[str, int] = {}
    for row in rows:
        name = _SIGNAL_NAMES.get(row["signal"], row["signal"])
        signal_counts[name] = signal_counts.get(name, 0) + 1
    gate_counts: dict[str, int] = {}
    for row in entries:
        gate_counts[row["gate_reached"]] = gate_counts.get(row["gate_reached"], 0) + 1

    risk_events = sum(1 for row in rows if row["risk_event"] == "1")
    lines = [f"Recorded ticks: {len(rows):,}. Signals: " + ", ".join(f"{name} {count:,}" for name, count in sorted(signal_counts.items())) + ".",
             "Highest gate passed by entry signals (0 = refused before the gates, Kill Switch latched): "
             + (", ".join(f"gate {gate}: {count:,}" for gate, count in sorted(gate_counts.items())) or "none") + ".",
             f"Ticks on which the risk pre-check raised a Kill Switch event: {risk_events:,}.", ""]
    lines += percentile_table("Queue delay before T1, every tick (publisher timestamp to T1)", [int(row["ingest_lag_ns"]) for row in rows], unit="ms")
    lines += percentile_table("Tick-to-decision, every tick (T1 to the end of the alpha step)", [int(row["tick_to_decision_ns"]) for row in rows])
    lines += percentile_table("Tick-to-order, entry signals through the Risk Gates (T1 to T2)", [int(row["tick_to_order_ns"]) for row in entries])
    lines += stage_table(rows)
    if meta:
        samples = meta.get("rss_mb_samples", [])
        lines += ["**Run**", "",
                  f"- Python {meta.get('python')} ({meta.get('implementation')}), {meta.get('platform')}, pyzmq {meta.get('pyzmq')}.",
                  f"- Received {meta.get('received', 0):,} ticks, warm-up {meta.get('warmup_ticks', 0):,} not recorded, recorded {meta.get('recorded', 0):,}, "
                  f"malformed {meta.get('malformed', 0):,}, dropped (buffer full) {meta.get('dropped_buffer_full', 0):,}. Stopped: {meta.get('stopped_because')}.",
                  f"- One `perf_counter_ns()` call costs about {meta.get('perf_counter_call_overhead_ns')} ns here; each latency contains 2 to 5 of them.",
                  f"- GC collections during the run, generations 0/1/2: {meta.get('gc_collections_during_run')}.",
                  "- Resident memory (MB) by ticks received: " + ", ".join(f"{s['received']:,}: {s['rss_mb']}" for s in samples) + "."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="t2t_benchmark.csv written by scripts/latency_profile.py")
    args = parser.parse_args()
    if not args.csv_path.exists():
        print(f"file not found: {args.csv_path}")
        return 2
    rows = load_rows(args.csv_path)
    if not rows:
        print(f"{args.csv_path} has no rows")
        return 1
    meta_path = args.csv_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    print(render(rows, meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())

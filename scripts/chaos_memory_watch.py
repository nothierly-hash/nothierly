"""External RSS memory-drift watcher across every Chaos Shadow Run process (mission Directive 4).

Complements :mod:`core.chaos.memory_profiler` (in-process, per-agent,
tracemalloc+RSS): that module can only see the one process it is
imported into, and tracemalloc specifically only sees Python-allocated
memory, missing native-extension growth (``ccxt.pro``'s aiohttp/websocket
buffers, ``pyzmq``, ``pydantic-core``) that a real, unattended 24h run
needs covered too. This script instead watches real OS-level RSS for an
explicit list of PIDs from *outside* those processes, on a fixed sample
interval, for a fixed total duration, and renders the same
GO/ATTENTION/NO-GO verdict style as :mod:`scripts.analyze_shadow_run` --
this project's own established, honest pattern (see that script's own
module docstring on why it uses the same shape rather than a fresh
ad-hoc format).

Usage::

    python scripts/chaos_memory_watch.py --pid 12345 --pid 12346 --pid 12347 \\
        --duration-hours 24 --drift-threshold-pct 5 --out logs/chaos_memory_report.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

Verdict = Literal["GO", "ATTENTION", "NO_GO"]


@dataclass
class _ProcessSample:
    pid: int
    label: str
    baseline_rss_bytes: int | None = None
    samples: list[tuple[float, int]] = field(default_factory=list)  # (unix_ts, rss_bytes)
    alive: bool = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pid", type=int, action="append", default=[], dest="pids", help="A PID to watch. Repeatable.")
    parser.add_argument("--label", action="append", default=[], dest="labels", help="Optional label for the matching --pid (same order). Defaults to the PID itself.")
    parser.add_argument("--duration-hours", type=float, default=24.0, help="Total watch duration in hours (default: 24.0, the mission brief's own target).")
    parser.add_argument("--sample-interval-seconds", type=float, default=300.0, help="Seconds between RSS samples (default: 300 = 5 minutes).")
    parser.add_argument("--drift-threshold-pct", type=float, default=5.0, help="Absolute RSS drift from baseline, in %%, that fails the run (default: 5.0, the mission brief's own default).")
    parser.add_argument("--out", default="logs/chaos_memory_report.json", help="Where to write the final JSON report.")
    return parser.parse_args()


def _rss_bytes(pid: int) -> int | None:
    if psutil is None:
        return None
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except psutil.Error:
        return None


def main() -> int:
    args = _parse_args()
    if psutil is None:
        print("psutil is required for this script (already a real dependency of core.watchdog elsewhere in this project).")
        return 2
    if not args.pids:
        print("At least one --pid is required.")
        return 2

    labels = args.labels + [str(pid) for pid in args.pids[len(args.labels):]]
    processes = [_ProcessSample(pid=pid, label=label) for pid, label in zip(args.pids, labels)]

    deadline = time.monotonic() + args.duration_hours * 3600.0
    start_wall = datetime.now(timezone.utc)
    print(f"[chaos_memory_watch] Watching {len(processes)} process(es) for {args.duration_hours:.2f}h, "
          f"sampling every {args.sample_interval_seconds:.0f}s. Ctrl+C to stop early and still get a report.")

    try:
        while time.monotonic() < deadline and any(p.alive for p in processes):
            now = time.time()
            for proc in processes:
                if not proc.alive:
                    continue
                rss = _rss_bytes(proc.pid)
                if rss is None:
                    proc.alive = False
                    print(f"[chaos_memory_watch] PID {proc.pid} ({proc.label}) no longer reachable -- stopped tracking it.")
                    continue
                if proc.baseline_rss_bytes is None:
                    proc.baseline_rss_bytes = rss
                proc.samples.append((now, rss))
            # min(..., remaining-until-deadline), not a flat sleep: sleeping the full
            # interval regardless of the deadline means a short validation run (e.g. 90s
            # duration, the default 300s interval) oversleeps its own watch window --
            # found via a real Chaos Shadow Run (2026-09-18), where the orchestrator's
            # own SIGTERM (a hard, non-graceful `TerminateProcess` on Windows, which never
            # runs this function's own post-loop report-writing code) killed this process
            # while still asleep, silently losing Directive 4 (memory drift) entirely
            # instead of reporting on the one baseline sample actually collected.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.sample_interval_seconds, remaining))
    except KeyboardInterrupt:
        print("\n[chaos_memory_watch] Stopped early by user -- reporting on samples collected so far.")

    report = _build_report(processes, start_wall=start_wall, duration_hours_target=args.duration_hours, drift_threshold_pct=args.drift_threshold_pct)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_report(report)
    return {"GO": 0, "ATTENTION": 1, "NO_GO": 2}[report["verdict"]]


def _build_report(
    processes: list[_ProcessSample], *, start_wall: datetime, duration_hours_target: float, drift_threshold_pct: float,
) -> dict:
    per_process = []
    worst_drift_pct: float | None = None
    any_crashed = False

    for proc in processes:
        if not proc.alive:
            any_crashed = True
        if len(proc.samples) < 2 or proc.baseline_rss_bytes is None:
            per_process.append({
                "pid": proc.pid, "label": proc.label, "alive_for_report": proc.alive,
                "samples_collected": len(proc.samples), "drift_pct": None,
            })
            continue
        final_rss = proc.samples[-1][1]
        drift_pct = (final_rss - proc.baseline_rss_bytes) / proc.baseline_rss_bytes * 100.0
        if worst_drift_pct is None or abs(drift_pct) > abs(worst_drift_pct):
            worst_drift_pct = drift_pct
        per_process.append({
            "pid": proc.pid, "label": proc.label, "alive_for_report": proc.alive,
            "samples_collected": len(proc.samples),
            "baseline_rss_bytes": proc.baseline_rss_bytes, "final_rss_bytes": final_rss,
            "drift_pct": round(drift_pct, 3),
        })

    observed_hours = 0.0
    for proc in processes:
        if len(proc.samples) >= 2:
            observed_hours = max(observed_hours, (proc.samples[-1][0] - proc.samples[0][0]) / 3600.0)

    verdict: Verdict
    if any_crashed:
        verdict = "NO_GO"
        reason = "At least one watched process became unreachable (crashed or exited) before the watch window ended."
    elif observed_hours < duration_hours_target * 0.95:
        verdict = "ATTENTION"
        reason = f"Only {observed_hours:.2f}h observed vs a {duration_hours_target:.2f}h target -- re-run for the full window before concluding."
    elif worst_drift_pct is not None and abs(worst_drift_pct) > drift_threshold_pct:
        verdict = "NO_GO"
        reason = f"Worst observed RSS drift {worst_drift_pct:.2f}%% exceeds the {drift_threshold_pct:.1f}%% threshold -- suspected memory leak."
    else:
        verdict = "GO"
        reason = f"Every watched process stayed within {drift_threshold_pct:.1f}%% RSS drift over {observed_hours:.2f}h, with no crashes."

    return {
        "start_utc": start_wall.isoformat(),
        "duration_hours_target": duration_hours_target,
        "duration_hours_observed": round(observed_hours, 3),
        "drift_threshold_pct": drift_threshold_pct,
        "worst_drift_pct": round(worst_drift_pct, 3) if worst_drift_pct is not None else None,
        "processes": per_process,
        "verdict": verdict,
        "verdict_reason": reason,
    }


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("CHAOS SHADOW RUN -- MEMORY DRIFT REPORT (Directive 4)")
    print("=" * 60)
    print(f"Observed: {report['duration_hours_observed']:.2f}h / target {report['duration_hours_target']:.2f}h")
    for proc in report["processes"]:
        drift = proc.get("drift_pct")
        drift_str = f"{drift:+.2f}%" if drift is not None else "n/a (too few samples)"
        print(f"  PID {proc['pid']} ({proc['label']}): {proc['samples_collected']} samples, drift {drift_str}, "
              f"{'alive' if proc['alive_for_report'] else 'LOST DURING RUN'}")
    print("-" * 60)
    label = {"GO": "GO", "ATTENTION": "ATTENTION (JAUNE)", "NO_GO": "NO-GO (ROUGE)"}[report["verdict"]]
    print(f"VERDICT: {label}")
    print(report["verdict_reason"])


if __name__ == "__main__":
    raise SystemExit(main())

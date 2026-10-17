"""Orchestrator for the full "Chaos Shadow Run" (mission: ZMQ saturation + Binance fault injection + Fleet_Command_Arbiter stress + memory-drift watch).

Wires up, as real OS subprocesses (each already a real, independent
launcher elsewhere in this repo -- this script adds no new trading
logic of its own, only sequencing + lifecycle):

1. ``scripts/run_fake_redis.py`` -- throwaway local Redis, matching
   ``.env``'s ``REDIS_PORT`` (use ``--use-real-redis`` to skip this and
   point at a real Redis instead, for an actual unattended overnight
   run against durable state).
2. ``scripts/run_zmq_broker.py`` -- required for the chaos injector's
   published signals, and any agent's own publishes, to actually reach
   anything (see that script's own docstring).
3. ``scripts/run_market_data_agent_chaos.py`` -- the real Binance feed,
   fault-injected (Directive 2).
4. ``scripts/run_execution_agent.py`` -- ``Fleet_Command_Arbiter``
   (Directive 1's target for lag, Directive 3's target for gate/margin
   stress), loop-lag and memory monitors enabled.
5. ``scripts/run_alpha_engine.py`` -- shares the same chaos-fed Redis
   market metrics, so its own real AEGIS staleness breaker gets
   exercised too (see ``core/chaos/binance_fault_injector.py``'s module
   docstring for why corrupted ticks surface as *staleness* downstream,
   not corrupted values -- ``MarketDataAgent`` itself absorbs/rejects
   the corruption before ever writing to Redis). Skippable with
   ``--skip-alpha-engine``.
6. ``scripts/chaos_injector.py`` -- the signal flood + gate-telemetry
   observer (Directives 1 and 3).
7. ``scripts/chaos_memory_watch.py`` -- external RSS drift watch across
   every PID above, for the full run (Directive 4).

At the end (whether the run completes its full duration or is stopped
early with Ctrl+C), runs ``scripts/chaos_shadow_report.py`` and prints
the consolidated verdict.

**Honest scope note**: this launches everything needed for a real,
unattended run of any length, including the mission's own literal 24h --
but this script itself does not run for 24h "for free"; whoever starts
it (a person, or a scheduled task) needs to leave this process running,
same as any other long-lived job. A short ``--duration-hours`` value is
how this exact tooling gets validated in minutes before trusting it for
a real overnight run.

Usage::

    # Real 24h run:
    python scripts/run_chaos_shadow_run.py --duration-hours 24 --rate 75 \\
        --max-portfolio-exposure-pct 0.001

    # Quick validation burst:
    python scripts/run_chaos_shadow_run.py --duration-hours 0.05 --rate 75 \\
        --max-portfolio-exposure-pct 0.001
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--instrument", default="BTC/USDT")
    parser.add_argument("--rate", type=float, default=75.0, help="Chaos injector signals/sec (mission range 50-100).")
    parser.add_argument("--consensus-fraction", type=float, default=0.3)
    parser.add_argument("--max-portfolio-exposure-pct", type=float, default=0.001, help="Deliberately low Gate-5 ceiling -- the fat-finger-equivalent stress lever (default: 0.001).")
    parser.add_argument("--nan-probability", type=float, default=0.10)
    parser.add_argument("--flash-crash-probability", type=float, default=0.05)
    parser.add_argument("--flash-crash-fraction", type=float, default=0.30)
    parser.add_argument("--malformed-probability", type=float, default=0.10)
    parser.add_argument("--loop-lag-threshold-ms", type=float, default=25.0)
    parser.add_argument("--memory-drift-threshold-pct", type=float, default=5.0)
    parser.add_argument("--memory-sample-interval-seconds", type=float, default=300.0)
    parser.add_argument("--redis-port", type=int, default=16399)
    parser.add_argument("--use-real-redis", action="store_true", help="Skip step 1 (fake redis) and use whatever REDIS_HOST/REDIS_PORT is already configured.")
    parser.add_argument(
        "--reset-kill-switch", action="store_true",
        help=(
            "Explicitly run scripts/chaos_reset_kill_switch.py once, right after Redis is up. "
            "Without this, a fresh Redis (fake or real) has no SYSTEM_STATUS key yet, and "
            "core.security.is_system_safe's own documented 'fail closed on any doubt' default "
            "blocks every real order attempt upstream of Gates 4-7 -- a real, correct, but "
            "different rejection path than Directive 3's margin/exposure stress. This flag is "
            "the one place this orchestrator makes that normally-human-only decision, and only "
            "because you passed it explicitly -- see that script's own docstring."
        ),
    )
    parser.add_argument("--skip-alpha-engine", action="store_true")
    parser.add_argument("--log-dir", default=None, help="Defaults to logs/chaos_shadow_run_<timestamp>/.")
    return parser.parse_args()


def _spawn(name: str, args: list[str], *, log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [_PYTHON, *args], cwd=str(_PROJECT_ROOT), stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[orchestrator] started {name} (pid={proc.pid}) -> {log_path}")
    return proc


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else _PROJECT_ROOT / "logs" / f"chaos_shadow_run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen] = {}
    log_files: dict[str, Path] = {}

    # **Real isolation bug, found live (2026-09-23) via an actual chaos burst**: every child below
    # used to inherit this process's own environment unchanged, so --redis-port only ever controlled
    # what port the orchestrator's OWN spawned fake_redis binds to -- none of the real agents
    # (execution_agent, alpha_engine, market_data_chaos) or chaos_reset_kill_switch.py were ever told
    # to CONNECT to it. Every one of them fell back to core.config.get_settings()'s own default (this
    # project's real, shared, standing Redis on .env's REDIS_PORT) -- confirmed live: a "quick
    # validation burst" run this way reset SYSTEM_STATUS and threw 3 minutes of fault injection at
    # that same shared instance instead of a throwaway one. child_env is now the one, real source of
    # truth every spawned child (and the one-off chaos_reset_kill_switch.py subprocess.run below) is
    # given -- REDIS_HOST/REDIS_PORT overridden to the isolated fake_redis this script itself started,
    # unless --use-real-redis was passed, in which case child_env is left exactly as documented
    # ("whatever REDIS_HOST/REDIS_PORT is already configured").
    child_env = os.environ.copy()
    if not args.use_real_redis:
        child_env["REDIS_HOST"] = "127.0.0.1"
        child_env["REDIS_PORT"] = str(args.redis_port)

    def spawn(name: str, script: str, script_args: list[str]) -> None:
        log_path = log_dir / f"{name}.log"
        procs[name] = _spawn(name, [str(_PROJECT_ROOT / "scripts" / script), *script_args], log_path=log_path, env=child_env)
        log_files[name] = log_path

    try:
        if not args.use_real_redis:
            spawn("fake_redis", "run_fake_redis.py", ["--port", str(args.redis_port)])
            time.sleep(2)

        if args.reset_kill_switch:
            print(f"[orchestrator] --reset-kill-switch passed: running scripts/chaos_reset_kill_switch.py once, explicitly, "
                  f"against {child_env.get('REDIS_HOST', '<.env default>')}:{child_env.get('REDIS_PORT', '<.env default>')}.")
            subprocess.run(
                [_PYTHON, str(_PROJECT_ROOT / "scripts" / "chaos_reset_kill_switch.py")],
                cwd=str(_PROJECT_ROOT), check=True, env=child_env,
            )
        else:
            print("[orchestrator] NOTE: SYSTEM_STATUS is not being reset (pass --reset-kill-switch to do so explicitly).")
            print("[orchestrator]       Every real order attempt will be blocked upstream by the global Kill Switch's own")
            print("[orchestrator]       documented fail-closed default -- real, correct, but it means Gates 4-7 (Margin/")
            print("[orchestrator]       RiskManager/PortfolioRiskParityManager/StochasticCore) will not be exercised this run.")

        spawn("zmq_broker", "run_zmq_broker.py", [])
        time.sleep(2)

        spawn(
            "market_data_chaos", "run_market_data_agent_chaos.py",
            [
                "--symbol", args.instrument, "--nan-probability", str(args.nan_probability),
                "--flash-crash-probability", str(args.flash_crash_probability),
                "--flash-crash-fraction", str(args.flash_crash_fraction),
                "--malformed-probability", str(args.malformed_probability),
            ],
        )
        time.sleep(3)

        spawn(
            "execution_agent", "run_execution_agent.py",
            [
                "--instruments", args.instrument,
                "--max-portfolio-exposure-pct", str(args.max_portfolio_exposure_pct),
                "--enable-loop-lag-monitor", "--loop-lag-threshold-ms", str(args.loop_lag_threshold_ms),
                "--enable-memory-monitor", "--memory-drift-threshold-pct", str(args.memory_drift_threshold_pct),
                "--memory-sample-interval-seconds", str(args.memory_sample_interval_seconds),
            ],
        )
        time.sleep(2)

        if not args.skip_alpha_engine:
            spawn("alpha_engine", "run_alpha_engine.py", ["--symbol", args.instrument])
            time.sleep(2)

        duration_seconds = args.duration_hours * 3600.0
        gate_report_path = log_dir / "chaos_gate_report.jsonl"
        spawn(
            "chaos_injector", "chaos_injector.py",
            [
                "--instrument", args.instrument, "--rate", str(args.rate),
                "--consensus-fraction", str(args.consensus_fraction),
                "--duration-seconds", str(duration_seconds),
                "--loop-lag-threshold-ms", str(args.loop_lag_threshold_ms),
                "--report-path", str(gate_report_path),
            ],
        )

        memory_report_path = log_dir / "chaos_memory_report.json"
        memory_pid_args: list[str] = []
        for name in ("market_data_chaos", "execution_agent") + (() if args.skip_alpha_engine else ("alpha_engine",)):
            memory_pid_args += ["--pid", str(procs[name].pid), "--label", name]
        spawn(
            "memory_watch", "chaos_memory_watch.py",
            [
                *memory_pid_args, "--duration-hours", str(args.duration_hours),
                "--sample-interval-seconds", str(args.memory_sample_interval_seconds),
                "--drift-threshold-pct", str(args.memory_drift_threshold_pct),
                "--out", str(memory_report_path),
            ],
        )

        print(f"[orchestrator] Chaos Shadow Run in progress for {args.duration_hours:.3f}h "
              f"({duration_seconds:.0f}s). Logs: {log_dir}")
        print("[orchestrator] Ctrl+C to stop early -- the report still runs on whatever was collected.")

        deadline = time.monotonic() + duration_seconds + 15.0
        while time.monotonic() < deadline:
            if procs["chaos_injector"].poll() is not None:
                break
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\n[orchestrator] Stopping early on Ctrl+C.")
    finally:
        for name, proc in procs.items():
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 10.0
        for name, proc in procs.items():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[orchestrator] Every subprocess stopped.")

    report_out = log_dir / "chaos_shadow_report.json"
    report_args = [
        str(_PROJECT_ROOT / "scripts" / "chaos_shadow_report.py"),
        "--out", str(report_out),
    ]
    if "market_data_chaos" in log_files:
        report_args += ["--market-data-log", str(log_files["market_data_chaos"])]
    if "execution_agent" in log_files:
        report_args += ["--execution-log", str(log_files["execution_agent"])]
    report_args += ["--gate-report", str(log_dir / "chaos_gate_report.jsonl")]
    memory_report = log_dir / "chaos_memory_report.json"
    if memory_report.exists():
        report_args += ["--memory-report", str(memory_report)]

    result = subprocess.run([_PYTHON, *report_args], cwd=str(_PROJECT_ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

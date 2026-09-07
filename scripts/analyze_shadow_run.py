"""Post-Shadow-Run diagnostic for AlphaMarketMakerAgent's real logs -- GO/NO-GO for real capital.

**Corrections to the brief, verified against the real code first** (this
project's established discipline):

1. **Log format is real structured JSON, not free text with ``[TAG]``
   markers.** ``core/logger.py``'s ``configure_logging`` renders every log
   call through ``structlog.processors.JSONRenderer()`` by default -- one
   JSON object per line, with real field names (``event``, ``level``,
   ``timestamp``, ``service``), not bracketed strings like
   ``"[STALE DATA]"`` or ``"[SHADOW TRADE]"``. This script parses each
   line as JSON and matches on ``event``/``level`` -- the exact real event
   names :class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`
   emits (grepped from the source below, not guessed), not regex
   substrings that would silently match nothing against this codebase's
   real output.

2. **No PnL to extract from a shadow run -- by design, not by omission.**
   ``AlphaMarketMakerAgent._run_simulated_cycle`` (what actually runs
   under ``AgentControlState.SHADOW``/``MOCK_MODE``) deliberately never
   computes or logs a PnL number -- see that method's own docstring:
   "never a fabricated number displayed as if it were real" (this
   project's established discipline, see ``src/dashboard/ems_pages/
   agent_fleet.py``'s module docstring for the same rule applied
   elsewhere). A PnL-extraction regex against a real shadow run's logs
   would always find zero matches -- not evidence of a quiet, profitable
   night, just evidence the metric was never populated.

3. **The load-bearing correction: a SHADOW run does not exercise three of
   the four hardening guardrails at all.** ``_run_cycle`` branches into
   ``_run_simulated_cycle`` and returns the instant ``simulate`` is
   ``True``, *before* ever reaching ``_place_resting_order``/
   ``_hedge_excess_inventory`` -- the only two call sites that invoke
   ``_check_fat_finger``, ``_check_order_rate_limit``, or the daily-loss
   check in ``_record_fill``. A clean 24h shadow run proves real things
   (process stability, Redis/Telegram connectivity, staleness detection,
   the Master Toggle responding) -- it does **not** prove the fat-finger
   cap, the order-rate breaker, or the daily-loss halt work, because none
   of that code path runs under SHADOW. Those three are validated by
   ``tests/test_alpha_engine.py``'s real unit tests instead (which call
   them directly). This script reports both halves honestly rather than
   implying one covers the other.

4. **No SOR/leg-risk check** -- this agent never touches
   :mod:`~src.execution.smart_order_router` at all; "Rollback"/leg-risk
   are a different component's real concepts, not this one's.

**Reusable, not just a CLI script**: :func:`compute_diagnostics` is the
real, pure analysis core (no I/O, no printing) -- both this file's own
``main()`` and :mod:`~src.dashboard.ems_pages.shadow_run_analytics`'s
dashboard page call it, so the dashboard's "GO/NO-GO" reading is computed
by the exact same logic as this script's, never a second,
independently-drifting reimplementation.

Usage::

    # Overnight, in its own terminal (stdout only -- no file handler is configured by
    # scripts/run_alpha_engine.py, matching every other launcher in this project; Docker's own log
    # driver captures stdout in a real deployment, a plain shell redirect does the same job locally):
    python scripts/run_alpha_engine.py --symbol BTC/USDT > logs/alpha_shadow_run.log 2>&1

    # The next morning:
    python scripts/analyze_shadow_run.py logs/alpha_shadow_run.log

    # Or from the dashboard: src/dashboard/ems_pages/shadow_run_analytics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

# Real, verified-not-assumed bug this line fixes: Windows' default console codepage (cp1252) cannot
# encode the emoji this report prints (confirmed live -- a bare `print()` of "🔬" raises
# UnicodeEncodeError and crashes this script before it ever reaches a verdict). reconfigure() is
# Python 3.7+, available on every version this project's own pyproject.toml already requires.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

# Real event names AlphaMarketMakerAgent (and, for agent_tick_failed, its real BaseAgent parent)
# actually emit -- see src/agents/alpha_engine/agent.py and core/base_agent.py. Grouped by what
# each one means for a go/no-go read, not by a guessed keyword.

# An exception genuinely escaped somewhere real work happens -- always worth a NO-GO look.
_UNEXPECTED_FAILURE_EVENTS: dict[str, str] = {
    "agent_tick_failed": "Exception non gérée pendant un cycle (BaseAgent.run's own catch-all)",
    "alpha_engine_quote_failed": "Échec de placement d'une cotation",
    "alpha_engine_hedge_order_failed": "Échec de l'ordre de couverture",
    "alpha_engine_emergency_liquidate_order_failed": "Échec de l'ordre Emergency Liquidate",
    "alpha_engine_market_metrics_unparseable": "Donnée de marché reçue mais illisible (bug de schéma probable)",
}

# The 4 hardening guardrails tripping -- see module docstring point 3 on why SHADOW mode should
# never actually produce these (structurally unreachable), so seeing one here is itself a real
# anomaly worth investigating, not proof the guardrail "worked as intended" during this run.
_GUARDRAIL_TRIP_EVENTS: dict[str, str] = {
    "alpha_engine_fat_finger_blocked": "Fat-finger : taille d'ordre hors limite",
    "alpha_engine_fat_finger_forced_off": "Fat-finger : moteur forcé OFF",
    "alpha_engine_order_rate_limit_breached": "Emballement de commandes : moteur forcé OFF",
    "alpha_engine_max_daily_loss_hit": "Perte max/jour atteinte : moteur forcé OFF",
    "alpha_engine_hedge_blocked_by_kill_switch_manual_intervention_required": "Hedge bloqué par le Kill Switch global",
    "alpha_engine_emergency_liquidate_blocked_by_kill_switch": "Emergency Liquidate bloqué par le Kill Switch global",
}

_STALE_DATA_EVENT = "alpha_engine_stale_data_detected"
_SIMULATED_QUOTE_EVENT = "alpha_engine_simulated_quote"
_STARTED_EVENT = "alpha_engine_started"
_NO_DAILY_LOSS_LIMIT_EVENT = "alpha_engine_no_daily_loss_limit_configured"

# A gap this many times the expected tick this run's own logs show is treated as "the process
# likely hung or was silently restarted", not just a slow cycle -- see _find_timestamp_gaps.
_GAP_MULTIPLE_OF_TICK_INTERVAL = 10.0
_STALE_DATA_ATTENTION_THRESHOLD = 50  # matches this brief's own real-world-reasonable threshold

Verdict = Literal["GO", "ATTENTION", "NO_GO"]


@dataclass
class ShadowRunDiagnostics:
    """The real, structured result of analyzing one log -- both the CLI and the dashboard page render this, never a second parallel computation."""

    total_json_lines: int
    total_duration_hours: float
    simulated_quote_count: int
    stale_data_count: int
    daily_loss_limit_configured: bool
    mixed_mock_mode_seen: set[bool]
    event_counts: Counter[str]
    unexpected_failures: list[tuple[int, str, str]] = field(default_factory=list)
    guardrail_trips: list[tuple[int, str, str]] = field(default_factory=list)
    unparseable_lines: list[tuple[int, str]] = field(default_factory=list)
    gaps: list[tuple[datetime, float]] = field(default_factory=list)
    verdict: Verdict = "NO_GO"
    verdict_reason: str = ""

    @property
    def exit_code(self) -> int:
        return {"GO": 0, "ATTENTION": 1, "NO_GO": 2}[self.verdict]


def _iter_log_records(lines: Iterable[str]) -> Iterator[tuple[int, dict[str, Any] | None, str]]:
    """Yield ``(line_number, parsed_json_or_None, raw_line)`` for every non-blank line."""
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.rstrip("\n")
        if not stripped.strip():
            continue
        try:
            record: dict[str, Any] | None = json.loads(stripped)
        except json.JSONDecodeError:
            record = None
        yield line_number, record, stripped


def compute_diagnostics(lines: Iterable[str]) -> ShadowRunDiagnostics:
    """The real, pure analysis core -- no I/O, no printing. See module docstring."""
    event_counts: Counter[str] = Counter()
    unexpected_failures: list[tuple[int, str, str]] = []
    guardrail_trips: list[tuple[int, str, str]] = []
    stale_data_count = 0
    simulated_quote_count = 0
    daily_loss_limit_configured = True  # innocent until the startup warning says otherwise
    timestamps: list[datetime] = []
    unparseable_lines: list[tuple[int, str]] = []
    mixed_mock_mode_seen: set[bool] = set()

    for line_number, record, raw_line in _iter_log_records(lines):
        if record is None:
            # A line that isn't valid JSON at all -- most likely a raw Python traceback that
            # bypassed structlog (e.g. a crash before configure_logging ran, or asyncio's own
            # "Task exception was never retrieved" printed straight to stderr). Never silently
            # skipped -- see module docstring's "report both halves honestly" principle.
            if "Traceback" in raw_line or "Exception" in raw_line:
                unparseable_lines.append((line_number, raw_line[:200]))
            continue

        event = record.get("event", "")
        level = record.get("level", "")
        event_counts[event] += 1

        ts_raw = record.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                timestamps.append(datetime.fromisoformat(ts_raw.replace("Z", "+00:00")))
            except ValueError:
                pass

        if event == _STARTED_EVENT:
            if "mock_mode" in record:
                mixed_mock_mode_seen.add(bool(record["mock_mode"]))
        elif event == _NO_DAILY_LOSS_LIMIT_EVENT:
            daily_loss_limit_configured = False
        elif event == _STALE_DATA_EVENT:
            stale_data_count += 1
        elif event == _SIMULATED_QUOTE_EVENT:
            simulated_quote_count += 1
        elif event in _UNEXPECTED_FAILURE_EVENTS:
            unexpected_failures.append((line_number, event, _UNEXPECTED_FAILURE_EVENTS[event]))
        elif event in _GUARDRAIL_TRIP_EVENTS:
            guardrail_trips.append((line_number, event, _GUARDRAIL_TRIP_EVENTS[event]))

        if level in ("error", "critical") and event not in _UNEXPECTED_FAILURE_EVENTS and event not in _GUARDRAIL_TRIP_EVENTS and event != _NO_DAILY_LOSS_LIMIT_EVENT:
            # Anything else logged at error/critical this script doesn't already have a specific
            # bucket for -- caught here so a future new critical/error event in the source doesn't
            # silently fall through unclassified.
            unexpected_failures.append((line_number, event, "Événement critique/erreur non catégorisé"))

    total_duration_hours = (
        (max(timestamps) - min(timestamps)).total_seconds() / 3600.0 if len(timestamps) >= 2 else 0.0
    )
    gaps = _find_timestamp_gaps(timestamps)

    diagnostics = ShadowRunDiagnostics(
        total_json_lines=sum(event_counts.values()),
        total_duration_hours=total_duration_hours,
        simulated_quote_count=simulated_quote_count,
        stale_data_count=stale_data_count,
        daily_loss_limit_configured=daily_loss_limit_configured,
        mixed_mock_mode_seen=mixed_mock_mode_seen,
        event_counts=event_counts,
        unexpected_failures=unexpected_failures,
        guardrail_trips=guardrail_trips,
        unparseable_lines=unparseable_lines,
        gaps=gaps,
    )
    diagnostics.verdict, diagnostics.verdict_reason = _decide_verdict(diagnostics)
    return diagnostics


def _decide_verdict(d: ShadowRunDiagnostics) -> tuple[Verdict, str]:
    if d.unexpected_failures or d.unparseable_lines:
        return "NO_GO", (
            "Des échecs réels et non attendus sont apparus. Investigue chaque ligne listée "
            "avant d'envisager du capital réel."
        )
    if d.guardrail_trips:
        return "NO_GO", (
            "Un guardrail s'est déclenché alors qu'il devrait être structurellement inatteignable "
            "en SHADOW -- soit le moteur n'était pas réellement en SHADOW/MOCK_MODE pendant tout le "
            "run, soit c'est un vrai bug. À investiguer avant tout capital réel."
        )
    if d.total_duration_hours < 20.0:
        return "ATTENTION", (
            f"Seulement {d.total_duration_hours:.1f}h de run réellement observées -- relance un vrai "
            "cycle complet de 24h avant de conclure."
        )
    if d.stale_data_count > _STALE_DATA_ATTENTION_THRESHOLD:
        return "ATTENTION", (
            f"{d.stale_data_count} coupures de flux détectées -- connexion réseau/API pas assez "
            "stable pour du live pendant qu'un flux réel doit rester à jour."
        )
    if d.gaps:
        return "ATTENTION", (
            "Des interruptions du flux de logs sont apparues -- confirme qu'il ne s'agit pas d'un "
            "vrai plantage silencieux avant de continuer."
        )
    return "GO", (
        "Stabilité du processus, détection de staleness, et connectivité Redis/Telegram : propres. "
        "Rappel honnête (voir module docstring point 3) : ce run ne prouve RIEN sur le fat-finger "
        "cap, le coupe-circuit de débit, ou l'arrêt sur perte max/jour -- ces trois-là sont validés "
        "par les tests réels de tests/test_alpha_engine.py, pas par ce run. Considère un passage en "
        "mode sandbox/testnet de l'exchange (si supporté) pour exercer ces chemins de code réels "
        "sans capital réel, avant le premier vrai LIVE."
    )


def _find_timestamp_gaps(timestamps: list[datetime]) -> list[tuple[datetime, float]]:
    """Real gaps between consecutive log timestamps larger than a sane multiple of the expected tick.

    A genuine signal a naive keyword scan would miss entirely: the process silently hanging or
    being killed and restarted leaves a real, measurable hole in the timestamp sequence, whether or
    not any single log line ever says the word "crash".
    """
    if len(timestamps) < 2:
        return []
    ordered = sorted(timestamps)
    intervals = [(ordered[i] - ordered[i - 1]).total_seconds() for i in range(1, len(ordered))]
    median_interval = sorted(intervals)[len(intervals) // 2] if intervals else 1.0
    threshold = max(median_interval * _GAP_MULTIPLE_OF_TICK_INTERVAL, 60.0)
    return [(ordered[i], intervals[i - 1]) for i in range(1, len(ordered)) if intervals[i - 1] > threshold]


def print_report(d: ShadowRunDiagnostics, *, source_label: str) -> None:
    """Render the exact same CLI report this script always has -- now from a ShadowRunDiagnostics instead of computing inline."""
    print(f"🔬 [DIAGNOSTIC] Analyse du run réel : {source_label}\n")
    print("=" * 60)
    print("📊 RAPPORT D'AUDIT -- SHADOW RUN (données réelles, aucune valeur simulée)")
    print("=" * 60)
    print(f"Lignes de log JSON valides analysées : {d.total_json_lines}")
    print(f"Durée réelle observée (1er -> dernier timestamp) : {d.total_duration_hours:.1f}h")
    if len(d.mixed_mock_mode_seen) > 1:
        print("⚠️  Ce fichier contient plusieurs démarrages, certains en MOCK_MODE et d'autres non -- "
              "vérifie qu'il ne mélange pas plusieurs runs différents.")
    print(f"Cotations simulées calculées (mode SHADOW réel) : {d.simulated_quote_count}")
    print("-" * 60)

    print("🛡️  GUARDRAILS -- voir module docstring point 3 : structurellement inatteignables en SHADOW,")
    print("    donc TOUTE occurrence ici est une vraie anomalie à investiguer, pas un test réussi.")
    if d.guardrail_trips:
        for line_number, event, label in d.guardrail_trips:
            print(f"   ⚠️  Ligne {line_number} [{event}] : {label}")
    else:
        print("   Aucun déclenchement -- attendu et correct pour un run en SHADOW.")
    print(f"Coupures de flux de marché détectées (staleness réelle) : {d.stale_data_count}")
    print("-" * 60)

    print(f"❌ ÉCHECS RÉELS NON ATTENDUS : {len(d.unexpected_failures)}")
    for line_number, event, label in d.unexpected_failures[:10]:
        print(f"   > Ligne {line_number} [{event}] : {label}")
    if len(d.unexpected_failures) > 10:
        print(f"   ... et {len(d.unexpected_failures) - 10} de plus (voir le fichier de log directement).")

    if d.unparseable_lines:
        print(f"\n⚠️  {len(d.unparseable_lines)} ligne(s) ressemblant à une trace Python brute (hors JSON) :")
        for line_number, preview in d.unparseable_lines[:5]:
            print(f"   > Ligne {line_number} : {preview}")

    if d.gaps:
        print(f"\n⚠️  {len(d.gaps)} interruption(s) suspecte(s) dans le flux de logs (processus probablement figé/relancé) :")
        for gap_start, gap_seconds in d.gaps[:5]:
            print(f"   > Vers {gap_start.isoformat()} : silence de {gap_seconds / 60:.1f} minutes")

    print("-" * 60)
    print(f"MAX_DAILY_LOSS_USD configuré au démarrage : {'✅ Oui' if d.daily_loss_limit_configured else '⚠️ NON -- voir alpha_engine_no_daily_loss_limit_configured'}")
    print("=" * 60)

    verdict_label = {"GO": "✅ VERDICT (PARTIEL) : GO", "ATTENTION": "⚠️ VERDICT : ATTENTION (JAUNE)", "NO_GO": "🚨 VERDICT : NO-GO (ROUGE)"}[d.verdict]
    print(f"{verdict_label} pour ce que ce run peut réellement prouver." if d.verdict == "GO" else verdict_label + ".")
    print(d.verdict_reason)


def analyze(path: Path) -> int:
    """CLI entry point: read ``path``, compute diagnostics, print the report. Returns the real exit code."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        diagnostics = compute_diagnostics(f)
    print_report(diagnostics, source_label=str(path))
    return diagnostics.exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file", type=Path, help="Path to the captured stdout log file (see module docstring's Usage).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.log_file.exists():
        print(f"❌ Erreur : fichier introuvable : {args.log_file}")
        sys.exit(2)
    sys.exit(analyze(args.log_file))


if __name__ == "__main__":
    main()

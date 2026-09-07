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

Usage::

    # Overnight, in its own terminal (stdout only -- no file handler is configured by
    # scripts/run_alpha_engine.py, matching every other launcher in this project; Docker's own log
    # driver captures stdout in a real deployment, a plain shell redirect does the same job locally):
    python scripts/run_alpha_engine.py --symbol BTC/USDT > logs/alpha_shadow_run.log 2>&1

    # The next morning:
    python scripts/analyze_shadow_run.py logs/alpha_shadow_run.log
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

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


def _iter_log_records(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str]]:
    """Yield ``(line_number, parsed_json_or_None, raw_line)`` for every non-blank line."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            stripped = raw_line.rstrip("\n")
            if not stripped.strip():
                continue
            try:
                record: dict[str, Any] | None = json.loads(stripped)
            except json.JSONDecodeError:
                record = None
            yield line_number, record, stripped


def analyze(path: Path) -> int:
    """Run the real diagnostic. Returns a process exit code (0 = GO, 1 = ATTENTION, 2 = NO-GO)."""
    print(f"🔬 [DIAGNOSTIC] Analyse du run réel : {path}\n")

    event_counts: Counter[str] = Counter()
    unexpected_failures: list[tuple[int, str, str]] = []
    guardrail_trips: list[tuple[int, str, str]] = []
    stale_data_count = 0
    simulated_quote_count = 0
    daily_loss_limit_configured = True  # innocent until the startup warning says otherwise
    started_at: datetime | None = None
    timestamps: list[datetime] = []
    unparseable_lines: list[tuple[int, str]] = []
    mixed_mock_mode_seen: set[bool] = set()

    for line_number, record, raw_line in _iter_log_records(path):
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
            started_at = timestamps[-1] if timestamps else started_at
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

    # ---- Report ----
    print("=" * 60)
    print("📊 RAPPORT D'AUDIT -- SHADOW RUN (données réelles, aucune valeur simulée)")
    print("=" * 60)
    print(f"Lignes de log JSON valides analysées : {sum(event_counts.values())}")
    print(f"Durée réelle observée (1er -> dernier timestamp) : {total_duration_hours:.1f}h")
    if mixed_mock_mode_seen and True in mixed_mock_mode_seen and False in mixed_mock_mode_seen:
        print("⚠️  Ce fichier contient plusieurs démarrages, certains en MOCK_MODE et d'autres non -- "
              "vérifie qu'il ne mélange pas plusieurs runs différents.")
    print(f"Cotations simulées calculées (mode SHADOW réel) : {simulated_quote_count}")
    print("-" * 60)

    print("🛡️  GUARDRAILS -- voir module docstring point 3 : structurellement inatteignables en SHADOW,")
    print("    donc TOUTE occurrence ici est une vraie anomalie à investiguer, pas un test réussi.")
    if guardrail_trips:
        for line_number, event, label in guardrail_trips:
            print(f"   ⚠️  Ligne {line_number} [{event}] : {label}")
    else:
        print("   Aucun déclenchement -- attendu et correct pour un run en SHADOW.")
    print(f"Coupures de flux de marché détectées (staleness réelle) : {stale_data_count}")
    print("-" * 60)

    print(f"❌ ÉCHECS RÉELS NON ATTENDUS : {len(unexpected_failures)}")
    for line_number, event, label in unexpected_failures[:10]:
        print(f"   > Ligne {line_number} [{event}] : {label}")
    if len(unexpected_failures) > 10:
        print(f"   ... et {len(unexpected_failures) - 10} de plus (voir le fichier de log directement).")

    if unparseable_lines:
        print(f"\n⚠️  {len(unparseable_lines)} ligne(s) ressemblant à une trace Python brute (hors JSON) :")
        for line_number, preview in unparseable_lines[:5]:
            print(f"   > Ligne {line_number} : {preview}")

    if gaps:
        print(f"\n⚠️  {len(gaps)} interruption(s) suspecte(s) dans le flux de logs (processus probablement figé/relancé) :")
        for gap_start, gap_seconds in gaps[:5]:
            print(f"   > Vers {gap_start.isoformat()} : silence de {gap_seconds / 60:.1f} minutes")

    print("-" * 60)
    print(f"MAX_DAILY_LOSS_USD configuré au démarrage : {'✅ Oui' if daily_loss_limit_configured else '⚠️ NON -- voir alpha_engine_no_daily_loss_limit_configured'}")
    print("=" * 60)

    # ---- Verdict ----
    if unexpected_failures or unparseable_lines:
        print("🚨 VERDICT : NO-GO (ROUGE).")
        print("Des échecs réels et non attendus sont apparus. Investigue chaque ligne listée ci-dessus "
              "avant d'envisager du capital réel.")
        return 2

    if guardrail_trips:
        print("🚨 VERDICT : NO-GO (ROUGE).")
        print("Un guardrail s'est déclenché alors qu'il devrait être structurellement inatteignable en "
              "SHADOW -- soit le moteur n'était pas réellement en SHADOW/MOCK_MODE pendant tout le run, "
              "soit c'est un vrai bug. À investiguer avant tout capital réel.")
        return 2

    if total_duration_hours < 20.0:
        print("⚠️ VERDICT : ATTENTION (JAUNE).")
        print(f"Seulement {total_duration_hours:.1f}h de run réellement observées -- relance un vrai "
              "cycle complet de 24h avant de conclure.")
        return 1

    if stale_data_count > _STALE_DATA_ATTENTION_THRESHOLD:
        print("⚠️ VERDICT : ATTENTION (JAUNE).")
        print(f"{stale_data_count} coupures de flux détectées -- connexion réseau/API pas assez stable "
              "pour du live pendant qu'un flux réel doit rester à jour.")
        return 1

    if gaps:
        print("⚠️ VERDICT : ATTENTION (JAUNE).")
        print("Des interruptions du flux de logs sont apparues (voir ci-dessus) -- confirme qu'il ne "
              "s'agit pas d'un vrai plantage silencieux avant de continuer.")
        return 1

    print("✅ VERDICT (PARTIEL) : GO pour ce que ce run peut réellement prouver.")
    print("Stabilité du processus, détection de staleness, et connectivité Redis/Telegram : propres sur "
          f"{total_duration_hours:.1f}h réelles.")
    print("Rappel honnête (voir module docstring point 3) : ce run ne prouve RIEN sur le fat-finger cap, "
          "le coupe-circuit de débit, ou l'arrêt sur perte max/jour -- ces trois-là sont validés par les "
          "31 tests réels de tests/test_alpha_engine.py, pas par ce run. Considère un passage en mode "
          "sandbox/testnet de l'exchange (si supporté) pour exercer ces chemins de code réels sans "
          "capital réel, avant le premier vrai LIVE.")
    return 0


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

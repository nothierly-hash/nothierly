"""Shadow Run Analytics -- the real post-vol audit for AlphaMarketMakerAgent's overnight SHADOW runs.

**Real data, real analysis -- no fabricated trade history** (this
project's own established discipline, see ``src/dashboard/ems_pages/
agent_fleet.py``'s module docstring on why an earlier draft's invented
Tier-2 progress numbers were rejected). This page uploads the real log
file captured from a real ``python scripts/run_alpha_engine.py > ...``
overnight run and parses it with :func:`~scripts.analyze_shadow_run.
compute_diagnostics` -- the exact same function
``scripts/analyze_shadow_run.py``'s own CLI uses, not a second,
independently-drifting reimplementation.

**Two corrections to the brief, stated upfront** (this codebase's
established discipline):

1. This is deliberately *not* rendered as a "WORM Ledger" table. The
   real, cryptographically hash-chained WORM ledger is
   :mod:`src.execution.audit_ledger`, already shown on its own real page
   (``audit_trail.py``) -- a different, tamper-evident structure with a
   real integrity guarantee. A SHADOW run's log is a plain, ephemeral
   text file with no such property; calling it "WORM" here would
   overclaim what this page actually shows.
2. There is no "slippage moyen simulé" KPI. ``AlphaMarketMakerAgent``'s
   own ``_run_simulated_cycle`` (what actually runs under SHADOW/
   MOCK_MODE) places no real order and fills nothing -- there is no real
   fill price to compute a slippage number from, simulated or otherwise.
   A fabricated slippage figure would look like real evidence in a page
   whose entire purpose is deciding whether to risk real capital; this
   page shows the real KPIs a shadow run's logs can actually support
   instead (see :class:`~scripts.analyze_shadow_run.ShadowRunDiagnostics`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import streamlit as st

from scripts.analyze_shadow_run import ShadowRunDiagnostics, compute_diagnostics
from src.dashboard.theme import status_badge

st.title("🌙 Shadow Run Analytics")
st.caption(
    "Analyse réelle des logs d'un run overnight en mode SHADOW -- lit le fichier réellement produit "
    "par `python scripts/run_alpha_engine.py > logs/alpha_shadow_run.log 2>&1`, jamais de données "
    "simulées à sa place."
)

with st.expander("ℹ️ Comment produire le fichier à analyser"):
    st.code(
        "python scripts/run_alpha_engine.py --symbol BTC/USDT > logs/alpha_shadow_run.log 2>&1",
        language="bash",
    )
    st.caption(
        "Laisse tourner (idéalement 24h, en mode SHADOW ou avec MOCK_MODE=true), puis charge le "
        "fichier résultant ci-dessous. Voir scripts/analyze_shadow_run.py pour la version ligne de "
        "commande de cette même analyse."
    )

uploaded_file = st.file_uploader(
    "Fichier de log à analyser", type=["log", "txt"],
    help="Le fichier capturé depuis la sortie réelle de scripts/run_alpha_engine.py.",
)

if uploaded_file is not None:
    if st.button("🔬 Lancer l'analyse du Shadow Run", type="primary"):
        raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
        st.session_state["shadow_run_diagnostics"] = compute_diagnostics(raw_text.splitlines())
        st.session_state["shadow_run_filename"] = uploaded_file.name

diagnostics: ShadowRunDiagnostics | None = st.session_state.get("shadow_run_diagnostics")

if diagnostics is None:
    st.info(
        "**Aucune analyse réelle chargée pour l'instant.** Charge un fichier de log ci-dessus et "
        "lance l'analyse -- rien n'est affiché en attendant, jamais un historique fictif à sa place."
    )
else:
    filename = st.session_state.get("shadow_run_filename", "?")
    st.markdown("---")
    st.subheader(f"📊 Résultat -- `{filename}`")

    if diagnostics.verdict == "GO":
        verdict_badge_kind: Literal["SAFE", "PANIC", "PENDING"] = "SAFE"
        verdict_label = "GO (partiel)"
    elif diagnostics.verdict == "ATTENTION":
        verdict_badge_kind = "PENDING"
        verdict_label = "ATTENTION"
    else:
        verdict_badge_kind = "PANIC"
        verdict_label = "NO-GO"
    st.markdown(f"### Verdict : {status_badge(verdict_badge_kind, verdict_label)}", unsafe_allow_html=True)
    st.markdown(diagnostics.verdict_reason)

    st.markdown("---")
    st.subheader("📡 KPIs réels")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Durée réelle observée", f"{diagnostics.total_duration_hours:.1f} h")
    col2.metric("Cotations simulées calculées", diagnostics.simulated_quote_count)
    col3.metric("Coupures de flux (staleness)", diagnostics.stale_data_count)
    col4.metric("Échecs réels non attendus", len(diagnostics.unexpected_failures))

    col5, col6, col7 = st.columns(3)
    col5.metric("Guardrails déclenchés", len(diagnostics.guardrail_trips))
    col6.metric("Interruptions suspectes", len(diagnostics.gaps))
    if diagnostics.market_data_circuit_breaker_trips:
        st.caption(f"🔌 Disjoncteur de données de marché : {len(diagnostics.market_data_circuit_breaker_trips)} déclenchement(s) réel(s) -- défense correcte, pas une anomalie (voir `scripts/analyze_shadow_run.py`).")
    if diagnostics.daily_loss_limit_configured is None:
        daily_loss_value = "❔ Indéterminé"
    elif diagnostics.daily_loss_limit_configured:
        daily_loss_value = "✅ Oui"
    else:
        daily_loss_value = "⚠️ Non"
    col7.metric("Perte max/jour configurée", daily_loss_value)
    if diagnostics.daily_loss_limit_configured is None:
        st.caption(
            "Indéterminé : ce run est entièrement en MOCK_MODE -- AlphaMarketMakerAgent.on_start ne "
            "journalise l'avertissement « pas de plafond configuré » qu'en mode réel (voir "
            "src/agents/alpha_engine/agent.py::on_start), donc son absence ici ne prouve rien."
        )
    st.caption(
        "Pas de « slippage moyen simulé » ici -- le mode SHADOW ne place aucun ordre réel et ne "
        "remplit rien ; il n'existe aucun prix d'exécution, simulé ou non, dont dériver un slippage. "
        "Voir le docstring de ce module."
    )

    if diagnostics.guardrail_trips:
        st.markdown("---")
        st.error(
            "⚠️ Un ou plusieurs guardrails se sont déclenchés pendant ce run SHADOW -- "
            "structurellement inattendu (voir `scripts/analyze_shadow_run.py`, point 3)."
        )
        st.dataframe(
            pd.DataFrame(diagnostics.guardrail_trips, columns=["Ligne", "Événement", "Détail"]),
            width="stretch", hide_index=True,
        )

    if diagnostics.unexpected_failures:
        st.markdown("---")
        st.error("❌ Échecs réels non attendus")
        st.dataframe(
            pd.DataFrame(diagnostics.unexpected_failures, columns=["Ligne", "Événement", "Détail"]),
            width="stretch", hide_index=True,
        )

    if diagnostics.gaps:
        st.markdown("---")
        st.warning("⚠️ Interruptions suspectes dans le flux de logs (processus probablement figé/relancé)")
        st.dataframe(
            pd.DataFrame(
                [(ts.isoformat(), f"{seconds / 60:.1f} min") for ts, seconds in diagnostics.gaps],
                columns=["Vers", "Durée du silence"],
            ),
            width="stretch", hide_index=True,
        )

    st.markdown("---")
    st.subheader("📋 Journal des événements réels")
    st.caption("Comptage réel par type d'événement, extrait directement du fichier -- jamais un historique de trades fictifs.")
    event_rows: list[tuple[str, int]] = diagnostics.event_counts.most_common()
    events_df = pd.DataFrame(event_rows, columns=["Événement", "Occurrences"])
    st.dataframe(events_df, width="stretch", hide_index=True)

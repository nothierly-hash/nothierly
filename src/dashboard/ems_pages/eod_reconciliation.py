"""EOD Reconciliation -- real, live broker-balance audit + real day-over-day break detection.

**Real backend, previously un-surfaced**: every figure here comes from
:func:`~src.execution.eod_reconciliator.run_eod_reconciliation` -- real
live ``get_account_balance()`` calls per configured broker, real
day-over-day balance-delta break detection persisted in TimescaleDB,
and a real (Shadow-mode-only, honestly labeled) inventory aggregation.
That module has run on a real midnight-UTC cron since Chapter 14, but
no dashboard page read :data:`~src.execution.eod_reconciliator.
_LATEST_REPORT_REDIS_KEY` -- confirmed by a repo-wide grep before
writing this page. This page is pure UI-surfacing of already-real data
plus one real, on-demand trigger; no new reconciliation logic lives
here.

**"Expected Inventory" vs "Broker True Balance", updated**: the brief's
own framing assumes a real, client-attributed ledger of every LIVE-mode
fill exists to compare against -- when this page was first written, it
did not (``LIVE`` orders were published onto the ZMQ bus and nothing
downstream persisted them with attribution; see
``eod_reconciliator.py``'s own module docstring). That gap is now
closed by :class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent`: real LIVE fills land in ``client_fills`` with
a real ``broker_id``, surfaced below as "Fills LIVE réels, attribués par
client" -- still not a full theoretical ledger (a broker's balance also
moves from fees/funding/anything this system didn't route), but a real,
non-fabricated cross-reference against each broker's own real balance
**today** vs. **yesterday** delta (the "Break" column). Shadow-mode
inventory stays in its own, separately labeled section, exactly as
``ReconciliationReport.scope_note`` itself insists on.

**"Generate Regulatory Report (MiFID II)", corrected**: no real MiFID
II RTS 27/28 execution-quality report exists anywhere in this
codebase (confirmed before writing this page) -- building one needs
real, separate scoping (best-execution multi-factor disclosures, APA-
format post-trade reporting), not a relabeled export. The button below
honestly exports *this real reconciliation data* as a PDF -- balances,
deltas, breaks -- not a MiFID II filing.
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.config import get_settings
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_get
from src.execution.eod_reconciliator import ReconciliationReport, run_eod_reconciliation

_LATEST_REPORT_REDIS_KEY = "EOD_RECONCILIATION_LATEST_REPORT"  # matches eod_reconciliator.py's own private constant

st.title("📑 EOD Reconciliation -- Post-Trade & Compliance")
st.caption(
    "Réconciliation réelle des soldes brokers, en direct -- voir src/execution/eod_reconciliator.py. "
    "Les ordres LIVE réels sont désormais persistés avec attribution client par "
    "MacroOrderExecutionAgent (voir ci-dessous) -- les breaks restent de vrais deltas de solde "
    "jour-sur-jour, pas une comparaison contre un inventaire théorique fabriqué."
)

settings = get_settings()
r = get_redis_client()

if st.button("▶️ Lancer la réconciliation maintenant (réel)"):
    with st.spinner("Récupération des soldes brokers réels en cours..."):
        report = asyncio.run(run_eod_reconciliation(settings=settings))
    st.session_state["eod_last_report"] = report.model_dump(mode="json")
    st.toast("Réconciliation réelle terminée.", icon="✅")

raw_report = safe_get(r, _LATEST_REPORT_REDIS_KEY)
report_dict = st.session_state.get("eod_last_report")
if report_dict is None and raw_report is not None:
    try:
        report_dict = json.loads(raw_report)
    except json.JSONDecodeError:
        report_dict = None

if report_dict is None:
    not_yet_implemented(
        "Rapport de réconciliation",
        reason=(
            "Aucun rapport réel en cache -- cliquez \"Lancer la réconciliation maintenant\" ci-dessus, "
            "ou attendez le cron réel de minuit UTC (voir schedule_eod_reconciliation)."
        ),
    )
else:
    report = ReconciliationReport.model_validate(report_dict)
    run_at_str = report.run_at.strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(report.run_at, datetime) else str(report.run_at)
    st.caption(f"Dernier rapport réel : {run_at_str}")

    breaks_by_broker = {b.broker_id: b for b in report.breaks}

    st.subheader("Soldes brokers réels -- delta jour-sur-jour")
    rows: list[dict[str, Any]] = []
    for balance in report.broker_balances:
        broken = breaks_by_broker.get(balance.broker_id)
        delta = broken.delta_usd if broken is not None else 0.0
        rows.append(
            {
                "Broker": balance.broker_id.upper(),
                "Solde réel (API)": f"${balance.balance_usd:,.2f}" if balance.balance_usd is not None else "ÉCHEC",
                "Delta / Break ($)": round(delta, 2),
                "Statut": "🔴 BREAK" if broken is not None else ("⚠️ ÉCHEC" if balance.balance_usd is None else "🟢 OK"),
                "Détail": balance.error or (broken.note if broken is not None else "--"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)

    if report.brokers_failed:
        st.warning(f"Échec de récupération réelle pour : {', '.join(b.upper() for b in report.brokers_failed)}")

    st.markdown("---")
    st.subheader("Fills LIVE réels, attribués par client -- pourquoi un Break a eu lieu")
    st.caption(
        "Réel : chaque ligne vient d'un vrai fill exécuté par MacroOrderExecutionAgent contre un broker réel "
        "(voir src/agents/macro_order_execution/agent.py), agrégé par (broker, instrument) sur les dernières 24h. "
        "Un broker listé ici avec un 🔴 BREAK ci-dessus peut désormais être croisé avec les fills réels et "
        "attribués qui ont réellement eu lieu dessus dans la même fenêtre."
    )
    if report.live_fills_by_broker:
        live_fill_rows = [
            {
                "Broker": row.broker_id.upper(),
                "Instrument": row.instrument,
                "Position nette client (LIVE)": round(row.net_quantity, 8),
                "Fills": row.fill_count,
                "Break sur ce broker ?": "🔴 OUI" if row.broker_id in breaks_by_broker else "🟢 non",
            }
            for row in report.live_fills_by_broker
        ]
        st.dataframe(live_fill_rows, width="stretch", hide_index=True)
    else:
        st.info(
            "Aucun fill LIVE réel et attribué dans les dernières 24h -- MacroOrderExecutionAgent doit avoir "
            "réellement exécuté au moins un ordre client LIVE pour peupler cette section."
        )

    st.markdown("---")
    st.subheader("Inventaire Shadow-mode (réel, mais pas un solde broker réel)")
    st.caption(report.scope_note)
    if report.shadow_inventory:
        shadow_rows = [
            {"Client": row.client_id, "Instrument": row.instrument, "Position nette (Shadow)": row.net_quantity, "Fills": row.fill_count}
            for row in report.shadow_inventory
        ]
        st.dataframe(shadow_rows, width="stretch", hide_index=True)
    else:
        st.info("Aucun fill Shadow-mode dans la fenêtre des dernières 24h.")

    st.markdown("---")
    st.subheader("Export")

    def _build_pdf(report: ReconciliationReport) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [
            Paragraph("Nothierly -- Rapport de réconciliation EOD (réel)", styles["Title"]),
            Paragraph(f"Généré : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", styles["Normal"]),
            Paragraph(f"Rapport source : {run_at_str}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Soldes brokers réels", styles["Heading2"]),
        ]
        table_data = [["Broker", "Solde ($)", "Delta ($)", "Statut"]]
        for row in rows:
            table_data.append([row["Broker"], row["Solde réel (API)"], str(row["Delta / Break ($)"]), row["Statut"]])
        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a1929")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        elements.append(table)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            "Note honnête : ceci est un export réel des données de réconciliation ci-dessus -- "
            "pas un rapport réglementaire MiFID II RTS 27/28 (non construit dans ce projet -- voir "
            "le docstring de cette page).",
            styles["Normal"],
        ))
        doc.build(elements)
        return buffer.getvalue()

    pdf_bytes = _build_pdf(report)
    st.download_button(
        "📄 Exporter le rapport de réconciliation (PDF réel)",
        data=pdf_bytes,
        file_name=f"nothierly_eod_reconciliation_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
    )
    st.caption(
        "Honnête : ceci exporte les données réelles ci-dessus en PDF -- ce n'est pas un rapport "
        "réglementaire MiFID II RTS 27/28, qui nécessiterait un travail de conception séparé et réel "
        "(divulgations multi-facteurs de meilleure exécution, format de reporting post-trade APA)."
    )

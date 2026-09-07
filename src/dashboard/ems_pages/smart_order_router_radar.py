"""Smart Order Router -- Radar Live: real-time observation of real, already-executed routing decisions.

**Real data only**: every row read here comes from
:data:`~core.schemas.SOR_ROUTING_LOG_REDIS_KEY`, a Redis list
:func:`~src.execution.smart_order_router.log_routing_decision` pushes
one real entry to every time :func:`~src.execution.smart_order_router.
execute_allocation_plan` actually routes an order (see that function's
own docstring -- never for a "Calculer le plan" preview alone). The
donut chart below shows the *last* real logged decision's real
per-venue split; the console lists real recent decisions in
chronological order; the "économies ce mois-ci" counter sums each
entry's real, always-non-negative
:attr:`~src.execution.smart_order_router.AllocationPlan.
savings_vs_worst_level_usd` for entries timestamped within the current
calendar month -- never an invented bps figure (see that property's
own docstring for exactly what it measures and why it's a real,
directly computable quantity, not an estimate).

**Where to actually route an order**: this page is deliberately
observation-only (no form here) -- routing itself happens on
``src/dashboard/ems_pages/liquidity_routing.py``'s own real
"Allocation de capital par route de liquidité" section, which already
persists here via the same real ``redis_client=`` parameter. Splitting
"configure/act" from "observe" this way mirrors this dashboard's own
established Command Center vs. per-page pattern elsewhere.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from core.schemas import SOR_ROUTING_LOG_REDIS_KEY
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_lrange

st.title("📡 Smart Order Router -- Radar Live")
st.caption(
    "Observation en temps réel des décisions de routage réellement exécutées -- "
    "voir src/execution/smart_order_router.py. Aucune donnée fabriquée : cette page "
    "est vide tant qu'aucun ordre réel n'a été routé."
)

r = get_redis_client()
raw_entries = safe_lrange(r, SOR_ROUTING_LOG_REDIS_KEY, 100)

entries: list[Any] = []
for raw in raw_entries:
    try:
        entries.append(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        continue

if not entries:
    not_yet_implemented(
        "Radar de routage en direct",
        reason=(
            "Aucune décision de routage réelle enregistrée pour l'instant -- routez un ordre réel via "
            "la page Liquidity Routing (section \"Allocation de capital par route de liquidité\") pour "
            "peupler ce radar avec des données réelles."
        ),
    )
else:
    latest = entries[0]  # LPUSH -- newest first
    st.subheader(f"Dernière décision réelle -- {latest['symbol']} ({latest['side']})")

    allocations = latest["allocations"]
    assert isinstance(allocations, list)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[str(a["broker_id"]).upper() for a in allocations],
                values=[float(a["quantity"]) for a in allocations],
                hole=0.55,
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), showlegend=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Alloué : {latest['allocated_quantity']} / {latest['requested_quantity']} demandé -- "
        + ("entièrement rempli." if latest["fully_filled"] else "rempli partiellement (profondeur réelle insuffisante).")
    )

    st.markdown("---")
    st.subheader("Économies réelles vs. pire niveau exécuté")
    now = datetime.now(timezone.utc)
    month_savings = sum(
        float(entry["savings_usd"])
        for entry in entries
        if datetime.fromtimestamp(float(entry["timestamp"]), tz=timezone.utc).month == now.month
        and datetime.fromtimestamp(float(entry["timestamp"]), tz=timezone.utc).year == now.year
    )
    col1, col2 = st.columns(2)
    col1.metric("Économies ce mois-ci (réel)", f"${month_savings:,.2f}")
    col2.metric("Décisions enregistrées", len(entries))
    st.caption(
        "Real, toujours ≥ 0 : le prix effectif du pire niveau réellement utilisé par le water-filling, "
        "appliqué à la quantité totale, moins le coût réel effectivement payé -- voir "
        "AllocationPlan.savings_vs_worst_level_usd."
    )

    st.markdown("---")
    st.subheader("Console -- décisions réelles récentes")
    console_rows = []
    for entry in entries[:30]:
        timestamp_str = datetime.fromtimestamp(float(entry["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        venues_str = ", ".join(
            f"{a['broker_id'].upper()} ({a['quantity']:.6g} @ {a['effective_price']:.2f})" for a in entry["allocations"]
        )
        console_rows.append(
            {
                "Horodatage": timestamp_str,
                "Symbole": entry["symbol"],
                "Sens": entry["side"],
                "Venues": venues_str,
                "Économie ($)": round(float(entry["savings_usd"]), 2),
            }
        )
    st.dataframe(console_rows, width="stretch", hide_index=True, height=420)

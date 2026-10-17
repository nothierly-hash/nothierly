"""Bilan -- P&L temps réel & courbe d'équité de la position propre de Nothierly.

**Real, load-bearing scope correction, verified before writing any of this**:
the brief asks for "gains/pertes de trades", Win Rate et Profit Factor --
concepts that only make sense for a book that actually opens and closes
directional positions. This codebase runs two structurally different
businesses, and only one of them has that:

1. :class:`~src.execution.reporting_engine.ClientMonthlyMetrics` (B2B
   execution-as-a-service, see that module's own docstring point 2):
   Nothierly never takes a directional position *for* a client, so a
   client's fills have no PnL/win-rate to report -- ``gross_pnl_usd`` is
   honestly ``None`` there unless the client reports their own result.
2. :class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`: Nothierly's
   *own* market-making book, with real FIFO cost-basis tracking
   (``_apply_fill``) and real realized PnL per fill (``_record_fill`` --
   see that method's own module docstring point 7 for the one disclosed
   approximation: fills are detected as a position delta between two
   polled broker reads, priced at that cycle's mid, not the fill's exact
   execution price).

This page is built on (2) -- the only real source of "trades gagnants/
perdants" in this codebase. ``_record_fill`` now also pushes one real
entry per fill to :data:`~core.schemas.ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX`
(added alongside this page -- the pre-existing
``ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX`` snapshot has no history, only
today's running totals, and a real equity curve needs a real time series).

**Honestly empty right now**: the deployed engine runs under
``MOCK_MODE=true`` (Binance testnet) -- ``_run_simulated_cycle`` returns
before ever reaching fill detection (module docstring point 3 on that
agent), so zero real fills exist yet and every KPI below is a real,
correctly-computed zero, not a placeholder. This page renders that
state cleanly rather than crashing or fabricating a curve -- it starts
populating the moment the engine's real (non-mock) path detects its
first fill.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.schemas import ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES, ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX
from src.dashboard.ems_pages._shared import get_redis_client, safe_keys, safe_lrange
from src.dashboard.theme import CRIMSON, MINT, STEEL_GRAY

st.title("💰 Bilan -- P&L & Courbe d'Équité")
st.caption(
    "P&L réel de la position propre de Nothierly (AlphaMarketMakerAgent, market-making) -- voir "
    "src/agents/alpha_engine/agent.py::_record_fill. Ceci n'est PAS un P&L client : Nothierly ne prend "
    "jamais de position directionnelle pour le compte d'un client B2B, donc ce chiffre n'existe "
    "structurellement pas (voir src/execution/reporting_engine.py). Aucune donnée simulée : cette page "
    "reste à zéro tant qu'aucun fill réel (hors MOCK_MODE) n'a été détecté."
)

refresh_col, _ = st.columns([1, 3])
with refresh_col:
    refresh_seconds = st.select_slider(
        "Rafraîchissement auto", options=[2, 3, 4, 5], value=3, format_func=lambda s: f"{s}s"
    )
st_autorefresh(interval=refresh_seconds * 1000, key="bilan_autorefresh")

r = get_redis_client()
agent_keys = safe_keys(r, f"{ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX}*")

entries: list[dict[str, Any]] = []
for key in agent_keys:
    agent_id = key[len(ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX) :]
    for raw in safe_lrange(r, key, ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES):
        try:
            record = json.loads(raw)
            record["_agent_id"] = agent_id
            record["_delta"] = Decimal(str(record["realized_pnl_delta_usd"]))
            entries.append(record)
        except (json.JSONDecodeError, TypeError, KeyError, InvalidOperation):
            continue

entries.sort(key=lambda e: float(e["timestamp"]))  # oldest first, for a real chronological equity curve

gains_cumules = sum((e["_delta"] for e in entries if e["_delta"] > 0), Decimal(0))
pertes_cumulees = sum((e["_delta"] for e in entries if e["_delta"] < 0), Decimal(0))
pnl_net = gains_cumules + pertes_cumulees
win_count = sum(1 for e in entries if e["_delta"] > 0)
loss_count = sum(1 for e in entries if e["_delta"] < 0)
decisive_count = win_count + loss_count
win_rate_pct = (win_count / decisive_count * 100.0) if decisive_count > 0 else None
profit_factor = (float(gains_cumules) / abs(float(pertes_cumulees))) if pertes_cumulees != 0 else None

st.markdown("---")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Gains cumulés", f"${float(gains_cumules):,.2f}")
kpi2.metric("Pertes cumulées", f"-${abs(float(pertes_cumulees)):,.2f}")
kpi3.metric("P&L Net", f"${float(pnl_net):,.2f}" if pnl_net >= 0 else f"-${abs(float(pnl_net)):,.2f}")
with kpi4:
    win_rate_label = f"{win_rate_pct:.1f}%" if win_rate_pct is not None else "N/A"
    profit_factor_label = f"{profit_factor:.2f}" if profit_factor is not None else ("∞" if gains_cumules > 0 else "N/A")
    st.metric("Win Rate / Profit Factor", f"{win_rate_label} / {profit_factor_label}")

if not entries:
    st.info(
        "**En attente des premiers signaux** -- aucun fill réel enregistré pour l'instant "
        f"({len(agent_keys)} moteur(s) alpha détecté(s), tous à 0 fill réel). Les métriques ci-dessus "
        "sont honnêtement à zéro, pas des placeholders. Voir le caption en haut de page pour pourquoi "
        "(MOCK_MODE=true sur ce déploiement)."
    )
    st.stop()

st.caption(
    f"{len(entries)} fill(s) réel(s) sur {len(agent_keys)} moteur(s) -- "
    f"{win_count} gagnant(s), {loss_count} perdant(s)"
    + (f", {len(entries) - decisive_count} neutre(s)" if len(entries) > decisive_count else "")
    + "."
)

st.markdown("---")
st.subheader("Courbe d'équité (P&L cumulé réel)")

timestamps = [datetime.fromtimestamp(float(e["timestamp"]), tz=timezone.utc) for e in entries]
cumulative: list[float] = []
running = 0.0
for e in entries:
    running += float(e["_delta"])
    cumulative.append(running)

curve_color = MINT if cumulative[-1] >= 0 else CRIMSON
fig = go.Figure()
fig.add_hline(y=0, line_dash="dash", line_color=STEEL_GRAY, annotation_text="Seuil de rentabilité")
fig.add_trace(
    go.Scatter(
        x=timestamps, y=cumulative, mode="lines", name="P&L cumulé",
        line=dict(color=curve_color, width=2.5),
        fill="tozeroy", fillcolor=f"rgba({'0,230,118' if curve_color == MINT else '255,23,68'},0.12)",
    )
)
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
    xaxis_title="Horodatage réel des exécutions", yaxis_title="P&L cumulé ($)",
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.markdown("---")
st.subheader("Fills réels récents")
console_rows = [
    {
        "Horodatage": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Moteur": e["_agent_id"],
        "Symbole": e.get("symbol", "?"),
        "Qté fill": e.get("fill_qty", "?"),
        "Prix fill": e.get("fill_price", "?"),
        "PnL réalisé ($)": round(float(e["_delta"]), 2),
    }
    for ts, e in zip(reversed(timestamps), reversed(entries))
][:50]
st.dataframe(console_rows, width="stretch", hide_index=True, height=380)

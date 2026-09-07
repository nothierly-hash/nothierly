"""Command Center -- executive summary: PnL, TCA, order book state.

**Real data, honestly scoped**: AUM/positions/system status are read
from the same real Redis keys ``src/dashboard/app.py`` already uses
(``NOTHIERLY_MARKET_STATE``, position keys, ``SYSTEM_STATUS``).

**3D order-book depth topography -- corrected, a real surface now, not
a stale gap-note**: this section previously said no live L2 depth
snapshot was written to Redis for a dashboard to read, only
top-of-book scalars. That was true when first written; fixed by
:mod:`~src.agents.market_data.l2_surface_packer` (real, capped
per-symbol history under ``l2_surface:{symbol}``, fed by
:class:`~src.agents.market_data.agent.MarketDataAgent`'s own already-
computed :class:`~core.schemas.OrderBookSnapshot`, previously discarded)
and :func:`~src.dashboard.ems_pages._l2_surface_chart.
build_l2_surface_figure` (real bid/ask surfaces on a real price axis,
meeting near the real mid-price). Still shown as an explicit "not yet
available" state, honestly, whenever no ``MarketDataAgent`` has
published for the selected symbol in this environment -- never an
invented surface.

**TCA -- corrected, a real computation now, not a stale gap-note**:
this section previously claimed no fill/execution history was
persisted anywhere. That was true when first written, but went stale
once Shadow Mode started writing real, client-attributed fills (with
real ``fee_usd``/``slippage_saved_usd``) to
:data:`_FILLS_BLOTTER_REDIS_KEY` -- see
``api_middleware/gateway.py``'s ``_write_shadow_fill`` -- the same real
list ``src/dashboard/ems_pages/order_blotter.py`` already reads for its
own "Total Fills" metric. Found stale via a live audit of every
dashboard page; fixed to compute real totals from those fields. Real
production fills (:meth:`~src.agents.execution.agent.ExecutionAgent.
_log_fill`) do not carry ``fee_usd``/``slippage_saved_usd`` at all
(that agent only logs ``instrument``/``action``/``price``/``units``/
``latency_ms``) -- so this total is honestly scoped to SHADOW-mode
fills only, stated explicitly below the numbers, not silently
presented as covering every fill in the blotter.
"""

from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX
from core.security import STATUS_PANIC, SYSTEM_STATUS_KEY
from src.agents.market_data.l2_surface_packer import read_l2_surface_history
from src.dashboard.ems_pages._l2_surface_chart import build_l2_surface_figure
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_get, safe_get_json, safe_lrange
from src.dashboard.theme import status_badge

_MONITORED_SYMBOLS = ("BTC/USDT", "EUR_USD", "XAU_USD")
_MARKET_STATE_REDIS_KEY = "NOTHIERLY_MARKET_STATE"
_ACTIVE_POSITIONS_REDIS_KEY = "NOTHIERLY_ACTIVE_POSITIONS"
_EQUITY_CURVE_REDIS_KEY = "NOTHIERLY_EQUITY_CURVE"
_FILLS_BLOTTER_REDIS_KEY = "NOTHIERLY_FILLS_BLOTTER"

st.title("📊 Command Center")
st.caption("Résumé exécutif -- statut système, capital, carnet d'ordres.")

r = get_redis_client()

system_status = safe_get(r, SYSTEM_STATUS_KEY)
badge = status_badge("PANIC" if system_status == STATUS_PANIC else "SAFE")
st.markdown(f"**SYSTEM_STATUS:** {badge}", unsafe_allow_html=True)

# Same real "no fabricated numbers" pattern as src/dashboard/app.py:
# AUM comes from the real equity-curve's latest sample (None if no
# trading history yet), gross exposure from real active positions.
raw_equity_points = safe_lrange(r, _EQUITY_CURVE_REDIS_KEY, 1)
current_aum = None
if raw_equity_points:
    try:
        current_aum = json.loads(raw_equity_points[0]).get("balance")
    except (json.JSONDecodeError, TypeError):
        pass

active_positions: list[dict[str, object]] = []
raw_positions = safe_get(r, _ACTIVE_POSITIONS_REDIS_KEY)
if raw_positions:
    try:
        active_positions = json.loads(raw_positions)
    except json.JSONDecodeError:
        pass

gross_notional = 0.0
for position in active_positions:
    units, entry_price = position.get("units"), position.get("entry_price")
    if isinstance(units, (int, float)) and isinstance(entry_price, (int, float)):
        gross_notional += abs(float(units) * float(entry_price))
risk_exposure_pct = (gross_notional / current_aum * 100.0) if current_aum else None

col1, col2, col3 = st.columns(3)
col1.metric("AUM (réel)", f"${current_aum:,.2f}" if current_aum else "N/A")
col2.metric("Exposition brute", f"{risk_exposure_pct:.1f}%" if risk_exposure_pct is not None else "N/A")
col3.metric("Positions ouvertes", str(len(active_positions)))

st.markdown("---")
st.subheader("Transaction Cost Analysis (TCA)")
raw_fills = safe_lrange(r, _FILLS_BLOTTER_REDIS_KEY, 200)
total_fee_usd = 0.0
total_slippage_saved_usd = 0.0
shadow_fill_count = 0
total_fill_count = 0
for raw_fill in raw_fills:
    try:
        fill_entry = json.loads(raw_fill)
    except (json.JSONDecodeError, TypeError):
        continue
    total_fill_count += 1
    fee_usd, slippage_saved_usd = fill_entry.get("fee_usd"), fill_entry.get("slippage_saved_usd")
    if fee_usd is None and slippage_saved_usd is None:
        continue  # a real production ExecutionAgent fill -- see module docstring, never carries these fields
    shadow_fill_count += 1
    total_fee_usd += float(fee_usd or 0.0)
    total_slippage_saved_usd += float(slippage_saved_usd or 0.0)

if shadow_fill_count > 0:
    tca_col1, tca_col2 = st.columns(2)
    tca_col1.metric("Frais réels payés (SHADOW)", f"${total_fee_usd:,.2f}")
    tca_col2.metric("Slippage réel évité (SHADOW)", f"${total_slippage_saved_usd:,.2f}")
    st.caption(
        f"Calculé sur {shadow_fill_count}/{total_fill_count} fills du journal réel NOTHIERLY_FILLS_BLOTTER -- "
        "uniquement ceux soumis en execution_mode=SHADOW (voir _write_shadow_fill dans api_middleware/gateway.py). "
        "Les fills réels de production (ExecutionAgent._log_fill) ne calculent pas encore ces champs."
    )
else:
    not_yet_implemented(
        "Frais économisés / slippage évité",
        reason="Aucun fill SHADOW (portant fee_usd/slippage_saved_usd) dans NOTHIERLY_FILLS_BLOTTER pour l'instant -- soumettez un ordre en execution_mode=SHADOW via le gateway pour peupler des données réelles.",
    )

st.markdown("---")
st.subheader("Order Book -- Top of Book (temps réel)")
rows = []
for symbol in _MONITORED_SYMBOLS:
    metrics = safe_get_json(r, f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
    if metrics:
        rows.append((symbol, metrics.get("spread"), metrics.get("order_book_imbalance"), metrics.get("mid_price")))

if rows:
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Symbole", "Spread", "OBI", "Mid Price"], fill_color="rgba(0,229,255,0.15)", align="left"),
        cells=dict(values=list(zip(*rows)), align="left"),
    )])
    fig.update_layout(height=180, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
else:
    st.info("Aucune donnée MarketMetrics fraîche en direct (aucun MarketDataAgent actif dans cet environnement).")

st.markdown("---")
st.subheader("Topographie 3D du carnet d'ordres")
l2_symbol = st.selectbox("Symbole", _MONITORED_SYMBOLS, key="l2_surface_symbol")
l2_history = read_l2_surface_history(r, l2_symbol)
if l2_history:
    st.plotly_chart(build_l2_surface_figure(l2_history), width="stretch", config={"displayModeBar": False})
    st.caption(
        f"{len(l2_history)} snapshot(s) réel(s) -- src.agents.market_data.l2_surface_packer, "
        f"profondeur réelle de {l2_history[0].depth} niveaux par côté (ORDER_BOOK_TOP_N)."
    )
else:
    not_yet_implemented(
        "Topographie 3D du carnet d'ordres complet",
        reason=f"Aucun snapshot L2 pour {l2_symbol} dans Redis (l2_surface:{l2_symbol}) -- aucun MarketDataAgent actif pour ce symbole dans cet environnement pour le moment.",
    )

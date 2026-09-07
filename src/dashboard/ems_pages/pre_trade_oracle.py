"""Pre-Trade Oracle -- real UI for pricing.pre_trade_oracle.estimate_total_implementation_cost. Chapter 13, Étape 5.9.

**Real data path, matching liquidity_routing.py's own established
pattern**: "Récupérer le carnet en direct" builds a real, momentarily
connected :class:`~src.execution.broker_gateway.BaseBroker` via
``BrokerFactory.create`` + ``asyncio.run(...)`` (Streamlit pages are
sync scripts; every other page that talks to a live broker already
does this, see ``liquidity_routing.py``'s ping buttons) and calls its
real ``get_orderbook_snapshot`` -- a genuine top-5-level L2 snapshot
from the connected venue, not a fabricated one. A failure (bad
credentials, exchange unreachable) is shown honestly; the quote is
simply not computed until a real book is fetched, or Mode Démo is
switched on for a synthetic illustration book -- the same explicit,
banner'd opt-in every other page in this app uses, never a silent
substitute.

``tick_size``/``lot_size`` on the constructed ``UniversalAsset`` are
display-only placeholders (:func:`pricing.pre_trade_oracle.
estimate_total_implementation_cost` never reads either field) -- see
that module for what actually drives the quote.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.nothierly_ontology import AssetClass, OrderSide, OrderStatus, OrderType, UniversalAsset, UniversalOrder
from core.schemas import OrderBookLevel, OrderBookSnapshot
from pricing.pre_trade_oracle import InsufficientLiquidityError, estimate_total_implementation_cost, simulate_cost_curve
from src.dashboard.ems_pages._shared import demo_mode_toggle
from src.execution.broker_gateway import BrokerFactory

st.title("🔮 Pre-Trade Oracle")
st.caption("Coût réel total estimé avant envoi -- spread, slippage, frais, funding, taxe -- contre un carnet L2 réel.")

_EXCHANGES = ["binance", "bybit", "okx", "deribit", "oanda"]
_ASSET_CLASS_BY_EXCHANGE = {
    "binance": AssetClass.CRYPTO_SPOT, "bybit": AssetClass.CRYPTO_SPOT,
    "okx": AssetClass.CRYPTO_SPOT, "deribit": AssetClass.CRYPTO_SPOT, "oanda": AssetClass.FOREX,
}


def _demo_book(symbol: str, *, mid: float = 65_000.0) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=symbol, timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=mid - i * mid * 0.0005, volume=0.4 + i * 0.15) for i in range(5)],
        asks=[OrderBookLevel(price=mid + i * mid * 0.0005, volume=0.4 + i * 0.15) for i in range(5)],
    )


st.subheader("1. Carnet d'ordres")
col_ex, col_sym, col_fetch = st.columns([1, 1, 1])
with col_ex:
    exchange_id = st.selectbox("Venue", _EXCHANGES)
with col_sym:
    symbol = st.text_input("Symbole", value="BTC/USDT" if exchange_id != "oanda" else "EUR_USD")
with col_fetch:
    st.write("")
    st.write("")
    fetch_clicked = st.button("📡 Récupérer le carnet en direct", width="stretch")

demo_on = demo_mode_toggle(key="oracle_demo", label="🎭 Mode Démo (carnet simulé)")

if fetch_clicked:
    async def _fetch() -> tuple[OrderBookSnapshot | None, str | None]:
        try:
            broker = await BrokerFactory.create(exchange_id)
            async with broker:
                return await broker.get_orderbook_snapshot(symbol), None
        except Exception as exc:  # noqa: BLE001 -- any real connector failure surfaces honestly below, never silently
            return None, str(exc)

    snapshot, error = asyncio.run(_fetch())
    if snapshot is not None:
        st.session_state["oracle_book"] = snapshot
        st.session_state["oracle_book_source"] = f"réel ({exchange_id})"
        st.success(f"Carnet réel récupéré pour {symbol} sur {exchange_id}.")
    else:
        st.error(f"Échec de récupération du carnet réel : {error}")

if demo_on:
    st.session_state["oracle_book"] = _demo_book(symbol)
    st.session_state["oracle_book_source"] = "simulé (Mode Démo)"

book: OrderBookSnapshot | None = st.session_state.get("oracle_book")
book_source: str = st.session_state.get("oracle_book_source", "")

if book is None:
    st.info("Aucun carnet réel chargé -- cliquez sur **Récupérer le carnet en direct** ou activez le Mode Démo.")
    st.stop()

st.caption(f"Carnet actif : {book.symbol} -- source : {book_source} -- {len(book.bids)} niveaux bid / {len(book.asks)} niveaux ask.")

st.subheader("2. Ordre")
c1, c2, c3, c4 = st.columns(4)
with c1:
    side = st.selectbox("Sens", ["BUY", "SELL"])
with c2:
    quantity_str = st.text_input("Quantité", value="1.0")
with c3:
    order_type = st.selectbox("Type", ["MARKET", "LIMIT"])
with c4:
    limit_price_str = st.text_input("Prix limite", value="", disabled=order_type == "MARKET")

c5, c6, c7 = st.columns(3)
with c5:
    jurisdiction_choice = st.selectbox("Juridiction (taxe)", ["Aucune", "FR", "UK", "US"])
with c6:
    volume_str = st.text_input("Volume client 30j (USD, pour palier de frais)", value="0")
with c7:
    is_perp = st.checkbox("Perpétuel (funding)", value=False, help="Aucun adaptateur de ce projet ne place réellement de perpétuel aujourd'hui -- voir pricing/pre_trade_oracle.py. Activé ici uniquement pour illustrer le calcul.")

funding_rate_str, holding_hours_str = "0", "0"
if is_perp:
    c8, c9 = st.columns(2)
    with c8:
        funding_rate_str = st.text_input("Funding rate / 8h (ex: 0.0001)", value="0.0001")
    with c9:
        holding_hours_str = st.text_input("Durée de détention prévue (h)", value="24")

compute_clicked = st.button("⚡ Calculer le coût réel total", type="primary")

if not compute_clicked:
    st.stop()

try:
    quantity = Decimal(quantity_str)
    limit_price = Decimal(limit_price_str) if order_type == "LIMIT" and limit_price_str else None
    client_volume = Decimal(volume_str or "0")
    funding_rate = Decimal(funding_rate_str or "0") if is_perp else None
    holding_hours = Decimal(holding_hours_str or "0") if is_perp else Decimal(0)
except InvalidOperation:
    st.error("Valeur numérique invalide.")
    st.stop()

if quantity <= 0:
    st.error("La quantité doit être positive.")
    st.stop()
if order_type == "LIMIT" and limit_price is None:
    st.error("Un ordre LIMIT nécessite un prix limite.")
    st.stop()

asset_class = AssetClass.CRYPTO_PERPETUAL if is_perp else _ASSET_CLASS_BY_EXCHANGE.get(exchange_id, AssetClass.CRYPTO_SPOT)
base, _, quote = symbol.partition("/") if "/" in symbol else symbol.partition("_")
asset = UniversalAsset(
    symbol=symbol, asset_class=asset_class, base_currency=base or symbol, quote_currency=quote or "USD",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)
order = UniversalOrder(
    # This order is a hypothetical cost preview, never actually submitted to a broker or reported to a
    # regulator -- algo_id/regulatory_client_shortcode are honest sentinels marking it as such, not real
    # regulatory-reportable values (see UniversalOrder's own docstring on what these fields are really for).
    order_id="oracle-preview", broker_id=exchange_id, asset=asset,
    algo_id="PRE_TRADE_ORACLE_PREVIEW", decision_timestamp_utc=datetime.now(timezone.utc),
    regulatory_client_shortcode="INTERNAL_PREVIEW",
    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
    order_type=OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT,
    quantity=quantity, limit_price=limit_price, status=OrderStatus.PENDING,
)
jurisdiction = None if jurisdiction_choice == "Aucune" else jurisdiction_choice

try:
    quote_result = estimate_total_implementation_cost(
        order, book, jurisdiction=jurisdiction, client_30day_volume_usd=client_volume,
        funding_rate_per_8h=funding_rate, holding_period_hours=holding_hours,
    )
except InsufficientLiquidityError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("3. Devis réel")
m1, m2, m3 = st.columns(3)
m1.metric("Prix moyen simulé", f"${quote_result.simulated_avg_fill_price:,.4f}")
m2.metric("Prix de référence (mid)", f"${quote_result.mid_price_reference:,.4f}")
m3.metric("Prix de rentabilité (break-even)", f"${quote_result.break_even_price:,.4f}")

if quote_result.unfilled_quantity > 0:
    st.warning(f"⚠️ {quote_result.unfilled_quantity} unités non couvertes par la profondeur réelle capturée -- ce devis est un plancher.")

st.markdown("#### Répartition du coût réel")
breakdown = {
    "Impact de marché (slippage)": float(quote_result.market_impact_usd),
    "Frais d'exchange": float(quote_result.exchange_fee_usd),
    "Funding": float(quote_result.funding_cost_usd),
    "Taxe": float(quote_result.tax_usd),
}
fig = go.Figure(go.Bar(
    x=list(breakdown.values()), y=list(breakdown.keys()), orientation="h",
    marker_color=["#00E5FF", "#2962FF", "#FF1744", "#5B6B7C"], text=[f"${v:,.2f}" for v in breakdown.values()], textposition="auto",
))
fig.update_layout(
    height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E8EAED", xaxis_title="USD",
)
st.plotly_chart(fig, width="stretch")

st.metric("Coût total réel estimé", f"${quote_result.total_cost_usd:,.2f}", help=f"Palier de frais appliqué : {quote_result.fee_tier_applied}")

for note in quote_result.notes:
    st.info(note)

st.markdown("#### Courbe coût vs. taille (impact de marché, calcul numpy vectorisé)")
max_depth = float(sum(lvl.volume for lvl in (book.asks if side == "BUY" else book.bids)))
sizes = np.linspace(max_depth * 0.05, max_depth * 1.5, 20)
costs_bps = simulate_cost_curve(OrderSide.BUY if side == "BUY" else OrderSide.SELL, book, sizes)
curve_fig = go.Figure(go.Scatter(x=sizes, y=costs_bps, mode="lines+markers", line=dict(color="#00E676")))
curve_fig.update_layout(
    height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#E8EAED", xaxis_title="Taille de l'ordre", yaxis_title="Impact de marché (bps)",
)
st.plotly_chart(curve_fig, width="stretch")
st.caption("Les tailles au-delà de la profondeur réelle capturée (5 niveaux) apparaissent comme un trou dans la courbe -- jamais extrapolées.")

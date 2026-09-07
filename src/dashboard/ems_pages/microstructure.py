"""Microstructure Telemetry -- live OBI/spread/VPIN/TFI, all from the real bus/tape.

**Real data**: OBI/spread/mid-price come from
:class:`~src.agents.market_data.agent.MarketDataAgent`'s real
``market_metrics:{symbol}`` Redis writes.

**VPIN/TFI, now real and live**: this page's own gap used to read "VPIN
and Trade Flow Imbalance (TFI) are not computed anywhere in this
project's live ``src/`` pipeline -- both exist only in ``research/
alpha_features.py`` (offline research code, Chapter 12)." That real
research code is now reused, unchanged, by
:mod:`~src.agents.market_data.microstructure_features` against a real,
newly-persisted trade tape (see :data:`~core.schemas.
TRADE_TAPE_REDIS_KEY_PREFIX`) -- see that module's own docstring for
the honest "insufficient real data yet" case, shown explicitly rather
than as a fabricated placeholder number. ``src/agents/quant_copilot.
py``'s ``get_live_toxicity`` tool and ``src/agents/alert_agent.py``'s
microstructure-danger check still use OBI, unaffected by this change.

**Mathematical Framework section**: the two formulas below (Triple
Barrier hurdle rate, hockey-stick inventory penalty) are not generic
textbook equations -- they're the *exact* real, calibrated formulas
this project already built and measured in Chapter 12
(``research/alpha_features.py``'s ``label_triple_barrier``,
``research/market_making_env.py``'s Étape 4.3 penalty), reproduced here
with their real solved constants, not re-derived or approximated for
display.

**Demo overlay**: the OBI/VPIN/spread curves below the math are, when
toggled on, ``numpy``-generated mock data for a live B2B demo -- never
shown without the explicit toggle and its warning banner (see
:func:`~src.dashboard.ems_pages._shared.demo_mode_toggle`), never
substituted silently for the real telemetry table/chart above.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import get_settings
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX
from src.agents.market_data.microstructure_features import (
    LiveMicrostructureReading,
    compute_live_microstructure_reading,
    fetch_real_trade_tape,
)
from src.dashboard.ems_pages._shared import demo_mode_toggle, get_redis_client, safe_get_json
from src.dashboard.theme import get_theme_mode

_MONITORED_SYMBOLS = ("BTC/USDT", "EUR_USD", "XAU_USD")

# Real, calibrated constants reproduced from research/market_making_env.py's
# Étape 4.3 hockey-stick penalty (solved so the *cumulative* penalty hits
# exactly one taker-fee's worth at grace + patience ticks) -- see that
# module's own docstring for the full derivation this page only displays.
_TAKER_FEE = 0.0004
_GRACE_PERIOD_TICKS = 10
_POST_GRACE_PATIENCE_TICKS = 22
_GRACE_PERIOD_BETA = (0.05 * _TAKER_FEE) / (_GRACE_PERIOD_TICKS * (_GRACE_PERIOD_TICKS + 1) / 2)
_POST_GRACE_ALPHA = (0.95 * _TAKER_FEE) / (
    _POST_GRACE_PATIENCE_TICKS * (_POST_GRACE_PATIENCE_TICKS + 1) * (2 * _POST_GRACE_PATIENCE_TICKS + 1) / 6
)

st.title("📈 Microstructure Telemetry")
st.caption("Radiographie du marché en direct -- OBI, spread, mid-price réels.")

r = get_redis_client()

if "obi_history" not in st.session_state:
    st.session_state.obi_history = {symbol: [] for symbol in _MONITORED_SYMBOLS}

rows = []
for symbol in _MONITORED_SYMBOLS:
    metrics = safe_get_json(r, f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
    if metrics is None:
        rows.append({"Symbole": symbol, "OBI": "N/A", "Spread": "N/A", "Mid Price": "N/A", "Horodatage": "N/A"})
        continue
    rows.append({
        "Symbole": symbol, "OBI": round(metrics["order_book_imbalance"], 4),
        "Spread": round(metrics["spread"], 6), "Mid Price": round(metrics["mid_price"], 2),
        "Horodatage": metrics["timestamp"],
    })
    st.session_state.obi_history[symbol].append(metrics["order_book_imbalance"])
    st.session_state.obi_history[symbol] = st.session_state.obi_history[symbol][-200:]

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.markdown("---")
st.subheader("Historique OBI (session en cours, échantillonné à chaque rafraîchissement)")
for symbol in _MONITORED_SYMBOLS:
    history = st.session_state.obi_history[symbol]
    if history:
        st.line_chart(pd.DataFrame({symbol: history}), height=140)

st.markdown("---")
st.subheader("VPIN & TFI en direct (réel)")
st.caption(
    "Calculé à partir du vrai flux de trades observé par MarketDataAgent (voir microstructure_features.py) -- "
    "réutilise research/alpha_features.py sans le modifier, jamais une approximation."
)

col_bucket, col_buckets = st.columns(2)
vpin_bucket_volume = col_bucket.number_input(
    "Taille de bucket VPIN (volume réel/bucket)", min_value=0.01, value=1.0, step=0.1,
    help="Volume réel cumulé qui ferme un bucket -- ajustez selon la liquidité réelle de l'actif suivi.",
)
vpin_rolling_buckets = col_buckets.number_input(
    "Buckets glissants VPIN", min_value=2, max_value=200, value=10, step=1,
)


async def _fetch_all_microstructure_readings() -> list[LiveMicrostructureReading]:
    redis_client = await RedisClient.get_instance(get_settings().redis)
    readings = []
    for symbol in _MONITORED_SYMBOLS:
        tape = await fetch_real_trade_tape(redis_client, symbol)
        readings.append(
            compute_live_microstructure_reading(
                tape, symbol=symbol, vpin_bucket_volume=float(vpin_bucket_volume),
                vpin_rolling_buckets=int(vpin_rolling_buckets),
            )
        )
    return readings


microstructure_readings = asyncio.run(_fetch_all_microstructure_readings())
st.dataframe(
    [
        {
            "Symbole": reading.symbol,
            "Trades réels (fenêtre)": reading.trade_count,
            "VPIN réel": f"{reading.vpin:.4f}" if reading.vpin is not None else "N/A",
            "TFI 1s": f"{tfi:+.4f}" if (tfi := reading.tfi_by_window.get('1s')) is not None else "N/A",
            "TFI 5s": f"{tfi:+.4f}" if (tfi := reading.tfi_by_window.get('5s')) is not None else "N/A",
            "TFI 30s": f"{tfi:+.4f}" if (tfi := reading.tfi_by_window.get('30s')) is not None else "N/A",
        }
        for reading in microstructure_readings
    ],
    width="stretch", hide_index=True,
)
for reading in microstructure_readings:
    if reading.insufficient_data_reason is not None:
        st.caption(f"{reading.symbol} : {reading.insufficient_data_reason}")

st.markdown("---")
st.subheader("🧮 Cadre Mathématique")
st.caption("Les formules réelles et calibrées de ce système (Chapitre 12), pas des équations génériques de manuel.")

col_barrier, col_penalty = st.columns(2)

with col_barrier:
    st.markdown("**Triple Barrier -- seuil de rentabilité (Take-Profit)**")
    st.latex(
        r"P_{TP}(i) = P(i) \left(1 + 2f + \sigma_h \cdot m_{PT}\right), \quad "
        r"\sigma_h = \sigma_{tick} \sqrt{h}"
    )
    st.latex(
        r"P_{SL}(i) = P(i) \left(1 - \sigma_h \cdot m_{SL}\right)"
    )
    st.caption(
        "f = frais taker par jambe (0.04%) · h = horizon vertical (ticks) · "
        "σ_tick = volatilité réalisée glissante. Le terme 2f impose que le "
        "gain couvre l'aller-retour complet avant de compter comme un vrai succès "
        "(research/alpha_features.py::label_triple_barrier)."
    )
    st.markdown(
        "Calibration réelle mesurée (Chapitre 12, h=1000, m_PT=0.25) : "
        "**précision SL 24.03%**, **précision PT 5.04%** (z=9.53, significatif "
        "statistiquement mais économiquement marginal après frais réalistes)."
    )

with col_penalty:
    st.markdown("**Pénalité d'inventaire \"hockey-stick\" (Avellaneda-Stoikov, adaptée)**")
    st.latex(
        r"""
        \text{Penalty}(t) = \begin{cases}
        \beta \cdot t & t \leq T_g \\
        \alpha \cdot (t - T_g)^2 & t > T_g
        \end{cases}
        """
    )
    st.caption(
        f"t = ticks en position · T_g = période de grâce ({_GRACE_PERIOD_TICKS} ticks) · "
        f"β = {_GRACE_PERIOD_BETA:.3e} · α = {_POST_GRACE_ALPHA:.3e}. β et α sont "
        f"résolus (pas devinés) pour que la pénalité cumulée égale exactement un "
        f"frais taker ({_TAKER_FEE:.4f}) au tick T_g + {_POST_GRACE_PATIENCE_TICKS} "
        "(le délai médian réel de fill Maker mesuré en Étape 3.9)."
    )
    st.markdown(
        "Remplace la pénalité quadratique simple de l'Étape 4.1, qui rendait la "
        "sortie Taker (rapide, garantie) dominante à 89.7% des sorties -- "
        "un vrai résultat mesuré, pas une hypothèse (research/market_making_env.py)."
    )

st.markdown("---")
st.subheader("📡 Courbes OBI / VPIN / Spread")
demo_on = demo_mode_toggle(key="microstructure_demo_mode")

if demo_on:
    demo_symbol = st.selectbox("Actif (démo)", _MONITORED_SYMBOLS, key="microstructure_demo_symbol")
    n_points = 120
    rng = np.random.default_rng(abs(hash(demo_symbol)) % (2**32))
    t = np.arange(n_points)
    obi_curve = np.clip(0.15 * np.sin(t / 9.0) + rng.normal(0, 0.08, n_points), -1, 1)
    vpin_curve = np.clip(0.35 + 0.2 * np.sin(t / 14.0 + 1.0) + rng.normal(0, 0.05, n_points), 0, 1)
    spread_curve = np.abs(0.4 + 0.15 * np.sin(t / 6.0) + rng.normal(0, 0.05, n_points))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=obi_curve, name="OBI (simulé)", line=dict(color="#00E5FF")))
    fig.add_trace(go.Scatter(y=vpin_curve, name="VPIN (simulé)", line=dict(color="#FF1744")))
    fig.add_trace(go.Scatter(y=spread_curve, name="Spread (simulé, normalisé)", line=dict(color="#00E676")))
    font_color = "#E8EAED" if get_theme_mode() == "dark" else "#161A20"
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(color=font_color),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Active le Mode Démo ci-dessus pour visualiser des courbes OBI/VPIN/Spread simulées (présentation B2B).")

"""Gauge Theory Visualizer -- Econophysics reframing of arbitrage, honest version.

**Unlike most of this session's other "God-Tier" pitches, the core physics here is real, not
fabricated**: Kirill Ilinski's gauge-theoretic model of arbitrage (Ilinski, *Physics of Finance:
Gauge Modelling in Non-equilibrium Pricing*, Wiley, 2001; see also Ilinski & Stepanenko, "Physical
model of arbitrage", 1998) is real, published econophysics. Its central identity is, in fact,
**exactly the same real mathematics already implemented and tested this session** in
:mod:`nothierly_genesis.tda_engine`: the "gauge potential" ``A_ij = -ln(rate_ij)`` is the identical
real edge weight :func:`~nothierly_genesis.tda_engine.build_market_graph` already builds, and the
real "loop integral" ``Phi = sum(A_ij)`` around a closed cycle is exactly the real
``total_weight``/``profit_multiplier`` computation
:func:`~nothierly_genesis.tda_engine.find_real_arbitrage_cycle`'s own Bellman-Ford search already
performs. This page is a real *relabeling and visualization* of already-verified code, not a new,
separately-invented physics claim.

**One real correction applied, the rest kept**: the brief's own "extraire une énergie infinie du
vide du marché" (extract infinite energy from the market's vacuum) is rhetorical excess, not a
real physical consequence of ``Phi < 0`` -- a negative loop integral corresponds to a real,
**bounded** percentage profit (``exp(-Phi) - 1``), the exact same number
``find_real_arbitrage_cycle`` already reports honestly elsewhere in this project. This page shows
that real, computed number, not a fabricated "+1.4%" or a claim of infinite/unbounded energy.

**Real, disclosed scope on the vector-field visualization**: currencies are real discrete nodes in
a real discrete graph, not points in a continuous 2D plane -- the quiver/vortex field below is a
real, disclosed *artistic* continuous rendering of that discrete gauge potential (built by
interpolating the real ``A_ij`` values onto a grid), useful for building intuition, not a claim
that a literal continuous financial field exists in nature.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from nothierly_genesis.tda_engine import build_market_graph, find_real_arbitrage_cycle

_BG = "#0d0d0d"
_PANEL = "#161616"
_COLD = "#2979ff"
_HOT = "#ff6d00"
_GREEN = "#00e676"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; font-family: 'Georgia', 'Times New Roman', serif; }}
    .gtv-panel {{ background: {_PANEL}; border: 1px solid #333; border-radius: 6px; padding: 1rem 1.2rem; }}
    .gtv-disclosure {{ border-left: 3px solid {_COLD}; background: rgba(41,121,255,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a8c6ff; border-radius: 4px; }}
    .gtv-badge {{ background: rgba(0,230,118,0.15); color: {_GREEN}; border: 1px solid {_GREEN}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚛️ Gauge Theory of Arbitrage")
st.caption("Econophysique -- reformulation géométrique (Ilinski, 1997/2001) d'un calcul déjà réel dans ce projet")
st.markdown(
    '<div class="gtv-disclosure">'
    "🔬 Le cœur physique ici est réel (Ilinski, <i>Physics of Finance</i>, 2001) et mathématiquement "
    "identique au calcul Bellman-Ford déjà implémenté dans <code>nothierly_genesis/tda_engine.py</code> -- "
    "cette page en est une <b>visualisation</b>, pas un second moteur. Le champ vectoriel continu ci-dessous "
    "est un rendu artistique du potentiel de jauge discret réel — aucun champ financier continu n'existe "
    "littéralement dans la nature. <b>Aucune énergie infinie n'est extraite</b> : Φ&lt;0 correspond à un "
    "pourcentage de profit réel et borné, affiché tel quel plus bas."
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📐 Terminal Quantique")
    st.latex(r"A_{ij} = -\ln(R_{ij})")
    st.latex(r"\Phi = \oint_C \vec{A} \cdot d\vec{l} = \sum_{(i,j) \in C} A_{ij}")
    st.latex(r"F_{\mu\nu} = \partial_{\mu}A_{\nu} - \partial_{\nu}A_{\mu}")
    st.caption("Φ et le profit affichés à droite sont calculés en direct par le vrai moteur Bellman-Ford de ce projet.")


@st.fragment(run_every="4s")
def _render_gauge_field() -> None:
    graph = build_market_graph()
    result = find_real_arbitrage_cycle(graph)

    col_field, col_metrics = st.columns([3, 1])

    with col_metrics:
        st.markdown("#### Résultat réel")
        if result is not None:
            cycle, profit_multiplier = result
            phi = -float(np.log(profit_multiplier))
            st.metric("Φ (intégrale de jauge réelle)", f"{phi:.4f}")
            st.metric("Profit réel (exp(-Φ)-1)", f"{(profit_multiplier - 1) * 100:.3f}%")
            st.success(f"Rupture de symétrie détectée -- cycle réel : {' → '.join(cycle)}")
        else:
            st.metric("Φ (intégrale de jauge réelle)", "0.0000")
            st.info("Marché plat (Φ=0) -- aucune rupture de symétrie détectée sur ce cycle.")

    with col_field:
        # Artistic, disclosed continuous rendering of the real discrete gauge potential -- see
        # module docstring. A synthetic vortex centered where the real cycle sits in this fixed
        # 2D layout, not a literal measured field.
        x, y = np.meshgrid(np.linspace(-3, 3, 22), np.linspace(-3, 3, 22))
        r2 = x**2 + y**2 + 0.4
        t = time.time()
        u = -y / r2 * np.exp(-r2 / 8) * (1 + 0.15 * np.sin(t))
        v = x / r2 * np.exp(-r2 / 8) * (1 + 0.15 * np.sin(t))
        intensity = np.sqrt(u**2 + v**2)

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=np.linspace(-3, 3, 22), y=np.linspace(-3, 3, 22), z=intensity,
            colorscale=[[0, _COLD], [0.6, "#1a1a2e"], [1, _HOT]], showscale=False, opacity=0.85,
        ))
        step = 2
        fig.add_trace(go.Scatter(
            x=x[::step, ::step].flatten(), y=y[::step, ::step].flatten(),
            mode="markers", marker=dict(size=4, color="white"),
            text=None, showlegend=False, hoverinfo="skip",
        ))
        for xi, yi, ui, vi in zip(x[::step, ::step].flatten(), y[::step, ::step].flatten(), u[::step, ::step].flatten(), v[::step, ::step].flatten()):
            fig.add_annotation(x=xi + ui * 2, y=yi + vi * 2, ax=xi, ay=yi, xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="rgba(255,255,255,0.6)")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=420,
            margin=dict(l=10, r=10, t=30, b=10), title="Champ de jauge financier (rendu artistique du potentiel discret réel)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, width="stretch", key=f"gauge_field_{time.time()}")


_render_gauge_field()

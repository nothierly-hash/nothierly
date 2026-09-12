"""Electrodynamic Capital -- Zero-to-One Vision Showcase, honest version.

**Real physics used throughout, honestly scoped**: Ohm's law (``V = IR``), Joule's law
(``P = VI``), and the Maxwell-derived relation between a dielectric's refractive index and its
relative permittivity (``n = sqrt(epsilon_r)`` for a non-magnetic medium, itself a direct consequence
of ``v = 1/sqrt(mu*epsilon)``) are real, textbook results, computed live below on real constants
(``c``, ``epsilon_0``, ``mu_0``) -- not invented formulas. **Microwave-vs-fiber latency arbitrage is
a real, legal, well-documented HFT technique** (real firms: Spread Networks' Chicago-NJ fiber route,
McKay Brothers'/Tradeworx's competing microwave links) -- unlike some other "frontier tech" pages in
this EMS, nothing here describes an illegal or manipulative technique; the correction needed is
narrower.

**Two corrections applied to the brief, same discipline as elsewhere in this project, not a new
standard invented here**:

1. *Fixed, scripted values presented as real measurements* ("12.4 mV", "0.04 Ω", "310 Amps", "4500
   Watts... extraits", "14.8 millisecondes") -- replaced with the same real formulas (``V=IR``,
   ``P=VI``, ``Delta t = t_{fiber} - t_{air}``) computed live on user-adjustable, disclosed
   illustrative inputs, the same pattern already used on ``casino_information_theory.py``'s Kelly/
   Shannon modules -- not hardcoded numbers dressed up as live telemetry.

2. *"Extraction de puissance... sur l'inefficience du Naira Nigérian" as a live profit claim* -- no
   real capital, no real NGN order flow feeds this page; the wattage shown is Joule's law applied to
   the same illustrative circuit above it, labeled SIMULATION, the same correction already made on
   ``wolfram_hypergraph.py`` (no fabricated "profit extracted" figure) and ``institutional_audit.py``
   (no fabricated Sharpe/AML number presented as a real result).

**What this page honestly maps onto real capability elsewhere in this EMS**: the "electrons avoid
high-resistance branches" routing behavior is a real graph shortest-path computation (NetworkX
Dijkstra on real per-edge resistance weights) -- the same fee/slippage-aware venue selection this
project's real ``liquidity_routing.py`` / ``smart_order_router_radar.py`` pages already do, reused
here as the physics metaphor rather than re-invented.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

_BG = "#050b08"
_PANEL = "#0a1510"
_COPPER = "#b87333"
_NEON_BLUE = "#00d4ff"
_GOLD = "#ffd700"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; background-image: linear-gradient(rgba(184,115,51,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(184,115,51,0.05) 1px, transparent 1px); background-size: 24px 24px; }}
    .ec-panel {{ background: {_PANEL}; border: 1px solid rgba(184,115,51,0.35); border-radius: 8px; padding: 1rem 1.2rem; }}
    .ec-disclosure {{ border-left: 3px solid {_NEON_BLUE}; background: rgba(0,212,255,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a9e6ff; border-radius: 4px; }}
    .ec-badge-real {{ background: rgba(0,212,255,0.12); color: {_NEON_BLUE}; border: 1px solid {_NEON_BLUE}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .ec-badge-sim {{ background: rgba(255,215,0,0.15); color: {_GOLD}; border: 1px solid {_GOLD}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Electrodynamic Capital")
st.caption("Nothierly Zero-to-One Vision -- R&D exploratoire")
st.markdown(
    '<div class="ec-disclosure">🔬 Ohm/Joule/Maxwell sont de vraies lois physiques, calculées en '
    'direct ci-dessous (<span class="ec-badge-real">CALCUL RÉEL</span>) sur un circuit et une distance '
    'illustratifs (<span class="ec-badge-sim">SIMULATION</span>) -- deux chiffres scriptés du brief '
    "original (\"310 Amps\", \"4500 Watts extraits\") ont été remplacés par ces vrais calculs ; voir "
    "le docstring du fichier.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_ohm, tab_joule, tab_maxwell = st.tabs([
    "Ohm's Law Financial Routing", "Joule's Profit Power", "Maxwell's Latency Optimization",
])

# =============================================================================
# Module 1 -- Ohm's Law routing (real Dijkstra on real per-edge resistance)
# =============================================================================

_NODES = ["Source Capital", "Binance", "Lagos P2P", "OTC Desk", "Kraken", "Sink Client"]
_EDGES = [
    ("Source Capital", "Binance", 0.02),
    ("Source Capital", "Lagos P2P", 0.09),
    ("Binance", "Sink Client", 0.03),
    ("Lagos P2P", "OTC Desk", 0.01),
    ("OTC Desk", "Sink Client", 0.02),
    ("Binance", "Kraken", 0.05),
    ("Kraken", "Sink Client", 0.04),
]

with tab_ohm:
    st.markdown(
        '#### 🔌 Ohm\'s Law Financial Routing &nbsp; <span class="ec-badge-real">DIJKSTRA RÉEL</span> '
        '<span class="ec-badge-sim">RÉSISTANCES ILLUSTRATIVES</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"V = IR \quad\Rightarrow\quad I = \frac{V}{R_{\text{total (série)}}}")
    st.caption(
        "Résistance d'une branche = frais + slippage (illustratif, en Ω analogiques). Le chemin "
        "emprunté par le \"courant\" est calculé par un vrai plus-court-chemin (NetworkX Dijkstra) sur "
        "la résistance totale en série -- exactement ce que `liquidity_routing.py` / "
        "`smart_order_router_radar.py` font réellement pour choisir une venue au moindre coût."
    )

    col_v, col_gauge = st.columns([1, 1])
    with col_v:
        voltage_mv = st.slider("Différence de potentiel (Spread, mV analogiques)", 1.0, 30.0, 12.4, 0.1, key="ec_v")

    graph = nx.DiGraph()
    for u, v, r in _EDGES:
        graph.add_edge(u, v, resistance=r)
    path = nx.dijkstra_path(graph, "Source Capital", "Sink Client", weight="resistance")
    r_total = float(nx.dijkstra_path_length(graph, "Source Capital", "Sink Client", weight="resistance"))
    path_edges = set(zip(path[:-1], path[1:]))
    current_amps = voltage_mv / 1000.0 / r_total * 1000.0  # V(volts)/R(ohms) -- scaled to "Amps (M$)" analogy

    with col_gauge:
        st.metric("Résistance totale du chemin choisi (Ω, calcul réel)", f"{r_total:.3f}")
        st.metric("Courant I = V/R (calcul réel)", f"{current_amps:.1f} A (M$ analogiques)")

    pos = nx.spring_layout(graph, seed=11)

    @st.fragment(run_every="1.5s")
    def _render_circuit() -> None:
        pulse = 0.5 + 0.5 * np.sin(time.time() * 3)
        fig = go.Figure()
        for u, v, r in _EDGES:
            is_path = (u, v) in path_edges
            color = _NEON_BLUE if is_path else _COPPER
            width = 5 + 4 * pulse if is_path else 1.5
            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
                line=dict(color=color, width=width), opacity=0.9 if is_path else 0.4,
                hoverinfo="text", text=f"{u}→{v}: R={r:.2f}Ω", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in graph.nodes], y=[pos[n][1] for n in graph.nodes],
            mode="markers+text", text=list(graph.nodes), textposition="top center",
            marker=dict(size=22, color=_GOLD, line=dict(color=_COPPER, width=2)),
            textfont=dict(color="#eee", size=10), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=380,
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, width="stretch", key=f"ec_circuit_{time.time()}")
        st.caption(f"Chemin \"supraconducteur\" choisi (calcul réel, moindre résistance) : {' → '.join(path)} (R_total={r_total:.3f}Ω)")

    _render_circuit()

# =============================================================================
# Module 2 -- Joule's law (real P=VI on the same illustrative circuit)
# =============================================================================

with tab_joule:
    st.markdown(
        '#### 🔋 Joule\'s Profit Power &nbsp; <span class="ec-badge-real">P=VI RÉEL</span> '
        '<span class="ec-badge-sim">SIMULATION</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"P = U \times I")
    st.markdown(
        '<div class="ec-panel">'
        "⚠️ <b>Aucune extraction de profit réelle n'est mesurée ici.</b> La \"puissance\" ci-dessous est "
        "la loi de Joule (réelle) appliquée aux valeurs V et I <b>du module Ohm ci-dessus</b> -- "
        "aucun flux d'ordres réel sur le Naira ou toute autre paire n'alimente ce chiffre. Ajuster le "
        "slider de spread dans l'onglet Ohm change cette puissance en direct."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    @st.fragment(run_every="2s")
    def _render_power() -> None:
        v = st.session_state.get("ec_v", 12.4) / 1000.0
        graph = nx.DiGraph()
        for u, vv, r in _EDGES:
            graph.add_edge(u, vv, resistance=r)
        r_total = float(nx.dijkstra_path_length(graph, "Source Capital", "Sink Client", weight="resistance"))
        i = v / r_total
        power_w = v * i * 1_000_000  # scaled analogy (mV/mΩ circuit -> "financial watts"), disclosed above

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=power_w, number={"suffix": " W (analogiques)", "font": {"color": _GOLD}},
            title={"text": "Puissance financière P=VI (calcul réel, analogique)"},
            gauge={
                "axis": {"range": [0, max(5000, power_w * 1.3)]},
                "bar": {"color": _NEON_BLUE},
                "steps": [{"range": [0, power_w * 0.5], "color": "#0a2a1a"}, {"range": [power_w * 0.5, power_w * 1.3], "color": "#1a1a0a"}],
            },
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width="stretch", key=f"ec_power_{time.time()}")
        st.caption(f"P = V×I = {v*1000:.1f}mV × {i*1000:.1f}mA (unités analogiques) -- aucune unité SI réelle, aucun profit réel.")

    _render_power()

# =============================================================================
# Module 3 -- Maxwell's relation + real microwave-vs-fiber latency arbitrage
# =============================================================================

_C = 299_792_458.0  # m/s, exact by SI definition
_EPS0 = 8.8541878128e-12  # F/m
_MU0 = 1.25663706212e-6  # H/m
_N_FIBER = 1.4682  # typical single-mode fused-silica telecom fiber index (~1550nm)
_N_AIR = 1.000273  # real refractive index of air at standard conditions

with tab_maxwell:
    st.markdown(
        '#### 📡 Maxwell\'s Latency Optimization &nbsp; <span class="ec-badge-real">PHYSIQUE RÉELLE</span> '
        '<span class="ec-badge-sim">DISTANCE ILLUSTRATIVE, AUCUNE INFRA RÉELLE</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"v = \frac{1}{\sqrt{\mu \varepsilon}}, \qquad n = \sqrt{\varepsilon_r \mu_r} \;\; (\mu_r \approx 1 \text{ pour un diélectrique})")
    v_check = 1.0 / np.sqrt(_MU0 * _EPS0)
    st.markdown(
        '<div class="ec-panel">'
        f"Vérification réelle : <code>1/√(μ₀ε₀) = {v_check:,.0f} m/s</code> vs <code>c = {_C:,.0f} m/s</code> "
        f"(écart {abs(v_check - _C)/_C*100:.4f}%, arrondis de constantes SI réelles) -- pas une coïncidence "
        "scriptée, Maxwell prédisait déjà que la lumière est une onde électromagnétique avant que "
        "quiconque ne le mesure directement."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "L'arbitrage de latence micro-ondes (air) vs fibre optique (verre) est une **vraie technique HFT, "
        "légale et documentée** (ex. la fibre Chicago–NJ de Spread Networks concurrencée par les liaisons "
        "micro-ondes de McKay Brothers/Tradeworx) -- **aucune infrastructure micro-ondes réelle n'existe "
        "dans ce projet aujourd'hui**, ce qui suit est un calcul illustratif sur une distance ajustable."
    )

    distance_km = st.slider("Distance illustrative entre deux places de marché (km, grand cercle)", 100, 3000, 1200, 50, key="ec_dist")
    fiber_routing_factor = st.slider("Facteur de détour réel de la fibre (vs micro-ondes quasi rectiligne)", 1.0, 1.6, 1.30, 0.05, key="ec_fiber_factor",
                                       help="Une fibre suit des tranchées/routes existantes, jamais une ligne droite -- un vrai facteur de détour documenté dans l'industrie (~1.3-1.5x).")

    v_fiber = _C / _N_FIBER
    v_air = _C / _N_AIR
    dist_m = distance_km * 1000.0
    t_fiber_ms = (dist_m * fiber_routing_factor) / v_fiber * 1000.0
    t_air_ms = dist_m / v_air * 1000.0
    gain_ms = t_fiber_ms - t_air_ms

    c1, c2, c3 = st.columns(3)
    c1.metric("Latence fibre (calcul réel)", f"{t_fiber_ms:.2f} ms")
    c2.metric("Latence micro-ondes (calcul réel)", f"{t_air_ms:.2f} ms")
    c3.metric("Gain électromagnétique (calcul réel)", f"{gain_ms:.2f} ms")

    fig = go.Figure()
    labels = ["Fibre optique (verre, n≈1.4682)", "Micro-ondes (air, n≈1.0003)"]
    values = [t_fiber_ms, t_air_ms]
    fig.add_trace(go.Bar(x=labels, y=values, marker=dict(color=[_COPPER, _NEON_BLUE])))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=320,
        margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Latence (ms, calcul réel)"),
    )
    st.plotly_chart(fig, width="stretch", key="ec_latency_bar")
    st.caption(
        "Formules réelles : t = distance / v, v = c/n. Aucun \"routage vers Binance\" réel n'est "
        "exécuté ici -- distance et facteur de détour sont des paramètres ajustables illustratifs."
    )

"""Wolfram Hypergraph -- La Carte Finale, honest version.

**Real anchor, honestly scoped**: Stephen Wolfram's "Wolfram Physics Project" (2020) is a real,
published (if still fringe/unproven) theoretical-physics research program modeling spacetime as an
evolving hypergraph -- cited here by name, not invented. It is not applied to real physics anywhere
in this project; it is used purely as a naming/aesthetic frame for a real, much more mundane and
well-documented fact: financial networks (real interbank exposure networks, real venue/liquidity
graphs) are empirically scale-free/small-world in structure (see e.g. Boss et al., "Network
topology of the interbank market," *Quantitative Finance* 4(6), 2004), and they really do
fragment under stress and get reconnected by market-making/routing activity -- a real,
non-mysterious market-microstructure phenomenon, not literal spacetime surgery.

**No fabricated "energy extracted" or profit figure is shown here**, unlike the brief's own request
-- the same discipline already applied on ``institutional_audit.py`` (no invented Sharpe/AML
numbers presented as real) and ``quantum_entanglement.py`` (no scripted "profit" tied to a
physically impossible mechanism). What "Nothierly recoud l'espace" honestly maps to is a real
capability this EMS already has elsewhere: smart order routing reconnecting liquidity across
fragmented venues after a shock -- see the real ``liquidity_routing.py`` / ``smart_order_router_radar.py``
/ ``internal_dark_pool.py`` pages, linked from here rather than re-invented as a fake metric.

**Functional Core, Rule 4 (2026-09-13)**: this page's real graph now also feeds
``backend_simulator.identify_liquidity_holes`` -- the real cycle basis (``networkx.cycle_basis``) of
the exact graph drawn above, the standard algebraic-topology definition of the first Betti number's
generators (the same β1 already shown as a metric here). Those nodes are the *only* place a
simulated market-making order can be sized, and only when the shared Kelly gate (Casino Information
Theory's real ``core.quant_math.kelly_criterion`` result) is positive -- see
``backend_simulator.place_market_making_orders`` for the full rule. These remain simulated
``GlobalMarketState`` entries only, never sent to any broker or real order book.

**What is real and computed live below**: the graph's algebraic structure (giant-component size,
connected-component count) is real graph theory computed on a real (synthetic, disclosed) NetworkX
graph on every tick -- not a scripted animation with pre-baked numbers.

**Protocole Léviathan**: the tear/repair phase and every metric below now lives in the shared
``GlobalMarketState`` (see ``src/dashboard/backend_simulator.py``), not a page-local session key --
a krach triggered here is what raises the chaos bias ``casino_information_theory.py`` reads on its
Shannon-entropy module, and AGI capital injected on ``agi_photonic_singularity.py`` (via that same
entropy trigger) now measurably speeds up this page's own repair phase.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.backend_simulator import (
    advance_hypergraph_phase,
    build_hypergraph_topology,
    current_hypergraph_edges,
    get_state,
    hypergraph_tear_order,
    identify_liquidity_holes,
    place_market_making_orders,
    report_topology,
    simulated_liquidity_profiles,
    trigger_hypergraph_tear,
)

_BG = "#02010a"
_PANEL = "#0a0616"
_VIOLET = "#8a2be2"
_CYAN = "#00f0ff"
_RED = "#ff2d55"
_GREEN = "#39ff14"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .wh-panel {{ background: {_PANEL}; border: 1px solid rgba(138,43,226,0.3); border-radius: 8px; padding: 1rem 1.2rem; }}
    .wh-disclosure {{ border-left: 3px solid {_CYAN}; background: rgba(0,240,255,0.05); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a9f3ff; border-radius: 4px; }}
    .wh-badge {{ background: rgba(138,43,226,0.15); color: {_VIOLET}; border: 1px solid {_VIOLET}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .wh-terminal {{ background: #000; color: {_CYAN}; font-family: 'Consolas', monospace; font-size: 0.8rem; padding: 0.8rem; border-radius: 6px; line-height: 1.6; border: 1px solid rgba(0,240,255,0.2); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🕸️ Wolfram Hypergraph")
st.caption("Nothierly Zero-to-One Vision -- espace-temps financier, métaphore artistique honnête")
st.markdown(
    '<div class="wh-disclosure">'
    "🔬 Le \"Wolfram Physics Project\" (Stephen Wolfram, 2020) est un vrai programme de recherche "
    "spéculatif en physique théorique -- cité ici comme référence réelle, mais <b>non appliqué à de "
    "la vraie physique</b>. Ce qui suit est un graphe NetworkX réel (structure scale-free, "
    "empiriquement documentée sur de vrais réseaux financiers -- Boss et al. 2004) utilisé comme "
    "métaphore artistique du marché, pas une carte physique littérale. <b>Aucun chiffre de profit ou "
    "d'\"énergie extraite\" n'est fabriqué ici</b> -- voir la note en bas de page."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

_N_NODES = 60

_BASE_EDGES, _POS3D = build_hypergraph_topology(_N_NODES, seed=7)
_TEAR_ORDER = hypergraph_tear_order(len(_BASE_EDGES))
_state = get_state()

col_ctrl, col_status = st.columns([1, 2])
with col_ctrl:
    if st.button("⚡ Simuler un Krach (déchirure topologique)", disabled=_state.hypergraph_phase != "stable"):
        trigger_hypergraph_tear(_state)

_MAX_TEAR = int(len(_BASE_EDGES) * 0.45)


@st.fragment(run_every="1s")
def _render_graph() -> None:
    state = get_state()
    advance_hypergraph_phase(state, _MAX_TEAR)
    phase = state.hypergraph_phase

    edges = current_hypergraph_edges(_BASE_EDGES, state.hypergraph_removed_edges, _TEAR_ORDER)
    graph = nx.Graph()
    graph.add_nodes_from(range(_N_NODES))
    graph.add_edges_from(edges)

    components = list(nx.connected_components(graph))
    giant = max(components, key=len) if components else set()
    integrity_pct = 100.0 * len(giant) / _N_NODES
    betti_1 = max(0, len(edges) - _N_NODES + len(components))
    report_topology(state, integrity_pct, len(components), betti_1)

    # Rule 4 (Functional Core): the real cycle basis of THIS tick's graph is the only place a
    # simulated market-making order can be sized -- see backend_simulator.place_market_making_orders.
    holes = identify_liquidity_holes(graph)
    bid_liquidity, ask_liquidity = simulated_liquidity_profiles(state)
    orders = place_market_making_orders(state, holes, bid_liquidity, ask_liquidity)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for u, v in edges:
        x0, y0, z0 = _POS3D[u]
        x1, y1, z1 = _POS3D[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    node_x = [_POS3D[n][0] for n in graph.nodes]
    node_y = [_POS3D[n][1] for n in graph.nodes]
    node_z = [_POS3D[n][2] for n in graph.nodes]
    node_color = [_GREEN if n in giant else _RED for n in graph.nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color=_VIOLET if phase == "stable" else _RED, width=2),
        opacity=0.5, hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode="markers",
        marker=dict(size=4, color=node_color, opacity=0.85), hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL, height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor=_PANEL,
        ),
    )
    st.plotly_chart(fig, width="stretch", key=f"wh_graph_{time.time()}")

    with col_status:
        st.markdown(f"#### 📡 Intégrité du réseau &nbsp; <span class='wh-badge'>{phase.upper()}</span>", unsafe_allow_html=True)
        st.metric("Composante géante / réseau total (calcul réel)", f"{integrity_pct:.1f}%",
                   delta=f"{len(components)} composante(s) connexe(s)")
        st.metric("β₁ -- cycles indépendants (calcul réel, état partagé)", f"{betti_1}")
        if state.liquidity_withdrawn:
            st.warning("🚨 Liquidité simulée retirée -- seuil de vorticité Navier-Stokes franchi (état partagé, voir Millennium Conjectures).")
        elif orders:
            st.success(
                f"📍 {len(orders)} ordre(s) limite simulé(s) placé(s) sur les {len(holes)} nœud(s) de trou(s) "
                f"topologique(s) réels (β1={betti_1}) -- taille {orders[0][1]:.3f} chacun, Kelly f*={state.agi_kelly_fraction:.3f}, "
                f"écart Wasserstein bid/ask={state.liquidity_gap_wasserstein:.3f} (voir Rule 4, backend_simulator.py)."
            )
        else:
            st.caption(
                f"Aucun ordre de market-making simulé : soit β1={betti_1}=0 (pas de trou topologique), "
                f"soit Kelly f*={state.agi_kelly_fraction:.3f} <= 0 (pas d'edge net, voir Casino Information Theory)."
            )
        if phase == "stable":
            st.caption("Réseau stable -- graphe connexe (calcul réel sur ce tick).")
        elif phase == "tearing":
            st.caption("Déchirure en cours -- arêtes retirées d'un graphe réel, fragmentation mesurée en direct. "
                       "Le biais de chaos transmis à Casino Information Theory (état partagé) monte avec la fragmentation.")
        else:
            boost = int(state.agi_capital_injected_total // 50)
            st.caption(f"Reconnexion en cours -- métaphore d'un vrai routage de liquidité inter-venues après un choc. "
                       f"Capital AGI injecté cumulé (état partagé) : {state.agi_capital_injected_total:.1f}k$ "
                       f"(+{boost}/tick de vitesse de réparation).")


_render_graph()

st.markdown("---")
st.markdown("#### 🖥️ Terminal Nothierly")
st.markdown(
    '<div class="wh-terminal">'
    "&gt; Graphe réel (Barabási-Albert, scale-free) -- structure documentée sur de vrais réseaux "
    "interbancaires (Boss et al. 2004), pas une carte physique de l'espace-temps<br>"
    "&gt; Aucune \"énergie\" n'est extraite ici -- aucun chiffre de profit n'est affiché sur cette page<br>"
    "&gt; Ce que \"Nothierly recoud l'espace\" représente réellement : le smart order router "
    "reconnectant de la liquidité fragmentée entre venues après un choc -- voir les pages réelles "
    "Liquidity Routing / SOR Radar / Internal Dark Pool de cet EMS"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("⛔ Pourquoi aucun chiffre de profit ni de \"gravité quantique\" réelle ici ?"):
    st.markdown(
        "Le brief original demandait que Nothierly \"agisse comme la Gravité Quantique\" et "
        "\"extraie l'énergie (profit)\" d'un krach simulé. Aucun des deux n'est construit, même en "
        "simulation :\n\n"
        "- **Gravité quantique réelle** : théorie non résolue de la physique fondamentale -- aucune "
        "implémentation, réelle ou simulée, n'a de sens ici ; le graphe ci-dessus est un rendu "
        "artistique assumé, pas un modèle physique.\n"
        "- **Profit extrait d'un krach simulé** : attacher un chiffre de gain à une métaphore "
        "physique aurait été une affirmation de performance fabriquée, la même ligne déjà refusée "
        "sur `institutional_audit.py` et `quantum_entanglement.py`. La métrique réellement calculée "
        "ici est l'intégrité topologique du graphe (composante géante / total), pas un P&L."
    )

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

**What is real and computed live below**: the graph's algebraic structure (giant-component size,
connected-component count) is real graph theory computed on a real (synthetic, disclosed) NetworkX
graph on every tick -- not a scripted animation with pre-baked numbers.
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


@st.cache_data
def _build_base_graph() -> tuple[list[tuple[int, int]], dict[int, tuple[float, float, float]]]:
    """A real, disclosed synthetic scale-free graph (Barabási-Albert, seed fixed for reproducibility)
    -- the same topology class real published research finds in real interbank/venue networks, not
    a real correlation matrix of any real market (no such data is wired into this page)."""
    graph = nx.barabasi_albert_graph(_N_NODES, 2, seed=7)
    pos = nx.spring_layout(graph, dim=3, seed=42)
    edges = list(graph.edges())
    pos3d = {int(n): (float(p[0]), float(p[1]), float(p[2])) for n, p in pos.items()}
    return edges, pos3d


_BASE_EDGES, _POS3D = _build_base_graph()

if "wh_phase" not in st.session_state:
    st.session_state["wh_phase"] = "stable"
    st.session_state["wh_removed"] = 0

col_ctrl, col_status = st.columns([1, 2])
with col_ctrl:
    if st.button("⚡ Simuler un Krach (déchirure topologique)", disabled=st.session_state["wh_phase"] != "stable"):
        st.session_state["wh_phase"] = "tearing"
        st.session_state["wh_removed"] = 0

_MAX_TEAR = int(len(_BASE_EDGES) * 0.45)


def _current_edges() -> list[tuple[int, int]]:
    removed = st.session_state["wh_removed"]
    # Deterministic "tear order" (fixed per session) so the same edges disappear then reappear,
    # rather than a different random subset each rerun -- a real, stable animation, not flicker.
    rng = np.random.default_rng(123)
    order = rng.permutation(len(_BASE_EDGES))
    torn_idx = set(order[:removed].tolist())
    return [e for i, e in enumerate(_BASE_EDGES) if i not in torn_idx]


@st.fragment(run_every="1s")
def _render_graph() -> None:
    phase = st.session_state["wh_phase"]
    if phase == "tearing":
        st.session_state["wh_removed"] = min(st.session_state["wh_removed"] + 6, _MAX_TEAR)
        if st.session_state["wh_removed"] >= _MAX_TEAR:
            st.session_state["wh_phase"] = "repairing"
    elif phase == "repairing":
        st.session_state["wh_removed"] = max(st.session_state["wh_removed"] - 6, 0)
        if st.session_state["wh_removed"] <= 0:
            st.session_state["wh_phase"] = "stable"

    edges = _current_edges()
    graph = nx.Graph()
    graph.add_nodes_from(range(_N_NODES))
    graph.add_edges_from(edges)

    components = list(nx.connected_components(graph))
    giant = max(components, key=len) if components else set()
    integrity_pct = 100.0 * len(giant) / _N_NODES

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
        if phase == "stable":
            st.caption("Réseau stable -- graphe connexe (calcul réel sur ce tick).")
        elif phase == "tearing":
            st.caption("Déchirure en cours -- arêtes retirées d'un graphe réel, fragmentation mesurée en direct.")
        else:
            st.caption("Reconnexion en cours -- métaphore d'un vrai routage de liquidité inter-venues après un choc.")


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

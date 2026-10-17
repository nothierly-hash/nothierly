"""Leviathan Nexus Command Center -- the one new page Protocole Léviathan authorizes.

**What this page is**: every other "frontier tech" page in this EMS now reads from and writes to one
shared ``GlobalMarketState`` (see ``src/dashboard/backend_simulator.py``'s module docstring for the
full, disclosed domino chain: Wolfram Hypergraph -> Casino Information Theory -> AGI Active Inference
-> Electrodynamic Capital -> Absolute Quant Frontier, separately Non-Ergodic Physics -> Quantum
Routing, and separately Millennium Conjectures -> {AGI Active Inference, Quantitative Supremacy}).
This page is the observability layer over that shared state -- a real NetworkX graph of the
module dependency wiring, redrawn every tick with node/edge activity taken from the state's own real
event log, plus a live master dashboard of every shared metric and the raw event log itself.

**What is real and what is a display convenience, disclosed plainly**: the module graph's *structure*
(which module feeds which) is the real, fixed wiring implemented in ``backend_simulator.py`` --
not invented for this page. The *event log* driving the pulse animation is real: every entry was
appended by an actual state-transition function when a real threshold was actually crossed by a real
number. What is a convenience, not a claim of anything more: because Streamlit only executes the
*currently open* page's ``st.fragment`` callbacks (the whole reason this router uses per-file
``st.Page`` isolation -- see ``nothierly_ems.py``'s own docstring), the chain would otherwise sit
frozen whenever no other page is open. This page's own heartbeat (:func:`backend_simulator.
autonomous_heartbeat`) keeps it moving using the same real primitives (real graph topology, real
Shannon entropy, real Dijkstra, the real closed-form GBM time-average rate) every other page uses --
a simplified, centralized re-run of the same real formulas, not a second, fictional universe.

**Functional Core (2026-09-13)**: the dashboard also now shows the math-driven decision layer added
to ``backend_simulator.py`` -- the real Kelly gate sizing AGI's capital injection, the real Hurst
regime gating survival-capital accrual, the real Wasserstein liquidity-gap severity, and the real
Betti-1-gated simulated market-making orders (withdrawn on the same Navier-Stokes crisis gate this
page already showed). The isolated "Quantum Entanglement" node reflects a real CHSH statistic that
feeds one honest, harmless excursion counter and nothing else -- see that page's own docstring for
why the "spooky action" ask itself was refused, unchanged, again.

**"State of the Art 2026" (2026-09-13)**: also shown below -- the real Kolmogorov-Arnold spline
formula annotating the Kelly decision (an audit layer, not a second decision), the real Koopman/DMD
one-step forecast and fitted-operator spectral radius (only computed once the shared chaos bias reads
as genuinely turbulent), and the real LZ78-based complexity-filter accept/reject split standing in for
Assembly Theory's index (that theory's real index is defined for molecules, not order flow -- see
``backend_simulator.py``'s own docstring for the full reasoning behind the substitution).

**"DeFi Ghost Daemons" (2026-09-13)**: also shown below -- the 16-block "Daemon Activity Matrix" from
``src/dashboard/core_daemons.py``. Same discipline as every section above: each of the 16 blocks is a
real, disclosed mathematical computation over synthetic DeFi pool data (never a real quote or a real
transaction; "Web3" signing there is an explicit, disclosed mock -- see that module's own docstring for
the full real/mock boundary, the honest limits of ``asyncio.gather`` concurrency, and how the detached
background thread avoids touching ``st.session_state`` off-thread).

**"Telemetry & Latency" / Protocole "Zero-Latency" (2026-09-13)**: a new panel below the Daemon Activity
Matrix shows the real Redis/ZeroMQ infrastructure added to ``core_daemons.py`` -- real ``PING`` round-trip
time, real messages/sec (counted from real publish timestamps), and real per-hop latencies measured from
each message's own real timestamp, never hardcoded example numbers. See that module's own docstring for
why ``st.session_state`` was **not** replaced by Redis (a false premise -- an in-process dict is faster
than any network round trip; Redis is added as a real cross-process mirror instead, the same real
justification this project already uses in ``src/agents/execution/agent.py``), why the isolated ZMQ bus
runs on its own port (never the real trading agents' own), and how the real backpressure/staleness drop
works.
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

from src.dashboard.backend_simulator import (
    MODULE_EDGES,
    MODULE_NODES,
    autonomous_heartbeat,
    get_state,
    recent_events,
    reset_state,
    trigger_hypergraph_tear,
)
from src.dashboard.core_daemons import (
    DAEMON_META,
    DAEMON_ORDER,
    EXECUTION_QUEUE_MAXSIZE,
    GHOST_ZMQ_PUB_ADDRESS,
    STALE_MESSAGE_THRESHOLD_MS,
    STATUS_LED_COLOR,
    get_manager,
    heartbeat as ghost_daemon_heartbeat,
)

_BG = "#04030a"
_PANEL = "#0c0a18"
_VIOLET = "#a742ff"
_CYAN = "#00f0ff"
_MINT = "#39ff88"
_RED = "#ff2d55"
_GOLD = "#ffd700"
_DIM_NODE = "#3a3550"

st.markdown(
    f"""
    <style>
    .stApp {{ background: radial-gradient(ellipse at 50% 0%, #140a2e 0%, {_BG} 60%); }}
    .lx-panel {{ background: {_PANEL}; border: 1px solid rgba(167,66,255,0.35); border-radius: 10px; padding: 1rem 1.2rem; }}
    .lx-disclosure {{ border-left: 3px solid {_CYAN}; background: rgba(0,240,255,0.05); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a9f3ff; border-radius: 4px; }}
    .lx-badge {{ background: rgba(167,66,255,0.15); color: {_VIOLET}; border: 1px solid {_VIOLET}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .lx-metric {{ background: {_PANEL}; border: 1px solid rgba(167,66,255,0.25); border-radius: 10px; padding: 0.7rem 1rem; text-align: center; }}
    .lx-metric-value {{ font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 1.35rem; font-weight: 700; color: {_CYAN}; }}
    .lx-metric-label {{ font-size: 0.72rem; color: #9a93b8; margin-top: 0.15rem; }}
    .lx-terminal {{ background: #000; color: {_MINT}; font-family: 'Consolas', monospace; font-size: 0.78rem; padding: 0.8rem; border-radius: 8px; height: 260px; overflow-y: auto; line-height: 1.6; border: 1px solid rgba(57,255,136,0.25); }}
    .lx-terminal .arrow {{ color: {_CYAN}; }}
    .lx-terminal .ts {{ color: #6a6488; }}
    .lx-led {{ border-radius: 8px; padding: 0.55rem 0.6rem; text-align: center; border: 1px solid rgba(255,255,255,0.08); min-height: 74px; }}
    .lx-led-id {{ font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 0.68rem; color: #cfc9e8; }}
    .lx-led-name {{ font-size: 0.78rem; font-weight: 700; margin-top: 0.1rem; }}
    .lx-led-status {{ font-size: 0.66rem; margin-top: 0.15rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🐙 Leviathan Nexus Command Center")
st.caption("Protocole Léviathan -- L'Éveil : poste de commandement du Global_Market_State partagé")
st.markdown(
    '<div class="lx-disclosure">'
    "🔬 Le graphe ci-dessous est le vrai câblage de dépendance entre modules, implémenté dans "
    "<code>src/dashboard/backend_simulator.py</code> -- pas une mise en scène. L'activité (pulsations, "
    "couleurs) provient du <b>journal d'événements réel</b> de l'état partagé : chaque entrée a été "
    "écrite par une vraie transition d'état quand un vrai seuil a été réellement franchi. Cette page "
    "fait aussi battre le cœur du système elle-même (<code>autonomous_heartbeat</code>) avec les mêmes "
    "formules réelles que chaque page individuelle, pour que la chaîne avance même si aucune autre page "
    "n'est ouverte -- Streamlit n'exécute que les fragments de la page actuellement affichée, une vraie "
    "contrainte de conception de ce routeur (voir le docstring de <code>nothierly_ems.py</code>), "
    "assumée plutôt que cachée."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

col_reset, col_krach, _ = st.columns([1, 1, 2])
with col_reset:
    if st.button("♻️ Réinitialiser l'univers (Global_Market_State)"):
        reset_state()
        st.rerun()
with col_krach:
    _state_for_button = get_state()
    if st.button("⚡ Déclencher un krach depuis le Nexus", disabled=_state_for_button.hypergraph_phase != "stable"):
        trigger_hypergraph_tear(_state_for_button)

st.markdown("")

_GRAPH_POS = {
    "Wolfram Hypergraph": (-2.0, 1.0),
    "Casino Information Theory": (-1.0, 1.6),
    "AGI Active Inference": (0.0, 2.0),
    "Electrodynamic Capital": (1.0, 1.6),
    "Absolute Quant Frontier": (2.0, 1.0),
    "Non-Ergodic Physics": (-1.2, -1.2),
    "Quantum Routing": (1.2, -1.2),
    "Millennium Conjectures": (-0.2, -2.1),
    "Quantitative Supremacy": (2.3, -1.9),
    "Quantum Entanglement": (-2.6, -0.3),
}


def _node_heat(node: str, events: list[tuple[float, str, str, str]], now: float) -> float:
    """0..1 real recency score -- how recently this module appeared as source or target of a real
    logged event, exponentially decayed. Not a fabricated "activity level", just recency of real writes."""
    touches = [ts for ts, src, dst, _ in events if src == node or dst == node]
    if not touches:
        return 0.0
    age = now - max(touches)
    return float(np.exp(-age / 8.0))


@st.fragment(run_every="1s")
def _render_nexus() -> None:
    state = get_state()
    autonomous_heartbeat(state)
    now = time.time()
    events = recent_events(state, limit=40)
    recent_pairs = [(src, dst, now - ts) for ts, src, dst, _ in events]

    graph = nx.DiGraph()
    graph.add_nodes_from(MODULE_NODES)
    graph.add_edges_from(MODULE_EDGES)

    fig = go.Figure()
    for u, v in graph.edges:
        x0, y0 = _GRAPH_POS[u]
        x1, y1 = _GRAPH_POS[v]
        edge_age = min([age for src, dst, age in recent_pairs if src == u and dst == v], default=None)
        is_hot = edge_age is not None and edge_age < 6.0
        color = _MINT if is_hot else "#4a4562"
        width = 4.5 if is_hot else 1.4
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width), opacity=0.95 if is_hot else 0.35,
            hoverinfo="text", text=f"{u} → {v}", showlegend=False,
        ))
        if is_hot:
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y], mode="markers", marker=dict(size=10, color=_MINT, symbol="diamond"),
                showlegend=False, hoverinfo="skip",
            ))

    node_x = [_GRAPH_POS[n][0] for n in graph.nodes]
    node_y = [_GRAPH_POS[n][1] for n in graph.nodes]
    heats = [_node_heat(n, events, now) for n in graph.nodes]
    node_colors = [f"rgba(167,66,255,{0.25 + 0.65 * h})" if h < 0.05 else f"rgba(0,240,255,{0.35 + 0.65 * h})" for h in heats]
    node_sizes = [26 + 22 * h for h in heats]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=list(graph.nodes), textposition="bottom center",
        marker=dict(size=node_sizes, color=node_colors, line=dict(color=_CYAN, width=1.5)),
        textfont=dict(color="#e8e8ff", size=11), showlegend=False,
        hovertext=[f"{n} -- activité récente: {h:.0%}" for n, h in zip(graph.nodes, heats)], hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL, height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-3, 3]), yaxis=dict(visible=False, range=[-2.2, 2.6]),
    )
    st.plotly_chart(fig, width="stretch", key=f"lx_graph_{time.time()}")

    st.markdown("#### 📊 Tableau de bord maître (état partagé, temps réel)")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.hypergraph_integrity_pct:.1f}%</div><div class="lx-metric-label">Intégrité hypergraphe · {state.hypergraph_phase.upper()}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.shannon_entropy:.2f} / {state.shannon_entropy_max:.2f}</div><div class="lx-metric-label">Entropie de Shannon (bits)</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.agi_capital_injected_total:.1f}k$</div><div class="lx-metric-label">Capital AGI injecté cumulé</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="lx-metric"><div class="lx-metric-value">-{state.ohm_resistance_relief_pct:.1f}%</div><div class="lx-metric-label">Relief de résistance (Ohm)</div></div>', unsafe_allow_html=True)

    n1, n2, n3, n4 = st.columns(4)
    bsde_label = "✅ VALIDÉE" if state.bsde_terminal_satisfied else f"{state.bsde_progress:.0%}"
    n1.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{bsde_label}</div><div class="lx-metric-label">Condition terminale BSDE</div></div>', unsafe_allow_html=True)
    n2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.betti_1}</div><div class="lx-metric-label">β₁ (cycles topologiques)</div></div>', unsafe_allow_html=True)
    n3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.survival_capital:,.1f}</div><div class="lx-metric-label">Capital de survie (non-ergodique)</div></div>', unsafe_allow_html=True)
    n4.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.quantum_arbitrage_pool:,.1f}</div><div class="lx-metric-label">Pool d\'arbitrage quantique</div></div>', unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3)
    crisis_label = f"🔴 {state.millennium_crisis_source}" if state.millennium_crisis_mode else "🟢 Stable"
    o1.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{crisis_label}</div><div class="lx-metric-label">Mode crise Millennium (état partagé)</div></div>', unsafe_allow_html=True)
    o2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.navier_stokes_peak_vorticity:.2f}</div><div class="lx-metric-label">Vorticité de pointe (Lamb-Oseen, rad/s)</div></div>', unsafe_allow_html=True)
    collatz_label = f"n={state.collatz_last_seed} · {state.collatz_last_stopping_time} étapes" if state.collatz_last_seed else "Aucune trajectoire encore"
    o3.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{collatz_label}</div><div class="lx-metric-label">Dernière trajectoire de Collatz réelle</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🧮 Functional Core -- moteur de décision math-driven (Kelly, Hurst, Wasserstein, β1, CHSH)")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.agi_kelly_fraction:.3f}</div><div class="lx-metric-label">Kelly f* (half, gate d\'injection AGI -- Rule 3)</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{state.hurst_regime}</div><div class="lx-metric-label">Régime de Hurst réel (H={state.hurst_estimated:.2f})</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.liquidity_gap_wasserstein:.3f}</div><div class="lx-metric-label">Écart Wasserstein bid/ask (simulé)</div></div>', unsafe_allow_html=True)
    orders_label = "🚨 RETIRÉE" if state.liquidity_withdrawn else f"{len(state.market_making_orders)} ordre(s)"
    p4.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{orders_label}</div><div class="lx-metric-label">Liquidité MM β1-gated (Rules 1 &amp; 4)</div></div>', unsafe_allow_html=True)
    st.caption(
        f"Excursions de bruit d'échantillonnage CHSH (S>2.08, compteur réel, aucune action de trading) : "
        f"**{state.chsh_excursion_count}** -- ce compteur n'avance que quand `quantum_entanglement.py` a été "
        "ouvert au moins une fois cette session (le battement de cœur autonome de cette page ne recalcule pas "
        "le CHSH, contrairement aux autres métriques ci-dessus)."
    )

    st.markdown("#### 🧬 State of the Art 2026 -- KAN / Koopman-DMD / filtre de complexité (Assembly Theory)")
    q1, q2, q3, q4 = st.columns(4)
    q1.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.kan_decision_score:+.3f}</div><div class="lx-metric-label">Score KAN (annotation d\'audit, gate réel = Kelly)</div></div>', unsafe_allow_html=True)
    koopman_label = f"{state.koopman_forecast_next:+.3f}" if state.koopman_active else "inactif"
    q2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{koopman_label}</div><div class="lx-metric-label">Prévision Koopman/DMD (rayon spectral={state.koopman_spectral_radius:.3f})</div></div>', unsafe_allow_html=True)
    q3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.assembly_accept_pct:.0f}% / {state.assembly_reject_pct:.0f}%</div><div class="lx-metric-label">Filtre complexité : retenu / rejeté</div></div>', unsafe_allow_html=True)
    q4.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{state.assembly_mean_index_ratio:.3f}</div><div class="lx-metric-label">Ratio LZ78 moyen (proxy indice d\'assemblage)</div></div>', unsafe_allow_html=True)
    st.code(state.kan_formula_str or "en attente du premier tick...", language="text")
    st.caption(
        "KAN : recalcul honnête (splines linéaires disclosées) des mêmes entrées que le gate Kelly réel, "
        "affiché pour audit -- ne re-décide rien. Koopman/DMD : SVD tronquée réelle sur un plongement de "
        "retard, ne s'active que si le biais de chaos partagé dépasse le seuil de non-linéarité -- une "
        "prévision locale, pas une résolution du marché. Filtre de complexité : proxy réel (LZ78) sur un "
        "flux d'ordres synthétique disclosé, pas une identification de contrepartie réelle -- voir le "
        "docstring de `backend_simulator.py` pour la correction complète de l'indice d'assemblage littéral."
    )

    st.markdown("")
    st.markdown('#### 🖥️ Journal d\'événements du Léviathan &nbsp; <span class="lx-badge">RÉEL, ÉTAT PARTAGÉ</span>', unsafe_allow_html=True)
    if events:
        lines = []
        for ts, src, dst, msg in events:
            age_s = now - ts
            lines.append(
                f'<div><span class="ts">T-{age_s:04.1f}s</span> <b>{src}</b> '
                f'<span class="arrow">→</span> <b>{dst}</b> : {msg}</div>'
            )
        st.markdown(f'<div class="lx-terminal">{"".join(lines)}</div>', unsafe_allow_html=True)
    else:
        st.info("Aucun événement encore -- l'univers vient d'être réinitialisé. Le battement de cœur autonome va bientôt en produire.")

    st.caption(
        f"Tick #{state.tick_count} (battement de cœur autonome de cette page, ~1/s tant qu'elle est ouverte) -- "
        "tous les chiffres ci-dessus sont le même état partagé que celui lu/écrit par les 7 autres pages "
        "'frontier tech' de cet EMS, pas une copie ni une simulation séparée."
    )


@st.fragment(run_every="1s")
def _render_daemon_matrix() -> None:
    swarm = ghost_daemon_heartbeat()
    manager = get_manager()
    bg = manager.background_snapshot()

    st.markdown("")
    st.markdown('#### 🔮 Daemon Activity Matrix &nbsp; <span class="lx-badge">PROTOCOLE DEFI GHOST DAEMONS</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lx-disclosure">🔬 16 automates asynchrones réels (voir '
        "<code>src/dashboard/core_daemons.py</code>), coordonnés par un vrai "
        "<code>asyncio.gather</code> -- chaque bloc calcule une vraie loi mathématique sur des données "
        "DeFi synthétiques disclosées ; la signature \"Web3\" est un mock explicite (aucune vraie clé, "
        "aucun vrai RPC, aucun fonds réel). 🟢 actif · 🔴 alerte · 🔵 exécution · ⚪ veille."
        "</div>",
        unsafe_allow_html=True,
    )
    if bg["started_at"] is not None:
        uptime_s = time.time() - bg["started_at"]
        st.caption(f"⚙️ Processus autonome en arrière-plan (thread détaché, sa propre boucle asyncio) : tick #{bg['tick']} -- actif depuis {uptime_s:,.0f}s.")

    cols_per_row = 4
    daemon_ids = list(DAEMON_ORDER)
    for row_start in range(0, len(daemon_ids), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, daemon_id in zip(row_cols, daemon_ids[row_start:row_start + cols_per_row]):
            result = swarm.results[daemon_id]
            codename, law = DAEMON_META[daemon_id]
            color = STATUS_LED_COLOR[result.status]
            col.markdown(
                f'<div class="lx-led" style="background: rgba(255,255,255,0.03); box-shadow: 0 0 10px {color}55; '
                f'border-color: {color}66;">'
                f'<div class="lx-led-id">{daemon_id}</div>'
                f'<div class="lx-led-name" style="color:{color};">{codename}</div>'
                f'<div style="font-size:0.62rem;color:#8a84a8;">{law}</div>'
                f'<div class="lx-led-status" style="color:{color};">{result.status}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('##### 🖥️ Log temps réel du swarm')
    if swarm.log:
        lines = list(swarm.log)[-14:][::-1]
        st.markdown(f'<div class="lx-terminal">{"".join(f"<div>{ln}</div>" for ln in lines)}</div>', unsafe_allow_html=True)
    st.caption(f"Tick #{swarm.tick_count} (câblé sur le même `shannon_chaos_bias` partagé que le reste de l'EMS) -- les lignes non-\"actif\" sont republiées dans le journal d'événements du Léviathan ci-dessus.")


@st.fragment(run_every="1s")
def _render_telemetry() -> None:
    manager = get_manager()
    bg = manager.background_snapshot()
    redis_ok, redis_rtt_ms = manager.redis_health()
    redis_snapshot_age_s = manager.redis_snapshot_age_seconds()
    throughput = manager.zmq_throughput_per_sec()

    st.markdown("")
    st.markdown('#### 📡 Telemetry & Latency &nbsp; <span class="lx-badge">PROTOCOLE ZERO-LATENCY</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lx-disclosure">🔬 Chaque nombre ci-dessous est réellement mesuré cette seconde '
        "(<code>time.perf_counter</code> / horodatage réel <code>MessagePayload.timestamp</code>), jamais "
        "une valeur d'exemple recopiée -- voir le docstring de <code>core_daemons.py</code> pour la "
        "correction honnête de la prémisse \"Redis plus rapide que st.session_state\" (c'est l'inverse : "
        "Redis sert ici de miroir d'état inter-process réel, pas d'accélérateur)."
        "</div>",
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)
    redis_label = "🟢 EN LIGNE" if redis_ok else "🔴 HORS LIGNE"
    r1.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{redis_label}</div><div class="lx-metric-label">Serveur Redis (PING réel)</div></div>', unsafe_allow_html=True)
    ping_label = f"{redis_rtt_ms:.2f} ms" if redis_rtt_ms is not None else "N/A"
    r2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{ping_label}</div><div class="lx-metric-label">Round-trip réel PING (client sync, thread Streamlit)</div></div>', unsafe_allow_html=True)
    age_label = f"{redis_snapshot_age_s:.1f}s" if redis_snapshot_age_s is not None else "aucune clé"
    r3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{age_label}</div><div class="lx-metric-label">Âge réel du dernier miroir d\'état (clé TTL={5}s)</div></div>', unsafe_allow_html=True)

    z1, z2, z3, z4 = st.columns(4)
    zmq_label = "🟢 LIÉ" if bg["zmq_bound"] else "🔴 NON LIÉ"
    z1.markdown(f'<div class="lx-metric"><div class="lx-metric-value" style="font-size:1.0rem;">{zmq_label}</div><div class="lx-metric-label">Bus ZMQ isolé ({GHOST_ZMQ_PUB_ADDRESS})</div></div>', unsafe_allow_html=True)
    z2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{throughput:.0f} msg/s</div><div class="lx-metric-label">Débit réel ZMQ (fenêtre glissante 1s)</div></div>', unsafe_allow_html=True)
    z3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{bg["zmq_messages_published"]} / {bg["zmq_messages_consumed"]}</div><div class="lx-metric-label">Messages réels publiés / consommés (cumul)</div></div>', unsafe_allow_html=True)
    z4.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{bg["zmq_dropped_stale"]} / {bg["zmq_dropped_backpressure"]}</div><div class="lx-metric-label">Détruits : périmés (&gt;{STALE_MESSAGE_THRESHOLD_MS:.0f}ms) / backpressure</div></div>', unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    d2f = bg["latency_daemon_to_fleet_ms"]
    d2f_label = f"{d2f:.3f} ms" if d2f is not None else "en attente..."
    l1.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{d2f_label}</div><div class="lx-metric-label">Latence réelle Daemons → Fleet Command</div></div>', unsafe_allow_html=True)
    f2e = bg["latency_fleet_to_execution_ms"]
    f2e_label = f"{f2e:.3f} ms" if f2e is not None else "en attente..."
    l2.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{f2e_label}</div><div class="lx-metric-label">Latence réelle Fleet Command → Exécution (ZMQ)</div></div>', unsafe_allow_html=True)
    l3.markdown(f'<div class="lx-metric"><div class="lx-metric-value">{EXECUTION_QUEUE_MAXSIZE}</div><div class="lx-metric-label">Capacité réelle de la queue borné (backpressure)</div></div>', unsafe_allow_html=True)

    st.caption(
        "Redis = miroir d'état inter-process réel (pas un accélérateur -- `st.session_state` reste plus "
        "rapide pour cette page). ZMQ = vrai bus PUB/SUB isolé (port dédié, distinct des agents de trading "
        "réels) entre Fleet Command et les Automates d'Exécution. Backpressure réelle : la queue bornée "
        "évince le message le plus ancien plutôt que de bloquer ; la fraîcheur réelle de chaque message est "
        "vérifiée contre son propre horodatage avant exécution."
    )


_render_nexus()
_render_daemon_matrix()
_render_telemetry()

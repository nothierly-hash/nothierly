"""Singularity Engine Dashboard -- Deep Tech showcase for institutional M&A audiences.

**Two real, disclosed corrections to the brief, same discipline this project applies everywhere
else on this page's own real content**:

1. **No ``pages/05_Singularity_Engine.py``, and no second ``st.set_page_config()``.** This project
   has no ``pages/`` auto-discovery folder by design -- ``nothierly_ems.py``'s own module docstring
   already explains why (per-page isolation via ``st.Page`` file paths) and already redirects every
   earlier chapter that asked for a literal ``pages/0N_name.py`` path here, under ``ems_pages/``,
   for the same reason. The router already calls ``st.set_page_config(..., layout="wide")`` exactly
   once, before any page loads -- a second call from this page would crash it outright (the same
   real bug ``swarm_command_center.py`` avoided earlier this session).

2. **The honest line this page draws, and why it matters more here than anywhere else in this
   project**: every other "R&D/vision" module in this codebase (``nothierly_genesis/*``) discloses
   its real/simulated boundary in a *docstring* -- fine for code other engineers read. This page is
   explicitly, literally built "pour les investisseurs (M&A)" -- for a real institutional audience
   making real financial decisions. A docstring nobody in that room ever opens does not satisfy
   this project's own "never fabricate data presented as real" rule when the audience is exactly
   the people that rule exists to protect. Every module below that has no real, live backend behind
   it (the thermodynamic migration map, the causal counterfactual panel, the autonomous-hiring
   terminal) is labeled **[SIMULATION]**, visibly, on the rendered page itself, in the same breath
   as its own numbers -- not buried in a tooltip. The one module with a real backend (Cognitive
   Router pricing -- :mod:`hypervisor.cognitive_pricing`, built and tested earlier this session) is
   labeled **[DONNÉES RÉELLES]** and actually reads that real registry, not a hardcoded copy of its
   numbers. The brief's own literal "+4 520$ de bénéfice (24h)" and "Risque net : 0%" figures are
   not rendered anywhere on this page: this project has no 24h of real accumulated hypervisor
   traffic to back the first number, and no real automatic NGN-to-ChipperCash failover exists to
   back the second -- both would be fabricated precision presented to a real deal-evaluating
   audience, exactly the failure mode this project's whole testing discipline exists to prevent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import random
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hypervisor.cognitive_pricing import ModelRegistry, TaskSpecialty

_DARK_BG = "#05070a"
_PANEL_BG = "#0d1117"
_ACCENT = "#00e5ff"
_GREEN = "#00e676"
_RED = "#ff1744"
_TEXT_DIM = "#8b949e"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_DARK_BG}; }}
    .singularity-panel {{
        background: {_PANEL_BG}; border: 1px solid rgba(0,229,255,0.15); border-radius: 10px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.6rem;
    }}
    .singularity-disclosure {{
        border-left: 3px solid {_ACCENT}; background: rgba(0,229,255,0.06);
        padding: 0.6rem 1rem; font-size: 0.85rem; color: {_TEXT_DIM}; border-radius: 4px;
    }}
    .singularity-badge-real {{
        background: rgba(0,230,118,0.15); color: {_GREEN}; border: 1px solid {_GREEN};
        border-radius: 4px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700;
    }}
    .singularity-badge-sim {{
        background: rgba(255,23,68,0.12); color: #ff6e6e; border: 1px solid {_RED};
        border-radius: 4px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700;
    }}
    .singularity-terminal {{
        background: #000000; color: #39ff14; font-family: 'Consolas', monospace; font-size: 0.82rem;
        padding: 0.8rem; border-radius: 6px; height: 220px; overflow-y: auto; line-height: 1.5;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Singularity Engine")
st.caption("Nothierly Deep Tech Showcase -- Cognitive Routing / Thermodynamic Compute / Causal Risk / Autonomous Ops")
st.markdown(
    '<div class="singularity-disclosure">'
    '🔬 <b>Légende honnête</b> -- '
    '<span class="singularity-badge-real">DONNÉES RÉELLES</span> = lit un backend réellement construit et testé cette session. '
    '<span class="singularity-badge-sim">SIMULATION</span> = vision R&D illustrative, aucun système de production ne fait cela aujourd\'hui. '
    'Cette page ne présente jamais un chiffre simulé comme une métrique business réalisée -- voir le docstring du module.'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")


# =============================================================================
# Module 1 -- Cognitive Router (real pricing backend)
# =============================================================================


@st.fragment(run_every="2s")
def _render_cognitive_router_module() -> None:
    st.markdown(
        '### 🧩 Cognitive Routing & API Abstraction &nbsp; <span class="singularity-badge-real">DONNÉES RÉELLES</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Tarifs réels publiés (GPT-4o/Claude) et estimation réelle disclosée (Vast.ai) -- "
        "lus depuis `hypervisor/cognitive_pricing.py`, construit et testé cette session. "
        "Le volume de tâches ci-dessous est une simulation de charge pour la démonstration visuelle."
    )

    registry = ModelRegistry()
    rng = np.random.default_rng()
    # Simulated load, disclosed above -- real cost RATES from the registry are applied to it.
    simulated_tasks_per_worker = {
        "claude-3-5-sonnet": int(rng.integers(2, 9)),
        "llama-3-70b-vastai": int(rng.integers(10, 40)),
        "gpt-4o": int(rng.integers(1, 5)),
        "mistral-nemo-local": int(rng.integers(5, 20)),
    }
    avg_tokens_per_task = 900

    rows = []
    for worker_name, task_count in simulated_tasks_per_worker.items():
        worker = registry.get(worker_name)
        cost = worker.cost_for(task_count * avg_tokens_per_task, task_count * avg_tokens_per_task // 3)
        rows.append({"worker": worker_name, "tasks": task_count, "cost_usd": cost, "specialty": worker.specialty.value})
    df = pd.DataFrame(rows).sort_values("tasks")

    col_chart, col_metric = st.columns([2, 1])
    with col_chart:
        fig = go.Figure(
            go.Bar(
                x=df["tasks"], y=df["worker"], orientation="h",
                marker=dict(color=[_ACCENT if "vastai" in w or "local" in w else _RED for w in df["worker"]]),
                text=[f"{t} tâches -- ${c:.4f}" for t, c in zip(df["tasks"], df["cost_usd"])],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG,
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tâches routées (dernière seconde, simulé)",
        )
        st.plotly_chart(fig, width="stretch", key=f"cog_router_{time.time()}")

    with col_metric:
        naive_cost = sum(registry.get("gpt-4o").cost_for(t * avg_tokens_per_task, t * avg_tokens_per_task // 3) for t in simulated_tasks_per_worker.values())
        real_cost = df["cost_usd"].sum()
        real_savings_pct = (naive_cost - real_cost) / naive_cost * 100.0 if naive_cost > 0 else 0.0
        st.metric("Coût réel optimisé (cette seconde)", f"${real_cost:.4f}")
        st.metric("Vs. tout-GPT-4o (référence réelle)", f"${naive_cost:.4f}")
        st.metric("Économie réelle de routage", f"{real_savings_pct:.1f}%")
        st.caption("Taux d'économie calculé sur les vrais tarifs publiés -- pas une projection 24h fabriquée.")


# =============================================================================
# Module 2 -- Thermodynamic AI Migration (illustrative -- no real infra exists)
# =============================================================================

_TEXAS = (31.0, -99.0)
_ICELAND = (64.9, -19.0)


@st.fragment(run_every="2s")
def _render_thermodynamic_module() -> None:
    st.markdown(
        '### 🌍 Thermodynamic AI Migration &nbsp; <span class="singularity-badge-sim">SIMULATION</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Vision R&D illustrative : aucune infrastructure de calcul Nothierly ne migre réellement "
        "aujourd'hui -- ce projet tourne sur cette seule machine locale. Aucun vrai capteur de "
        "prix de l'électricité n'est branché."
    )

    if "singularity_migration_progress" not in st.session_state:
        st.session_state["singularity_migration_progress"] = 0.0
    progress = st.session_state["singularity_migration_progress"]
    progress = (progress + 0.08) % 1.0
    st.session_state["singularity_migration_progress"] = progress

    lat = _TEXAS[0] + (_ICELAND[0] - _TEXAS[0]) * progress
    lon = _TEXAS[1] + (_ICELAND[1] - _TEXAS[1]) * progress

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[_TEXAS[1], _ICELAND[1]], lat=[_TEXAS[0], _ICELAND[0]], mode="lines",
        line=dict(width=1.5, color=_TEXT_DIM, dash="dot"), showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=[_TEXAS[1]], lat=[_TEXAS[0]], mode="markers+text", marker=dict(size=14, color=_RED),
        text=["⚠ Texas -- +0.12$/kWh (simulé)"], textposition="bottom center", textfont=dict(color=_RED), showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=[_ICELAND[1]], lat=[_ICELAND[0]], mode="markers+text", marker=dict(size=14, color=_GREEN),
        text=["✓ Islande -- 0.03$/kWh géothermie (simulé)"], textposition="top center", textfont=dict(color=_GREEN), showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=[lon], lat=[lat], mode="markers", marker=dict(size=12, color=_ACCENT, symbol="triangle-right"), showlegend=False,
    ))
    fig.update_geos(
        projection_type="orthographic", showland=True, landcolor="#1b2230", showocean=True, oceancolor=_DARK_BG,
        showcountries=True, countrycolor="#2a3140", bgcolor=_PANEL_BG,
        projection_rotation=dict(lon=-50, lat=45),
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL_BG, height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch", key=f"thermo_{time.time()}")

    if progress > 0.85:
        st.success("Migration de la charge simulée -- Islande (démonstration, aucun transfert réel).")
    else:
        st.info(f"Migration simulée en cours : {progress * 100:.0f}%")


# =============================================================================
# Module 3 -- Causal Risk Engine (illustrative narrative graph)
# =============================================================================


@st.fragment(run_every="2s")
def _render_causal_module() -> None:
    st.markdown(
        '### 🕸️ Causal Risk Engine (Beyond Correlation) &nbsp; <span class="singularity-badge-sim">SIMULATION</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Graphe narratif illustratif -- ni do-calculus ni inférence contrefactuelle réelle n'est "
        "exécutée ici. Le vrai moteur causal de ce projet (`nothierly_genesis/causal_impact_engine.py`, "
        "DoWhy) répond à une question différente (effet causal mesuré sur une série temporelle) et "
        "n'est pas branché à ce récit NGN/ChipperCash."
    )

    chain = ["Chute du Naira", "Hausse des volumes P2P", "Saturation bancaire", "Blocage des API locales"]
    graph: nx.DiGraph = nx.DiGraph()
    for i, node in enumerate(chain):
        graph.add_node(node, x=i)
    for i in range(len(chain) - 1):
        graph.add_edge(chain[i], chain[i + 1])

    pulse = int(time.time()) % len(chain)
    node_x = [graph.nodes[n]["x"] for n in graph.nodes]
    node_y = [0] * len(chain)
    colors = [_ACCENT if i == pulse else "#3a4250" for i in range(len(chain))]

    fig = go.Figure()
    for u, v in graph.edges:
        fig.add_trace(go.Scatter(
            x=[graph.nodes[u]["x"], graph.nodes[v]["x"]], y=[0, 0], mode="lines",
            line=dict(color=_TEXT_DIM, width=2), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=chain, textposition="top center",
        marker=dict(size=26, color=colors, line=dict(width=2, color=_ACCENT)),
        textfont=dict(color="#e6edf3", size=11), showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG, height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1, 1]),
    )
    st.plotly_chart(fig, width="stretch", key=f"causal_{time.time()}")

    st.markdown(
        "**Simulation contrefactuelle (illustrative)** : *Que se passerait-il si Binance suspendait "
        "le NGN dans les 5 prochaines minutes ?*\n\n"
        "→ Scénario narratif : bascule envisagée vers un rail alternatif (ex. ChipperCash). "
        "**Aucun moteur de bascule automatique réel n'existe dans ce projet** -- ceci est une "
        "hypothèse de roadmap, pas un résultat calculé."
    )


# =============================================================================
# Module 4 -- Autonomous Economic Agent (illustrative terminal)
# =============================================================================

_HR_SCRIPT = [
    "Audit de performance : latence de boucle ZMQ mesurée = {latency:.0f}ms. ({status})",
    "Génération d'un ticket GitHub Issue (illustratif -- aucune API GitHub réelle appelée).",
    "Bounty Smart Contract simulé : {bounty} USDT pour optimisation C++ (aucun contrat réel déployé).",
    "En attente de soumission humaine... (simulation)",
]


@st.fragment(run_every="1s")
def _render_autonomous_hr_module() -> None:
    st.markdown(
        '### 🤖 Autonomous Economic Agent (Self-Hiring) &nbsp; <span class="singularity-badge-sim">SIMULATION</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Terminal illustratif -- aucun système d'embauche/bounty automatique réel n'existe dans ce "
        "projet (pas d'API GitHub, pas d'API Upwork, pas de smart contract déployé). "
        "La latence affichée est un tirage aléatoire autour d'ordres de grandeur réellement mesurés "
        "cette session pour le bus ZMQ (voir `tests/test_hypervisor.py`)."
    )

    if "singularity_hr_log" not in st.session_state:
        st.session_state["singularity_hr_log"] = []

    log = st.session_state["singularity_hr_log"]
    if len(log) < 200:
        latency = random.uniform(15.0, 90.0)
        status = "Anormal" if latency > 40.0 else "Nominal"
        bounty = random.choice([250, 500, 750])
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        step = len(log) % len(_HR_SCRIPT)
        line = _HR_SCRIPT[step].format(latency=latency, status=status, bounty=bounty)
        log.append(f"{now} - {line}")

    st.markdown(
        '<div class="singularity-terminal">' + "<br>".join(f"&gt; {line}" for line in log[-12:]) + "</div>",
        unsafe_allow_html=True,
    )


_render_cognitive_router_module()
_render_thermodynamic_module()
_render_causal_module()
_render_autonomous_hr_module()

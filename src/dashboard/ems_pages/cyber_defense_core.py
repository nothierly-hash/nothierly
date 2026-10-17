"""Cyber Defense Core -- vision showcase, honest version.

**Real, disclosed scope**: ASLR (address space layout randomization) and honeypot/deception
technology are real, established defensive security techniques -- and Fully Homomorphic Encryption
(``Enc(A) x Enc(B) = Enc(A x B)``) is real, published cryptography (real schemes: CKKS, BFV; real
libraries: Microsoft SEAL, OpenFHE). None of the three are implemented in this project: there is no
real polymorphic binary mutation, no real attacker-facing sandbox/honeypot, and no real FHE
pipeline anywhere in this codebase. Every claim below is labeled as a conceptual simulation, not an
operating security control -- presenting an unbuilt security capability as active protection would
be a false security claim, the same discipline already applied on
``institutional_audit.py``. The brief's own precise "0.0000001%" reverse-engineering probability
is not reproduced as a real figure (unmeasurable, fabricated precision); shown instead as a
labeled illustrative gauge.
"""

from __future__ import annotations

import hashlib
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

_BG = "#050505"
_PANEL = "#0d0f0d"
_MATRIX_GREEN = "#00ff41"
_ALERT_RED = "#ff1744"
_GRAY = "#666"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; font-family: 'Consolas', monospace; }}
    .cdc-panel {{ background: {_PANEL}; border: 1px solid #1a3a1a; border-radius: 6px; padding: 0.9rem 1rem; }}
    .cdc-disclosure {{ border-left: 3px solid {_ALERT_RED}; background: rgba(255,23,68,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #ffb3b3; border-radius: 4px; }}
    .cdc-badge {{ background: rgba(255,23,68,0.12); color: #ff8080; border: 1px solid {_ALERT_RED}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.7rem; font-weight: 700; }}
    .cdc-terminal {{ background: #000; color: {_MATRIX_GREEN}; font-size: 0.78rem; padding: 0.7rem; border-radius: 6px; height: 240px; overflow-y: auto; line-height: 1.5; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ Cyber Defense Core")
st.caption("Vision R&D -- Défense active, honnêtement scopée")
st.markdown(
    '<div class="cdc-disclosure">'
    "⚠️ <b>Aucun des trois systèmes ci-dessous n'est implémenté dans ce projet</b> -- "
    "ASLR/honeypots/FHE sont des techniques de sécurité réelles et publiées, mais aucune mutation "
    "binaire, aucun sandbox d'attaquant, aucun pipeline FHE réel n'existe ici. Présenter cette page "
    "comme une protection active serait une fausse déclaration de sécurité."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

col1, col2, col3 = st.columns(3)

# =============================================================================
# Column 1 -- Polymorphic Kernel (simulation)
# =============================================================================

with col1:
    st.markdown('#### 🧬 Noyau Polymorphe &nbsp; <span class="cdc-badge">SIMULATION</span>', unsafe_allow_html=True)

    @st.fragment(run_every="2s")
    def _render_polymorphic() -> None:
        if "cdc_hash_log" not in st.session_state:
            st.session_state["cdc_hash_log"] = []
        log = st.session_state["cdc_hash_log"]
        if len(log) < 100:
            seed = f"{time.time()}-{random.random()}".encode()
            new_hash = hashlib.sha256(seed).hexdigest()[:12]
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            log.append(f"{now} Rotation ASLR (simulée)... Signature : 0x{new_hash} (mutation)")
        st.markdown('<div class="cdc-terminal">' + "<br>".join(f"&gt; {line}" for line in log[-10:]) + "</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=0.01, number={"suffix": "%"},
            title={"text": "Probabilité de reverse-engineering (illustrative)"},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": _MATRIX_GREEN}},
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=200, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", key=f"cdc_gauge_{time.time()}")
        st.caption("Chiffre illustratif non mesurable -- aucun noyau polymorphe réel n'existe.")

    _render_polymorphic()

# =============================================================================
# Column 2 -- Shadow Matrix honeypot (simulation)
# =============================================================================

with col2:
    st.markdown('#### 🕸️ Shadow Matrix (Honeypot) &nbsp; <span class="cdc-badge">SIMULATION</span>', unsafe_allow_html=True)

    @st.fragment(run_every="3s")
    def _render_shadow_matrix() -> None:
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_node("Core Engine", pos=(0, 0.6))
        graph.add_node("Attaquant (simulé)", pos=(-1, 0))
        graph.add_node("Shadow Sandbox", pos=(1, -0.6))
        graph.add_edge("Attaquant (simulé)", "Core Engine")
        graph.add_edge("Core Engine", "Shadow Sandbox")

        pulse = int(time.time()) % 2 == 0
        colors = {"Core Engine": "#2979ff", "Attaquant (simulé)": _ALERT_RED if pulse else _GRAY, "Shadow Sandbox": _GRAY}

        pos = nx.get_node_attributes(graph, "pos")
        fig = go.Figure()
        for u, v in graph.edges:
            fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines", line=dict(color=_GRAY, width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in graph.nodes], y=[pos[n][1] for n in graph.nodes],
            mode="markers+text", text=list(graph.nodes), textposition="bottom center",
            marker=dict(size=28, color=[colors[n] for n in graph.nodes]), textfont=dict(color="white", size=10), showlegend=False,
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=260, margin=dict(l=10, r=10, t=10, b=30),
                           xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1, 1]))
        st.plotly_chart(fig, width="stretch", key=f"shadow_matrix_{time.time()}")

        if "cdc_intrusion_log" not in st.session_state:
            st.session_state["cdc_intrusion_log"] = []
        log = st.session_state["cdc_intrusion_log"]
        if len(log) < 100 and pulse:
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            fake_ip = f"194.55.{random.randint(0,255)}.{random.randint(0,255)}"
            log.append(f"{now} Intrusion simulée (IP fictive {fake_ip}) déviée vers le sandbox (démo).")
        st.markdown('<div class="cdc-terminal" style="height:100px;">' + "<br>".join(f"&gt; {line}" for line in log[-4:]) + "</div>", unsafe_allow_html=True)

    _render_shadow_matrix()

# =============================================================================
# Column 3 -- Homomorphic Encryption (conceptual)
# =============================================================================

with col3:
    st.markdown('#### 🔒 Chiffrement Homomorphe (FHE) &nbsp; <span class="cdc-badge">SIMULATION / VISION R&D</span>', unsafe_allow_html=True)
    st.markdown('<div class="cdc-panel">Nœud GPU (concept) -- traite une matrice restant chiffrée de bout en bout.</div>', unsafe_allow_html=True)
    st.latex(r"Enc(A) \times Enc(B) = Enc(A \times B)")
    st.caption(
        "Propriété cryptographique réelle (schémas CKKS/BFV réels, ex. Microsoft SEAL) -- "
        "**aucun pipeline FHE réel n'est intégré à ce projet**. Aucun calcul Vast.ai réel ne "
        "s'exécute ici sous chiffrement."
    )
    st.markdown(
        '<div class="cdc-panel" style="margin-top:0.6rem;color:#666;">'
        "Statut (conceptuel) : exécution zéro-confiance visée -- non implémentée aujourd'hui."
        "</div>",
        unsafe_allow_html=True,
    )

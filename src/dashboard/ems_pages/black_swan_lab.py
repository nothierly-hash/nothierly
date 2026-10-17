"""Black Swan Lab -- Zero-to-One Vision Showcase, honest version.

**One tab refused outright, not built even as a simulation**: the brief's own "Ghost Network"
concept -- splitting a large order into ~500 micro-transactions disguised with fake merchant
labels ("Spotify Subscription", "Uber Ride") specifically to stay invisible to detection systems
("Central Bank AI Detection Rate: 0.00%") -- is a real description of transaction *structuring*, a
real financial crime (illegal under 31 U.S.C. 5324 in the US, and equivalent statutes elsewhere)
regardless of the underlying funds' origin. Building this, even labeled "simulation", would hand
over a working design for evading AML detection -- refused for the same reason this session
already refused an earlier "adversarial market chaos" request. See this session's own conversation
for the full reasoning; the tab below explains this in place of the content, rather than silently
vanishing.

**The other two tabs are real, legitimate R&D vision** (sentiment-driven trading signals are
already partially real in this project -- see ``analyst_agent.py``/``political_agent.py``'s own
real FinBERT sentiment classification; DNA-based cold storage is real, published security
research, just not something this project has built) -- shown here as clearly labeled, honestly
scoped conceptual visualizations, same discipline as ``singularity_engine.py``.
"""

from __future__ import annotations

import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_BG = "#02040a"
_PANEL = "#0a0e1a"
_NEON_GREEN = "#39ff14"
_NEON_PURPLE = "#b026ff"
_NEON_CYAN = "#00fff9"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .bsl-panel {{ background: {_PANEL}; border: 1px solid rgba(176,38,255,0.25); border-radius: 8px; padding: 1rem 1.2rem; }}
    .bsl-disclosure {{ border-left: 3px solid {_NEON_CYAN}; background: rgba(0,255,249,0.05); padding: 0.6rem 1rem; font-size: 0.85rem; color: #9fdff0; border-radius: 4px; }}
    .bsl-refused {{ border-left: 3px solid #ff2222; background: rgba(255,34,34,0.08); padding: 1rem; font-size: 0.9rem; color: #ffb3b3; border-radius: 4px; }}
    .bsl-badge-sim {{ background: rgba(176,38,255,0.15); color: {_NEON_PURPLE}; border: 1px solid {_NEON_PURPLE}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .bsl-terminal {{ background: #000; color: {_NEON_GREEN}; font-family: 'Consolas', monospace; font-size: 0.8rem; padding: 0.8rem; border-radius: 6px; height: 260px; overflow-y: auto; line-height: 1.5; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌌 Black Swan Lab")
st.caption("Nothierly Zero-to-One Vision -- R&D exploratoire")
st.markdown(
    '<div class="bsl-disclosure">🔬 Toutes les sections de cette page sont des visions R&D '
    'conceptuelles (<span class="bsl-badge-sim">SIMULATION</span>) -- aucune n\'est un système de '
    "production. Un onglet du brief original a été refusé ; voir l'explication dans cet onglet.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_ghost, tab_psycho, tab_dna = st.tabs(["Ghost Network (refusé)", "Psychohistory Engine", "Post-Quantum DNA Vault"])

with tab_ghost:
    st.markdown(
        '<div class="bsl-refused">'
        "⛔ <b>Ce module n'a pas été construit, y compris en simulation.</b><br><br>"
        "Le concept demandé -- éclater un ordre de 1 000 000$ en ~500 micro-transactions "
        "maquillées en achats anodins (\"Spotify Subscription\", \"Uber Ride\") pour rester "
        "invisible aux systèmes de détection des banques centrales -- décrit une vraie technique "
        "de <b>structuring</b> (démantèlement de transactions), un vrai délit financier "
        "(illégal aux États-Unis sous 31 U.S.C. § 5324, et sous des lois équivalentes ailleurs), "
        "indépendamment de l'origine des fonds. Fournir ce plan, même en simulation, revient à "
        "livrer une méthode fonctionnelle d'évasion AML. C'est une ligne que je ne franchis pas."
        "</div>",
        unsafe_allow_html=True,
    )

with tab_psycho:
    st.markdown('#### 🧠 Psychohistory Engine &nbsp; <span class="bsl-badge-sim">SIMULATION</span>', unsafe_allow_html=True)
    st.caption(
        "Vision R&D illustrative d'un signal de sentiment social en argot local. Ce projet a "
        "une vraie classification de sentiment (FinBERT, 3 classes réelles POSITIVE/NEGATIVE/NEUTRAL "
        "-- voir `core/schemas.py`) appliquée à de vrais titres de presse dans `analyst_agent.py`/"
        "`political_agent.py` -- **pas** à des messages WhatsApp/Telegram scrapés, qui n'existent "
        "nulle part dans ce projet."
    )

    col_terminal, col_gauge = st.columns([3, 2])

    _MESSAGES = [
        ("Omo, Naira is melting today!", "NGN", "NEGATIVE"),
        ("Dólar blue explotó loco!", "ARS", "NEGATIVE"),
        ("Tout va bien, le marché est calme", "EUR", "NEUTRAL"),
        ("Everyone is buying USDT right now!!", "TRY", "POSITIVE"),
    ]

    @st.fragment(run_every="2s")
    def _render_terminal() -> None:
        if "bsl_psycho_log" not in st.session_state:
            st.session_state["bsl_psycho_log"] = []
        log = st.session_state["bsl_psycho_log"]
        if len(log) < 100:
            msg, fiat, sentiment = random.choice(_MESSAGES)
            now = datetime.now(timezone.utc).strftime("%H:%M:%S")
            log.append(f"{now} [{fiat}] \"{msg}\" -> sentiment (simulé)={sentiment}")
        st.markdown('<div class="bsl-terminal">' + "<br>".join(f"&gt; {line}" for line in log[-14:]) + "</div>", unsafe_allow_html=True)

    with col_terminal:
        st.markdown("**Terminal d'écoute (simulation)**")
        _render_terminal()

    @st.fragment(run_every="2s")
    def _render_gauge() -> None:
        rng = np.random.default_rng()
        value = float(rng.integers(60, 96))
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            title={"text": "Probabilité de dévaluation simulée (4h)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": _NEON_PURPLE},
                   "steps": [{"range": [0, 50], "color": "#1a0a1a"}, {"range": [50, 100], "color": "#3a0a3a"}]},
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=260, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", key=f"psycho_gauge_{time.time()}")
        st.caption("Chiffre simulé, tiré aléatoirement pour la démonstration -- aucun modèle prédictif réel ne calcule cette probabilité.")

    with col_gauge:
        st.markdown("**Résultat prédictif (simulation)**")
        _render_gauge()

with tab_dna:
    st.markdown('#### 🧬 Post-Quantum DNA Vault &nbsp; <span class="bsl-badge-sim">SIMULATION / VISION R&D</span>', unsafe_allow_html=True)
    st.caption(
        "Visualisation conceptuelle -- le stockage de clés dans de l'ADN synthétisé est un vrai "
        "champ de recherche en sécurité, mais aucune synthèse ADN réelle n'existe dans ce projet. "
        "L'hélice ci-dessous est une vraie construction 3D paramétrique (mathématiques réelles), "
        "pas une donnée biologique réelle."
    )

    fake_key_hex = "0x4F8A9B2C1D3E5F60"
    binary_stream = "".join(format(int(c, 16), "04b") for c in fake_key_hex[2:18])
    dna_map = {"00": "A", "01": "T", "10": "C", "11": "G"}
    dna_sequence = "-".join(dna_map[binary_stream[i:i + 2]] for i in range(0, len(binary_stream) - 1, 2))

    st.code(f"Private Key (Hex, factice) : {fake_key_hex}\nBinary Stream               : {binary_stream}\nDNA Sequence (synthèse simulée) : {dna_sequence}")

    theta = np.linspace(0, 6 * np.pi, 200)
    z = np.linspace(0, 4, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=np.cos(theta), y=np.sin(theta), z=z, mode="lines", line=dict(color=_NEON_GREEN, width=5), name="Brin A"))
    fig.add_trace(go.Scatter3d(x=np.cos(theta + np.pi), y=np.sin(theta + np.pi), z=z, mode="lines", line=dict(color=_NEON_CYAN, width=5), name="Brin B"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL, height=420, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    st.plotly_chart(fig, width="stretch", key="dna_helix")

    if st.button("🔴 Simulate EMP / Quantum Attack"):
        st.success("Attaque simulée échouée (scénario scripté). Aucun système biologique réel n'existe -- ceci est une démonstration d'interface, pas un test de sécurité.")

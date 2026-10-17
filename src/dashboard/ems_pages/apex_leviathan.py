"""Apex Leviathan -- Zero-to-One Vision Showcase, honest version.

**One module refused outright, not built even as a simulation**: the brief's own "vol de compute via
WebAssembly" concept -- harvesting visitors'/third parties' CPU or GPU cycles through an in-browser
WASM module without their informed, opt-in consent -- is a real description of **browser
cryptojacking**, a well-documented category of malware (the same technique Coinhive was built on
before being shut down industry-wide in 2019) and unauthorized-access computer crime in most
jurisdictions (e.g. the US Computer Fraud and Abuse Act, 18 U.S.C. 1030). Framing it as "stolen
compute" in the brief is not a euphemism this page can launder by calling it a simulation: even a
fake progress bar promoting the idea would be advertising a real attack technique as a Nothierly
product feature. Refused for the same reason ``black_swan_lab.py``'s "Ghost Network" tab was refused
-- see that module's docstring for the identical reasoning applied there first.

**The other module is real, legitimate R&D vision**: algorithmic, crypto-collateralized stablecoins
are a real, well-studied category (real schemes: MakerDAO's DAI, real failures: TerraUSD/LUNA's
May 2022 collapse, a directly relevant cautionary precedent for anyone proposing a new one) -- shown
here as a clearly labeled, honestly scoped conceptual visualization with its real-world failure mode
disclosed up front, not a working token, smart contract, or financial instrument. Same discipline as
``institutional_audit.py``'s MPC vault and ``cyber_defense_core.py``'s FHE module: real underlying
concept, zero real implementation in this project, said plainly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_BG = "#03050a"
_PANEL = "#0a0d16"
_NEON_AMBER = "#ffb000"
_NEON_CYAN = "#00e5ff"
_ALERT_RED = "#ff1744"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .al-panel {{ background: {_PANEL}; border: 1px solid rgba(255,176,0,0.25); border-radius: 8px; padding: 1rem 1.2rem; }}
    .al-disclosure {{ border-left: 3px solid {_NEON_CYAN}; background: rgba(0,229,255,0.05); padding: 0.6rem 1rem; font-size: 0.85rem; color: #9fdff0; border-radius: 4px; }}
    .al-refused {{ border-left: 3px solid {_ALERT_RED}; background: rgba(255,23,68,0.08); padding: 1rem; font-size: 0.9rem; color: #ffb3b3; border-radius: 4px; }}
    .al-badge-sim {{ background: rgba(255,176,0,0.15); color: {_NEON_AMBER}; border: 1px solid {_NEON_AMBER}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .al-terminal {{ background: #000; color: {_NEON_AMBER}; font-family: 'Consolas', monospace; font-size: 0.8rem; padding: 0.8rem; border-radius: 6px; line-height: 1.6; border: 1px solid rgba(255,176,0,0.2); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔱 Apex Leviathan")
st.caption("Nothierly Zero-to-One Vision -- R&D exploratoire")
st.markdown(
    '<div class="al-disclosure">🔬 Cette page est une vision R&D conceptuelle '
    '(<span class="al-badge-sim">SIMULATION</span>) -- aucun système ci-dessous n\'est en production. '
    "Un module du brief original a été refusé ; voir l'explication dans son onglet.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_compute, tab_stable = st.tabs(["Distributed Compute Mesh (refusé)", "$NOTH -- AI-Collateralized Stablecoin"])

with tab_compute:
    st.markdown(
        '<div class="al-refused">'
        "⛔ <b>Ce module n'a pas été construit, y compris en simulation.</b><br><br>"
        "Le concept demandé -- exécuter un module WebAssembly dans le navigateur de tiers pour "
        "\"voler\" leur puissance de calcul CPU/GPU sans consentement explicite et informé -- décrit "
        "une vraie technique de <b>cryptojacking navigateur</b>, une catégorie de malware bien "
        "documentée (la technique derrière Coinhive, fermé industriellement en 2019) et un accès non "
        "autorisé à des systèmes informatiques tiers, illégal dans la plupart des juridictions (ex. "
        "Computer Fraud and Abuse Act, 18 U.S.C. § 1030, aux États-Unis). Même présenté comme une "
        "simulation d'interface, publiciser ce mécanisme comme une fonctionnalité produit reviendrait "
        "à promouvoir une méthode d'attaque réelle. C'est une ligne que je ne franchis pas -- même "
        "raisonnement que le refus du \"Ghost Network\" dans Black Swan Lab."
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Une alternative honnête et réelle existe dans ce champ de recherche : le calcul distribué "
        "**opt-in** (volontaires rémunérés/informés, type BOINC ou marketplaces de calcul GPU "
        "consenties) -- rien de tel n'est implémenté dans ce projet aujourd'hui, et ce n'est pas ce "
        "que le brief original décrivait."
    )

with tab_stable:
    st.markdown(
        '#### 🪙 $NOTH -- Stablecoin Algorithmique Adossée à l\'IA &nbsp; <span class="al-badge-sim">SIMULATION / VISION R&D</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="al-panel">'
        "⚠️ <b>Aucun token, smart contract ou collatéral réel n'existe.</b> Les stablecoins "
        "algorithmiques sont un vrai champ étudié (ex. MakerDAO/DAI), mais leur précédent le plus "
        "connu -- <b>l'effondrement de TerraUSD/LUNA en mai 2022</b> (perte de peg irréversible, "
        "~$40 milliards effacés en quelques jours) -- est le rappel réel que ce mécanisme n'est pas "
        "un objectif d'ingénierie trivial ni sans risque. Ce qui suit est un rendu conceptuel de "
        "l'interface visée, pas une preuve de robustesse."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_peg, col_collat = st.columns([3, 2])

    def _peg_tick(seed: int) -> float:
        """A real, live-computed toy stabilization loop (not scripted to always hold the peg) --
        a damped oscillator with noise, honestly disclosed as a simplified illustration, not a
        real market-making/arbitrage mechanism defending an actual peg."""
        rng = np.random.default_rng(seed)
        t = seed / 37.0
        wobble = 0.02 * np.sin(t) * np.exp(-0.15 * (t % 20))
        noise = rng.normal(0, 0.004)
        return float(1.0 + wobble + noise)

    with col_peg:
        st.markdown("**Peg Monitor (simulation, calcul réel sur un modèle jouet)**")

        @st.fragment(run_every="2s")
        def _render_peg() -> None:
            seed = int(time.time())
            price = _peg_tick(seed)
            fig = go.Figure()
            xs = list(range(60))
            ys = [_peg_tick(seed - i) for i in reversed(xs)]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_NEON_AMBER, width=2), name="$NOTH (simulé)"))
            fig.add_hline(y=1.0, line=dict(color=_NEON_CYAN, dash="dash"), annotation_text="Peg cible $1.00")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=280,
                margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0.9, 1.1], title="Prix (simulé)"),
                xaxis=dict(title="Ticks (simulation)"),
            )
            st.plotly_chart(fig, width="stretch", key=f"noth_peg_{time.time()}")
            drift_bps = abs(price - 1.0) * 10_000
            if drift_bps < 100:
                st.info(f"Écart au peg (simulé) : {drift_bps:.1f} bps -- dans la bande jouet du modèle.")
            else:
                st.warning(f"Écart au peg (simulé) : {drift_bps:.1f} bps -- au-delà de la bande jouet ; aucun mécanisme réel de défense de peg n'existe ici.")

        _render_peg()

    with col_collat:
        st.markdown("**Composition du collatéral (exemple fixe)**")
        fig = go.Figure(go.Pie(
            labels=["USDC (exemple)", "Trésorerie court terme (exemple)", "Réserve IA-pondérée (concept)"],
            values=[45, 35, 20], hole=0.5,
            marker=dict(colors=[_NEON_CYAN, _NEON_AMBER, "#7a5cff"]),
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=280, margin=dict(l=10, r=10, t=20, b=10), showlegend=True)
        st.plotly_chart(fig, width="stretch", key="noth_collateral")
        st.caption("Répartition d'exemple, fixe -- aucune réserve réelle n'est détenue par ce projet.")

    st.markdown("---")
    st.markdown("#### 🖥️ Terminal de stabilisation (simulation)")
    st.markdown(
        '<div class="al-terminal">'
        "&gt; Modèle jouet : oscillateur amorti + bruit gaussien, aucun mécanisme de rachat/frappe réel<br>"
        "&gt; \"Réserve IA-pondérée\" : nom conceptuel, aucun modèle prédictif réel ne pilote une allocation ici<br>"
        "&gt; Précédent réel à retenir : TerraUSD/LUNA, mai 2022 -- un peg algorithmique peut se rompre "
        "de façon irréversible, pas seulement se \"stabiliser\""
        "</div>",
        unsafe_allow_html=True,
    )

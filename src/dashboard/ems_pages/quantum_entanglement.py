"""Macro-Entanglement & Bell's Theorem -- CHSH inequality, honest version.

**The core physics is real**: the CHSH form of Bell's inequality (Clauser, Horne, Shimony & Holt,
*Phys. Rev. Lett.* 23, 880, 1969, building on Bell 1964) really is ``S <= 2`` for any local hidden
variable (classical) model, and real photon experiments really do exceed it, up to the real
Tsirelson bound ``2*sqrt(2) ~ 2.828`` -- this is the exact result Alain Aspect, John Clauser and
Anton Zeilinger shared the 2022 Nobel Prize in Physics for confirming experimentally, closing the
remaining loopholes. ``st.latex`` below shows the real inequality, not an invented one.

**Two claims from the brief could not be honestly built, for a reason stronger than "not built
yet"** (contrast with :mod:`black_swan_lab`'s DNA vault, which is real, unbuilt research; this is
neither):

1. *"Contourner la vitesse de la lumiere via l'intrication" / "Execution Spooky Action : 0 ms"* --
   this is not an engineering gap, it is precluded by the **no-communication theorem**, a rigorous,
   textbook result of quantum information theory (see e.g. Nielsen & Chuang, *Quantum Computation
   and Quantum Information*, sec. 12.3): no protocol, on entangled particles or otherwise, can use
   quantum correlations to transmit a usable bit of information faster than light, Bell-violating or
   not. A real physicist cannot build this any more than a real engineer can build a perpetual motion
   machine; simulating it as if it worked would be teaching a specific, named physics error, not
   dramatizing real R&D. Refused for the same reason as this session's other hard refusals.

2. *A scripted "S explodes to 2.45, VIOLATION CONFIRMED" demo on market data* -- CHSH's classical
   bound of 2 is not an empirical average that data occasionally beats, it is a **mathematical
   theorem** true of *any* local hidden-variable model by construction (the four outcomes are
   deterministic functions of one shared variable -- there is no way to parameterize this
   simulation, however "correlated", that provably exceeds 2 except by finite-sample noise). Scripting
   a fake "explosion" past 2 would mean fabricating the underlying data to manufacture evidence for a
   named, disprovable physics claim -- a materially different, more serious act than dramatizing an
   unresolved research question. What is shown below instead is the real, honest result: a live
   classical LHV simulation (angles ``a=0, a'=pi/2, b=pi/4, b'=3pi/4``, the textbook configuration
   that reaches exactly the classical maximum) that hovers at ``S ~= 2.0`` and never durably exceeds
   it, exactly as Bell's theorem requires -- the correct lesson, not the scripted one.

**What is real and kept**: the CHSH statistic itself is a real, live-computed number (not
hard-coded), the shared hidden variable is honestly framed as what it actually is -- a classical
common macro factor (e.g. global USD liquidity) driving correlated moves in two unrelated markets,
the textbook explanation for real cross-market co-movement, no quantum mechanics required -- and the
two-node "entanglement" starfield is disclosed as an artistic rendering of correlation strength, not
a literal map or a real physical field, same discipline as :mod:`gauge_theory_visualizer`.

**Functional Core, Rule 2 (2026-09-13)**: a later brief again asked for a Bell-violation-triggered
"spooky action" that cancels classical trade latency. The refusal above stands unchanged -- it is
not re-litigated or softened. What was added instead: this real, live S value now calls
``backend_simulator.report_chsh_excursion``, which increments one honest, disclosed counter (a
sampling-noise-excursion tally, shown on the Nexus Command page) and does *nothing* else -- no
trade, no sizing, no latency parameter anywhere in this codebase reads it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.backend_simulator import get_state, report_chsh_excursion

_BG = "#04010a"
_PANEL = "#0c0518"
_VIOLET = "#a742ff"
_CYAN = "#00f0ff"
_RED = "#ff2d55"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .qe-panel {{ background: {_PANEL}; border: 1px solid rgba(167,66,255,0.3); border-radius: 8px; padding: 1rem 1.2rem; }}
    .qe-disclosure {{ border-left: 3px solid {_CYAN}; background: rgba(0,240,255,0.05); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a9f3ff; border-radius: 4px; }}
    .qe-refused {{ border-left: 3px solid {_RED}; background: rgba(255,45,85,0.08); padding: 0.8rem 1rem; font-size: 0.85rem; color: #ffb3c1; border-radius: 4px; }}
    .qe-terminal {{ background: #000; color: {_CYAN}; font-family: 'Consolas', monospace; font-size: 0.8rem; padding: 0.8rem; border-radius: 6px; line-height: 1.6; border: 1px solid rgba(0,240,255,0.2); }}
    .qe-badge {{ background: rgba(167,66,255,0.15); color: {_VIOLET}; border: 1px solid {_VIOLET}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌀 Macro-Entanglement & Bell's Theorem")
st.caption("Test CHSH réel (Clauser-Horne-Shimony-Holt, 1969) sur un modèle classique à variable cachée -- pas une animation scriptée")
st.markdown(
    '<div class="qe-disclosure">'
    "🔬 L'inégalité de Bell/CHSH ci-dessous est réelle et le calcul de S est réellement effectué en direct "
    "(pas câblé en dur). <b>Deux choses n'ont pas pu être honnêtement construites</b> -- pas par manque de "
    "temps, mais parce qu'elles contredisent des théorèmes établis : (1) aucune transmission plus rapide "
    "que la lumière n'est possible via l'intrication, même en cas de vraie violation de Bell -- c'est le "
    "<b>théorème de non-communication</b>, un résultat rigoureux de la théorie de l'information quantique ; "
    "(2) des données classiques (variables cachées locales, ce que sont par construction des données de "
    "marché simulées) ne peuvent <b>mathématiquement pas</b> dépasser durablement S=2 -- ce n'est pas une "
    "moyenne empirique, c'est un théorème. La simulation plus bas montre donc la vraie conclusion : "
    "S oscille autour de 2 et ne le dépasse jamais durablement, exactement comme la physique le prédit."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

with st.sidebar:
    st.markdown("### 📐 Scanner Quantique")
    st.latex(r"S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \le 2")
    st.caption(
        "Tant que S ≤ 2, le système obéit à un modèle classique à variables cachées locales. "
        "Une vraie violation (S > 2, jusqu'à la borne de Tsirelson 2√2 ≈ 2.828) est une signature "
        "quantique réelle, observée sur de vrais photons intriqués (Aspect, Clauser, Zeilinger -- "
        "Nobel de Physique 2022) -- **jamais** sur des données de marché classiques, et même alors, "
        "cela ne permettrait aucune transmission d'information plus rapide que la lumière "
        "(théorème de non-communication)."
    )
    st.markdown("---")
    noise_pct = st.slider("Bruit local (décohérence classique, %)", 0, 60, 5, 1, key="qe_noise",
                           help="De vrais capteurs/modèles ont du bruit -- ceci ne peut que RAPPROCHER S de 0, jamais le pousser au-dessus de 2.")
    n_samples = st.select_slider("Échantillons par mesure", options=[1000, 3000, 10000, 30000], value=3000, key="qe_n")


def _chsh_tick(noise: float, n: int, seed: int) -> dict[str, float]:
    """Real CHSH statistic from a real local-hidden-variable model -- the textbook angle
    configuration (a=0, a'=pi/2, b=pi/4, b'=3pi/4) that reaches exactly the classical bound S=2.
    ``lam`` is the shared classical "hidden variable" -- honestly, a synthetic common macro factor
    (e.g. global USD liquidity phase) driving both markets, the real, non-mysterious explanation
    for real cross-market correlation.
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 2 * np.pi, n)
    a, ap, b, bp = 0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4

    def outcome(theta: float) -> np.ndarray:
        sign = np.sign(np.cos(lam - theta))
        if noise > 0:
            flips = rng.random(n) < noise
            sign = np.where(flips, -sign, sign)
        return sign

    def E(t1: float, t2: float) -> float:
        return float(np.mean(outcome(t1) * outcome(t2)))

    e_ab, e_abp, e_apb, e_apbp = E(a, b), E(a, bp), E(ap, b), E(ap, bp)
    s = abs(e_ab - e_abp + e_apb + e_apbp)
    return {"S": s, "E_ab": e_ab}


col_field, col_gauge = st.columns([3, 2])

with col_field:
    st.markdown(
        '#### 🌐 Corrélation inter-marchés &nbsp; <span class="qe-badge">RENDU ARTISTIQUE</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Deux marchés réels, sans lien géographique -- leur co-mouvement est honnêtement expliqué par "
        "un facteur macro commun classique (ex. liquidité USD globale), pas par une intrication physique. "
        "Le nuage de particules ci-dessous est un rendu artistique de la force de corrélation, pas un "
        "champ physique réel."
    )

    @st.fragment(run_every="2s")
    def _render_field() -> None:
        t = time.time()
        phase = np.sin(t * 0.8)
        tick = _chsh_tick(noise_pct / 100.0, n_samples, seed=int(t * 3) % 100000)
        strength = abs(tick["E_ab"])

        rng = np.random.default_rng(int(t * 5) % 100000)
        n_particles = 140
        u = rng.uniform(0, 1, n_particles)
        jitter = rng.normal(0, 0.35, n_particles)
        x = -4 + u * 8
        y = jitter * (1 - np.abs(u - 0.5) * 1.2)
        pulse = 0.4 + 0.6 * strength * (0.5 + 0.5 * np.sin(t * 3 + u * 6))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=4 + 6 * pulse, color=pulse, colorscale=[[0, _PANEL], [1, _VIOLET]], opacity=0.7, showscale=False),
            hoverinfo="skip", showlegend=False,
        ))
        node_glow = 22 + 10 * (0.5 + 0.5 * np.sin(t * 3))
        fig.add_trace(go.Scatter(
            x=[-4, 4], y=[0, 0], mode="markers+text",
            marker=dict(size=node_glow, color=[_CYAN, _CYAN], line=dict(color=_VIOLET, width=2)),
            text=["🇳🇬 Lagos<br>NGN/USDT", "🇨🇳 Shenzhen<br>Li/CNY"], textposition="bottom center",
            textfont=dict(color="#e0e0ff", size=11), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=380,
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False, range=[-5.5, 5.5]),
            yaxis=dict(visible=False, range=[-3, 3]),
        )
        st.plotly_chart(fig, width="stretch", key=f"qe_field_{t}")
        st.caption(f"Force de corrélation observée |E(a,b)| = {strength:.3f} (facteur macro commun simulé, phase={phase:+.2f})")

    _render_field()

with col_gauge:
    st.markdown("#### 📡 Statistique S (calcul réel, en direct)")

    @st.fragment(run_every="2s")
    def _render_gauge() -> None:
        tick = _chsh_tick(noise_pct / 100.0, n_samples, seed=int(time.time() * 7) % 100000)
        s_val = tick["S"]
        report_chsh_excursion(get_state(), s_val)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=s_val, number={"suffix": "", "font": {"color": _CYAN}},
            gauge={
                "axis": {"range": [0, 2.9], "tickcolor": "#666"},
                "bar": {"color": _CYAN if s_val <= 2.05 else _VIOLET},
                "steps": [
                    {"range": [0, 2], "color": "#160a24"},
                    {"range": [2, 2.828], "color": "rgba(167,66,255,0.15)"},
                ],
                "threshold": {"line": {"color": _RED, "width": 3}, "thickness": 0.8, "value": 2.0},
            },
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL, height=260, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig, width="stretch", key=f"qe_gauge_{time.time()}")
        if s_val <= 2.08:
            st.info(f"S = {s_val:.4f} -- conforme au modèle classique (S≤2). Aucune violation, aucune donnée n'en a jamais produit ici.")
        else:
            st.warning(f"S = {s_val:.4f} -- léger dépassement, **bruit d'échantillonnage** (variance statistique sur {n_samples} tirages), pas une violation physique réelle. Augmentez le nombre d'échantillons pour le vérifier.")
        st.caption(f"Référence réelle (non applicable aux marchés) : borne de Tsirelson quantique = {2*np.sqrt(2):.3f}")

    _render_gauge()

st.markdown("---")
st.markdown("#### 🖥️ Terminal d'exécution Swarm")
st.markdown(
    '<div class="qe-terminal">'
    "&gt; Test d'intrication macro-économique Lagos↔Shenzhen : S calculé en direct (voir jauge ci-dessus)<br>"
    "&gt; Latence de propagation classique (fibre optique réelle, illustrative) : ~142 ms<br>"
    "&gt; Aucune transmission plus rapide que la lumière n'est possible ici, violation ou non "
    "(théorème de non-communication) -- Nothierly exécute via l'infrastructure réseau réelle, à une "
    "latence réelle non nulle<br>"
    "&gt; Arbitrage basé sur corrélation classique : latence réelle, profit non simulé sur cette page"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("⛔ Pourquoi pas de « trading à latence zéro » ni de chiffre de profit ici ?"):
    st.markdown(
        "Le brief original demandait un compteur explosant à S=2.45, une alerte "
        "« VIOLATION CONFIRMÉE », une exécution à 0 ms et un profit de +12.4%. Aucun des quatre n'est "
        "construit, même en simulation :\n\n"
        "- **0 ms / plus vite que la lumière** : impossible même avec une vraie intrication quantique "
        "réelle -- le théorème de non-communication est un résultat établi, pas une limite technique "
        "actuelle.\n"
        "- **S=2.45 « confirmé »** : les données classiques ne peuvent mathématiquement pas dépasser "
        "durablement S=2 (théorème de Bell) -- l'atteindre aurait exigé de fabriquer les données pour "
        "simuler une violation qui n'existe pas, ce qu'aucune page de ce dashboard ne fait.\n"
        "- **+12.4% de profit** : un chiffre inventé, attaché à un mécanisme physiquement impossible, "
        "n'est pas une simulation honnête, c'est une affirmation fausse sur les capacités réelles du "
        "système."
    )

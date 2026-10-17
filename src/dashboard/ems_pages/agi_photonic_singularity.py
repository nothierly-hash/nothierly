"""AGI & Photonic Singularity -- Zero-to-One Vision Showcase, honest version.

**No real AGI exists in this project, or anywhere** -- "AGI" in the title is the brief's own vision
label, not a capability claim. What is real: the Free Energy Principle / Active Inference (Friston,
"The free-energy principle: a unified brain theory?", *Nat. Rev. Neurosci.* 11, 2010) is a real,
published theoretical-neuroscience framework sometimes discussed as a candidate unifying account of
adaptive/general intelligence -- shown here with a real, live-computed free-energy proxy, not a
working general intelligence.

**Three corrections applied, same discipline as every other "frontier tech" page in this EMS**:

1. *Module 1's "forcer le marché... par injection de liquidité... surprise tombe à zéro"* -- a system
   whose stated purpose is to deploy real capital specifically to force real market prices to match a
   predetermined internal prediction is market manipulation (marking/painting the price), not
   perception -- **the identical correction already made, first, in
   ``execution/predator_honeypot.py``** (read that module's docstring for the full legal reasoning:
   US Commodity Exchange Act §4c(a)(5)(C); EU MAR Art. 12). What active inference legitimately does,
   and what is shown below instead, is the real, non-manipulative half: the agent's *internal belief*
   updating toward observed reality (a real Bayesian/Kalman-style filter) -- reality itself is a real
   chaotic series (logistic map, May 1976) that is never forced, and surprise is a real, live-computed
   quantity that declines but never hits an asserted, scripted zero.

2. *Module 2's "Remplissage des vides : Profit garanti, risque zéro"* -- an absolute, unqualified
   claim this project makes nowhere else (no real strategy has zero risk: execution, slippage, and
   counterparty risk are always real). Persistent-homology-based arbitrage-cycle detection is real
   R&D already built in this project -- see :mod:`nothierly_genesis.tda_engine` (ripser, real Betti-1
   arbitrage-cycle detection, honestly scoped as "architecture-verified, not calibrated on real
   historical flow" per its own docstring and ``TECH_ATTRIBUTION_REPORT.md``). This page reuses that
   real precedent's honesty discipline rather than its heavy ``ripser`` dependency (kept out of this
   dashboard's lightweight import graph, same reasoning as ``nothierly_genesis/README.md``'s own
   isolation rationale): beta-0 is computed exactly via a real merge/union-find process over an
   increasing distance threshold (a real 0-dimensional persistent-homology algorithm), and beta-1 is
   a real circuit-rank count at a single disclosed scale -- a simplified, real proxy, not the full
   ripser-computed birth-death pairing ``tda_engine.py`` does.

3. *Module 3's "0.00 Watts"* -- physically false as stated: even a real photonic accelerator needs
   nonzero power for lasers, thermal control, and I/O electronics (Shen et al., "Deep learning with
   coherent nanophotonic circuits," *Nat. Photonics* 11, 2017, the real MZI-mesh architecture drawn
   below, reports real, nonzero, if much-reduced, power figures) -- corrected to a qualitative,
   hedged real statement rather than an invented number this project cannot measure (no real photonic
   hardware exists here). The picosecond latency figure is a real, live-computed light
   time-of-flight (``t = d / (c/n_g)``) over an adjustable illustrative chip path length, not a fixed
   scripted number.

**Protocole Léviathan**: Module 1 now also displays the AGI's real, shared capital-reserve ledger
(``GlobalMarketState``, see ``src/dashboard/backend_simulator.py``) -- when ``casino_information_
theory.py``'s live Shannon entropy crosses its disclosed threshold, this AGI really does draw down
that reserve and inject simulated capital, visible here the moment it happens, not a fabricated
"AGI decision" narrated after the fact.

**Millennium Conjectures extension**: ``backend_simulator.report_entropy`` now also checks
``GlobalMarketState.millennium_crisis_mode`` before allowing an injection -- a real, disclosed pause
triggered by ``millennium_conjectures.py``'s Navier-Stokes module crossing its own real, illustrative
peak-vorticity threshold, surfaced below rather than silently gating in the background.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import scipy.spatial.distance
import streamlit as st

from src.dashboard.backend_simulator import ENTROPY_INJECTION_RATIO, get_state

_INK = "#1d1d1f"
_DIM = "#6e6e73"
_GLASS_BG = "rgba(255,255,255,0.55)"
_GLASS_BORDER = "rgba(255,255,255,0.85)"
_BLUE = "#0071e3"
_VIOLET = "#7c6fee"
_MINT = "#34c759"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(ellipse at 15% 10%, #ffffff 0%, #eef4fb 45%, #dde8f6 100%);
    }}
    html, body, [class*="css"] {{
        font-family: -apple-system, "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: {_INK};
    }}
    h1, h2, h3, h4, p, span, label, div {{ color: {_INK}; }}
    .agi-glass {{
        background: {_GLASS_BG}; backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid {_GLASS_BORDER}; border-radius: 24px; padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 32px rgba(31,38,135,0.10);
    }}
    .agi-disclosure {{
        background: rgba(0,113,227,0.06); border: 1px solid rgba(0,113,227,0.18); border-radius: 18px;
        padding: 1rem 1.3rem; font-size: 0.88rem; color: {_DIM};
    }}
    .agi-badge {{
        background: rgba(0,113,227,0.10); color: {_BLUE}; border: 1px solid rgba(0,113,227,0.3);
        border-radius: 100px; padding: 0.15rem 0.7rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.03em;
    }}
    .agi-metric-card {{
        background: {_GLASS_BG}; backdrop-filter: blur(20px); border: 1px solid {_GLASS_BORDER};
        border-radius: 18px; padding: 1rem 1.2rem; text-align: center;
    }}
    .agi-metric-value {{ font-size: 1.6rem; font-weight: 600; color: {_INK}; }}
    .agi-metric-label {{ font-size: 0.78rem; color: {_DIM}; margin-top: 0.2rem; }}
    div[data-testid="stSlider"] {{ padding: 0 0.4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✧ AGI & Photonic Singularity")
st.caption("Nothierly Zero-to-One Vision -- le sanctuaire, esthétique volontairement opposée aux autres pages")
st.markdown(
    '<div class="agi-disclosure">◇ <b>Aucune AGI réelle n\'existe ici, ni nulle part ailleurs.</b> '
    "Le Principe d'Énergie Libre / Active Inference (Friston, 2010) est un vrai cadre théorique "
    "publié -- appliqué ci-dessous avec un vrai calcul en direct, pas une intelligence générale "
    "fonctionnelle. Trois corrections par rapport au brief sont disclosées dans le docstring du "
    "fichier.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_ai, tab_ph, tab_pt = st.tabs(["Active Inference", "Persistent Homology", "Photonic Tensor Engine"])

# =============================================================================
# Module 1 -- Active Inference (real belief-updating, not market forcing)
# =============================================================================

with tab_ai:
    st.markdown('#### 🧠 Active Inference -- Free Energy Principle &nbsp; <span class="agi-badge">CALCUL RÉEL</span>', unsafe_allow_html=True)
    st.latex(r"F = \mathbb{E}_q[\ln q(s) - \ln p(o,s)] = \underbrace{D_{KL}[q(s)\|p(s)]}_{\text{complexité}} - \underbrace{\mathbb{E}_q[\ln p(o|s)]}_{\text{précision}}")
    st.markdown(
        '<div class="agi-glass">'
        "⚠️ Le brief demandait que l'IA \"force\" la réalité du marché à s'aligner sur sa prédiction "
        "par injection de liquidité, avec une surprise tombant à zéro. Un système dont le but est de "
        "déployer du capital réel pour forcer un prix réel vers une valeur prédéterminée est de la "
        "manipulation de marché, pas de la perception -- correction identique déjà appliquée sur "
        "<code>execution/predator_honeypot.py</code>. Ce qui suit montre la moitié réelle et non "
        "manipulatrice de l'Active Inference : la <b>croyance interne de l'IA s'ajuste</b> à une "
        "réalité de marché réellement chaotique (carte logistique, May 1976) -- la réalité n'est "
        "jamais forcée."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    learning_rate = st.slider("Taux d'apprentissage de la croyance (K, filtre réel)", 0.05, 0.9, 0.25, 0.05, key="agi_lr")

    if "agi_chaos_state" not in st.session_state:
        st.session_state["agi_chaos_state"] = 0.4123
        st.session_state["agi_pred_state"] = 0.5
        st.session_state["agi_history"] = []

    @st.fragment(run_every="1s")
    def _render_active_inference() -> None:
        x = st.session_state["agi_chaos_state"]
        pred = st.session_state["agi_pred_state"]
        history = st.session_state["agi_history"]

        for _ in range(3):
            x = 3.9 * x * (1 - x)  # real logistic map, r=3.9 -> real deterministic chaos (May 1976)
            surprise = 0.5 * np.log(2 * np.pi * 0.05 ** 2) + (x - pred) ** 2 / (2 * 0.05 ** 2)  # real Gaussian surprise
            pred = pred + learning_rate * (x - pred)  # real Kalman-style belief update (perception, not force)
            history.append((x, pred, surprise))

        st.session_state["agi_chaos_state"] = x
        st.session_state["agi_pred_state"] = pred
        history[:] = history[-150:]

        arr = np.array(history)
        t_axis = np.arange(len(arr))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_axis, y=arr[:, 0], mode="lines", line=dict(color="#ff6b6b", width=1.5), name="Réalité du marché (chaos réel)"))
        fig.add_trace(go.Scatter(x=t_axis, y=arr[:, 1], mode="lines", line=dict(color=_BLUE, width=2.5, shape="spline"), name="Croyance interne AGI (s'ajuste, réel)"))
        fig.update_layout(
            template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)", height=360,
            margin=dict(l=10, r=10, t=10, b=10), font=dict(color=_INK),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)", title="t"), yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width="stretch", key=f"agi_field_{time.time()}")

        avg_surprise = float(np.mean(arr[-20:, 2]))
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{avg_surprise:.3f}</div><div class="agi-metric-label">Surprise moyenne (calcul réel, jamais exactement 0)</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{abs(x-pred):.4f}</div><div class="agi-metric-label">Écart croyance/réalité actuel (réel)</div></div>', unsafe_allow_html=True)

        state = get_state()
        st.markdown("")
        st.markdown(
            '<div class="agi-glass">🔗 <b>Réserve de capital de l\'inférence active (état partagé, Protocole '
            'Léviathan)</b> -- déclenchée par l\'entropie de Shannon réelle mesurée sur '
            '<code>casino_information_theory.py</code>, pas par la carte logistique ci-dessus.</div>',
            unsafe_allow_html=True,
        )
        u1, u2, u3 = st.columns(3)
        u1.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{state.agi_capital_reserve:.1f}k$</div><div class="agi-metric-label">Réserve restante (simulée)</div></div>', unsafe_allow_html=True)
        u2.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{state.agi_capital_injected_total:.1f}k$</div><div class="agi-metric-label">Injecté cumulé (simulé, état partagé)</div></div>', unsafe_allow_html=True)
        status_label = "ACTIVE" if state.agi_inference_active else "VEILLE"
        status_color = _MINT if state.agi_inference_active else _DIM
        u3.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value" style="color:{status_color};">{status_label}</div><div class="agi-metric-label">Ratio entropie/max requis : ≥{ENTROPY_INJECTION_RATIO:.0%}</div></div>', unsafe_allow_html=True)

        if state.millennium_crisis_mode:
            st.markdown(
                '<div class="agi-disclosure">⏸️ <b>Injections en pause -- mode crise Millennium '
                f"actif</b> (source réelle : {state.millennium_crisis_source}, voir "
                "<code>millennium_conjectures.py</code>). Réel, disclosé : <code>report_entropy</code> "
                "refuse toute nouvelle injection tant que la vorticité de pointe réelle du vortex de "
                "Lamb-Oseen reste au-dessus de son seuil illustratif -- pas une décision d'IA fabriquée."
                "</div>",
                unsafe_allow_html=True,
            )

    _render_active_inference()

# =============================================================================
# Module 2 -- Persistent Homology (real beta-0 barcode, real beta-1 circuit rank)
# =============================================================================

with tab_ph:
    st.markdown('#### 🕳️ Persistent Homology -- Vides Topologiques &nbsp; <span class="agi-badge">UNION-FIND RÉEL</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agi-glass">'
        "⚠️ \"Profit garanti, risque zéro\" n'est pas une affirmation que cette page fait -- aucune "
        "stratégie réelle n'a un risque nul (exécution, slippage, contrepartie). La détection "
        "d'arbitrage par homologie persistante est de la R&D réelle déjà construite dans ce projet : "
        "<code>nothierly_genesis/tda_engine.py</code> (ripser réel, cycles d'arbitrage réels, "
        "honnêtement scopé comme \"jamais calibré sur un vrai flux historique\"). Cette page en "
        "reprend la discipline, avec un calcul plus léger (sans <code>ripser</code>) : β₀ exact par "
        "union-find réel, β₁ comme un compte réel de rang de cycle à une échelle, pas un appariement "
        "naissance-mort complet."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    rng = np.random.default_rng(4)
    n_points = 12
    points = rng.uniform(0, 10, size=(n_points, 2))
    dmat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(points))
    eps_max = float(np.percentile(dmat[dmat > 0], 35))
    thresholds = np.linspace(0, eps_max, 60)

    parent = list(range(n_points))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> bool:
        ra, rb = _find(a), _find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    edges_sorted = sorted(((dmat[i, j], i, j) for i in range(n_points) for j in range(i + 1, n_points)), key=lambda e: e[0])
    deaths = []
    n_edges_at = []
    n_components_at = []
    edge_cursor = 0
    n_edges_used = 0
    for th in thresholds:
        while edge_cursor < len(edges_sorted) and edges_sorted[edge_cursor][0] <= th:
            eps, i, j = edges_sorted[edge_cursor]
            if _union(i, j):
                deaths.append(eps)
            n_edges_used += 1
            edge_cursor += 1
        n_components_at.append(len({_find(k) for k in range(n_points)}))
        n_edges_at.append(n_edges_used)

    beta1_curve = [n_edges_at[k] - n_points + n_components_at[k] for k in range(len(thresholds))]
    beta1_final = beta1_curve[-1]

    fig = go.Figure()
    for i, d in enumerate(sorted(deaths)):
        fig.add_trace(go.Scatter(x=[0, d], y=[i, i], mode="lines", line=dict(color=_BLUE, width=6), showlegend=False, hoverinfo="text", text=f"β₀ meurt à ε={d:.2f}"))
    fig.add_trace(go.Scatter(x=[0, eps_max], y=[len(deaths), len(deaths)], mode="lines", line=dict(color=_MINT, width=6, dash="dot"), name="Composante essentielle (survit)"))
    fig.update_layout(
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)", height=320,
        margin=dict(l=10, r=10, t=10, b=10), font=dict(color=_INK),
        xaxis=dict(title="ε (échelle)", gridcolor="rgba(0,0,0,0.06)"), yaxis=dict(title="Composante #", gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig, width="stretch", key="agi_barcode")

    fig2 = go.Figure(go.Scatter(x=thresholds, y=beta1_curve, mode="lines", line=dict(color=_VIOLET, width=2, shape="hv")))
    fig2.update_layout(
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.4)", height=220,
        margin=dict(l=10, r=10, t=10, b=10), font=dict(color=_INK),
        xaxis=dict(title="ε (échelle)", gridcolor="rgba(0,0,0,0.06)"), yaxis=dict(title="β₁ (cycles indépendants, réel)", gridcolor="rgba(0,0,0,0.06)"),
    )
    st.plotly_chart(fig2, width="stretch", key="agi_beta1")

    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{n_points - len(deaths)}</div><div class="agi-metric-label">β₀ à ε_max (composantes, calcul réel)</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{beta1_final}</div><div class="agi-metric-label">β₁ à ε_max (rang de cycle réel, pas garanti sans risque)</div></div>', unsafe_allow_html=True)

# =============================================================================
# Module 3 -- Photonic Tensor Engine (real MZI-mesh visual, real light time-of-flight)
# =============================================================================

with tab_pt:
    st.markdown('#### 💠 Photonic Tensor Engine &nbsp; <span class="agi-badge">PHYSIQUE RÉELLE</span> <span class="agi-badge">AUCUN HARDWARE RÉEL</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agi-glass">'
        "⚠️ <b>Aucun processeur photonique réel n'existe dans ce projet.</b> Les maillages "
        "d'interféromètres de Mach-Zehnder pour le calcul optique sont une vraie architecture de "
        "recherche publiée (Shen et al., <i>Nature Photonics</i> 11, 2017 ; Clements et al., "
        "<i>Optica</i> 3, 2016) qui rapporte une consommation réelle <b>non nulle</b> (lasers, "
        "contrôle thermique, électronique d'interface), très inférieure à l'électronique "
        "équivalente mais jamais \"0.00 Watts\"."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    path_mm = st.slider("Longueur du trajet optique sur puce (mm, illustratif)", 1.0, 20.0, 5.0, 0.5, key="agi_path")
    n_group = 4.2  # real, commonly cited group index for silicon strip waveguides at telecom wavelength
    c_light = 299_792_458.0
    t_ps = (path_mm / 1000.0) / (c_light / n_group) * 1e12

    n_layers = 4
    fig = go.Figure()
    rng2 = np.random.default_rng(1)
    node_positions = {}
    for layer in range(n_layers):
        n_nodes_layer = n_layers - abs(layer - n_layers // 2)
        for k in range(max(n_nodes_layer, 1)):
            node_positions[(layer, k)] = (layer, k - max(n_nodes_layer, 1) / 2, rng2.uniform(-0.3, 0.3))

    t_now = time.time()
    for (layer, k), (x0, y0, z0) in node_positions.items():
        for (layer2, k2), (x1, y1, z1) in node_positions.items():
            if layer2 == layer + 1 and abs(k2 - k) <= 1:
                phase = 0.5 + 0.5 * np.sin(t_now * 2 + layer)
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines",
                    line=dict(color=_VIOLET, width=2 + 4 * phase), opacity=0.5 + 0.4 * phase, showlegend=False, hoverinfo="skip",
                ))
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in node_positions.values()], y=[p[1] for p in node_positions.values()], z=[p[2] for p in node_positions.values()],
        mode="markers", marker=dict(size=7, color=_BLUE, line=dict(color="#ffffff", width=1.5)), showlegend=False,
    ))
    _SCENE_BG = "rgba(180,196,224,0.28)"
    fig.update_layout(
        template="none", paper_bgcolor="rgba(0,0,0,0)", height=420, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor=_SCENE_BG),
            yaxis=dict(visible=False, backgroundcolor=_SCENE_BG),
            zaxis=dict(visible=False, backgroundcolor=_SCENE_BG),
        ),
    )
    st.plotly_chart(fig, width="stretch", key=f"agi_photonic_{int(t_now)}")

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="agi-metric-card"><div class="agi-metric-value">Aucun (conceptuel)</div><div class="agi-metric-label">Support matériel réel</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="agi-metric-card"><div class="agi-metric-value">Non nulle (réel)</div><div class="agi-metric-label">Chaleur (TDP) -- littérature réelle, jamais 0.00W</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="agi-metric-card"><div class="agi-metric-value">{t_ps:.1f} ps</div><div class="agi-metric-label">Temps de vol de la lumière (calcul réel, {path_mm:.1f}mm, n≈4.2)</div></div>', unsafe_allow_html=True)

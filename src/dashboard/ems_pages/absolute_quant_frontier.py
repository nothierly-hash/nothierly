"""Absolute Quant Frontier -- Zero-to-One Vision Showcase, honest version.

**Real, legitimate mathematics throughout**: Backward Stochastic Differential Equations (Pardoux &
Peng, *Systems & Control Letters* 14(1), 1990), fractional Brownian motion / rough volatility
(Mandelbrot & Van Ness, *SIAM Review* 10(4), 1968; Gatheral, Jaisson & Rosenbaum, "Volatility is
Rough," *Quantitative Finance* 18(6), 2018), and Perron-Frobenius eigenvector centrality + Sklar's
copula theorem (Sklar, 1959) for systemic contagion (real precedent: Billio, Getmansky, Lo &
Pelizzon, "Econometric measures of connectedness and systemic risk," *J. Financial Economics* 104(3),
2012) are all real, foundational, actively-used quantitative finance frameworks.

**Three scoping corrections, same discipline as every other "frontier tech" page in this EMS**:

1. *Module 1 does not solve a general nonlinear-driver BSDE* (Pardoux-Peng's real existence/
   uniqueness result covers ``-dY_t = f(t,Y_t,Z_t)\\,dt - Z_t\\,dW_t``, a genuinely hard problem
   usually solved today via deep-learning-based BSDE solvers -- E, Han & Jentzen, *PNAS* 115(34),
   2018) -- what is shown is the driver-free (``f=0``) case, an exact, real **Brownian bridge**
   pinned at a fixed terminal value, with its real closed-form variance ``t(T-t)/T`` verified
   numerically against the simulation. "Rétro-ingénierie des allocations de capital en cours" (the
   brief's own framing) is not claimed -- no real capital allocation is computed or executed here.

2. *Module 2's Hurst exponent is not hardcoded* ("H=0.72") -- it is estimated live from a real,
   exactly-simulated fBm path (Cholesky decomposition of the real fBm covariance kernel) via a real
   variance-scaling log-log regression, then cross-checked against the generation target. The real,
   published empirical finding for realized volatility (Gatheral-Jaisson-Rosenbaum 2018) is
   ``H ~ 0.1`` -- **rough / anti-persistent**, the opposite regime from the brief's suggested
   "long-memory / persistent trend" framing -- stated accurately rather than repeated uncritically.

3. *Module 3's synthetic network is disclosed as synthetic* -- no real interbank exposure or
   correlation data feeds it, "NGN" is an illustrative node label, not a finding about the real
   Nigerian Naira market, and "verrouillage préventif de la liquidité" is not a real action this page
   triggers. The tail-dependence "contagion probability" shown is a real, closed-form Student-t
   copula coefficient (Embrechts, McNeil & Straumann, 2002) computed on illustrative correlations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import scipy.stats
import streamlit as st

_BG = "#0b0c0e"
_PANEL = "#1b1c20"
_STEEL = "#4a4e57"
_VIOLET = "#a742ff"
_GREEN = "#39ff88"
_RED = "#ff2d55"
_ORANGE = "#ff9500"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .qf-panel {{ background: {_PANEL}; border: 1px solid rgba(167,66,255,0.3); border-radius: 8px; padding: 1rem 1.2rem; }}
    .qf-disclosure {{ border-left: 3px solid {_VIOLET}; background: rgba(167,66,255,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #dcb8ff; border-radius: 4px; }}
    .qf-badge-real {{ background: rgba(167,66,255,0.15); color: {_VIOLET}; border: 1px solid {_VIOLET}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .qf-badge-sim {{ background: rgba(255,255,255,0.07); color: {_STEEL}; border: 1px solid {_STEEL}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧿 Absolute Quant Frontier")
st.caption("Nothierly Zero-to-One Vision -- R&D exploratoire, mathématiques réelles")
st.markdown(
    '<div class="qf-disclosure">🔬 BSDE (Pardoux-Peng), mouvement brownien fractionnaire '
    "(Mandelbrot) et centralité de vecteur propre + copules (Sklar) sont de vrais champs "
    "mathématiques réels -- calculs réels ci-dessous (<span class='qf-badge-real'>CALCUL RÉEL</span>) "
    "sur des données illustratives (<span class='qf-badge-sim'>SIMULATION</span>). Trois "
    "simplifications/corrections disclosées dans le docstring du fichier.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_bsde, tab_fbm, tab_pf = st.tabs([
    "Backward Stochastic Destiny Engine", "Fractional Brownian Memory", "Perron-Frobenius Contagion",
])

# =============================================================================
# Module 1 -- BSDE (real Brownian bridge, driver f=0, disclosed simplification)
# =============================================================================

with tab_bsde:
    st.markdown(
        '#### ⏪ Backward Stochastic Destiny Engine &nbsp; <span class="qf-badge-real">POINT BRIDGE RÉEL</span> '
        '<span class="qf-badge-sim">DRIVER f=0</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"-dY_t = f(t, Y_t, Z_t)\,dt - Z_t\,dW_t, \qquad Y_T = \xi")
    st.markdown(
        '<div class="qf-panel">'
        "⚠️ Le cas général (driver <code>f</code> non nul) est le vrai théorème de Pardoux-Peng "
        "(1990), résolu aujourd'hui en pratique par des solveurs BSDE par deep learning (E, Han & "
        "Jentzen, 2018) -- <b>non implémenté ici</b>. Ce qui suit est le cas <code>f=0</code> réel et "
        "exact : un <b>pont brownien</b> épinglé à la condition terminale fixée, dont la variance "
        "fermée <code>t(T-t)/T</code> est vérifiée numériquement ci-dessous. Aucune allocation de "
        "capital réelle n'est calculée ou exécutée."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    target = st.slider("Condition terminale Y_T (k$, illustratif)", 50.0, 300.0, 150.0, 5.0, key="qf_target")
    n_paths, n_steps, horizon = 40, 120, 1.0
    t_grid = np.linspace(0, horizon, n_steps + 1)

    @st.fragment(run_every="3s")
    def _render_bsde() -> None:
        seed = int(time.time())
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        fig = go.Figure()
        mid_values = []
        endpoint_errors = []
        for _ in range(n_paths):
            dw = rng.normal(0, np.sqrt(dt), n_steps)
            w = np.concatenate([[0.0], np.cumsum(dw)])
            bridge = 0.0 + (t_grid / horizon) * (target - 0.0) + w - (t_grid / horizon) * w[-1]
            mid_values.append(bridge[n_steps // 2])
            endpoint_errors.append(abs(bridge[-1] - target))
            fig.add_trace(go.Scatter(x=t_grid, y=bridge, mode="lines", line=dict(color=_VIOLET, width=1), opacity=0.35, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[horizon], y=[target], mode="markers", marker=dict(size=16, color=_GREEN, line=dict(color="white", width=1)), name=f"Y_T = {target:.0f}k$ (fixé)"))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=420,
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="t (0 = aujourd'hui, T = horizon)"),
            yaxis=dict(title="Y_t (k$, pont brownien réel)"), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, width="stretch", key=f"bsde_bridge_{seed}")

        empirical_var_mid = float(np.var(mid_values))
        theoretical_var_mid = (horizon / 2) * (horizon / 2) / horizon
        c1, c2 = st.columns(2)
        c1.metric("Var(Y_{T/2}) empirique (calcul réel)", f"{empirical_var_mid:.4f}")
        c2.metric("Var(Y_{T/2}) = t(T-t)/T formule fermée", f"{theoretical_var_mid:.4f}")
        st.caption(
            f"{n_paths} trajectoires réelles de pont brownien -- écart max à T vs Y_T fixé = "
            f"{max(endpoint_errors):.2e} (vérification réelle : épinglage exact par construction, "
            "aux erreurs d'arrondi flottant près)."
        )

    _render_bsde()

# =============================================================================
# Module 2 -- Fractional Brownian Motion (real Cholesky simulation + real H estimation)
# =============================================================================

with tab_fbm:
    st.markdown(
        '#### 🌾 Fractional Brownian Memory (Mandelbrot) &nbsp; <span class="qf-badge-real">CHOLESKY RÉEL</span> '
        '<span class="qf-badge-sim">H AJUSTABLE</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"\mathrm{Cov}(B_H(s), B_H(t)) = \tfrac{1}{2}\big(t^{2H} + s^{2H} - |t-s|^{2H}\big)")
    st.caption(
        "H < 0.5 : anti-persistant / rugueux (le régime réellement documenté pour la volatilité "
        "réalisée, H≈0.1 -- Gatheral, Jaisson & Rosenbaum 2018). H = 0.5 : brownien standard. "
        "H > 0.5 : persistant / tendanciel. L'exposant affiché ci-dessous est **estimé en direct**, "
        "pas câblé en dur."
    )

    h_target = st.slider("H cible pour la génération (paramètre du modèle)", 0.10, 0.90, 0.50, 0.05, key="qf_h")

    @st.fragment(run_every="4s")
    def _render_fbm() -> None:
        n_t, n_paths_fbm = 90, 25
        t = np.linspace(0.01, 1.0, n_t)
        cov = 0.5 * (t[:, None] ** (2 * h_target) + t[None, :] ** (2 * h_target) - np.abs(t[:, None] - t[None, :]) ** (2 * h_target))
        cov += np.eye(n_t) * 1e-10
        chol = np.linalg.cholesky(cov)
        seed = int(time.time())
        rng = np.random.default_rng(seed)
        z = rng.normal(0, 1, (n_t, n_paths_fbm))
        paths = chol @ z  # each column: one real, exactly-covariance-matched fBm sample path

        fig = go.Figure(go.Surface(z=paths.T, x=t, y=np.arange(n_paths_fbm), colorscale=[[0, _PANEL], [0.5, _VIOLET], [1, "#ffffff"]], showscale=False))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, height=440, margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(xaxis=dict(title="t"), yaxis=dict(title="trajectoire #"), zaxis=dict(title="B_H(t)")),
        )
        st.plotly_chart(fig, width="stretch", key=f"fbm_surface_{seed}")

        # Real Hurst estimator: Var(increment at lag L) scales as L^(2H) -- log-log slope = 2H.
        one_path = paths[:, 0]
        lags = np.array([1, 2, 3, 5, 8])
        variances = np.array([np.var(one_path[lag:] - one_path[:-lag]) for lag in lags])
        slope = float(np.polyfit(np.log(lags), np.log(variances), 1)[0])
        h_estimated = slope / 2.0

        c1, c2, c3 = st.columns(3)
        c1.metric("H cible (génération)", f"{h_target:.2f}")
        c2.metric("H estimé (régression log-log réelle)", f"{h_estimated:.3f}")
        regime = "anti-persistant/rugueux" if h_estimated < 0.45 else ("persistant/tendanciel" if h_estimated > 0.55 else "brownien standard")
        c3.metric("Régime (déduit du calcul réel)", regime)
        st.caption(f"Écart estimation vs cible : {abs(h_estimated - h_target):.3f} (bruit d'estimation réel sur {n_t} points, pas une coïncidence scriptée).")

    _render_fbm()

# =============================================================================
# Module 3 -- Perron-Frobenius contagion (real eigenvector centrality + real copula tail dep.)
# =============================================================================

with tab_pf:
    st.markdown(
        '#### 🕸️ Perron-Frobenius Systemic Contagion &nbsp; <span class="qf-badge-real">EIGENVECTOR RÉEL</span> '
        '<span class="qf-badge-sim">RÉSEAU SYNTHÉTIQUE</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"Av = \lambda_{\max} v, \qquad \lambda_{\text{tail}} = 2\, t_{\nu+1}\!\left(-\sqrt{\tfrac{(\nu+1)(1-\rho)}{1+\rho}}\right)")
    st.markdown(
        '<div class="qf-panel">'
        "⚠️ Réseau synthétique (Barabási-Albert), <b>aucune donnée réelle d'exposition "
        "interbancaire</b>. \"NGN\" est une étiquette illustrative, pas une mesure du vrai marché du "
        "Naira. Le coefficient de dépendance de queue (copule de Student, formule réelle fermée -- "
        "Embrechts, McNeil &amp; Straumann 2002) est calculé sur des corrélations illustratives. "
        "<b>Aucun verrouillage de liquidité réel n'est déclenché ici.</b>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    _LABELS = ["NGN", "USDT", "EUR", "BTC", "ARS", "TRY", "BRL", "ZAR", "EGP", "KES", "GHS", "XOF"]
    graph = nx.barabasi_albert_graph(len(_LABELS), 2, seed=5)
    mapping = {i: _LABELS[i] for i in range(len(_LABELS))}
    graph = nx.relabel_nodes(graph, mapping)
    adjacency = nx.to_numpy_array(graph)

    eigvals, eigvecs = np.linalg.eig(adjacency)
    idx_max = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[idx_max].real)
    v_max = np.abs(eigvecs[:, idx_max].real)
    v_max /= v_max.sum()
    residual = float(np.linalg.norm(adjacency @ eigvecs[:, idx_max].real - lambda_max * eigvecs[:, idx_max].real))

    hub_idx = int(np.argmax(v_max))
    hub_label = _LABELS[hub_idx]
    neighbors = list(graph.neighbors(hub_label))

    rng = np.random.default_rng(9)
    nu = 4
    tail_deps = {}
    for nb in neighbors:
        rho = float(rng.uniform(0.3, 0.85))
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        tail_dep = float(2 * scipy.stats.t.cdf(arg, df=nu + 1))
        tail_deps[nb] = (rho, tail_dep)

    pos = nx.spring_layout(graph, seed=2)
    fig = go.Figure()
    for u, v in graph.edges:
        fig.add_trace(go.Scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines", line=dict(color=_STEEL, width=1), opacity=0.4, showlegend=False, hoverinfo="skip"))
    node_colors = []
    node_text = []
    for label in graph.nodes:
        if label == hub_label:
            node_colors.append(_RED)
            node_text.append(f"{label} (centralité max)")
        elif label in tail_deps:
            rho, td = tail_deps[label]
            node_colors.append(_ORANGE)
            node_text.append(f"{label} (λ_tail={td:.3f})")
        else:
            node_colors.append(_STEEL)
            node_text.append(label)
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in graph.nodes], y=[pos[n][1] for n in graph.nodes],
        mode="markers+text", text=list(graph.nodes), textposition="top center",
        marker=dict(size=[14 + 40 * v_max[list(graph.nodes).index(n)] for n in graph.nodes], color=node_colors),
        textfont=dict(color="#eee", size=10), hovertext=node_text, hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL, height=440, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, width="stretch", key="pf_network")

    c1, c2, c3 = st.columns(3)
    c1.metric("Nœud de centralité max (calcul réel)", hub_label)
    c2.metric("λ_max (calcul réel)", f"{lambda_max:.3f}")
    c3.metric("Résidu ‖Av-λv‖ (vérification réelle)", f"{residual:.2e}")
    st.caption(
        "Voisins en orange = dépendance de queue réelle (copule de Student, ν=4) calculée sur une "
        "corrélation illustrative par arête : " +
        ", ".join(f"{nb} (ρ={r:.2f}, λ_tail={td:.3f})" for nb, (r, td) in tail_deps.items()) +
        ". Aucune probabilité de contagion réelle mesurée sur un vrai marché."
    )

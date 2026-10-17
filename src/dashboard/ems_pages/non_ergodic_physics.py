"""Non-Ergodic Physics & Absorbing Barriers -- ergodicity economics, honest version.

**The core physics here is real, published economics/physics**, same discipline as
:mod:`gauge_theory_visualizer`: Ole Peters' "ergodicity economics" (Peters, O. *"The ergodicity
problem in economics."* Nature Physics 15, 1216-1221, 2019; Peters & Gell-Mann, *"Evaluating
gambles using dynamics."* Chaos 26, 023103, 2016) is a real, peer-reviewed critique of how
mainstream finance conflates the *ensemble average* (``E[X_t] = X_0 e^{mu t}``, an average over
many parallel universes that no single trader ever lives in) with the *time average* (the single
real trajectory one actual account follows, whose long-run growth rate is the real, well-known
``mu - sigma^2/2`` -- this is exact geometric Brownian motion (GBM) mathematics, not an invented
formula). The simulation below is a real, vectorized GBM Monte Carlo with a real absorbing barrier
at -90%, not a scripted animation.

**Two claims from the brief needed honest treatment, same pattern as this session's other
"God-Tier" pages** (see :mod:`black_swan_lab`, :mod:`gauge_theory_visualizer`):

1. ``"Capital absorbe par Nothierly suite aux liquidations : +452M USDT"`` -- a fund closing does
   not mechanically transfer its capital to a named competitor; nothing in a GBM survival
   simulation demonstrates a specific real-world dollar transfer. Kept as a real, *computed*
   number instead: the aggregate initial capital destroyed across the simulated cohort that hit
   the barrier, in the same illustrative base-100 units as the simulation, explicitly labeled as
   theoretical capital destroyed -- not dollars any real entity received.
2. ``"Probabilite de ruine Nothierly : 0.00000%"`` -- no real trading system has an exactly-zero
   ruin probability. What is real is the asymptotic property Kelly/time-average optimization
   gives *under this idealized GBM model* (no jumps, no fat tails, continuous rebalancing): ruin
   probability over a finite horizon goes to a real, computed, typically-tiny-but-not-exactly-zero
   number, shown as an actual empirical result of a second Monte Carlo cohort run with safer
   parameters -- with the idealization disclosed rather than presented as an absolute guarantee.

The "1000 competing funds" are synthetic simulated entities parameterized by the user's own
sliders, not real named funds; the "Nothierly" trajectory is one real GBM draw from a
lower-volatility parameterization of the exact same model, not a live account or a backtest.

**Protocole Léviathan**: as the real Nothierly trajectory advances along its own already-simulated
time-average curve, each real, non-negative step of growth is accrued into the shared
``GlobalMarketState.survival_capital`` (see ``src/dashboard/backend_simulator.py``) -- ergodicity
economics' own point, made cross-page: capital that *survives* the non-ergodic dynamics above
partially funds ``quantum_inspired_routing.py``'s real TT-SVD rank budget, not a fabricated transfer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.backend_simulator import accrue_survival_capital, get_state

_BG = "#050505"
_PANEL = "#0d0d0d"
_GHOST = "#4a4a4a"
_RED = "#ff1744"
_CYAN = "#00e5ff"
_GOLD = "#ffc400"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .nep-panel {{ background: {_PANEL}; border: 1px solid #2a2a2a; border-radius: 8px; padding: 1rem 1.2rem; }}
    .nep-disclosure {{ border-left: 3px solid {_GOLD}; background: rgba(255,196,0,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #f0d999; border-radius: 4px; }}
    .nep-badge {{ background: rgba(0,229,255,0.15); color: {_CYAN}; border: 1px solid {_CYAN}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💀 Non-Ergodic Physics & Absorbing Barriers")
st.caption("Ergodicity economics (Peters, Nature Physics 2019) -- simulation Monte Carlo réelle, pas une animation scriptée")
st.markdown(
    '<div class="nep-disclosure">'
    "🔬 Les équations et la dynamique ci-dessous sont une vraie physique/économie publiée "
    "(Ole Peters, <i>ergodicity economics</i>) -- <b>réellement simulée</b> par mouvement brownien "
    "géométrique (GBM) avec une vraie barrière absorbante à -90%. Les 1000 « fonds » sont des entités "
    "synthétiques paramétrées par vos curseurs, pas de vrais fonds nommés. Le capital « détruit » "
    "affiché plus bas est une somme réelle calculée sur la simulation, pas un flux de capitaux réel "
    "vers Nothierly. La « probabilité de ruine » de Nothierly est un résultat empirique réel de ce "
    "modèle idéalisé (sans sauts ni queues épaisses) -- pas une garantie absolue en marché réel."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

with st.sidebar:
    st.markdown("### 📐 Terminal Non-Ergodique")
    st.latex(r"\mathbb{E}[X_t] = X_0\, e^{\mu t}")
    st.latex(r"\lim_{t \to \infty} \frac{1}{t}\ln(X_t) = \mu - \frac{\sigma^2}{2}")
    st.caption(
        "Le terme -σ²/2 est la pénalité de volatilité (réelle, exacte pour un GBM). Si σ est trop "
        "élevé, l'espérance d'ensemble (haut) continue de monter alors que le taux de croissance "
        "temporel réel (bas) devient négatif : la trajectoire individuelle fait faillite, même si "
        "la moyenne sur des milliers d'univers parallèles imaginaires ne le montre jamais."
    )
    st.markdown("---")
    st.markdown("**Cohorte concurrente**")
    mu_pct = st.slider("Rendement attendu μ (%/an)", 0.0, 20.0, 8.0, 0.5, key="nep_mu")
    sigma_pct = st.slider("Volatilité σ (%/an)", 5.0, 80.0, 45.0, 1.0, key="nep_sigma")
    st.markdown("**Cohorte Nothierly (optimisation temporelle)**")
    mu_n_pct = st.slider("Rendement attendu μ_N (%/an)", 0.0, 20.0, 11.0, 0.5, key="nep_mu_n")
    sigma_n_pct = st.slider("Volatilité σ_N (%/an)", 3.0, 40.0, 14.0, 1.0, key="nep_sigma_n")
    seed = st.number_input("Graine aléatoire", min_value=0, max_value=999_999, value=42, step=1, key="nep_seed")


@st.cache_data(show_spinner="Simulation Monte Carlo en cours...")
def _simulate(mu: float, sigma: float, mu_n: float, sigma_n: float, seed: int) -> dict[str, np.ndarray]:
    """Real vectorized GBM Monte Carlo with a real absorbing barrier at -90% of X0."""
    rng = np.random.default_rng(seed)
    x0 = 100.0
    n_paths, n_steps, dt = 1000, 120, 1.0 / 12.0  # 10 years, monthly steps
    barrier = 0.10 * x0

    def run_cohort(mu_: float, sigma_: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        z = rng.standard_normal((n, n_steps))
        log_returns = (mu_ - 0.5 * sigma_**2) * dt + sigma_ * np.sqrt(dt) * z
        log_paths = np.cumsum(log_returns, axis=1)
        paths = x0 * np.exp(log_paths)
        paths = np.hstack([np.full((n, 1), x0), paths])
        dead = np.cumsum(paths <= barrier, axis=1) > 0
        # Two views on purpose: a real 0.0 sentinel for the "is this fund dead" metrics below, and a
        # NaN sentinel for anything plotted on a log axis -- log(0) is -inf and silently corrupts
        # Plotly's own autorange (observed: it blew the y-axis out to 1e10 with real data topping out
        # at ~5000). NaN instead reads as a real gap, i.e. the line disappearing at the moment of death.
        return np.where(dead, 0.0, paths), np.where(dead, np.nan, paths)

    competitor_paths, competitor_paths_plot = run_cohort(mu, sigma, n_paths)
    nothierly_cohort, nothierly_cohort_plot = run_cohort(mu_n, sigma_n, n_paths)
    nothierly_single = nothierly_cohort_plot[0]

    t_grid = np.arange(n_steps + 1) * dt
    ensemble_illusion = x0 * np.exp(mu * t_grid)
    time_average_curve = x0 * np.exp((mu_n - 0.5 * sigma_n**2) * t_grid)

    liquidated_mask = competitor_paths[:, -1] == 0.0
    nothierly_ruin_mask = nothierly_cohort[:, -1] == 0.0

    return {
        "t_grid": t_grid,
        "competitor_paths_plot": competitor_paths_plot,
        "nothierly_single": nothierly_single,
        "ensemble_illusion": ensemble_illusion,
        "time_average_curve": time_average_curve,
        "liquidated_count": np.array([int(liquidated_mask.sum())]),
        "capital_destroyed": np.array([float(liquidated_mask.sum() * x0)]),
        "nothierly_ruin_count": np.array([int(nothierly_ruin_mask.sum())]),
        "n_paths": np.array([n_paths]),
    }


sim = _simulate(mu_pct / 100.0, sigma_pct / 100.0, mu_n_pct / 100.0, sigma_n_pct / 100.0, int(seed))

t_grid = sim["t_grid"]
competitor_paths_plot = sim["competitor_paths_plot"]
n_paths = int(sim["n_paths"][0])


@st.fragment(run_every="4s")
def _accrue_survival_capital() -> None:
    """Walks one step further along the already-simulated real Nothierly trajectory each tick,
    accruing only its real, non-negative growth into the shared survival-capital pool -- reusing
    the exact path this page already computed above, not drawing new random numbers."""
    idx_key = "nep_survival_idx"
    idx = st.session_state.get(idx_key, 0)
    path = sim["nothierly_single"]
    if idx + 1 < len(path) and not (np.isnan(path[idx]) or np.isnan(path[idx + 1])):
        delta = float(path[idx + 1] - path[idx])
        accrue_survival_capital(get_state(), max(0.0, delta))
    st.session_state[idx_key] = (idx + 1) % (len(path) - 1)

    state = get_state()
    st.caption(
        f"🔗 Capital de survie cumulé (état partagé, Protocole Léviathan) : **{state.survival_capital:,.1f}** "
        f"unités -- finance {state.quantum_arbitrage_pool:,.1f} unités du pool d'arbitrage de "
        "`quantum_inspired_routing.py`."
    )


_accrue_survival_capital()

fig = go.Figure()

# One real trace for all 1000 competitor paths, NaN-separated for rendering speed -- same 1000
# real simulated trajectories, not a decorative subsample.
x_ghost = np.tile(np.append(t_grid, np.nan), n_paths)
y_ghost = np.concatenate([np.append(competitor_paths_plot[i], np.nan) for i in range(n_paths)])
fig.add_trace(go.Scattergl(
    x=x_ghost, y=y_ghost, mode="lines", line=dict(color=_GHOST, width=0.6),
    opacity=0.35, hoverinfo="skip", showlegend=True, name=f"{n_paths} fonds concurrents simulés",
))

fig.add_trace(go.Scatter(
    x=t_grid, y=sim["ensemble_illusion"], mode="lines", line=dict(color=_RED, width=2.5, dash="dash"),
    name="Moyenne d'ensemble (espérance classique, illusoire)",
))

fig.add_trace(go.Scatter(
    x=t_grid, y=sim["nothierly_single"], mode="lines", line=dict(color=_CYAN, width=3.5),
    name="Nothierly -- une trajectoire GBM réelle (σ réduit)",
))

fig.add_trace(go.Scatter(
    x=t_grid, y=sim["time_average_curve"], mode="lines", line=dict(color=_GOLD, width=1.5, dash="dot"),
    name="Nothierly -- courbe théorique μ_N - σ_N²/2 (moyenne temporelle exacte)",
))

fig.add_hline(y=10.0, line=dict(color=_RED, width=1, dash="dot"))
# A data-anchored annotation_text on add_hline interacts badly with a log y-axis in this Plotly
# version -- observed: it silently blows the autorange out to ~1e10 with real data topping out at
# ~5000 (reproduced and confirmed in isolation). A paper-anchored label avoids the axis entirely.
fig.add_annotation(
    text="Barrière absorbante (-90%)", xref="paper", yref="paper", x=0.99, y=0.04,
    showarrow=False, font=dict(color=_RED, size=11), align="right",
)

fig.update_layout(
    template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=520,
    margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Années", yaxis_title="Capital (base 100, échelle log)",
    yaxis_type="log",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(fig, width="stretch", key="nep_main_chart")

st.markdown("#### 🪦 Tableau de bord de survie (calculé sur cette simulation)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Fonds concurrents liquidés",
        f"{int(sim['liquidated_count'][0])} / {n_paths}",
        help="Nombre réel de trajectoires simulées ayant touché la barrière absorbante à -90% sur 10 ans.",
    )
with col2:
    st.metric(
        "Capital théorique détruit (base 100/fonds)",
        f"{sim['capital_destroyed'][0]:,.0f} unités",
        help="Somme du capital initial des fonds liquidés dans cette simulation -- une mesure de capital détruit, "
        "pas un flux de capitaux réel vers Nothierly : rien dans ce modèle ne transfère l'argent d'un fonds mort à un tiers.",
    )
with col3:
    ruin_pct = 100.0 * sim["nothierly_ruin_count"][0] / n_paths
    st.metric(
        "Ruine Nothierly (cohorte μ_N/σ_N, empirique)",
        f"{ruin_pct:.4f}%",
        help="Fraction réelle de la cohorte Nothierly ayant touché la barrière dans cette simulation. Asymptotiquement "
        "→0 sous ce modèle GBM idéalisé quand μ_N > σ_N²/2 -- pas une garantie absolue : un marché réel a des sauts, "
        "des queues épaisses et un risque de contrepartie que ce modèle ne capture pas.",
    )

st.caption(
    "Changez les curseurs dans la barre latérale pour rejouer la simulation avec d'autres hypothèses de μ/σ -- "
    "tous les chiffres ci-dessus sont recalculés en direct sur les vraies trajectoires, aucun n'est câblé en dur."
)

"""Millennium Conjectures & Unresolved Math -- Zero-to-One Vision Showcase, honest version.

**None of the five Clay Millennium Prize Problems shown below is solved, proven, refuted, or even
touched by this page.** ``P vs NP``, the Riemann Hypothesis, Navier-Stokes existence/smoothness, and
the Hodge Conjecture are real, unsolved (as of this writing) problems (Collatz is a separate, also
real and unsolved conjecture, not one of the seven Clay problems). Claiming a trading system derives
a "scientific, absolute edge" from an *open* problem is incoherent -- an open problem has no known
general solution to exploit. What follows is real, cited, correctly-computed mathematics from each
area, applied honestly to disclosed synthetic/illustrative data, in the same "chalkboard" showcase
spirit as this project's other Zero-to-One pages -- not a resolution of anything.

**Five corrections applied, same discipline as every other "frontier tech" page in this EMS**
(:mod:`non_ergodic_physics`, :mod:`agi_photonic_singularity`, :mod:`quantitative_supremacy`, etc.):

1. *P vs NP module's "résolution heuristique en 0.8 ms"* -- a specific, fabricated performance number
   with no computation behind it. Replaced with a real, live-measured benchmark: an exact brute-force
   Held-Karp-free traveling-salesman solver (``O((n-1)!)``, real ``itertools.permutations``) against a
   real polynomial-time nearest-neighbor heuristic (``O(n^2)``), both timed with
   ``time.perf_counter()`` on the same real random instance. **No claim that this, or any heuristic,
   proves or refutes P≠NP** -- nearest-neighbor also carries no guaranteed constant approximation
   ratio in general (unlike, say, Christofides' 3/2 under the triangle inequality); the ratio shown is
   an empirical measurement on one instance, not a theorem.

2. *Riemann Zeta module's "les pics de volatilité s'alignent sur les zéros non triviaux"* -- an
   unfalsifiable, unevidenced causal claim this project makes nowhere else. What is real: the first
   ``N`` non-trivial zeros, located live via ``mpmath.zetazero`` (the real Riemann-Siegel/Turing-method
   algorithm, not hardcoded constants), a real 3D ``|zeta(s)|`` surface, and the real Montgomery-
   Odlyzko law (Montgomery 1973; Odlyzko 1987) -- the celebrated, real result that zeta-zero spacing
   statistics match Gaussian Unitary Ensemble (GUE) random-matrix eigenvalue spacing. This page runs a
   real two-sample Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``) comparing real zeta-zero spacings
   against a real GUE ensemble *and* against synthetic, disclosed-as-synthetic Poisson "market shock"
   timestamps -- honestly showing uncorrelated synthetic shocks do **not** reproduce the real
   level-repulsion signature, rather than fabricating a positive "alignment."

3. *Navier-Stokes module's "vitesse tend vers l'infini (singularité)"* -- presented as a real physical
   event triggering real capital withdrawal. What is real: the Lamb-Oseen vortex (Lamb, 1932; Oseen,
   1911), an exact closed-form solution of the 2D incompressible Navier-Stokes vorticity equation, with
   a real, live-computed peak vorticity. Disclosed explicitly: this 2D exact solution is smooth and
   globally regular for every ``t > 0`` -- it is **not** a demonstration of the real, still-open 3D
   blow-up question the Clay Institute actually poses. The "alert" below is a disclosed, illustrative
   threshold on this real number (see ``backend_simulator.report_navier_stokes_vorticity``), not a
   physical singularity or a real trade.

4. *Hodge module's "décomposition de la macro-économie mondiale en cycles algébriques P2P"* -- the
   real Hodge Conjecture concerns algebraic cycles on smooth projective varieties and has nothing to
   do with graphs or economics. What is shown instead is a real, different, similarly-named field:
   *combinatorial/discrete Hodge theory* (Jiang, Lim, Yao & Ye, "Statistical ranking and combinatorial
   Hodge theory," *Math. Programming* 127, 2011 -- the real basis of HodgeRank), applied here to its
   own textbook use case: decomposing a synthetic pairwise FX/crypto log-rate graph into a consistent
   "fair price" gradient component plus a real cyclic/inconsistent (triangular-arbitrage-style)
   residual, computed via real linear algebra (``numpy.linalg.lstsq`` on the graph incidence matrix) --
   on disclosed synthetic rates, not live quotes.

5. *Collatz module's "vente à découvert déclenchée avant le retour à la parité intrinsèque"* -- an
   unvalidated, unbacktested trading rule asserted as if proven. The Collatz conjecture itself is real
   and genuinely unsolved (Erdős: "mathematics is not yet ripe for such questions"; computationally
   verified for enormous ranges -- Barina, "Convergence verification of the Collatz conjecture,"
   *J. Supercomputing*, 2020). Its real sawtooth trajectories are shown as a disclosed, playful visual
   analogy for boom/bust dynamics -- **no backtest, no claimed edge, no real order is implied.**

**Protocole Léviathan**: the Navier-Stokes module's real peak vorticity (driven live by the shared
Shannon chaos bias from ``casino_information_theory.py``, or by a manual slider) is reported through
``backend_simulator.report_navier_stokes_vorticity`` -- crossing the disclosed threshold really pauses
AGI Active Inference's capital injection (see that function's use in ``report_entropy``) and really
widens the disclosed cost matrix in ``quantitative_supremacy.py``'s Optimal Transport LP, visible on
both of those pages the moment it happens, not narrated after the fact.
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import mpmath
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import scipy.stats
import streamlit as st

from src.dashboard.backend_simulator import (
    NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD,
    get_state,
    lamb_oseen_peak_vorticity,
    report_navier_stokes_vorticity,
)

_BG = "#0c0f13"
_SLATE = "#12161c"
_CHALK = "#eef1f0"
_CHALK_DIM = "#9aa3a8"
_CYAN = "#5ad8ff"
_GOLD = "#ffc94a"
_RED = "#ff5d6c"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    html, body, [class*="css"] {{ font-family: 'JetBrains Mono', 'Consolas', monospace; color: {_CHALK}; }}
    .mc-panel {{ background: {_SLATE}; border: 1px solid rgba(90,216,255,0.18); border-radius: 6px; padding: 1rem 1.3rem; }}
    .mc-disclosure {{ border-left: 3px solid {_GOLD}; background: rgba(255,201,74,0.06); padding: 0.7rem 1.1rem; font-size: 0.85rem; color: #f2debb; border-radius: 4px; }}
    .mc-badge-real {{ background: rgba(90,216,255,0.12); color: {_CYAN}; border: 1px solid {_CYAN}; border-radius: 3px; padding: 0.05rem 0.55rem; font-size: 0.7rem; font-weight: 700; }}
    .mc-badge-sim {{ background: rgba(255,255,255,0.06); color: {_CHALK_DIM}; border: 1px solid {_CHALK_DIM}; border-radius: 3px; padding: 0.05rem 0.55rem; font-size: 0.7rem; font-weight: 700; }}
    .mc-badge-alert {{ background: rgba(255,93,108,0.14); color: {_RED}; border: 1px solid {_RED}; border-radius: 3px; padding: 0.05rem 0.55rem; font-size: 0.7rem; font-weight: 700; }}
    .mc-metric {{ background: {_SLATE}; border: 1px solid rgba(90,216,255,0.18); border-radius: 8px; padding: 0.8rem 1rem; text-align: center; }}
    .mc-metric-value {{ font-size: 1.35rem; font-weight: 700; color: {_GOLD}; }}
    .mc-metric-label {{ font-size: 0.72rem; color: {_CHALK_DIM}; margin-top: 0.15rem; }}
    .mc-log {{ background: #000; color: {_CYAN}; font-size: 0.78rem; padding: 0.7rem 0.9rem; border-radius: 6px; line-height: 1.6; border: 1px solid rgba(90,216,255,0.2); }}
    div[data-testid="stMetricValue"] {{ color: {_GOLD} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("∫ Millennium Conjectures & Unresolved Math")
st.caption("Nothierly Zero-to-One Vision -- mathématiques réelles, problèmes réellement non résolus")
st.markdown(
    '<div class="mc-disclosure">🔬 <b>Aucun des cinq problèmes ci-dessous n\'est résolu, prouvé ou '
    "réfuté ici</b> -- P vs NP, l'Hypothèse de Riemann, Navier-Stokes et Hodge sont de vrais problèmes "
    "du prix du Millénaire (Clay Mathematics Institute) toujours ouverts ; Collatz est une autre vraie "
    "conjecture non résolue. Un problème ouvert n'a, par définition, aucune solution générale connue à "
    "exploiter. Ce qui suit est un vrai calcul (<span class='mc-badge-real'>CALCUL RÉEL</span>), cité, "
    "appliqué à des données synthétiques disclosées (<span class='mc-badge-sim'>SIMULATION</span>) -- "
    "voir le docstring du fichier pour les cinq corrections détaillées.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_np, tab_zeta, tab_ns, tab_hodge, tab_collatz = st.tabs([
    "P vs NP", "Riemann Zeta", "Navier-Stokes", "Hodge", "Collatz",
])

# =============================================================================
# Module 1 -- P vs NP Complexity Arbitrage (real measured brute force vs real heuristic)
# =============================================================================

with tab_np:
    st.markdown('#### 🧮 P vs NP -- Complexity Arbitrage &nbsp; <span class="mc-badge-real">BENCHMARK RÉEL, PAS UNE FORMULE</span>', unsafe_allow_html=True)
    st.latex(r"P \neq NP \quad \text{(conjecturé, non prouvé)}")
    st.caption(
        "Instance réelle : un petit voyageur de commerce (TSP) euclidien sur des points aléatoires -- "
        "résolu exactement par force brute (tous les chemins) et approximativement par plus-proche-voisin, "
        "les deux réellement chronométrés, pas des complexités théoriques assumées."
    )

    max_n = st.slider("Taille maximale n (attention : (n-1)! chemins en force brute)", 4, 10, 8, 1, key="mc_np_max_n")
    seed_np = st.number_input("Graine aléatoire (instance)", 0, 999_999, 7, key="mc_np_seed")

    def _tour_length(tour: tuple[int, ...], dist: np.ndarray) -> float:
        return float(sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))

    @st.cache_data(show_spinner="Résolution combinatoire réelle en cours (force brute + heuristique)...")
    def _complexity_benchmark(max_n: int, seed: int) -> dict[str, list[float]]:
        rng = np.random.default_rng(seed)
        ns = list(range(3, max_n + 1))
        brute_times, heuristic_times, approx_ratios, brute_ops = [], [], [], []
        for n in ns:
            pts = rng.uniform(0, 100, size=(n, 2))
            dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

            t0 = time.perf_counter()
            best_len = None
            for perm in itertools.permutations(range(1, n)):
                length = _tour_length((0,) + perm, dist)
                if best_len is None or length < best_len:
                    best_len = length
            brute_times.append(time.perf_counter() - t0)
            brute_ops.append(float(math.factorial(n - 1)))

            t0 = time.perf_counter()
            unvisited = set(range(1, n))
            tour = [0]
            while unvisited:
                last = tour[-1]
                nxt = min(unvisited, key=lambda j: dist[last, j])
                tour.append(nxt)
                unvisited.remove(nxt)
            heuristic_times.append(time.perf_counter() - t0)
            approx_ratios.append(_tour_length(tuple(tour), dist) / best_len if best_len else 1.0)

        return {
            "ns": ns, "brute_times": brute_times, "heuristic_times": heuristic_times,
            "approx_ratios": approx_ratios, "brute_ops": brute_ops,
        }

    bench = _complexity_benchmark(max_n, int(seed_np))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench["ns"], y=bench["brute_times"], mode="lines+markers", line=dict(color=_RED, width=2.5), name="Force brute réelle -- O((n-1)!)"))
    fig.add_trace(go.Scatter(x=bench["ns"], y=bench["heuristic_times"], mode="lines+markers", line=dict(color=_CYAN, width=2.5), name="Heuristique réelle (plus proche voisin) -- O(n²)"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, plot_bgcolor=_SLATE, height=420,
        margin=dict(l=10, r=10, t=30, b=10), xaxis_title="n (villes)", yaxis_title="Temps mesuré réel (s, échelle log)",
        yaxis_type="log", legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, width="stretch", key="mc_np_chart")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Force brute réelle à n={max_n}", f"{bench['brute_times'][-1]*1000:,.1f} ms", help=f"{bench['brute_ops'][-1]:,.0f} permutations réellement énumérées.")
    c2.metric(f"Heuristique réelle à n={max_n}", f"{bench['heuristic_times'][-1]*1000:,.3f} ms")
    speedup = bench["brute_times"][-1] / bench["heuristic_times"][-1] if bench["heuristic_times"][-1] > 0 else float("inf")
    c3.metric("Accélération mesurée (réelle)", f"{speedup:,.0f}×")
    st.markdown(
        '<div class="mc-panel">⚠️ Cette accélération réelle et ce ratio d\'approximation mesuré '
        f"(<b>{bench['approx_ratios'][-1]:.3f}×</b> l'optimal sur cette instance, sans garantie théorique "
        "générale) illustrent l'écart classe-de-complexité entre un solveur exact exponentiel et une "
        "heuristique polynomiale -- <b>ni preuve ni réfutation de P≠NP</b>. Le TT-SVD réel de "
        "<code>quantum_inspired_routing.py</code> est un autre exemple réel du même paradigme "
        "(approximation polynomiale d'un problème dont la version générale est NP-difficile)."
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# Module 2 -- Riemann Zeta Volatility Harmonics (real zetazero, real GUE, real KS tests)
# =============================================================================

with tab_zeta:
    st.markdown('#### ζ Riemann Zeta -- Volatility Harmonics &nbsp; <span class="mc-badge-real">mpmath RÉEL</span>', unsafe_allow_html=True)
    st.latex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ premier}} \frac{1}{1 - p^{-s}}")
    st.caption(
        "Hypothèse de Riemann (non prouvée) : tous les zéros non triviaux ont Re(s)=1/2. Les zéros "
        "ci-dessous sont localisés en direct par mpmath.zetazero (méthode de Riemann-Siegel/Turing, "
        "réelle), pas des constantes recopiées."
    )

    n_zeros = st.slider("Nombre de zéros réels à localiser (mpmath.zetazero)", 10, 60, 30, 5, key="mc_zeta_n")

    @st.cache_data(show_spinner="Localisation réelle des zéros non triviaux de zêta (mpmath)...")
    def _real_zeta_zeros(n: int) -> list[float]:
        return [float(mpmath.zetazero(k).imag) for k in range(1, n + 1)]

    @st.cache_data(show_spinner="Évaluation réelle de |ζ(s)| sur la bande critique (mpmath)...")
    def _zeta_surface(t_max: float, n_sigma: int = 26, n_t: int = 55) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sigmas = np.linspace(0.05, 0.95, n_sigma)
        ts = np.linspace(0.5, t_max, n_t)
        mag = np.zeros((n_sigma, n_t))
        for i, s in enumerate(sigmas):
            for j, t in enumerate(ts):
                mag[i, j] = float(abs(mpmath.zeta(complex(s, t))))
        return sigmas, ts, mag

    zeros = np.array(_real_zeta_zeros(n_zeros))
    sigmas, ts, mag = _zeta_surface(float(zeros[-1]) + 5.0)

    fig = go.Figure(data=[go.Surface(x=ts, y=sigmas, z=mag, colorscale="Ice", showscale=False, opacity=0.92)])
    fig.add_trace(go.Scatter3d(
        x=zeros, y=np.full_like(zeros, 0.5), z=np.zeros_like(zeros),
        mode="markers", marker=dict(size=4, color=_GOLD), name="Zéros réels (mpmath.zetazero)",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, height=460, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title="t (Im s)", yaxis_title="σ (Re s)", zaxis_title="|ζ(s)|",
            xaxis=dict(backgroundcolor=_SLATE), yaxis=dict(backgroundcolor=_SLATE), zaxis=dict(backgroundcolor=_SLATE),
        ),
    )
    st.plotly_chart(fig, width="stretch", key="mc_zeta_surface")
    st.caption(f"|ζ(s)| réellement évaluée s'effondre vers 0 à σ=1/2 exactement aux {n_zeros} zéros réels localisés ci-dessus (marqueurs or) -- une propriété numérique vérifiable, pas une mise en scène.")

    st.markdown("##### Loi de Montgomery-Odlyzko : espacement des zéros vs matrices aléatoires GUE (réel)")
    st.markdown(
        '<div class="mc-panel">🔬 Résultat réel et célèbre (Montgomery 1973 ; confirmation numérique '
        "d'Odlyzko 1987) : la statistique d'espacement des zéros de zêta ressemble à celle des valeurs "
        "propres de matrices hermitiennes aléatoires (Gaussian Unitary Ensemble). Testé ici en direct par "
        "un vrai test de Kolmogorov-Smirnov -- <b>aucune affirmation n'est faite sur de vrais chocs de "
        "marché</b> : les horodatages de \"chocs\" ci-dessous sont un processus de Poisson synthétique "
        "disclosé, pas des données réelles."
        "</div>",
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner="Génération réelle de l'ensemble GUE (matrices hermitiennes aléatoires)...")
    def _gue_unfolded_spacings(matrix_size: int, n_trials: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_trials):
            a = rng.standard_normal((matrix_size, matrix_size)) + 1j * rng.standard_normal((matrix_size, matrix_size))
            h = (a + a.conj().T) / 2.0  # real GUE Hermitian matrix
            eigs = np.sort(np.linalg.eigvalsh(h))
            diffs = np.diff(eigs)
            out.append(diffs / diffs.mean())
        return np.concatenate(out)

    zeta_diffs = np.diff(zeros)
    zeta_unfolded = zeta_diffs / zeta_diffs.mean()  # simplified unfolding (sample mean), disclosed below
    gue_spacings = _gue_unfolded_spacings(matrix_size=40, n_trials=60, seed=11)

    rng_shock = np.random.default_rng(int(seed_np) + 1)
    n_shocks = max(len(zeta_unfolded), 30)
    shock_gaps = rng_shock.exponential(scale=1.0, size=n_shocks)  # real synthetic Poisson process, mean-1 spacing

    ks_zeta_gue = scipy.stats.ks_2samp(zeta_unfolded, gue_spacings)
    ks_zeta_poisson = scipy.stats.ks_2samp(zeta_unfolded, shock_gaps)
    ks_shocks_gue = scipy.stats.ks_2samp(shock_gaps, gue_spacings)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=gue_spacings, histnorm="probability density", name="Espacements GUE (réel)", marker_color=_CYAN, opacity=0.55, nbinsx=30))
    fig2.add_trace(go.Histogram(x=zeta_unfolded, histnorm="probability density", name="Espacements zéros ζ (réel, déplié simplifié)", marker_color=_GOLD, opacity=0.65, nbinsx=20))
    fig2.add_trace(go.Histogram(x=shock_gaps, histnorm="probability density", name="Chocs marché synthétiques (Poisson, SIMULATION)", marker_color=_CHALK_DIM, opacity=0.45, nbinsx=30))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, plot_bgcolor=_SLATE, height=360, barmode="overlay",
        margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Espacement déplié (unités locales moyennes)", legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig2, width="stretch", key="mc_zeta_spacings")

    c1, c2, c3 = st.columns(3)
    c1.metric("KS zéros ζ vs GUE (réel)", f"D={ks_zeta_gue.statistic:.3f}", f"p={ks_zeta_gue.pvalue:.3f}")
    c2.metric("KS zéros ζ vs Poisson synth. (réel)", f"D={ks_zeta_poisson.statistic:.3f}", f"p={ks_zeta_poisson.pvalue:.3f}")
    c3.metric("KS chocs synth. vs GUE (réel)", f"D={ks_shocks_gue.statistic:.3f}", f"p={ks_shocks_gue.pvalue:.3f}")
    st.markdown(
        '<div class="mc-panel">Test réel, pas une affirmation : un processus de Poisson pur (nos '
        "chocs de marché synthétiques) n'a, par construction, aucune répulsion de niveau -- il ne "
        "reproduit pas la statistique réelle des zéros de zêta/GUE. <b>Aucune preuve statistique ici ne "
        "rapproche de vrais chocs de marché des zéros de zêta</b> ; unfolding simplifié (moyenne globale "
        f"de l'échantillon des {n_zeros} zéros), pas la formule asymptotique complète de Riemann-von "
        "Mangoldt -- disclosé, pas une reproduction de l'analyse d'Odlyzko sur des millions de zéros."
        "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# Module 3 -- Navier-Stokes Fluid Singularity (real Lamb-Oseen vortex, real peak vorticity)
# =============================================================================

with tab_ns:
    st.markdown('#### 🌀 Navier-Stokes -- Fluid Singularity (Krach Monitor) &nbsp; <span class="mc-badge-real">SOLUTION ANALYTIQUE RÉELLE</span>', unsafe_allow_html=True)
    st.latex(r"\rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}")
    st.markdown(
        '<div class="mc-panel">⚠️ Le vrai problème du Millénaire porte sur l\'existence/régularité en '
        "3D et reste non résolu. Ci-dessous : le vortex de Lamb-Oseen, une <b>vraie solution analytique "
        "exacte</b> de l'équation de vorticité de Navier-Stokes 2D (Lamb, 1932 ; Oseen, 1911) -- lisse "
        "et globalement régulière pour tout t&gt;0. Elle ne démontre <b>aucune</b> singularité 3D réelle. "
        "Le seuil d'alerte est disclosé et illustratif, appliqué à une vraie vorticité de pointe calculée "
        "en forme fermée, pas un phénomène physique de rupture."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    live_mode = st.checkbox("Piloter Γ depuis l'entropie de Shannon partagée (Casino Information Theory, réel)", value=True, key="mc_ns_live")
    nu = st.slider("Viscosité cinématique ν (illustrative)", 0.2, 1.5, 0.6, 0.05, key="mc_ns_nu")
    t_since = st.slider("Temps depuis la formation du tourbillon t (illustratif)", 0.1, 3.0, 1.0, 0.1, key="mc_ns_t")
    if not live_mode:
        gamma = st.slider("Force du tourbillon Γ (proxy manuel du déséquilibre du carnet d'ordres)", 2.0, 60.0, 10.0, 1.0, key="mc_ns_gamma")

    @st.fragment(run_every="2s")
    def _render_navier_stokes(nu: float, t_since: float, live_mode: bool) -> None:
        state = get_state()
        gamma_local = 4.0 + 40.0 * state.shannon_chaos_bias if live_mode else st.session_state.get("mc_ns_gamma", 10.0)

        extent, grid_n = 6.0, 26
        xs = np.linspace(-extent, extent, grid_n)
        ys = np.linspace(-extent, extent, grid_n)
        xx, yy = np.meshgrid(xs, ys)
        r = np.sqrt(xx**2 + yy**2)
        r_safe = np.where(r < 1e-6, 1e-6, r)
        u_theta = gamma_local / (2 * np.pi * r_safe) * (1 - np.exp(-r_safe**2 / (4 * nu * t_since)))
        u = -u_theta * yy / r_safe
        v = u_theta * xx / r_safe

        peak_vorticity = lamb_oseen_peak_vorticity(gamma_local, nu, t_since)
        crisis_active = report_navier_stokes_vorticity(state, peak_vorticity)

        stream_fig = ff.create_streamline(xs, ys, u, v, arrow_scale=0.08, density=1.1, line=dict(color=_CYAN, width=1.2))
        stream_fig.update_layout(
            template="plotly_dark", paper_bgcolor=_SLATE, plot_bgcolor=_SLATE, height=440,
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(stream_fig, width="stretch", key=f"mc_ns_stream_{time.time()}")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="mc-metric"><div class="mc-metric-value">{gamma_local:.2f}</div><div class="mc-metric-label">Γ (force du tourbillon, {"piloté par l\'entropie réelle" if live_mode else "manuel"})</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="mc-metric"><div class="mc-metric-value">{peak_vorticity:.2f} rad/s</div><div class="mc-metric-label">Vorticité de pointe (forme fermée réelle)</div></div>', unsafe_allow_html=True)
        badge = '<span class="mc-badge-alert">CRISE ACTIVE</span>' if crisis_active else '<span class="mc-badge-real">STABLE</span>'
        c3.markdown(f'<div class="mc-metric"><div class="mc-metric-value">{badge}</div><div class="mc-metric-label">Seuil illustratif : {NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD:.1f} rad/s</div></div>', unsafe_allow_html=True)

        if crisis_active:
            st.markdown(
                '<div class="mc-panel">🔴 <b>Turbulence détectée sur la vorticité réelle du modèle</b> -- '
                f"seuil illustratif {NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD:.1f} rad/s franchi. Effet réel, "
                "disclosé, sur l'état partagé (Protocole Léviathan) : les injections de capital simulées de "
                "<code>agi_photonic_singularity.py</code> sont mises en pause, et le coût de friction du LP "
                "d'<code>quantitative_supremacy.py</code> (Optimal Transport) est réellement majoré -- "
                "<b>aucun retrait de capital réel n'a lieu</b>."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Vorticité réelle sous le seuil illustratif -- aucun effet sur les autres pages en ce moment.")

    _render_navier_stokes(nu, t_since, live_mode)

# =============================================================================
# Module 4 -- Hodge Cycle Macro-Mapping (real combinatorial Hodge decomposition)
# =============================================================================

with tab_hodge:
    st.markdown('#### Ω Hodge -- Combinatorial Hodge Decomposition &nbsp; <span class="mc-badge-real">ALGÈBRE LINÉAIRE RÉELLE</span>', unsafe_allow_html=True)
    st.latex(r"H^{k,k}(X, \mathbb{Q}) = \text{Hdg}^k(X) \cap H^{2k}(X, \mathbb{Q})")
    st.markdown(
        '<div class="mc-panel">⚠️ La vraie Conjecture de Hodge (ci-dessus) porte sur des cycles '
        "algébriques sur des variétés projectives complexes lisses -- non résolue, et sans rapport avec "
        "des graphes ou de la macro-économie. Ce qui suit est un domaine réel <b>différent</b>, portant "
        "un nom proche : la théorie de Hodge combinatoire/discrète (Jiang, Lim, Yao &amp; Ye, "
        "<i>Math. Programming</i> 2011), réellement calculée ci-dessous sur un graphe synthétique de taux "
        "de change (pas de cotations réelles) pour détecter des incohérences cycliques de type arbitrage "
        "triangulaire."
        "</div>",
        unsafe_allow_html=True,
    )
    st.latex(r"Y = \underbrace{B\,\hat p}_{\text{gradient (prix cohérent)}} + \underbrace{r}_{\text{cyclique (incohérence)}}, \qquad \text{curl}_{ijk} = Y_{ij} + Y_{jk} + Y_{ki}")

    assets = ["USD", "EUR", "BTC", "USDT", "ARS"]
    noise_scale = st.slider("Bruit de cotation inter-plateforme (σ, synthétique)", 0.0, 0.5, 0.12, 0.01, key="mc_hodge_noise")
    seed_hodge = st.number_input("Graine aléatoire (taux synthétiques)", 0, 999_999, 3, key="mc_hodge_seed")

    @st.cache_data(show_spinner="Génération réelle des taux synthétiques et décomposition de Hodge...")
    def _hodge_decompose(assets: tuple[str, ...], noise_scale: float, seed: int):
        n = len(assets)
        rng = np.random.default_rng(seed)
        true_log_price = rng.normal(0, 1.0, n)
        true_log_price -= true_log_price[0]
        pairs = list(itertools.combinations(range(n), 2))
        y = np.array([(true_log_price[j] - true_log_price[i]) + rng.normal(0, noise_scale) for i, j in pairs])

        incidence = np.zeros((len(pairs), n))
        for idx, (i, j) in enumerate(pairs):
            incidence[idx, i], incidence[idx, j] = -1.0, 1.0
        p_reduced, *_ = np.linalg.lstsq(incidence[:, 1:], y, rcond=None)
        p_hat = np.concatenate([[0.0], p_reduced])
        grad = incidence @ p_hat
        residual = y - grad

        pair_index = {p: idx for idx, p in enumerate(pairs)}

        def flow(i: int, j: int) -> float:
            return y[pair_index[(i, j)]] if i < j else -y[pair_index[(j, i)]]

        curls = {(i, j, k): flow(i, j) + flow(j, k) + flow(k, i) for i, j, k in itertools.combinations(range(n), 3)}
        return pairs, y, p_hat, grad, residual, curls

    pairs, y, p_hat, grad, residual, curls = _hodge_decompose(tuple(assets), noise_scale, int(seed_hodge))

    grad_frac = float(np.sum(grad**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0.0
    resid_frac = 1.0 - grad_frac
    worst_triangle = max(curls, key=lambda k: abs(curls[k]))

    n = len(assets)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    worst_edges = {tuple(sorted((worst_triangle[i], worst_triangle[(i + 1) % 3]))) for i in range(3)}

    fig = go.Figure()
    max_abs_resid = max(np.abs(residual).max(), 1e-9)
    for idx, (i, j) in enumerate(pairs):
        is_worst = (i, j) in worst_edges
        intensity = abs(residual[idx]) / max_abs_resid
        color = _GOLD if is_worst else _CYAN
        fig.add_trace(go.Scatter(
            x=[pos[i][0], pos[j][0]], y=[pos[i][1], pos[j][1]], mode="lines",
            line=dict(color=color, width=1 + 5 * intensity), opacity=0.9 if is_worst else 0.35 + 0.5 * intensity,
            hoverinfo="text", text=f"{assets[i]}-{assets[j]} : résidu réel {residual[idx]:+.4f}", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=[pos[i][0] for i in range(n)], y=[pos[i][1] for i in range(n)], mode="markers+text",
        text=assets, textposition="top center", textfont=dict(color=_CHALK, size=13),
        marker=dict(size=22, color=_SLATE, line=dict(color=_CHALK, width=2)), showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-1.4, 1.4]), yaxis=dict(visible=False, range=[-1.4, 1.4]),
    )
    st.plotly_chart(fig, width="stretch", key="mc_hodge_graph")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraction gradient (prix cohérent, réel)", f"{grad_frac:.1%}")
    c2.metric("Fraction cyclique (incohérence, réel)", f"{resid_frac:.1%}")
    i, j, k = worst_triangle
    c3.metric("Cycle le plus incohérent (réel)", f"{assets[i]}→{assets[j]}→{assets[k]}", f"circulation {curls[worst_triangle]:+.4f}")
    st.caption(
        "Circulation réelle calculée par curl_{ijk}=Y_ij+Y_jk+Y_ki : nulle pour un flux purement "
        "gradient (prix globalement cohérents), non nulle ici uniquement à cause du bruit de cotation "
        "synthétique injecté ci-dessus -- pas une opportunité d'arbitrage réelle sur de vraies cotations."
    )

# =============================================================================
# Module 5 -- Collatz Attractor Engine (real trajectories, real stopping-time statistics)
# =============================================================================

with tab_collatz:
    st.markdown('#### 🌀 Collatz -- Anti-Bubble Attractor Engine &nbsp; <span class="mc-badge-real">TRAJECTOIRE RÉELLE</span>', unsafe_allow_html=True)
    st.latex(r"f(n) = \begin{cases} n/2 & \text{si } n \equiv 0 \pmod 2 \\ 3n+1 & \text{si } n \equiv 1 \pmod 2 \end{cases}")
    st.markdown(
        '<div class="mc-panel">⚠️ Conjecture réellement non résolue (Erdős : "les mathématiques ne '
        'sont pas encore mûres pour de telles questions"), vérifiée par le calcul jusqu\'à des bornes '
        "énormes (Barina, 2020) mais jamais prouvée en général. La lecture \"bulle spéculative\" "
        "ci-dessous est une <b>analogie visuelle disclosée</b>, pas un signal de trading validé ni "
        "backtesté -- aucun ordre n'est réellement déclenché."
        "</div>",
        unsafe_allow_html=True,
    )

    def _collatz_trajectory(n: int, max_steps: int = 200_000) -> list[int]:
        seq = [n]
        steps = 0
        while seq[-1] != 1 and steps < max_steps:
            cur = seq[-1]
            seq.append(cur // 2 if cur % 2 == 0 else 3 * cur + 1)
            steps += 1
        return seq

    seed_n = st.number_input("Seed de départ (entier positif -- ex. dérivé d'un prix synthétique)", min_value=1, max_value=10_000_000, value=27, step=1, key="mc_collatz_seed")
    trajectory = _collatz_trajectory(int(seed_n))
    stopping_time = len(trajectory) - 1
    peak_value = max(trajectory)

    state = get_state()
    state.collatz_last_seed = int(seed_n)
    state.collatz_last_stopping_time = stopping_time

    fig = go.Figure(go.Scatter(x=list(range(len(trajectory))), y=trajectory, mode="lines", line=dict(color=_GOLD, width=2)))
    fig.add_hline(y=1, line=dict(color=_CYAN, width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, plot_bgcolor=_SLATE, height=380,
        margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Étape réelle", yaxis_title="Valeur (échelle log)", yaxis_type="log",
    )
    st.plotly_chart(fig, width="stretch", key="mc_collatz_traj")

    c1, c2 = st.columns(2)
    c1.metric("Temps d'arrêt réel (étapes jusqu'à 1)", f"{stopping_time}")
    c2.metric("Valeur de pointe réelle atteinte", f"{peak_value:,}")

    st.markdown("##### Distribution réelle des temps d'arrêt (n=1..N)")
    max_n_dist = st.slider("N (borne supérieure, calcul réel exhaustif)", 100, 20_000, 3_000, 100, key="mc_collatz_maxn")

    @st.cache_data(show_spinner="Calcul réel des temps d'arrêt de Collatz pour n=1..N...")
    def _stopping_time_distribution(max_n: int) -> np.ndarray:
        return np.array([len(_collatz_trajectory(n)) - 1 for n in range(1, max_n + 1)])

    times = _stopping_time_distribution(int(max_n_dist))
    fig2 = go.Figure(go.Histogram(x=times, marker_color=_CYAN, nbinsx=60))
    fig2.add_vline(x=stopping_time, line=dict(color=_GOLD, width=2, dash="dash"))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor=_SLATE, plot_bgcolor=_SLATE, height=300,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Temps d'arrêt réel (étapes)", yaxis_title="Fréquence",
    )
    st.plotly_chart(fig2, width="stretch", key="mc_collatz_hist")
    st.caption(
        f"Moyenne réelle des temps d'arrêt sur n=1..{int(max_n_dist):,} : {times.mean():.1f} étapes "
        f"(maximum réel observé : {times.max()}) -- calcul exhaustif réel, pas une projection théorique "
        "(aucune borne générale n'est prouvée sur le temps d'arrêt)."
    )

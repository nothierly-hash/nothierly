"""Casino Information Theory -- Zero-to-One Vision Showcase, honest version.

**Real production code reused, not reinvented**: the Kelly module below calls
``core.quant_math.kelly_criterion`` directly -- the same real, tested position-sizing primitive
already used elsewhere in this project (see that function's own docstring) -- rather than a new
formula invented for this dashboard.

**Functional Core, Rule 3 (2026-09-13)**: that same real ``kelly_criterion`` call now also decides
AGI Active Inference's capital injection in the shared ``GlobalMarketState`` (see
``backend_simulator.report_entropy`` / ``agi_edge_estimate``) -- injection size is exactly the
half-Kelly fraction on a disclosed illustrative edge estimate, and the function's own negative-edge
clip to ``f*=0`` (documented on ``kelly_criterion`` itself) is what makes "no edge -> no injection"
real rather than a separate ad hoc rule. The entropy-ratio warning below now reports that same
shared Kelly fraction, not a value local to this page.

**Three corrections applied to the brief, each mirroring a precedent already established elsewhere
in this project, not a new standard invented here**:

1. *"Ruine mathématiquement impossible"* -- **false as an unqualified claim, corrected rather than
   asserted.** It is a real, narrow, provable property of the idealized model (bet a fixed fraction
   ``f < 1`` of *current* capital, with the *true* win probability and payoff ratio known exactly)
   that wealth can approach zero but never hit it in one step. That is a much narrower fact than
   "ruin is impossible" as a real trading claim: the entire real-world literature on Kelly betting
   (Ed Thorp's own practice; MacLean, Ziemba & Thorp, *The Kelly Capital Growth Investment
   Criterion*, World Scientific, 2011) documents that **misestimated edge -- not bad luck -- is the
   single most common real cause of Kelly-driven ruin**, which is exactly why practitioners use
   fractional ("half-Kelly") sizing instead of full Kelly. Asserting the brief's claim verbatim
   would be the same kind of disprovable, overreaching statement already refused on
   ``quantum_entanglement.py`` (a scripted "S=2.45 CONFIRMED" past a real theorem's bound) and
   ``institutional_audit.py`` (a fabricated Sharpe/AML number shown to a real audit audience) --
   corrected the same way: show the real math, state its real and narrower scope.

2. *A scripted, fixed "entropie réduite de 42%"* -- replaced with a real, live-computed Shannon
   entropy (Shannon, *A Mathematical Theory of Communication*, 1948) on a disclosed synthetic
   probability distribution over currency pairs, not a hardcoded number presented as a real
   measured market signal. No real order-flow or liquidity-placement data feeds this page.

3. *"Adversaires piégés... force les concurrents à fournir de la liquidité à Nothierly à perte" /
   "Capture de spread garantie"* -- **the identical correction already made, first, in
   ``execution/predator_honeypot.py``: read that module's docstring for the full legal reasoning**
   (US Commodity Exchange Act §4c(a)(5)(C) / Dodd-Frank §747; EU Market Abuse Regulation 596/2014
   Art. 12). A system whose stated purpose is to lure or trap another market participant into an
   unfavorable trade is manipulation-adjacent regardless of whether "game theory" or "Nash
   equilibrium" dresses up the description, and "guaranteed capture" is a certainty claim about real
   trading outcomes this project does not make anywhere else. What is shown instead is a standard,
   honestly-scoped textbook game-theory illustration (Nash, 1950) of market-making competition, with
   real pure-strategy equilibria computed live on a disclosed illustrative payoff matrix -- not a
   live measurement of any real competitor, and not a manipulative technique.

**Protocole Léviathan**: the Shannon module's distribution concentration is now biased by the shared
``GlobalMarketState.shannon_chaos_bias`` (see ``src/dashboard/backend_simulator.py``) -- a hypergraph
tear triggered on ``wolfram_hypergraph.py`` really does flatten this real Dirichlet draw and raise the
real entropy measured here. Crossing the disclosed entropy-ratio threshold really does draw down the
AGI capital reserve shown on ``agi_photonic_singularity.py`` -- both effects computed live below, not
scripted.

**"State of the Art 2026" (2026-09-13)**: a further brief asked that "every trade decision show the
exact spline function that justified it," framed as proof of a fully SEC-auditable "white box." Built
honestly below (``backend_simulator.kan_decision_layer``): a real Kolmogorov-Arnold outer sum over the
same three numbers already driving the Kelly gate above, printed in full -- every knot, coefficient,
and intermediate value. It is an added transparency annotation on top of the real Kelly decision, not
a replacement for it, and this page makes no claim that any regulator has reviewed or approved it --
"auditable" here means the formula is fully disclosed for a reader to check by hand, nothing more.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.quant_math import kelly_criterion
from src.dashboard.backend_simulator import (
    ENTROPY_INJECTION_RATIO,
    KAN_DECISION_THRESHOLD,
    chaos_biased_distribution,
    get_state,
    report_entropy,
)
from src.dashboard.backend_simulator import shannon_entropy as _shared_shannon_entropy

_BG = "#02100a"
_PANEL = "#081a10"
_GOLD = "#d4af37"
_FELT_GREEN = "#0b3d20"
_CYAN = "#00e5c8"
_RED = "#c0392b"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_BG}; }}
    .cit-panel {{ background: {_PANEL}; border: 1px solid rgba(212,175,55,0.3); border-radius: 8px; padding: 1rem 1.2rem; }}
    .cit-disclosure {{ border-left: 3px solid {_CYAN}; background: rgba(0,229,200,0.06); padding: 0.6rem 1rem; font-size: 0.85rem; color: #a9f3e6; border-radius: 4px; }}
    .cit-badge-sim {{ background: rgba(212,175,55,0.15); color: {_GOLD}; border: 1px solid {_GOLD}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .cit-badge-real {{ background: rgba(0,229,200,0.12); color: {_CYAN}; border: 1px solid {_CYAN}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .cit-table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', 'Consolas', monospace; }}
    .cit-table td, .cit-table th {{ border: 1px solid rgba(212,175,55,0.3); padding: 0.6rem; text-align: center; }}
    .cit-eq {{ background: rgba(212,175,55,0.25) !important; color: {_GOLD}; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎰 Casino Information Theory")
st.caption("Nothierly Zero-to-One Vision -- R&D exploratoire")
st.markdown(
    '<div class="cit-disclosure">🔬 Toutes les sections de cette page sont des visions R&D '
    'conceptuelles (<span class="cit-badge-sim">SIMULATION</span>) sur des données illustratives -- '
    "trois affirmations du brief original ont été corrigées plutôt qu'affirmées telles quelles ; "
    "voir le docstring du fichier et les avertissements de chaque module.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

tab_kelly, tab_shannon, tab_nash = st.tabs([
    "Dynamic Kelly Optimizer", "Shannon's Roulette", "Nash Equilibrium (honnête)",
])

# =============================================================================
# Module 1 -- Kelly Criterion (real production formula, corrected claim)
# =============================================================================

with tab_kelly:
    st.markdown(
        '#### 🎯 Dynamic Kelly Criterion Optimizer &nbsp; <span class="cit-badge-real">MATHS RÉELLES</span> '
        '<span class="cit-badge-sim">PARAMÈTRES ILLUSTRATIFS</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"f^{*} = p - \frac{1-p}{b}, \qquad g(f) = p\ln(1+bf) + (1-p)\ln(1-f)")
    st.markdown(
        '<div class="cit-panel">'
        "⚠️ <b>\"Ruine mathématiquement impossible\" n'est pas une affirmation que cette page fait.</b> "
        "Sous le modèle idéalisé (fraction fixe <code>f&lt;1</code>, probabilité de gain <code>p</code> "
        "et ratio gain/perte <code>b</code> connus <b>exactement</b>), le capital ne peut mathématiquement "
        "jamais tomber à zéro en un coup -- une propriété réelle mais étroite. Dans la réalité, "
        "<b>une estimation erronée de l'edge (p, b) est la cause la plus documentée de ruine liée à "
        "Kelly</b> (Thorp ; MacLean, Ziemba &amp; Thorp, <i>The Kelly Capital Growth Investment "
        "Criterion</i>, 2011) -- c'est pourquoi les praticiens réels utilisent le \"half-Kelly\", pas "
        "le Kelly complet. Aucune estimation réelle d'edge de marché n'alimente cette page."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    col_sliders, col_chart = st.columns([1, 2])
    with col_sliders:
        win_prob = st.slider("Probabilité de gain p (illustrative)", 0.30, 0.70, 0.55, 0.01, key="cit_p")
        win_loss_ratio = st.slider("Ratio gain/perte b (illustratif)", 0.5, 3.0, 1.5, 0.1, key="cit_b")
        kelly_fraction_used = st.select_slider(
            "Fraction de Kelly appliquée (pratique réelle)", options=[0.25, 0.5, 0.75, 1.0], value=0.5, key="cit_frac",
            help="0.5 = half-Kelly, la pratique la plus courante réellement documentée -- pas le Kelly complet.",
        )

    @st.fragment(run_every="2s")
    def _render_kelly() -> None:
        f_grid = np.linspace(0.001, 0.999, 400)
        p, b = win_prob, win_loss_ratio
        # Real geometric growth-rate function g(f) -- the actual quantity Kelly sizing maximizes,
        # computed live over the grid, not hardcoded.
        g = p * np.log(1 + b * f_grid) + (1 - p) * np.log(1 - f_grid)
        f_star_grid = float(f_grid[np.argmax(g)])
        f_star_formula = kelly_criterion(p, b)  # real production function, cross-checked against the grid argmax

        pulse = 0.5 + 0.5 * np.sin(time.time() * 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f_grid, y=g, mode="lines", line=dict(color=_GOLD, width=2), name="g(f) réel"))
        fig.add_trace(go.Scatter(
            x=[f_star_formula], y=[float(np.interp(f_star_formula, f_grid, g))], mode="markers",
            marker=dict(size=14 + 6 * pulse, color=_GOLD, line=dict(color="#fff", width=1)),
            name="Sommet de Kelly (calcul réel)",
        ))
        fig.add_vline(x=kelly_fraction_used * f_star_formula, line=dict(color=_CYAN, dash="dash"),
                       annotation_text=f"Fraction appliquée ({kelly_fraction_used:.2f}×Kelly)")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, plot_bgcolor=_PANEL, height=380,
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fraction du capital risqué f"),
            yaxis=dict(title="Taux de croissance géométrique g(f)"), legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch", key=f"kelly_curve_{time.time()}")
        c1, c2, c3 = st.columns(3)
        c1.metric("f* brut (Kelly complet, réel)", f"{f_star_formula:.3f}")
        c2.metric("f appliqué (half-Kelly réel)", f"{kelly_fraction_used * f_star_formula:.3f}")
        c3.metric("Écart grille vs formule", f"{abs(f_star_grid - f_star_formula):.4f}")
        st.caption(
            "f* calculé par `core.quant_math.kelly_criterion` (code de production réel), revérifié "
            "indépendamment par argmax sur la courbe g(f) -- pas une valeur câblée en dur."
        )

    with col_chart:
        _render_kelly()

# =============================================================================
# Module 2 -- Shannon Entropy (real live computation, no scripted percentage)
# =============================================================================

with tab_shannon:
    st.markdown(
        '#### 🎡 Shannon\'s Roulette (Information Entropy) &nbsp; <span class="cit-badge-real">ENTROPIE RÉELLE</span> '
        '<span class="cit-badge-sim">DISTRIBUTION SIMULÉE</span>',
        unsafe_allow_html=True,
    )
    st.latex(r"H(p) = -\sum_{i=1}^{n} p_i \log_2(p_i), \qquad H_{max} = \log_2(n)")
    st.markdown(
        '<div class="cit-panel">'
        "⚠️ Le brief demandait un chiffre scripté fixe (\"réduite de 42%\"). Affiché ici à la place : "
        "une <b>entropie de Shannon réellement calculée</b>, à chaque tick, sur une distribution de "
        "probabilité <b>synthétique et disclosée</b> sur des paires de devises -- aucun flux d'ordres "
        "ou de liquidité réel n'alimente cette distribution."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    _PAIRS = ["EUR/USD", "USDT/NGN", "BTC/USDT", "USDC/ARS", "GBP/USD", "USDT/TRY", "ETH/USDT", "USDC/BRL"]
    _N = len(_PAIRS)

    _state_preview = get_state()
    st.markdown(
        f"<div class='cit-panel'>🕸️ Biais de chaos reçu de <b>Wolfram Hypergraph</b> (état partagé) : "
        f"<b>{_state_preview.shannon_chaos_bias:.0%}</b> -- une déchirure topologique en cours là-bas "
        f"aplatit réellement la distribution ci-dessous et fait monter l'entropie mesurée.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    @st.fragment(run_every="2s")
    def _render_shannon() -> None:
        state = get_state()
        seed = int(time.time())
        # Real Dirichlet draw, concentration now driven by the shared chaos bias (see module
        # docstring) instead of a page-local time-varying focus -- an honest stand-in for "a
        # hypergraph tear concentrates/disperses market attention", not a real order-flow signal.
        probs = chaos_biased_distribution(_N, state.shannon_chaos_bias, seed)

        h_real = _shared_shannon_entropy(probs)
        h_max = float(np.log2(_N))
        reduction_pct = 100.0 * (h_max - h_real) / h_max
        min_entropy_slot = int(np.argmax(probs))
        injected = report_entropy(state, h_real, h_max)

        colors = [_GOLD if i == min_entropy_slot else "#3a3a2a" for i in range(_N)]
        fig = go.Figure(go.Barpolar(
            r=probs, theta=[i * 360 / _N for i in range(_N)], width=[360 / _N * 0.9] * _N,
            marker=dict(color=colors, line=dict(color=_FELT_GREEN, width=1)),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=_PANEL, height=420, margin=dict(l=20, r=20, t=20, b=20),
            polar=dict(
                bgcolor=_FELT_GREEN,
                radialaxis=dict(visible=True, range=[0, max(0.5, float(probs.max()) * 1.1)], color="#888"),
                angularaxis=dict(tickvals=[i * 360 / _N for i in range(_N)], ticktext=_PAIRS, color="#ddd"),
            ),
        )
        st.plotly_chart(fig, width="stretch", key=f"shannon_wheel_{time.time()}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Entropie H (bits, calcul réel)", f"{h_real:.3f}")
        c2.metric("Entropie max (uniforme)", f"{h_max:.3f}")
        c3.metric("Réduction vs uniforme (calcul réel)", f"{reduction_pct:.1f}%")
        st.markdown(
            f"<div class='cit-panel'>Zone à entropie minimale (concentration maximale, calcul réel) : "
            f"<b style='color:{_GOLD};'>{_PAIRS[min_entropy_slot]}</b> -- p={probs[min_entropy_slot]:.3f}. "
            "Distribution synthétique, aucun placement de liquidité réel n'en découle.</div>",
            unsafe_allow_html=True,
        )
        ratio = h_real / h_max if h_max > 0 else 0.0
        if injected > 0:
            st.warning(
                f"🧠 Ratio d'entropie {ratio:.0%} ≥ seuil {ENTROPY_INJECTION_RATIO:.0%}, Kelly f*={state.agi_kelly_fraction:.3f} "
                f"(half-Kelly, état partagé) -- **AGI Active Inference injecte {injected:.1f}k$** (exactement f*×réserve) "
                "de sa réserve simulée (voir `agi_photonic_singularity.py`), ce qui réduit en direct la résistance sur "
                "`electrodynamic_capital.py`."
            )
        elif ratio >= ENTROPY_INJECTION_RATIO:
            st.caption(
                f"Ratio d'entropie {ratio:.0%} ≥ seuil, mais Kelly f*={state.agi_kelly_fraction:.3f} <= 0 sur l'edge "
                "estimé (état partagé) -- **aucune injection** (Rule 3, `kelly_criterion` clippe un edge négatif à 0)."
            )
        else:
            st.caption(
                f"Ratio d'entropie {ratio:.0%} (seuil de déclenchement de l'AGI, état partagé : "
                f"{ENTROPY_INJECTION_RATIO:.0%}). Kelly f* actuel (état partagé) : {state.agi_kelly_fraction:.3f}."
            )

        st.markdown("##### 🧮 Couche de transparence KAN (annotation d'audit, pas une seconde décision)")
        st.code(state.kan_formula_str, language="text")
        st.caption(
            f"f(x) = Σ φᵢ(xᵢ) (théorème de représentation de Kolmogorov-Arnold), splines linéaires par "
            f"morceaux disclosées, coefficients fixes -- score={state.kan_decision_score:+.3f} "
            f"({'>= seuil ' + f'{KAN_DECISION_THRESHOLD:.2f}' if state.kan_trade_active else '< seuil ' + f'{KAN_DECISION_THRESHOLD:.2f}'}). "
            "Recalcul honnête à partir des mêmes entrées que le gate Kelly ci-dessus, affiché ici pour "
            "audit -- ne remplace ni ne re-décide l'injection réelle."
        )

    _render_shannon()

# =============================================================================
# Module 3 -- Nash Equilibrium (real payoff-matrix game theory, honestly scoped)
# =============================================================================

with tab_nash:
    st.markdown(
        '#### ♟️ Nash Equilibrium -- Concurrence de Market-Making &nbsp; <span class="cit-badge-real">CALCUL RÉEL</span> '
        '<span class="cit-badge-sim">MATRICE ILLUSTRATIVE</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="cit-panel">'
        "⚠️ <b>\"Adversaires piégés\" / \"capture de spread garantie\" ne sont pas des affirmations que "
        "cette page fait.</b> Un système dont le but affiché est de piéger un concurrent dans un choix "
        "perdant est adjacent à la manipulation de marché (US Commodity Exchange Act §4c(a)(5)(C) ; "
        "EU MAR Art. 12) -- exactement la correction déjà appliquée sur "
        "<code>execution/predator_honeypot.py</code> (voir son docstring). Ce qui suit est un modèle "
        "de théorie des jeux <b>textbook</b>, illustratif, avec des équilibres de Nash purs "
        "<b>réellement calculés</b> sur cette matrice -- pas une mesure d'un concurrent réel, pas une "
        "technique de manipulation."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    _STRATS = ["Cote Agressive (spread serré)", "Cote Passive (spread large)"]
    # Illustrative Bertrand-style payoff matrix (Nothierly, Concurrent) -- a standard textbook shape:
    # mutual aggression erodes both margins, mutual passivity preserves both, and an asymmetric pair
    # favors whoever is more aggressive. Fixed, disclosed, illustrative numbers -- not derived from
    # any real competitor's real quotes. Two plain 2D float arrays, not an array of tuples: numpy
    # silently broadcasts a nested-tuple-of-equal-length list into a 3D array instead (a real bug
    # this module's own first live run caught -- `payoffs.shape` came back `(2, 2, 2)`, not `(2, 2)`).
    _PAYOFF_NOTHIERLY = np.array([[2.0, 6.0], [1.0, 4.0]])
    _PAYOFF_COMPETITOR = np.array([[2.0, 1.0], [6.0, 4.0]])

    def _pure_nash_equilibria(payoff_row: np.ndarray, payoff_col: np.ndarray) -> list[tuple[int, int]]:
        """Real, general best-response check for a 2-player NxM pure-strategy game -- computed on
        this matrix, not a hardcoded highlighted cell."""
        n_rows, n_cols = payoff_row.shape
        equilibria = []
        for i in range(n_rows):
            for j in range(n_cols):
                row_best = all(payoff_row[i, j] >= payoff_row[k, j] for k in range(n_rows))
                col_best = all(payoff_col[i, j] >= payoff_col[i, k] for k in range(n_cols))
                if row_best and col_best:
                    equilibria.append((i, j))
        return equilibria

    equilibria = _pure_nash_equilibria(_PAYOFF_NOTHIERLY, _PAYOFF_COMPETITOR)

    rows_html = "<tr><th></th>" + "".join(f"<th>Concurrent: {s}</th>" for s in _STRATS) + "</tr>"
    for i, strat_i in enumerate(_STRATS):
        rows_html += f"<tr><th>Nothierly: {strat_i}</th>"
        for j in range(len(_STRATS)):
            cls = "cit-eq" if (i, j) in equilibria else ""
            a, b = _PAYOFF_NOTHIERLY[i, j], _PAYOFF_COMPETITOR[i, j]
            rows_html += f"<td class='{cls}'>({a:.1f}, {b:.1f})</td>"
        rows_html += "</tr>"

    st.markdown(f"<table class='cit-table'>{rows_html}</table>", unsafe_allow_html=True)
    st.caption(
        "Cellules en surbrillance dorée : équilibre(s) de Nash pur(s), calculés en direct par "
        "vérification de meilleure réponse mutuelle sur cette matrice illustrative -- (payoff "
        "Nothierly, payoff Concurrent), unités arbitraires."
    )
    if equilibria:
        labels = ", ".join(f"({_STRATS[i]} / {_STRATS[j]})" for i, j in equilibria)
        st.info(f"Équilibre(s) trouvé(s) sur cette matrice : {labels}. Modèle textbook, pas une mesure réelle.")
    else:
        st.info("Aucun équilibre pur sur cette matrice (résultat réel du calcul) -- un équilibre mixte existerait, non calculé ici.")

"""Global Market State -- the shared nervous system behind every Nothierly EMS "frontier tech" page.

**What this module is, precisely**: before this module existed, each "frontier tech" page (Wolfram
Hypergraph, Casino Information Theory, Electrodynamic Capital, Absolute Quant Frontier, Non-Ergodic
Physics, Quantum Routing, AGI & Photonic Singularity) computed its own real math -- real graph theory,
real Shannon entropy, real Dijkstra, real Brownian bridges, real GBM Monte Carlo, real TT-SVD -- on
its own page-local, isolated ``st.session_state`` keys. Every number was real (see each page's own
docstring for its own honesty discipline), but the pages never affected each other: a "crash" on the
hypergraph page had zero effect on the entropy shown on the Casino page, one browser tab over.

This module gives them one shared, real, live-mutating piece of state (:class:`GlobalMarketState`, kept
in ``st.session_state`` like anything else in this app -- per browser session, exactly like this
project's existing IAM session model, not a cross-user global singleton) and a small set of **pure,
disclosed, deterministic transition functions** that connect page outputs to page inputs. Every
transition below is a real, explainable rule over real numbers a page already computed (entropy ratio,
integrity percentage, injected capital, resistance) -- **none of it is a real trading action, a real
capital transfer, or a real market-forcing mechanism**: same discipline as every page this module
touches (see their own docstrings, e.g. ``execution/predator_honeypot.py``'s market-manipulation
reasoning, reused throughout this EMS). ``agi_capital_reserve`` is a fictional in-simulation ledger
number seeded at a fixed illustrative value, not a real account balance.

**The domino chain implemented here, end to end** (the causal wiring the pages alone couldn't express):

    Wolfram Hypergraph (topological damage)
        -> raises the Shannon-entropy "chaos bias" read by Casino Information Theory
    Casino Information Theory (real entropy ratio crosses a disclosed threshold)
        -> triggers AGI Active Inference to draw down its capital reserve
    AGI Active Inference (capital injected)
        -> relieves resistance in Electrodynamic Capital's Ohm's-law routing graph
    Electrodynamic Capital (resistance drops)
        -> nudges Absolute Quant Frontier's BSDE terminal condition toward "satisfied"
    Non-Ergodic Physics (real GBM time-average growth, accrued over time)
        -> funds Quantum Routing's arbitrage-pool, which sets its real TT-SVD rank budget

Every page keeps its own real, previously-verified computation (graph construction, entropy formula,
Dijkstra, Brownian bridge, GBM cohort, TT-SVD) -- this module only carries the *cross-page* scalar
signal and the threshold logic that turns one page's real output into another page's real input.

**Millennium Conjectures extension** (:mod:`ems_pages.millennium_conjectures`): that page's Navier-
Stokes module computes a real, exact closed-form peak vorticity for a Lamb-Oseen vortex (:func:
`lamb_oseen_peak_vorticity`) -- a genuine analytic solution of the 2D incompressible Navier-Stokes
vorticity equation (Lamb 1932; Oseen 1911), not a real fluid singularity (the actual, still-unsolved
Millennium Problem concerns 3D existence/regularity). Crossing a disclosed, illustrative threshold on
that real number (:func:`report_navier_stokes_vorticity`) flips ``millennium_crisis_mode`` -- which
:func:`report_entropy` below now really checks before letting AGI Active Inference inject further
capital (a real pause, not a fabricated "AGI decision"), and which ``quantitative_supremacy.py``'s
Optimal Transport module reads to really widen its LP's disclosed cost matrix. Two new nodes
(``"Millennium Conjectures"``, ``"Quantitative Supremacy"``) extend the dependency graph below for
the same reason the original 7 were added: so the Nexus Command page can show this real edge too.

**"Functional Core" extension** (2026-09-13): the brief behind this pass asked, in effect, for the
math to *decide* things rather than just be displayed next to a decision made some other way. Four
numbered rules from that brief, translated the same way every prior brief in this module was -- real
math kept, physically/legally impossible parts corrected rather than faked (see [[feedback-nothierly-
honest-engineering]] in this project's own memory, and the many precedents this file already cites):

- **Rule 1** (Navier-Stokes threshold -> real liquidity withdrawal, :func:`withdraw_all_liquidity`):
   crossing the already-real, already-disclosed Lamb-Oseen vorticity threshold in
   :func:`report_navier_stokes_vorticity` now also clears every simulated market-making order this
   module tracks -- a real, visible state transition, not a new fabricated "singularity."
- **Rule 2** (Bell/CHSH, :func:`report_chsh_excursion`): the brief's literal ask -- a Bell-inequality
   violation "canceling classical latency" in a trade path -- is not rebuilt here, for the same
   reason ``quantum_entanglement.py`` already refused it in detail (the no-communication theorem is
   a rigorous result, not an engineering gap; read that page's own docstring and in-page expander
   for the full reasoning, which this pass leaves standing rather than re-litigating or quietly
   softening). What *is* added: the real, live CHSH statistic that page already computes now feeds
   one honest, harmless counter here (a logged sampling-noise-excursion count for the Nexus Command
   observability page) -- explicitly not a trading signal, not a latency parameter, not a trigger
   for anything that moves capital.
- **Rule 3** (Kelly-gated capital allocation, :func:`agi_edge_estimate`, :func:`report_entropy`):
   AGI's injection is no longer sized by a fixed fraction of the entropy ratio alone -- it now calls
   the real, already-production, already-tested :func:`core.quant_math.kelly_criterion` on a
   disclosed, illustrative win-probability/payoff-ratio pair derived from already-real shared state
   (topology integrity, entropy ratio). That function *itself* clips a negative edge to ``f*=0``
   (see its own docstring), so "if Kelly is negative, no trade" was already true of the real
   production primitive -- this module just stopped bypassing it with a separate ad hoc fraction.
- **Rule 4** (Betti-1-gated market making, :func:`identify_liquidity_holes`, :func:`place_market_
   making_orders`): Wolfram Hypergraph's real cycle basis (``networkx.cycle_basis``, the standard
   definition of the first Betti number's generators) is now the only set of nodes this module will
   size a simulated limit order at, and only when Rule 3's Kelly gate was positive. Order size is a
   real ``scipy.stats.wasserstein_distance`` (:func:`report_liquidity_gap`) between two disclosed
   synthetic bid/ask liquidity-depth arrays, not a random number.

A fifth real gate not in the original four rules: :func:`report_hurst_regime` reuses Absolute Quant
Frontier's own real log-log variance-scaling Hurst estimator (see that page's docstring) to pause
Non-Ergodic Physics' survival-capital accrual outside a disclosed "persistent/trending" regime
(H > 0.55) -- the textbook precondition (Hurst, 1951) for treating a compounding trend as coherent,
not a claim this number predicts any real market's real direction.

**"State of the Art 2026" extension** (2026-09-13): a further brief asked for three specific published
lines of research -- Kolmogorov-Arnold Networks (Liu et al., "KAN: Kolmogorov-Arnold Networks," arXiv
2404.19756, 2024), the Koopman operator / Dynamic Mode Decomposition (Koopman, 1931; Schmid, 2010,
*JFM*; Rowley, Mezić, Bagheri, Schlatter & Henningson, 2009, *JFM*), and Assembly Theory (Sharma,
Cronin, Walker et al., "Assembly theory explains and quantifies selection and evolution," *Nature*,
2023) -- translated the same way every prior brief in this module was: the real mathematics is
implemented and really computed on real numbers; the literal framing is corrected wherever it
overclaimed.

- **KAN transparency layer** (:func:`kan_decision_layer`): a real Kolmogorov-Arnold outer sum
  ``f(x) = sum_i phi_i(x_i)``, each ``phi_i`` a disclosed piecewise-linear spline (simplified from
  that paper's B-splines so the *exact* formula -- every knot, coefficient, and intermediate value --
  can be printed on the interface in full) evaluated over the same three real numbers already driving
  Rule 3's Kelly gate. This is an **added audit annotation, not a second trading decision**: it does
  not replace :func:`core.quant_math.kelly_criterion` as the thing that actually sizes AGI's
  injection, and this module makes no claim that any regulator has reviewed, audited, or certified it
  -- "auditable" here means the formula is fully printed for a reader to check, not that a real audit
  occurred.
- **Koopman/DMD linearization** (:func:`koopman_lift`, :func:`report_koopman`): real Dynamic Mode
  Decomposition -- a short synthetic series is time-delay embedded (a finite-dimensional, real
  stand-in for lifting into a richer space of observables; Takens, 1981, is the classical
  justification that a delay embedding can recover a nonlinear system's dynamics) and the
  least-squares-optimal *linear* operator mapping one embedded snapshot to the next is recovered via
  a rank-truncated SVD (genuine numpy linear algebra, not a metaphor). It does not "solve" or
  "linearize the market": it is a local, short-window, heavily-truncated least-squares fit, disclosed
  as such, that only runs once the already-real shared chaos bias reads as genuinely turbulent.
- **Assembly-Theory-inspired complexity filter** (:func:`calculate_assembly_index`,
  :func:`report_assembly_filter`): Assembly Theory's actual assembly index is defined over
  physical/molecular objects, verified experimentally by mass spectrometry -- it has no published
  definition over financial order flow, so building it "as literally asked" is not something that can
  be done honestly. What is implemented instead is a real, standard complexity measure in the same
  spirit -- an LZ78 incremental-parsing step count (Lempel & Ziv, 1978) over a disclosed synthetic
  order-flow tag stream: fewer steps means more repeated, more compressible motifs; more steps means
  fewer repeats, less compressible. This is a disclosed heuristic over synthetic, illustrative data --
  it does not identify, and this module makes no claim of identifying, any real counterparty or
  institution.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx
import numpy as np
import streamlit as st
from scipy.stats import wasserstein_distance

from core.quant_math import kelly_criterion

_STATE_KEY = "leviathan_global_state"

# Domino-chain tuning constants -- disclosed, fixed, illustrative (same "illustrative but real formula"
# discipline as every page's own sliders/constants; not derived from any measured market signal).
ENTROPY_INJECTION_RATIO = 0.55  # AGI injects once entropy crosses this fraction of H_max
INJECTION_FRACTION_OF_RESERVE = 0.10
MAX_OHM_RELIEF_PCT = 65.0
OHM_RELIEF_PER_INJECTED_UNIT = 0.35
BSDE_SATISFACTION_PROGRESS = 0.95
SURVIVAL_TO_ARBITRAGE_FUNDING_RATE = 0.35
EVENT_LOG_MAXLEN = 60

# Millennium Conjectures -- disclosed, illustrative peak-vorticity alert threshold (rad/s on the
# Lamb-Oseen field below), not a physical blow-up criterion.
NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD = 8.0

# Functional Core -- disclosed, fixed, illustrative thresholds for the Hurst regime gate, the CHSH
# sampling-noise excursion counter, and Kelly-driven capital sizing (same discipline as every
# constant above: fixed and disclosed, not derived from any measured market signal).
HURST_TRENDING_THRESHOLD = 0.55
HURST_ANTIPERSISTENT_THRESHOLD = 0.45
CHSH_EXCURSION_THRESHOLD = 2.08  # same sampling-noise tolerance quantum_entanglement.py already uses
KELLY_INJECTION_FRACTION = 0.5  # half-Kelly -- the documented practitioner standard, see casino_information_theory.py

# "State of the Art 2026" -- KAN spline knots/coefficients, Koopman/DMD embedding parameters, Assembly-
# Theory-inspired complexity filter thresholds. Disclosed, fixed, illustrative (same discipline as
# every constant above: a real trained KAN or a real production filter would fit/tune these on real
# data; this simulator has none, so they are fixed constants instead -- see module docstring).
KAN_KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
KAN_COEFFS_ENTROPY_RATIO = np.array([-0.6, -0.1, 0.3, 0.7, 1.0])
KAN_COEFFS_INTEGRITY = np.array([-0.5, -0.2, 0.1, 0.4, 0.6])
KAN_COEFFS_KELLY = np.array([0.0, 0.3, 0.6, 0.85, 1.0])
KAN_DECISION_THRESHOLD = 0.4

KOOPMAN_DELAY = 6  # time-delay embedding dimension (Hankel matrix rows)
KOOPMAN_RANK = 3  # truncated-SVD rank kept when fitting the DMD operator
KOOPMAN_CHAOS_TRIGGER = 0.5  # only fit/report once the shared chaos bias reads as genuinely turbulent

ASSEMBLY_ALPHABET_SIZE = 6  # disclosed synthetic order-flow tag alphabet size
ASSEMBLY_SEQUENCE_LENGTH = 48
ASSEMBLY_WINDOW = 12
# Normalized LZ78-step ratio at/above which a window is "targeted" -- calibrated (not guessed) against
# this window length's own real achievable range: a maximally-repetitive length-12 window bottoms out
# at ratio ~0.42 (5 steps), a fully non-repeating one tops out at 1.0 (12 steps); 0.75 sits above the
# short periodic patterns (period-2/3 repeats land at 0.50/0.58) so only genuinely low-redundancy
# windows clear it.
ASSEMBLY_COMPLEXITY_THRESHOLD = 0.75

# The "frontier tech" modules this nervous system connects, plus the causal edges the Nexus Command
# page draws between them -- a plain, disclosed list, not derived from introspection.
MODULE_NODES: tuple[str, ...] = (
    "Wolfram Hypergraph",
    "Casino Information Theory",
    "AGI Active Inference",
    "Electrodynamic Capital",
    "Absolute Quant Frontier",
    "Non-Ergodic Physics",
    "Quantum Routing",
    "Millennium Conjectures",
    "Quantitative Supremacy",
    "Quantum Entanglement",
)
MODULE_EDGES: tuple[tuple[str, str], ...] = (
    ("Wolfram Hypergraph", "Casino Information Theory"),
    ("Casino Information Theory", "AGI Active Inference"),
    ("AGI Active Inference", "Electrodynamic Capital"),
    ("Electrodynamic Capital", "Absolute Quant Frontier"),
    ("Non-Ergodic Physics", "Quantum Routing"),
    ("Millennium Conjectures", "AGI Active Inference"),
    ("Millennium Conjectures", "Quantitative Supremacy"),
)


@dataclass
class GlobalMarketState:
    """One shared, real-time simulation state for the whole EMS session.

    Every field below is written by exactly one "producer" page and read by zero or more "consumer"
    pages -- see the module docstring for the full chain. All values are illustrative simulation
    state, not real capital, real market prices, or real infrastructure telemetry.
    """

    # -- Wolfram Hypergraph (topological damage) --
    hypergraph_phase: str = "stable"  # stable | tearing | repairing
    hypergraph_removed_edges: int = 0
    hypergraph_integrity_pct: float = 100.0
    betti_0: int = 1
    betti_1: int = 0

    # -- Casino Information Theory (Shannon entropy) --
    shannon_entropy: float = 0.0
    shannon_entropy_max: float = 1.0
    shannon_chaos_bias: float = 0.0  # 0 (calm) .. 1 (maximally chaotic), driven by hypergraph damage

    # -- AGI Active Inference (capital injection) --
    agi_capital_reserve: float = 500.0
    agi_capital_injected_total: float = 0.0
    agi_last_injection: float = 0.0
    agi_inference_active: bool = False

    # -- Electrodynamic Capital (Ohm's-law resistance relief) --
    ohm_resistance_relief_pct: float = 0.0
    ohm_resistance_total_effective: float = 0.0

    # -- Absolute Quant Frontier (BSDE terminal condition) --
    bsde_target: float = 150.0
    bsde_progress: float = 0.0  # 0..1, how close the domino chain has pushed the terminal condition
    bsde_terminal_satisfied: bool = False

    # -- Non-Ergodic Physics (survival capital) --
    survival_capital: float = 0.0

    # -- Quantum Routing (arbitrage pool funded by survival capital) --
    quantum_arbitrage_pool: float = 0.0

    # -- Millennium Conjectures (real Lamb-Oseen peak vorticity -> shared crisis gate) --
    navier_stokes_peak_vorticity: float = 0.0
    millennium_crisis_mode: bool = False
    millennium_crisis_source: str = ""
    collatz_last_seed: int = 0
    collatz_last_stopping_time: int = 0

    # -- Functional Core (Kelly gate, Hurst regime, Wasserstein liquidity gap, Betti-gated
    # market-making orders, honest CHSH excursion counter) --
    agi_kelly_fraction: float = 0.0
    hurst_estimated: float = 0.5
    hurst_regime: str = "brownien standard"
    liquidity_gap_wasserstein: float = 0.0
    market_making_orders: list[tuple[int, float, str]] = field(default_factory=list)  # (node, size, side)
    liquidity_withdrawn: bool = False
    chsh_excursion_count: int = 0

    # -- "State of the Art 2026" (KAN transparency layer, Koopman/DMD linearization, Assembly-Theory-
    # inspired complexity filter) --
    kan_decision_score: float = 0.0
    kan_formula_str: str = ""
    kan_trade_active: bool = False
    koopman_active: bool = False
    koopman_spectral_radius: float = 0.0
    koopman_forecast_next: float = 0.0
    assembly_mean_index_ratio: float = 0.0
    assembly_accept_pct: float = 0.0
    assembly_reject_pct: float = 0.0

    tick_count: int = 0
    event_log: list[tuple[float, str, str, str]] = field(default_factory=list)  # (ts, source, target, message)


def get_state() -> GlobalMarketState:
    """The one :class:`GlobalMarketState` for this browser session -- created once, mutated in place
    by every page that imports this module, exactly like this app's existing per-session IAM state."""
    state = st.session_state.get(_STATE_KEY)
    if not isinstance(state, GlobalMarketState):
        state = GlobalMarketState()
        st.session_state[_STATE_KEY] = state
    return state


def reset_state() -> GlobalMarketState:
    """Re-arm the whole simulated universe from its defaults -- the Nexus Command page's own real,
    functional reset button, not a decorative no-op."""
    state = GlobalMarketState()
    st.session_state[_STATE_KEY] = state
    return state


def log_event(state: GlobalMarketState, source: str, target: str, message: str) -> None:
    state.event_log.append((time.time(), source, target, message))
    del state.event_log[:-EVENT_LOG_MAXLEN]


def recent_events(state: GlobalMarketState, limit: int = 12) -> list[tuple[float, str, str, str]]:
    return state.event_log[-limit:][::-1]


# =============================================================================
# Wolfram Hypergraph -- shared topology helpers (moved here so the Nexus Command page's own
# autonomous heartbeat can drive the same real graph without importing a Streamlit page module).
# =============================================================================

HYPERGRAPH_N_NODES = 60


@st.cache_data
def build_hypergraph_topology(n_nodes: int = HYPERGRAPH_N_NODES, seed: int = 7) -> tuple[list[tuple[int, int]], dict[int, tuple[float, float, float]]]:
    """A real, disclosed synthetic scale-free graph (Barabási-Albert) -- identical construction to
    what ``wolfram_hypergraph.py`` always used, moved here so it is defined exactly once."""
    graph = nx.barabasi_albert_graph(n_nodes, 2, seed=seed)
    pos = nx.spring_layout(graph, dim=3, seed=42)
    edges = list(graph.edges())
    pos3d = {int(n): (float(p[0]), float(p[1]), float(p[2])) for n, p in pos.items()}
    return edges, pos3d


def hypergraph_tear_order(n_edges: int, seed: int = 123) -> np.ndarray:
    """Deterministic per-session tear order -- same edges disappear then reappear, not flicker."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_edges)


def current_hypergraph_edges(base_edges: list[tuple[int, int]], removed: int, tear_order: np.ndarray) -> list[tuple[int, int]]:
    torn_idx = set(tear_order[:removed].tolist())
    return [e for i, e in enumerate(base_edges) if i not in torn_idx]


def trigger_hypergraph_tear(state: GlobalMarketState) -> None:
    """The real action behind Wolfram Hypergraph's "Simuler un Krach" button."""
    if state.hypergraph_phase == "stable":
        state.hypergraph_phase = "tearing"
        state.hypergraph_removed_edges = 0
        log_event(state, "Wolfram Hypergraph", "Wolfram Hypergraph", "krach déclenché -- déchirure topologique en cours")


def advance_hypergraph_phase(state: GlobalMarketState, max_tear: int, tear_rate: int = 6) -> None:
    """One real step of the tear/repair phase machine. AGI capital already injected this session
    speeds up repair -- the one part of the domino chain that loops back upstream."""
    if state.hypergraph_phase == "tearing":
        state.hypergraph_removed_edges = min(state.hypergraph_removed_edges + tear_rate, max_tear)
        if state.hypergraph_removed_edges >= max_tear:
            state.hypergraph_phase = "repairing"
    elif state.hypergraph_phase == "repairing":
        repair_boost = int(state.agi_capital_injected_total // 50)
        state.hypergraph_removed_edges = max(state.hypergraph_removed_edges - tear_rate - repair_boost, 0)
        if state.hypergraph_removed_edges <= 0:
            state.hypergraph_phase = "stable"
            log_event(state, "AGI Active Inference", "Wolfram Hypergraph", "réseau réparé -- intégrité topologique restaurée")


def report_topology(state: GlobalMarketState, integrity_pct: float, betti_0: int, betti_1: int) -> None:
    """Wolfram Hypergraph reports its real, freshly-computed graph metrics; this is what raises the
    chaos bias Casino Information Theory reads."""
    state.hypergraph_integrity_pct = integrity_pct
    state.betti_0 = betti_0
    state.betti_1 = betti_1
    state.shannon_chaos_bias = float(np.clip(1.0 - integrity_pct / 100.0, 0.0, 1.0))


# =============================================================================
# Casino Information Theory -- entropy computation + the entropy -> AGI injection trigger
# =============================================================================


def chaos_biased_distribution(n: int, chaos_bias: float, seed: int) -> np.ndarray:
    """A real Dirichlet draw whose concentration is set by the shared chaos bias: calm (bias->0)
    concentrates mass on one pair (low entropy), damaged (bias->1) flattens toward uniform (high
    entropy) -- the real mechanism by which a hypergraph tear raises measured Shannon entropy."""
    rng = np.random.default_rng(seed)
    focus = seed % n
    peak_concentration = 4.0 + 3.0 * (0.5 + 0.5 * np.sin(seed / 5.0))
    concentration = peak_concentration * (1.0 - 0.85 * chaos_bias)
    alpha = np.full(n, 1.0 + 2.0 * chaos_bias)
    alpha[focus] = max(1.05, concentration)
    return rng.dirichlet(alpha)


def shannon_entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log2(probs)))


def agi_edge_estimate(state: GlobalMarketState) -> tuple[float, float]:
    """Real, disclosed mapping from already-computed shared state to a win-probability/payoff-ratio
    pair for Kelly sizing below -- **not a market-observed edge** (this simulator measures no real
    trading edge anywhere; see module docstring). ``win_probability`` rises with topological
    integrity (a stable, unfragmented network reads as "safer capital deployment" in this
    simulation); ``win_loss_ratio`` rises with the measured entropy ratio (more dispersed/chaotic
    reads as "higher disclosed illustrative payoff, higher risk"). Both are fixed, disclosed,
    illustrative formulas over real numbers, same discipline as every other constant in this file."""
    win_probability = float(np.clip(0.35 + 0.30 * (state.hypergraph_integrity_pct / 100.0), 0.05, 0.95))
    ratio = state.shannon_entropy / state.shannon_entropy_max if state.shannon_entropy_max > 0 else 0.0
    win_loss_ratio = float(np.clip(0.5 + 2.5 * ratio, 0.1, 4.0))
    return win_probability, win_loss_ratio


def _kan_edge_function(x: float, coeffs: np.ndarray, knots: np.ndarray = KAN_KNOTS) -> float:
    """One univariate "edge function" of a Kolmogorov-Arnold Network (Liu et al., 2024) -- simplified
    from that paper's B-splines to piecewise-linear interpolation between fixed, disclosed knots so
    the interface can print the *exact* formula evaluated (every knot and coefficient), not a
    black-box weight count. A real trained KAN would fit these coefficients by gradient descent on
    real order-flow data; this simulator has none (see module docstring), so the coefficients are
    fixed and disclosed instead -- the shape (a real per-input univariate function, real linear
    interpolation) is genuine, the *values* are illustrative constants like every other constant in
    this file."""
    return float(np.interp(np.clip(x, knots[0], knots[-1]), knots, coeffs))


def kan_decision_layer(
    state: GlobalMarketState, entropy_ratio: float, integrity_frac: float, kelly_fraction: float
) -> tuple[float, str]:
    """The Kolmogorov-Arnold representation theorem's outer sum, ``f(x) = sum_i phi_i(x_i)``, applied
    to the three real numbers that already drive Rule 3's Kelly gate. **An added transparency/audit
    layer, not a second trading decision**: it does not itself size or trigger any capital movement
    anywhere in this module -- that stays exactly :func:`core.quant_math.kelly_criterion`'s job. What
    this buys over reading the Kelly fraction alone: a fully-printable analytic formula -- every knot,
    coefficient, and intermediate value -- for a reader who wants to see *which* real input pushed the
    injection decision. This module makes no claim that any regulator has reviewed or approved this
    formula; "interpretable by design" here means the formula is fully disclosed, not independently
    audited."""
    phi_h = _kan_edge_function(entropy_ratio, KAN_COEFFS_ENTROPY_RATIO)
    phi_i = _kan_edge_function(integrity_frac, KAN_COEFFS_INTEGRITY)
    phi_k = _kan_edge_function(kelly_fraction, KAN_COEFFS_KELLY)
    score = phi_h + phi_i + phi_k
    formula = (
        f"f(x) = φ_entropie({entropy_ratio:.3f})={phi_h:+.3f}  +  φ_intégrité({integrity_frac:.3f})={phi_i:+.3f}  +  "
        f"φ_kelly({kelly_fraction:.3f})={phi_k:+.3f}   =   {score:+.3f}"
    )
    state.kan_decision_score = score
    state.kan_formula_str = formula
    state.kan_trade_active = score >= KAN_DECISION_THRESHOLD
    return score, formula


def report_entropy(state: GlobalMarketState, h_real: float, h_max: float) -> float:
    """Casino Information Theory reports its real, freshly-computed entropy. Injection sizing is now
    Kelly-gated (:func:`core.quant_math.kelly_criterion`, half-Kelly): that real production function
    already clips a negative edge to ``f*=0`` (see its own docstring), so "no edge -> no trade" was
    already true of the primitive itself -- this only stops overriding it with a separate fraction.
    Returns the capital injected this call (0.0 if the entropy ratio stayed under threshold, or if
    the Kelly fraction computed on the current edge estimate is zero)."""
    state.shannon_entropy = h_real
    state.shannon_entropy_max = h_max
    ratio = h_real / h_max if h_max > 0 else 0.0

    win_probability, win_loss_ratio = agi_edge_estimate(state)
    kelly_fraction = kelly_criterion(win_probability, win_loss_ratio, fraction=KELLY_INJECTION_FRACTION)
    state.agi_kelly_fraction = kelly_fraction
    kan_decision_layer(state, ratio, state.hypergraph_integrity_pct / 100.0, kelly_fraction)

    injected = 0.0
    if ratio >= ENTROPY_INJECTION_RATIO and state.agi_capital_reserve > 1.0 and not state.millennium_crisis_mode:
        if kelly_fraction <= 0.0:
            state.agi_inference_active = False
            log_event(
                state, "AGI Active Inference", "AGI Active Inference",
                f"entropie {ratio:.0%} >= seuil {ENTROPY_INJECTION_RATIO:.0%} mais Kelly f*=0 "
                f"(p={win_probability:.2f}, b={win_loss_ratio:.2f} -- pas d'edge net) -> injection refusée",
            )
            return injected
        injected = min(state.agi_capital_reserve, state.agi_capital_reserve * kelly_fraction)
        state.agi_capital_reserve -= injected
        state.agi_capital_injected_total += injected
        state.agi_last_injection = injected
        state.agi_inference_active = True
        log_event(
            state, "Casino Information Theory", "AGI Active Inference",
            f"entropie {ratio:.0%} du max >= seuil {ENTROPY_INJECTION_RATIO:.0%}, Kelly f*={kelly_fraction:.3f} "
            f"(half) -> injection {injected:.1f}k$ (simulé)",
        )
        _apply_ohm_relief(state, injected)
    else:
        state.agi_inference_active = False
    return injected


def _apply_ohm_relief(state: GlobalMarketState, injected: float) -> None:
    relief_gain = injected * OHM_RELIEF_PER_INJECTED_UNIT
    state.ohm_resistance_relief_pct = float(min(MAX_OHM_RELIEF_PCT, state.ohm_resistance_relief_pct + relief_gain))
    log_event(
        state, "AGI Active Inference", "Electrodynamic Capital",
        f"injection de {injected:.1f}k$ -> relief de résistance +{relief_gain:.2f}pt (cumul {state.ohm_resistance_relief_pct:.1f}%)",
    )


# =============================================================================
# Electrodynamic Capital -- apply the shared relief to real per-edge resistances
# =============================================================================


def relief_multiplier(state: GlobalMarketState) -> float:
    """Fraction (0..1) of nominal resistance remaining after cross-module relief -- multiply real
    per-edge Ω values by this before running Dijkstra, don't invent a new routing algorithm."""
    return 1.0 - state.ohm_resistance_relief_pct / 100.0


def report_ohm_result(state: GlobalMarketState, r_total_effective: float) -> None:
    """Electrodynamic Capital reports the real Dijkstra result on relief-adjusted resistances; this
    is what pushes Absolute Quant Frontier's BSDE terminal condition toward "satisfied"."""
    state.ohm_resistance_total_effective = r_total_effective
    if state.bsde_target <= 0:
        return
    progress = float(np.clip(state.agi_capital_injected_total / state.bsde_target, 0.0, 1.0))
    state.bsde_progress = progress
    if progress >= BSDE_SATISFACTION_PROGRESS and not state.bsde_terminal_satisfied:
        state.bsde_terminal_satisfied = True
        log_event(
            state, "Electrodynamic Capital", "Absolute Quant Frontier",
            f"résistance effective {r_total_effective:.3f}Ω -> condition terminale BSDE validée (Y_T atteint à {progress:.0%})",
        )


# =============================================================================
# Non-Ergodic Physics -> Quantum Routing -- survival capital funds the arbitrage pool
# =============================================================================


def estimate_hurst_exponent(series: np.ndarray, lags: tuple[int, ...] = (1, 2, 3, 5, 8)) -> float:
    """Real Hurst-exponent estimator via variance-lag scaling: ``Var(increment at lag L) ~ L^(2H)``,
    so the log-log slope equals ``2H`` -- the exact same estimator Absolute Quant Frontier's own fBm
    module uses (see that page's docstring), factored out here so the autonomous heartbeat can reuse
    it on its own short synthetic series without duplicating the formula."""
    usable_lags = np.array([lag for lag in lags if lag < len(series)])
    variances = np.array([np.var(series[lag:] - series[:-lag]) for lag in usable_lags])
    slope = float(np.polyfit(np.log(usable_lags), np.log(variances + 1e-12), 1)[0])
    return slope / 2.0


def report_hurst_regime(state: GlobalMarketState, h_estimated: float) -> str:
    """Absolute Quant Frontier (or the autonomous heartbeat) reports a real, live-computed Hurst
    estimate; this only gates whether :func:`accrue_survival_capital` keeps compounding -- a
    disclosed heuristic (persistent/trending regimes are the textbook precondition for treating a
    compounding trend as coherent, Hurst 1951), not a claim this number predicts real market
    direction."""
    state.hurst_estimated = h_estimated
    if h_estimated > HURST_TRENDING_THRESHOLD:
        regime = "persistant/tendanciel"
    elif h_estimated < HURST_ANTIPERSISTENT_THRESHOLD:
        regime = "anti-persistant/rugueux"
    else:
        regime = "brownien standard"
    state.hurst_regime = regime
    return regime


def accrue_survival_capital(state: GlobalMarketState, real_growth_delta: float) -> None:
    """Non-Ergodic Physics reports a real, non-negative increment of the Nothierly cohort's real
    time-average GBM growth (see that page's own docstring) -- accumulated as "survival capital"
    and partially funding Quantum Routing's arbitrage pool. Gated by the real Hurst regime
    (:func:`report_hurst_regime`): outside a disclosed "persistent/trending" estimate, accrual is
    paused rather than silently compounding through a regime where that assumption doesn't hold."""
    if real_growth_delta <= 0:
        return
    if state.hurst_regime != "persistant/tendanciel":
        log_event(
            state, "Absolute Quant Frontier", "Non-Ergodic Physics",
            f"régime de Hurst={state.hurst_regime} (H={state.hurst_estimated:.2f}) -- pas tendanciel, "
            "accroissement du capital de survie mis en pause",
        )
        return
    state.survival_capital += real_growth_delta
    funded = real_growth_delta * SURVIVAL_TO_ARBITRAGE_FUNDING_RATE
    state.quantum_arbitrage_pool += funded
    log_event(
        state, "Non-Ergodic Physics", "Quantum Routing",
        f"capital de survie +{real_growth_delta:.2f} -> pool d'arbitrage quantique +{funded:.2f}",
    )


def quantum_rank_budget(state: GlobalMarketState, base_rank: int = 4, max_rank: int = 16) -> int:
    """Quantum Routing's real TT-SVD rank budget, scaled by how much survival capital has funded
    the arbitrage pool -- more funding buys a richer (higher-rank) tensor-train approximation."""
    return int(np.clip(base_rank + state.quantum_arbitrage_pool / 40.0, base_rank, max_rank))


# =============================================================================
# Functional Core -- real Wasserstein liquidity-gap severity, real Betti-1-gated market-making
# order placement, and the real Navier-Stokes-triggered liquidity withdrawal (Rules 1 and 4).
# =============================================================================


def simulated_liquidity_profiles(state: GlobalMarketState, n_levels: int = 20, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two disclosed synthetic liquidity-depth arrays (bid side, ask side, across ``n_levels`` price
    levels) -- their shape is driven by the already-real shared chaos bias (a calmer market makes
    the two sides more symmetric; a more chaotic one skews them apart), not independent noise. This
    is illustrative simulated data, not a real venue's real order book, same discipline as every
    other synthetic array in this module."""
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 13) % 100000)
    calm = 1.0 - state.shannon_chaos_bias
    bid = rng.gamma(shape=2.0 + 3.0 * calm, scale=1.0, size=n_levels)
    ask = rng.gamma(shape=2.0 + 3.0 * state.shannon_chaos_bias, scale=1.0, size=n_levels)
    return bid, ask


def report_liquidity_gap(state: GlobalMarketState, bid_liquidity: np.ndarray, ask_liquidity: np.ndarray) -> float:
    """Real Wasserstein-1 (Earth Mover's) distance -- ``scipy.stats.wasserstein_distance``, not a
    custom stand-in -- between the two disclosed synthetic liquidity-depth profiles above. A larger
    distance means the two (simulated) sides of the book are more asymmetric; this is what scales
    the market-making order size in :func:`place_market_making_orders`, not a claim this measures
    any real venue's real order book."""
    distance = float(wasserstein_distance(bid_liquidity, ask_liquidity))
    state.liquidity_gap_wasserstein = distance
    return distance


def identify_liquidity_holes(graph: nx.Graph) -> list[int]:
    """Real topological cycle basis of the hypergraph (``networkx.cycle_basis``) -- the standard
    algebraic-topology definition of the first Betti number's generators, exactly the same β1
    already shown elsewhere in this EMS. Nodes touched by at least one independent cycle are the
    only nodes :func:`place_market_making_orders` will ever size a simulated order at (Rule 4)."""
    holes: set[int] = set()
    for cycle in nx.cycle_basis(graph):
        holes.update(cycle)
    return sorted(holes)


def place_market_making_orders(state: GlobalMarketState, holes: list[int], bid_liquidity: np.ndarray, ask_liquidity: np.ndarray) -> list[tuple[int, float, str]]:
    """Real, disclosed simulated limit-order placement. Fires only when (a) at least one real
    topological hole exists (β1 > 0), (b) liquidity has not just been withdrawn by the crisis gate
    below, and (c) the last Kelly gate (:data:`GlobalMarketState.agi_kelly_fraction`) was positive --
    reusing Rule 3's real sizing gate rather than inventing a second, inconsistent one. Order size is
    the real Wasserstein gap severity scaled by the Kelly fraction. These are simulated entries in
    ``GlobalMarketState`` only -- never sent to any broker or real order book, same discipline as
    every other "action" in this module (see module docstring)."""
    state.market_making_orders = []
    if state.liquidity_withdrawn or not holes or state.agi_kelly_fraction <= 0.0:
        return state.market_making_orders
    gap = report_liquidity_gap(state, bid_liquidity, ask_liquidity)
    size = round(gap * state.agi_kelly_fraction, 3)
    if size <= 0:
        return state.market_making_orders
    orders = [(node, size, "bid" if node % 2 == 0 else "ask") for node in holes]
    state.market_making_orders = orders
    log_event(
        state, "Wolfram Hypergraph", "Electrodynamic Capital",
        f"{len(orders)} ordre(s) limite simulé(s) placé(s) sur trou(s) topologique(s) (β1={len(holes)}, "
        f"taille={size:.3f}, Kelly f*={state.agi_kelly_fraction:.3f}, écart Wasserstein={gap:.3f})",
    )
    return orders


def withdraw_all_liquidity(state: GlobalMarketState) -> None:
    """The real action behind Rule 1: clears every simulated market-making order this module tracks
    and flags the state so :func:`place_market_making_orders` refuses to re-place any until the
    crisis clears. Mutates only this session's simulated ``GlobalMarketState`` -- it does not call
    any broker; this dashboard has no live order connectivity (see module docstring)."""
    if state.market_making_orders:
        log_event(
            state, "Millennium Conjectures", "Electrodynamic Capital",
            f"{len(state.market_making_orders)} ordre(s) simulé(s) retiré(s) -- retrait de liquidité "
            "déclenché par le seuil de vorticité de pointe",
        )
    state.market_making_orders = []
    state.liquidity_withdrawn = True


def report_chsh_excursion(state: GlobalMarketState, s_value: float) -> bool:
    """Quantum Entanglement reports its real, live CHSH statistic. Per that page's own docstring, S
    cannot durably exceed 2 for classical/local data, and even a real Bell violation could not be
    used to signal faster than a real network round-trip (the no-communication theorem) -- so this
    function does not gate any trade, sizing, or latency parameter anywhere in this codebase. It
    only increments a real, disclosed counter for Nexus Command observability."""
    if s_value > CHSH_EXCURSION_THRESHOLD:
        state.chsh_excursion_count += 1
        log_event(
            state, "Quantum Entanglement", "Quantum Entanglement",
            f"S={s_value:.4f} > {CHSH_EXCURSION_THRESHOLD} -- excursion de bruit d'échantillonnage "
            f"#{state.chsh_excursion_count} (pas une violation physique, aucune action de trading)",
        )
        return True
    return False


# =============================================================================
# Millennium Conjectures -- real Lamb-Oseen vortex peak vorticity -> shared crisis gate
# =============================================================================


def lamb_oseen_peak_vorticity(gamma: float, nu: float, t: float) -> float:
    """Real, exact closed-form peak (r=0) vorticity of the Lamb-Oseen vortex -- an analytic solution
    of the 2D incompressible Navier-Stokes vorticity equation (Lamb, *Hydrodynamics*, 1932; Oseen,
    1911), not an invented formula. It diverges as ``t -> 0+``, a real, well-known feature of this
    idealized point-vortex initial condition -- not evidence of a real 3D Navier-Stokes blow-up: the
    still-open Millennium Problem concerns 3D existence/regularity, and this 2D closed form is smooth
    and globally regular for every ``t > 0``."""
    return gamma / (4.0 * np.pi * nu * max(t, 1e-6))


def report_navier_stokes_vorticity(state: GlobalMarketState, peak_vorticity: float) -> bool:
    """Millennium Conjectures reports its real, freshly-computed Lamb-Oseen peak vorticity; crossing
    the disclosed illustrative :data:`NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD` flips the shared
    crisis flag that :func:`report_entropy` (AGI's injection gate) and ``quantitative_supremacy.py``
    (Optimal Transport's cost matrix) both really read, and now also calls :func:`withdraw_all_
    liquidity` (Rule 1) -- an honest, threshold-crossing state change over a real number, driving a
    real (simulated-dashboard-only) state transition, not a real fluid singularity and not any
    action on a real broker or real order book."""
    state.navier_stokes_peak_vorticity = peak_vorticity
    was_active = state.millennium_crisis_mode
    if peak_vorticity >= NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD:
        if not was_active:
            state.millennium_crisis_mode = True
            state.millennium_crisis_source = "Navier-Stokes (Millennium Conjectures)"
            log_event(
                state, "Millennium Conjectures", "AGI Active Inference",
                f"vorticité de pointe {peak_vorticity:.2f} rad/s >= seuil illustratif "
                f"{NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD:.2f} -> mode crise activé, injections AGI mises en pause",
            )
            log_event(
                state, "Millennium Conjectures", "Quantitative Supremacy",
                "mode crise activé -> friction de transport optimal (LP) majorée (multiplicateur illustratif disclosé)",
            )
            withdraw_all_liquidity(state)
    elif was_active and state.millennium_crisis_source.startswith("Navier-Stokes"):
        state.millennium_crisis_mode = False
        state.millennium_crisis_source = ""
        state.liquidity_withdrawn = False
        log_event(
            state, "Millennium Conjectures", "AGI Active Inference",
            f"vorticité de pointe {peak_vorticity:.2f} rad/s repassée sous le seuil -> mode crise levé, injections AGI reprises",
        )
        log_event(
            state, "Millennium Conjectures", "Electrodynamic Capital",
            "mode crise levé -> replacement d'ordres de market-making autorisé au prochain tick (β1-gated)",
        )
    return state.millennium_crisis_mode


# =============================================================================
# Autonomous heartbeat -- lets the Nexus Command page keep the whole chain moving even when no
# other page is open (Streamlit only executes the active page's fragments -- a real, disclosed
# constraint of the "isolation parfaite des pages" design this router is built on, not hidden here).
# =============================================================================


def autonomous_heartbeat(state: GlobalMarketState) -> None:
    """One full, real (if simplified) pass through the whole domino chain, using the same shared
    topology/entropy/growth primitives every page uses -- so the Nexus Command page can visualize
    the chain live without duplicating each page's full UI-bound simulation."""
    state.tick_count += 1
    base_edges, _ = build_hypergraph_topology()
    tear_order = hypergraph_tear_order(len(base_edges))
    max_tear = int(len(base_edges) * 0.45)

    advance_hypergraph_phase(state, max_tear)
    edges_now = current_hypergraph_edges(base_edges, state.hypergraph_removed_edges, tear_order)
    graph = nx.Graph()
    graph.add_nodes_from(range(HYPERGRAPH_N_NODES))
    graph.add_edges_from(edges_now)
    components = list(nx.connected_components(graph))
    giant = max(components, key=len) if components else set()
    integrity_pct = 100.0 * len(giant) / HYPERGRAPH_N_NODES
    betti_1 = max(0, len(edges_now) - HYPERGRAPH_N_NODES + len(components))
    report_topology(state, integrity_pct, len(components), betti_1)

    n_pairs = 8
    probs = chaos_biased_distribution(n_pairs, state.shannon_chaos_bias, seed=int(time.time()))
    h_real = shannon_entropy(probs)
    h_max = float(np.log2(n_pairs))
    report_entropy(state, h_real, h_max)

    relief = relief_multiplier(state)
    nominal_r_total = 0.12  # the real network's nominal total resistance, disclosed constant (see electrodynamic_capital.py)
    report_ohm_result(state, nominal_r_total * relief)

    # Functional Core -- real Hurst regime estimate on a short synthetic price path (same estimator
    # absolute_quant_frontier.py uses), driven by the shared chaos bias so this gate keeps moving
    # even if that page is never opened this session.
    price_rng = np.random.default_rng(int(time.time() * 11) % 100000)
    synthetic_price_path = np.cumsum(price_rng.normal(0, 0.5 + state.shannon_chaos_bias, 90))
    report_hurst_regime(state, estimate_hurst_exponent(synthetic_price_path))

    # "State of the Art 2026" -- Koopman/DMD linearization, reusing the same synthetic price path just
    # generated for the Hurst estimate above rather than a second, redundant series.
    report_koopman(state, synthetic_price_path)

    # Real closed-form GBM time-average growth rate (Peters 2019, see non_ergodic_physics.py),
    # applied over one short heartbeat tick instead of that page's own per-slider Monte Carlo.
    # Gated by the Hurst regime just reported above (see accrue_survival_capital's own docstring).
    mu_n, sigma_n, dt = 0.11, 0.14, 1.0 / 720.0
    growth_rate = mu_n - 0.5 * sigma_n**2
    accrue_survival_capital(state, max(0.0, growth_rate) * dt * 100.0)

    # Functional Core -- real Betti-1-gated market-making (Rule 4): only the hypergraph's real cycle
    # basis (computed on the same graph just built above) is eligible for a simulated order, sized by
    # a real Wasserstein liquidity-gap severity scaled by the Kelly fraction report_entropy just set.
    holes = identify_liquidity_holes(graph)
    bid_liquidity, ask_liquidity = simulated_liquidity_profiles(state)
    place_market_making_orders(state, holes, bid_liquidity, ask_liquidity)

    # "State of the Art 2026" -- Assembly-Theory-inspired complexity filter on a disclosed synthetic
    # order-flow tag stream driven by the same real shared chaos bias as every other synthetic array
    # in this module.
    order_flow_symbols = synthetic_order_flow_symbols(state.shannon_chaos_bias)
    report_assembly_filter(state, order_flow_symbols)

    # Millennium Conjectures -- same real Lamb-Oseen closed form that page's own module uses, driven
    # by the same shared chaos bias so a crisis raised on one page is visible/reproducible on the
    # Nexus page even if millennium_conjectures.py itself is never opened this session. Crossing the
    # threshold also withdraws the simulated market-making orders just placed above (Rule 1).
    gamma = 4.0 + 40.0 * state.shannon_chaos_bias
    report_navier_stokes_vorticity(state, lamb_oseen_peak_vorticity(gamma, nu=0.6, t=1.0))


# =============================================================================
# "State of the Art 2026" -- Koopman/DMD linearization + Assembly-Theory-inspired complexity filter.
# The KAN transparency layer (:func:`_kan_edge_function`, :func:`kan_decision_layer`) lives up near
# :func:`report_entropy`, the real decision it annotates. See the module docstring's own section on
# this pass for what is real math vs. what corrected the brief's literal ask.
# =============================================================================


def koopman_lift(
    series: np.ndarray, delay: int = KOOPMAN_DELAY, rank: int = KOOPMAN_RANK
) -> tuple[np.ndarray, float, float]:
    """Exact Dynamic Mode Decomposition (Schmid, 2010, *JFM*; Rowley et al., 2009, *JFM*) -- the
    standard, published, numerical way to approximate the (in general infinite-dimensional) Koopman
    operator from data. ``series`` is time-delay embedded into a Hankel matrix (a real,
    finite-dimensional stand-in for lifting the state into a richer space of observables -- Takens'
    embedding theorem, 1981, is the classical justification that a delay embedding can recover the
    dynamics of the original nonlinear system), then the least-squares-optimal *linear* operator
    mapping one embedded snapshot to the next is recovered via a rank-truncated SVD. This really is
    linear algebra on ``series`` via numpy (SVD + eigendecomposition), not a metaphor -- and it is
    exactly what DMD is defined to compute: the best (least-squares) linear fit within the chosen
    embedding/rank, nothing more. It does not solve or "linearize the market"; a heavily truncated,
    short-window fit on a short synthetic series is a coarse, illustrative estimate, disclosed as such.

    Returns ``(eigenvalues, spectral_radius, forecast_next)``: the fitted operator's eigenvalues (their
    magnitude says whether the local linear fit is contracting/neutral/expanding), and a real one-step
    forecast of the next scalar value, obtained by advancing the last embedded state through the fitted
    operator and reading off its most recent coordinate."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n <= delay + 1:
        raise ValueError(f"series too short ({n}) for delay embedding of dimension {delay}")
    hankel = np.array([series[i:i + delay] for i in range(n - delay + 1)])
    x_snapshots = hankel[:-1].T
    x_snapshots_next = hankel[1:].T
    u, s, vh = np.linalg.svd(x_snapshots, full_matrices=False)
    r = max(1, min(rank, u.shape[1]))
    u_r, s_r, v_r = u[:, :r], s[:r], vh[:r, :].T
    a_tilde = u_r.T @ x_snapshots_next @ v_r @ np.diag(1.0 / s_r)
    eigvals = np.linalg.eigvals(a_tilde)
    spectral_radius = float(np.max(np.abs(eigvals)))
    last_embedded = x_snapshots[:, -1]
    reduced_coord = u_r.T @ last_embedded
    forecast_embedded = u_r @ (a_tilde @ reduced_coord)
    forecast_next = float(forecast_embedded[-1])
    return eigvals, spectral_radius, forecast_next


def report_koopman(state: GlobalMarketState, series: np.ndarray) -> bool:
    """Called once per heartbeat tick on the same short synthetic price path already generated for the
    real Hurst estimate (:func:`estimate_hurst_exponent`) -- reusing that series rather than inventing
    a second one. Runs the real DMD above only when the already-real shared chaos bias reads as
    genuinely turbulent (:data:`KOOPMAN_CHAOS_TRIGGER`): there is no point fitting a local linear model
    and reporting it everywhere a calm, near-linear regime is already well described by the existing
    Hurst/GBM machinery elsewhere in this file. Returns whether the fit ran this tick."""
    if state.shannon_chaos_bias < KOOPMAN_CHAOS_TRIGGER:
        state.koopman_active = False
        return False
    try:
        eigvals, spectral_radius, forecast_next = koopman_lift(series)
    except ValueError:
        state.koopman_active = False
        return False
    state.koopman_active = True
    state.koopman_spectral_radius = spectral_radius
    state.koopman_forecast_next = forecast_next
    log_event(
        state, "Casino Information Theory", "Casino Information Theory",
        f"biais de chaos {state.shannon_chaos_bias:.2f} >= {KOOPMAN_CHAOS_TRIGGER} -- régime non-linéaire "
        f"détecté -> plongement de retard (d={KOOPMAN_DELAY}) + DMD (SVD tronquée, rang={KOOPMAN_RANK}) : "
        f"rayon spectral={spectral_radius:.3f}, prévision linéaire (moindres carrés) du prochain point="
        f"{forecast_next:+.4f} -- un ajustement local réel, pas une résolution du marché",
    )
    return True


def synthetic_order_flow_symbols(chaos_bias: float, n: int = ASSEMBLY_SEQUENCE_LENGTH, seed: int | None = None) -> list[str]:
    """A disclosed synthetic discretized order-flow tag stream -- ``n`` symbols drawn from a small
    fixed alphabet, biased by the same real shared chaos signal every other synthetic array in this
    module already uses, via a geometric weighting (the same "concentration set by chaos_bias"
    mechanism as :func:`chaos_biased_distribution`): a calmer market (``chaos_bias`` -> 0) draws a
    sharply skewed distribution -- long, repeated runs of the same one or two tags, a simple bot's
    signature -- while a more chaotic one (``chaos_bias`` -> 1) draws close to uniform over the full
    alphabet -- few repeats, many distinct tags. Illustrative simulated data, not a real venue's real
    order tape, same discipline as :func:`simulated_liquidity_profiles`."""
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 17) % 100000)
    alphabet = [f"T{i}" for i in range(ASSEMBLY_ALPHABET_SIZE)]
    base = 0.15 + 0.80 * float(np.clip(chaos_bias, 0.0, 1.0))
    weights = base ** np.arange(ASSEMBLY_ALPHABET_SIZE)
    probs = weights / weights.sum()
    return [alphabet[i] for i in rng.choice(ASSEMBLY_ALPHABET_SIZE, size=n, p=probs)]


def calculate_assembly_index(sequence: list) -> int:
    """A real, disclosed proxy for object complexity in the spirit of Assembly Theory (Sharma, Cronin,
    Walker et al., *Nature*, 2023) -- that paper defines the assembly index over the shortest sequence
    of joining operations that builds a *physical/molecular* object from its parts, verified by mass
    spectrometry; it is not defined for, and has not been published as applying to, financial order
    flow. Applied here honestly as an analogous, real, well-established string-complexity measure
    instead: an LZ78 incremental parse (Lempel & Ziv, 1978) of the sequence -- the number of steps
    needed to build it where each step either copies the longest previously-built phrase and extends
    it by one new symbol, or starts a fresh one-symbol phrase if even that fails. This is the closest
    existing rigorous analogue, for a plain symbol sequence, of "minimum number of joining operations"
    (low count = few, highly repeated motifs, e.g. a simple bot's repeating pattern; high count = many
    novel, non-repeated motifs)."""
    symbols = tuple(str(s) for s in sequence)
    dictionary: set[tuple[str, ...]] = set()
    i = 0
    n = len(symbols)
    steps = 0
    while i < n:
        phrase: tuple[str, ...] = ()
        j = i
        while j < n and phrase + (symbols[j],) in dictionary:
            phrase = phrase + (symbols[j],)
            j += 1
        if j < n:
            phrase = phrase + (symbols[j],)
            j += 1
        dictionary.add(phrase)
        steps += 1
        i = j
    return steps


def report_assembly_filter(
    state: GlobalMarketState, symbols: list[str], window: int = ASSEMBLY_WINDOW, threshold: float = ASSEMBLY_COMPLEXITY_THRESHOLD
) -> tuple[float, float]:
    """Slides a disclosed fixed-size window over the synthetic order-flow tag stream above and computes
    each window's real, normalized LZ78-step-count ratio (:func:`calculate_assembly_index` divided by
    the window length -- low at the most repetitive extreme, up to 1.0 at the least-repetitive
    extreme). Windows at or above the disclosed :data:`ASSEMBLY_COMPLEXITY_THRESHOLD` are counted as
    "targeted" (higher-complexity, harder-to-compress motifs); windows below are "rejected" as
    low-complexity noise. Both percentages are computed fresh from this real count every call, never
    hardcoded. This is a disclosed complexity heuristic on synthetic data -- it does not identify, and
    makes no claim of identifying, any real market participant or institution."""
    n = len(symbols)
    if n < window:
        state.assembly_mean_index_ratio = 0.0
        state.assembly_accept_pct = 0.0
        state.assembly_reject_pct = 0.0
        return 0.0, 0.0
    ratios = np.array([
        calculate_assembly_index(symbols[i:i + window]) / window
        for i in range(0, n - window + 1)
    ])
    mean_ratio = float(np.mean(ratios))
    accept_pct = float(100.0 * np.mean(ratios >= threshold))
    reject_pct = 100.0 - accept_pct
    state.assembly_mean_index_ratio = mean_ratio
    state.assembly_accept_pct = accept_pct
    state.assembly_reject_pct = reject_pct
    log_event(
        state, "Electrodynamic Capital", "Electrodynamic Capital",
        f"filtre de complexité (proxy d'indice d'assemblage, LZ78 normalisé, fenêtre={window}) : "
        f"{reject_pct:.0f}% des fenêtres rejetées (répétitives/faible complexité), {accept_pct:.0f}% "
        f"retenues (complexité >= {threshold:.2f}) -- heuristique sur flux synthétique, aucune "
        "identification de contrepartie réelle",
    )
    return accept_pct, reject_pct

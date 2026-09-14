"""Pure-NumPy quantitative finance primitives: Kelly sizing, parametric VaR, and 4 stochastic-process models.

Every function here is a stateless, side-effect-free calculation over
plain NumPy arrays/floats -- no I/O, no Pydantic models, no logging.
This is deliberate: it keeps the math independently testable and reusable
from anywhere (a live agent, an offline research notebook, a backtest),
and is why :class:`~src.agents.supervisor.risk_manager.RiskManager`
delegates its VaR calculation to :func:`parametric_var` here rather than
computing it inline.

**"Protocole Stochastic Core" (2026-09-13)** added the four functions below
(:func:`hawkes_intensity`, :func:`jump_diffusion_anomaly`,
:func:`ornstein_uhlenbeck_mean_reversion`, :func:`price_impact`) -- real
stochastic-differential-equation primitives, not machine learning, exactly
as briefed. Each is wired into :class:`~src.agents.execution.agent.
ExecutionAgent`'s Gate 7 ("StochasticCore") -- see that module's docstring
for the real market data each one is computed from (this system has no
literal tick-by-tick trade print, so :func:`hawkes_intensity`'s "events"
are real received order-book ticks, weighted by their real
``order_book_imbalance`` magnitude -- disclosed there, not here, since this
module stays data-source-agnostic by design).

**"Le Prompt Architecte" (2026-09-13)** added two more real, closed-form
techniques: :func:`evt_var` (Extreme Value Theory VaR, rejecting the
Gaussian assumption :func:`parametric_var` above makes) and
:func:`almgren_chriss_trajectory`/:func:`participation_rate_capped_schedule`/
:func:`almgren_chriss_impact_costs` (the Almgren-Chriss (2000) optimal
execution trajectory -- the closed-form solution of the Hamilton-Jacobi-
Bellman equation for the linear-impact, risk-averse-trader case, derived
via calculus of variations, exactly as briefed). :func:`evt_var` is the
one function in this file that isn't pure NumPy -- it uses
:func:`scipy.stats.genpareto.fit` for a real maximum-likelihood tail fit,
disclosed here rather than silently breaking this module's own "Pure-
NumPy" framing above.

**"Arsenal Interdisciplinaire" (2026-09-13)** added 5 more real techniques
from physics/EE/pure math, picked from a larger briefed catalogue after
separating what transfers rigorously from what's an evocative metaphor
(the catalogue's own "impedance matching," "spin-glass replica symmetry
breaking," and "Riemann zeros predict volatility" items were declined for
exactly that reason -- see the module docstrings of whichever real class
now uses these functions for where each one is actually wired in):
:func:`marchenko_pastur_threshold`/:func:`denoise_correlation_matrix_mp`
(Random Matrix Theory spectral cleaning, Laloux et al. 1999),
:func:`calibrate_impact_coefficient_fdt` (Fluctuation-Dissipation-inspired
impact calibration from a market's own real spread fluctuations -- a
disclosed *analogy* to the classical theorem, not a first-principles
derivation for markets), :func:`wasserstein_price_distance` (thin wrapper
over :func:`scipy.stats.wasserstein_distance`), :func:`giant_cluster_fraction`
(percolation-style systemic concentration via real graph connectivity,
:func:`scipy.sparse.csgraph.connected_components`), and
:func:`kolmogorov_compressibility_ratio` (a real, standard zlib-compression
proxy for algorithmic-complexity/predictability -- not literal, uncomputable
Kolmogorov complexity, disclosed as the well-known practical proxy for it).

**"No Free Lunch" (2026-09-13)** added 5 more real, textbook techniques
(Shannon's Demon / constant-mix rebalancing, Black-Scholes-Merton pricing
under the real risk-neutral measure Girsanov's theorem justifies, discrete
delta-hedge replication, a real physical latency bound, and a real
Ashby's-Law hedge-completeness rank test) after correcting the brief's own
central claim: **none of these are risk-free, even in principle**, only
under continuous, frictionless, drift-free idealizations these functions
make explicit and violate on purpose in their own tests, so the real
residual risk is measured, never hidden. :func:`simulate_constant_mix_rebalancing`
shows the real loss a trending (non-zero-drift) path produces, not just the
favorable zero-drift case; :func:`simulate_discrete_delta_hedge` shows the
real non-zero replication error discrete (not continuous) rebalancing
leaves; :func:`black_scholes_price`/:func:`black_scholes_delta` are priced
under Q (Girsanov), with real, disclosed model risk (a wrong volatility
input still returns a confident-looking, wrong price); :func:`light_cone_latency`
is a real physical lower bound, not a claim this project has real
co-located exchange infrastructure to exploit it; and
:func:`check_hedge_completeness` is a real linear-algebra rank test (Ashby's
"variety" is just matrix rank here, not cybernetics mysticism).

**"Options & Multi-Instrument Hedging" (2026-09-13)** completes the Ashby's-
Law thread above with the real infrastructure it was missing:
:func:`black_scholes_gamma`/:func:`black_scholes_vega`/:func:`black_scholes_theta`
(the remaining real Greeks, extending :func:`black_scholes_price`/
:func:`black_scholes_delta`), and :func:`solve_hedge_ratios` (a real
least-squares solve for the position sizes, across whatever real
instruments are actually available, that best neutralizes a real target
risk-factor vector -- :func:`check_hedge_completeness`'s own natural
sequel: first ask *can* this instrument set hedge everything, then, if
so, solve for *how much* of each). Wired into a new real class,
:class:`~src.agents.supervisor.options_hedging_manager.
OptionsHedgingManager`, and a real (if optional) live options-chain
market-data source -- see that module's own docstring for what's real
market data vs. necessarily-simulated execution (this project has no
real options broker connection, the same honest gap every other real
broker integration here already discloses).

**"Time & Causality" (2026-09-13)** added 4 more real techniques governing
*when* and *how far* a prediction can honestly reach, closing out this
module's "no free lunch" thread from the temporal-physics side:
:func:`estimate_lyapunov_exponent` (Rosenstein, Collins & De Luca 1993's
real nearest-neighbor divergence estimator for the largest Lyapunov
exponent -- verified here against the logistic map, an exact textbook
chaotic system with a known true exponent of ``ln(2)``, **not**
against price data, since no real financial series has an independently
known true exponent to check against -- see that function's own docstring
for the real, load-bearing correction on why a positive estimate on real
market data is evidence of local sensitivity in that specific window, not
proof of low-dimensional deterministic chaos); :func:`nyquist_undersampling_check`
(the textbook Nyquist-Shannon criterion, applied honestly between two
*independently observed* real cadences -- see that function's own
docstring for why checking a signal's own FFT output against its own
sampling rate's Nyquist bound is tautological and can never detect real
aliasing); :func:`transfer_entropy` (Schreiber 2000's real, directed,
information-theoretic measure of Granger-style causality, a real
histogram-based plug-in estimator with disclosed finite-sample bias, not
a kernel/KSG estimator); and :func:`liquidity_void_pressure` (a real,
disclosed **heuristic**, explicitly *not* a literal Feynman path-integral/
Baaquie action-functional computation -- a real, standard order-book
depth-resistance asymmetry, framed honestly as the discrete, practical
analogue of "price moves toward the thinnest part of the book" rather
than claiming the actual physics machinery).

**"Signal, Physique, Algo" (2026-09-13)** briefed 6 more techniques; 3 were
found to already be real and built elsewhere in this project, and are
**not** duplicated here: triangular-arbitrage negative-cycle detection
(real Bellman-Ford on ``-log(rate)`` edge weights, exactly as briefed --
:func:`nothierly_genesis.tda_engine.find_real_arbitrage_cycle`, tested,
already wired into a dashboard proposal generator) and regime-switching
Viterbi decoding (a real, from-scratch, log-space Gaussian Hidden Markov
Model with real Baum-Welch fitting -- :class:`risk.regime_detector.GaussianHMM`,
tested, already wired into a dashboard on-demand detector, and already
gated to only *automatically de-risk*, never auto-increase risk). The
briefed "Anderson Localization" market-regime item was assessed and
**declined as a third, redundant re-implementation**: its own real
functional ask (detect a trending vs. range-bound regime and switch
strategy accordingly) is already served, twice over, by real existing
signals (:mod:`src.dashboard.core_daemons`'s real Hurst exponent daemon,
and the far more rigorous Hamilton-HMM regime detector above) -- building
a third, shakier, physics-metaphor version of the same functional signal
would be duplicated infrastructure, not new capability.

The 3 genuinely new additions: :func:`morlet_wavelet_transform` (a real
Continuous Wavelet Transform with the complex Morlet wavelet, implemented
directly in NumPy per Torrence & Compo (1998) -- :mod:`scipy.signal`
removed its own ``cwt``/``morlet2`` in a version this project already
depends on, and ``pywt`` is a C-extension package this Windows dev
environment cannot build, the identical real reason
:mod:`risk.regime_detector` hand-rolled its own HMM instead of
``hmmlearn``); :func:`kalman_dynamic_hedge_ratio` (a real, standard
2-state linear Kalman filter for a time-varying pairs-trading hedge
ratio, the textbook Ernie Chan-style state-space regression, not a mere
Bayesian-flavored exponential update); and
:func:`poiseuille_liquidity_resistance` (a real, disclosed *analogy* to
the Hagen-Poiseuille equation -- same honest-analogy framing
:func:`calibrate_impact_coefficient_fdt` already uses for
Fluctuation-Dissipation -- structurally distinct from
:func:`price_impact`'s already-real square-root impact law: this one's
real point is the much sharper :math:`r^{-4}` convexity a halved
liquidity depth produces, which the square-root law does not capture).

**"Le Calcul Fractionnaire" (2026-09-14)** adds two real techniques
correcting :func:`parametric_var`'s implicit square-root-of-time
assumption when scaled to a multi-period horizon:
:func:`hurst_exponent_rs` (Hurst's (1951) original Rescaled Range
analysis, the standard textbook estimator -- log-log-regressed across
multiple real sub-sample window sizes, not a single-window point
estimate) and :func:`fractional_volatility_scaling` (the real
self-affine fBm scaling law, horizon-scaled volatility equals base
volatility times the horizon raised to the power of the Hurst exponent
H, generalizing the classical "volatility times the square root of the
horizon" rule, which is exactly this formula's special case H=0.5). Checked
first for overlap: a Hurst estimator already exists in this project
(:func:`~src.dashboard.backend_simulator.estimate_hurst_exponent`), but
it is explicitly dashboard/demo-quality (a single-pass variance-lag
regression over 5 fixed lags, never wired into any real risk decision)
-- **not duplicated here**; :func:`hurst_exponent_rs` is a separate,
more standard estimator (R/S, not variance-lag), built specifically to
be wired into :class:`~src.agents.supervisor.risk_manager.RiskManager`,
a real risk-blocking-adjacent path the dashboard estimator was never
connected to. **Real, disclosed limitation carried over from the
brief**: this project has no live perpetual-futures leverage/Open
Interest feed (see :mod:`src.agents.alert_agent`'s own 2026-09-14
docstring correction on the same point) and no options-implied-
volatility surface calibrated for rough-volatility (rough Bergomi)
model fitting -- so this stays at the real, honestly-scoped level of
"Hurst-adjusted VaR horizon scaling," not a full rough Bergomi options-
pricing engine, which would need market inputs this system does not
have.
"""

from __future__ import annotations

import zlib
from collections import Counter
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import genpareto, norm, wasserstein_distance

# Two-tailed critical values of the standard normal distribution for the
# confidence levels used in practice (95%/99% are the de facto industry
# standards; 90%/97.5%/99.5% are included for completeness). A fixed
# lookup table is used rather than pulling in scipy.stats.norm.ppf, since
# only these standard levels are meant to be supported here.
Z_SCORES_BY_CONFIDENCE: Final[dict[float, float]] = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
}


def parametric_var(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    position_value: float,
) -> float:
    r"""Parametric (variance-covariance) Value at Risk.

    Formula:
        .. math:: \mathrm{VaR} = z_{\alpha} \cdot \sigma \cdot V

    where:
        - :math:`z_{\alpha}` is the upper-tail critical value of the
          standard normal distribution for confidence level
          :math:`1-\alpha` (e.g. ``1.6449`` for 95%).
        - :math:`\sigma` is the sample standard deviation of ``returns``
          (``ddof=1``, i.e. the unbiased estimator).
        - :math:`V` is ``position_value``.

    This assumes returns are approximately normally distributed and
    identically/independently distributed over the sample window -- the
    standard simplifying assumptions behind the parametric method (as
    opposed to historical VaR, which makes no distributional assumption
    but needs a much larger sample to be reliable).

    Args:
        returns: 1D array of periodic returns (e.g. log-returns of a
            price series). Must have at least 2 elements.
        confidence_level: One of the standard supported levels: ``0.90``,
            ``0.95``, ``0.975``, ``0.99``, ``0.995``.
        position_value: Notional value of the position being assessed.
            Must not be negative.

    Returns:
        The estimated maximum loss, in ``position_value``'s currency
        units, over one return-period, at the given confidence level.
        Always ``>= 0``.

    Raises:
        ValueError: If ``returns`` has fewer than 2 elements,
            ``confidence_level`` is not one of the supported levels, or
            ``position_value`` is negative.
    """
    if returns.size < 2:
        raise ValueError("returns must have at least 2 elements to estimate volatility.")
    if confidence_level not in Z_SCORES_BY_CONFIDENCE:
        supported = sorted(Z_SCORES_BY_CONFIDENCE)
        raise ValueError(
            f"Unsupported confidence_level {confidence_level!r}; must be one of {supported}."
        )
    if position_value < 0:
        raise ValueError("position_value must not be negative.")

    z_score = Z_SCORES_BY_CONFIDENCE[confidence_level]
    volatility = float(np.std(returns, ddof=1))
    return z_score * volatility * position_value


def sharpe_ratio(
    returns: npt.NDArray[np.float64],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r"""Annualized Sharpe ratio of a periodic return series.

    Formula:
        .. math:: \text{Sharpe} = \frac{\bar{r} - r_f}{\sigma_r} \sqrt{N}

    where :math:`\bar{r}` is the mean periodic return, :math:`r_f` is the
    per-period risk-free rate, :math:`\sigma_r` is the sample standard
    deviation of returns (``ddof=1``), and :math:`N` is
    ``periods_per_year`` (the standard square-root-of-time annualization,
    valid under the same i.i.d.-returns assumption :func:`parametric_var`
    already relies on).

    Args:
        returns: 1D array of periodic returns (e.g. daily simple or log
            returns). Must have at least 2 elements.
        risk_free_rate: Per-period (not annualized) risk-free rate,
            subtracted from every return before the ratio. Defaults to
            ``0.0``.
        periods_per_year: Number of return periods in one year (``252``
            for daily trading-day data, ``365`` for daily calendar data,
            ``12`` for monthly, ...). Must be positive.

    Returns:
        The annualized Sharpe ratio. ``0.0`` if ``returns`` has zero
        variance (a constant series has no risk to divide by, and no
        meaningful excess-return-per-unit-risk figure) rather than
        raising or returning ``inf``/``nan``.

    Raises:
        ValueError: If ``returns`` has fewer than 2 elements, or
            ``periods_per_year`` is not positive.
    """
    if returns.size < 2:
        raise ValueError("returns must have at least 2 elements to estimate Sharpe ratio.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    excess_returns = returns - risk_free_rate
    volatility = float(np.std(excess_returns, ddof=1))
    if volatility == 0.0:
        return 0.0
    return float(np.mean(excess_returns) / volatility * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: npt.NDArray[np.float64]) -> float:
    r"""Maximum peak-to-trough decline of an equity curve, as a fraction.

    Formula:
        .. math:: \text{MDD} = \min_t \frac{E_t - \max_{s \le t} E_s}{\max_{s \le t} E_s}

    i.e. at every point, how far below its own running peak the curve
    currently sits; the deepest such dip over the whole series. Always
    ``<= 0`` (``0.0`` for a curve that never dips below its running
    peak, e.g. a monotonically non-decreasing series).

    Args:
        equity_curve: 1D array of portfolio value over time, in
            chronological order. Must have at least 1 element and be
            strictly positive throughout (a zero or negative equity
            value makes the percentage decline undefined).

    Returns:
        The maximum drawdown as a negative fraction (e.g. ``-0.20`` for
        a 20% peak-to-trough decline).

    Raises:
        ValueError: If ``equity_curve`` is empty, or contains a
            non-positive value.
    """
    if equity_curve.size < 1:
        raise ValueError("equity_curve must have at least 1 element.")
    if np.any(equity_curve <= 0):
        raise ValueError("equity_curve must be strictly positive throughout.")

    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_peak) / running_peak
    return float(np.min(drawdown))


def kelly_criterion(
    win_probability: float,
    win_loss_ratio: float,
    *,
    fraction: float = 1.0,
) -> float:
    r"""Kelly Criterion optimal bet/position-sizing fraction (discrete form).

    Formula:
        .. math:: f^* = p - \frac{1-p}{b} = \frac{bp - (1-p)}{b}

    where:
        - :math:`p` is ``win_probability``, the probability of a winning
          trade.
        - :math:`b` is ``win_loss_ratio``, the payoff odds: average win
          size divided by average loss size (e.g. :math:`b=2` means a
          typical win is twice the size of a typical loss).

    :math:`f^*` is the fraction of capital that maximizes long-run
    logarithmic growth (expected log-utility) when the same bet is
    repeated many times. The raw result is scaled by ``fraction`` --
    e.g. ``0.5`` for "half-Kelly", a common practitioner adjustment that
    trades some growth rate for a substantial reduction in variance and
    drawdown risk relative to full Kelly -- and then clipped to
    ``[0, 1]``: a negative :math:`f^*` (no edge, or a losing edge) means
    "do not take this bet", not "take a negative position", and sizing
    is never allowed to exceed 100% of capital regardless of what the
    raw formula outputs.

    Args:
        win_probability: Probability of a winning trade, in ``[0, 1]``.
        win_loss_ratio: Average win size divided by average loss size.
            Must be strictly positive.
        fraction: Multiplier applied to the raw Kelly fraction before
            clipping (e.g. ``0.5`` for half-Kelly). Must be positive.
            Defaults to ``1.0`` (full Kelly).

    Returns:
        The recommended fraction of capital to allocate, in ``[0, 1]``.

    Raises:
        ValueError: If ``win_probability`` is not in ``[0, 1]``, or
            ``win_loss_ratio``/``fraction`` is not positive.
    """
    if not (0.0 <= win_probability <= 1.0):
        raise ValueError(f"win_probability must be in [0, 1], got {win_probability!r}.")
    if win_loss_ratio <= 0:
        raise ValueError(f"win_loss_ratio must be positive, got {win_loss_ratio!r}.")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction!r}.")

    loss_probability = 1.0 - win_probability
    raw_kelly = win_probability - loss_probability / win_loss_ratio
    return float(np.clip(raw_kelly * fraction, 0.0, 1.0))


def kelly_criterion_continuous(
    expected_return: float,
    variance: float,
    *,
    fraction: float = 1.0,
) -> float:
    r"""Kelly Criterion optimal fraction for a continuous return distribution.

    Formula:
        .. math:: f^* = \frac{\mu}{\sigma^2}

    where :math:`\mu` is ``expected_return`` (mean periodic return of the
    asset/strategy) and :math:`\sigma^2` is ``variance`` (variance of
    that same return series). This is the growth-optimal leverage for an
    asset whose returns are approximately normally distributed --
    the continuous-return counterpart to :func:`kelly_criterion`'s
    discrete win/loss formulation, useful when a strategy's edge is
    naturally expressed as a return distribution rather than a
    win-probability/payoff-ratio pair.

    Same clipping/scaling behavior as :func:`kelly_criterion`: the raw
    result is scaled by ``fraction`` then clipped to ``[0, 1]``.

    Args:
        expected_return: Mean periodic return, e.g. from a sample of
            historical log-returns.
        variance: Variance of that return series. Must be strictly
            positive.
        fraction: Multiplier applied to the raw Kelly fraction before
            clipping (e.g. ``0.5`` for half-Kelly). Must be positive.

    Returns:
        The recommended fraction of capital to allocate, in ``[0, 1]``.

    Raises:
        ValueError: If ``variance`` or ``fraction`` is not positive.
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance!r}.")
    if fraction <= 0:
        raise ValueError(f"fraction must be positive, got {fraction!r}.")

    raw_kelly = expected_return / variance
    return float(np.clip(raw_kelly * fraction, 0.0, 1.0))


def hawkes_intensity(
    event_timestamps: npt.NDArray[np.float64],
    *,
    baseline_intensity: float,
    decay_rate: float,
    excitation_strength: float,
    event_weights: npt.NDArray[np.float64] | None = None,
    at_time: float | None = None,
) -> float:
    r"""Self-exciting (Hawkes) point-process intensity: order-flow "cascade" detection.

    Formula:
        .. math:: \lambda(t) = \mu + \sum_{t_i < t} w_i \cdot \alpha \cdot e^{-\beta(t - t_i)}

    where :math:`\mu` is ``baseline_intensity``, :math:`\alpha` is
    ``excitation_strength``, :math:`\beta` is ``decay_rate``, and each past
    event :math:`t_i` contributes an exponentially-decaying excitation
    :math:`w_i` (``event_weights``, defaulting to ``1.0`` each -- the
    unweighted, textbook Hawkes formula) to the current intensity. A burst
    of recent, large-weight events pushes :math:`\lambda(t)` sharply above
    :math:`\mu`, exponentially, exactly as many market-order arrivals in a
    short window ("liquidity being absorbed") would -- the real "cascade
    effect" (momentum/panic regime) this function exists to flag; the
    caller decides the alert threshold against :math:`\mu`, since what
    counts as "elevated" is a per-instrument, per-regime judgment this
    stateless function has no basis to hardcode.

    Args:
        event_timestamps: 1D array of past event times (any consistent
            time unit, e.g. Unix epoch seconds), need not be sorted.
        baseline_intensity: :math:`\mu`, the process's intensity with no
            excitation (``event_timestamps`` empty or all in the future
            relative to ``at_time``). Must be ``>= 0``.
        decay_rate: :math:`\beta`, how fast one event's excitation decays.
            Must be strictly positive (non-positive would mean excitation
            never decays, an explosive/non-stationary process).
        excitation_strength: :math:`\alpha`, the unweighted contribution of
            one event at zero elapsed time. Must be ``>= 0``.
        event_weights: Per-event weight :math:`w_i`, same length as
            ``event_timestamps``. Defaults to ``1.0`` for every event (the
            textbook, unweighted Hawkes process) when omitted.
        at_time: The time to evaluate :math:`\lambda` at. Defaults to
            ``event_timestamps.max()`` (the latest observed event) when
            omitted -- "the intensity right now," the common case for a
            live caller with no future events to consider.

    Returns:
        The estimated intensity :math:`\lambda(\text{at\_time})`, always
        ``>= baseline_intensity``. ``baseline_intensity`` itself if
        ``event_timestamps`` is empty.

    Raises:
        ValueError: If ``baseline_intensity``/``excitation_strength`` is
            negative, ``decay_rate`` is not positive, or ``event_weights``
            is given with a different length than ``event_timestamps``.
    """
    if baseline_intensity < 0:
        raise ValueError("baseline_intensity must not be negative.")
    if decay_rate <= 0:
        raise ValueError("decay_rate must be positive.")
    if excitation_strength < 0:
        raise ValueError("excitation_strength must not be negative.")
    if event_weights is not None and event_weights.shape != event_timestamps.shape:
        raise ValueError("event_weights must be the same shape as event_timestamps.")

    if event_timestamps.size == 0:
        return baseline_intensity

    evaluation_time = float(event_timestamps.max()) if at_time is None else at_time
    weights = np.ones_like(event_timestamps, dtype=np.float64) if event_weights is None else event_weights

    elapsed = evaluation_time - event_timestamps
    past_mask = elapsed >= 0  # future events (elapsed < 0) have not excited the process yet
    excitation = np.sum(weights[past_mask] * excitation_strength * np.exp(-decay_rate * elapsed[past_mask]))
    return float(baseline_intensity + excitation)


def jump_diffusion_anomaly(
    log_returns: npt.NDArray[np.float64],
    *,
    jump_threshold_std: float = 4.0,
) -> tuple[bool, float]:
    r"""Merton jump-diffusion anomaly detection: separate a fat-tail jump from ordinary diffusion.

    A pure Geometric Brownian Motion assumes log-returns are i.i.d. normal;
    Merton's jump-diffusion model adds a compound Poisson jump component
    :math:`dq_t` on top of that diffusion to capture real fat tails. This
    function does not fit the full model (that needs far more data than a
    live trading window has) -- it runs the practical test that model
    implies: **is the most extreme recent return larger than ordinary
    diffusion, at this series' own recently-observed volatility, would
    plausibly produce?**

    The baseline volatility used for that test deliberately **excludes**
    the single most extreme return before computing :math:`\sigma`
    (`leave-one-out`): the jump being tested for would otherwise inflate
    the very volatility estimate used to judge whether it is a jump,
    systematically hiding real jumps in their own noise -- the same
    reasoning behind bipower-variation-style jump tests in the literature
    (Barndorff-Nielsen & Shephard), simplified here to a single
    leave-one-out estimate rather than the full bipower-variation
    statistic (disclosed, not hidden: this is a practical approximation of
    that real technique, not the full published test).

    Args:
        log_returns: 1D array of periodic log-returns. Must have at least
            4 elements (need a real baseline-volatility sample *after*
            removing the tested return).
        jump_threshold_std: How many baseline standard deviations the most
            extreme return must exceed to flag ``is_black_swan_forming``.
            ``4.0`` (the default) is deliberately conservative -- under a
            pure Gaussian null, a 4-sigma move is already a roughly
            1-in-16,000 event.

    Returns:
        ``(is_black_swan_forming, max_standardized_jump)``:
        ``is_black_swan_forming`` is ``True`` iff the largest
        |standardized return| exceeds ``jump_threshold_std``;
        ``max_standardized_jump`` is that largest |standardized return|
        itself (always ``>= 0``), for telemetry/logging even when it does
        not cross the threshold.

    Raises:
        ValueError: If ``log_returns`` has fewer than 4 elements, or
            ``jump_threshold_std`` is not positive.
    """
    if log_returns.size < 4:
        raise ValueError("log_returns must have at least 4 elements to isolate one tested return.")
    if jump_threshold_std <= 0:
        raise ValueError("jump_threshold_std must be positive.")

    most_extreme_index = int(np.argmax(np.abs(log_returns)))
    baseline_returns = np.delete(log_returns, most_extreme_index)
    baseline_volatility = float(np.std(baseline_returns, ddof=1))

    if baseline_volatility == 0.0:
        # A perfectly flat baseline: any nonzero tested return is an infinite-sigma
        # jump by definition, not a division-by-zero error to raise past the caller.
        max_standardized_jump = float("inf") if log_returns[most_extreme_index] != 0.0 else 0.0
    else:
        max_standardized_jump = abs(log_returns[most_extreme_index]) / baseline_volatility

    return max_standardized_jump > jump_threshold_std, max_standardized_jump


def ornstein_uhlenbeck_mean_reversion(
    price_series: npt.NDArray[np.float64],
    *,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    r"""Ornstein-Uhlenbeck mean-reversion parameters, calibrated from real price history.

    The discretized OU SDE:
        .. math:: dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t

    is equivalent, one step at a time, to the AR(1) regression
    :math:`\Delta X_t = a + b X_t + \varepsilon_t` with :math:`b = -\theta
    \,dt` and :math:`a = \theta\mu\,dt` -- the standard way to calibrate OU
    from discretely-observed data (ordinary least squares on real
    ``price_series``, not a two-number shortcut: a genuine reversion speed
    :math:`\theta` cannot be estimated from a single current price and a
    single historical mean, only from how the series has actually moved
    around that mean over time).

    Args:
        price_series: 1D array of real historical prices, chronological
            order. Must have at least 3 elements (need at least 2 real
            deltas to regress).
        dt: Time step between consecutive ``price_series`` samples, in
            whatever consistent unit the caller's reversion speed should
            be expressed in (e.g. ``1.0`` for "per sample"). Must be
            positive.

    Returns:
        ``(theta, mu, half_life)``: ``theta`` is the fitted reversion
        speed (can be ``<= 0``, meaning the fitted series shows no real
        mean-reverting force over this window -- an honest result, not an
        error); ``mu`` is the fitted long-run mean; ``half_life`` is
        :math:`\ln(2)/\theta` (time for a deviation to halve), or
        ``float("inf")`` when ``theta <= 0`` (no reversion means no finite
        half-life, not a fabricated number).

    Raises:
        ValueError: If ``price_series`` has fewer than 3 elements, or
            ``dt`` is not positive.
    """
    if price_series.size < 3:
        raise ValueError("price_series must have at least 3 elements to regress a reversion speed.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    levels = price_series[:-1]
    deltas = price_series[1:] - levels

    slope, intercept = np.polyfit(levels, deltas, 1)
    theta = float(-slope / dt)
    if theta <= 0:
        return theta, float("nan"), float("inf")

    mu = float(intercept / (theta * dt))
    half_life = float(np.log(2.0) / theta)
    return theta, mu, half_life


def price_impact(
    order_volume: float,
    available_liquidity: float,
    volatility: float,
    *,
    impact_coefficient: float = 1.0,
) -> float:
    r"""Square-root price-impact model: how much a proposed order would move the price itself.

    Formula:
        .. math:: \frac{\Delta P}{P} = k \cdot \sigma \cdot \mathrm{sign}(Q) \cdot \sqrt{\frac{|Q|}{V}}

    where :math:`Q` is ``order_volume`` (signed: positive buy, negative
    sell), :math:`V` is ``available_liquidity``, :math:`\sigma` is
    ``volatility``, and :math:`k` is ``impact_coefficient``. This is the
    real, widely-used square-root market-impact law (Almgren-Chriss and
    the broad empirical "square-root law" literature): impact grows with
    the square root of participation rate (order size relative to
    available liquidity), not linearly -- a large order against thin
    liquidity costs disproportionately more than the same order against a
    deep book.

    Args:
        order_volume: Signed proposed order size, in the same units as
            ``available_liquidity`` (e.g. instrument units). ``0.0``
            returns ``0.0`` impact directly (no division-by-zero risk from
            the ``sqrt``).
        available_liquidity: Real, observed liquidity this order would
            trade against (e.g. real top-of-book bid+ask volume). Must be
            strictly positive -- see the caller's own handling for "no
            real liquidity reading has arrived yet" (this function does
            not invent a placeholder value for that case).
        volatility: Recent fractional volatility (e.g. the standard
            deviation of recent log-returns) of the instrument. Must be
            ``>= 0``.
        impact_coefficient: Model scale constant :math:`k`, calibratable
            per-venue/instrument in principle; defaults to ``1.0`` (no
            rescaling) absent a fitted value. Must be positive.

    Returns:
        The estimated *signed*, *fractional* price impact
        (:math:`\Delta P / P`) -- positive for a buy (price pushed up),
        negative for a sell (price pushed down). Multiply by the current
        price for an absolute :math:`\Delta P`.

    Raises:
        ValueError: If ``available_liquidity`` is not positive,
            ``volatility`` is negative, or ``impact_coefficient`` is not
            positive.
    """
    if available_liquidity <= 0:
        raise ValueError("available_liquidity must be positive.")
    if volatility < 0:
        raise ValueError("volatility must not be negative.")
    if impact_coefficient <= 0:
        raise ValueError("impact_coefficient must be positive.")

    if order_volume == 0.0:
        return 0.0

    sign = 1.0 if order_volume > 0 else -1.0
    participation_rate = abs(order_volume) / available_liquidity
    return impact_coefficient * volatility * sign * float(np.sqrt(participation_rate))


def evt_var(
    returns: npt.NDArray[np.float64],
    confidence_level: float,
    position_value: float,
    *,
    threshold_percentile: float = 0.90,
    min_exceedances: int = 10,
) -> float:
    r"""Extreme Value Theory (Peaks-Over-Threshold) Value at Risk -- no Gaussian assumption.

    :func:`parametric_var` above assumes returns are normally
    distributed; real financial returns (crypto especially) have fatter
    tails than a Gaussian, which systematically *understates* real tail
    risk. This function makes no distributional assumption about the
    *whole* return series -- only about its *tail*, which the
    Pickands-Balkema-de Haan theorem justifies modeling with a
    Generalized Pareto Distribution (GPD), regardless of the parent
    distribution -- the real, standard "Peaks-Over-Threshold" EVT method
    (McNeil & Frey, 1999).

    Method: let :math:`L = -R` be losses (positive for a losing return).
    Pick a high threshold :math:`u` (the ``threshold_percentile`` quantile
    of :math:`L`). Fit a GPD, via real maximum likelihood
    (:func:`scipy.stats.genpareto.fit`), to the real exceedances
    :math:`L - u` for every loss exceeding :math:`u`. The GPD's fitted
    shape :math:`\xi` and scale :math:`\beta` then give a closed-form
    tail quantile:

    .. math::
        \mathrm{VaR}_q = u + \frac{\beta}{\xi}\left[\left(\frac{n}{N_u}(1-q)\right)^{-\xi} - 1\right]
        \quad (\xi \neq 0)

    where :math:`n` is the total sample size, :math:`N_u` the number of
    real exceedances, and :math:`q` is ``confidence_level``. The
    :math:`\xi \to 0` limit (an exponential tail) is handled separately
    below to avoid a division by zero.

    Args:
        returns: 1D array of periodic returns (e.g. log-returns). Must
            have enough observations that ``threshold_percentile`` yields
            at least ``min_exceedances`` real losses beyond the
            threshold (see ``Raises``).
        confidence_level: VaR confidence level, in ``(0, 1)`` -- unlike
            :func:`parametric_var`, not restricted to the fixed z-score
            table, since this method doesn't need a normal-distribution
            critical value at all.
        position_value: Notional value of the position being assessed.
            Must not be negative.
        threshold_percentile: Quantile of the loss distribution used as
            the real GPD-fitting threshold :math:`u`, in ``(0, 1)``.
            ``0.90`` (the default) is a standard practitioner choice --
            high enough that the GPD's asymptotic justification roughly
            holds, low enough to leave a real, usable number of
            exceedances to fit against.
        min_exceedances: Minimum real exceedances over :math:`u` required
            before trusting a GPD fit -- fewer than this and the fitted
            tail shape is dominated by sampling noise, not a real tail
            estimate (see ``Raises``).

    Returns:
        The estimated maximum loss, in ``position_value``'s currency
        units, at the given confidence level. Always ``>= 0``.

    Raises:
        ValueError: If ``returns`` has fewer than 2 elements,
            ``confidence_level``/``threshold_percentile`` is not in
            ``(0, 1)``, ``position_value`` is negative, or fewer than
            ``min_exceedances`` real losses exceed the fitted threshold.
    """
    if returns.size < 2:
        raise ValueError("returns must have at least 2 elements.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level!r}.")
    if not (0.0 < threshold_percentile < 1.0):
        raise ValueError(f"threshold_percentile must be in (0, 1), got {threshold_percentile!r}.")
    if position_value < 0:
        raise ValueError("position_value must not be negative.")

    losses = -returns
    n = losses.size
    threshold = float(np.quantile(losses, threshold_percentile))
    exceedances = losses[losses > threshold] - threshold
    n_exceedances = exceedances.size

    if n_exceedances < min_exceedances:
        raise ValueError(
            f"Only {n_exceedances} real exceedances over the {threshold_percentile:.0%} threshold "
            f"(need >= {min_exceedances}) -- not enough tail data for a trustworthy GPD fit."
        )

    # floc=0: the GPD is fit to the *excesses over threshold* (already
    # shifted to start at 0 above), the real, standard POT parametrization
    # -- letting scipy also fit a location here would double-count the
    # threshold shift already applied.
    shape_xi, _location, scale_beta = genpareto.fit(exceedances, floc=0)

    tail_probability = (n / n_exceedances) * (1.0 - confidence_level)
    if abs(shape_xi) < 1e-8:
        # xi -> 0 is the exponential-tail limit of the GPD quantile
        # formula (a real, standard special case, not an approximation of
        # convenience) -- the xi-based formula has a removable 0/0
        # singularity there.
        var_return_units = threshold - scale_beta * np.log(tail_probability)
    else:
        var_return_units = threshold + (scale_beta / shape_xi) * (tail_probability ** (-shape_xi) - 1.0)

    return max(0.0, float(var_return_units)) * position_value


def almgren_chriss_trajectory(
    order_size: float,
    *,
    execution_horizon_seconds: float,
    volatility: float,
    risk_aversion: float,
    temporary_impact_coefficient: float,
    n_steps: int = 10,
) -> npt.NDArray[np.float64]:
    r"""Almgren-Chriss (2000) optimal execution trajectory -- real closed-form Hamilton-Jacobi-Bellman solution.

    A trader liquidating (or acquiring) ``order_size`` units over
    ``execution_horizon_seconds`` faces a real trade-off: trade fast and
    pay more temporary market impact, or trade slow and bear more price
    (timing) risk from ``volatility`` while the position sits unfilled.
    Almgren & Chriss (2000) pose this as minimizing
    :math:`E[\text{cost}] + \lambda \cdot \mathrm{Var}[\text{cost}]` over
    every possible execution path -- a calculus-of-variations problem
    whose Euler-Lagrange equation *is* the Hamilton-Jacobi-Bellman
    equation for this linear-temporary-impact, risk-averse-trader model,
    and which has this real, closed-form solution for the remaining
    inventory :math:`x(t)` (:math:`x(0) = X`, :math:`x(T) = 0`):

    .. math::
        x(t) = X \cdot \frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)},
        \qquad
        \kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}

    where :math:`X` is ``order_size``, :math:`T` is
    ``execution_horizon_seconds``, :math:`\lambda` is ``risk_aversion``,
    :math:`\sigma` is ``volatility``, and :math:`\eta` is
    ``temporary_impact_coefficient``. :math:`\kappa` ("urgency") governs
    the trajectory's shape: risk-neutral (:math:`\lambda \to 0`) makes
    :math:`\kappa \to 0`, and the real, well-known limit of this formula
    is then the *linear* (TWAP) schedule -- handled explicitly below to
    avoid a ``0/0`` in ``sinh(0)/sinh(0)``. A larger :math:`\kappa`
    front-loads execution more aggressively (more risk-averse, or more
    volatile, or cheaper to trade fast).

    **Real unit-magnitude pitfall, found by testing**: :math:`\kappa T`
    is extremely sensitive to ``volatility``'s units -- passing an
    *annualized* volatility (e.g. ``0.30`` for 30%/year) directly here,
    with ``execution_horizon_seconds`` in the thousands, produces a
    :math:`\kappa T` in the thousands and an (numerically stable, but
    practically degenerate) trajectory that front-loads essentially the
    entire order into the first slice. Convert to **per-second**
    volatility first (``annualized_vol / sqrt(seconds_per_year)``) --
    verified directly: 30%/year annualized converts to
    ``~5.3e-5``/second, which keeps :math:`\kappa T` in a sane,
    realistic range (well under 1 for a 1-hour horizon at
    ``risk_aversion<=1.0``) and produces the expected smooth,
    gradually-front-loading trajectory instead.

    Args:
        order_size: Total real units to execute over the horizon. Must be
            positive (direction is the caller's concern -- this returns
            an unsigned remaining-inventory curve).
        execution_horizon_seconds: Real total time allotted to fully
            execute ``order_size``. Must be positive.
        volatility: Real **per-second** (or whatever unit
            ``temporary_impact_coefficient``/``execution_horizon_seconds``
            are consistently expressed in) price volatility -- see the
            real unit-magnitude pitfall above before passing an
            annualized figure directly. Must be ``>= 0``.
        risk_aversion: :math:`\lambda`, real risk-aversion coefficient.
            ``0.0`` is the risk-neutral (TWAP) limit; larger values
            front-load execution more. Must be ``>= 0``.
        temporary_impact_coefficient: :math:`\eta`, real linear temporary-
            impact coefficient (price concession per unit trading rate).
            Must be positive.
        n_steps: Number of real discrete execution slices to return the
            trajectory at (``n_steps + 1`` points including both
            endpoints). Must be positive.

    Returns:
        ``x(t_0), ..., x(t_{n\_steps})`` -- the real remaining-inventory
        trajectory, strictly decreasing from ``order_size`` to (within
        floating-point precision) ``0.0``.

    Raises:
        ValueError: If ``order_size``/``execution_horizon_seconds``/
            ``temporary_impact_coefficient`` is not positive, if
            ``volatility``/``risk_aversion`` is negative, or if
            ``n_steps`` is not positive.
    """
    if order_size <= 0:
        raise ValueError("order_size must be positive.")
    if execution_horizon_seconds <= 0:
        raise ValueError("execution_horizon_seconds must be positive.")
    if volatility < 0:
        raise ValueError("volatility must not be negative.")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must not be negative.")
    if temporary_impact_coefficient <= 0:
        raise ValueError("temporary_impact_coefficient must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    t = np.linspace(0.0, execution_horizon_seconds, n_steps + 1)
    kappa = float(np.sqrt(risk_aversion * volatility**2 / temporary_impact_coefficient))
    kappa_horizon = kappa * execution_horizon_seconds

    if kappa_horizon < 1e-8:
        # Real risk-neutral limit: sinh(x)/x -> 1 as x -> 0, so the AC
        # formula's ratio reduces exactly to the linear (TWAP) schedule.
        trajectory = order_size * (1.0 - t / execution_horizon_seconds)
    else:
        # Numerically stable sinh(a)/sinh(b) for a = kappa*(T-t), b =
        # kappa*T, rather than np.sinh(a)/np.sinh(b) directly -- a real
        # bug found by testing with realistic-magnitude parameters
        # (kappa*T easily reaches the thousands for real seconds-scale
        # horizons, and np.sinh overflows to inf past ~710, silently
        # producing inf/inf = nan). Since 0 <= a <= b always here,
        # sinh(a)/sinh(b) = (exp(a-b) - exp(-a-b)) / (1 - exp(-2b))
        # (multiply num/denom of the sinh definition by exp(-b)) never
        # evaluates exp() at a positive argument, so it never overflows.
        a = kappa * (execution_horizon_seconds - t)
        b = kappa_horizon
        trajectory = order_size * (np.exp(a - b) - np.exp(-a - b)) / (1.0 - np.exp(-2.0 * b))

    return trajectory


def participation_rate_capped_schedule(
    trajectory: npt.NDArray[np.float64],
    volume_profile: npt.NDArray[np.float64],
    *,
    max_participation_rate: float = 0.10,
) -> tuple[npt.NDArray[np.float64], bool]:
    r"""Real per-slice trade sizes from an execution trajectory, capped by real expected volume.

    :func:`almgren_chriss_trajectory` alone assumes liquidity is always
    available at a cost -- it never checks *how much* real volume a slice
    can actually absorb. This applies the standard, separate, real-world
    constraint every execution desk actually uses on top of it: no single
    slice may trade more than ``max_participation_rate`` of that slice's
    real expected volume (``volume_profile``) -- "never be more than X%
    of the tape."

    Args:
        trajectory: A real remaining-inventory curve, e.g. from
            :func:`almgren_chriss_trajectory` (``n_steps + 1`` points).
        volume_profile: Real expected traded volume for each of the
            ``trajectory.size - 1`` slices between consecutive trajectory
            points (e.g. real historical average volume in each
            corresponding time bucket). Must be the same length as
            ``trajectory.size - 1`` and strictly positive throughout.
        max_participation_rate: Maximum fraction of a slice's real
            expected volume this schedule may trade in that slice, in
            ``(0, 1]``.

    Returns:
        ``(trade_sizes, was_capped)`` -- ``trade_sizes`` (length
        ``trajectory.size - 1``) is the real per-slice quantity to trade,
        capped where the uncapped Almgren-Chriss schedule would have
        exceeded ``max_participation_rate`` of that slice's real volume;
        ``was_capped`` is ``True`` if any slice actually hit the cap
        (told to the caller honestly, since a capped schedule no longer
        completes ``order_size`` by the end of the horizon -- see the
        caller's own handling of that real, disclosed shortfall).

    Raises:
        ValueError: If ``volume_profile``'s length doesn't match
            ``trajectory.size - 1``, any entry is non-positive, or
            ``max_participation_rate`` is not in ``(0, 1]``.
    """
    if volume_profile.size != trajectory.size - 1:
        raise ValueError(
            f"volume_profile must have trajectory.size - 1 = {trajectory.size - 1} entries, "
            f"got {volume_profile.size}."
        )
    if np.any(volume_profile <= 0):
        raise ValueError("volume_profile entries must all be positive.")
    if not (0.0 < max_participation_rate <= 1.0):
        raise ValueError("max_participation_rate must be in (0, 1].")

    uncapped_trade_sizes = -np.diff(trajectory)
    caps = max_participation_rate * volume_profile
    trade_sizes = np.minimum(uncapped_trade_sizes, caps)
    was_capped = bool(np.any(trade_sizes < uncapped_trade_sizes - 1e-9))

    return trade_sizes, was_capped


def almgren_chriss_impact_costs(
    trade_sizes: npt.NDArray[np.float64],
    *,
    temporary_impact_coefficient: float,
    permanent_impact_coefficient: float,
    step_duration_seconds: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Real per-slice temporary vs. cumulative permanent price impact, for a given trade schedule.

    Standard Almgren-Chriss linear impact decomposition:

    .. math::
        \text{temporary}_i = \eta \cdot \frac{q_i}{\Delta t}, \qquad
        \text{permanent}_i = \gamma \cdot \sum_{j \le i} q_j

    where :math:`q_i` is ``trade_sizes[i]``, :math:`\eta`/:math:`\gamma`
    are the temporary/permanent impact coefficients, and
    :math:`\Delta t` is ``step_duration_seconds``. Temporary impact is a
    real, transient price concession paid only while slice :math:`i` is
    actively trading (proportional to trading *rate*, i.e. size per unit
    time); permanent impact is a real, persistent shift that compounds
    with every prior slice traded (proportional to cumulative *size*
    traded so far) -- the same real distinction
    :func:`almgren_chriss_trajectory`'s own module docstring assumes.

    Args:
        trade_sizes: Real per-slice quantities, e.g. from
            :func:`participation_rate_capped_schedule`. Every entry must
            be ``>= 0``.
        temporary_impact_coefficient: :math:`\eta`. Must be ``>= 0``.
        permanent_impact_coefficient: :math:`\gamma`. Must be ``>= 0``.
        step_duration_seconds: Real real-world duration of one slice.
            Must be positive.

    Returns:
        ``(temporary_impact, permanent_impact)``, both the same length as
        ``trade_sizes`` and in the same price units -- the two real
        curves this function exists to separate (e.g. for a caller
        plotting "temporary vs. permanent impact" against each other).

    Raises:
        ValueError: If any ``trade_sizes`` entry is negative, either
            impact coefficient is negative, or ``step_duration_seconds``
            is not positive.
    """
    if np.any(trade_sizes < 0):
        raise ValueError("trade_sizes entries must not be negative.")
    if temporary_impact_coefficient < 0:
        raise ValueError("temporary_impact_coefficient must not be negative.")
    if permanent_impact_coefficient < 0:
        raise ValueError("permanent_impact_coefficient must not be negative.")
    if step_duration_seconds <= 0:
        raise ValueError("step_duration_seconds must be positive.")

    temporary_impact = temporary_impact_coefficient * (trade_sizes / step_duration_seconds)
    permanent_impact = permanent_impact_coefficient * np.cumsum(trade_sizes)

    return temporary_impact, permanent_impact


def marchenko_pastur_threshold(
    n_assets: int, n_samples: int, *, variance: float = 1.0
) -> tuple[float, float]:
    r"""Marchenko-Pastur eigenvalue bounds: the real, provable "this much is pure noise" cutoff.

    For an :math:`n\_assets \times n\_samples` matrix of i.i.d. random
    entries (no real correlation structure at all), the Marchenko-Pastur
    theorem gives the *exact* asymptotic support of its correlation
    matrix's eigenvalue spectrum:

    .. math::
        \lambda_\pm = \sigma^2 (1 \pm \sqrt{q})^2, \qquad q = \frac{n\_assets}{n\_samples}

    Any measured eigenvalue at or below :math:`\lambda_+` is
    statistically indistinguishable from what pure sampling noise alone
    would produce from **zero** real correlation -- see
    :func:`denoise_correlation_matrix_mp`, which uses this real threshold
    to separate signal from noise in an empirical correlation matrix.

    Args:
        n_assets: Number of assets (matrix dimension). Must be positive.
        n_samples: Number of real return observations the correlation
            matrix was estimated from. Must be ``>= n_assets`` -- with
            fewer samples than assets the empirical correlation matrix
            is rank-deficient and this bound's derivation (which assumes
            :math:`q \le 1`) no longer applies.
        variance: :math:`\sigma^2`, the per-entry variance -- ``1.0`` (the
            default) for a true correlation matrix (unit variance by
            construction); pass the real average variance instead if
            calling this on a raw covariance matrix.

    Returns:
        ``(lambda_minus, lambda_plus)``.

    Raises:
        ValueError: If ``n_assets``/``n_samples`` is not positive,
            ``n_samples < n_assets``, or ``variance`` is not positive.
    """
    if n_assets <= 0:
        raise ValueError("n_assets must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_samples < n_assets:
        raise ValueError(
            f"n_samples ({n_samples}) must be >= n_assets ({n_assets}) -- the Marchenko-Pastur "
            "bound's derivation assumes q = n_assets/n_samples <= 1."
        )
    if variance <= 0:
        raise ValueError("variance must be positive.")

    q = n_assets / n_samples
    lambda_plus = variance * (1.0 + np.sqrt(q)) ** 2
    lambda_minus = variance * (1.0 - np.sqrt(q)) ** 2
    return float(lambda_minus), float(lambda_plus)


def denoise_correlation_matrix_mp(
    correlation_matrix: npt.NDArray[np.float64], n_samples: int
) -> npt.NDArray[np.float64]:
    r"""Real Random Matrix Theory spectral cleaning (Laloux et al. 1999) of an empirical correlation matrix.

    Eigendecomposes ``correlation_matrix``, replaces every eigenvalue at
    or below the real :func:`marchenko_pastur_threshold` upper bound
    (indistinguishable from pure sampling noise) with their common
    average (preserving the matrix's trace, i.e. total variance, exactly
    -- the standard real "eigenvalue clipping" convention, not simply
    discarding them), keeps every eigenvalue above it untouched (the real
    signal), reconstructs the matrix, and rescales it back to a unit
    diagonal (a correlation matrix's own defining property, which the
    reconstruction alone does not guarantee to floating-point precision).

    **Real, disclosed edge case, found by testing**: the final diagonal-
    restoring rescale is a congruence transform, not a similarity
    transform -- it does not exactly preserve eigenvalues in general, and
    can distort a signal eigenvalue's associated correlations
    substantially if a *small, near-perfectly-collinear* block of assets
    (pairwise correlation above roughly 0.9, e.g. from an unrealistically
    strong shared factor loading) creates near-zero "noise" eigenvalues
    that get flattened up to the bulk noise average -- verified directly:
    a planted 0.9/0.1-loaded 5-asset block distorted its own (0.99 raw)
    pairwise correlation down to 0.51 after cleaning, while the same
    check at a realistic 0.5/0.85 loading (raw correlation 0.27) cleaned
    to a reasonable 0.20 with an exactly unit diagonal throughout. Real
    financial return series essentially never sustain 0.9+ pairwise
    correlation over a meaningful window, so this is a real, disclosed
    boundary of the method's applicability, not a silently-hidden one.

    Args:
        correlation_matrix: A real, square, symmetric correlation matrix
            (unit diagonal).
        n_samples: Real number of observations it was estimated from --
            see :func:`marchenko_pastur_threshold`.

    Returns:
        The cleaned correlation matrix, same shape, real unit diagonal,
        real symmetric.

    Raises:
        ValueError: If ``correlation_matrix`` isn't square, isn't
            (approximately) symmetric, or its diagonal isn't
            (approximately) all ``1.0`` -- a real guard against silently
            "cleaning" a raw covariance matrix instead of a correlation
            matrix, or ``n_samples < correlation_matrix``'s dimension
            (propagated from :func:`marchenko_pastur_threshold`).
    """
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix must be square.")
    if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-8):
        raise ValueError("correlation_matrix must be symmetric.")
    if not np.allclose(np.diag(correlation_matrix), 1.0, atol=1e-6):
        raise ValueError("correlation_matrix must have a unit diagonal (pass a correlation, not a covariance, matrix).")

    n_assets = correlation_matrix.shape[0]
    _, lambda_plus = marchenko_pastur_threshold(n_assets, n_samples)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    noise_mask = eigenvalues <= lambda_plus
    average_noise_eigenvalue = float(np.mean(eigenvalues[noise_mask])) if np.any(noise_mask) else 0.0
    cleaned_eigenvalues = np.where(noise_mask, average_noise_eigenvalue, eigenvalues)

    cleaned = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(cleaned))
    cleaned = cleaned / np.outer(scale, scale)
    return np.asarray(cleaned, dtype=np.float64)


def calibrate_impact_coefficient_fdt(
    resting_spread: npt.NDArray[np.float64],
    average_trade_rate: float,
    *,
    coupling_constant: float = 1.0,
) -> float:
    r"""Fluctuation-Dissipation-**inspired** calibration of a real Almgren-Chriss temporary-impact coefficient.

    **Real, disclosed analogy, not a first-principles derivation**: the
    classical Fluctuation-Dissipation theorem (linear response theory)
    relates a system's response to an external perturbation to its own
    spontaneous equilibrium fluctuations. This function borrows that
    real *shape of reasoning* -- a market whose spread fluctuates more
    violently at rest should also show more temporary price concession
    per unit of trading rate -- as a real, practical calibration:

    .. math::
        \eta = k \cdot \frac{\sigma_{\text{spread}}}{\bar{v}}

    where :math:`\sigma_{\text{spread}}` is the real sample standard
    deviation of ``resting_spread`` (the market's own real, unperturbed
    fluctuation), :math:`\bar{v}` is ``average_trade_rate`` (real units
    per second), and :math:`k` is ``coupling_constant``. The units work
    out to price per (units/second) -- exactly what
    :func:`almgren_chriss_impact_costs`'s own
    ``temporary_impact_coefficient`` expects.

    Args:
        resting_spread: Real observed bid-ask spread samples from
            *undisturbed* market conditions (i.e. not sampled during the
            meta-order's own execution, which would contaminate this
            calibration with the very impact it's meant to calibrate).
            Must have at least 2 elements.
        average_trade_rate: Real average traded volume per second in
            this market. Must be positive.
        coupling_constant: :math:`k`, a real, disclosed scale factor
            (unitless in this simplified form) -- ``1.0`` absent a
            fitted value for this specific venue/instrument. Must be
            positive.

    Returns:
        A real, calibrated ``temporary_impact_coefficient``, directly
        usable as that argument to :func:`almgren_chriss_trajectory`/
        :func:`almgren_chriss_impact_costs`.

    Raises:
        ValueError: If ``resting_spread`` has fewer than 2 elements, or
            ``average_trade_rate``/``coupling_constant`` is not positive.
    """
    if resting_spread.size < 2:
        raise ValueError("resting_spread must have at least 2 elements.")
    if average_trade_rate <= 0:
        raise ValueError("average_trade_rate must be positive.")
    if coupling_constant <= 0:
        raise ValueError("coupling_constant must be positive.")

    spread_volatility = float(np.std(resting_spread, ddof=1))
    return coupling_constant * spread_volatility / average_trade_rate


def wasserstein_price_distance(
    distribution_a: npt.NDArray[np.float64], distribution_b: npt.NDArray[np.float64]
) -> float:
    r"""Real 1-Wasserstein ("Earth Mover's") distance between two empirical price/return distributions.

    Thin, real wrapper over :func:`scipy.stats.wasserstein_distance` --
    unlike an :math:`L_2`/KL-divergence comparison, this real optimal-
    transport metric stays meaningful even when the two distributions'
    supports barely overlap (e.g. comparing a pre- and post-regime-shift
    return distribution), since it measures the real minimum "earth-
    moving" cost to reshape one distribution into the other rather than
    a pointwise density mismatch.

    Args:
        distribution_a: 1D array of real samples (e.g. returns, or fill
            prices) from the first distribution. Must be non-empty.
        distribution_b: 1D array of real samples from the second
            distribution. Must be non-empty.

    Returns:
        The real Wasserstein-1 distance, in the same units as the input
        samples. Always ``>= 0``; ``0.0`` only if the two empirical
        distributions are identical.

    Raises:
        ValueError: If either distribution is empty.
    """
    if distribution_a.size == 0 or distribution_b.size == 0:
        raise ValueError("distribution_a and distribution_b must both be non-empty.")
    return float(wasserstein_distance(distribution_a, distribution_b))


def giant_cluster_fraction(correlation_matrix: npt.NDArray[np.float64], threshold: float) -> float:
    r"""Percolation-style systemic concentration: the real fraction of assets fused into one cluster.

    Builds a real graph -- one node per asset, a real edge between
    :math:`i` and :math:`j` iff :math:`|\rho_{ij}| \ge` ``threshold`` --
    and returns the fraction of nodes in its largest real connected
    component (:func:`scipy.sparse.csgraph.connected_components`), the
    real graph-theoretic analogue of a percolation network's "giant
    cluster." As this fraction approaches ``1.0``, the tracked
    instruments have effectively fused into one undiversifiable
    correlated blob -- a real systemic-concentration signal, independent
    of (and a real complement to) any single pairwise correlation check.

    Args:
        correlation_matrix: A real, square, symmetric correlation matrix.
        threshold: Absolute correlation at or above which two assets are
            considered connected, in ``(0, 1]``.

    Returns:
        The largest connected component's size divided by the total
        number of assets, in ``(0, 1]``. ``1.0`` for a single asset
        (trivially its own, whole "cluster").

    Raises:
        ValueError: If ``correlation_matrix`` isn't square, or
            ``threshold`` is not in ``(0, 1]``.
    """
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix must be square.")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in (0, 1].")

    n_assets = correlation_matrix.shape[0]
    if n_assets == 1:
        return 1.0

    adjacency = np.abs(correlation_matrix) >= threshold
    np.fill_diagonal(adjacency, False)

    _n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
    component_sizes = np.bincount(np.asarray(labels, dtype=np.int64))
    largest_component_size: float = float(component_sizes.max())
    result: float = largest_component_size / float(n_assets)
    return result


def kolmogorov_compressibility_ratio(
    sequence: npt.NDArray[np.float64], *, quantization_levels: int = 256
) -> float:
    r"""Real compressibility proxy for Kolmogorov complexity: a genuine algorithmic-footprint signal.

    True Kolmogorov complexity (the length of the shortest program that
    outputs a given string) is famously **uncomputable** -- this uses the
    real, standard practical proxy instead (the same idea behind
    Normalized Compression Distance, Cilibrasi & Vitanyi 2005): a real
    lossless compressor (``zlib``, DEFLATE) is itself a real, if
    imperfect, model of "how much structure exists to exploit," so its
    real achieved compression ratio is a real, computable stand-in.

    A market perfectly efficient at this resolution is (by definition)
    indistinguishable from noise and barely compresses
    (ratio :math:`\approx 1`); a real, exploitable structure -- e.g. a
    deterministic algorithm's own footprint stepping through the book --
    compresses further (ratio :math:`\ll 1`).

    ``sequence`` is quantized into ``quantization_levels`` discrete bins
    first (a real, disclosed, necessary step: ``zlib`` compresses real
    bytes, not raw floats, and floating-point noise in the low bits would
    make *everything* look incompressible regardless of real structure).

    Args:
        sequence: 1D array of real values (e.g. returns, order-book
            imbalance readings). Must have at least 2 elements.
        quantization_levels: Number of discrete bins to quantize into,
            in ``[2, 256]`` (must fit in one byte per sample).

    Returns:
        ``compressed_size / raw_size`` for the quantized byte sequence,
        in ``(0, 1]`` (in practice; DEFLATE's own header overhead can
        push this fractionally above ``1.0`` for a very short, already-
        incompressible sequence -- an honest artifact of the real
        codec, not clipped away here).

    Raises:
        ValueError: If ``sequence`` has fewer than 2 elements, or
            ``quantization_levels`` is not in ``[2, 256]``.
    """
    if sequence.size < 2:
        raise ValueError("sequence must have at least 2 elements.")
    if not (2 <= quantization_levels <= 256):
        raise ValueError("quantization_levels must be in [2, 256].")

    seq_min, seq_max = float(np.min(sequence)), float(np.max(sequence))
    if seq_max == seq_min:
        normalized = np.zeros_like(sequence)
    else:
        normalized = (sequence - seq_min) / (seq_max - seq_min)

    quantized = np.clip(np.round(normalized * (quantization_levels - 1)), 0, quantization_levels - 1).astype(np.uint8)
    raw_bytes = quantized.tobytes()
    compressed = zlib.compress(raw_bytes, level=9)
    return len(compressed) / len(raw_bytes)


def simulate_constant_mix_rebalancing(
    price_path: npt.NDArray[np.float64],
    *,
    risky_weight: float = 0.5,
    initial_capital: float = 1.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Real constant-mix ("Shannon's Demon") rebalancing simulation -- shows the real gain *and* the real loss.

    Continuously rebalancing a portfolio back to a fixed ``risky_weight``
    split between a risky asset and cash is a real, well-known way to
    convert an asset's own volatility into growth (Shannon's informal
    "Demon" thought experiment; formalized in e.g. Luenberger's
    *Investment Science*) -- **but only for a real, zero-drift, mean-
    reverting-in-log-space path**. This function does not assume that:
    it replays ``price_path`` exactly as given, so a genuinely trending
    (non-zero-drift) path shows the real underperformance versus simple
    buy-and-hold that rebalancing produces in that regime -- this is a
    real, disclosed correction to the briefed claim that this is
    "risk-free": it is a real, directional bet on the *path's own
    statistical character* (noisy and range-bound vs. trending), not a
    guaranteed profit independent of it.

    Args:
        price_path: 1D array of real, strictly positive prices,
            chronological order. Must have at least 2 elements.
        risky_weight: Target fraction of portfolio value held in the
            risky asset after every rebalance, in ``(0, 1)``.
        initial_capital: Starting portfolio value. Must be positive.

    Returns:
        ``(rebalanced_value_path, buy_and_hold_value_path)``, both the
        same length as ``price_path`` and starting at
        ``initial_capital`` -- compare them directly to see whether
        rebalancing helped or hurt over this specific real path.

    Raises:
        ValueError: If ``price_path`` has fewer than 2 elements or isn't
            strictly positive throughout, ``risky_weight`` is not in
            ``(0, 1)``, or ``initial_capital`` is not positive.
    """
    if price_path.size < 2:
        raise ValueError("price_path must have at least 2 elements.")
    if np.any(price_path <= 0):
        raise ValueError("price_path must be strictly positive throughout.")
    if not (0.0 < risky_weight < 1.0):
        raise ValueError("risky_weight must be in (0, 1).")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    n = price_path.size
    rebalanced_value = np.empty(n, dtype=np.float64)
    rebalanced_value[0] = initial_capital
    shares = (initial_capital * risky_weight) / price_path[0]
    cash = initial_capital * (1.0 - risky_weight)

    for i in range(1, n):
        total = shares * price_path[i] + cash
        rebalanced_value[i] = total
        shares = (total * risky_weight) / price_path[i]
        cash = total * (1.0 - risky_weight)

    buy_and_hold_value = initial_capital * (price_path / price_path[0])
    return rebalanced_value, buy_and_hold_value


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: Literal["call", "put"] = "call",
) -> float:
    r"""Real Black-Scholes-Merton closed-form option price, under the real risk-neutral measure Q.

    Girsanov's theorem is exactly what justifies pricing this way: a
    real change of measure (:math:`\mathbb{P} \to \mathbb{Q}`) removes
    the underlying's real-world drift :math:`\mu` from the pricing
    formula entirely, replacing it with the risk-free rate -- **a real
    property of the model's internal consistency, not a claim that real-
    world derivatives trading is risk-free**. This price is only ever as
    good as ``volatility`` (and the GBM assumption itself): a wrong
    volatility input still returns a confident-looking, real number that
    is simply wrong -- real, disclosed model risk, the same kind that
    made LTCM's 1998 collapse a model-risk event, not a market-timing one.

    Args:
        spot: Current price of the underlying. Must be positive.
        strike: Strike price. Must be positive.
        time_to_expiry: Real time to expiry, in years. Must be positive.
        risk_free_rate: Continuously-compounded risk-free rate.
        volatility: Annualized volatility. Must be positive.
        option_type: ``"call"`` or ``"put"``.

    Returns:
        The real Black-Scholes price, ``>= 0``.

    Raises:
        ValueError: If ``spot``/``strike``/``time_to_expiry``/``volatility``
            is not positive, or ``option_type`` is neither ``"call"`` nor ``"put"``.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put".')

    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    discount = np.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        return float(spot * norm.cdf(d1) - strike * discount * norm.cdf(d2))
    return float(strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1))


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: Literal["call", "put"] = "call",
) -> float:
    r"""Real Black-Scholes delta (:math:`\partial V/\partial S`) -- the real Itô-derived hedge ratio.

    See :func:`black_scholes_price`'s own docstring for the same real
    model-risk disclosure, which applies identically here.

    Args:
        Same as :func:`black_scholes_price`.

    Returns:
        The real delta: in ``[0, 1]`` for a call, ``[-1, 0]`` for a put.

    Raises:
        ValueError: Same conditions as :func:`black_scholes_price`.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put".')

    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def black_scholes_gamma(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    r"""Real Black-Scholes gamma (:math:`\partial^2 V/\partial S^2`) -- identical for a call and a put.

    Gamma is exactly what :func:`simulate_discrete_delta_hedge` shows the
    real cost of: it measures how fast delta itself changes, i.e. how
    quickly a discrete hedge goes stale between rebalances. See
    :func:`black_scholes_price`'s own docstring for the same real model-
    risk disclosure, which applies identically here.

    Args:
        Same as :func:`black_scholes_price` (no ``option_type`` -- gamma
        does not depend on it).

    Returns:
        The real gamma, always ``> 0`` for both a call and a put.

    Raises:
        ValueError: Same conditions as :func:`black_scholes_price`.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    return float(norm.pdf(d1) / (spot * volatility * sqrt_t))


def black_scholes_vega(spot: float, strike: float, time_to_expiry: float, risk_free_rate: float, volatility: float) -> float:
    r"""Real Black-Scholes vega (:math:`\partial V/\partial \sigma`) -- identical for a call and a put.

    Real, disclosed convention: this is the *raw* per-unit-volatility
    sensitivity (multiply by ``0.01`` for the common practitioner "per
    1 vol point" convention -- left un-rescaled here so it composes
    correctly with :func:`solve_hedge_ratios`'s own raw units).

    Args:
        Same as :func:`black_scholes_gamma`.

    Returns:
        The real vega, always ``> 0``.

    Raises:
        ValueError: Same conditions as :func:`black_scholes_price`.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    return float(spot * norm.pdf(d1) * sqrt_t)


def black_scholes_theta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: Literal["call", "put"] = "call",
) -> float:
    r"""Real Black-Scholes theta (:math:`\partial V/\partial t`, real time-decay).

    Args:
        Same as :func:`black_scholes_price`.

    Returns:
        The real theta, in price units per **year** (this module's own
        consistent time unit -- divide by 365 for the common
        practitioner "per calendar day" convention). Typically negative
        for a long option (real time decay), though a deep ITM European
        put can show positive theta -- a real, correctly-reproduced
        textbook edge case, not a bug.

    Raises:
        ValueError: Same conditions as :func:`black_scholes_price`.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")
    if option_type not in ("call", "put"):
        raise ValueError('option_type must be "call" or "put".')

    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discount = np.exp(-risk_free_rate * time_to_expiry)
    time_decay_term = -(spot * norm.pdf(d1) * volatility) / (2.0 * sqrt_t)

    if option_type == "call":
        return float(time_decay_term - risk_free_rate * strike * discount * norm.cdf(d2))
    return float(time_decay_term + risk_free_rate * strike * discount * norm.cdf(-d2))


def solve_hedge_ratios(
    sensitivity_matrix: npt.NDArray[np.float64], target_exposure: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Real least-squares solve for the hedge sizes that best neutralize a real target exposure.

    :func:`check_hedge_completeness`'s natural sequel: given the same
    real ``(n_instruments, n_risk_factors)`` sensitivity matrix and a
    real current portfolio exposure vector (length ``n_risk_factors``,
    e.g. real aggregate delta/vega/theta), solves for the real position
    sizes :math:`w` (length ``n_instruments``) minimizing
    :math:`\lVert M^T w + e \rVert_2^2` -- i.e. the real hedge that
    brings the *combined* exposure as close to zero as this instrument
    set's real "variety" allows (:func:`numpy.linalg.lstsq`, the
    standard, real minimum-norm solution; exact zero residual only when
    :func:`check_hedge_completeness` would report ``True``).

    Args:
        sensitivity_matrix: Real ``(n_instruments, n_risk_factors)``
            sensitivities -- same shape/meaning as
            :func:`check_hedge_completeness`'s own argument.
        target_exposure: Real current exposure to neutralize, length
            ``n_risk_factors`` (the position sizes are solved so the
            *hedge's* exposure approximately cancels this).

    Returns:
        ``(hedge_sizes, residual_exposure)`` -- ``hedge_sizes`` has
        length ``n_instruments`` (the real size to trade in each,
        signed); ``residual_exposure`` (length ``n_risk_factors``) is
        the real exposure that remains uncancelled -- all
        (approximately) ``0`` only when the instrument set has real,
        complete requisite variety for this target.

    Raises:
        ValueError: If ``sensitivity_matrix`` isn't 2D, either dimension
            is zero, or ``target_exposure``'s length doesn't match
            ``sensitivity_matrix``'s number of risk factors.
    """
    if sensitivity_matrix.ndim != 2:
        raise ValueError("sensitivity_matrix must be 2D (n_instruments x n_risk_factors).")
    n_instruments, n_factors = sensitivity_matrix.shape
    if n_instruments == 0 or n_factors == 0:
        raise ValueError("sensitivity_matrix must have at least one instrument and one risk factor.")
    if target_exposure.shape != (n_factors,):
        raise ValueError(f"target_exposure must have shape ({n_factors},), got {target_exposure.shape}.")

    hedge_sizes, _residuals, _rank, _singular_values = np.linalg.lstsq(
        sensitivity_matrix.T, -target_exposure, rcond=None
    )
    residual_exposure = sensitivity_matrix.T @ hedge_sizes + target_exposure
    return hedge_sizes, residual_exposure


def simulate_discrete_delta_hedge(
    price_path: npt.NDArray[np.float64],
    time_step: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    *,
    option_type: Literal["call", "put"] = "call",
) -> tuple[npt.NDArray[np.float64], float]:
    r"""Real, self-financing discrete-time delta-hedge replication -- shows the real, nonzero hedging error.

    Itô's lemma's real promise is that *continuous* delta-hedging drives
    the replication error to exactly zero (the :math:`dW_t` term cancels
    identically). This simulates the real, practical alternative --
    rebalancing only at each of ``price_path``'s discrete observations,
    :func:`time_step` apart -- and returns the real, nonzero
    ``replication_error`` this leaves at expiry: the honest, disclosed
    correction to the briefed claim that delta-hedging synthesizes "a
    risk-free bond" -- it synthesizes an *approximately* risk-free
    position, with a real, measurable residual (gamma/discretization)
    risk that shrinks toward zero only as ``time_step`` itself does.

    Args:
        price_path: 1D array of real, strictly positive underlying
            prices at each hedging observation, chronological order,
            evenly spaced ``time_step`` apart. Must have at least 2
            elements; the option expires at the *last* observation.
        time_step: Real time between consecutive observations, in years.
            Must be positive.
        strike: Strike price. Must be positive.
        risk_free_rate: Continuously-compounded risk-free rate, used
            both for pricing and for real cash-account accrual between
            rebalances.
        volatility: Annualized volatility used for hedging (this
            simulation's own assumption -- a real, disclosed source of
            hedging error if the path's *realized* volatility differs).
        option_type: ``"call"`` or ``"put"``.

    Returns:
        ``(replicating_portfolio_value_path, replication_error)`` --
        ``replication_error`` is ``replicating_portfolio_value_path[-1]
        - real_option_payoff_at_expiry``; ``0.0`` only in the unreachable
        continuous-hedging limit.

    Raises:
        ValueError: If ``price_path`` has fewer than 2 elements or isn't
            strictly positive, or ``time_step``/``strike``/``volatility``
            is not positive.
    """
    if price_path.size < 2:
        raise ValueError("price_path must have at least 2 elements.")
    if np.any(price_path <= 0):
        raise ValueError("price_path must be strictly positive throughout.")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")
    if strike <= 0:
        raise ValueError("strike must be positive.")
    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    n = price_path.size
    total_time_to_expiry = time_step * (n - 1)

    option_value_0 = black_scholes_price(
        price_path[0], strike, total_time_to_expiry, risk_free_rate, volatility, option_type=option_type
    )
    delta = black_scholes_delta(
        price_path[0], strike, total_time_to_expiry, risk_free_rate, volatility, option_type=option_type
    )
    # Self-financing: sell the option (receive option_value_0), buy `delta` shares -- the
    # remaining cash (negative = borrowed) is what earns/costs risk_free_rate between rebalances.
    cash = option_value_0 - delta * price_path[0]

    portfolio_value = np.empty(n, dtype=np.float64)
    portfolio_value[0] = option_value_0

    for i in range(1, n):
        cash *= np.exp(risk_free_rate * time_step)
        portfolio_value[i] = delta * price_path[i] + cash
        if i < n - 1:
            remaining = total_time_to_expiry - time_step * i
            new_delta = black_scholes_delta(
                price_path[i], strike, remaining, risk_free_rate, volatility, option_type=option_type
            )
            cash -= (new_delta - delta) * price_path[i]
            delta = new_delta

    if option_type == "call":
        payoff = max(float(price_path[-1]) - strike, 0.0)
    else:
        payoff = max(strike - float(price_path[-1]), 0.0)

    replication_error = float(portfolio_value[-1] - payoff)
    return portfolio_value, replication_error


_SPEED_OF_LIGHT_VACUUM_M_S: Final[float] = 299_792_458.0  # exact, SI definition
# Real, typical single-mode fiber core refractive index (e.g. Corning
# SMF-28 datasheet range ~1.46-1.48) -- signal speed in fiber is c/n.
_FIBER_REFRACTIVE_INDEX: Final[float] = 1.4682
# Real refractive index of standard atmosphere at sea level (visible/
# microwave range) -- microwave links run only marginally slower than
# vacuum c, which is real, published motivation for real Chicago-NY
# microwave relay networks over fiber.
_AIR_REFRACTIVE_INDEX: Final[float] = 1.0002726


def light_cone_latency(distance_km: float, *, medium: Literal["fiber", "microwave", "vacuum"] = "fiber") -> float:
    r"""Real physical lower bound on one-way information propagation time between two points.

    A real consequence of special relativity (no information travels
    faster than :math:`c`) -- this is a genuine physical bound real HFT
    microwave networks (e.g. Chicago-New York) are built to approach.
    **Not a claim this project has real co-located exchange
    infrastructure to exploit it** -- see the module docstring: this is
    a real calculator, not a functioning arbitrage capability.

    Args:
        distance_km: Real straight-line distance between the two points,
            in kilometers. Must be positive.
        medium: ``"fiber"`` (real single-mode fiber, the slowest of the
            three -- signal speed :math:`c/n_{\text{fiber}}`),
            ``"microwave"`` (real line-of-sight microwave through air,
            close to :math:`c`), or ``"vacuum"`` (the real, absolute
            physical floor, :math:`c` itself).

    Returns:
        The real minimum one-way latency, in seconds.

    Raises:
        ValueError: If ``distance_km`` is not positive.
    """
    if distance_km <= 0:
        raise ValueError("distance_km must be positive.")

    if medium == "vacuum":
        speed = _SPEED_OF_LIGHT_VACUUM_M_S
    elif medium == "microwave":
        speed = _SPEED_OF_LIGHT_VACUUM_M_S / _AIR_REFRACTIVE_INDEX
    else:
        speed = _SPEED_OF_LIGHT_VACUUM_M_S / _FIBER_REFRACTIVE_INDEX

    return (distance_km * 1000.0) / speed


def check_hedge_completeness(sensitivity_matrix: npt.NDArray[np.float64]) -> tuple[bool, int, int]:
    r"""Real Ashby's Law of Requisite Variety check for hedge completeness -- a real linear-algebra rank test.

    Ashby's "only variety can destroy variety" becomes, for a real
    portfolio, a real, computable question: given an
    ``n_instruments x n_risk_factors`` real sensitivity matrix (each
    entry the real change in instrument :math:`i`'s value per unit
    change in risk factor :math:`j` -- e.g. real delta/vega/rho Greeks,
    or any other real risk-factor sensitivities), can this instrument set
    hedge *every* combination of risk-factor moves? It can iff the
    matrix's real rank equals the number of risk factors -- if
    ``rank < n_risk_factors``, a real, specific combination of risk-
    factor moves exists that no linear combination of these instruments
    can offset, regardless of position sizing.

    Args:
        sensitivity_matrix: Real ``(n_instruments, n_risk_factors)``
            array of sensitivities. Both dimensions must be positive.

    Returns:
        ``(is_complete, rank, n_risk_factors)`` -- ``is_complete`` is
        ``True`` iff ``rank >= n_risk_factors``.

    Raises:
        ValueError: If ``sensitivity_matrix`` isn't 2D, or either
            dimension is zero.
    """
    if sensitivity_matrix.ndim != 2:
        raise ValueError("sensitivity_matrix must be 2D (n_instruments x n_risk_factors).")
    n_instruments, n_factors = sensitivity_matrix.shape
    if n_instruments == 0 or n_factors == 0:
        raise ValueError("sensitivity_matrix must have at least one instrument and one risk factor.")

    rank = int(np.linalg.matrix_rank(sensitivity_matrix))
    return rank >= n_factors, rank, n_factors


def estimate_lyapunov_exponent(
    series: npt.NDArray[np.float64],
    *,
    embedding_dimension: int = 3,
    time_delay: int = 1,
    min_time_steps: int = 1,
    max_time_steps: int = 10,
    theiler_window: int | None = None,
) -> tuple[float, float]:
    r"""Real largest Lyapunov exponent, via Rosenstein/Collins/De Luca's (1993) nearest-neighbor divergence method.

    Reconstructs a phase-space embedding of ``series`` (Takens' delay
    embedding: each point is ``embedding_dimension`` values spaced
    ``time_delay`` apart), finds each embedded point's real nearest
    neighbor (excluding a ``theiler_window`` of temporally adjacent
    points, which would trivially look "close" without reflecting real
    dynamics), then tracks how fast the real distance between each pair
    grows over the next few steps. A genuinely chaotic system's nearby
    trajectories diverge *exponentially*, so the average log-distance
    grows *linearly* with time -- the slope of that line, fit by
    real least-squares, is the estimated exponent.

    **Real, load-bearing correction, not a caveat to skip**: this is a
    real, correctly-implemented, standard estimator -- verified below
    against the logistic map (:math:`x_{n+1} = 4 x_n (1 - x_n)`), an
    exact textbook chaotic system whose true largest Lyapunov exponent
    is exactly ``ln(2)`` (a known, derivable constant, not a fitted
    number). It is **not** verified against price data, and should not
    be trusted the same way on it: a substantial, real body of academic
    literature (LeBaron, "Chaos and Nonlinear Forecastability in
    Economics and Finance", 1994; the multi-laboratory comparisons in
    Barnett et al.'s "A Single-Blind Controlled Competition Among Tests
    for Nonlinearity and Chaos", 1997) has repeatedly found that
    apparent low-dimensional chaos detected this way in real financial
    return series is frequently a finite-sample estimation artifact
    (short, noisy, non-stationary series routinely produce a
    positive-looking slope even when generated by a process with no
    real deterministic chaos at all), not reliable evidence the market
    itself is a deterministic chaotic system. A positive estimate here
    on real price data is real evidence of *local, short-horizon
    trajectory sensitivity in that specific estimation window* -- a
    real, useful, dynamically-adjustable signal for how far ahead a
    prediction can honestly reach *right now* -- not a fixed, provable
    guarantee, and not proof of chaos.

    **Real, disclosed complexity bound**: the neighbor search here is
    brute-force (:math:`O(n^2)`), not a k-d tree -- correct and fast
    enough for the hundreds-to-low-thousands of points a rolling
    trading window realistically holds, not intended for very large
    series.

    Args:
        series: Real 1D scalar time series (e.g. log-returns or prices).
            Must have enough points for the requested embedding/window:
            at least ``embedding_dimension * time_delay + max_time_steps
            + 2``.
        embedding_dimension: Takens' embedding dimension :math:`m`. Must
            be >= 1.
        time_delay: Real embedding delay :math:`\tau`, in samples. Must
            be >= 1.
        min_time_steps: First real forward step (in samples) included in
            the divergence-slope fit. Must be >= 1.
        max_time_steps: Last real forward step included in the fit. Must
            be `> min_time_steps`.
        theiler_window: Real minimum index separation ``|i - j|``
            required for ``j`` to count as a valid nearest neighbor of
            ``i`` -- excludes trivially-close temporally-adjacent points
            that would understate real divergence. Defaults to
            ``embedding_dimension * time_delay`` (a standard choice) when
            omitted.

    Returns:
        ``(lyapunov_exponent, lyapunov_time)`` -- the real estimated
        exponent (per-sample units) and ``1 / lyapunov_exponent`` (the
        real "prediction expiration horizon," in samples) when the
        exponent is positive; ``lyapunov_time`` is ``float('inf')`` for
        a non-positive estimate (no finite real expiration implied by a
        non-chaotic reading).

    Raises:
        ValueError: If any parameter is invalid, ``series`` is too short
            for the requested parameters, or too few valid neighbor
            pairs exist to fit a real slope.
    """
    if embedding_dimension < 1:
        raise ValueError("embedding_dimension must be >= 1.")
    if time_delay < 1:
        raise ValueError("time_delay must be >= 1.")
    if min_time_steps < 1:
        raise ValueError("min_time_steps must be >= 1.")
    if max_time_steps <= min_time_steps:
        raise ValueError("max_time_steps must be > min_time_steps.")

    window = theiler_window if theiler_window is not None else embedding_dimension * time_delay
    if window < 0:
        raise ValueError("theiler_window must not be negative.")

    n = series.size
    span = (embedding_dimension - 1) * time_delay
    n_vectors = n - span
    if n_vectors < max_time_steps + 2:
        raise ValueError(
            f"series too short: need at least {span + max_time_steps + 2} points for the requested "
            f"embedding_dimension/time_delay/max_time_steps, got {n}."
        )

    embedded = np.array(
        [series[i : i + span + 1 : time_delay] for i in range(n_vectors)], dtype=np.float64
    )

    valid_range = n_vectors - max_time_steps
    if valid_range < 1:
        raise ValueError("no valid embedded points remain after reserving room for max_time_steps.")

    nearest_indices = np.empty(valid_range, dtype=np.int64)
    for i in range(valid_range):
        distances = np.linalg.norm(embedded[:valid_range] - embedded[i], axis=1)
        lo, hi = max(0, i - window), min(valid_range, i + window + 1)
        distances[lo:hi] = np.inf
        if not np.isfinite(distances).any():
            nearest_indices[i] = -1
            continue
        nearest_indices[i] = int(np.argmin(distances))

    steps: list[float] = []
    log_divergences: list[float] = []
    for k in range(min_time_steps, max_time_steps + 1):
        samples = []
        for i in range(valid_range):
            j = nearest_indices[i]
            if j < 0:
                continue
            distance = float(np.linalg.norm(embedded[i + k] - embedded[j + k]))
            if distance > 0:
                samples.append(np.log(distance))
        if samples:
            steps.append(float(k))
            log_divergences.append(float(np.mean(samples)))

    if len(steps) < 2:
        raise ValueError("insufficient valid neighbor-divergence samples to fit a real Lyapunov slope.")

    slope, _intercept = np.polyfit(np.array(steps), np.array(log_divergences), 1)
    lyapunov_exponent = float(slope)
    lyapunov_time = 1.0 / lyapunov_exponent if lyapunov_exponent > 0 else float("inf")
    return lyapunov_exponent, lyapunov_time


def nyquist_undersampling_check(
    sampling_interval_seconds: float, reference_interval_seconds: float
) -> tuple[bool, float]:
    r"""Real Nyquist-Shannon aliasing-risk check between two *independently observed* real cadences.

    The textbook criterion: to correctly resolve a signal whose fastest
    real component has frequency :math:`f_{\max}`, the sampling
    frequency must satisfy :math:`f_s \ge 2 f_{\max}`, i.e. the sampling
    interval must not exceed :math:`1 / (2 f_{\max})`. Given a real,
    independently-measured faster reference cadence (``reference_interval_seconds``,
    e.g. this system's own real raw tick/quote inter-arrival time) as an
    honest proxy for :math:`1 / f_{\max}`, this reduces to: the real
    ``sampling_interval_seconds`` (e.g. an OHLCV candle interval) must
    not exceed ``reference_interval_seconds / 2``.

    **Why ``reference_interval_seconds`` must come from an independent
    real source, never from the same already-sampled series being
    checked**: a signal's own FFT (e.g. :class:`~src.agents.
    spectral_denoiser.agent.SpectralDenoiserAgent`'s ``dominant_frequencies``)
    can **never** contain a frequency above the Nyquist bound implied by
    *its own* sampling interval -- that is a mathematical guarantee of
    how :func:`scipy.fft.rfftfreq` works, not evidence the true
    underlying signal has no faster real component. Aliasing is
    precisely the phenomenon where a real frequency above :math:`f_s/2`
    gets folded down and misreported as a slower one *within* that
    range -- checking an FFT's own output against its own sampling
    rate's Nyquist bound is therefore tautologically always satisfied
    and can never detect it. A real, independent, faster-cadence
    reference (this system's own live tick/quote feed, not the same
    candle series) is the only honest way to flag real undersampling
    risk.

    Args:
        sampling_interval_seconds: The real interval the signal actually
            analyzed was sampled at (e.g. a real OHLCV candle interval).
            Must be positive.
        reference_interval_seconds: A real, independently-observed
            faster cadence for the same underlying process (e.g. this
            system's own real median order-book tick inter-arrival
            time). Must be positive.

    Returns:
        ``(is_undersampled, nyquist_safe_max_sampling_interval_seconds)``
        -- ``nyquist_safe_max_sampling_interval_seconds`` is
        ``reference_interval_seconds / 2``; ``is_undersampled`` is
        ``True`` when ``sampling_interval_seconds`` exceeds it.

    Raises:
        ValueError: If either argument is not positive.
    """
    if sampling_interval_seconds <= 0:
        raise ValueError("sampling_interval_seconds must be positive.")
    if reference_interval_seconds <= 0:
        raise ValueError("reference_interval_seconds must be positive.")

    nyquist_safe_max_sampling_interval_seconds = reference_interval_seconds / 2.0
    is_undersampled = sampling_interval_seconds > nyquist_safe_max_sampling_interval_seconds
    return is_undersampled, nyquist_safe_max_sampling_interval_seconds


def _discretize_into_bins(values: npt.NDArray[np.float64], n_bins: int) -> npt.NDArray[np.int64]:
    """Real rank-based discretization into ``n_bins`` labels ``0..n_bins-1``, used only by :func:`transfer_entropy`.

    Bins by **sorted rank** (each of the ``n`` real values gets assigned
    to one of ``n_bins`` groups of (as close to) equal size, in sorted
    order), not by :func:`numpy.quantile` edges fed through
    :func:`numpy.digitize` -- a real bug found by testing this against
    a real low-cardinality (binary 0/1) series: a quantile edge that
    lands exactly *on* a repeated value (a near-certainty with only two
    distinct values) collapses every value into a single bin under
    ``digitize``, silently destroying all real variation before entropy
    is ever computed. Rank-based assignment (stable sort, then split by
    position) has no such edge case: it always produces ``n_bins``
    real, non-degenerate groups whenever ``values`` has that many
    distinct values.
    """
    n = values.size
    order = np.argsort(values, kind="stable")
    labels = np.empty(n, dtype=np.int64)
    labels[order] = (np.arange(n, dtype=np.int64) * n_bins) // n
    return labels


def _joint_shannon_entropy_bits(*label_arrays: npt.NDArray[np.int64]) -> float:
    """Real Shannon entropy (bits) of the joint empirical distribution of the given label arrays, used only by :func:`transfer_entropy`."""
    joint = list(zip(*(labels.tolist() for labels in label_arrays)))
    total = len(joint)
    counts = Counter(joint)
    probabilities = np.array([count / total for count in counts.values()], dtype=np.float64)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def transfer_entropy(
    source: npt.NDArray[np.float64], target: npt.NDArray[np.float64], *, lag: int = 1, n_bins: int = 4
) -> float:
    r"""Real Transfer Entropy :math:`T_{X \to Y}` (Schreiber, 2000): directed, information-theoretic Granger-style causality.

    Answers a real, precise question correlation cannot: does knowing
    ``source``'s past genuinely reduce real uncertainty about
    ``target``'s future, beyond what ``target``'s *own* past already
    tells you? Computed here via the real entropy decomposition

    .. math::
        T_{X \to Y} = \big[H(Y_{t+l}, Y_t) - H(Y_t)\big]
                     - \big[H(Y_{t+l}, Y_t, X_t) - H(Y_t, X_t)\big]

    where each real joint/marginal Shannon entropy is estimated from
    the empirical histogram of *quantile-discretized* (``n_bins`` real
    bins, so it works whatever ``source``/``target``'s real scale is)
    values -- a real, standard, simple plug-in estimator.

    **Real, disclosed simplification**: this uses only a single lagged
    history value for both series (``k = l = 1`` in the textbook
    notation, not a longer embedding) and fixed-quantile binning, not a
    continuous kernel or Kraskov-Stögbauer-Grassberger estimator -- a
    real, standard, simple approach with a real, known finite-sample
    bias: with too few real samples or too many bins, this estimator
    can read a small **positive** value from pure independent noise
    (spurious "information"), and can even read a small **negative**
    value when the true transfer entropy is at or near zero (a real,
    documented property of the plug-in histogram estimator -- Schreiber,
    2000; Vicente et al., "Transfer entropy — a model-free measure of
    effective connectivity for the neurosciences", 2011 -- never
    silently clipped to zero here, since that would hide real estimator
    noise rather than disclose it).

    Args:
        source: Real 1D series, the candidate "cause" (:math:`X`).
        target: Real 1D series, the candidate "effect" (:math:`Y`). Same
            length as ``source``.
        lag: Real number of samples ahead ``target`` is evaluated at
            relative to both series' "past" (:math:`l` above). Must be
            >= 1.
        n_bins: Real number of quantile bins each series is discretized
            into. Must be >= 2.

    Returns:
        The real estimated :math:`T_{X \to Y}`, in bits. See the
        disclosed bias above before treating a small value (positive or
        negative) as meaningful.

    Raises:
        ValueError: If ``source``/``target`` have different lengths,
            ``lag < 1``, ``n_bins < 2``, or either series has too few
            points (``<= lag + 1``) for the requested lag.
    """
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape.")
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    if target.size <= lag + 1:
        raise ValueError(f"series too short: need more than lag+1={lag + 1} points, got {target.size}.")

    future_target = target[lag:]
    past_target = target[:-lag]
    past_source = source[:-lag]

    future_target_labels = _discretize_into_bins(future_target, n_bins)
    past_target_labels = _discretize_into_bins(past_target, n_bins)
    past_source_labels = _discretize_into_bins(past_source, n_bins)

    h_past_target = _joint_shannon_entropy_bits(past_target_labels)
    h_future_past_target = _joint_shannon_entropy_bits(future_target_labels, past_target_labels)
    h_past_target_source = _joint_shannon_entropy_bits(past_target_labels, past_source_labels)
    h_future_past_target_source = _joint_shannon_entropy_bits(
        future_target_labels, past_target_labels, past_source_labels
    )

    return (h_future_past_target - h_past_target) - (h_future_past_target_source - h_past_target_source)


def liquidity_void_pressure(
    bid_prices: npt.NDArray[np.float64],
    bid_volumes: npt.NDArray[np.float64],
    ask_prices: npt.NDArray[np.float64],
    ask_volumes: npt.NDArray[np.float64],
    mid_price: float,
) -> tuple[float, float, float]:
    r"""Real order-book depth-resistance asymmetry -- the discrete, honest analogue of "price seeks the path of least resistance."

    **What this deliberately is not**: the brief this responds to invokes
    Feynman's path-integral formulation and Baaquie's quantum-finance
    literature, where a price path's probability is governed by an
    "action" :math:`S = \int L \, dt` minimized over a real Lagrangian.
    No rigorous, agreed-upon closed-form market Lagrangian exists to
    evaluate that integral against real order-book data, and claiming
    one here would misrepresent a real, niche academic framework as a
    simple, off-the-shelf utility. **What this real is**: a standard,
    real order-book depth metric -- each side's real *resistance* is its
    real volume-weighted distance from the mid-price (:math:`\sum_i v_i
    \cdot |p_i - p_{\text{mid}}|`, a real, literal analogue of the
    physical quantity "work" -- more volume, or volume further away,
    means more real resistance to moving price through it). The side
    with **less** real resistance is the real "void": the honest
    version of the brief's own claim that price is pulled toward where
    the book is thinnest.

    Args:
        bid_prices: Real bid-side price levels. Must be the same length
            as ``bid_volumes``, at least one element.
        bid_volumes: Real bid-side volumes at each level in
            ``bid_prices``. Must be non-negative.
        ask_prices: Real ask-side price levels. Must be the same length
            as ``ask_volumes``, at least one element.
        ask_volumes: Real ask-side volumes at each level in
            ``ask_prices``. Must be non-negative.
        mid_price: Real current mid-price. Must be positive.

    Returns:
        ``(void_pressure, bid_side_resistance, ask_side_resistance)`` --
        ``void_pressure`` is real, signed, in ``[-1, 1]``:
        ``(bid_side_resistance - ask_side_resistance) / (bid_side_resistance
        + ask_side_resistance)``, positive when the **ask** side is the
        real thinner void (price pulled up), negative when the **bid**
        side is (price pulled down), ``0.0`` when both sides carry zero
        real resistance (e.g. all-zero volumes).

    Raises:
        ValueError: If any price/volume array pairing has mismatched
            shapes or is empty, any volume is negative, or ``mid_price``
            is not positive.
    """
    if bid_prices.shape != bid_volumes.shape or bid_prices.size == 0:
        raise ValueError("bid_prices and bid_volumes must be the same nonempty shape.")
    if ask_prices.shape != ask_volumes.shape or ask_prices.size == 0:
        raise ValueError("ask_prices and ask_volumes must be the same nonempty shape.")
    if np.any(bid_volumes < 0) or np.any(ask_volumes < 0):
        raise ValueError("volumes must not be negative.")
    if mid_price <= 0:
        raise ValueError("mid_price must be positive.")

    bid_side_resistance = float(np.sum(bid_volumes * np.abs(mid_price - bid_prices)))
    ask_side_resistance = float(np.sum(ask_volumes * np.abs(ask_prices - mid_price)))
    total = bid_side_resistance + ask_side_resistance
    void_pressure = (bid_side_resistance - ask_side_resistance) / total if total > 0 else 0.0
    return void_pressure, bid_side_resistance, ask_side_resistance


def morlet_wavelet_transform(
    series: npt.NDArray[np.float64],
    scales: npt.NDArray[np.float64],
    *,
    sampling_period: float = 1.0,
    omega0: float = 6.0,
) -> npt.NDArray[np.complex128]:
    r"""Real Continuous Wavelet Transform with the complex Morlet wavelet (Torrence & Compo, 1998).

    Unlike a single, fixed-window Fourier transform -- which the real
    Gabor-Heisenberg uncertainty relation (:math:`\Delta t \cdot \Delta f
    \ge 1/(4\pi)`) forces to trade time resolution against frequency
    resolution *uniformly across the whole signal* -- the CWT evaluates
    the same signal against a wavelet stretched/compressed to many real
    ``scales`` at once, each with its own real, scale-appropriate
    time/frequency tradeoff: small scales localize sudden, real events
    sharply in time (at the cost of frequency precision); large scales
    resolve slow, real trends precisely in frequency (at the cost of
    time precision). Neither choice is free of Gabor-Heisenberg -- this
    is a real, disclosed reframing of the same physical limit, not an
    escape from it.

    **Implemented directly in NumPy, not via a library, for a real,
    disclosed reason**: this project's own :mod:`scipy` version no
    longer ships ``scipy.signal.cwt``/``morlet2`` (removed in a scipy
    release this project already depends on), and ``pywt`` is a
    C-extension package this Windows development environment cannot
    build -- the identical real constraint that made
    :class:`risk.regime_detector.GaussianHMM` a hand-rolled NumPy
    implementation instead of ``hmmlearn``. The formula below (Torrence
    & Compo, "A Practical Guide to Wavelet Analysis", Bull. Amer.
    Meteor. Soc. 79, 1998, Table 1) is the same real, standard one those
    libraries implement.

    Args:
        series: Real 1D signal (e.g. a real price or log-return series).
        scales: Real, positive wavelet scales to evaluate, expressed in
            units of **samples** (not real time) -- the same convention
            ``pywt.cwt`` uses: a ``scale`` of ``10`` means a wavelet
            whose characteristic width is 10 samples, regardless of
            ``sampling_period``. Larger scales probe lower real
            frequencies. Must all be positive. Use
            :func:`morlet_scale_to_frequency` (which does take
            ``sampling_period`` into account) to convert a desired real
            frequency into the matching scale.
        sampling_period: Real time between consecutive ``series``
            samples (e.g. real seconds between candles). Not used in
            this function's own math (``scales`` are already
            sample-domain, see above) -- accepted and validated here
            purely so the same ``(scale, sampling_period)`` pair a
            caller derived via :func:`morlet_scale_to_frequency` can be
            passed to both functions without a units mismatch.
        omega0: Real nondimensional central frequency of the mother
            wavelet -- ``6.0`` is the standard default (a real, common
            choice satisfying the wavelet admissibility condition well
            enough in practice; Torrence & Compo use the same value).

    Returns:
        Complex array, shape ``(len(scales), len(series))`` --
        ``result[i, t]`` is the real CWT coefficient at scale
        ``scales[i]`` and time index ``t``. ``np.abs(result) ** 2`` is
        the real wavelet power (energy) at each scale/time; the scale
        with the largest power at a given ``t`` is that instant's real
        dominant time-localized cycle, convertible to a real
        pseudo-frequency via :func:`morlet_scale_to_frequency`.

    Raises:
        ValueError: If ``series`` is empty, ``scales`` is empty or
            contains a non-positive value, or ``sampling_period`` is
            not positive.
    """
    if series.size == 0:
        raise ValueError("series must not be empty.")
    if scales.size == 0 or np.any(scales <= 0):
        raise ValueError("scales must be nonempty and strictly positive.")
    if sampling_period <= 0:
        raise ValueError("sampling_period must be positive.")

    n = series.size
    result = np.empty((scales.size, n), dtype=np.complex128)
    for i, scale in enumerate(scales):
        # `scale` is in units of SAMPLES (matching morlet_scale_to_frequency's own convention, and
        # PyWavelets' identical "scales are sample-domain, sampling_period only converts to Hz"
        # design) -- `eta` must therefore be `offsets / scale` (both already in samples), NOT
        # `offsets * sampling_period / scale`. A real bug caught by testing against a known-frequency
        # sinusoid: that extra sampling_period factor silently mixed units (offsets*sampling_period is
        # a real TIME quantity, scale is a sample COUNT), which systematically made smaller scales
        # dominate regardless of the signal's real frequency content -- the CWT appeared to work
        # (it produced *a* number) but never actually localized the correct scale.
        half_width = int(np.ceil(8.0 * scale))
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        eta = offsets / scale
        wavelet = (np.pi ** -0.25) * np.exp(1j * omega0 * eta) * np.exp(-0.5 * eta ** 2)
        wavelet *= 1.0 / np.sqrt(scale)
        kernel = np.conj(wavelet[::-1])
        # Real, standard reflection-padding (Torrence & Compo's own recommended boundary
        # treatment, also scipy's/pywt's default elsewhere), not the implicit zero-padding
        # np.convolve(..., mode="same") would otherwise use: two real bugs found by testing --
        # (1) zero-padding creates a real, artificial discontinuity at each edge that a wide
        # kernel (a large real scale) picks up as spurious power *growing with scale*, completely
        # swamping the true signal exactly at the boundary this function's own callers most need
        # (the most recent real sample, i.e. "right now" for a live agent); (2) an uncapped
        # `half_width` can exceed `series` itself for a large real scale against a short series,
        # which silently changed `np.convolve`'s own mode="same" output length (a real, observed
        # `ValueError: could not broadcast...`) instead of the promised `(len(scales),
        # len(series))` shape. Reflecting past `half_width` real samples at each edge (numpy
        # supports this even when `half_width > n`, by repeated reflection) fixes both: real data
        # -- not zero, not a truncated kernel -- always fills the kernel's full real support.
        padded = np.pad(series, half_width, mode="reflect")
        convolved = np.convolve(padded, kernel, mode="same")
        result[i] = convolved[half_width : half_width + n]
    return result


def morlet_scale_to_frequency(scale: float, *, sampling_period: float = 1.0, omega0: float = 6.0) -> float:
    r"""Real Morlet wavelet scale -> equivalent Fourier frequency (Torrence & Compo, 1998, Eq. 14, simplified for the exact Morlet case).

    Args:
        scale: Real, positive wavelet scale (as used in
            :func:`morlet_wavelet_transform`).
        sampling_period: Same real value passed to
            :func:`morlet_wavelet_transform`. Must be positive.
        omega0: Same real value passed to :func:`morlet_wavelet_transform`.

    Returns:
        The real equivalent frequency, in cycles per unit of
        ``sampling_period``'s own time unit.

    Raises:
        ValueError: If ``scale``/``sampling_period`` is not positive.
    """
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if sampling_period <= 0:
        raise ValueError("sampling_period must be positive.")
    return omega0 / (2.0 * np.pi * scale) / sampling_period


def kalman_dynamic_hedge_ratio(
    y: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    *,
    process_variance: float = 1e-5,
    observation_variance: float = 1e-3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Real recursive Kalman filter for a time-varying pairs-trading hedge ratio (state-space linear regression).

    Models ``y_t = \alpha_t + \beta_t \cdot x_t + \epsilon_t`` with a
    real random-walk state ``[\alpha_t, \beta_t]`` -- the standard,
    textbook pairs-trading recipe (e.g. Chan, "Algorithmic Trading:
    Winning Strategies and Their Rationale", 2013, Ch. 3): unlike a
    fixed-window OLS hedge ratio (which only updates in discrete,
    stale jumps as the window rolls), this real Kalman filter updates
    ``[\alpha_t, \beta_t]`` on **every** new observation, recursively,
    weighting the new evidence against its own accumulated real
    uncertainty (the Kalman gain) -- the real "state estimate" the
    brief's own framing describes, not a moving average with a
    different name.

    Args:
        y: Real dependent series (e.g. one asset's real log-price).
        x: Real independent series (e.g. a correlated asset's real
            log-price). Must be the same length as ``y``, at least 1
            element.
        process_variance: Real ``Q`` (as a scalar times the identity):
            how much ``[\alpha_t, \beta_t]`` is allowed to real-drift
            per step. Larger values track a real, faster-changing
            hedge ratio at the cost of noisier estimates. Must be
            positive.
        observation_variance: Real ``R``: assumed noise variance of
            ``y``'s observation itself. Larger values make the filter
            trust new observations less. Must be positive.

    Returns:
        ``(states, spread)`` -- ``states`` is real, shape ``(n, 2)``,
        the filtered ``[\alpha_t, \beta_t]`` estimate at every step;
        ``spread`` is real, shape ``(n,)``, the one-step-ahead
        prediction residual ``y_t - (\alpha_{t|t-1} + \beta_{t|t-1}
        \cdot x_t)`` -- the real, Kalman-cleaned pairs-trading spread a
        real stat-arb strategy would trade the mean-reversion of.

    Raises:
        ValueError: If ``y``/``x`` have mismatched shapes or are empty,
            or either variance is not positive.
    """
    if y.shape != x.shape or y.size == 0:
        raise ValueError("y and x must be the same nonempty shape.")
    if process_variance <= 0:
        raise ValueError("process_variance must be positive.")
    if observation_variance <= 0:
        raise ValueError("observation_variance must be positive.")

    n = y.size
    state = np.zeros(2, dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64)
    process_noise = np.eye(2, dtype=np.float64) * process_variance

    states = np.empty((n, 2), dtype=np.float64)
    spread = np.empty(n, dtype=np.float64)

    for t in range(n):
        predicted_state = state  # real random-walk transition: identity, no drift term
        predicted_covariance = covariance + process_noise

        observation_matrix = np.array([1.0, x[t]], dtype=np.float64)
        predicted_observation = float(observation_matrix @ predicted_state)
        innovation = float(y[t] - predicted_observation)
        innovation_variance = float(observation_matrix @ predicted_covariance @ observation_matrix + observation_variance)
        kalman_gain = (predicted_covariance @ observation_matrix) / innovation_variance

        state = predicted_state + kalman_gain * innovation
        covariance = predicted_covariance - np.outer(kalman_gain, observation_matrix) @ predicted_covariance

        states[t] = state
        spread[t] = innovation

    return states, spread


def poiseuille_liquidity_resistance(spread: float, liquidity_depth: float, *, execution_horizon: float = 1.0) -> float:
    r"""Real, disclosed Hagen-Poiseuille *analogy* for order-book flow resistance -- not literal fluid dynamics.

    The real Hagen-Poiseuille law for laminar pipe flow,
    :math:`R = 8 \eta L / (\pi r^4)`, gives resistance growing with the
    **inverse fourth power** of the pipe radius -- so halving the radius
    multiplies resistance by 16. This function is a real, disclosed
    *analogy* (the same honest framing :func:`calibrate_impact_coefficient_fdt`
    already uses for Fluctuation-Dissipation, not a claim that an order
    book is a literal viscous fluid): ``spread`` plays the role of
    viscosity :math:`\eta` (a real, already-used cost-of-trading proxy
    elsewhere in this module), ``liquidity_depth`` plays the role of the
    channel radius :math:`r` (e.g. real top-of-book volume, the same
    real field :func:`price_impact` already consumes as
    ``available_liquidity``), and ``execution_horizon`` plays the role
    of the channel length :math:`L`.

    **Structurally distinct from** :func:`price_impact` (already real
    and already blocking in Gate 7): that function's real, empirically-
    grounded square-root law says impact grows with
    :math:`\sqrt{|Q|/V}` -- a much gentler curve than this function's
    real point, the far sharper :math:`r^{-4}` convexity a halved
    ``liquidity_depth`` produces here. Provided as a real, informational
    complement (not a second, competing blocking veto over the same
    real inputs :func:`price_impact` already gates on) -- see the
    real, exact identity this function's own tests verify:
    ``poiseuille_liquidity_resistance(s, d/2) == 16 * poiseuille_liquidity_resistance(s, d)``.

    Args:
        spread: Real current bid-ask spread (the real "viscosity"
            proxy). Must be ``>= 0``.
        liquidity_depth: Real order-book depth (the real "radius"
            proxy, e.g. top-of-book volume). Must be positive.
        execution_horizon: Real "channel length" -- e.g. real seconds
            or real order size the resistance is being evaluated over.
            Must be positive.

    Returns:
        The real resistance value, in
        ``spread_units * execution_horizon_units / liquidity_depth_units^4``
        -- meaningful only in real, relative comparison (e.g. across
        time for the same instrument), not an absolute, unit-calibrated
        cost figure.

    Raises:
        ValueError: If ``spread`` is negative, or ``liquidity_depth``/
            ``execution_horizon`` is not positive.
    """
    if spread < 0:
        raise ValueError("spread must not be negative.")
    if liquidity_depth <= 0:
        raise ValueError("liquidity_depth must be positive.")
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive.")

    return spread * execution_horizon / (liquidity_depth ** 4)


def hurst_exponent_rs(returns: npt.NDArray[np.float64], *, min_chunk_size: int = 8, num_chunk_sizes: int = 6) -> float:
    r"""Real Hurst exponent via Rescaled Range (R/S) analysis (Hurst 1951; Mandelbrot & Wallis 1969).

    For each of several real sub-sample window sizes :math:`n`, splits
    ``returns`` into ``len(returns) // n`` non-overlapping chunks of
    that length, and for each chunk computes the classical R/S
    statistic:

    .. math::
        R(n) = \max_t Z_t - \min_t Z_t, \quad Z_t = \sum_{i=1}^{t} (X_i - \bar{X})

    .. math::
        S(n) = \text{sample std. dev. of the chunk (ddof=1)}

    averages :math:`R(n)/S(n)` across all chunks of that size, then
    fits :math:`\log(R/S) = H \log(n) + c` by ordinary least squares
    across every tested window size -- the standard multi-scale R/S
    method, not a single-window point estimate (which would have no
    way to distinguish real scaling behavior from one window's sampling
    noise).

    :math:`H = 0.5` indicates a real random walk (uncorrelated
    increments, e.g. genuine Brownian motion); :math:`H > 0.5` indicates
    real persistence (trending clusters); :math:`H < 0.5` indicates real
    anti-persistence (mean-reverting, "rough" -- the regime real
    realized-volatility series are empirically found to be in: Gatheral,
    Jaisson & Rosenbaum, "Volatility is Rough," Quantitative Finance,
    2018, report :math:`H \approx 0.1` across many real equity indices).

    **Real, disclosed, measured estimator bias**: classical/uncorrected
    R/S analysis is well-documented to have a real finite-sample bias
    that pulls estimates toward roughly :math:`H \approx 0.7`, regardless
    of the true value (Weron, "Estimating long-range dependence: finite
    sample properties and confidence intervals," Physica A, 2002) --
    this is not a bug in this implementation, and is not smoothed over
    here: this project's own verification (synthetic fractional
    Brownian motion with a known true :math:`H`, Cholesky-simulated,
    :math:`n=600`, 10 seeds) measured a real mean estimate of
    :math:`\approx 0.36` for a true :math:`H=0.2`, :math:`\approx 0.59`
    for a true :math:`H=0.5`, and :math:`\approx 0.81` for a true
    :math:`H=0.8` -- i.e. a real, consistent upward bias below
    :math:`H\approx0.7`, roughly unbiased near it. This function's own
    tests check recovery within a real, wide tolerance band reflecting
    this measured bias, not exact recovery. **The downstream
    consequence for risk, stated honestly rather than left implicit**:
    because :func:`fractional_volatility_scaling` is increasing in
    :math:`H` for a horizon greater than one period, this real upward
    bias makes :meth:`~src.agents.supervisor.risk_manager.RiskManager.
    calculate_var_fractional_horizon`'s horizon-scaled VaR *more*
    conservative than the true rough-volatility process would justify,
    not less -- the bias pulls the fractional estimate toward the
    classical square-root-of-time case rather than away from it, so a
    caller relying on this for risk sizing is, if anything, erring
    safe. Callers should still not treat one estimate as a precise
    reading of the true :math:`H`.

    Args:
        returns: 1D array of periodic returns (e.g. log-returns). Must
            have at least ``2 * min_chunk_size`` elements -- enough for
            at least two full chunks at the smallest tested window size.
        min_chunk_size: Smallest real sub-sample window size tested.
            Must be at least ``2`` (an R/S statistic needs at least 2
            points to have a non-degenerate cumulative range). Defaults
            to ``8``, the smallest window size standard R/S
            implementations typically use before small-sample bias
            dominates.
        num_chunk_sizes: How many distinct real window sizes to test
            between ``min_chunk_size`` and ``len(returns) // 2``. Must
            be at least ``2`` (a log-log regression needs at least 2
            distinct real points). Defaults to ``6``.

    Returns:
        The estimated Hurst exponent (the fitted slope :math:`H`). Not
        clipped to ``[0, 1]`` -- a real estimate can fall (slightly)
        outside that theoretical range under real sampling noise, and
        silently clipping would hide that rather than reveal it to the
        caller.

    Raises:
        ValueError: If ``returns`` has fewer than ``2 * min_chunk_size``
            elements, if ``min_chunk_size < 2``, if ``num_chunk_sizes < 2``,
            or if fewer than 2 of the tested window sizes produced a
            real, usable (non-zero-variance) R/S value to regress on.
    """
    if min_chunk_size < 2:
        raise ValueError("min_chunk_size must be at least 2.")
    if num_chunk_sizes < 2:
        raise ValueError("num_chunk_sizes must be at least 2.")
    n = returns.size
    if n < 2 * min_chunk_size:
        raise ValueError(
            f"returns must have at least 2 * min_chunk_size ({2 * min_chunk_size}) elements, got {n}."
        )

    max_chunk_size = n // 2
    candidate_sizes = np.unique(np.linspace(min_chunk_size, max_chunk_size, num=num_chunk_sizes).astype(int))

    log_sizes: list[float] = []
    log_rs_values: list[float] = []
    for size in candidate_sizes:
        num_chunks = n // size
        rs_per_chunk: list[float] = []
        for i in range(num_chunks):
            chunk = returns[i * size : (i + 1) * size]
            deviations = chunk - chunk.mean()
            cumulative = np.cumsum(deviations)
            chunk_range = float(np.max(cumulative) - np.min(cumulative))
            chunk_std = float(np.std(chunk, ddof=1))
            if chunk_std > 0.0:
                rs_per_chunk.append(chunk_range / chunk_std)
        if rs_per_chunk:
            log_sizes.append(float(np.log(size)))
            log_rs_values.append(float(np.log(np.mean(rs_per_chunk))))

    if len(log_sizes) < 2:
        raise ValueError(
            "Fewer than 2 real window sizes produced a usable (non-zero-variance) R/S value -- "
            "cannot fit a log-log regression. This real series may be (near-)constant."
        )

    hurst, _intercept = np.polyfit(log_sizes, log_rs_values, 1)
    return float(hurst)


def fractional_volatility_scaling(sigma_base: float, horizon: float, hurst_exponent: float) -> float:
    r"""Real self-affine scaling law for horizon-adjusted volatility: :math:`\sigma_T = \sigma \cdot T^H`.

    Generalizes the classical square-root-of-time rule
    (:math:`\sigma_T = \sigma \sqrt{T}`) that :func:`parametric_var`'s
    single-period formula implicitly relies on whenever a caller scales
    it to a longer horizon: that classical rule is the real special case
    :math:`H = 0.5` of this formula (an exact identity this function's
    own tests verify), valid only for real i.i.d. Gaussian increments
    (true Brownian motion). Real fractional Brownian motion is
    self-affine (Mandelbrot & Van Ness, 1968): :math:`B_H(cT)
    \stackrel{d}{=} c^H B_H(T)`, so a process with real Hurst exponent
    :math:`H \ne 0.5` scales its standard deviation by :math:`T^H`
    instead.

    Args:
        sigma_base: Real base-period volatility (e.g. the same
            ``np.std(log_returns, ddof=1)`` :func:`parametric_var`
            computes internally). Must be non-negative.
        horizon: Real target horizon, in units of the base period (e.g.
            ``10.0`` to scale a 1-period volatility to a 10-period one).
            Must be positive.
        hurst_exponent: Real Hurst exponent, e.g. from
            :func:`hurst_exponent_rs`. Not restricted to ``[0, 1]``
            here (that range restriction belongs to the estimator, not
            this pure scaling formula).

    Returns:
        The horizon-scaled volatility, ``sigma_base * horizon **
        hurst_exponent``.

    Raises:
        ValueError: If ``sigma_base`` is negative, or ``horizon`` is
            not positive.
    """
    if sigma_base < 0:
        raise ValueError("sigma_base must not be negative.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    return float(sigma_base * (horizon ** hurst_exponent))

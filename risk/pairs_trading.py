"""Stat-Arb Screening Layer: which real pairs are cointegrated, and what the spread is saying right now.

**The real, confirmed gap this closes**: this project already has both
of pairs trading's other two real building blocks --
:func:`~core.quant_math.kalman_dynamic_hedge_ratio` (a time-varying
hedge ratio, wired into :meth:`~src.agents.execution.agent.
ExecutionAgent.compute_kalman_pairs_spread`) and
:func:`~core.quant_math.ornstein_uhlenbeck_mean_reversion`
(mean-reversion speed/half-life) -- but
:meth:`~src.agents.execution.agent.ExecutionAgent.
compute_kalman_pairs_spread`'s own docstring already states the real
reason neither is enough on its own: "this project has no existing
stat-arb/pairs-trading execution agent" to consume them, and neither
function answers *which* instruments should be paired in the first
place. Two merely-correlated series can drift apart forever; only a
real cointegration relationship (:func:`~core.quant_math.
engle_granger_cointegration_test`) gives mean-reversion a structural
reason to hold. This module is that missing screening + signal layer --
**not** a new autonomous execution agent, the same real, disclosed
scope boundary ``compute_kalman_pairs_spread`` already drew, still open
after this addition: :func:`pairs_trading_signal` below returns a real
recommendation, it never places an order, checks inventory, or talks to
a broker (this project's established "pure decision logic, network/
execution wiring is the caller's job" pattern -- see
:mod:`execution.predator_honeypot`, :mod:`risk.regime_detector`).

**Screening pipeline** (:func:`screen_cointegrated_pairs`):

1. Real log-returns, correlation matrix, then
   :func:`~core.quant_math.denoise_correlation_matrix_mp` (Random
   Matrix Theory spectral cleaning already built for this project's
   portfolio-risk code) -- shortlists candidate pairs by real,
   noise-filtered correlation instead of testing every possible pair
   for cointegration (which would be both slow and, at any fixed
   significance level, guarantee some real false positives purely from
   multiple testing across a large universe).
2. :func:`~core.quant_math.engle_granger_cointegration_test` on each
   shortlisted candidate's real log-price history -- confirms real
   cointegration, not just correlation.
3. :func:`~core.quant_math.ornstein_uhlenbeck_mean_reversion` on the
   confirmed pair's real static residual spread -- the real half-life a
   trader needs to size a holding-period expectation, not just a
   binary cointegrated/not-cointegrated flag.

**Real, disclosed simplification**: candidate shortlisting uses the
static (single, whole-window) denoised correlation, not a rolling one
-- a pair whose relationship is itself time-varying enough to need
rolling re-screening is a real, further extension, not built here (this
module's own real job is deciding *whether* a pair is a real candidate
at all; :func:`~core.quant_math.kalman_dynamic_hedge_ratio` already
handles the *hedge ratio itself* varying through time once a pair has
been confirmed).
"""

from __future__ import annotations

from enum import Enum
from itertools import combinations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.quant_math import (
    denoise_correlation_matrix_mp,
    engle_granger_cointegration_test,
    ornstein_uhlenbeck_mean_reversion,
)

_log = get_logger(__name__)


class CointegratedPair(BaseModel):
    """One real, confirmed cointegrated pair -- see :func:`screen_cointegrated_pairs`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol_a: str
    symbol_b: str
    denoised_correlation: float = Field(ge=-1.0, le=1.0)
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    critical_value: float
    half_life: float = Field(gt=0.0)


def screen_cointegrated_pairs(
    log_prices: dict[str, npt.NDArray[np.float64]],
    *,
    correlation_threshold: float = 0.6,
    adf_lags: int = 1,
    significance: str = "5%",
) -> list[CointegratedPair]:
    """Real cointegration screen over every pair in ``log_prices`` -- see module docstring for the 3-step pipeline.

    Args:
        log_prices: Real, chronologically-ordered log-price history per
            symbol. Series of differing lengths are truncated to their
            common trailing length (the most recent shared window) --
            same real "align by trailing length" convention as
            :meth:`~src.agents.execution.agent.ExecutionAgent.
            compute_kalman_pairs_spread`.
        correlation_threshold: Minimum absolute *denoised* correlation
            (see module docstring) a pair must clear to even be tested
            for cointegration. Must be in ``(0, 1]``.
        adf_lags: Forwarded to :func:`~core.quant_math.
            engle_granger_cointegration_test`.
        significance: Forwarded to :func:`~core.quant_math.
            engle_granger_cointegration_test`.

    Returns:
        Every real pair that cleared both the correlation shortlist and
        the Engle-Granger cointegration test, as
        :class:`CointegratedPair`, sorted by ADF statistic ascending
        (the most strongly cointegrated pair first). A pair whose
        confirmed residual spread shows no real mean reversion at all
        (:func:`~core.quant_math.ornstein_uhlenbeck_mean_reversion`'s
        own ``theta <= 0`` case, ``half_life = inf``) is dropped here --
        Engle-Granger's residual ADF test and this OLS-on-levels
        half-life estimate are two different real regressions on the
        same series and can, on real finite samples, disagree at the
        margin; a pair with no finite half-life has no real, actionable
        holding-period expectation regardless of what the ADF test
        alone concluded, so it is excluded rather than surfaced with a
        fabricated ``half_life``.

    Raises:
        ValueError: If ``log_prices`` has fewer than 2 symbols, any
            series is empty, or ``correlation_threshold`` is not in
            ``(0, 1]``.
    """
    if len(log_prices) < 2:
        raise ValueError("log_prices must contain at least 2 symbols to screen any pair.")
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold must be in (0, 1].")

    symbols = list(log_prices.keys())
    common_length = min(series.size for series in log_prices.values())
    if common_length < 3:
        raise ValueError("Every symbol needs at least 3 real observations after aligning to the common trailing length.")

    aligned = np.column_stack([log_prices[symbol][-common_length:] for symbol in symbols])
    returns = np.diff(aligned, axis=0)
    correlation_matrix = np.corrcoef(returns, rowvar=False)
    np.fill_diagonal(correlation_matrix, 1.0)  # kill float-noise diagonal drift, same precedent as hierarchical_risk_parity_weights
    denoised = denoise_correlation_matrix_mp(correlation_matrix, n_samples=returns.shape[0])

    results: list[CointegratedPair] = []
    for i, j in combinations(range(len(symbols)), 2):
        denoised_correlation = float(denoised[i, j])
        if abs(denoised_correlation) < correlation_threshold:
            continue

        symbol_a, symbol_b = symbols[i], symbols[j]
        hedge_ratio, intercept, residuals, adf_statistic, critical_value, is_cointegrated = engle_granger_cointegration_test(
            aligned[:, i], aligned[:, j], adf_lags=adf_lags, significance=significance,
        )
        if not is_cointegrated:
            continue

        theta, _mu, half_life = ornstein_uhlenbeck_mean_reversion(residuals)
        if theta <= 0.0 or not np.isfinite(half_life):
            _log.info(
                "pairs_screen_cointegrated_but_no_finite_half_life", symbol_a=symbol_a, symbol_b=symbol_b,
                adf_statistic=adf_statistic,
            )
            continue

        results.append(
            CointegratedPair(
                symbol_a=symbol_a, symbol_b=symbol_b, denoised_correlation=denoised_correlation,
                hedge_ratio=hedge_ratio, intercept=intercept, adf_statistic=adf_statistic,
                critical_value=critical_value, half_life=half_life,
            )
        )

    results.sort(key=lambda pair: pair.adf_statistic)
    return results


class PairSignal(str, Enum):
    LONG_SPREAD = "LONG_SPREAD"  # spread abnormally low -- buy symbol_a, sell hedge_ratio * symbol_b
    SHORT_SPREAD = "SHORT_SPREAD"  # spread abnormally high -- sell symbol_a, buy hedge_ratio * symbol_b
    EXIT = "EXIT"  # spread has reverted close to its own mean -- close any open pairs position
    HOLD = "HOLD"  # neither an entry nor an exit condition -- no real action this cycle


class SpreadSignal(BaseModel):
    """One real, current stat-arb signal for one already-confirmed cointegrated pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_spread: float
    z_score: float
    signal: PairSignal


def pairs_trading_signal(
    spread_history: npt.NDArray[np.float64], *, entry_z: float = 2.0, exit_z: float = 0.5,
) -> SpreadSignal:
    r"""Real entry/exit signal from a real spread's own z-score against its own trailing mean/std.

    Intended to be called with :meth:`~src.agents.execution.agent.
    ExecutionAgent.compute_kalman_pairs_spread`'s own real, live,
    Kalman-filtered ``spread`` array (already the real, cleaned
    pairs-trading spread that function's own docstring describes) --
    this function does not compute a spread itself, it only turns one
    the caller already has into a real trading decision.

    ``z_score = (spread_history[-1] - mean(spread_history)) /
    std(spread_history)``, i.e. how many real standard deviations the
    *current* spread sits from its own trailing average.

    Args:
        spread_history: Real, chronologically-ordered spread values,
            most recent last. Must have at least 2 elements and a
            nonzero standard deviation (a perfectly constant spread has
            no real z-score to compute).
        entry_z: :math:`|z| \geq` this opens a position betting on
            reversion. Must be positive.
        exit_z: :math:`|z| \leq` this closes an open position (the
            spread has reverted). Must be ``>= 0`` and ``< entry_z`` --
            an ``exit_z`` at or above ``entry_z`` would mean every real
            entry signal is also, contradictorily, an exit signal.

    Returns:
        A real :class:`SpreadSignal`. Between ``exit_z`` and
        ``entry_z`` the real signal is :attr:`PairSignal.HOLD` -- a
        real, deliberate dead zone, not a forced choice between
        entering deeper or exiting early on a spread that has neither
        clearly reverted nor clearly extended.

    Raises:
        ValueError: If ``spread_history`` has fewer than 2 elements, a
            zero (or near-zero, see the function's own floor) standard
            deviation, or ``entry_z``/``exit_z`` violate the constraints
            above.
    """
    if spread_history.size < 2:
        raise ValueError("spread_history must have at least 2 elements to compute a real standard deviation.")
    if entry_z <= 0.0:
        raise ValueError("entry_z must be positive.")
    if exit_z < 0.0 or exit_z >= entry_z:
        raise ValueError("exit_z must be >= 0 and strictly less than entry_z.")

    mean = float(np.mean(spread_history))
    std = float(np.std(spread_history))
    if std < 1e-12:
        raise ValueError("spread_history has a real, effectively-zero standard deviation -- no real z-score to compute.")

    current_spread = float(spread_history[-1])
    z_score = (current_spread - mean) / std

    if z_score >= entry_z:
        signal = PairSignal.SHORT_SPREAD
    elif z_score <= -entry_z:
        signal = PairSignal.LONG_SPREAD
    elif abs(z_score) <= exit_z:
        signal = PairSignal.EXIT
    else:
        signal = PairSignal.HOLD

    return SpreadSignal(current_spread=current_spread, z_score=z_score, signal=signal)

"""Pure-NumPy quantitative finance primitives: Kelly sizing and parametric VaR.

Every function here is a stateless, side-effect-free calculation over
plain NumPy arrays/floats -- no I/O, no Pydantic models, no logging.
This is deliberate: it keeps the math independently testable and reusable
from anywhere (a live agent, an offline research notebook, a backtest),
and is why :class:`~src.agents.supervisor.risk_manager.RiskManager`
delegates its VaR calculation to :func:`parametric_var` here rather than
computing it inline.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

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

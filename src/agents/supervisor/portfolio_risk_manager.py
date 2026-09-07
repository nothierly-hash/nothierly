"""Correlation-aware portfolio risk: the Global Portfolio Manager (Phase 3, "Le Stratège").

:class:`~src.agents.supervisor.risk_manager.RiskManager` already enforces
a flat ceiling on *total* notional exposure, instrument-agnostic. That
misses a real failure mode: two instruments that are highly correlated
(e.g. ``BTC/USDT`` and ``ETH/USDT`` tend to move together) are, from a
risk standpoint, close to *the same bet twice* -- a flat exposure ceiling
that has room for both individually can still leave the portfolio far
more concentrated than its notional total suggests.

:class:`PortfolioRiskParityManager` adds two capabilities on top of that
existing ceiling, never replacing it:

1. :meth:`check_correlated_exposure` -- a **second, independent** gate:
   refuses a proposed position if it, combined with existing positions on
   instruments correlated above a threshold, would exceed a tighter
   exposure ceiling reserved for correlated clusters specifically.
2. :meth:`compute_risk_parity_weights` -- given a covariance matrix,
   returns capital weights where every instrument contributes *equally*
   to total portfolio risk (Equal Risk Contribution / ERC), rather than
   equal notional -- the standard "risk parity" allocation.

Pure, stateless-with-respect-to-positions in the same sense as
:class:`~src.agents.supervisor.risk_manager.RiskManager`: this class
tracks price history for its own correlation/covariance estimates, but
current positions/capital are always supplied by the caller, never owned
here.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_LOOKBACK_WINDOW = 100
_DEFAULT_CORRELATION_THRESHOLD = 0.7
_DEFAULT_CORRELATED_EXPOSURE_PCT = 0.15


class InsufficientPriceHistoryError(Exception):
    """Raised when a correlation/covariance estimate is requested without enough shared history."""


class CorrelatedExposureError(Exception):
    """Raised when a proposed position would breach the correlated-cluster exposure ceiling.

    Like :class:`~src.agents.supervisor.risk_manager.RiskLimitExceededError`,
    this is a hard stop: callers must treat it as blocking the order.
    """


class PortfolioRiskParityManager:
    """Tracks per-instrument price history; checks correlated exposure and computes ERC weights."""

    def __init__(
        self,
        *,
        lookback_window: int = _DEFAULT_LOOKBACK_WINDOW,
        correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
        correlated_exposure_pct: float = _DEFAULT_CORRELATED_EXPOSURE_PCT,
    ) -> None:
        """Configure the correlation model and the correlated-cluster ceiling.

        Args:
            lookback_window: Number of recent prices kept per instrument
                for the return/correlation/covariance estimates.
            correlation_threshold: Absolute Pearson correlation above
                which two instruments are treated as "the same bet" by
                :meth:`check_correlated_exposure`. Must be in ``(0, 1]``.
            correlated_exposure_pct: Maximum combined notional exposure
                across a correlated cluster, as a fraction of capital
                (e.g. ``0.15`` for 15%) -- deliberately tighter than
                :class:`~src.agents.supervisor.risk_manager.
                RiskManager`'s own (uncorrelated-blind) ceiling, since
                this one targets concentration risk specifically.

        Raises:
            ValueError: If ``lookback_window`` is too small to ever
                produce a return series, or either percentage argument
                is out of its valid range.
        """
        if lookback_window < 3:
            raise ValueError("lookback_window must be >= 3 (need at least 2 returns).")
        if not (0.0 < correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be in (0, 1].")
        if not (0.0 < correlated_exposure_pct <= 1.0):
            raise ValueError("correlated_exposure_pct must be in (0, 1].")

        self._lookback_window = lookback_window
        self._correlation_threshold = correlation_threshold
        self._correlated_exposure_pct = correlated_exposure_pct
        self._price_history: dict[str, deque[float]] = {}

    def record_price(self, instrument: str, price: float) -> None:
        """Feed ``instrument``'s latest price into its rolling history.

        Args:
            instrument: Which instrument this price is for.
            price: Latest observed price. Must be strictly positive.

        Raises:
            ValueError: If ``price`` is not positive.
        """
        if price <= 0:
            raise ValueError("price must be positive.")
        history = self._price_history.setdefault(instrument, deque(maxlen=self._lookback_window))
        history.append(price)

    def has_sufficient_history(self, instrument: str) -> bool:
        """Whether ``instrument`` has enough recorded prices to compute a return series."""
        return len(self._price_history.get(instrument, ())) >= 3

    def _log_returns(self, instrument: str) -> npt.NDArray[np.float64]:
        """Log-returns of ``instrument``'s recorded price history, oldest-recorded-first."""
        prices = np.array(self._price_history[instrument], dtype=np.float64)
        return np.diff(np.log(prices))

    def compute_correlation(self, instrument_a: str, instrument_b: str) -> float:
        """Pearson correlation between two instruments' recent log-returns.

        The two instruments' return series are compared over their
        shared most-recent window: the shorter of the two histories, so
        an instrument with less history than another simply narrows the
        comparison window rather than blocking it entirely.

        Args:
            instrument_a: First instrument. Must have been fed via
                :meth:`record_price` enough times (see
                :meth:`has_sufficient_history`).
            instrument_b: Second instrument, same requirement.

        Returns:
            Correlation in ``[-1, 1]``. Instruments correlate perfectly
            with themselves (``instrument_a == instrument_b`` always
            returns ``1.0``, computed the same way as any other pair).

        Raises:
            InsufficientPriceHistoryError: If either instrument lacks
                enough history for at least 2 overlapping returns.
        """
        if not self.has_sufficient_history(instrument_a):
            raise InsufficientPriceHistoryError(
                f"Not enough price history for {instrument_a!r} to compute correlation."
            )
        if not self.has_sufficient_history(instrument_b):
            raise InsufficientPriceHistoryError(
                f"Not enough price history for {instrument_b!r} to compute correlation."
            )

        returns_a = self._log_returns(instrument_a)
        returns_b = self._log_returns(instrument_b)
        window = min(returns_a.size, returns_b.size)
        if window < 2:
            raise InsufficientPriceHistoryError(
                f"{instrument_a!r} and {instrument_b!r} don't share enough overlapping "
                f"history yet (need >= 2 overlapping returns, have {window})."
            )

        # Most-recent `window` returns from each -- the shared, comparable
        # tail, regardless of which series has more total history.
        correlation_matrix = np.corrcoef(returns_a[-window:], returns_b[-window:])
        return float(correlation_matrix[0, 1])

    def check_correlated_exposure(
        self,
        instrument: str,
        proposed_notional: float,
        *,
        positions: dict[str, float],
        capital: float,
    ) -> None:
        """Block a proposed position if its correlated cluster would breach the ceiling.

        Sums ``proposed_notional`` with the notional of every *existing*
        position (from ``positions``) whose instrument's absolute
        correlation with ``instrument`` exceeds ``correlation_threshold``
        -- instruments without enough shared history to compute a
        correlation are conservatively treated as correlated (fails
        closed: an unknown relationship is not assumed to be safe).

        Args:
            instrument: The instrument a new position is being proposed
                for.
            proposed_notional: Notional value (absolute) of the proposed
                position.
            positions: Every other currently open position, as
                ``{instrument: absolute_notional}``. ``instrument``
                itself, if present (e.g. scaling into an existing
                position), is included in the correlated sum like any
                other entry.
            capital: Total portfolio capital the exposure percentage is
                measured against.

        Raises:
            CorrelatedExposureError: If the correlated cluster's total
                notional (existing + proposed) would exceed
                ``correlated_exposure_pct * capital``.
            ValueError: If ``proposed_notional`` is negative or
                ``capital`` is not positive.
        """
        if proposed_notional < 0:
            raise ValueError("proposed_notional must not be negative.")
        if capital <= 0:
            raise ValueError("capital must be positive.")

        correlated_notional = proposed_notional
        correlated_with: list[str] = []
        for other_instrument, other_notional in positions.items():
            if other_notional == 0.0:
                continue
            try:
                correlation = self.compute_correlation(instrument, other_instrument)
                is_correlated = abs(correlation) >= self._correlation_threshold
            except InsufficientPriceHistoryError:
                # Fails closed: an unknown relationship is not assumed safe.
                is_correlated = True
                correlation = float("nan")
            if is_correlated:
                correlated_notional += abs(other_notional)
                correlated_with.append(other_instrument)

        # No correlated peer found: this check has nothing to say about a
        # lone, uncorrelated position -- sizing/exposure for an isolated
        # position is RiskManager.check_exposure_limit's job, not this
        # correlation-specific ceiling's. Only a genuine correlated
        # cluster (this instrument plus >=1 correlated existing position)
        # is evaluated against correlated_exposure_pct.
        if not correlated_with:
            return

        max_correlated_exposure = self._correlated_exposure_pct * capital
        if correlated_notional > max_correlated_exposure:
            _log.warning(
                "correlated_exposure_limit_exceeded",
                instrument=instrument,
                correlated_with=correlated_with,
                correlated_notional=correlated_notional,
                max_correlated_exposure=max_correlated_exposure,
                correlated_exposure_pct=self._correlated_exposure_pct,
            )
            raise CorrelatedExposureError(
                f"Proposed {instrument} position would bring the correlated cluster "
                f"{[instrument, *correlated_with]} to {correlated_notional:.2f} notional, "
                f"exceeding the {self._correlated_exposure_pct:.1%} limit "
                f"({max_correlated_exposure:.2f}) on capital {capital:.2f}."
            )

    def compute_risk_parity_weights(
        self,
        instruments: list[str],
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
    ) -> dict[str, float]:
        """Equal Risk Contribution (ERC) capital weights across ``instruments``.

        Finds weights (summing to ``1.0``, each ``> 0``) such that every
        instrument contributes an equal share of total portfolio
        variance -- the standard risk-parity allocation, and a strict
        generalization of naive inverse-volatility weighting: the two
        agree exactly when every pairwise correlation is ``0``, and ERC
        additionally accounts for correlation when it isn't.

        Solved via a multiplicative fixed-point iteration (Choueifaty-
        style): each instrument's weight is scaled by the square root of
        its target-to-actual risk-contribution ratio, then renormalized,
        repeated until every contribution is within ``tolerance`` of
        equal or ``max_iterations`` is reached.

        Args:
            instruments: Which instruments to weight. Every one must
                have sufficient price history (see
                :meth:`has_sufficient_history`) against every other, for
                the covariance matrix this is computed from.
            max_iterations: Maximum fixed-point iterations.
            tolerance: Convergence threshold on the largest weight
                change between iterations.

        Returns:
            ``{instrument: weight}``, weights summing to ``1.0``.

        Raises:
            ValueError: If ``instruments`` has fewer than 2 entries.
            InsufficientPriceHistoryError: If any instrument lacks
                sufficient history, or a pair doesn't share enough
                overlapping history for the covariance estimate.
        """
        if len(instruments) < 2:
            raise ValueError("instruments must have at least 2 entries.")

        n = len(instruments)
        covariance_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    returns = self._log_returns(instruments[i])
                    if returns.size < 2:
                        raise InsufficientPriceHistoryError(
                            f"Not enough price history for {instruments[i]!r}."
                        )
                    variance = float(np.var(returns, ddof=1))
                else:
                    correlation = self.compute_correlation(instruments[i], instruments[j])
                    returns_i = self._log_returns(instruments[i])
                    returns_j = self._log_returns(instruments[j])
                    std_i = float(np.std(returns_i, ddof=1))
                    std_j = float(np.std(returns_j, ddof=1))
                    variance = correlation * std_i * std_j
                covariance_matrix[i, j] = variance
                covariance_matrix[j, i] = variance

        weights = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(max_iterations):
            portfolio_variance = float(weights @ covariance_matrix @ weights)
            marginal_contribution = covariance_matrix @ weights
            risk_contribution = weights * marginal_contribution
            target_contribution = portfolio_variance / n

            adjustment = np.sqrt(target_contribution / np.maximum(risk_contribution, 1e-18))
            new_weights = np.clip(weights * adjustment, 1e-12, None)
            new_weights /= new_weights.sum()

            if float(np.max(np.abs(new_weights - weights))) < tolerance:
                weights = new_weights
                break
            weights = new_weights

        return dict(zip(instruments, (float(w) for w in weights)))

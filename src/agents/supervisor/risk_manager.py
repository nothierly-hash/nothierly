"""Financial risk calculations: Value at Risk and exposure limits.

:class:`RiskManager` is a pure, stateless-with-respect-to-portfolio risk
calculation utility: it computes Value at Risk (VaR) from a rolling
window of order book mid-prices, and enforces a hard exposure ceiling.
It deliberately does *not* track positions or execute anything -- order
execution and portfolio/position tracking don't exist yet in this system
(see the chapter's scope), so ``current_exposure`` is always supplied by
the caller rather than owned here.

The actual VaR formula lives in :mod:`core.quant_math` (a dependency-free
math library reusable outside this class, e.g. from an offline research
script) -- this class is just the stateful "rolling price window per
symbol" wrapper around it.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import numpy.typing as npt

from core.logger import get_logger
from core.quant_math import Z_SCORES_BY_CONFIDENCE, evt_var, parametric_var
from core.schemas import OrderBookSnapshot

_log = get_logger(__name__)


class InsufficientHistoryError(Exception):
    """Raised when VaR is requested for a symbol without enough price history yet."""


class RiskLimitExceededError(Exception):
    """Raised when a proposed order would breach the configured exposure limit.

    This is the "hard stop": callers must treat this as blocking the
    order, not merely a warning.
    """


class RiskManager:
    """Computes parametric VaR and enforces a portfolio exposure ceiling.

    Usage pattern: feed every new :class:`~core.schemas.OrderBookSnapshot`
    through :meth:`record_order_book_snapshot` as it arrives (e.g. from
    :class:`~src.agents.market_data.agent.MarketDataAgent`), then call
    :meth:`calculate_var` and :meth:`check_exposure_limit` before sizing
    or approving a new order.
    """

    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        lookback_window: int = 30,
        max_portfolio_exposure_pct: float = 0.05,
    ) -> None:
        """Configure the risk model.

        Args:
            confidence_level: VaR confidence level. Must be one of the
                standard levels in :data:`core.quant_math.
                Z_SCORES_BY_CONFIDENCE` (``0.90``, ``0.95``, ``0.975``,
                ``0.99``, ``0.995``).
            lookback_window: Number of recent mid-prices kept per symbol
                for the volatility estimate (so ``lookback_window - 1``
                returns).
            max_portfolio_exposure_pct: Maximum total notional exposure,
                as a fraction of capital (e.g. ``0.05`` for 5%), enforced
                by :meth:`check_exposure_limit`.

        Raises:
            ValueError: If ``confidence_level`` isn't one of the standard
                supported levels, or if ``lookback_window`` is too small
                to ever produce a return series.
        """
        if confidence_level not in Z_SCORES_BY_CONFIDENCE:
            supported = sorted(Z_SCORES_BY_CONFIDENCE)
            raise ValueError(
                f"Unsupported confidence_level {confidence_level!r}; must be one of {supported}."
            )
        if lookback_window < 3:
            raise ValueError("lookback_window must be >= 3 (need at least 2 returns).")
        if not (0 < max_portfolio_exposure_pct <= 1):
            raise ValueError("max_portfolio_exposure_pct must be in (0, 1].")

        self._confidence_level = confidence_level
        self._lookback_window = lookback_window
        self._max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self._mid_price_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )

    def record_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Feed a new snapshot's mid-price into the rolling window used for VaR.

        Args:
            snapshot: A validated order book snapshot (see
                :class:`~core.schemas.OrderBookSnapshot`).
        """
        mid_price = (snapshot.best_bid.price + snapshot.best_ask.price) / 2.0
        self._mid_price_history[snapshot.symbol].append(mid_price)

    def has_sufficient_history(self, symbol: str) -> bool:
        """Whether :meth:`calculate_var` can currently produce a result for ``symbol``."""
        return len(self._mid_price_history.get(symbol, ())) >= 3

    def calculate_var(self, symbol: str, position_value: float) -> float:
        """Estimate Value at Risk via the parametric (variance-covariance) method.

        Computes the standard deviation of recent log-returns on
        ``symbol``'s order book mid-price, then scales it by the
        confidence level's z-score and the position's notional value:
        ``VaR = z * sigma * position_value``.

        Args:
            symbol: Trading pair symbol to compute VaR for. Must have had
                at least 3 snapshots recorded via
                :meth:`record_order_book_snapshot`.
            position_value: Notional value of the position being risk-
                assessed (in the same currency units as the VaR result).

        Returns:
            The estimated maximum loss, in ``position_value``'s currency
            units, over one snapshot-to-snapshot period, at the
            configured confidence level.

        Raises:
            InsufficientHistoryError: If fewer than 3 snapshots have been
                recorded for ``symbol`` (i.e. fewer than 2 returns).
            ValueError: If ``position_value`` is negative.
        """
        if position_value < 0:
            raise ValueError("position_value must not be negative.")
        if not self.has_sufficient_history(symbol):
            raise InsufficientHistoryError(
                f"Not enough order book history for {symbol!r} to compute VaR "
                f"(have {len(self._mid_price_history.get(symbol, ()))}, need >= 3)."
            )

        prices: npt.NDArray[np.float64] = np.array(
            self._mid_price_history[symbol], dtype=np.float64
        )
        log_returns = np.diff(np.log(prices))

        var = parametric_var(log_returns, self._confidence_level, position_value)
        _log.debug(
            "var_calculated",
            symbol=symbol,
            confidence_level=self._confidence_level,
            position_value=position_value,
            var=var,
        )
        return var

    def calculate_var_evt(
        self,
        symbol: str,
        position_value: float,
        *,
        threshold_percentile: float = 0.90,
        min_exceedances: int = 10,
    ) -> float:
        """Estimate Value at Risk via Extreme Value Theory (Peaks-Over-Threshold/GPD) -- no Gaussian assumption.

        A real, additional VaR estimate alongside :meth:`calculate_var`,
        not a replacement for it -- see :func:`core.quant_math.evt_var`'s
        own docstring for the full method (McNeil & Frey, 1999) and why
        it doesn't share :meth:`calculate_var`'s Gaussian-returns
        assumption.

        **Real prerequisite, disclosed**: this needs substantially more
        history than :meth:`calculate_var` does -- fitting a trustworthy
        tail distribution needs real exceedances (see
        ``min_exceedances``), which this class's own default
        ``lookback_window`` (``30``) essentially never provides at the
        default ``threshold_percentile`` (``0.90`` of 30 samples leaves
        only ~3 tail points). Construct this :class:`RiskManager` with a
        real, substantially larger ``lookback_window`` (e.g. ``200+``) to
        actually use this method -- :meth:`calculate_var` remains fully
        usable at the smaller default either way.

        Args:
            symbol: Trading pair symbol to compute VaR for.
            position_value: Notional value of the position being risk-
                assessed. Must not be negative.
            threshold_percentile: See :func:`core.quant_math.evt_var`.
            min_exceedances: See :func:`core.quant_math.evt_var`.

        Returns:
            The estimated maximum loss, in ``position_value``'s currency
            units, at this instance's configured confidence level.

        Raises:
            InsufficientHistoryError: If fewer than 3 snapshots have been
                recorded for ``symbol``.
            ValueError: If ``position_value`` is negative, or (propagated
                from :func:`core.quant_math.evt_var`) too few real
                exceedances exist over the fitted threshold for a
                trustworthy tail estimate -- see the real prerequisite
                above.
        """
        if position_value < 0:
            raise ValueError("position_value must not be negative.")
        if not self.has_sufficient_history(symbol):
            raise InsufficientHistoryError(
                f"Not enough order book history for {symbol!r} to compute VaR "
                f"(have {len(self._mid_price_history.get(symbol, ()))}, need >= 3)."
            )

        prices: npt.NDArray[np.float64] = np.array(
            self._mid_price_history[symbol], dtype=np.float64
        )
        log_returns = np.diff(np.log(prices))

        var = evt_var(
            log_returns, self._confidence_level, position_value,
            threshold_percentile=threshold_percentile, min_exceedances=min_exceedances,
        )
        _log.debug(
            "var_evt_calculated",
            symbol=symbol,
            confidence_level=self._confidence_level,
            position_value=position_value,
            var=var,
        )
        return var

    def check_exposure_limit(
        self, *, current_exposure: float, proposed_order_value: float, capital: float
    ) -> None:
        """Block a proposed order if it would breach the exposure ceiling.

        Args:
            current_exposure: Current total notional exposure across the
                portfolio (absolute value, all positions summed), before
                this order.
            proposed_order_value: Notional value of the order being
                considered.
            capital: Total portfolio capital (equity) the exposure
                percentage is measured against.

        Raises:
            RiskLimitExceededError: If ``current_exposure +
                proposed_order_value`` would exceed
                ``max_portfolio_exposure_pct * capital``.
            ValueError: If ``capital`` is not positive, or either
                exposure value is negative.
        """
        if capital <= 0:
            raise ValueError("capital must be positive.")
        if current_exposure < 0 or proposed_order_value < 0:
            raise ValueError("current_exposure and proposed_order_value must not be negative.")

        max_exposure = self._max_portfolio_exposure_pct * capital
        projected_exposure = current_exposure + proposed_order_value

        if projected_exposure > max_exposure:
            _log.warning(
                "risk_limit_exceeded",
                current_exposure=current_exposure,
                proposed_order_value=proposed_order_value,
                projected_exposure=projected_exposure,
                max_exposure=max_exposure,
                max_portfolio_exposure_pct=self._max_portfolio_exposure_pct,
            )
            raise RiskLimitExceededError(
                f"Proposed order would bring total exposure to {projected_exposure:.2f}, "
                f"exceeding the {self._max_portfolio_exposure_pct:.1%} limit "
                f"({max_exposure:.2f}) on capital {capital:.2f}."
            )

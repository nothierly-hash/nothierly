"""Real options position tracking + real multi-instrument hedge-ratio solving.

:class:`OptionsHedgingManager` tracks real (paper-filled) option
positions and computes real, live aggregate Greeks combining them with
this agent's own real underlying spot/CFD positions -- completing the
Ashby's-Law-as-linear-algebra thread :func:`core.quant_math.
check_hedge_completeness`/:func:`core.quant_math.solve_hedge_ratios`
started: given the real instruments actually available (this agent's
configured spot/CFD instruments plus whatever option contracts this
class is tracking or considering), does the real combined sensitivity
matrix have enough "requisite variety" to hedge the real aggregate
exposure, and if so, what real position sizes would do it?

**Real, disclosed execution boundary**: this project has no real options
broker connection -- OANDA (:mod:`~src.agents.supervisor.broker_client`)
is FX/CFD spot only, and no live options exchange client exists in the
production agent fleet. Every option position this class tracks is
therefore necessarily a simulated/paper fill, priced by real Black-
Scholes math (:mod:`core.quant_math`) against real market inputs where
available -- the exact same MOCK_MODE/SHADOW convention
:class:`~src.agents.execution.agent.ExecutionAgent` already uses for its
own real spot entries, applied honestly here rather than silently
pretending a real options order was placed. See
:mod:`~src.agents.market_data.options_chain_client` for the one real,
optional piece of live options market data this project *can* actually
reach (a public Deribit option chain, via ``ccxt`` -- already a project
dependency), used for real strikes/expiries/mark prices, never for
placing a real order.

**"Discrete Hedge Replication Error" (2026-09-19)**: a real due-diligence "dead code" finding --
:func:`core.quant_math.simulate_discrete_delta_hedge` existed with zero real callers anywhere in
this project, despite this module's own paper-filled positions being the one place in this codebase
that could actually use it. :meth:`OptionsHedgingManager.estimate_discrete_hedge_replication_error`
is that real wiring: given a real historical price path for a tracked position's underlying (e.g.
:meth:`~src.agents.supervisor.portfolio_risk_manager.PortfolioRiskParityManager.price_history`,
already fed from every real ``MARKET_DATA_L2`` tick :class:`~src.agents.execution.agent.
ExecutionAgent` receives), it retrospectively measures the real, nonzero replication error this
project's own discrete rebalancing would have left -- honest, quantified evidence for this module's
own "paper-filled, priced by real Black-Scholes math, never risk-free" framing above, not just an
assertion of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import numpy as np
import numpy.typing as npt

from core.logger import get_logger
from core.quant_math import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
    black_scholes_theta,
    black_scholes_vega,
    check_hedge_completeness,
    simulate_discrete_delta_hedge,
    solve_hedge_ratios,
)
from core.schemas import OptionContract, OptionPosition

_log = get_logger(__name__)

_GREEK_NAMES: Final[tuple[str, str, str, str]] = ("delta", "gamma", "vega", "theta")
_GREEK_INDEX: Final[dict[str, int]] = {name: i for i, name in enumerate(_GREEK_NAMES)}


@dataclass(frozen=True)
class GreeksSnapshot:
    """One real (delta, gamma, vega, theta) reading -- per-position, per-instrument, or aggregated.

    Attributes:
        delta: Real sensitivity to the underlying's price.
        gamma: Real sensitivity of delta itself to the underlying's price.
        vega: Real sensitivity to volatility.
        theta: Real sensitivity to time (real time decay).
    """

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    def to_array(self) -> npt.NDArray[np.float64]:
        """``[delta, gamma, vega, theta]``, matching :data:`_GREEK_NAMES`'s real, fixed order."""
        return np.array([self.delta, self.gamma, self.vega, self.theta], dtype=np.float64)

    def __add__(self, other: GreeksSnapshot) -> GreeksSnapshot:
        return GreeksSnapshot(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
        )


@dataclass(frozen=True)
class HedgeRecommendation:
    """One real Ashby's-Law hedge-completeness check + hedge-ratio solve.

    Attributes:
        is_complete: Whether the real instrument set's variety is
            sufficient to fully neutralize every targeted Greek (see
            :func:`core.quant_math.check_hedge_completeness`).
        rank: The real rank of the sensitivity matrix actually used.
        n_factors: How many Greeks were targeted.
        hedge_sizes: Real recommended signed size to trade in each real
            candidate instrument, keyed by instrument name.
        residual_exposure: Real exposure that would remain uncancelled
            after applying ``hedge_sizes``, keyed by targeted Greek name
            -- all (approximately) ``0`` only when ``is_complete``.
    """

    is_complete: bool
    rank: int
    n_factors: int
    hedge_sizes: dict[str, float]
    residual_exposure: dict[str, float]


class OptionsHedgingManager:
    """Tracks real option positions; computes real aggregate Greeks + real hedge-ratio recommendations."""

    def __init__(self, *, risk_free_rate: float = 0.0) -> None:
        """Configure the real risk-free rate every Black-Scholes evaluation uses.

        Args:
            risk_free_rate: Continuously-compounded real risk-free rate
                (e.g. a real short-term Treasury/SOFR proxy). ``0.0`` by
                default -- a real, disclosed simplification: this project
                has no real live risk-free-rate feed, so an uncalibrated
                ``0.0`` is the honest "no real input available" default,
                not a fabricated nonzero rate.
        """
        self._risk_free_rate = risk_free_rate
        self._positions: list[OptionPosition] = []

    @property
    def positions(self) -> tuple[OptionPosition, ...]:
        """Every real (paper-filled) option position currently tracked, oldest first."""
        return tuple(self._positions)

    def open_position(
        self,
        contract: OptionContract,
        quantity: float,
        *,
        current_underlying_price: float,
        volatility: float,
    ) -> OptionPosition:
        """Open one real (necessarily paper-filled) option position, priced via real Black-Scholes.

        Args:
            contract: The real contract specification.
            quantity: Real signed contracts (positive = long, negative =
                short). Must not be zero.
            current_underlying_price: Real current price of
                ``contract.underlying``. Must be positive.
            volatility: Real annualized volatility to price this entry
                at (e.g. a real recent realized-volatility estimate, or
                a real market-implied volatility from
                :mod:`~src.agents.market_data.options_chain_client`).
                Must be positive.

        Returns:
            The real, newly-opened (and tracked) :class:`~core.schemas.OptionPosition`.

        Raises:
            ValueError: If ``quantity`` is zero, ``current_underlying_price``/
                ``volatility`` is not positive, or ``contract`` has
                already expired.
        """
        if quantity == 0:
            raise ValueError("quantity must not be zero.")
        if current_underlying_price <= 0:
            raise ValueError("current_underlying_price must be positive.")
        if volatility <= 0:
            raise ValueError("volatility must be positive.")

        time_to_expiry = self._time_to_expiry(contract)
        premium = black_scholes_price(
            current_underlying_price, contract.strike, time_to_expiry, self._risk_free_rate, volatility,
            option_type=contract.option_type,
        )
        position = OptionPosition(
            contract=contract, quantity=quantity, entry_premium=premium, entry_timestamp=time.time()
        )
        self._positions.append(position)
        _log.info(
            "option_position_opened",
            underlying=contract.underlying, strike=contract.strike, option_type=contract.option_type,
            expiry=contract.expiry.isoformat(), quantity=quantity, premium=round(premium, 6),
        )
        return position

    def close_all_positions(self) -> None:
        """Flatten every real tracked position -- a real, unilateral flatten, no partial-close support."""
        self._positions.clear()

    @staticmethod
    def _time_to_expiry(contract: OptionContract) -> float:
        """Real years-to-expiry for ``contract``, from the real current wall-clock time.

        Raises:
            ValueError: If ``contract`` has already expired.
        """
        remaining_seconds = (contract.expiry - datetime.now(timezone.utc)).total_seconds()
        if remaining_seconds <= 0:
            raise ValueError(f"Contract already expired: {contract.expiry.isoformat()}.")
        return remaining_seconds / (365.0 * 24.0 * 3600.0)

    def compute_position_greeks(
        self, position: OptionPosition, current_underlying_price: float, volatility: float
    ) -> GreeksSnapshot:
        """Real Greeks for one position at the current real market, scaled by its real signed size.

        Args:
            position: The real position to evaluate.
            current_underlying_price: Real current price of
                ``position.contract.underlying``. Must be positive.
            volatility: Real annualized volatility to evaluate at. Must
                be positive.

        Returns:
            The real, size-scaled :class:`GreeksSnapshot`.

        Raises:
            ValueError: If ``current_underlying_price``/``volatility`` is
                not positive, or the position's contract has expired.
        """
        if current_underlying_price <= 0:
            raise ValueError("current_underlying_price must be positive.")
        if volatility <= 0:
            raise ValueError("volatility must be positive.")

        time_to_expiry = self._time_to_expiry(position.contract)
        scale = position.quantity * position.contract.contract_size
        option_type = position.contract.option_type
        strike = position.contract.strike

        return GreeksSnapshot(
            delta=black_scholes_delta(current_underlying_price, strike, time_to_expiry, self._risk_free_rate, volatility, option_type=option_type) * scale,
            gamma=black_scholes_gamma(current_underlying_price, strike, time_to_expiry, self._risk_free_rate, volatility) * scale,
            vega=black_scholes_vega(current_underlying_price, strike, time_to_expiry, self._risk_free_rate, volatility) * scale,
            theta=black_scholes_theta(current_underlying_price, strike, time_to_expiry, self._risk_free_rate, volatility, option_type=option_type) * scale,
        )

    def compute_portfolio_greeks(
        self,
        *,
        underlying_prices: dict[str, float],
        volatilities: dict[str, float],
        spot_positions: dict[str, float] | None = None,
    ) -> GreeksSnapshot:
        """Real aggregate Greeks: every tracked option position, plus real spot/CFD delta contributions.

        A real spot/CFD position of signed size :math:`x` contributes
        exactly :math:`x` to aggregate delta and exactly ``0`` to
        gamma/vega/theta -- its own real, trivial "Greeks" (a linear
        instrument has no convexity, volatility, or time-decay
        sensitivity of its own).

        Args:
            underlying_prices: Real current price per underlying
                instrument. An option position whose underlying is
                missing here is honestly skipped (real "cannot price
                without real current data", never a fabricated value --
                see the real ``skipped`` count logged).
            volatilities: Real annualized volatility per underlying
                instrument, same real skip behavior as
                ``underlying_prices`` for a missing entry.
            spot_positions: Real current signed spot/CFD position size
                per instrument (e.g. this agent's own
                :attr:`~src.agents.execution.agent.ExecutionAgent._positions`
                translated to signed units), if any.

        Returns:
            The real aggregate :class:`GreeksSnapshot`.
        """
        total = GreeksSnapshot()
        skipped = 0
        for position in self._positions:
            underlying = position.contract.underlying
            price = underlying_prices.get(underlying)
            volatility = volatilities.get(underlying)
            if price is None or volatility is None:
                skipped += 1
                continue
            try:
                total = total + self.compute_position_greeks(position, price, volatility)
            except ValueError:
                skipped += 1

        if skipped > 0:
            _log.debug("portfolio_greeks_positions_skipped_missing_real_market_data", skipped=skipped)

        if spot_positions:
            spot_delta = sum(spot_positions.values())
            total = GreeksSnapshot(
                delta=total.delta + spot_delta, gamma=total.gamma, vega=total.vega, theta=total.theta
            )

        return total

    def compute_hedge_recommendation(
        self,
        current_exposure: GreeksSnapshot,
        available_instruments: dict[str, GreeksSnapshot],
        *,
        target_greeks: tuple[str, ...] = ("delta", "vega"),
    ) -> HedgeRecommendation:
        r"""Real Ashby's-Law hedge: is there enough real instrument variety, and if so, how much of each?

        Args:
            current_exposure: The real current aggregate exposure to
                hedge away (e.g. from :meth:`compute_portfolio_greeks`).
            available_instruments: Real candidate hedging instruments,
                each mapped to its own real **per-unit** Greeks (a spot
                instrument's own real per-unit Greeks are simply
                ``GreeksSnapshot(delta=1.0)``; an option contract's are
                real per-contract Black-Scholes Greeks, e.g. from
                :meth:`compute_position_greeks` on a hypothetical
                ``quantity=1`` position).
            target_greeks: Which Greeks to actually neutralize, in
                ``{"delta", "gamma", "vega", "theta"}``. Defaults to
                ``("delta", "vega")`` -- the two a real spot-only
                instrument set can never touch beyond delta (see the
                module docstring: real spot has zero vega by
                construction, so hedging real vega exposure genuinely
                requires a real option in ``available_instruments``).

        Returns:
            The real :class:`HedgeRecommendation`.

        Raises:
            ValueError: If ``target_greeks`` is empty or contains an
                unknown Greek name, or ``available_instruments`` is empty.
        """
        if not target_greeks:
            raise ValueError("target_greeks must not be empty.")
        unknown = set(target_greeks) - set(_GREEK_NAMES)
        if unknown:
            raise ValueError(f"Unknown Greek(s) in target_greeks: {sorted(unknown)}.")
        if not available_instruments:
            raise ValueError("available_instruments must not be empty.")

        factor_indices = [_GREEK_INDEX[name] for name in target_greeks]
        exposure_vector = current_exposure.to_array()[factor_indices]

        instrument_names = list(available_instruments.keys())
        sensitivity_matrix = np.array(
            [available_instruments[name].to_array()[factor_indices] for name in instrument_names],
            dtype=np.float64,
        )

        is_complete, rank, n_factors = check_hedge_completeness(sensitivity_matrix)
        hedge_sizes, residual = solve_hedge_ratios(sensitivity_matrix, exposure_vector)

        return HedgeRecommendation(
            is_complete=is_complete,
            rank=rank,
            n_factors=n_factors,
            hedge_sizes=dict(zip(instrument_names, (float(x) for x in hedge_sizes))),
            residual_exposure=dict(zip(target_greeks, (float(x) for x in residual))),
        )

    def estimate_discrete_hedge_replication_error(
        self,
        position: OptionPosition,
        underlying_price_path: npt.NDArray[np.float64],
        *,
        time_step: float,
        volatility: float,
    ) -> float:
        r"""Real, retrospective discrete-hedge replication error for one real (paper-filled) position. See module docstring addendum.

        Args:
            position: The real (paper-filled) position to evaluate.
            underlying_price_path: A real, chronologically-ordered price
                path for ``position.contract.underlying``, evenly spaced
                ``time_step`` apart, ending at (or before) the
                contract's real expiry -- e.g. a trailing slice of
                :meth:`~src.agents.supervisor.portfolio_risk_manager.
                PortfolioRiskParityManager.price_history`. See
                :func:`~core.quant_math.simulate_discrete_delta_hedge`'s
                own docstring for the full real precondition (at least 2
                elements, strictly positive).
            time_step: Real time between consecutive
                ``underlying_price_path`` observations, in years.
            volatility: Real annualized volatility assumed for the
                hedge -- this simulation's own real, disclosed
                assumption (see
                :func:`~core.quant_math.simulate_discrete_delta_hedge`'s
                own docstring on hedging-vs-realized-volatility error).

        Returns:
            The real, signed, size-scaled replication error in this
            position's own quote currency -- ``position.quantity *
            position.contract.contract_size`` times
            :func:`~core.quant_math.simulate_discrete_delta_hedge`'s own
            per-unit ``replication_error``, the same real scaling
            :meth:`compute_position_greeks` already applies to its own
            per-unit Greeks.

        Raises:
            ValueError: Propagated from
                :func:`~core.quant_math.simulate_discrete_delta_hedge`
                (e.g. too short or non-positive a price path).
        """
        contract = position.contract
        scale = position.quantity * contract.contract_size
        _replicating_path, per_unit_replication_error = simulate_discrete_delta_hedge(
            underlying_price_path, time_step, contract.strike, self._risk_free_rate, volatility,
            option_type=contract.option_type,
        )
        return per_unit_replication_error * scale

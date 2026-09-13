"""Optimal meta-order execution scheduling: real Almgren-Chriss slicing + real impact cost curves.

:class:`OptimalExecutionScheduler` is a pure, stateless-with-respect-to-
position scheduling utility -- the same "stateless real-math wrapper
around :mod:`core.quant_math`" pattern :class:`~src.agents.supervisor.
risk_manager.RiskManager` and :class:`~src.agents.supervisor.
portfolio_risk_manager.PortfolioRiskParityManager` already establish. It
computes *how a proposed meta-order should be sliced* over time to
minimize the real trade-off between market-impact cost and timing risk
(Almgren-Chriss, 2000), and the real temporary/permanent impact cost
curve that schedule implies. It does not place any order itself, track
any position, or know this system's real broker client -- see
:meth:`~src.agents.execution.agent.ExecutionAgent.
compute_optimal_trajectory`, the only real caller, for the real,
disclosed scope boundary: this is a computed recommendation, not yet
wired into this system's live order-placement path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from core.logger import get_logger
from core.quant_math import (
    almgren_chriss_impact_costs,
    almgren_chriss_trajectory,
    calibrate_impact_coefficient_fdt,
    participation_rate_capped_schedule,
)

_log = get_logger(__name__)

# Real free parameters of the Almgren-Chriss model this class was NOT
# given a real, per-instrument-calibrated value for (no historical
# execution-cost dataset exists yet in this system to fit them against
# -- a real, disclosed limitation, the same kind this project's other
# uncalibrated model defaults already carry, e.g. core.quant_math's own
# Hawkes-process constants). Illustrative, conservative defaults, not
# fabricated precision.
_DEFAULT_RISK_AVERSION: float = 1e-6
_DEFAULT_TEMPORARY_IMPACT_COEFFICIENT: float = 0.1
_DEFAULT_PERMANENT_IMPACT_COEFFICIENT: float = 0.05
_DEFAULT_N_STEPS: int = 10
_DEFAULT_MAX_PARTICIPATION_RATE: float = 0.10


@dataclass(frozen=True)
class ExecutionSchedule:
    """One real Almgren-Chriss execution schedule + its real impact cost curve.

    Attributes:
        trajectory: Real remaining-inventory curve (``n_steps + 1``
            points, ``trajectory[0] == order_size``).
        trade_sizes: Real per-slice quantity to trade (``n_steps``
            points), after the real participation-rate cap.
        temporary_impact: Real per-slice temporary price impact
            (``n_steps`` points) -- reverts after each slice.
        permanent_impact: Real cumulative permanent price impact
            (``n_steps`` points) -- persists, monotonically non-decreasing.
        was_participation_capped: Whether any real slice's trade size was
            reduced below what the uncapped Almgren-Chriss schedule
            wanted, by the real expected-volume participation cap -- if
            ``True``, the schedule does not fully complete ``order_size``
            by the end of the horizon (a real, honest shortfall, not
            silently hidden).
        total_estimated_temporary_cost: Real
            ``sum(temporary_impact[i] * trade_sizes[i])`` -- the real
            temporary-impact-only cost this schedule is estimated to
            incur, in price-times-quantity units. Deliberately excludes
            permanent impact (a persistent price *shift*, not
            unambiguously a realized "cost" in the same sense -- see the
            two separate curves above for the real breakdown instead of
            a single blended, harder-to-defend total).
    """

    trajectory: npt.NDArray[np.float64]
    trade_sizes: npt.NDArray[np.float64]
    temporary_impact: npt.NDArray[np.float64]
    permanent_impact: npt.NDArray[np.float64]
    was_participation_capped: bool
    total_estimated_temporary_cost: float


class OptimalExecutionScheduler:
    """Computes a real Almgren-Chriss meta-order execution schedule + its real impact cost curve."""

    def __init__(
        self,
        *,
        risk_aversion: float = _DEFAULT_RISK_AVERSION,
        temporary_impact_coefficient: float = _DEFAULT_TEMPORARY_IMPACT_COEFFICIENT,
        permanent_impact_coefficient: float = _DEFAULT_PERMANENT_IMPACT_COEFFICIENT,
        max_participation_rate: float = _DEFAULT_MAX_PARTICIPATION_RATE,
        n_steps: int = _DEFAULT_N_STEPS,
    ) -> None:
        """Configure the Almgren-Chriss model's real free parameters.

        Args:
            risk_aversion: See :func:`core.quant_math.
                almgren_chriss_trajectory`'s ``risk_aversion``.
            temporary_impact_coefficient: See that function's
                ``temporary_impact_coefficient``.
            permanent_impact_coefficient: See :func:`core.quant_math.
                almgren_chriss_impact_costs`'s ``permanent_impact_coefficient``.
            max_participation_rate: See :func:`core.quant_math.
                participation_rate_capped_schedule`'s
                ``max_participation_rate``.
            n_steps: How many discrete execution slices to schedule.
        """
        self._risk_aversion = risk_aversion
        self._temporary_impact_coefficient = temporary_impact_coefficient
        self._permanent_impact_coefficient = permanent_impact_coefficient
        self._max_participation_rate = max_participation_rate
        self._n_steps = n_steps

    def calibrate_temporary_impact_from_market_data(
        self,
        resting_spread: npt.NDArray[np.float64],
        average_trade_rate: float,
        *,
        coupling_constant: float = 1.0,
    ) -> float:
        r"""Real, Fluctuation-Dissipation-inspired recalibration of :attr:`_temporary_impact_coefficient`.

        Replaces the constructor's uncalibrated
        ``temporary_impact_coefficient`` default with a real value fit
        from this instrument's own real, recently observed market data --
        see :func:`core.quant_math.calibrate_impact_coefficient_fdt` for
        the real formula and its honestly-disclosed status as an analogy
        to the classical theorem, not a first-principles derivation for
        markets. A real, callable capability -- not auto-invoked on a
        timer by this class itself, since "how often to recalibrate" is
        a real, separate policy decision for the caller (e.g.
        :class:`~src.agents.execution.agent.ExecutionAgent`, which
        already tracks real recent spread history) to make.

        Args:
            resting_spread: Real observed spread samples from
                *undisturbed* conditions -- see that function's own
                docstring on why contaminating this with the meta-
                order's own impact would be circular.
            average_trade_rate: Real average traded volume per second.
            coupling_constant: See that function's own ``coupling_constant``.

        Returns:
            The newly calibrated coefficient (also stored on this
            instance for every subsequent :meth:`compute_schedule` call).

        Raises:
            ValueError: Propagated from :func:`core.quant_math.
                calibrate_impact_coefficient_fdt` for any invalid input.
        """
        calibrated = calibrate_impact_coefficient_fdt(
            resting_spread, average_trade_rate, coupling_constant=coupling_constant
        )
        _log.info(
            "execution_scheduler_temporary_impact_recalibrated",
            previous_coefficient=self._temporary_impact_coefficient,
            calibrated_coefficient=calibrated,
        )
        self._temporary_impact_coefficient = calibrated
        return calibrated

    def compute_schedule(
        self,
        order_size: float,
        volatility: float,
        volume_profile: npt.NDArray[np.float64],
        *,
        execution_horizon_seconds: float,
    ) -> ExecutionSchedule:
        """Compute one real execution schedule for a proposed meta-order.

        Args:
            order_size: Total real units to execute. Must be positive.
            volatility: Real **per-second** price volatility -- see
                :func:`core.quant_math.almgren_chriss_trajectory`'s own
                docstring on the real unit-magnitude pitfall found by
                testing this (an annualized figure passed here directly
                produces a practically degenerate, front-load-everything
                schedule).
            volume_profile: Real expected traded volume for each of this
                instance's ``n_steps`` slices. Must have exactly
                ``n_steps`` entries, all positive.
            execution_horizon_seconds: Real total time allotted to fully
                execute ``order_size``. Must be positive.

        Returns:
            The real computed :class:`ExecutionSchedule`.

        Raises:
            ValueError: Propagated from :mod:`core.quant_math` for any
                invalid input (see those functions' own docstrings) --
                including ``volume_profile`` not having exactly
                ``n_steps`` entries.
        """
        trajectory = almgren_chriss_trajectory(
            order_size,
            execution_horizon_seconds=execution_horizon_seconds,
            volatility=volatility,
            risk_aversion=self._risk_aversion,
            temporary_impact_coefficient=self._temporary_impact_coefficient,
            n_steps=self._n_steps,
        )
        trade_sizes, was_capped = participation_rate_capped_schedule(
            trajectory, volume_profile, max_participation_rate=self._max_participation_rate
        )
        step_duration_seconds = execution_horizon_seconds / self._n_steps
        temporary_impact, permanent_impact = almgren_chriss_impact_costs(
            trade_sizes,
            temporary_impact_coefficient=self._temporary_impact_coefficient,
            permanent_impact_coefficient=self._permanent_impact_coefficient,
            step_duration_seconds=step_duration_seconds,
        )
        total_estimated_temporary_cost = float(np.sum(temporary_impact * trade_sizes))

        if was_capped:
            _log.warning(
                "execution_schedule_participation_capped",
                order_size=order_size,
                scheduled_total=float(np.sum(trade_sizes)),
                detail="Real expected volume was too thin to complete this order within the horizon at the uncapped Almgren-Chriss rate.",
            )

        return ExecutionSchedule(
            trajectory=trajectory,
            trade_sizes=trade_sizes,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            was_participation_capped=was_capped,
            total_estimated_temporary_cost=total_estimated_temporary_cost,
        )

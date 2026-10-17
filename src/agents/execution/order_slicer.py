"""Order Slicer: real TWAP (Time-Weighted Average Price) execution slicing with stochastic jitter.

:class:`OrderSlicer` is a pure, stateless-with-respect-to-position scheduling/pacing utility -- the
same "stateless real-math wrapper, real integration lives in the calling agent" pattern
:class:`~src.agents.supervisor.optimal_execution_scheduler.OptimalExecutionScheduler` already
establishes. Given a total signed order size, it computes a real, deterministic set of integer
sub-orders ("slices") that sum exactly to that total, then paces placing them one at a time over a
real wall-clock horizon with real random jitter between slices.

**Real, disclosed relationship to the Almgren-Chriss scheduler**: this project already has a more
mathematically sophisticated meta-order scheduler (:class:`~src.agents.supervisor.
optimal_execution_scheduler.OptimalExecutionScheduler`, real Almgren-Chriss risk/impact trade-off +
participation-rate cap) -- but that one is explicitly *not* wired into the live order-placement
path (see its own module docstring: "a computed recommendation, not yet wired into this system's
live order-placement path"). This class is deliberately simpler and *is* wired into the live path
(see :meth:`~src.agents.execution.agent.ExecutionAgent._execute_sliced_entry`, the only real caller)
-- literal equal-size, equal-time TWAP slicing with jitter, matching what was actually asked for:
reduce real market-impact/slippage on a large real order and avoid an obviously-regular firing
cadence, not reproduce Almgren-Chriss's own risk-aversion trade-off. A future session could let this
class consult ``OptimalExecutionScheduler`` for non-uniform, participation-capped slice sizing
instead of uniform TWAP -- real, disclosed, separate follow-up work, not silently implied here.

**Real safety boundary**: this class does not itself know about this system's Kill Switch,
``AgentControlState``, or the real per-instrument lock that guards ``PositionState`` mutations -- it
only knows how to size and pace slices. The caller's ``place_slice`` callback is the sole real
integration point, and is expected to perform every one of those real safety checks itself, wrapped
around the real broker call, before returning (see :meth:`ExecutionAgent._execute_sliced_entry`) --
a slow, real, minutes-long execution must stay interruptible mid-flight by the same real Kill
Switch/operator controls that gate a single order, not just checked once at the very start.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Final

from core.logger import get_logger
from core.security import FinancialCallAbortedError

_log = get_logger(__name__)

DEFAULT_NUM_SLICES: Final[int] = 10
DEFAULT_EXECUTION_HORIZON_SECONDS: Final[float] = 300.0  # 5 minutes
DEFAULT_JITTER_SECONDS: Final[float] = 0.4  # a few hundred ms of real stochastic pacing noise
_MIN_SLEEP_SECONDS: Final[float] = 0.01  # a real floor so jitter can never produce a zero/negative sleep


class SlicingAbortedError(Exception):
    """Raised by a ``place_slice`` callback to stop remaining slices, for a real, benign, non-broker reason.

    Distinct from :class:`~core.security.FinancialCallAbortedError` (a real broker/Kill-Switch
    failure) -- both are treated identically by :meth:`OrderSlicer.execute` (stop, return the real
    partial result), but this one is for a caller-side reason unrelated to the broker call itself
    (e.g. the position this slicing was growing was closed by a concurrent, unrelated real exit).
    """


@dataclass(frozen=True)
class OrderSlice:
    """One real, integer-unit slice of a larger meta-order.

    Attributes:
        index: 1-based position in the real slicing plan (for
            human-readable progress -- "tranche 3/10").
        units: Real signed integer units to trade for this slice alone
            (matches :attr:`~src.agents.execution.agent.PositionState.
            units_held`'s sign convention: positive = buy, negative =
            sell).
    """

    index: int
    units: int


@dataclass(frozen=True)
class SlicingResult:
    """The real outcome of one :meth:`OrderSlicer.execute` call -- always honest about a partial fill.

    Attributes:
        filled_units: Real signed total units actually filled across
            every slice that completed before this execution finished
            or aborted. Equals the originally requested total only if
            ``aborted_early`` is ``False``.
        slices_completed: How many of ``slices_planned`` real slices
            actually placed a real order.
        slices_planned: Total real slices this execution was planned
            for (see :meth:`OrderSlicer.plan`).
        aborted_early: Whether this execution stopped before every
            planned slice fired (a real Kill Switch trip, operator
            control-state change, concurrent position close, or broker
            failure -- see ``place_slice``'s own contract). ``False``
            means every planned slice filled.
    """

    filled_units: int
    slices_completed: int
    slices_planned: int
    aborted_early: bool


class OrderSlicer:
    """Plans and paces one real meta-order as N real market-order slices, TWAP-spaced with jitter."""

    def __init__(
        self,
        *,
        num_slices: int = DEFAULT_NUM_SLICES,
        execution_horizon_seconds: float = DEFAULT_EXECUTION_HORIZON_SECONDS,
        jitter_seconds: float = DEFAULT_JITTER_SECONDS,
        rng: random.Random | None = None,
    ) -> None:
        """Configure this slicer's real TWAP pacing parameters.

        Args:
            num_slices: How many discrete slices to split a qualifying
                order into. Must be positive.
            execution_horizon_seconds: Real total wall-clock time to
                spread ``num_slices`` slices across. Must be positive.
            jitter_seconds: Real uniform random noise, in seconds, added
                to (or subtracted from) each slice's otherwise equal
                TWAP interval -- see the module docstring on why (an
                adversarial HFT strategy watching for an exactly regular
                firing cadence should not find one). Must be ``>= 0``;
                ``0`` degenerates to a literal, unjittered TWAP.
            rng: Real source of randomness for jitter. Defaults to a
                fresh, process-local :class:`random.Random` -- not
                :mod:`secrets`/cryptographic randomness, since this
                jitter's only job is breaking a detectable regular
                cadence, not resisting an adversary who can already
                observe the real fills themselves.

        Raises:
            ValueError: If ``num_slices`` or ``execution_horizon_seconds``
                is not positive, or ``jitter_seconds`` is negative.
        """
        if num_slices < 1:
            raise ValueError(f"num_slices must be positive, got {num_slices}.")
        if execution_horizon_seconds <= 0:
            raise ValueError(f"execution_horizon_seconds must be positive, got {execution_horizon_seconds}.")
        if jitter_seconds < 0:
            raise ValueError(f"jitter_seconds must be >= 0, got {jitter_seconds}.")
        self._num_slices = num_slices
        self._execution_horizon_seconds = execution_horizon_seconds
        self._jitter_seconds = jitter_seconds
        self._rng = rng if rng is not None else random.Random()

    def plan(self, total_units: int) -> list[OrderSlice]:
        """Real, deterministic sizing: up to :attr:`_num_slices` integer slices summing exactly to ``total_units``.

        Uses cumulative rounding (each slice is ``round(total * i / n) -
        round(total * (i-1) / n)``), not naive ``total // n`` repeated:
        that would silently drop up to ``n - 1`` real units to integer-
        division truncation (real money left unexecuted, never
        resubmitted). Cumulative rounding guarantees the real slices sum
        to exactly ``total_units`` and distributes the real one-unit
        remainders evenly across the plan rather than dumping them all
        onto the first or last slice.

        Args:
            total_units: Real signed total to slice. Must be nonzero.

        Returns:
            Between 1 and :attr:`_num_slices` slices (a slice that would
            round to 0 real units is dropped, so a ``total_units``
            smaller in magnitude than :attr:`_num_slices` produces
            fewer, larger slices rather than any real zero-unit order)
            -- always at least one, and their ``units`` always sum to
            exactly ``total_units``.

        Raises:
            ValueError: If ``total_units`` is 0.
        """
        if total_units == 0:
            raise ValueError("total_units must not be 0.")
        sign = 1 if total_units > 0 else -1
        magnitude = abs(total_units)

        slices: list[OrderSlice] = []
        previous_cumulative = 0
        for i in range(1, self._num_slices + 1):
            cumulative_target = round(magnitude * i / self._num_slices)
            slice_magnitude = cumulative_target - previous_cumulative
            previous_cumulative = cumulative_target
            if slice_magnitude != 0:
                slices.append(OrderSlice(index=len(slices) + 1, units=sign * slice_magnitude))
        return slices

    async def execute(
        self,
        total_units: int,
        *,
        place_slice: Callable[[int], Awaitable[object]],
        on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> SlicingResult:
        """Real TWAP execution loop: place each planned slice, jittered-sleep between, stop honestly on any abort.

        Args:
            total_units: See :meth:`plan`.
            place_slice: Awaited once per slice with that slice's real
                signed integer ``units``. Expected to perform every real
                safety check (Kill Switch, operator control state,
                concurrent position-state consistency -- see the module
                docstring) and the real broker call itself, atomically,
                before returning; raising :class:`SlicingAbortedError`
                or :class:`~core.security.FinancialCallAbortedError`
                stops all remaining slices honestly (this method never
                retries a slice that raised).
            on_progress: Awaited after each slice that successfully
                placed, with ``(slices_completed, slices_planned)`` --
                e.g. to persist live "34% (17/50 tranches tirées)"
                telemetry. Never awaited for a slice that raised.

        Returns:
            The real, honest :class:`SlicingResult` -- ``aborted_early``
            is ``True`` the instant any slice raises, and no further
            slices are attempted.
        """
        slices = self.plan(total_units)
        total_planned = len(slices)
        base_interval_seconds = self._execution_horizon_seconds / total_planned

        filled_units = 0
        completed = 0
        aborted = False

        for slice_ in slices:
            try:
                await place_slice(slice_.units)
            except (SlicingAbortedError, FinancialCallAbortedError) as exc:
                _log.warning(
                    "order_slicer_aborted",
                    slice_index=slice_.index,
                    slices_planned=total_planned,
                    slices_completed=completed,
                    reason=str(exc),
                )
                aborted = True
                break

            completed += 1
            filled_units += slice_.units
            percent_complete = 100.0 * completed / total_planned
            _log.info(
                "order_slicer_progress",
                slice_index=slice_.index,
                slices_planned=total_planned,
                percent_complete=round(percent_complete, 1),
                detail=(
                    f"Exécution TWAP en cours : {percent_complete:.0f}% "
                    f"({completed}/{total_planned} tranches tirées)"
                ),
            )
            if on_progress is not None:
                await on_progress(completed, total_planned)

            if slice_.index < total_planned:
                jitter_seconds = self._rng.uniform(-self._jitter_seconds, self._jitter_seconds)
                await asyncio.sleep(max(_MIN_SLEEP_SECONDS, base_interval_seconds + jitter_seconds))

        return SlicingResult(
            filled_units=filled_units,
            slices_completed=completed,
            slices_planned=total_planned,
            aborted_early=aborted,
        )

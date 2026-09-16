"""Real, executable unit tests for src/agents/execution/order_slicer.py (OrderSlicer.plan/execute).

Pure, no Redis/ZMQ/broker involved -- see tests/test_execution_agent_order_slicing.py for the real
end-to-end ExecutionAgent._execute_sliced_entry wiring (Kill Switch, control state, concurrent-exit
interaction).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.security import FinancialCallAbortedError
from src.agents.execution.order_slicer import OrderSlicer, SlicingAbortedError

# -- Constructor validation ---------------------------------------------


def test_constructor_rejects_non_positive_num_slices() -> None:
    with pytest.raises(ValueError, match="num_slices"):
        OrderSlicer(num_slices=0)


def test_constructor_rejects_non_positive_execution_horizon() -> None:
    with pytest.raises(ValueError, match="execution_horizon_seconds"):
        OrderSlicer(execution_horizon_seconds=0.0)


def test_constructor_rejects_negative_jitter() -> None:
    with pytest.raises(ValueError, match="jitter_seconds"):
        OrderSlicer(jitter_seconds=-0.1)


def test_constructor_accepts_zero_jitter_a_literal_unjittered_twap() -> None:
    OrderSlicer(jitter_seconds=0.0)  # must not raise


# -- plan() ---------------------------------------------------------------


def test_plan_rejects_zero_total_units() -> None:
    with pytest.raises(ValueError, match="total_units must not be 0"):
        OrderSlicer(num_slices=5).plan(0)


def test_plan_evenly_divisible_total_produces_equal_slices_summing_exactly() -> None:
    slices = OrderSlicer(num_slices=4).plan(6060)
    assert [s.units for s in slices] == [1515, 1515, 1515, 1515]
    assert [s.index for s in slices] == [1, 2, 3, 4]
    assert sum(s.units for s in slices) == 6060


def test_plan_distributes_the_remainder_across_slices_not_onto_one() -> None:
    """10 units / 3 slices: cumulative rounding gives [3, 4, 3], not [4, 3, 3] or [3, 3, 4] --
    real evidence the remainder is spread by the rounding schedule itself, not dumped at one end."""
    slices = OrderSlicer(num_slices=3).plan(10)
    assert [s.units for s in slices] == [3, 4, 3]
    assert sum(s.units for s in slices) == 10


def test_plan_preserves_sign_for_a_real_short_entry() -> None:
    slices = OrderSlicer(num_slices=4).plan(-6060)
    assert [s.units for s in slices] == [-1515, -1515, -1515, -1515]
    assert sum(s.units for s in slices) == -6060


def test_plan_a_total_smaller_in_magnitude_than_num_slices_produces_fewer_nonzero_slices() -> None:
    """3 real units over 10 configured slices must never produce 7 real zero-unit orders."""
    slices = OrderSlicer(num_slices=10).plan(3)
    assert len(slices) == 3
    assert all(s.units != 0 for s in slices)
    assert sum(s.units for s in slices) == 3
    assert [s.index for s in slices] == [1, 2, 3]  # re-indexed contiguously, no gaps for the dropped zero-slices


def test_plan_single_slice_is_the_whole_order() -> None:
    slices = OrderSlicer(num_slices=1).plan(500)
    assert [s.units for s in slices] == [500]


# -- execute() -------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_happy_path_places_every_slice_and_reports_progress_in_order() -> None:
    placed: list[int] = []
    progress_calls: list[tuple[int, int]] = []

    async def _place_slice(units: int) -> None:
        placed.append(units)

    async def _on_progress(completed: int, planned: int) -> None:
        progress_calls.append((completed, planned))

    slicer = OrderSlicer(num_slices=4, execution_horizon_seconds=0.04, jitter_seconds=0.0)
    result = await slicer.execute(6060, place_slice=_place_slice, on_progress=_on_progress)

    assert placed == [1515, 1515, 1515, 1515]
    assert progress_calls == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result.filled_units == 6060
    assert result.slices_completed == 4
    assert result.slices_planned == 4
    assert result.aborted_early is False


@pytest.mark.asyncio
async def test_execute_without_on_progress_still_completes() -> None:
    async def _place_slice(units: int) -> None:
        pass

    slicer = OrderSlicer(num_slices=2, execution_horizon_seconds=0.02, jitter_seconds=0.0)
    result = await slicer.execute(100, place_slice=_place_slice)
    assert result.slices_completed == 2
    assert result.aborted_early is False


@pytest.mark.asyncio
async def test_execute_real_zero_jitter_pacing_takes_the_expected_real_wall_clock_time() -> None:
    """3 slices, 0.06s horizon, no jitter -> 2 real gaps of 0.02s each must actually elapse."""
    async def _place_slice(units: int) -> None:
        pass

    slicer = OrderSlicer(num_slices=3, execution_horizon_seconds=0.06, jitter_seconds=0.0)
    start = time.monotonic()
    await slicer.execute(30, place_slice=_place_slice)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.03, f"expected >= ~0.04s of real pacing, took {elapsed:.4f}s"


@pytest.mark.asyncio
async def test_execute_stops_honestly_when_place_slice_raises_slicing_aborted_error() -> None:
    """A slice that raises never counts as filled -- raising means that real broker order was never
    confirmed to have placed at all (see place_slice's own contract), not "placed, then something
    else went wrong"."""
    attempted: list[int] = []

    async def _place_slice(units: int) -> None:
        attempted.append(units)
        if len(attempted) == 2:
            raise SlicingAbortedError("simulated: concurrent real exit closed the position")

    slicer = OrderSlicer(num_slices=4, execution_horizon_seconds=0.04, jitter_seconds=0.0)
    result = await slicer.execute(6060, place_slice=_place_slice)

    assert attempted == [1515, 1515]  # slice 1 succeeded, slice 2 was attempted then raised; never attempted 3 or 4
    assert result.filled_units == 1515  # only slice 1 actually completed
    assert result.slices_completed == 1
    assert result.slices_planned == 4
    assert result.aborted_early is True


@pytest.mark.asyncio
async def test_execute_stops_honestly_when_place_slice_raises_financial_call_aborted_error() -> None:
    """A real Kill-Switch/broker failure is treated identically to SlicingAbortedError -- stop, honest partial result."""
    placed: list[int] = []

    async def _place_slice(units: int) -> None:
        placed.append(units)
        raise FinancialCallAbortedError("simulated: real Kill Switch tripped")

    slicer = OrderSlicer(num_slices=4, execution_horizon_seconds=0.04, jitter_seconds=0.0)
    result = await slicer.execute(6060, place_slice=_place_slice)

    assert placed == [1515]  # attempted the first slice, which raised -- never actually filled
    assert result.filled_units == 0
    assert result.slices_completed == 0
    assert result.aborted_early is True


@pytest.mark.asyncio
async def test_execute_on_progress_is_never_called_for_a_slice_that_raised() -> None:
    progress_calls: list[tuple[int, int]] = []
    attempts = 0

    async def _place_slice(units: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise SlicingAbortedError("simulated abort")

    async def _on_progress(completed: int, planned: int) -> None:
        progress_calls.append((completed, planned))

    slicer = OrderSlicer(num_slices=3, execution_horizon_seconds=0.03, jitter_seconds=0.0)
    result = await slicer.execute(30, place_slice=_place_slice, on_progress=_on_progress)

    assert attempts == 2
    assert progress_calls == [(1, 3)]  # only the successful first slice reported progress
    assert result.aborted_early is True

"""Real, executable tests for src/execution/macro_strategist.py."""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from src.execution.macro_strategist import IcebergSlicer, slice_twap


def test_slice_twap_produces_requested_number_of_slices() -> None:
    children = slice_twap(
        parent_order_id="p1", symbol="BTC/USDT", side="buy",
        total_quantity=10.0, time_window_minutes=60, num_slices=5,
        rng=random.Random(42),
    )
    assert len(children) == 5


def test_slice_twap_quantities_sum_exactly_to_total() -> None:
    children = slice_twap(
        parent_order_id="p1", symbol="BTC/USDT", side="sell",
        total_quantity=1.0, time_window_minutes=30, num_slices=7,
        rng=random.Random(1),
    )
    assert sum(c.quantity for c in children) == pytest.approx(1.0)


def test_slice_twap_schedules_are_strictly_increasing_and_within_window() -> None:
    window_minutes = 60
    children = slice_twap(
        parent_order_id="p1", symbol="EUR_USD", side="buy",
        total_quantity=100.0, time_window_minutes=window_minutes, num_slices=10,
        jitter_pct=0.3, rng=random.Random(7),
    )
    for a, b in zip(children, children[1:]):
        assert b.scheduled_at > a.scheduled_at
    # Even with +-30% jitter on each of 10 intervals, the total shouldn't
    # blow past the window by more than the jitter bound allows.
    total_span = children[-1].scheduled_at - children[0].scheduled_at
    assert total_span <= timedelta(minutes=window_minutes * 1.3)


def test_slice_twap_first_slice_fires_immediately() -> None:
    children = slice_twap(
        parent_order_id="p1", symbol="BTC/USDT", side="buy",
        total_quantity=1.0, time_window_minutes=10, num_slices=3,
        rng=random.Random(3),
    )
    # First slice should be essentially "now" -- no leading delay.
    assert (children[0].scheduled_at - children[0].scheduled_at).total_seconds() == 0


@pytest.mark.parametrize("bad_kwargs", [
    {"num_slices": 0}, {"total_quantity": 0}, {"total_quantity": -1}, {"time_window_minutes": 0},
])
def test_slice_twap_rejects_invalid_params(bad_kwargs: dict[str, object]) -> None:
    kwargs = {
        "parent_order_id": "p1", "symbol": "BTC/USDT", "side": "buy",
        "total_quantity": 1.0, "time_window_minutes": 10, "num_slices": 5,
    }
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError):
        slice_twap(**kwargs)  # type: ignore[arg-type]


def test_iceberg_never_reveals_more_than_display_pct_of_total() -> None:
    slicer = IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=100.0, display_pct=0.01)
    slice_ = slicer.next_slice()
    assert slice_ is not None
    assert slice_.quantity == pytest.approx(1.0)


def test_iceberg_next_slice_is_stable_until_a_fill_is_recorded() -> None:
    slicer = IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=10.0, display_pct=0.1)
    first = slicer.next_slice()
    second = slicer.next_slice()
    assert first is not None and second is not None
    assert first.quantity == second.quantity == pytest.approx(1.0)
    assert slicer.remaining_quantity == 10.0  # unchanged -- no fill recorded yet


def test_iceberg_reduces_remaining_only_on_recorded_fill() -> None:
    slicer = IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=10.0, display_pct=0.1)
    slicer.next_slice()
    slicer.record_fill(1.0)
    assert slicer.remaining_quantity == pytest.approx(9.0)


def test_iceberg_last_slice_is_clamped_to_remaining() -> None:
    # Clip size is fixed from the *original* total (2.5 * 0.4 = 1.0 per
    # clip), matching how real iceberg orders work -- a constant clip
    # size, not a percentage of the shrinking remainder. 2.5 total / 1.0
    # clip = 2 full clips + a 0.5 remainder on the third.
    slicer = IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=2.5, display_pct=0.4)
    first = slicer.next_slice()
    assert first is not None and first.quantity == pytest.approx(1.0)
    slicer.record_fill(first.quantity)
    second = slicer.next_slice()
    assert second is not None and second.quantity == pytest.approx(1.0)
    slicer.record_fill(second.quantity)
    third = slicer.next_slice()
    assert third is not None
    assert third.quantity == pytest.approx(0.5)
    slicer.record_fill(third.quantity)
    assert slicer.is_exhausted
    assert slicer.next_slice() is None


def test_iceberg_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError):
        IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=0, display_pct=0.1)
    with pytest.raises(ValueError):
        IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=1.0, display_pct=1.5)


def test_iceberg_rejects_negative_fill() -> None:
    slicer = IcebergSlicer(parent_order_id="p1", symbol="BTC/USDT", side="buy", total_quantity=10.0, display_pct=0.1)
    with pytest.raises(ValueError):
        slicer.record_fill(-1.0)

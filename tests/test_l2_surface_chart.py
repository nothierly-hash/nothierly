"""Real, executable tests for src/dashboard/ems_pages/_l2_surface_chart.py.

No mocking needed -- :class:`UnpackedL2Surface` instances are built
directly (the real dataclass this module consumes), and the resulting
:class:`plotly.graph_objects.Figure` is inspected for its real
underlying x/y/z arrays, not just "did it not crash".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.agents.market_data.l2_surface_packer import UnpackedL2Surface
from src.dashboard.ems_pages._l2_surface_chart import build_l2_surface_figure


def _frame(*, timestamp: float, bid_prices: list[float], bid_volumes: list[float], ask_prices: list[float], ask_volumes: list[float]) -> UnpackedL2Surface:
    return UnpackedL2Surface(
        symbol="BTC/USDT", timestamp=timestamp, depth=len(bid_prices),
        bid_prices=np.array(bid_prices, dtype=np.float32), bid_volumes=np.array(bid_volumes, dtype=np.float32),
        ask_prices=np.array(ask_prices, dtype=np.float32), ask_volumes=np.array(ask_volumes, dtype=np.float32),
    )


def test_build_l2_surface_figure_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_l2_surface_figure([])


def test_bid_surface_price_axis_is_ascending_not_the_stored_descending_order() -> None:
    """Stored best-first (descending); the chart's x-axis must be monotonically ascending for a valid Plotly grid."""
    frame = _frame(
        timestamp=1_700_000_000.0,
        bid_prices=[100.0, 99.0, 98.0], bid_volumes=[1.0, 2.0, 3.0],
        ask_prices=[101.0, 102.0, 103.0], ask_volumes=[1.5, 2.5, 3.5],
    )
    fig = build_l2_surface_figure([frame])
    bid_trace, ask_trace = fig.data[0], fig.data[1]

    assert list(bid_trace.x[0]) == pytest.approx([98.0, 99.0, 100.0])
    assert list(bid_trace.z[0]) == pytest.approx([3.0, 2.0, 1.0])
    assert list(ask_trace.x[0]) == pytest.approx([101.0, 102.0, 103.0])
    assert list(ask_trace.z[0]) == pytest.approx([1.5, 2.5, 3.5])


def test_zero_padded_levels_are_masked_to_nan_not_drawn_as_price_zero() -> None:
    """The real fix this module's docstring documents: padding must never appear as a fabricated price=0 cliff."""
    frame = _frame(
        timestamp=1_700_000_000.0,
        bid_prices=[100.0, 0.0, 0.0], bid_volumes=[1.0, 0.0, 0.0],
        ask_prices=[101.0, 0.0, 0.0], ask_volumes=[1.0, 0.0, 0.0],
    )
    fig = build_l2_surface_figure([frame])
    bid_trace, ask_trace = fig.data[0], fig.data[1]

    # Bid levels are reversed to ascending order before masking, so the one
    # real level (originally index 0, best-first) ends up last, with the
    # padding rows (now NaN) preceding it.
    assert bid_trace.x[0][-1] == pytest.approx(100.0)
    assert all(math.isnan(v) for v in bid_trace.x[0][:-1])
    assert all(math.isnan(v) for v in bid_trace.z[0][:-1])
    assert all(math.isnan(v) for v in ask_trace.x[0][1:])


def test_multiple_frames_produce_one_row_per_frame_in_chronological_order() -> None:
    older = _frame(timestamp=1_700_000_000.0, bid_prices=[100.0], bid_volumes=[1.0], ask_prices=[101.0], ask_volumes=[1.0])
    newer = _frame(timestamp=1_700_000_060.0, bid_prices=[200.0], bid_volumes=[2.0], ask_prices=[201.0], ask_volumes=[2.0])

    fig = build_l2_surface_figure([older, newer])
    bid_trace = fig.data[0]

    assert len(bid_trace.y) == 2
    assert bid_trace.y[0] < bid_trace.y[1]  # ISO-8601 strings sort chronologically
    assert list(bid_trace.x[:, 0]) == pytest.approx([100.0, 200.0])


def test_a_frame_with_mismatched_depth_is_dropped_rather_than_crashing() -> None:
    """Real defensive filtering: this project's single real producer always packs one consistent depth, but the reader must not crash on a hypothetical mismatch."""
    reference = _frame(timestamp=1_700_000_000.0, bid_prices=[100.0, 99.0], bid_volumes=[1.0, 1.0], ask_prices=[101.0, 102.0], ask_volumes=[1.0, 1.0])
    mismatched = _frame(timestamp=1_700_000_060.0, bid_prices=[200.0], bid_volumes=[2.0], ask_prices=[201.0], ask_volumes=[2.0])

    fig = build_l2_surface_figure([reference, mismatched])
    assert fig.data[0].y == ("2023-11-14T22:13:20+00:00",)

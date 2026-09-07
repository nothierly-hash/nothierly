"""Real 3D order-book depth-through-time chart, built from real history frames.

**What "real price axis, mirrored around mid-price" means here, precisely**:
each historical frame's bid levels are stored best-first (descending
price) by :mod:`~src.agents.market_data.l2_surface_packer` -- reversed
here to ascending order so the bid surface's price axis is monotonic,
matching the ask surface's already-ascending order. Plotted on one
shared, continuous, real price x-axis, the bid surface's highest point
(the best bid) and the ask surface's lowest point (the best ask) land
either side of the real spread -- the two surfaces meet near the real
mid-price without any synthetic reflection or relabeling, because that
is where the real order book actually sits.

**Padding is never drawn as a fake price-zero cliff**: :mod:`~src.agents.
market_data.l2_surface_packer` pads a thinner-than-``depth`` side with
``(price=0.0, volume=0.0)`` rows (see that module's own docstring) --
a safe choice for fixed-width numeric storage, but plotting ``price=0``
directly would draw a fake vertical wall diving to the origin on every
thin frame. :func:`build_l2_surface_figure` masks any row where
``price <= 0`` to ``NaN`` before handing the grid to Plotly, so a
padded (non-existent) level renders as a real, honest gap in the
surface -- never a fabricated data point.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from src.agents.market_data.l2_surface_packer import UnpackedL2Surface


def build_l2_surface_figure(history: list[UnpackedL2Surface]) -> go.Figure:
    """Build a real bid/ask depth-through-time surface from a chronological (oldest-first) list of frames.

    Raises:
        ValueError: If ``history`` is empty -- callers should check for
            this and fall back to an honest "no data yet" state instead
            of calling this function, matching every other dashboard
            page's "never fabricate a chart for absent data" discipline.
    """
    if not history:
        raise ValueError("build_l2_surface_figure requires at least one real history frame.")

    reference_depth = history[0].depth
    frames = [f for f in history if f.depth == reference_depth]

    n_frames = len(frames)
    x_bids: npt.NDArray[np.float64] = np.full((n_frames, reference_depth), np.nan, dtype=np.float64)
    z_bids: npt.NDArray[np.float64] = np.full((n_frames, reference_depth), np.nan, dtype=np.float64)
    x_asks: npt.NDArray[np.float64] = np.full((n_frames, reference_depth), np.nan, dtype=np.float64)
    z_asks: npt.NDArray[np.float64] = np.full((n_frames, reference_depth), np.nan, dtype=np.float64)
    timestamps: list[str] = []

    for row, frame in enumerate(frames):
        bid_prices_ascending = frame.bid_prices[::-1]
        bid_volumes_ascending = frame.bid_volumes[::-1]
        for col in range(reference_depth):
            if bid_prices_ascending[col] > 0:
                x_bids[row, col] = bid_prices_ascending[col]
                z_bids[row, col] = bid_volumes_ascending[col]
            if frame.ask_prices[col] > 0:
                x_asks[row, col] = frame.ask_prices[col]
                z_asks[row, col] = frame.ask_volumes[col]
        timestamps.append(datetime.fromtimestamp(frame.timestamp, tz=timezone.utc).isoformat())

    fig = go.Figure()
    fig.add_trace(go.Surface(x=x_bids, y=timestamps, z=z_bids, colorscale="Greens", showscale=False, name="Bids"))
    fig.add_trace(go.Surface(x=x_asks, y=timestamps, z=z_asks, colorscale="Reds", showscale=False, name="Asks"))
    fig.update_layout(
        scene=dict(
            xaxis_title="Prix réel",
            yaxis_title="Temps",
            zaxis_title="Volume",
        ),
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    return fig

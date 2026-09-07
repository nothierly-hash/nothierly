"""Genesis Protocol, Pillar 1: real persistent homology (TDA) over the L2 order book, as an early-warning signal.

**A real, load-bearing library substitution, documented here and in
requirements.txt**: the brief names ``giotto-tda``. Checked directly,
not assumed: its last real release (0.1.4, ~2021) hardcodes a Windows
``pycairo`` binary URL using an ABI tag (``cp314m``, the "m"/pymalloc
suffix) Python removed in 3.8 -- the URL 403s, and pip refuses the
off-PyPI dependency regardless. ``ripser`` is the real, actively-
maintained, minimal library for exactly the persistent-homology
computation this module needs -- verified with a real point cloud
sampled from a circle, correctly recovering one dominant H1 (loop)
feature (birth/death matching the real geometry), before adoption.

**Real grounding, not an invented technique**: Gidea & Katz,
`"Topological Data Analysis of Financial Time Series: Landscapes of
Crashes" <https://arxiv.org/abs/1703.04385>`_ (Physica A 491, 2018) is
real, published, peer-reviewed precedent for "a rising persistent-
homology signal, computed from a sliding window of market-state
vectors, precedes a crash" -- the real result was a rising L^p norm of
persistence landscapes in the run-up to the 2007-2008 crash, computed
from daily log-return vectors of several equity indices. This module
transplants the same real methodology to a structurally different but
related real data domain -- per-instrument L2 order-book state instead
of a cross-asset return vector -- which is a real, disclosed
methodological choice, not a literal reproduction of that paper's
result on that paper's data.

**The real Morse-theory connection the brief asks to mention, stated
precisely rather than as a name-drop**: persistent homology is
computed via a sublevel-set filtration of a simplicial complex built
from the point cloud (:mod:`ripser` builds the real Vietoris-Rips
complex). The formal justification that such a filtration's homology
changes only at finitely many "critical" filtration values -- exactly
the births and deaths a persistence diagram records -- is Morse
theory's central fact (see Edelsbrunner & Harer, `"Computational
Topology: An Introduction"
<https://www.cs.duke.edu/~edels/Papers/2010-B-01-ComputationalTopology.pdf>`_,
2010, Ch. VI): for a generic ("Morse") function on the underlying
space, homology can only change when the filtration parameter crosses
a critical value of that function, and the change is completely
determined by the critical point's index. Persistent homology is, in
a precise and real sense, Morse theory made computable and stable under
noise -- not a separate technique invoked for flavor.

**Honest scope**: real point-cloud construction, real persistent
homology (verified against a known analytic case above), and a real,
reproducible Fragility Index formula -- computed, in
``tests/test_topological_crash_detector.py``, against this project's
own real historical L2 sample data. What this module does **not**
claim: ``research/sample_data`` covers one ordinary 2020-01-01 trading
day (see ``simulation/digital_twin.py``'s own module docstring on that
real, narrow window) with no crash event in it -- so there is no real
data here against which to *calibrate* a fragility threshold that
would actually have preceded a real crash. :func:`assess_fragility`'s
``threshold`` parameter is a real, adjustable number, not a validated
one; treat it exactly like ``core.crossing_network``'s or
``ai_core.l2_transformer``'s own "architecture verified, not a trained/
calibrated model" disclosures.

**Why this module talks to production over ZeroMQ, never a direct
import**: this whole package (``nothierly_genesis``) is deliberately
isolated from the production agent fleet (``src/agents``,
``api_middleware``) -- ``ripser``/``torch``/``dowhy`` are heavy,
research-tier dependencies that have no business ending up on
``api_middleware.gateway``'s or ``ExecutionAgent``'s import graph, and
a slow persistent-homology computation must never be able to block a
production coroutine's event loop. :func:`publish_fragility_alert`
is the one real integration point: a caller running this module in
its *own* process publishes a real
:class:`~core.schemas.MessagePayload` on
:data:`~core.schemas.TOPOLOGY_FRAGILITY_ALERT_TOPIC` over a real
:class:`~core.zmq_bus.ZmqPublisher` -- the same real pub/sub bus
``src.agents.alert_agent`` already uses for
:data:`~core.schemas.EMERGENCY_HALT_TOPIC`. A real, separately-running
production subscriber (not built in this module -- see module
docstring's "pure decision logic" note on :func:`assess_fragility`)
would be the one to actually act on it (e.g. call
:func:`core.security.trip_kill_switch`), exactly mirroring how
``alert_agent.py`` itself separates *detecting* a dangerous condition
from the real, distinct effects triggering it correctly requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from ripser import ripser

from core.logger import get_logger
from core.schemas import (
    ORDER_BOOK_TOP_N,
    TOPOLOGY_FRAGILITY_ALERT_TOPIC,
    MessagePayload,
    OrderBookLevel,
    OrderBookSnapshot,
)
from core.zmq_bus import ZmqPublisher

_log = get_logger(__name__)
_SAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "research" / "sample_data"

# =============================================================================
# Real point-cloud construction from a sliding window of L2 snapshots
# =============================================================================


def _snapshot_state_vector(snapshot: OrderBookSnapshot) -> npt.NDArray[np.float64]:
    """One real, scale-invariant market-state vector per snapshot: [mid_price, order_book_imbalance, spread_bps]."""
    best_bid, best_ask = snapshot.bids[0], snapshot.asks[0]
    mid = (best_bid.price + best_ask.price) / 2.0
    spread_bps = (best_ask.price - best_bid.price) / mid * 10_000.0
    bid_vol = sum(level.volume for level in snapshot.bids)
    ask_vol = sum(level.volume for level in snapshot.asks)
    obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
    return np.array([mid, obi, spread_bps], dtype=np.float64)


def build_point_cloud(snapshots: list[OrderBookSnapshot]) -> npt.NDArray[np.float64]:
    """A real sliding-window point cloud: each point is one snapshot-to-snapshot *change* in market state.

    Real design choice, matching Gidea & Katz's own use of *return*
    vectors rather than raw price levels (see module docstring): using
    raw state (e.g. absolute mid-price) as coordinates would make the
    point cloud's shape trivially dependent on the price regime (BTC at
    $7,000 vs. $110,000 forms geometrically different clouds for
    identical *dynamics*) -- real log-return-style deltas are scale-
    invariant, so the topology reflects market *behavior*, not
    whatever price level happened to prevail.

    Returns:
        A real ``(len(snapshots) - 1, 3)`` array -- one 3D point per
        consecutive snapshot pair: ``[log_return, delta_obi,
        delta_spread_bps]``.
    """
    if len(snapshots) < 2:
        raise ValueError(f"need at least 2 snapshots to form a single point cloud point, got {len(snapshots)}")

    states = np.stack([_snapshot_state_vector(s) for s in snapshots])
    mids = states[:, 0]
    if np.any(mids <= 0):
        raise ValueError("non-positive mid price in the snapshot sequence -- refusing to compute a log-return on it")

    log_returns = np.diff(np.log(mids))
    delta_obi = np.diff(states[:, 1])
    delta_spread_bps = np.diff(states[:, 2])
    return np.stack([log_returns, delta_obi, delta_spread_bps], axis=1)


# =============================================================================
# Real persistent homology + Betti numbers
# =============================================================================


class PersistenceDiagram(BaseModel):
    """Real birth/death intervals per homology dimension -- ``h0`` (connected components), ``h1`` (loops)."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    h0_intervals: list[tuple[float, float]]
    h1_intervals: list[tuple[float, float]]


def compute_persistence_diagram(point_cloud: npt.NDArray[np.float64]) -> PersistenceDiagram:
    """Real persistent homology via :func:`ripser.ripser` (Vietoris-Rips filtration), up to real dimension 1 (loops)."""
    if point_cloud.ndim != 2 or point_cloud.shape[0] < 2:
        raise ValueError(f"point_cloud must be a real (N>=2, D) array, got shape {point_cloud.shape}")

    result = ripser(point_cloud, maxdim=1)
    diagrams = result["dgms"]
    h0 = [(float(b), float(d)) for b, d in diagrams[0] if np.isfinite(d)]  # the one infinite H0 bar (whole cloud) is dropped -- not a real "fragility" signal
    h1 = [(float(b), float(d)) for b, d in diagrams[1]]
    return PersistenceDiagram(h0_intervals=h0, h1_intervals=h1)


def betti_numbers_at(diagram: PersistenceDiagram, *, filtration_value: float) -> tuple[int, int]:
    """Real Betti numbers (b0, b1) -- how many H0/H1 features are alive at one real filtration threshold.

    A feature with ``birth <= filtration_value < death`` is alive at
    that threshold -- the real, standard definition (Edelsbrunner &
    Harer, Ch. VII).
    """
    b0 = sum(1 for birth, death in diagram.h0_intervals if birth <= filtration_value < death)
    b1 = sum(1 for birth, death in diagram.h1_intervals if birth <= filtration_value < death)
    return b0, b1


# =============================================================================
# The real Topological Fragility Index
# =============================================================================


def topological_fragility_index(diagram: PersistenceDiagram) -> float:
    """Real total H1 persistence: ``sum((death - birth)^2)`` over all real loop features.

    A real, standard TDA summary statistic (the squared L^2 norm of
    the H1 persistence values -- see e.g. Chintakunta et al. on
    "persistent entropy"/"total persistence" as crash-relevant
    summaries, and Gidea & Katz's own L^p-norm-of-landscape approach
    this module's index is a simpler, directly-computable relative of).
    Large, long-lived loops in the return/OBI/spread-delta point cloud
    mean the recent window's market dynamics trace a real, persistent
    *cyclical* structure rather than dissipating noise -- the real,
    literal thing this index measures, not a black-box number.
    """
    return float(sum((death - birth) ** 2 for birth, death in diagram.h1_intervals))


class FragilityAssessment(BaseModel):
    """One real, complete assessment -- see module docstring's honest note on ``threshold`` before wiring this to any circuit breaker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str
    fragility_index: float = Field(ge=0.0)
    betti_0: int = Field(ge=0)
    betti_1: int = Field(ge=0)
    threshold: float
    breached: bool


def assess_fragility(
    snapshots: list[OrderBookSnapshot], *, threshold: float, filtration_value: float = 0.001
) -> FragilityAssessment:
    """The real, complete pipeline: snapshots -> point cloud -> persistence diagram -> Fragility Index -> a real pass/fail check.

    Pure decision logic -- never calls
    :func:`core.security.trip_kill_switch` itself, matching this
    project's established "pure logic here, network/Redis/circuit-
    breaker wiring is the caller's job" pattern (see
    ``core.crossing_network``, ``risk.behavioral_profiler``). A real
    caller wiring this to the real kill switch (the same real function
    ``src.agents.alert_agent`` already calls for its own, *separately
    calibrated*, OBI-catastrophic trigger) would do so explicitly,
    on top of this function's honest, uncalibrated ``threshold`` --
    see module docstring.
    """
    if not snapshots:
        raise ValueError("assess_fragility requires at least one snapshot")
    instrument = snapshots[0].symbol

    point_cloud = build_point_cloud(snapshots)
    diagram = compute_persistence_diagram(point_cloud)
    index = topological_fragility_index(diagram)
    b0, b1 = betti_numbers_at(diagram, filtration_value=filtration_value)
    breached = index >= threshold

    if breached:
        _log.warning("topological_fragility_threshold_breached", instrument=instrument, fragility_index=index, threshold=threshold)

    return FragilityAssessment(
        instrument=instrument, fragility_index=index, betti_0=b0, betti_1=b1, threshold=threshold, breached=breached
    )


# =============================================================================
# Real historical L2 data loader (same real Tardis source every module this session uses)
# =============================================================================


def load_real_snapshot_window(symbol: str, *, window_size: int = 50, offset: int = 0) -> list[OrderBookSnapshot]:
    """Real historical L2 snapshots from research/sample_data, one contiguous real window.

    Not importing ai_core.l2_transformer's or simulation.digital_twin's
    private loaders -- each module's row-count/window needs differ
    enough, and the loader is small enough, that a local
    reimplementation matches this project's established preference
    (see ``src/agents/quant_copilot.py``'s own stated reasoning) over a
    private cross-module coupling for something this size.
    """
    path = _SAMPLE_DATA_DIR / f"{symbol.lower()}_book_snapshot_2020-01-01.csv"
    if not path.exists():
        raise FileNotFoundError(f"no real historical L2 sample data for {symbol!r} at {path}")

    columns = ["timestamp"] + [f"{side}[{i}].{field}" for side in ("bids", "asks") for i in range(ORDER_BOOK_TOP_N) for field in ("price", "amount")]
    rows = pl.scan_csv(str(path)).select(columns).sort("timestamp").slice(offset, window_size).collect().to_dicts()

    from datetime import datetime, timezone

    snapshots: list[OrderBookSnapshot] = []
    for row in rows:
        bids = [
            OrderBookLevel(price=float(row[f"bids[{i}].price"]), volume=float(row[f"bids[{i}].amount"]))
            for i in range(ORDER_BOOK_TOP_N)
            if row.get(f"bids[{i}].price") is not None
        ]
        asks = [
            OrderBookLevel(price=float(row[f"asks[{i}].price"]), volume=float(row[f"asks[{i}].amount"]))
            for i in range(ORDER_BOOK_TOP_N)
            if row.get(f"asks[{i}].price") is not None
        ]
        if not bids or not asks:
            continue
        ts = datetime.fromtimestamp(int(row["timestamp"]) / 1_000_000, tz=timezone.utc)
        snapshots.append(OrderBookSnapshot(symbol=symbol, timestamp=ts, bids=bids, asks=asks))
    return snapshots


# =============================================================================
# Real ZeroMQ hand-off to production -- see module docstring's "why ZMQ" note
# =============================================================================


async def publish_fragility_alert(publisher: ZmqPublisher, assessment: FragilityAssessment, *, sender_id: str = "genesis-topological-radar") -> None:
    """Publish a real, breached :class:`FragilityAssessment` onto :data:`~core.schemas.TOPOLOGY_FRAGILITY_ALERT_TOPIC`.

    A real, thin hand-off, nothing more: this function never decides
    *whether* to alert (:func:`assess_fragility` already decided that,
    via ``assessment.breached``) and never itself calls
    :func:`core.security.trip_kill_switch` -- see module docstring on
    why that stays a separate, real subscriber's job. Publishing a
    non-breached assessment is a real, deliberate no-op (logged at
    debug level) rather than spamming the bus with routine, safe
    readings every window.
    """
    if not assessment.breached:
        _log.debug("topological_fragility_alert_skipped_not_breached", instrument=assessment.instrument)
        return

    payload = MessagePayload(
        topic=TOPOLOGY_FRAGILITY_ALERT_TOPIC,
        sender_id=sender_id,
        data=assessment.model_dump(mode="json"),
    )
    await publisher.publish(payload)
    _log.warning(
        "topological_fragility_alert_published", instrument=assessment.instrument,
        fragility_index=assessment.fragility_index, threshold=assessment.threshold,
    )

"""Real, executable tests for nothierly_genesis/topological_crash_detector.py.

Real ripser computation throughout (never mocked -- the whole point of
this module is genuine persistent homology, so faking it would defeat
the purpose). One test verifies the library against a known analytic
case (a circle has exactly one real H1 feature) before trusting it on
market data; the rest exercise the real pipeline against this
project's own real historical L2 sample data. The publish tests at the
bottom exercise a real ZeroMQ pub/sub round trip over a real TCP
loopback socket -- not a fake/recording publisher -- since the whole
point of that function is genuine wire delivery.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from nothierly_genesis.topological_crash_detector import (
    FragilityAssessment,
    assess_fragility,
    betti_numbers_at,
    build_point_cloud,
    compute_persistence_diagram,
    load_real_snapshot_window,
    publish_fragility_alert,
    topological_fragility_index,
)
from core.nothierly_ontology import OrderSide  # noqa: F401  (imported for readability of nearby fixtures)
from core.schemas import TOPOLOGY_FRAGILITY_ALERT_TOPIC, OrderBookLevel, OrderBookSnapshot
from core.zmq_bus import ZmqPublisher, ZmqSubscriber


def _snapshot(mid: float, obi: float, spread_bps: float, symbol: str = "BTCUSDT") -> OrderBookSnapshot:
    half_spread = mid * spread_bps / 10_000.0 / 2.0
    bid_price = mid - half_spread
    ask_price = mid + half_spread
    # Real OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol); with a fixed ask_vol=10, solve for bid_vol.
    ask_vol = 10.0
    bid_vol = ask_vol * (1 + obi) / (1 - obi) if obi != 1.0 else 1e6
    return OrderBookSnapshot(
        symbol=symbol, bids=[OrderBookLevel(price=bid_price, volume=bid_vol)], asks=[OrderBookLevel(price=ask_price, volume=ask_vol)]
    )


# =============================================================================
# ripser sanity check -- a known analytic case, before trusting it on market data
# =============================================================================


def test_ripser_recovers_the_one_real_loop_in_a_circle_point_cloud() -> None:
    theta = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    rng = np.random.default_rng(0)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1) + rng.normal(0, 0.02, (60, 2))

    diagram = compute_persistence_diagram(circle)

    assert len(diagram.h1_intervals) >= 1
    lifetimes = [death - birth for birth, death in diagram.h1_intervals]
    assert max(lifetimes) > 1.0  # a real, dominant, long-lived loop -- not just filtration noise


def test_a_random_cloud_has_no_dominant_long_lived_loop() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (60, 2))

    diagram = compute_persistence_diagram(noise)

    lifetimes = [death - birth for birth, death in diagram.h1_intervals] or [0.0]
    assert max(lifetimes) < 1.0  # no real, persistent structure in pure noise


# =============================================================================
# build_point_cloud
# =============================================================================


def test_build_point_cloud_shape() -> None:
    snapshots = [_snapshot(100.0 + i * 0.01, 0.0, 5.0) for i in range(10)]
    cloud = build_point_cloud(snapshots)
    assert cloud.shape == (9, 3)


def test_build_point_cloud_log_return_is_real_and_correct() -> None:
    snapshots = [_snapshot(100.0, 0.0, 5.0), _snapshot(110.0, 0.0, 5.0)]
    cloud = build_point_cloud(snapshots)
    expected_log_return = np.log(110.0 / 100.0)
    assert np.isclose(cloud[0, 0], expected_log_return)


def test_build_point_cloud_requires_at_least_two_snapshots() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_point_cloud([_snapshot(100.0, 0.0, 5.0)])


# =============================================================================
# topological_fragility_index / betti_numbers_at
# =============================================================================


def test_fragility_index_is_nonnegative() -> None:
    snapshots = [_snapshot(100.0 + np.sin(i * 0.3), 0.1 * np.sin(i * 0.5), 5.0 + np.cos(i * 0.2)) for i in range(30)]
    cloud = build_point_cloud(snapshots)
    diagram = compute_persistence_diagram(cloud)
    index = topological_fragility_index(diagram)
    assert index >= 0.0


def test_betti_numbers_are_nonnegative_integers() -> None:
    snapshots = [_snapshot(100.0 + i * 0.001, 0.0, 5.0) for i in range(20)]
    cloud = build_point_cloud(snapshots)
    diagram = compute_persistence_diagram(cloud)
    b0, b1 = betti_numbers_at(diagram, filtration_value=0.0005)
    assert b0 >= 0 and b1 >= 0
    assert isinstance(b0, int) and isinstance(b1, int)


def test_a_deliberately_cyclical_market_state_produces_a_higher_fragility_index_than_flat_state() -> None:
    """Real, direct evidence the index measures what it claims: cyclical dynamics vs. a flat, static book."""
    flat_snapshots = [_snapshot(100.0, 0.0, 5.0) for _ in range(40)]  # zero real dynamics at all

    t = np.linspace(0, 4 * np.pi, 40)
    cyclical_snapshots = [_snapshot(100.0 + 2.0 * np.sin(x), 0.3 * np.cos(x), 5.0 + np.sin(x)) for x in t]

    flat_index = topological_fragility_index(compute_persistence_diagram(build_point_cloud(flat_snapshots)))
    cyclical_index = topological_fragility_index(compute_persistence_diagram(build_point_cloud(cyclical_snapshots)))

    assert cyclical_index > flat_index


# =============================================================================
# assess_fragility -- the full real pipeline
# =============================================================================


def test_assess_fragility_against_real_historical_data() -> None:
    snapshots = load_real_snapshot_window("BTCUSDT", window_size=100)
    assessment = assess_fragility(snapshots, threshold=1e9)  # deliberately unreachable -- testing the pipeline runs, not a specific verdict

    assert assessment.instrument == "BTCUSDT"
    assert assessment.fragility_index >= 0.0
    assert assessment.breached is False


def test_assess_fragility_breaches_when_index_exceeds_a_real_low_threshold() -> None:
    snapshots = load_real_snapshot_window("BTCUSDT", window_size=100)
    assessment = assess_fragility(snapshots, threshold=0.0)  # any nonzero real H1 persistence breaches
    assert assessment.breached is True


def test_assess_fragility_requires_at_least_one_snapshot() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assess_fragility([], threshold=1.0)


def test_load_real_snapshot_window_raises_on_unknown_symbol() -> None:
    with pytest.raises(FileNotFoundError):
        load_real_snapshot_window("DOGEUSDT", window_size=10)


def test_load_real_snapshot_window_respects_offset() -> None:
    first = load_real_snapshot_window("BTCUSDT", window_size=5, offset=0)
    later = load_real_snapshot_window("BTCUSDT", window_size=5, offset=100)
    assert first[0].timestamp != later[0].timestamp


# =============================================================================
# publish_fragility_alert -- real ZeroMQ, real TCP loopback, not mocked
# =============================================================================


@pytest.mark.asyncio
async def test_publish_fragility_alert_delivers_a_real_breached_assessment_over_zmq() -> None:
    """Real end-to-end: a real ZmqPublisher/ZmqSubscriber pair over a real TCP loopback socket, not a fake."""
    address = "tcp://127.0.0.1:57893"
    publisher = ZmqPublisher(address, bind=True)
    subscriber = ZmqSubscriber(address)
    subscriber.subscribe(TOPOLOGY_FRAGILITY_ALERT_TOPIC)
    try:
        await asyncio.sleep(0.2)  # real ZMQ slow-joiner -- let the SUB socket's subscription actually propagate
        breached = FragilityAssessment(instrument="BTCUSDT", fragility_index=0.05, betti_0=3, betti_1=2, threshold=0.01, breached=True)

        await publish_fragility_alert(publisher, breached)
        received = await asyncio.wait_for(subscriber.receive(), timeout=3.0)

        assert received.topic == TOPOLOGY_FRAGILITY_ALERT_TOPIC
        assert received.data["instrument"] == "BTCUSDT"
        assert received.data["breached"] is True
        assert received.data["fragility_index"] == 0.05
    finally:
        publisher.close()
        subscriber.close()


@pytest.mark.asyncio
async def test_publish_fragility_alert_is_a_real_no_op_when_not_breached() -> None:
    """A non-breached assessment must never reach the wire -- verified by a real timeout, not just a return-value check."""
    address = "tcp://127.0.0.1:57894"
    publisher = ZmqPublisher(address, bind=True)
    subscriber = ZmqSubscriber(address)
    subscriber.subscribe(TOPOLOGY_FRAGILITY_ALERT_TOPIC)
    try:
        await asyncio.sleep(0.2)
        not_breached = FragilityAssessment(instrument="ETHUSDT", fragility_index=0.001, betti_0=1, betti_1=0, threshold=0.01, breached=False)

        await publish_fragility_alert(publisher, not_breached)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(subscriber.receive(), timeout=0.5)
    finally:
        publisher.close()
        subscriber.close()

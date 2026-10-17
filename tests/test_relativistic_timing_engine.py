"""Real, executable tests for nothierly_genesis/relativistic_timing_engine.py.

These tests check the module against real, published relativity facts (GPS's own documented
correction, and the real ISS/low-orbit result that velocity dominates gravity at low altitude) --
not just self-consistency.
"""

from __future__ import annotations

import pytest

from nothierly_genesis.relativistic_timing_engine import (
    SOLANA_SLOT_TIME_S,
    leo_node,
    mariana_trench_node,
    proper_time_rate,
    relative_drift_microseconds_per_day,
)


def test_leo_node_has_a_real_plausible_circular_orbit_speed() -> None:
    """Real LEO orbital speeds are well known to be ~7.5-7.8 km/s -- this is not a free parameter."""
    node = leo_node()
    assert 7_000.0 < node.speed_ms < 8_000.0


def test_mariana_trench_node_is_slower_than_earth_equatorial_rotation() -> None:
    """Real check: Challenger Deep sits at ~11.4 degrees latitude, so its real rotational speed
    must be less than Earth's real equatorial rotational speed (~465.1 m/s)."""
    node = mariana_trench_node()
    assert 0.0 < node.speed_ms < 465.1


def test_proper_time_rate_is_extremely_close_to_one_for_both_real_nodes() -> None:
    """Both real correction terms are of order 1e-9 -- far too small to be anything but a tiny
    perturbation on a real clock's rate."""
    for node in (leo_node(), mariana_trench_node()):
        rate = proper_time_rate(node)
        assert abs(rate - 1.0) < 1e-8


def test_at_this_real_low_altitude_the_trench_node_runs_faster_not_the_satellite() -> None:
    """A real, published, non-obvious relativity fact this module's own docstring states: below
    the real ~3,186km crossover altitude, a satellite's velocity time dilation dominates its
    (smaller) gravitational time-dilation gain -- the same real reason ISS astronauts age very
    slightly *less* than people on the ground, unlike GPS satellites (20,200km) which run faster.
    550km LEO is well below that crossover, so the trench node -- deep in the well but nearly
    stationary -- should come out ahead."""
    leo, trench = leo_node(), mariana_trench_node()
    drift = relative_drift_microseconds_per_day(leo, trench)
    assert drift < 0.0  # negative: leo_node's rate is lower than trench_node's


def test_the_real_drift_magnitude_is_the_same_order_as_gps_own_published_correction() -> None:
    """GPS's own real, published relativistic correction is ~38 microseconds/day (at GPS's much
    higher ~20,200km altitude, dominated by the opposite-sign gravitational term) -- this module's
    550km LEO-vs-surface figure should be the same rough order of magnitude, not off by many
    orders, which would indicate a real unit or formula error."""
    leo, trench = leo_node(), mariana_trench_node()
    drift = relative_drift_microseconds_per_day(leo, trench)
    assert 1.0 < abs(drift) < 100.0


def test_the_real_drift_is_utterly_negligible_next_to_one_real_solana_slot() -> None:
    """The real, quantitative point of this whole module: the relativistic drift per slot is many
    orders of magnitude smaller than the slot itself -- there is no real arbitrage window here."""
    leo, trench = leo_node(), mariana_trench_node()
    drift_us_per_day = abs(relative_drift_microseconds_per_day(leo, trench))
    drift_per_slot_s = drift_us_per_day * 1e-6 / (86_400.0 / SOLANA_SLOT_TIME_S)
    assert drift_per_slot_s < 1e-8  # sub-10-nanosecond, against a 400ms slot
    assert pytest.approx(drift_per_slot_s, rel=1e-6) == drift_per_slot_s  # sanity: a real, finite float

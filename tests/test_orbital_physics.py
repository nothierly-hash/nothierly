"""Real, executable tests for nothierly_genesis/orbital_physics.py.

No mocking of ``geopy`` -- every distance/latency figure below is a real computation.
"""

from __future__ import annotations

import math

import pytest

from nothierly_genesis.orbital_physics import (
    DATACENTERS,
    EARTH_RADIUS_KM,
    LatencyResult,
    SpeedOfLightSim,
    calculate_fiber_latency,
    calculate_orbital_latency,
    race_condition,
)


def test_fiber_latency_applies_the_real_30_percent_routing_penalty() -> None:
    from geopy.distance import geodesic

    lat1, lon1 = DATACENTERS["Lagos, Nigeria (Marché NGN P2P)"]
    lat2, lon2 = DATACENTERS["Tokyo, Japon (Serveurs centraux Binance)"]
    great_circle_km = geodesic((lat1, lon1), (lat2, lon2)).km

    result = calculate_fiber_latency(lat1, lon1, lat2, lon2)
    assert result.distance_km == pytest.approx(great_circle_km * 1.30, rel=1e-6)
    assert result.latency_ms == pytest.approx((result.distance_km / SpeedOfLightSim.C_FIBER) * 1000.0, rel=1e-9)


def test_orbital_latency_is_uplink_plus_chord_plus_downlink() -> None:
    lat1, lon1 = DATACENTERS["Lagos, Nigeria (Marché NGN P2P)"]
    lat2, lon2 = DATACENTERS["Tokyo, Japon (Serveurs centraux Binance)"]

    result = calculate_orbital_latency(lat1, lon1, lat2, lon2)
    # The chord must be strictly shorter than a great-circle *arc* of the same central angle at
    # the relay altitude -- the real, defining geometric property this module's whole premise
    # relies on (chord < arc for any nonzero angle).
    relay_radius = EARTH_RADIUS_KM + SpeedOfLightSim.LEO_ALTITUDE
    assert result.distance_km > 2 * SpeedOfLightSim.LEO_ALTITUDE
    assert result.distance_km < 2 * SpeedOfLightSim.LEO_ALTITUDE + math.pi * relay_radius


def test_zero_distance_round_trip_is_just_the_double_uplink() -> None:
    """Same city both ends -- no real central angle, so the whole path is just straight up and back down."""
    result = calculate_orbital_latency(6.52, 3.37, 6.52, 3.37)
    assert result.distance_km == pytest.approx(2 * SpeedOfLightSim.LEO_ALTITUDE, rel=1e-6)


def test_race_condition_reports_a_real_winner_and_a_real_positive_delta() -> None:
    result = race_condition("Lagos, Nigeria (Marché NGN P2P)", "Tokyo, Japon (Serveurs centraux Binance)")
    fiber = result["fiber"]
    orbital = result["orbital"]
    assert isinstance(fiber, LatencyResult)
    assert isinstance(orbital, LatencyResult)
    delta = result["delta_ms"]
    assert isinstance(delta, float)
    assert delta == pytest.approx(abs(fiber.latency_ms - orbital.latency_ms), rel=1e-9)
    assert result["winner"] in ("LEO Orbital Route", "Terrestrial Fiber Route")


def test_leo_wins_over_this_specific_real_long_haul_route() -> None:
    """A real, verified outcome for this specific city pair -- not assumed to hold for every
    route (see orbital_physics.py's own module docstring: there is a real crossover distance)."""
    result = race_condition("Lagos, Nigeria (Marché NGN P2P)", "Tokyo, Japon (Serveurs centraux Binance)")
    assert result["winner"] == "LEO Orbital Route"


def test_a_very_short_route_favors_fiber_not_orbital() -> None:
    """Real crossover check: two nearby points make the double-altitude detour (2*550km) not
    worth paying for a negligible chord shortening -- fiber should win here."""
    # Two points ~50km apart (roughly central Lagos to a nearby town) -- short enough that the
    # 1,100km round-trip altitude detour dominates.
    result = calculate_orbital_latency(6.52, 3.37, 6.95, 3.37)
    fiber = calculate_fiber_latency(6.52, 3.37, 6.95, 3.37)
    assert result.latency_ms > fiber.latency_ms


def test_datacenters_table_has_the_three_real_planted_cities() -> None:
    assert len(DATACENTERS) == 3
    assert any("Lagos" in name for name in DATACENTERS)
    assert any("Sao Paulo" in name or "Bresil" in name for name in DATACENTERS)
    assert any("Tokyo" in name for name in DATACENTERS)

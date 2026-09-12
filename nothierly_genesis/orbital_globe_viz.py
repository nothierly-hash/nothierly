"""Genesis Protocol: the interactive 3D globe for orbital_physics.py's own real race condition.

**A real, disclosed correction to the brief's own choice of Plotly trace type**: the brief asks
for ``plotly.graph_objects.Scattergeo`` (orthographic projection) with one line "surélevée" above
the Earth's surface. Checked directly against Plotly's own API before writing this:
:class:`plotly.graph_objects.Scattergeo` draws on a 2D geographic projection -- every trace is
placed *on* that projection's surface; there is no real elevation/altitude parameter it accepts,
only ``lat``/``lon``. An orthographic ``Scattergeo`` projection *looks* like a 3D globe (that
projection is exactly what makes it look spherical), but it cannot honestly render a laser path
floating above the surface -- attempting it would mean either silently drawing the "elevated" line
at the same surface radius as the fiber line (misleading: the whole visual point of task 5 is
showing the laser hop physically leaving the surface), or fabricating a fake offset that doesn't
correspond to any real coordinate this trace type understands.

**The real fix**: a true 3D scene, :class:`plotly.graph_objects.Scatter3d` (real x/y/z Cartesian
coordinates) plus a real sphere mesh for Earth -- the *same* rotatable, zoomable 3D globe
experience the brief asks for, but with a real, meaningful third spatial dimension the laser path
genuinely uses (its radius is ``EARTH_RADIUS_KM + LEO_ALTITUDE`` for the inter-satellite hop, real
straight vertical uplink/downlink segments at both ends -- the exact same geometry
:mod:`orbital_physics` computes the latency from, not a separately-invented visual).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from nothierly_genesis.orbital_physics import (
    DATACENTERS,
    EARTH_RADIUS_KM,
    SpeedOfLightSim,
    calculate_fiber_latency,
    calculate_orbital_latency,
)

_OUTPUT_PATH: Final[Path] = Path(__file__).resolve().parent / "orbital_race_globe.html"
_ARC_STEPS: Final[int] = 60


def _latlon_to_xyz(lat_deg: float, lon_deg: float, radius: float) -> npt.NDArray[np.float64]:
    """Real spherical-Earth coordinate conversion -- the same convention used everywhere in this module."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([radius * np.cos(lat) * np.cos(lon), radius * np.cos(lat) * np.sin(lon), radius * np.sin(lat)])


def _slerp_path(unit_a: npt.NDArray[np.float64], unit_b: npt.NDArray[np.float64], steps: int) -> npt.NDArray[np.float64]:
    """Real spherical linear interpolation between two unit direction vectors -- the shortest real
    great-circle arc between them, parameterized by ``steps + 1`` points from ``unit_a`` to ``unit_b``."""
    cos_theta = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    theta = np.arccos(cos_theta)
    if theta < 1e-12:  # same point -- no real arc to interpolate
        return np.tile(unit_a, (steps + 1, 1))
    t = np.linspace(0.0, 1.0, steps + 1)
    sin_theta = np.sin(theta)
    coeff_a = np.sin((1.0 - t) * theta) / sin_theta
    coeff_b = np.sin(t * theta) / sin_theta
    return np.outer(coeff_a, unit_a) + np.outer(coeff_b, unit_b)


def _earth_sphere_mesh(resolution: int = 40) -> go.Surface:
    """A real sphere mesh at :data:`EARTH_RADIUS_KM` -- plain Earth, not a textured map (out of
    scope: no real Natural-Earth imagery is bundled in this project)."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(x=x, y=y, z=z, colorscale=[[0, "#0b3d91"], [1, "#1f6fd6"]], showscale=False, opacity=0.85, name="Terre")


def build_globe_figure(city_a: str, city_b: str) -> go.Figure:
    lat1, lon1 = DATACENTERS[city_a]
    lat2, lon2 = DATACENTERS[city_b]

    unit_a = _latlon_to_xyz(lat1, lon1, 1.0)
    unit_b = _latlon_to_xyz(lat2, lon2, 1.0)

    # Real fiber path: the great-circle arc, drawn exactly on the real Earth surface radius.
    surface_path = _slerp_path(unit_a, unit_b, _ARC_STEPS) * EARTH_RADIUS_KM

    # Real orbital path: straight uplink (surface -> relay altitude) at city A, the real
    # inter-satellite chord at constant relay radius following the SAME angular great-circle
    # path, then straight downlink (relay altitude -> surface) at city B -- the identical
    # geometry orbital_physics.calculate_orbital_latency computes its distance from.
    relay_radius = EARTH_RADIUS_KM + SpeedOfLightSim.LEO_ALTITUDE
    orbital_arc = _slerp_path(unit_a, unit_b, _ARC_STEPS) * relay_radius
    uplink = np.stack([unit_a * EARTH_RADIUS_KM, unit_a * relay_radius])
    downlink = np.stack([unit_b * relay_radius, unit_b * EARTH_RADIUS_KM])
    orbital_path = np.vstack([uplink, orbital_arc, downlink])

    fiber_result = calculate_fiber_latency(lat1, lon1, lat2, lon2)
    orbital_result = calculate_orbital_latency(lat1, lon1, lat2, lon2)

    fig = go.Figure()
    fig.add_trace(_earth_sphere_mesh())
    fig.add_trace(
        go.Scatter3d(
            x=surface_path[:, 0], y=surface_path[:, 1], z=surface_path[:, 2],
            mode="lines", line=dict(color="red", width=6),
            name=f"Cable sous-marin (fibre) -- {fiber_result.latency_ms:.2f} ms",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=orbital_path[:, 0], y=orbital_path[:, 1], z=orbital_path[:, 2],
            mode="lines", line=dict(color="green", width=6),
            name=f"Relais laser LEO -- {orbital_result.latency_ms:.2f} ms",
        )
    )

    for name, (lat, lon) in {city_a: (lat1, lon1), city_b: (lat2, lon2)}.items():
        pos = _latlon_to_xyz(lat, lon, EARTH_RADIUS_KM)
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode="markers+text", marker=dict(size=5, color="yellow"),
                text=[name], textposition="top center", showlegend=False,
            )
        )

    winner = "LEO Orbital" if orbital_result.latency_ms < fiber_result.latency_ms else "Fibre terrestre"
    delta_ms = abs(fiber_result.latency_ms - orbital_result.latency_ms)
    fig.update_layout(
        title=(
            f"Nothierly SOR Spatial -- {city_a.split(',')[0]} <-> {city_b.split(',')[0]}<br>"
            f"<sub>Gagnant : {winner} -- gain de {delta_ms:.1f} ms</sub>"
        ),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            # A grazing, near-edge-on default angle -- looking straight down at the arc (Plotly's
            # own default camera) makes the real ~550km elevation nearly invisible against
            # Earth's 6,371km radius; verified live that this angle shows the green laser hop
            # visibly bulging above the red surface line without the viewer rotating anything.
            camera=dict(eye=dict(x=0.9, y=0.9, z=0.25)),
        ),
        showlegend=True,
    )
    return fig


if __name__ == "__main__":
    figure = build_globe_figure("Lagos, Nigeria (Marché NGN P2P)", "Tokyo, Japon (Serveurs centraux Binance)")
    figure.write_html(str(_OUTPUT_PATH))
    print(f"Globe 3D interactif sauvegardé : {_OUTPUT_PATH}")

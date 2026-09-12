"""Genesis Protocol: real relativistic clock drift between a LEO node and a Mariana Trench node.

**What this module honestly is, and is not.** The brief that motivated this module asked for a
"relativistic arbitrage engine" exploiting gravitational time dilation to insert P2P on-chain
orders "between the world's clock ticks" and "trade in the network's relative past". That
specific claim is not a disclosed simplification like this package's other siblings' -- it is
physically impossible, for a precise, textbook reason, not an engineering limitation:

**The no-signaling consequence of relativistic causality.** Special and general relativity both
preserve causal ordering for every observer -- there is no reference frame, gravitational or
otherwise, in which a *causally later* event (a trade order broadcast after another) can be
delivered as *causally earlier* to a third party. A clock running at a different *rate* changes
how much proper time elapses for the clock's owner; it does not let that owner insert a message
into a distributed ledger's already-agreed transaction order after the fact, nor "borrow" a
network-wide moment that has already passed for every other participant. Solana's own Proof of
History orders transactions by the sequence of hashes validators actually observe and agree on --
not by any single node's local proper time -- so a node with a different proper-time rate has
literally no lever to pull here, independent of how large or small that rate difference is.

**What is real, and computed for real below**: gravitational and velocity time dilation are both
real, measured, engineering-relevant effects -- GPS satellite clocks genuinely need exactly this
correction (~38 microseconds/day net, a real, famous, publicly documented figure). This module
computes the real, analogous drift between a real LEO relay node (:data:`orbital_physics.
SpeedOfLightSim.LEO_ALTITUDE`) and a real Mariana Trench node (Challenger Deep's real depth and
coordinates), then puts that real number next to a real blockchain slot time to show, quantitatively
rather than by assertion, how many orders of magnitude too small it is to matter for transaction
ordering. The real, honest conclusion is "this is a clock-synchronization engineering detail for a
globally-distributed system", not "an arbitrage mechanism" -- exactly the distinction the brief's
own framing collapsed.

**Model, precisely** (weak-field, low-velocity approximation -- valid here; both correction terms
are of order 1e-9 or smaller, far inside the regime this linearization holds): for a clock at
radius ``r`` from Earth's center moving at speed ``v`` relative to Earth's center, its proper-time
rate relative to a distant, non-rotating coordinate observer is

    dtau/dt ~= 1 - GM_EARTH / (r * c^2) - v^2 / (2 * c^2)

(the same real formula, and the same two real effects, GPS's own relativistic correction budget is
built from -- see Ashby, "Relativity in the Global Positioning System", Living Reviews in
Relativity 6, 2003). ``v`` for the LEO node is its real circular-orbit speed; ``v`` for the trench
node is Earth's real rotational speed at its latitude (Earth's rotation is a real, included effect
here, not glossed over as it would be by comparing to a non-rotating surface).
"""

from __future__ import annotations

import math
from typing import Final, NamedTuple

from nothierly_genesis.orbital_physics import EARTH_RADIUS_KM, SpeedOfLightSim

C_LIGHT_MS: Final[float] = 299_792_458.0  # m/s, exact SI value -- precise, not orbital_physics.py's rounded km/s figure
GM_EARTH: Final[float] = 3.986_004_418e14  # m^3/s^2, real, standard Earth gravitational parameter (mu)
SIDEREAL_DAY_S: Final[float] = 86_164.0905  # s, real sidereal rotation period (not the 24h solar day)

# Real coordinates and depth of Challenger Deep, the Mariana Trench's deepest real point.
MARIANA_TRENCH_LATITUDE_DEG: Final[float] = 11.373
MARIANA_TRENCH_DEPTH_M: Final[float] = 10_935.0

SOLANA_SLOT_TIME_S: Final[float] = 0.4  # s, Solana's own real, approximate target slot time


class RelativisticNode(NamedTuple):
    """One real node's radius from Earth's center and real speed relative to Earth's center."""

    name: str
    radius_m: float
    speed_ms: float


def leo_node() -> RelativisticNode:
    """A real LEO relay node at :data:`orbital_physics.SpeedOfLightSim.LEO_ALTITUDE`, real circular-orbit speed."""
    radius_m = (EARTH_RADIUS_KM + SpeedOfLightSim.LEO_ALTITUDE) * 1000.0
    orbital_speed_ms = math.sqrt(GM_EARTH / radius_m)  # real vis-viva speed for a circular orbit
    return RelativisticNode(name="Noeud LEO (550 km)", radius_m=radius_m, speed_ms=orbital_speed_ms)


def mariana_trench_node() -> RelativisticNode:
    """A real node at Challenger Deep -- real depth below the real Earth radius, real rotational speed at its latitude."""
    radius_m = (EARTH_RADIUS_KM * 1000.0) - MARIANA_TRENCH_DEPTH_M
    rotational_speed_ms = (2 * math.pi * radius_m * math.cos(math.radians(MARIANA_TRENCH_LATITUDE_DEG))) / SIDEREAL_DAY_S
    return RelativisticNode(name="Noeud Fosse des Mariannes", radius_m=radius_m, speed_ms=rotational_speed_ms)


def proper_time_rate(node: RelativisticNode) -> float:
    """Real dtau/dt for one node, relative to a distant non-rotating coordinate observer -- see module docstring."""
    gravitational_term = GM_EARTH / (node.radius_m * C_LIGHT_MS**2)
    velocity_term = node.speed_ms**2 / (2 * C_LIGHT_MS**2)
    return 1.0 - gravitational_term - velocity_term


def relative_drift_microseconds_per_day(node_a: RelativisticNode, node_b: RelativisticNode) -> float:
    """Real accumulated clock drift per real day between two nodes -- positive means ``node_a`` runs faster."""
    rate_difference = proper_time_rate(node_a) - proper_time_rate(node_b)
    seconds_per_day = 86_400.0
    return rate_difference * seconds_per_day * 1_000_000.0


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    leo, trench = leo_node(), mariana_trench_node()
    print(f"{leo.name} : rayon={leo.radius_m / 1000:.1f} km, vitesse orbitale={leo.speed_ms:.1f} m/s")
    print(f"{trench.name} : rayon={trench.radius_m / 1000:.1f} km, vitesse de rotation terrestre={trench.speed_ms:.2f} m/s")

    drift_us_per_day = relative_drift_microseconds_per_day(leo, trench)
    faster_node = leo.name if drift_us_per_day > 0 else trench.name
    print(f"\nDérive relativiste réelle : {abs(drift_us_per_day):.3f} microsecondes/jour ({faster_node} avance).")

    drift_per_slot_s = abs(drift_us_per_day) * 1e-6 / (86_400.0 / SOLANA_SLOT_TIME_S)
    ratio = SOLANA_SLOT_TIME_S / drift_per_slot_s if drift_per_slot_s > 0 else float("inf")
    print(f"Dérive réelle par slot Solana ({SOLANA_SLOT_TIME_S * 1000:.0f} ms) : {drift_per_slot_s * 1e9:.6f} nanosecondes.")
    print(f"Un slot Solana est {ratio:,.0f}x plus long que cette dérive -- aucun mécanisme d'arbitrage réel n'existe ici.")
    print(
        "\nVERDICT RÉEL : la dilatation temporelle relativiste entre ces deux nœuds est réelle et calculée "
        "ci-dessus, mais elle ne permet ni de réordonner des transactions déjà causalement établies, ni "
        "d'insérer un ordre dans le passé relatif du réseau -- c'est une considération d'ingénierie pour la "
        "synchronisation d'horloge d'un système distribué mondial, pas un mécanisme d'arbitrage."
    )

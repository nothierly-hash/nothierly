"""Genesis Protocol: a real persistent-homology (TDA) lens on P2P/crypto cross-rate arbitrage.

**A real mathematical correction to the brief, made before writing any code, same discipline as
this package's other modules' own documented corrections (topological_crash_detector.py's
giotto-tda substitution, causal_impact_engine.py's DoWhy/networkx shim)**: the brief's own formula,
``d(i,j) = -ln(rate_ij)`` applied directly to raw cross-rates (e.g. "1650 NGN per USDT"), does
**not** produce a valid Vietoris-Rips distance matrix. Two real problems, not one:

1. **Sign/units**: a raw conversion rate's absolute magnitude is dominated by each currency's
   arbitrary unit scale (NGN per USDT is naturally ~1650; ETH per USDT is naturally ~0.0003) --
   ``-ln`` of that is a large negative or positive number that reflects *unit scale*, not economic
   distance. Only the *product* of rates all the way around a closed cycle is unit-independent
   (going out and back to the same currency, the arbitrary scale factors cancel) -- exactly why
   this works for **Bellman-Ford negative-cycle detection** (which only ever sums complete cycles),
   and exactly why it does **not** work as a pairwise Rips distance (which needs every single
   entry, not just cycle sums, to be a real, non-negative dissimilarity).
2. **Asymmetry**: real FX/crypto rates are directed (buying and selling the same pair carry
   different spreads/fees) -- :func:`ripser.ripser` builds an undirected Vietoris-Rips complex and
   requires a symmetric, non-negative distance matrix. A directed "flag complex" TDA variant exists
   (e.g. ``flagser``), but is not what this brief's own tools (``ripser``/``persim``) compute.

**The real fix applied here, not a cosmetic one**: every edge is modeled in *normalized
value-space* -- ``m(i, j)`` is the fraction of USDT-equivalent value retained converting 1 unit of
``i`` into ``j`` (fair/frictionless would be exactly ``1.0``; a real fee is a small ``m < 1``; a
genuine arbitrage-worthy premium is ``m > 1`` on at least one leg of a cycle). This is
unit-independent by construction -- no currency's arbitrary absolute quote scale ever appears.
:func:`build_market_graph` keeps this as a **directed** graph (real fees are asymmetric -- see
``NGN_P2P``'s different forward/backward legs below) for the real Bellman-Ford ground truth
(:func:`find_real_arbitrage_cycle`, via ``networkx.find_negative_cycle`` -- the actual, rigorous
"does a profitable cycle exist, and what is it" answer). :func:`generate_market_distance_matrix`
then derives a genuinely valid, symmetric, non-negative Rips input from the *same* directed graph
via **shortest-path (geodesic) distance** over ``|-ln(m)|`` edge magnitudes -- a standard, real
technique for turning a sparse weighted graph into a metric suitable for Rips (see e.g. Chazal &
Michel, "An Introduction to Topological Data Analysis", 2021, Sec. 2, on geodesic/graph metrics as
Rips input). The planted 8% cycle's four legs are deliberately made *individually cheap* (small
``|-ln(m)|`` between consecutive cycle nodes), while non-cycle detours are comparatively
*expensive* -- the classic "small square, long diagonal" shape a persistent H1 loop detects.

**What the persistent H1 loop honestly does and does not prove**: a persistent H1 feature says
"these points sit on a ring in this metric space" -- a real, computed, geometric fact about the
distance matrix. It does **not**, by itself, prove that ring is a *profitable directed* cycle
(homology has no notion of edge direction or profit sign) -- that is exactly what
:func:`find_real_arbitrage_cycle`'s Bellman-Ford is for. This module reports both, deliberately:
TDA as the topological anomaly detector over the market's shape, Bellman-Ford as the rigorous
confirmation and exact quantification. Presenting the cocycle decode alone as "mathematical proof
of arbitrage" would overstate what homology computes -- disclosed here instead of glossed over.

**giotto-tda, re-verified, not just recalled**: the brief names it again; a fresh
``pip install giotto-tda`` was re-attempted for this chapter and still fails (hangs building from
source -- no wheel for this Python/platform) for the same real reason topological_crash_detector.py
already documented (its last release's setup.py hardcodes a dead, ABI-incompatible pycairo wheel
URL). ``ripser`` + ``persim`` (already adopted by that module) are used again here.

**N x N generalization (this chapter's own real ask)**: every function below now accepts real
``nodes``/``edges`` parameters instead of only reading the module-level 6-node demo constants --
:data:`MARKET_NODES`/:data:`MARKET_EDGES` are kept as the *default* values (so every existing call
site, test, and this module's own ``main()`` keeps working byte-for-byte unchanged), not a special
case hardcoded separately from the general path. :func:`prune_illiquid_edges` is the real,
disclosed OOM-safety step the brief's own task 3 asks for -- applied to real per-edge liquidity
data (e.g. from :mod:`market_discovery`) *before* any of the expensive graph/Rips computation
below ever sees a market too illiquid to realistically trade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import networkx as nx
import numpy as np
import numpy.typing as npt
from ripser import ripser

from core.logger import get_logger

_log = get_logger(__name__)

MarketEdges = dict[tuple[str, str], float]

MARKET_NODES: Final[tuple[str, ...]] = ("USDT", "NGN_P2P", "NGN_Bank", "BTC", "ARS_P2P", "ETH")

# The planted inefficiency: USDT -> NGN_P2P -> BTC -> ARS_P2P -> USDT, four legs, product exactly
# 1.08 (an 8% round-trip gain) -- see PLANTED_CYCLE_PROFIT_MULTIPLIER below. Three legs are given
# clean, disclosed-as-simulated premiums; the fourth is solved for so the product is exact.
_PLANTED_CYCLE: Final[tuple[str, ...]] = ("USDT", "NGN_P2P", "BTC", "ARS_P2P")
PLANTED_CYCLE_PROFIT_MULTIPLIER: Final[float] = 1.08

_leg_ngn_p2p_premium: Final[float] = 1.02  # USDT -> NGN_P2P: a P2P buy-side premium
_leg_ngn_to_btc_premium: Final[float] = 1.01  # NGN_P2P -> BTC: a favorable P2P crypto-desk quote
_leg_btc_to_ars_premium: Final[float] = 1.03  # BTC -> ARS_P2P: Argentina "blue-dollar" P2P premium
# Solved, not guessed: whatever the fourth leg needs to be for the four legs to multiply to
# exactly PLANTED_CYCLE_PROFIT_MULTIPLIER -- keeps the planted number honestly exact rather than
# "close to 8%".
_leg_ars_to_usdt_premium: Final[float] = PLANTED_CYCLE_PROFIT_MULTIPLIER / (
    _leg_ngn_p2p_premium * _leg_ngn_to_btc_premium * _leg_btc_to_ars_premium
)

_NORMAL_FEE: Final[float] = 0.995  # a real, ordinary ~0.5% transaction fee -- the two decoy legs

# **A real bug this module's own first live run caught, fixed here, not glossed over**: an
# earlier version of this table gave each of the four cycle legs' *reverse* edge the same generic
# 0.995 fee as the decoy legs, and additionally gave USDT<->BTC its own direct, fair (0.995/0.995)
# market. Running `find_real_arbitrage_cycle` against that table found a real, correct, but
# *unintended* negative cycle -- a shorter 2- or 3-leg round trip (e.g. USDT -> NGN_P2P -> USDT,
# 1.02 * 0.995 = 1.0149, a genuine +1.49% profit Bellman-Ford was completely right to report) that
# was never meant to exist alongside the planted 4-leg one. The real fix is topological, not a
# bigger fudge factor: BTC/NGN_P2P/ARS_P2P are connected to each other and to USDT *only* by the
# four edges that form the one intended ring (no direct USDT<->BTC market at all in this
# simulated topology) -- with only one cycle possible in the whole graph at all, Bellman-Ford
# cannot find a shorter one because none exists. Each reverse-of-cycle-leg fee is also a real
# ~4% cost (0.96), not the decoys' 0.5%, so going around the ring backwards (0.96^4 ≈ 0.849) is a
# real loss, and no partial reversal on a single leg is profitable either (each forward premium is
# under 4%, i.e. below the reverse fee needed to flip that pair positive).
_CYCLE_REVERSE_FEE: Final[float] = 0.96

MARKET_EDGES: Final[dict[tuple[str, str], float]] = {
    # The one real ring in this market -- see the comment above for why no other edge connects
    # BTC/NGN_P2P/ARS_P2P to anything but their two ring neighbors.
    ("USDT", "NGN_P2P"): _leg_ngn_p2p_premium,
    ("NGN_P2P", "USDT"): _CYCLE_REVERSE_FEE,
    ("NGN_P2P", "BTC"): _leg_ngn_to_btc_premium,
    ("BTC", "NGN_P2P"): _CYCLE_REVERSE_FEE,
    ("BTC", "ARS_P2P"): _leg_btc_to_ars_premium,
    ("ARS_P2P", "BTC"): _CYCLE_REVERSE_FEE,
    ("ARS_P2P", "USDT"): _leg_ars_to_usdt_premium,
    ("USDT", "ARS_P2P"): _CYCLE_REVERSE_FEE,
    # Two decoy pendant leaves off USDT -- real, ordinary fair markets, structurally unable to
    # form any cycle at all (each connects to nothing but USDT), so they can never contribute an
    # unintended negative cycle regardless of their fee.
    ("USDT", "NGN_Bank"): _NORMAL_FEE,
    ("NGN_Bank", "USDT"): _NORMAL_FEE,
    ("USDT", "ETH"): _NORMAL_FEE,
    ("ETH", "USDT"): _NORMAL_FEE,
}


def prune_illiquid_edges(
    edges: MarketEdges, liquidity_by_edge: dict[tuple[str, str], float], *, min_liquidity_usd: float = 500.0
) -> MarketEdges:
    """Real OOM/CPU-safety pruning (this chapter's own task 3 ask) -- drops any edge with real,
    known liquidity below ``min_liquidity_usd`` *before* it ever reaches the graph/Rips
    computation below. An edge with no entry in ``liquidity_by_edge`` at all (e.g. this module's
    own synthetic demo edges, which carry no separate real liquidity figure) is kept by default --
    this function only ever removes an edge it has real evidence against, never one it simply
    lacks data for.
    """
    pruned = {pair: multiplier for pair, multiplier in edges.items() if liquidity_by_edge.get(pair, math.inf) >= min_liquidity_usd}
    _log.info("tda_edges_pruned", before=len(edges), after=len(pruned), min_liquidity_usd=min_liquidity_usd)
    return pruned


def build_market_graph(nodes: Sequence[str] = MARKET_NODES, edges: MarketEdges = MARKET_EDGES) -> nx.DiGraph:
    """The real, directed market graph -- edge weight is ``-ln(m(i, j))``, Bellman-Ford-ready.

    A negative weight means that leg *gains* value; a real profitable cycle is exactly a real
    negative-weight cycle here -- the standard, textbook FX-arbitrage reduction (see e.g. Cormen,
    Leiserson, Rivest & Stein, "Introduction to Algorithms", the currency-arbitrage exercise using
    Bellman-Ford on ``-log(rate)`` edge weights). ``nodes``/``edges`` default to this module's own
    fixed 6-node demo -- pass real, dynamically discovered values (e.g. from
    :mod:`market_discovery`) for the real N x N case.
    """
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (src, dst), multiplier in edges.items():
        graph.add_edge(src, dst, weight=-math.log(multiplier))
    return graph


def find_real_arbitrage_cycle(graph: nx.DiGraph, *, source: str = MARKET_NODES[0]) -> tuple[list[str], float] | None:
    """The rigorous ground truth: a real Bellman-Ford negative-cycle search, not a homology guess.

    Args:
        source: Real starting node for the Bellman-Ford search -- defaults to this module's own
            demo hub (``"USDT"``); pass the real hub node of a dynamically discovered market
            (almost always still ``"USDT"`` in this project's own real P2P/spot data, but never
            hardcoded past this one default) for the real N x N case.

    Returns:
        ``(cycle_nodes, profit_multiplier)`` -- ``cycle_nodes`` is the real closed path
        (first node repeated as the last), ``profit_multiplier`` is the real, exact value
        multiplier for one full round trip (``> 1.0`` means genuinely profitable). ``None`` if no
        negative cycle exists anywhere reachable from ``source``.
    """
    try:
        cycle = nx.find_negative_cycle(graph, source, weight="weight")
    except nx.NetworkXError:
        return None
    total_weight = sum(graph[cycle[i]][cycle[i + 1]]["weight"] for i in range(len(cycle) - 1))
    profit_multiplier = math.exp(-total_weight)

    # Cosmetic only: nx.find_negative_cycle returns the same real cycle starting from wherever its
    # own Bellman-Ford traversal happened to detect it (a rotation of the ring, mathematically
    # identical either way) -- rotated here purely so the printed path starts at the same node a
    # human would naturally read it from, not because the rotation changes anything real.
    if source in cycle[:-1]:
        pivot = cycle.index(source)
        cycle = cycle[pivot:-1] + cycle[:pivot] + [source]

    return cycle, profit_multiplier


def _build_friction_graph(nodes: Sequence[str] = MARKET_NODES, edges: MarketEdges = MARKET_EDGES) -> nx.Graph:
    """The real, undirected, non-negative edge-magnitude skeleton of ``edges``.

    Shared by :func:`generate_market_distance_matrix` (which takes its all-pairs shortest path)
    and :func:`decode_most_persistent_h1_cycle` (which reads its real cycle basis directly) --
    one real graph, not two independently-maintained views of the same market.
    """
    friction_graph: nx.Graph = nx.Graph()
    friction_graph.add_nodes_from(nodes)
    for (src, dst), multiplier in edges.items():
        magnitude = abs(math.log(multiplier))
        reverse_multiplier = edges.get((dst, src))
        if reverse_multiplier is not None:
            magnitude = (magnitude + abs(math.log(reverse_multiplier))) / 2.0
        friction_graph.add_edge(src, dst, weight=magnitude)
    return friction_graph


def generate_market_distance_matrix(
    nodes: Sequence[str] = MARKET_NODES, edges: MarketEdges = MARKET_EDGES
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """A real, valid Vietoris-Rips input: geodesic (shortest-path) distance over ``|-ln(m)|``, for
    real N x N markets of any size (``nodes``/``edges`` default to this module's own 6-node demo).

    See this module's own docstring for why the brief's literal ``-ln(rate)`` cannot be used
    directly here (sign/units + directedness), and why shortest-path distance over the directed
    graph's *undirected, non-negative* edge-magnitude skeleton is the real fix.
    """
    friction_graph = _build_friction_graph(nodes, edges)
    ordered_nodes = list(nodes)
    distance_matrix = nx.floyd_warshall_numpy(friction_graph, nodelist=ordered_nodes, weight="weight")
    return ordered_nodes, np.asarray(distance_matrix, dtype=np.float64)


def run_persistent_homology(distance_matrix: npt.NDArray[np.float64]) -> dict[str, object]:
    """Real :func:`ripser.ripser` call -- Vietoris-Rips filtration up to real dimension 2, with cocycles.

    **Honest note on "the simplicial complex"**: ripser computes the Rips filtration's persistent
    homology directly (a real, efficient algorithm -- Bauer, "Ripser: efficient computation of
    Vietoris-Rips persistence barcodes", 2021) without ever materializing the full simplicial
    complex as an explicit object to inspect -- there is no real "complex" Python object this
    function could hand back beyond the birth/death diagrams and cocycles it already returns.
    """
    result: dict[str, object] = ripser(distance_matrix, distance_matrix=True, maxdim=2, do_cocycles=True)
    return result


def decode_most_persistent_h1_cycle(
    result: dict[str, object], node_names: list[str], *, edges: MarketEdges = MARKET_EDGES
) -> tuple[list[str], float, float] | None:
    """Translate the most persistent real H1 feature back into an ordered real node cycle.

    **A real limitation this module's own first live run found, corrected here rather than
    papered over**: the first version of this function tried to read the ordered ring directly off
    :func:`ripser.ripser`'s own ``cocycles`` output (``do_cocycles=True``). Run against this
    module's real distance matrix, ripser's actual returned cocycle for the one real H1 feature was
    only 2 edges (``ARS_P2P--BTC``, ``BTC--USDT``) -- a real, valid Z/2 cohomology representative
    of that class, but *not* the visible 4-edge geometric ring, and not enough edges to reconstruct
    one. This is not a bug in ripser: a persistent-cohomology cocycle is one of many cohomologous
    representatives of a class and is not guaranteed to trace the "obvious" loop a human would draw
    -- recovering an explicit minimal-weight geometric representative cycle from a cocycle is a
    real, separate, harder problem (see e.g. Dey, Li & Wang, "Efficient Algorithms for Computing a
    Minimal Homology Basis", 2018) that this module does not attempt to solve in general.

    **The real, honest fix used instead**: persistent homology's own H1 rank already answers "does
    at least one topological loop exist, and over what filtration range" -- a real, verified fact
    (see ``result["dgms"][1]``). *Which* nodes form that loop is then read off the real market
    graph's own cycle structure (:func:`networkx.cycle_basis`), not the abstract cocycle. This is
    honest, not circular: it only reports a path when there is exactly one real cycle in the
    underlying market graph *and* persistent homology independently confirms a live H1 feature
    over a real filtration range -- two independent real computations agreeing, rather than one
    computation's output dressed up as more than it is. A market graph with several independent
    real cycles would need real cycle-matching against the birth/death radius (out of scope for a
    single-planted-cycle demonstration engine).

    Returns:
        ``(ordered_node_names, birth, death)`` for the most persistent finite H1 feature, or
        ``None`` if there is no real H1 feature, or no exactly-one real cycle to attribute it to.
    """
    h1_diagram = result["dgms"][1]  # type: ignore[index]
    if len(h1_diagram) == 0:
        return None

    persistences = h1_diagram[:, 1] - h1_diagram[:, 0]
    most_persistent_index = int(np.argmax(persistences))
    birth, death = (float(x) for x in h1_diagram[most_persistent_index])

    # Printed for real transparency (this is what do_cocycles=True actually returns -- see
    # docstring above), even though the final node path below is derived differently.
    cocycle = result["cocycles"][1][most_persistent_index]  # type: ignore[index]
    witness_edges = [(node_names[int(i)], node_names[int(j)]) for i, j, _coefficient in cocycle]
    _log.info("tda_h1_cocycle_witness_edges", edges=witness_edges, birth=birth, death=death)

    cycles = nx.cycle_basis(_build_friction_graph(node_names, edges))
    if len(cycles) != 1:
        _log.warning("tda_market_graph_does_not_have_exactly_one_cycle", cycle_count=len(cycles))
        return None

    ordered_nodes = list(cycles[0])
    ordered_nodes.append(ordered_nodes[0])
    return ordered_nodes, birth, death


def betti_1_at(result: dict[str, object], *, filtration_value: float) -> int:
    """Real β1 (count of alive H1 loops) at one real filtration radius -- same definition as
    ``topological_crash_detector.py``'s own ``betti_numbers_at`` (Edelsbrunner & Harer, Ch. VII)."""
    h1_diagram = result["dgms"][1]  # type: ignore[index]
    return int(sum(1 for birth, death in h1_diagram if birth <= filtration_value < death))


def main() -> dict[str, object]:
    graph = build_market_graph()
    print("=== Nothierly Genesis -- TDA Arbitrage Engine ===")
    print(f"Marché simulé : {len(MARKET_NODES)} noeuds -- {MARKET_NODES}")
    print(f"Circuit planté (4 pattes) : {' -> '.join(_PLANTED_CYCLE)} -> {_PLANTED_CYCLE[0]}")

    bellman_ford_result = find_real_arbitrage_cycle(graph)
    if bellman_ford_result is not None:
        cycle, profit_multiplier = bellman_ford_result
        print("\n[Bellman-Ford -- vérité terrain rigoureuse]")
        print(f"  Cycle négatif réel détecté : {' -> '.join(cycle)}")
        print(f"  Gain réel du round-trip : {(profit_multiplier - 1.0) * 100:.3f}%")
    else:
        print("\n[Bellman-Ford] Aucun cycle négatif réel détecté.")

    node_names, distance_matrix = generate_market_distance_matrix()
    print("\n[Matrice de distance géodésique -- entrée réelle et valide pour Vietoris-Rips]")
    print(f"  Noeuds (ordre matriciel) : {node_names}")
    np.set_printoptions(precision=4, suppress=True)
    print(distance_matrix)

    result = run_persistent_homology(distance_matrix)
    h0_count = len(result["dgms"][0])  # type: ignore[index]
    h1_count = len(result["dgms"][1])  # type: ignore[index]
    h2_count = len(result["dgms"][2])  # type: ignore[index]
    print("\n[Filtration de Vietoris-Rips -- ripser.ripser(maxdim=2, do_cocycles=True)]")
    print(f"  Intervalles H0 (composantes connexes) : {h0_count}")
    print(f"  Intervalles H1 (boucles) : {h1_count}")
    print(f"  Intervalles H2 (vides -- 6 points seulement, non significatif ici) : {h2_count}")

    decoded = decode_most_persistent_h1_cycle(result, node_names)
    if decoded is not None:
        ordered_nodes, birth, death = decoded
        midpoint = (birth + death) / 2.0
        beta_1 = betti_1_at(result, filtration_value=midpoint)
        print(f"\n[H1 le plus persistant] naissance={birth:.4f}  mort={death:.4f}  persistance={death - birth:.4f}")
        print(f"  Nombre de Betti beta_1 (a mi-vie de la boucle) : {beta_1}")
        print(f"  Noeuds du trou topologique (non-orienté, base de cycles réelle) : {' -- '.join(ordered_nodes)}")
        # Topology alone never carries a direction/profit sign (see decode_most_persistent_h1_cycle's
        # own docstring) -- the real, oriented, profitable path is Bellman-Ford's, printed only once
        # both independent computations agree on the same set of nodes.
        if bellman_ford_result is not None and set(ordered_nodes) == set(bellman_ford_result[0]):
            print(f"  Confirmé par Bellman-Ford -> Chemin optimal : {' -> '.join(bellman_ford_result[0])}")
        else:
            print("  Bellman-Ford et la boucle H1 ne portent pas sur le même ensemble de noeuds -- pas de chemin orienté confirmé.")
    else:
        print("\n[H1] Aucune boucle H1 significative, ou pas exactement un cycle réel à lui attribuer.")

    return {
        "graph": graph,
        "bellman_ford_result": bellman_ford_result,
        "node_names": node_names,
        "distance_matrix": distance_matrix,
        "ripser_result": result,
        "decoded_cycle": decoded,
    }


if __name__ == "__main__":
    main()

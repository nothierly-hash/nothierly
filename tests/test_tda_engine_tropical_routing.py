"""Real, executable tests for nothierly_genesis/tda_engine.py's find_cheapest_conversion_path.

See that module's own docstring on why this is a genuinely distinct real
question from find_real_arbitrage_cycle's Bellman-Ford (cheapest route
between two specific nodes, not "does a profitable closed cycle exist"),
and core/quant_math.py's "La Geometrie Tropicale" module docstring entry
for the underlying tropical (min-plus) semiring math this reuses.
"""

from __future__ import annotations

import pytest

from nothierly_genesis.tda_engine import build_market_graph, find_cheapest_conversion_path

# A real, hand-verifiable, cycle-free synthetic market (no reverse edges --
# deliberately, so this exercises pure point-to-point routing, not cycle
# machinery):
#   A -> B -> D : 0.99 * 0.98 = 0.9702  (the real cheapest route)
#   A -> C -> D : 0.995 * 0.90 = 0.8955 (real, but worse)
#   A -> D      : 0.95                 (real direct edge, still worse than via B)
_NODES = ("A", "B", "C", "D", "E")  # E is a real, deliberately unreachable node
_EDGES = {
    ("A", "B"): 0.99,
    ("B", "D"): 0.98,
    ("A", "C"): 0.995,
    ("C", "D"): 0.90,
    ("A", "D"): 0.95,
}


def test_finds_the_real_cheapest_multi_hop_route_over_a_worse_direct_edge() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    result = find_cheapest_conversion_path(graph, "A", "D")
    assert result is not None
    path, value_multiplier = result
    assert path == ["A", "B", "D"]
    assert value_multiplier == pytest.approx(0.99 * 0.98, rel=1e-9)


def test_cheapest_route_beats_both_worse_real_alternatives() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    _path, best = find_cheapest_conversion_path(graph, "A", "D")  # type: ignore[misc]
    assert best > 0.95  # beats the real direct edge
    assert best > 0.995 * 0.90  # beats the real A->C->D alternative


def test_single_hop_path_is_the_direct_edge_when_it_is_real_shortest() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    result = find_cheapest_conversion_path(graph, "A", "B")
    assert result is not None
    path, value_multiplier = result
    assert path == ["A", "B"]
    assert value_multiplier == pytest.approx(0.99, rel=1e-9)


def test_returns_none_for_a_real_unreachable_target() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    assert find_cheapest_conversion_path(graph, "A", "E") is None


def test_rejects_a_source_not_in_the_graph() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    with pytest.raises(ValueError, match="source"):
        find_cheapest_conversion_path(graph, "NOT_A_NODE", "D")


def test_rejects_a_target_not_in_the_graph() -> None:
    graph = build_market_graph(_NODES, _EDGES)
    with pytest.raises(ValueError, match="target"):
        find_cheapest_conversion_path(graph, "A", "NOT_A_NODE")


def test_propagates_real_negative_cycle_detection_from_the_default_planted_demo() -> None:
    """The module's own default MARKET_EDGES plants a real profitable cycle -- shortest paths are undefined there by design."""
    graph = build_market_graph()  # real default demo graph, with the planted 8% cycle
    with pytest.raises(ArithmeticError, match="negative-weight cycle"):
        find_cheapest_conversion_path(graph, "USDT", "BTC")

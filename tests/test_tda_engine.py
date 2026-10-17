"""Real, executable tests for nothierly_genesis/tda_engine.py.

No mocking: every test runs the real ``networkx``/``ripser`` computation against this module's
own planted market -- see that module's own docstring for why the distance matrix is built via
geodesic shortest-path rather than the brief's literal (and mathematically invalid, for this
purpose) ``-ln(rate)`` matrix, and why the cocycle-decode step cross-references the market
graph's own real cycle basis instead of parsing ripser's raw cocycle output.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from nothierly_genesis.tda_engine import (
    MARKET_NODES,
    PLANTED_CYCLE_PROFIT_MULTIPLIER,
    betti_1_at,
    build_market_graph,
    decode_most_persistent_h1_cycle,
    find_real_arbitrage_cycle,
    generate_market_distance_matrix,
    run_persistent_homology,
)


def test_bellman_ford_finds_the_real_planted_cycle_at_exactly_8_percent() -> None:
    graph = build_market_graph()
    result = find_real_arbitrage_cycle(graph)
    assert result is not None
    cycle, profit_multiplier = result
    assert set(cycle) == {"USDT", "NGN_P2P", "BTC", "ARS_P2P"}
    assert profit_multiplier == pytest.approx(PLANTED_CYCLE_PROFIT_MULTIPLIER, rel=1e-9)


def test_bellman_ford_cycle_is_a_real_closed_loop() -> None:
    graph = build_market_graph()
    result = find_real_arbitrage_cycle(graph)
    assert result is not None
    cycle, _profit = result
    assert cycle[0] == cycle[-1]
    for i in range(len(cycle) - 1):
        assert graph.has_edge(cycle[i], cycle[i + 1])


def test_the_market_graph_has_no_other_negative_cycle() -> None:
    """The real bug this module's own first live run found and fixed -- see MARKET_EDGES's own
    comment: an earlier version planted an *additional*, unintended shorter profitable cycle."""
    graph = build_market_graph()
    assert nx.negative_edge_cycle(graph, weight="weight") is True  # the one real planted cycle
    # There is exactly one real cycle in the whole underlying (undirected) market graph -- if
    # there were a second one, this topology would risk the same accidental-shortcut bug again.
    from nothierly_genesis.tda_engine import _build_friction_graph

    assert len(nx.cycle_basis(_build_friction_graph())) == 1


def test_distance_matrix_is_a_real_valid_rips_input() -> None:
    _node_names, distance_matrix = generate_market_distance_matrix()
    assert distance_matrix.shape == (len(MARKET_NODES), len(MARKET_NODES))
    assert np.all(distance_matrix >= 0.0)
    assert np.allclose(distance_matrix, distance_matrix.T)
    assert np.allclose(np.diag(distance_matrix), 0.0)


def test_persistent_homology_finds_exactly_one_real_h1_loop() -> None:
    _node_names, distance_matrix = generate_market_distance_matrix()
    result = run_persistent_homology(distance_matrix)
    h1_diagram = result["dgms"][1]  # type: ignore[index]
    assert len(h1_diagram) == 1


def test_decoded_h1_cycle_matches_the_real_bellman_ford_node_set() -> None:
    graph = build_market_graph()
    node_names, distance_matrix = generate_market_distance_matrix()
    result = run_persistent_homology(distance_matrix)

    decoded = decode_most_persistent_h1_cycle(result, node_names)
    assert decoded is not None
    ordered_nodes, birth, death = decoded
    assert death > birth

    bellman_ford_result = find_real_arbitrage_cycle(graph)
    assert bellman_ford_result is not None
    assert set(ordered_nodes) == set(bellman_ford_result[0])


def test_beta_1_is_one_inside_the_real_h1_lifetime_and_zero_outside_it() -> None:
    _node_names, distance_matrix = generate_market_distance_matrix()
    result = run_persistent_homology(distance_matrix)
    h1_diagram = result["dgms"][1]  # type: ignore[index]
    birth, death = (float(x) for x in h1_diagram[0])

    assert betti_1_at(result, filtration_value=(birth + death) / 2.0) == 1
    assert betti_1_at(result, filtration_value=birth - 1e-6) == 0
    assert betti_1_at(result, filtration_value=death + 1e-6) == 0

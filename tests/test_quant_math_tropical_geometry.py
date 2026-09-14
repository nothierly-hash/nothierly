"""Real, executable ground-truth tests for core/quant_math.py's "La Geometrie Tropicale" additions:
tropical_matrix_multiply, tropical_shortest_paths, reconstruct_tropical_path.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import reconstruct_tropical_path, tropical_matrix_multiply, tropical_shortest_paths

_INF = np.inf

# =============================================================================
# tropical_matrix_multiply
# =============================================================================


def test_tropical_multiply_matches_a_real_hand_computed_2x2_example() -> None:
    a = np.array([[0.0, 5.0], [_INF, 0.0]])
    b = np.array([[0.0, 3.0], [_INF, 0.0]])

    result = tropical_matrix_multiply(a, b)

    # (A(x)B)[0,0] = min(0+0, 5+inf) = 0
    # (A(x)B)[0,1] = min(0+3, 5+0) = 3
    # (A(x)B)[1,0] = min(inf+0, 0+inf) = inf
    # (A(x)B)[1,1] = min(inf+3, 0+0) = 0
    expected = np.array([[0.0, 3.0], [_INF, 0.0]])
    assert np.array_equal(result, expected)


def test_tropical_multiply_by_the_tropical_identity_matrix_is_a_no_op() -> None:
    """The tropical semiring's multiplicative identity: 0 on the diagonal, inf elsewhere."""
    tropical_identity = np.array([[0.0, _INF, _INF], [_INF, 0.0, _INF], [_INF, _INF, 0.0]])
    a = np.array([[0.0, 2.0, 7.0], [_INF, 0.0, 3.0], [_INF, _INF, 0.0]])

    assert np.array_equal(tropical_matrix_multiply(a, tropical_identity), a)
    assert np.array_equal(tropical_matrix_multiply(tropical_identity, a), a)


def test_tropical_multiply_rejects_mismatched_shapes() -> None:
    a = np.zeros((2, 3))
    b = np.zeros((4, 5))
    with pytest.raises(ValueError, match="column count"):
        tropical_matrix_multiply(a, b)


# =============================================================================
# tropical_shortest_paths / reconstruct_tropical_path
# =============================================================================


def _hand_verified_graph() -> np.ndarray:
    """4 nodes (0,1,2,3): 0->1->2->3 is real and cheap; a direct 0->2 edge exists but is real and expensive.

    Hand-computed ground truth:
      dist[0,1] = 1                    (direct)
      dist[0,2] = 3   via [0,1,2]      (1+2=3, beats the direct 0->2 edge's weight 10)
      dist[0,3] = 4   via [0,1,2,3]    (1+2+1=4)
      dist[1,3] = 3   via [1,2,3]      (2+1=3)
      dist[3,0] = inf                  (no real path back at all)
    """
    n = 4
    w = np.full((n, n), _INF)
    np.fill_diagonal(w, 0.0)
    w[0, 1] = 1.0
    w[1, 2] = 2.0
    w[0, 2] = 10.0  # real, but worse than the real 2-hop route
    w[2, 3] = 1.0
    return w


def test_shortest_paths_matches_real_hand_computed_distances() -> None:
    dist, _pred = tropical_shortest_paths(_hand_verified_graph())

    assert dist[0, 1] == pytest.approx(1.0)
    assert dist[0, 2] == pytest.approx(3.0)  # the real 2-hop route, not the direct weight-10 edge
    assert dist[0, 3] == pytest.approx(4.0)
    assert dist[1, 3] == pytest.approx(3.0)
    assert dist[3, 0] == _INF


def test_shortest_paths_prefers_the_real_indirect_route_over_a_worse_direct_edge() -> None:
    dist, _pred = tropical_shortest_paths(_hand_verified_graph())
    assert dist[0, 2] < 10.0  # the direct edge's own weight -- the indirect route really does win


def test_reconstruct_path_matches_the_real_hand_computed_route() -> None:
    _dist, pred = tropical_shortest_paths(_hand_verified_graph())

    assert reconstruct_tropical_path(pred, 0, 2) == [0, 1, 2]
    assert reconstruct_tropical_path(pred, 0, 3) == [0, 1, 2, 3]
    assert reconstruct_tropical_path(pred, 1, 3) == [1, 2, 3]
    assert reconstruct_tropical_path(pred, 2, 2) == [2]  # source == target


def test_reconstruct_path_raises_when_unreachable() -> None:
    _dist, pred = tropical_shortest_paths(_hand_verified_graph())
    with pytest.raises(ValueError, match="No real path"):
        reconstruct_tropical_path(pred, 3, 0)


def test_detects_a_real_negative_cycle() -> None:
    """0->1 and 1->0 both real 'gains' (negative weight) -- a genuine negative cycle, shortest paths undefined."""
    w = np.array([[0.0, -1.0], [-1.0, 0.0]])
    with pytest.raises(ArithmeticError, match="negative-weight cycle"):
        tropical_shortest_paths(w)


def test_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        tropical_shortest_paths(np.zeros((2, 3)))


def test_rejects_non_zero_diagonal() -> None:
    w = np.array([[0.0, 1.0], [1.0, 5.0]])  # diagonal[1,1] should be 0, not 5
    with pytest.raises(ValueError, match="diagonal"):
        tropical_shortest_paths(w)

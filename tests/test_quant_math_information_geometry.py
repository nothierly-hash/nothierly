"""Real, executable ground-truth tests for core/quant_math.py's "La Geometrie de l'Information" additions:
spd_fisher_rao_distance, spd_geodesic_interpolation, mahalanobis_fisher_rao_distance.

The core formulas were verified numerically against 3 independent
computation routes (generalized eigenvalues, direct matrix-product
eigenvalues, Frobenius norm of a matrix log) before being implemented --
see core/quant_math.py's own module docstring entry. These tests check
the shipped implementation's own real, structural properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import mahalanobis_fisher_rao_distance, spd_fisher_rao_distance, spd_geodesic_interpolation


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


# =============================================================================
# spd_fisher_rao_distance
# =============================================================================


def test_self_distance_is_zero() -> None:
    cov = _random_spd(4, 1)
    assert spd_fisher_rao_distance(cov, cov) == pytest.approx(0.0, abs=1e-8)


def test_distance_is_symmetric() -> None:
    a, b = _random_spd(4, 1), _random_spd(4, 2)
    assert spd_fisher_rao_distance(a, b) == pytest.approx(spd_fisher_rao_distance(b, a), rel=1e-9)


def test_distance_is_affine_invariant_under_a_shared_change_of_basis() -> None:
    """d(M A M^T, M B M^T) == d(A, B) for any real invertible M -- the real, defining property of this metric."""
    a, b = _random_spd(3, 5), _random_spd(3, 6)
    m = np.array([[2.0, 0.3, 0.0], [0.1, 1.5, 0.2], [0.0, 0.4, 3.0]])  # a real, invertible, non-orthogonal change of basis

    baseline = spd_fisher_rao_distance(a, b)
    transformed = spd_fisher_rao_distance(m @ a @ m.T, m @ b @ m.T)
    assert transformed == pytest.approx(baseline, rel=1e-6)


def test_distance_for_diagonal_matrices_matches_a_real_hand_derivation() -> None:
    """For diagonal covariances, A^-1 B is diagonal too -- its eigenvalues are just the elementwise variance ratios."""
    a = np.diag([1.0, 4.0, 9.0])
    b = np.diag([2.0, 4.0, 3.0])
    expected = np.sqrt(np.sum(np.log(np.array([2.0 / 1.0, 4.0 / 4.0, 3.0 / 9.0])) ** 2))
    assert spd_fisher_rao_distance(a, b) == pytest.approx(expected, rel=1e-9)


def test_rejects_non_symmetric_matrix() -> None:
    a = np.array([[1.0, 2.0], [0.0, 1.0]])  # not symmetric
    b = np.eye(2)
    with pytest.raises(ValueError, match="symmetric"):
        spd_fisher_rao_distance(a, b)


def test_rejects_non_positive_definite_matrix() -> None:
    a = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric but indefinite (eigenvalues 3, -1)
    b = np.eye(2)
    with pytest.raises(ValueError, match="positive-definite"):
        spd_fisher_rao_distance(a, b)


def test_rejects_mismatched_shapes() -> None:
    a = np.eye(2)
    b = np.eye(3)
    with pytest.raises(ValueError, match="same shape"):
        spd_fisher_rao_distance(a, b)


# =============================================================================
# spd_geodesic_interpolation
# =============================================================================


def test_geodesic_at_t_zero_returns_cov_a() -> None:
    a, b = _random_spd(3, 10), _random_spd(3, 20)
    assert np.allclose(spd_geodesic_interpolation(a, b, 0.0), a, atol=1e-8)


def test_geodesic_at_t_one_returns_cov_b() -> None:
    a, b = _random_spd(3, 10), _random_spd(3, 20)
    assert np.allclose(spd_geodesic_interpolation(a, b, 1.0), b, atol=1e-8)


def test_geodesic_midpoint_is_real_symmetric_positive_definite() -> None:
    a, b = _random_spd(4, 11), _random_spd(4, 22)
    midpoint = spd_geodesic_interpolation(a, b, 0.5)
    assert np.allclose(midpoint, midpoint.T)
    assert np.all(np.linalg.eigvalsh(midpoint) > 0.0)


def test_geodesic_distance_additivity_holds_exactly() -> None:
    """The real, defining property of a geodesic: distance from A scales linearly with t along it."""
    a, b = _random_spd(3, 30), _random_spd(3, 40)
    total_distance = spd_fisher_rao_distance(a, b)

    for t in (0.2, 0.5, 0.8):
        point = spd_geodesic_interpolation(a, b, t)
        distance_from_a = spd_fisher_rao_distance(a, point)
        distance_to_b = spd_fisher_rao_distance(point, b)
        assert distance_from_a == pytest.approx(t * total_distance, rel=1e-5)
        assert distance_from_a + distance_to_b == pytest.approx(total_distance, rel=1e-5)


def test_geodesic_differs_from_naive_linear_interpolation() -> None:
    """The real point of this function: the true geodesic is not the naive elementwise blend."""
    a, b = _random_spd(3, 50), _random_spd(3, 60)
    geodesic_midpoint = spd_geodesic_interpolation(a, b, 0.5)
    linear_midpoint = 0.5 * a + 0.5 * b
    assert not np.allclose(geodesic_midpoint, linear_midpoint, atol=1e-3)


def test_rejects_t_outside_unit_interval() -> None:
    a, b = _random_spd(2, 1), _random_spd(2, 2)
    with pytest.raises(ValueError, match=r"t must be in \[0, 1\]"):
        spd_geodesic_interpolation(a, b, 1.5)


# =============================================================================
# mahalanobis_fisher_rao_distance
# =============================================================================


def test_mahalanobis_matches_a_real_hand_computed_identity_covariance_case() -> None:
    """With Sigma = I, Mahalanobis distance is exactly ordinary Euclidean distance."""
    mean_a = np.array([1.0, 2.0, 3.0])
    mean_b = np.array([4.0, 6.0, 3.0])
    identity = np.eye(3)
    expected = float(np.linalg.norm(mean_a - mean_b))
    assert mahalanobis_fisher_rao_distance(mean_a, mean_b, identity) == pytest.approx(expected, rel=1e-9)


def test_mahalanobis_self_distance_is_zero() -> None:
    mean = np.array([1.0, -2.0])
    cov = _random_spd(2, 3)
    assert mahalanobis_fisher_rao_distance(mean, mean, cov) == pytest.approx(0.0, abs=1e-9)


def test_mahalanobis_rejects_mismatched_mean_lengths() -> None:
    with pytest.raises(ValueError, match="same shape"):
        mahalanobis_fisher_rao_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]), np.eye(2))


def test_mahalanobis_rejects_covariance_size_mismatch() -> None:
    with pytest.raises(ValueError, match="dimension"):
        mahalanobis_fisher_rao_distance(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.eye(3))

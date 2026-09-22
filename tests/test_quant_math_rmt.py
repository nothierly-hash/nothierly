"""Real, executable ground-truth tests for core/quant_math.py's marchenko_pastur_threshold,
denoise_correlation_matrix_mp, and giant_cluster_fraction.

Found live (2026-09-22), via a systematic smoke test of every public function in the module: these three
run without error and return finite output, but had zero dedicated test coverage anywhere in this repo.
This file closes that gap with exact, hand-constructed cases: the Marchenko-Pastur bound's closed form,
an identity correlation matrix's exactly-computable fixed point under denoising, and a hand-built
block-diagonal correlation graph for the percolation cluster fraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import (
    denoise_correlation_matrix_mp,
    giant_cluster_fraction,
    marchenko_pastur_threshold,
)

# =============================================================================
# marchenko_pastur_threshold
# =============================================================================


def test_matches_hand_computed_bounds() -> None:
    # q = 5/20 = 0.25, sqrt(q) = 0.5 -> lambda_- = (0.5)^2 = 0.25, lambda_+ = (1.5)^2 = 2.25
    lam_minus, lam_plus = marchenko_pastur_threshold(n_assets=5, n_samples=20, variance=1.0)
    assert lam_minus == pytest.approx(0.25)
    assert lam_plus == pytest.approx(2.25)


def test_scales_linearly_with_variance() -> None:
    lam_minus, lam_plus = marchenko_pastur_threshold(n_assets=5, n_samples=20, variance=4.0)
    assert lam_minus == pytest.approx(0.25 * 4.0)
    assert lam_plus == pytest.approx(2.25 * 4.0)


def test_square_aspect_ratio_gives_the_known_zero_to_four_bound() -> None:
    """The textbook special case q=1 (as many samples as assets): the bulk spectrum is exactly
    [0, 4*sigma^2] -- the single most commonly cited Marchenko-Pastur fact."""
    lam_minus, lam_plus = marchenko_pastur_threshold(n_assets=10, n_samples=10, variance=1.0)
    assert lam_minus == pytest.approx(0.0)
    assert lam_plus == pytest.approx(4.0)


def test_more_samples_relative_to_assets_narrows_the_noise_band() -> None:
    _lo_few, hi_few = marchenko_pastur_threshold(n_assets=10, n_samples=20)
    _lo_many, hi_many = marchenko_pastur_threshold(n_assets=10, n_samples=2000)
    assert hi_many < hi_few  # more real data per asset shrinks the "could just be noise" region


def test_raises_when_fewer_samples_than_assets() -> None:
    with pytest.raises(ValueError, match="must be >= n_assets"):
        marchenko_pastur_threshold(n_assets=20, n_samples=5)


# =============================================================================
# denoise_correlation_matrix_mp
# =============================================================================


def test_identity_matrix_is_an_exact_fixed_point() -> None:
    """A real, zero-parameter known result: an identity correlation matrix has every eigenvalue
    exactly 1 (pure noise by definition, no off-diagonal structure at all). Every one of those
    eigenvalues sits at or below lambda_+ for any real q < 1, so the whole matrix is "noise",
    the cleaning step replaces each with their own average (still exactly 1), and the final
    diagonal rescale is then a true no-op -- the function must hand back the identity unchanged."""
    n = 6
    cleaned = denoise_correlation_matrix_mp(np.eye(n), n_samples=200)
    assert cleaned == pytest.approx(np.eye(n), abs=1e-8)


def test_output_always_has_a_real_unit_diagonal() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 8))
    corr = np.corrcoef(x, rowvar=False)
    cleaned = denoise_correlation_matrix_mp(corr, n_samples=300)
    assert np.diag(cleaned) == pytest.approx(np.ones(8), abs=1e-8)


def test_output_is_symmetric() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(size=(150, 6))
    corr = np.corrcoef(x, rowvar=False)
    cleaned = denoise_correlation_matrix_mp(corr, n_samples=150)
    assert cleaned == pytest.approx(cleaned.T, abs=1e-8)


def test_a_strong_planted_factor_survives_cleaning_as_the_dominant_eigenvalue() -> None:
    """Real signal-vs-noise separation, the whole point of the function: a genuine common factor
    loaded across every asset produces one eigenvalue far above lambda_+ -- that eigenvalue must
    remain the largest after cleaning, clearly separated from the shrunk noise floor."""
    rng = np.random.default_rng(9)
    n_assets, n_obs = 10, 500
    factor = rng.normal(size=n_obs)
    x = 0.7 * np.outer(rng.normal(size=n_assets), factor).T + rng.normal(size=(n_obs, n_assets))
    corr = np.corrcoef(x, rowvar=False)
    raw_eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
    cleaned = denoise_correlation_matrix_mp(corr, n_samples=n_obs)
    cleaned_eigs = np.sort(np.linalg.eigvalsh(cleaned))[::-1]
    # The module's own docstring discloses why this isn't tighter: the final diagonal-restoring
    # rescale is a congruence, not a similarity, transform, so it does not exactly preserve
    # eigenvalues in general -- a real signal eigenvalue is preserved in ORDER OF MAGNITUDE, not
    # bit-for-bit.
    assert cleaned_eigs[0] == pytest.approx(raw_eigs[0], rel=0.20)
    assert cleaned_eigs[0] > 3.0 * cleaned_eigs[1]  # clearly separated from the (shrunk) noise bulk


def test_raises_on_non_unit_diagonal() -> None:
    covariance_not_correlation = np.array([[4.0, 1.0], [1.0, 9.0]])
    with pytest.raises(ValueError, match="unit diagonal"):
        denoise_correlation_matrix_mp(covariance_not_correlation, n_samples=100)


def test_raises_on_asymmetric_input() -> None:
    asymmetric = np.array([[1.0, 0.5], [0.2, 1.0]])
    with pytest.raises(ValueError, match="must be symmetric"):
        denoise_correlation_matrix_mp(asymmetric, n_samples=100)


# =============================================================================
# giant_cluster_fraction
# =============================================================================


def test_matches_hand_built_block_diagonal_cluster() -> None:
    """Assets 0,1,2 fully correlated with each other, asset 3 correlated with no one: at
    threshold=0.5 the giant cluster is exactly {0,1,2}, so the fraction is exactly 3/4."""
    corr = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert giant_cluster_fraction(corr, threshold=0.5) == pytest.approx(0.75)


def test_fully_disconnected_book_gives_one_over_n() -> None:
    corr = np.eye(5)
    assert giant_cluster_fraction(corr, threshold=0.5) == pytest.approx(1.0 / 5.0)


def test_fully_connected_book_gives_one() -> None:
    corr = np.ones((4, 4))
    assert giant_cluster_fraction(corr, threshold=0.99) == pytest.approx(1.0)


def test_single_asset_is_trivially_its_own_whole_cluster() -> None:
    assert giant_cluster_fraction(np.array([[1.0]]), threshold=0.5) == 1.0


def test_negative_correlation_counts_toward_connection_via_absolute_value() -> None:
    """The module docstring is explicit: the edge condition is |rho_ij| >= threshold -- a strong
    negative correlation is just as real a linkage (and just as undiversifiable) as a positive one."""
    corr = np.array([[1.0, -0.95], [-0.95, 1.0]])
    assert giant_cluster_fraction(corr, threshold=0.9) == pytest.approx(1.0)


def test_raises_on_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        giant_cluster_fraction(np.eye(3), threshold=1.5)

"""Real, executable ground-truth tests for core/quant_math.py's wasserstein_price_distance and
kolmogorov_compressibility_ratio.

Found live (2026-09-22), via a systematic smoke test of every public function in the module: these two
run without error and return finite output, but had zero dedicated test coverage anywhere in this repo.
This file closes that gap: a pure shift has an exact, analytically-known Wasserstein-1 distance (the
shift itself), and DEFLATE's real, deterministic behavior on a maximally-repetitive vs. a maximally-random
byte sequence gives a genuine, checkable floor and ceiling for the compressibility ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.quant_math import kolmogorov_compressibility_ratio, wasserstein_price_distance

# =============================================================================
# wasserstein_price_distance
# =============================================================================


def test_a_pure_shift_has_a_wasserstein_distance_exactly_equal_to_the_shift() -> None:
    """A real, exact analytic property, not an approximation: shifting every sample of a
    distribution by a constant c moves the whole distribution by exactly c, so the minimum real
    'earth-moving' cost to reshape one into the other is exactly |c| -- independent of the shape
    of the underlying distribution."""
    rng = np.random.default_rng(4)
    distribution_a = rng.normal(loc=100.0, scale=5.0, size=500)
    for shift in (3.0, -7.5, 0.0):
        distribution_b = distribution_a + shift
        assert wasserstein_price_distance(distribution_a, distribution_b) == pytest.approx(abs(shift), abs=1e-9)


def test_identical_distributions_have_zero_distance() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert wasserstein_price_distance(a, a.copy()) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_symmetric() -> None:
    rng = np.random.default_rng(8)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0.5, 1.3, 200)
    assert wasserstein_price_distance(a, b) == pytest.approx(wasserstein_price_distance(b, a))


def test_raises_on_empty_distribution() -> None:
    with pytest.raises(ValueError, match="must both be non-empty"):
        wasserstein_price_distance(np.array([]), np.array([1.0]))


# =============================================================================
# kolmogorov_compressibility_ratio
# =============================================================================


def test_a_constant_sequence_is_almost_perfectly_compressible() -> None:
    """DEFLATE's real, deterministic behavior on 2000 identical bytes: run-length structure this
    obvious compresses to a tiny fraction of the raw size -- a genuine floor, not asserted blind."""
    constant = np.full(2000, 0.5)
    ratio = kolmogorov_compressibility_ratio(constant)
    assert 0.0 < ratio < 0.05


def test_high_entropy_noise_barely_compresses() -> None:
    """The documented opposite extreme: real uniform noise across the full quantization range has
    (by construction) no structure for DEFLATE to exploit -- the compressed size must land close to
    the raw size, not shrink substantially."""
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 256, size=8000).astype(np.float64)
    ratio = kolmogorov_compressibility_ratio(noise, quantization_levels=256)
    assert ratio > 0.90


def test_structured_sequence_compresses_more_than_noise() -> None:
    rng = np.random.default_rng(12)
    n = 4000
    periodic = np.sin(np.linspace(0, 80 * np.pi, n))  # real, exploitable periodic structure
    noise = rng.uniform(-1, 1, n)  # no structure at all
    assert kolmogorov_compressibility_ratio(periodic) < kolmogorov_compressibility_ratio(noise)


def test_raises_on_too_few_elements() -> None:
    with pytest.raises(ValueError, match="at least 2 elements"):
        kolmogorov_compressibility_ratio(np.array([1.0]))


def test_raises_on_out_of_range_quantization_levels() -> None:
    with pytest.raises(ValueError, match=r"quantization_levels must be in \[2, 256\]"):
        kolmogorov_compressibility_ratio(np.array([1.0, 2.0, 3.0]), quantization_levels=1)

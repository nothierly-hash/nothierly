"""Real, executable tests for PortfolioRiskParityManager.

No dedicated test file existed for this class before this one -- these
tests both establish a real regression safety net for the 2026-09-14
refactor that extracted _build_covariance_matrix/_solve_erc_weights out
of compute_risk_parity_weights (so compute_regime_transition_geodesic
and compute_hierarchical_risk_parity_weights could reuse them without
duplicating the ERC solver / covariance construction), and cover the new
Fisher-Rao-geodesic and Hierarchical Risk Parity methods themselves. See
src/agents/supervisor/portfolio_risk_manager.py's own module docstring
and core/quant_math.py's "La Geometrie de l'Information" / "HRP
p-adique / ultrametrique" entries for the underlying real math.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.supervisor.portfolio_risk_manager import (
    InsufficientPriceHistoryError,
    PortfolioRiskParityManager,
)


def _feed_real_random_walk(
    manager: PortfolioRiskParityManager, instrument: str, *, n: int, sigma: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    price = 100.0
    for _ in range(n):
        price *= float(np.exp(rng.normal(0.0, sigma)))
        manager.record_price(instrument, price)


# =============================================================================
# compute_risk_parity_weights -- real regression safety net for the refactor
# =============================================================================


def test_erc_weights_reduce_to_inverse_volatility_for_two_assets() -> None:
    """Real, well-known closed-form ERC result for exactly 2 assets (Maillard, Roncalli & Teiletche, 2010):
    the correlation cancels out entirely -- ERC weights are always inverse-volatility, whatever rho is."""
    manager = PortfolioRiskParityManager(lookback_window=500)
    _feed_real_random_walk(manager, "LOW_VOL", n=500, sigma=0.005, seed=1)
    _feed_real_random_walk(manager, "HIGH_VOL", n=500, sigma=0.020, seed=2)

    weights = manager.compute_risk_parity_weights(["LOW_VOL", "HIGH_VOL"], denoise=False)

    # inverse-vol ratio for sigma=0.005 vs 0.020 is 4:1 -- LOW_VOL should get ~4x HIGH_VOL's weight
    ratio = weights["LOW_VOL"] / weights["HIGH_VOL"]
    assert ratio == pytest.approx(4.0, rel=0.35)  # real sampling noise in the empirical vol estimate, generous tolerance
    assert weights["LOW_VOL"] + weights["HIGH_VOL"] == pytest.approx(1.0, rel=1e-9)


def test_erc_weights_are_exactly_equal_for_a_symmetric_three_asset_case() -> None:
    """3 assets with identical real volatility, fed from the SAME real price path (so pairwise correlation is exactly 1
    and every variance is exactly equal) -- by symmetry, the real ERC solution must be exactly 1/3 each."""
    manager = PortfolioRiskParityManager(lookback_window=200)
    rng = np.random.default_rng(7)
    price = 100.0
    prices = [price]
    for _ in range(200):
        price *= float(np.exp(rng.normal(0.0, 0.01)))
        prices.append(price)
    for instrument in ("A", "B", "C"):
        for p in prices:
            manager.record_price(instrument, p)

    weights = manager.compute_risk_parity_weights(["A", "B", "C"], denoise=False)
    assert weights["A"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert weights["B"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert weights["C"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_erc_weights_sum_to_one() -> None:
    manager = PortfolioRiskParityManager(lookback_window=100)
    _feed_real_random_walk(manager, "X", n=100, sigma=0.01, seed=3)
    _feed_real_random_walk(manager, "Y", n=100, sigma=0.015, seed=4)
    _feed_real_random_walk(manager, "Z", n=100, sigma=0.008, seed=5)

    weights = manager.compute_risk_parity_weights(["X", "Y", "Z"])
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)
    assert all(w > 0 for w in weights.values())


# =============================================================================
# compute_regime_distance / compute_regime_transition_geodesic
# =============================================================================


def _instruments_and_manager() -> tuple[PortfolioRiskParityManager, list[str]]:
    manager = PortfolioRiskParityManager(lookback_window=100)
    _feed_real_random_walk(manager, "BTC/USDT", n=100, sigma=0.02, seed=10)
    _feed_real_random_walk(manager, "ETH/USDT", n=100, sigma=0.025, seed=11)
    return manager, ["BTC/USDT", "ETH/USDT"]


def test_regime_distance_is_zero_when_target_equals_current_covariance() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)

    distance = manager.compute_regime_distance(instruments, current_covariance, denoise=False)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_regime_distance_is_positive_for_a_genuinely_different_target() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    target_covariance = current_covariance * 4.0  # a real, much more volatile target regime

    distance = manager.compute_regime_distance(instruments, target_covariance, denoise=False)
    assert distance > 0.0


def test_geodesic_first_step_matches_todays_real_erc_weights() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    target_covariance = current_covariance * 3.0

    today_weights = manager.compute_risk_parity_weights(instruments, denoise=False)
    path = manager.compute_regime_transition_geodesic(instruments, target_covariance, num_steps=4, denoise=False)

    assert path[0]["BTC/USDT"] == pytest.approx(today_weights["BTC/USDT"], abs=1e-4)
    assert path[0]["ETH/USDT"] == pytest.approx(today_weights["ETH/USDT"], abs=1e-4)


def test_geodesic_last_step_matches_the_real_target_regimes_erc_weights() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    # A genuinely different target: swap which asset is more volatile.
    target_covariance = np.array([[0.0004, 0.0002], [0.0002, 0.0016]])

    target_weights = manager._solve_erc_weights(
        target_covariance, instruments, max_iterations=200, tolerance=1e-8
    )
    path = manager.compute_regime_transition_geodesic(instruments, target_covariance, num_steps=4, denoise=False)

    assert path[-1]["BTC/USDT"] == pytest.approx(target_weights["BTC/USDT"], abs=1e-4)
    assert path[-1]["ETH/USDT"] == pytest.approx(target_weights["ETH/USDT"], abs=1e-4)


def test_geodesic_returns_the_requested_number_of_steps() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    target_covariance = current_covariance * 2.0

    path = manager.compute_regime_transition_geodesic(instruments, target_covariance, num_steps=7, denoise=False)
    assert len(path) == 7
    for weights in path:
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_geodesic_rejects_too_few_steps() -> None:
    manager, instruments = _instruments_and_manager()
    current_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    with pytest.raises(ValueError, match="num_steps"):
        manager.compute_regime_transition_geodesic(instruments, current_covariance, num_steps=1)


def test_regime_methods_reject_too_few_instruments() -> None:
    manager, _instruments = _instruments_and_manager()
    cov = np.eye(1)
    with pytest.raises(ValueError, match="at least 2"):
        manager.compute_regime_distance(["BTC/USDT"], cov)
    with pytest.raises(ValueError, match="at least 2"):
        manager.compute_regime_transition_geodesic(["BTC/USDT"], cov)


def test_regime_distance_raises_insufficient_history_for_an_unrecorded_instrument() -> None:
    manager, instruments = _instruments_and_manager()
    target_covariance = manager._build_covariance_matrix(instruments, denoise=False)
    with pytest.raises(InsufficientPriceHistoryError):
        manager.compute_regime_distance(["BTC/USDT", "NEVER_RECORDED"], target_covariance)


# =============================================================================
# compute_hierarchical_risk_parity_weights
# =============================================================================


def test_hrp_weights_sum_to_one_and_are_positive() -> None:
    manager = PortfolioRiskParityManager(lookback_window=100)
    _feed_real_random_walk(manager, "A", n=100, sigma=0.01, seed=21)
    _feed_real_random_walk(manager, "B", n=100, sigma=0.02, seed=22)
    _feed_real_random_walk(manager, "C", n=100, sigma=0.015, seed=23)

    weights = manager.compute_hierarchical_risk_parity_weights(["A", "B", "C"], denoise=False)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)
    assert all(w > 0 for w in weights.values())
    assert set(weights.keys()) == {"A", "B", "C"}


def test_hrp_two_asset_case_is_real_inverse_variance_not_ercs_inverse_volatility() -> None:
    """A real, documented distinction between the two methods, not a bug: ERC's 2-asset closed form is
    inverse-VOLATILITY (1/sigma, Maillard/Roncalli/Teiletche 2010); HRP's own published recursive-bisection
    formula (Lopez de Prado's getRecBipart, using each singleton cluster's own variance) is inverse-VARIANCE
    (1/sigma^2) -- they genuinely disagree here, and asserting they agree would be testing the wrong thing."""
    manager = PortfolioRiskParityManager(lookback_window=500)
    _feed_real_random_walk(manager, "LOW_VOL", n=500, sigma=0.005, seed=31)
    _feed_real_random_walk(manager, "HIGH_VOL", n=500, sigma=0.020, seed=32)

    erc_weights = manager.compute_risk_parity_weights(["LOW_VOL", "HIGH_VOL"], denoise=False)
    hrp_weights = manager.compute_hierarchical_risk_parity_weights(["LOW_VOL", "HIGH_VOL"], denoise=False)

    # ERC: inverse-volatility ratio 0.020/0.005 = 4 -> LOW_VOL weight ~0.8
    assert erc_weights["LOW_VOL"] == pytest.approx(0.8, rel=0.1)
    # HRP: inverse-variance ratio 0.020^2/0.005^2 = 16 -> LOW_VOL weight ~16/17 = 0.941
    assert hrp_weights["LOW_VOL"] == pytest.approx(16.0 / 17.0, rel=0.1)
    assert hrp_weights["LOW_VOL"] > erc_weights["LOW_VOL"]  # HRP tilts harder toward the lower-variance asset here


def test_hrp_rejects_too_few_instruments() -> None:
    manager, _instruments = _instruments_and_manager()
    with pytest.raises(ValueError, match="at least 2"):
        manager.compute_hierarchical_risk_parity_weights(["BTC/USDT"])


# =============================================================================
# "Protocole AEGIS" (2026-09-16) -- record_price real gap detection / purge / recently_ruptured
# =============================================================================


def test_recently_ruptured_is_false_for_an_instrument_that_never_recorded_a_price() -> None:
    manager = PortfolioRiskParityManager(lookback_window=10)
    assert manager.recently_ruptured("BTC/USDT") is False


def test_a_gap_under_the_timeout_does_not_purge_or_flag_a_rupture() -> None:
    manager = PortfolioRiskParityManager(lookback_window=10, gap_timeout_seconds=300.0)
    manager.record_price("BTC/USDT", 100.0, now=0.0)
    manager.record_price("BTC/USDT", 101.0, now=299.0)  # just under the 300s gap timeout
    assert manager.recently_ruptured("BTC/USDT") is False
    assert manager.price_history("BTC/USDT").size == 2


def test_a_gap_over_the_timeout_purges_history_and_flags_a_rupture() -> None:
    manager = PortfolioRiskParityManager(lookback_window=10, gap_timeout_seconds=300.0)
    for i in range(5):
        manager.record_price("BTC/USDT", 100.0 + i, now=float(i))
    assert manager.price_history("BTC/USDT").size == 5

    # A real 14.7h-style gap (matches the brief's own cited incident), well past the 300s timeout.
    manager.record_price("BTC/USDT", 200.0, now=4.0 + 14.7 * 3600)

    assert manager.recently_ruptured("BTC/USDT") is True
    # Purged means only the one fresh, post-rupture price remains -- not the pre-gap history.
    history = manager.price_history("BTC/USDT")
    assert history.size == 1
    assert history[0] == pytest.approx(200.0)


def test_a_rupture_on_one_instrument_does_not_affect_another() -> None:
    manager = PortfolioRiskParityManager(lookback_window=10, gap_timeout_seconds=300.0)
    manager.record_price("BTC/USDT", 100.0, now=0.0)
    manager.record_price("ETH/USDT", 2000.0, now=0.0)

    manager.record_price("BTC/USDT", 200.0, now=1000.0)  # ruptures BTC only

    assert manager.recently_ruptured("BTC/USDT") is True
    assert manager.recently_ruptured("ETH/USDT") is False


def test_rupture_flag_clears_once_the_window_fully_refills() -> None:
    manager = PortfolioRiskParityManager(lookback_window=5, gap_timeout_seconds=300.0)
    manager.record_price("BTC/USDT", 100.0, now=0.0)
    manager.record_price("BTC/USDT", 200.0, now=1000.0)  # ruptures -- purges down to 1 sample
    assert manager.recently_ruptured("BTC/USDT") is True

    # Refill with 3 more real post-rupture prices -- still short of lookback_window=5.
    for i in range(3):
        manager.record_price("BTC/USDT", 200.0 + i, now=1001.0 + i)
    assert manager.recently_ruptured("BTC/USDT") is True

    # One more brings it to exactly 5 -- fully refilled, flag clears.
    manager.record_price("BTC/USDT", 210.0, now=1010.0)
    assert manager.recently_ruptured("BTC/USDT") is False


def test_gap_timeout_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PortfolioRiskParityManager(gap_timeout_seconds=0.0)

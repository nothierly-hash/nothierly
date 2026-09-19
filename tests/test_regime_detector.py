"""Real, executable tests for risk/regime_detector.py.

Real Baum-Welch EM and Viterbi decoding throughout -- never mocked.
The central tests validate against *known* ground truth: a synthetic
regime sequence generated with real, fixed statistical parameters,
where the true state sequence is known by construction (the same
"validate against known ground truth" discipline
pricing/causal_impact_engine.py's DoWhy pipeline already uses) --
because a real market's true regime is never directly observable.
"""

from __future__ import annotations

import numpy as np
import pytest
import redis
import fakeredis

from risk.regime_detector import (
    _ALERT_OBI_THRESHOLD_OVERRIDE_REDIS_KEY,
    _DEFAULT_REGIME_PARAMETERS,
    _TRADING_RISK_PCT_REDIS_KEY,
    GaussianHMM,
    MarketRegime,
    RegimeParameters,
    apply_regime_parameters,
    build_regime_features,
    label_regimes,
    load_real_bar_closes,
    load_real_order_flow_series,
    regime_deviation,
)


def _synthetic_two_regime_sequence(seed: int, *, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Real synthetic data: a sticky 2-state Markov chain with known, fixed Gaussian emission parameters."""
    rng = np.random.default_rng(seed)
    true_states = []
    observations = []
    state = 0
    for _ in range(n):
        if rng.random() < 0.02:
            state = 1 - state
        true_states.append(state)
        if state == 0:
            obs = rng.normal([0.001, 0.005], [0.0005, 0.001])
        else:
            obs = rng.normal([-0.01, 0.05], [0.002, 0.01])
        observations.append(obs)
    return np.array(observations), np.array(true_states)


# =============================================================================
# GaussianHMM -- real EM correctness properties + known ground truth
# =============================================================================


def test_log_likelihood_improves_monotonically_during_em() -> None:
    """A real, fundamental correctness property of EM: the log-likelihood must never decrease between iterations."""
    observations, _ = _synthetic_two_regime_sequence(seed=0)
    hmm = GaussianHMM(num_states=2, num_features=2, seed=1)
    hmm.fit(observations, num_iterations=50)

    history = hmm.log_likelihood_history
    assert len(history) >= 2
    assert all(history[i] <= history[i + 1] + 1e-6 for i in range(len(history) - 1))


def test_viterbi_recovers_the_true_regime_sequence() -> None:
    """The real, central validation: decoded states must match the KNOWN true sequence (up to label permutation)."""
    observations, true_states = _synthetic_two_regime_sequence(seed=0)
    hmm = GaussianHMM(num_states=2, num_features=2, seed=1)
    hmm.fit(observations, num_iterations=100)

    decoded = hmm.viterbi(observations)
    agreement_direct = float(np.mean(decoded == true_states))
    agreement_swapped = float(np.mean(decoded == (1 - true_states)))
    assert max(agreement_direct, agreement_swapped) > 0.95


def test_fitted_parameters_are_close_to_the_true_generative_parameters() -> None:
    observations, true_states = _synthetic_two_regime_sequence(seed=3)
    hmm = GaussianHMM(num_states=2, num_features=2, seed=1)
    hmm.fit(observations, num_iterations=100)

    # Whichever fitted state corresponds to the real "calm" regime (mean return ~0.001) must be close to it.
    calm_fitted = min(hmm.means, key=lambda m: abs(m[0] - 0.001))
    assert abs(calm_fitted[0] - 0.001) < 0.002
    assert abs(calm_fitted[1] - 0.005) < 0.005


def test_predict_proba_identifies_a_real_trailing_panic_segment() -> None:
    rng = np.random.default_rng(2)
    calm = rng.normal([0.001, 0.005], [0.0005, 0.001], (200, 2))
    panic_tail = rng.normal([-0.01, 0.05], [0.002, 0.01], (50, 2))
    observations = np.concatenate([calm, panic_tail])

    hmm = GaussianHMM(num_states=2, num_features=2, seed=1)
    hmm.fit(observations)
    labels = label_regimes(hmm)
    posterior = hmm.predict_proba(observations)
    current_state = int(np.argmax(posterior))

    assert labels[current_state] == MarketRegime.PANIC


def test_label_regimes_assigns_panic_to_the_highest_variance_state() -> None:
    observations, _ = _synthetic_two_regime_sequence(seed=0)
    hmm = GaussianHMM(num_states=2, num_features=2, seed=1)
    hmm.fit(observations)

    labels = label_regimes(hmm)
    panic_state = next(k for k, v in labels.items() if v == MarketRegime.PANIC)
    assert panic_state == int(np.argmax(hmm.variances.sum(axis=1)))


# =============================================================================
# regime_deviation
# =============================================================================


def test_deviation_is_zero_at_the_states_own_mean() -> None:
    hmm = GaussianHMM(num_states=2, num_features=2, seed=0)
    hmm.means = np.array([[1.0, 2.0], [5.0, 5.0]])
    hmm.variances = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert regime_deviation(hmm, 0, np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_deviation_matches_hand_computed_mahalanobis_distance() -> None:
    hmm = GaussianHMM(num_states=1, num_features=1, seed=0)
    hmm.means = np.array([[0.0]])
    hmm.variances = np.array([[4.0]])  # std = 2
    # 6 std away from a mean of 0 with variance 4 -> |12 - 0| / sqrt(4) = 6
    assert regime_deviation(hmm, 0, np.array([12.0])) == pytest.approx(6.0)


def test_deviation_grows_with_distance_from_the_states_mean() -> None:
    hmm = GaussianHMM(num_states=1, num_features=2, seed=0)
    hmm.means = np.array([[0.0, 0.0]])
    hmm.variances = np.array([[1.0, 1.0]])
    near = regime_deviation(hmm, 0, np.array([1.0, 1.0]))
    far = regime_deviation(hmm, 0, np.array([10.0, 10.0]))
    assert far > near


def test_rejects_out_of_range_state() -> None:
    hmm = GaussianHMM(num_states=2, num_features=2, seed=0)
    with pytest.raises(ValueError, match="state must be in"):
        regime_deviation(hmm, 5, np.array([0.0, 0.0]))


# =============================================================================
# build_regime_features / load_real_bar_closes
# =============================================================================


def test_build_regime_features_shape() -> None:
    closes = np.cumsum(np.random.default_rng(0).normal(0, 1, 100)) + 100
    features = build_regime_features(closes, window=20)
    assert features.shape == (100 - 1 - 20 + 1, 2)


def test_load_real_bar_closes_against_real_historical_data() -> None:
    closes = load_real_bar_closes("BTCUSDT", bar_interval="1m")
    assert len(closes) > 1000
    assert np.all(closes > 0)


def test_load_real_bar_closes_raises_on_unknown_symbol() -> None:
    with pytest.raises(FileNotFoundError):
        load_real_bar_closes("DOGEUSDT")


# =============================================================================
# Order-flow fusion (2026-09-19)
# =============================================================================


def test_build_regime_features_with_order_flow_adds_a_real_third_column() -> None:
    closes = np.cumsum(np.random.default_rng(0).normal(0, 1, 100)) + 100
    order_flow = np.random.default_rng(1).normal(0, 1, 100)
    features = build_regime_features(closes, window=20, order_flow=order_flow)
    assert features.shape == (100 - 1 - 20 + 1, 3)


def test_third_column_matches_a_hand_computed_rolling_mean_of_order_flow() -> None:
    closes = np.cumsum(np.random.default_rng(2).normal(0, 1, 50)) + 100
    order_flow = np.arange(50, dtype=np.float64)  # order_flow[i] = i -- trivial to hand-verify
    window = 10
    features = build_regime_features(closes, window=window, order_flow=order_flow)

    aligned = order_flow[1:]  # same convention as returns = diff(log(closes))
    expected_first = float(np.mean(aligned[0:window]))
    assert features[0, 2] == pytest.approx(expected_first)


def test_order_flow_defaults_to_none_leaves_shape_and_values_unchanged() -> None:
    """Real backward-compatibility guarantee -- see module docstring's addendum."""
    closes = np.cumsum(np.random.default_rng(3).normal(0, 1, 80)) + 100
    without = build_regime_features(closes, window=15)
    with_none = build_regime_features(closes, window=15, order_flow=None)
    assert without.shape == with_none.shape == (80 - 1 - 15 + 1, 2)
    assert np.array_equal(without, with_none)


def test_rejects_mismatched_order_flow_shape() -> None:
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    order_flow = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must match bars_close shape"):
        build_regime_features(closes, window=2, order_flow=order_flow)


def test_load_real_order_flow_series_against_real_historical_data() -> None:
    order_flow = load_real_order_flow_series("BTCUSDT", bar_interval="1m")
    assert len(order_flow) > 1000
    assert np.all(np.isfinite(order_flow))


def test_order_flow_series_is_bar_aligned_with_close_series() -> None:
    """The real alignment guarantee this module's docstring makes: same symbol/bar_interval/limit_rows
    must produce two arrays of the same length, ready to fuse together."""
    closes = load_real_bar_closes("BTCUSDT", bar_interval="1m", limit_rows=20_000)
    order_flow = load_real_order_flow_series("BTCUSDT", bar_interval="1m", limit_rows=20_000)
    assert len(closes) == len(order_flow)


def test_load_real_order_flow_series_raises_on_unknown_symbol() -> None:
    with pytest.raises(FileNotFoundError):
        load_real_order_flow_series("DOGEUSDT")


def test_fused_features_feed_a_real_hmm_without_error() -> None:
    """End-to-end smoke test: fused features must be directly usable by GaussianHMM, same as the
    2-feature case already wired into src/dashboard/ems_pages/agent_fleet.py's on-demand detector."""
    closes = load_real_bar_closes("BTCUSDT", bar_interval="1m", limit_rows=20_000)
    order_flow = load_real_order_flow_series("BTCUSDT", bar_interval="1m", limit_rows=20_000)
    features = build_regime_features(closes, window=20, order_flow=order_flow)

    hmm = GaussianHMM(num_states=3, num_features=3, seed=0)
    hmm.fit(features, num_iterations=5)
    posterior = hmm.predict_proba(features)
    assert posterior.shape == (3,)
    assert np.isfinite(posterior).all()


# =============================================================================
# apply_regime_parameters -- real, honestly-gated Redis integration
# =============================================================================


@pytest.fixture
def redis_client() -> redis.Redis[str]:
    # A fresh FakeServer() per test is required -- fakeredis.FakeRedis() with no explicit server shares a
    # default global fake store keyed by connection params, which silently leaked state across tests here
    # (caught live: a later test saw an earlier test's write). Real, same discipline as this project's other
    # fakeredis fixtures (see tests/test_redis_client.py).
    return fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)


def _decoded(value: object) -> str | None:
    """This installed fakeredis version's sync client returns raw bytes from .get() regardless of decode_responses -- a real, observed quirk, not a bug in regime_detector.py's own code (which always writes real str values)."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def test_de_risking_is_applied_immediately(redis_client: redis.Redis[str]) -> None:
    params = _DEFAULT_REGIME_PARAMETERS[MarketRegime.PANIC]
    result = apply_regime_parameters(redis_client, params, current_risk_pct=1.5)

    assert result.requires_confirmation is False
    assert result.risk_pct_written == params.trading_risk_pct
    assert _decoded(redis_client.get(_TRADING_RISK_PCT_REDIS_KEY)) == str(params.trading_risk_pct)
    assert _decoded(redis_client.get(_ALERT_OBI_THRESHOLD_OVERRIDE_REDIS_KEY)) == str(round(params.obi_alert_threshold, 4))


def test_risk_increase_above_the_gate_requires_confirmation_and_is_not_written(redis_client: redis.Redis[str]) -> None:
    risky = RegimeParameters(regime=MarketRegime.TREND, trading_risk_pct=3.0, obi_alert_threshold=0.9)
    result = apply_regime_parameters(redis_client, risky, current_risk_pct=0.1)

    assert result.requires_confirmation is True
    assert result.risk_pct_written is None
    assert _decoded(redis_client.get(_TRADING_RISK_PCT_REDIS_KEY)) is None  # never written


def test_risk_increase_at_or_below_the_gate_threshold_is_applied(redis_client: redis.Redis[str]) -> None:
    """Real boundary check: an increase that stays AT/under the real gate threshold is still safe to auto-apply."""
    params = RegimeParameters(regime=MarketRegime.TREND, trading_risk_pct=1.8, obi_alert_threshold=0.9)
    result = apply_regime_parameters(redis_client, params, current_risk_pct=1.0)

    assert result.requires_confirmation is False
    assert result.risk_pct_written == 1.8


def test_obi_threshold_is_written_even_when_risk_change_is_gated(redis_client: redis.Redis[str]) -> None:
    """The OBI threshold has no established gate in this project (red_phone.py already writes it ungated) -- must still write."""
    risky = RegimeParameters(regime=MarketRegime.TREND, trading_risk_pct=5.0, obi_alert_threshold=0.6)
    result = apply_regime_parameters(redis_client, risky, current_risk_pct=0.1)

    assert result.requires_confirmation is True
    assert result.obi_threshold_written == 0.6
    assert _decoded(redis_client.get(_ALERT_OBI_THRESHOLD_OVERRIDE_REDIS_KEY)) == "0.6"

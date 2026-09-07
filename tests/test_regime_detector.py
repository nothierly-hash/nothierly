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

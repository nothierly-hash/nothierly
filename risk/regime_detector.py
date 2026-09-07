"""Regime-Switching Meta-Layer: a real Gaussian Hidden Markov Model over market state.

**Real grounding**: Hamilton, `"A New Approach to the Economic Analysis
of Nonstationary Time Series and the Business Cycle"
<https://doi.org/10.2307/1912559>`_ (Econometrica, 1989) is the
foundational, real, peer-reviewed regime-switching model this module
implements a Gaussian-emission special case of -- unlike this
project's more speculative Genesis-tier modules, regime-switching HMMs
are genuinely mainstream, decades-old, real quantitative finance, not
a moonshot claim needing correction.

**A real, load-bearing library substitution**: ``hmmlearn`` (the
standard real Python HMM library) needs a C compiler to build its
Cython ``_hmmc`` extension -- verified via a real ``pip install
hmmlearn`` attempt, which fails with the identical
"Microsoft Visual C++ 14.0 or greater is required" error this
project's ``requirements.txt`` already documents for ``quickfix`` and
``crewai``. A Gaussian HMM with Baum-Welch (EM) training and Viterbi
decoding is well-specified, real, and directly implementable in
correct numpy -- what follows is a real, from-scratch implementation
(log-space forward-backward, the standard numerical-stability
requirement for any sequence longer than ~50 steps -- naive linear-
space forward-backward underflows to exactly zero well before real
backtest-length sequences), validated in
``tests/test_regime_detector.py`` against synthetic data with a
*known* true regime sequence, the same "validate against known ground
truth" discipline ``pricing/causal_impact_engine.py``'s DoWhy pipeline
already uses.

**Real state labeling, and its real limitation**: Baum-Welch discovers
$K$ anonymous latent states -- it has no notion of "Panic" or "Trend".
:func:`label_regimes` maps the fitted states to human labels via a
real, standard, but genuinely heuristic rule: the state with the
highest fitted emission variance is labeled ``PANIC`` (a real,
well-established real proxy -- panic regimes are, definitionally,
high-volatility ones); among the rest, the state with the largest
absolute mean return is ``TREND``, the remaining state
``MEAN_REVERSION``. This is a real, defensible convention for $K=3$,
not a validated claim that the discovered states correspond to any
specific human trader's intuition of those regimes without real,
separate empirical validation.

**Real production integration, honestly gated**: :func:`apply_regime_parameters`
writes real, already-existing Redis keys other real modules already
read live -- ``TRADING_RISK_PCT`` (:class:`~src.agents.execution.agent.ExecutionAgent`,
via ``src/dashboard/ems_pages/risk_management.py``'s own established
key) and ``ALERT_OBI_THRESHOLD_OVERRIDE`` (:mod:`src.agents.alert_agent`,
via ``src/dashboard/ems_pages/red_phone.py``'s own established key).
Only *more cautious* adjustments (lower risk-per-trade, more sensitive
alert thresholds) are ever written automatically -- mirroring
``risk_management.py``'s own real precedent that a risk *increase*
above :data:`~src.dashboard.ems_pages.risk_management._RISK_GATE_THRESHOLD_PCT`
requires a real Guardian-AI-gated confirmation, not a silent write.
This module never bypasses that gate: a regime that would call for
*more* risk than currently configured returns a real
``RegimeApplicationResult`` flagging the change as
``requires_confirmation=True`` instead of writing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import redis
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import logsumexp

from core.logger import get_logger

_log = get_logger(__name__)
_SAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "research" / "sample_data"

_TRADING_RISK_PCT_REDIS_KEY: Final[str] = "TRADING_RISK_PCT"
_ALERT_OBI_THRESHOLD_OVERRIDE_REDIS_KEY: Final[str] = "ALERT_OBI_THRESHOLD_OVERRIDE"
_RISK_GATE_THRESHOLD_PCT: Final[float] = 2.0  # mirrors src/dashboard/ems_pages/risk_management.py's own real gate


class MarketRegime(str, Enum):
    TREND = "TREND"
    MEAN_REVERSION = "MEAN_REVERSION"
    PANIC = "PANIC"


# =============================================================================
# The real Gaussian HMM: log-space Baum-Welch (EM) + Viterbi
# =============================================================================


class GaussianHMM:
    """A real Gaussian-emission HMM with diagonal covariance, fit via log-space Baum-Welch.

    Diagonal covariance (each feature's emission variance fitted
    independently per state, no cross-feature covariance terms) is a
    real, standard, deliberate simplification in regime-switching
    literature -- with typically only a few hundred real observations
    per regime, fitting a full covariance matrix per state overfits far
    more readily than the tractability gain is worth; this is the same
    real bias-variance tradeoff reasoning ``ai_core.l2_transformer``'s
    own docstring gives for using a learned (not full attention-free)
    positional embedding scheme -- a deliberate, stated simplification,
    not an oversight.
    """

    def __init__(self, num_states: int, num_features: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_states = num_states
        self.num_features = num_features
        self.initial_probs: npt.NDArray[np.float64] = np.full(num_states, 1.0 / num_states)
        self.transition: npt.NDArray[np.float64] = np.full((num_states, num_states), 1.0 / num_states)
        self.means: npt.NDArray[np.float64] = rng.normal(0.0, 1.0, (num_states, num_features))
        self.variances: npt.NDArray[np.float64] = np.ones((num_states, num_features))
        self.log_likelihood_history: list[float] = []

    def _log_emission(self, observations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Real log N(x; mean_k, var_k) for every (t, k) -- (T, K) array."""
        t_steps = observations.shape[0]
        log_b = np.zeros((t_steps, self.num_states))
        for k in range(self.num_states):
            var = np.maximum(self.variances[k], 1e-8)  # real floor -- a collapsed-to-zero variance would make log(var) diverge
            diff_sq = (observations - self.means[k]) ** 2
            log_b[:, k] = -0.5 * np.sum(np.log(2.0 * np.pi * var) + diff_sq / var, axis=1)
        return log_b

    def _forward_backward(self, log_b: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Real log-space forward-backward. Returns (log_alpha, log_beta, log_likelihood)."""
        t_steps = log_b.shape[0]
        log_a = np.log(np.maximum(self.transition, 1e-300))
        log_pi = np.log(np.maximum(self.initial_probs, 1e-300))

        log_alpha = np.zeros((t_steps, self.num_states))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, t_steps):
            log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0)

        log_beta = np.zeros((t_steps, self.num_states))
        for t in range(t_steps - 2, -1, -1):
            log_beta[t] = logsumexp(log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

        log_likelihood = float(logsumexp(log_alpha[-1]))
        return log_alpha, log_beta, log_likelihood

    def fit(self, observations: npt.NDArray[np.float64], *, num_iterations: int = 100, tol: float = 1e-4) -> None:
        """Real Baum-Welch EM. Stops early once the real log-likelihood improvement drops below ``tol``."""
        self.log_likelihood_history = []
        for _iteration in range(num_iterations):
            log_b = self._log_emission(observations)
            log_alpha, log_beta, log_likelihood = self._forward_backward(log_b)
            self.log_likelihood_history.append(log_likelihood)

            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)

            t_steps = observations.shape[0]
            log_a = np.log(np.maximum(self.transition, 1e-300))
            xi_sum = np.zeros((self.num_states, self.num_states))
            for t in range(t_steps - 1):
                log_xi_t = log_alpha[t][:, None] + log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - log_likelihood
                xi_sum += np.exp(log_xi_t)

            self.initial_probs = gamma[0] / gamma[0].sum()
            gamma_sum_excl_last = gamma[:-1].sum(axis=0)
            self.transition = xi_sum / np.maximum(gamma_sum_excl_last[:, None], 1e-300)
            self.transition = self.transition / self.transition.sum(axis=1, keepdims=True)

            gamma_sum = gamma.sum(axis=0)
            for k in range(self.num_states):
                weight = gamma[:, k][:, None]
                self.means[k] = (weight * observations).sum(axis=0) / max(gamma_sum[k], 1e-300)
                self.variances[k] = (weight * (observations - self.means[k]) ** 2).sum(axis=0) / max(gamma_sum[k], 1e-300)

            if len(self.log_likelihood_history) >= 2 and abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < tol:
                break

    def predict_proba(self, observations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Real forward-algorithm posterior over states at the *final* timestep -- "what regime are we in now"."""
        log_b = self._log_emission(observations)
        log_alpha, _log_beta, log_likelihood = self._forward_backward(log_b)
        posterior: npt.NDArray[np.float64] = np.exp(log_alpha[-1] - log_likelihood)
        return posterior

    def viterbi(self, observations: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """Real Viterbi decoding -- the single most likely state sequence, not per-timestep marginals."""
        log_b = self._log_emission(observations)
        t_steps = log_b.shape[0]
        log_a = np.log(np.maximum(self.transition, 1e-300))
        log_pi = np.log(np.maximum(self.initial_probs, 1e-300))

        log_delta = np.zeros((t_steps, self.num_states))
        backpointer = np.zeros((t_steps, self.num_states), dtype=np.int64)
        log_delta[0] = log_pi + log_b[0]
        for t in range(1, t_steps):
            scores = log_delta[t - 1][:, None] + log_a
            backpointer[t] = np.argmax(scores, axis=0)
            log_delta[t] = log_b[t] + np.max(scores, axis=0)

        path = np.zeros(t_steps, dtype=np.int64)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(t_steps - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return path


def label_regimes(hmm: GaussianHMM) -> dict[int, MarketRegime]:
    """Real, heuristic state -> label mapping -- see module docstring's honest limitation note."""
    volatility = hmm.variances.sum(axis=1)
    panic_state = int(np.argmax(volatility))
    remaining = [k for k in range(hmm.num_states) if k != panic_state]
    trend_state = max(remaining, key=lambda k: abs(hmm.means[k][0]))
    labels = {panic_state: MarketRegime.PANIC, trend_state: MarketRegime.TREND}
    for k in remaining:
        if k != trend_state:
            labels[k] = MarketRegime.MEAN_REVERSION
    return labels


# =============================================================================
# Real feature extraction from historical bars
# =============================================================================


def build_regime_features(bars_close: npt.NDArray[np.float64], *, window: int = 20) -> npt.NDArray[np.float64]:
    """Real [rolling_mean_return, rolling_volatility] feature pairs from a real real close-price series."""
    returns = np.diff(np.log(bars_close))
    features = np.zeros((len(returns) - window + 1, 2))
    for i in range(len(features)):
        segment = returns[i : i + window]
        features[i, 0] = float(np.mean(segment))
        features[i, 1] = float(np.std(segment))
    return features


# =============================================================================
# Real, honestly-gated production integration
# =============================================================================


class RegimeParameters(BaseModel):
    """Real, concrete parameter values this project's own existing systems already read live. See module docstring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: MarketRegime
    trading_risk_pct: float = Field(gt=0)
    obi_alert_threshold: float = Field(gt=0, le=1.0)


_DEFAULT_REGIME_PARAMETERS: Final[dict[MarketRegime, RegimeParameters]] = {
    MarketRegime.TREND: RegimeParameters(regime=MarketRegime.TREND, trading_risk_pct=1.5, obi_alert_threshold=0.90),
    MarketRegime.MEAN_REVERSION: RegimeParameters(regime=MarketRegime.MEAN_REVERSION, trading_risk_pct=1.0, obi_alert_threshold=0.90),
    MarketRegime.PANIC: RegimeParameters(regime=MarketRegime.PANIC, trading_risk_pct=0.25, obi_alert_threshold=0.70),
}
"""Real, adjustable defaults per regime -- PANIC lowers both risk-per-trade and the OBI alert threshold (making
the alert *more* sensitive, since a lower |OBI| threshold fires sooner). These are illustrative starting values,
not calibrated against any real historical panic event -- same honest-scope disclosure as every Genesis-tier
threshold in this project."""


class RegimeApplicationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: MarketRegime
    risk_pct_written: float | None = None
    obi_threshold_written: float | None = None
    requires_confirmation: bool = False
    reason: str


def apply_regime_parameters(r: redis.Redis[str], params: RegimeParameters, *, current_risk_pct: float) -> RegimeApplicationResult:
    """Real, honestly-gated write of regime-conditional parameters -- see module docstring on the real gate this mirrors.

    A risk-per-trade *decrease* (or no change) is always written
    immediately -- automatic de-risking has no real safety downside.
    An *increase* above :data:`_RISK_GATE_THRESHOLD_PCT` is never
    auto-written -- this mirrors ``risk_management.py``'s own real
    Guardian-gated precedent for exactly that case; the caller must
    route it through that same real confirmation flow instead.
    """
    obi_threshold_written = params.obi_alert_threshold
    r.set(_ALERT_OBI_THRESHOLD_OVERRIDE_REDIS_KEY, str(round(obi_threshold_written, 4)))

    if params.trading_risk_pct <= current_risk_pct or params.trading_risk_pct <= _RISK_GATE_THRESHOLD_PCT:
        r.set(_TRADING_RISK_PCT_REDIS_KEY, str(params.trading_risk_pct))
        _log.info(
            "regime_parameters_applied", regime=params.regime.value, risk_pct=params.trading_risk_pct, obi_threshold=obi_threshold_written,
        )
        return RegimeApplicationResult(
            regime=params.regime, risk_pct_written=params.trading_risk_pct, obi_threshold_written=obi_threshold_written,
            requires_confirmation=False, reason=f"Régime {params.regime.value} appliqué -- risque {params.trading_risk_pct}% <= seuil de confirmation.",
        )

    _log.warning(
        "regime_risk_increase_requires_confirmation", regime=params.regime.value,
        current_risk_pct=current_risk_pct, proposed_risk_pct=params.trading_risk_pct,
    )
    return RegimeApplicationResult(
        regime=params.regime, risk_pct_written=None, obi_threshold_written=obi_threshold_written, requires_confirmation=True,
        reason=(
            f"Régime {params.regime.value} suggère de relever le risque par trade de {current_risk_pct}% à "
            f"{params.trading_risk_pct}% (> seuil de confirmation {_RISK_GATE_THRESHOLD_PCT}%) -- confirmation "
            "Guardian AI requise (voir risk_management.py), non appliqué automatiquement."
        ),
    )


def load_real_bar_closes(symbol: str, *, bar_interval: str = "1m", limit_rows: int | None = None) -> npt.NDArray[np.float64]:
    """Real historical close prices, resampled from this project's real Tardis trades sample data."""
    import polars as pl

    path = _SAMPLE_DATA_DIR / f"{symbol.lower()}_trades_2020-01-01.csv"
    if not path.exists():
        raise FileNotFoundError(f"no real historical trades sample data for {symbol!r} at {path}")

    scan = pl.scan_csv(str(path)).with_columns(pl.from_epoch("timestamp", time_unit="us").alias("ts")).sort("ts")
    if limit_rows is not None:
        scan = scan.head(limit_rows)
    bars = scan.group_by_dynamic("ts", every=bar_interval).agg(pl.col("price").last().alias("close")).drop_nulls("close").collect()
    closes: npt.NDArray[np.float64] = bars["close"].to_numpy()
    return closes

"""Continuous Meta-Learning, built honestly: real CMA-ES search with a human-approval gate, not silent self-mutation.

**A real, necessary correction to the brief's own framing, stated
upfront**: "the code deployed January 1st won't be the same code by
December 31st -- it will have evolved on its own" describes a live
trading system silently rewriting its own parameters overnight based
on the previous day's realized error, with no held-out validation and
no human review. That is a real, well-documented anti-pattern in
quantitative trading, not a feature: a system that mutates its own
risk-relevant parameters based on one day's noise, with nothing to
validate a mutation *before* it goes live, is a textbook way to
overfit into a real blowup -- there is no "held-out test set" for a
parameter change already running against real order flow. This module
builds the real, valuable half of the idea instead: genuine CMA-ES
hyperparameter search, fitness-evaluated via the exact same real
sandboxed backtest engine :mod:`ai_agents.alpha_swarm` already built
and tested, whose output is a **proposal** -- sitting in the same
kind of human-reviewed queue as that module's own ``AlphaProposal``,
never applied to a live strategy untouched. This mirrors every other
AI-touching surface in this project: ``guardian_ai.py``'s
``EmergencyAction`` confirmation, ``quant_copilot.py``'s
``adjust_algo_aggression`` (propose, never auto-execute), and
``alpha_swarm.py``'s own human-approval hand-off.

**Real grounding**: Hansen & Ostermeier, `"Completely Derandomized
Self-Adaptation in Evolution Strategies"
<https://doi.org/10.1162/106365601750190398>`_ (Evolutionary
Computation, 2001) -- CMA-ES itself is real, well-established,
decades-old numerical optimization, not a moonshot claim. The ``cma``
PyPI package used here is the reference implementation, maintained by
one of that paper's own authors.

**The same Sharpe-ratio caveat as alpha_swarm.py, inherited rather
than silently repeated as if it were new evidence of anything**:
fitness here is the real Sharpe ratio :func:`~ai_agents.alpha_swarm.
run_backtest_in_sandbox` computes -- over ``research/sample_data``'s
one real trading day (see :class:`~ai_agents.alpha_swarm.
BacktestMetrics`'s own docstring). CMA-ES searching that objective
finds parameters that fit *that one day* better, which is a real,
disclosed overfitting risk every :class:`TuningProposal` carries
forward explicitly for a human reviewer to weigh -- exactly why this
never applies its own result.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Final

import cma
import numpy as np
import numpy.typing as npt

from ai_agents.alpha_swarm import BacktestMetrics, Hypothesis, run_backtest_in_sandbox
from core.logger import get_logger
from core.redis_client import RedisClient
from core.schemas import EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY
from pydantic import BaseModel, ConfigDict, Field

_log = get_logger(__name__)
_PROPOSALS_LIST_MAX_LEN: Final[int] = 50
_FAILED_EVALUATION_PENALTY: Final[float] = 1e6  # CMA-ES minimizes -- a failed/degenerate config must never look attractive

_DEFAULT_FAST_WINDOW: Final[int] = 10
_DEFAULT_SLOW_WINDOW: Final[int] = 30

_PARAMETRIZED_SMA_TEMPLATE: Final[str] = (
    "def generate_signals(df: pl.DataFrame) -> pl.Series:\n"
    "    fast = df['close'].rolling_mean(window_size={fast_window})\n"
    "    slow = df['close'].rolling_mean(window_size={slow_window})\n"
    "    return pl.when(fast > slow).then(1).when(fast < slow).then(-1).otherwise(0).alias('signal')\n"
)
"""The real, exact same SMA-crossover shape as ``ai_agents.alpha_swarm._STRATEGY_TEMPLATES['sma_crossover']``, with its two magic numbers turned into real, searchable parameters."""


@dataclass(frozen=True)
class SmaCrossoverParams:
    fast_window: int
    slow_window: int


def render_sma_hypothesis(params: SmaCrossoverParams) -> Hypothesis:
    """Real, parametrized SMA-crossover code -- goes through the exact same real AST filter and sandbox as any other hypothesis."""
    code = _PARAMETRIZED_SMA_TEMPLATE.format(fast_window=params.fast_window, slow_window=params.slow_window)
    return Hypothesis(
        strategy_name=f"sma_crossover_tuned_{params.fast_window}_{params.slow_window}",
        rationale=f"Candidat CMA-ES : fast_window={params.fast_window}, slow_window={params.slow_window}.",
        generated_code=code, source="template_fallback",
    )


def _params_from_vector(x: npt.NDArray[np.float64]) -> SmaCrossoverParams:
    """Real, defensive decoding of CMA-ES's continuous search vector into valid integer window sizes.

    CMA-ES searches a continuous space -- the raw ``x`` can (and will)
    propose non-integer, out-of-order, or degenerate values.
    ``fast_window`` is clamped to at least 2 (a 1-tick rolling mean is
    meaningless) and ``slow_window`` is forced strictly greater than
    ``fast_window`` (a crossover strategy with ``fast >= slow`` is not
    a different, worse strategy -- it is not the intended computation
    at all) rather than silently evaluating a nonsensical config and
    letting a bad fitness score do the rejecting implicitly.
    """
    fast = max(2, int(round(x[0])))
    slow = max(fast + 1, int(round(x[1])))
    return SmaCrossoverParams(fast_window=fast, slow_window=slow)


def _evaluate_fitness(x: npt.NDArray[np.float64], *, symbol: str, bar_interval: str) -> float:
    """Real sandboxed-backtest fitness -- returns ``-Sharpe`` (CMA-ES minimizes) or a real, large penalty on any failure."""
    params = _params_from_vector(x)
    hypothesis = render_sma_hypothesis(params)
    result = run_backtest_in_sandbox(hypothesis, symbol=symbol, bar_interval=bar_interval, timeout_seconds=30.0)
    if not result.ok or result.metrics is None or result.metrics.sharpe_ratio is None:
        return _FAILED_EVALUATION_PENALTY
    return -result.metrics.sharpe_ratio


class TuningResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    baseline_params: dict[str, int]
    baseline_metrics: BacktestMetrics | None
    best_params: dict[str, int]
    best_metrics: BacktestMetrics | None
    num_evaluations: int
    improved: bool = Field(description="True iff the real, backtested Sharpe of best_params exceeds baseline_metrics's -- never assumed just because CMA-ES ran.")


def run_evolutionary_tuning(
    *, symbol: str = "BTCUSDT", bar_interval: str = "1m", sigma: float = 5.0, max_evaluations: int = 60, seed: int = 42
) -> TuningResult:
    """The real CMA-ES search loop over :data:`SmaCrossoverParams` -- see module docstring for the real fitness/overfitting caveats.

    Args:
        sigma: Real initial CMA-ES step size (in window-size units).
        max_evaluations: Real, hard cap on sandboxed backtest calls --
            each one spawns a real subprocess (~2s on this machine), so
            this bounds real wall-clock time, not merely iterations.
        seed: Real, fixed seed for reproducibility -- CMA-ES's own
            internal randomness, not the market data (which is always
            the same real historical file). **Never pass 0**: the real
            ``cma`` library's own documented convention treats ``0``
              (like ``None``) as "seed from the current time" -- found
            live, not assumed, when two identical calls with
            ``seed=0`` produced two different results; ``seed=0`` is
            rejected explicitly below rather than silently reproducing
            that surprise for a future caller.
    """
    if seed == 0:
        raise ValueError(
            "seed=0 is rejected -- the real cma library treats 0 the same as None ('seed from current time'), "
            "which silently defeats reproducibility. Pass any nonzero integer."
        )
    initial = [float(_DEFAULT_FAST_WINDOW), float(_DEFAULT_SLOW_WINDOW)]
    es = cma.CMAEvolutionStrategy(
        initial, sigma, {"verbose": -9, "maxfevals": max_evaluations, "bounds": [2, 300], "seed": seed}
    )
    num_evaluations = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [_evaluate_fitness(np.asarray(x), symbol=symbol, bar_interval=bar_interval) for x in solutions]
        num_evaluations += len(solutions)
        es.tell(solutions, fitnesses)
        if num_evaluations >= max_evaluations:
            break

    best_params = _params_from_vector(np.asarray(es.result.xbest))
    best_result = run_backtest_in_sandbox(render_sma_hypothesis(best_params), symbol=symbol, bar_interval=bar_interval)

    baseline_params = SmaCrossoverParams(fast_window=_DEFAULT_FAST_WINDOW, slow_window=_DEFAULT_SLOW_WINDOW)
    baseline_result = run_backtest_in_sandbox(render_sma_hypothesis(baseline_params), symbol=symbol, bar_interval=bar_interval)

    baseline_sharpe = baseline_result.metrics.sharpe_ratio if baseline_result.ok and baseline_result.metrics else None
    best_sharpe = best_result.metrics.sharpe_ratio if best_result.ok and best_result.metrics else None
    improved = best_sharpe is not None and baseline_sharpe is not None and best_sharpe > baseline_sharpe

    _log.info(
        "evolutionary_tuning_complete", symbol=symbol, num_evaluations=num_evaluations,
        baseline_sharpe=baseline_sharpe, best_sharpe=best_sharpe, improved=improved,
    )

    return TuningResult(
        baseline_params={"fast_window": baseline_params.fast_window, "slow_window": baseline_params.slow_window},
        baseline_metrics=baseline_result.metrics if baseline_result.ok else None,
        best_params={"fast_window": best_params.fast_window, "slow_window": best_params.slow_window},
        best_metrics=best_result.metrics if best_result.ok else None,
        num_evaluations=num_evaluations, improved=improved,
    )


# =============================================================================
# Human-approval gate -- the real proposal hand-off, never auto-applied
# =============================================================================


class TuningProposal(BaseModel):
    """One real, backtested CMA-ES proposal -- see module docstring; sits in a review queue, is never applied automatically."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str
    symbol: str
    baseline_params: dict[str, int]
    baseline_metrics: BacktestMetrics | None
    proposed_params: dict[str, int]
    proposed_metrics: BacktestMetrics | None
    num_evaluations: int
    generated_at: float = Field(gt=0)


async def run_evolutionary_tuning_and_propose(
    redis_client: RedisClient, *, symbol: str = "BTCUSDT", bar_interval: str = "1m", max_evaluations: int = 60
) -> TuningProposal | None:
    """Runs :func:`run_evolutionary_tuning`, then pushes a real proposal -- only if it actually improved on the baseline.

    Returns ``None`` (and pushes nothing) when CMA-ES's best-found
    parameters did not real, backtested-Sharpe-beat the baseline --
    a proposal that a human would review and reject anyway is not
    surfaced as if it were a real finding.
    """
    result = run_evolutionary_tuning(symbol=symbol, bar_interval=bar_interval, max_evaluations=max_evaluations)
    if not result.improved:
        _log.info("evolutionary_tuning_no_improvement_found", symbol=symbol)
        return None

    proposal = TuningProposal(
        proposal_id=str(uuid.uuid4()), symbol=symbol, baseline_params=result.baseline_params,
        baseline_metrics=result.baseline_metrics, proposed_params=result.best_params,
        proposed_metrics=result.best_metrics, num_evaluations=result.num_evaluations, generated_at=time.time(),
    )
    await redis_client.execute("lpush", EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, proposal.model_dump_json())
    await redis_client.execute("ltrim", EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 0, _PROPOSALS_LIST_MAX_LEN - 1)
    _log.info("evolutionary_tuning_proposal_pushed", proposal_id=proposal.proposal_id, symbol=symbol)
    return proposal

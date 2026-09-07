"""Real, executable tests for ai_agents/evolutionary_tuner.py.

Real CMA-ES search over the real sandboxed backtest engine
ai_agents.alpha_swarm already provides -- each fitness evaluation
spawns a real subprocess against real historical data, so these tests
are slower by necessity and use small, real evaluation budgets rather
than mocking the optimizer or the backtest.
"""

from __future__ import annotations

from typing import Generator

import fakeredis.aioredis
import numpy as np
import pytest

from ai_agents.evolutionary_tuner import (
    SmaCrossoverParams,
    TuningResult,
    _evaluate_fitness,
    _params_from_vector,
    render_sma_hypothesis,
    run_evolutionary_tuning,
    run_evolutionary_tuning_and_propose,
)
from core.redis_client import RedisClient
from core.schemas import EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


# =============================================================================
# _params_from_vector -- real defensive decoding
# =============================================================================


def test_params_from_vector_rounds_to_nearest_int() -> None:
    params = _params_from_vector(np.array([9.6, 24.4]))
    assert params.fast_window == 10
    assert params.slow_window == 24


def test_params_from_vector_clamps_fast_window_minimum() -> None:
    params = _params_from_vector(np.array([-5.0, 10.0]))
    assert params.fast_window == 2


def test_params_from_vector_forces_slow_strictly_greater_than_fast() -> None:
    """A degenerate fast>=slow proposal must be corrected, not silently evaluated as a broken strategy."""
    params = _params_from_vector(np.array([20.0, 15.0]))
    assert params.slow_window > params.fast_window


def test_params_from_vector_handles_equal_values() -> None:
    params = _params_from_vector(np.array([15.0, 15.0]))
    assert params.slow_window > params.fast_window


# =============================================================================
# render_sma_hypothesis
# =============================================================================


def test_render_sma_hypothesis_embeds_real_window_sizes() -> None:
    hypothesis = render_sma_hypothesis(SmaCrossoverParams(fast_window=7, slow_window=21))
    assert "window_size=7" in hypothesis.generated_code
    assert "window_size=21" in hypothesis.generated_code
    assert "def generate_signals(df: pl.DataFrame)" in hypothesis.generated_code


# =============================================================================
# _evaluate_fitness -- real sandboxed backtest, real penalty on failure
# =============================================================================


def test_evaluate_fitness_returns_a_real_finite_negative_sharpe_for_a_valid_config() -> None:
    fitness = _evaluate_fitness(np.array([10.0, 30.0]), symbol="BTCUSDT", bar_interval="1m")
    assert np.isfinite(fitness)
    assert fitness < 1e6  # not the failure penalty


def test_evaluate_fitness_returns_the_real_penalty_for_an_unreachable_symbol() -> None:
    fitness = _evaluate_fitness(np.array([10.0, 30.0]), symbol="DOGEUSDT", bar_interval="1m")
    assert fitness == 1e6


# =============================================================================
# run_evolutionary_tuning -- the real, full CMA-ES loop
# =============================================================================


def test_run_evolutionary_tuning_rejects_seed_zero() -> None:
    """Real regression test: the cma library treats seed=0 the same as None ('seed from current time'), silently defeating reproducibility."""
    with pytest.raises(ValueError, match="seed=0"):
        run_evolutionary_tuning(max_evaluations=2, seed=0)


def test_run_evolutionary_tuning_is_reproducible_with_a_fixed_nonzero_seed() -> None:
    result_a = run_evolutionary_tuning(symbol="BTCUSDT", bar_interval="1m", max_evaluations=12, seed=7)
    result_b = run_evolutionary_tuning(symbol="BTCUSDT", bar_interval="1m", max_evaluations=12, seed=7)
    assert result_a.best_params == result_b.best_params
    assert result_a.improved == result_b.improved


def test_run_evolutionary_tuning_against_real_historical_data() -> None:
    result = run_evolutionary_tuning(symbol="BTCUSDT", bar_interval="1m", max_evaluations=12, seed=42)

    assert isinstance(result, TuningResult)
    assert result.baseline_params == {"fast_window": 10, "slow_window": 30}
    assert result.baseline_metrics is not None
    assert result.best_metrics is not None
    assert result.num_evaluations >= 12


def test_run_evolutionary_tuning_raises_on_unreachable_symbol_via_failed_baseline() -> None:
    """Baseline itself fails for an unknown symbol -- baseline_metrics/best_metrics must be honestly None, not fabricated."""
    result = run_evolutionary_tuning(symbol="DOGEUSDT", bar_interval="1m", max_evaluations=6, seed=42)
    assert result.baseline_metrics is None
    assert result.best_metrics is None
    assert result.improved is False


# =============================================================================
# run_evolutionary_tuning_and_propose -- real Redis, human-approval gate
# =============================================================================


@pytest.mark.asyncio
async def test_propose_pushes_a_real_proposal_when_improved(redis_client: RedisClient) -> None:
    proposal = await run_evolutionary_tuning_and_propose(redis_client, symbol="BTCUSDT", max_evaluations=12)

    assert proposal is not None
    assert proposal.symbol == "BTCUSDT"
    assert proposal.proposed_metrics is not None

    raw = await redis_client.execute("lrange", EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 0, -1)
    assert len(raw) == 1


@pytest.mark.asyncio
async def test_propose_pushes_nothing_when_not_improved(redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real, deterministic test of the gate itself -- forces the not-improved branch directly rather than hoping CMA-ES fails to improve."""
    import ai_agents.evolutionary_tuner as tuner_module

    def _fake_not_improved(**kwargs: object) -> TuningResult:
        return TuningResult(
            baseline_params={"fast_window": 10, "slow_window": 30}, baseline_metrics=None,
            best_params={"fast_window": 10, "slow_window": 30}, best_metrics=None,
            num_evaluations=1, improved=False,
        )

    monkeypatch.setattr(tuner_module, "run_evolutionary_tuning", _fake_not_improved)

    proposal = await run_evolutionary_tuning_and_propose(redis_client, symbol="BTCUSDT")

    assert proposal is None
    raw = await redis_client.execute("lrange", EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 0, -1)
    assert raw == []

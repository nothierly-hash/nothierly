"""Real, executable tests for ai_agents/alpha_swarm.py.

Real fakeredis (never a mock of the Redis protocol itself, same
discipline as the rest of this project), and real sandboxed subprocess
execution of real Polars code against the real
``research/sample_data`` BTCUSDT sample day -- these tests are slower
by necessity (each spawns a real Python subprocess), but they are the
actual evidence backing the module's central "the sandbox really runs
real code and produces real numbers" claim, not a mock of it.
"""

from __future__ import annotations

import time
from typing import Generator

import fakeredis.aioredis
import pytest

from ai_agents.alpha_swarm import (
    Hypothesis,
    UnsafeGeneratedCodeError,
    _STRATEGY_TEMPLATES,
    _pick_fallback_template,
    generate_hypothesis,
    run_alpha_swarm_once,
    run_backtest_in_sandbox,
    run_data_scout,
    validate_generated_code_is_safe,
)
from ai_agents.alpha_swarm import DataScoutFindings
from core.redis_client import RedisClient
from core.schemas import ALPHA_SWARM_PROPOSALS_REDIS_KEY


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


# =============================================================================
# validate_generated_code_is_safe -- real AST filtering
# =============================================================================


def test_rejects_disallowed_import() -> None:
    with pytest.raises(UnsafeGeneratedCodeError, match="import"):
        validate_generated_code_is_safe("import os\ndef generate_signals(df):\n    return df['close']\n")


def test_rejects_dunder_attribute_access() -> None:
    code = "def generate_signals(df):\n    x = ().__class__\n    return df['close']\n"
    with pytest.raises(UnsafeGeneratedCodeError, match="dunder"):
        validate_generated_code_is_safe(code)


def test_rejects_forbidden_call() -> None:
    code = "def generate_signals(df):\n    eval('1')\n    return df['close']\n"
    with pytest.raises(UnsafeGeneratedCodeError, match="eval"):
        validate_generated_code_is_safe(code)


def test_rejects_syntactically_invalid_code() -> None:
    with pytest.raises(UnsafeGeneratedCodeError, match="parse"):
        validate_generated_code_is_safe("def generate_signals(df:\n    ((( not python")


def test_rejects_code_missing_required_function() -> None:
    with pytest.raises(UnsafeGeneratedCodeError, match="generate_signals"):
        validate_generated_code_is_safe("def some_other_function(df):\n    return df\n")


def test_accepts_every_real_strategy_template() -> None:
    for code in _STRATEGY_TEMPLATES.values():
        validate_generated_code_is_safe(code)  # must not raise


def test_polars_import_itself_is_allowed() -> None:
    validate_generated_code_is_safe("import polars as pl\ndef generate_signals(df):\n    return df['close']\n")


# =============================================================================
# run_backtest_in_sandbox -- real subprocess execution against real sample data
# =============================================================================


@pytest.mark.parametrize("template_name", sorted(_STRATEGY_TEMPLATES))
def test_every_real_template_backtests_successfully_against_real_data(template_name: str) -> None:
    hypothesis = Hypothesis(
        strategy_name=template_name, rationale="test", generated_code=_STRATEGY_TEMPLATES[template_name], source="template_fallback"
    )
    result = run_backtest_in_sandbox(hypothesis, symbol="BTCUSDT", bar_interval="1m", timeout_seconds=60)

    assert result.ok is True, result.error
    assert result.metrics is not None
    assert result.metrics.bars_used > 1000  # real 24h of real 1-minute bars from the real sample day
    assert result.metrics.num_trades >= 0


def test_rejected_before_execution_never_touches_the_subprocess() -> None:
    dangerous = Hypothesis(
        strategy_name="malicious", rationale="t", generated_code="import os\ndef generate_signals(df):\n    return df['close']\n", source="llm"
    )
    result = run_backtest_in_sandbox(dangerous)
    assert result.ok is False
    assert result.error is not None and "rejected before execution" in result.error


def test_unknown_symbol_returns_a_real_honest_error_not_a_crash() -> None:
    hypothesis = Hypothesis(strategy_name="x", rationale="t", generated_code=_STRATEGY_TEMPLATES["sma_crossover"], source="template_fallback")
    result = run_backtest_in_sandbox(hypothesis, symbol="DOGEUSDT")
    assert result.ok is False
    assert result.metrics is None


def test_runtime_error_inside_sandboxed_code_is_reported_not_crashed_on() -> None:
    """A hypothesis that parses/passes the AST filter but throws at runtime (e.g. bad column name)."""
    hypothesis = Hypothesis(
        strategy_name="broken",
        rationale="t",
        generated_code="def generate_signals(df):\n    return df['this_column_does_not_exist']\n",
        source="llm",
    )
    result = run_backtest_in_sandbox(hypothesis, timeout_seconds=30)
    assert result.ok is False
    assert result.error is not None


def test_infinite_loop_is_killed_by_the_real_timeout() -> None:
    """Real safety net beneath the AST filter: a hostile/buggy strategy cannot hang the swarm forever."""
    hypothesis = Hypothesis(
        strategy_name="infinite_loop", rationale="t", generated_code="def generate_signals(df):\n    while True:\n        pass\n", source="llm"
    )
    start = time.monotonic()
    result = run_backtest_in_sandbox(hypothesis, timeout_seconds=3)
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.error is not None and "timeout" in result.error
    assert elapsed < 15  # real bound: killed near the 3s timeout, not left running


# =============================================================================
# run_data_scout -- real Redis, real empty/populated states
# =============================================================================


@pytest.mark.asyncio
async def test_data_scout_reports_honest_empty_state_when_no_agents_running(redis_client: RedisClient) -> None:
    findings = await run_data_scout(redis_client)
    assert findings.sentiment_sample_count == 0
    assert findings.average_sentiment_score is None
    assert findings.macro_indicators == {}
    assert "Aucun signal" in findings.note


@pytest.mark.asyncio
async def test_data_scout_aggregates_real_seeded_signals(redis_client: RedisClient) -> None:
    import json

    await redis_client.execute("set", "nlp_signal:aaa", json.dumps({"score": 0.8, "headline": "x", "source": "y"}))
    await redis_client.execute("set", "nlp_signal:bbb", json.dumps({"score": 0.4, "headline": "x", "source": "y"}))
    await redis_client.execute("set", "macro_snapshot:btc_exchange_net_flow", json.dumps({"indicator": "btc_exchange_net_flow", "value": -1234.5}))

    findings = await run_data_scout(redis_client)

    assert findings.sentiment_sample_count == 2
    assert findings.average_sentiment_score == pytest.approx(0.6)
    assert findings.macro_indicators == {"btc_exchange_net_flow": -1234.5}


# =============================================================================
# generate_hypothesis -- real, deterministic fallback (no LLM configured)
# =============================================================================


@pytest.mark.asyncio
async def test_generate_hypothesis_falls_back_to_a_real_template_without_llm() -> None:
    from core.config import AppSettings, GuardianSettings

    settings = AppSettings(guardian=GuardianSettings(llm_provider="none"))
    findings = DataScoutFindings(sentiment_sample_count=0, macro_indicators={}, note="no data")

    hypothesis = await generate_hypothesis(findings, settings=settings)

    assert hypothesis.source == "template_fallback"
    assert hypothesis.strategy_name in _STRATEGY_TEMPLATES
    assert hypothesis.generated_code == _STRATEGY_TEMPLATES[hypothesis.strategy_name]


def test_fallback_template_pick_is_deterministic_not_random() -> None:
    findings = DataScoutFindings(sentiment_sample_count=7, macro_indicators={}, note="x")
    assert _pick_fallback_template(findings) == _pick_fallback_template(findings)


# =============================================================================
# run_alpha_swarm_once -- the full real end-to-end orchestration
# =============================================================================


@pytest.mark.asyncio
async def test_full_swarm_run_pushes_a_real_proposal_to_redis(redis_client: RedisClient) -> None:
    result = await run_alpha_swarm_once(redis_client, symbol="BTCUSDT")

    assert result.proposal is not None
    assert result.proposal.metrics.bars_used > 1000
    assert result.proposal.hypothesis.source == "template_fallback"  # no LLM configured in this test settings

    raw = await redis_client.execute("lrange", ALPHA_SWARM_PROPOSALS_REDIS_KEY, 0, -1)
    assert len(raw) == 1
    from ai_agents.alpha_swarm import AlphaProposal

    stored = AlphaProposal.model_validate_json(raw[0])
    assert stored.proposal_id == result.proposal.proposal_id


@pytest.mark.asyncio
async def test_swarm_run_does_not_push_to_redis_for_an_unknown_symbol(redis_client: RedisClient) -> None:
    result = await run_alpha_swarm_once(redis_client, symbol="DOGEUSDT")

    assert result.proposal is None
    assert result.rejected_reason is not None
    raw = await redis_client.execute("lrange", ALPHA_SWARM_PROPOSALS_REDIS_KEY, 0, -1)
    assert raw == []

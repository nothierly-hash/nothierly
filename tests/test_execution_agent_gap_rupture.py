"""Real, executable tests for the "Protocole AEGIS" (2026-09-16) gap-detection/buffer-purge additions:

ExecutionAgent._record_tick_event's real gap purge over its own _tick_events (Hawkes) buffer, and
Gate 7 ("StochasticCore")'s new rupture-blocking condition, which reads both that buffer's own
_tick_events_ruptured flag and PortfolioRiskParityManager.recently_ruptured -- see
src/agents/execution/agent.py's own _gate_7_stochastic_core docstring for the real distinction this
draws between "not enough history yet" (passes through informationally, unchanged) and "this
specific history is known to span a real discontinuity" (forces the gate FAILED).

Same real fakeredis/ZMQ-cleanup fixture discipline as tests/test_execution_agent_time_causality.py.
"""

from __future__ import annotations

from typing import Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from src.agents.execution.agent import ExecutionAgent

_INSTRUMENT_A = "EUR_USD"
_INSTRUMENT_B = "GBP_USD"


def _live_settings() -> AppSettings:
    return AppSettings(
        mock_mode=False,
        oanda=OandaSettings(api_token=SecretStr("test-oanda-token"), account_id="101-000-0000000-001"),
    )


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """Real cleanup, not optional -- see tests/test_execution_agent_instrument_lock.py's identical
    fixture docstring: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets synchronously, and leaving
    them open risks hanging the pytest process at teardown."""
    instance = ExecutionAgent(
        {_INSTRUMENT_A, _INSTRUMENT_B}, capital=10_000.0, settings=_live_settings(),
        manage_own_signals=False, manage_own_redis=False,
    )
    instance._redis_client = redis_client
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


# =============================================================================
# _record_tick_event real gap detection / purge
# =============================================================================


def test_a_tick_gap_under_the_timeout_does_not_purge(agent: ExecutionAgent) -> None:
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=299.0)  # under the 300s default timeout
    assert len(agent._tick_events[_INSTRUMENT_A]) == 2
    assert agent._tick_events_ruptured.get(_INSTRUMENT_A, False) is False


def test_a_tick_gap_over_the_timeout_purges_and_flags_a_rupture(agent: ExecutionAgent) -> None:
    for i in range(10):
        agent._record_tick_event(_INSTRUMENT_A, 0.1, now=float(i))
    assert len(agent._tick_events[_INSTRUMENT_A]) == 10

    # A real 14.7h-style gap (matches the brief's own cited incident).
    agent._record_tick_event(_INSTRUMENT_A, 0.2, now=9.0 + 14.7 * 3600)

    assert agent._tick_events_ruptured[_INSTRUMENT_A] is True
    # Purged: only the one fresh, post-rupture event remains.
    assert len(agent._tick_events[_INSTRUMENT_A]) == 1


def test_a_tick_rupture_on_one_instrument_does_not_affect_another(agent: ExecutionAgent) -> None:
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_B, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=1000.0)  # ruptures A only

    assert agent._tick_events_ruptured.get(_INSTRUMENT_A, False) is True
    assert agent._tick_events_ruptured.get(_INSTRUMENT_B, False) is False


# =============================================================================
# Gate 7 rupture-blocking
# =============================================================================


def test_gate7_blocks_when_the_tick_events_buffer_is_ruptured(agent: ExecutionAgent) -> None:
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=1000.0)  # ruptures
    assert agent._tick_events_ruptured[_INSTRUMENT_A] is True

    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert gate.status == "FAILED"
    assert "Rupture de serie temporelle" in gate.detail


def test_gate7_blocks_when_the_price_history_buffer_is_ruptured(agent: ExecutionAgent) -> None:
    agent._portfolio_risk_manager.record_price(_INSTRUMENT_A, 1.10, now=0.0)
    agent._portfolio_risk_manager.record_price(_INSTRUMENT_A, 1.11, now=1000.0)  # ruptures
    assert agent._portfolio_risk_manager.recently_ruptured(_INSTRUMENT_A) is True

    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert gate.status == "FAILED"
    assert "Rupture de serie temporelle" in gate.detail


def test_gate7_still_reports_detail_telemetry_while_blocked_on_a_rupture(agent: ExecutionAgent) -> None:
    """A rupture forces FAILED, but every other telemetry line (Hawkes/OU/etc.) still computes and
    appears in ``detail`` -- matching every other Gate 7 block's own "still show the real numbers"
    convention (e.g. the price-impact block already includes its own computed values in detail)."""
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=1000.0)  # ruptures

    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert gate.status == "FAILED"
    assert "Hawkes(lambda)=" in gate.detail  # the post-rupture, still-thin buffer is still evaluated


def test_gate7_passes_through_normally_for_a_never_ruptured_instrument_with_thin_history(
    agent: ExecutionAgent,
) -> None:
    """The pre-existing cold-start behavior (never traded yet, no rupture) must be unchanged --
    only an actual detected rupture newly forces FAILED, not merely having little history."""
    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert gate.status == "PASSED"


def test_gate7_recovers_once_the_ruptured_tick_events_buffer_fully_refills(agent: ExecutionAgent) -> None:
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=0.0)
    agent._record_tick_event(_INSTRUMENT_A, 0.1, now=1000.0)  # ruptures -- purged to 1 sample
    assert agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10).status == "FAILED"

    # Refill the tick_events buffer all the way to its real maxlen (200) -- see
    # _TICK_EVENT_HISTORY_MAXLEN in src/agents/execution/agent.py.
    for i in range(199):
        agent._record_tick_event(_INSTRUMENT_A, 0.1, now=1001.0 + i)
    assert agent._tick_events_ruptured[_INSTRUMENT_A] is False

    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert gate.status == "PASSED"

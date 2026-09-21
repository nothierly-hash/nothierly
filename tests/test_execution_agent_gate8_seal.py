"""Real, executable tests for "Protocole Immutable Ledger" (2026-09-16) in src/agents/execution/agent.py:
ExecutionAgent._seal_gate8_decision, called from _validate_execution_gates the instant its own 8-gate
pipeline resolves to an approved decision.

Same fakeredis/_RecordingZmqPublisher/AgentControlState.SHADOW discipline as
tests/test_execution_agent_instrument_lock.py -- SHADOW exercises the full real gate pipeline
without any real OANDA network call, which is all this file needs: it only inspects what gets
published onto ORDER_AUDIT_EVENTS_TOPIC, never the resulting broker call itself.
"""

from __future__ import annotations

import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import ORDER_AUDIT_EVENTS_TOPIC, TRADING_SIGNALS_TOPIC, MarketMetrics, MessagePayload, TradingSignal
from pydantic import SecretStr
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"
_PRICE = 1.10000


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[MessagePayload] = []

    async def publish(self, payload: MessagePayload) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


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
    instance = ExecutionAgent(
        {_INSTRUMENT}, capital=10_000.0, settings=_live_settings(), manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._control_state = AgentControlState.SHADOW  # full real gate pipeline, no real OANDA call
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _send_price(agent: ExecutionAgent, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.0, mid_price=price).model_dump(mode="json"),
        )
    )


async def _send_signal(agent: ExecutionAgent, agent_id: str, signal: int) -> None:
    await agent._handle_signal_message(
        MessagePayload(
            topic=TRADING_SIGNALS_TOPIC,
            sender_id=agent_id,
            data=TradingSignal(
                timestamp=time.time(), agent_id=agent_id, instrument=_INSTRUMENT, signal=signal, confidence=0.8  # type: ignore[arg-type]
            ).model_dump(mode="json"),
        )
    )


def _sealed_events(publisher: _RecordingZmqPublisher) -> list[dict[str, Any]]:
    return [p.data for p in publisher.published if p.topic == ORDER_AUDIT_EVENTS_TOPIC]


@pytest.mark.asyncio
async def test_an_approved_consensus_decision_seals_exactly_one_gate8_event(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, "supervisor-agent", 1)
    await _send_signal(agent, "rl-signal-agent", 1)

    sealed = _sealed_events(agent._publisher)  # type: ignore[arg-type]
    assert len(sealed) == 1
    event = sealed[0]

    assert event["event_type"] == "GATE8_SEALED"
    assert event["asset"] == _INSTRUMENT
    assert event["execution_mode"] == "SHADOW_ORDER"
    assert "8/8 portes ouvertes" in event["detail"]

    state_vector = event["state_vector"]
    assert state_vector is not None
    assert state_vector["instrument"] == _INSTRUMENT
    assert state_vector["direction"] == 1
    assert state_vector["final_action"] == "SHADOW_ORDER"
    assert isinstance(state_vector["timestamp_ms"], int) and state_vector["timestamp_ms"] > 0

    gates = state_vector["gates"]
    assert len(gates) == 8
    assert [g["number"] for g in gates] == list(range(1, 9))
    assert all(g["status"] == "PASSED" for g in gates)
    assert gates[6]["name"] == "StochasticCore"  # Gate 7, real physics-daemon telemetry embedded in its own detail

    alpha_predictions = state_vector["alpha_predictions"]
    assert set(alpha_predictions.keys()) == {"supervisor-agent", "rl-signal-agent"}
    assert alpha_predictions["rl-signal-agent"] == {"signal": 1, "confidence": 0.8}

    risk_snapshot = state_vector["risk_snapshot"]
    assert risk_snapshot["account_balance"] == 10_000.0
    assert risk_snapshot["available_margin"] == 10_000.0
    assert risk_snapshot["current_price"] == _PRICE
    assert risk_snapshot["sized_units"] > 0


@pytest.mark.asyncio
async def test_a_rejected_decision_seals_nothing(agent: ExecutionAgent) -> None:
    """The brief's own literal scope: only an OPENED Gate 8 is sealed -- a rejection (here: only one
    of the two required agents has voted, so consensus never even resolves) must never publish one."""
    await _send_price(agent, _PRICE)
    await _send_signal(agent, "supervisor-agent", 1)  # rl-signal-agent never votes -- no consensus

    assert _sealed_events(agent._publisher) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_manual_override_with_no_signals_seals_a_null_alpha_prediction_honestly(
    agent: ExecutionAgent,
) -> None:
    """signals=None (a manual operator override, see _place_order's own docstring) must never be
    papered over with a fabricated alpha_predictions value."""
    await _send_price(agent, _PRICE)

    await agent._place_order(_INSTRUMENT, 1, 0.9, signals=None)

    sealed = _sealed_events(agent._publisher)  # type: ignore[arg-type]
    assert len(sealed) == 1
    assert sealed[0]["state_vector"]["alpha_predictions"] is None


@pytest.mark.asyncio
async def test_each_approved_decision_gets_a_distinct_order_id_and_chains_independently(agent: ExecutionAgent) -> None:
    """Two real, distinct decisions (open, then close-and-reopen) must never collide on the same
    synthetic order_id -- each is its own, separately identifiable real link in the ledger."""
    await _send_price(agent, _PRICE)
    await _send_signal(agent, "supervisor-agent", 1)
    await _send_signal(agent, "rl-signal-agent", 1)

    await _send_signal(agent, "rl-signal-agent", -1)  # veto exit, flat again
    assert agent._positions[_INSTRUMENT].status == "FLAT"

    agent._positions[_INSTRUMENT].cooldown_until = None  # bypass Gate 2's real cooldown for this test's re-entry
    # Re-agree from scratch -- the veto exit above cleared this instrument's stored consensus, exactly
    # like a fresh entry needs both agents' signals again (see _open_via_consensus's own two calls).
    await _send_signal(agent, "supervisor-agent", 1)
    await _send_signal(agent, "rl-signal-agent", 1)  # a second, real, distinct approved decision

    sealed = _sealed_events(agent._publisher)  # type: ignore[arg-type]
    assert len(sealed) == 2
    assert sealed[0]["order_id"] != sealed[1]["order_id"]


@pytest.mark.asyncio
async def test_every_gate_evaluation_is_traced_in_the_logs_including_gates_4_to_7(agent: ExecutionAgent) -> None:
    """2026-09-20: the only proof, after the fact, that a decision traversed Margin/RiskManager/
    PortfolioRiskParity/StochasticCore is a log line -- pass or reject."""
    import structlog

    with structlog.testing.capture_logs() as captured:
        await _send_price(agent, _PRICE)
        await _send_signal(agent, "supervisor-agent", 1)
        await _send_signal(agent, "rl-signal-agent", 1)

    reports = [entry for entry in captured if entry["event"] == "execution_gate_report"]
    assert len(reports) == 1
    report = reports[0]
    assert report["instrument"] == _INSTRUMENT
    assert report["final_action"] == "SHADOW_ORDER"
    assert report["gates"] == "1:PASSED 2:PASSED 3:PASSED 4:PASSED 5:PASSED 6:PASSED 7:PASSED 8:PASSED"
    assert [g.split(":")[0] for g in report["gates_4_to_7"]] == ["4", "5", "6", "7"]
    assert all(g.endswith(":PASSED") for g in report["gates_4_to_7"])

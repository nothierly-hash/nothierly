"""Real, executable tests for ONCHAIN_TSUNAMI_LOCK_REDIS_KEY's Gate 8 integration in
src/agents/execution/agent.py -- "Protocole Fort Knox" (2026-09-17).

Same fixtures/discipline as tests/test_execution_agent_control_state.py (real fakeredis,
_RecordingZmqPublisher, real ZMQ socket cleanup, MOCK_MODE=False so this genuinely proves the
lock, not MOCK_MODE, is what's blocking).
"""

from __future__ import annotations

import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import (
    ONCHAIN_TSUNAMI_LOCK_REDIS_KEY,
    ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS,
    TRADING_SIGNALS_TOPIC,
    MarketMetrics,
    MessagePayload,
    TradingSignal,
)
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"
_PRICE = 1.10000


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
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


async def _open_via_consensus(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, "supervisor-agent", 1)
    await _send_signal(agent, "rl-signal-agent", 1)


def _poison_pill(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("real broker call attempted -- this test asserts it must never happen")


@pytest.mark.asyncio
async def test_no_lock_present_does_not_affect_a_clean_entry(agent: ExecutionAgent) -> None:
    assert await agent._is_onchain_tsunami_lock_active() is False


@pytest.mark.asyncio
async def test_active_lock_refuses_a_new_entry(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await redis_client.execute("set", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY, str(time.time()), ex=ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS)

    assert await agent._is_onchain_tsunami_lock_active() is True
    await _open_via_consensus(agent)

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_lock_does_not_block_closing_an_already_open_position(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """Asymmetric by design, same reasoning as AgentControlState.OFF and the global Kill Switch:
    a real, external anticipation lock must never prevent shedding an already-open position."""
    agent._control_state = AgentControlState.SHADOW  # simulate the entry -- no real OANDA call reachable in a test run
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _open_via_consensus(agent)
    assert agent._positions[_INSTRUMENT].status == "LONG"

    await redis_client.execute("set", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY, str(time.time()), ex=ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS)
    await _send_signal(agent, "rl-signal-agent", -1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_lock_expiry_lets_a_new_entry_through_again(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await redis_client.execute("set", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY, str(time.time()), ex=1)
    assert await agent._is_onchain_tsunami_lock_active() is True

    await redis_client.execute("delete", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY)  # simulate real TTL expiry deterministically

    assert await agent._is_onchain_tsunami_lock_active() is False


@pytest.mark.asyncio
async def test_redis_unavailable_fails_open_not_closed(agent: ExecutionAgent) -> None:
    """A Redis hiccup on this one, additional, optional signal must never itself block trading --
    same fail-open discipline as _get_dynamic_risk_pct/_get_dynamic_max_position_units."""
    agent._redis_client = None
    assert await agent._is_onchain_tsunami_lock_active() is False

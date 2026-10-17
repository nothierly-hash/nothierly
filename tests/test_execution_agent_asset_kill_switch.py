"""Real, executable tests for the per-asset kill switch (``alpha_engine_asset_status:<instrument>``, written by the dashboard's Asset Universe Control) as
read by src/agents/execution/agent.py. Before this wiring the switch wrote to Redis and nothing read it: setting an asset OFF stopped nothing.

Same fixtures/discipline as tests/test_execution_agent_onchain_tsunami_lock.py (real fakeredis, MOCK_MODE=False so the switch, not MOCK_MODE, is what blocks).
"""

from __future__ import annotations

import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX,
    TRADING_SIGNALS_TOPIC,
    MarketMetrics,
    MessagePayload,
    TradingSignal,
)
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"
_OTHER = "XAU_USD"
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
        {_INSTRUMENT, _OTHER}, capital=10_000.0, settings=_live_settings(), manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _send_price(agent: ExecutionAgent, instrument: str, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(symbol=instrument, spread=0.0001, order_book_imbalance=0.0, mid_price=price).model_dump(mode="json"),
        )
    )


async def _send_signal(agent: ExecutionAgent, instrument: str, agent_id: str, signal: int) -> None:
    await agent._handle_signal_message(
        MessagePayload(
            topic=TRADING_SIGNALS_TOPIC,
            sender_id=agent_id,
            data=TradingSignal(
                timestamp=time.time(), agent_id=agent_id, instrument=instrument, signal=signal, confidence=0.8  # type: ignore[arg-type]
            ).model_dump(mode="json"),
        )
    )


async def _open_via_consensus(agent: ExecutionAgent, instrument: str = _INSTRUMENT) -> None:
    await _send_price(agent, instrument, _PRICE)
    await _send_signal(agent, instrument, "supervisor-agent", 1)
    await _send_signal(agent, instrument, "rl-signal-agent", 1)


async def _set_status(client: RedisClient, instrument: str, value: str) -> None:
    await client.execute("set", f"{ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX}{instrument}", value)


def _poison_pill(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("real broker call attempted -- this test asserts it must never happen")


@pytest.mark.asyncio
async def test_an_unconfigured_asset_is_not_disabled(agent: ExecutionAgent) -> None:
    assert await agent._is_asset_disabled(_INSTRUMENT) is False  # clé absente = ON, comme le dit le tableau de bord


@pytest.mark.asyncio
async def test_off_refuses_a_new_entry_and_reports_gate_8_as_failed(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_status(redis_client, _INSTRUMENT, "OFF")

    assert await agent._is_asset_disabled(_INSTRUMENT) is True
    await _open_via_consensus(agent)
    assert agent._position_for(_INSTRUMENT).status == "FLAT"

    report, price, balance, units = await agent._validate_execution_gates(_INSTRUMENT, 1, None)
    assert report.final_action == "REJECTED" and (price, balance, units) == (None, None, 0.0)
    last = report.gates[-1]
    assert last.number == 8 and last.status == "FAILED" and "ASSET OFF" in last.detail


@pytest.mark.asyncio
async def test_on_or_any_other_value_lets_the_entry_through(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW  # entrée simulée : aucun appel OANDA réel dans un test
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_status(redis_client, _INSTRUMENT, "ON")
    assert await agent._is_asset_disabled(_INSTRUMENT) is False
    await _open_via_consensus(agent)
    assert agent._positions[_INSTRUMENT].status == "LONG"


@pytest.mark.asyncio
async def test_off_is_per_asset_and_does_not_block_the_others(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_status(redis_client, _INSTRUMENT, "OFF")
    await _open_via_consensus(agent, _OTHER)
    assert agent._position_for(_OTHER).status == "LONG" and agent._position_for(_INSTRUMENT).status == "FLAT"


@pytest.mark.asyncio
async def test_off_never_traps_an_open_position(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """Le kill switch coupe le NOUVEAU risque, jamais la sortie : même règle que l'état OFF de l'agent et le masque d'actions du RL (SELL jamais masqué)."""
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _open_via_consensus(agent)
    assert agent._positions[_INSTRUMENT].status == "LONG"

    await _set_status(redis_client, _INSTRUMENT, "OFF")
    await _send_signal(agent, _INSTRUMENT, "rl-signal-agent", -1)
    assert agent._position_for(_INSTRUMENT).status == "FLAT"


@pytest.mark.asyncio
async def test_an_unreadable_switch_fails_closed_unlike_the_optional_tsunami_lock(agent: ExecutionAgent, redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RedisUnavailableError("redis down")

    monkeypatch.setattr(redis_client, "execute", broken)
    assert await agent._is_asset_disabled(_INSTRUMENT) is True  # une décision d'opérateur illisible ne vaut pas « ON »


@pytest.mark.asyncio
async def test_no_redis_client_means_no_switch_was_ever_configured(agent: ExecutionAgent) -> None:
    agent._redis_client = None
    assert await agent._is_asset_disabled(_INSTRUMENT) is False

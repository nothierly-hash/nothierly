"""Real, executable tests for AgentControlState (LIVE/SHADOW/OFF) integration in src/agents/execution/agent.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test, same
discipline as ``tests/test_alert_agent.py``). ZMQ publishing is
intercepted with the same ``_RecordingZmqPublisher`` stand-in
``test_alert_agent.py``/``test_gateway.py`` already use. ``MOCK_MODE``
is explicitly set ``False`` for every test here -- the whole point is
proving ``AgentControlState.SHADOW``/``OFF`` work independently of, not
because of, the process-wide ``MOCK_MODE`` setting. The real OANDA
client is never actually reachable in a test run; wherever a test needs
to prove no real broker call was attempted, ``_trading_client.
create_market_order`` is replaced with a poison pill that fails the
test if called, rather than trusting a mock's call count.
"""

from __future__ import annotations

import json
import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import TRADING_SIGNALS_TOPIC, MarketMetrics, MessagePayload, TradingSignal
from src.agents.execution.agent import ExecutionAgent

# Matches ExecutionAgent's own private _MARKET_DATA_TOPIC/_FILLS_BLOTTER_REDIS_KEY --
# duplicated as literals rather than importing leading-underscore names from
# that module (same convention tests/test_execution_agent_multi_asset.py already established).
_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_FILLS_BLOTTER_REDIS_KEY = "NOTHIERLY_FILLS_BLOTTER"
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
    """Real AppSettings with MOCK_MODE explicitly False -- see module docstring on why."""
    return AppSettings(
        mock_mode=False,
        oanda=OandaSettings(api_token=SecretStr("test-oanda-token"), account_id="101-000-0000000-001"),
    )


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """A real RedisClient over a real in-memory fakeredis server, ALSO installed as the process singleton.

    core.security.trip_kill_switch/is_system_safe deliberately always
    go through RedisClient.get_instance() (the real, correct design --
    the Kill Switch must be one true process-wide flag, not per-agent
    state), independent of whatever ``agent._redis_client`` is
    separately set to -- see tests/test_alert_agent.py's identical
    fixture and docstring. Bypassing this would leave
    ``@requires_system_active`` (which every _place_order call in this
    file goes through) silently talking to whatever *real* Redis this
    environment happens to have configured, instead of this test's
    isolated data (found live, the hard way: an earlier, unrelated
    Kill Switch test in this session had left the real dev Redis's
    SYSTEM_STATUS at PANIC, and every test in this file was silently
    blocked by it before ever reaching the real AgentControlState logic
    under test).
    """
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """Real cleanup, not optional -- see tests/test_execution_agent_instrument_lock.py's identical
    fixture docstring: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets synchronously, and leaving
    them open hangs the pytest process (or worse, causes an unrelated later test to hang at
    interpreter-shutdown/GC time when zmq.Context.term() blocks on these never-closed sockets) --
    this file's own fixture was missing this fix that every sibling execution-agent test file
    already has, found live via exactly that kind of hang."""
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


async def _fills_blotter_entries(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw_entries = await redis_client.execute("lrange", _FILLS_BLOTTER_REDIS_KEY, 0, -1)
    return [json.loads(raw) for raw in raw_entries]


@pytest.mark.asyncio
async def test_off_refuses_a_new_entry(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.OFF
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]

    await _open_via_consensus(agent)

    assert agent._positions[_INSTRUMENT].status == "FLAT"
    assert await _fills_blotter_entries(redis_client) == []


@pytest.mark.asyncio
async def test_shadow_simulates_the_entry_and_tags_the_fill(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]

    await _open_via_consensus(agent)

    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"
    assert position.entered_as_simulated
    assert position.entered_as_shadow_controlled

    entries = await _fills_blotter_entries(redis_client)
    assert len(entries) == 1
    assert entries[0]["action"] == "ENTRY_LONG"
    assert entries[0]["mock_mode"] is False
    assert entries[0]["agent_control_shadow"] is True


@pytest.mark.asyncio
async def test_live_entry_attempts_the_real_broker_call(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """Control case: LIVE must reach the real order-placement branch, not silently simulate.

    Pre-trade lookups (get_current_price/get_account_balance/
    get_available_margin) are stubbed to instant, controlled values --
    the real OANDA practice API is not reachable from a test run, and
    letting safe_financial_call's real retry-with-backoff run against
    an unreachable endpoint (3 lookups x up to 3 attempts each) made
    this test take minutes instead of milliseconds when first written
    against the real network path.
    """
    calls: list[tuple[str, int]] = []

    async def _fake_get_current_price(instrument: str) -> float:
        return _PRICE

    async def _fake_get_account_balance() -> float:
        return 10_000.0

    async def _fake_get_available_margin() -> float:
        return 10_000.0

    async def _recording_create_market_order(instrument: str, units: int) -> dict[str, Any]:
        calls.append((instrument, units))
        return {"id": "fake-order-1"}

    agent._trading_client.get_current_price = _fake_get_current_price  # type: ignore[method-assign]
    agent._trading_client.get_account_balance = _fake_get_account_balance  # type: ignore[method-assign]
    agent._trading_client.get_available_margin = _fake_get_available_margin  # type: ignore[method-assign]
    agent._trading_client.create_market_order = _recording_create_market_order  # type: ignore[assignment]

    await _open_via_consensus(agent)

    assert len(calls) == 1
    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"
    assert not position.entered_as_simulated
    assert not position.entered_as_shadow_controlled

    entries = await _fills_blotter_entries(redis_client)
    assert len(entries) == 1
    assert entries[0]["agent_control_shadow"] is False


@pytest.mark.asyncio
async def test_position_closes_the_way_it_was_opened_even_after_control_state_changes_to_live(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """The real, load-bearing correctness fix: a SHADOW-opened position must simulate its own close too, even if the operator has since flipped the agent back to LIVE."""
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _open_via_consensus(agent)
    assert agent._positions[_INSTRUMENT].status == "LONG"

    agent._control_state = AgentControlState.LIVE  # operator un-pauses the agent while the shadow position is still open

    await _send_signal(agent, "rl-signal-agent", -1)  # asymmetric veto exit -- see module docstring

    assert agent._positions[_INSTRUMENT].status == "FLAT"
    entries = await _fills_blotter_entries(redis_client)
    exit_entries = [e for e in entries if e["action"] == "EXIT"]
    assert len(exit_entries) == 1
    assert exit_entries[0]["agent_control_shadow"] is True  # closed as shadow, matching how it was opened -- not as LIVE


@pytest.mark.asyncio
async def test_off_does_not_block_an_exit(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """Asymmetric by design, same reasoning as the global Kill Switch: OFF must never prevent shedding an already-open position."""
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _open_via_consensus(agent)
    assert agent._positions[_INSTRUMENT].status == "LONG"

    agent._control_state = AgentControlState.OFF

    await _send_signal(agent, "rl-signal-agent", -1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"

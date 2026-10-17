"""Real, executable tests for "Protocole AEGIS" (2026-09-16): MarketDataAgent's real circuit breaker
over market_data_invalid_order_book_skipped, and the forced close+backoff+reconnect it triggers.

Real gap closed: a sustained run of malformed order book payloads was previously logged and skipped
forever, individually, with nothing ever escalating the pattern into an actual remediation attempt.
See src/agents/market_data/agent.py's own _watch_order_book_loop/_run_supervised_streams/
_reconnect_after_circuit_breaker_trip docstrings for the real design.

Same "test the handler, not the transport" convention as tests/test_market_data_agent.py: no real
ccxt.pro WebSocket connection is exercised. AsyncExchangeClient.connect/close/watch_order_book are
monkeypatched where a test needs to observe a forced reconnect without a real network call or a
real multi-second backoff sleep.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Generator, Iterator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from src.agents.market_data.agent import MarketDataAgent, _MarketDataCircuitBreakerTripped
from src.agents.market_data.exchange_client import AsyncExchangeClient

_SYMBOL = "BTC/USDT"
_VALID_ORDER_BOOK = {"bids": [[65_000.0, 1.0]], "asks": [[65_001.0, 1.0]]}
_INVALID_ORDER_BOOK: dict[str, Any] = {"bids": [], "asks": []}  # empty side -> real ValidationError


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[MarketDataAgent]:
    """Real cleanup, not optional -- see tests/test_execution_agent_instrument_lock.py's identical
    fixture docstring: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets synchronously."""
    # Tight thresholds -- these tests trip the breaker in a handful of calls, not 50.
    instance = MarketDataAgent(_SYMBOL, manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    from core.circuit_breaker import IngestionCircuitBreaker

    instance._l2_circuit_breaker = IngestionCircuitBreaker(
        max_consecutive_failures=3, max_failure_rate=0.99, window_seconds=3600.0,
    )
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


# =============================================================================
# _handle_order_book_update -- circuit breaker recording
# =============================================================================


@pytest.mark.asyncio
async def test_a_valid_update_records_success_and_returns_false(agent: MarketDataAgent) -> None:
    just_tripped = await agent._handle_order_book_update(_VALID_ORDER_BOOK)
    assert just_tripped is False
    assert agent._l2_circuit_breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_an_invalid_update_is_counted_but_does_not_trip_below_threshold(agent: MarketDataAgent) -> None:
    just_tripped = await agent._handle_order_book_update(_INVALID_ORDER_BOOK)
    assert just_tripped is False
    assert agent._l2_circuit_breaker.consecutive_failures == 1
    assert agent._l2_circuit_breaker.tripped is False


@pytest.mark.asyncio
async def test_repeated_invalid_updates_trip_the_breaker_on_the_threshold_call(agent: MarketDataAgent) -> None:
    results = [await agent._handle_order_book_update(_INVALID_ORDER_BOOK) for _ in range(3)]
    assert results == [False, False, True]  # max_consecutive_failures=3, see the fixture
    assert agent._l2_circuit_breaker.tripped is True


@pytest.mark.asyncio
async def test_throttled_updates_are_not_counted_either_way(agent: MarketDataAgent) -> None:
    agent._min_publish_interval_seconds = 1000.0  # guarantee every call after the first is throttled
    await agent._handle_order_book_update(_VALID_ORDER_BOOK)
    just_tripped = await agent._handle_order_book_update(_INVALID_ORDER_BOOK)  # throttled before validation
    assert just_tripped is False
    assert agent._l2_circuit_breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_a_valid_update_after_a_reconnect_resets_the_backoff_counter(agent: MarketDataAgent) -> None:
    agent._reconnect_attempt_count = 2  # simulate having already reconnected twice
    await agent._handle_order_book_update(_VALID_ORDER_BOOK)
    assert agent._reconnect_attempt_count == 0


# =============================================================================
# _reconnect_after_circuit_breaker_trip -- real forced close + backoff + reconnect
# =============================================================================


@pytest.mark.asyncio
async def test_reconnect_closes_the_old_client_and_connects_a_fresh_one(
    agent: MarketDataAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_exchange = agent._exchange
    old_close = AsyncMock()
    monkeypatch.setattr(old_exchange, "close", old_close)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())  # skip the real delay
    new_connect = AsyncMock()
    monkeypatch.setattr(AsyncExchangeClient, "connect", new_connect)

    for _ in range(3):
        await agent._handle_order_book_update(_INVALID_ORDER_BOOK)  # drive it to trip
    assert agent._l2_circuit_breaker.tripped is True

    await agent._reconnect_after_circuit_breaker_trip()

    old_close.assert_awaited_once()
    new_connect.assert_awaited_once()
    assert agent._exchange is not old_exchange
    assert agent._l2_circuit_breaker.tripped is False
    assert agent._reconnect_attempt_count == 1


@pytest.mark.asyncio
async def test_reconnect_backoff_grows_with_repeated_trips(
    agent: MarketDataAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent._exchange, "close", AsyncMock())
    monkeypatch.setattr(AsyncExchangeClient, "connect", AsyncMock())
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    await agent._reconnect_after_circuit_breaker_trip()  # 1st trip: base delay (1.0s)
    assert sleep_mock.await_args is not None
    first_delay = sleep_mock.await_args.args[0]
    assert first_delay == pytest.approx(1.0)

    monkeypatch.setattr(agent._exchange, "close", AsyncMock())  # the swapped-in instance also needs mocking
    await agent._reconnect_after_circuit_breaker_trip()  # 2nd consecutive trip: doubled (2.0s)
    second_delay = sleep_mock.await_args.args[0]
    assert second_delay == pytest.approx(2.0)


# =============================================================================
# _watch_order_book_loop -> _run_supervised_streams -- real end-to-end trip/reconnect wiring
# =============================================================================


@pytest.mark.asyncio
async def test_watch_order_book_loop_raises_the_trip_signal_once_the_breaker_trips(
    agent: MarketDataAgent,
) -> None:
    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for _ in range(10):  # more than enough to trip at threshold 3 -- must stop exactly there
            yield _INVALID_ORDER_BOOK

    agent._exchange.watch_order_book = lambda *a, **kw: _stream()  # type: ignore[method-assign]

    with pytest.raises(_MarketDataCircuitBreakerTripped):
        await agent._watch_order_book_loop()

    assert agent._l2_circuit_breaker.consecutive_failures == 3  # stopped exactly at the trip, not all 10

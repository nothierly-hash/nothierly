"""Real, executable tests for "Protocole AEGIS" (2026-09-16): AlphaMarketMakerAgent's new
market-data circuit breaker.

Real gap closed: a real 51.5h shadow-run window produced 34,637 alpha_engine_market_metrics_
unparseable + 36,211 alpha_engine_stale_data_detected warnings -- correctly logged, correctly
fail-closed (this engine never traded on bad data), but nothing ever escalated the *pattern* into a
decision to stop retrying at the normal tick cadence forever. These tests prove
_read_fresh_market_metrics now feeds every unparseable/stale outcome into
_market_data_circuit_breaker, and that a trip forces AgentControlState.OFF via the exact same
_force_off idiom as the existing order-rate-limit/fat-finger breakers.

Own dedicated fixture (not tests/test_alpha_engine.py's shared one): that file's own `_make_agent`
immediately replaces the real ZMQ publisher with a recording fake, with no capture-before-replace
teardown -- constructing directly here instead, so the real publisher/subscriber/context this
project's BaseAgent.__init__ always opens can be captured and closed properly (see
tests/test_execution_agent_instrument_lock.py's identical fixture docstring for why this matters).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import RedisSettings
from core.notifier import TelegramNotifier
from core.redis_client import RedisClient
from core.schemas import AGENT_CONTROL_STATE_REDIS_KEY_PREFIX, MARKET_METRICS_REDIS_KEY_PREFIX
from src.agents.alpha_engine.agent import AlphaMarketMakerAgent


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[AlphaMarketMakerAgent]:
    """Real cleanup, not optional -- see tests/test_execution_agent_instrument_lock.py's identical
    fixture docstring: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets synchronously."""
    instance = AlphaMarketMakerAgent(
        "BTC/USDT", exchange_id="fake", manage_own_signals=False, manage_own_redis=False,
        # Tight thresholds -- these tests trip the breaker in a handful of calls, not 50.
        max_consecutive_market_data_errors=3, max_market_data_error_rate=0.99,
        market_data_error_window_seconds=3600.0,
    )
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    # Unconfigured (no bot_token/chat_id) -- .send() logs and returns False, never touches the
    # network -- same discipline as tests/test_alpha_engine.py's own fixture. _run_cycle's stale/
    # unparseable branches assert self._notifier is not None on their first real occurrence.
    instance._notifier = TelegramNotifier(instance._settings.telegram)
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def _redis(agent: AlphaMarketMakerAgent) -> RedisClient:
    assert agent._redis_client is not None
    return agent._redis_client


async def _seed_unparseable_market_metrics(agent: AlphaMarketMakerAgent) -> None:
    await _redis(agent).execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT", "not-real-json{{{")


async def _seed_stale_market_metrics(agent: AlphaMarketMakerAgent) -> None:
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=agent._max_market_data_age_seconds + 10.0)
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps(
            {"symbol": "BTC/USDT", "timestamp": timestamp.isoformat(), "spread": 1.0, "order_book_imbalance": 0.0, "mid_price": 65000.0}
        ),
    )


async def _seed_fresh_market_metrics(agent: AlphaMarketMakerAgent) -> None:
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps(
            {
                "symbol": "BTC/USDT", "timestamp": datetime.now(timezone.utc).isoformat(),
                "spread": 1.0, "order_book_imbalance": 0.0, "mid_price": 65000.0,
            }
        ),
    )


@pytest.mark.asyncio
async def test_unparseable_reads_are_counted_toward_the_circuit_breaker(agent: AlphaMarketMakerAgent) -> None:
    await _seed_unparseable_market_metrics(agent)
    await agent._read_fresh_market_metrics()
    assert agent._market_data_circuit_breaker.consecutive_failures == 1
    assert agent._market_data_circuit_breaker.tripped is False


@pytest.mark.asyncio
async def test_a_fresh_valid_read_resets_the_consecutive_streak(agent: AlphaMarketMakerAgent) -> None:
    await _seed_unparseable_market_metrics(agent)
    await agent._read_fresh_market_metrics()
    assert agent._market_data_circuit_breaker.consecutive_failures == 1

    await _seed_fresh_market_metrics(agent)
    await agent._read_fresh_market_metrics()
    assert agent._market_data_circuit_breaker.consecutive_failures == 0


@pytest.mark.asyncio
async def test_repeated_unparseable_reads_trip_the_breaker(agent: AlphaMarketMakerAgent) -> None:
    await _seed_unparseable_market_metrics(agent)
    for _ in range(3):  # max_consecutive_market_data_errors=3, see the fixture
        await agent._read_fresh_market_metrics()
    assert agent._market_data_circuit_breaker.tripped is True


@pytest.mark.asyncio
async def test_repeated_stale_reads_also_trip_the_breaker(agent: AlphaMarketMakerAgent) -> None:
    """Both real conditions from the brief (unparseable AND stale) feed the SAME breaker -- a mixed
    run of either failure mode escalates together, not as two independently-tracked counters."""
    await _seed_stale_market_metrics(agent)
    for _ in range(3):
        await agent._read_fresh_market_metrics()
    assert agent._market_data_circuit_breaker.tripped is True


@pytest.mark.asyncio
async def test_a_circuit_breaker_trip_forces_the_engine_off(agent: AlphaMarketMakerAgent) -> None:
    """The real end-to-end wiring: _run_cycle notices a trip and calls the same _force_off idiom
    every other circuit breaker in this engine uses (order-rate-limit, fat-finger). Starts LIVE --
    AgentControlState's own documented default, unasserted here to avoid a real mypy literal-
    narrowing quirk across the awaited _run_cycle() calls below (the same property read twice with
    two different literal comparisons in one function)."""
    await _seed_unparseable_market_metrics(agent)

    for _ in range(3):
        await agent._run_cycle()

    assert agent.control_state == AgentControlState.OFF
    persisted = await _redis(agent).execute("get", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent.agent_id}")
    # Real, separately-documented fakeredis-vs-real-Redis decode_responses quirk -- see
    # tests/test_alpha_engine.py's own identical handling.
    if isinstance(persisted, bytes):
        persisted = persisted.decode()
    assert persisted == AgentControlState.OFF.value


@pytest.mark.asyncio
async def test_the_engine_stays_on_below_the_trip_threshold(agent: AlphaMarketMakerAgent) -> None:
    await _seed_unparseable_market_metrics(agent)
    for _ in range(2):  # one below max_consecutive_market_data_errors=3
        await agent._run_cycle()
    assert agent.control_state == AgentControlState.LIVE

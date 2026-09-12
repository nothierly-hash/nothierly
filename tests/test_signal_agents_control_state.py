"""Real, executable tests for AgentControlState (OFF/SHADOW) suppression in the two signal agents.

Both :class:`~src.agents.rl_signal.agent.RlSignalAgent` and
:class:`~src.agents.supervisor.agent.XgboostSignalAgent` never place an
order themselves -- their whole real effect on the system is the
:class:`~core.schemas.TradingSignal` they publish onto the real bus.
These tests verify the one thing that matters for
``AgentControlState``: a signal computed under ``OFF``/``SHADOW`` must
never reach that real bus, regardless of what the underlying PPO/
XGBoost inference (both real, untrained-skeleton models here -- no
network/model-file dependency) actually decided.

Real Redis via fakeredis (fresh ``FakeServer()`` per test). Neither
agent's ``_run_signal_cycle`` touches the global Kill Switch (only
``publish()``, a plain ZMQ call) -- unlike ``ExecutionAgent``'s tests,
no ``RedisClient._instance`` singleton juggling is needed here.
"""

from __future__ import annotations

from typing import Any, Generator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.message_integrity import publish_signed_market_metrics
from core.redis_client import RedisClient
from core.schemas import MarketMetrics
from src.agents.rl_signal.agent import RlSignalAgent
from src.agents.supervisor.agent import XgboostSignalAgent

_INSTRUMENT = "EUR_USD"


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


async def _seed_market_metrics(redis_client: RedisClient) -> None:
    # Signed write (core/message_integrity.py): XgboostSignalAgent._read_market_metrics now verifies a
    # signature before accepting a reading -- a plain unsigned SET (the old body of this helper) would be
    # rejected exactly like tests/red_team_hacker.py's own poisoning attack is.
    metrics = MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.1, mid_price=1.10000)
    await publish_signed_market_metrics(redis_client, _INSTRUMENT, metrics, ttl_seconds=10)


@pytest.fixture
def rl_agent(redis_client: RedisClient) -> RlSignalAgent:
    instance = RlSignalAgent(_INSTRUMENT, manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    return instance


@pytest.fixture
def xgb_agent(redis_client: RedisClient) -> XgboostSignalAgent:
    instance = XgboostSignalAgent(_INSTRUMENT, manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    return instance


@pytest.mark.asyncio
async def test_rl_signal_off_never_publishes(rl_agent: RlSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    rl_agent._control_state = AgentControlState.OFF

    await rl_agent._run_signal_cycle()

    assert rl_agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_rl_signal_shadow_never_publishes(rl_agent: RlSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    rl_agent._control_state = AgentControlState.SHADOW

    await rl_agent._run_signal_cycle()

    assert rl_agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_rl_signal_live_publishes_a_real_signal(rl_agent: RlSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    assert rl_agent.control_state == AgentControlState.LIVE  # the real default

    await rl_agent._run_signal_cycle()

    published = rl_agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].topic == "TRADING_SIGNALS"
    assert published[0].data["instrument"] == _INSTRUMENT


@pytest.mark.asyncio
async def test_xgboost_signal_off_never_publishes(xgb_agent: XgboostSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    xgb_agent._control_state = AgentControlState.OFF

    await xgb_agent._run_signal_cycle()

    assert xgb_agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_xgboost_signal_shadow_never_publishes(xgb_agent: XgboostSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    xgb_agent._control_state = AgentControlState.SHADOW

    await xgb_agent._run_signal_cycle()

    assert xgb_agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_xgboost_signal_live_publishes_a_real_signal(xgb_agent: XgboostSignalAgent, redis_client: RedisClient) -> None:
    await _seed_market_metrics(redis_client)
    assert xgb_agent.control_state == AgentControlState.LIVE

    await xgb_agent._run_signal_cycle()

    published = xgb_agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].topic == "TRADING_SIGNALS"
    assert published[0].data["instrument"] == _INSTRUMENT

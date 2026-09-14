"""Real, executable ground-truth tests for AlertAgent._check_liquidity_collapse_danger.

See src/agents/alert_agent.py's module docstring, points 7-9, for the
real corrections this check makes to the "Surfusion"/liquidation-cascade
brief: no live Open Interest/leverage feed exists in this system, so
this only detects the real, live half of that idea -- a book's bid-side
depth collapsing relative to its own recent normal, sourced from the
real L2 surface history src.agents.market_data.l2_surface_packer
already persists.

Same fakeredis/_RecordingZmqPublisher/_RecordingTwilioClient discipline
as tests/test_alert_agent.py -- real in-memory Redis, no real network
call ever placed.
"""

from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from typing import Any, Generator

import fakeredis.aioredis
import msgpack
import numpy as np
import pytest

from core.config import TwilioSettings
from core.redis_client import RedisClient
from core.schemas import L2_SURFACE_REDIS_KEY_PREFIX, ORDER_BOOK_TOP_N, OrderBookLevel, OrderBookSnapshot
from src.agents import alert_agent as alert_agent_module
from src.agents.alert_agent import AlertAgent, LIQUIDITY_COLLAPSE_THRESHOLD_OVERRIDE_REDIS_KEY
from src.agents.market_data.l2_surface_packer import pack_snapshot


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


class _RecordingTwilioClient:
    calls: list[dict[str, str]] = []

    def __init__(self, account_sid: str, auth_token: str) -> None:
        self.messages = self

    def create(self, *, body: str, from_: str, to: str) -> None:
        _RecordingTwilioClient.calls.append({"body": body, "from_": from_, "to": to})


@pytest.fixture(autouse=True)
def reset_twilio_recording() -> None:
    _RecordingTwilioClient.calls = []


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """Real RedisClient over real in-memory fakeredis, also installed as the process singleton -- same reasoning as test_alert_agent.py's identical fixture."""
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> AlertAgent:
    instance = AlertAgent(manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._settings.twilio = TwilioSettings(
        account_sid="ACtest", auth_token=__import__("pydantic").SecretStr("t"),
        phone_number="+15550000000", admin_phone_number="+15551111111",
    )
    return instance


def _redis(agent: AlertAgent) -> RedisClient:
    assert agent._redis_client is not None
    return agent._redis_client


def _snapshot(symbol: str, *, bid_prices: list[float], bid_volumes: list[float], mid: float = 65000.0) -> OrderBookSnapshot:
    """One real, valid OrderBookSnapshot -- ask side is a fixed, healthy placebo (only bid depth is under test)."""
    bids = [OrderBookLevel(price=p, volume=v) for p, v in zip(bid_prices, bid_volumes)]
    asks = [OrderBookLevel(price=mid + 1.0 + i, volume=10.0) for i in range(len(bid_prices))]
    return OrderBookSnapshot(symbol=symbol, timestamp=datetime.now(timezone.utc), bids=bids, asks=asks)


async def _push_frame(agent: AlertAgent, symbol: str, *, bid_prices: list[float], bid_volumes: list[float]) -> None:
    encoded = pack_snapshot(_snapshot(symbol, bid_prices=bid_prices, bid_volumes=bid_volumes))
    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}{symbol}"
    await _redis(agent).execute("lpush", key, encoded)


_HEALTHY_BIDS = ([64990.0, 64980.0, 64970.0, 64960.0, 64950.0], [10.0, 10.0, 10.0, 10.0, 10.0])


@pytest.mark.asyncio
async def test_ignores_symbol_with_insufficient_history(agent: AlertAgent) -> None:
    """Fewer than the required reference frames -- must never fire (not enough real data to know what's 'normal' yet)."""
    prices, volumes = _HEALTHY_BIDS
    for _ in range(3):  # below _LIQUIDITY_COLLAPSE_MIN_REFERENCE_FRAMES
        await _push_frame(agent, "BTC/USDT", bid_prices=prices, bid_volumes=volumes)

    await agent._check_liquidity_collapse_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ignores_stable_healthy_depth(agent: AlertAgent) -> None:
    """A book whose depth fluctuates normally (never collapses) must never fire."""
    prices, volumes = _HEALTHY_BIDS
    for _ in range(8):
        await _push_frame(agent, "BTC/USDT", bid_prices=prices, bid_volumes=volumes)

    await agent._check_liquidity_collapse_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fires_on_a_real_sharp_depth_collapse(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact real shape the brief describes: same price, same-looking book, but the real depth has quietly emptied out."""
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    healthy_prices, healthy_volumes = _HEALTHY_BIDS
    for _ in range(6):  # real, healthy reference history
        await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=healthy_volumes)

    # The newest frame: same price levels, but volume collapsed to 5% of normal -- a real >=70% drop.
    collapsed_volumes = [v * 0.05 for v in healthy_volumes]
    await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=collapsed_volumes)

    await agent._check_liquidity_collapse_danger()

    published = agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].topic == "EMERGENCY_HALT"
    assert published[0].data["category"] == "MICROSTRUCTURE"
    assert len(_RecordingTwilioClient.calls) == 1


@pytest.mark.asyncio
async def test_does_not_fire_on_a_moderate_depth_dip_below_threshold(agent: AlertAgent) -> None:
    """A real but moderate dip (well under the 70% default threshold) must not trigger a false alarm."""
    healthy_prices, healthy_volumes = _HEALTHY_BIDS
    for _ in range(6):
        await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=healthy_volumes)

    mild_dip_volumes = [v * 0.70 for v in healthy_volumes]  # 30% drop, under the 70% threshold
    await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=mild_dip_volumes)

    await agent._check_liquidity_collapse_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_live_redis_override_lowers_the_threshold(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same live-tunable pattern as OBI_THRESHOLD_OVERRIDE_REDIS_KEY: a dashboard-set override actually changes what fires."""
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    healthy_prices, healthy_volumes = _HEALTHY_BIDS
    for _ in range(6):
        await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=healthy_volumes)

    mild_dip_volumes = [v * 0.70 for v in healthy_volumes]  # 30% drop
    await _push_frame(agent, "BTC/USDT", bid_prices=healthy_prices, bid_volumes=mild_dip_volumes)

    await _redis(agent).execute("set", LIQUIDITY_COLLAPSE_THRESHOLD_OVERRIDE_REDIS_KEY, "0.25")
    await agent._check_liquidity_collapse_danger()

    assert len(agent._publisher.published) == 1  # type: ignore[attr-defined]


def _push_raw_zero_depth_frame(symbol: str) -> str:
    """A frame whose bid arrays are all-zero -- unreachable through a real, pydantic-validated OrderBookSnapshot
    (OrderBookLevel requires volume > 0), but a real defensive case for a corrupted/malformed encoded payload.
    Built via the exact same wire format pack_snapshot uses, bypassing OrderBookSnapshot's own validation on purpose."""
    zeros = np.zeros(ORDER_BOOK_TOP_N, dtype=np.float32)
    payload = {
        "symbol": symbol, "t": time.time(), "depth": ORDER_BOOK_TOP_N,
        "b_p": zeros.tobytes(), "b_v": zeros.tobytes(), "a_p": zeros.tobytes(), "a_v": zeros.tobytes(),
    }
    return base64.b64encode(msgpack.packb(payload, use_bin_type=True)).decode("ascii")


@pytest.mark.asyncio
async def test_ignores_a_book_with_never_any_real_depth(agent: AlertAgent) -> None:
    """Reference median of exactly 0 must never be treated as '100% collapsed' -- there was nothing real to collapse from."""
    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}BTC/USDT"
    for _ in range(7):
        await _redis(agent).execute("lpush", key, _push_raw_zero_depth_frame("BTC/USDT"))

    await agent._check_liquidity_collapse_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]

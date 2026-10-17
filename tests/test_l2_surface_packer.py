"""Real, executable tests for src/agents/market_data/l2_surface_packer.py.

Redis is faked with ``fakeredis`` (same discipline as ``tests/
test_alert_agent.py``/``tests/test_regime_detector.py`` -- a real
in-memory Redis implementation, fresh ``FakeServer()`` per test),
configured with ``decode_responses=True`` on purpose: this is the exact
real configuration that broke a naive raw-MessagePack-bytes design
(verified live against the real dev Redis before this module was
written -- see its own docstring) and that these tests must keep
proving works, not just assume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator

import fakeredis
import fakeredis.aioredis
import pytest
import redis

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import L2_SURFACE_REDIS_KEY_PREFIX, OrderBookLevel, OrderBookSnapshot
from src.agents.market_data.l2_surface_packer import (
    L2_SURFACE_HISTORY_MAX_LEN,
    pack_snapshot,
    publish_l2_surface,
    read_l2_surface_history,
    unpack_snapshot,
)


def _snapshot(*, symbol: str = "BTC/USDT", bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
        asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
    )


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """A real RedisClient over a real in-memory fakeredis server -- see test_alert_agent.py's identical fixture."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def sync_redis() -> redis.Redis[str]:
    return fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)


# =============================================================================
# pack_snapshot / unpack_snapshot -- round-trip fidelity
# =============================================================================


def test_pack_unpack_round_trip_is_exact_for_a_full_book() -> None:
    snapshot = _snapshot(
        bids=[(100.0, 1.0), (99.5, 2.0), (99.0, 3.0), (98.5, 4.0), (98.0, 5.0)],
        asks=[(100.5, 1.5), (101.0, 2.5), (101.5, 3.5), (102.0, 4.5), (102.5, 5.5)],
    )
    encoded = pack_snapshot(snapshot)
    unpacked = unpack_snapshot(encoded)

    assert unpacked.symbol == "BTC/USDT"
    assert unpacked.depth == 5
    assert list(unpacked.bid_prices) == pytest.approx([100.0, 99.5, 99.0, 98.5, 98.0])
    assert list(unpacked.bid_volumes) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert list(unpacked.ask_prices) == pytest.approx([100.5, 101.0, 101.5, 102.0, 102.5])
    assert list(unpacked.ask_volumes) == pytest.approx([1.5, 2.5, 3.5, 4.5, 5.5])


def test_pack_unpack_pads_a_thinner_than_depth_side_with_zero_volume_rows() -> None:
    """A thin book (fewer real levels than `depth`) must not silently shift the remaining levels' meaning -- see module docstring."""
    snapshot = _snapshot(bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.0)])
    unpacked = unpack_snapshot(pack_snapshot(snapshot, depth=5))

    assert list(unpacked.bid_prices) == pytest.approx([100.0, 99.0, 0.0, 0.0, 0.0])
    assert list(unpacked.bid_volumes) == pytest.approx([1.0, 2.0, 0.0, 0.0, 0.0])
    assert list(unpacked.ask_prices) == pytest.approx([101.0, 0.0, 0.0, 0.0, 0.0])
    assert list(unpacked.ask_volumes) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


def test_pack_snapshot_output_is_a_real_ascii_safe_string() -> None:
    """The exact property that fixes the real UnicodeDecodeError found against decode_responses=True Redis -- see module docstring."""
    snapshot = _snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    encoded = pack_snapshot(snapshot)
    assert isinstance(encoded, str)
    encoded.encode("ascii")  # must not raise


def test_unpack_snapshot_rejects_malformed_input() -> None:
    with pytest.raises(Exception):  # noqa: B017 -- real base64/msgpack decode errors, not a single stable type
        unpack_snapshot("not valid base64 msgpack !!!")


# =============================================================================
# publish_l2_surface / read_l2_surface_history -- real Redis round trip
# =============================================================================


@pytest.mark.asyncio
async def test_publish_then_read_round_trips_through_real_decode_responses_redis() -> None:
    """The real, load-bearing integration claim: base64-wrapped msgpack survives BOTH this project's real Redis client configurations.

    Both the async publisher and the sync reader are pointed at the SAME
    in-memory ``FakeServer`` (not two independent fixtures) so this test
    actually exercises one shared key, not two islands.
    """
    shared_server = fakeredis.FakeServer()
    async_client = RedisClient(RedisSettings())
    async_client._redis = fakeredis.aioredis.FakeRedis(server=shared_server, decode_responses=True)
    sync_client: redis.Redis[str] = fakeredis.FakeRedis(server=shared_server, decode_responses=True)

    snapshot = _snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    await publish_l2_surface(async_client, snapshot)
    history = read_l2_surface_history(sync_client, "BTC/USDT")

    assert len(history) == 1
    assert history[0].symbol == "BTC/USDT"
    assert list(history[0].bid_prices)[:1] == pytest.approx([100.0])


@pytest.mark.asyncio
async def test_publish_writes_under_the_real_shared_key_prefix(redis_client: RedisClient) -> None:
    snapshot = _snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    await publish_l2_surface(redis_client, snapshot)

    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}BTC/USDT"
    assert await redis_client.execute("llen", key) == 1


@pytest.mark.asyncio
async def test_publish_caps_history_length_via_ltrim(redis_client: RedisClient) -> None:
    for i in range(L2_SURFACE_HISTORY_MAX_LEN + 10):
        await publish_l2_surface(redis_client, _snapshot(bids=[(100.0 + i, 1.0)], asks=[(101.0 + i, 1.0)]), max_history=L2_SURFACE_HISTORY_MAX_LEN)

    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}BTC/USDT"
    assert await redis_client.execute("llen", key) == L2_SURFACE_HISTORY_MAX_LEN


def test_read_l2_surface_history_orders_oldest_first(sync_redis: redis.Redis[str]) -> None:
    """LPUSH stores newest-first; the reader must reverse it back to a real chronological time axis."""
    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}BTC/USDT"
    older = pack_snapshot(_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]))
    newer = pack_snapshot(_snapshot(bids=[(200.0, 1.0)], asks=[(201.0, 1.0)]))
    sync_redis.lpush(key, older)
    sync_redis.lpush(key, newer)

    history = read_l2_surface_history(sync_redis, "BTC/USDT")
    assert [round(float(h.bid_prices[0])) for h in history] == [100, 200]


def test_read_l2_surface_history_skips_malformed_entries_without_raising(sync_redis: redis.Redis[str]) -> None:
    key = f"{L2_SURFACE_REDIS_KEY_PREFIX}BTC/USDT"
    good = pack_snapshot(_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]))
    sync_redis.lpush(key, "not real msgpack")
    sync_redis.lpush(key, good)

    history = read_l2_surface_history(sync_redis, "BTC/USDT")
    assert len(history) == 1
    assert history[0].bid_prices[0] == pytest.approx(100.0)


def test_read_l2_surface_history_is_empty_for_an_unknown_symbol(sync_redis: redis.Redis[str]) -> None:
    assert read_l2_surface_history(sync_redis, "NO/SUCH") == []

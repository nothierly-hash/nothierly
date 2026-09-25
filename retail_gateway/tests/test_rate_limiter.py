from __future__ import annotations

import time
from unittest.mock import AsyncMock

import fakeredis
import pytest
from fastapi import HTTPException

from retail_gateway.security import enforce_rate_limit, get_clock
from retail_gateway.tests.fake_clock import FakeClock


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    # A fresh FakeServer per test: fakeredis.FakeAsyncRedis() with no explicit `server=` reuses
    # one process-wide in-memory server by default, which silently leaked rate-limiter counters
    # across these tests (same bucket_id + same real wall-clock window -> a later test started
    # already partway through the previous test's count) -- a real bug caught by running this
    # suite, not a hypothetical one.
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)


@pytest.mark.asyncio
async def test_blocks_requests_over_the_limit(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)
    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_buckets_are_isolated_per_identity(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)
    # A different bucket_id (a different client IP) has its own, untouched quota.
    await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="5.6.7.8", limit=3, window_seconds=60, clock=clock)


@pytest.mark.asyncio
async def test_fails_closed_when_redis_is_unreachable(clock: FakeClock) -> None:
    """A rate limiter that lets traffic through when it can't check the count is not a rate
    limiter -- see security.py's own module docstring on this fail-closed contract.
    """
    broken_redis = AsyncMock()
    broken_redis.incr.side_effect = ConnectionError("redis unreachable")
    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(broken_redis, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_the_window_is_the_whole_interval_and_rolls_over(clock: FakeClock) -> None:
    """Still blocked half a window later, allowed again once the clock crosses into the next window."""
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)

    clock.advance(30)
    with pytest.raises(HTTPException):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)

    clock.advance(30)
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60, clock=clock)


def test_the_production_clock_dependency_is_the_wall_clock() -> None:
    assert get_clock() is time.time

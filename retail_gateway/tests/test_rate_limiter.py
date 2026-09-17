from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis
import pytest
from fastapi import HTTPException

from retail_gateway.security import enforce_rate_limit


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    # A fresh FakeServer per test: fakeredis.FakeAsyncRedis() with no explicit `server=` reuses
    # one process-wide in-memory server by default, which silently leaked rate-limiter counters
    # across these tests (same bucket_id + same real wall-clock window -> a later test started
    # already partway through the previous test's count) -- a real bug caught by running this
    # suite, not a hypothetical one.
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60)


@pytest.mark.asyncio
async def test_blocks_requests_over_the_limit() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60)
    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_buckets_are_isolated_per_identity() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60)
    # A different bucket_id (a different client IP) has its own, untouched quota.
    await enforce_rate_limit(redis_client, key_prefix="test:", bucket_id="5.6.7.8", limit=3, window_seconds=60)


@pytest.mark.asyncio
async def test_fails_closed_when_redis_is_unreachable() -> None:
    """A rate limiter that lets traffic through when it can't check the count is not a rate
    limiter -- see security.py's own module docstring on this fail-closed contract.
    """
    broken_redis = AsyncMock()
    broken_redis.incr.side_effect = ConnectionError("redis unreachable")
    with pytest.raises(HTTPException) as exc_info:
        await enforce_rate_limit(broken_redis, key_prefix="test:", bucket_id="1.2.3.4", limit=3, window_seconds=60)
    assert exc_info.value.status_code == 429

from __future__ import annotations

import time

import fakeredis
import pytest
from fastapi import HTTPException

from retail_gateway.security import enforce_rate_limit_with_backoff

_WINDOW_SECONDS = 60


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


async def _hit(redis_client: fakeredis.FakeAsyncRedis, *, bucket_id: str = "1.2.3.4") -> None:
    await enforce_rate_limit_with_backoff(
        redis_client,
        key_prefix="test:",
        bucket_id=bucket_id,
        limit=3,
        window_seconds=_WINDOW_SECONDS,
        backoff_base_seconds=100,
        backoff_max_seconds=1000,
    )


def _retry_after(exc: HTTPException) -> str:
    assert exc.headers is not None
    return exc.headers["Retry-After"]


async def _clear_lockout_and_window(redis_client: fakeredis.FakeAsyncRedis, *, bucket_id: str = "1.2.3.4") -> None:
    """Simulate real time passing: clear both the lockout and the still-live fixed-window counter.

    Deleting only the lockout key would leave the current fixed-window count still over `limit`
    from the prior phase (a fast unit test never crosses a real 60s window boundary on its own),
    so the very next "should succeed" call would immediately re-trip -- that is real, correct
    behavior of the limiter (a bot flooding within one window keeps compounding), just not what
    these specific tests are isolating.
    """
    current_window = int(time.time()) // _WINDOW_SECONDS
    await redis_client.delete(f"test:lockout:{bucket_id}", f"test:{bucket_id}:{current_window}")


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client)


@pytest.mark.asyncio
async def test_first_violation_locks_out_for_the_base_duration() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client)
    with pytest.raises(HTTPException) as exc_info:
        await _hit(redis_client)
    assert exc_info.value.status_code == 429
    assert _retry_after(exc_info.value) == "100"


@pytest.mark.asyncio
async def test_lockout_blocks_further_calls_even_within_a_fresh_window() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client)
    with pytest.raises(HTTPException):
        await _hit(redis_client)
    # Still locked out -- the plain fixed-window count resetting on its own does not matter while
    # the separate lockout key is still live. This is exactly the real gap a bare fixed window has
    # against a bot that just waits out one window and retries at the same rate (mission Directive 3).
    with pytest.raises(HTTPException) as exc_info:
        await _hit(redis_client)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_repeated_violations_escalate_the_lockout_exponentially() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client)
    with pytest.raises(HTTPException) as first_violation:
        await _hit(redis_client)
    assert _retry_after(first_violation.value) == "100"

    # Simulates time passing, to observe the *next* violation's own escalated duration without a
    # real 100s sleep in the test suite.
    await _clear_lockout_and_window(redis_client)
    for _ in range(3):
        await _hit(redis_client)
    with pytest.raises(HTTPException) as second_violation:
        await _hit(redis_client)
    assert _retry_after(second_violation.value) == "200"  # base * 2**1


@pytest.mark.asyncio
async def test_lockout_never_exceeds_the_configured_maximum() -> None:
    redis_client = _isolated_fake_redis()
    for violation_number in range(6):
        for _ in range(3):
            await _hit(redis_client)
        with pytest.raises(HTTPException) as exc_info:
            await _hit(redis_client)
        await _clear_lockout_and_window(redis_client)
    assert int(_retry_after(exc_info.value)) <= 1000


@pytest.mark.asyncio
async def test_buckets_are_isolated_per_identity() -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, bucket_id="1.2.3.4")
    with pytest.raises(HTTPException):
        await _hit(redis_client, bucket_id="1.2.3.4")
    # A different bucket (e.g. a different IP, or the identifier bucket vs the IP bucket) is untouched.
    await _hit(redis_client, bucket_id="5.6.7.8")

from __future__ import annotations

import fakeredis
import pytest
from fastapi import HTTPException

from retail_gateway.security import enforce_rate_limit_with_backoff
from retail_gateway.tests.fake_clock import FakeClock

_WINDOW_SECONDS = 60


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


async def _hit(redis_client: fakeredis.FakeAsyncRedis, clock: FakeClock, *, bucket_id: str = "1.2.3.4") -> None:
    await enforce_rate_limit_with_backoff(
        redis_client,
        key_prefix="test:",
        bucket_id=bucket_id,
        limit=3,
        window_seconds=_WINDOW_SECONDS,
        backoff_base_seconds=100,
        backoff_max_seconds=1000,
        clock=clock,
    )


def _retry_after(exc: HTTPException) -> str:
    assert exc.headers is not None
    return exc.headers["Retry-After"]


async def _clear_lockout_and_window(redis_client: fakeredis.FakeAsyncRedis, clock: FakeClock, *, bucket_id: str = "1.2.3.4") -> None:
    """Simulate real time passing: clear the lockout and move the clock into the next fixed window.

    The lockout is a Redis TTL (real time, 100 s here), so a fast test cannot wait it out and it is
    deleted directly. The window counter is keyed on the injected clock, so the next window is reached
    by advancing it -- otherwise the current window's count would still be over `limit` from the prior
    phase and the very next "should succeed" call would immediately re-trip, which is real, correct
    behavior of the limiter (a bot flooding within one window keeps compounding), just not what these
    specific tests are isolating.
    """
    await redis_client.delete(f"test:lockout:{bucket_id}")
    clock.advance(_WINDOW_SECONDS)


@pytest.mark.asyncio
async def test_allows_requests_under_the_limit(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, clock)


@pytest.mark.asyncio
async def test_first_violation_locks_out_for_the_base_duration(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, clock)
    with pytest.raises(HTTPException) as exc_info:
        await _hit(redis_client, clock)
    assert exc_info.value.status_code == 429
    assert _retry_after(exc_info.value) == "100"


@pytest.mark.asyncio
async def test_lockout_blocks_further_calls_even_within_a_fresh_window(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, clock)
    with pytest.raises(HTTPException):
        await _hit(redis_client, clock)
    # Still locked out after the clock crosses into a fresh window -- the plain fixed-window count
    # resetting on its own does not matter while the separate lockout key is still live. This is
    # exactly the real gap a bare fixed window has against a bot that just waits out one window and
    # retries at the same rate (mission Directive 3).
    clock.advance(_WINDOW_SECONDS)
    with pytest.raises(HTTPException) as exc_info:
        await _hit(redis_client, clock)
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_repeated_violations_escalate_the_lockout_exponentially(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, clock)
    with pytest.raises(HTTPException) as first_violation:
        await _hit(redis_client, clock)
    assert _retry_after(first_violation.value) == "100"

    # Simulates time passing, to observe the *next* violation's own escalated duration without a
    # real 100s sleep in the test suite.
    await _clear_lockout_and_window(redis_client, clock)
    for _ in range(3):
        await _hit(redis_client, clock)
    with pytest.raises(HTTPException) as second_violation:
        await _hit(redis_client, clock)
    assert _retry_after(second_violation.value) == "200"  # base * 2**1


@pytest.mark.asyncio
async def test_lockout_never_exceeds_the_configured_maximum(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for violation_number in range(6):
        for _ in range(3):
            await _hit(redis_client, clock)
        with pytest.raises(HTTPException) as exc_info:
            await _hit(redis_client, clock)
        await _clear_lockout_and_window(redis_client, clock)
    assert int(_retry_after(exc_info.value)) <= 1000


@pytest.mark.asyncio
async def test_buckets_are_isolated_per_identity(clock: FakeClock) -> None:
    redis_client = _isolated_fake_redis()
    for _ in range(3):
        await _hit(redis_client, clock, bucket_id="1.2.3.4")
    with pytest.raises(HTTPException):
        await _hit(redis_client, clock, bucket_id="1.2.3.4")
    # A different bucket (e.g. a different IP, or the identifier bucket vs the IP bucket) is untouched.
    await _hit(redis_client, clock, bucket_id="5.6.7.8")

from __future__ import annotations

import fakeredis
import pytest

from retail_gateway.otp_store import generate_otp_code, store_otp, verify_and_consume_otp


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


def test_generate_otp_code_is_the_right_length_and_zero_padded() -> None:
    codes = {generate_otp_code(length=6) for _ in range(200)}
    assert all(len(code) == 6 and code.isdigit() for code in codes)
    assert len(codes) > 1  # real randomness, not a constant


@pytest.mark.asyncio
async def test_verify_accepts_the_correct_code_once() -> None:
    redis_client = _isolated_fake_redis()
    await store_otp(redis_client, identifier="user@example.com", code="123456", ttl_seconds=300, max_attempts=5)
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="123456") is True
    # Single-use: the same correct code no longer verifies once consumed.
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="123456") is False


@pytest.mark.asyncio
async def test_verify_rejects_wrong_code() -> None:
    redis_client = _isolated_fake_redis()
    await store_otp(redis_client, identifier="user@example.com", code="123456", ttl_seconds=300, max_attempts=5)
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="000000") is False


@pytest.mark.asyncio
async def test_verify_returns_false_for_an_identifier_with_no_pending_code() -> None:
    redis_client = _isolated_fake_redis()
    assert await verify_and_consume_otp(redis_client, identifier="nobody@example.com", code="123456") is False


@pytest.mark.asyncio
async def test_exhausting_max_attempts_invalidates_the_code_even_though_it_was_never_guessed_correctly() -> None:
    redis_client = _isolated_fake_redis()
    await store_otp(redis_client, identifier="user@example.com", code="123456", ttl_seconds=300, max_attempts=3)
    for _ in range(3):
        assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="000000") is False
    # The code is gone even though it was correct all along -- attempts are exhausted.
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="123456") is False


@pytest.mark.asyncio
async def test_requesting_a_new_code_invalidates_the_previous_one() -> None:
    redis_client = _isolated_fake_redis()
    await store_otp(redis_client, identifier="user@example.com", code="111111", ttl_seconds=300, max_attempts=5)
    await store_otp(redis_client, identifier="user@example.com", code="222222", ttl_seconds=300, max_attempts=5)
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="111111") is False
    assert await verify_and_consume_otp(redis_client, identifier="user@example.com", code="222222") is True

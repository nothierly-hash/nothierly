from __future__ import annotations

from datetime import datetime, timezone

import fakeredis
import pytest
from pydantic import SecretStr

from retail_gateway.config import JwtSettings
from retail_gateway.security import InvalidTokenError, create_access_token
from retail_gateway.refresh_tokens import (
    create_refresh_token,
    decode_refresh_token,
    is_refresh_token_active,
    register_refresh_token,
    revoke_refresh_token,
)


def _isolated_fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


def _settings() -> JwtSettings:
    return JwtSettings(secret_key=SecretStr("test-secret-key-at-least-32-bytes-long"), refresh_token_expire_days=7)


def test_refresh_token_round_trip() -> None:
    settings = _settings()
    token, expires_at, jti = create_refresh_token(subject="user-123", settings=settings)
    subject, decoded_jti = decode_refresh_token(token, settings=settings)
    assert subject == "user-123"
    assert decoded_jti == jti
    assert expires_at > datetime.now(timezone.utc)


def test_an_access_token_is_never_accepted_as_a_refresh_token() -> None:
    """The real reason for the `typ` claim -- see refresh_tokens.py's own module docstring."""
    settings = _settings()
    access_token, _ = create_access_token(subject="user-123", settings=settings)
    with pytest.raises(InvalidTokenError):
        decode_refresh_token(access_token, settings=settings)


def test_refresh_token_rejects_bad_signature() -> None:
    settings = _settings()
    token, _, _ = create_refresh_token(subject="user-123", settings=settings)
    other_settings = JwtSettings(secret_key=SecretStr("a-different-secret-also-at-least-32-bytes"))
    with pytest.raises(InvalidTokenError):
        decode_refresh_token(token, settings=other_settings)


@pytest.mark.asyncio
async def test_registered_token_is_active_until_revoked() -> None:
    redis_client = _isolated_fake_redis()
    await register_refresh_token(redis_client, user_id="user-123", jti="jti-1", ttl_seconds=3600)
    assert await is_refresh_token_active(redis_client, user_id="user-123", jti="jti-1") is True

    await revoke_refresh_token(redis_client, user_id="user-123", jti="jti-1")
    assert await is_refresh_token_active(redis_client, user_id="user-123", jti="jti-1") is False


@pytest.mark.asyncio
async def test_an_unregistered_jti_is_never_active() -> None:
    redis_client = _isolated_fake_redis()
    assert await is_refresh_token_active(redis_client, user_id="user-123", jti="never-registered") is False

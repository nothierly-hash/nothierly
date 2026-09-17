from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from pydantic import SecretStr

from retail_gateway.config import JwtSettings
from retail_gateway.security import InvalidTokenError, create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_round_trip() -> None:
    hashed = hash_password("Correct-Horse-Battery-9")
    assert hashed != "Correct-Horse-Battery-9"
    assert verify_password(password_hash=hashed, password="Correct-Horse-Battery-9")


def test_password_verify_rejects_wrong_password() -> None:
    hashed = hash_password("Correct-Horse-Battery-9")
    assert not verify_password(password_hash=hashed, password="wrong-password")


def _settings(expire_minutes: int = 1440) -> JwtSettings:
    return JwtSettings(secret_key=SecretStr("test-secret-key-at-least-32-bytes-long"), access_token_expire_minutes=expire_minutes)


def test_jwt_round_trip() -> None:
    settings = _settings()
    token, expires_at = create_access_token(subject="user-123", settings=settings)
    assert decode_access_token(token, settings=settings) == "user-123"
    assert expires_at > datetime.now(timezone.utc)


def test_jwt_rejects_bad_signature() -> None:
    settings = _settings()
    token, _ = create_access_token(subject="user-123", settings=settings)
    other_settings = JwtSettings(secret_key=SecretStr("a-different-secret-also-at-least-32-bytes"))
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, settings=other_settings)


def test_jwt_rejects_expired_token() -> None:
    settings = _settings()
    expired_payload = {
        "sub": "user-123",
        "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    expired_token = jwt.encode(expired_payload, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)
    with pytest.raises(InvalidTokenError):
        decode_access_token(expired_token, settings=settings)


def test_jwt_rejects_missing_subject_claim() -> None:
    settings = _settings()
    malformed_token = jwt.encode(
        {"iat": datetime.now(timezone.utc), "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        settings.secret_key.get_secret_value(),
        algorithm=settings.algorithm,
    )
    with pytest.raises(InvalidTokenError):
        decode_access_token(malformed_token, settings=settings)

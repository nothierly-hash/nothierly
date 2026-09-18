"""Refresh-token issuance, verification, and revocation.

A refresh token is a signed JWT (same HMAC key/algorithm as the access token, a distinct
``"typ": "refresh"`` claim so one can never be replayed as the other) **plus** a Redis
allow-list entry keyed by its ``jti`` -- the JWT signature alone proves authenticity but cannot
be revoked before its own expiry; the allow-list is what makes :func:`revoke_refresh_token`
(logout) and rotation-on-refresh real rather than cosmetic. Never returned in a JSON response
body -- ``main.py`` sets it only as an ``HttpOnly``/``Secure``/``SameSite=Strict`` cookie, per
the brief's own explicit requirement that client-side JavaScript must never be able to read it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Final

import jwt

from retail_gateway.config import JwtSettings
from retail_gateway.security import InvalidTokenError

_TOKEN_TYPE_CLAIM: Final[str] = "refresh"
_REFRESH_REDIS_KEY_PREFIX: Final[str] = "refresh_token:"


def _redis_key(*, user_id: str, jti: str) -> str:
    return f"{_REFRESH_REDIS_KEY_PREFIX}{user_id}:{jti}"


def create_refresh_token(*, subject: str, settings: JwtSettings) -> tuple[str, datetime, str]:
    """Mint a signed refresh token. Returns ``(token, expires_at, jti)`` -- the caller still needs
    ``jti`` to register the matching Redis allow-list entry (see module docstring)."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.refresh_token_expire_days)
    jti = str(uuid.uuid4())
    payload: dict[str, Any] = {"sub": subject, "iat": now, "exp": expires_at, "typ": _TOKEN_TYPE_CLAIM, "jti": jti}
    token = jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)
    return token, expires_at, jti


def decode_refresh_token(token: str, *, settings: JwtSettings) -> tuple[str, str]:
    """Verify signature/expiry/token-type, return ``(subject, jti)``. Raises :class:`InvalidTokenError`."""
    try:
        payload = jwt.decode(token, settings.secret_key.get_secret_value(), algorithms=[settings.algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
    if payload.get("typ") != _TOKEN_TYPE_CLAIM:
        raise InvalidTokenError("Not a refresh token.")
    subject, jti = payload.get("sub"), payload.get("jti")
    if not isinstance(subject, str) or not isinstance(jti, str):
        raise InvalidTokenError("Refresh token payload missing 'sub'/'jti'.")
    return subject, jti


async def register_refresh_token(redis_client: Any, *, user_id: str, jti: str, ttl_seconds: int) -> None:
    """Add ``jti`` to the allow-list -- a refresh token is only honored while its entry exists."""
    await redis_client.set(_redis_key(user_id=user_id, jti=jti), "1", ex=ttl_seconds)


async def is_refresh_token_active(redis_client: Any, *, user_id: str, jti: str) -> bool:
    return bool(await redis_client.exists(_redis_key(user_id=user_id, jti=jti)))


async def revoke_refresh_token(redis_client: Any, *, user_id: str, jti: str) -> None:
    """Remove ``jti`` from the allow-list -- logout, and the old token on every rotation."""
    await redis_client.delete(_redis_key(user_id=user_id, jti=jti))

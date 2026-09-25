"""Password hashing, JWT issuance/verification, and the Redis rate limiter.

**Argon2id, not bcrypt**: the brief offered either; this project already made and verified
this exact call for ``core/iam.py`` (Argon2id -- the PHC-winning, GPU/ASIC-resistant KDF,
``argon2-cffi`` confirmed installing and hashing/verifying cleanly on this machine). Re-litigating
that choice here for no new reason would trade a verified decision for an unverified one --
the same "don't re-decide a settled question" discipline this project's Dockerfile already
applies to its own Python-version pin.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Final

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request, status
from redis.asyncio import Redis

from retail_gateway.config import JwtSettings

_hasher: Final = PasswordHasher()


# -- password hashing ---------------------------------------------------------------------


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(*, password_hash: str, password: str) -> bool:
    try:
        _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    return True


# -- JWT ------------------------------------------------------------------------------------


class InvalidTokenError(Exception):
    """Raised by :func:`decode_access_token` on any invalid/expired token -- never a raw ``jwt.*`` exception."""


def create_access_token(*, subject: str, settings: JwtSettings) -> tuple[str, datetime]:
    """Mint a signed access token for ``subject`` (the user's UUID, as a string).

    Returns ``(token, expires_at)`` -- the caller (``main.py``) surfaces ``expires_at`` in the
    response body so a client knows exactly when to re-authenticate, rather than having to
    decode the token itself to find out.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "iat": now, "exp": expires_at}
    token = jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)
    return token, expires_at


def decode_access_token(token: str, *, settings: JwtSettings) -> str:
    """Verify signature + expiry, return the ``sub`` claim (the user's UUID)."""
    try:
        payload = jwt.decode(token, settings.secret_key.get_secret_value(), algorithms=[settings.algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise InvalidTokenError("Token payload missing a valid 'sub' claim.")
    return subject


# -- rate limiting ----------------------------------------------------------------------------
#
# Same real O(1) fixed-window algorithm as p2p_api.py's own `_enforce_rate_limit` (one INCR,
# occasionally one EXPIRE) -- reimplemented standalone here rather than imported, for the air-gap
# reason documented in retail_gateway/db.py. Fail-closed on Redis downtime: a rate limiter that
# lets traffic through when it can't check the count is not a rate limiter.
#
# The window is ``int(clock()) // window_seconds``, and the clock is always passed in by the caller
# (see ``get_clock``) instead of being read here, so a test can pin it. A test that fires N+1 requests
# "in one window" against the real clock straddles a window boundary now and then and fails at random.

Clock = Callable[[], float]


def get_clock() -> Clock:
    """FastAPI dependency: the wall clock (epoch seconds) the rate limiter buckets requests by."""
    return time.time


async def enforce_rate_limit(
    redis_client: Redis[str],
    *,
    key_prefix: str,
    bucket_id: str,
    limit: int,
    window_seconds: int,
    clock: Clock,
) -> None:
    """Raise HTTP 429 if ``bucket_id`` has exceeded ``limit`` requests in the current window,
    or if Redis itself could not be reached (fail-closed, not fail-open).
    """
    current_window = int(clock()) // window_seconds
    redis_key = f"{key_prefix}{bucket_id}:{current_window}"
    try:
        current_count = await redis_client.incr(redis_key)
        if current_count == 1:
            await redis_client.expire(redis_key, window_seconds * 2)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limiter temporarily unavailable -- request rejected, not silently allowed through.",
        )

    if current_count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window_seconds}s.",
        )


async def enforce_rate_limit_with_backoff(
    redis_client: Redis[str],
    *,
    key_prefix: str,
    bucket_id: str,
    limit: int,
    window_seconds: int,
    backoff_base_seconds: int,
    backoff_max_seconds: int,
    clock: Clock,
) -> None:
    """Same fixed-window check as :func:`enforce_rate_limit`, plus an escalating lockout on repeat offenders.

    The real toll-fraud/SMS-pumping shield (mission Directive 3): a plain fixed window resets
    every ``window_seconds`` no matter how many times it was already exceeded, which lets a bot
    just wait out one window and retry at the same rate forever. Each *violation* here instead
    doubles a separate lockout TTL (``backoff_base_seconds * 2**violations``, capped at
    ``backoff_max_seconds``) -- a bucket that keeps re-offending is locked out for longer and
    longer, until a human clears it or the violation streak itself expires
    (``backoff_max_seconds`` after the last violation, via the same key's own TTL).

    Raises:
        HTTPException: 429, with a ``Retry-After`` header, either because a lockout from a
            previous violation is still active or because this call is the violation that just
            triggered a new one. Fail-closed on Redis downtime, same as :func:`enforce_rate_limit`.
    """
    lockout_key = f"{key_prefix}lockout:{bucket_id}"
    violations_key = f"{key_prefix}violations:{bucket_id}"

    try:
        lockout_ttl = await redis_client.ttl(lockout_key)
    except Exception:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limiter temporarily unavailable -- request rejected, not silently allowed through.")

    if lockout_ttl and lockout_ttl > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Locked out after repeated violations. Retry in {lockout_ttl}s.",
            headers={"Retry-After": str(lockout_ttl)},
        )

    current_window = int(clock()) // window_seconds
    redis_key = f"{key_prefix}{bucket_id}:{current_window}"
    try:
        current_count = await redis_client.incr(redis_key)
        if current_count == 1:
            await redis_client.expire(redis_key, window_seconds * 2)
    except Exception:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limiter temporarily unavailable -- request rejected, not silently allowed through.")

    if current_count > limit:
        violations = await redis_client.incr(violations_key)
        await redis_client.expire(violations_key, backoff_max_seconds)
        lockout_seconds = min(backoff_base_seconds * (2 ** (violations - 1)), backoff_max_seconds)
        await redis_client.set(lockout_key, "1", ex=lockout_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window_seconds}s. Locked out for {lockout_seconds}s.",
            headers={"Retry-After": str(lockout_seconds)},
        )


def client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"

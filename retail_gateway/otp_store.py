"""Ephemeral OTP generation/storage -- Redis only, never Postgres (the brief's own explicit prohibition).

A code is stored as ``(argon2_hash, attempts_remaining)`` under one Redis key with a hard TTL
(``OtpSettings.ttl_seconds``, default 300s) -- the key's own expiry *is* the code's expiry, no
separate "issued_at" bookkeeping needed. Verification is single-use: a correct guess deletes the
key immediately (:func:`verify_and_consume_otp`), so a leaked/replayed code cannot be reused even
inside its own TTL window. A wrong guess decrements ``attempts_remaining`` instead of leaving the
code guessable indefinitely -- a 6-digit code is only 1,000,000 possibilities, too few to allow
unlimited attempts within a 5-minute window (see ``OtpSettings.otp_verify_max_attempts`` on
``RateLimitSettings``, checked here).
"""

from __future__ import annotations

import json
import secrets
from typing import Final

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from redis.asyncio import Redis

_hasher: Final = PasswordHasher()
_OTP_KEY_PREFIX: Final[str] = "otp:"


def generate_otp_code(*, length: int) -> str:
    """A cryptographically secure, zero-padded decimal code -- ``secrets.randbelow``, never ``random``."""
    upper_bound = 10**length
    return str(secrets.randbelow(upper_bound)).zfill(length)


def _redis_key(identifier: str) -> str:
    return f"{_OTP_KEY_PREFIX}{identifier}"


async def store_otp(redis_client: Redis[str], *, identifier: str, code: str, ttl_seconds: int, max_attempts: int) -> None:
    """Hash ``code`` (Argon2id) and store it against ``identifier`` with a hard TTL.

    Overwrites any code already pending for this identifier -- requesting a new OTP invalidates
    the previous one, so a user who lost the first message is never stuck choosing which of two
    live codes is current.
    """
    payload = json.dumps({"hash": _hasher.hash(code), "attempts_remaining": max_attempts})
    await redis_client.set(_redis_key(identifier), payload, ex=ttl_seconds)


async def verify_and_consume_otp(redis_client: Redis[str], *, identifier: str, code: str) -> bool:
    """Return ``True`` and delete the stored code on a correct, still-live guess; ``False`` otherwise.

    A wrong guess is not free: ``attempts_remaining`` is decremented in place (same TTL,
    untouched) and the code is deleted outright once it hits zero, closing the brute-force
    window early rather than waiting out the rest of the TTL.
    """
    key = _redis_key(identifier)
    raw = await redis_client.get(key)
    if raw is None:
        return False

    record = json.loads(raw)
    try:
        _hasher.verify(record["hash"], code)
    except VerifyMismatchError:
        remaining = record["attempts_remaining"] - 1
        if remaining <= 0:
            await redis_client.delete(key)
        else:
            ttl = await redis_client.ttl(key)
            record["attempts_remaining"] = remaining
            await redis_client.set(key, json.dumps(record), ex=ttl if ttl > 0 else 1)
        return False

    await redis_client.delete(key)
    return True

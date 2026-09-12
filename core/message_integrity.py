"""HMAC-SHA256 signing/verification for Redis-resident values an attacker with mere Redis
write access must not be able to forge.

Real gap this closes, found by ``tests/red_team_hacker.py``: ``RedisSettings.password``
(see ``core/config.py``) gates who can *connect* to Redis, never who is allowed to
*author* one specific value once connected. A compromised or merely misconfigured
process sharing the same Redis instance (any agent, any tenant on a shared deployment,
or simply the fact that local/dev Redis has no password by default) can already
authenticate and still forge ``SYSTEM_STATUS`` (the global Kill Switch) or a symbol's
``MarketMetrics`` cache entry. Signing is therefore a second, independent control layered
on top of ``REDIS_PASSWORD``/``REDIS_USE_TLS`` -- not a replacement for them.

Design choice, deliberately **not** an opaque signed envelope replacing the stored value:
every signed value keeps its real, existing plaintext representation at its real, existing
key (e.g. ``SYSTEM_STATUS`` still holds literally ``"SAFE"``/``"PANIC"``, a MarketMetrics
key still holds the real, unmodified ``MarketMetrics.model_dump_json()``), with the HMAC
written to a *separate* companion key (``<key>:sig``). This is what lets every existing
plain reader across this codebase (dashboard pages displaying SYSTEM_STATUS, every
:class:`~core.schemas.MarketMetrics` consumer that has not yet adopted
:func:`read_verified_market_metrics`) keep working completely unchanged -- only the
security-critical callers that opt in (:mod:`core.security`,
:class:`~src.agents.supervisor.agent.XgboostSignalAgent`) gain the stronger guarantee.
An opaque envelope would have forced updating every one of those call sites in the same
patch to avoid breaking them outright; a companion key does not.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Final

from core.config import get_settings
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics

MARKET_METRICS_SIG_SUFFIX: Final[str] = ":sig"


class SignatureVerificationError(ValueError):
    """Raised when a value's companion signature is missing or does not match.

    Callers should treat this exactly like a missing/corrupt reading (fail
    closed), never like a value that merely failed a Pydantic shape check
    -- a signature mismatch means the value was not written by a holder of
    ``VAULT_MASTER_KEY``, i.e. it may be actively hostile, not just malformed.
    """


def _signing_key() -> bytes:
    # Reuses VaultSettings.master_key (core/config.py) -- already a required secret every
    # real deployment of this project provisions (see that class's own docstring), rather
    # than inventing a second one an operator could forget to set. Distinct purpose
    # (message authenticity, not credential-at-rest encryption) but the same trust
    # boundary: whoever holds this key is already trusted with every stored broker
    # credential, so trusting it to author SYSTEM_STATUS/MarketMetrics too adds no new
    # exposure.
    return get_settings().vault.master_key.get_secret_value().encode("utf-8")


def sign(payload: str | bytes) -> str:
    """Return the hex HMAC-SHA256 of ``payload`` under ``VAULT_MASTER_KEY``.

    Accepts ``bytes`` as well as ``str``: ``redis.asyncio.Redis`` with
    ``decode_responses=True`` (what ``core.redis_client.RedisClient`` always
    configures) reliably returns ``str`` in production, but ``fakeredis``
    2.37.1's async client does not always honor that flag (a real,
    reproducible library quirk -- see ``tests/red_team_hacker.py``'s
    ``_DecodingFakeRedis`` for the same issue found independently). Signing
    is a security boundary, not a place to let a test-double quirk raise
    ``AttributeError`` instead of the real pass/fail verdict it should be
    reporting.
    """
    payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hmac.new(_signing_key(), payload_bytes, hashlib.sha256).hexdigest()


def verify(payload: str | bytes, signature: str | bytes | None) -> bool:
    """Return whether ``signature`` is a valid HMAC-SHA256 of ``payload``.

    Uses :func:`hmac.compare_digest` -- constant-time, so a forged
    signature cannot be brute-forced one byte at a time via a response-time
    timing side-channel, which a naive ``==`` comparison would admit.
    ``None``/empty ``signature`` (companion key absent -- e.g. a value
    written before this module existed, or by a caller that never signed
    it) is always rejected, never treated as "no signature required".
    """
    if not signature:
        return False
    signature_str = signature.decode("utf-8") if isinstance(signature, bytes) else signature
    return hmac.compare_digest(sign(payload), signature_str)


async def publish_signed_market_metrics(
    redis_client: RedisClient, symbol: str, metrics: MarketMetrics, *, ttl_seconds: int
) -> None:
    """Write ``metrics`` under its real, existing key, plus a signed companion key.

    The one real writer -- :class:`~src.agents.market_data.agent.MarketDataAgent` --
    should call this instead of a bare ``redis_client.execute("set", ...,
    metrics.model_dump_json())``. Both keys share ``ttl_seconds`` so the signature never
    outlives (or expires before) the reading it authenticates.
    """
    key = f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}"
    raw = metrics.model_dump_json()
    await asyncio.gather(
        redis_client.execute("set", key, raw, ex=ttl_seconds),
        redis_client.execute("set", f"{key}{MARKET_METRICS_SIG_SUFFIX}", sign(raw), ex=ttl_seconds),
    )


async def read_verified_market_metrics(redis_client: RedisClient, symbol: str) -> MarketMetrics | None:
    """Read + verify a symbol's :class:`~core.schemas.MarketMetrics`, or raise on tamper.

    Returns ``None`` only when the primary key is genuinely absent (no
    reading published yet, or its TTL expired) -- the same, unchanged
    contract every existing unsigned reader already relies on. Raises
    :class:`SignatureVerificationError` when a reading *is* present but its
    companion signature is missing or does not match -- the real Redis
    cache-poisoning scenario :mod:`tests.red_team_hacker` demonstrates
    (Attack 1): a value written directly to the real key, without going
    through :func:`publish_signed_market_metrics`, now fails verification
    instead of being silently accepted as genuine.

    **Known, disclosed scope**: only callers that adopt this helper gain
    the guarantee. Every other real reader of this key
    (:class:`~src.agents.rl_signal.agent.RlSignalAgent`,
    :mod:`src.agents.quant_copilot`, :mod:`src.agents.alert_agent`,
    :mod:`src.agents.alpha_engine.agent`, :mod:`src.execution.
    shadow_crossing_engine`, ``api_middleware/gateway.py``, and the
    Command Center / Microstructure Radar dashboard pages) still reads the
    plain key directly and is not yet protected -- migrating each is the
    same two-line change demonstrated here, left as a disclosed follow-up
    rather than rewritten unprompted in this patch.
    """
    key = f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}"
    raw, signature = await asyncio.gather(
        redis_client.execute("get", key),
        redis_client.execute("get", f"{key}{MARKET_METRICS_SIG_SUFFIX}"),
    )
    if raw is None:
        return None
    if not verify(raw, signature):
        raise SignatureVerificationError(
            f"MarketMetrics for {symbol!r} failed signature verification -- possible cache poisoning "
            f"(the value at {key!r} was not written by publish_signed_market_metrics)."
        )
    return MarketMetrics.model_validate_json(raw)

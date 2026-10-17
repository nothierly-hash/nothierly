"""Process-wide async Redis client for the Retail Gateway's rate limiter.

Deliberately simpler than ``core/redis_client.py``'s full backoff-reconnection singleton: this
service's real availability requirement is narrower (a rate-limiter check on two low-traffic
endpoints, not a continuous multi-agent trading bus) -- ``redis.asyncio.Redis``'s own built-in
``retry_on_timeout``/``health_check_interval`` already cover the real failure mode (a transient
blip), and :func:`retail_gateway.security.enforce_rate_limit` fails closed (HTTP 429) on any
exception regardless, so an unhandled reconnect storm cannot silently let unlimited traffic
through either way.
"""

from __future__ import annotations

from redis.asyncio import Redis

from retail_gateway.config import RetailRedisSettings

_client: Redis[str] | None = None


def get_redis(settings: RetailRedisSettings) -> Redis[str]:
    """Return the process-wide client, creating it (and only it -- one real TCP connection pool
    per process) on first call.
    """
    global _client
    if _client is None:
        _client = Redis.from_url(
            settings.url(),
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30,
        )
    return _client


async def reset_redis_for_tests() -> None:
    """Test-only: drop the cached client so a fresh one (e.g. against ``fakeredis``) is built next call."""
    global _client
    if _client is not None:
        # `close()`, not `aclose()`: same real stub gap as core/redis_client.py's own `close()`
        # method documents -- `aclose()` is the non-deprecated spelling at runtime, but redis-py's
        # bundled stubs don't declare it on the generic `Redis[str]` alias. Functionally identical.
        await _client.close()
    _client = None

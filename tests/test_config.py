"""Real, executable tests for core/config.py.

Focused on :meth:`AppSettings._require_redis_auth_in_production` --
added in response to a real red-team finding (``tests/red_team_hacker.py``):
this project's Redis has no password/TLS by default, and nothing
previously stopped a real ``ENVIRONMENT=production`` process (started
outside ``docker-compose.yml``, whose own ``redis-broker`` service
already refuses to start without ``REDIS_PASSWORD``) from silently
running with an unauthenticated Redis connection.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from core.config import AppSettings, Environment, RedisSettings


def test_production_without_redis_password_is_rejected() -> None:
    with pytest.raises(ValidationError, match="REDIS_PASSWORD"):
        AppSettings(environment=Environment.PRODUCTION, redis=RedisSettings(password=None, use_tls=True))


def test_production_without_tls_is_rejected() -> None:
    with pytest.raises(ValidationError, match="REDIS_USE_TLS"):
        AppSettings(
            environment=Environment.PRODUCTION,
            redis=RedisSettings(password=SecretStr("a-real-strong-password"), use_tls=False),
        )


def test_production_with_password_and_tls_is_accepted() -> None:
    settings = AppSettings(
        environment=Environment.PRODUCTION,
        redis=RedisSettings(password=SecretStr("a-real-strong-password"), use_tls=True),
    )
    assert settings.environment == Environment.PRODUCTION
    assert settings.redis.password is not None


def test_development_never_requires_redis_auth() -> None:
    """The exact real local dev shape (see this project's own .env: no REDIS_PASSWORD) must stay untouched."""
    settings = AppSettings(environment=Environment.DEVELOPMENT, redis=RedisSettings(password=None, use_tls=False))
    assert settings.redis.password is None
    assert settings.redis.use_tls is False


def test_staging_never_requires_redis_auth() -> None:
    """Deliberately scoped to PRODUCTION only -- a staging environment legitimately may not have real Redis
    auth wired up yet, and this validator must not block it."""
    settings = AppSettings(environment=Environment.STAGING, redis=RedisSettings(password=None, use_tls=False))
    assert settings.environment == Environment.STAGING


def test_an_empty_database_url_falls_back_to_the_individual_timescale_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """`DATABASE_URL=` (vide, comme dans .env.example) ne doit pas écraser TIMESCALE_HOST : sinon asyncpg reçoit un DSN vide et se rabat sur 127.0.0.1."""
    from core.config import TimescaleSettings

    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("TIMESCALE_HOST", "timescaledb")
    monkeypatch.setenv("TIMESCALE_USER", "u")
    monkeypatch.setenv("TIMESCALE_PASSWORD", "p")
    monkeypatch.setenv("TIMESCALE_DATABASE", "d")
    settings = TimescaleSettings(_env_file=None)
    assert settings.connection_url == "postgresql://u:p@timescaledb:5432/d"
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@elsewhere:5432/z")
    assert TimescaleSettings(_env_file=None).connection_url == "postgresql://x:y@elsewhere:5432/z"

"""Settings for the Retail Gateway -- deliberately its own :class:`~pydantic_settings.BaseSettings`
tree, not an addition to :mod:`core.config`.

**Air-gap, applied to configuration too**: ``core/config.py``'s ``get_settings()`` eagerly
constructs and validates *every* settings group on first call (``RedisSettings``, ``BrokerSettings``,
``IbkrSettings``, ...) -- see that module's own docstring. Importing it here would mean this
B2C service's own startup depends on the trading core's entire settings surface validating
cleanly (e.g. broker credentials it has no reason to know about), the exact coupling the mission's
own "jamais partager le même processeur ni la même boucle asynchrone" rule exists to prevent.
This module reads only ``RETAIL_*``-prefixed environment variables and has zero import of
anything under ``core/``, ``src/``, or any trading-agent package.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Connection settings for the Retail Gateway's own Postgres instance.

    A plain, vanilla ``postgres:16-alpine`` -- **not** the trading core's TimescaleDB (see
    ``docker-compose.retail.yml``'s own comment on why: sharing that database would mean a
    schema migration or a load spike on one side can affect the other, exactly the coupling
    the air-gap rule forbids). Same host/port/user/password + optional ``DATABASE_URL``
    override shape as ``core.config.TimescaleSettings``, for the same real reason that module
    documents (most Postgres tooling expects a bare ``DATABASE_URL``) -- the shape is
    reused, the connection target is not.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Retail Postgres server hostname or IP.")
    port: int = Field(default=5432, ge=1, le=65535, description="Retail Postgres server port.")
    database: str = Field(default="retail_gateway", description="Database name.")
    user: str = Field(default="retail_gateway", description="Database user.")
    password: SecretStr | None = Field(default=None, description="Database password.")
    database_url: str | None = Field(
        default=None,
        validation_alias="RETAIL_DATABASE_URL",
        description="Full connection string, overrides the individual fields above when set.",
    )
    min_pool_size: int = Field(default=2, ge=1, description="asyncpg pool floor.")
    max_pool_size: int = Field(default=10, ge=1, description="asyncpg pool ceiling.")

    @property
    def connection_url(self) -> str:
        if self.database_url:
            return self.database_url
        password_part = f":{self.password.get_secret_value()}" if self.password else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"


class RetailRedisSettings(BaseSettings):
    """The Retail Gateway's own Redis instance -- rate-limiter state + the Celery broker/result
    backend for the welcome-email queue. Deliberately a **separate container** from both the
    trading core's shared broker (``redis-broker`` in ``docker-compose.yml``) and the EMS
    dashboard's IAM Redis (``core.iam``) -- see ``docker-compose.retail.yml``'s own comment.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Retail Redis hostname or IP.")
    port: int = Field(default=6379, ge=1, le=65535, description="Retail Redis port.")
    password: SecretStr | None = Field(default=None, description="Retail Redis password, if required.")
    db: int = Field(default=0, ge=0, description="Redis logical DB index for rate-limiter keys.")
    celery_broker_db: int = Field(
        default=1, ge=0, description="Separate logical DB for the Celery broker (queued task messages)."
    )
    celery_result_backend_db: int = Field(
        default=2, ge=0, description="Separate logical DB for Celery task results/state."
    )

    def url(self, *, db: int | None = None) -> str:
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db if db is None else db}"


class JwtSettings(BaseSettings):
    """JWT issuance/verification settings. Unlike the fail-soft optional groups in
    ``core/config.py`` (email, Twilio, ...), ``secret_key`` has **no default** and is required
    -- an auth service silently booting with a guessable/empty signing key is a real
    vulnerability class (forgeable session tokens), not a degraded-but-safe fallback.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_JWT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    secret_key: SecretStr = Field(description="HMAC signing key for access tokens/refresh tokens. Required -- no default.")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm.")
    access_token_expire_minutes: int = Field(
        default=15, gt=0,
        description="Access-token lifetime -- short-lived by design (Identity Gateway hardening); "
        "see refresh_token_expire_days for the session-continuity half of that trade-off.",
    )
    refresh_token_expire_days: int = Field(
        default=7, gt=0,
        description="Refresh-token lifetime. Issued only inside an HttpOnly/Secure/SameSite=Strict "
        "cookie (see main.py) -- never readable by client-side JavaScript.",
    )


class RateLimitSettings(BaseSettings):
    """Per-endpoint abuse limits, enforced by the Redis fixed-window limiter in
    :mod:`retail_gateway.security` (same real algorithm ``p2p_api.py``'s own
    ``_enforce_rate_limit`` already established for this project: ``INCR``/``EXPIRE``,
    fail-closed on Redis downtime -- reimplemented standalone here rather than imported,
    for the same air-gap reason the whole package avoids importing ``p2p_api.py``/``core.*``).
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_RATE_LIMIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    signup_per_hour_per_ip: int = Field(default=5, gt=0, description="Max /signup calls per IP per rolling hour.")
    login_per_minute_per_ip: int = Field(default=10, gt=0, description="Max /login calls per IP per rolling minute.")
    otp_requests_per_window: int = Field(
        default=3, gt=0, description="Max /otp/request calls per bucket (IP, and separately per identifier) per otp_request_window_seconds."
    )
    otp_request_window_seconds: int = Field(default=900, gt=0, description="OTP request window (default: 15 minutes, matching the brief).")
    otp_backoff_base_seconds: int = Field(
        default=900, gt=0,
        description="Lockout duration after the *first* window violation -- doubles per further "
        "violation (see security.enforce_rate_limit_with_backoff) up to otp_backoff_max_seconds. "
        "The real toll-fraud/SMS-pumping shield: a bot that keeps hammering /otp/request past the "
        "plain window limit faces an exponentially growing lockout, not just a flat per-window cap.",
    )
    otp_backoff_max_seconds: int = Field(default=86_400, gt=0, description="Lockout ceiling (default: 24h).")
    otp_verify_max_attempts: int = Field(
        default=5, gt=0, description="Max /otp/verify guesses against one issued code before it is invalidated outright -- a 6-digit code has only 1,000,000 possibilities, too few to leave unlimited-attempt within a 5-minute TTL."
    )


class EmailSettings(BaseSettings):
    """Transactional email provider config for the welcome-email Celery task.

    Fail-soft by default (``provider="console"``): an unconfigured deployment still accepts
    signups and queues the task, it just logs the email instead of sending it -- same
    "degrade, don't crash" contract ``core.config.EmailSettings`` already established for the
    reporting engine. ``resend`` is the one real, sending provider implemented in
    :mod:`retail_gateway.email_provider` (see that module's own docstring on why: a plain
    HTTPS POST via stdlib ``urllib``, no new SDK dependency). ``ses`` is a real, honestly
    disclosed, **not-yet-implemented** option -- same interface, swap in a
    :class:`~retail_gateway.email_provider.EmailProvider` implementation using ``boto3`` (already
    a dependency of the main project, not added here) when a real AWS account exists to test
    against; not faked here.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_EMAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["console", "resend", "ses"] = Field(default="console")
    resend_api_key: SecretStr | None = Field(default=None)
    from_address: str = Field(default="welcome@nothierly.example")


class SmsSettings(BaseSettings):
    """SMS provider config for the OTP-delivery Celery task. Same fail-soft "console by default"
    contract as :class:`EmailSettings` -- an unconfigured deployment still accepts OTP requests
    and queues the task, it just logs the code instead of texting it.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_SMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["console", "twilio"] = Field(default="console")
    twilio_account_sid: str | None = Field(default=None)
    twilio_auth_token: SecretStr | None = Field(default=None)
    twilio_from_number: str | None = Field(default=None, description="E.164 sender number, e.g. +15551234567.")


class OtpSettings(BaseSettings):
    """One-time-password generation/storage settings. The OTP value itself never touches
    Postgres (see repository.py/otp_store.py) -- it lives only as an Argon2 hash in Redis with
    a hard TTL, per the brief's own explicit prohibition on plaintext storage.
    """

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_OTP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    length: int = Field(default=6, ge=4, le=10, description="Digit count.")
    ttl_seconds: int = Field(default=300, gt=0, description="Redis TTL for a stored (hashed) OTP -- default: 5 minutes, matching the brief.")


class RetailGatewaySettings(BaseSettings):
    """Top-level settings container -- mirrors ``core.config.AppSettings``'s own composition
    pattern (one field per sub-settings group), scoped to only what this microservice needs.
    """

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RetailRedisSettings = Field(default_factory=RetailRedisSettings)
    jwt: JwtSettings = Field(default_factory=JwtSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    sms: SmsSettings = Field(default_factory=SmsSettings)
    otp: OtpSettings = Field(default_factory=OtpSettings)


@lru_cache
def get_settings() -> RetailGatewaySettings:
    """Process-wide cached settings instance -- same ``lru_cache`` pattern as ``core.config.get_settings``."""
    return RetailGatewaySettings()

"""Strictly-typed, environment-driven application configuration.

This module is the single source of truth for runtime configuration. It
loads values from process environment variables (and an optional ``.env``
file at the project root) via ``pydantic-settings``, and validates every
value eagerly so that a misconfigured deployment fails at startup with a
clear, actionable error instead of failing later inside the trading loop.

Typical usage::

    from core.config import get_settings

    settings = get_settings()
    redis_client = redis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
    )

Environment variables are namespaced by prefix:

- ``REDIS_*``   -> :class:`RedisSettings`
- ``ZMQ_*``     -> :class:`ZmqSettings`
- ``BROKER_*``  -> :class:`BrokerSettings`
- everything else -> :class:`AppSettings`
"""

from __future__ import annotations

import re
from enum import Enum
from functools import lru_cache
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ZMQ_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<scheme>tcp|ipc|inproc)://(?P<host>[^:]+)(:(?P<port>\d+))?$"
)


class ConfigurationError(RuntimeError):
    """Raised when the application configuration is invalid or incomplete.

    This wraps the underlying ``pydantic.ValidationError`` with a message
    that is safe to surface in logs/CLI output (no secret values), while
    still pointing at exactly which setting is missing or malformed.
    """


class Environment(str, Enum):
    """Deployment environment discriminator."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


def _validate_zmq_address(value: str) -> str:
    """Validate a ZeroMQ endpoint address of the form ``scheme://host:port``.

    Args:
        value: The raw endpoint string to validate (e.g.
            ``"tcp://127.0.0.1:5555"``).

    Returns:
        The unmodified, validated address string.

    Raises:
        ValueError: If the scheme is unsupported, the host is not a valid
            IPv4 address (for ``tcp``/``inproc`` schemes) or hostname, or
            the port is outside the valid ``1-65535`` range.
    """
    match = _ZMQ_ADDRESS_PATTERN.match(value)
    if match is None:
        raise ValueError(
            f"Invalid ZMQ address {value!r}; expected format "
            "'tcp://<host>:<port>' or 'ipc:///path/to/socket'."
        )

    scheme = match.group("scheme")
    host = match.group("host")
    port = match.group("port")

    if scheme in ("tcp", "inproc") and host not in ("*", "localhost"):
        try:
            IPv4Address(host)
        except AddressValueError as exc:
            raise ValueError(
                f"Invalid ZMQ host {host!r} in address {value!r}: {exc}"
            ) from exc

    if scheme in ("tcp", "inproc"):
        if port is None:
            raise ValueError(
                f"ZMQ address {value!r} must specify a port for scheme {scheme!r}."
            )
        port_number = int(port)
        if not (1 <= port_number <= 65535):
            raise ValueError(
                f"ZMQ port {port_number} in address {value!r} is out of "
                "the valid range 1-65535."
            )

    return value


class RedisSettings(BaseSettings):
    """Connection settings for the Redis message/state broker.

    Populated from environment variables prefixed with ``REDIS_`` (e.g.
    ``REDIS_HOST``, ``REDIS_PORT``).
    """

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Redis server hostname or IP.")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis server port.")
    db: int = Field(default=0, ge=0, le=15, description="Redis logical database index.")
    password: SecretStr | None = Field(
        default=None, description="Redis AUTH password, if required."
    )
    use_tls: bool = Field(default=False, description="Whether to connect over TLS.")


class ZmqSettings(BaseSettings):
    """Connection settings for the ZeroMQ inter-agent message bus.

    Populated from environment variables prefixed with ``ZMQ_`` (e.g.
    ``ZMQ_PUB_ADDRESS``, ``ZMQ_SUB_ADDRESS``).
    """

    model_config = SettingsConfigDict(
        env_prefix="ZMQ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pub_address: str = Field(
        default="tcp://127.0.0.1:5555",
        description=(
            "When broker_frontend_address is unset: address this process "
            "BINDS to PUBlish messages on (one agent talking to itself, "
            "the standalone/demo/test topology). Ignored when a broker "
            "frontend is configured -- see broker_frontend_address."
        ),
    )
    sub_address: str = Field(
        default="tcp://127.0.0.1:5556",
        description=(
            "When broker_backend_address is unset: address this process "
            "CONNECTS to, to SUBscribe to messages. Ignored when a broker "
            "backend is configured -- see broker_backend_address."
        ),
    )
    broker_frontend_address: str | None = Field(
        default=None,
        description=(
            "Address of a running ~core.zmq_broker.ZmqBroker's XSUB "
            "frontend. When set, every agent's publisher CONNECTS here "
            "instead of binding pub_address -- required whenever more "
            "than one agent (process or task) shares this configuration, "
            "since only one of them could otherwise bind the same port."
        ),
    )
    broker_backend_address: str | None = Field(
        default=None,
        description=(
            "Address of a running ~core.zmq_broker.ZmqBroker's XPUB "
            "backend. When set, every agent's subscriber connects here "
            "instead of sub_address."
        ),
    )
    high_water_mark: int = Field(
        default=1000,
        ge=1,
        description="Max queued messages per socket before ZMQ starts dropping/blocking.",
    )

    @field_validator("pub_address", "sub_address")
    @classmethod
    def _check_address(cls, value: str) -> str:
        return _validate_zmq_address(value)

    @field_validator("broker_frontend_address", "broker_backend_address", mode="before")
    @classmethod
    def _blank_broker_address_means_unset(cls, value: str | None) -> str | None:
        # pydantic-settings maps an env var set to an empty string (e.g.
        # `ZMQ_BROKER_FRONTEND_ADDRESS=` in a copied .env.example, the
        # standalone/non-Docker default) to `""`, not `None`, for a plain
        # `str | None` field -- unlike `SecretStr | None` fields
        # elsewhere in this module, which pydantic already treats an
        # empty value as unset for. Without this, every non-broker
        # deployment would fail validation on startup.
        return None if value == "" else value

    @field_validator("broker_frontend_address", "broker_backend_address")
    @classmethod
    def _check_broker_address(cls, value: str | None) -> str | None:
        return value if value is None else _validate_zmq_address(value)


class BrokerSettings(BaseSettings):
    """Credentials and endpoint for the brokerage/exchange API.

    Populated from environment variables prefixed with ``BROKER_``. Both
    ``api_key`` and ``api_secret`` are required with no default: if either
    is absent from the environment/``.env`` file, settings loading fails
    immediately with a :class:`ConfigurationError`.
    """

    model_config = SettingsConfigDict(
        env_prefix="BROKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = Field(description="Brokerage/exchange API key. Required.")
    api_secret: SecretStr = Field(description="Brokerage/exchange API secret. Required.")
    base_url: str = Field(
        default="https://api.broker.example.com",
        description="Base URL of the brokerage/exchange REST API.",
    )


class VaultSettings(BaseSettings):
    """Master key for :class:`~src.execution.security_vault.SecurityVault`.

    Populated from environment variables prefixed with ``VAULT_``.
    Required, no default: per-exchange broker credentials added at
    runtime through the dashboard's credential form (see
    ``src/dashboard/ems_pages/liquidity_routing.py``) are encrypted
    with this key before being written to Redis -- an unset key would
    mean either refusing to store credentials at all or falling back to
    plaintext, and this project's established convention (see
    :class:`OandaSettings`, :class:`BrokerSettings`) is to fail fast at
    startup instead of silently degrading a security-relevant default.
    """

    model_config = SettingsConfigDict(
        env_prefix="VAULT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    master_key: SecretStr = Field(
        description="Passphrase used to derive the Fernet key encrypting broker credentials at rest. Required."
    )


class IbkrSettings(BaseSettings):
    """Connection parameters for Interactive Brokers' TWS/IB Gateway -- not an API key/secret pair.

    Populated from environment variables prefixed with ``IBKR_``. Unlike
    every other broker settings group in this module, IBKR authenticates
    via the *running TWS/IB Gateway desktop application itself* (the
    operator logs into that app directly), not a key this process
    presents over the wire -- so every field here has a sane default and
    none are required: a project with no IBKR account configured must
    still boot cleanly, unlike :class:`OandaSettings`/
    :class:`BrokerSettings`, which back a broker every deployment of
    this project actually depends on.
    """

    model_config = SettingsConfigDict(
        env_prefix="IBKR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="Host TWS/IB Gateway is listening on.")
    port: int = Field(
        default=7497,
        description="7497=TWS paper (default, safe), 7496=TWS live, 4002=IB Gateway paper, 4001=IB Gateway live.",
    )
    client_id: int = Field(default=1, description="IBKR API client ID -- must be unique per simultaneous connection.")


class OandaSettings(BaseSettings):
    """Credentials for the OANDA v20 forex broker API.

    Populated from environment variables prefixed with ``OANDA_``. OANDA's
    auth model is a single personal access token plus an account ID (not
    a key/secret pair like :class:`BrokerSettings`' CCXT-oriented shape),
    hence a dedicated settings group rather than reusing that one.
    Required, with no default, for the same reason as
    :class:`BrokerSettings`: :class:`~src.agents.supervisor.
    broker_client.OandaClient` cannot function at all without them, so
    failing fast at startup is preferable to failing on the first order.
    """

    model_config = SettingsConfigDict(
        env_prefix="OANDA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_token: SecretStr = Field(description="OANDA v20 API personal access token. Required.")
    account_id: str = Field(min_length=1, description="OANDA account ID, e.g. '101-004-1234567-001'.")
    environment: Literal["practice", "live"] = Field(
        default="practice",
        description=(
            "'practice' (paper trading, no real funds -- the default) or "
            "'live' (real money). Must be set to 'live' explicitly."
        ),
    )


class VastAiSettings(BaseSettings):
    """Credentials and SSH key paths for Vast.ai GPU rental, used only by
    the offline :mod:`scripts.run_remote_training` pipeline (:mod:`src.
    mlops`) -- no live trading agent touches this.

    Populated from environment variables prefixed with ``VASTAI_``.
    Optional: the pipeline simply cannot run without these, but their
    absence must never affect anything else in this system starting up
    (unlike a broker credential, this is a purely offline convenience
    tool).
    """

    model_config = SettingsConfigDict(
        env_prefix="VASTAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = Field(default=None, description="Vast.ai API key.")
    ssh_public_key_path: Path | None = Field(
        default=None,
        description="Local path to the SSH public key authorized on rented instances.",
    )
    ssh_private_key_path: Path | None = Field(
        default=None,
        description="Local path to the matching SSH private key, used to connect to them.",
    )


class MacroApiSettings(BaseSettings):
    """API credentials for macroeconomic / on-chain data providers.

    Populated from environment variables prefixed with ``MACRO_``. Unlike
    :class:`BrokerSettings`, these are intentionally *optional*: a
    provider left unconfigured doesn't block startup or raise a
    :class:`ConfigurationError` -- :class:`~src.agents.macro.agent.
    MacroAgent` simply falls back to a simulated reading for that
    indicator instead, so a partially-configured macro data pipeline
    never blocks the rest of the trading system.
    """

    model_config = SettingsConfigDict(
        env_prefix="MACRO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fred_api_key: SecretStr | None = Field(
        default=None, description="FRED (Federal Reserve Economic Data) API key."
    )
    glassnode_api_key: SecretStr | None = Field(
        default=None, description="Glassnode on-chain data API key."
    )


class NewsApiSettings(BaseSettings):
    """API credentials for the financial news headline provider.

    Populated from environment variables prefixed with ``NEWS_``.
    Optional, following the same rationale as :class:`MacroApiSettings`:
    :class:`~src.agents.nlp.agent.NlpAgent` falls back to a simulated
    headline feed when unset, rather than failing to start.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEWS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = Field(
        default=None, description="Financial news API key (e.g. NewsAPI.org)."
    )


class GuardianSettings(BaseSettings):
    """API credentials/config for the Guardian AI (Chief Risk Officer) module.

    Populated from environment variables prefixed with ``GUARDIAN_``.
    Optional, following the same rationale as :class:`MacroApiSettings`/
    :class:`NewsApiSettings`: :mod:`src.agents.guardian_ai` falls back to
    a deterministic, template-based risk assessment when no provider is
    configured (``llm_provider="none"``, the default), rather than
    failing to start or silently skipping the safety check a dangerous
    action is supposed to trigger.
    """

    model_config = SettingsConfigDict(
        env_prefix="GUARDIAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["openai", "anthropic", "none"] = Field(
        default="none",
        description=(
            "Which LLM API src.agents.guardian_ai.assess_risk calls. "
            "'none' (the default) uses the built-in template-based "
            "fallback instead of any network call."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the configured llm_provider. Required (enforced at call time, not startup) if llm_provider != 'none'.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model identifier for the configured provider. Defaults "
            "(when unset) to a small, low-latency model per provider -- "
            "see src.agents.guardian_ai's _DEFAULT_MODEL_BY_PROVIDER."
        ),
    )


class DashboardSettings(BaseSettings):
    """Login credentials for ``src/dashboard/app.py``'s :func:`check_password` gate.

    Populated from environment variables prefixed with ``DASHBOARD_``.
    Both fields default to ``None`` -- ``check_password`` treats either
    being unset as "refuse every login attempt", never as "no password
    required", so there is no accidental-bypass state.

    Deliberately its own :class:`BaseSettings` (reads ``.env`` itself)
    rather than the dashboard reading ``os.environ`` directly: pydantic-
    settings' ``env_file=".env"`` support is what actually loads ``.env``
    into this process at all -- plain ``os.environ.get(...)`` only sees
    variables the launching shell itself exported, so a value that only
    exists in ``.env`` would silently never match, no matter what is
    typed at the login form.
    """

    model_config = SettingsConfigDict(
        env_prefix="DASHBOARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    user: str | None = Field(default=None, description="Required username for the dashboard login form.")
    password: SecretStr | None = Field(
        default=None, description="Required password for the dashboard login form."
    )


class R2Settings(BaseSettings):
    """Cloudflare R2 (S3-compatible) data lake credentials, for offline backtest/training data.

    Used by :mod:`src.data_pipeline.data_lake_sync` -- entirely separate
    from the live trading system (ZMQ/Redis/TimescaleDB): this is where
    historical Parquet datasets for offline backtesting/model training
    are stored, not anything a live agent reads from.

    Deliberately its own :class:`BaseSettings` reading ``.env`` itself,
    same reasoning as :class:`DashboardSettings`: plain ``os.environ.
    get(...)`` (or a raw ``boto3.client(...)`` call relying on boto3's
    own environment auto-detection) only sees variables the launching
    shell itself exported, silently missing a value that only exists in
    ``.env``. ``access_key_id``/``secret_access_key`` deliberately use
    the unprefixed ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` names
    (via ``validation_alias``, the same mechanism :class:`TimescaleSettings
    .database_url` already uses to break out of its class's own prefix)
    rather than ``R2_ACCESS_KEY_ID``/``R2_SECRET_ACCESS_KEY``: those are
    R2's actual documented credential variable names (R2 is S3-API-
    compatible by design, and Cloudflare's own dashboard calls them
    exactly that), so matching them exactly is what lets a user paste
    credentials from Cloudflare's UI without renaming anything.
    """

    model_config = SettingsConfigDict(
        env_prefix="R2_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Required alongside access_key_id/secret_access_key's
        # validation_alias below: without this, pydantic-settings only
        # accepts the *alias* (AWS_ACCESS_KEY_ID) as an __init__ keyword,
        # not the field's own Python name -- fine for env-var loading,
        # but breaks constructing R2Settings(access_key_id=...) directly
        # in code (every test in tests/test_data_lake_sync.py does this).
        populate_by_name=True,
    )

    account_id: str | None = Field(
        default=None,
        description="Cloudflare account ID -- combined with the fixed R2 domain to form endpoint_url.",
    )
    bucket_name: str | None = Field(default=None, description="R2 bucket holding historical datasets.")
    access_key_id: str | None = Field(
        default=None, validation_alias="AWS_ACCESS_KEY_ID", description="R2 S3 API access key ID."
    )
    secret_access_key: SecretStr | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY", description="R2 S3 API secret access key."
    )

    @property
    def endpoint_url(self) -> str | None:
        """The R2 S3-API endpoint URL, or ``None`` if ``account_id`` is unset.

        Cloudflare's fixed, documented URL shape for R2's S3-compatible
        API -- not configurable per-bucket, derived entirely from the
        account ID.
        """
        if self.account_id is None:
            return None
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class TwilioSettings(BaseSettings):
    """Credentials for ``src/agents/alert_agent.py``'s SMS circuit-breaker alerts ("The Red Phone").

    Populated from environment variables prefixed with ``TWILIO_``,
    except ``admin_phone_number`` -- the brief's own literal env var
    name is the unprefixed ``ADMIN_PHONE_NUMBER``, read via
    ``validation_alias`` the same way :class:`R2Settings`.
    ``access_key_id``/``secret_access_key`` break out of their class's
    own prefix. All four fields default to ``None``: unconfigured means
    "cannot send SMS", not "send anyway with an empty destination" --
    :func:`~src.agents.alert_agent.send_sms_alert` logs and skips rather
    than raising when any are missing, so a misconfigured deployment
    loses only the SMS channel, never crashes the agent watching for a
    real danger.
    """

    model_config = SettingsConfigDict(
        env_prefix="TWILIO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,  # required alongside admin_phone_number's validation_alias -- see R2Settings for the identical reasoning
    )

    account_sid: str | None = Field(default=None, description="Twilio Account SID.")
    auth_token: SecretStr | None = Field(default=None, description="Twilio Auth Token.")
    phone_number: str | None = Field(default=None, description="Twilio-provisioned sending number, e.g. '+15551234567'.")
    admin_phone_number: str | None = Field(
        default=None, validation_alias="ADMIN_PHONE_NUMBER", description="Destination number for critical alerts."
    )


class EmailSettings(BaseSettings):
    """SMTP credentials for ``src/execution/reporting_engine.py``'s monthly client report emails.

    Populated from environment variables prefixed with ``EMAIL_``. Same
    all-optional, fail-soft shape as :class:`TwilioSettings`: an
    unconfigured deployment can still generate real report PDFs, it
    just cannot email them -- :func:`~src.execution.reporting_engine.
    send_report_email` logs and skips rather than raising, the same
    contract :func:`~src.agents.alert_agent.send_sms_alert` already
    established for a missing notification channel. Plain ``smtplib``
    (stdlib, already installed everywhere) rather than a provider SDK
    (AWS SES, SendGrid): this project has no existing account with
    either, and SMTP is the one transport that works against literally
    any provider (including both of those, via their SMTP relay) without
    picking a specific vendor dependency for a single new feature.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    smtp_host: str | None = Field(default=None, description="SMTP server host.")
    smtp_port: int = Field(default=587, description="SMTP port -- 587 (STARTTLS, the default) or 465 (implicit TLS).")
    smtp_user: str | None = Field(default=None, description="SMTP auth username.")
    smtp_password: SecretStr | None = Field(default=None, description="SMTP auth password.")
    from_address: str | None = Field(default=None, description="Sender address reports are mailed from.")


class ApiMiddlewareSettings(BaseSettings):
    """Credentials and limits for ``api_middleware/gateway.py``'s B2B order-ingestion API.

    Populated from environment variables prefixed with
    ``API_MIDDLEWARE_``. ``api_keys`` is empty by default -- deliberately
    fails closed (every request rejected) rather than accidentally
    accepting unauthenticated traffic when unconfigured, the same
    fail-closed default this project already uses for
    :class:`DashboardSettings`.
    """

    model_config = SettingsConfigDict(
        env_prefix="API_MIDDLEWARE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_keys: SecretStr | None = Field(
        default=None,
        description=(
            "Comma-separated list of valid client API keys (checked against "
            "the X-API-Key header). Unset means no key validates -- every "
            "request is rejected, not accepted."
        ),
    )
    rate_limit_per_minute: int = Field(
        default=60,
        gt=0,
        description="Max requests per API key per rolling 60s window, enforced via Redis.",
    )
    self_base_url: str = Field(
        default="http://127.0.0.1:8000",
        description=(
            "This gateway's own externally-reachable base URL -- used only by the "
            "/ws/copilot WebSocket handler's self-call back into POST "
            "/api/v1/orders/{order_id}/adjust once a pending aggression adjustment is "
            "confirmed (see src.agents.quant_copilot.confirm_pending_aggression_adjustment)."
        ),
    )
    cors_allowed_origins: str | None = Field(
        default=None,
        description=(
            "Comma-separated list of browser origins allowed to call this API "
            "cross-origin (e.g. 'https://app.example.com,https://staging.example.com'). "
            "Unset (the default) means no CORSMiddleware is installed at all -- this "
            "API stays server-to-server-only (its real integrations today: internal "
            "Python clients, no browser JS caller) rather than defaulting to a "
            "permissive origin list. Set this only once a real browser-based client "
            "needs to call the API directly."
        ),
    )


class ReconciliationSettings(BaseSettings):
    """Configuration for ``src/execution/eod_reconciliator.py``'s daily broker-balance reconciliation.

    Populated from environment variables prefixed with ``RECONCILIATION_``.
    ``broker_ids`` is empty by default -- deliberately fails closed (no
    brokers reconciled) rather than guessing which of this project's
    real broker adapters (see ``BrokerFactory``) an operator actually
    has live credentials for, the same "explicit, never fabricated"
    convention ``ApiMiddlewareSettings.api_keys`` already uses.
    """

    model_config = SettingsConfigDict(
        env_prefix="RECONCILIATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    broker_ids: str | None = Field(
        default=None,
        description="Comma-separated broker_ids to reconcile at EOD (e.g. 'binance,oanda,ibkr'). Unset means none are reconciled.",
    )
    break_threshold_usd: float = Field(
        default=50.0, gt=0,
        description="Minimum unexplained day-over-day balance delta (USD) to flag as a real Break, not noise from rounding/fees.",
    )


class FixGatewaySettings(BaseSettings):
    """Configuration for ``api_middleware/fix_server.py``'s real FIX 4.4 order-entry Acceptor.

    Populated from environment variables prefixed with ``FIX_``.
    ``allowed_client_sender_comp_ids`` is empty by default -- deliberately
    fails closed (every Logon rejected) rather than accepting a session
    from an unrecognized counterparty, the same convention
    ``ApiMiddlewareSettings.api_keys`` already uses for the REST gateway.
    """

    model_config = SettingsConfigDict(
        env_prefix="FIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Interface this Acceptor binds to. Defaults to loopback-only -- safe by default, matching AppSettings.mock_mode's own reasoning; set to 0.0.0.0 explicitly to accept real external client connections.",
    )
    port: int = Field(default=5001, ge=1, le=65535)
    our_comp_id: str = Field(
        default="NOTHIERLY",
        description="This Acceptor's own SenderCompID on outbound messages (Logon ack, ExecutionReport, Heartbeat) -- what clients must set as their TargetCompID.",
    )
    allowed_client_sender_comp_ids: str | None = Field(
        default=None,
        description="Comma-separated list of client SenderCompIDs allowed to Logon. Unset means every Logon is rejected, not accepted.",
    )
    heartbeat_interval_seconds: int = Field(default=30, gt=0, description="Negotiated HeartBtInt (tag 108) this Acceptor requires from every session.")


class TimescaleSettings(BaseSettings):
    """Connection settings for the TimescaleDB database.

    Populated from environment variables prefixed with ``TIMESCALE_``,
    used by :class:`~src.agents.archivist.agent.ArchivistAgent` (its
    ``bus_messages`` hypertable) and, as of the data pipeline chapter, by
    :mod:`src.data_pipeline` as well (``ohlcv_candles``/
    ``macro_indicators``/``computed_features`` -- see
    :mod:`src.data_pipeline.schema`) -- the same database, different
    tables, one shared connection story (:mod:`core.timescale_client`).
    """

    model_config = SettingsConfigDict(
        env_prefix="TIMESCALE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1", description="TimescaleDB server hostname or IP.")
    port: int = Field(default=5432, ge=1, le=65535, description="TimescaleDB server port.")
    database: str = Field(default="trading", description="Database name.")
    user: str = Field(default="postgres", description="Database user.")
    password: SecretStr | None = Field(default=None, description="Database password, if required.")
    # A bare (unprefixed) DATABASE_URL, read via validation_alias rather
    # than the class's own TIMESCALE_ prefix -- matches the connection
    # string convention most Postgres/TimescaleDB tooling (including the
    # docker-compose service this project's own data pipeline chapter
    # adds) expects, without requiring a second, differently-named env
    # var. Optional: connection_url below falls back to composing one
    # from host/port/database/user/password when this is unset, so
    # existing ArchivistAgent-only deployments (TIMESCALE_* only, no
    # DATABASE_URL) keep working unchanged.
    database_url: str | None = Field(
        default=None,
        validation_alias="DATABASE_URL",
        description="Full connection string (postgresql://user:pass@host:port/db). Overrides the individual fields above when set.",
    )

    @property
    def connection_url(self) -> str:
        """The connection string every TimescaleDB caller should actually use.

        Prefers :attr:`database_url` verbatim when set; otherwise
        composes one from the individual fields. Centralizing this here
        (rather than in each caller) means ``core.timescale_client``,
        :class:`~src.agents.archivist.agent.ArchivistAgent`, and every
        :mod:`src.data_pipeline` module agree on the same URL by
        construction, never by convention.
        """
        if self.database_url is not None:
            return self.database_url
        password_part = f":{self.password.get_secret_value()}" if self.password is not None else ""
        return f"postgresql://{self.user}{password_part}@{self.host}:{self.port}/{self.database}"


class AppSettings(BaseSettings):
    """Top-level application settings, aggregating every settings group.

    Populated from unprefixed environment variables (e.g. ``ENVIRONMENT``,
    ``LOG_LEVEL``) plus the nested :class:`RedisSettings`,
    :class:`ZmqSettings`, :class:`BrokerSettings`, :class:`OandaSettings`,
    :class:`VastAiSettings`, :class:`MacroApiSettings`,
    :class:`NewsApiSettings`, :class:`TimescaleSettings`,
    :class:`GuardianSettings`, :class:`DashboardSettings`,
    :class:`R2Settings`, and :class:`ApiMiddlewareSettings` groups, each
    of which independently reads its own prefixed environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment (development/staging/production).",
    )
    log_level: str = Field(default="INFO", description="Root log level.")
    service_name: str = Field(
        default="trading-engine", description="Service name reported in structured logs."
    )
    mock_mode: bool = Field(
        default=True,
        description=(
            "When True (the default -- safe by default, matching this "
            "project's other financial-action defaults), "
            "~src.agents.execution.agent.ExecutionAgent simulates order "
            "execution (logs the intended order, never calls OANDA) "
            "instead of placing a real one. Independent of OANDA's own "
            "practice/live environment switch: MOCK_MODE governs whether "
            "the broker is called *at all*, not which of its own "
            "environments is called. Must be explicitly set to false to "
            "place any real (including OANDA-practice) order."
        ),
    )

    redis: RedisSettings = Field(default_factory=RedisSettings)
    zmq: ZmqSettings = Field(default_factory=ZmqSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    oanda: OandaSettings = Field(default_factory=OandaSettings)
    ibkr: IbkrSettings = Field(default_factory=IbkrSettings)
    vastai: VastAiSettings = Field(default_factory=VastAiSettings)
    macro: MacroApiSettings = Field(default_factory=MacroApiSettings)
    news: NewsApiSettings = Field(default_factory=NewsApiSettings)
    timescale: TimescaleSettings = Field(default_factory=TimescaleSettings)
    guardian: GuardianSettings = Field(default_factory=GuardianSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    r2: R2Settings = Field(default_factory=R2Settings)
    api_middleware: ApiMiddlewareSettings = Field(default_factory=ApiMiddlewareSettings)
    reconciliation: ReconciliationSettings = Field(default_factory=ReconciliationSettings)
    fix_gateway: FixGatewaySettings = Field(default_factory=FixGatewaySettings)
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    vault: VaultSettings = Field(default_factory=VaultSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(
                f"Invalid log_level {value!r}; must be one of {sorted(valid_levels)}."
            )
        return normalized

    @model_validator(mode="after")
    def _require_redis_auth_in_production(self) -> AppSettings:
        """Real, minimal environment-aware enforcement -- reuses the ``environment``/``redis`` fields
        already on this class rather than a second, parallel config mechanism.

        A red-team audit found this project's Redis has no password/TLS
        by default (real, confirmed: a bare local run trivially allows
        cache poisoning and Kill Switch spoofing -- see ``tests/
        red_team_hacker.py``). ``docker-compose.yml``'s own
        ``redis-broker`` service already refuses to start without
        ``REDIS_PASSWORD`` set -- this closes the same gap for any
        process started *outside* that Compose stack (a bare ``python
        scripts/run_*.py``, a bare-metal/systemd deployment, ...),
        which had no equivalent guard until now. Local development is
        deliberately untouched: ``environment`` defaults to
        ``Environment.DEVELOPMENT``, under which this never fires.
        """
        if self.environment == Environment.PRODUCTION and (self.redis.password is None or not self.redis.use_tls):
            missing = []
            if self.redis.password is None:
                missing.append("REDIS_PASSWORD")
            if not self.redis.use_tls:
                missing.append("REDIS_USE_TLS=true")
            raise ValueError(
                f"ENVIRONMENT=production requires {' and '.join(missing)} -- refusing to start with an "
                "unauthenticated/unencrypted Redis connection in production."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load, validate, and cache the application settings singleton.

    The result is cached (``lru_cache``) so settings are parsed once per
    process. Use ``get_settings.cache_clear()`` in tests to force a reload.

    Returns:
        The validated :class:`AppSettings` instance.

    Raises:
        ConfigurationError: If any required environment variable is
            missing or fails validation (e.g. a missing broker API key, an
            out-of-range Redis port, or a malformed ZMQ address).
    """
    try:
        return AppSettings()
    except ValidationError as exc:
        missing_or_invalid = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(
            "Application configuration is invalid or incomplete. "
            f"Check your environment/.env file. Details: {missing_or_invalid}"
        ) from exc

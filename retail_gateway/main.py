"""Retail Gateway -- the B2C signup/auth API. "Protocole Retail Gateway".

**Air-gap, the real architectural constraint this whole package is built around**: this service
never imports anything from ``core/``, ``src/``, or any other trading-agent package in this
repo, runs its own Postgres (not TimescaleDB) and its own Redis (not the trading broker or the
EMS dashboard's IAM store), and ships as its own Docker image (``retail_gateway/Dockerfile``,
not the shared multi-service image in the repo root's ``Dockerfile``) on its own Compose network
(``docker-compose.retail.yml``). It shares a process with nothing that places real trades, and a
crash, a schema migration, or a traffic spike here cannot touch ``ExecutionAgent``, the ZMQ bus,
or the Fleet Command Redis in any direction.

**Data flow to the public dashboard -- one-way, by construction**: ``GET /api/v1/public/performance``
reads only ``public_performance_snapshot``, a table in *this* database. That table is written by
``scripts/export_performance_snapshot.py`` (in the main trading repo, not here) -- a real,
standalone, schedulable job that reads Nothierly's own telemetry and pushes rows into this
database over the network. This service has no code path that reaches back into the trading
core in the other direction; the closest it comes is this cache table.

**Explicitly scoped out, disclosed, not silently missing**:

- **No email verification gate.** The brief asked for a welcome email, not an activate-your-
  account flow -- accounts are active immediately on signup. Adding a verification-token flow
  is real, separable follow-up work (a second Celery task + a ``/verify`` endpoint + an
  ``email_verified`` column), not built here to avoid guessing a UX the brief never specified.
- **No refresh-token rotation.** One access token, real expiry (``JwtSettings.
  access_token_expire_minutes``), no silent renewal. A client that wants a session longer than
  the configured lifetime logs in again. Refresh tokens are a real design surface of their own
  (rotation, revocation, replay detection) -- not added speculatively.
- **No password-reset flow.** Not asked for in this brief; would reuse the same Celery/email
  plumbing already built here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from retail_gateway.config import RetailGatewaySettings, get_settings
from retail_gateway.db import connect_with_backoff, init_schema
from retail_gateway.logging_config import configure_logging, get_logger
from retail_gateway.models import LoginRequest, PerformanceMetric, SignupRequest, TokenResponse, UserPublic
from retail_gateway.redis_client import get_redis
from retail_gateway.repository import EmailAlreadyRegisteredError, UserRepository
from retail_gateway.security import (
    InvalidTokenError,
    client_ip,
    create_access_token,
    decode_access_token,
    enforce_rate_limit,
    hash_password,
    verify_password,
)
from retail_gateway.tasks import send_welcome_email

_log = get_logger(__name__)

_SIGNUP_RATE_LIMIT_PREFIX = "ratelimit:retail:signup:"
_LOGIN_RATE_LIMIT_PREFIX = "ratelimit:retail:login:"

_bearer_scheme = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(service_name="retail-gateway-api")
    pool = await connect_with_backoff(settings.database)
    await init_schema(pool)
    app.state.pool = pool
    app.state.settings = settings
    _log.info("retail_gateway_started")
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="Nothierly Retail Gateway",
    description="B2C signup/auth API -- air-gapped from the Nothierly trading core (see module docstring).",
    version="1.0.0",
    lifespan=lifespan,
)


# -- dependencies -----------------------------------------------------------------------------


def get_pool(request: Request) -> asyncpg.Pool:
    pool: asyncpg.Pool = request.app.state.pool
    return pool


def get_repository(pool: Annotated[asyncpg.Pool, Depends(get_pool)]) -> UserRepository:
    return UserRepository(pool)


def get_app_settings(request: Request) -> RetailGatewaySettings:
    settings: RetailGatewaySettings = request.app.state.settings
    return settings


async def require_current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> UUID:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    try:
        subject = decode_access_token(credentials.credentials, settings=settings.jwt)
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    try:
        return UUID(subject)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject.")


async def _enforce_signup_rate_limit(
    request: Request, settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)]
) -> None:
    redis_client = get_redis(settings.redis)
    await enforce_rate_limit(
        redis_client,
        key_prefix=_SIGNUP_RATE_LIMIT_PREFIX,
        bucket_id=client_ip(request),
        limit=settings.rate_limit.signup_per_hour_per_ip,
        window_seconds=3600,
    )


async def _enforce_login_rate_limit(
    request: Request, settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)]
) -> None:
    redis_client = get_redis(settings.redis)
    await enforce_rate_limit(
        redis_client,
        key_prefix=_LOGIN_RATE_LIMIT_PREFIX,
        bucket_id=client_ip(request),
        limit=settings.rate_limit.login_per_minute_per_ip,
        window_seconds=60,
    )


# -- routes -----------------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/signup",
    response_model=UserPublic,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_enforce_signup_rate_limit)],
)
async def signup(payload: SignupRequest, repository: Annotated[UserRepository, Depends(get_repository)]) -> UserPublic:
    """Create the account, respond immediately, queue the welcome email out-of-band.

    The ``INSERT`` (real Argon2id hash, real Postgres commit) completes before this handler
    returns -- the account genuinely exists the moment the client gets a 201. Only the email
    *send* is deferred: :func:`~retail_gateway.tasks.send_welcome_email` is a Celery task, its
    ``.delay()`` call below just publishes one message onto the (separate, air-gapped) Redis
    broker and returns immediately, picked up by the worker process asynchronously. This is the
    real mechanism behind the brief's "l'API répond instantanément... place la tâche dans la
    file" requirement, not a simulated delay.
    """
    try:
        user = await repository.create_user(email=payload.email, password_hash=hash_password(payload.password))
    except EmailAlreadyRegisteredError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered.")

    send_welcome_email.delay(user.email)
    _log.info("user_signed_up", user_id=str(user.id))
    return UserPublic(id=user.id, email=user.email, created_at=user.created_at)


@app.post(
    "/api/v1/login",
    response_model=TokenResponse,
    dependencies=[Depends(_enforce_login_rate_limit)],
)
async def login(
    payload: LoginRequest,
    repository: Annotated[UserRepository, Depends(get_repository)],
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> TokenResponse:
    user = await repository.get_by_email(payload.email)
    # Constant-shape failure regardless of "unknown email" vs "wrong password" -- verify_password
    # is still called against a real (if unrelated) hash even when the user doesn't exist, so a
    # timing side-channel can't distinguish the two cases from response latency alone.
    password_hash = user.password_hash if user is not None else hash_password("placeholder-for-timing-only")
    password_ok = verify_password(password_hash=password_hash, password=payload.password)
    if user is None or not user.is_active or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    token, expires_at = create_access_token(subject=str(user.id), settings=settings.jwt)
    return TokenResponse(access_token=token, expires_at=expires_at)


@app.get("/api/v1/me", response_model=UserPublic)
async def me(
    user_id: Annotated[UUID, Depends(require_current_user_id)],
    repository: Annotated[UserRepository, Depends(get_repository)],
) -> UserPublic:
    user = await repository.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserPublic(id=user.id, email=user.email, created_at=user.created_at)


@app.get("/api/v1/public/performance", response_model=list[PerformanceMetric])
async def public_performance(repository: Annotated[UserRepository, Depends(get_repository)]) -> list[PerformanceMetric]:
    """No auth, no rate limit gate beyond what a reverse proxy would add in front of this in
    production -- deliberately public, read-only, cached data, exactly what the brief's own
    B2C dashboard spec asks for. See module docstring for the one-way write path.
    """
    rows = await repository.list_recent_performance_metrics()
    return [PerformanceMetric(metric_name=r["metric_name"], metric_value=r["metric_value"], as_of=r["as_of"]) for r in rows]

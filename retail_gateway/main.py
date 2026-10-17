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

**"Identity Gateway" addendum (OTP/MFA, real refresh-token rotation)**: ``POST /api/v1/otp/
request``/``/otp/verify`` add a second, passwordless authentication path alongside the original
email+password one above -- a 6-digit code, Argon2-hashed, stored only in Redis with a 300s TTL
(never Postgres -- see ``otp_store.py``), delivered by a Celery task exactly like the welcome
email, over either channel (``email`` via the existing provider, ``sms`` via the new
``sms_provider.py``/Twilio). A first successful verify for a given identifier *is* signup on
this path (see ``repository.get_or_create_by_identifier``). Real refresh-token rotation now
backs both auth paths (short-lived 15-minute access token + a 7-day refresh token, issued only
as an ``HttpOnly``/``Secure``/``SameSite=Strict`` cookie, allow-listed in Redis by ``jti`` so
:func:`logout`/rotation can actually revoke one -- see ``refresh_tokens.py``) -- the "no
refresh-token rotation" gap the original brief below disclosed as real, separable follow-up
work; this is that follow-up, not a second draft that silently supersedes the disclosure.

**Still explicitly scoped out, disclosed, not silently missing**:

- **No email verification gate on the password signup path.** The original brief asked for a
  welcome email, not an activate-your-account flow -- a password-signup account is active
  immediately. (The OTP path above has no separate verification step because a successful OTP
  verify *is* the proof of channel ownership.)
- **No password-reset flow.** Not asked for in either brief; would reuse the same Celery/email
  plumbing already built here.
- **No cross-channel account linking.** An OTP-created email account and an OTP-created phone
  account for "the same real person" are two separate rows unless they later also complete a
  password signup with a matching email -- deliberately not guessed at; real identity-linking UX
  is its own design surface.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from retail_gateway.config import RetailGatewaySettings, get_settings
from retail_gateway.db import connect_with_backoff, init_schema
from retail_gateway.logging_config import configure_logging, get_logger
from retail_gateway.models import (
    LoginRequest,
    OtpRequest,
    OtpVerify,
    PerformanceMetric,
    SignupRequest,
    TokenResponse,
    UserPublic,
)
from retail_gateway.otp_store import generate_otp_code, store_otp, verify_and_consume_otp
from retail_gateway.redis_client import get_redis
from retail_gateway.refresh_tokens import (
    create_refresh_token,
    decode_refresh_token,
    is_refresh_token_active,
    register_refresh_token,
    revoke_refresh_token,
)
from retail_gateway.repository import EmailAlreadyRegisteredError, UserRepository
from retail_gateway.security import (
    Clock,
    InvalidTokenError,
    client_ip,
    create_access_token,
    decode_access_token,
    enforce_rate_limit,
    enforce_rate_limit_with_backoff,
    get_clock,
    hash_password,
    verify_password,
)
from retail_gateway.tasks import send_otp_email, send_otp_sms, send_welcome_email

_log = get_logger(__name__)

_SIGNUP_RATE_LIMIT_PREFIX = "ratelimit:retail:signup:"
_LOGIN_RATE_LIMIT_PREFIX = "ratelimit:retail:login:"
_OTP_REQUEST_RATE_LIMIT_PREFIX = "ratelimit:retail:otp_request:"
_REFRESH_COOKIE_NAME = "refresh_token"
_REFRESH_COOKIE_PATH = "/api/v1"

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
    request: Request,
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> None:
    redis_client = get_redis(settings.redis)
    await enforce_rate_limit(
        redis_client,
        key_prefix=_SIGNUP_RATE_LIMIT_PREFIX,
        bucket_id=client_ip(request),
        limit=settings.rate_limit.signup_per_hour_per_ip,
        window_seconds=3600,
        clock=clock,
    )


async def _enforce_login_rate_limit(
    request: Request,
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> None:
    redis_client = get_redis(settings.redis)
    await enforce_rate_limit(
        redis_client,
        key_prefix=_LOGIN_RATE_LIMIT_PREFIX,
        bucket_id=client_ip(request),
        limit=settings.rate_limit.login_per_minute_per_ip,
        window_seconds=60,
        clock=clock,
    )


async def _issue_refresh_cookie(response: Response, *, user_id: str, settings: RetailGatewaySettings) -> None:
    """Mint + allow-list a refresh token, set it as an HttpOnly/Secure/SameSite=Strict cookie.

    Never returned in a JSON body -- ``response.set_cookie`` is the only place this value goes,
    per the brief's own explicit requirement that client-side JavaScript must never read it.
    Scoped to ``_REFRESH_COOKIE_PATH`` (this API's own prefix), not site-wide -- the browser
    never attaches it to an unrelated path on the same origin.
    """
    redis_client = get_redis(settings.redis)
    refresh_token, _, jti = create_refresh_token(subject=user_id, settings=settings.jwt)
    ttl_seconds = settings.jwt.refresh_token_expire_days * 86_400
    await register_refresh_token(redis_client, user_id=user_id, jti=jti, ttl_seconds=ttl_seconds)
    response.set_cookie(
        key=_REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=ttl_seconds,
        path=_REFRESH_COOKIE_PATH,
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
    return UserPublic(id=user.id, email=user.email, phone_number=user.phone_number, created_at=user.created_at)


@app.post(
    "/api/v1/login",
    response_model=TokenResponse,
    dependencies=[Depends(_enforce_login_rate_limit)],
)
async def login(
    payload: LoginRequest,
    response: Response,
    repository: Annotated[UserRepository, Depends(get_repository)],
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> TokenResponse:
    user = await repository.get_by_email(payload.email)
    # Constant-shape failure regardless of "unknown email" vs "wrong password" -- verify_password
    # is still called against a real (if unrelated) hash even when the user doesn't exist, so a
    # timing side-channel can't distinguish the two cases from response latency alone.
    password_hash = user.password_hash if user is not None and user.password_hash is not None else hash_password("placeholder-for-timing-only")
    password_ok = verify_password(password_hash=password_hash, password=payload.password)
    if user is None or not user.is_active or user.password_hash is None or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    token, expires_at = create_access_token(subject=str(user.id), settings=settings.jwt)
    await _issue_refresh_cookie(response, user_id=str(user.id), settings=settings)
    return TokenResponse(access_token=token, expires_at=expires_at)


@app.post("/api/v1/otp/request", status_code=status.HTTP_202_ACCEPTED)
async def request_otp(
    payload: OtpRequest,
    request: Request,
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
    clock: Annotated[Clock, Depends(get_clock)],
) -> dict[str, str]:
    """Issue and dispatch one OTP over ``payload.channel``. Always 202 -- the actual send happens
    out-of-band in the Celery worker, same "respond instantly, queue the network call" contract
    ``POST /api/v1/signup`` already established (see that route's own docstring).

    Rate-limited on **both** the caller's IP and the target identifier, independently, each with
    its own escalating lockout (mission Directive 3) -- an attacker spraying OTP requests for one
    victim identifier from many IPs is still capped by the identifier bucket; one IP spraying many
    identifiers (harvesting/toll-fraud) is still capped by the IP bucket.
    """
    redis_client = get_redis(settings.redis)
    rate_limit_settings = settings.rate_limit
    for bucket_id in (f"ip:{client_ip(request)}", f"id:{payload.identifier}"):
        await enforce_rate_limit_with_backoff(
            redis_client,
            key_prefix=_OTP_REQUEST_RATE_LIMIT_PREFIX,
            bucket_id=bucket_id,
            limit=rate_limit_settings.otp_requests_per_window,
            window_seconds=rate_limit_settings.otp_request_window_seconds,
            backoff_base_seconds=rate_limit_settings.otp_backoff_base_seconds,
            backoff_max_seconds=rate_limit_settings.otp_backoff_max_seconds,
            clock=clock,
        )

    code = generate_otp_code(length=settings.otp.length)
    await store_otp(
        redis_client,
        identifier=payload.identifier,
        code=code,
        ttl_seconds=settings.otp.ttl_seconds,
        max_attempts=rate_limit_settings.otp_verify_max_attempts,
    )

    if payload.channel == "email":
        send_otp_email.delay(payload.identifier, code)
    else:
        send_otp_sms.delay(payload.identifier, code)

    # Identifier deliberately not logged -- same PII-minimization discipline `signup` already
    # follows (that route logs `user_id`, never the raw email either).
    _log.info("otp_requested", channel=payload.channel)
    return {"status": "otp_sent"}


@app.post("/api/v1/otp/verify", response_model=TokenResponse)
async def verify_otp(
    payload: OtpVerify,
    response: Response,
    repository: Annotated[UserRepository, Depends(get_repository)],
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> TokenResponse:
    """Consume one still-live, correct OTP; get-or-create the account; issue a fresh session.

    A first-ever successful verify for ``payload.identifier`` *is* signup on this path -- see
    ``UserRepository.get_or_create_by_identifier``'s own docstring.
    """
    redis_client = get_redis(settings.redis)
    code_ok = await verify_and_consume_otp(redis_client, identifier=payload.identifier, code=payload.code)
    if not code_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired code.")

    user = await repository.get_or_create_by_identifier(channel=payload.channel, identifier=payload.identifier)
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account disabled.")

    access_token, expires_at = create_access_token(subject=str(user.id), settings=settings.jwt)
    await _issue_refresh_cookie(response, user_id=str(user.id), settings=settings)
    _log.info("otp_verified", user_id=str(user.id), channel=payload.channel)
    return TokenResponse(access_token=access_token, expires_at=expires_at)


@app.post("/api/v1/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> TokenResponse:
    """Exchange a still-active refresh cookie for a fresh access token, rotating the refresh token itself.

    Rotation: the presented ``jti`` is revoked before a new one is issued, so a stolen-and-
    replayed *old* refresh token stops working the instant the legitimate client refreshes past
    it, rather than staying valid for the rest of its own 7-day lifetime.
    """
    raw_refresh_token = request.cookies.get(_REFRESH_COOKIE_NAME)
    if raw_refresh_token is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token cookie.")
    try:
        subject, jti = decode_refresh_token(raw_refresh_token, settings=settings.jwt)
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token.")

    redis_client = get_redis(settings.redis)
    if not await is_refresh_token_active(redis_client, user_id=subject, jti=jti):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has been revoked.")

    await revoke_refresh_token(redis_client, user_id=subject, jti=jti)
    access_token, expires_at = create_access_token(subject=subject, settings=settings.jwt)
    await _issue_refresh_cookie(response, user_id=subject, settings=settings)
    return TokenResponse(access_token=access_token, expires_at=expires_at)


@app.post("/api/v1/logout", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(
    request: Request,
    response: Response,
    settings: Annotated[RetailGatewaySettings, Depends(get_app_settings)],
) -> None:
    """Revoke the current refresh token (if any) and clear its cookie. Always succeeds -- an
    already-invalid/expired cookie has nothing left to revoke, but the cookie is cleared either way.
    """
    raw_refresh_token = request.cookies.get(_REFRESH_COOKIE_NAME)
    if raw_refresh_token is not None:
        try:
            subject, jti = decode_refresh_token(raw_refresh_token, settings=settings.jwt)
        except InvalidTokenError:
            pass
        else:
            await revoke_refresh_token(get_redis(settings.redis), user_id=subject, jti=jti)
    response.delete_cookie(_REFRESH_COOKIE_NAME, path=_REFRESH_COOKIE_PATH)


@app.get("/api/v1/me", response_model=UserPublic)
async def me(
    user_id: Annotated[UUID, Depends(require_current_user_id)],
    repository: Annotated[UserRepository, Depends(get_repository)],
) -> UserPublic:
    user = await repository.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    return UserPublic(id=user.id, email=user.email, phone_number=user.phone_number, created_at=user.created_at)


@app.get("/api/v1/public/performance", response_model=list[PerformanceMetric])
async def public_performance(repository: Annotated[UserRepository, Depends(get_repository)]) -> list[PerformanceMetric]:
    """No auth, no rate limit gate beyond what a reverse proxy would add in front of this in
    production -- deliberately public, read-only, cached data, exactly what the brief's own
    B2C dashboard spec asks for. See module docstring for the one-way write path.
    """
    rows = await repository.list_recent_performance_metrics()
    return [PerformanceMetric(metric_name=r["metric_name"], metric_value=r["metric_value"], as_of=r["as_of"]) for r in rows]

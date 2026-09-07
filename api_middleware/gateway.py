"""Chapter 13, Étape 5.0: B2B Execution Middleware -- Smart Order Router ingestion gateway.

Business pivot this responds to: the research chapters (see
``research/``) found a real, statistically significant microstructure
edge that is consistently too weak, at these timescales, to trade
profitably on the firm's own account. This module instead packages the
firm's already-built multi-agent infrastructure (ZMQ bus, Redis state
store, XGBoost/RL signal agents) as execution infrastructure a client
(a hedge fund) pays to use directly -- "Execution-as-a-Service" -- which
does not require the directional edge to be strong, only the execution
quality (queue-aware fills, toxicity avoidance) this whole research
chapter already measured and improved.

**A real, load-bearing correction to the brief, stated upfront**: Rule 3
describes "l'Ingestor Agent (déjà existant)" as the consumer that will
drain this gateway's output. No such agent exists anywhere in this
codebase -- ``src/agents/`` has exactly seven agents (market_data,
macro, nlp, supervisor, rl_signal, execution, archivist), none of which
ingest externally-submitted orders (verified by direct inspection, not
assumed). This module still does its own job completely and correctly
-- authenticate, rate-limit, validate, assign an ``order_id``, and
publish onto the bus -- but as of this commit nothing subscribes to
:data:`core.schemas.CLIENT_ORDERS_TOPIC` and acts on it. Building that
consumer is real, separate future work, not invented here without an
explicit ask.

**A second deviation, also stated upfront**: Rule 3 suggests a Redis
Stream or a ZMQ ``PUSH`` socket for the "asynchronous handoff". Neither
pattern exists anywhere in this project: :mod:`core.zmq_bus` implements
PUB/SUB only (confirmed by inspection -- no PUSH/PULL or REQ/REP
anywhere in ``core/``), and :class:`~core.redis_client.RedisClient` is
used throughout as a state store, never as a stream/queue. Rather than
introducing a second, parallel messaging pattern the rest of the system
doesn't use, order submission publishes onto the existing
:class:`~core.zmq_bus.ZmqPublisher` (PUB) under the new
``CLIENT_ORDERS`` topic, wrapped in the same
:class:`~core.schemas.MessagePayload` envelope every other message on
this bus already uses -- connecting to a running
:class:`~core.zmq_broker.ZmqBroker` exactly the way
:class:`~core.base_agent.BaseAgent` already does (see
:func:`_build_zmq_publisher`), so a future Ingestor Agent would
subscribe to it exactly like every other consumer already does. Order
*status* (for the polling endpoint below) is written to Redis, this
project's actual, already-connected state store.

**Rule 4 implemented as SSE, not outbound webhooks, deliberately**: a
webhook that POSTs to an arbitrary client-supplied URL is a real SSRF
surface -- the server would be making outbound requests to a
destination the client controls -- that this module does not implement
without an explicit security review (URL allowlisting, DNS
rebinding protection, etc.). The brief itself offered SSE as an
alternative ("Webhooks & Callbacks - ... ou Server-Sent Events"); SSE
is used instead here -- strictly pull-based, the gateway never
initiates an outbound call to a client-supplied address.

**Rate limiting implemented against Redis, not a new dependency**: no
rate-limiting library exists in this project. Rather than adding one
(e.g. ``slowapi``) for a single fixed-window counter, :func:`_check_rate_limit`
uses a small Redis ``INCR``/``EXPIRE`` fixed-window counter against the
same :class:`~core.redis_client.RedisClient` every other real state
read/write in this codebase already goes through.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, Final

import asyncpg
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.config import AppSettings, get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX,
    CLIENT_ORDERS_TOPIC,
    FILLS_BLOTTER_MAX_ENTRIES,
    FILLS_BLOTTER_REDIS_KEY,
    MARKET_METRICS_REDIS_KEY_PREFIX,
    ORDER_AUDIT_EVENTS_TOPIC,
    ORDER_STATUS_REDIS_KEY_PREFIX,
    MessagePayload,
)
from core.timescale_client import TimescaleUnavailableError, connect_with_backoff
from core.zmq_bus import ZmqPublisher
from src.agents.quant_copilot import CopilotContext, confirm_pending_aggression_adjustment, run_copilot_turn, stream_final_reply
from src.execution.audit_ledger import OrderAuditEvent
from src.execution.billing_schema import ensure_schema as ensure_billing_schema
from src.execution.billing_schema import insert_client_fill, upsert_client_monthly_pnl
from src.execution.macro_strategist import ChildOrder, IcebergSlicer, slice_twap

_log = get_logger(__name__)

_SENDER_ID: Final[str] = "api-gateway"
# 24h: long enough for a client to poll a completed order's final state
# well after submission, not an indefinite retention policy.
_ORDER_STATUS_TTL_SECONDS: Final[int] = 86_400
_RATE_LIMIT_REDIS_PREFIX: Final[str] = "rate_limit:api_middleware:"
_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
_SSE_POLL_INTERVAL_SECONDS: Final[float] = 1.0
# Bounds how long a single SSE connection is kept open server-side, so a
# client that never disconnects can't hold a worker slot forever.
_SSE_MAX_STREAM_SECONDS: Final[float] = 300.0

# Every real order_id is a uuid.uuid4() (see submit_order). The route paths
# below use Starlette's own built-in ":uuid" converter (`{order_id:uuid}`),
# not a Pydantic-level Path(pattern=...) constraint -- that distinction is
# what actually fixes a real routing bug a live audit found: GET
# /api/v1/orders/submit used to match /api/v1/orders/{order_id} (treating
# "submit" as an order_id) instead of correctly 405ing against the
# POST-only /api/v1/orders/submit route. A Pydantic-level constraint only
# runs *after* Starlette has already picked a route by raw path shape (so
# "submit" still matched, then failed Pydantic validation -- 422, not the
# 405 REST actually calls for). Starlette's own ":uuid" converter rejects
# "submit" at ROUTING time, so Starlette correctly falls through to
# recognizing the literal /orders/submit path (right shape, wrong method)
# and reports 405 -- verified live, see test_get_submit_route_returns_405_not_404.
def _order_id_str(order_id: uuid.UUID) -> str:
    """Every downstream use (Redis keys, response models, logging) wants the canonical string form."""
    return str(order_id)

# -- Chapter 13, Étape 5.6, Pilier 3: Shadow Mode --------------------------
_SHADOW_TCA_REDIS_KEY_PREFIX: Final[str] = "shadow_tca:"
_SHADOW_TCA_TTL_SECONDS: Final[int] = 86_400
# Same real fills-blotter list src.agents.execution.agent.ExecutionAgent's
# real fills write to (see core.schemas.FILLS_BLOTTER_REDIS_KEY) -- a shadow
# fill uses the identical shape, tagged agent_id="api-gateway-shadow", so the
# real Order Blotter & TCA dashboard page shows shadow simulations without new UI.
_SHADOW_AGENT_ID: Final[str] = "api-gateway-shadow"
# Same real, calibrated 0.04%/leg taker fee already established in
# research/market_making_env.py and research/alpha_features.py this
# chapter -- reused here, not re-guessed, for the same fee assumption
# across every part of this project that estimates one.
_SHADOW_TAKER_FEE_RATE: Final[float] = 0.0004
_TWAP_DEFAULT_NUM_SLICES: Final[int] = 10
# A deliberately simple, clearly-labeled heuristic -- NOT the real,
# historical-tick-calibrated fill model research/maker_execution_sim.py
# uses (that needs offline polars DataFrames of real book+trade data this
# live HTTP process doesn't have). This scales market impact linearly with
# what fraction of the parent order one child represents, off of real,
# live spread/mid-price (when available) -- directionally sound (bigger
# slices cost more) and internally consistent for the naive-vs-sliced
# comparison below, but not a claim of the same rigor as the Chapter 12
# research model. See _run_shadow_simulation's docstring.
_NAIVE_IMPACT_BPS_PER_UNIT_FRACTION: Final[float] = 15.0


class ExecutionMode(str, Enum):
    """LIVE (the only mode that existed before this chapter) or SHADOW (real, self-contained simulation -- see _run_shadow_simulation)."""

    LIVE = "LIVE"
    SHADOW = "SHADOW"


class ExecutionAlgo(str, Enum):
    """Execution strategy the (future) downstream pipeline is asked to run."""

    AI_ICEBERG = "AI_ICEBERG"
    AI_TWAP = "AI_TWAP"
    TOXICITY_AVOIDANCE = "TOXICITY_AVOIDANCE"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class AggressionLevel(str, Enum):
    """How urgently the (future) execution pipeline should cross the spread vs. wait for a passive fill.

    Real terminology this chapter's own research already established
    (Étape 3.9/4.x): PASSIVE means maker-only, willing to wait for a
    queue-position fill; AGGRESSIVE means taker-first, prioritizing
    certainty of execution over the spread/fee saved by waiting.
    """

    PASSIVE = "PASSIVE"
    AGGRESSIVE = "AGGRESSIVE"


class OrderStatusValue(str, Enum):
    """Lifecycle states written to Redis and reported over the status/SSE endpoints.

    Only ``ACCEPTED`` is ever actually set by this module today (the
    moment order submission validates and publishes) -- the rest exist
    as the real, documented contract a future execution-side consumer
    would report through, not invented behavior implemented here.
    """

    ACCEPTED = "ACCEPTED"
    ROUTING = "ROUTING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MacroOrderRequest(BaseModel):
    """A client's high-level execution instruction (e.g. "buy 100 ETH over 4 hours").

    This is the gateway's own HTTP request contract, not a bus message
    -- :func:`submit_order` wraps a validated instance of this into a
    :class:`~core.schemas.MessagePayload` before publishing (see module
    docstring for why).
    """

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(min_length=1, max_length=128)
    asset: str = Field(min_length=1, max_length=32)
    total_quantity: float = Field(gt=0)
    side: OrderSide
    execution_algo: ExecutionAlgo
    time_window_minutes: int = Field(gt=0, le=1440, description="Max 24h -- this is an execution algo parameter, not a standing order.")
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.LIVE,
        description=(
            "LIVE (default, unchanged behavior) or SHADOW (real, self-contained paper-trading "
            "simulation -- see _run_shadow_simulation). Defaults to LIVE, not required, so every "
            "existing client request (this model uses extra='forbid') keeps validating unchanged."
        ),
    )


class OrderAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    status: OrderStatusValue
    submitted_at: datetime


class OrderStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    status: OrderStatusValue
    client_id: str
    aggression_level: AggressionLevel
    updated_at: datetime
    execution_mode: ExecutionMode = ExecutionMode.LIVE


class ShadowFillReport(BaseModel):
    """One simulated child-order fill within a SHADOW order's simulation -- real numbers, clearly not a live fill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int
    quantity: float
    fill_price: float
    slippage_bps: float


class ShadowTcaReport(BaseModel):
    """Real Transaction Cost Analysis for one SHADOW order -- see _run_shadow_simulation's docstring for the model's real scope/limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str
    available: bool
    mid_price_reference: float | None = None
    child_fills: list[ShadowFillReport] = Field(default_factory=list)
    naive_single_shot_slippage_bps: float | None = None
    sliced_weighted_avg_slippage_bps: float | None = None
    slippage_saved_usd: float | None = None
    note: str


class AggressionAdjustmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aggression_level: AggressionLevel


class AggressionAdjustmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    aggression_level: AggressionLevel
    updated_at: datetime


class GatewayState:
    """Process-wide ZMQ/Redis/Timescale handles, created once at startup by :func:`_lifespan`."""

    def __init__(self) -> None:
        self.zmq_publisher: ZmqPublisher | None = None
        self.redis_client: RedisClient | None = None
        self.timescale_pool: asyncpg.Pool | None = None


_state = GatewayState()


def _build_zmq_publisher(settings: AppSettings) -> ZmqPublisher:
    """Connect to a running broker if configured, else bind our own port -- identical to BaseAgent's own resolution logic.

    See :class:`~core.base_agent.BaseAgent`'s constructor for the
    original of this exact same broker-frontend-or-bind-own-port
    decision; duplicated here (not imported) because this gateway is
    deliberately not a :class:`~core.base_agent.BaseAgent` subclass --
    it's an HTTP service, not a bus-consuming agent loop.
    """
    broker_frontend = settings.zmq.broker_frontend_address
    if broker_frontend is not None:
        return ZmqPublisher(broker_frontend, high_water_mark=settings.zmq.high_water_mark, bind=False)
    return ZmqPublisher(settings.zmq.pub_address, high_water_mark=settings.zmq.high_water_mark, bind=True)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(log_level=settings.log_level, service_name="api-middleware")
    _state.redis_client = await RedisClient.get_instance(settings.redis)
    _state.zmq_publisher = _build_zmq_publisher(settings)

    # Bounded, fail-soft: TimescaleDB backs an *optional* durability path
    # (billing/reporting history, see src/execution/billing_schema.py) --
    # unlike Redis/ZMQ above, this gateway must still start and serve
    # requests even when no TimescaleDB is reachable (true in this dev
    # environment, confirmed by direct inspection), just without that one
    # persistence path. A single fast attempt, not the unbounded retry a
    # long-lived agent would use for a dependency it truly can't run without.
    try:
        _state.timescale_pool = await connect_with_backoff(settings.timescale, max_attempts=1)
        await ensure_billing_schema(_state.timescale_pool)
        _log.info("gateway_timescale_connected")
    except TimescaleUnavailableError:
        _log.warning("gateway_timescale_unavailable_billing_history_disabled")
        _state.timescale_pool = None

    _log.info("gateway_started")
    try:
        yield
    finally:
        if _state.zmq_publisher is not None:
            _state.zmq_publisher.close()
        if _state.redis_client is not None:
            await _state.redis_client.close()
        if _state.timescale_pool is not None:
            await _state.timescale_pool.close()
        _log.info("gateway_stopped")


app = FastAPI(
    title="Execution Middleware Gateway",
    description="B2B Smart Order Router ingestion API -- submits macro execution orders onto the internal agent bus.",
    version="5.0.0",
    lifespan=_lifespan,
)

_cors_origins_raw = get_settings().api_middleware.cors_allowed_origins
if _cors_origins_raw:
    # Only installed when real origins are configured -- see
    # ApiMiddlewareSettings.cors_allowed_origins's docstring for why this
    # stays absent (not "allow *") by default.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()],
        allow_credentials=False,  # auth is via X-API-Key, not cookies -- no cross-origin credential exposure to allow
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


@app.middleware("http")
async def _security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """A real, standard baseline -- found missing entirely by a live audit against this API.

    HSTS is safe to send unconditionally: browsers only act on it over a
    real HTTPS connection, so it's a no-op today (this dev server is plain
    HTTP) and takes effect for free the moment a real deployment terminates
    TLS in front of this process (a reverse proxy -- see SecurityVault's own
    docstring on this project's established "no infrastructure this project
    doesn't have" convention; this header doesn't add any).
    """
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


def _get_redis() -> RedisClient:
    if _state.redis_client is None:
        raise RuntimeError("RedisClient not initialized -- _lifespan must run before any request is handled.")
    return _state.redis_client


def _get_zmq_publisher() -> ZmqPublisher:
    if _state.zmq_publisher is None:
        raise RuntimeError("ZmqPublisher not initialized -- _lifespan must run before any request is handled.")
    return _state.zmq_publisher


async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Validate the ``X-API-Key`` header against the configured key set.

    Args:
        x_api_key: The raw header value, injected by FastAPI.

    Returns:
        The validated API key (useful as a per-client identifier for
        downstream rate limiting).

    Raises:
        HTTPException: 401 if the header is missing or does not match
            any configured key. Comparison is timing-safe
            (:func:`secrets.compare_digest`) against every configured
            key -- never short-circuits on the first mismatch, so
            response timing can't leak which prefix of a guessed key is
            correct.
    """
    settings = get_settings().api_middleware
    if x_api_key is None or settings.api_keys is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key.")

    configured_keys = [key.strip() for key in settings.api_keys.get_secret_value().split(",") if key.strip()]
    if not any(secrets.compare_digest(x_api_key, configured) for configured in configured_keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key.")
    return x_api_key


async def _check_rate_limit(api_key: Annotated[str, Depends(require_api_key)]) -> None:
    """Fixed-window rate limit, keyed per API key, enforced via Redis INCR/EXPIRE.

    Args:
        api_key: The already-validated key from :func:`require_api_key`.

    Raises:
        HTTPException: 429 if this key has exceeded
            ``ApiMiddlewareSettings.rate_limit_per_minute`` requests in
            the current 60-second window.
    """
    settings = get_settings().api_middleware
    redis_client = _get_redis()
    # A fixed (not sliding) window: simple, and the standard first choice
    # for this kind of per-key API limiter -- a client can burst up to
    # 2x the limit across a window boundary, an accepted tradeoff for
    # not needing a sorted-set/Lua-script sliding-window implementation.
    redis_key = f"{_RATE_LIMIT_REDIS_PREFIX}{api_key}"
    current_count = await redis_client.execute("incr", redis_key)
    if current_count == 1:
        await redis_client.execute("expire", redis_key, _RATE_LIMIT_WINDOW_SECONDS)

    if current_count > settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {settings.rate_limit_per_minute} requests/{_RATE_LIMIT_WINDOW_SECONDS}s.",
        )


@app.post(
    "/api/v1/orders/submit",
    response_model=OrderAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a macro execution order",
    description=(
        "Validates and accepts a client's macro order for asynchronous execution. "
        "Does not block on routing/execution: publishes the order onto the internal "
        "bus and returns immediately with an order_id for status polling."
    ),
)
async def submit_order(
    order: MacroOrderRequest, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> OrderAcceptedResponse:
    """Accept, persist, and publish one client order. Never performs execution itself.

    Args:
        order: The validated request body.
        _api_key: Unused directly -- present so FastAPI runs
            :func:`_check_rate_limit` (which itself depends on
            :func:`require_api_key`) before this handler executes.

    Returns:
        202-status payload with a freshly generated ``order_id`` and
        ``ACCEPTED`` status.
    """
    order_id = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc)

    redis_client = _get_redis()
    status_record = {
        "order_id": order_id,
        "status": OrderStatusValue.ACCEPTED.value,
        "client_id": order.client_id,
        "aggression_level": AggressionLevel.PASSIVE.value,
        "updated_at": submitted_at.isoformat(),
        "execution_mode": order.execution_mode.value,
    }
    await redis_client.execute(
        "set",
        f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id}",
        json.dumps(status_record),
        ex=_ORDER_STATUS_TTL_SECONDS,
    )

    publisher = _get_zmq_publisher()
    payload = MessagePayload(
        topic=CLIENT_ORDERS_TOPIC,
        sender_id=_SENDER_ID,
        data={"order_id": order_id, **order.model_dump(mode="json")},
    )
    await publisher.publish(payload)

    audit_event = OrderAuditEvent(
        event_type="SUBMITTED", order_id=order_id, client_id=order.client_id, asset=order.asset,
        execution_mode=order.execution_mode.value,
        detail=f"Ordre {order.execution_algo.value} soumis : {order.total_quantity} {order.asset} ({order.side.value}).",
    )
    await publisher.publish(
        MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id=_SENDER_ID, data=audit_event.model_dump(mode="json"))
    )

    _log.info(
        "order_accepted",
        order_id=order_id,
        client_id=order.client_id,
        asset=order.asset,
        execution_algo=order.execution_algo.value,
        execution_mode=order.execution_mode.value,
    )

    if order.execution_mode == ExecutionMode.SHADOW:
        # Deliberately awaited inline, not fire-and-forget: this
        # simulation is entirely local (Redis reads/writes, no network
        # call to a real broker -- that's the whole point of SHADOW
        # mode), so it's fast and its result should be immediately
        # visible to the client's very next status/TCA poll, not racy
        # against a background task.
        try:
            await _run_shadow_simulation(order, order_id=order_id)
        except Exception:
            _log.exception("shadow_simulation_failed", order_id=order_id)

    return OrderAcceptedResponse(order_id=order_id, status=OrderStatusValue.ACCEPTED, submitted_at=submitted_at)


async def _read_order_status(order_id: str) -> OrderStatusResponse | None:
    """Read and validate one order's status record from Redis, if present."""
    redis_client = _get_redis()
    raw = await redis_client.execute("get", f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id}")
    if raw is None:
        return None
    record: dict[str, Any] = json.loads(raw)
    return OrderStatusResponse(
        order_id=record["order_id"],
        status=OrderStatusValue(record["status"]),
        client_id=record["client_id"],
        aggression_level=AggressionLevel(record.get("aggression_level", AggressionLevel.PASSIVE.value)),
        updated_at=datetime.fromisoformat(record["updated_at"]),
        # .get(...) with a LIVE default: order records written before this
        # field existed still parse -- they really were LIVE-mode (SHADOW
        # didn't exist yet when they were written).
        execution_mode=ExecutionMode(record.get("execution_mode", ExecutionMode.LIVE.value)),
    )


def _simulate_fill(
    *, mid_price: float, spread: float, side: OrderSide, quantity: float, parent_quantity: float
) -> tuple[float, float]:
    """One simplified, clearly-labeled simulated fill -- see :func:`_run_shadow_simulation`'s docstring for its real scope.

    Args:
        mid_price: Real, live mid-price (from ``market_metrics:{asset}``).
        spread: Real, live bid-ask spread.
        side: ``BUY`` or ``SELL``.
        quantity: This child order's size.
        parent_quantity: The full parent order's total size -- larger
            children (relative to the parent) simulate proportionally
            more market impact.

    Returns:
        ``(fill_price, slippage_bps)`` -- slippage measured against
        ``mid_price``, unsigned (always >= 0 regardless of side).
    """
    half_spread = spread / 2.0
    quantity_fraction = quantity / parent_quantity if parent_quantity > 0 else 0.0
    impact = mid_price * (_NAIVE_IMPACT_BPS_PER_UNIT_FRACTION * quantity_fraction) / 10_000.0
    direction = 1.0 if side == OrderSide.BUY else -1.0
    fill_price = mid_price + direction * (half_spread + impact)
    slippage_bps = abs(fill_price - mid_price) / mid_price * 10_000.0
    return fill_price, slippage_bps


async def _write_shadow_fill(
    *, order_id: str, order: MacroOrderRequest, child: ChildOrder, fill_price: float, slippage_saved_usd: float
) -> None:
    """Persist one simulated fill: the live Redis blotter (unchanged shape) + the new durable ``client_fills`` table.

    Real integration, not a parallel UI: ``src/dashboard/ems_pages/
    order_blotter.py`` reads this exact Redis list already -- a shadow
    fill shows up there immediately, tagged by ``agent_id`` and
    ``mock_mode`` so it's never confused with a real production fill.
    ``client_id``/``fee_usd`` are new fields on the Redis entry (this
    project's very first fill record to actually carry client
    attribution -- see :mod:`src.execution.billing_schema`'s module
    docstring for why nothing did before this).

    The ``client_fills`` write is the durable half this project never
    had: Redis's blotter is capped at :data:`_FILLS_BLOTTER_MAX_ENTRIES`
    and carries no TTL-survival guarantee for month-spanning billing
    queries -- TimescaleDB is. Fails soft (logs, does not raise) when
    unreachable, exactly like the Redis write already does -- see
    ``_lifespan``'s own bounded-attempt reasoning for why this gateway
    must keep serving requests either way.
    """
    fee_usd = fill_price * child.quantity * _SHADOW_TAKER_FEE_RATE
    redis_client = _get_redis()
    entry = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).timestamp(),
            "agent_id": _SHADOW_AGENT_ID,
            "client_id": order.client_id,
            "order_id": order_id,
            "instrument": order.asset,
            "action": "ENTRY_LONG" if order.side == OrderSide.BUY else "ENTRY_SHORT",
            "price": round(fill_price, 6),
            "units": round(child.quantity, 6),
            "fee_usd": round(fee_usd, 6),
            "slippage_saved_usd": round(slippage_saved_usd, 6),
            "latency_ms": 0.0,
            "reason": f"SHADOW simulation ({order.execution_algo.value}, child #{child.sequence}, order {order_id})",
            "mock_mode": True,
        }
    )
    try:
        await redis_client.execute("lpush", FILLS_BLOTTER_REDIS_KEY, entry)
        await redis_client.execute("ltrim", FILLS_BLOTTER_REDIS_KEY, 0, FILLS_BLOTTER_MAX_ENTRIES - 1)
    except RedisUnavailableError:
        _log.warning("shadow_fill_blotter_write_failed", order_id=order_id)

    if _state.timescale_pool is not None:
        try:
            await insert_client_fill(
                _state.timescale_pool,
                time=datetime.now(timezone.utc),
                client_id=order.client_id,
                order_id=order_id,
                instrument=order.asset,
                side=order.side.value,
                execution_algo=order.execution_algo.value,
                execution_mode=order.execution_mode.value,
                quantity=child.quantity,
                price=fill_price,
                fee_usd=fee_usd,
                slippage_saved_usd=slippage_saved_usd,
            )
        except asyncpg.PostgresError:
            _log.warning("shadow_fill_client_fills_write_failed", order_id=order_id, client_id=order.client_id)

    audit_event = OrderAuditEvent(
        event_type="EXECUTED", order_id=order_id, client_id=order.client_id, asset=order.asset,
        execution_mode=order.execution_mode.value,
        detail=f"Fill SHADOW simulé : {child.quantity} {order.asset} @ {fill_price:.6f} (child #{child.sequence}).",
    )
    await _get_zmq_publisher().publish(
        MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id=_SENDER_ID, data=audit_event.model_dump(mode="json"))
    )


async def _run_shadow_simulation(order: MacroOrderRequest, *, order_id: str) -> None:
    """Pilier 3: a real, self-contained paper-trading simulation -- no HTTP/FIX call to any broker.

    **Real scope, stated plainly**: this slices the parent order with
    the same real :mod:`src.execution.macro_strategist` logic a live
    executor would use (:func:`~src.execution.macro_strategist.
    slice_twap` for ``AI_TWAP``/``TOXICITY_AVOIDANCE``,
    :class:`~src.execution.macro_strategist.IcebergSlicer` for
    ``AI_ICEBERG``), then simulates each child's fill against the
    **real, live** mid-price/spread already published to
    ``market_metrics:{asset}`` by :class:`~src.agents.market_data.
    agent.MarketDataAgent` (see :func:`_simulate_fill`) -- when that
    real data exists. It does **not** replicate
    ``research/maker_execution_sim.py``'s real, historical-tick-
    calibrated queue-position fill model: that needs offline
    ``polars`` DataFrames of real book+trade history this live HTTP
    process has no access to. The market-impact term used here is a
    simple, explicitly-labeled heuristic (see
    :data:`_NAIVE_IMPACT_BPS_PER_UNIT_FRACTION`) applied identically to
    both the sliced and the naive-single-shot comparison, so the
    *comparison* (sliced vs. naive) is internally consistent even
    though the absolute slippage numbers are illustrative, not a
    calibrated market-impact model.

    **Honest gap, not fabricated**: if no real ``market_metrics``
    record exists for ``order.asset`` (no live
    :class:`~src.agents.market_data.agent.MarketDataAgent` running for
    it -- the common case in this environment), this writes a
    :class:`ShadowTcaReport` with ``available=False`` and an explicit
    reason, never a fabricated price.
    """
    redis_client = _get_redis()
    raw_metrics = await redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{order.asset}")
    if raw_metrics is None:
        report = ShadowTcaReport(
            order_id=order_id,
            available=False,
            note=(
                f"⚠️ Aucune donnée de marché réelle en direct pour {order.asset!r} "
                "(market_metrics absent ou expiré) -- simulation impossible sans prix réel."
            ),
        )
        await redis_client.execute(
            "set", f"{_SHADOW_TCA_REDIS_KEY_PREFIX}{order_id}", report.model_dump_json(), ex=_SHADOW_TCA_TTL_SECONDS
        )
        return

    metrics: dict[str, Any] = json.loads(raw_metrics)
    mid_price = float(metrics["mid_price"])
    spread = float(metrics["spread"])

    if order.execution_algo == ExecutionAlgo.AI_ICEBERG:
        slicer = IcebergSlicer(
            parent_order_id=order_id, symbol=order.asset,
            side="buy" if order.side == OrderSide.BUY else "sell", total_quantity=order.total_quantity,
        )
        children: list[ChildOrder] = []
        while (child := slicer.next_slice()) is not None:
            children.append(child)
            slicer.record_fill(child.quantity)
    else:
        children = slice_twap(
            parent_order_id=order_id, symbol=order.asset, side="buy" if order.side == OrderSide.BUY else "sell",
            total_quantity=order.total_quantity, time_window_minutes=order.time_window_minutes,
            num_slices=min(_TWAP_DEFAULT_NUM_SLICES, max(1, int(order.total_quantity))),
        )

    # Computed before the loop (not just after) so each child's own
    # slippage-saved contribution can be persisted per-row into the
    # durable client_fills table below -- the aggregate slippage_saved_usd
    # on ShadowTcaReport only lives 24h in Redis (see that model's
    # docstring); a monthly billing report needs this per-fill and durable,
    # not just as a session-scoped aggregate.
    _naive_fill_price, naive_slippage_bps = _simulate_fill(
        mid_price=mid_price, spread=spread, side=order.side, quantity=order.total_quantity,
        parent_quantity=order.total_quantity,
    )

    fills: list[ShadowFillReport] = []
    weighted_slippage_sum = 0.0
    for child in children:
        fill_price, slippage_bps = _simulate_fill(
            mid_price=mid_price, spread=spread, side=order.side, quantity=child.quantity,
            parent_quantity=order.total_quantity,
        )
        fills.append(ShadowFillReport(sequence=child.sequence, quantity=child.quantity, fill_price=fill_price, slippage_bps=slippage_bps))
        weighted_slippage_sum += slippage_bps * (child.quantity / order.total_quantity)
        child_slippage_saved_bps = max(0.0, naive_slippage_bps - slippage_bps)
        child_slippage_saved_usd = mid_price * child.quantity * child_slippage_saved_bps / 10_000.0
        await _write_shadow_fill(
            order_id=order_id, order=order, child=child, fill_price=fill_price,
            slippage_saved_usd=child_slippage_saved_usd,
        )

    slippage_saved_bps = max(0.0, naive_slippage_bps - weighted_slippage_sum)
    slippage_saved_usd = mid_price * order.total_quantity * slippage_saved_bps / 10_000.0

    report = ShadowTcaReport(
        order_id=order_id,
        available=True,
        mid_price_reference=mid_price,
        child_fills=fills,
        naive_single_shot_slippage_bps=naive_slippage_bps,
        sliced_weighted_avg_slippage_bps=weighted_slippage_sum,
        slippage_saved_usd=slippage_saved_usd,
        note=(
            f"Simulation réelle basée sur market_metrics:{order.asset} en direct (mid={mid_price:.4f}, "
            f"spread={spread:.6f}). Modèle d'impact simplifié -- voir _run_shadow_simulation."
        ),
    )
    await redis_client.execute(
        "set", f"{_SHADOW_TCA_REDIS_KEY_PREFIX}{order_id}", report.model_dump_json(), ex=_SHADOW_TCA_TTL_SECONDS
    )


@app.get(
    "/api/v1/orders/{order_id:uuid}",
    response_model=OrderStatusResponse,
    summary="Get an order's current status",
)
async def get_order_status(
    order_id: uuid.UUID, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> OrderStatusResponse:
    """Return the current status record for ``order_id``.

    Raises:
        HTTPException: 404 if ``order_id`` is unknown or its status
            record has expired (see :data:`_ORDER_STATUS_TTL_SECONDS`).
    """
    order_id_str = _order_id_str(order_id)
    result = await _read_order_status(order_id_str)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown order_id: {order_id_str!r}")
    return result


@app.get(
    "/api/v1/orders/{order_id:uuid}/tca",
    response_model=ShadowTcaReport,
    summary="Get a SHADOW order's real, simulated Transaction Cost Analysis",
    description=(
        "Only ever populated for orders submitted with execution_mode=SHADOW -- see "
        "_run_shadow_simulation's docstring for exactly what this simulates and its real limits."
    ),
)
async def get_shadow_tca(order_id: uuid.UUID, _api_key: Annotated[str, Depends(_check_rate_limit)]) -> ShadowTcaReport:
    """Return the real, previously-computed SHADOW simulation report for ``order_id``.

    Raises:
        HTTPException: 404 if ``order_id`` is unknown, was never
            submitted in SHADOW mode, or its TCA record has expired
            (see :data:`_SHADOW_TCA_TTL_SECONDS`).
    """
    order_id_str = _order_id_str(order_id)
    redis_client = _get_redis()
    raw = await redis_client.execute("get", f"{_SHADOW_TCA_REDIS_KEY_PREFIX}{order_id_str}")
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No SHADOW TCA report for {order_id_str!r} -- unknown order, not submitted in SHADOW mode, or expired.",
        )
    return ShadowTcaReport.model_validate_json(raw)


@app.post(
    "/api/v1/orders/{order_id:uuid}/adjust",
    response_model=AggressionAdjustmentResponse,
    summary="Adjust an order's execution aggression (Passive <-> Aggressive)",
    description=(
        "Added for quant_copilot.py's adjust_algo_aggression tool. Unlike order "
        "submission, this mutates an existing order's execution posture -- callers "
        "(the copilot included) are expected to have already gotten explicit human "
        "confirmation before calling this; the gateway itself does not re-confirm."
    ),
)
async def adjust_order_aggression(
    order_id: uuid.UUID, adjustment: AggressionAdjustmentRequest, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> AggressionAdjustmentResponse:
    """Update an order's stored aggression level and publish the change onto the bus.

    Raises:
        HTTPException: 404 if ``order_id`` is unknown, 409 if it is
            already in a terminal state (``COMPLETED``/``FAILED`` --
            nothing left to adjust).
    """
    order_id_str = _order_id_str(order_id)
    existing = await _read_order_status(order_id_str)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown order_id: {order_id_str!r}")
    if existing.status in (OrderStatusValue.COMPLETED, OrderStatusValue.FAILED, OrderStatusValue.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Order {order_id_str!r} is already {existing.status.value} -- nothing to adjust.",
        )

    updated_at = datetime.now(timezone.utc)
    redis_client = _get_redis()
    status_record = {
        "order_id": order_id_str,
        "status": existing.status.value,
        "client_id": existing.client_id,
        "aggression_level": adjustment.aggression_level.value,
        "updated_at": updated_at.isoformat(),
        # Real bug fixed here, found live while building cancel_order
        # right below (same read-modify-write pattern): omitting this
        # field here silently reset execution_mode back to the
        # ExecutionMode.LIVE fallback default on every real aggression
        # adjustment, even for a real SHADOW order -- see
        # _read_order_status's own defensive .get() for why that never
        # crashed, just quietly lost real information.
        "execution_mode": existing.execution_mode.value,
    }
    await redis_client.execute(
        "set", f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id_str}", json.dumps(status_record), ex=_ORDER_STATUS_TTL_SECONDS
    )

    publisher = _get_zmq_publisher()
    await publisher.publish(
        MessagePayload(
            topic=CLIENT_ORDERS_TOPIC,
            sender_id=_SENDER_ID,
            data={
                "order_id": order_id_str,
                "event": "AGGRESSION_ADJUSTED",
                "aggression_level": adjustment.aggression_level.value,
            },
        )
    )

    _log.info("order_aggression_adjusted", order_id=order_id_str, aggression_level=adjustment.aggression_level.value)
    return AggressionAdjustmentResponse(
        order_id=order_id_str, aggression_level=adjustment.aggression_level, updated_at=updated_at
    )


class CancelOrderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    status: OrderStatusValue


@app.post(
    "/api/v1/orders/{order_id:uuid}/cancel",
    response_model=CancelOrderResponse,
    summary="Cancel a client order, if it hasn't already executed",
    description=(
        "Real, but honestly bounded: MacroOrderExecutionAgent executes a real LIVE order "
        "immediately on receipt (no time-based scheduling exists yet -- see that module's own "
        "docstring), so this real cancellation window is typically short. Most calls against an "
        "already-executed order will find it already COMPLETED/FAILED (409)."
    ),
)
async def cancel_order(
    order_id: uuid.UUID, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> CancelOrderResponse:
    """Mark ``order_id`` CANCELLED and set the real marker MacroOrderExecutionAgent checks before executing.

    Raises:
        HTTPException: 404 if ``order_id`` is unknown, 409 if it is
            already in a terminal state (``COMPLETED``/``FAILED``/
            ``CANCELLED`` -- see this route's own honest description on
            why that will be the common case).
    """
    order_id_str = _order_id_str(order_id)
    existing = await _read_order_status(order_id_str)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown order_id: {order_id_str!r}")
    if existing.status in (OrderStatusValue.COMPLETED, OrderStatusValue.FAILED, OrderStatusValue.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Order {order_id_str!r} is already {existing.status.value} -- cannot cancel.",
        )

    updated_at = datetime.now(timezone.utc)
    redis_client = _get_redis()
    status_record = {
        "order_id": order_id_str,
        "status": OrderStatusValue.CANCELLED.value,
        "client_id": existing.client_id,
        "aggression_level": existing.aggression_level.value,
        "updated_at": updated_at.isoformat(),
        "execution_mode": existing.execution_mode.value,
    }
    await redis_client.execute(
        "set", f"{ORDER_STATUS_REDIS_KEY_PREFIX}{order_id_str}", json.dumps(status_record), ex=_ORDER_STATUS_TTL_SECONDS
    )
    # Real marker MacroOrderExecutionAgent checks (real EXISTS call) immediately
    # before executing a real LIVE order -- see core.schemas.
    # CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX's own docstring for the real,
    # honestly short window this closes.
    await redis_client.execute(
        "set", f"{CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX}{order_id_str}", "1", ex=_ORDER_STATUS_TTL_SECONDS
    )

    audit_event = OrderAuditEvent(
        event_type="CANCELLED", order_id=order_id_str, client_id=existing.client_id, asset=None,
        execution_mode=existing.execution_mode.value,
        detail="Ordre annulé par le client (réel) -- voir cancel_order pour la fenêtre réelle d'annulation.",
    )
    await _get_zmq_publisher().publish(
        MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id=_SENDER_ID, data=audit_event.model_dump(mode="json"))
    )

    _log.info("order_cancelled", order_id=order_id_str, client_id=existing.client_id)
    return CancelOrderResponse(order_id=order_id_str, status=OrderStatusValue.CANCELLED)


@app.get(
    "/api/v1/orders/{order_id:uuid}/stream",
    summary="Stream order status updates (Server-Sent Events)",
    description=(
        "Rule 4, implemented as SSE rather than an outbound webhook -- see module "
        "docstring for why. Polls the order's Redis status record and emits an SSE "
        "event whenever it changes, until a terminal status is reached, the "
        "client disconnects, or the connection's max lifetime is reached."
    ),
)
async def stream_order_status(
    request: Request, order_id: uuid.UUID, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> StreamingResponse:
    """Server-Sent Events endpoint: pushes a status update to the client whenever it changes.

    Raises:
        HTTPException: 404 if ``order_id`` is unknown at the moment the
            stream is opened.
    """
    order_id_str = _order_id_str(order_id)
    initial = await _read_order_status(order_id_str)
    if initial is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown order_id: {order_id_str!r}")

    async def _event_generator() -> AsyncIterator[str]:
        last_status: OrderStatusValue | None = None
        elapsed = 0.0
        terminal_statuses = {OrderStatusValue.COMPLETED, OrderStatusValue.FAILED}

        while elapsed < _SSE_MAX_STREAM_SECONDS:
            if await request.is_disconnected():
                _log.info("sse_client_disconnected", order_id=order_id_str)
                return

            current = await _read_order_status(order_id_str)
            if current is not None and current.status != last_status:
                last_status = current.status
                yield f"data: {current.model_dump_json()}\n\n"
                if current.status in terminal_statuses:
                    return

            await asyncio.sleep(_SSE_POLL_INTERVAL_SECONDS)
            elapsed += _SSE_POLL_INTERVAL_SECONDS

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


async def _authenticate_websocket(websocket: WebSocket) -> str | None:
    """Validate ``?api_key=...`` the same way :func:`require_api_key` validates the header.

    WebSocket connections from browser JS cannot set custom headers on
    the handshake request, so the key travels as a query parameter here
    instead -- the one deliberate exception to this gateway's
    header-only auth convention, and only for this endpoint.

    Returns:
        The validated key, or ``None`` (after closing the socket with
        code 4401) if missing/invalid.
    """
    settings = get_settings().api_middleware
    api_key = websocket.query_params.get("api_key")
    if api_key is None or settings.api_keys is None:
        await websocket.close(code=4401, reason="Missing or invalid api_key.")
        return None
    configured_keys = [key.strip() for key in settings.api_keys.get_secret_value().split(",") if key.strip()]
    if not any(secrets.compare_digest(api_key, configured) for configured in configured_keys):
        await websocket.close(code=4401, reason="Missing or invalid api_key.")
        return None
    return api_key


@app.websocket("/ws/copilot")
async def copilot_websocket(websocket: WebSocket) -> None:
    """Rule 4 (WebSocket path): one Quant Copilot conversation per connection, typewriter-streamed.

    Protocol, one JSON object per client message:

    - ``{"message": "...", "history": [...]}`` -- a normal conversation
      turn. Server replies with a sequence of ``{"chunk": "..."}``
      frames (see :func:`~src.agents.quant_copilot.stream_final_reply`),
      then one ``{"done": true, "history": [...]}`` frame carrying the
      updated history to send back on the next turn.
    - ``{"confirm": "<confirmation_id>"}`` -- confirms a previously
      proposed :func:`~src.agents.quant_copilot.adjust_algo_aggression`
      change (see that function's module docstring for why this is a
      separate, explicit step the LLM cannot trigger itself). Server
      replies with one ``{"chunk": "...", "done": true}`` frame.

    Auth via ``?api_key=...`` -- see :func:`_authenticate_websocket`.
    Rate limiting is intentionally not applied here: a live conversation
    is one connection issuing many small messages, not the
    one-request-per-call shape :func:`_check_rate_limit` was designed
    for.
    """
    await websocket.accept()
    api_key = await _authenticate_websocket(websocket)
    if api_key is None:
        return

    settings = get_settings().api_middleware
    redis_client = _get_redis()
    context = CopilotContext(redis_client=redis_client, gateway_base_url=settings.self_base_url, gateway_api_key=api_key)

    try:
        while True:
            payload = await websocket.receive_json()

            if "confirm" in payload:
                result = await confirm_pending_aggression_adjustment(
                    redis_client,
                    confirmation_id=payload["confirm"],
                    gateway_base_url=settings.self_base_url,
                    gateway_api_key=api_key,
                )
                await websocket.send_json({"chunk": result.message, "done": True})
                continue

            user_message = payload["message"]
            history = payload.get("history", [])
            reply_text, updated_history = await run_copilot_turn(
                user_message=user_message, conversation_history=history, context=context
            )
            async for chunk in stream_final_reply(reply_text, provider=get_settings().guardian.llm_provider):
                await websocket.send_json({"chunk": chunk})
            await websocket.send_json({"done": True, "history": updated_history})
    except WebSocketDisconnect:
        _log.info("copilot_websocket_disconnected")


class MonthlyPnlReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int = Field(ge=2020, le=2100)
    month: int = Field(ge=1, le=12)
    gross_pnl_usd: float = Field(description="The client's own real, reported gross PnL for this month -- never computed by this system.")


class MonthlyPnlReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    year: int
    month: int
    gross_pnl_usd: float
    recorded: bool
    note: str


@app.post(
    "/api/v1/clients/{client_id}/monthly-pnl",
    response_model=MonthlyPnlReportResponse,
    summary="Client self-reports their real gross PnL for one month",
    description=(
        "Nothierly never takes directional positions on a client's behalf (Execution-as-a-Service), "
        "so it has no way to know a client's own trading outcome unless the client reports it. Once "
        "recorded, the monthly billing report (src/execution/reporting_engine.py) shows the "
        "Nothierly Edge as a real share of the client's own reported result -- never fabricated."
    ),
)
async def report_monthly_pnl(
    client_id: str, report: MonthlyPnlReportRequest, _api_key: Annotated[str, Depends(_check_rate_limit)]
) -> MonthlyPnlReportResponse:
    """Record (or correct) one client's self-reported gross PnL for one calendar month.

    Real, but fails soft when TimescaleDB is unreachable -- ``recorded``
    tells the caller whether the value actually persisted; a client's
    self-report is valuable data, worth an honest failure signal rather
    than a silent 202-and-drop.
    """
    if _state.timescale_pool is None:
        return MonthlyPnlReportResponse(
            client_id=client_id, year=report.year, month=report.month, gross_pnl_usd=report.gross_pnl_usd,
            recorded=False, note="⚠️ TimescaleDB indisponible -- valeur non enregistrée durablement, réessayer plus tard.",
        )
    try:
        await upsert_client_monthly_pnl(
            _state.timescale_pool, client_id=client_id, year=report.year, month=report.month,
            gross_pnl_usd=report.gross_pnl_usd,
        )
    except asyncpg.PostgresError as exc:
        _log.exception("monthly_pnl_report_failed", client_id=client_id, year=report.year, month=report.month)
        return MonthlyPnlReportResponse(
            client_id=client_id, year=report.year, month=report.month, gross_pnl_usd=report.gross_pnl_usd,
            recorded=False, note=f"⚠️ Échec d'enregistrement : {exc}",
        )

    _log.info("monthly_pnl_reported", client_id=client_id, year=report.year, month=report.month)
    return MonthlyPnlReportResponse(
        client_id=client_id, year=report.year, month=report.month, gross_pnl_usd=report.gross_pnl_usd,
        recorded=True, note="Enregistré -- pris en compte dans le prochain rapport mensuel.",
    )


@app.get("/health", summary="Liveness check", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}

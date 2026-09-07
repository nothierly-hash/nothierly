"""Wire-format contracts for messages exchanged on the inter-agent bus.

Every message published on the ZeroMQ bus (see :mod:`core.zmq_bus`) is a
validated :class:`MessagePayload`. Serializing/deserializing exclusively
through this model guarantees that no malformed, mistyped, or
schema-violating message can ever transit between agents: a publisher
cannot construct an invalid payload, and a subscriber immediately rejects
anything that doesn't conform on receipt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_TOPIC_PATTERN = r"^[A-Z][A-Z0-9_]*$"

# Top-of-book depth every OrderBookSnapshot is truncated to. Kept as a
# module constant so MarketMetrics' OBI docstring and any future consumer
# can reference the same number instead of a magic literal.
ORDER_BOOK_TOP_N: int = 5


class MalformedMessageError(ValueError):
    """Raised when raw bytes received from the bus fail schema validation.

    Wraps the underlying ``pydantic.ValidationError`` so callers can catch
    a single, stable exception type regardless of which field failed
    validation.
    """


def _require_timezone_aware(value: datetime) -> datetime:
    """Shared validator body: reject naive datetimes across every schema.

    Args:
        value: The datetime to check.

    Returns:
        ``value`` unchanged, if it is timezone-aware.

    Raises:
        ValueError: If ``value.tzinfo`` is ``None``.
    """
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware (naive datetimes are rejected).")
    return value


class MessagePayload(BaseModel):
    """Generic, transport-agnostic envelope for a single bus message.

    Attributes:
        timestamp: Timezone-aware UTC creation time of the message.
            Defaults to "now" if not supplied. Naive datetimes are
            rejected to prevent ambiguous, timezone-less timestamps from
            entering the system.
        topic: Routing topic used for ZMQ SUB-side filtering (e.g.
            ``"MARKET_DATA"``, ``"NLP_SIGNAL"``). Must be
            ``UPPER_SNAKE_CASE`` to keep topic naming consistent across
            every agent.
        sender_id: Stable identifier of the agent that produced the
            message (e.g. ``"execution-agent-1"``).
        data: Arbitrary, message-specific payload. Kept as a plain
            ``dict`` here since its shape is defined by each topic, not by
            the transport envelope.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    topic: str = Field(min_length=1, pattern=_TOPIC_PATTERN)
    sender_id: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)

    @field_validator("sender_id")
    @classmethod
    def _require_non_blank_sender_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sender_id must not be blank or whitespace-only.")
        return value

    def serialize(self) -> bytes:
        """Serialize this payload to UTF-8 JSON bytes for wire transport.

        Returns:
            The JSON-encoded payload as ``bytes``, ready to be sent as a
            ZeroMQ message frame.
        """
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def deserialize(cls, raw: bytes) -> MessagePayload:
        """Parse and validate raw wire bytes into a :class:`MessagePayload`.

        Args:
            raw: UTF-8 JSON bytes, typically received from a ZeroMQ
                socket frame.

        Returns:
            The validated :class:`MessagePayload`.

        Raises:
            MalformedMessageError: If ``raw`` is not valid UTF-8 JSON, or
                does not conform to the schema (missing/extra fields,
                wrong types, invalid topic format, naive timestamp, etc.).
        """
        try:
            return cls.model_validate_json(raw)
        except ValidationError as exc:
            raise MalformedMessageError(
                f"Received message failed schema validation: {exc}"
            ) from exc


class OrderBookLevel(BaseModel):
    """A single price/volume level of an order book side.

    Attributes:
        price: Limit price at this level. Must be strictly positive.
        volume: Quantity available at ``price``. Must be strictly
            positive (a level with zero volume is, by definition, not
            present in the book).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    price: float = Field(gt=0)
    volume: float = Field(gt=0)


class OrderBookSnapshot(BaseModel):
    """A validated, top-N snapshot of one symbol's order book.

    Guarantees enforced at construction time: at most
    :data:`ORDER_BOOK_TOP_N` levels per side, at least one level per side
    (both sides are required to compute a spread/imbalance), bids sorted
    best-first (descending price), asks sorted best-first (ascending
    price), and a non-crossed book (best bid <= best ask).

    Attributes:
        symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
        timestamp: Timezone-aware UTC time the snapshot was captured.
        bids: Buy-side levels, best (highest price) first.
        asks: Sell-side levels, best (lowest price) first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bids: list[OrderBookLevel] = Field(min_length=1, max_length=ORDER_BOOK_TOP_N)
    asks: list[OrderBookLevel] = Field(min_length=1, max_length=ORDER_BOOK_TOP_N)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)

    @field_validator("bids")
    @classmethod
    def _validate_bids_sorted(cls, value: list[OrderBookLevel]) -> list[OrderBookLevel]:
        if any(value[i].price < value[i + 1].price for i in range(len(value) - 1)):
            raise ValueError("bids must be sorted best-first (descending price).")
        return value

    @field_validator("asks")
    @classmethod
    def _validate_asks_sorted(cls, value: list[OrderBookLevel]) -> list[OrderBookLevel]:
        if any(value[i].price > value[i + 1].price for i in range(len(value) - 1)):
            raise ValueError("asks must be sorted best-first (ascending price).")
        return value

    @model_validator(mode="after")
    def _validate_not_crossed(self) -> OrderBookSnapshot:
        if self.bids[0].price > self.asks[0].price:
            raise ValueError(
                f"Crossed order book for {self.symbol!r}: "
                f"best bid {self.bids[0].price} > best ask {self.asks[0].price}."
            )
        return self

    @property
    def best_bid(self) -> OrderBookLevel:
        """The highest-priced buy level (``bids[0]``)."""
        return self.bids[0]

    @property
    def best_ask(self) -> OrderBookLevel:
        """The lowest-priced sell level (``asks[0]``)."""
        return self.asks[0]


class MarketMetrics(BaseModel):
    """Derived, order-book-level market microstructure metrics.

    Attributes:
        symbol: Exchange trading pair symbol this snapshot describes.
        timestamp: Timezone-aware UTC time the underlying snapshot was
            captured.
        spread: Best ask price minus best bid price. Always ``>= 0`` for
            a valid (non-crossed) book.
        order_book_imbalance: Volume imbalance between the two sides,
            aggregated over the top :data:`ORDER_BOOK_TOP_N` levels:
            ``(bid_volume - ask_volume) / (bid_volume + ask_volume)``.
            Ranges from ``-1`` (all volume on the ask side) to ``+1``
            (all volume on the bid side); ``0`` means perfectly balanced.
        mid_price: Midpoint between best bid and best ask
            (``(best_bid + best_ask) / 2``). Without this, no downstream
            consumer of this schema (VaR's price-return calculation,
            offline training's future-return labels, ...) has any way to
            reconstruct an absolute price level -- spread/imbalance alone
            describe the book's *shape*, not *where* it is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spread: float = Field(ge=0)
    order_book_imbalance: float = Field(ge=-1, le=1)
    mid_price: float = Field(gt=0)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)

    @classmethod
    def from_order_book_snapshot(cls, snapshot: OrderBookSnapshot) -> MarketMetrics:
        """Derive spread, imbalance, and mid-price from a validated snapshot.

        Args:
            snapshot: The order book snapshot to compute metrics from.

        Returns:
            A new :class:`MarketMetrics` for the same symbol/timestamp.
        """
        spread = snapshot.best_ask.price - snapshot.best_bid.price
        bid_volume = sum(level.volume for level in snapshot.bids)
        ask_volume = sum(level.volume for level in snapshot.asks)
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        mid_price = (snapshot.best_bid.price + snapshot.best_ask.price) / 2.0
        return cls(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            spread=spread,
            order_book_imbalance=imbalance,
            mid_price=mid_price,
        )


class MacroSnapshot(BaseModel):
    """A single macroeconomic or on-chain indicator reading.

    Attributes:
        timestamp: Timezone-aware UTC time this reading was captured (or
            published by the source, if that differs from ingestion
            time).
        indicator: Indicator identifier, ``UPPER_SNAKE_CASE`` (e.g.
            ``"US_10Y_YIELD"``, ``"BTC_EXCHANGE_FLOW"``), for the same
            naming-consistency reasons as :attr:`MessagePayload.topic`.
        value: The reading itself.
        source: Origin of this reading (e.g. ``"FRED"``,
            ``"Glassnode"``), for provenance/debugging when multiple
            providers feed the same indicator.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    indicator: str = Field(min_length=1, pattern=_TOPIC_PATTERN)
    value: float
    source: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)


class SentimentPolarity(str, Enum):
    """The three sentiment classes FinBERT (and this system) classifies into."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class SentimentSignal(BaseModel):
    """A single NLP sentiment reading for one piece of financial news.

    Attributes:
        timestamp: Timezone-aware UTC time this signal was produced.
        headline: The news headline/text that was analyzed.
        source: Origin of the headline (e.g. ``"Reuters"``,
            ``"Bloomberg"``), for provenance/debugging.
        polarity: The classified sentiment.
        score: The model's confidence in ``polarity``, in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    headline: str = Field(min_length=1)
    source: str = Field(min_length=1)
    polarity: SentimentPolarity
    score: float = Field(ge=0, le=1)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)


class SupervisorAction(str, Enum):
    """The three actions the supervisor's decision engine can output."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SuperviserDecision(BaseModel):
    """The supervisor's fused trading decision for one evaluation cycle.

    Attributes:
        timestamp: Timezone-aware UTC time this decision was produced.
        action: The chosen action.
        confidence_score: The decision engine's confidence in ``action``,
            in ``[0, 1]`` (e.g. derived from
            :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel`'s
            output probability).
        target_size: Position size this decision calls for. Must be
            ``0`` for ``HOLD`` (a hold that also sizes a position is a
            contradiction) and strictly positive for ``BUY``/``SELL``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: SupervisorAction
    confidence_score: float = Field(ge=0, le=1)
    target_size: float = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)

    @model_validator(mode="after")
    def _validate_target_size_matches_action(self) -> SuperviserDecision:
        if self.action == SupervisorAction.HOLD and self.target_size != 0:
            raise ValueError("target_size must be 0 when action is HOLD.")
        if self.action != SupervisorAction.HOLD and self.target_size == 0:
            raise ValueError(f"target_size must be > 0 when action is {self.action.value}.")
        return self


class TradeSide(str, Enum):
    """Which side of the market a confirmed trade executed on.

    A separate enum from :class:`SupervisorAction`: an *executed* trade
    can only ever be a buy or a sell, unlike a *decision*, which can also
    be ``HOLD``.
    """

    BUY = "BUY"
    SELL = "SELL"


class TradeExecution(BaseModel):
    """A confirmed order execution, published for archival/audit purposes.

    Attributes:
        timestamp: Timezone-aware UTC time the order was executed.
        symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
        side: ``BUY`` or ``SELL``.
        cost: Notional order size in quote-currency units, as passed to
            :meth:`~src.agents.market_data.exchange_client.
            AsyncExchangeClient.create_market_order_by_cost`.
        order_id: Exchange-assigned order identifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = Field(min_length=1)
    side: TradeSide
    cost: float = Field(gt=0)
    order_id: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)


# Topic every signal-generating agent (the XGBoost-based
# ~src.agents.supervisor.agent and the RL-based
# ~src.agents.rl_signal.agent) publishes a TradingSignal on, and the sole
# topic ~src.agents.execution.agent.ExecutionAgent subscribes to. Kept
# here (not locally per-agent, unlike this project's other topic
# constants) because two independent producers and one consumer must
# agree on the exact same string -- a single source of truth removes any
# chance of that drifting.
TRADING_SIGNALS_TOPIC: Final[str] = "TRADING_SIGNALS"


class TradingSignal(BaseModel):
    """One agent's directional opinion on one instrument -- never an order.

    Published by a signal-generating agent, consumed by
    :class:`~src.agents.execution.agent.ExecutionAgent`, which is the
    *only* thing in this system authorized to act on it -- and even then,
    only once it has independently arbitrated consensus between every
    signal agent (see that class's docstring). A single ``TradingSignal``
    on its own authorizes nothing.

    Attributes:
        timestamp: Unix epoch seconds this signal was generated at (wall
            clock, not ``time.monotonic()``: signals cross process/agent
            boundaries over the ZMQ bus, so the freshness check that
            consumes this must be comparable regardless of which
            process/thread produced it).
        agent_id: Which agent produced this opinion (e.g.
            ``"supervisor-agent"``, ``"rl-signal-agent"``) -- the
            consensus check needs to tell independent agents' opinions
            apart, not just count messages.
        instrument: The instrument this opinion is about (e.g.
            ``"EUR_USD"``).
        signal: ``-1`` (sell), ``0`` (neutral/hold), or ``1`` (buy).
        confidence: The producing agent's confidence in ``signal``, in
            ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(gt=0)
    agent_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    signal: Literal[-1, 0, 1]
    confidence: float = Field(ge=0, le=1)


# Topic api_middleware/gateway.py publishes each validated, authenticated
# client macro-order onto. Two independent real subscribers as of this
# writing: src.execution.shadow_crossing_engine.ShadowCrossingEngine
# (SHADOW-relevant observation) and
# src.agents.macro_order_execution.agent.MacroOrderExecutionAgent (the
# real "Ingestor Agent" this comment used to say didn't exist -- it
# consumes execution_mode=LIVE orders and really executes them). Kept
# here, not locally in api_middleware/, for the same single-source-of-
# truth reason as TRADING_SIGNALS_TOPIC: every subscriber must agree on
# the exact same string.
CLIENT_ORDERS_TOPIC: Final[str] = "CLIENT_ORDERS"

# Topic every real order-lifecycle emitter publishes a real
# src.execution.audit_ledger.OrderAuditEvent onto (SUBMITTED at
# api_middleware/gateway.py's submit_order, EXECUTED at both
# gateway.py's _write_shadow_fill and MacroOrderExecutionAgent's real
# fills, REJECTED at MacroOrderExecutionAgent's real refusal paths,
# CANCELLED at gateway.py's real cancel_order route -- see that
# module's own docstring for exactly what is wired here). Consumed by
# src.execution.audit_ledger.CryptographicAuditLedger, which chains and
# persists every event -- see that class's own docstring for the real
# tamper-evidence (not tamper-prevention) guarantee this actually provides.
ORDER_AUDIT_EVENTS_TOPIC: Final[str] = "ORDER_AUDIT_EVENTS"

# Redis STRING key prefix (+ order_id), real, short-TTL marker --
# api_middleware/gateway.py's cancel_order route sets this the moment a
# client really cancels an order; MacroOrderExecutionAgent checks it
# (real EXISTS call) immediately before executing a real LIVE order and
# skips execution if present. Real, but honestly bounded: that agent
# executes immediately on receipt (no time-based scheduling exists yet
# -- see its own module docstring), so this real cancellation window is
# typically short; most cancel attempts will find the order already
# terminal (see cancel_order's own 409 handling for that real case).
CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX: Final[str] = "cancelled_order:"

# Topic api_middleware/fix_server.py publishes each accepted, translated
# NewOrderSingle onto, as a full core.nothierly_ontology.UniversalOrder
# payload -- deliberately a *different* topic from CLIENT_ORDERS_TOPIC
# above, not a reuse of it: that topic's real payload shape is
# MacroOrderRequest (client_id/asset/total_quantity/...), not
# UniversalOrder -- publishing both shapes under one topic name would
# silently break whatever future consumer is built expecting one
# consistent shape. Same "no real subscriber exists yet" honesty as
# CLIENT_ORDERS_TOPIC above.
FIX_INBOUND_ORDERS_TOPIC: Final[str] = "FIX_INBOUND_ORDERS"

# Redis list ai_agents/alpha_swarm.py's run_alpha_swarm_once pushes each
# real, backtested strategy proposal onto (LPUSH, capped via LTRIM --
# same "bounded Redis list, not an immutable audit store" honesty as
# every other *_ALERTS/*_LOG key in this project). Read by
# src/dashboard/ems_pages/quant_copilot_chat.py's "Alpha Swarm --
# Propositions en attente" section for human approve/reject -- a
# cross-module shared key, same reason FIX_INBOUND_ORDERS_TOPIC above
# lives here rather than in either module alone.
ALPHA_SWARM_PROPOSALS_REDIS_KEY: Final[str] = "ALPHA_SWARM_PROPOSALS"

# Redis list ai_agents/evolutionary_tuner.py's run_evolutionary_tuning_and_propose
# pushes each real, CMA-ES-discovered, backtest-validated parameter
# proposal onto (LPUSH, capped via LTRIM -- same bounded-list pattern as
# ALPHA_SWARM_PROPOSALS_REDIS_KEY above). A separate key, not a reuse of
# that one: the payload shape is a parameter-tuning proposal
# (TuningProposal), not a new-strategy proposal (AlphaProposal) -- same
# "different real payload shape needs a different topic/key" reasoning
# as FIX_INBOUND_ORDERS_TOPIC vs CLIENT_ORDERS_TOPIC. Read by
# src/dashboard/ems_pages/quant_copilot_chat.py's "Evolutionary Tuner"
# section for human approve/reject -- never applied automatically, see
# that module's own docstring.
EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY: Final[str] = "EVOLUTIONARY_TUNING_PROPOSALS"

# Chapter 13, Étape 5.2 ("The Red Phone"): broadcast by src/agents/
# alert_agent.py when it trips the circuit breaker. Deliberately
# separate from, and complementary to, core.security's real
# SYSTEM_STATUS Redis kill switch (SAFE/PANIC) -- that flag already gates
# every *new* consensus order via @requires_system_active; it does not,
# by itself, make any already-running agent proactively act. This topic
# is what tells ExecutionAgent to actively flatten every open position
# right now (see ExecutionAgent.cancel_all_positions), not just stop
# opening new ones. Alert_agent's circuit breaker calls
# core.security.trip_kill_switch() AND publishes this -- two real,
# distinct effects, not a duplicate of the same one.
EMERGENCY_HALT_TOPIC: Final[str] = "EMERGENCY_HALT"

# Published by nothierly_genesis.topological_crash_detector when a real
# Topological Fragility Index breaches its (uncalibrated -- see that
# module's own docstring) threshold. Deliberately a separate topic from
# EMERGENCY_HALT_TOPIC, not a reuse of it: the Genesis research tier is
# architecturally isolated from the production agent fleet (heavy
# torch/ripser/dowhy imports must never end up on api_middleware/
# gateway.py's or ExecutionAgent's import graph -- see that module's own
# "why ZMQ, not a direct import" note) and runs as its own separate
# process, publishing an *alert* a real, separately-running production
# subscriber (not built in this module) would react to -- it never calls
# core.security.trip_kill_switch itself.
TOPOLOGY_FRAGILITY_ALERT_TOPIC: Final[str] = "TOPOLOGY_FRAGILITY_ALERT"

FEATURES_TOPIC: Final[str] = "FEATURES"

# Redis key prefix (+ instrument) FeatureStore additionally persists each
# vector under, for the two signal agents' poll-based reads -- see
# src.agents.supervisor.feature_confirmation and FeatureStore's own
# docstring for why this exists alongside the ZMQ publish above.
FEATURE_VECTOR_REDIS_KEY_PREFIX: Final[str] = "feature_vector:"

# Redis key prefix (+ agent_id) core.base_agent.BaseAgent.run's tick loop
# writes a fresh timestamp under, with a short TTL, every tick -- a
# missing/expired key means that agent process is down or stalled, not
# just "hasn't reported in yet". Read by
# src.agents.quant_copilot.get_active_agents_status.
HEARTBEAT_REDIS_KEY_PREFIX: Final[str] = "heartbeat:"

# Redis STRING key prefix (+ agent_id) an agent's real, current
# core.base_agent.AgentControlState ("LIVE"/"SHADOW"/"OFF") is persisted
# under -- read once by core.base_agent.BaseAgent._listen_for_control_commands
# at startup (so an agent restarting after a dashboard "OFF" click resumes
# in the correct state, not silently back to LIVE) and written by whichever
# dashboard control surface changes it (src/dashboard/ems_pages/agent_fleet.py).
# No TTL: unlike the heartbeat above, an operator's control decision must
# persist across a Redis restart or an agent being offline, not expire.
AGENT_CONTROL_STATE_REDIS_KEY_PREFIX: Final[str] = "agent_control_state:"

# Redis Pub/Sub channel prefix (+ agent_id) a NEW control state is
# PUBLISHed to for immediate effect on an already-running agent -- Pub/Sub,
# not a second read of the STRING key above, so a live agent's in-memory
# core.base_agent.BaseAgent._control_state updates the instant an operator
# acts, without adding a per-tick Redis round trip to the hot loop (see
# that method's own docstring for why: a background listener task updates
# an in-memory attribute once, and every real gating check afterward reads
# that attribute, not Redis). One-shot commands, not persisted state --
# same real Pub/Sub-vs-list-or-string reasoning as
# src.agents.execution.agent's own _MANUAL_TRADE_CHANNEL.
AGENT_CONTROL_CHANNEL_PREFIX: Final[str] = "agent_control_channel:"

# Redis key prefix (+ symbol) src.agents.market_data.agent.MarketDataAgent
# writes a fresh MarketMetrics JSON blob under, with a short TTL, on every
# throttled book update. Read by
# src.agents.quant_copilot.get_live_toxicity for the real (OBI) half of
# that tool's report.
MARKET_METRICS_REDIS_KEY_PREFIX: Final[str] = "market_metrics:"

# Redis key prefix (+ symbol) src.agents.market_data.l2_surface_packer's
# publish_l2_surface writes a capped (LPUSH+LTRIM) real history of
# compact, base64-wrapped MessagePack-encoded L2 snapshots under --
# the full-depth companion to MARKET_METRICS_REDIS_KEY_PREFIX above
# (that key persists only top-of-book scalars; this one carries the
# real per-level price/volume arrays Command Center's own "3D order-
# book topography" section needed and, until now, did not have a real
# producer for). Read by src/dashboard/ems_pages/command_center.py.
L2_SURFACE_REDIS_KEY_PREFIX: Final[str] = "l2_surface:"

# Redis LIST key prefix (+ symbol) (LPUSH + LTRIM, newest first)
# src.agents.market_data.agent.MarketDataAgent pushes one real
# {timestamp (ms), side, amount, price} trade record to on every real
# trade its ccxt.pro trade-stream observes -- previously "consumed and
# logged", never persisted (see that agent's own module docstring on
# why: no derived signal was computed from trade flow before this).
# Read by src.agents.market_data.microstructure_features to compute
# real, live VPIN (research.alpha_features.compute_vpin) and TFI
# (research.alpha_features.compute_tfi) from this real rolling tape --
# see that module's own docstring for src/dashboard/ems_pages/
# microstructure.py's previously-disclosed "VPIN/TFI only exist in
# offline research code" gap, now closed.
TRADE_TAPE_REDIS_KEY_PREFIX: Final[str] = "trade_tape:"
TRADE_TAPE_MAX_ENTRIES: Final[int] = 5_000

# Redis key prefix (+ order_id) api_middleware/gateway.py writes each
# client order's status JSON blob under. Read by
# src.agents.quant_copilot.get_tca_report and
# src.agents.quant_copilot.adjust_algo_aggression, in addition to the
# gateway's own status/stream endpoints.
ORDER_STATUS_REDIS_KEY_PREFIX: Final[str] = "order_status:"

# Redis LIST key (LPUSH + LTRIM, newest first) every real fill-producing
# execution path pushes one record to: api_middleware/gateway.py's
# _write_shadow_fill (Shadow-mode simulated fills) and
# src.agents.macro_order_execution.agent.MacroOrderExecutionAgent (real
# LIVE fills). Read by src/dashboard/ems_pages/order_blotter.py. Was
# independently duplicated as a private constant in both writers before
# this one, shared home -- promoted here once a second real writer
# needed the exact same key.
FILLS_BLOTTER_REDIS_KEY: Final[str] = "NOTHIERLY_FILLS_BLOTTER"
FILLS_BLOTTER_MAX_ENTRIES: Final[int] = 300

# Redis LIST key (LPUSH + LTRIM, newest first) src.execution.smart_order_router's
# execute_allocation_plan pushes one real, capped record to on every real
# routing decision it actually executes (never for a "Calculer le plan"
# preview alone -- see that function's own docstring) -- symbol, side,
# per-venue quantity/effective price, and the real, always-non-negative
# AllocationPlan.savings_vs_worst_level_usd figure. Read by
# src/dashboard/ems_pages/smart_order_router_radar.py for its real
# "Radar Live" console and monthly savings counter.
SOR_ROUTING_LOG_REDIS_KEY: Final[str] = "SOR_ROUTING_LOG"
SOR_ROUTING_LOG_MAX_ENTRIES: Final[int] = 200

# Redis STRING counter (INCR), real and cumulative --
# src.agents.macro_order_execution.agent.MacroOrderExecutionAgent
# increments this every time a real execution_mode=LIVE order tagged
# execution_algo=TOXICITY_AVOIDANCE is actually reduced or fully refused
# by execution.predator_honeypot.AdverseFlowDefenseAgent's real posture
# (fed real, live VPIN -- see that agent's own docstring). Read by
# src/dashboard/ems_pages/order_blotter.py -- the real counter that used
# to be demo-only ("Toxic Orders Evaded (Count, démo)").
TOXIC_ORDERS_EVADED_REDIS_KEY: Final[str] = "TOXIC_ORDERS_EVADED_COUNT"

# Redis LIST key (LPUSH + LTRIM, newest first)
# src.execution.shadow_crossing_engine.ShadowCrossingEngine pushes one real,
# persisted record to every time its internal (real, reused, not
# reimplemented) core.crossing_network.CrossingNetwork finds a real shadow
# match between two orders observed on CLIENT_ORDERS_TOPIC -- timestamp,
# instrument, quantity, real mid-price at match time, and a real,
# documented theoretical fee-saved estimate. Never a real executed cross
# (see that module's own docstring on why: no real order is ever
# intercepted or redirected). Read by
# src/dashboard/ems_pages/internal_dark_pool.py.
DARK_POOL_SHADOW_MATCHES_REDIS_KEY: Final[str] = "DARK_POOL_SHADOW_MATCHES"
DARK_POOL_SHADOW_MATCHES_MAX_ENTRIES: Final[int] = 200

# Redis STRING key src.execution.audit_ledger.CryptographicAuditLedger
# overwrites with the current chain tip (seq_id + current_hash + written_at
# JSON) after every real event it persists to the durable order_audit_trail
# TimescaleDB table. A real, independent second witness to the chain's tip,
# stored in a different system than the hash-chained rows themselves -- an
# attacker who edits TimescaleDB rows and recomputes the whole downstream
# chain still has to separately, consistently falsify this key too, or a
# verification pass that cross-checks both will disagree. Not a substitute
# for real external/offline anchoring (see that module's own docstring on
# what this does and does not prove), just a real, cheap improvement over
# trusting one single store alone.
AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY: Final[str] = "AUDIT_LEDGER_CHAIN_TIP"

# Redis LIST key (LPUSH, uncapped -- a real compliance event must never be
# silently dropped for capacity reasons, unlike every LTRIM-capped list
# above) src.execution.audit_ledger.CryptographicAuditLedger pushes a raw
# OrderAuditEvent onto only when persisting it to TimescaleDB has *already*
# failed after real bounded retries -- see that module's own docstring on
# why this exists (ArchivistAgent's own "logged and dropped, not retried"
# batch semantics are not an acceptable loss model for a compliance-relevant
# event) and its honest limits (queued here for manual replay, not
# automatically reprocessed).
AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY: Final[str] = "AUDIT_LEDGER_DEAD_LETTER_QUEUE"

# Redis key prefix (+ agent_id) src.agents.alpha_engine.agent.AlphaMarketMakerAgent
# writes a real, short-TTL JSON telemetry blob under, on every tick
# (inventory, avg entry price, session realized/unrealized PnL, 24h trade
# count, market-data staleness, current AgentControlState) -- read by
# src/dashboard/ems_pages/autonomous_command_center.py. TTL'd like the
# heartbeat above (not the no-TTL control-state key): a missing/expired
# key here should read as "no live telemetry", not a stale last value
# left behind after the engine crashed.
ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX: Final[str] = "alpha_engine_telemetry:"

# Redis Pub/Sub channel prefix (+ agent_id) the dashboard's "Emergency
# Liquidate" button PUBLISHes a one-shot command to -- deliberately a
# *separate* channel from AGENT_CONTROL_CHANNEL_PREFIX above, not an
# overload of AgentControlState.OFF: OFF only means "place no new
# quotes", the same passive meaning every other agent already gives it
# (see AgentControlState's own docstring) -- it does not, by itself,
# mean "flatten every open position right now". Emergency Liquidate is
# the strictly stronger, explicit action; AlphaMarketMakerAgent sets its
# own control state to OFF as *part of* handling this message, it does
# not treat OFF as implying it.
ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX: Final[str] = "alpha_engine_emergency_liquidate:"

# Redis key prefix (+ agent_id) src.agents.alpha_engine.agent.AlphaMarketMakerAgent persists its
# own inventory baseline under -- no TTL (same reasoning as
# AGENT_CONTROL_STATE_REDIS_KEY_PREFIX: a real operator decision, here "how much of the account's
# real balance existed before this engine ever started", must survive a Redis restart or the
# engine being offline, not expire). Real bug this fixes, found while building the cold-start
# sanitizer: recomputing the baseline fresh from the broker's current balance on every on_start()
# (the original implementation) silently "forgot" whatever inventory the engine itself had
# legitimately accumulated across a crash/restart -- treating its own prior real exposure as
# pre-existing, cap-exempt balance instead of counting it. Persisting the baseline once, on this
# engine's real first-ever start, and reusing it across every subsequent restart, is what makes the
# inventory cap keep meaning what it says after a crash.
ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX: Final[str] = "alpha_engine_baseline_qty:"

# Redis key prefix (+ registry symbol, e.g. "EUR_USD") src/dashboard/ems_pages/
# asset_universe_control.py writes a real "ON"/"OFF" per-asset kill switch under. No TTL --
# same reasoning as AGENT_CONTROL_STATE_REDIS_KEY_PREFIX: a real operator decision must survive a
# Redis restart, not silently revert to "ON". **Not yet consumed by AlphaMarketMakerAgent** (which
# still only trades one hardcoded self._symbol per process, gated solely by its own
# AgentControlState) -- written here, real and functional, so a future multi-symbol engine (or a
# per-symbol read added to _run_cycle) has a real key to read rather than needing this page rebuilt.
ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX: Final[str] = "alpha_engine_asset_status:"

# Redis key prefix (+ registry symbol) the same page writes a real JSON
# {"max_allowable_spread": "<Decimal str>", "max_position_size": "<Decimal str>", "updated_at": <float>}
# dynamic risk override under -- meant to shadow (not replace) core.asset_universe.ASSET_UNIVERSE's
# static config without a restart. Same "not yet consumed" honesty note as the key above.
ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX: Final[str] = "alpha_engine_asset_risk_override:"


class FeatureVector(BaseModel):
    """One instrument's latest normalized technical-indicator snapshot.

    Published by :class:`~src.data_pipeline.feature_store.FeatureStore`
    on :data:`FEATURES_TOPIC`, consumed by every signal-generating agent
    (:class:`~src.agents.supervisor.agent.XgboostSignalAgent`,
    :class:`~src.agents.rl_signal.agent.RlSignalAgent`).

    Deliberately **one flat topic for every instrument**, not a
    per-instrument topic like ``features.BTC_USD``: this ``instrument``
    field is what a subscriber filters on, exactly the same "flat topic +
    field, not topic-per-symbol" choice already made for
    :data:`TRADING_SIGNALS_TOPIC` and ``MARKET_DATA_L2`` (see
    :class:`~src.agents.execution.agent.ExecutionAgent`'s module
    docstring) -- and, mechanically, the only choice ``_TOPIC_PATTERN``
    above actually allows (it requires an uppercase name with no ``.``).

    Every field is already normalized to ``[-1, 1]`` by
    :class:`~src.data_pipeline.feature_engine.FeatureEngine` before this
    is constructed -- see that module's docstring for exactly how each
    one gets there (RSI's fixed-bound rescale vs. every other field's
    rolling Z-score).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str = Field(min_length=1)
    timestamp: float = Field(gt=0)
    ema_fast: float = Field(ge=-1, le=1)
    ema_slow: float = Field(ge=-1, le=1)
    rsi: float = Field(ge=-1, le=1)
    bb_upper: float = Field(ge=-1, le=1)
    bb_lower: float = Field(ge=-1, le=1)
    atr: float = Field(ge=-1, le=1)

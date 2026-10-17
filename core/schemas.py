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
        top_of_book_liquidity: Real summed volume across every level of
            both sides of the underlying snapshot (``sum(bid volumes) +
            sum(ask volumes)``), added for :class:`~src.agents.execution.
            agent.ExecutionAgent`'s Gate 7 ("StochasticCore") price-impact
            model -- see :func:`core.quant_math.price_impact`'s own
            docstring on why a real liquidity reading (not a stand-in
            proxy) matters there. ``0.0`` (the default) means "unknown,"
            not "zero real liquidity" -- only a caller building this via
            :meth:`from_order_book_snapshot` (i.e. from a real snapshot)
            ever sets a genuine positive value; anything constructing this
            model directly (every pre-existing call site, including every
            test in this repo) keeps working unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spread: float = Field(ge=0)
    order_book_imbalance: float = Field(ge=-1, le=1)
    mid_price: float = Field(gt=0)
    top_of_book_liquidity: float = Field(ge=0, default=0.0)

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
            top_of_book_liquidity=bid_volume + ask_volume,
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

# ZMQ Pub/Sub topic core.exception_manager broadcasts a fatal, unhandled exception's traceback onto
# (event loop's own last-resort loop.set_exception_handler) or core.watchdog broadcasts a triggered
# shutdown reason onto -- see both modules' own docstrings. **Deliberately NOT ORDER_AUDIT_EVENTS_TOPIC
# above**: OrderAuditEvent's schema is order-lifecycle-shaped (a required order_id, a closed
# event_type Literal of SUBMITTED/EXECUTED/REJECTED/CANCELLED) and CryptographicAuditLedger's real
# hash chain is a compliance record for order events specifically -- a system-fatal exception is
# neither an order event nor something that schema can honestly represent, so this gets its own topic
# rather than a forced-fit `event_type`. **Real, disclosed gap**: no consumer/persister exists for
# this topic yet (unlike ORDER_AUDIT_EVENTS_TOPIC's CryptographicAuditLedger) -- this is a real,
# functional ZMQ broadcast any future subscriber can pick up, not yet chained into durable WORM
# storage the way order events are.
SYSTEM_FATAL_EXCEPTION_TOPIC: Final[str] = "SYSTEM_FATAL_EXCEPTION"

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

# Topic src.agents.spectral_denoiser.agent.SpectralDenoiserAgent publishes each instrument's latest
# FFT-denoised price signal on -- see DenoisedPriceSignal below for the payload shape. No consumer
# wired in yet (a new, standalone signal source, not yet required by ExecutionAgent's consensus).
DENOISED_PRICE_SIGNAL_TOPIC: Final[str] = "DENOISED_PRICE_SIGNAL"

# =============================================================================
# Swarm AI -- Scout / Analyst / Risk Gate reactive pipeline (scout_agent.py /
# analyst_agent.py / risk_gate_agent.py). ZeroMQ carries the real instant
# event; Redis carries the real state each event refers to -- neither topic
# below ever embeds the full anomaly/analysis payload itself, only the
# `anomaly_id` a subscriber uses to read the real record from Redis. Same
# "small event, real state lives in Redis" discipline every other real
# ZMQ topic in this project already follows (see e.g. TRADING_SIGNALS_TOPIC's
# own MessagePayload.data shape).
# =============================================================================

# Real, statistical (z-score) anomaly on p2p_api.py's own accumulated P2P
# premium history -- published by scout_agent.py the moment a real deviation
# crosses threshold for one of the five tracked currencies.
SWARM_ANOMALY_DETECTED_TOPIC: Final[str] = "SWARM_ANOMALY_DETECTED"
# Real Redis HASH key prefix (+ anomaly_id) scout_agent.py writes the real
# detection details to (currency, latest/baseline premium, z-score,
# detected_at) -- read by analyst_agent.py on receiving the event above.
SWARM_ANOMALY_REDIS_KEY_PREFIX: Final[str] = "swarm:anomalies:"

# Real outcome of analyst_agent.py's own reactive check (political_agent.py's
# real RSS fetch + real LLM crisis scoring, reused as a library call -- see
# that module's own docstring for why it is not itself a ZMQ participant).
SWARM_ANALYSIS_DONE_TOPIC: Final[str] = "SWARM_ANALYSIS_DONE"
# Real Redis HASH key prefix (+ anomaly_id) analyst_agent.py writes its real
# outcome to -- status is one of "CRISIS_CONFIRMED"/"NO_CRISIS"/
# "NO_MATCHING_HEADLINES"/"LLM_UNCONFIGURED", never a fabricated verdict when
# no real LLM provider is configured (see political_agent.py's own module
# docstring on this exact discipline).
SWARM_ANALYSIS_REDIS_KEY_PREFIX: Final[str] = "swarm:analysis:"

# Real final green light from risk_gate_agent.py -- only published when
# analyst_agent.py's own status is "CRISIS_CONFIRMED" *and* the currency's
# real, current P2P liquidity (p2p_api.py's own already-computed
# avg_liquidity_usdt field) clears the configured floor. A human operator
# (dashboard), not an autonomous trader, is the real subscriber -- this
# project holds no real execution capability for P2P/fiat-rail arbitrage
# (see p2p_api.py's own module docstring on what this data can and cannot
# back), so the swarm's real output is a real, actionable alert, not a
# fabricated auto-trade.
SWARM_ARBITRAGE_APPROVED_TOPIC: Final[str] = "SWARM_ARBITRAGE_APPROVED"
# Real, bounded Redis LIST (LPUSH + LTRIM, newest first) of every real
# approval risk_gate_agent.py has ever published -- read by the dashboard's
# Swarm Console section. Same bounded-growth discipline as every other real
# event log in this project (see e.g. ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX).
SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY: Final[str] = "swarm:approved_opportunities"
SWARM_APPROVED_OPPORTUNITIES_MAX_ENTRIES: Final[int] = 500

# Real, separate channel/list for nothierly_genesis/saga_proposal_generator.py's own output --
# deliberately NOT the same topic/key as SWARM_ARBITRAGE_APPROVED_TOPIC above: that one's entries
# are scout/analyst/risk-gate's real single-fiat crisis alerts (fiat, crisis_score, sentiment...);
# a TDA-derived multi-leg saga proposal has a structurally different shape (a cycle, per-leg
# liquidity, a saga_steps breakdown) and reusing the same list would mix two incompatible record
# shapes the dashboard renders differently. Same disclosed scope as that topic's own comment: a
# real, actionable proposal for a human operator to review -- this project has no real execution
# capability wired to it, and nothing here auto-executes.
SWARM_TDA_SAGA_PROPOSED_TOPIC: Final[str] = "SWARM_TDA_SAGA_PROPOSED"
SWARM_TDA_SAGA_REDIS_KEY: Final[str] = "swarm:tda_saga_proposals"
SWARM_TDA_SAGA_MAX_ENTRIES: Final[int] = 200

# --- Generic, currency-agnostic Redis key patterns (market_discovery.py's own consumers) -------
# Real, disclosed scope: this is the go-forward pattern for NEW dynamic-market code
# (market_discovery.py, the generalized nothierly_genesis/tda_engine.py) -- it does not retrofit
# every existing hardcoded key above (P2P_HISTORY_ZSET_KEY_PREFIX, SWARM_APPROVED_OPPORTUNITIES_
# REDIS_KEY, ...), which real, already-tested production code paths still read/write today.
# Migrating those is real, separate, larger follow-up work, not silently implied by adding these.
def p2p_premium_redis_key(fiat_code: str, asset_symbol: str = "USDT") -> str:
    """``market:p2p:{fiat}:{asset}:premium`` -- one real key per (fiat, asset) pair, built at call
    time from whatever :mod:`market_discovery` actually discovered, never a hardcoded currency."""
    return f"market:p2p:{fiat_code.upper()}:{asset_symbol.upper()}:premium"


def spot_price_redis_key(base_symbol: str, quote_symbol: str) -> str:
    """``market:spot:{base}_{quote}:price`` -- one real key per real ccxt spot pair discovered."""
    return f"market:spot:{base_symbol.upper()}_{quote_symbol.upper()}:price"

# Redis key prefix (+ instrument) FeatureStore additionally persists each
# vector under, for the two signal agents' poll-based reads -- see
# src.agents.supervisor.feature_confirmation and FeatureStore's own
# docstring for why this exists alongside the ZMQ publish above.
FEATURE_VECTOR_REDIS_KEY_PREFIX: Final[str] = "feature_vector:"

# Redis STRING key prefix (+ instrument), TTL'd, SpectralDenoiserAgent
# persists its latest DenoisedPriceSignal under -- same "ZMQ publish +
# Redis cache, both real, neither the source of truth for the other"
# pattern as FEATURE_VECTOR_REDIS_KEY_PREFIX above (a late poller reads
# this; a live subscriber reads the ZMQ publish; both see the same data).
DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX: Final[str] = "denoised_price_signal:"

# Redis STRING key prefix (+ instrument), TTL'd, ExecutionAgent.
# compute_optimal_trajectory persists its latest real OptimalExecutionSchedule
# under -- Redis-only (see that model's own docstring on why this isn't
# also a ZMQ publish), read by a real dashboard impact-curve chart.
OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX: Final[str] = "optimal_execution_schedule:"

# Redis STRING key (no per-instrument suffix -- one real, single portfolio-level snapshot), TTL'd,
# ExecutionAgent.compute_portfolio_greeks persists its latest real PortfolioGreeks under --
# Redis-only, same reasoning as OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX above.
PORTFOLIO_GREEKS_REDIS_KEY: Final[str] = "nothierly_portfolio_greeks"

# Redis STRING key prefix (+ instrument), TTL'd, ExecutionAgent._execute_sliced_entry persists its
# latest real OrderSlicingProgress under -- "Protocole Stealth Slicing" (2026-09-16): unlike
# OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX above (a computed plan, never actually executed over
# time), this key tracks a real, in-flight or completed TWAP slicing execution -- real broker calls
# already placed, not a preview. Read by src/dashboard/ems_pages/fleet_command_nexus.py's Module 10.
ORDER_SLICING_PROGRESS_REDIS_KEY_PREFIX: Final[str] = "order_slicing_progress:"

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

# Redis LIST key prefix (+ agent_id) (LPUSH + LTRIM, newest first) AlphaMarketMakerAgent._record_fill
# pushes one real entry to on every real fill it detects (see that method's own docstring point 7 on
# how a fill is detected/priced) -- timestamp, fill_qty/price, and the resulting realized_pnl_delta_usd/
# session_realized_pnl_usd. ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX above is a short-TTL *snapshot*
# (today's/session's running totals only, gone the moment the engine stops publishing); this is the
# real history a P&L equity curve needs and the snapshot alone cannot provide. Read by
# src/dashboard/ems_pages/bilan.py. Honestly empty until this engine's real (non-SHADOW/mock) path
# detects its first fill -- see that page's own docstring for why no other component in this codebase
# can stand in for it (src/execution/reporting_engine.py's own docstring: this system takes no
# directional position on a B2B client's behalf, so client fills have no PnL/win-rate to report either).
ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX: Final[str] = "alpha_engine_pnl_history:"
ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES: Final[int] = 5000

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

# Redis STRING key ("true"/"false"), no TTL -- same reasoning as AGENT_CONTROL_STATE_REDIS_KEY_PREFIX:
# a real operator decision must survive a Redis restart, not silently revert. This is the live master
# switch for src.mlops.vast_client.VastAIClient -- src/dashboard/ems_pages/finops.py's toggle writes
# here so the running nothierly-agents/nothierly-dashboard containers see the change on their very
# next call, with no restart. core.config.VastAiSettings.enabled (env var VASTAI_ENABLED) is only the
# cold-start fallback used the first time this key has never been written -- see
# src.mlops.vast_client.is_vast_ai_enabled for the real precedence and its fail-closed behavior on a
# Redis error (this is a cost-control switch: any doubt about its state must block, not allow).
VAST_AI_ENABLED_REDIS_KEY: Final[str] = "vast_ai_enabled"

# Real, dynamic lookup for political_agent.py's proprietary LLM circuit breaker: a rented Vast.ai
# instance gets a new public IP:port every rental, so this can never be a static .env value the way
# GUARDIAN_API_KEY is. scripts/deploy_custom_llm.py writes this (a full "http://host:port" base URL,
# no trailing slash) only once its vLLM server is confirmed answering /health -- never on a bare
# `rent_instance` success alone, so this key can never point at a container still loading its model.
# Absent/empty is a real, valid, common state ("no proprietary LLM deployed right now"), not an
# error -- political_agent.py's own fallback path already handles it as exactly that.
PROPRIETARY_LLM_ENDPOINT_REDIS_KEY: Final[str] = "llm:proprietary_endpoint"

# Must match, verbatim, the `--model` this project's own vLLM onstart_script loads (see
# scripts/deploy_custom_llm.py) -- vLLM's OpenAI-compatible server validates the request body's
# `model` field against what it actually loaded and rejects a mismatch. Shared here, once, so the
# deploy script and political_agent.py's own proprietary call can never drift apart.
PROPRIETARY_LLM_MODEL_NAME: Final[str] = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"

# Real, observed outcome of political_agent.py's most recent LLM call (one of "proprietary",
# "fallback", "unconfigured") + its timestamp, as a JSON string -- written by _assess_headline
# itself (best-effort, same "never fatal" discipline as its own heartbeat write) so the dashboard's
# "Statut de l'IA" indicator reflects what actually just happened, never a guess derived from
# PROPRIETARY_LLM_ENDPOINT_REDIS_KEY's mere presence (an endpoint can be configured yet still be
# unreachable -- only a real recent call outcome can honestly distinguish the two). No TTL: a stale
# value (older than roughly one poll cycle) is itself a real signal the agent has stopped scoring
# anything, surfaced via this key's own timestamp field, not by the key silently disappearing.
LLM_STATUS_REDIS_KEY: Final[str] = "llm:status"

# p2p_api.py's real time-travel history for sandbox.py's 30-day-delayed lookup -- one ZSET per
# fiat (real Redis "{prefix}{fiat}" key, e.g. "p2p:history_zset:ARS"), member = that scan's own
# JSON-serialized P2PCandidate, score = the scan's real unix timestamp. Never TTL'd (a sorted set
# has no per-member expiry) -- bounded instead via ZREMRANGEBYSCORE in the same write, see
# p2p_api.py's own _store_history_snapshot for why (same "don't let something grow unbounded
# forever" discipline the 2026-09-10 disk-space incident made non-negotiable in this project).
P2P_HISTORY_ZSET_KEY_PREFIX: Final[str] = "p2p:history_zset:"
# Real retention window: covers the 30-day sandbox lookback plus a real margin so a slightly late
# write (a missed cache refresh cycle, a brief outage) can never leave a fiat with zero history
# right when the sandbox needs it.
P2P_HISTORY_RETENTION_SECONDS: Final[int] = 40 * 24 * 60 * 60

# Real Redis Pub/Sub channel p2p_api.py's own _run_full_scan publishes its complete, real
# P2PPremiumsResponse JSON to, the instant a fresh scan completes -- p2p_websocket.py's own
# WebSocket endpoint subscribes to this and forwards each message to every connected client, real
# push instead of a client having to poll GET /api/v1/p2p-premiums on its own schedule. Same real
# payload shape either way -- a WebSocket client and a REST poller see byte-identical data, only
# the delivery mechanism differs. Best-effort, fire-and-forget (see _run_full_scan's own docstring
# on why a Redis hiccup here must never affect the real, synchronous REST response path).
P2P_PREMIUMS_PUBSUB_CHANNEL: Final[str] = "p2p:premiums:updates"

# Real, O(1)-readable snapshot of the *complete* latest P2PPremiumsResponse (one JSON blob),
# written by p2p_api.py's own _publish_scan_update alongside its existing Pub/Sub push -- neither
# the Pub/Sub channel above (a transient push, nothing to read once no one was listening) nor the
# per-fiat P2P_HISTORY_ZSET_KEY_PREFIX (a ranged ZSET query, not a single-key lookup) can answer
# "what is the current state right now" in one GET. src/graphql_api/schema.py's get_premium
# resolver is this key's real reader -- see that module's own docstring for why a GraphQL resolver
# specifically needs a real O(1) read here rather than triggering (or waiting on) a fresh scan.
P2P_LATEST_SNAPSHOT_REDIS_KEY: Final[str] = "p2p:premiums:latest_snapshot"

# Published by src.agents.whale_forensics.agent.WhaleForensicsDaemon (Protocole Fort Knox, 2026-09-17)
# the instant a real, live-observed ERC-20 Transfer event (decoded from a real Web3 event log, not
# simulated) moves more than WHALE_ALERT_THRESHOLD_USD notional into a watched address -- see that
# module's own docstring for the real, disclosed scope (a curated watchlist of known exchange/bridge
# addresses, not exhaustive chain-wide coverage). Consumed by src.agents.execution.agent.ExecutionAgent's
# Gate 8 (see ONCHAIN_TSUNAMI_LOCK_REDIS_KEY below) and rendered on fleet_command_nexus.py.
ONCHAIN_TSUNAMI_WARNING_TOPIC: Final[str] = "ONCHAIN_TSUNAMI_WARNING"

# Real Redis STRING key, short-TTL (see WhaleForensicsDaemon's own _LOCK_TTL_SECONDS), written
# alongside every real ONCHAIN_TSUNAMI_WARNING_TOPIC publish -- a bounded "anticipation window", not
# a second, independent kill switch (see core.security's own SYSTEM_STATUS, which this deliberately
# does NOT touch: a single large on-chain transfer is real signal, not the same category of
# already-confirmed danger AlertAgent's three conditions represent, and this project's own kill
# switch is human-reset-only by design -- auto-tripping it from an unattended on-chain listener would
# be a real, disproportionate operational risk). ExecutionAgent.Gate 8 checks this key's mere
# presence (EXISTS, not its value) the same way it already checks AgentControlState.OFF -- see that
# method's own docstring for the exact wiring.
ONCHAIN_TSUNAMI_LOCK_REDIS_KEY: Final[str] = "onchain_tsunami_lock"
ONCHAIN_TSUNAMI_LOCK_TTL_SECONDS: Final[int] = 600


class WhaleTransferAlert(BaseModel):
    """One real, decoded, over-threshold ERC-20 Transfer -- the payload of :data:`ONCHAIN_TSUNAMI_WARNING_TOPIC`.

    Attributes:
        timestamp: Timezone-aware UTC time this transfer was observed
            (real wall-clock at detection, not the block's own
            timestamp -- see WhaleForensicsDaemon's own docstring on why).
        chain: Which real chain this was observed on (e.g. ``"ethereum"``).
        token_symbol: The real, configured symbol for the token contract
            this event was decoded from (e.g. ``"WETH"``, ``"USDC"``) --
            never a token this daemon wasn't explicitly configured to
            watch (see its own watchlist-driven design).
        from_address: Real sender address (lowercased, EIP-55 not
            enforced -- comparison-only, never displayed as a checksum).
        to_address: Real recipient address.
        amount_usd: Real notional value, converted from the event's raw
            token amount using either a stablecoin's disclosed 1:1 USD
            assumption or a real, live mid-price already published by
            this project's own MarketDataAgent (see that conversion's
            own docstring for exactly which).
        tx_hash: Real transaction hash the event was emitted in.
        block_number: Real block number the transaction was mined in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chain: str = Field(min_length=1)
    token_symbol: str = Field(min_length=1)
    from_address: str = Field(min_length=1)
    to_address: str = Field(min_length=1)
    amount_usd: float = Field(gt=0)
    tx_hash: str = Field(min_length=1)
    block_number: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)


# Real port for src/grpc_service/grpc_server.py's TradeExecutor service -- the internal,
# binary-protocol EaaS layer (see that module's own docstring for its real, disclosed scope:
# a latency demonstration of the transport, not a second real execution engine).
GRPC_TRADE_EXECUTOR_PORT: Final[int] = 50051


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


class DenoisedPriceSignal(BaseModel):
    """One instrument's latest real FFT-denoised close-price window.

    Published by :class:`~src.agents.spectral_denoiser.agent.
    SpectralDenoiserAgent` on :data:`DENOISED_PRICE_SIGNAL_TOPIC`, and
    persisted under :data:`DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX` -- the
    same "ZMQ publish + Redis cache" pairing :class:`FeatureVector`
    already establishes.

    **This is the wire/cache shape, not the PyTorch tensor itself**:
    :meth:`~src.agents.spectral_denoiser.agent.SpectralDenoiserAgent.
    _to_feature_tensor` builds a real ``torch.Tensor`` from
    ``denoised_window`` for any in-process caller doing feature
    engineering directly against this agent -- but ZMQ/Redis only ever
    carry JSON (the same real serialization boundary every other
    schema/topic in this module already has), so a downstream consumer
    reading this off the wire reconstructs its own tensor from the plain
    float list here (``torch.tensor(signal.denoised_window,
    dtype=torch.float32)``), never receives one directly.

    Attributes:
        instrument: Which instrument this denoised window is for.
        timestamp: Unix epoch seconds this signal was computed at.
        denoised_window: The most recent ``publish_window_size`` real
            denoised close prices, chronological order (oldest first) --
            not the full FFT input window, to keep this message small for
            low-latency delivery; a consumer needing the full window
            reads TimescaleDB directly, the same real source this agent
            itself reads from.
        dominant_frequencies: Real dominant frequency components (cycles
            per second, derived from the real observed candle spacing --
            never an assumed fixed interval), the ``top_k_frequencies``
            most energetic components of the real FFT, descending by
            magnitude.
        dominant_magnitudes: Each entry in ``dominant_frequencies``'s real
            spectral magnitude, same order/length.
        noise_energy_removed_pct: Real fraction (0-100) of total spectral
            energy discarded by the top-K low-pass filter -- how much of
            the original signal was judged "noise" this cycle, not a
            fabricated confidence figure.
        window_size: How many real candles the FFT was computed over this
            cycle (the real input window size, which can be smaller than
            the agent's configured ``lookback_candles`` early in an
            instrument's history).
        cwt_dominant_period_seconds: "Signal, Physique, Algo" (2026-09-14)
            addition -- real, **time-localized** dominant real cycle
            length (in seconds), from a real Morlet Continuous Wavelet
            Transform evaluated only at the most recent real candle
            (see :func:`core.quant_math.morlet_wavelet_transform`).
            Complementary to (never a replacement for)
            ``dominant_frequencies`` above, which describes the whole
            real fetched window at once via a single, real Fourier
            transform -- this instead answers "what real cycle length
            dominates *right now*," honestly reframing (not escaping)
            the same Gabor-Heisenberg time/frequency tradeoff that
            single-window FFT is fixed to one point of. ``None`` when
            the real fetched window was too short to evaluate any real
            candidate scale.
        cwt_dominant_power: Real wavelet power at
            ``cwt_dominant_period_seconds`` -- how energetically that
            specific real cycle is currently present, not a normalized
            confidence score. ``None`` under the same condition as
            ``cwt_dominant_period_seconds``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str = Field(min_length=1)
    timestamp: float = Field(gt=0)
    denoised_window: list[float] = Field(min_length=1)
    dominant_frequencies: list[float] = Field(default_factory=list)
    dominant_magnitudes: list[float] = Field(default_factory=list)
    noise_energy_removed_pct: float = Field(ge=0, le=100)
    window_size: int = Field(gt=0)
    cwt_dominant_period_seconds: float | None = Field(default=None, gt=0)
    cwt_dominant_power: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_dominant_arrays_same_length(self) -> DenoisedPriceSignal:
        if len(self.dominant_frequencies) != len(self.dominant_magnitudes):
            raise ValueError("dominant_frequencies and dominant_magnitudes must be the same length.")
        return self


class OptimalExecutionSchedule(BaseModel):
    """One real Almgren-Chriss execution schedule + impact cost curve for a proposed meta-order.

    Cached under :data:`OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX` by
    :meth:`~src.agents.execution.agent.ExecutionAgent.
    compute_optimal_trajectory` -- the wire/cache shape of
    :class:`~src.agents.supervisor.optimal_execution_scheduler.
    ExecutionSchedule`, the same "internal dataclass vs. JSON-serializable
    cache model" split :class:`DenoisedPriceSignal` already establishes.
    Not published on ZMQ (unlike every other schema in this module) --
    this is a computed planning artifact for an operator/dashboard to
    consult, not an opinion another agent consensus-arbitrates over.

    Attributes:
        instrument: Which instrument this schedule was computed for.
        timestamp: Unix epoch seconds this schedule was computed at.
        order_size: The real total quantity this schedule was computed
            for.
        execution_horizon_seconds: The real total time this schedule
            spans.
        trajectory: Real remaining-inventory curve (``n_steps + 1``
            points).
        trade_sizes: Real per-slice trade quantity (``n_steps`` points),
            after the real participation-rate cap.
        temporary_impact: Real per-slice temporary price impact
            (``n_steps`` points).
        permanent_impact: Real cumulative permanent price impact
            (``n_steps`` points).
        was_participation_capped: Whether real expected volume was too
            thin to complete ``order_size`` within the horizon at the
            uncapped rate -- see :class:`~src.agents.supervisor.
            optimal_execution_scheduler.ExecutionSchedule`'s own
            docstring.
        total_estimated_temporary_cost: See that same class's docstring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str = Field(min_length=1)
    timestamp: float = Field(gt=0)
    order_size: float = Field(gt=0)
    execution_horizon_seconds: float = Field(gt=0)
    trajectory: list[float] = Field(min_length=2)
    trade_sizes: list[float] = Field(min_length=1)
    temporary_impact: list[float] = Field(min_length=1)
    permanent_impact: list[float] = Field(min_length=1)
    was_participation_capped: bool
    total_estimated_temporary_cost: float = Field(ge=0)


class OrderSlicingProgress(BaseModel):
    """Real, live (or final) progress of one in-flight TWAP slicing execution -- an actual execution, not a plan.

    Cached under :data:`ORDER_SLICING_PROGRESS_REDIS_KEY_PREFIX` by
    :meth:`~src.agents.execution.agent.ExecutionAgent._execute_sliced_entry`
    -- see that method's own docstring for the real "Protocole Stealth
    Slicing" mechanism this reports on (real TWAP + jitter, wired into
    the live order-placement path for any entry whose notional exceeds
    the configured discretion threshold). Distinct from
    :class:`OptimalExecutionSchedule`: that one is a computed
    recommendation nothing has actually executed; every slice this
    model counts already placed a real (or, before Gate 8 resolved to a
    live order, this path is never reached at all) broker order.

    Attributes:
        instrument: Which instrument this slicing execution is for.
        direction: ``1`` (BUY/long) or ``-1`` (SELL/short) -- matches
            :meth:`~src.agents.execution.agent.ExecutionAgent.
            _place_order`'s own ``direction`` convention.
        timestamp: Unix epoch seconds this progress snapshot was
            written at.
        slices_completed: How many real slices have placed a real
            order so far.
        slices_planned: Total real slices this execution was planned
            for.
        percent_complete: ``100 * slices_completed / slices_planned``,
            precomputed for direct display (e.g. "34% (17/50 tranches
            tirées)").
        status: ``"IN_PROGRESS"`` while slices are still being placed;
            ``"COMPLETED"`` once every planned slice filled;
            ``"ABORTED"`` if it stopped early (Kill Switch, operator
            control-state change, or a concurrent real exit -- see
            :class:`~src.agents.execution.order_slicer.
            SlicingAbortedError`) before every planned slice filled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str = Field(min_length=1)
    direction: int = Field(ge=-1, le=1)
    timestamp: float = Field(gt=0)
    slices_completed: int = Field(ge=0)
    slices_planned: int = Field(gt=0)
    percent_complete: float = Field(ge=0, le=100)
    status: Literal["IN_PROGRESS", "COMPLETED", "ABORTED"]

    @field_validator("direction")
    @classmethod
    def _validate_direction_nonzero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("direction must be -1 or 1, never 0 (neutral has no real order to slice).")
        return value


class OptionContract(BaseModel):
    """A real options contract specification -- not a live market/broker connection.

    Real, disclosed execution boundary: this project has no real options
    broker connection (OANDA is FX/CFD spot only) -- see
    :class:`~src.agents.supervisor.options_hedging_manager.
    OptionsHedgingManager`'s own module docstring for why every position
    against this contract is necessarily a simulated/paper fill, the same
    honest MOCK_MODE/SHADOW convention every other broker gap in this
    project already discloses, never a real order.

    Attributes:
        underlying: The real underlying instrument (e.g. ``"BTC/USDT"``).
        strike: Real strike price. Must be positive.
        expiry: Real, timezone-aware expiry timestamp. Must be in the future.
        option_type: ``"call"`` or ``"put"``.
        contract_size: Real underlying units per contract (e.g. Deribit
            crypto options are typically 1 underlying unit per contract).
            Must be positive.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    underlying: str = Field(min_length=1)
    strike: float = Field(gt=0)
    expiry: datetime
    option_type: Literal["call", "put"]
    contract_size: float = Field(gt=0, default=1.0)

    @field_validator("expiry")
    @classmethod
    def _validate_expiry(cls, value: datetime) -> datetime:
        return _require_timezone_aware(value)


class OptionPosition(BaseModel):
    """One real (necessarily paper-filled) options position -- see :class:`OptionContract`'s own docstring.

    Attributes:
        contract: The real contract specification.
        quantity: Real signed number of contracts (positive = long,
            negative = short). Must not be zero.
        entry_premium: Real per-contract premium at entry (from a real
            Black-Scholes evaluation against real market inputs at the
            time -- see :meth:`~src.agents.supervisor.
            options_hedging_manager.OptionsHedgingManager.open_position`).
            Must be ``>= 0``.
        entry_timestamp: Unix epoch seconds this position was opened at.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: OptionContract
    quantity: float
    entry_premium: float = Field(ge=0)
    entry_timestamp: float = Field(gt=0)

    @field_validator("quantity")
    @classmethod
    def _validate_quantity_nonzero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("quantity must not be zero.")
        return value


class PortfolioGreeks(BaseModel):
    """Real aggregate Greeks across every tracked position (options + underlying spot/CFD).

    Cached under :data:`PORTFOLIO_GREEKS_REDIS_KEY_PREFIX` -- Redis-only,
    like :class:`OptimalExecutionSchedule`, not a ZMQ-published opinion.

    Attributes:
        timestamp: Unix epoch seconds this snapshot was computed at.
        delta: Real aggregate delta (a real spot/CFD position of size
            :math:`x` contributes exactly :math:`x` -- its own real,
            trivial "delta").
        gamma: Real aggregate gamma (spot/CFD positions contribute
            exactly ``0`` -- gamma is a real second-derivative property
            options alone carry in this system).
        vega: Real aggregate vega (same real spot/CFD ``0``-contribution
            reasoning as gamma).
        theta: Real aggregate theta.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(gt=0)
    delta: float
    gamma: float
    vega: float
    theta: float

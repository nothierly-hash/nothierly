"""Chapter 13, Étape 5.8, Pilier 1: a cross-asset order/position/instrument model.

**Scope, stated honestly upfront**: this project touches exactly three
real broker integrations today -- crypto spot (via ``ccxt``,
:class:`~src.execution.broker_gateway.CcxtBrokerAdapter`), forex/
commodity CFDs (via OANDA, :class:`~src.execution.broker_gateway.
OandaBrokerAdapter` -- symbol-agnostic, so it already places a real
:data:`AssetClass.COMMODITY` order (``XAU_USD``, ``BCO_USD``, ...)
exactly the same way it places a ``FOREX`` one, see
``core/asset_universe.py`` for the real registry that actually
classifies them that way), and equities (via Interactive Brokers,
:class:`~src.execution.broker_gateway.IBKRBrokerAdapter`, which can
also reach forex/crypto through IBKR itself). This module's real job is
unifying *those*, each of which arrives from its underlying SDK
as a loosely-typed ``dict[str, Any]`` (raw ``ccxt``, raw OANDA v20, raw
``ib_async``) with a different field name for the same concept (ccxt's
``amount`` vs. OANDA's ``units`` vs. ``ib_async``'s ``totalQuantity``,
for one example) -- :class:`UniversalOrder`/:class:`UniversalPosition`
give every consumer in this codebase (the cross-margin engine, a future
risk dashboard, the billing engine) exactly one shape to read,
regardless of which broker produced it. Crypto perpetuals/futures and
listed options are modeled here as real :class:`AssetClass` values
because the schema generalizes to them cleanly, but **no adapter in
this project actually places a perpetual or options order today** --
included for the schema's own completeness, not as a claim that
trading them already works end to end.

**On "surpasses ccxt or the Bloomberg API"**: it doesn't, and this
module doesn't claim to -- ``ccxt`` alone normalizes 100+ real
exchanges' wire protocols with a team and years behind it; the
Bloomberg Terminal is an entire market-data/analytics/compliance
platform. What this module actually is: the one place *this specific
codebase's* three real broker integrations get normalized into a
single internal representation, so code that reasons about "a
position" or "an order" (the margin engine below, for one) never has
to branch on which broker it came from. A real, narrow, useful
improvement over three raw ``dict[str, Any]`` shapes -- not a
competitor to either of those systems.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class AssetClass(str, Enum):
    """Every asset class this ontology can describe -- see module docstring on which are actually tradeable here today."""

    CRYPTO_SPOT = "CRYPTO_SPOT"
    CRYPTO_PERPETUAL = "CRYPTO_PERPETUAL"  # schema-complete; no adapter places these today
    FOREX = "FOREX"
    EQUITY = "EQUITY"
    COMMODITY = "COMMODITY"  # real OANDA CFDs (XAU_USD, BCO_USD, ...) -- see core/asset_universe.py's
    # module docstring: OandaBrokerAdapter already places these today (it is symbol-agnostic), they
    # were just previously left unclassified/mislabeled as FOREX instead of getting their own value.


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class UniversalAsset(BaseModel):
    """One instrument, described the same way regardless of venue.

    The real, concrete reason ``tick_size``/``lot_size`` both exist as
    separate concepts (not collapsed into one "precision" field): a
    forex lot (OANDA, standard lot = 100,000 units, tradeable in whole
    units) and a crypto tick (ccxt, e.g. BTC/USDT priced to 1 cent,
    tradeable in fractional BTC) constrain two *different* things --
    ``tick_size`` bounds the **price** granularity a limit order can
    specify; ``lot_size`` bounds the **quantity** granularity an order
    can be sized in. Collapsing them (as a naive single-asset-class
    model might) would silently misdescribe whichever asset class the
    model wasn't originally built for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Venue-native symbol, e.g. 'BTC/USDT', 'EUR_USD', 'AAPL'.")
    asset_class: AssetClass
    base_currency: str = Field(min_length=1, description="e.g. 'BTC' for BTC/USDT, 'EUR' for EUR_USD, 'AAPL' for a share.")
    quote_currency: str = Field(min_length=1, description="e.g. 'USDT' for BTC/USDT, 'USD' for EUR_USD or AAPL.")
    tick_size: Decimal = Field(gt=0, description="Minimum real price increment, e.g. Decimal('0.01').")
    lot_size: Decimal = Field(gt=0, description="Minimum real quantity increment, e.g. Decimal('0.00001') BTC, Decimal('1') for a stock share, Decimal('100000') for one OANDA standard lot.")
    contract_multiplier: Decimal = Field(
        default=Decimal(1),
        gt=0,
        description="1 for spot/equity/forex-in-units; the real per-contract unit count for a future/perpetual (unused by any adapter today, kept for schema completeness -- see module docstring).",
    )


class UniversalOrder(BaseModel):
    """One order, described the same way regardless of which broker it was (or will be) placed on.

    ``broker_native_order_id``/``broker_id`` are kept alongside the
    universal fields deliberately, not discarded: a real operator
    debugging a discrepancy needs to trace back to the exact ccxt/OANDA
    /``ib_async`` record this came from, and this model's whole point is
    normalization *for reasoning*, not erasure of the real provenance.

    **Algo-tagging fields** (``algo_id``/``decision_timestamp_utc``/
    ``regulatory_client_shortcode``) mirror the real shape MiFID II's
    RTS 6 (algorithmic-trading real-time monitoring, requires a
    per-algorithm identifier) and RTS 22 (transaction reporting,
    requires a short code identifying the person/algorithm responsible
    for the investment decision) actually ask for -- stated precisely
    so this isn't overclaimed: three required fields on one internal
    ontology model make orders *taggable* for a real regulatory report;
    they are not, by themselves, a MiFID II transaction-reporting
    pipeline (no such reporting/submission mechanism exists anywhere in
    this codebase). ``decision_timestamp_utc`` is deliberately distinct
    from ``submitted_at`` below: RTS 25 cares about when the algorithm
    *decided*, which can precede actual submission (e.g. a TWAP child
    order sliced and queued before it is actually sent).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1, description="This system's own order identifier (e.g. a fresh UUID4 at submission).")
    broker_native_order_id: str | None = Field(default=None, description="The broker's own order ID, once known -- None before submission ack.")
    broker_id: str = Field(min_length=1, description="Which BaseBroker produced/will execute this -- 'oanda', 'binance', 'ibkr', ...")
    client_id: str | None = Field(default=None, description="The B2B client this order belongs to, if any (see api_middleware/gateway.py).")
    algo_id: str = Field(min_length=1, description="Which Nothierly algorithm produced this order, e.g. 'RL_MAKER_V2', 'AI_TWAP' -- RTS 6-style algo identifier.")
    decision_timestamp_utc: datetime = Field(description="When the algorithm decided to place this order (microsecond precision), not when it was submitted -- see class docstring.")
    regulatory_client_shortcode: str = Field(min_length=1, description="RTS 22-style short code for the person/entity responsible for the investment decision -- distinct from the internal client_id.")
    asset: UniversalAsset
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0, description="Requested size, in the asset's own base-currency units (never lots pre-multiplied).")
    limit_price: Decimal | None = Field(
        default=None, validate_default=True, description="Required for LIMIT orders, None for MARKET."
    )
    status: OrderStatus
    filled_quantity: Decimal = Field(default=Decimal(0), ge=0)
    avg_fill_price: Decimal | None = Field(default=None, ge=0)
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("submitted_at", "decision_timestamp_utc")
    @classmethod
    def _require_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("submitted_at/decision_timestamp_utc must be timezone-aware.")
        return value

    @field_validator("limit_price")
    @classmethod
    def _limit_price_required_for_limit_orders(cls, value: Decimal | None, info: ValidationInfo) -> Decimal | None:
        if info.data.get("order_type") == OrderType.LIMIT and value is None:
            raise ValueError("limit_price is required when order_type is LIMIT.")
        return value


class UniversalPosition(BaseModel):
    """One open position, described the same way regardless of venue.

    ``quantity`` is signed (positive = long, negative = short) -- the
    one sign convention every consumer of this model can rely on,
    instead of each broker's own convention (OANDA signs ``units``
    itself; ccxt spot balances are always positive with side implied by
    which currency you're holding; ``ib_async`` positions carry their
    own signed ``position`` field already). ``margin_used_usd`` is
    ``None`` for a genuinely unmargined position (a plain crypto spot
    holding) -- see :mod:`risk.cross_margin_engine`'s own docstring for
    why that distinction is load-bearing, not just a formality.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_id: str = Field(min_length=1)
    asset: UniversalAsset
    quantity: Decimal = Field(description="Signed: positive = long, negative = short.")
    avg_entry_price: Decimal = Field(gt=0)
    mark_price: Decimal = Field(gt=0, description="Current real price this position is marked at.")
    realized_pnl_usd: Decimal = Field(default=Decimal(0))
    unrealized_pnl_usd: Decimal = Field(description="(mark_price - avg_entry_price) * quantity, in USD-equivalent terms.")
    margin_used_usd: Decimal | None = Field(
        default=None, description="None for an unmargined (spot) position -- a real distinct state, not zero."
    )

    @property
    def side(self) -> Literal["LONG", "SHORT"]:
        return "LONG" if self.quantity >= 0 else "SHORT"

    @property
    def notional_usd(self) -> Decimal:
        return abs(self.quantity) * self.mark_price


class MarginSnapshot(BaseModel):
    """One broker's real account balance/margin state at one instant -- see :mod:`risk.cross_margin_engine`'s own docstring for why the ``None`` case matters.

    ``margin_used_usd``/``margin_available_usd``/``margin_utilization_pct``
    are all ``None`` together for a venue with no margin concept at all
    (this project's crypto spot trading via
    :class:`~src.execution.broker_gateway.CcxtBrokerAdapter`) -- a real,
    distinct state from "0% utilized", which would wrongly imply margin
    exists and is merely unused right now.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_id: str = Field(min_length=1)
    account_balance_usd: Decimal = Field(ge=0)
    margin_used_usd: Decimal | None = Field(default=None, ge=0)
    margin_available_usd: Decimal | None = Field(default=None, ge=0)

    @property
    def margin_utilization_pct(self) -> Decimal | None:
        """``margin_used / (margin_used + margin_available) * 100``, or ``None`` for an unmargined venue."""
        if self.margin_used_usd is None or self.margin_available_usd is None:
            return None
        total = self.margin_used_usd + self.margin_available_usd
        if total == 0:
            return Decimal(0)
        return (self.margin_used_usd / total) * Decimal(100)


# =============================================================================
# Real normalizers -- raw broker dict -> UniversalOrder/UniversalPosition.
# Deliberately live here (not scattered across each adapter): one place to
# audit every field-name mapping this project actually relies on.
# =============================================================================


def order_from_ccxt(
    raw: dict[str, Any],
    *,
    broker_id: str,
    asset: UniversalAsset,
    algo_id: str,
    decision_timestamp_utc: datetime,
    regulatory_client_shortcode: str,
    client_id: str | None = None,
) -> UniversalOrder:
    """Normalize one raw ``ccxt`` order/fill structure (as returned by ``AsyncExchangeClient``'s order methods).

    ``algo_id``/``decision_timestamp_utc``/``regulatory_client_shortcode``
    have no real source in a raw ccxt dict -- a broker has no concept of
    which Nothierly algorithm decided to trade, or a regulatory short
    code. Required keyword args here, same as ``client_id`` already was,
    not defaulted: the caller (whichever agent/order-flow actually knows
    this context) must supply real values, never a fabricated one.
    """
    status_map = {"open": OrderStatus.OPEN, "closed": OrderStatus.FILLED, "canceled": OrderStatus.CANCELLED, "rejected": OrderStatus.REJECTED}
    raw_status = str(raw.get("status") or "closed").lower()
    return UniversalOrder(
        order_id=str(raw.get("clientOrderId") or raw.get("id")),
        broker_native_order_id=str(raw.get("id")) if raw.get("id") is not None else None,
        broker_id=broker_id,
        client_id=client_id,
        algo_id=algo_id,
        decision_timestamp_utc=decision_timestamp_utc,
        regulatory_client_shortcode=regulatory_client_shortcode,
        asset=asset,
        side=OrderSide.BUY if str(raw.get("side", "buy")).lower() == "buy" else OrderSide.SELL,
        order_type=OrderType.LIMIT if str(raw.get("type", "market")).lower() == "limit" else OrderType.MARKET,
        quantity=Decimal(str(raw.get("amount", 0))),
        limit_price=Decimal(str(raw["price"])) if raw.get("price") is not None else None,
        status=status_map.get(raw_status, OrderStatus.OPEN),
        filled_quantity=Decimal(str(raw.get("filled", 0))),
        avg_fill_price=Decimal(str(raw["average"])) if raw.get("average") is not None else None,
    )


def order_from_oanda(
    raw: dict[str, Any],
    *,
    asset: UniversalAsset,
    algo_id: str,
    decision_timestamp_utc: datetime,
    regulatory_client_shortcode: str,
    client_id: str | None = None,
) -> UniversalOrder:
    """Normalize one raw OANDA order-fill dict (as returned by ``OandaClient.create_market_order``). See order_from_ccxt for the new required args' rationale."""
    units = Decimal(str(raw.get("units", 0)))
    return UniversalOrder(
        order_id=str(raw.get("id")),
        broker_native_order_id=str(raw.get("id")),
        broker_id="oanda",
        client_id=client_id,
        algo_id=algo_id,
        decision_timestamp_utc=decision_timestamp_utc,
        regulatory_client_shortcode=regulatory_client_shortcode,
        asset=asset,
        side=OrderSide.BUY if units >= 0 else OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=abs(units),
        status=OrderStatus.FILLED,
        filled_quantity=abs(units),
        avg_fill_price=Decimal(str(raw["price"])) if raw.get("price") is not None else None,
    )

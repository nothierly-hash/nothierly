"""Pydantic response models for the Nothierly SDK.

Every model uses ``ConfigDict(extra="ignore")`` deliberately, unlike the
backend's own ``extra="forbid"`` models these mirror: this SDK validates
only the shape a client's code actually depends on, not the full response
byte-for-byte. The backend adding a new field in a future release must
never break an SDK version already shipped to an institutional client
(forward compatibility, per this mission's own requirement) -- ``forbid``
would turn every backend addition into a breaking SDK change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class P2PCandidate(BaseModel):
    """One country's real, live P2P premium reading.

    Mirrors ``p2p_api.py``'s own ``P2PCandidate`` field-for-field -- see
    that module's docstring for exactly how each value is computed.
    """

    model_config = ConfigDict(extra="ignore")

    country_fiat: str
    p2p_price: float
    forex_rate: float
    usdt_spot_price: float
    premium_percentage: float
    p2p_ads_sampled: int
    usd_equivalent_price: float
    avg_liquidity_usdt: float
    avg_merchant_completion_rate: float
    avg_merchant_order_count: float


class CrossCountryOpportunity(BaseModel):
    """One real pairwise USD-equivalent spread between two countries' P2P markets.

    A price signal, not proof of executable arbitrage -- see
    ``p2p_api.py``'s own ``CrossCountryOpportunity`` docstring for the
    real, disclosed execution gap (no cross-border payment access is
    implied).
    """

    model_config = ConfigDict(extra="ignore")

    buy_country_fiat: str
    sell_country_fiat: str
    buy_usd_equivalent: float
    sell_usd_equivalent: float
    spread_percentage: float


class P2PPremiumsResponse(BaseModel):
    """Full ``GET /api/v1/p2p-premiums`` response."""

    model_config = ConfigDict(extra="ignore")

    timestamp: str
    top_candidates: list[P2PCandidate] = Field(default_factory=list)
    cross_country_opportunities: list[CrossCountryOpportunity] = Field(default_factory=list)
    # currency -> human-readable reason it was excluded (no real Binance P2P/forex data found for
    # it) -- a dict, not a list, matching the real backend's own P2PPremiumsResponse.excluded.
    excluded: dict[str, str] = Field(default_factory=dict)
    cache_age_seconds: float = 0.0


class MacroEvent(BaseModel):
    """One historical crisis-scoring event for a single currency.

    Mirrors ``political_agent.py``'s own real ``CrisisAssessment`` model --
    the shape actually written to the ``macro:global_events`` Redis list.
    """

    model_config = ConfigDict(extra="ignore")

    currency_affected: str
    crisis_score: float
    sentiment: Literal["bearish", "bullish"]
    event_summary: str


class MacroSentimentResponse(BaseModel):
    """Real-time + historical macro/political risk read for one fiat currency.

    **Disclosure**: as of this SDK version, no Nothierly API endpoint
    serves this yet -- ``political_agent.py`` already computes and stores
    this exact data in Redis (``macro:risk_score:{currency}``,
    ``macro:global_events``), but nothing exposes it over REST. This model
    is real and ready for when that endpoint ships; see
    :meth:`nothierly.client.NothierlyClient.get_macro_sentiment`'s own
    docstring.
    """

    model_config = ConfigDict(extra="ignore")

    fiat: str
    current_risk_score: float | None = None
    recent_events: list[MacroEvent] = Field(default_factory=list)


class LiquidityDepthLevel(BaseModel):
    """One price level of an order book side."""

    model_config = ConfigDict(extra="ignore")

    price: float
    size: float


class LiquidityDepthResponse(BaseModel):
    """L2 order-book depth snapshot for one symbol.

    **Disclosure**: no Nothierly API endpoint serves this yet either --
    modeled here against the standard bids/asks price-level shape every
    execution venue uses (a universal convention, not a Nothierly-specific
    contract), purely for this SDK's forward-compatibility requirement.
    """

    model_config = ConfigDict(extra="ignore")

    symbol: str
    bids: list[LiquidityDepthLevel] = Field(default_factory=list)
    asks: list[LiquidityDepthLevel] = Field(default_factory=list)
    timestamp: str | None = None

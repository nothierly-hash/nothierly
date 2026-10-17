"""Real, executable tests for pricing/pre_trade_oracle.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from core.nothierly_ontology import AssetClass, OrderSide, OrderStatus, OrderType, UniversalAsset, UniversalOrder
from core.schemas import OrderBookLevel, OrderBookSnapshot
from pricing.pre_trade_oracle import FeeTier, estimate_total_implementation_cost, simulate_cost_curve

_BTC = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)
_BTC_PERP = UniversalAsset(
    symbol="BTC-PERP", asset_class=AssetClass.CRYPTO_PERPETUAL, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.001"),
)


def _order(*, side: OrderSide = OrderSide.BUY, quantity: Decimal = Decimal("1.5"), order_type: OrderType = OrderType.MARKET, asset: UniversalAsset = _BTC, limit_price: Decimal | None = None) -> UniversalOrder:
    return UniversalOrder(
        order_id="o1", broker_id="binance", asset=asset, side=side, order_type=order_type,
        quantity=quantity, limit_price=limit_price, status=OrderStatus.PENDING,
        algo_id="AI_TWAP", decision_timestamp_utc=datetime.now(timezone.utc), regulatory_client_shortcode="CLI001",
    )


def _book(*, bid: float = 64990, ask: float = 65000, depth: int = 2) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=bid - i * 10, volume=1.0) for i in range(depth)],
        asks=[OrderBookLevel(price=ask + i * 10, volume=1.0) for i in range(depth)],
    )


def test_market_buy_walks_the_real_ask_side() -> None:
    book = _book(depth=2)  # asks: 65000@1.0, 65010@1.0
    order = _order(quantity=Decimal("1.5"))
    quote = estimate_total_implementation_cost(order, book)
    assert quote.filled_quantity == Decimal("1.5")
    assert quote.unfilled_quantity == 0
    expected_avg = (Decimal("65000") * 1 + Decimal("65010") * Decimal("0.5")) / Decimal("1.5")
    assert quote.simulated_avg_fill_price == expected_avg


def test_market_sell_walks_the_real_bid_side() -> None:
    book = _book(depth=2)  # bids: 64990@1.0, 64980@1.0
    order = _order(side=OrderSide.SELL, quantity=Decimal("1.5"))
    quote = estimate_total_implementation_cost(order, book)
    expected_avg = (Decimal("64990") * 1 + Decimal("64980") * Decimal("0.5")) / Decimal("1.5")
    assert quote.simulated_avg_fill_price == expected_avg


def test_partial_fill_when_book_depth_insufficient() -> None:
    book = _book(depth=2)  # only 2.0 total ask volume
    order = _order(quantity=Decimal("5.0"))
    quote = estimate_total_implementation_cost(order, book)
    assert quote.filled_quantity == Decimal("2.0")
    assert quote.unfilled_quantity == Decimal("3.0")
    assert any("insuffisante" in note for note in quote.notes)


def test_single_level_book_still_fills_correctly() -> None:
    """core.schemas.OrderBookSnapshot enforces min_length=1 on both sides -- a real, valid snapshot can
    never have zero levels, so InsufficientLiquidityError's guard is real defensive code (kept for a
    caller that bypasses pydantic validation) but is not reachable through this constructor, and this
    test does not claim otherwise -- it only verifies the real one-level boundary case fills correctly."""
    book = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=64990, volume=1.0)], asks=[OrderBookLevel(price=65000, volume=1.0)],
    )
    order = _order(quantity=Decimal("0.5"))
    quote = estimate_total_implementation_cost(order, book)
    assert quote.filled_quantity == Decimal("0.5")


def test_limit_order_has_no_market_impact() -> None:
    book = _book()
    order = _order(order_type=OrderType.LIMIT, quantity=Decimal("1.0"), limit_price=Decimal("64995"))
    quote = estimate_total_implementation_cost(order, book)
    assert quote.market_impact_usd == 0
    assert quote.simulated_avg_fill_price == Decimal("64995")
    assert any("LIMIT" in note for note in quote.notes)


def test_fee_tier_resolution_real_volume_thresholds() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"))
    low_volume_quote = estimate_total_implementation_cost(order, book, client_30day_volume_usd=Decimal(0))
    high_volume_quote = estimate_total_implementation_cost(order, book, client_30day_volume_usd=Decimal("100000000"))
    assert high_volume_quote.exchange_fee_usd < low_volume_quote.exchange_fee_usd


def test_jurisdiction_none_means_zero_tax() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"))
    quote = estimate_total_implementation_cost(order, book, jurisdiction=None)
    assert quote.tax_usd == 0


def test_unknown_jurisdiction_is_zero_tax_with_a_real_note() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"))
    quote = estimate_total_implementation_cost(order, book, jurisdiction="ZZ")
    assert quote.tax_usd == 0
    assert any("ZZ" in note for note in quote.notes)


def test_uk_jurisdiction_applies_real_published_rate() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"))
    quote = estimate_total_implementation_cost(order, book, jurisdiction="UK")
    expected_tax = quote.simulated_avg_fill_price * Decimal("1.0") * Decimal("0.005")
    assert quote.tax_usd == expected_tax


def test_funding_cost_zero_for_spot_even_if_rate_supplied() -> None:
    """Spot has no funding concept -- a supplied rate must be ignored, not silently applied."""
    book = _book()
    order = _order(quantity=Decimal("1.0"))  # _BTC = CRYPTO_SPOT
    quote = estimate_total_implementation_cost(order, book, funding_rate_per_8h=Decimal("0.0001"), holding_period_hours=Decimal("24"))
    assert quote.funding_cost_usd == 0
    assert any("aucun concept de funding" in note for note in quote.notes)


def test_funding_cost_applies_for_real_perpetual_with_rate_supplied() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"), asset=_BTC_PERP)
    quote = estimate_total_implementation_cost(order, book, funding_rate_per_8h=Decimal("0.0001"), holding_period_hours=Decimal("24"))
    # 24h / 8h = 3 real funding periods
    expected = quote.simulated_avg_fill_price * Decimal("1.0") * Decimal("0.0001") * Decimal(3)
    assert quote.funding_cost_usd == expected
    assert quote.funding_cost_usd > 0


def test_funding_cost_zero_for_perpetual_without_rate_supplied() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"), asset=_BTC_PERP)
    quote = estimate_total_implementation_cost(order, book)
    assert quote.funding_cost_usd == 0
    assert any("funding" in note.lower() for note in quote.notes)


def test_break_even_price_above_fill_price_for_a_buy() -> None:
    book = _book()
    order = _order(side=OrderSide.BUY, quantity=Decimal("1.0"))
    quote = estimate_total_implementation_cost(order, book, jurisdiction="FR")
    assert quote.break_even_price > quote.simulated_avg_fill_price


def test_break_even_price_below_fill_price_for_a_sell() -> None:
    book = _book()
    order = _order(side=OrderSide.SELL, quantity=Decimal("1.0"))
    quote = estimate_total_implementation_cost(order, book, jurisdiction="FR")
    assert quote.break_even_price < quote.simulated_avg_fill_price


def test_custom_fee_tiers_are_honored() -> None:
    book = _book()
    order = _order(quantity=Decimal("1.0"))
    zero_fee_tier = (FeeTier(min_30day_volume_usd=Decimal(0), taker_fee_rate=Decimal(0), maker_fee_rate=Decimal(0)),)
    quote = estimate_total_implementation_cost(order, book, fee_tiers=zero_fee_tier)
    assert quote.exchange_fee_usd == 0


# -- simulate_cost_curve (numpy) --------------------------------


def test_simulate_cost_curve_returns_real_costs_within_depth() -> None:
    book = _book(depth=3)  # 3.0 total ask volume
    quantities = np.array([0.5, 1.0, 2.0])
    costs_bps = simulate_cost_curve(OrderSide.BUY, book, quantities)
    assert costs_bps.shape == quantities.shape
    assert not np.isnan(costs_bps).any()
    # Real, monotonic: a bigger order walks further up the book -> more impact.
    assert costs_bps[0] <= costs_bps[1] <= costs_bps[2]


def test_simulate_cost_curve_returns_nan_beyond_real_captured_depth() -> None:
    book = _book(depth=2)  # 2.0 total ask volume
    quantities = np.array([1.0, 5.0])  # 5.0 exceeds real depth
    costs_bps = simulate_cost_curve(OrderSide.BUY, book, quantities)
    assert not np.isnan(costs_bps[0])
    assert np.isnan(costs_bps[1])  # never a fabricated extrapolated cost


def test_simulate_cost_curve_sell_side_uses_real_bids() -> None:
    book = _book(depth=2)
    quantities = np.array([0.5])
    buy_costs = simulate_cost_curve(OrderSide.BUY, book, quantities)
    sell_costs = simulate_cost_curve(OrderSide.SELL, book, quantities)
    assert buy_costs[0] > 0
    assert sell_costs[0] > 0

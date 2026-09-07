"""Real, executable tests for core/nothierly_ontology.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.nothierly_ontology import (
    AssetClass,
    MarginSnapshot,
    OrderSide,
    OrderStatus,
    OrderType,
    UniversalAsset,
    UniversalOrder,
    UniversalPosition,
    order_from_ccxt,
    order_from_oanda,
)

_BTC = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)
_NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_margin_snapshot_none_for_unmargined_venue() -> None:
    snapshot = MarginSnapshot(broker_id="binance", account_balance_usd=Decimal("10000"))
    assert snapshot.margin_used_usd is None
    assert snapshot.margin_available_usd is None
    assert snapshot.margin_utilization_pct is None


def test_margin_snapshot_computes_real_utilization_pct() -> None:
    snapshot = MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"),
        margin_used_usd=Decimal("2500"), margin_available_usd=Decimal("7500"),
    )
    assert snapshot.margin_utilization_pct == Decimal("25")


def test_margin_snapshot_handles_zero_total_margin_without_division_error() -> None:
    snapshot = MarginSnapshot(broker_id="oanda", account_balance_usd=Decimal("0"), margin_used_usd=Decimal("0"), margin_available_usd=Decimal("0"))
    assert snapshot.margin_utilization_pct == Decimal("0")


def test_universal_position_side_and_notional() -> None:
    long_position = UniversalPosition(
        broker_id="binance", asset=_BTC, quantity=Decimal("1.5"), avg_entry_price=Decimal("60000"),
        mark_price=Decimal("65000"), unrealized_pnl_usd=Decimal("7500"),
    )
    assert long_position.side == "LONG"
    assert long_position.notional_usd == Decimal("97500")

    short_position = long_position.model_copy(update={"quantity": Decimal("-1.5")})
    assert short_position.side == "SHORT"
    assert short_position.notional_usd == Decimal("97500")  # abs()


def test_universal_order_limit_requires_limit_price() -> None:
    with pytest.raises(ValidationError):
        UniversalOrder(
            order_id="o1", broker_id="binance", asset=_BTC, side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal("1"), status=OrderStatus.OPEN,
            algo_id="RL_MAKER_V2", decision_timestamp_utc=_NOW, regulatory_client_shortcode="CLI001",
        )


def test_universal_order_market_does_not_require_limit_price() -> None:
    order = UniversalOrder(
        order_id="o1", broker_id="binance", asset=_BTC, side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal("1"), status=OrderStatus.FILLED,
        algo_id="RL_MAKER_V2", decision_timestamp_utc=_NOW, regulatory_client_shortcode="CLI001",
    )
    assert order.limit_price is None


def test_universal_order_rejects_naive_decision_timestamp() -> None:
    with pytest.raises(ValidationError):
        UniversalOrder(
            order_id="o1", broker_id="binance", asset=_BTC, side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("1"), status=OrderStatus.FILLED,
            algo_id="RL_MAKER_V2", decision_timestamp_utc=datetime(2026, 9, 3, 12, 0, 0),  # no tzinfo
            regulatory_client_shortcode="CLI001",
        )


def test_universal_order_decision_timestamp_keeps_microsecond_precision() -> None:
    precise = datetime(2026, 9, 3, 12, 0, 0, 123456, tzinfo=timezone.utc)
    order = UniversalOrder(
        order_id="o1", broker_id="binance", asset=_BTC, side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal("1"), status=OrderStatus.FILLED,
        algo_id="RL_MAKER_V2", decision_timestamp_utc=precise, regulatory_client_shortcode="CLI001",
    )
    assert order.decision_timestamp_utc.microsecond == 123456


def test_order_from_ccxt_maps_real_fields() -> None:
    raw = {"id": "abc", "clientOrderId": "my-1", "side": "sell", "type": "limit", "amount": 0.25, "price": 65500.0, "filled": 0.25, "average": 65500.0, "status": "closed"}
    order = order_from_ccxt(
        raw, broker_id="bybit", asset=_BTC,
        algo_id="AI_TWAP", decision_timestamp_utc=_NOW, regulatory_client_shortcode="CLI001",
    )
    assert order.order_id == "my-1"
    assert order.broker_native_order_id == "abc"
    assert order.broker_id == "bybit"
    assert order.side == OrderSide.SELL
    assert order.quantity == Decimal("0.25")
    assert order.status == OrderStatus.FILLED
    assert order.algo_id == "AI_TWAP"
    assert order.regulatory_client_shortcode == "CLI001"


def test_order_from_oanda_infers_side_from_signed_units() -> None:
    buy_order = order_from_oanda(
        {"id": "tx1", "units": 500, "price": "1.0800"}, asset=_BTC,
        algo_id="AI_TWAP", decision_timestamp_utc=_NOW, regulatory_client_shortcode="CLI001",
    )
    sell_order = order_from_oanda(
        {"id": "tx2", "units": -500, "price": "1.0800"}, asset=_BTC,
        algo_id="AI_TWAP", decision_timestamp_utc=_NOW, regulatory_client_shortcode="CLI001",
    )
    assert buy_order.side == OrderSide.BUY
    assert sell_order.side == OrderSide.SELL
    assert buy_order.quantity == sell_order.quantity == Decimal("500")

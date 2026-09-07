"""Real, executable tests for risk/cross_margin_engine.py.

No real broker is contacted -- a minimal, real (not mocked) BaseBroker
subclass with controllable margin/position state stands in for a
connected adapter, exercising the engine's own real aggregation/
risk-reduction logic against it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from core.nothierly_ontology import AssetClass, MarginSnapshot, UniversalAsset, UniversalPosition
from core.schemas import OrderBookSnapshot
from risk.cross_margin_engine import CrossMarginEngine
from src.execution.broker_gateway import BaseBroker

_EUR_USD = UniversalAsset(
    symbol="EUR_USD", asset_class=AssetClass.FOREX, base_currency="EUR", quote_currency="USD",
    tick_size=Decimal("0.0001"), lot_size=Decimal("1"),
)
_BTC_USDT = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)


class _FakeBroker(BaseBroker):
    """A real, minimal BaseBroker implementation with fully controllable state -- not a mock of the ABC's methods."""

    def __init__(
        self, broker_id: str, *, margin_snapshot: MarginSnapshot, positions: list[UniversalPosition] | None = None
    ) -> None:
        self.broker_id = broker_id
        self._margin_snapshot = margin_snapshot
        self._positions = positions or []
        self.close_calls: list[tuple[str, str, float]] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def place_market_order(self, symbol: str, side: Any, quantity: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        self.close_calls.append((symbol, side, quantity))
        return {"id": f"{self.broker_id}-close-1", "clientOrderId": client_order_id or "x", "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": 1.0, "status": "closed", "units": quantity if side == "buy" else -quantity, "price": "1.0"}

    async def place_limit_order(self, symbol: str, side: Any, quantity: float, price: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        raise NotImplementedError

    async def get_current_price(self, symbol: str) -> float:
        return 1.0

    async def get_account_balance(self) -> float:
        return float(self._margin_snapshot.account_balance_usd)

    async def get_margin_snapshot(self) -> MarginSnapshot:
        return self._margin_snapshot

    async def get_open_positions(self) -> list[UniversalPosition]:
        return self._positions


def _position(asset: UniversalAsset, *, quantity: Decimal, mark_price: Decimal) -> UniversalPosition:
    return UniversalPosition(
        broker_id="x", asset=asset, quantity=quantity, avg_entry_price=mark_price, mark_price=mark_price,
        unrealized_pnl_usd=Decimal(0),
    )


@pytest.mark.asyncio
async def test_global_report_excludes_spot_from_utilization_ratio() -> None:
    oanda = _FakeBroker("oanda", margin_snapshot=MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"), margin_used_usd=Decimal("2000"), margin_available_usd=Decimal("8000"),
    ))
    binance = _FakeBroker("binance", margin_snapshot=MarginSnapshot(broker_id="binance", account_balance_usd=Decimal("5000")))

    engine = CrossMarginEngine({"oanda": oanda, "binance": binance})
    report = await engine.compute_global_margin_report()

    assert report.total_balance_usd == Decimal("15000")  # both count toward balance
    assert report.total_margin_used_usd == Decimal("2000")  # only oanda counts toward margin
    assert report.global_utilization_pct == Decimal("20")  # 2000 / (2000+8000) * 100
    assert report.unmargined_broker_ids == ["binance"]


@pytest.mark.asyncio
async def test_global_report_is_zero_when_no_margined_broker_connected() -> None:
    binance = _FakeBroker("binance", margin_snapshot=MarginSnapshot(broker_id="binance", account_balance_usd=Decimal("5000")))
    engine = CrossMarginEngine({"binance": binance})
    report = await engine.compute_global_margin_report()
    assert report.global_utilization_pct == Decimal("0")
    assert report.unmargined_broker_ids == ["binance"]


@pytest.mark.asyncio
async def test_enforce_margin_safety_does_nothing_below_critical_threshold() -> None:
    oanda = _FakeBroker("oanda", margin_snapshot=MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"), margin_used_usd=Decimal("2000"), margin_available_usd=Decimal("8000"),
    ), positions=[_position(_EUR_USD, quantity=Decimal("1000"), mark_price=Decimal("1.08"))])

    engine = CrossMarginEngine({"oanda": oanda}, critical_threshold_pct=Decimal(90))
    closed = await engine.enforce_margin_safety()

    assert closed == []
    assert oanda.close_calls == []


@pytest.mark.asyncio
async def test_enforce_margin_safety_closes_largest_position_on_the_same_critical_broker() -> None:
    """Real check that risk reduction happens on the SAME venue that's in danger -- never a cross-broker fund move (see module docstring)."""
    oanda = _FakeBroker("oanda", margin_snapshot=MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"), margin_used_usd=Decimal("9500"), margin_available_usd=Decimal("500"),
    ), positions=[
        _position(_EUR_USD, quantity=Decimal("1000"), mark_price=Decimal("1.08")),
    ])
    binance = _FakeBroker("binance", margin_snapshot=MarginSnapshot(broker_id="binance", account_balance_usd=Decimal("50000")),
        positions=[_position(_BTC_USDT, quantity=Decimal("1.0"), mark_price=Decimal("65000"))])

    engine = CrossMarginEngine({"oanda": oanda, "binance": binance}, critical_threshold_pct=Decimal(90))
    closed = await engine.enforce_margin_safety()

    assert len(closed) == 1
    assert oanda.close_calls == [("EUR_USD", "sell", 1000.0)]
    assert binance.close_calls == []  # never touched -- no fund transfer, no cross-broker action


@pytest.mark.asyncio
async def test_enforce_margin_safety_picks_the_largest_notional_position() -> None:
    oanda = _FakeBroker("oanda", margin_snapshot=MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"), margin_used_usd=Decimal("9500"), margin_available_usd=Decimal("500"),
    ), positions=[
        _position(_EUR_USD, quantity=Decimal("100"), mark_price=Decimal("1.08")),  # notional ~108
        _position(_EUR_USD.model_copy(update={"symbol": "GBP_USD"}), quantity=Decimal("-5000"), mark_price=Decimal("1.25")),  # notional 6250
    ])
    engine = CrossMarginEngine({"oanda": oanda}, critical_threshold_pct=Decimal(90))
    closed = await engine.enforce_margin_safety()

    assert len(closed) == 1
    assert oanda.close_calls == [("GBP_USD", "buy", 5000.0)]  # short position -> closed by buying


@pytest.mark.asyncio
async def test_compute_liquidatable_spot_value_sums_only_unmargined_broker_positions() -> None:
    oanda = _FakeBroker("oanda", margin_snapshot=MarginSnapshot(
        broker_id="oanda", account_balance_usd=Decimal("10000"), margin_used_usd=Decimal("2000"), margin_available_usd=Decimal("8000"),
    ), positions=[_position(_EUR_USD, quantity=Decimal("1000"), mark_price=Decimal("1.08"))])
    binance = _FakeBroker("binance", margin_snapshot=MarginSnapshot(broker_id="binance", account_balance_usd=Decimal("50000")),
        positions=[_position(_BTC_USDT, quantity=Decimal("2.0"), mark_price=Decimal("65000"))])

    engine = CrossMarginEngine({"oanda": oanda, "binance": binance})
    liquidatable = await engine.compute_liquidatable_spot_value_usd()

    assert liquidatable == Decimal("130000")  # only binance's 2 BTC @ 65000 -- oanda is margined, excluded

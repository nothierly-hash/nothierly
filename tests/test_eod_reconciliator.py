"""Real, executable tests for src/execution/eod_reconciliator.py.

No real broker or Timescale/Redis is contacted. Balance-fetching tests
substitute a real, minimal ``BaseBroker`` subclass (same discipline as
``tests/test_cross_margin_engine.py``) in place of ``BrokerFactory.
create``. The pure aggregation/break-detection functions
(``detect_breaks``, ``compute_theoretical_shadow_inventory``,
``rollup_inventory_by_instrument``) take plain data and need no I/O at
all -- see module docstring's own "pure functions, real regression
tests without a live DB/broker" framing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from core.schemas import OrderBookSnapshot
from src.execution.broker_gateway import BaseBroker
from src.execution.eod_reconciliator import (
    BrokerBalanceResult,
    ShadowInventoryRow,
    compute_live_fills_by_broker,
    compute_theoretical_shadow_inventory,
    detect_breaks,
    fetch_broker_balances,
    rollup_inventory_by_instrument,
)

_NOW = datetime(2026, 9, 3, 0, 0, 0, tzinfo=timezone.utc)


class _FakeBroker(BaseBroker):
    """A real, minimal BaseBroker with controllable balance/failure -- not a mock of the ABC's methods."""

    def __init__(self, *, balance: float | None = None, raises: Exception | None = None) -> None:
        self._balance = balance
        self._raises = raises
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def place_market_order(self, symbol: str, side: Any, quantity: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def place_limit_order(self, symbol: str, side: Any, quantity: float, price: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        raise NotImplementedError

    async def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError

    async def get_account_balance(self) -> float:
        if self._raises is not None:
            raise self._raises
        assert self._balance is not None
        return self._balance

    async def get_margin_snapshot(self) -> Any:
        raise NotImplementedError

    async def get_open_positions(self) -> list[Any]:
        raise NotImplementedError


# -- fetch_broker_balances (crash resilience) --------------------------------


@pytest.mark.asyncio
async def test_fetch_broker_balances_real_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_create(broker_id: str, *, settings: Any = None, redis_client: Any = None) -> BaseBroker:
        return _FakeBroker(balance=12345.67)

    monkeypatch.setattr("src.execution.eod_reconciliator.BrokerFactory.create", _fake_create)
    results = await fetch_broker_balances(["binance"])
    assert len(results) == 1
    assert results[0].broker_id == "binance"
    assert results[0].balance_usd == 12345.67
    assert results[0].error is None


@pytest.mark.asyncio
async def test_fetch_broker_balances_one_failure_does_not_block_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real crash-resilience requirement: one broker's API outage must not prevent fetching the rest."""

    async def _fake_create(broker_id: str, *, settings: Any = None, redis_client: Any = None) -> BaseBroker:
        if broker_id == "broken_exchange":
            return _FakeBroker(raises=ConnectionError("real simulated API outage"))
        return _FakeBroker(balance=500.0)

    monkeypatch.setattr("src.execution.eod_reconciliator.BrokerFactory.create", _fake_create)
    results = await fetch_broker_balances(["binance", "broken_exchange", "oanda"])

    by_id = {r.broker_id: r for r in results}
    assert by_id["binance"].balance_usd == 500.0
    assert by_id["oanda"].balance_usd == 500.0
    assert by_id["broken_exchange"].balance_usd is None
    assert by_id["broken_exchange"].error is not None
    assert "real simulated API outage" in by_id["broken_exchange"].error


@pytest.mark.asyncio
async def test_fetch_broker_balances_factory_construction_failure_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real regression guard: a failure inside BrokerFactory.create itself (not just get_account_balance) must also be caught."""

    async def _fake_create(broker_id: str, *, settings: Any = None, redis_client: Any = None) -> BaseBroker:
        raise ValueError("unknown broker_id")

    monkeypatch.setattr("src.execution.eod_reconciliator.BrokerFactory.create", _fake_create)
    results = await fetch_broker_balances(["not_a_real_broker"])
    assert results[0].balance_usd is None
    assert "unknown broker_id" in (results[0].error or "")


# -- detect_breaks --------------------------------


def test_detect_breaks_flags_large_unexplained_delta() -> None:
    previous = {"binance": 100_000.0}
    current = [BrokerBalanceResult(broker_id="binance", balance_usd=99_000.0, fetched_at=_NOW)]
    breaks = detect_breaks(previous_balances=previous, current_results=current, threshold_usd=50.0)
    assert len(breaks) == 1
    assert breaks[0].delta_usd == -1000.0
    assert breaks[0].severity == "BREAK"


def test_detect_breaks_ignores_small_delta_under_threshold() -> None:
    previous = {"binance": 100_000.0}
    current = [BrokerBalanceResult(broker_id="binance", balance_usd=100_010.0, fetched_at=_NOW)]
    breaks = detect_breaks(previous_balances=previous, current_results=current, threshold_usd=50.0)
    assert breaks == []


def test_detect_breaks_no_prior_snapshot_produces_no_break() -> None:
    """A broker reconciled for the first time has nothing real to compare against -- not a fabricated Break."""
    current = [BrokerBalanceResult(broker_id="new_broker", balance_usd=5000.0, fetched_at=_NOW)]
    breaks = detect_breaks(previous_balances={}, current_results=current, threshold_usd=50.0)
    assert breaks == []


def test_detect_breaks_failed_fetch_produces_no_break() -> None:
    """A None balance (failed fetch) must never be silently treated as a $0 balance / real Break."""
    previous = {"binance": 100_000.0}
    current = [BrokerBalanceResult(broker_id="binance", balance_usd=None, error="timeout", fetched_at=_NOW)]
    breaks = detect_breaks(previous_balances=previous, current_results=current, threshold_usd=50.0)
    assert breaks == []


# -- compute_theoretical_shadow_inventory --------------------------------


def test_compute_shadow_inventory_nets_buys_and_sells() -> None:
    fills = [
        {"client_id": "c1", "instrument": "BTC/USDT", "side": "BUY", "quantity": 1.5},
        {"client_id": "c1", "instrument": "BTC/USDT", "side": "SELL", "quantity": 0.5},
        {"client_id": "c1", "instrument": "ETH/USDT", "side": "BUY", "quantity": 2.0},
        {"client_id": "c2", "instrument": "BTC/USDT", "side": "SELL", "quantity": 1.0},
    ]
    rows = compute_theoretical_shadow_inventory(fills)
    by_key = {(r.client_id, r.instrument): r for r in rows}

    assert by_key[("c1", "BTC/USDT")].net_quantity == pytest.approx(1.0)
    assert by_key[("c1", "BTC/USDT")].fill_count == 2
    assert by_key[("c1", "ETH/USDT")].net_quantity == pytest.approx(2.0)
    assert by_key[("c2", "BTC/USDT")].net_quantity == pytest.approx(-1.0)
    assert all(r.is_shadow_only for r in rows)


def test_compute_shadow_inventory_empty_input_returns_empty_list() -> None:
    assert compute_theoretical_shadow_inventory([]) == []


def test_compute_shadow_inventory_side_is_case_insensitive() -> None:
    fills = [{"client_id": "c1", "instrument": "BTC/USDT", "side": "buy", "quantity": 1.0}]
    rows = compute_theoretical_shadow_inventory(fills)
    assert rows[0].net_quantity == pytest.approx(1.0)


# -- compute_live_fills_by_broker --------------------------------


def test_compute_live_fills_by_broker_nets_buys_and_sells_per_broker() -> None:
    fills = [
        {"broker_id": "binance", "instrument": "BTC/USDT", "side": "BUY", "quantity": 1.5},
        {"broker_id": "binance", "instrument": "BTC/USDT", "side": "SELL", "quantity": 0.5},
        {"broker_id": "binance", "instrument": "ETH/USDT", "side": "BUY", "quantity": 2.0},
        {"broker_id": "okx", "instrument": "BTC/USDT", "side": "SELL", "quantity": 1.0},
    ]
    rows = compute_live_fills_by_broker(fills)
    by_key = {(r.broker_id, r.instrument): r for r in rows}

    assert by_key[("binance", "BTC/USDT")].net_quantity == pytest.approx(1.0)
    assert by_key[("binance", "BTC/USDT")].fill_count == 2
    assert by_key[("binance", "ETH/USDT")].net_quantity == pytest.approx(2.0)
    assert by_key[("okx", "BTC/USDT")].net_quantity == pytest.approx(-1.0)


def test_compute_live_fills_by_broker_empty_input_returns_empty_list() -> None:
    assert compute_live_fills_by_broker([]) == []


def test_compute_live_fills_by_broker_side_is_case_insensitive() -> None:
    fills = [{"broker_id": "binance", "instrument": "BTC/USDT", "side": "buy", "quantity": 1.0}]
    rows = compute_live_fills_by_broker(fills)
    assert rows[0].net_quantity == pytest.approx(1.0)


# -- rollup_inventory_by_instrument --------------------------------


def test_rollup_sums_across_clients_per_instrument() -> None:
    rows = [
        ShadowInventoryRow(client_id="c1", instrument="BTC/USDT", net_quantity=1.0, fill_count=1),
        ShadowInventoryRow(client_id="c2", instrument="BTC/USDT", net_quantity=-0.5, fill_count=1),
        ShadowInventoryRow(client_id="c1", instrument="ETH/USDT", net_quantity=3.0, fill_count=1),
    ]
    totals = rollup_inventory_by_instrument(rows)
    assert totals["BTC/USDT"] == pytest.approx(0.5)
    assert totals["ETH/USDT"] == pytest.approx(3.0)


def test_rollup_empty_input_returns_empty_dict() -> None:
    assert rollup_inventory_by_instrument([]) == {}

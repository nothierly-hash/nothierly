"""Real, executable tests for network/resilience_mesh.py.

Real timing (small, fast durations), not a mocked clock -- consistent
with this project's "prefer a real behavior over a mock" discipline.
No real broker/exchange is contacted.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from ccxt.base.errors import RateLimitExceeded

from core.schemas import OrderBookLevel, OrderBookSnapshot
from network.resilience_mesh import RateLimitMesh, StreamRecoveryManager, TokenBucket
from src.execution.broker_gateway import BaseBroker


def _book(bid: float, ask: float) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=bid, volume=1.0)], asks=[OrderBookLevel(price=ask, volume=1.0)],
    )


class _FakeBrokerForRecovery(BaseBroker):
    """Minimal real BaseBroker whose get_orderbook_snapshot returns a controllable, real snapshot."""

    def __init__(self, snapshots: list[OrderBookSnapshot]) -> None:
        self.broker_id = "test"
        self._snapshots = snapshots
        self._call_count = 0

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def place_market_order(self, symbol: str, side: Any, quantity: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def place_limit_order(self, symbol: str, side: Any, quantity: float, price: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        snapshot = self._snapshots[min(self._call_count, len(self._snapshots) - 1)]
        self._call_count += 1
        return snapshot

    async def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError

    async def get_account_balance(self) -> float:
        raise NotImplementedError

    async def get_margin_snapshot(self) -> Any:
        raise NotImplementedError

    async def get_open_positions(self) -> list[Any]:
        raise NotImplementedError


# -- TokenBucket --------------------------------------------------------


@pytest.mark.asyncio
async def test_token_bucket_allows_immediate_calls_within_capacity() -> None:
    bucket = TokenBucket(capacity=3, refill_rate_per_second=10.0)
    start = time.monotonic()
    for _ in range(3):
        await bucket.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.05  # all 3 within the initial capacity -- no real waiting


@pytest.mark.asyncio
async def test_token_bucket_real_blocks_once_capacity_exhausted() -> None:
    bucket = TokenBucket(capacity=1, refill_rate_per_second=10.0)  # refills 1 token per 100ms
    await bucket.acquire()  # consumes the only token
    start = time.monotonic()
    await bucket.acquire()  # must really wait ~100ms for a refill
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08  # real wait happened, not skipped


@pytest.mark.asyncio
async def test_token_bucket_reduce_and_restore_rate() -> None:
    bucket = TokenBucket(capacity=10, refill_rate_per_second=10.0)
    bucket.reduce_rate(0.5)
    assert bucket.refill_rate_per_second == 5.0
    bucket.restore_rate(10.0)
    assert bucket.refill_rate_per_second == 10.0


# -- RateLimitMesh --------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_mesh_requires_registration() -> None:
    mesh = RateLimitMesh()
    with pytest.raises(ValueError):
        await mesh.throttled_call("unregistered", lambda: asyncio.sleep(0))


@pytest.mark.asyncio
async def test_rate_limit_mesh_real_call_succeeds() -> None:
    mesh = RateLimitMesh()
    mesh.register_broker("binance", requests_per_second=10.0)
    result = await mesh.throttled_call("binance", lambda: _return_42())
    assert result == 42


async def _return_42() -> int:
    return 42


@pytest.mark.asyncio
async def test_rate_limit_mesh_halves_rate_on_real_rate_limit_error() -> None:
    mesh = RateLimitMesh()
    mesh.register_broker("binance", requests_per_second=10.0)

    async def _raises_rate_limit() -> None:
        raise RateLimitExceeded("binance says slow down")

    with pytest.raises(RateLimitExceeded):
        await mesh.throttled_call("binance", _raises_rate_limit)

    assert mesh._buckets["binance"].refill_rate_per_second == 5.0


# -- StreamRecoveryManager --------------------------------------------------------


def test_is_stale_false_immediately_after_construction() -> None:
    broker = _FakeBrokerForRecovery([_book(100.0, 100.5)])
    manager = StreamRecoveryManager(broker, "BTC/USDT", gap_timeout_seconds=3.0)
    assert manager.is_stale is False


@pytest.mark.asyncio
async def test_is_stale_true_after_real_gap_timeout() -> None:
    broker = _FakeBrokerForRecovery([_book(100.0, 100.5)])
    manager = StreamRecoveryManager(broker, "BTC/USDT", gap_timeout_seconds=0.05)
    await asyncio.sleep(0.08)
    assert manager.is_stale is True


@pytest.mark.asyncio
async def test_record_tick_resets_staleness() -> None:
    broker = _FakeBrokerForRecovery([_book(100.0, 100.5)])
    manager = StreamRecoveryManager(broker, "BTC/USDT", gap_timeout_seconds=0.05)
    await asyncio.sleep(0.08)
    assert manager.is_stale is True
    manager.record_tick(_book(101.0, 101.5))
    assert manager.is_stale is False


@pytest.mark.asyncio
async def test_reconcile_detects_real_price_delta() -> None:
    broker = _FakeBrokerForRecovery([_book(105.0, 105.5)])
    manager = StreamRecoveryManager(broker, "BTC/USDT")
    manager.record_tick(_book(100.0, 100.5))

    fresh, delta = await manager.reconcile()
    assert fresh.best_bid.price == 105.0
    assert delta is not None
    assert delta["old_best_bid"] == 100.0
    assert delta["new_best_bid"] == 105.0
    assert manager.reconciliation_count == 1


@pytest.mark.asyncio
async def test_reconcile_reports_no_delta_when_price_unchanged() -> None:
    broker = _FakeBrokerForRecovery([_book(100.0, 100.5)])
    manager = StreamRecoveryManager(broker, "BTC/USDT")
    manager.record_tick(_book(100.0, 100.5))

    _fresh, delta = await manager.reconcile()
    assert delta is None


@pytest.mark.asyncio
async def test_guarded_stream_flags_recovery_after_a_real_gap() -> None:
    """The real end-to-end scenario: a stream goes quiet past the gap timeout, the next tick triggers a real reconcile."""

    async def _slow_stream() -> Any:
        yield _book(100.0, 100.5)
        await asyncio.sleep(0.08)  # exceeds the 0.05s gap_timeout_seconds below -- a real gap
        yield _book(103.0, 103.5)

    broker = _FakeBrokerForRecovery([_book(103.0, 103.5)])  # what the REST resync will return
    manager = StreamRecoveryManager(broker, "BTC/USDT", gap_timeout_seconds=0.05)

    results = [item async for item in manager.guarded_stream(_slow_stream())]

    assert len(results) == 2
    assert results[0][0].best_bid.price == 100.0
    assert results[0][1] is False  # was_recovered
    assert results[1][0].best_bid.price == 103.0
    assert results[1][1] is True  # was_recovered
    assert manager.reconciliation_count == 1

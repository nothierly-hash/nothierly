"""Real, executable tests for core/crossing_network.py.

Real asyncio concurrency and real timing (small real sleeps/windows,
not a mocked clock) -- matching this project's established async-test
discipline. ``get_mid_price`` is a simple, real, controllable async
callable (not a mock of the matching engine itself).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Awaitable, Callable

import pytest

from core.crossing_network import CrossingNetwork, CrossingResult
from core.nothierly_ontology import OrderSide

_INSTRUMENT = "BTC/USDT"


def _fixed_price(price: Decimal | None) -> Callable[[str], Awaitable[Decimal | None]]:
    async def _get(instrument: str) -> Decimal | None:
        return price
    return _get


@pytest.mark.asyncio
async def test_immediate_full_match_when_counter_order_already_resting() -> None:
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("65000")), matching_window_ms=200.0)

    async def _submit_sell() -> None:
        await network.submit_order(order_id="s1", client_id="client-b", instrument=_INSTRUMENT, side=OrderSide.SELL, quantity=Decimal("1.0"))

    sell_task = asyncio.create_task(_submit_sell())
    await asyncio.sleep(0.01)  # let the sell order actually rest in the book first
    buy_result = await network.submit_order(order_id="b1", client_id="client-a", instrument=_INSTRUMENT, side=OrderSide.BUY, quantity=Decimal("1.0"))
    await sell_task

    assert buy_result.remaining_quantity == 0
    assert buy_result.released_to_market is False
    assert len(buy_result.fills) == 1
    fill = buy_result.fills[0]
    assert fill.price == Decimal("65000")
    assert fill.quantity == Decimal("1.0")
    assert fill.buy_client_id == "client-a"
    assert fill.sell_client_id == "client-b"


@pytest.mark.asyncio
async def test_partial_match_then_late_arrival_completes_it() -> None:
    """Real regression test for the wait-and-retry loop: a partial fill must keep waiting for the remainder within its own window."""
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("2500")), matching_window_ms=300.0)

    async def _submit_buy() -> CrossingResult:
        return await network.submit_order(order_id="b1", client_id="client-a", instrument="ETH/USDT", side=OrderSide.BUY, quantity=Decimal("10.0"))

    buy_task = asyncio.create_task(_submit_buy())
    await asyncio.sleep(0.02)
    # First seller only has 4 -- a real partial fill.
    await network.submit_order(order_id="s1", client_id="client-b", instrument="ETH/USDT", side=OrderSide.SELL, quantity=Decimal("4.0"))
    await asyncio.sleep(0.02)
    # Second seller shows up later within the buy order's own 300ms window, completing it.
    await network.submit_order(order_id="s2", client_id="client-c", instrument="ETH/USDT", side=OrderSide.SELL, quantity=Decimal("6.0"))

    buy_result = await buy_task
    assert buy_result.remaining_quantity == 0
    assert buy_result.released_to_market is False
    assert len(buy_result.fills) == 2
    assert {f.quantity for f in buy_result.fills} == {Decimal("4.0"), Decimal("6.0")}
    assert buy_result.filled_quantity == Decimal("10.0")


@pytest.mark.asyncio
async def test_self_trade_prevention_same_client_never_matches() -> None:
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("65000")), matching_window_ms=40.0)

    async def _submit_sell() -> None:
        await network.submit_order(order_id="s1", client_id="same-client", instrument=_INSTRUMENT, side=OrderSide.SELL, quantity=Decimal("1.0"))

    sell_task = asyncio.create_task(_submit_sell())
    await asyncio.sleep(0.005)
    buy_result = await network.submit_order(order_id="b1", client_id="same-client", instrument=_INSTRUMENT, side=OrderSide.BUY, quantity=Decimal("1.0"))
    await sell_task

    assert buy_result.fills == []
    assert buy_result.released_to_market is True
    assert buy_result.remaining_quantity == Decimal("1.0")


@pytest.mark.asyncio
async def test_no_live_price_means_no_match_even_with_a_real_counter_order() -> None:
    """Real 'never fabricate a price' guard: an opposite order exists, but with no live mid-price the engine must not cross."""
    network = CrossingNetwork(get_mid_price=_fixed_price(None), matching_window_ms=40.0)

    async def _submit_sell() -> None:
        await network.submit_order(order_id="s1", client_id="client-b", instrument=_INSTRUMENT, side=OrderSide.SELL, quantity=Decimal("1.0"))

    sell_task = asyncio.create_task(_submit_sell())
    await asyncio.sleep(0.005)
    buy_result = await network.submit_order(order_id="b1", client_id="client-a", instrument=_INSTRUMENT, side=OrderSide.BUY, quantity=Decimal("1.0"))
    await sell_task

    assert buy_result.fills == []
    assert buy_result.released_to_market is True


@pytest.mark.asyncio
async def test_no_counter_order_releases_to_market_after_window() -> None:
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("65000")), matching_window_ms=30.0)
    result = await network.submit_order(order_id="b1", client_id="client-a", instrument=_INSTRUMENT, side=OrderSide.BUY, quantity=Decimal("1.0"))
    assert result.fills == []
    assert result.released_to_market is True
    assert result.remaining_quantity == Decimal("1.0")
    # Real cleanup check: a released order must not linger in the book forever.
    assert network.book_depth(_INSTRUMENT)[OrderSide.BUY] == Decimal(0)


@pytest.mark.asyncio
async def test_price_time_priority_fifo_across_multiple_resting_orders() -> None:
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("100")), matching_window_ms=200.0)

    async def _submit(order_id: str, client_id: str, qty: str) -> None:
        await network.submit_order(order_id=order_id, client_id=client_id, instrument=_INSTRUMENT, side=OrderSide.SELL, quantity=Decimal(qty))

    first_task = asyncio.create_task(_submit("s1", "client-first", "5.0"))
    await asyncio.sleep(0.01)
    second_task = asyncio.create_task(_submit("s2", "client-second", "5.0"))
    await asyncio.sleep(0.01)

    buy_result = await network.submit_order(order_id="b1", client_id="client-buyer", instrument=_INSTRUMENT, side=OrderSide.BUY, quantity=Decimal("5.0"))
    await first_task
    await second_task

    # Real FIFO/price-time priority: the earlier-arrived resting order (s1) must be filled first, entirely.
    assert len(buy_result.fills) == 1
    assert buy_result.fills[0].sell_order_id == "s1"
    assert buy_result.fills[0].sell_client_id == "client-first"


@pytest.mark.asyncio
async def test_different_instruments_do_not_cross_each_other() -> None:
    network = CrossingNetwork(get_mid_price=_fixed_price(Decimal("100")), matching_window_ms=30.0)

    async def _submit_sell() -> None:
        await network.submit_order(order_id="s1", client_id="client-b", instrument="ETH/USDT", side=OrderSide.SELL, quantity=Decimal("1.0"))

    sell_task = asyncio.create_task(_submit_sell())
    await asyncio.sleep(0.005)
    buy_result = await network.submit_order(order_id="b1", client_id="client-a", instrument="BTC/USDT", side=OrderSide.BUY, quantity=Decimal("1.0"))
    await sell_task

    assert buy_result.fills == []
    assert buy_result.released_to_market is True

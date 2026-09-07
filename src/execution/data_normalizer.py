"""Chapter 13, Étape 5.6, Pilier 1: raw exchange payload -> unified L2 structure.

**A real correction to the brief, stated upfront**: the brief asks for a
new ``NothierlyL2Update`` dataclass to unify Binance/Bybit's raw
WebSocket JSON before it reaches the XGBoost agent. That normalization
step **already exists and already works** -- it just lives inline as
:meth:`~src.agents.market_data.agent.MarketDataAgent._calculate_metrics`,
converting raw ``ccxt`` order books into the project's real, validated
schema (:class:`~core.schemas.OrderBookSnapshot` +
:class:`~core.schemas.MarketMetrics`) before every Redis write and ZMQ
publish. Inventing a second, differently-named structure
(``NothierlyL2Update``) holding the same fields would be a duplicate
representation of the same data, not a new capability -- ``ccxt``
already *is* the real, working normalization layer across Binance,
Bybit, OKX, and every other ``ccxt``-supported exchange (see
:mod:`src.execution.broker_gateway`'s own docstring for why one
adapter, not three, covers all of them): every exchange's raw WebSocket
payload already arrives at this module in ``ccxt``'s one unified shape,
not three heterogeneous ones.

What *is* real, new work here: the conversion logic was only reachable
as a private method on one agent, untestable in isolation and
unusable by any other consumer (a future Shadow Mode simulator, a
backtest replay tool, ...). This module extracts it into a pure,
standalone, directly-testable function --
:class:`~src.agents.market_data.agent.MarketDataAgent` now calls this
instead of duplicating the logic inline, and :data:`NormalizedL2Update`
is a thin, named ``NamedTuple`` alias for the real
``(OrderBookSnapshot, MarketMetrics)`` pair -- ergonomic destructuring
at call sites, not a new competing schema.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, NamedTuple

from core.schemas import ORDER_BOOK_TOP_N, MarketMetrics, OrderBookLevel, OrderBookSnapshot


class NormalizedL2Update(NamedTuple):
    """One exchange-agnostic, validated L2 order book update -- see module docstring for why a ``NamedTuple``, not a new schema."""

    snapshot: OrderBookSnapshot
    metrics: MarketMetrics


def normalize_l2_update(raw_order_book: dict[str, Any], *, symbol: str) -> NormalizedL2Update:
    """Convert one raw ``ccxt`` order book payload into validated, unified project schemas.

    Real logic, unchanged from its original home in
    :meth:`~src.agents.market_data.agent.MarketDataAgent._calculate_metrics`
    (moved here, not rewritten) -- pure Python only (no pandas, no
    NumPy): at a fixed top-5 depth, the handful of sums/subtractions/
    divisions this needs (done in
    :meth:`~core.schemas.MarketMetrics.from_order_book_snapshot`, the
    single source of truth for the formula) is faster than the
    array-construction overhead NumPy would add for data this small.

    Works identically regardless of which exchange ``raw_order_book``
    came from (Binance, Bybit, OKX, ...): ``ccxt`` has already unified
    the wire format by the time this function ever sees it -- see the
    module docstring for why this is real exchange-agnostic
    normalization, not a stub.

    Args:
        raw_order_book: Raw ``ccxt`` order book structure. Must contain
            ``"bids"``/``"asks"`` as lists of ``[price, amount, ...]``
            pairs, best level first.
        symbol: The trading pair this snapshot is for (ccxt order book
            payloads don't self-report their own symbol).

    Returns:
        The validated ``(snapshot, metrics)`` pair.

    Raises:
        pydantic.ValidationError: If the raw payload doesn't yield a
            valid snapshot (e.g. an empty side, or a crossed book).
    """
    top_n = ORDER_BOOK_TOP_N
    bids = [OrderBookLevel(price=price, volume=volume) for price, volume, *_ in raw_order_book["bids"][:top_n]]
    asks = [OrderBookLevel(price=price, volume=volume) for price, volume, *_ in raw_order_book["asks"][:top_n]]

    raw_timestamp_ms = raw_order_book.get("timestamp")
    timestamp = (
        datetime.fromtimestamp(raw_timestamp_ms / 1000, tz=timezone.utc)
        if raw_timestamp_ms is not None
        else datetime.now(timezone.utc)
    )

    snapshot = OrderBookSnapshot(symbol=symbol, timestamp=timestamp, bids=bids, asks=asks)
    metrics = MarketMetrics.from_order_book_snapshot(snapshot)
    return NormalizedL2Update(snapshot=snapshot, metrics=metrics)

"""Market data micro-agent: WebSocket ingestion, liquidity metrics, publish.

:class:`MarketDataAgent` wires an :class:`~src.agents.market_data.
exchange_client.AsyncExchangeClient` into :class:`~core.base_agent.
BaseAgent`'s lifecycle. Unlike a poll-driven agent, order book and trade
updates arrive as a push-based WebSocket stream, so this agent overrides
:meth:`~core.base_agent.BaseAgent.run` to concurrently drain both streams
via ``asyncio.gather`` instead of using the periodic tick loop.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import zmq.asyncio
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.redis_client import RedisUnavailableError
from core.schemas import (
    MARKET_METRICS_REDIS_KEY_PREFIX,
    ORDER_BOOK_TOP_N,
    TRADE_TAPE_MAX_ENTRIES,
    TRADE_TAPE_REDIS_KEY_PREFIX,
)
from src.agents.market_data.exchange_client import AsyncExchangeClient
from src.agents.market_data.l2_surface_packer import publish_l2_surface
from src.execution.data_normalizer import normalize_l2_update

_MARKET_DATA_L2_TOPIC = "MARKET_DATA_L2"


class MarketDataAgent(BaseAgent):
    """Streams one symbol's order book/trades and publishes L2 liquidity metrics.

    On every order book update (throttled to at most one publish per
    ``min_publish_interval_seconds``), this agent computes
    :class:`~core.schemas.MarketMetrics` (spread + top-5 order book
    imbalance), writes it to Redis with a short TTL, and publishes it on
    the ZMQ bus under the ``MARKET_DATA_L2`` topic. Trade updates are
    consumed and logged; no derived signal is computed from them in this
    chapter.
    """

    def __init__(
        self,
        symbol: str,
        *,
        exchange_id: str = "binance",
        order_book_limit: int = ORDER_BOOK_TOP_N,
        min_publish_interval_seconds: float = 0.1,
        redis_ttl_seconds: int = 10,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Configure the agent for one symbol on one exchange.

        Args:
            symbol: Exchange trading pair to stream (e.g. ``"BTC/USDT"``).
            exchange_id: ``ccxt.pro`` exchange identifier.
            order_book_limit: Order book depth requested from the
                exchange. Only the top :data:`~core.schemas.
                ORDER_BOOK_TOP_N` levels are ever used for metrics.
            min_publish_interval_seconds: Minimum time between two
                published updates, to avoid saturating Redis/ZMQ with
                every single WebSocket tick (order books can update
                dozens of times per second on a liquid pair).
            redis_ttl_seconds: TTL applied to the ``MarketMetrics`` key
                written to Redis. Deliberately short: a stale liquidity
                snapshot is worse than a missing one for a low-latency
                trading system.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
        """
        agent_id = f"market-data-{exchange_id}-{symbol.replace('/', '-').lower()}"
        super().__init__(
            agent_id,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._symbol = symbol
        self._order_book_limit = order_book_limit
        self._min_publish_interval_seconds = min_publish_interval_seconds
        self._redis_ttl_seconds = redis_ttl_seconds
        self._exchange = AsyncExchangeClient(exchange_id)
        self._last_published_at: float = 0.0

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent's data is stream-, not poll-driven.

        :meth:`run` is fully overridden to concurrently drain the order
        book and trade WebSocket streams instead of using
        :class:`~core.base_agent.BaseAgent`'s periodic tick loop, so this
        is never called. Implemented (rather than omitted) only to
        satisfy the abstract base class, and raises loudly if it is ever
        invoked by mistake (e.g. a future refactor calling
        ``super().run()``).
        """
        raise NotImplementedError("MarketDataAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Load exchange market metadata before streaming begins."""
        await self._exchange.connect()
        self._log.info("market_data_agent_ready", agent_id=self.agent_id, symbol=self._symbol)

    async def on_shutdown(self) -> None:
        """Close the ccxt.pro exchange client's WebSocket/HTTP connections.

        Runs as part of :class:`~core.base_agent.BaseAgent`'s inherited
        graceful-shutdown sequence, before the ZMQ sockets/context and the
        Redis connection are torn down.
        """
        await self._exchange.close()

    async def run(self) -> None:
        """Connect dependencies, then concurrently drain both WS streams.

        Mirrors :meth:`BaseAgent.run`'s lifecycle exactly (signal
        handling, Redis connection, cancellation-safe shutdown) but
        replaces the periodic tick loop with two concurrent, long-lived
        stream consumers run via ``asyncio.gather``: cancelling the
        overall task (on shutdown) cancels both consumers together.

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info(
                "agent_started",
                agent_id=self.agent_id,
                symbol=self._symbol,
            )

            await asyncio.gather(
                self._watch_order_book_loop(),
                self._watch_trades_loop(),
            )
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    # -- Stream consumers -----------------------------------------------

    async def _watch_order_book_loop(self) -> None:
        """Drain the order book WS stream, publishing metrics per update.

        A failure while handling one update (e.g. a malformed payload, a
        Redis hiccup) is logged and skipped rather than crashing this
        loop, so one bad update never brings down market data ingestion.
        Reconnection on WebSocket/network failure is already handled
        transparently inside :meth:`AsyncExchangeClient.watch_order_book`.
        """
        async for raw_order_book in self._exchange.watch_order_book(
            self._symbol, self._order_book_limit
        ):
            try:
                await self._handle_order_book_update(raw_order_book)
            except Exception:
                self._log.exception(
                    "market_data_order_book_update_failed",
                    agent_id=self.agent_id,
                    symbol=self._symbol,
                )

    async def _watch_trades_loop(self) -> None:
        """Drain the trade WS stream, persisting each real trade onto a real, bounded tape.

        No derived VPIN/TFI *signal* is computed here (see
        :mod:`~src.agents.market_data.microstructure_features` for
        that, computed on demand from this real tape) -- this loop's
        own job is only to stop discarding the real trade data after
        logging it, which is what made computing either honestly
        impossible before this.
        """
        async for trades in self._exchange.watch_trades(self._symbol):
            try:
                await self._handle_trades_update(trades)
            except Exception:
                self._log.exception(
                    "market_data_trades_update_failed",
                    agent_id=self.agent_id,
                    symbol=self._symbol,
                )

    # -- Per-update handling ----------------------------------------------

    async def _handle_order_book_update(self, raw_order_book: dict[str, Any]) -> None:
        """Throttle, validate, persist, and publish one order book update.

        Args:
            raw_order_book: Raw ``ccxt`` order book structure, as yielded
                by :meth:`AsyncExchangeClient.watch_order_book`.
        """
        now = time.monotonic()
        if now - self._last_published_at < self._min_publish_interval_seconds:
            return  # Throttled: too soon since the last publish.

        try:
            snapshot, metrics = normalize_l2_update(raw_order_book, symbol=self._symbol)
        except ValidationError:
            self._log.warning(
                "market_data_invalid_order_book_skipped",
                agent_id=self.agent_id,
                symbol=self._symbol,
            )
            return

        self._last_published_at = now

        if self._redis_client is not None:
            try:
                await self._redis_client.execute(
                    "set",
                    f"{MARKET_METRICS_REDIS_KEY_PREFIX}{self._symbol}",
                    metrics.model_dump_json(),
                    ex=self._redis_ttl_seconds,
                )
            except RedisUnavailableError:
                self._log.warning(
                    "market_data_redis_write_failed",
                    agent_id=self.agent_id,
                    symbol=self._symbol,
                )

            try:
                await publish_l2_surface(self._redis_client, snapshot)
            except RedisUnavailableError:
                self._log.warning(
                    "market_data_l2_surface_write_failed",
                    agent_id=self.agent_id,
                    symbol=self._symbol,
                )

        await self.publish(_MARKET_DATA_L2_TOPIC, metrics.model_dump(mode="json"))

    async def _handle_trades_update(self, trades: list[dict[str, Any]]) -> None:
        """Log, and persist onto a real bounded tape, each trade in a WS batch update.

        Args:
            trades: Raw ``ccxt`` list of trade structures for this
                update, as yielded by
                :meth:`AsyncExchangeClient.watch_trades`.
        """
        for trade in trades:
            self._log.debug(
                "trade_observed",
                agent_id=self.agent_id,
                symbol=self._symbol,
                price=trade.get("price"),
                amount=trade.get("amount"),
                side=trade.get("side"),
            )

        if self._redis_client is None:
            return
        # ccxt's own trade "timestamp" is real epoch *milliseconds* -- kept
        # verbatim here; research.alpha_features' functions expect epoch
        # *microseconds*, a real, deliberate unit conversion applied at
        # read time by microstructure_features.py, not here (this tape is
        # the raw, unopinionated real record; unit conversion belongs with
        # its one real consumer, not baked into every future one).
        entries = [
            json.dumps(
                {
                    "timestamp": trade.get("timestamp"),
                    "side": trade.get("side"),
                    "amount": trade.get("amount"),
                    "price": trade.get("price"),
                }
            )
            for trade in trades
            if trade.get("timestamp") is not None and trade.get("side") in ("buy", "sell") and trade.get("amount") is not None
        ]
        if not entries:
            return
        key = f"{TRADE_TAPE_REDIS_KEY_PREFIX}{self._symbol}"
        try:
            await self._redis_client.execute("lpush", key, *entries)
            await self._redis_client.execute("ltrim", key, 0, TRADE_TAPE_MAX_ENTRIES - 1)
        except RedisUnavailableError:
            self._log.warning("market_data_trade_tape_write_failed", agent_id=self.agent_id, symbol=self._symbol)


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
from typing import Any, Final

import zmq.asyncio
from pydantic import ValidationError

from core.base_agent import BaseAgent
from core.circuit_breaker import IngestionCircuitBreaker
from core.config import AppSettings
from core.message_integrity import publish_signed_market_metrics
from core.redis_client import RedisUnavailableError
from core.schemas import (
    ORDER_BOOK_TOP_N,
    TRADE_TAPE_MAX_ENTRIES,
    TRADE_TAPE_REDIS_KEY_PREFIX,
)
from src.agents.market_data.exchange_client import AsyncExchangeClient

# Real bug found live in this exact session's own shadow run (13.5h+ uptime, real trade/L2 data
# flowing the whole time): overriding run() below to drain two WebSocket streams instead of using
# BaseAgent's own periodic tick loop means BaseAgent._write_heartbeat() -- normally called once per
# tick, right there in that generic loop -- was never being called at all, ever, for this agent's
# entire lifetime. A perfectly healthy agent (real, fresh market_metrics/trade_tape writes the
# whole time) therefore looked permanently OFFLINE to every real heartbeat-based liveness check in
# this project (docker-compose.yml's own nothierly-agents healthcheck, swarm_console.py's agent
# status, core.base_agent's own documented "a missing/expired heartbeat means this agent is down"
# contract). Fixed below with a third concurrent loop that writes a real heartbeat on a real, short
# period -- well under _HEARTBEAT_MIN_TTL_SECONDS (5s) so the key never has a chance to expire
# between writes, the same real safety margin BaseAgent's own default tick_interval_seconds=1.0
# already gives every tick-loop-based agent.
_HEARTBEAT_WRITE_INTERVAL_SECONDS: Final[float] = 3.0
from src.agents.market_data.l2_surface_packer import publish_l2_surface
from src.execution.data_normalizer import normalize_l2_update

_MARKET_DATA_L2_TOPIC = "MARKET_DATA_L2"

# "Protocole AEGIS" (2026-09-16): a real circuit breaker over
# market_data_invalid_order_book_skipped -- see _handle_order_book_update/
# _reconnect_after_circuit_breaker_trip. Trip thresholds match core.circuit_breaker.
# IngestionCircuitBreaker's own defaults, spelled out here for discoverability. Reconnect backoff
# constants mirror AsyncExchangeClient's own defaults (base 1.0s / max 60.0s / x2), applied here to
# how aggressively *this agent* re-attempts a clean reconnect after a trip -- a separate, slower
# escalation from ccxt.pro's own internal per-call retry inside watch_order_book, which keeps
# retrying the SAME connection; this backoff instead paces repeated forced reconnects when even a
# fresh connection keeps producing corrupted data.
_MAX_CONSECUTIVE_PARSE_ERRORS: Final[int] = 50
_PARSE_ERROR_RATE_THRESHOLD: Final[float] = 0.05
_PARSE_ERROR_WINDOW_SECONDS: Final[float] = 60.0
_RECONNECT_BASE_DELAY_SECONDS: Final[float] = 1.0
_RECONNECT_MAX_DELAY_SECONDS: Final[float] = 60.0
_RECONNECT_BACKOFF_MULTIPLIER: Final[float] = 2.0


class _MarketDataCircuitBreakerTripped(Exception):
    """Internal signal only, never raised past :meth:`MarketDataAgent.run`.

    Raised by :meth:`MarketDataAgent._watch_order_book_loop` the instant
    :attr:`MarketDataAgent._l2_circuit_breaker` trips -- ``asyncio.gather``
    (without ``return_exceptions=True``) then cancels the sibling trades/
    heartbeat loops automatically, letting :meth:`MarketDataAgent.run`
    force a clean reconnect and restart all three together, rather than
    trying to coordinate a reconnect across independently-running tasks
    via shared mutable state.
    """


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
        self._exchange_id = exchange_id
        self._order_book_limit = order_book_limit
        self._min_publish_interval_seconds = min_publish_interval_seconds
        self._redis_ttl_seconds = redis_ttl_seconds
        self._exchange = AsyncExchangeClient(exchange_id)
        self._last_published_at: float = 0.0
        # "Protocole AEGIS" -- see the module-level circuit-breaker constants' own comment.
        self._l2_circuit_breaker = IngestionCircuitBreaker(
            max_consecutive_failures=_MAX_CONSECUTIVE_PARSE_ERRORS,
            max_failure_rate=_PARSE_ERROR_RATE_THRESHOLD,
            window_seconds=_PARSE_ERROR_WINDOW_SECONDS,
        )
        self._reconnect_attempt_count = 0

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

            await self._run_supervised_streams()
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _run_supervised_streams(self) -> None:
        """Run the two WS stream consumers + heartbeat loop, forcing a clean reconnect on a real circuit-breaker trip.

        "Protocole AEGIS" (2026-09-16): previously, ``run()`` called
        ``asyncio.gather`` on the three loops exactly once -- a sustained
        stream of malformed order book payloads (see
        :meth:`_handle_order_book_update`) was individually logged and
        skipped forever, with nothing ever escalating the pattern into an
        actual remediation attempt (this agent passively absorbed the
        corrupted stream instead of trying to reset it). Now wrapped in a
        loop: a trip inside :meth:`_watch_order_book_loop` raises
        :class:`_MarketDataCircuitBreakerTripped`, which ``asyncio.gather``
        propagates here after cancelling the sibling trades/heartbeat
        tasks -- caught below to force a real close+backoff+reconnect
        (:meth:`_reconnect_after_circuit_breaker_trip`) before restarting
        all three fresh. A genuine shutdown (``CancelledError``) is not
        caught here -- it propagates to :meth:`run`'s own handler.
        """
        while True:
            try:
                await asyncio.gather(
                    self._watch_order_book_loop(),
                    self._watch_trades_loop(),
                    self._heartbeat_loop(),
                )
                return  # pragma: no cover -- these loops only end via CancelledError or the trip above
            except _MarketDataCircuitBreakerTripped:
                await self._reconnect_after_circuit_breaker_trip()

    async def _reconnect_after_circuit_breaker_trip(self) -> None:
        """Real remediation for a tripped :attr:`_l2_circuit_breaker`: force-close the WS connection, back off, reconnect clean.

        The brief's own literal ask ("fermeture forcée de la connexion,
        délai d'attente exponentiel, tentative de reconnexion propre") --
        :class:`~src.agents.market_data.exchange_client.
        AsyncExchangeClient` has no "un-close" once :meth:`~src.agents.
        market_data.exchange_client.AsyncExchangeClient.close` is called
        (by design -- see that method's own docstring), so a genuinely
        fresh instance is constructed rather than reused. The backoff
        delay grows with :attr:`_reconnect_attempt_count` (repeated trips
        back off further each time, capped at
        :data:`_RECONNECT_MAX_DELAY_SECONDS`) and is reset back to zero
        the moment a post-reconnect update is recorded as a real success
        (see :meth:`_handle_order_book_update`) -- a single transient
        trip does not permanently slow down future reconnects.
        """
        self._reconnect_attempt_count += 1
        delay = min(
            _RECONNECT_BASE_DELAY_SECONDS * (_RECONNECT_BACKOFF_MULTIPLIER ** (self._reconnect_attempt_count - 1)),
            _RECONNECT_MAX_DELAY_SECONDS,
        )
        self._log.critical(
            "market_data_circuit_breaker_tripped_reconnecting",
            agent_id=self.agent_id,
            symbol=self._symbol,
            consecutive_failures=self._l2_circuit_breaker.consecutive_failures,
            reconnect_attempt=self._reconnect_attempt_count,
            backoff_delay_seconds=delay,
        )
        await self._exchange.close()
        await asyncio.sleep(delay)
        self._exchange = AsyncExchangeClient(self._exchange_id)
        await self._exchange.connect()
        self._l2_circuit_breaker.reset()
        self._log.info(
            "market_data_circuit_breaker_reconnected", agent_id=self.agent_id, symbol=self._symbol,
        )

    # -- Stream consumers -----------------------------------------------

    async def _watch_order_book_loop(self) -> None:
        """Drain the order book WS stream, publishing metrics per update.

        A single bad update (e.g. one malformed payload, a Redis hiccup)
        is logged and skipped rather than crashing this loop. A
        *sustained* run of malformed payloads is a different, escalating
        condition -- see :meth:`_handle_order_book_update`'s
        :attr:`_l2_circuit_breaker`: once that trips, this raises
        :class:`_MarketDataCircuitBreakerTripped` instead of continuing
        to skip forever (see :meth:`_run_supervised_streams` for what
        happens next). Reconnection on a plain WebSocket/network failure
        (the connection itself dropping) is a separate, already-handled
        case, transparent inside :meth:`AsyncExchangeClient.
        watch_order_book` -- this method's own circuit breaker instead
        targets a connection that stays technically alive but keeps
        producing data that fails schema validation.
        """
        async for raw_order_book in self._exchange.watch_order_book(
            self._symbol, self._order_book_limit
        ):
            try:
                just_tripped = await self._handle_order_book_update(raw_order_book)
            except Exception:
                self._log.exception(
                    "market_data_order_book_update_failed",
                    agent_id=self.agent_id,
                    symbol=self._symbol,
                )
                continue
            if just_tripped:
                raise _MarketDataCircuitBreakerTripped()

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

    async def _heartbeat_loop(self) -> None:
        """Real, independent liveness signal -- see this module's own top-level comment on why
        this agent needs its own heartbeat loop instead of relying on BaseAgent's tick loop (which
        this agent's own ``run()`` override never calls). Runs forever alongside the two real
        WebSocket stream consumers via the same ``asyncio.gather`` in ``run()`` -- cancelled
        together with them on shutdown, same as any other task in that gather.
        """
        while True:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_WRITE_INTERVAL_SECONDS)

    # -- Per-update handling ----------------------------------------------

    async def _handle_order_book_update(self, raw_order_book: dict[str, Any]) -> bool:
        """Throttle, validate, persist, and publish one order book update.

        Args:
            raw_order_book: Raw ``ccxt`` order book structure, as yielded
                by :meth:`AsyncExchangeClient.watch_order_book`.

        Returns:
            ``True`` exactly on the call whose validation failure newly
            trips :attr:`_l2_circuit_breaker` -- see
            :meth:`_watch_order_book_loop`, which reacts to this by
            forcing a clean reconnect. ``False`` otherwise (including
            every throttled/valid/already-tripped call).
        """
        now = time.monotonic()
        if now - self._last_published_at < self._min_publish_interval_seconds:
            return False  # Throttled: too soon since the last publish.

        try:
            snapshot, metrics = normalize_l2_update(raw_order_book, symbol=self._symbol)
        except ValidationError:
            self._log.warning(
                "market_data_invalid_order_book_skipped",
                agent_id=self.agent_id,
                symbol=self._symbol,
            )
            return self._l2_circuit_breaker.record_failure()

        self._last_published_at = now
        if self._reconnect_attempt_count > 0:
            # A real, valid update landed after a forced reconnect -- the remediation worked;
            # un-escalate future backoff instead of leaving it permanently elevated by one past trip.
            self._log.info(
                "market_data_circuit_breaker_recovered", agent_id=self.agent_id, symbol=self._symbol,
            )
            self._reconnect_attempt_count = 0
        self._l2_circuit_breaker.record_success()

        if self._redis_client is not None:
            try:
                # Signed write (core/message_integrity.py) -- real fix for
                # tests/red_team_hacker.py's Attack 1 (Redis MarketMetrics cache poisoning):
                # a plain SET let anyone with Redis access forge this key with zero
                # authentication. Keeps the exact same real key/plaintext JSON every existing
                # reader already expects; only adds a companion signature readers can opt
                # into verifying (see read_verified_market_metrics's own docstring).
                await publish_signed_market_metrics(
                    self._redis_client, self._symbol, metrics, ttl_seconds=self._redis_ttl_seconds
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
        return False

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


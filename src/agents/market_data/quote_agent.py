"""Live top-of-book quotes for FOREX (OANDA) and US STOCKS (Alpaca) -> the same ``MarketMetrics`` cache and ``MARKET_DATA_L2`` topic the crypto agents use.

Until this module, ``EUR_USD``/``XAU_USD`` had no price feed at all (see ``scripts/run_backend.py``): their signal and execution agents never saw a tick. This agent
fills exactly that gap, read-only: it never places an order and needs no trading permission.

**What a quote is, and is not.** Both sources give the best bid/ask only (top of book), not a depth ladder. ``MarketMetrics`` is therefore built from level 1:
``spread = ask - bid``, ``order_book_imbalance = (bid_size - ask_size) / (bid_size + ask_size)`` (0 when both are 0), ``top_of_book_liquidity = bid_size + ask_size``.
Sizes are in the source's own units (OANDA: units of the base currency; Alpaca: round lots), so do not compare the liquidity or the imbalance across asset classes.

**Sources** (both free for a paper/practice account; you create the account, this code only needs its read-only credentials):

- :class:`OandaQuoteSource` -- v20 pricing stream, ``GET {stream_host}/v3/accounts/{account}/pricing/stream?instruments=EUR_USD,XAU_USD``, ``Authorization: Bearer``;
  at most 4 prices per second per instrument, a ``HEARTBEAT`` every 5 s (OANDA documentation). The practice host ``stream-fxpractice.oanda.com`` and live host
  ``stream-fxtrade.oanda.com`` come from OANDA's usual endpoints, not from the page that was read while writing this: configurable, to be verified at first run.
- :class:`AlpacaQuoteSource` -- ``wss://stream.data.alpaca.markets/v2/iex`` (free IEX feed), ``{"action": "auth", "key", "secret"}`` then
  ``{"action": "subscribe", "quotes": [...]}``; messages are JSON arrays whose quote items have ``T == "q"``, ``S``, ``bp``, ``bs``, ``ap``, ``as``, ``t`` (Alpaca
  documentation). The free IEX feed is a fraction of the consolidated market and has no data outside US market hours.

**Not verified against the real servers** (no account was available when this was written): the parsers follow the documented shapes and are tested on those shapes only.
The first live run is the real test; both sources raise (never guess) on an authentication or HTTP error, and the agent reconnects with a capped backoff.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Protocol

import aiohttp
import websockets
import zmq.asyncio

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.message_integrity import publish_signed_market_metrics
from core.redis_client import RedisUnavailableError
from core.schemas import MarketMetrics

_MARKET_DATA_L2_TOPIC: Final[str] = "MARKET_DATA_L2"
OANDA_STREAM_HOSTS: Final[Mapping[str, str]] = {"practice": "https://stream-fxpractice.oanda.com", "live": "https://stream-fxtrade.oanda.com"}
ALPACA_DATA_WS: Final[str] = "wss://stream.data.alpaca.markets/v2/{feed}"
_HEARTBEAT_WRITE_INTERVAL_SECONDS: Final[float] = 5.0
_MAX_BACKOFF_SECONDS: Final[float] = 60.0


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    time: datetime


class QuoteSourceError(Exception):
    """The source refused the connection or sent an error (authentication, HTTP status, closed stream)."""


class QuoteSource(Protocol):
    name: str

    def stream(self) -> AsyncIterator[Quote]: ...


# -- Pure parsers (tested on the documented message shapes) -----------------------------------


def _parse_time(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    if "." in text:  # nanosecond precision (OANDA, Alpaca): datetime keeps microseconds
        head, _, tail = text.partition(".")
        digits = ""
        rest = ""
        for index, char in enumerate(tail):
            if char.isdigit():
                digits += char
            else:
                rest = tail[index:]
                break
        text = f"{head}.{digits[:6].ljust(6, '0')}{rest}"
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _positive(value: Any) -> float:
    number = float(value)
    if not (number > 0) or number == float("inf"):
        raise ValueError("price must be finite and > 0")
    return number


def parse_oanda_line(line: str) -> Quote | None:
    """One line of the OANDA pricing stream: a quote, or ``None`` for a heartbeat, an untradeable price or anything that is not a price.

    Raises ``ValueError``/``KeyError`` on a price message whose fields are missing or unreadable (never a half-quote).
    """
    message = json.loads(line)
    if not isinstance(message, dict) or message.get("type") == "HEARTBEAT":
        return None
    if "bids" not in message or "asks" not in message:
        return None  # not a price message (OANDA documents only PRICE and HEARTBEAT on this stream)
    if message.get("status", "tradeable") != "tradeable":
        return None
    bid, ask = message["bids"][0], message["asks"][0]
    return Quote(
        str(message["instrument"]), _positive(bid["price"]), _positive(ask["price"]), float(bid.get("liquidity", 0.0)), float(ask.get("liquidity", 0.0)),
        _parse_time(message["time"]),
    )


def parse_alpaca_message(item: Mapping[str, Any]) -> Quote | None:
    """One item of an Alpaca message array: a quote (``T == "q"``), or ``None`` for control messages (``success``, ``subscription``)."""
    if item.get("T") != "q":
        return None
    return Quote(
        str(item["S"]), _positive(item["bp"]), _positive(item["ap"]), float(item.get("bs", 0.0)), float(item.get("as", 0.0)), _parse_time(item["t"]),
    )


def quote_to_metrics(quote: Quote) -> MarketMetrics | None:
    """``None`` for a crossed or locked-negative book (bid > ask): a malformed quote is skipped, never published."""
    if quote.bid > quote.ask:
        return None
    total = quote.bid_size + quote.ask_size
    imbalance = (quote.bid_size - quote.ask_size) / total if total > 0 else 0.0
    return MarketMetrics(
        symbol=quote.symbol, timestamp=quote.time, spread=quote.ask - quote.bid, order_book_imbalance=max(-1.0, min(1.0, imbalance)),
        mid_price=(quote.bid + quote.ask) / 2.0, top_of_book_liquidity=total,
    )


# -- Sources -----------------------------------------------------------------------------------


class OandaQuoteSource:
    name = "oanda"

    def __init__(self, account_id: str, token: str, instruments: list[str], *, environment: str = "practice", host: str | None = None) -> None:
        if not instruments or not account_id or not token:
            raise ValueError("account_id, token and at least one instrument are required")
        if environment not in OANDA_STREAM_HOSTS and host is None:
            raise ValueError("environment must be 'practice' or 'live'")
        self._url = f"{host or OANDA_STREAM_HOSTS[environment]}/v3/accounts/{account_id}/pricing/stream"
        self._params = {"instruments": ",".join(instruments)}
        self._headers = {"Authorization": f"Bearer {token}"}  # never logged, never part of an exception message

    async def stream(self) -> AsyncIterator[Quote]:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=30)  # OANDA heartbeats every 5 s: 30 s of silence = dead link
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self._url, params=self._params, headers=self._headers) as response:
                if response.status != 200:
                    raise QuoteSourceError(f"OANDA stream refused: HTTP {response.status}")
                async for raw in response.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        quote = parse_oanda_line(line)
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise QuoteSourceError(f"unreadable OANDA price message: {type(exc).__name__}") from exc
                    if quote is not None:
                        yield quote
        raise QuoteSourceError("OANDA stream closed")


class AlpacaQuoteSource:
    name = "alpaca"

    def __init__(self, key: str, secret: str, symbols: list[str], *, feed: str = "iex", url: str | None = None) -> None:
        if not symbols or not key or not secret:
            raise ValueError("key, secret and at least one symbol are required")
        self._url = url or ALPACA_DATA_WS.format(feed=feed)
        self._auth = json.dumps({"action": "auth", "key": key, "secret": secret})  # never logged
        self._subscribe = json.dumps({"action": "subscribe", "quotes": symbols})

    async def stream(self) -> AsyncIterator[Quote]:
        async with websockets.connect(self._url, ping_interval=20, ping_timeout=20) as connection:
            await connection.send(self._auth)
            await connection.send(self._subscribe)
            async for raw in connection:
                try:
                    items = json.loads(raw)
                    for item in items if isinstance(items, list) else [items]:
                        if item.get("T") == "error":
                            raise QuoteSourceError(f"Alpaca error {item.get('code')}: {str(item.get('msg'))[:80]}")
                        quote = parse_alpaca_message(item)
                        if quote is not None:
                            yield quote
                except QuoteSourceError:
                    raise
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise QuoteSourceError(f"unreadable Alpaca message: {type(exc).__name__}") from exc
        raise QuoteSourceError("Alpaca stream closed")


# -- Agent -------------------------------------------------------------------------------------


class QuoteMarketDataAgent(BaseAgent):
    """Publishes ``MarketMetrics`` (Redis cache + ``MARKET_DATA_L2``) for every symbol a :class:`QuoteSource` streams. One agent per source."""

    def __init__(
        self,
        source: QuoteSource,
        *,
        min_publish_interval_seconds: float = 0.5,
        redis_ttl_seconds: int = 30,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        manage_own_redis: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        super().__init__(
            f"market-data-{source.name}", settings=settings, manage_own_signals=manage_own_signals, zmq_context=zmq_context, manage_own_redis=manage_own_redis,
        )
        self._source = source
        self._min_interval = min_publish_interval_seconds
        self._ttl = redis_ttl_seconds
        self._last_published: dict[str, float] = {}
        self._sleep = sleep
        self.stats = {"published": 0, "throttled": 0, "skipped": 0, "reconnects": 0}

    async def on_tick(self) -> None:
        raise NotImplementedError("QuoteMarketDataAgent overrides run() and never ticks.")

    async def handle_quote(self, quote: Quote, *, now: float | None = None) -> bool:
        """Throttle per symbol, validate, cache and publish one quote. ``True`` if it was published."""
        moment = time.monotonic() if now is None else now
        if moment - self._last_published.get(quote.symbol, -1e9) < self._min_interval:
            self.stats["throttled"] += 1
            return False
        metrics = quote_to_metrics(quote)
        if metrics is None:
            self.stats["skipped"] += 1
            self._log.warning("quote_skipped_crossed_book", agent_id=self.agent_id, symbol=quote.symbol)
            return False
        self._last_published[quote.symbol] = moment
        if self._redis_client is not None:
            try:
                await publish_signed_market_metrics(self._redis_client, quote.symbol, metrics, ttl_seconds=self._ttl)
            except RedisUnavailableError:
                self._log.warning("quote_redis_write_failed", agent_id=self.agent_id, symbol=quote.symbol)
        await self.publish(_MARKET_DATA_L2_TOPIC, metrics.model_dump(mode="json"))
        self.stats["published"] += 1
        return True

    async def consume_once(self) -> None:
        """Drain the source until it ends or raises."""
        async for quote in self._source.stream():
            await self.handle_quote(quote)

    async def supervise(self, *, max_attempts: int | None = None) -> None:
        """Reconnect forever (or ``max_attempts`` times, for tests) with a capped exponential backoff; the backoff resets after a connection that delivered quotes."""
        failures = 0
        attempts = 0
        while max_attempts is None or attempts < max_attempts:
            attempts += 1
            before = self.stats["published"] + self.stats["throttled"] + self.stats["skipped"]
            try:
                await self.consume_once()
            except asyncio.CancelledError:
                raise
            except (QuoteSourceError, aiohttp.ClientError, websockets.WebSocketException, OSError, asyncio.TimeoutError) as exc:
                self._log.warning("quote_source_error", agent_id=self.agent_id, source=self._source.name, error=type(exc).__name__, detail=str(exc)[:120])
            delivered = self.stats["published"] + self.stats["throttled"] + self.stats["skipped"] > before
            failures = 0 if delivered else failures + 1
            self.stats["reconnects"] += 1
            await self._sleep(min(_MAX_BACKOFF_SECONDS, 2.0 ** min(failures, 6)))

    async def _heartbeat_loop(self) -> None:
        while True:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_WRITE_INTERVAL_SECONDS)

    async def run(self) -> None:
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")
        self._main_task = asyncio.current_task()
        self._install_signal_handlers()
        try:
            await self._connect_redis()
            await self.on_start()
            self._running = True
            self._log.info("agent_started", agent_id=self.agent_id, source=self._source.name)
            await asyncio.gather(self.supervise(), self._heartbeat_loop())
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

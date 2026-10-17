"""Forex (OANDA) and US stocks (Alpaca) quote feeds: parsers on the documented shapes, metrics maths, per-symbol throttling, cache + bus publication, supervised reconnection,
and the two real network paths against LOCAL servers that speak the documented protocols (an aiohttp streaming endpoint, a websockets endpoint).

Not tested here, and said so in the module docstring: the real OANDA/Alpaca servers. The first live run is that test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Generator, Iterator
from datetime import datetime, timezone
from typing import Any

import fakeredis.aioredis
import pytest
import websockets
from aiohttp import web

from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX
from src.agents.market_data import quote_agent as qa

OANDA_PRICE = {
    "type": "PRICE", "instrument": "EUR_USD", "time": "2026-10-17T14:30:00.123456789Z", "status": "tradeable",
    "bids": [{"price": "1.08500", "liquidity": 10_000_000}], "asks": [{"price": "1.08512", "liquidity": 5_000_000}],
    "closeoutBid": "1.08490", "closeoutAsk": "1.08522",
}
ALPACA_QUOTE = {"T": "q", "S": "AAPL", "bp": 230.10, "bs": 3, "ap": 230.14, "as": 1, "t": "2026-10-17T14:30:00.123456789Z"}


def _quote(symbol: str = "EUR_USD", bid: float = 1.0850, ask: float = 1.0851, bid_size: float = 4.0, ask_size: float = 1.0) -> qa.Quote:
    return qa.Quote(symbol, bid, ask, bid_size, ask_size, datetime(2026, 10, 17, 14, 30, tzinfo=timezone.utc))


# -- parsers -----------------------------------------------------------------------------------


def test_oanda_price_lines_are_parsed_and_everything_else_is_ignored() -> None:
    quote = qa.parse_oanda_line(json.dumps(OANDA_PRICE))
    assert quote is not None and (quote.symbol, quote.bid, quote.ask, quote.bid_size, quote.ask_size) == ("EUR_USD", 1.085, 1.08512, 10_000_000.0, 5_000_000.0)
    assert quote.time == datetime(2026, 10, 17, 14, 30, 0, 123456, tzinfo=timezone.utc)  # précision nanoseconde ramenée à la microseconde
    assert qa.parse_oanda_line(json.dumps({"type": "HEARTBEAT", "time": "2026-10-17T14:30:05.0Z"})) is None
    assert qa.parse_oanda_line(json.dumps({**OANDA_PRICE, "status": "non-tradeable"})) is None  # prix non négociable : pas de cotation
    assert qa.parse_oanda_line(json.dumps({"something": "else"})) is None
    for bad in ({**OANDA_PRICE, "bids": []}, {**OANDA_PRICE, "bids": [{"price": "0"}]}, {**OANDA_PRICE, "time": "pas une date"}):
        with pytest.raises((ValueError, KeyError, IndexError)):
            qa.parse_oanda_line(json.dumps(bad))


def test_alpaca_quotes_are_parsed_and_control_messages_ignored() -> None:
    quote = qa.parse_alpaca_message(ALPACA_QUOTE)
    assert quote is not None and (quote.symbol, quote.bid, quote.ask, quote.bid_size, quote.ask_size) == ("AAPL", 230.10, 230.14, 3.0, 1.0)
    assert qa.parse_alpaca_message({"T": "success", "msg": "connected"}) is None
    assert qa.parse_alpaca_message({"T": "subscription", "quotes": ["AAPL"]}) is None
    with pytest.raises((ValueError, KeyError)):
        qa.parse_alpaca_message({**ALPACA_QUOTE, "bp": -1})


def test_metrics_are_computed_from_the_top_of_book_and_a_crossed_book_is_refused() -> None:
    metrics = qa.quote_to_metrics(_quote(bid=1.0850, ask=1.0852, bid_size=3.0, ask_size=1.0))
    assert metrics is not None
    assert metrics.spread == pytest.approx(0.0002) and metrics.mid_price == pytest.approx(1.0851)
    assert metrics.order_book_imbalance == pytest.approx(0.5) and metrics.top_of_book_liquidity == 4.0   # (3-1)/(3+1)
    empty = qa.quote_to_metrics(_quote(bid_size=0.0, ask_size=0.0))
    assert empty is not None and empty.order_book_imbalance == 0.0
    assert qa.quote_to_metrics(_quote(bid=1.0860, ask=1.0850)) is None  # bid > ask : jamais publié


# -- agent -------------------------------------------------------------------------------------


class ScriptedSource:
    name = "scripted"

    def __init__(self, *scripts: list[qa.Quote | Exception]) -> None:
        self._scripts = list(scripts)
        self.connections = 0

    async def stream(self) -> AsyncIterator[qa.Quote]:
        script = self._scripts[min(self.connections, len(self._scripts) - 1)]
        self.connections += 1
        for item in script:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def make_agent(redis_client: RedisClient) -> Iterator[Any]:
    agents: list[qa.QuoteMarketDataAgent] = []

    async def no_sleep(_: float) -> None:
        return None

    def make(source: Any, **kwargs: Any) -> qa.QuoteMarketDataAgent:
        agent = qa.QuoteMarketDataAgent(
            source, settings=AppSettings(mock_mode=True, guardian=GuardianSettings(llm_provider="none")), manage_own_signals=False, manage_own_redis=False,
            sleep=no_sleep, **kwargs,
        )
        agent._redis_client = redis_client
        published: list[tuple[str, dict[str, Any]]] = []

        async def capture(topic: str, data: dict[str, Any]) -> None:
            published.append((topic, data))

        agent.publish = capture  # type: ignore[method-assign]
        agent.published = published  # type: ignore[attr-defined]
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent._publisher.close()
        agent._subscriber.close()
        agent._zmq_context.term()


@pytest.mark.asyncio
async def test_a_quote_is_cached_signed_and_published_and_each_symbol_is_throttled_separately(make_agent: Any, redis_client: RedisClient) -> None:
    agent = make_agent(ScriptedSource([]), min_publish_interval_seconds=1.0)
    assert await agent.handle_quote(_quote("EUR_USD"), now=100.0) is True
    assert await agent.handle_quote(_quote("EUR_USD"), now=100.4) is False      # trop tôt pour ce symbole
    assert await agent.handle_quote(_quote("XAU_USD", bid=2400.0, ask=2400.5), now=100.4) is True  # un autre symbole a son propre rythme
    assert await agent.handle_quote(_quote("EUR_USD"), now=101.2) is True
    assert agent.stats == {"published": 3, "throttled": 1, "skipped": 0, "reconnects": 0}
    assert [topic for topic, _ in agent.published] == ["MARKET_DATA_L2"] * 3
    cached = json.loads(await redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}XAU_USD"))
    assert cached["symbol"] == "XAU_USD" and cached["spread"] == pytest.approx(0.5)
    assert await redis_client.execute("ttl", f"{MARKET_METRICS_REDIS_KEY_PREFIX}EUR_USD") > 0  # TTL courte : une cotation périmée disparaît


@pytest.mark.asyncio
async def test_a_crossed_quote_is_skipped_and_never_reaches_redis_or_the_bus(make_agent: Any, redis_client: RedisClient) -> None:
    agent = make_agent(ScriptedSource([]))
    assert await agent.handle_quote(_quote(bid=1.0860, ask=1.0850), now=1.0) is False
    assert agent.stats["skipped"] == 1 and agent.published == []
    assert await redis_client.execute("get", f"{MARKET_METRICS_REDIS_KEY_PREFIX}EUR_USD") is None


@pytest.mark.asyncio
async def test_the_agent_reconnects_after_a_source_error_and_keeps_publishing(make_agent: Any) -> None:
    source = ScriptedSource([_quote(bid=1.0, ask=1.1), qa.QuoteSourceError("closed")], [_quote(bid=1.2, ask=1.3)])
    agent = make_agent(source, min_publish_interval_seconds=0.0)
    await agent.supervise(max_attempts=2)
    assert source.connections == 2 and agent.stats["published"] == 2 and agent.stats["reconnects"] == 2
    assert [d["mid_price"] for _, d in agent.published] == pytest.approx([1.05, 1.25])


@pytest.mark.asyncio
async def test_the_backoff_grows_without_data_and_resets_after_a_connection_that_delivered(make_agent: Any) -> None:
    waits: list[float] = []

    async def record(delay: float) -> None:
        waits.append(delay)

    source = ScriptedSource([qa.QuoteSourceError("refused")], [qa.QuoteSourceError("refused")], [qa.QuoteSourceError("refused")], [_quote(), qa.QuoteSourceError("x")], [qa.QuoteSourceError("refused")])
    agent = make_agent(source, min_publish_interval_seconds=0.0)
    agent._sleep = record
    await agent.supervise(max_attempts=5)
    assert waits == [2.0, 4.0, 8.0, 1.0, 2.0]  # 2^1, 2^2, 2^3 sans donnée ; 2^0 après une connexion qui a livré ; puis ça repart de 2^1


# -- sources against local servers speaking the documented protocols -------------------------------


async def _serve_oanda(handler: Any) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_get("/v3/accounts/{account}/pricing/stream", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


@pytest.mark.asyncio
async def test_the_oanda_source_streams_quotes_sends_the_bearer_token_and_skips_heartbeats() -> None:
    seen: dict[str, str] = {}

    async def handler(request: web.Request) -> web.StreamResponse:
        seen.update(auth=request.headers.get("Authorization", ""), instruments=request.query.get("instruments", ""), account=request.match_info["account"])
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write((json.dumps({"type": "HEARTBEAT", "time": "2026-10-17T14:30:05Z"}) + "\n").encode())
        await response.write((json.dumps(OANDA_PRICE) + "\n").encode())
        await response.write(b"\n")
        await response.write((json.dumps({**OANDA_PRICE, "instrument": "XAU_USD", "bids": [{"price": "2400.0", "liquidity": 1}], "asks": [{"price": "2400.4", "liquidity": 1}]}) + "\n").encode())
        return response

    runner, host = await _serve_oanda(handler)
    try:
        source = qa.OandaQuoteSource("101-001-1234567-001", "secret-token", ["EUR_USD", "XAU_USD"], host=host)
        quotes: list[qa.Quote] = []
        with pytest.raises(qa.QuoteSourceError, match="closed"):  # la fin du flux est une erreur : l'agent doit se reconnecter
            async for quote in source.stream():
                quotes.append(quote)
    finally:
        await runner.cleanup()
    assert [q.symbol for q in quotes] == ["EUR_USD", "XAU_USD"] and quotes[1].ask == 2400.4
    assert seen == {"auth": "Bearer secret-token", "instruments": "EUR_USD,XAU_USD", "account": "101-001-1234567-001"}


@pytest.mark.asyncio
async def test_the_oanda_source_refuses_loudly_on_http_errors_and_never_leaks_the_token() -> None:
    async def handler(request: web.Request) -> web.StreamResponse:
        return web.Response(status=401, text="Insufficient authorization")

    runner, host = await _serve_oanda(handler)
    try:
        source = qa.OandaQuoteSource("acc", "super-secret-token", ["EUR_USD"], host=host)
        with pytest.raises(qa.QuoteSourceError) as caught:
            async for _ in source.stream():
                pass
    finally:
        await runner.cleanup()
    assert "401" in str(caught.value) and "super-secret-token" not in str(caught.value)


@pytest.mark.asyncio
async def test_the_alpaca_source_authenticates_subscribes_and_parses_batched_messages() -> None:
    received: list[dict[str, Any]] = []

    async def server(connection: Any) -> None:
        await connection.send(json.dumps([{"T": "success", "msg": "connected"}]))
        for _ in range(2):
            received.append(json.loads(await connection.recv()))
        await connection.send(json.dumps([{"T": "success", "msg": "authenticated"}]))
        await connection.send(json.dumps([{"T": "subscription", "quotes": ["AAPL"]}, ALPACA_QUOTE, {**ALPACA_QUOTE, "S": "MSFT", "bp": 410.0, "ap": 410.05}]))
        await connection.close()

    async with websockets.serve(server, "127.0.0.1", 0) as ws_server:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        source = qa.AlpacaQuoteSource("KEY", "SECRET", ["AAPL", "MSFT"], url=f"ws://127.0.0.1:{port}")
        quotes: list[qa.Quote] = []
        with pytest.raises(qa.QuoteSourceError):
            async for quote in source.stream():
                quotes.append(quote)
    assert received == [{"action": "auth", "key": "KEY", "secret": "SECRET"}, {"action": "subscribe", "quotes": ["AAPL", "MSFT"]}]
    assert [q.symbol for q in quotes] == ["AAPL", "MSFT"] and quotes[1].bid == 410.0


@pytest.mark.asyncio
async def test_the_alpaca_source_raises_on_an_error_message_without_echoing_credentials() -> None:
    async def server(connection: Any) -> None:
        await connection.recv()
        await connection.send(json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]))
        await asyncio.sleep(0.2)

    async with websockets.serve(server, "127.0.0.1", 0) as ws_server:
        port = next(iter(ws_server.sockets)).getsockname()[1]
        source = qa.AlpacaQuoteSource("KEY123", "SECRET456", ["AAPL"], url=f"ws://127.0.0.1:{port}")
        with pytest.raises(qa.QuoteSourceError) as caught:
            async for _ in source.stream():
                pass
    assert "402" in str(caught.value) and "SECRET456" not in str(caught.value) and "KEY123" not in str(caught.value)


def test_sources_refuse_missing_credentials_or_instruments() -> None:
    with pytest.raises(ValueError):
        qa.OandaQuoteSource("", "token", ["EUR_USD"])
    with pytest.raises(ValueError):
        qa.OandaQuoteSource("acc", "token", [])
    with pytest.raises(ValueError):
        qa.AlpacaQuoteSource("key", "", ["AAPL"])
    with pytest.raises(ValueError):
        qa.OandaQuoteSource("acc", "token", ["EUR_USD"], environment="sandbox")

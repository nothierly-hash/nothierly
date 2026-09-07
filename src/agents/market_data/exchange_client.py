"""Resilient async WebSocket exchange client, built on ``ccxt.pro``.

:class:`AsyncExchangeClient` wraps a single ``ccxt.pro`` exchange instance
and exposes just the pieces a market-data agent needs: an async
initialization step (loading market metadata) and two auto-reconnecting
async generators, :meth:`~AsyncExchangeClient.watch_order_book` and
:meth:`~AsyncExchangeClient.watch_trades`.

``ccxt.pro`` ships no type stubs, so its symbols are treated as ``Any``
by MyPy (see the ``ccxt.*`` override in ``pyproject.toml``); every value
that crosses that boundary here is immediately given an explicit,
correctly-typed home (or handed to a Pydantic schema in
:mod:`core.schemas` in a later chapter) rather than left as ``Any``.

This module intentionally contains no trading/business logic: it only
knows how to stay connected and hand back raw exchange payloads.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from types import TracebackType
from typing import Any, Final, Literal

import ccxt.pro as ccxtpro
from ccxt.base.errors import AuthenticationError, ExchangeError, NetworkError, RequestTimeout

from core.logger import get_logger
from core.security import requires_system_active

# RequestTimeout is technically a subclass of NetworkError already, but it
# is listed explicitly (as required) for clarity at the call site about
# exactly which failure modes this client treats as recoverable.
_RECOVERABLE_CCXT_ERRORS: Final[tuple[type[Exception], ...]] = (
    NetworkError,
    ExchangeError,
    RequestTimeout,
)
# AuthenticationError is a subclass of ExchangeError above, but a real bug
# found via live testing (Chapter 13, Étape 5.6): retrying it "forever with
# backoff" like every other ExchangeError means a genuinely invalid API
# key never surfaces an error at all -- connect() just hangs, retrying a
# credential that can never become valid by waiting longer. Excluded from
# the recoverable set below and re-raised immediately instead.
_PERMANENT_CCXT_ERRORS: Final[tuple[type[Exception], ...]] = (AuthenticationError,)


class AsyncExchangeClient:
    """Async, auto-reconnecting wrapper around one ``ccxt.pro`` exchange.

    Every network-facing call (``load_markets``, ``watch_order_book``,
    ``watch_trades``) goes through the same resilient retry loop: on a
    recoverable ``ccxt`` error the failure is logged via structlog and
    retried after a capped exponential backoff (with jitter), forever,
    rather than propagating the error to the caller. Non-``ccxt``
    exceptions (programming errors, cancellation, ...) are never caught
    here and propagate normally.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool = False,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 60.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        """Construct (but do not yet connect) a ``ccxt.pro`` exchange client.

        Args:
            exchange_id: The ``ccxt.pro`` exchange identifier (e.g.
                ``"binance"``, ``"kraken"``). Must name a class exposed by
                ``ccxt.pro``.
            api_key: Optional API key. Not required for public market-data
                streams (order book, trades) on virtually every exchange;
                only needed for private/authenticated channels.
            api_secret: Optional API secret, paired with ``api_key``.
            sandbox: If ``True``, switches the exchange into its
                testnet/sandbox mode, where supported.
            base_delay_seconds: Initial reconnect delay after a failure.
            max_delay_seconds: Upper bound the backoff delay is capped at.
            backoff_multiplier: Factor the delay is multiplied by after
                each consecutive failure.

        Raises:
            ValueError: If ``exchange_id`` is not a known ``ccxt.pro``
                exchange.
        """
        exchange_class = getattr(ccxtpro, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"ccxt.pro has no exchange named {exchange_id!r}.")

        config: dict[str, Any] = {"enableRateLimit": True}
        if api_key is not None:
            config["apiKey"] = api_key
        if api_secret is not None:
            config["secret"] = api_secret

        # ccxt ships no type stubs; this is genuinely `Any` as far as
        # MyPy is concerned. Kept private so the untyped surface never
        # leaks past this class's own methods.
        self._exchange: Any = exchange_class(config)
        if sandbox:
            self._exchange.set_sandbox_mode(True)

        self._exchange_id = exchange_id
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._backoff_multiplier = backoff_multiplier
        self._log = get_logger(f"{__name__}.{exchange_id}")
        self._closed = False

    async def connect(self) -> None:
        """Asynchronously load this exchange's market metadata.

        Most ``ccxt.pro`` exchanges need their market metadata (symbol
        precision, limits, ...) loaded before ``watch_*`` calls can
        correctly normalize incoming WebSocket payloads. This retries
        indefinitely with exponential backoff on recoverable errors, so
        it only returns once the exchange is actually reachable.
        """
        await self._call_with_backoff(
            self._exchange.load_markets,
            operation="load_markets",
            context={},
        )
        self._log.info("exchange_connected", exchange=self._exchange_id)

    async def watch_order_book(self, symbol: str, limit: int = 5) -> AsyncIterator[dict[str, Any]]:
        """Yield order book updates for ``symbol`` forever.

        Internally calls ``ccxt.pro``'s ``watch_order_book`` in a loop:
        each iteration awaits the next WebSocket update. If the
        connection drops or the exchange returns a recoverable error, the
        failure is logged and the call is retried with exponential
        backoff — transparently to the caller, which just keeps
        iterating.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            limit: Requested order book depth from the exchange.

        Yields:
            The raw ``ccxt`` order book structure (a ``dict`` with
            ``bids``/``asks``/``timestamp``/... keys). Not yet validated
            against :class:`core.schemas.OrderBookSnapshot`; that
            conversion belongs to the agent that consumes this stream.
        """
        while not self._closed:
            yield await self._call_with_backoff(
                lambda: self._exchange.watch_order_book(symbol, limit),
                operation="watchOrderBook",
                context={"symbol": symbol, "limit": limit},
            )

    async def watch_trades(self, symbol: str) -> AsyncIterator[list[dict[str, Any]]]:
        """Yield trade updates for ``symbol`` forever.

        Same resilience behavior as :meth:`watch_order_book`: connection
        drops and recoverable ``ccxt`` errors are logged and retried with
        exponential backoff, transparently to the caller.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).

        Yields:
            The raw ``ccxt`` list of trade structures for each WebSocket
            update. Not yet validated against a Pydantic schema; that
            conversion belongs to the agent that consumes this stream.
        """
        while not self._closed:
            yield await self._call_with_backoff(
                lambda: self._exchange.watch_trades(symbol),
                operation="watchTrades",
                context={"symbol": symbol},
            )

    @requires_system_active
    async def create_market_order_by_cost(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        cost: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a market order sized by quote-currency cost, not base quantity.

        Guarded by the global Kill Switch (see :mod:`core.security`):
        refuses to place a new order while ``SYSTEM_STATUS`` is
        ``PANIC``. :meth:`cancel_all_orders` is deliberately *not*
        guarded the same way -- an emergency must still be able to
        unwind existing exposure.

        Deliberately uses ccxt's cost-based unified methods (e.g.
        Binance's ``quoteOrderQty``) instead of the base
        ``create_order(..., amount=...)`` call: ``amount`` is denominated
        in the *base* asset (e.g. BTC for BTC/USDT), which would require
        converting a notional/currency-denominated size into a quantity
        via the current price -- a value this client has no reliable way
        to obtain independently at call time. Sizing by cost sidesteps
        that conversion (and the staleness/race risk of using a
        possibly-outdated price for it) entirely.

        Unlike :meth:`watch_order_book`/:meth:`watch_trades`, this makes
        exactly **one** attempt and never auto-retries: retrying an order
        placement whose response was lost to a network blip risks
        submitting a *second*, duplicate order on top of one that may
        have actually succeeded. A ``client_order_id`` is still attached
        as defense-in-depth so that exchanges honoring it reject/dedupe
        an accidental retry at a higher layer instead of double-filling.
        Callers should treat a raised exception here as "unknown outcome,
        do not blindly retry" and surface it (e.g. to an operator or a
        circuit breaker), not swallow-and-resubmit it.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            cost: Notional order size in quote-currency units (e.g. USDT).
            client_order_id: Idempotency hint forwarded to the exchange.
                Defaults to a fresh UUID4 if omitted.

        Returns:
            The raw ccxt order structure returned by the exchange.

        Raises:
            ~core.security.KillSwitchActiveError: If the global Kill
                Switch is active.
            ccxt.NetworkError: On a connection failure or timeout.
            ccxt.ExchangeError: On an exchange-side rejection (e.g.
                insufficient balance, invalid symbol).
        """
        params = {"clientOrderId": client_order_id or str(uuid.uuid4())}
        order: dict[str, Any]
        if side == "buy":
            order = await self._exchange.create_market_buy_order_with_cost(symbol, cost, params)
        else:
            order = await self._exchange.create_market_sell_order_with_cost(symbol, cost, params)
        return order

    @requires_system_active
    async def create_market_order(
        self, symbol: str, side: Literal["buy", "sell"], amount: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        """Place a market order sized by base-asset quantity (unit-based), not quote cost.

        Added alongside :class:`~src.execution.broker_gateway.
        CcxtBrokerAdapter` (Chapter 13, Étape 5.6), which needs a
        unit-based sizing method to satisfy :class:`~src.execution.
        broker_gateway.BaseBroker`'s cross-broker ``place_market_order``
        contract -- :meth:`create_market_order_by_cost` above sizes by
        quote-currency notional instead, correct for its own original
        caller but not a fit for an abstraction that also has to match
        :class:`~src.agents.supervisor.broker_client.OandaClient`'s
        unit-based ``create_market_order(instrument, units)``. Same
        single-attempt, Kill-Switch-gated, ``client_order_id``-tagged
        contract as :meth:`create_market_order_by_cost` -- see that
        method's docstring for why retrying a lost order-placement
        response is unsafe.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            amount: Order size in base-asset units (e.g. BTC for
                BTC/USDT). Must be pre-computed by the caller -- this
                method does no price lookup or conversion.
            client_order_id: Idempotency hint forwarded to the exchange.
                Defaults to a fresh UUID4 if omitted.

        Returns:
            The raw ccxt order structure returned by the exchange.
        """
        params = {"clientOrderId": client_order_id or str(uuid.uuid4())}
        order: dict[str, Any] = await self._exchange.create_order(symbol, "market", side, amount, None, params)
        return order

    @requires_system_active
    async def create_limit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a resting limit order -- the real Maker-side primitive this client lacked entirely before.

        Every order-placement method on this class before this one was
        market-only (immediate taker fill); a market-making system built
        on Avellaneda-Stoikov quoting fundamentally needs to *rest*
        orders at a chosen price, not just sweep the book. Same
        single-attempt, Kill-Switch-gated, ``client_order_id``-tagged
        contract as :meth:`create_market_order` -- see
        :meth:`create_market_order_by_cost`'s docstring for why.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            side: ``"buy"`` or ``"sell"``.
            amount: Order size in base-asset units.
            price: Limit price.
            client_order_id: Idempotency hint forwarded to the exchange.
                Defaults to a fresh UUID4 if omitted.

        Returns:
            The raw ccxt order structure returned by the exchange.
        """
        params = {"clientOrderId": client_order_id or str(uuid.uuid4())}
        order: dict[str, Any] = await self._exchange.create_order(symbol, "limit", side, amount, price, params)
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel one specific open order by ID.

        Unlike :meth:`create_market_order`/:meth:`create_limit_order`,
        cancellation is naturally idempotent (see :meth:`cancel_all_orders`'s
        docstring for the same reasoning) -- retried indefinitely via
        :meth:`_call_with_backoff` rather than making one attempt.

        Args:
            order_id: Exchange-assigned order ID to cancel.
            symbol: Trading pair the order was placed on -- required by
                most exchanges' cancel endpoints even though the order
                ID alone is technically unique.

        Returns:
            The raw ccxt cancelled-order structure.
        """
        result: dict[str, Any] = await self._call_with_backoff(
            lambda: self._exchange.cancel_order(order_id, symbol),
            operation="cancelOrder",
            context={"order_id": order_id, "symbol": symbol},
        )
        return result

    async def get_current_price(self, symbol: str) -> float:
        """Fetch the current last-traded (or mid, if unavailable) price for ``symbol`` via one-shot REST.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).

        Returns:
            The current price. Retried indefinitely on recoverable
            errors (a plain price lookup carries none of order
            placement's double-submission risk).
        """
        ticker: dict[str, Any] = await self._call_with_backoff(
            lambda: self._exchange.fetch_ticker(symbol), operation="fetchTicker", context={"symbol": symbol}
        )
        price = ticker.get("last") or ticker.get("close")
        if price is None:
            bid, ask = ticker.get("bid"), ticker.get("ask")
            if bid is not None and ask is not None:
                price = (float(bid) + float(ask)) / 2.0
        if price is None:
            raise ValueError(f"Ticker for {symbol!r} on {self._exchange_id!r} has no last/close/bid+ask price.")
        return float(price)

    def get_taker_fee_rate(self, symbol: str, *, default: float = 0.001) -> float:
        """Real taker-fee rate (fraction of notional, e.g. ``0.001`` = 0.1%) for ``symbol`` -- for Smart Order Router's fee-inclusive effective-price routing.

        Reads ``ccxt``'s own real, unified per-market fee schedule
        (populated by :meth:`connect`'s ``load_markets()`` call), not a
        value invented here -- see a real, live check of this exact
        field (``exchange.describe()["fees"]["trading"]["taker"]``,
        e.g. Binance spot = ``0.001``) before this method was written.
        Synchronous: unlike every other accessor in this class, this
        needs no network call -- ``load_markets()`` already fetched and
        cached it.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            default: Used only if this exchange's market metadata for
                ``symbol`` genuinely has no ``taker`` fee reported (rare,
                but real for exchanges without static public fee
                schedules) -- a documented fallback, not a silent
                per-market fabrication.
        """
        market = self._exchange.markets.get(symbol) if self._exchange.markets else None
        taker = market.get("taker") if market is not None else None
        return float(taker) if taker is not None else default

    async def get_account_balance(self) -> float:
        """Fetch total account equity in the exchange's reported quote currency, via one-shot REST.

        Returns:
            The ``"total"`` free+used balance across every asset ccxt's
            unified ``fetch_balance`` reports, summed. A real
            approximation, not a precise USD-denominated net worth
            (ccxt's unified balance does not itself convert every asset
            to one currency) -- callers needing a single-currency figure
            must convert themselves using :meth:`get_current_price`.
        """
        balance: dict[str, Any] = await self._call_with_backoff(
            self._exchange.fetch_balance, operation="fetchBalance", context={}
        )
        total = balance.get("total", {})
        if not isinstance(total, dict):
            return 0.0
        return sum(float(amount) for amount in total.values() if amount)

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Real, honest interpretation of "open positions" for **spot** trading -- see caller's own docstring.

        Added alongside :mod:`risk.cross_margin_engine` (Chapter 13,
        Étape 5.8). ``ccxt``'s unified ``fetch_positions`` is a real
        method, but it only returns anything for margin/futures/swap
        markets -- this project only ever places **spot** orders (see
        :meth:`create_market_order`/:meth:`create_limit_order`), which
        have no "position" in that sense at all. The honest equivalent
        for spot is a non-zero balance in any non-quote asset -- that
        *is* your real, held exposure, just never leveraged/marginable.
        Every returned entry has no natural "entry price" (a spot
        balance doesn't remember what it was bought at the way a
        margin position does), so ``avg_price`` is always ``None`` here
        -- never fabricated.

        Returns:
            One dict per non-zero, non-quote-currency balance:
            ``symbol`` (bare asset code, e.g. ``"BTC"``), ``amount``,
            ``avg_price`` (always ``None`` -- see above).
        """
        balance: dict[str, Any] = await self._call_with_backoff(
            self._exchange.fetch_balance, operation="fetchBalance", context={}
        )
        total = balance.get("total", {})
        if not isinstance(total, dict):
            return []
        return [
            {"symbol": asset, "amount": float(amount), "avg_price": None}
            for asset, amount in total.items()
            if amount and abs(float(amount)) > 1e-12
        ]

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        """Fetch one order book snapshot via one-shot REST (not a stream).

        Distinct from :meth:`watch_order_book`: that method is an
        infinite, auto-reconnecting WebSocket generator for a
        long-lived streaming consumer (:class:`~src.agents.market_data.
        agent.MarketDataAgent`); this is a single point-in-time REST
        call for a caller that just needs "the book right now" (e.g.
        :class:`~src.execution.broker_gateway.CcxtBrokerAdapter`'s
        ``get_orderbook_snapshot``), without holding a WebSocket
        connection open.

        Args:
            symbol: Exchange trading pair symbol (e.g. ``"BTC/USDT"``).
            limit: Requested order book depth from the exchange.

        Returns:
            The raw ccxt order book structure (same shape as
            :meth:`watch_order_book`'s yielded values) -- not yet
            validated against :class:`core.schemas.OrderBookSnapshot`;
            see :mod:`src.execution.data_normalizer` for that
            conversion.
        """
        book: dict[str, Any] = await self._call_with_backoff(
            lambda: self._exchange.fetch_order_book(symbol, limit),
            operation="fetchOrderBook",
            context={"symbol": symbol, "limit": limit},
        )
        return book

    async def cancel_all_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Cancel every open order, optionally scoped to one symbol.

        Unlike order *placement*, cancellation is naturally idempotent
        (cancelling an already-cancelled/filled order is a no-op on
        essentially every exchange), so -- unlike
        :meth:`create_market_order_by_cost` -- this *does* go through
        :meth:`_call_with_backoff`'s indefinite retry-with-backoff. This
        is deliberate: this method is the kill switch's emergency action,
        and it must keep trying until it actually reaches the exchange
        rather than give up after one network blip.

        Args:
            symbol: If given, cancel only this symbol's open orders;
                otherwise cancel every open order on the account.

        Returns:
            The raw ccxt list of cancelled-order structures.
        """
        cancelled: list[dict[str, Any]] = await self._call_with_backoff(
            lambda: self._exchange.cancel_all_orders(symbol),
            operation="cancelAllOrders",
            context={"symbol": symbol},
        )
        return cancelled

    async def _call_with_backoff(
        self,
        call: Callable[[], Awaitable[Any]],
        *,
        operation: str,
        context: dict[str, Any],
    ) -> Any:
        """Await ``call()``, retrying forever with backoff on *recoverable* ccxt errors.

        Args:
            call: A zero-argument callable returning an awaitable (e.g.
                ``self._exchange.load_markets`` or a lambda wrapping a
                parameterized ``watch_*`` call).
            operation: Short label identifying the call, for logging.
            context: Extra structured fields (e.g. ``symbol``) to attach
                to log events for this call.

        Returns:
            Whatever ``call()`` eventually returns, once successful.

        Raises:
            ccxt.base.errors.AuthenticationError: Immediately, never
                retried -- see :data:`_PERMANENT_CCXT_ERRORS`. A real
                bug found via live testing (Chapter 13, Étape 5.6): an
                invalid API key can never become valid by waiting
                longer, so retrying it "forever with backoff" like a
                transient network error just hangs the caller with no
                visible failure, ever.
        """
        delay = self._base_delay_seconds
        while True:
            try:
                return await call()
            except _PERMANENT_CCXT_ERRORS as exc:
                self._log.error(
                    "exchange_permanent_error_not_retrying",
                    exchange=self._exchange_id,
                    operation=operation,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    **context,
                )
                raise
            except _RECOVERABLE_CCXT_ERRORS as exc:
                jittered_delay = delay * random.uniform(0.5, 1.5)
                self._log.warning(
                    "exchange_ws_error_reconnecting",
                    exchange=self._exchange_id,
                    operation=operation,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    delay_seconds=round(jittered_delay, 2),
                    **context,
                )
                await asyncio.sleep(jittered_delay)
                delay = min(delay * self._backoff_multiplier, self._max_delay_seconds)

    async def close(self) -> None:
        """Close the underlying WebSocket connections and HTTP session.

        Safe to call multiple times; also stops any in-progress
        :meth:`watch_order_book`/:meth:`watch_trades` generator loops.
        """
        if self._closed:
            return
        self._closed = True
        await self._exchange.close()
        self._log.info("exchange_closed", exchange=self._exchange_id)

    async def __aenter__(self) -> AsyncExchangeClient:
        """Connect, real resource cleanup guaranteed even on failure.

        Same real bug (and fix) as
        :meth:`~src.execution.broker_gateway.BaseBroker.__aenter__` --
        see that method's own docstring. ``async with`` never calls
        ``__aexit__`` when ``__aenter__`` itself raises, which used to
        leak this object's real ``aiohttp.ClientSession`` (opened at
        construction time, before ``connect()``'s ``load_markets()``
        call can fail) whenever a caller used this class directly via
        ``async with``. No real caller currently does (``
        CcxtBrokerAdapter`` calls ``connect()``/``close()`` directly,
        not through this context manager) -- fixed anyway for the same
        real correctness, not left as a dormant landmine for the first
        future caller that does.
        """
        try:
            await self.connect()
        except Exception:
            await self.close()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

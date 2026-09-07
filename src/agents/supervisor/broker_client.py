"""Async wrapper around OANDA's v20 REST broker API.

OANDA's official ``v20`` Python library is **synchronous** (blocking
socket I/O) -- there is no async variant. Every call in
:class:`OandaClient` is therefore dispatched to a dedicated
``ThreadPoolExecutor`` via ``loop.run_in_executor``, the exact same
non-blocking-bridge mechanic :class:`~src.agents.nlp.sentiment_engine.
SentimentEngine` (chapter 4.2) uses for FinBERT inference: there the
blocking work was CPU-bound compute, here it's blocking network I/O, but
the reason it matters -- keeping the event loop free to service every
other agent's I/O while this call is in flight -- is identical.

This module contains no trading/decision logic: it only knows how to
place/cancel orders and read account state on OANDA, and how to fail
loudly and specifically (timeout vs. rejection vs. any other error) when
something goes wrong.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import v20

from core.logger import get_logger

_log = get_logger(__name__)

_PRACTICE_HOSTNAME = "api-fxpractice.oanda.com"
_LIVE_HOSTNAME = "api-fxtrade.oanda.com"


class OandaError(Exception):
    """Base class for every error this client raises."""


class OandaTimeoutError(OandaError):
    """Raised when a request to OANDA does not complete within the configured timeout."""


class OandaRequestError(OandaError):
    """Raised for any non-2xx HTTP response, or a network-level failure below that."""


class OandaOrderRejectedError(OandaError):
    """Raised when OANDA's matching engine rejects an order.

    This is distinct from :class:`OandaRequestError`: OANDA can return
    HTTP 201 (the order was successfully *submitted*) while the order
    itself is still rejected by the engine (e.g. insufficient margin,
    market halted, invalid instrument state) -- the HTTP status alone
    does not tell you whether the order actually filled.
    """


class OandaClient:
    """Async-friendly wrapper around one OANDA v20 account.

    Must be used as an async context manager (or have :meth:`connect`
    awaited) before making requests, matching this codebase's convention
    for every other async-resource-owning client (see
    :class:`~core.http_client.ResilientHttpClient`,
    :class:`~src.agents.market_data.exchange_client.AsyncExchangeClient`).
    """

    def __init__(
        self,
        *,
        api_token: str,
        account_id: str,
        environment: Literal["practice", "live"] = "practice",
        request_timeout_seconds: float = 10.0,
        max_workers: int = 2,
        context: v20.Context | None = None,
    ) -> None:
        """Configure the client.

        Args:
            api_token: OANDA v20 personal access token.
            account_id: OANDA account ID (e.g. ``"101-004-1234567-001"``).
            environment: ``"practice"`` (paper trading, the default) or
                ``"live"`` (real money).
            request_timeout_seconds: Per-request timeout.
            max_workers: Size of the dedicated ``ThreadPoolExecutor``
                every blocking ``v20`` call is dispatched to.
            context: Inject a pre-built ``v20.Context`` (e.g. a test
                double). Defaults to constructing a real one against
                OANDA's REST API for ``environment``.
        """
        hostname = _PRACTICE_HOSTNAME if environment == "practice" else _LIVE_HOSTNAME
        self._context = context or v20.Context(
            hostname, token=api_token, datetime_format="RFC3339"
        )
        self._account_id = account_id
        self._environment = environment
        self._request_timeout_seconds = request_timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="oanda-client")
        self._closed = False

        if environment == "live":
            _log.warning(
                "oanda_client_live_environment",
                account_id=account_id,
                detail="This client will place REAL orders with REAL funds.",
            )

    async def connect(self) -> None:
        """No-op, present for lifecycle-convention consistency.

        ``v20.Context`` has no separate connect step -- each request
        opens its own HTTPS connection. Kept as an async method (rather
        than omitted) so this class matches the ``connect()``/``close()``
        /``async with`` shape every other client in this codebase has.
        """
        return None

    async def close(self) -> None:
        """Shut down the dedicated thread pool. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        await asyncio.get_running_loop().run_in_executor(None, self._executor.shutdown, True)

    async def __aenter__(self) -> OandaClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- Account state -----------------------------------------------

    async def get_available_margin(self) -> float:
        """Fetch the account's currently available margin.

        Returns:
            Available margin, in the account's home currency.

        Raises:
            OandaTimeoutError: On a network timeout.
            OandaRequestError: On any other request failure.
        """
        response = await self._call(self._context.account.summary, self._account_id)
        account = response.body.get("account")
        if account is None:
            raise OandaRequestError(f"OANDA account summary response missing 'account': {response.body}")
        return float(account.marginAvailable)

    async def get_account_balance(self) -> float:
        """Fetch the account's total equity (distinct from :meth:`get_available_margin`).

        ``balance`` is total account equity; ``marginAvailable`` is that
        minus margin already committed to open positions. Risk-based
        position sizing (risk_amount = balance * risk_pct) needs the
        former -- sizing off available margin would shrink every new
        order's risk budget by however much existing positions already
        happen to be using, which has nothing to do with how much of the
        account's own capital this trade should risk.

        Returns:
            Account balance, in the account's home currency.

        Raises:
            OandaTimeoutError: On a network timeout.
            OandaRequestError: On any other request failure.
        """
        response = await self._call(self._context.account.summary, self._account_id)
        account = response.body.get("account")
        if account is None:
            raise OandaRequestError(f"OANDA account summary response missing 'account': {response.body}")
        return float(account.balance)

    async def get_margin_summary(self) -> dict[str, float]:
        """Fetch ``balance``/``marginUsed``/``marginAvailable`` from one real account-summary call.

        Added alongside :mod:`risk.cross_margin_engine` (Chapter 13,
        Étape 5.8): that engine needs a precise, real ``marginUsed``
        figure, not an approximation derived from
        :meth:`get_account_balance`/:meth:`get_available_margin`
        separately (``balance`` isn't simply ``marginUsed +
        marginAvailable`` on OANDA -- unrealized P&L folds into
        ``balance`` differently). All three come from OANDA's real
        account-summary response in a single call, exactly as it
        reports them, not derived/estimated here.

        Returns:
            ``{"balance": ..., "margin_used": ..., "margin_available": ...}``,
            all in the account's home currency.

        Raises:
            OandaTimeoutError: On a network timeout.
            OandaRequestError: On any other request failure.
        """
        response = await self._call(self._context.account.summary, self._account_id)
        account = response.body.get("account")
        if account is None:
            raise OandaRequestError(f"OANDA account summary response missing 'account': {response.body}")
        return {
            "balance": float(account.balance),
            "margin_used": float(account.marginUsed),
            "margin_available": float(account.marginAvailable),
        }

    async def get_current_price(self, instrument: str) -> float:
        """Fetch the current mid price for ``instrument``.

        Args:
            instrument: OANDA instrument name (e.g. ``"EUR_USD"``).

        Returns:
            The mid price: ``(best_bid + best_ask) / 2``.

        Raises:
            OandaTimeoutError: On a network timeout.
            OandaRequestError: On any other request failure, or if no
                price is currently available for ``instrument``.
        """
        response = await self._call(
            self._context.pricing.get, self._account_id, instruments=instrument
        )
        prices = response.body.get("prices") or []
        if not prices:
            raise OandaRequestError(f"OANDA returned no current price for {instrument!r}.")

        price = prices[0]
        best_bid = float(price.bids[0].price)
        best_ask = float(price.asks[0].price)
        return (best_bid + best_ask) / 2.0

    async def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch every currently open position on this account.

        Added alongside :mod:`risk.cross_margin_engine` (Chapter 13,
        Étape 5.8), which needs a real, cross-broker view of open
        exposure -- this client had no position-listing method before
        (only order placement/cancellation and account-level
        balance/margin totals). OANDA's v20 API's real
        ``position.list_open`` endpoint, not invented here.

        Returns:
            One dict per open position: ``instrument``, signed ``units``
            (positive long, negative short -- OANDA reports long/short
            unit counts separately; this collapses them into this
            client's own single-signed-quantity convention, matching
            every other method here), ``avg_price``, ``unrealized_pnl``.

        Raises:
            OandaTimeoutError: On a network timeout.
            OandaRequestError: On any other request failure.
        """
        response = await self._call(self._context.position.list_open, self._account_id)
        positions = response.body.get("positions") or []

        result: list[dict[str, Any]] = []
        for position in positions:
            long_units = float(getattr(position.long, "units", 0) or 0)
            short_units = float(getattr(position.short, "units", 0) or 0)
            signed_units = long_units + short_units  # OANDA reports short units already negative
            side_leg = position.long if long_units != 0 else position.short
            result.append(
                {
                    "instrument": position.instrument,
                    "units": signed_units,
                    "avg_price": float(getattr(side_leg, "averagePrice", 0) or 0),
                    "unrealized_pnl": float(getattr(position, "unrealizedPL", 0) or 0),
                }
            )
        return result

    # -- Order execution -----------------------------------------------

    async def create_market_order(
        self, instrument: str, units: int, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        """Place a MARKET order.

        **Not** Kill-Switch-guarded here, deliberately, and unlike this
        method's chapter-8-era version: OANDA has no separate
        "close position" endpoint -- both opening *and* closing a
        position go through this exact call (an offsetting market
        order). Blanket-blocking it here would also block a stop-loss/
        take-profit/agent-veto close during a PANIC, stranding open
        exposure instead of letting it be torn down -- the opposite of
        what the switch exists for (the same reasoning
        :meth:`cancel_all_orders` already followed). The Kill Switch
        gate now lives one layer up, on the *entry-only* call site in
        :class:`~src.agents.execution.agent.ExecutionAgent`
        (``_place_order``); every close path calls this method directly,
        ungated.

        Args:
            instrument: OANDA instrument name (e.g. ``"EUR_USD"``).
            units: Positive to buy, negative to sell -- OANDA encodes
                direction in the sign of ``units``, unlike this system's
                own ``BUY``/``SELL`` action enum.
            client_order_id: Idempotency hint attached as a client
                extension. Defaults to a fresh UUID4.

        Returns:
            The OANDA ``orderFillTransaction``, as a plain dict (id,
            price, units, time, ...).

        Raises:
            ValueError: If ``units`` is ``0``.
            OandaTimeoutError: On a network timeout.
            OandaOrderRejectedError: If OANDA's matching engine rejected
                the order (see the class docstring on why this is
                distinct from an HTTP-level failure).
            OandaRequestError: On any other request failure.
        """
        if units == 0:
            raise ValueError("units must not be 0.")

        order_id = client_order_id or str(uuid.uuid4())
        response = await self._call(
            self._context.order.market,
            self._account_id,
            instrument=instrument,
            units=units,
            timeInForce="FOK",
            clientExtensions={"id": order_id},
        )

        reject_transaction = response.body.get("orderRejectTransaction")
        if reject_transaction is not None:
            reason = getattr(reject_transaction, "rejectReason", "UNKNOWN")
            _log.warning(
                "oanda_order_rejected",
                instrument=instrument,
                units=units,
                reject_reason=reason,
            )
            raise OandaOrderRejectedError(f"OANDA rejected the order for {instrument!r}: {reason}")

        fill_transaction = response.body.get("orderFillTransaction")
        if fill_transaction is None:
            # Submitted but not immediately filled (e.g. queued) -- FOK
            # market orders should not normally reach this state, but
            # surfacing it as a rejection is safer than silently
            # returning an empty/partial result to the caller.
            raise OandaOrderRejectedError(
                f"OANDA order for {instrument!r} did not produce a fill "
                f"(response: {list(response.body.keys())})."
            )

        _log.info(
            "oanda_order_filled",
            instrument=instrument,
            units=units,
            order_id=getattr(fill_transaction, "id", order_id),
            price=getattr(fill_transaction, "price", None),
        )
        return {
            "id": getattr(fill_transaction, "id", order_id),
            "instrument": instrument,
            "units": units,
            "price": getattr(fill_transaction, "price", None),
            "time": getattr(fill_transaction, "time", None),
        }

    async def cancel_all_orders(self, instrument: str | None = None) -> list[dict[str, Any]]:
        """Cancel every pending order, optionally scoped to one instrument.

        OANDA has no single "cancel all" endpoint (unlike ccxt's unified
        API used elsewhere in this system): this lists every pending
        order, then cancels each individually. This is the Kill Switch's
        emergency action when using this client, so every cancellation
        that fails is logged and the rest are still attempted -- a single
        stuck order must never prevent the others from being cancelled.

        Args:
            instrument: If given, cancel only this instrument's pending
                orders; otherwise cancel every pending order on the
                account.

        Returns:
            One entry per order that was successfully cancelled.

        Raises:
            OandaTimeoutError: If *listing* pending orders times out.
            OandaRequestError: If *listing* pending orders fails outright.
        """
        response = await self._call(self._context.order.list_pending, self._account_id)
        pending_orders = response.body.get("orders") or []
        if instrument is not None:
            pending_orders = [o for o in pending_orders if getattr(o, "instrument", None) == instrument]

        cancelled: list[dict[str, Any]] = []
        for order in pending_orders:
            order_id = getattr(order, "id", None)
            if order_id is None:
                continue
            try:
                await self._call(self._context.order.cancel, self._account_id, order_id)
                cancelled.append({"id": order_id, "instrument": getattr(order, "instrument", None)})
            except OandaError:
                _log.exception("oanda_order_cancel_failed", order_id=order_id)

        return cancelled

    # -- Internal call wrapper -------------------------------------------

    async def _call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch one blocking ``v20`` call to the executor, with a timeout.

        Args:
            func: A bound ``v20.Context`` method (e.g.
                ``self._context.order.market``).
            *args: Positional arguments forwarded to ``func``.
            **kwargs: Keyword arguments forwarded to ``func``.

        Returns:
            The ``v20`` response object (with ``.status`` and ``.body``).

        Raises:
            OandaTimeoutError: If the call does not complete within
                ``request_timeout_seconds``.
            OandaRequestError: If the call raises any other exception, or
                returns a non-2xx HTTP status.
        """
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)

        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(self._executor, call), timeout=self._request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            _log.warning("oanda_request_timeout", timeout_seconds=self._request_timeout_seconds)
            raise OandaTimeoutError(
                f"OANDA request timed out after {self._request_timeout_seconds}s."
            ) from exc
        except OandaError:
            raise
        except Exception as exc:
            _log.warning("oanda_request_failed", error_type=type(exc).__name__, error=str(exc))
            raise OandaRequestError(f"OANDA request failed: {exc}") from exc

        if response.status // 100 != 2:
            _log.warning("oanda_request_http_error", status=response.status)
            raise OandaRequestError(f"OANDA request failed: HTTP {response.status}.")

        return response

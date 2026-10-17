"""Real, public, read-only Deribit options-chain market data connector, via ``ccxt``.

Deribit is, as of this module's writing, the one real exchange this
project's own :mod:`~src.agents.market_data.exchange_client` dependency
(``ccxt``/``ccxt.pro``) confirms real, live support for via its unified
``fetch_option_chain`` method (verified directly against the real,
live public endpoint before this module was written -- ``ex.has
['fetchOptionChain']`` is ``True`` for ``ccxt.pro.deribit``, and a real
call returns hundreds of real, currently-listed BTC contracts).

**Real, disclosed scope boundary** -- same convention as
:mod:`~src.agents.supervisor.options_hedging_manager`'s own module
docstring: this is a **read-only** market-data connector. It never
places, modifies, or cancels a real order; it exists purely to feed
:class:`~src.agents.supervisor.options_hedging_manager.
OptionsHedgingManager`'s ``volatility``/pricing inputs with real,
live market-implied numbers instead of a caller's manual guess, so a
(still always paper-filled) option position opened via
:meth:`~src.agents.execution.agent.ExecutionAgent.open_option_position`
can be priced against a real, current market-implied volatility.

**Real, disclosed unit/parsing conventions**:

- Deribit quotes options *inverse* (premium denominated in the
  underlying coin, e.g. BTC, not USD) -- :attr:`OptionQuote.
  mark_price_quote_ccy` is this project's own real conversion
  (``mark_price_in_coin * underlying_price``) into an approximate
  USD-equivalent premium, not a value Deribit itself reports directly.
- Deribit's real, documented contract-naming convention
  (``{CURRENCY}-{DMMMYY}-{STRIKE}-{C|P}``, e.g. ``"BTC-16SEP26-77500-
  C"``) is parsed directly from ``info["instrument_name"]`` rather than
  ccxt's own unified ``symbol`` field, since it is Deribit's own
  human-readable spec and changes shape less often across ccxt
  versions.
- Every Deribit option **expires at 08:00 UTC** on its expiry date --
  a real, documented exchange convention (Deribit's own contract
  specifications), not a guessed default; this is baked into
  :data:`_DERIBIT_EXPIRY_HOUR_UTC` rather than left as midnight.
- :class:`~core.schemas.OptionContract.underlying` is populated here as
  ``f"{CURRENCY}/USDT"`` (e.g. ``"BTC/USDT"``) -- a real, disclosed
  **approximation**: Deribit's actual underlying index is a BTC-USD
  (not USDT) synthetic. Treating USDT as at-parity with USD is the
  same simplification this project's own spot/CFD trading already
  makes everywhere it treats ``BTC/USDT`` prices as USD-denominated;
  it is *not* exact (USDT can and does trade at a small premium/
  discount to USD), and is never assumed to be within this module --
  it is surfaced here as a known, bounded approximation.
- Deribit's real ``mark_iv`` field is a percentage (e.g. ``33.84`` for
  a real 33.84% annualized mark implied volatility) -- divided by
  ``100`` here so :attr:`OptionQuote.mark_implied_volatility` is
  already in the decimal-fraction units every ``core.quant_math``
  Black-Scholes function expects (e.g. ``0.3384``), never left for a
  caller to silently misinterpret as a raw percentage.

Any contract this module cannot honestly parse or price (a malformed
``instrument_name``, a missing real mark price) is skipped, logged at
debug level with a count -- never fabricated -- exactly the same
honest-skip discipline :meth:`OptionsHedgingManager.
compute_portfolio_greeks` already applies to missing market data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal

import ccxt.pro as ccxtpro

from core.logger import get_logger
from core.schemas import OptionContract

_log = get_logger(__name__)

# Deribit's own instrument-name date format, e.g. "16SEP26" -> day 16, Sep, 2026.
_DERIBIT_INSTRUMENT_DATE_FORMAT: Final[str] = "%d%b%y"
# Deribit options expire/settle at 08:00 UTC -- see module docstring.
_DERIBIT_EXPIRY_HOUR_UTC: Final[int] = 8
_TYPE_CODE: Final[dict[str, Literal["call", "put"]]] = {"C": "call", "P": "put"}


@dataclass(frozen=True)
class OptionQuote:
    """One real, live Deribit option market quote.

    Attributes:
        contract: The real, parsed :class:`~core.schemas.OptionContract`.
        underlying_price: Real current price of the underlying, as
            reported by Deribit alongside this quote.
        mark_price_quote_ccy: Real mark price, converted from Deribit's
            native coin-denominated quoting into an approximate
            quote-currency premium (see module docstring).
        mark_implied_volatility: Real mark IV as a decimal fraction
            (e.g. ``0.35`` for 35%), or ``None`` if Deribit reported
            none for this contract (illiquid / no recent mark) --
            honestly absent, never a fabricated fallback.
        open_interest: Real open interest, in contracts.
    """

    contract: OptionContract
    underlying_price: float
    mark_price_quote_ccy: float
    mark_implied_volatility: float | None
    open_interest: float


class DeribitOptionsChainClient:
    """Real, read-only, public Deribit options-chain connector, via ``ccxt.pro``.

    Uses no API key/secret -- ``fetch_option_chain`` is a real public
    endpoint, same as :meth:`~src.agents.market_data.exchange_client.
    AsyncExchangeClient.get_current_price`'s ``fetch_ticker``. Deliberately
    does **not** reuse :class:`~src.agents.market_data.exchange_client.
    AsyncExchangeClient`: that class's resilience contract (infinite
    retry-with-backoff on every call) is built for a long-lived streaming
    consumer; this connector is a simple, bounded, one-shot REST fetch an
    operator or dashboard calls on demand, where an unbounded retry loop
    would be the wrong failure mode (a real, temporary Deribit outage
    should surface as a real, visible error here, not hang forever).
    """

    def __init__(self) -> None:
        self._exchange = ccxtpro.deribit({"enableRateLimit": True})
        self._log = get_logger(__name__)

    async def fetch_chain(self, underlying_currency: str) -> list[OptionQuote]:
        """Fetch the real, live, full option chain currently listed for one currency.

        Args:
            underlying_currency: Deribit currency code, e.g. ``"BTC"``
                or ``"ETH"``.

        Returns:
            Every real contract this call could honestly parse and
            price (see module docstring for what gets skipped, and
            why). Order is whatever Deribit/ccxt returned it in --
            not sorted.
        """
        raw_chain = await self._exchange.fetch_option_chain(underlying_currency)
        quotes: list[OptionQuote] = []
        skipped = 0
        for option in raw_chain.values():
            quote = self._parse_option(option)
            if quote is None:
                skipped += 1
                continue
            quotes.append(quote)
        if skipped:
            self._log.debug(
                "options_chain_contracts_skipped", underlying=underlying_currency, skipped=skipped,
                total=len(raw_chain),
            )
        return quotes

    @staticmethod
    def _parse_option(option: dict[str, Any]) -> OptionQuote | None:
        """Parse one real raw ``ccxt`` option entry, honestly returning ``None`` if it cannot be trusted."""
        info = option.get("info", {})
        instrument_name = info.get("instrument_name")
        underlying_price = option.get("underlyingPrice")
        if underlying_price is None:
            underlying_price = info.get("underlying_price")
        mark_price_native = option.get("markPrice")
        if mark_price_native is None:
            mark_price_native = info.get("mark_price")

        if not instrument_name or underlying_price is None or mark_price_native is None:
            return None

        parts = instrument_name.split("-")
        if len(parts) != 4:
            return None
        currency, expiry_raw, strike_raw, type_code = parts

        option_type = _TYPE_CODE.get(type_code)
        if option_type is None:
            return None
        try:
            expiry_date = datetime.strptime(expiry_raw, _DERIBIT_INSTRUMENT_DATE_FORMAT)
            strike = float(strike_raw)
        except ValueError:
            return None

        expiry = expiry_date.replace(hour=_DERIBIT_EXPIRY_HOUR_UTC, tzinfo=timezone.utc)

        mark_iv_raw = info.get("mark_iv")
        mark_iv = float(mark_iv_raw) / 100.0 if mark_iv_raw is not None else None

        contract = OptionContract(
            underlying=f"{currency}/USDT",
            strike=strike,
            expiry=expiry,
            option_type=option_type,
        )
        open_interest = option.get("openInterest")
        if open_interest is None:
            open_interest = info.get("open_interest") or 0.0

        return OptionQuote(
            contract=contract,
            underlying_price=float(underlying_price),
            mark_price_quote_ccy=float(mark_price_native) * float(underlying_price),
            mark_implied_volatility=mark_iv,
            open_interest=float(open_interest),
        )

    async def close(self) -> None:
        """Close the underlying ``ccxt`` HTTP session. Safe to call once fetching is done."""
        await self._exchange.close()

    async def __aenter__(self) -> DeribitOptionsChainClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()


def select_nearest_contract(
    quotes: Sequence[OptionQuote],
    *,
    target_strike: float,
    option_type: Literal["call", "put"],
    min_time_to_expiry_days: float = 1.0,
) -> OptionQuote | None:
    """Pick the real quote closest to a target strike, among real contracts of the requested type.

    A small, genuinely useful filter over :meth:`DeribitOptionsChainClient.
    fetch_chain`'s real output -- e.g. finding a real near-the-money
    contract to hedge a given delta exposure with, without a caller
    having to hand-scan hundreds of real listed contracts.

    Args:
        quotes: Real quotes, e.g. from :meth:`DeribitOptionsChainClient.fetch_chain`.
        target_strike: Real strike to get as close to as possible.
        option_type: Restrict the search to this real option type.
        min_time_to_expiry_days: Exclude any real contract expiring
            sooner than this (avoids selecting a contract already too
            close to expiry to be practically tradable/hedgeable).

    Returns:
        The real matching :class:`OptionQuote` closest to
        ``target_strike``, or ``None`` if no real contract satisfies
        ``option_type``/``min_time_to_expiry_days``.
    """
    now = datetime.now(timezone.utc)
    min_seconds = min_time_to_expiry_days * 24.0 * 3600.0
    candidates = [
        quote
        for quote in quotes
        if quote.contract.option_type == option_type
        and (quote.contract.expiry - now).total_seconds() >= min_seconds
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda quote: abs(quote.contract.strike - target_strike))

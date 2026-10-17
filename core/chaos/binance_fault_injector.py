"""Fault-injecting wrapper around :class:`~src.agents.market_data.exchange_client.AsyncExchangeClient` (Directive 2).

**Why a wrapper, not an edit to ``exchange_client.py`` itself**: that
module is shared, unconditionally, by every real caller in this project
(``MarketDataAgent``, ``AlphaMarketMakerAgent``'s emergency-liquidate
path, ``CcxtBrokerAdapter``) -- baking corruption into it, even behind a
flag, means one forgotten env var stands between "Shadow Run chaos test"
and "real order pipeline fed garbage". :class:`ChaosExchangeClient`
instead composes over a real, already-connected ``AsyncExchangeClient``
and is only ever constructed by a ``scripts/run_*_chaos.py`` launcher
that exists purely for this test -- the production launcher
(``scripts/run_market_data_agent.py``) and every other real caller never
import this module at all.

**Honest scope note on "malformed JSON" (mission Directive 2)**: by the
time any code in this project sees a WebSocket message, ``ccxt.pro`` has
already parsed it into a Python ``dict`` -- genuinely malformed JSON
bytes on the wire fail inside ``ccxt.pro``'s own transport, never reach
this codebase, and would just look like the ordinary
``NetworkError``/``OSError`` reconnect path ``AsyncExchangeClient``
already handles (see that module's own docstring). The realistic
equivalent of "the exchange sent us garbage" from *this* codebase's
vantage point is a structurally-broken dict downstream of ccxt's own
parser (missing/wrong-typed keys) -- :meth:`ChaosExchangeClient.
watch_order_book` injects exactly that, alongside real ``NaN`` prices
and simulated flash-crash jumps, which are the two corruption modes that
actually need to reach ``MarketMetrics`` computation to exercise it.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from typing import Any, Final

from core.logger import get_logger
from src.agents.market_data.exchange_client import AsyncExchangeClient

_log = get_logger(__name__)

_DEFAULT_NAN_PROBABILITY: Final[float] = 0.10
_DEFAULT_FLASH_CRASH_PROBABILITY: Final[float] = 0.05
_DEFAULT_MALFORMED_PROBABILITY: Final[float] = 0.10
_DEFAULT_FLASH_CRASH_FRACTION: Final[float] = 0.30


class ChaosExchangeClient:
    """Wraps a real, connected :class:`AsyncExchangeClient`; corrupts a configurable fraction of its order-book stream.

    Every method other than :meth:`watch_order_book` delegates straight
    through to the wrapped client unchanged -- this only touches the one
    stream :class:`~src.agents.market_data.agent.MarketDataAgent` derives
    :class:`~core.schemas.MarketMetrics` from, which is the real surface
    the mission's AEGIS/circuit-breaker chaos targets.
    """

    def __init__(
        self,
        real_client: AsyncExchangeClient,
        *,
        nan_probability: float = _DEFAULT_NAN_PROBABILITY,
        flash_crash_probability: float = _DEFAULT_FLASH_CRASH_PROBABILITY,
        malformed_probability: float = _DEFAULT_MALFORMED_PROBABILITY,
        flash_crash_fraction: float = _DEFAULT_FLASH_CRASH_FRACTION,
        rng: random.Random | None = None,
    ) -> None:
        """
        Args:
            real_client: An already-constructed ``AsyncExchangeClient``
                (this wrapper never calls ``connect()``/``close()``
                itself -- the caller owns that lifecycle, unchanged).
            nan_probability: Fraction of yielded order-book updates whose
                best bid/ask is replaced with ``float("nan")``.
            flash_crash_probability: Fraction of yielded updates whose
                entire book is shifted by
                ``+/- flash_crash_fraction`` relative to the last real
                price seen, simulating a one-tick flash crash/spike.
            malformed_probability: Fraction of yielded updates
                structurally broken downstream of ccxt's own parser
                (missing ``bids``/``asks`` keys) -- see module docstring
                for why this is the realistic equivalent of "malformed
                JSON" at this layer.
            flash_crash_fraction: Magnitude of the simulated flash-crash
                move, as a fraction of the last real price (mission's
                own 30% default).
            rng: Injectable for deterministic tests; defaults to the
                module-global ``random`` behavior via a fresh
                ``random.Random()``.
        """
        if not 0.0 <= nan_probability <= 1.0:
            raise ValueError("nan_probability must be in [0, 1].")
        if not 0.0 <= flash_crash_probability <= 1.0:
            raise ValueError("flash_crash_probability must be in [0, 1].")
        if not 0.0 <= malformed_probability <= 1.0:
            raise ValueError("malformed_probability must be in [0, 1].")
        self._real_client = real_client
        self._nan_probability = nan_probability
        self._flash_crash_probability = flash_crash_probability
        self._malformed_probability = malformed_probability
        self._flash_crash_fraction = flash_crash_fraction
        self._rng = rng or random.Random()
        self._last_real_mid: float | None = None
        self._log = get_logger(f"{__name__}.ChaosExchangeClient")
        self.injected_count = 0
        self.total_count = 0

    async def watch_order_book(self, symbol: str, limit: int = 5) -> AsyncIterator[dict[str, Any]]:
        """Same contract as :meth:`AsyncExchangeClient.watch_order_book`, with a chance of corruption per update."""
        async for book in self._real_client.watch_order_book(symbol, limit):
            self.total_count += 1
            bids, asks = book.get("bids"), book.get("asks")
            if isinstance(bids, list) and bids and isinstance(asks, list) and asks:
                self._last_real_mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0

            roll = self._rng.random()
            if roll < self._malformed_probability:
                yield self._corrupt_malformed(book, symbol)
            elif roll < self._malformed_probability + self._nan_probability:
                yield self._corrupt_nan(book, symbol)
            elif roll < self._malformed_probability + self._nan_probability + self._flash_crash_probability:
                yield self._corrupt_flash_crash(book, symbol)
            else:
                yield book

    def _corrupt_nan(self, book: dict[str, Any], symbol: str) -> dict[str, Any]:
        self.injected_count += 1
        corrupted = dict(book)
        nan = float("nan")
        corrupted["bids"] = [[nan, nan]] + list(book.get("bids") or [])[1:]
        corrupted["asks"] = [[nan, nan]] + list(book.get("asks") or [])[1:]
        self._log.warning("chaos_injected_nan_price", symbol=symbol)
        return corrupted

    def _corrupt_flash_crash(self, book: dict[str, Any], symbol: str) -> dict[str, Any]:
        self.injected_count += 1
        reference = self._last_real_mid or 1.0
        direction = self._rng.choice((-1.0, 1.0))
        shocked_mid = reference * (1.0 + direction * self._flash_crash_fraction)
        corrupted = dict(book)
        corrupted["bids"] = [[shocked_mid * 0.999, level[1]] for level in (book.get("bids") or [[0, 0]])]
        corrupted["asks"] = [[shocked_mid * 1.001, level[1]] for level in (book.get("asks") or [[0, 0]])]
        self._log.warning(
            "chaos_injected_flash_crash", symbol=symbol,
            reference_mid=reference, shocked_mid=shocked_mid, fraction=self._flash_crash_fraction,
        )
        return corrupted

    def _corrupt_malformed(self, book: dict[str, Any], symbol: str) -> dict[str, Any]:
        """Structurally broken downstream of ccxt's own parser -- see module docstring's honest scope note."""
        self.injected_count += 1
        self._log.warning("chaos_injected_malformed_payload", symbol=symbol)
        return {"timestamp": book.get("timestamp"), "garbage": True}

    def __getattr__(self, name: str) -> Any:
        """Delegate everything else (``watch_trades``, ``connect``, ``close``, order methods, ...) unchanged."""
        return getattr(self._real_client, name)

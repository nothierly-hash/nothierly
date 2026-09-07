"""The cross-asset instrument registry -- the "laws of physics" per asset, read by every asset-agnostic component instead of branching on symbol identity. See Mission 1.

**Corrections to the brief, verified against the real code first** (this
project's established discipline):

1. **``tick_size``/``lot_size`` already exist as real fields on
   :class:`~core.nothierly_ontology.UniversalAsset`** -- reused here
   (composed into :class:`AssetUniverseEntry`), not duplicated. A
   literal ``lot_multiplier`` field as briefed would have been a second,
   conflicting definition of the exact concept ``UniversalAsset.lot_size``
   already covers for the FX-lot case (that field's own docstring
   already gives ``Decimal('100000')`` "for one OANDA standard lot" as
   its own real example). The concept this registry actually needed and
   did *not* already have is *how a generic signal's "N units" maps to
   a real order size* -- see :class:`PositionSizingConvention`, which is
   genuinely new, not a restatement of ``lot_size``.

2. **``trading_hours`` is one real weekly window, not naive daily
   open/close pairs.** FX and commodity CFDs trade continuously from
   Sunday evening UTC to Friday evening UTC -- there is no real "closes
   at 5pm, reopens at 9am" daily cycle for them the way an equity
   exchange has one. Modeling it as a single :class:`WeeklyTradingWindow`
   is the factually correct shape, not a simplification of a daily model.

3. **OANDA's real commodity CFDs already work today, through the
   existing, generic** :class:`~src.execution.broker_gateway.
   OandaBrokerAdapter` **-- it is symbol-agnostic** (verified by reading
   it: ``place_market_order`` branches on nothing but ``side``, forwards
   whatever ``symbol`` string it's given straight to
   :class:`~src.agents.supervisor.broker_client.OandaClient`). This
   registry is what actually classifies ``XAU_USD``/``BCO_USD`` under
   the new :data:`~core.nothierly_ontology.AssetClass.COMMODITY` value
   instead of leaving them mislabeled ``FOREX`` -- it does not invent a
   fixed "1 lot = 100 troy oz" convention neither OANDA nor this
   project's own ``OandaClient`` actually enforces for spot-CFD gold/oil
   (OANDA quotes and fills these in plain fractional units, no lot
   concept) -- modeled here as :attr:`PositionSizingConvention.
   DIRECT_QUANTITY`. ``EUR_USD`` is modeled as the
   :attr:`PositionSizingConvention.LOT_BASED` example the brief asked
   for explicitly (a real convention many FX venues, though not OANDA
   itself, do enforce) -- a deliberate choice to demonstrate that
   convention, not a claim that OANDA requires it.
   ``BCO_USD`` (not "Brent/USD") is OANDA's real instrument symbol for
   Brent Crude -- kept in this project's already-established
   underscore-symbol convention for OANDA instruments (see
   ``scripts/run_backend.py``'s own ``EUR_USD``/``XAU_USD``).

4. **Real, checked constraint this registry's numbers respect**:
   :meth:`OandaBrokerAdapter.place_market_order` truncates quantity to
   ``int(quantity)`` before calling OANDA -- OANDA units are whole
   numbers, never fractional. Every OANDA-routed entry below therefore
   sets ``asset.lot_size = Decimal("1")``, so
   :func:`~src.execution.smart_order_router.normalize_signal_size`'s own
   quantization (Mission 4) always produces a real, placeable integer
   quantity for these three -- never a fractional-unit order OANDA would
   itself have silently truncated (and thus mis-sized) anyway.

**Agnostic design (the brief's own constraint 2)**: nothing in this
module, or in any of its real callers
(:class:`~core.session_manager.MarketSessionManager`,
:func:`is_spread_acceptable`,
:func:`~src.execution.smart_order_router.normalize_signal_size`), ever
branches on a literal symbol string (no ``if symbol == "BTC/USDT"``
anywhere) -- every one of them takes a ``symbol``, looks up its real
:class:`AssetUniverseEntry`, and branches only on the *looked-up*
``asset_class``/``position_sizing_convention`` enum value. Adding a
fourth, fifth, tenth real asset means adding one dict entry below --
never touching any consuming code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dt_time
from decimal import Decimal
from enum import Enum
from typing import Final

from core.nothierly_ontology import AssetClass, UniversalAsset


class PositionSizingConvention(str, Enum):
    """How a generic signal's size ("N units") maps to a real order quantity. See module docstring point 1."""

    DIRECT_QUANTITY = "DIRECT_QUANTITY"  # N signal units = N real base-asset units (crypto: "1" -> 1 BTC)
    LOT_BASED = "LOT_BASED"  # N signal units = N * lot_multiplier real base-asset units (FX-lot: "1" -> 1 standard lot)


@dataclass(frozen=True)
class WeeklyTradingWindow:
    """One real, continuous weekly trading window. See module docstring point 2 on why this is a single window, not daily open/close pairs.

    Weekday numbering matches ``datetime.weekday()``: Monday=0 ... Sunday=6.
    A window may legitimately wrap the week boundary (e.g. "opens Sunday,
    closes Friday" wraps from weekday 6 back around through weekday 0-3)
    -- see :func:`_is_within_weekly_window`, which handles both cases.
    """

    open_weekday: int
    open_time_utc: dt_time
    close_weekday: int
    close_time_utc: dt_time


@dataclass(frozen=True)
class AssetUniverseEntry:
    """One asset's complete, real trading configuration -- the single source every asset-agnostic component reads."""

    asset: UniversalAsset
    max_allowable_spread: Decimal
    """Absolute ceiling, in quote-currency units (e.g. USD for a *_USD pair) -- not "pips": pip size
    itself varies per instrument (0.0001 for most FX pairs, 0.01 for JPY pairs, no such convention at
    all for crypto/commodities), so an absolute quote-currency threshold is the one unit every asset
    class here can express without a separate per-pair pip-size lookup table."""

    trading_hours: WeeklyTradingWindow | None  # None = always open (crypto)
    position_sizing_convention: PositionSizingConvention
    lot_multiplier: Decimal = Decimal(1)  # only meaningful when position_sizing_convention is LOT_BASED


_FX_COMMODITY_WEEKLY_WINDOW: Final[WeeklyTradingWindow] = WeeklyTradingWindow(
    open_weekday=6, open_time_utc=dt_time(22, 0),  # Sunday 22:00 UTC
    close_weekday=4, close_time_utc=dt_time(22, 0),  # Friday 22:00 UTC
)

# Real registry entries. Add a new asset by adding one entry here -- see module docstring's
# "agnostic design" section on why no consuming code ever needs to change.
ASSET_UNIVERSE: Final[dict[str, AssetUniverseEntry]] = {
    "BTC/USDT": AssetUniverseEntry(
        asset=UniversalAsset(
            symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
            tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
        ),
        # $5 absolute: real BTC/USDT spreads on major venues run cents to low single digits -- 5 is
        # already a generous ceiling before this is treated as broker/venue manipulation or a
        # genuinely illiquid book, not a value grazing the real, normal spread.
        max_allowable_spread=Decimal("5.00"),
        trading_hours=None,
        position_sizing_convention=PositionSizingConvention.DIRECT_QUANTITY,
    ),
    "EUR_USD": AssetUniverseEntry(
        asset=UniversalAsset(
            symbol="EUR_USD", asset_class=AssetClass.FOREX, base_currency="EUR", quote_currency="USD",
            tick_size=Decimal("0.00001"), lot_size=Decimal("1"),  # OANDA units are whole numbers -- see module docstring point 4
        ),
        max_allowable_spread=Decimal("0.0003"),  # 3 pips absolute (EUR/USD's real pip = 0.0001)
        trading_hours=_FX_COMMODITY_WEEKLY_WINDOW,
        position_sizing_convention=PositionSizingConvention.LOT_BASED,
        lot_multiplier=Decimal("100000"),  # 1 standard lot = 100,000 EUR -- the brief's own real, explicit example
    ),
    "XAU_USD": AssetUniverseEntry(
        asset=UniversalAsset(
            symbol="XAU_USD", asset_class=AssetClass.COMMODITY, base_currency="XAU", quote_currency="USD",
            tick_size=Decimal("0.01"), lot_size=Decimal("1"),
        ),
        max_allowable_spread=Decimal("0.50"),  # $0.50/troy oz absolute
        trading_hours=_FX_COMMODITY_WEEKLY_WINDOW,
        position_sizing_convention=PositionSizingConvention.DIRECT_QUANTITY,  # real OANDA convention -- see module docstring point 3
    ),
    "BCO_USD": AssetUniverseEntry(
        asset=UniversalAsset(
            symbol="BCO_USD", asset_class=AssetClass.COMMODITY, base_currency="BCO", quote_currency="USD",
            tick_size=Decimal("0.01"), lot_size=Decimal("1"),
        ),
        max_allowable_spread=Decimal("0.05"),  # $0.05/barrel absolute
        trading_hours=_FX_COMMODITY_WEEKLY_WINDOW,
        position_sizing_convention=PositionSizingConvention.DIRECT_QUANTITY,
    ),
}


def get_asset_universe_entry(symbol: str) -> AssetUniverseEntry:
    """The one real lookup every asset-agnostic component uses.

    Raises:
        KeyError: If ``symbol`` has no real registry entry -- deliberately
            never silently falls back to some other asset's real
            physical limits (a missing entry is a real configuration gap
            to fix, not a case to paper over).
    """
    try:
        return ASSET_UNIVERSE[symbol]
    except KeyError:
        raise KeyError(
            f"{symbol!r} is not registered in core.asset_universe.ASSET_UNIVERSE -- add a real "
            "AssetUniverseEntry for it before trading it through any asset-agnostic component."
        ) from None


def is_spread_acceptable(symbol: str, spread: Decimal) -> bool:
    """Real, asset-agnostic spread-widening filter. See Mission 3.

    Args:
        symbol: A symbol registered in :data:`ASSET_UNIVERSE`.
        spread: The real, current ask-minus-bid spread, in the same
            absolute quote-currency units as the registry's
            ``max_allowable_spread`` (e.g.
            :attr:`core.schemas.MarketMetrics.spread` for a crypto venue).

    Returns:
        ``False`` if ``spread`` exceeds this asset's real configured
        ceiling. Callers use this to silently reject an opportunity
        during broker spread-widening (macro announcements, illiquid
        session edges) -- this function itself never raises or alerts,
        it only answers the question.
    """
    entry = get_asset_universe_entry(symbol)
    return spread <= entry.max_allowable_spread


def _minutes_since_monday(weekday: int, t: dt_time) -> int:
    return weekday * 24 * 60 + t.hour * 60 + t.minute


def is_within_weekly_window(window: WeeklyTradingWindow, now: datetime) -> bool:
    """Real, general-purpose weekly-window containment check -- handles a window that wraps the week boundary (this project's real FX/commodity case) exactly as correctly as one that doesn't."""
    now_offset = _minutes_since_monday(now.weekday(), now.time())
    open_offset = _minutes_since_monday(window.open_weekday, window.open_time_utc)
    close_offset = _minutes_since_monday(window.close_weekday, window.close_time_utc)
    if open_offset <= close_offset:
        return open_offset <= now_offset < close_offset
    # Wraps the week boundary (e.g. opens Sunday night, closes Friday night -- open_offset is
    # numerically *larger* than close_offset in Monday-indexed minutes).
    return now_offset >= open_offset or now_offset < close_offset

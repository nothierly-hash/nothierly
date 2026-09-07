"""Real, executable tests for core/asset_universe.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from core.asset_universe import (
    ASSET_UNIVERSE,
    PositionSizingConvention,
    get_asset_universe_entry,
    is_spread_acceptable,
    is_within_weekly_window,
)
from core.nothierly_ontology import AssetClass


def test_get_asset_universe_entry_returns_a_real_registered_asset() -> None:
    entry = get_asset_universe_entry("BTC/USDT")
    assert entry.asset.asset_class == AssetClass.CRYPTO_SPOT
    assert entry.trading_hours is None


def test_get_asset_universe_entry_raises_for_an_unregistered_symbol() -> None:
    with pytest.raises(KeyError, match="not registered"):
        get_asset_universe_entry("DOGE/USDT")


def test_eur_usd_is_registered_as_forex_lot_based() -> None:
    entry = get_asset_universe_entry("EUR_USD")
    assert entry.asset.asset_class == AssetClass.FOREX
    assert entry.position_sizing_convention == PositionSizingConvention.LOT_BASED
    assert entry.lot_multiplier == Decimal("100000")


def test_xau_and_bco_are_registered_as_commodity_not_forex() -> None:
    """The real correction this registry makes -- see module docstring point 3."""
    for symbol in ("XAU_USD", "BCO_USD"):
        entry = get_asset_universe_entry(symbol)
        assert entry.asset.asset_class == AssetClass.COMMODITY
        assert entry.position_sizing_convention == PositionSizingConvention.DIRECT_QUANTITY


def test_every_oanda_routed_entry_has_integer_lot_size() -> None:
    """Real, checked OANDA constraint -- see module docstring point 4 (place_market_order truncates to int(quantity))."""
    for symbol, entry in ASSET_UNIVERSE.items():
        if entry.asset.asset_class in (AssetClass.FOREX, AssetClass.COMMODITY):
            assert entry.asset.lot_size == Decimal("1"), symbol


# =============================================================================
# is_spread_acceptable
# =============================================================================


def test_spread_within_ceiling_is_acceptable() -> None:
    assert is_spread_acceptable("EUR_USD", Decimal("0.0002")) is True


def test_spread_widened_past_ceiling_is_rejected() -> None:
    """The real Mission 3 scenario: a broker widening the spread around an FOMC/NFP announcement."""
    assert is_spread_acceptable("EUR_USD", Decimal("0.0010")) is False


def test_spread_exactly_at_ceiling_is_acceptable() -> None:
    entry = get_asset_universe_entry("BTC/USDT")
    assert is_spread_acceptable("BTC/USDT", entry.max_allowable_spread) is True


# =============================================================================
# is_within_weekly_window -- the wrapping Sunday-22:00 -> Friday-22:00 UTC window
# =============================================================================


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc), True),  # Monday noon
        (datetime(2026, 9, 10, 3, 0, tzinfo=timezone.utc), True),  # Thursday 3am
        (datetime(2026, 9, 11, 21, 59, 59, tzinfo=timezone.utc), True),  # Friday, just before close
        (datetime(2026, 9, 11, 22, 0, 0, tzinfo=timezone.utc), False),  # Friday, exactly at close
        (datetime(2026, 9, 11, 23, 0, tzinfo=timezone.utc), False),  # Friday night
        (datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc), False),  # Saturday
        (datetime(2026, 9, 13, 21, 59, tzinfo=timezone.utc), False),  # Sunday, just before open
        (datetime(2026, 9, 13, 22, 0, 0, tzinfo=timezone.utc), True),  # Sunday, exactly at open
        (datetime(2026, 9, 13, 23, 0, tzinfo=timezone.utc), True),  # Sunday night
    ],
)
def test_fx_commodity_window_real_boundaries(now: datetime, expected: bool) -> None:
    entry = get_asset_universe_entry("EUR_USD")
    assert entry.trading_hours is not None
    assert is_within_weekly_window(entry.trading_hours, now) is expected


def test_crypto_has_no_weekly_window_at_all() -> None:
    assert get_asset_universe_entry("BTC/USDT").trading_hours is None

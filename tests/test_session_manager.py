"""Real, executable tests for core/session_manager.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.session_manager import MarketSessionManager

# See tests/test_asset_universe.py for the verified real weekdays these dates land on:
# 2026-09-07=Monday, -10=Thursday, -11=Friday, -12=Saturday, -13=Sunday.


@pytest.mark.parametrize(
    ("now", "expected_open"),
    [
        (datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc), True),  # Monday
        (datetime(2026, 9, 11, 20, 59, tzinfo=timezone.utc), True),  # Friday, before the unwind trigger
        (datetime(2026, 9, 11, 22, 0, tzinfo=timezone.utc), False),  # Friday, at the real close
        (datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc), False),  # Saturday
        (datetime(2026, 9, 13, 22, 0, tzinfo=timezone.utc), True),  # Sunday, real reopen
    ],
)
def test_is_market_open_for_eur_usd(now: datetime, expected_open: bool) -> None:
    assert MarketSessionManager.is_market_open("EUR_USD", now=now) is expected_open


def test_is_market_open_for_crypto_is_always_true() -> None:
    for now in [
        datetime(2026, 9, 12, 3, 0, tzinfo=timezone.utc),  # Saturday 3am -- would be closed for FX
        datetime(2026, 9, 11, 23, 59, tzinfo=timezone.utc),  # Friday night
    ]:
        assert MarketSessionManager.is_market_open("BTC/USDT", now=now) is True


# =============================================================================
# is_unwind_period -- the Friday Unwind, Mission 2's own explicit scenario
# =============================================================================


@pytest.mark.parametrize(
    ("now", "expected_unwind"),
    [
        (datetime(2026, 9, 11, 20, 59, 59, tzinfo=timezone.utc), False),  # Friday, just before the trigger
        (datetime(2026, 9, 11, 21, 0, 0, tzinfo=timezone.utc), True),  # Friday 21:00 UTC -- the exact real trigger
        (datetime(2026, 9, 11, 21, 30, tzinfo=timezone.utc), True),  # Friday, mid-unwind window, market still open
        (datetime(2026, 9, 11, 22, 30, tzinfo=timezone.utc), True),  # Friday, after the real close too
        (datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc), True),  # Saturday -- still flat, market closed
        (datetime(2026, 9, 13, 21, 0, tzinfo=timezone.utc), True),  # Sunday, before real reopen
        (datetime(2026, 9, 13, 22, 0, tzinfo=timezone.utc), False),  # Sunday, real reopen -- unwind over
        (datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc), False),  # Tuesday -- nowhere near the weekend
    ],
)
def test_is_unwind_period_for_eur_usd(now: datetime, expected_unwind: bool) -> None:
    assert MarketSessionManager.is_unwind_period("EUR_USD", now=now) is expected_unwind


def test_is_unwind_period_for_xau_usd_matches_eur_usd_same_calendar() -> None:
    """Both real FX and commodity CFDs share the exact same weekly window -- see core/asset_universe.py."""
    friday_trigger = datetime(2026, 9, 11, 21, 0, tzinfo=timezone.utc)
    assert MarketSessionManager.is_unwind_period("XAU_USD", now=friday_trigger) is True


def test_is_unwind_period_for_crypto_is_always_false() -> None:
    """Crypto never has a weekend to unwind before -- always False, any real Friday/Saturday/Sunday timestamp."""
    for now in [
        datetime(2026, 9, 11, 21, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
    ]:
        assert MarketSessionManager.is_unwind_period("BTC/USDT", now=now) is False

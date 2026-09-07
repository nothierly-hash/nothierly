"""MarketSessionManager -- real UTC session-awareness for TradFi assets, and the Friday Unwind. See Mission 2.

Crypto trades 24/7 by construction (every crypto entry in
:mod:`core.asset_universe` has ``trading_hours=None``) -- this module's
real job is only for the assets that genuinely close: FX and commodity
CFDs, both real today via :class:`~src.execution.broker_gateway.
OandaBrokerAdapter` (see that registry module's own docstring).

**A real, deliberate choice, stated upfront**: :meth:`MarketSessionManager.
is_unwind_period` returns ``True`` from the brief's own Friday-21:00-UTC
trigger all the way through the real weekend close (not just the single
instant ``21:00:00.000000``) -- "the desk should be flat" is a state to
stay in until the market genuinely reopens, not a one-shot flag a poll
cycle could step over and miss.
"""

from __future__ import annotations

from datetime import datetime, time as dt_time, timezone
from typing import Final

from core.asset_universe import get_asset_universe_entry, is_within_weekly_window

# The brief's own explicit trigger: Friday 21:00 UTC -- one real hour before this project's own
# weekly close (core.asset_universe._FX_COMMODITY_WEEKLY_WINDOW closes Friday 22:00 UTC), so a real
# desk starts flattening while the book is still liquid, not exactly as it closes.
_FRIDAY_UNWIND_HOUR_UTC: Final[int] = 21
_FRIDAY_WEEKDAY: Final[int] = 4  # datetime.weekday(): Monday=0 ... Friday=4


class MarketSessionManager:
    """Real, stateless UTC session gate.

    Every method takes ``now`` explicitly (defaults to the real current
    UTC time) -- exactly as testable as any other pure function in this
    codebase, never a hidden dependency on the wall clock a test can't
    control.
    """

    @staticmethod
    def is_market_open(symbol: str, *, now: datetime | None = None) -> bool:
        """``True`` if ``symbol``'s real market is open at ``now``.

        Args:
            symbol: A symbol registered in :data:`core.asset_universe.ASSET_UNIVERSE`.
            now: Defaults to ``datetime.now(timezone.utc)``.
        """
        now = now or datetime.now(timezone.utc)
        entry = get_asset_universe_entry(symbol)
        if entry.trading_hours is None:
            return True  # crypto -- always open
        return is_within_weekly_window(entry.trading_hours, now)

    @staticmethod
    def is_unwind_period(symbol: str, *, now: datetime | None = None) -> bool:
        """``True`` from the real Friday 21:00 UTC trigger through the actual weekend close. See Mission 2 / module docstring.

        Always ``False`` for an asset with no ``trading_hours`` (crypto
        never has a weekend to unwind before). Forces the caller to
        both liquidate existing exposure and refuse new entries for
        ``symbol`` while this is ``True`` -- the brief's own explicit
        requirement.
        """
        now = now or datetime.now(timezone.utc)
        entry = get_asset_universe_entry(symbol)
        if entry.trading_hours is None:
            return False
        if now.weekday() == _FRIDAY_WEEKDAY and now.time() >= dt_time(_FRIDAY_UNWIND_HOUR_UTC, 0):
            return True
        # Already past the real weekly close (Friday 22:00 UTC onward, through the actual weekend,
        # until the real Sunday reopen) -- the desk should already be flat throughout, not just at
        # the single Friday-21:00 instant.
        return not is_within_weekly_window(entry.trading_hours, now)

"""Nothierly Money Hunter: ranks real cross-country P2P spreads net of friction. Read-only, never trades.

**Scope**: reuses :func:`p2p_api._run_full_scan` (real Binance P2P ads, real forex rates, real USDT
spot) and keeps only the cross-country spreads that survive three filters:

1. both legs backed by at least :data:`MIN_LEG_LIQUIDITY_USDT` of real ad liquidity,
2. both legs' merchants above :data:`MIN_MERCHANT_COMPLETION_RATE`,
3. spread still positive after :data:`ROUND_TRIP_FRICTION_PCT` of assumed friction.

**Disclosed limits**: the friction figure is an assumption, not a measured cost -- P2P fees, payment
rail fees and FX slippage vary per merchant and rail. A positive ``net_spread_pct`` is a price
signal, not a guaranteed profit: moving fiat between two countries needs payment access in both,
which this module cannot verify. It places no orders.

Run: ``python nothierly_money_hunter.py``
"""

from __future__ import annotations

import asyncio
from typing import Final

from pydantic import BaseModel, ConfigDict

from core.logger import get_logger
from p2p_api import P2PCandidate, _run_full_scan

_log = get_logger(__name__)

MIN_LEG_LIQUIDITY_USDT: Final[float] = 500.0
MIN_MERCHANT_COMPLETION_RATE: Final[float] = 0.90
ROUND_TRIP_FRICTION_PCT: Final[float] = 1.0  # assumption, see module docstring


class HuntedOpportunity(BaseModel):
    """One cross-country spread that survived the liquidity/reliability/friction filters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    buy_fiat: str
    sell_fiat: str
    gross_spread_pct: float
    net_spread_pct: float
    limiting_liquidity_usdt: float


def hunt(
    candidates: list[P2PCandidate],
    *,
    min_liquidity_usdt: float = MIN_LEG_LIQUIDITY_USDT,
    min_completion_rate: float = MIN_MERCHANT_COMPLETION_RATE,
    friction_pct: float = ROUND_TRIP_FRICTION_PCT,
) -> list[HuntedOpportunity]:
    """Pure ranking step (no I/O): every ordered pair of eligible candidates, best net spread first."""
    eligible = [
        c for c in candidates
        if c.avg_liquidity_usdt >= min_liquidity_usdt and c.avg_merchant_completion_rate >= min_completion_rate
    ]
    found: list[HuntedOpportunity] = []
    for buy in eligible:
        for sell in eligible:
            if buy.country_fiat == sell.country_fiat or sell.usd_equivalent_price <= buy.usd_equivalent_price:
                continue
            gross = (sell.usd_equivalent_price / buy.usd_equivalent_price - 1.0) * 100.0
            net = gross - friction_pct
            if net <= 0.0:
                continue
            found.append(
                HuntedOpportunity(
                    buy_fiat=buy.country_fiat,
                    sell_fiat=sell.country_fiat,
                    gross_spread_pct=round(gross, 3),
                    net_spread_pct=round(net, 3),
                    limiting_liquidity_usdt=round(min(buy.avg_liquidity_usdt, sell.avg_liquidity_usdt), 2),
                )
            )
    found.sort(key=lambda o: o.net_spread_pct, reverse=True)
    return found


async def run_hunt() -> list[HuntedOpportunity]:
    """Live scan then :func:`hunt`. Needs network access to Binance P2P and the forex provider."""
    scan = await _run_full_scan()
    opportunities = hunt(scan.top_candidates)
    _log.info("money_hunter_completed", candidates=len(scan.top_candidates), opportunities=len(opportunities))
    return opportunities


if __name__ == "__main__":
    for opp in asyncio.run(run_hunt()):
        print(
            f"{opp.buy_fiat} -> {opp.sell_fiat}  gross {opp.gross_spread_pct:+.2f}%  "
            f"net {opp.net_spread_pct:+.2f}%  liquidity ${opp.limiting_liquidity_usdt:,.0f}"
        )

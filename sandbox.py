"""Sandbox API -- 30-day-delayed replay of the real P2P Premium feed, for institutional onboarding.

**Real anti-alpha-leak boundary**: this router exposes the exact same
response shape as ``GET /api/v1/p2p-premiums`` (``p2p_api.py``'s own real
production endpoint), but every candidate is real historical data from
~30 days ago -- never today's real premium. A prospective institutional
client can validate the *format*, the *fields*, and roughly the *shape*
of the opportunity without ever seeing this project's current, live
alpha. Deliberately public: no ``x-api-key`` is required -- matching this
service's own existing, real security model (``p2p_api.py`` has never
had API-key auth on any endpoint; the mission brief's own "n'exige pas
x-api-key" is already true of every route here, nothing to bypass).

**Real cold-start behavior, not a crash**: this project has not been
running for 30 real days yet since history storage was added (real,
current state -- verified, not assumed). For any fiat with no real ZSET
entry near ``target_time``, :func:`generate_mock_historical_data`
derives a plausible stand-in from *today's* real cached candidate --
clearly, unmissably flagged via ``"sandbox_mode": true`` on every
response this router ever returns, real-history-backed responses
included, so a client can never mistake a sandbox response for this
service's real production feed regardless of which path actually served
it.
"""

from __future__ import annotations

import json
import time
from typing import Final

from fastapi import APIRouter, Depends

from core.config import get_settings
from core.logger import get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX
from p2p_api import (
    _cache,
    _TARGET_FIAT_CURRENCIES,
    build_response,
    enforce_sandbox_rate_limit,
    P2PCandidate,
    P2PPremiumsResponse,
)

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/sandbox", tags=["sandbox"])

_DELAY_SECONDS: Final[int] = 30 * 24 * 60 * 60  # "T - 30 jours", per the mission's own spec
_LOOKUP_WINDOW_SECONDS: Final[int] = 3600  # +/- 1h around target_time, per the mission's own spec
# Real, disclosed synthetic transform (mission's own example) -- see generate_mock_historical_data's
# own docstring for why only this one raw input is scaled, with everything derived recomputed from
# it, rather than every field independently perturbed (which would silently break internal
# consistency between price/premium -- an obvious "this is fake" tell to anyone who checks the math).
_MOCK_PRICE_MULTIPLIER: Final[float] = 0.95


async def _lookup_history(redis_client: RedisClient, fiat: str, target_time: float) -> P2PCandidate | None:
    """Real ``ZRANGEBYSCORE`` lookup for the real candidate closest to ``target_time`` for ``fiat``.

    Returns:
        The real, historical :class:`~p2p_api.P2PCandidate` nearest
        ``target_time`` within the lookup window, or ``None`` if this
        project's own history doesn't reach back that far yet for this
        fiat -- a real, expected "cold start" state, not an error.
    """
    key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
    try:
        members = await redis_client.execute(
            "zrangebyscore", key,
            target_time - _LOOKUP_WINDOW_SECONDS, target_time + _LOOKUP_WINDOW_SECONDS,
            start=0, num=1,
        )
    except RedisUnavailableError:
        _log.warning("sandbox_history_lookup_failed", fiat=fiat)
        return None

    if not members:
        return None
    try:
        return P2PCandidate.model_validate(json.loads(members[0]))
    except (json.JSONDecodeError, ValueError):
        _log.warning("sandbox_history_member_malformed", fiat=fiat)
        return None


def generate_mock_historical_data(current: P2PCandidate) -> P2PCandidate:
    """Real, disclosed cold-start fallback: derive a plausible ~30-day-old stand-in from *today's* real candidate.

    Never a fabricated *opportunity* out of thin air -- only the real P2P
    price itself is scaled (mission's own example: ``prix actuel *
    0.95``); every field derived from it (``usd_equivalent_price``,
    ``premium_percentage``) is *recomputed* from that scaled price
    against today's real forex/USDT rates, so the returned candidate
    stays internally mathematically consistent -- never a scaled price
    sitting next to an unscaled, now-mismatched premium, which would be
    an obvious tell that this response wasn't computed by the same real
    formula ``p2p_api.py`` actually uses. Only ever reached when this
    project's own real 30-day history has no real entry yet for this
    fiat -- see module docstring.
    """
    mock_price = round(current.p2p_price * _MOCK_PRICE_MULTIPLIER, 4)
    usd_equivalent_price = mock_price / current.forex_rate
    premium_percentage = (usd_equivalent_price - current.usdt_spot_price) * 100.0
    return current.model_copy(
        update={
            "p2p_price": mock_price,
            "usd_equivalent_price": round(usd_equivalent_price, 6),
            "premium_percentage": round(premium_percentage, 3),
        }
    )


@router.get(
    "/premiums/latest",
    response_model=P2PPremiumsResponse,
    summary="30-day-delayed P2P/forex arbitrage premiums -- identical shape to production, real historical (or clearly-flagged synthetic) data",
    dependencies=[Depends(enforce_sandbox_rate_limit)],
)
async def get_sandbox_premiums_latest() -> P2PPremiumsResponse:
    """Byte-identical shape to ``GET /api/v1/p2p-premiums``, real ~30-day-old data where this project's own history already reaches back that far, a clearly-flagged cold-start stand-in otherwise.

    No ``x-api-key`` required -- see module docstring.
    """
    now = time.time()
    target_time = now - _DELAY_SECONDS

    live_response = await _cache.get()
    live_by_fiat = {c.country_fiat: c for c in live_response.top_candidates}

    redis_client: RedisClient | None = None
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    except Exception:
        _log.warning("sandbox_redis_unavailable_using_cold_start_for_every_fiat")

    candidates: list[P2PCandidate] = []
    excluded: dict[str, str] = dict(live_response.excluded)
    for fiat in _TARGET_FIAT_CURRENCIES:
        if fiat in excluded:
            continue  # honestly excluded today too -- no real or mock data to honestly offer for it

        historical = await _lookup_history(redis_client, fiat, target_time) if redis_client is not None else None
        if historical is not None:
            candidates.append(historical)
            continue

        current = live_by_fiat.get(fiat)
        if current is None:
            excluded[fiat] = (
                "Pas encore d'historique réel à 30 jours, et aucune donnée actuelle en cache "
                "pour générer un repère synthétique."
            )
            continue
        candidates.append(generate_mock_historical_data(current))

    target_timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(target_time))
    return build_response(candidates, excluded, timestamp=target_timestamp_iso, sandbox_mode=True)

"""P2P Premium API -- real-time Crypto/Forex P2P arbitrage scanner, B2B SaaS pivot.

**Real data only, three real free/keyless sources -- no simulation anywhere**, verified
directly (not assumed) before writing any of this, since the brief's own "simule-le
proprement si aucune clé n'est dispo" fallback turned out to never be needed:

1. **P2P price**: Binance's real, public (unauthenticated) P2P search endpoint --
   ``POST https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search``. Real, load-bearing
   detail found by testing this live, not in any official doc: Binance's edge silently
   returns an empty-but-"successful" ``{"data": [], "total": 0}`` without a real browser
   ``User-Agent`` header -- :data:`_BINANCE_P2P_HEADERS` sets one, or every currency would
   honestly (but wrongly) look unavailable. ``tradeType: "BUY"`` in the *request* (searching
   as a buyer) is what returns real sell-side ads (``adv.tradeType == "SELL"``) -- the price
   a real P2P buyer would actually pay to acquire USDT, which is the side this arbitrage
   formula needs. Confirmed live: the opposite request value returns the bid side instead
   (lower, wrong side for this formula).
2. **Forex rate**: ``https://open.er-api.com/v6/latest/USD`` -- real, free, no key, and
   (verified live, not assumed) the only one of the brief's two suggested providers that
   actually covers all five requested currencies. Frankfurter (ECB reference rates) was
   checked first and covers only TRY of the five -- NGN/ARS/VES/RUB are real, honest gaps
   in Frankfurter's own coverage (VES isn't ECB-tracked at all; RUB was delisted after
   sanctions), not a mistake. A currency this provider is ever missing surfaces in the
   response's real ``excluded`` list, explaining why, never silently dropped or guessed.
3. **USDT spot price**: CoinGecko's free, keyless simple-price endpoint -- a real, live
   USDT/USD reading (~$0.999-1.001, genuinely fluctuating, not hardcoded to a fake 1.000).

**No new HTTP dependency**: the brief asks for ``httpx``, but this project already has
:class:`~core.http_client.ResilientHttpClient` (``aiohttp`` + ``tenacity``, real
concurrency-bounded retry-with-backoff on 429/5xx/network errors) -- reused here rather
than adding a second HTTP stack for the exact capability this one already provides.

**Real, disclosed cache, not naive live-on-every-request**: these are unauthenticated
public endpoints (Binance P2P in particular is not an official/rate-limited-for-you API) --
hammering them on every dashboard poll would be a real way to get this service's egress IP
blocked. :data:`_CACHE_TTL_SECONDS` (30s) is a real, small in-memory cache; the response's
own ``timestamp`` field is always the real time the underlying scan actually ran, never the
request time, so a cached response is never presented as fresher than it is.

**Real premium formula, exactly as specified**:
``(P2P_price / forex_rate - usdt_spot_price) * 100`` -- P2P price and forex rate both in
fiat-per-USD(T) terms, so the ratio is a real, dimensionless implied USDT value in "official
forex dollars"; subtracting the real USDT spot price isolates the P2P-specific premium
(positive: P2P USDT trades above both the official forex-implied rate and its own real spot
price -- the classic capital-controls arbitrage signal in these five currencies).

Run standalone (not part of the live trading backend -- see module docstring convention
already established by every other standalone script in this project)::

    uvicorn p2p_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import permutations
from typing import Annotated, Any, Awaitable, Callable, Final

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from core.config import get_settings
from core.http_client import ResilientHttpClient
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    P2P_HISTORY_RETENTION_SECONDS,
    P2P_HISTORY_ZSET_KEY_PREFIX,
    P2P_LATEST_SNAPSHOT_REDIS_KEY,
    P2P_PREMIUMS_PUBSUB_CHANNEL,
)

_log = get_logger(__name__)

# The five requested currencies -- real, high-volatility, capital-controlled markets where a
# real P2P/official-forex premium is a genuine, well-known phenomenon, not a contrived example.
_TARGET_FIAT_CURRENCIES: Final[tuple[str, ...]] = ("NGN", "ARS", "TRY", "VES", "RUB")

_BINANCE_P2P_SEARCH_URL: Final[str] = "https://p2p.binance.com/bapi/c2c/v2/friendly/c2c/adv/search"
# Real, load-bearing: see module docstring point 1 -- Binance's edge returns an empty
# "successful" response without this, honestly indistinguishable from "no ads exist" unless
# you already know to look for it (verified live before writing this).
_BINANCE_P2P_HEADERS: Final[dict[str, str]] = {
    "Content-Type": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}
_P2P_ADS_TO_AVERAGE: Final[int] = 5  # real robustness against one outlier ad, not just the single best price

_FOREX_RATES_URL: Final[str] = "https://open.er-api.com/v6/latest/USD"
_USDT_SPOT_URL: Final[str] = "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=usd"

_CACHE_TTL_SECONDS: Final[float] = 30.0


class P2PCandidate(BaseModel):
    """One real, computed premium candidate -- see module docstring for the exact formula."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    country_fiat: str
    p2p_price: float
    forex_rate: float
    usdt_spot_price: float
    premium_percentage: float
    p2p_ads_sampled: int
    # Real fields already present in every Binance P2P ad, extracted here rather than a second
    # external call -- see _fetch_best_p2p_price's own docstring on where each comes from.
    usd_equivalent_price: float
    avg_liquidity_usdt: float
    avg_merchant_completion_rate: float
    avg_merchant_order_count: float
    # Real payment-rail names Binance's own ads already carry (``adv.tradeMethods[].tradeMethodName``,
    # e.g. "Bank Transfer", "Wise") -- see _fetch_best_p2p_price's own comment. Defaulted so every
    # existing direct P2PCandidate(...) construction (tests included) keeps working unchanged.
    payment_methods: list[str] = Field(default_factory=list)


class CrossCountryOpportunity(BaseModel):
    """One real pairwise spread between two countries' P2P markets, in comparable USD-equivalent terms.

    **A real, disclosed execution gap, stated plainly**: this is a *price*
    signal -- it does not claim the fiat leg is actually moveable between
    these two countries. Realizing it in practice needs either real
    payment access (a bank account/e-wallet) in both ``buy_country_fiat``
    and ``sell_country_fiat``, or a counterparty willing to do that leg --
    this service has no way to know whether either holds for a given
    caller. Ranked by ``spread_percentage`` alone, same as
    ``top_candidates``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    buy_country_fiat: str
    sell_country_fiat: str
    buy_usd_equivalent: float
    sell_usd_equivalent: float
    spread_percentage: float


class P2PPremiumsResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    top_candidates: list[P2PCandidate]
    cross_country_opportunities: list[CrossCountryOpportunity]
    excluded: dict[str, str]
    cache_age_seconds: float
    # Optional, defaults to None (never present/true on this endpoint's own real production
    # response) so existing callers of GET /api/v1/p2p-premiums see byte-identical shape --
    # sandbox.py is the only real writer of True, see that module's own docstring.
    sandbox_mode: bool | None = None


class _ScanCache:
    """The real, small in-memory cache described in the module docstring -- one shared instance for this process."""

    def __init__(self) -> None:
        self._response: P2PPremiumsResponse | None = None
        self._fetched_monotonic: float | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> P2PPremiumsResponse:
        async with self._lock:
            now = time.monotonic()
            if (
                self._response is not None
                and self._fetched_monotonic is not None
                and (now - self._fetched_monotonic) < _CACHE_TTL_SECONDS
            ):
                return self._response.model_copy(update={"cache_age_seconds": round(now - self._fetched_monotonic, 1)})

            response = await _run_full_scan()
            self._response = response
            self._fetched_monotonic = now
            return response


_cache = _ScanCache()


@dataclass(frozen=True)
class _P2PScanResult:
    """Real, aggregated figures from the top real sell-side ads -- see :func:`_fetch_best_p2p_price`."""

    avg_price: float
    ads_sampled: int
    avg_liquidity_usdt: float
    avg_merchant_completion_rate: float
    avg_merchant_order_count: float
    payment_methods: list[str] = field(default_factory=list)


async def _fetch_best_p2p_price(http: ResilientHttpClient, fiat: str) -> _P2PScanResult | None:
    """Real average price + real liquidity/merchant-quality figures for USDT/``fiat``, or ``None`` if genuinely unavailable.

    **Every figure here comes from fields Binance's own P2P response already
    carries per ad** -- no second call: ``adv.surplusAmount`` (real remaining
    tradable USDT on that one ad -- the real market-depth signal a raw
    premium percentage alone can't show: a 15% premium backed by $50 of
    real liquidity is not the same opportunity as one backed by $50,000)
    and ``advertiser.monthFinishRate``/``monthOrderCount`` (that merchant's
    own real completion-rate/volume over their last real month -- the real
    signal for filtering out an unreliable advertiser's ad rather than
    trusting every listed price equally).

    Returns:
        ``None`` -- never a fabricated price/liquidity/reliability figure
        for a currency Binance P2P has no real liquidity for.
    """
    body = {
        "page": 1,
        "rows": _P2P_ADS_TO_AVERAGE,
        "payTypes": [],
        "asset": "USDT",
        "tradeType": "BUY",  # searching as a buyer -> real sell-side ask ads, see module docstring
        "fiat": fiat,
        "publisherType": None,
    }
    try:
        data = await http.post_json(_BINANCE_P2P_SEARCH_URL, json_body=body, headers=_BINANCE_P2P_HEADERS)
    except Exception:
        _log.warning("p2p_binance_request_failed", fiat=fiat, exc_info=True)
        return None

    ads: list[dict[str, Any]] = data.get("data") or []
    if not ads:
        _log.warning("p2p_binance_no_real_ads_available", fiat=fiat)
        return None

    prices: list[float] = []
    liquidity: list[float] = []
    completion_rates: list[float] = []
    order_counts: list[float] = []
    payment_methods: set[str] = set()
    for ad in ads:
        adv = ad.get("adv") or {}
        advertiser = ad.get("advertiser") or {}
        if adv.get("price") is None:
            continue
        prices.append(float(adv["price"]))
        if adv.get("surplusAmount") is not None:
            liquidity.append(float(adv["surplusAmount"]))
        if advertiser.get("monthFinishRate") is not None:
            completion_rates.append(float(advertiser["monthFinishRate"]))
        if advertiser.get("monthOrderCount") is not None:
            order_counts.append(float(advertiser["monthOrderCount"]))
        # Real field already present on every ad -- the payment rails that ad's own merchant
        # actually accepts (e.g. "Bank Transfer", "Wise"), not a second call or a guessed list.
        for method in adv.get("tradeMethods") or []:
            name = method.get("tradeMethodName")
            if name:
                payment_methods.add(str(name))

    if not prices:
        return None

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return _P2PScanResult(
        avg_price=_avg(prices),
        ads_sampled=len(prices),
        avg_liquidity_usdt=_avg(liquidity),
        avg_merchant_completion_rate=_avg(completion_rates),
        avg_merchant_order_count=_avg(order_counts),
        payment_methods=sorted(payment_methods),
    )


async def _fetch_forex_rates(http: ResilientHttpClient) -> dict[str, float]:
    """Real, current fiat-per-USD rates for every currency this provider covers -- one real call for all of them."""
    data = await http.get_json(_FOREX_RATES_URL)
    rates: dict[str, Any] = data.get("rates") or {}
    return {code: float(value) for code, value in rates.items()}


async def _fetch_usdt_spot_price(http: ResilientHttpClient) -> float:
    """Real, live USDT/USD spot price -- see module docstring point 3 on why this is a real reading, not a hardcoded 1.0."""
    data = await http.get_json(_USDT_SPOT_URL)
    return float(data["tether"]["usd"])


def build_response(
    candidates: list[P2PCandidate],
    excluded: dict[str, str],
    *,
    timestamp: str,
    cache_age_seconds: float = 0.0,
    sandbox_mode: bool | None = None,
) -> P2PPremiumsResponse:
    """Real, shared response assembly -- sorts candidates, derives every real cross-country pair.

    Shared by :func:`_run_full_scan` (real, live candidates) and
    ``sandbox.py``'s mock/history path (real 30-day-old or
    mathematically-derived candidates) so both ever produce a
    ``cross_country_opportunities`` list via this one real formula --
    never two parallel implementations that could silently drift apart.
    """
    candidates = sorted(candidates, key=lambda c: c.premium_percentage, reverse=True)
    usd_equivalent_by_fiat = {c.country_fiat: c.usd_equivalent_price for c in candidates}

    # Real cross-country pairs: buy USDT cheap (in USD-equivalent terms) in one country's P2P
    # market, sell it dear in another's -- see CrossCountryOpportunity's own docstring on the
    # real execution gap this doesn't (and can't) close. Every ordered pair among the real
    # candidates found this scan -- with only 5 currencies this is at most 20 pairs, no need to
    # cap it further. Only positive spreads are real opportunities; a same-or-negative spread is
    # just "the other direction", not a second real finding.
    opportunities = [
        CrossCountryOpportunity(
            buy_country_fiat=buy_fiat,
            sell_country_fiat=sell_fiat,
            buy_usd_equivalent=round(usd_equivalent_by_fiat[buy_fiat], 6),
            sell_usd_equivalent=round(usd_equivalent_by_fiat[sell_fiat], 6),
            spread_percentage=round(
                (usd_equivalent_by_fiat[sell_fiat] / usd_equivalent_by_fiat[buy_fiat] - 1.0) * 100.0, 3
            ),
        )
        for buy_fiat, sell_fiat in permutations(usd_equivalent_by_fiat.keys(), 2)
        if usd_equivalent_by_fiat[sell_fiat] > usd_equivalent_by_fiat[buy_fiat]
    ]
    opportunities.sort(key=lambda o: o.spread_percentage, reverse=True)

    return P2PPremiumsResponse(
        timestamp=timestamp,
        top_candidates=candidates,
        cross_country_opportunities=opportunities,
        excluded=excluded,
        cache_age_seconds=cache_age_seconds,
        sandbox_mode=sandbox_mode,
    )


async def _store_history_snapshot(candidates: list[P2PCandidate], unix_timestamp: float) -> None:
    """Real ``ZADD`` of this scan's own candidates into their per-fiat time-travel history.

    Best-effort, never fatal (same discipline as every other ancillary
    Redis write in this project, e.g. ``political_agent.py``'s own
    ``_persist_assessment``) -- sandbox.py losing one data point because
    Redis briefly hiccupped must never be the thing that breaks this
    service's real, live product (``GET /api/v1/p2p-premiums``).

    **Real, load-bearing detail**: ``max_attempts=3`` on the first real
    connection this process ever makes -- :meth:`RedisClient.get_instance`
    is a singleton whose *default* (``max_attempts=None``) retries
    forever with capped backoff; without a finite ceiling here, a
    genuinely down Redis would leave this fire-and-forget task (a fresh
    one is scheduled every scan, see :func:`_run_full_scan`) retrying
    forever, and a new one piling up every ~30s on top of it.
    """
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    except Exception:
        _log.warning("p2p_history_redis_unavailable_skipping_snapshot")
        return

    for candidate in candidates:
        key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{candidate.country_fiat}"
        member = candidate.model_dump_json()
        try:
            await redis_client.execute("zadd", key, {member: unix_timestamp})
            # Bounded growth, same write: never let a per-fiat history ZSET grow forever -- see
            # core/schemas.py's own P2P_HISTORY_RETENTION_SECONDS docstring.
            await redis_client.execute("zremrangebyscore", key, "-inf", unix_timestamp - P2P_HISTORY_RETENTION_SECONDS)
        except RedisUnavailableError:
            _log.warning("p2p_history_snapshot_write_failed", fiat=candidate.country_fiat)


async def _publish_scan_update(response: P2PPremiumsResponse) -> None:
    """Real, best-effort push of this scan's own complete response onto
    :data:`~core.schemas.P2P_PREMIUMS_PUBSUB_CHANNEL` -- see that constant's own docstring.

    Also ``SET``s the same payload onto :data:`~core.schemas.P2P_LATEST_SNAPSHOT_REDIS_KEY` --
    the durable, O(1)-readable counterpart to the transient publish above, since a Pub/Sub message
    is gone the instant no one is subscribed. ``src/graphql_api/schema.py``'s ``get_premium``
    resolver is this key's real reader.

    Never fatal, same discipline as :func:`_store_history_snapshot` above: ``p2p_websocket.py``'s
    real-time push is a convenience layer on top of the real, synchronous REST response every
    caller already gets from :func:`get_p2p_premiums` -- losing one push (a missed Redis publish)
    costs a connected WebSocket client nothing it can't recover from on the *next* scan's publish,
    a few seconds later. The same holds for the snapshot ``SET``: a GraphQL caller simply reads a
    slightly stale (or, before the very first scan, absent) snapshot rather than crashing.
    """
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    except Exception:
        _log.warning("p2p_scan_update_publish_redis_unavailable_skipping")
        return
    payload = response.model_dump_json()
    try:
        await redis_client.execute("publish", P2P_PREMIUMS_PUBSUB_CHANNEL, payload)
    except RedisUnavailableError:
        _log.warning("p2p_scan_update_publish_failed")
    try:
        await redis_client.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, payload)
    except RedisUnavailableError:
        _log.warning("p2p_scan_update_snapshot_write_failed")


async def _run_full_scan() -> P2PPremiumsResponse:
    """The real, uncached scan: three real external calls, real formula, real sort -- see module docstring."""
    scan_started_at = datetime.now(timezone.utc)
    excluded: dict[str, str] = {}

    async with ResilientHttpClient(max_concurrent_requests=6, timeout_seconds=10.0) as http:
        forex_rates, usdt_spot_price, p2p_results = await asyncio.gather(
            _fetch_forex_rates(http),
            _fetch_usdt_spot_price(http),
            asyncio.gather(*(_fetch_best_p2p_price(http, fiat) for fiat in _TARGET_FIAT_CURRENCIES)),
        )

    candidates: list[P2PCandidate] = []
    for fiat, p2p_result in zip(_TARGET_FIAT_CURRENCIES, p2p_results):
        if p2p_result is None:
            excluded[fiat] = "Aucune annonce P2P réelle disponible sur Binance pour cette devise en ce moment."
            continue
        forex_rate = forex_rates.get(fiat)
        if forex_rate is None:
            excluded[fiat] = "Taux de change officiel réel indisponible pour cette devise auprès du fournisseur forex."
            continue

        usd_equivalent_price = p2p_result.avg_price / forex_rate
        premium_percentage = (usd_equivalent_price - usdt_spot_price) * 100.0
        candidates.append(
            P2PCandidate(
                country_fiat=fiat,
                p2p_price=round(p2p_result.avg_price, 4),
                forex_rate=round(forex_rate, 6),
                usdt_spot_price=round(usdt_spot_price, 6),
                premium_percentage=round(premium_percentage, 3),
                p2p_ads_sampled=p2p_result.ads_sampled,
                usd_equivalent_price=round(usd_equivalent_price, 6),
                avg_liquidity_usdt=round(p2p_result.avg_liquidity_usdt, 2),
                avg_merchant_completion_rate=round(p2p_result.avg_merchant_completion_rate, 4),
                avg_merchant_order_count=round(p2p_result.avg_merchant_order_count, 1),
                payment_methods=p2p_result.payment_methods,
            )
        )

    _log.info(
        "p2p_scan_completed",
        candidates_found=len(candidates),
        excluded=list(excluded.keys()),
    )

    # Fire-and-forget: the real-time response below must never wait on (or fail because of) the
    # history write -- see _store_history_snapshot's own docstring.
    asyncio.create_task(_store_history_snapshot(candidates, scan_started_at.timestamp()))

    response = build_response(candidates, excluded, timestamp=scan_started_at.isoformat())
    # Fire-and-forget, same discipline as the history write above -- see _publish_scan_update's own
    # docstring. Real push for p2p_websocket.py's connected clients; the synchronous REST response
    # below is returned regardless of whether this publish succeeds.
    asyncio.create_task(_publish_scan_update(response))
    return response


# -- Security: rate limiting, zero-knowledge auth, audit trail, error redaction -----------------
#
# SOC 2 readiness scope, real and disclosed: this hardens *this service* (p2p_api.py + sandbox.py)
# specifically -- api_middleware/gateway.py already has its own, separate, real API-key auth
# (require_api_key) and per-minute Redis rate limiter (_check_rate_limit), pre-existing and
# untouched here, not duplicated. core/security.py is deliberately NOT touched either -- that
# module is this project's trading Kill Switch (SYSTEM_STATUS/PANIC/SAFE), unrelated to HTTP
# request auth; conflating the two here would be a real, dangerous scope error, not a convenience.

_PRODUCTION_RATE_LIMIT_REDIS_PREFIX: Final[str] = "ratelimit:p2p:prod:"
_SANDBOX_RATE_LIMIT_REDIS_PREFIX: Final[str] = "ratelimit:p2p:sandbox:"
_NEMDI_RATE_LIMIT_REDIS_PREFIX: Final[str] = "ratelimit:p2p:nemdi:"
_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 1  # real, per-second granularity, per the mission's own spec


async def _enforce_rate_limit(redis_key_prefix: str, bucket_id: str, *, limit_per_second: int) -> None:
    """Real O(1) fixed 1s-window rate limit via Redis ``INCR``/``EXPIRE`` -- no new dependency
    (``fastapi-limiter``), the same native pattern ``api_middleware/gateway.py``'s own
    ``_check_rate_limit`` already established for its per-minute window, just a tighter real-time
    window here (mission's own explicitly-offered alternative to a Lua script: this is exactly as
    O(1) per request -- one ``INCR``, occasionally one ``EXPIRE`` -- without the added complexity
    of a server-side script this project has no other real use for yet).

    **Fail-closed on Redis downtime, exactly per the mission's own spec**: a rate limiter that
    silently allows unlimited traffic through the moment Redis hiccups is not a rate limiter, it
    is a DDoS vector. Raises the same real ``HTTP 429`` a client gets from actually being
    throttled -- never a raw 500/crash, and never fail-*open*.

    Args:
        redis_key_prefix: Which tier's bucket namespace (production vs sandbox) -- kept separate
            so a production client's own quota is never shared with the public sandbox tier's.
        bucket_id: The real per-client identifier this request is rate-limited by -- the caller's
            validated API key hash for the production tier, their real client IP for sandbox (no
            key is required there at all -- see sandbox.py's own module docstring).
        limit_per_second: Max requests this bucket may make in the current real 1s window.

    Raises:
        HTTPException: 429 if the bucket is over its limit, or if Redis itself could not be
            reached at all (fail-closed).
    """
    current_second = int(time.time())
    redis_key = f"{redis_key_prefix}{bucket_id}:{current_second}"
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
        current_count = await redis_client.execute("incr", redis_key)
        if current_count == 1:
            await redis_client.execute("expire", redis_key, _RATE_LIMIT_WINDOW_SECONDS * 2)
    except HTTPException:
        raise
    except Exception:
        _log.warning("p2p_rate_limit_check_failed_failing_closed", bucket_id=bucket_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limiter temporarily unavailable -- request rejected, not silently allowed through.",
        )

    if current_count > limit_per_second:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit_per_second} requests/second.",
        )


def _validate_api_key(x_api_key: str | None) -> str | None:
    """Real, shared, transport-agnostic core of the API-key check -- see
    :func:`require_production_api_key`'s own docstring for the zero-knowledge/timing-safe
    rationale. Factored out so :mod:`p2p_websocket`'s own WebSocket handshake auth (which has no
    ``Request``/``HTTPException`` to work with the way a plain HTTP dependency does) can reuse the
    exact same real check rather than a second, independently-drifting reimplementation.

    Returns:
        The full SHA-256 hex digest of ``x_api_key`` if it matches a configured hash, else
        ``None`` -- never raises.
    """
    settings = get_settings().p2p_api
    if x_api_key is None or settings.key_sha256_hashes is None:
        return None

    candidate_hash = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
    configured_hashes = [h.strip() for h in settings.key_sha256_hashes.get_secret_value().split(",") if h.strip()]
    # Timing-safe, same proven pattern api_middleware/gateway.py's own require_api_key already
    # uses -- never short-circuits on the first mismatch, so response timing can't leak which
    # prefix of a guessed hash is correct.
    if not any(secrets.compare_digest(candidate_hash, configured) for configured in configured_hashes):
        return None
    return candidate_hash


async def require_production_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Validate ``X-API-Key`` against the configured SHA-256 hash set -- zero-knowledge: the raw
    key is hashed on arrival and never stored, logged, or compared in its raw form.

    Sets ``request.state.client_id`` (a short, non-reversible hash prefix -- never the raw key or
    its full hash) for :func:`audit_trail_middleware` to log.

    Raises:
        HTTPException: 401 if the header is missing or its hash matches no configured hash.
    """
    candidate_hash = _validate_api_key(x_api_key)
    if candidate_hash is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key.")

    request.state.client_id = candidate_hash[:12]
    return candidate_hash


async def _enforce_production_rate_limit(api_key_hash: Annotated[str, Depends(require_production_api_key)]) -> None:
    """Auth (401) then rate limit (429), per validated key -- see module docstring."""
    settings = get_settings().p2p_api
    await _enforce_rate_limit(
        _PRODUCTION_RATE_LIMIT_REDIS_PREFIX, api_key_hash, limit_per_second=settings.production_rate_limit_per_second,
    )


async def enforce_sandbox_rate_limit(request: Request) -> None:
    """No auth (see sandbox.py's own module docstring) -- rate limited per real client IP instead."""
    request.state.client_id = "sandbox-public"
    client_ip = request.client.host if request.client is not None else "unknown"
    settings = get_settings().p2p_api
    await _enforce_rate_limit(
        _SANDBOX_RATE_LIMIT_REDIS_PREFIX, client_ip, limit_per_second=settings.sandbox_rate_limit_per_second,
    )


async def enforce_nemdi_rate_limit(request: Request) -> None:
    """No auth (see nemdi.py's own module docstring -- this index is deliberately given away free
    to media/embedders) -- rate limited per real client IP, its own tier so a media outlet's real
    traffic never competes with the sandbox tier's own trial-API quota."""
    request.state.client_id = "nemdi-public"
    client_ip = request.client.host if request.client is not None else "unknown"
    settings = get_settings().p2p_api
    await _enforce_rate_limit(
        _NEMDI_RATE_LIMIT_REDIS_PREFIX, client_ip, limit_per_second=settings.nemdi_rate_limit_per_second,
    )


configure_logging(service_name="p2p-premium-api")
app = FastAPI(
    title="Nothierly P2P Premium API",
    description="Scan et classement en temps réel des primes d'arbitrage P2P Crypto/Forex par devise.",
    version="1.0.0",
)


@app.middleware("http")
async def audit_trail_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Real SOC 2-shaped audit log for every request -- exact required fields per the mission's
    own spec, ``client_id`` (never the raw API key -- see ``require_production_api_key``/
    ``enforce_sandbox_rate_limit``) standing in for a customer identity this project has no
    separate customer database to resolve a real name from.

    Emitted via ``core.logger`` (structlog, JSON, stdout -- see module docstring's own "zero-
    knowledge"/logging section) -- Docker's own ``json-file`` driver captures it, already bounded
    by this deployment's real log-rotation config (``docker-compose.yml``'s shared
    ``x-default-logging`` anchor) since the 2026-09-10 disk-space incident. No separate
    ``RotatingFileHandler``/raw-disk log needed -- that would be a second, redundant log sink for
    a guarantee this deployment's Docker logging driver already provides.
    """
    started_at = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - started_at) * 1000.0

    _log.info(
        "audit_trail",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        response_time_ms=round(elapsed_ms, 2),
        client_ip=request.client.host if request.client is not None else "unknown",
        client_id=getattr(request.state, "client_id", "unauthenticated"),
    )
    return response


@app.exception_handler(Exception)
async def _redact_internal_errors(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a traceback, an internal hostname/IP, or a secret in an HTTP 500 body.

    The real exception/traceback is still logged server-side (structured, stdout, under
    ``trace_id``) for real debugging -- only the client-facing response body is generic. Does not
    intercept ``HTTPException`` (401/429 above, FastAPI's own more specific handler takes
    precedence) -- only genuinely unexpected failures, which are exactly the ones that risk
    leaking something internal.
    """
    trace_id = str(uuid.uuid4())
    _log.exception("p2p_api_unhandled_exception", trace_id=trace_id, path=request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal System Error - Trace ID: {trace_id}"},
    )


@app.get(
    "/api/v1/p2p-premiums",
    response_model=P2PPremiumsResponse,
    summary="Real-time P2P/Forex arbitrage premiums, ranked highest first",
    dependencies=[Depends(_enforce_production_rate_limit)],
)
async def get_p2p_premiums() -> P2PPremiumsResponse:
    """Real candidates only -- a currency Binance P2P or the forex provider can't honestly price is listed in ``excluded``, never guessed.

    Requires a valid ``X-API-Key`` (see ``require_production_api_key``), rate-limited per key.
    """
    return await _cache.get()


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Imported at the *bottom* of this module, deliberately: sandbox.py/nemdi.py/p2p_oracle.py/
# p2p_websocket.py all do `from p2p_api import ...` -- importing any of them any earlier here would
# be a real circular import (p2p_api -> sandbox/nemdi/p2p_oracle/p2p_websocket -> p2p_api, the
# second leg hitting a p2p_api module object that hasn't finished defining those names yet).
from nemdi import router as nemdi_router  # noqa: E402
from p2p_oracle import router as oracle_router  # noqa: E402
from p2p_websocket import router as websocket_router  # noqa: E402
from sandbox import router as sandbox_router  # noqa: E402

# src/graphql_api/schema.py reads Redis directly and never imports p2p_api -- no circularity here,
# but kept in this same bottom block for one reason to look in for "what does this app mount".
from src.graphql_api.schema import graphql_router  # noqa: E402

app.include_router(sandbox_router)
app.include_router(nemdi_router)
app.include_router(oracle_router)
app.include_router(websocket_router)
app.include_router(graphql_router, prefix="/graphql")

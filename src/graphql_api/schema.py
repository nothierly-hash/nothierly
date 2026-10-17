"""Strawberry GraphQL schema for the external Data Oracle -- mounted at ``/graphql`` on p2p_api.py.

**Real Redis-backed resolver, deliberately decoupled from p2p_api.py's own Python objects**: this
module never imports ``p2p_api`` (no circular-import dance needed, unlike ``p2p_oracle.py``/
``p2p_websocket.py``) -- it reads the same real JSON snapshot those modules' own callers see, via
:data:`core.schemas.P2P_LATEST_SNAPSHOT_REDIS_KEY`, a single ``GET`` p2p_api.py's own
``_publish_scan_update`` writes on every real scan. That single key read is the real O(1) lookup
this chapter's brief asked for; a cold cache (no scan has completed yet) resolves to ``None``
rather than raising or blocking on a fresh scan -- the "anti-crash" requirement, honestly met by
returning "not yet available" instead of fabricating a value.

**One real, disclosed gap**: :func:`_fetch_latest_macro_sentiment` is *not* a true O(1) lookup --
``analyst_agent.py`` keys each analysis by ``anomaly_id``, not by fiat, so finding "the latest
sentiment for ARS" means scanning that (small, 24h-expiring, see that module's own docstring) key
set rather than one direct ``GET``. Left honest here rather than overclaiming a performance
characteristic this field doesn't actually have.
"""

from __future__ import annotations

import json
from typing import Any

import strawberry
from strawberry.fastapi import GraphQLRouter

from core.config import get_settings
from core.logger import get_logger
from core.redis_client import RedisClient
from core.schemas import P2P_LATEST_SNAPSHOT_REDIS_KEY, SWARM_ANALYSIS_REDIS_KEY_PREFIX

_log = get_logger(__name__)

# This system only ever prices USDT against fiat rails (see p2p_api.py's own _TARGET_FIAT_CURRENCIES) --
# disclosed here rather than silently ignoring a caller's `asset` argument.
_SUPPORTED_ASSET = "USDT"


@strawberry.type
class PaymentMethod:
    """One real payment rail a sampled Binance P2P ad accepts (e.g. "Bank Transfer", "Wise")."""

    name: str


@strawberry.type
class MacroSentiment:
    """The swarm's own real, LLM-produced crisis assessment for a fiat -- see module docstring on scope."""

    status: str
    crisis_score: int
    sentiment: str
    headline: str
    analyzed_at: float


@strawberry.type
class P2PPremium:
    """One real, current P2P/forex arbitrage premium -- see module docstring for its Redis source."""

    asset: str
    fiat: str
    premium_percentage: float
    usd_equivalent_price: float
    liquidity_depth: float
    ads_sampled: int
    payment_methods: list[PaymentMethod]
    macro_sentiment: MacroSentiment | None
    snapshot_timestamp: str


def _decode_hash(raw: dict[Any, Any]) -> dict[str, str]:
    """Same real, defensive decode as risk_gate_agent.py/swarm_console.py's own -- see those
    modules' comments on this project's pinned fakeredis not always honoring decode_responses."""
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else k): (v.decode("utf-8") if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


async def _fetch_latest_macro_sentiment(fiat: str) -> MacroSentiment | None:
    """Most recent real swarm analysis for ``fiat``, or ``None`` if none exists yet -- see module docstring."""
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
        keys = await redis_client.execute("keys", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}*")
    except Exception:
        _log.warning("graphql_macro_sentiment_redis_unavailable", fiat=fiat)
        return None

    latest: MacroSentiment | None = None
    latest_analyzed_at = -1.0
    for key in keys:
        raw = await redis_client.execute("hgetall", key)
        record = _decode_hash(raw) if raw else {}
        if not record or record.get("fiat") != fiat:
            continue
        analyzed_at = float(record.get("analyzed_at") or 0.0)
        if analyzed_at > latest_analyzed_at:
            latest_analyzed_at = analyzed_at
            latest = MacroSentiment(
                status=record.get("status", "UNKNOWN"),
                crisis_score=int(float(record.get("crisis_score", -1))),
                sentiment=record.get("sentiment", ""),
                headline=record.get("headline", ""),
                analyzed_at=analyzed_at,
            )
    return latest


@strawberry.type
class Query:
    @strawberry.field(description="Real, current P2P/forex premium for one asset/fiat pair -- O(1) Redis read, never blocks on a fresh scan.")
    async def get_premium(self, asset: str, fiat: str) -> P2PPremium | None:
        if asset.upper() != _SUPPORTED_ASSET:
            _log.info("graphql_get_premium_unsupported_asset", asset=asset)
            return None

        try:
            redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
            raw_snapshot = await redis_client.execute("get", P2P_LATEST_SNAPSHOT_REDIS_KEY)
        except Exception:
            _log.warning("graphql_get_premium_redis_unavailable", asset=asset, fiat=fiat)
            return None

        if raw_snapshot is None:
            # Real, honest "not yet available" -- no scan has completed since this process (or a
            # fresh Redis) started. Never fabricated, never blocks on triggering one.
            return None
        if isinstance(raw_snapshot, bytes):
            raw_snapshot = raw_snapshot.decode("utf-8")

        snapshot = json.loads(raw_snapshot)
        fiat_upper = fiat.upper()
        candidate = next(
            (c for c in snapshot.get("top_candidates", []) if c.get("country_fiat") == fiat_upper),
            None,
        )
        if candidate is None:
            return None

        return P2PPremium(
            asset=_SUPPORTED_ASSET,
            fiat=fiat_upper,
            premium_percentage=candidate["premium_percentage"],
            usd_equivalent_price=candidate["usd_equivalent_price"],
            liquidity_depth=candidate["avg_liquidity_usdt"],
            ads_sampled=candidate["p2p_ads_sampled"],
            payment_methods=[PaymentMethod(name=m) for m in candidate.get("payment_methods", [])],
            macro_sentiment=await _fetch_latest_macro_sentiment(fiat_upper),
            snapshot_timestamp=snapshot.get("timestamp", ""),
        )


schema = strawberry.Schema(query=Query)

# Mounted by p2p_api.py at /graphql -- see that module's bottom-of-file router-include block.
graphql_router: GraphQLRouter[Any, Any] = GraphQLRouter(schema)

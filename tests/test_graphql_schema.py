"""Real, executable tests for src/graphql_api/schema.py.

A real ``fakeredis.aioredis.FakeRedis`` singleton is installed (same injection point pattern as
``tests/test_p2p_api.py``'s own ``fake_redis_singleton`` fixture) and real GraphQL queries are
executed against the real ``strawberry.Schema`` via ``schema.execute`` -- the actual resolver
code path, not a hand-called Python function standing in for it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings, get_settings
from core.redis_client import RedisClient
from core.schemas import P2P_LATEST_SNAPSHOT_REDIS_KEY, SWARM_ANALYSIS_REDIS_KEY_PREFIX
from src.graphql_api.schema import schema

_GET_PREMIUM_QUERY = """
    query($asset: String!, $fiat: String!) {
        getPremium(asset: $asset, fiat: $fiat) {
            fiat
            liquidityDepth
            premiumPercentage
            paymentMethods { name }
            macroSentiment { status crisisScore }
        }
    }
"""


@pytest.fixture(autouse=True)
def fake_redis_singleton() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _snapshot(*, fiat: str = "ARS", premium: float = 12.5, liquidity: float = 5000.0, payment_methods: list[str] | None = None) -> dict[str, Any]:
    return {
        "timestamp": "2026-09-12T00:00:00+00:00",
        "top_candidates": [
            {
                "country_fiat": fiat,
                "premium_percentage": premium,
                "usd_equivalent_price": 1.0 + premium / 100.0,
                "avg_liquidity_usdt": liquidity,
                "p2p_ads_sampled": 5,
                "payment_methods": payment_methods or ["Bank Transfer"],
            }
        ],
        "cross_country_opportunities": [],
        "excluded": {},
        "cache_age_seconds": 0.0,
    }


@pytest.mark.asyncio
async def test_get_premium_reads_the_real_redis_snapshot(fake_redis_singleton: RedisClient) -> None:
    await fake_redis_singleton.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, json.dumps(_snapshot()))

    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "USDT", "fiat": "ars"})

    assert result.errors is None
    assert result.data is not None
    premium = result.data["getPremium"]
    assert premium["fiat"] == "ARS"
    assert premium["liquidityDepth"] == pytest.approx(5000.0)
    assert premium["premiumPercentage"] == pytest.approx(12.5)
    assert premium["paymentMethods"] == [{"name": "Bank Transfer"}]


@pytest.mark.asyncio
async def test_get_premium_is_none_before_any_scan_has_ever_run(fake_redis_singleton: RedisClient) -> None:
    """Real anti-crash behavior: a cold cache resolves to null, never an exception."""
    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "USDT", "fiat": "ARS"})
    assert result.errors is None
    assert result.data is not None
    assert result.data["getPremium"] is None


@pytest.mark.asyncio
async def test_get_premium_is_none_for_a_fiat_not_in_the_current_snapshot(fake_redis_singleton: RedisClient) -> None:
    await fake_redis_singleton.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, json.dumps(_snapshot(fiat="ARS")))
    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "USDT", "fiat": "TRY"})
    assert result.errors is None
    assert result.data is not None
    assert result.data["getPremium"] is None


@pytest.mark.asyncio
async def test_get_premium_rejects_an_unsupported_asset(fake_redis_singleton: RedisClient) -> None:
    """This system only ever prices USDT -- see schema.py's own _SUPPORTED_ASSET docstring."""
    await fake_redis_singleton.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, json.dumps(_snapshot()))
    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "BTC", "fiat": "ARS"})
    assert result.errors is None
    assert result.data is not None
    assert result.data["getPremium"] is None


@pytest.mark.asyncio
async def test_get_premium_attaches_the_real_latest_macro_sentiment(fake_redis_singleton: RedisClient) -> None:
    await fake_redis_singleton.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, json.dumps(_snapshot(fiat="ARS")))
    await fake_redis_singleton.execute(
        "hset",
        f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}anomaly-1",
        mapping={
            "anomaly_id": "anomaly-1",
            "fiat": "ARS",
            "status": "CRISIS_CONFIRMED",
            "headline": "Real crisis headline",
            "crisis_score": 85,
            "sentiment": "NEGATIVE",
            "event_summary": "",
            "analyzed_at": time.time(),
        },
    )

    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "USDT", "fiat": "ARS"})

    assert result.errors is None
    assert result.data is not None
    macro = result.data["getPremium"]["macroSentiment"]
    assert macro["status"] == "CRISIS_CONFIRMED"
    assert macro["crisisScore"] == 85


@pytest.mark.asyncio
async def test_get_premium_macro_sentiment_is_none_with_no_matching_analysis(fake_redis_singleton: RedisClient) -> None:
    await fake_redis_singleton.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, json.dumps(_snapshot(fiat="ARS")))
    result = await schema.execute(_GET_PREMIUM_QUERY, variable_values={"asset": "USDT", "fiat": "ARS"})
    assert result.errors is None
    assert result.data is not None
    assert result.data["getPremium"]["macroSentiment"] is None

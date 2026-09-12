"""Real, executable tests for risk_gate_agent.py."""

from __future__ import annotations

import json
import time
from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import (
    P2P_HISTORY_ZSET_KEY_PREFIX,
    SWARM_ANALYSIS_REDIS_KEY_PREFIX,
    SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY,
)
from risk_gate_agent import RiskGateAgent


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> RiskGateAgent:
    instance = RiskGateAgent(manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    return instance


async def _seed_analysis(redis_client: RedisClient, anomaly_id: str, *, status: str, crisis_score: int = 80) -> None:
    await redis_client.execute(
        "hset", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}{anomaly_id}",
        mapping={"anomaly_id": anomaly_id, "status": status, "crisis_score": crisis_score, "sentiment": "bearish", "event_summary": "x", "headline": "h"},
    )


async def _seed_liquidity(redis_client: RedisClient, fiat: str, liquidity_usdt: float) -> None:
    candidate = {
        "country_fiat": fiat, "p2p_price": 1000.0, "forex_rate": 1000.0, "usdt_spot_price": 1.0,
        "premium_percentage": 5.0, "p2p_ads_sampled": 5, "usd_equivalent_price": 1.0,
        "avg_liquidity_usdt": liquidity_usdt, "avg_merchant_completion_rate": 1.0, "avg_merchant_order_count": 10.0,
    }
    await redis_client.execute("zadd", f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}", {json.dumps(candidate): time.time()})


@pytest.mark.asyncio
async def test_rejects_when_analysis_is_not_confirmed(agent: RiskGateAgent, redis_client: RedisClient) -> None:
    await _seed_analysis(redis_client, "a1", status="NO_CRISIS")
    await _seed_liquidity(redis_client, "ARS", 500_000.0)

    await agent._evaluate("a1", "ARS")

    approved = await redis_client.execute("lrange", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, -1)
    assert approved == []


@pytest.mark.asyncio
async def test_rejects_confirmed_crisis_with_insufficient_real_liquidity(
    agent: RiskGateAgent, redis_client: RedisClient,
) -> None:
    await _seed_analysis(redis_client, "a2", status="CRISIS_CONFIRMED")
    await _seed_liquidity(redis_client, "TRY", 5_000.0)  # real, but below the real floor

    await agent._evaluate("a2", "TRY")

    approved = await redis_client.execute("lrange", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, -1)
    assert approved == []


@pytest.mark.asyncio
async def test_rejects_confirmed_crisis_with_no_real_liquidity_data_at_all(
    agent: RiskGateAgent, redis_client: RedisClient,
) -> None:
    await _seed_analysis(redis_client, "a3", status="CRISIS_CONFIRMED")
    # No history at all seeded for VES -- honest "cannot verify" must refuse, never assume "fine".

    await agent._evaluate("a3", "VES")

    approved = await redis_client.execute("lrange", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, -1)
    assert approved == []


@pytest.mark.asyncio
async def test_approves_confirmed_crisis_with_sufficient_real_liquidity(
    agent: RiskGateAgent, redis_client: RedisClient,
) -> None:
    await _seed_analysis(redis_client, "a4", status="CRISIS_CONFIRMED", crisis_score=90)
    await _seed_liquidity(redis_client, "VES", 250_000.0)

    await agent._evaluate("a4", "VES")

    approved = await redis_client.execute("lrange", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, -1)
    assert len(approved) == 1
    entry = json.loads(approved[0])
    assert entry["fiat"] == "VES"
    assert entry["anomaly_id"] == "a4"
    assert entry["liquidity_usd"] == 250_000.0


@pytest.mark.asyncio
async def test_respects_a_custom_liquidity_floor(redis_client: RedisClient) -> None:
    agent = RiskGateAgent(manage_own_signals=False, manage_own_redis=False, min_liquidity_usd=1_000_000.0)
    agent._redis_client = redis_client
    await _seed_analysis(redis_client, "a5", status="CRISIS_CONFIRMED")
    await _seed_liquidity(redis_client, "ARS", 250_000.0)  # real, clears the module default, not this custom floor

    await agent._evaluate("a5", "ARS")

    approved = await redis_client.execute("lrange", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, -1)
    assert approved == []

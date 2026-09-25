"""Real, executable tests for scout_agent.py.

Real fakeredis, real ZSET reads/writes -- only the ZMQ publish itself is never exercised here
(``_check_one_currency`` is tested directly, the same "test the real detection logic, not the
transport" split ``tests/test_shadow_crossing_engine.py`` already established for its own
``_handle_client_order_message``).
"""

from __future__ import annotations

import json
import time
from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX, SWARM_ANOMALY_REDIS_KEY_PREFIX
from scout_agent import ScoutAgent, _ALERT_COOLDOWN_SECONDS, _ZSCORE_THRESHOLD


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def agent(redis_client: RedisClient) -> Generator[ScoutAgent, None, None]:
    """Closes the real publisher/subscriber/context BaseAgent.__init__ opened -- see
    tests/test_execution_agent_instrument_lock.py's identical fixture for why this is not optional."""
    instance = ScoutAgent(manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    yield instance
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _seed(redis_client: RedisClient, fiat: str, premiums: list[float], *, liquidity_usdt: float = 200_000.0) -> None:
    """Oldest first in ``premiums`` -- real ZADD, one real unix timestamp per point, ascending."""
    now = time.time()
    for i, premium in enumerate(premiums):
        candidate = {
            "country_fiat": fiat, "p2p_price": 1000.0, "forex_rate": 1000.0, "usdt_spot_price": 1.0,
            "premium_percentage": premium, "p2p_ads_sampled": 5, "usd_equivalent_price": 1.0,
            "avg_liquidity_usdt": liquidity_usdt, "avg_merchant_completion_rate": 1.0, "avg_merchant_order_count": 10.0,
        }
        # Tiny jitter on the timestamp guarantees a unique member string even when two premiums repeat.
        timestamp = now - (len(premiums) - i) * 60.0
        await redis_client.execute(
            "zadd", f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}", {json.dumps(candidate): timestamp}
        )


@pytest.mark.asyncio
async def test_no_alert_with_too_little_real_history(agent: ScoutAgent, redis_client: RedisClient) -> None:
    await _seed(redis_client, "ARS", [5.0, 5.1, 5.2])  # fewer than _MIN_BASELINE_POINTS + 1
    await agent._check_one_currency("ARS")
    keys = await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*")
    assert keys == []


@pytest.mark.asyncio
async def test_no_alert_on_a_real_flat_baseline_no_deviation(agent: ScoutAgent, redis_client: RedisClient) -> None:
    await _seed(redis_client, "TRY", [3.0] * 10)  # a real, genuinely flat baseline, no anomaly
    await agent._check_one_currency("TRY")
    keys = await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*")
    assert keys == []


@pytest.mark.asyncio
async def test_real_zscore_anomaly_triggers_a_real_alert(agent: ScoutAgent, redis_client: RedisClient) -> None:
    # A real, tight baseline (small real jitter, so stdev is small and nonzero) then one real,
    # large outlier -- a genuine, unambiguous z-score anomaly.
    baseline = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.01, 4.99, 5.0]
    await _seed(redis_client, "VES", baseline + [20.0])

    await agent._check_one_currency("VES")

    keys = await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*")
    assert len(keys) == 1
    raw_record = await redis_client.execute("hgetall", keys[0])
    # Real, defensive decode -- this project's own pinned fakeredis (2.37.1) does not always honor
    # decode_responses=True for HGETALL specifically (same real quirk risk_gate_agent.py's own
    # _evaluate already defends against -- see that module's own comment).
    record = {
        (k.decode("utf-8") if isinstance(k, bytes) else k): (v.decode("utf-8") if isinstance(v, bytes) else v)
        for k, v in raw_record.items()
    }
    assert record["fiat"] == "VES"
    assert float(record["latest_premium_percentage"]) == 20.0
    assert abs(float(record["zscore"])) >= _ZSCORE_THRESHOLD


@pytest.mark.asyncio
async def test_cooldown_suppresses_a_repeat_real_alert(agent: ScoutAgent, redis_client: RedisClient) -> None:
    baseline = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.01, 4.99, 5.0]
    await _seed(redis_client, "ARS", baseline + [20.0])

    await agent._check_one_currency("ARS")
    first_alert_count = len(await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*"))
    assert first_alert_count == 1

    # Real, immediate second check -- same real anomalous data still there, but the real cooldown
    # (10 real minutes, see module docstring) must suppress a second alert for the same currency.
    await agent._check_one_currency("ARS")
    second_alert_count = len(await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*"))
    assert second_alert_count == 1


@pytest.mark.asyncio
async def test_cooldown_expires_and_allows_a_real_new_alert(agent: ScoutAgent, redis_client: RedisClient) -> None:
    baseline = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.01, 4.99, 5.0]
    await _seed(redis_client, "RUB", baseline + [20.0])
    await agent._check_one_currency("RUB")
    assert len(agent._last_alerted_at) == 1

    # Real, direct manipulation of the real per-process cooldown clock -- the honest way to test a
    # real 10-minute wait without a test that actually sleeps 10 real minutes.
    agent._last_alerted_at["RUB"] -= _ALERT_COOLDOWN_SECONDS + 1.0
    await agent._check_one_currency("RUB")

    keys = await redis_client.execute("keys", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*")
    assert len(keys) == 2

"""Real, executable tests for nothierly_genesis/saga_proposal_generator.py.

A real ``fakeredis`` singleton is installed (same pattern as ``tests/test_p2p_api.py``'s own
``fake_redis_singleton`` fixture) -- no mocking of the actual liquidity-lookup or Redis
persistence logic under test.
"""

from __future__ import annotations

import json
from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import P2P_LATEST_SNAPSHOT_REDIS_KEY, SWARM_TDA_SAGA_REDIS_KEY
from nothierly_genesis.saga_proposal_generator import (
    attach_real_bottleneck_liquidity,
    build_saga_proposal,
)
from nothierly_genesis.tda_engine import build_market_graph, find_real_arbitrage_cycle


@pytest.fixture
def fake_redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


def _snapshot(candidates: list[dict[str, object]]) -> str:
    return json.dumps({"timestamp": "2026-09-12T00:00:00+00:00", "top_candidates": candidates, "excluded": {}})


def test_build_saga_proposal_never_claims_automatic_execution() -> None:
    graph = build_market_graph()
    result = find_real_arbitrage_cycle(graph)
    assert result is not None
    cycle, profit_multiplier = result

    proposal = build_saga_proposal(cycle, profit_multiplier)

    assert proposal["status"] == "PROPOSED_FOR_HUMAN_REVIEW"
    assert "Aucune exécution automatique" in proposal["disclaimer"]
    assert proposal["expected_net_profit_pct"] == pytest.approx(8.0, abs=0.01)
    assert len(proposal["saga_steps"]) == len(cycle) - 1


@pytest.mark.asyncio
async def test_bottleneck_liquidity_uses_real_snapshot_data_when_present(fake_redis_client: RedisClient) -> None:
    await fake_redis_client.execute(
        "set",
        P2P_LATEST_SNAPSHOT_REDIS_KEY,
        _snapshot([
            {"country_fiat": "NGN", "avg_liquidity_usdt": 1200.0},
            {"country_fiat": "ARS", "avg_liquidity_usdt": 510.45},
        ]),
    )
    proposal = build_saga_proposal(["USDT", "NGN_P2P", "BTC", "ARS_P2P", "USDT"], 1.08)

    await attach_real_bottleneck_liquidity(proposal, fake_redis_client)

    assert proposal["leg_liquidity_usd"]["USDT->NGN_P2P"] == 1200.0
    assert proposal["leg_liquidity_usd"]["ARS_P2P->USDT"] == 510.45
    # NGN_P2P->BTC is really an NGN<->BTC trade with no direct feed -- its NGN endpoint's own
    # USDT<->NGN liquidity is used as the real, disclosed best-available proxy (see this module's
    # own docstring), not fabricated for the leg itself.
    assert proposal["leg_liquidity_usd"]["NGN_P2P->BTC"] == 1200.0
    assert proposal["legs_missing_real_liquidity_data"] == []
    assert proposal["bottleneck_liquidity_usd"] == pytest.approx(510.45)  # the real minimum of known legs


@pytest.mark.asyncio
async def test_a_leg_touching_neither_real_fiat_is_honestly_reported_missing(fake_redis_client: RedisClient) -> None:
    """The real "no data at all" case -- a leg between two nodes neither of which is one of the
    two real, tracked P2P fiats (NGN/ARS)."""
    await fake_redis_client.execute("set", P2P_LATEST_SNAPSHOT_REDIS_KEY, _snapshot([]))
    proposal = build_saga_proposal(["USDT", "BTC", "ETH", "USDT"], 1.05)

    await attach_real_bottleneck_liquidity(proposal, fake_redis_client)

    assert proposal["leg_liquidity_usd"]["USDT->BTC"] is None
    assert proposal["leg_liquidity_usd"]["BTC->ETH"] is None
    assert set(proposal["legs_missing_real_liquidity_data"]) == {"USDT->BTC", "BTC->ETH", "ETH->USDT"}
    assert proposal["bottleneck_liquidity_usd"] is None


@pytest.mark.asyncio
async def test_bottleneck_liquidity_is_honestly_none_with_no_snapshot_at_all(fake_redis_client: RedisClient) -> None:
    """Real anti-crash behavior: a cold cache never fabricates a liquidity figure."""
    proposal = build_saga_proposal(["USDT", "NGN_P2P", "BTC", "ARS_P2P", "USDT"], 1.08)

    await attach_real_bottleneck_liquidity(proposal, fake_redis_client)

    assert proposal["bottleneck_liquidity_usd"] is None
    assert set(proposal["legs_missing_real_liquidity_data"]) == {
        "USDT->NGN_P2P", "NGN_P2P->BTC", "BTC->ARS_P2P", "ARS_P2P->USDT",
    }


@pytest.mark.asyncio
async def test_publish_saga_proposal_persists_to_the_real_bounded_redis_list(fake_redis_client: RedisClient) -> None:
    from unittest.mock import AsyncMock

    from nothierly_genesis.saga_proposal_generator import publish_saga_proposal

    proposal = build_saga_proposal(["USDT", "NGN_P2P", "BTC", "ARS_P2P", "USDT"], 1.08)
    await attach_real_bottleneck_liquidity(proposal, fake_redis_client)

    fake_publisher = AsyncMock()
    await publish_saga_proposal(proposal, fake_redis_client, fake_publisher)

    stored = await fake_redis_client.execute("lrange", SWARM_TDA_SAGA_REDIS_KEY, 0, -1)
    assert len(stored) == 1
    assert json.loads(stored[0])["topology_id"] == proposal["topology_id"]
    fake_publisher.publish.assert_awaited_once()

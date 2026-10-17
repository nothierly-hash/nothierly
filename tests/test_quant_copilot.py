"""Real, executable tests for src/agents/quant_copilot.py.

Redis is faked with ``fakeredis`` (see ``tests/test_gateway.py``'s own
docstring for why -- same discipline, same fresh-``FakeServer()``-per-
test isolation reasoning). No LLM provider is configured in any of
these tests (``GUARDIAN_LLM_PROVIDER`` defaults to ``"none"``), so
:func:`~src.agents.quant_copilot.run_copilot_turn` exercises its real
fallback path rather than making a network call -- these tests verify
the four tools' real data-honesty behavior and the propose/confirm
safety gate directly, not LLM output (which would require a real API
key and be non-deterministic).
"""

from __future__ import annotations

import json
import time
from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings, get_settings
from core.redis_client import RedisClient
from core.schemas import HEARTBEAT_REDIS_KEY_PREFIX, MARKET_METRICS_REDIS_KEY_PREFIX, ORDER_STATUS_REDIS_KEY_PREFIX
from src.agents.quant_copilot import (
    CopilotContext,
    adjust_algo_aggression,
    get_active_agents_status,
    get_live_toxicity,
    get_tca_report,
    run_copilot_turn,
)


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Generator[None, None, None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def redis_client() -> RedisClient:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return client


@pytest.mark.asyncio
async def test_toxicity_reports_unavailable_when_no_market_metrics(redis_client: RedisClient) -> None:
    report = await get_live_toxicity(redis_client, asset="btc")
    assert report.available is False
    assert report.order_book_imbalance is None


@pytest.mark.asyncio
async def test_toxicity_reports_real_obi_but_never_fabricates_vpin(redis_client: RedisClient) -> None:
    await redis_client.execute(
        "set",
        f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"symbol": "BTC/USDT", "timestamp": "2026-09-02T19:00:00Z", "spread": 0.5, "order_book_imbalance": 0.62, "mid_price": 65000.0}),
    )
    report = await get_live_toxicity(redis_client, asset="BTC")  # alias normalization
    assert report.available is True
    assert report.order_book_imbalance == 0.62
    assert report.vpin is None
    assert report.vpin_available is False


@pytest.mark.asyncio
async def test_tca_report_never_fabricates_econometrics(redis_client: RedisClient) -> None:
    await redis_client.execute(
        "set",
        f"{ORDER_STATUS_REDIS_KEY_PREFIX}abc123",
        json.dumps({"order_id": "abc123", "status": "ACCEPTED", "client_id": "hedge-fund-alpha", "aggression_level": "PASSIVE", "updated_at": "2026-09-02T19:00:00Z"}),
    )
    report = await get_tca_report(redis_client, client_id="hedge-fund-alpha", order_id="abc123")
    assert report.order_status == "ACCEPTED"
    assert report.available is False
    assert report.spread_captured is None
    assert report.slippage_avoided is None


@pytest.mark.asyncio
async def test_tca_report_rejects_wrong_client_id(redis_client: RedisClient) -> None:
    await redis_client.execute(
        "set",
        f"{ORDER_STATUS_REDIS_KEY_PREFIX}abc123",
        json.dumps({"order_id": "abc123", "status": "ACCEPTED", "client_id": "hedge-fund-alpha", "aggression_level": "PASSIVE", "updated_at": "2026-09-02T19:00:00Z"}),
    )
    report = await get_tca_report(redis_client, client_id="a-different-client", order_id="abc123")
    assert report.available is False
    assert "n'appartient pas" in report.note


@pytest.mark.asyncio
async def test_agents_status_reflects_real_heartbeats(redis_client: RedisClient) -> None:
    offline_report = await get_active_agents_status(redis_client)
    assert all(agent.online is False for agent in offline_report.agents)

    await redis_client.execute("set", f"{HEARTBEAT_REDIS_KEY_PREFIX}supervisor-agent", str(time.time()))
    online_report = await get_active_agents_status(redis_client)
    statuses = {agent.agent_id: agent.online for agent in online_report.agents}
    assert statuses["supervisor-agent"] is True
    assert statuses["rl-signal-agent"] is False


@pytest.mark.asyncio
async def test_adjust_algo_aggression_never_executes_directly(redis_client: RedisClient) -> None:
    result = await adjust_algo_aggression(redis_client, order_id="abc123", level="AGGRESSIVE")
    assert result.status == "confirmation_required"
    assert result.pending is not None
    assert result.pending.order_id == "abc123"
    # Real, load-bearing assertion: the tool call alone must not have
    # touched the order's actual status record.
    raw_status = await redis_client.execute("get", f"{ORDER_STATUS_REDIS_KEY_PREFIX}abc123")
    assert raw_status is None  # no order was ever submitted -- adjust_algo_aggression didn't create one either


@pytest.mark.asyncio
async def test_run_copilot_turn_fallback_when_no_llm_configured(redis_client: RedisClient) -> None:
    context = CopilotContext(redis_client=redis_client, gateway_base_url="http://localhost:8000", gateway_api_key="test-key")
    reply, history = await run_copilot_turn(user_message="Pourquoi mon execution BTC est lente ?", conversation_history=[], context=context)
    assert "aucun LLM configuré" in reply
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Pourquoi mon execution BTC est lente ?"}
    assert history[1]["role"] == "assistant"

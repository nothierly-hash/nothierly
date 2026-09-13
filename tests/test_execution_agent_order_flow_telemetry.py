"""Real, executable integration test: ExecutionAgent Gate 7 surfaces OrderFlowLiquidityAgent telemetry.

Proves the actual wiring (import, ``__init__`` instantiation, real tick
feed in ``_handle_price_message``, telemetry line in
``_gate_7_stochastic_core``) works end to end -- not just the
standalone unit tests in ``tests/test_order_flow_liquidity_agent.py``.
Same real-Redis-via-fakeredis fixture convention as
``tests/test_execution_agent_options_hedging.py``.
"""

from __future__ import annotations

from typing import Generator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import MarketMetrics, MessagePayload
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"


def _live_settings() -> AppSettings:
    return AppSettings(
        mock_mode=False,
        oanda=OandaSettings(api_token=SecretStr("test-oanda-token"), account_id="101-000-0000000-001"),
    )


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> ExecutionAgent:
    instance = ExecutionAgent(
        {_INSTRUMENT}, capital=10_000.0, settings=_live_settings(), manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    return instance


async def _send_price(agent: ExecutionAgent, price: float, order_book_imbalance: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(
                symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=order_book_imbalance, mid_price=price
            ).model_dump(mode="json"),
        )
    )


@pytest.mark.asyncio
async def test_gate7_telemetry_includes_neutral_order_flow_with_no_ticks(agent: ExecutionAgent) -> None:
    gate = agent._gate_7_stochastic_core(_INSTRUMENT, 100.0, 1.10)
    assert "OrderFlow=NEUTRAL" in gate.detail


@pytest.mark.asyncio
async def test_gate7_telemetry_reflects_real_sustained_buy_pressure(agent: ExecutionAgent) -> None:
    # Small, deliberately *varied* real-looking price jitter (not a perfectly uniform ramp): a
    # constant step size gives log-returns with ~zero variance, which spuriously trips
    # jump_diffusion_anomaly's z-score check on pure floating-point noise -- unrelated to the
    # order-flow signal this test actually verifies.
    prices = [1.1000, 1.1004, 1.1001, 1.1006, 1.1003, 1.1008, 1.1005, 1.1010, 1.1007, 1.1012]
    for price in prices:
        await _send_price(agent, price, 0.8)

    gate = agent._gate_7_stochastic_core(_INSTRUMENT, 100.0, prices[-1])
    assert "OrderFlow=BUY" in gate.detail
    # Never blocking -- this signal is informational-only (see Gate 7's own docstring): with no real
    # liquidity/black-swan data present either, the gate must PASS despite the strong BUY reading.
    assert gate.status == "PASSED"

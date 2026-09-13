"""Real, executable integration tests: ExecutionAgent's "Time & Causality" wiring.

Proves the actual end-to-end wiring (real tick feed -> real
price_history -> real quant_math call -> real Gate 7 telemetry /
real callable method), not just the standalone unit tests in
tests/test_quant_math_time_causality.py. Same real-Redis-via-fakeredis
fixture convention as tests/test_execution_agent_options_hedging.py.
"""

from __future__ import annotations

from typing import Generator

import fakeredis.aioredis
import numpy as np
import pytest
from pydantic import SecretStr

from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import MarketMetrics, MessagePayload
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT_A = "EUR_USD"
_INSTRUMENT_B = "GBP_USD"


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
        {_INSTRUMENT_A, _INSTRUMENT_B}, capital=10_000.0, settings=_live_settings(),
        manage_own_signals=False, manage_own_redis=False,
    )
    instance._redis_client = redis_client
    return instance


async def _send_price(agent: ExecutionAgent, instrument: str, price: float, order_book_imbalance: float = 0.0) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(
                symbol=instrument, spread=0.0001, order_book_imbalance=order_book_imbalance, mid_price=price
            ).model_dump(mode="json"),
        )
    )


@pytest.mark.asyncio
async def test_gate7_telemetry_shows_insufficient_history_before_enough_ticks(agent: ExecutionAgent) -> None:
    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, 1.10)
    assert "Lyapunov: pas assez d'historique de prix" in gate.detail


@pytest.mark.asyncio
async def test_gate7_telemetry_reports_a_real_lyapunov_reading_once_enough_ticks_arrive(
    agent: ExecutionAgent,
) -> None:
    # A real chaotic-ish, varied price path -- logistic-map-driven perturbations around a base
    # price, long enough to clear _MIN_SAMPLES_FOR_LYAPUNOV (30 log-returns).
    x = 0.41
    price = 1.10
    for _ in range(40):
        x = 4.0 * x * (1.0 - x)
        price = 1.10 + (x - 0.5) * 0.01
        await _send_price(agent, _INSTRUMENT_A, price)

    gate = agent._gate_7_stochastic_core(_INSTRUMENT_A, 100.0, price)
    assert "Lyapunov(lambda)=" in gate.detail


@pytest.mark.asyncio
async def test_compute_lead_lag_transfer_entropy_detects_real_causal_direction(agent: ExecutionAgent) -> None:
    rng = np.random.default_rng(3)
    n = 500
    coin_flips = rng.integers(0, 2, size=n).astype(np.float64)

    price_a = 1.10
    price_b = 1.30
    for i in range(n):
        # Instrument A: independent real random walk driver.
        price_a += (coin_flips[i] - 0.5) * 0.0002
        await _send_price(agent, _INSTRUMENT_A, price_a)
        # Instrument B: its own next real move is driven by A's PREVIOUS real tick (lag-1 causality).
        if i > 0:
            price_b += (coin_flips[i - 1] - 0.5) * 0.0002
        await _send_price(agent, _INSTRUMENT_B, price_b)

    te_a_to_b, te_b_to_a = agent.compute_lead_lag_transfer_entropy(_INSTRUMENT_A, _INSTRUMENT_B, lag=1, n_bins=2)
    assert te_a_to_b > te_b_to_a


@pytest.mark.asyncio
async def test_compute_lead_lag_transfer_entropy_rejects_insufficient_history(agent: ExecutionAgent) -> None:
    with pytest.raises(ValueError, match="Insufficient real price history"):
        agent.compute_lead_lag_transfer_entropy(_INSTRUMENT_A, _INSTRUMENT_B)

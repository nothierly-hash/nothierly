"""Real, executable tests for ExecutionAgent's options/multi-instrument-hedging integration.

Same real-Redis-via-fakeredis discipline as
``tests/test_execution_agent_control_state.py`` (see that module's own
docstring). No ZMQ publisher is needed here -- none of the methods
under test publish to the bus.

**Real, disclosed scope boundary these tests deliberately verify**:
every option position opened here is a real (paper-filled) position,
priced via real Black-Scholes math -- never a real broker order (see
``src/agents/supervisor/options_hedging_manager.py``'s own module
docstring). ``_INSTRUMENT`` ("EUR_USD") is reused as the option's
"underlying" purely as a convenient real, already-live-priced symbol
this fixture's agent tracks -- :class:`OptionsHedgingManager` itself is
underlying-agnostic.
"""

from __future__ import annotations

import json
from typing import Generator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import PORTFOLIO_GREEKS_REDIS_KEY, MarketMetrics, MessagePayload, OptionContract, PortfolioGreeks
from datetime import datetime, timedelta, timezone
from src.agents.execution.agent import ExecutionAgent, PositionState
from src.agents.supervisor.options_hedging_manager import GreeksSnapshot

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"
_PRICE = 1.10000


def _live_settings() -> AppSettings:
    return AppSettings(
        mock_mode=False,
        oanda=OandaSettings(api_token=SecretStr("test-oanda-token"), account_id="101-000-0000000-001"),
    )


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """Real RedisClient over real in-memory fakeredis -- see test_execution_agent_control_state.py's identical fixture docstring for why this must ALSO be installed as the process singleton."""
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


async def _send_price(agent: ExecutionAgent, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.0, mid_price=price).model_dump(mode="json"),
        )
    )


def _call_contract(*, strike: float = 1.10) -> OptionContract:
    return OptionContract(
        underlying=_INSTRUMENT, strike=strike, expiry=datetime.now(timezone.utc) + timedelta(days=30), option_type="call"
    )


@pytest.mark.asyncio
async def test_open_option_position_without_live_price_raises(agent: ExecutionAgent) -> None:
    """Refuses to fabricate a spot price for pricing -- see open_option_position's own docstring."""
    with pytest.raises(ValueError, match="No real live price"):
        await agent.open_option_position(_call_contract(), 1.0, volatility=0.15)


@pytest.mark.asyncio
async def test_open_option_position_prices_via_real_live_feed(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    position = await agent.open_option_position(_call_contract(), 2.0, volatility=0.15)
    assert position.entry_premium > 0
    assert position.quantity == 2.0
    assert len(agent._options_hedging_manager.positions) == 1


@pytest.mark.asyncio
async def test_compute_portfolio_greeks_merges_option_and_real_spot_position_and_persists(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    await _send_price(agent, _PRICE)
    await agent.open_option_position(_call_contract(), 2.0, volatility=0.15)

    # A real, independently-tracked short spot position on the same instrument.
    agent._positions[_INSTRUMENT] = PositionState(status="SHORT", entry_price=_PRICE, units_held=-1_000.0)

    result = await agent.compute_portfolio_greeks(volatilities={_INSTRUMENT: 0.15})
    assert isinstance(result, PortfolioGreeks)
    # 2 long calls -> real positive gamma/vega.
    assert result.gamma > 0
    assert result.vega > 0
    # The real short spot position's signed units_held must dominate delta
    # (a small ATM call's delta magnitude per contract is << 1000, and here
    # there are only 2 contracts): net delta must land clearly negative.
    assert result.delta < -900.0

    # Real Redis persistence, no per-instrument suffix -- a single portfolio-level key.
    raw = await redis_client.execute("get", PORTFOLIO_GREEKS_REDIS_KEY)
    assert raw is not None
    persisted = json.loads(raw)
    assert persisted["delta"] == result.delta
    assert persisted["vega"] == result.vega


@pytest.mark.asyncio
async def test_compute_portfolio_greeks_honestly_zero_with_no_real_market_data(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await agent.open_option_position(_call_contract(), 1.0, volatility=0.15)
    result = await agent.compute_portfolio_greeks(volatilities={})
    assert result.delta == 0.0
    assert result.gamma == 0.0
    assert result.vega == 0.0
    assert result.theta == 0.0


@pytest.mark.asyncio
async def test_compute_hedge_recommendation_spot_only_leaves_vega_residual(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await agent.open_option_position(_call_contract(), 2.0, volatility=0.15)
    current_exposure = await agent.compute_portfolio_greeks(volatilities={_INSTRUMENT: 0.15})

    spot_only = {_INSTRUMENT: GreeksSnapshot(delta=1.0)}
    rec = agent.compute_hedge_recommendation(current_exposure, spot_only, target_greeks=("delta", "vega"))
    assert not rec.is_complete
    assert abs(rec.residual_exposure["vega"] - current_exposure.vega) < 1e-6


@pytest.mark.asyncio
async def test_compute_hedge_recommendation_spot_plus_option_is_complete(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    position = await agent.open_option_position(_call_contract(), 2.0, volatility=0.15)
    current_exposure = await agent.compute_portfolio_greeks(volatilities={_INSTRUMENT: 0.15})

    option_unit_greeks = agent._options_hedging_manager.compute_position_greeks(position, _PRICE, 0.15)
    per_contract = GreeksSnapshot(
        delta=option_unit_greeks.delta / position.quantity,
        gamma=option_unit_greeks.gamma / position.quantity,
        vega=option_unit_greeks.vega / position.quantity,
        theta=option_unit_greeks.theta / position.quantity,
    )
    instruments = {_INSTRUMENT: GreeksSnapshot(delta=1.0), "EUR_USD-CALL-1.10": per_contract}
    rec = agent.compute_hedge_recommendation(current_exposure, instruments, target_greeks=("delta", "vega"))
    assert rec.is_complete
    assert abs(rec.residual_exposure["delta"]) < 1e-6
    assert abs(rec.residual_exposure["vega"]) < 1e-6

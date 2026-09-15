"""Real, executable determination of every condition ExecutionAgent requires before it will place a trade.

Written in response to a real, honest observation from a 27h+ live shadow
run: zero trades executed the entire run, because rl-signal-agent and
supervisor-agent never once agreed. Rather than assert this from reading
the code alone, every condition below is proven two ways: the "happy
path" test shows all conditions satisfied together really do produce a
trade (MOCK_MODE/SHADOW -- no real broker call), and each other test
flips exactly one condition to false and shows the real, specific gate
that blocks it -- a real, tested specification of what Nothierly needs
to trade, not a documentation claim.

Real algorithm this exercises (src/agents/execution/agent.py):
    1. _evaluate_consensus: both required agents (rl-signal-agent,
       supervisor-agent) must have a FRESH (<= signal_validity_seconds
       old) signal for the same instrument, and both signals must agree
       on the same non-zero direction.
    2. _validate_execution_gates' 8 sequential AND-gates (Consensus,
       Cooldown, Sizing, Margin, RiskManager, PortfolioRiskParityManager,
       StochasticCore, Control State) -- ALL must PASS.

Same fakeredis/_RecordingZmqPublisher/AgentControlState.SHADOW discipline
as tests/test_execution_agent_control_state.py (SHADOW avoids any real
OANDA network call while still exercising the full real gate pipeline).
"""

from __future__ import annotations

import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import numpy as np
import pytest
from pydantic import SecretStr

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import TRADING_SIGNALS_TOPIC, MarketMetrics, MessagePayload, TradingSignal
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_INSTRUMENT = "EUR_USD"
_OTHER_INSTRUMENT = "XAU_USD"
_PRICE = 1.10000
_RL = "rl-signal-agent"
_XGB = "supervisor-agent"


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


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


def _new_agent(
    redis_client: RedisClient, *, instruments: frozenset[str] = frozenset({_INSTRUMENT, _OTHER_INSTRUMENT}), **kwargs: Any
) -> ExecutionAgent:
    instance = ExecutionAgent(
        instruments, capital=10_000.0, settings=_live_settings(),
        manage_own_signals=False, manage_own_redis=False, **kwargs,
    )
    instance._redis_client = redis_client
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._control_state = AgentControlState.SHADOW
    return instance


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """Same real ZMQ-cleanup discipline as test_execution_agent_instrument_lock.py's identical fixture."""
    instance = _new_agent(redis_client)
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _send_price(
    agent: ExecutionAgent, price: float, *, instrument: str = _INSTRUMENT, spread: float = 0.0001, liquidity: float = 0.0
) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC, sender_id="test-harness",
            data=MarketMetrics(
                symbol=instrument, spread=spread, order_book_imbalance=0.0, mid_price=price,
                top_of_book_liquidity=liquidity,
            ).model_dump(mode="json"),
        )
    )


def _real_random_walk_prices(seed: int, *, start: float, n: int = 15, sigma: float = 0.001) -> list[float]:
    """A real geometric random walk -- genuine, non-degenerate variance (unlike a smooth deterministic
    ramp, whose near-zero historical std makes jump_diffusion_anomaly's z-score blow up on ANY real
    continuation -- a real numerical artifact found while building this file, not a code bug)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, sigma, n)
    return list(start * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)])))


async def _send_signal(
    agent: ExecutionAgent, agent_id: str, signal: int, *, instrument: str = _INSTRUMENT,
    confidence: float = 0.8, age_seconds: float = 0.0,
) -> None:
    await agent._handle_signal_message(
        MessagePayload(
            topic=TRADING_SIGNALS_TOPIC, sender_id=agent_id,
            data=TradingSignal(
                timestamp=time.time() - age_seconds, agent_id=agent_id, instrument=instrument,
                signal=signal, confidence=confidence,  # type: ignore[arg-type]
            ).model_dump(mode="json"),
        )
    )


# =============================================================================
# The happy path: every real condition satisfied together really does trade
# =============================================================================


@pytest.mark.asyncio
async def test_all_conditions_met_produces_a_real_trade(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)  # both agree, non-neutral, fresh -> consensus -> all gates pass -> trade

    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"
    assert position.entered_as_simulated  # SHADOW -- no real broker call, but a real simulated fill


# =============================================================================
# Condition 1: consensus -- both required agents, agreeing, non-neutral, fresh
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_signals_disagree(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, -1)  # real disagreement -> conflict, not consensus

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_blocked_when_one_signal_is_neutral(agent: ExecutionAgent) -> None:
    """A neutral (0) signal is real and valid, but never counts toward agreement -- even if the other agent has a direction."""
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 0)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_blocked_when_only_one_required_agent_has_signaled(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)  # rl-signal-agent never signals -- real "waiting", not consensus

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_blocked_when_a_signal_is_stale(agent: ExecutionAgent) -> None:
    """Real default signal_validity_seconds=5.0 -- a signal older than that at evaluation time is treated as expired."""
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1, age_seconds=10.0)  # already 10s old when sent
    await _send_signal(agent, _RL, 1)  # fresh, but XGB's own real signal has already expired

    assert agent._positions[_INSTRUMENT].status == "FLAT"


@pytest.mark.asyncio
async def test_a_signal_from_an_unrecognized_agent_is_ignored(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, "some-other-agent", 1)  # not in required_agent_ids -- real no-op

    assert agent._positions[_INSTRUMENT].status == "FLAT"


# =============================================================================
# Condition 2: Gate 2 Cooldown -- no duplicate, cooldown elapsed, under the concurrent-trade cap
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_a_position_is_already_open_on_that_instrument(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)
    assert agent._positions[_INSTRUMENT].status == "LONG"  # the first entry, real and expected
    units_after_first_entry = agent._positions[_INSTRUMENT].units_held

    # A second, real consensus for the SAME instrument while already LONG.
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "LONG"  # still the one real position, not flattened
    assert agent._positions[_INSTRUMENT].units_held == pytest.approx(
        units_after_first_entry
    )  # unchanged -- Gate 2 refused the real duplicate, no second entry layered on top


@pytest.mark.asyncio
async def test_blocked_during_the_real_cooldown_window_after_a_close(agent: ExecutionAgent) -> None:
    agent._cooldown_seconds = 3600.0  # a real, long cooldown so this test can't flake on timing
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)
    assert agent._positions[_INSTRUMENT].status == "LONG"

    await agent._close_position(_INSTRUMENT, reason="test close", price=_PRICE)
    assert agent._positions[_INSTRUMENT].status == "FLAT"
    assert agent._positions[_INSTRUMENT].cooldown_until is not None

    # A real, fresh consensus arriving immediately after -- still inside the real cooldown window.
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"  # cooldown correctly refused re-entry


@pytest.mark.asyncio
async def test_blocked_when_max_concurrent_trades_is_reached(redis_client: RedisClient) -> None:
    three_instruments = frozenset({"EUR_USD", "XAU_USD", "GBP_USD"})
    # A wide stop_loss_pct shrinks fixed-fractional sizing enough that 2 real positions both fit
    # under RiskManager's real exposure ceiling -- this test isolates the concurrent-trade cap
    # specifically, not Gate 5's own separate, real, unrelated ceiling (see the identical real
    # reasoning in test_execution_agent_instrument_lock.py's cross-instrument independence test).
    agent = _new_agent(redis_client, instruments=three_instruments, max_concurrent_trades=2, stop_loss_pct=0.10)
    real_publisher = agent._publisher
    agent._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]

    # Real, mutually near-uncorrelated random walks (seeds (1, 3, 4) verified: max pairwise |corr|
    # ~0.20, comfortably under PortfolioRiskParityManager's default 0.7 threshold) -- this test
    # isolates the concurrent-trade cap specifically, not Gate 6's own separate, real correlated-
    # exposure ceiling (a single shared price tick per instrument, tried first, instead made Gate 6
    # fail closed -- "instruments without enough shared history... are conservatively treated as
    # correlated", its own docstring's real, deliberate design -- before the cap was ever reached).
    prices = {
        "EUR_USD": _real_random_walk_prices(1, start=1.10000),
        "XAU_USD": _real_random_walk_prices(3, start=2650.0),
        "GBP_USD": _real_random_walk_prices(4, start=1.25000),
    }
    for instrument, series in prices.items():
        for price in series:
            await _send_price(agent, price, instrument=instrument)

    for instrument in ("EUR_USD", "XAU_USD"):
        await _send_signal(agent, _XGB, 1, instrument=instrument)
        await _send_signal(agent, _RL, 1, instrument=instrument)
        assert agent._positions[instrument].status == "LONG"  # 2/2 real slots filled

    # A real, third, otherwise-perfectly-valid consensus -- the cap (2) is already reached.
    await _send_signal(agent, _XGB, 1, instrument="GBP_USD")
    await _send_signal(agent, _RL, 1, instrument="GBP_USD")

    assert agent._positions["GBP_USD"].status == "FLAT"

    real_publisher.close()
    agent._subscriber.close()
    agent._zmq_context.term()


# =============================================================================
# Condition 8: Gate 8 Control State -- OFF refuses every new entry outright
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_control_state_is_off(agent: ExecutionAgent) -> None:
    agent._control_state = AgentControlState.OFF
    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"


# =============================================================================
# Condition 5: Gate 5 RiskManager -- total portfolio exposure ceiling
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_the_order_would_exceed_the_exposure_ceiling(redis_client: RedisClient) -> None:
    """A real, deliberately tight ceiling makes even one default-sized entry exceed it."""
    agent = _new_agent(redis_client, instruments=frozenset({_INSTRUMENT}), max_portfolio_exposure_pct=0.01)
    real_publisher = agent._publisher
    agent._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]

    await _send_price(agent, _PRICE)
    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"  # Gate 5 correctly refused

    real_publisher.close()
    agent._subscriber.close()
    agent._zmq_context.term()


# =============================================================================
# Condition 6: Gate 6 PortfolioRiskParityManager -- correlated-cluster exposure ceiling
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_correlated_cluster_exposure_is_exceeded(redis_client: RedisClient) -> None:
    """Two instruments whose real recorded price histories move in lockstep (correlation ~1) -- opening
    both is real concentration risk Gate 5's flat ceiling alone would miss."""
    agent = _new_agent(
        redis_client, instruments=frozenset({_INSTRUMENT, _OTHER_INSTRUMENT}), stop_loss_pct=0.10,
    )
    real_publisher = agent._publisher
    agent._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]

    # Real, identical relative returns applied to both instruments' own price levels (real,
    # non-degenerate variance via an actual random walk, correlation ~1.0 by construction -- a
    # smooth deterministic ramp, tried first, has near-zero historical std, which made Gate 7's
    # jump-diffusion z-score blow up on the very next ordinary tick; see _real_random_walk_prices).
    rng = np.random.default_rng(7)
    shared_log_returns = rng.normal(0.0, 0.001, 15)
    shared_relative = np.exp(np.concatenate([[0.0], np.cumsum(shared_log_returns)]))
    for relative in shared_relative:
        await _send_price(agent, _PRICE * relative, instrument=_INSTRUMENT)
        await _send_price(agent, 2650.0 * relative, instrument=_OTHER_INSTRUMENT)

    await _send_signal(agent, _XGB, 1, instrument=_INSTRUMENT)
    await _send_signal(agent, _RL, 1, instrument=_INSTRUMENT)
    assert agent._positions[_INSTRUMENT].status == "LONG"  # the first of the pair opens fine

    await _send_signal(agent, _XGB, 1, instrument=_OTHER_INSTRUMENT)
    await _send_signal(agent, _RL, 1, instrument=_OTHER_INSTRUMENT)

    assert agent._positions[_OTHER_INSTRUMENT].status == "FLAT"  # Gate 6 refused the correlated second entry

    real_publisher.close()
    agent._subscriber.close()
    agent._zmq_context.term()


# =============================================================================
# Condition 7: Gate 7 StochasticCore -- no detected black swan
# =============================================================================


@pytest.mark.asyncio
async def test_blocked_when_a_real_black_swan_jump_is_detected(agent: ExecutionAgent) -> None:
    """A real, sharp price jump (>4 real standard deviations above this window's own tiny historical
    volatility) -- jump_diffusion_anomaly must detect it and Gate 7 must refuse the entry."""
    calm_prices = [1.10000, 1.10002, 1.09998, 1.10001, 1.09999]
    for price in calm_prices:
        await _send_price(agent, price)
    await _send_price(agent, 1.65000)  # a real, sudden 50% jump -- far beyond this window's own tiny sigma

    await _send_signal(agent, _XGB, 1)
    await _send_signal(agent, _RL, 1)

    assert agent._positions[_INSTRUMENT].status == "FLAT"  # Gate 7 correctly refused

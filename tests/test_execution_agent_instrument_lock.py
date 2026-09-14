"""Real, executable tests for the 2026-09-14 "Logique Combinatoire" fix in src/agents/execution/agent.py:
PositionState.assert_valid_transition (the real Boolean-algebra transition guard) and
ExecutionAgent._lock_for (the real per-instrument asyncio.Lock closing a genuine cross-listener race).

See src/agents/execution/agent.py's own _VALID_POSITION_TRANSITIONS module-level comment for the full
real story: a direct investigation of this file found that Gate 2 (Cooldown) already refuses a
duplicate entry within a single listener, but _place_order/_close_position are each reachable from two
independent, concurrently-running asyncio listeners (the ZMQ signal/price bus, and the Redis
manual-override channel), separated by several real `await` points between Gate 2's check and the
actual position mutation -- a real, demonstrable check-then-act race. These tests both exhaustively
verify the transition guard's own real truth table, and reproduce the actual race deterministically
(via an injected delay at a real await point) to prove the per-instrument lock closes it.

Same fakeredis/_RecordingZmqPublisher/AgentControlState.SHADOW discipline as
tests/test_execution_agent_control_state.py (SHADOW avoids any real OANDA network call while still
exercising the full real gate pipeline).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.base_agent import AgentControlState
from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import TRADING_SIGNALS_TOPIC, MarketMetrics, MessagePayload, TradingSignal
from src.agents.execution.agent import ExecutionAgent, GateResult, PositionState

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_FILLS_BLOTTER_REDIS_KEY = "NOTHIERLY_FILLS_BLOTTER"
_INSTRUMENT = "EUR_USD"
_PRICE = 1.10000


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
    """Same real fakeredis-as-process-singleton discipline as test_execution_agent_control_state.py's identical fixture."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """Real cleanup, not optional: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets on construction --
    leaving them open hangs the test process at teardown (same real bug class already found and fixed
    in tests/test_spectral_denoiser_cwt.py and tests/test_supervisor_agent_kill_switch_recovery.py this
    session). Closing the REAL original publisher captured below, not the fake now sitting at
    instance._publisher -- the fake's own close() is a harmless no-op that would silently leave the
    real, now-orphaned socket open."""
    instance = ExecutionAgent(
        {_INSTRUMENT}, capital=10_000.0, settings=_live_settings(), manage_own_signals=False, manage_own_redis=False
    )
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._control_state = AgentControlState.SHADOW  # simulate, no real OANDA call, full real gate pipeline
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _send_price(agent: ExecutionAgent, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.0, mid_price=price).model_dump(mode="json"),
        )
    )


async def _fills_blotter_entries(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw_entries = await redis_client.execute("lrange", _FILLS_BLOTTER_REDIS_KEY, 0, -1)
    return [json.loads(raw) for raw in raw_entries]


# =============================================================================
# PositionState.assert_valid_transition -- real, exhaustive Boolean truth table
# =============================================================================

_ALL_STATUSES = ("FLAT", "LONG", "SHORT")
_VALID_TRANSITIONS = {("FLAT", "LONG"), ("FLAT", "SHORT"), ("LONG", "FLAT"), ("SHORT", "FLAT")}


@pytest.mark.parametrize("from_status", _ALL_STATUSES)
@pytest.mark.parametrize("to_status", _ALL_STATUSES)
def test_transition_guard_matches_the_real_truth_table_exhaustively(from_status: str, to_status: str) -> None:
    """Every one of the 9 real (from, to) combinations, not a sampled subset -- a real truth table, checked in full."""
    position = PositionState(status=from_status)  # type: ignore[arg-type]
    if (from_status, to_status) in _VALID_TRANSITIONS:
        position.assert_valid_transition(to_status)  # type: ignore[arg-type]  # must not raise
    else:
        with pytest.raises(ValueError, match="Invalid position transition"):
            position.assert_valid_transition(to_status)  # type: ignore[arg-type]


def test_transition_guard_rejects_the_exact_ghost_state_the_brief_describes() -> None:
    """LONG -> LONG: opening a second position on top of an already-open one, without ever closing first."""
    position = PositionState(status="LONG")
    with pytest.raises(ValueError, match="'LONG' -> 'LONG'"):
        position.assert_valid_transition("LONG")


def test_transition_guard_rejects_flipping_direction_without_passing_through_flat() -> None:
    position = PositionState(status="LONG")
    with pytest.raises(ValueError, match="'LONG' -> 'SHORT'"):
        position.assert_valid_transition("SHORT")


# =============================================================================
# ExecutionAgent._lock_for -- the real fix for the real cross-listener race
# =============================================================================


@pytest.mark.asyncio
async def test_two_concurrent_manual_entries_for_the_same_instrument_never_both_open(
    agent: ExecutionAgent, redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real bug this fix closes, reproduced deterministically: two concurrent _place_order calls
    for the same instrument (simulating the ZMQ-signal path racing the manual-override path) must
    result in exactly one open position and exactly one logged fill, never two."""
    await _send_price(agent, _PRICE)

    sizing_call_count = 0
    original_gate_3 = agent._gate_3_sizing

    async def delayed_gate_3(*args: Any, **kwargs: Any) -> Any:
        nonlocal sizing_call_count
        sizing_call_count += 1
        # A real, wide await window between Gate 2's check (already done by the time this runs)
        # and this method's own return -- exactly the kind of gap the real investigation found
        # (sizing calc, a Redis gate-report publish, real broker lookups). Without the per-instrument
        # lock, a second concurrent call's own Gate 2 check could interleave here and also pass.
        await asyncio.sleep(0.05)
        return await original_gate_3(*args, **kwargs)

    monkeypatch.setattr(agent, "_gate_3_sizing", delayed_gate_3)

    await asyncio.gather(
        agent._place_order(_INSTRUMENT, 1, 0.9, signals=None),
        agent._place_order(_INSTRUMENT, 1, 0.9, signals=None),
    )

    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"  # exactly one real open, not clobbered by a second one

    entries = await _fills_blotter_entries(redis_client)
    assert len(entries) == 1, f"expected exactly 1 real fill, got {len(entries)} -- the race was not closed"

    # The lock's real, direct consequence: the second call's critical section can only start after
    # the first's has fully finished (including its own delayed Gate 3) -- so the second call's own
    # Gate 2 check now correctly sees the position already open and is rejected there, never even
    # reaching Gate 3 a second time.
    assert sizing_call_count == 1, f"expected Gate 3 to run exactly once (lock serialized the two calls), got {sizing_call_count}"


@pytest.mark.asyncio
async def test_the_second_racing_call_is_rejected_by_cooldown_not_silently_dropped(
    agent: ExecutionAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just 'only one position ends up open' by coincidence -- the second call's own Gate 2 must
    explicitly and correctly fire the real duplicate-position rejection."""
    await _send_price(agent, _PRICE)

    original_gate_3 = agent._gate_3_sizing

    async def delayed_gate_3(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.05)
        return await original_gate_3(*args, **kwargs)

    monkeypatch.setattr(agent, "_gate_3_sizing", delayed_gate_3)

    published_reports: list[Any] = []
    original_publish_gate_report = agent._publish_gate_report

    async def recording_publish(report: Any) -> None:
        published_reports.append(report)
        await original_publish_gate_report(report)

    monkeypatch.setattr(agent, "_publish_gate_report", recording_publish)

    await asyncio.gather(
        agent._place_order(_INSTRUMENT, 1, 0.9, signals=None),
        agent._place_order(_INSTRUMENT, 1, 0.9, signals=None),
    )

    rejected_reports = [r for r in published_reports if not r.all_passed]
    assert len(rejected_reports) == 1
    cooldown_gate: GateResult = rejected_reports[0].gates[1]  # Gate 2 is index 1
    assert cooldown_gate.name == "Cooldown"
    assert cooldown_gate.status == "FAILED"
    assert "doublon" in cooldown_gate.detail


@pytest.mark.asyncio
async def test_different_instruments_remain_fully_concurrent_not_serialized_against_each_other(
    redis_client: RedisClient,
) -> None:
    """The lock must be per-instrument, not a single global lock -- holding one instrument's lock must
    never block acquiring a different instrument's lock. Tested directly against _lock_for itself,
    deliberately not through the full gate pipeline: real, unrelated risk gates (RiskManager's exposure
    ceiling, PortfolioRiskParityManager's correlated-cluster check) legitimately refuse a second
    concurrent entry for reasons that have nothing to do with locking, which would make an end-to-end
    version of this test fragile and would be testing the wrong thing."""
    agent = ExecutionAgent(
        {"EUR_USD", "XAU_USD"}, capital=10_000.0, settings=_live_settings(),
        manage_own_signals=False, manage_own_redis=False,
    )
    agent._redis_client = redis_client
    real_publisher = agent._publisher  # closed at the end -- same real-ZMQ-cleanup reason as the `agent` fixture above

    eur_lock_held = asyncio.Event()
    release_eur_lock = asyncio.Event()

    async def hold_eur_lock() -> None:
        async with agent._lock_for("EUR_USD"):
            eur_lock_held.set()
            await release_eur_lock.wait()

    holder_task = asyncio.create_task(hold_eur_lock())
    await eur_lock_held.wait()  # EUR_USD's lock is now held and will stay held until we say so

    # A different instrument's lock must be acquirable immediately, with EUR_USD's lock still held.
    acquired_xau_lock = await asyncio.wait_for(agent._lock_for("XAU_USD").acquire(), timeout=1.0)
    assert acquired_xau_lock is True
    agent._lock_for("XAU_USD").release()

    # Real, direct confirmation this is genuinely the SAME instrument's lock, not two different
    # asyncio.Lock objects handed out for the same key by accident.
    assert agent._lock_for("EUR_USD") is agent._lock_for("EUR_USD")
    assert agent._lock_for("EUR_USD") is not agent._lock_for("XAU_USD")

    release_eur_lock.set()
    await holder_task

    real_publisher.close()
    agent._subscriber.close()
    agent._zmq_context.term()

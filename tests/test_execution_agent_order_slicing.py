"""Real, executable tests for "Protocole Stealth Slicing" (2026-09-16) in src/agents/execution/agent.py:
ExecutionAgent._execute_sliced_entry, wired into _place_order for any real entry whose notional
exceeds ExecutionAgent._slicing_notional_threshold_usd.

Same fakeredis-as-process-singleton / _RecordingZmqPublisher / real-ZMQ-socket-cleanup discipline as
tests/test_execution_agent_instrument_lock.py. AgentControlState defaults to LIVE (unlike that file's
SHADOW-everywhere fixture) -- the whole point here is exercising the *real* broker-call path, since
slicing is never triggered for a simulated entry (see _execute_sliced_entry's own docstring). The
real OANDA client is never actually reachable in a test run; every test below replaces
_trading_client.create_market_order/get_current_price/get_account_balance/get_available_margin with
controlled fakes, the same convention tests/test_execution_agent_control_state.py's
test_live_entry_attempts_the_real_broker_call already established.

Every fixture agent is configured with a small, deterministic slicing plan (4 slices) and a real but
short execution horizon, so these tests run in well under a second while still exercising the real
asyncio.sleep pacing between slices, not a mocked-away one.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Generator, Iterator

import fakeredis.aioredis
import pytest
from pydantic import SecretStr

from core.config import AppSettings, OandaSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import (
    ORDER_SLICING_PROGRESS_REDIS_KEY_PREFIX,
    TRADING_SIGNALS_TOPIC,
    MarketMetrics,
    MessagePayload,
    TradingSignal,
)
import core.security as security_module
from core.security import trip_kill_switch
from src.agents.execution.agent import ExecutionAgent

_MARKET_DATA_TOPIC = "MARKET_DATA_L2"
_FILLS_BLOTTER_REDIS_KEY = "NOTHIERLY_FILLS_BLOTTER"
_INSTRUMENT = "EUR_USD"
_PRICE = 1.10000

# capital=10_000, _FALLBACK_RISK_PCT=0.01, stop_loss_pct default 0.015 -> risk_amount=100,
# sl_distance=1.1*0.015=0.0165, raw_units=100/0.0165=6060.60... -> int() truncates to 6060. Evenly
# divisible by 4 slices (cumulative rounding: 1515, 3030, 4545, 6060) -- deterministic, no rounding
# remainder to reason about in these assertions.
_EXPECTED_TOTAL_UNITS = 6060
_EXPECTED_SLICE_UNITS = 1515


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
    """Real fakeredis, also installed as the process singleton -- see test_execution_agent_control_state.py's
    identical fixture docstring for why (core.security.is_system_safe/trip_kill_switch must see this
    same data)."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _make_agent(redis_client: RedisClient, **slicing_kwargs: Any) -> ExecutionAgent:
    instance = ExecutionAgent(
        {_INSTRUMENT},
        capital=10_000.0,
        settings=_live_settings(),
        manage_own_signals=False,
        manage_own_redis=False,
        **slicing_kwargs,
    )
    instance._redis_client = redis_client
    return instance


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[ExecutionAgent]:
    """A low discretion threshold (well under this fixture's own ~$6,666 notional) so every test
    below deterministically takes the sliced path, with a short, real (not mocked-away) horizon."""
    instance = _make_agent(
        redis_client,
        slicing_notional_threshold_usd=100.0,
        slicing_num_slices=4,
        slicing_execution_horizon_seconds=0.08,
        slicing_jitter_seconds=0.01,
    )
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


async def _stub_broker_lookups(agent: ExecutionAgent) -> None:
    async def _fake_get_current_price(instrument: str) -> float:
        return _PRICE

    async def _fake_get_account_balance() -> float:
        return 10_000.0

    async def _fake_get_available_margin() -> float:
        return 10_000.0

    agent._trading_client.get_current_price = _fake_get_current_price  # type: ignore[method-assign]
    agent._trading_client.get_account_balance = _fake_get_account_balance  # type: ignore[method-assign]
    agent._trading_client.get_available_margin = _fake_get_available_margin  # type: ignore[method-assign]


async def _send_price(agent: ExecutionAgent, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(symbol=_INSTRUMENT, spread=0.0001, order_book_imbalance=0.0, mid_price=price).model_dump(mode="json"),
        )
    )


async def _send_signal(agent: ExecutionAgent, agent_id: str, signal: int) -> None:
    await agent._handle_signal_message(
        MessagePayload(
            topic=TRADING_SIGNALS_TOPIC,
            sender_id=agent_id,
            data=TradingSignal(
                timestamp=time.time(), agent_id=agent_id, instrument=_INSTRUMENT, signal=signal, confidence=0.8  # type: ignore[arg-type]
            ).model_dump(mode="json"),
        )
    )


async def _open_via_consensus(agent: ExecutionAgent) -> None:
    await _send_price(agent, _PRICE)
    await _send_signal(agent, "supervisor-agent", 1)
    await _send_signal(agent, "rl-signal-agent", 1)


async def _fills_blotter_entries(redis_client: RedisClient) -> list[dict[str, Any]]:
    raw_entries = await redis_client.execute("lrange", _FILLS_BLOTTER_REDIS_KEY, 0, -1)
    return [json.loads(raw) for raw in raw_entries]


async def _slicing_progress(redis_client: RedisClient) -> dict[str, Any] | None:
    raw = await redis_client.execute("get", f"{ORDER_SLICING_PROGRESS_REDIS_KEY_PREFIX}{_INSTRUMENT}")
    return json.loads(raw) if raw is not None else None


@pytest.mark.asyncio
async def test_large_real_entry_is_sliced_into_n_real_orders_not_one(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """The real, headline behavior: a real notional above the threshold produces N real broker
    calls, each smaller than the whole, not one single market order."""
    await _stub_broker_lookups(agent)
    calls: list[int] = []

    async def _fake_create_market_order(instrument: str, units: int) -> dict[str, Any]:
        calls.append(units)
        return {"id": f"fake-slice-{len(calls)}", "price": _PRICE}

    agent._trading_client.create_market_order = _fake_create_market_order  # type: ignore[assignment]

    await _open_via_consensus(agent)

    assert calls == [_EXPECTED_SLICE_UNITS] * 4, f"expected 4 equal real slices, got {calls}"
    assert sum(calls) == _EXPECTED_TOTAL_UNITS

    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"
    assert position.units_held == _EXPECTED_TOTAL_UNITS
    assert position.entry_price == _PRICE
    assert not position.entered_as_simulated

    entries = await _fills_blotter_entries(redis_client)
    assert len(entries) == 4
    assert all(e["action"] == "ENTRY_LONG" for e in entries)

    progress = await _slicing_progress(redis_client)
    assert progress is not None
    assert progress["status"] == "COMPLETED"
    assert progress["slices_completed"] == 4
    assert progress["slices_planned"] == 4
    assert progress["percent_complete"] == 100.0
    assert progress["direction"] == 1


@pytest.mark.asyncio
async def test_real_jittered_pacing_actually_takes_real_wall_clock_time(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """Not instant: 3 real gaps at ~0.02s base interval (0.08s horizon / 4 slices) must really elapse,
    proving this is a real asyncio.sleep-paced execution, not slices fired back-to-back."""
    await _stub_broker_lookups(agent)

    async def _fake_create_market_order(instrument: str, units: int) -> dict[str, Any]:
        return {"id": "fake", "price": _PRICE}

    agent._trading_client.create_market_order = _fake_create_market_order  # type: ignore[assignment]

    start = time.monotonic()
    await _open_via_consensus(agent)
    elapsed = time.monotonic() - start

    # 3 gaps * (0.02s base - 0.01s max jitter) is the real floor; generous slack for test-runner noise.
    assert elapsed >= 0.03, f"expected real TWAP pacing to take a non-trivial amount of time, took {elapsed:.4f}s"


@pytest.mark.asyncio
async def test_kill_switch_tripped_mid_slicing_aborts_remaining_slices_honestly(
    agent: ExecutionAgent, redis_client: RedisClient
) -> None:
    """The real safety property this whole feature exists to preserve: a real Kill Switch trip
    mid-execution must stop remaining slices, leaving the position honestly partial, not silently
    completed and not silently discarded."""
    security_module._min_accepted_epoch = 0  # test isolation -- see core/security.py's own docstring on this variable
    await _stub_broker_lookups(agent)
    calls: list[int] = []

    async def _fake_create_market_order(instrument: str, units: int) -> dict[str, Any]:
        calls.append(units)
        if len(calls) == 2:
            await trip_kill_switch("integration test: simulated real PANIC mid-slicing")
        return {"id": f"fake-slice-{len(calls)}", "price": _PRICE}

    agent._trading_client.create_market_order = _fake_create_market_order  # type: ignore[assignment]

    await _open_via_consensus(agent)

    assert len(calls) == 2, f"expected exactly 2 real slices before the trip blocked the 3rd, got {len(calls)}"

    position = agent._positions[_INSTRUMENT]
    assert position.status == "LONG"  # the 2 real slices that DID fill are honestly reflected, not rolled back
    assert position.units_held == 2 * _EXPECTED_SLICE_UNITS

    progress = await _slicing_progress(redis_client)
    assert progress is not None
    assert progress["status"] == "ABORTED"
    assert progress["slices_completed"] == 2
    assert progress["slices_planned"] == 4


@pytest.mark.asyncio
async def test_concurrent_real_exit_mid_slicing_aborts_remaining_slices(redis_client: RedisClient) -> None:
    """A second real, load-bearing safety property: this module's own opening invariant is that an
    exit is unilateral and immediate, never gated -- so a real stop-loss/veto exit landing between two
    slices must win, and slicing must never fight it by continuing to buy into a position that was
    just deliberately closed. Uses a longer, jitter-free horizon than the shared `agent` fixture so
    the exit can be timed deterministically into the real gap between slice 1 and slice 2."""
    instance = _make_agent(
        redis_client,
        slicing_notional_threshold_usd=100.0,
        slicing_num_slices=4,
        slicing_execution_horizon_seconds=0.4,  # 0.1333s/gap, jitter-free -- a wide, predictable window
        slicing_jitter_seconds=0.0,
    )
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    try:
        await _stub_broker_lookups(instance)
        calls: list[int] = []

        async def _fake_create_market_order(inst: str, units: int) -> dict[str, Any]:
            calls.append(units)
            return {"id": f"fake-slice-{len(calls)}", "price": _PRICE}

        instance._trading_client.create_market_order = _fake_create_market_order  # type: ignore[assignment]

        task = asyncio.create_task(_open_via_consensus(instance))
        # Slice 1 fires near-instantly; land well inside the ~0.1333s gap before slice 2, after
        # slice 1's own critical section has released the per-instrument lock.
        await asyncio.sleep(0.05)
        assert len(calls) == 1, "test assumption violated: slice 1 had not yet fired at the check point"

        await instance._close_position(_INSTRUMENT, reason="test: real concurrent veto exit", price=_PRICE)
        assert instance._positions[_INSTRUMENT].status == "FLAT"

        await task  # let the sliced entry's background execution finish (aborting on slice 2)

        assert len(calls) == 2, f"expected slice 1 (entry) + exactly 1 real close call, got {len(calls)}"
        # The position must stay FLAT -- slicing must not have reopened/re-added to it after the
        # real exit, even though its own planned total was only 1 of 4 slices in.
        assert instance._positions[_INSTRUMENT].status == "FLAT"

        progress = await _slicing_progress(redis_client)
        assert progress is not None
        assert progress["status"] == "ABORTED"
        assert progress["slices_completed"] == 1
        assert progress["slices_planned"] == 4
    finally:
        real_publisher.close()
        instance._subscriber.close()
        instance._zmq_context.term()


@pytest.mark.asyncio
async def test_small_real_entry_below_threshold_is_not_sliced(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    """Regression guard: the pre-existing single-market-order path must stay untouched for an entry
    that does not cross the discretion threshold."""
    agent._slicing_notional_threshold_usd = 1_000_000.0  # far above this fixture's ~$6,666 real notional
    await _stub_broker_lookups(agent)
    calls: list[int] = []

    async def _fake_create_market_order(instrument: str, units: int) -> dict[str, Any]:
        calls.append(units)
        return {"id": "fake-single-order"}

    agent._trading_client.create_market_order = _fake_create_market_order  # type: ignore[assignment]

    await _open_via_consensus(agent)

    assert len(calls) == 1, f"expected exactly 1 real order (unsliced path), got {len(calls)}"
    assert calls[0] == _EXPECTED_TOTAL_UNITS
    assert await _slicing_progress(redis_client) is None  # no slicing telemetry for the unsliced path

"""Real, executable tests for the per-asset risk override (``alpha_engine_asset_risk_override:<instrument>``, written by Asset Universe Control) as read by
src/agents/execution/agent.py: a spread ceiling that refuses entries, and a position-size cap. Before this wiring the form wrote to Redis and nothing read it.

Reuses the fixtures of tests/test_execution_agent_asset_kill_switch.py (real fakeredis, MOCK_MODE=False so the override, not MOCK_MODE, is what blocks).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from core.base_agent import AgentControlState
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX
from src.agents.execution.agent import TRADING_MAX_POSITION_UNITS_REDIS_KEY, AssetRiskOverrideUnreadableError, ExecutionAgent
from tests.test_execution_agent_asset_kill_switch import (  # noqa: F401 -- fixtures partagees
    _INSTRUMENT,
    _OTHER,
    _open_via_consensus,
    _poison_pill,
    _send_price,
    _send_signal,
    agent,
    redis_client,
)


async def _set_override(client: RedisClient, instrument: str, *, spread: Any = "0", size: Any = "0") -> None:
    payload = json.dumps({"max_allowable_spread": str(spread), "max_position_size": str(size), "updated_at": 1.0})
    await client.execute("set", f"{ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX}{instrument}", payload)


@pytest.mark.asyncio
async def test_no_override_means_no_restriction(agent: ExecutionAgent) -> None:
    assert await agent._read_asset_risk_override(_INSTRUMENT) is None


@pytest.mark.asyncio
async def test_zero_values_mean_no_override_on_that_limit(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    await _set_override(redis_client, _INSTRUMENT, spread="0.0", size="0.0")  # exactement ce qu'ecrit le formulaire sans override
    override = await agent._read_asset_risk_override(_INSTRUMENT)
    assert override is not None and override.max_spread is None and override.max_units is None
    await _set_override(redis_client, _INSTRUMENT, spread="0.0002", size="5")
    override = await agent._read_asset_risk_override(_INSTRUMENT)
    assert override is not None and override.max_spread == pytest.approx(0.0002) and override.max_units == 5.0


@pytest.mark.asyncio
async def test_a_spread_above_the_ceiling_refuses_a_new_entry_and_reports_gate_8(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_override(redis_client, _INSTRUMENT, spread="0.00005")  # le spread des ticks de test est 0.0001
    await _open_via_consensus(agent)
    assert agent._position_for(_INSTRUMENT).status == "FLAT"

    report, price, balance, units = await agent._validate_execution_gates(_INSTRUMENT, 1, None)
    assert report.final_action == "REJECTED" and (price, balance, units) == (None, None, 0.0)
    last = report.gates[-1]
    assert last.number == 8 and last.status == "FAILED" and "ASSET RISK OVERRIDE" in last.detail and "spread" in last.detail


@pytest.mark.asyncio
async def test_a_spread_within_the_ceiling_lets_the_entry_through(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_override(redis_client, _INSTRUMENT, spread="0.0002")
    await _open_via_consensus(agent)
    assert agent._position_for(_INSTRUMENT).status == "LONG"


@pytest.mark.asyncio
async def test_the_override_is_per_asset(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _set_override(redis_client, _INSTRUMENT, spread="0.00005")
    await _open_via_consensus(agent, _OTHER)  # un autre actif n'est pas concerne
    assert agent._position_for(_OTHER).status == "LONG"


@pytest.mark.asyncio
async def test_the_override_never_traps_an_open_position(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    agent._control_state = AgentControlState.SHADOW
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    await _open_via_consensus(agent)
    assert agent._position_for(_INSTRUMENT).status == "LONG"
    await _set_override(redis_client, _INSTRUMENT, spread="0.00005")  # plafond resserre APRES l'entree : la sortie reste toujours possible
    await _send_signal(agent, _INSTRUMENT, "rl-signal-agent", -1)
    assert agent._position_for(_INSTRUMENT).status == "FLAT"


@pytest.mark.asyncio
async def test_the_size_cap_reduces_the_calculated_size_and_combines_with_the_global_cap(agent: ExecutionAgent, redis_client: RedisClient) -> None:
    uncapped = await agent.calculate_position_size(_INSTRUMENT, 1.10, 1.09, 10_000.0)
    assert uncapped > 50
    await _set_override(redis_client, _INSTRUMENT, size="5")
    assert await agent.calculate_position_size(_INSTRUMENT, 1.10, 1.09, 10_000.0) == 5.0
    assert await agent.calculate_position_size(_OTHER, 1.10, 1.09, 10_000.0) == pytest.approx(uncapped)  # l'autre actif n'est pas plafonne
    await _set_override(redis_client, _INSTRUMENT, size="500000")  # plafond au-dessus de la taille calculee : sans effet
    assert await agent.calculate_position_size(_INSTRUMENT, 1.10, 1.09, 10_000.0) == pytest.approx(uncapped)
    await _set_override(redis_client, _INSTRUMENT, size="5")
    await redis_client.execute("set", TRADING_MAX_POSITION_UNITS_REDIS_KEY, "3")
    assert await agent.calculate_position_size(_INSTRUMENT, 1.10, 1.09, 10_000.0) == 3.0  # le plafond le plus strict l'emporte


@pytest.mark.asyncio
async def test_an_unreadable_override_fails_closed_for_the_gate_and_the_sizing(agent: ExecutionAgent, redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch) -> None:
    agent._trading_client.create_market_order = _poison_pill  # type: ignore[method-assign]
    key = f"{ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX}{_INSTRUMENT}"
    for bad in ("not json", "[1, 2]", json.dumps({"max_allowable_spread": "abc"}), json.dumps({"max_allowable_spread": "-1"}), json.dumps({"max_position_size": "nan"})):
        await redis_client.execute("set", key, bad)
        with pytest.raises(AssetRiskOverrideUnreadableError):
            await agent._read_asset_risk_override(_INSTRUMENT)
        assert await agent.calculate_position_size(_INSTRUMENT, 1.10, 1.09, 10_000.0) == 0.0
    await _open_via_consensus(agent)
    assert agent._position_for(_INSTRUMENT).status == "FLAT"
    report, *_ = await agent._validate_execution_gates(_INSTRUMENT, 1, None)
    assert report.final_action == "REJECTED" and "illisible" in report.gates[-1].detail

    await redis_client.execute("delete", key)

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RedisUnavailableError("redis down")

    monkeypatch.setattr(redis_client, "execute", broken)
    with pytest.raises(AssetRiskOverrideUnreadableError):
        await agent._read_asset_risk_override(_INSTRUMENT)


@pytest.mark.asyncio
async def test_no_redis_client_means_no_override(agent: ExecutionAgent) -> None:
    agent._redis_client = None
    assert await agent._read_asset_risk_override(_INSTRUMENT) is None

"""Real, executable tests for src/agents/supervisor/broker_client.py's new get_open_positions.

A minimal fake v20.Context stand-in (injected via OandaClient's own
``context=`` constructor parameter, built for exactly this kind of
testing) -- matches v20's real attribute-style response objects
(``.long.units``, not ``["long"]["units"]``), not a dict mock.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.agents.supervisor.broker_client import OandaClient


class _FakeResponse:
    def __init__(self, *, status: int = 200, body: dict[str, Any]) -> None:
        self.status = status
        self.body = body


class _FakePositionEndpoint:
    def __init__(self, positions: list[Any]) -> None:
        self._positions = positions

    def list_open(self, account_id: str) -> _FakeResponse:
        return _FakeResponse(body={"positions": self._positions})


class _FakeContext:
    def __init__(self, positions: list[Any]) -> None:
        self.position = _FakePositionEndpoint(positions)


def _leg(units: float, avg_price: float) -> SimpleNamespace:
    return SimpleNamespace(units=units, averagePrice=avg_price)


@pytest.mark.asyncio
async def test_get_open_positions_collapses_long_leg_to_positive_signed_units() -> None:
    positions = [
        SimpleNamespace(
            instrument="EUR_USD", unrealizedPL=42.5,
            long=_leg(1000.0, 1.0850), short=_leg(0.0, 0.0),
        )
    ]
    client = OandaClient(api_token="x", account_id="101-000-0000000-001", context=_FakeContext(positions))
    result = await client.get_open_positions()

    assert len(result) == 1
    assert result[0]["instrument"] == "EUR_USD"
    assert result[0]["units"] == 1000.0
    assert result[0]["avg_price"] == 1.0850
    assert result[0]["unrealized_pnl"] == 42.5


@pytest.mark.asyncio
async def test_get_open_positions_collapses_short_leg_to_negative_signed_units() -> None:
    positions = [
        SimpleNamespace(
            instrument="BTC_USD", unrealizedPL=-15.0,
            long=_leg(0.0, 0.0), short=_leg(-500.0, 65000.0),
        )
    ]
    client = OandaClient(api_token="x", account_id="101-000-0000000-001", context=_FakeContext(positions))
    result = await client.get_open_positions()

    assert result[0]["units"] == -500.0
    assert result[0]["avg_price"] == 65000.0


@pytest.mark.asyncio
async def test_get_open_positions_returns_empty_list_when_no_positions() -> None:
    client = OandaClient(api_token="x", account_id="101-000-0000000-001", context=_FakeContext([]))
    assert await client.get_open_positions() == []

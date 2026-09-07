"""Real, executable tests for src/agents/market_data/exchange_client.py's retry/backoff behavior.

Focused on the real bug found via live testing (Chapter 13, Étape 5.6):
AuthenticationError must fail immediately, never retry forever. No real
exchange is contacted -- ``_call_with_backoff`` is exercised directly
with a fake ``call`` callable.
"""

from __future__ import annotations

import pytest
from ccxt.base.errors import AuthenticationError, NetworkError

from src.agents.market_data.exchange_client import AsyncExchangeClient


def _client() -> AsyncExchangeClient:
    return AsyncExchangeClient("binance", base_delay_seconds=0.01, max_delay_seconds=0.02)


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried() -> None:
    call_count = 0

    async def always_fails_auth() -> None:
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("Invalid Api-Key ID.")

    client = _client()
    with pytest.raises(AuthenticationError):
        await client._call_with_backoff(always_fails_auth, operation="test", context={})
    assert call_count == 1  # exactly one attempt -- never retried


@pytest.mark.asyncio
async def test_recoverable_network_error_is_retried_until_success() -> None:
    call_count = 0

    async def fails_twice_then_succeeds() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise NetworkError("transient blip")
        return "ok"

    client = _client()
    result = await client._call_with_backoff(fails_twice_then_succeeds, operation="test", context={})
    assert result == "ok"
    assert call_count == 3

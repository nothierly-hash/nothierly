"""Real, executable tests for src/agents/market_data/exchange_client.py's retry/backoff behavior.

Focused on two real bugs found via live testing, not theorized:

1. Chapter 13, Étape 5.6: AuthenticationError must fail immediately, never retry forever.
2. A real multi-hour shadow run crash: a raw ``ConnectionResetError`` (``OSError``) leaking out of
   ccxt.pro's own internal websocket keepalive task, never translated into ``ccxt.NetworkError``,
   used to crash the whole agent process instead of being retried like every other transient
   network failure -- see :data:`~src.agents.market_data.exchange_client._RECOVERABLE_CCXT_ERRORS`'s
   own comment.

No real exchange is contacted -- ``_call_with_backoff`` is exercised directly with a fake ``call``
callable.
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


@pytest.mark.asyncio
async def test_raw_connection_reset_error_is_retried_not_crashed() -> None:
    """Real bug, found live: ccxt.pro's own internal ping keepalive can raise a raw
    ConnectionResetError that never passes through ccxt's own NetworkError wrapping -- this must be
    retried exactly like a real ccxt.NetworkError, never left to crash the caller."""
    call_count = 0

    async def fails_once_with_raw_os_error_then_succeeds() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionResetError("Cannot write to closing transport")
        return "ok"

    client = _client()
    result = await client._call_with_backoff(fails_once_with_raw_os_error_then_succeeds, operation="test", context={})
    assert result == "ok"
    assert call_count == 2

"""Real, executable tests for src/execution/broker_gateway.py.

Real Redis behavior via fakeredis (fresh FakeServer() per test, same
discipline as tests/test_gateway.py). Real encryption via
SecurityVault -- no mocking of the crypto itself. The underlying
AsyncExchangeClient/OandaClient network calls are never exercised here
(no real exchange in a test run) -- BrokerFactory.create only
*constructs* adapters, it never calls connect(), so no network I/O
happens in any of these tests.
"""

from __future__ import annotations

import json
from typing import Any, Generator

import fakeredis.aioredis
import pytest

from core.config import AppSettings
from core.redis_client import RedisClient
from src.execution.broker_gateway import (
    BROKER_CREDENTIALS_REDIS_KEY_PREFIX,
    BaseBroker,
    BrokerCapabilityError,
    BrokerFactory,
    CcxtBrokerAdapter,
    OandaBrokerAdapter,
    _resolve_ccxt_credentials,
    store_broker_credentials,
)
from src.execution.security_vault import SecurityVault


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


def _settings_with(
    *, broker_api_key: str = "", broker_api_secret: str = "", vault_key: str = "test-master-key"
) -> AppSettings:
    from core.config import BrokerSettings, OandaSettings, VaultSettings
    from pydantic import SecretStr

    return AppSettings(
        broker=BrokerSettings(api_key=SecretStr(broker_api_key), api_secret=SecretStr(broker_api_secret)),
        oanda=OandaSettings(api_token=SecretStr("test-oanda-token"), account_id="101-000-0000000-001"),
        vault=VaultSettings(master_key=SecretStr(vault_key)),
    )


class _FakeBrokerThatFailsToConnect(BaseBroker):
    """A real, minimal BaseBroker whose connect() deliberately raises -- to test __aenter__'s own cleanup contract, not a mock of it."""

    def __init__(self) -> None:
        self.broker_id = "fake-fails-to-connect"
        self.close_called = False

    async def connect(self) -> None:
        raise ConnectionError("simulated real connect() failure (e.g. a real invalid API key)")

    async def close(self) -> None:
        self.close_called = True

    async def place_market_order(self, symbol: str, side: Any, quantity: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def place_limit_order(self, symbol: str, side: Any, quantity: float, price: float, *, client_order_id: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> Any:
        raise NotImplementedError

    async def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError

    async def get_account_balance(self) -> float:
        raise NotImplementedError

    async def get_margin_snapshot(self) -> Any:
        raise NotImplementedError

    async def get_open_positions(self) -> list[Any]:
        raise NotImplementedError


@pytest.mark.asyncio
async def test_aenter_closes_real_resources_when_connect_fails() -> None:
    """Real regression test for a real bug found via live testing: a failing connect() inside `async with
    broker:` used to leak every resource already allocated before the failure, since Python's async with
    protocol never calls __aexit__ when __aenter__ itself raises. See BaseBroker.__aenter__'s own docstring."""
    broker = _FakeBrokerThatFailsToConnect()
    with pytest.raises(ConnectionError):
        async with broker:
            pytest.fail("must never reach the body -- connect() raises before __aenter__ returns")
    assert broker.close_called is True


@pytest.mark.asyncio
async def test_factory_creates_oanda_adapter() -> None:
    broker = await BrokerFactory.create("oanda", settings=_settings_with())
    assert isinstance(broker, OandaBrokerAdapter)
    assert broker.broker_id == "oanda"


@pytest.mark.asyncio
async def test_factory_creates_ccxt_adapter_for_any_exchange_id() -> None:
    for exchange_id in ("binance", "bybit", "okx"):
        broker = await BrokerFactory.create(exchange_id, settings=_settings_with())
        assert isinstance(broker, CcxtBrokerAdapter)
        assert broker.broker_id == exchange_id


@pytest.mark.asyncio
async def test_factory_is_case_insensitive() -> None:
    broker = await BrokerFactory.create("BiNaNcE", settings=_settings_with())
    assert broker.broker_id == "binance"


@pytest.mark.asyncio
async def test_oanda_adapter_place_limit_order_raises_capability_error() -> None:
    broker = OandaBrokerAdapter(api_token="x", account_id="101-000-0000000-001")
    with pytest.raises(BrokerCapabilityError):
        await broker.place_limit_order("EUR_USD", "buy", 100.0, 1.05)


@pytest.mark.asyncio
async def test_oanda_adapter_get_orderbook_snapshot_raises_capability_error() -> None:
    broker = OandaBrokerAdapter(api_token="x", account_id="101-000-0000000-001")
    with pytest.raises(BrokerCapabilityError):
        await broker.get_orderbook_snapshot("EUR_USD")


@pytest.mark.asyncio
async def test_oanda_adapter_cancel_order_raises_capability_error() -> None:
    broker = OandaBrokerAdapter(api_token="x", account_id="101-000-0000000-001")
    with pytest.raises(BrokerCapabilityError):
        await broker.cancel_order("some-order-id")


@pytest.mark.asyncio
async def test_resolve_ccxt_credentials_falls_back_to_env_default_when_no_redis() -> None:
    settings = _settings_with(broker_api_key="env-key", broker_api_secret="env-secret")
    api_key, api_secret = await _resolve_ccxt_credentials("binance", settings=settings, redis_client=None)
    assert (api_key, api_secret) == ("env-key", "env-secret")


@pytest.mark.asyncio
async def test_store_and_resolve_credentials_real_encrypt_decrypt_round_trip(redis_client: RedisClient) -> None:
    settings = _settings_with(broker_api_key="env-key", broker_api_secret="env-secret", vault_key="round-trip-key")
    vault = SecurityVault(settings.vault.master_key.get_secret_value())

    await store_broker_credentials(
        "bybit", api_key="real-bybit-key", api_secret="real-bybit-secret", redis_client=redis_client, vault=vault
    )

    # The stored record is genuinely ciphertext, not plaintext, in Redis.
    raw = await redis_client.execute("get", f"{BROKER_CREDENTIALS_REDIS_KEY_PREFIX}bybit")
    record = json.loads(raw)
    assert "real-bybit-key" not in record["api_key"]
    assert "real-bybit-secret" not in record["api_secret"]

    api_key, api_secret = await _resolve_ccxt_credentials("bybit", settings=settings, redis_client=redis_client)
    assert (api_key, api_secret) == ("real-bybit-key", "real-bybit-secret")


@pytest.mark.asyncio
async def test_resolve_ccxt_credentials_falls_back_when_vault_key_changed(redis_client: RedisClient) -> None:
    """A wrong/rotated master key must degrade to the env default, never crash order flow."""
    write_settings = _settings_with(vault_key="original-key")
    vault = SecurityVault(write_settings.vault.master_key.get_secret_value())
    await store_broker_credentials(
        "okx", api_key="secret-key", api_secret="secret-secret", redis_client=redis_client, vault=vault
    )

    read_settings = _settings_with(broker_api_key="fallback-key", broker_api_secret="fallback-secret", vault_key="rotated-key")
    api_key, api_secret = await _resolve_ccxt_credentials("okx", settings=read_settings, redis_client=redis_client)
    assert (api_key, api_secret) == ("fallback-key", "fallback-secret")


# =============================================================================
# Chapter 13, Étape 5.8: real margin/position methods, added for
# risk/cross_margin_engine.py -- the underlying AsyncExchangeClient/
# OandaClient calls are monkeypatched here (no real exchange), but the
# real conversion logic (spot quote-currency filtering, the honest
# unmargined-venue None, OANDA's real margin fields) is exercised for real.
# =============================================================================


@pytest.mark.asyncio
async def test_ccxt_adapter_margin_snapshot_is_honestly_unmargined() -> None:
    adapter = CcxtBrokerAdapter("binance")

    async def _fake_balance() -> float:
        return 12345.0

    adapter._client.get_account_balance = _fake_balance  # type: ignore[method-assign]
    snapshot = await adapter.get_margin_snapshot()

    assert snapshot.account_balance_usd == 12345.0
    assert snapshot.margin_used_usd is None
    assert snapshot.margin_available_usd is None
    assert snapshot.margin_utilization_pct is None


@pytest.mark.asyncio
async def test_ccxt_adapter_open_positions_skips_quote_currency_and_prices_the_rest() -> None:
    adapter = CcxtBrokerAdapter("binance")

    async def _fake_positions() -> list[dict[str, object]]:
        return [
            {"symbol": "USDT", "amount": 5000.0, "avg_price": None},
            {"symbol": "BTC", "amount": 0.5, "avg_price": None},
        ]

    async def _fake_price(symbol: str) -> float:
        assert symbol == "BTC/USDT"
        return 65000.0

    adapter._client.get_open_positions = _fake_positions  # type: ignore[method-assign]
    adapter._client.get_current_price = _fake_price  # type: ignore[method-assign]

    positions = await adapter.get_open_positions()

    assert len(positions) == 1  # USDT (cash) excluded
    assert positions[0].asset.base_currency == "BTC"
    assert positions[0].quantity == 0.5
    assert positions[0].mark_price == 65000.0
    assert positions[0].margin_used_usd is None  # spot -- honestly unmargined
    assert positions[0].unrealized_pnl_usd == 0  # no real cost basis tracked for spot -- never fabricated


@pytest.mark.asyncio
async def test_oanda_adapter_margin_snapshot_uses_real_reported_fields() -> None:
    adapter = OandaBrokerAdapter(api_token="x", account_id="101-000-0000000-001")

    async def _fake_margin_summary() -> dict[str, float]:
        return {"balance": 10000.0, "margin_used": 2500.0, "margin_available": 7500.0}

    adapter._client.get_margin_summary = _fake_margin_summary  # type: ignore[method-assign]
    snapshot = await adapter.get_margin_snapshot()

    assert snapshot.margin_used_usd == 2500.0
    assert snapshot.margin_available_usd == 7500.0
    assert snapshot.margin_utilization_pct == 25.0


@pytest.mark.asyncio
async def test_oanda_adapter_open_positions_reprices_via_live_lookup() -> None:
    adapter = OandaBrokerAdapter(api_token="x", account_id="101-000-0000000-001")

    async def _fake_positions() -> list[dict[str, object]]:
        return [{"instrument": "EUR_USD", "units": 1000.0, "avg_price": 1.0800, "unrealized_pnl": 15.0}]

    async def _fake_price(instrument: str) -> float:
        return 1.0850

    adapter._client.get_open_positions = _fake_positions  # type: ignore[method-assign]
    adapter._client.get_current_price = _fake_price  # type: ignore[method-assign]

    positions = await adapter.get_open_positions()

    assert len(positions) == 1
    assert positions[0].asset.base_currency == "EUR"
    # Decimal('1.08') != the float literal 1.08 (binary floating point can't
    # represent 1.08 exactly) -- compare as float, not a real bug in the adapter.
    assert float(positions[0].avg_entry_price) == 1.0800
    assert float(positions[0].mark_price) == 1.0850  # real live re-price, not just the stale entry price
    assert positions[0].side == "LONG"

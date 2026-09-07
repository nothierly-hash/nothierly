"""Real, executable tests for src/execution/broker_gateway.py's IBKRBrokerAdapter.

No real TWS/IB Gateway connection is exercised (none is running in a
test environment) -- these test what's genuinely testable without one:
the real symbol->Contract resolution heuristic, ABC compliance, and
BrokerFactory dispatch. ib_async must be imported inside a running
event loop (see broker_gateway.py's module docstring for why) --
every test here is `async def` for exactly that reason.
"""

from __future__ import annotations

import pytest

from src.execution.broker_gateway import BaseBroker, BrokerFactory, IBKRBrokerAdapter


@pytest.mark.asyncio
async def test_resolve_contract_forex_oanda_style() -> None:
    adapter = IBKRBrokerAdapter()
    contract = adapter._resolve_contract("EUR_USD")
    import ib_async

    assert isinstance(contract, ib_async.Forex)
    assert contract.pair() == "EURUSD"


@pytest.mark.asyncio
async def test_resolve_contract_forex_ccxt_style() -> None:
    adapter = IBKRBrokerAdapter()
    contract = adapter._resolve_contract("EUR/GBP")
    import ib_async

    assert isinstance(contract, ib_async.Forex)


@pytest.mark.asyncio
async def test_resolve_contract_crypto() -> None:
    adapter = IBKRBrokerAdapter()
    contract = adapter._resolve_contract("BTC/USD")
    import ib_async

    assert isinstance(contract, ib_async.Crypto)
    assert contract.symbol == "BTC"
    assert contract.currency == "USD"


@pytest.mark.asyncio
async def test_resolve_contract_stock_fallback() -> None:
    adapter = IBKRBrokerAdapter()
    contract = adapter._resolve_contract("AAPL")
    import ib_async

    assert isinstance(contract, ib_async.Stock)
    assert contract.symbol == "AAPL"
    assert contract.exchange == "SMART"


@pytest.mark.asyncio
async def test_ibkr_adapter_satisfies_base_broker_contract() -> None:
    adapter = IBKRBrokerAdapter()
    assert isinstance(adapter, BaseBroker)
    assert adapter.broker_id == "ibkr"


@pytest.mark.asyncio
async def test_ibkr_adapter_orderbook_snapshot_raises_capability_error() -> None:
    from src.execution.broker_gateway import BrokerCapabilityError

    adapter = IBKRBrokerAdapter()
    with pytest.raises(BrokerCapabilityError):
        await adapter.get_orderbook_snapshot("EUR_USD")


@pytest.mark.asyncio
async def test_factory_creates_ibkr_adapter_for_both_aliases() -> None:
    from core.config import AppSettings, IbkrSettings

    settings = AppSettings(ibkr=IbkrSettings(host="127.0.0.1", port=7497, client_id=7))
    for broker_id in ("ibkr", "interactive_brokers", "IBKR"):
        broker = await BrokerFactory.create(broker_id, settings=settings)
        assert isinstance(broker, IBKRBrokerAdapter)
        assert broker.broker_id == "ibkr"

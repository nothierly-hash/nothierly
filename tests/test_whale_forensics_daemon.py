"""Real, executable tests for src/agents/whale_forensics/agent.py.

``decode_transfer_log`` is pure and tested directly against real
``eth_getLogs``-shaped input (real topic/data encoding, hand-built). The
agent-level tests fake only the two real network surfaces it touches
(``web3.eth`` and Redis, via ``fakeredis``) -- everything else (settings
parsing, threshold evaluation, USD resolution, Gate-8 lock write) is real.

ZMQ cleanup follows this repo's own established pattern (see
tests/test_execution_agent_instrument_lock.py's ``agent`` fixture / the
"nothierly-zmq-test-leak-pattern" memory): close the REAL original
publisher/subscriber/context at teardown, never the fake standing in for it.
"""

from __future__ import annotations

from typing import Any, Iterator

import fakeredis.aioredis
import pytest
from web3 import Web3

from core.config import AppSettings, RedisSettings, Web3Settings
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX, ONCHAIN_TSUNAMI_LOCK_REDIS_KEY
from src.agents.whale_forensics.agent import (
    TRANSFER_EVENT_TOPIC,
    WhaleForensicsDaemon,
    decode_transfer_log,
    _parse_institutional_labels,
    _parse_watched_tokens,
)

_WETH_ADDRESS = "0x" + "aa" * 20
_USDC_ADDRESS = "0x" + "bb" * 20
_FROM_ADDRESS = "0x" + "11" * 20
_TO_ADDRESS = "0x" + "22" * 20


def _topic_from_address(address: str) -> bytes:
    """A real, left-padded-to-32-bytes indexed address topic, exactly how a real log encodes it."""
    return bytes(12) + bytes.fromhex(address[2:])


def _real_transfer_log(*, amount_raw: int, block_number: int = 12345, tx_hash: bytes = b"\x01" * 32) -> dict[str, Any]:
    return {
        "address": _WETH_ADDRESS,
        "topics": [bytes.fromhex(TRANSFER_EVENT_TOPIC[2:]), _topic_from_address(_FROM_ADDRESS), _topic_from_address(_TO_ADDRESS)],
        "data": amount_raw.to_bytes(32, "big"),
        "blockNumber": block_number,
        "transactionHash": tx_hash,
    }


# =============================================================================
# decode_transfer_log / parsing helpers (pure)
# =============================================================================


def test_parse_watched_tokens() -> None:
    assert _parse_watched_tokens(f"WETH:{_WETH_ADDRESS},USDC:{_USDC_ADDRESS}") == {
        _WETH_ADDRESS.lower(): "WETH",
        _USDC_ADDRESS.lower(): "USDC",
    }


def test_parse_watched_tokens_empty_string() -> None:
    assert _parse_watched_tokens("") == {}


def test_parse_institutional_labels() -> None:
    assert _parse_institutional_labels(f"Binance Hot Wallet:{_FROM_ADDRESS}") == {
        _FROM_ADDRESS.lower(): "Binance Hot Wallet"
    }


def test_decode_transfer_log_below_threshold_returns_none() -> None:
    # 10 real WETH at $2000/WETH = $20,000 -- well under a $50M threshold.
    log = _real_transfer_log(amount_raw=10 * 10**18)
    alert = decode_transfer_log(
        log, chain="ethereum", token_symbol="WETH", decimals=18, usd_price=2000.0, threshold_usd=50_000_000.0
    )
    assert alert is None


def test_decode_transfer_log_above_threshold_decodes_correctly() -> None:
    # 30,000 real WETH at $2000/WETH = $60,000,000 -- clears a $50M threshold.
    log = _real_transfer_log(amount_raw=30_000 * 10**18, block_number=999, tx_hash=b"\xab" * 32)
    alert = decode_transfer_log(
        log, chain="ethereum", token_symbol="WETH", decimals=18, usd_price=2000.0, threshold_usd=50_000_000.0
    )
    assert alert is not None
    assert alert.amount_usd == pytest.approx(60_000_000.0)
    assert alert.from_address == _FROM_ADDRESS.lower()
    assert alert.to_address == _TO_ADDRESS.lower()
    assert alert.block_number == 999
    assert alert.tx_hash == "0x" + "ab" * 32
    assert alert.token_symbol == "WETH"


def test_decode_transfer_log_respects_real_decimals() -> None:
    # 60,000,000 raw USDC units at 6 decimals = 60 real USDC tokens = $60 at $1 each.
    log = _real_transfer_log(amount_raw=60_000_000)
    below = decode_transfer_log(log, chain="ethereum", token_symbol="USDC", decimals=6, usd_price=1.0, threshold_usd=100.0)
    assert below is None
    above = decode_transfer_log(log, chain="ethereum", token_symbol="USDC", decimals=6, usd_price=1.0, threshold_usd=50.0)
    assert above is not None
    assert above.amount_usd == pytest.approx(60.0)


# =============================================================================
# WhaleForensicsDaemon (agent-level)
# =============================================================================


class _FakeEth:
    def __init__(self, *, block_number: int, logs_by_range: dict[tuple[int, int], list[dict[str, Any]]]) -> None:
        self.block_number = block_number
        self._logs_by_range = logs_by_range
        self.get_logs_calls: list[dict[str, Any]] = []

    def get_logs(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.get_logs_calls.append(params)
        return self._logs_by_range.get((params["fromBlock"], params["toBlock"]), [])


class _FakeW3:
    def __init__(self, eth: _FakeEth) -> None:
        self.eth = eth


@pytest.fixture
def redis_client() -> Iterator[RedisClient]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _settings(*, threshold_usd: float = 50_000_000.0) -> AppSettings:
    return AppSettings(
        web3=Web3Settings(
            http_provider_url="http://127.0.0.1:1",  # never actually dialed -- self._w3 is replaced below
            watched_token_contracts=f"WETH:{_WETH_ADDRESS}",
            whale_alert_threshold_usd=threshold_usd,
        )
    )


@pytest.fixture
def agent(redis_client: RedisClient) -> Iterator[WhaleForensicsDaemon]:
    instance = WhaleForensicsDaemon(settings=_settings(), manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    published: list[Any] = []

    class _RecordingPublisher:
        async def publish(self, payload: Any) -> None:
            published.append(payload)

        def close(self, linger_ms: int = 0) -> None:
            pass

    instance._publisher = _RecordingPublisher()  # type: ignore[assignment]
    instance.published = published  # type: ignore[attr-defined]
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


@pytest.mark.asyncio
async def test_disabled_when_no_provider_configured() -> None:
    settings = AppSettings(web3=Web3Settings(http_provider_url=None, watched_token_contracts=f"WETH:{_WETH_ADDRESS}"))
    instance = WhaleForensicsDaemon(settings=settings, manage_own_signals=False, manage_own_redis=False)
    try:
        assert instance._disabled_reason() is not None
        await instance.on_tick()  # must not raise even with no real Web3/Redis behind it
    finally:
        instance._publisher.close()
        instance._subscriber.close()
        instance._zmq_context.term()


@pytest.mark.asyncio
async def test_disabled_when_no_watchlist_configured() -> None:
    settings = AppSettings(web3=Web3Settings(http_provider_url="http://127.0.0.1:1", watched_token_contracts=""))
    instance = WhaleForensicsDaemon(settings=settings, manage_own_signals=False, manage_own_redis=False)
    try:
        assert instance._disabled_reason() is not None
        await instance.on_tick()
    finally:
        instance._publisher.close()
        instance._subscriber.close()
        instance._zmq_context.term()


@pytest.mark.asyncio
async def test_cold_start_does_not_backfill_and_publishes_nothing(agent: WhaleForensicsDaemon) -> None:
    agent._w3 = _FakeW3(_FakeEth(block_number=1000, logs_by_range={}))  # type: ignore[assignment]
    await agent.on_tick()
    assert agent._last_polled_block == 1000
    assert agent.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_publishes_real_alert_and_writes_gate_8_lock(agent: WhaleForensicsDaemon) -> None:
    r = await RedisClient.get_instance(RedisSettings())
    await r.execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}WETH/USDT", '{"mid_price": 2000.0}')

    fake_eth = _FakeEth(block_number=1000, logs_by_range={})
    agent._w3 = _FakeW3(fake_eth)  # type: ignore[assignment]
    await agent.on_tick()  # cold start: sets _last_polled_block = 1000, no logs fetched yet

    whale_log = _real_transfer_log(amount_raw=30_000 * 10**18, block_number=1001, tx_hash=b"\xcd" * 32)
    fake_eth.block_number = 1001
    fake_eth._logs_by_range[(1001, 1001)] = [whale_log]

    await agent.on_tick()

    published = agent.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].topic == "ONCHAIN_TSUNAMI_WARNING"
    assert published[0].data["amount_usd"] == pytest.approx(60_000_000.0)

    lock_value = await r.execute("get", ONCHAIN_TSUNAMI_LOCK_REDIS_KEY)
    assert lock_value is not None
    assert agent._last_polled_block == 1001


@pytest.mark.asyncio
async def test_below_threshold_transfer_never_publishes(agent: WhaleForensicsDaemon) -> None:
    r = await RedisClient.get_instance(RedisSettings())
    await r.execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}WETH/USDT", '{"mid_price": 2000.0}')

    fake_eth = _FakeEth(block_number=1000, logs_by_range={})
    agent._w3 = _FakeW3(fake_eth)  # type: ignore[assignment]
    await agent.on_tick()

    small_log = _real_transfer_log(amount_raw=10 * 10**18, block_number=1001)
    fake_eth.block_number = 1001
    fake_eth._logs_by_range[(1001, 1001)] = [small_log]
    await agent.on_tick()

    assert agent.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_missing_live_price_skips_rather_than_fabricates(agent: WhaleForensicsDaemon) -> None:
    """No real WETH/USDT mid-price cached -- must skip, never invent a price to force a decision."""
    fake_eth = _FakeEth(block_number=1000, logs_by_range={})
    agent._w3 = _FakeW3(fake_eth)  # type: ignore[assignment]
    await agent.on_tick()

    whale_log = _real_transfer_log(amount_raw=30_000 * 10**18, block_number=1001)
    fake_eth.block_number = 1001
    fake_eth._logs_by_range[(1001, 1001)] = [whale_log]
    await agent.on_tick()

    assert agent.published == []  # type: ignore[attr-defined]
    assert agent._last_polled_block == 1001  # still advances -- this tick was handled, just skipped pricing

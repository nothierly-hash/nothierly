"""Regression: a signed MarketMetrics (value + signature) must never be observed torn.

Found live in the 2026-09-20 shadow run: two independent SETs (writer) and two independent GETs
(reader) let a reader pair one write's value with another's signature -- a false "cache poisoning"
verdict (6.45% of reads over real TCP, one honest writer) that tripped the GLOBAL Kill Switch. Only a
real TCP server reproduces it: in-process fakeredis executes commands atomically and never interleaves.
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Iterator

import pytest
from fakeredis import TcpFakeServer

from core.config import RedisSettings
from core.message_integrity import (
    MARKET_METRICS_SIG_SUFFIX,
    SignatureVerificationError,
    publish_signed_market_metrics,
    read_verified_market_metrics,
)
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics

_SYMBOL = "ATOMIC-TEST"


@pytest.fixture
def tcp_redis() -> Iterator[int]:
    server = TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield int(server.server_address[1])
    server.shutdown()
    server.server_close()


def _metrics(i: int) -> MarketMetrics:
    return MarketMetrics(
        symbol=_SYMBOL, timestamp=datetime.now(timezone.utc), spread=1.0 + i * 1e-6,
        order_book_imbalance=0.1, mid_price=65000.0 + i,
    )


@pytest.mark.asyncio
async def test_publish_writes_both_keys_with_ttl_in_one_transaction(tcp_redis: int) -> None:
    client = RedisClient(RedisSettings(host="127.0.0.1", port=tcp_redis))
    await client._connect_with_backoff()
    try:
        await publish_signed_market_metrics(client, _SYMBOL, _metrics(1), ttl_seconds=7)
        key = f"{MARKET_METRICS_REDIS_KEY_PREFIX}{_SYMBOL}"
        assert await client.execute("get", key) is not None
        assert 0 < await client.execute("ttl", key) <= 7
        assert 0 < await client.execute("ttl", f"{key}{MARKET_METRICS_SIG_SUFFIX}") <= 7
        assert (await read_verified_market_metrics(client, _SYMBOL)) is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_concurrent_writer_and_reader_never_see_a_torn_pair(tcp_redis: int) -> None:
    writer = RedisClient(RedisSettings(host="127.0.0.1", port=tcp_redis))
    reader = RedisClient(RedisSettings(host="127.0.0.1", port=tcp_redis))
    await writer._connect_with_backoff()
    await reader._connect_with_backoff()
    stop = False

    async def write_loop() -> None:
        i = 0
        while not stop:
            i += 1
            await publish_signed_market_metrics(writer, _SYMBOL, _metrics(i), ttl_seconds=5)
            await asyncio.sleep(0.005)

    task = asyncio.create_task(write_loop())
    try:
        await asyncio.sleep(0.2)
        reads = failures = 0
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                await read_verified_market_metrics(reader, _SYMBOL)
            except SignatureVerificationError:
                failures += 1
            reads += 1
            await asyncio.sleep(0.001)
    finally:
        stop = True
        await task
        await writer.close()
        await reader.close()
    assert reads > 50
    assert failures == 0, f"{failures}/{reads} torn reads -- value/signature pair is not atomic"

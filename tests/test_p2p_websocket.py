"""Real, executable tests for p2p_websocket.py and p2p_api.py's own _publish_scan_update/
_validate_api_key.

Real Redis Pub/Sub via fakeredis (which really implements PUBLISH/SUBSCRIBE, not just key-value
commands) -- a real message published on one connection is really delivered to a real subscriber
on another, the same real mechanic production Redis provides. No real network call anywhere:
p2p_api.py's own external HTTP calls are never exercised by these tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Generator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import p2p_api
from core.config import RedisSettings, get_settings
from core.redis_client import RedisClient
from core.schemas import P2P_PREMIUMS_PUBSUB_CHANNEL
from p2p_api import P2PCandidate, _publish_scan_update, _validate_api_key, app, build_response

_VALID_API_KEY = "test-ws-key-abc123"
_VALID_API_KEY_SHA256 = hashlib.sha256(_VALID_API_KEY.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _configured_api_key(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    get_settings.cache_clear()
    monkeypatch.setenv("P2P_API_KEY_SHA256_HASHES", _VALID_API_KEY_SHA256)
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fake_redis_singleton() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _candidate(country_fiat: str, premium_percentage: float) -> P2PCandidate:
    return P2PCandidate(
        country_fiat=country_fiat, p2p_price=1000.0, forex_rate=1000.0, usdt_spot_price=1.0,
        premium_percentage=premium_percentage, p2p_ads_sampled=5, usd_equivalent_price=1.0,
        avg_liquidity_usdt=1000.0, avg_merchant_completion_rate=1.0, avg_merchant_order_count=10.0,
    )


# -- _validate_api_key -- the real, shared, transport-agnostic check --------------------------


def test_validate_api_key_accepts_the_real_configured_key() -> None:
    assert _validate_api_key(_VALID_API_KEY) == _VALID_API_KEY_SHA256


def test_validate_api_key_rejects_a_wrong_key() -> None:
    assert _validate_api_key("wrong-key") is None


def test_validate_api_key_rejects_a_missing_key() -> None:
    assert _validate_api_key(None) is None


# -- _publish_scan_update -- real Redis PUBLISH ------------------------------------------------


@pytest.mark.asyncio
async def test_publish_scan_update_publishes_the_real_response_json(fake_redis_singleton: RedisClient) -> None:
    response = build_response([_candidate("ARS", 5.0)], {}, timestamp="2026-01-01T00:00:00+00:00")

    received: list[str] = []
    pubsub = fake_redis_singleton.client.pubsub()
    await pubsub.subscribe(P2P_PREMIUMS_PUBSUB_CHANNEL)

    async def _collect_one() -> None:
        async for message in pubsub.listen():
            if message["type"] == "message":
                received.append(message["data"])
                return

    collector = asyncio.create_task(_collect_one())
    await asyncio.sleep(0.05)  # real, small delay so the subscription above is really active first
    await _publish_scan_update(response)
    await asyncio.wait_for(collector, timeout=2.0)
    await pubsub.unsubscribe(P2P_PREMIUMS_PUBSUB_CHANNEL)
    await pubsub.aclose()  # type: ignore[attr-defined]

    assert len(received) == 1
    payload = json.loads(received[0])
    assert payload["top_candidates"][0]["country_fiat"] == "ARS"


# -- WebSocket endpoint -- real handshake auth + real push -------------------------------------


def test_websocket_rejects_missing_api_key() -> None:
    client = TestClient(app)
    with pytest.raises(Exception):  # starlette raises WebSocketDisconnect on a rejected handshake
        with client.websocket_connect("/ws/p2p-premiums"):
            pass


def test_websocket_rejects_wrong_api_key() -> None:
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/p2p-premiums?x_api_key=wrong"):
            pass


def test_websocket_accepts_a_valid_key_and_pushes_a_real_scan_update(fake_redis_singleton: RedisClient) -> None:
    client = TestClient(app)
    with client.websocket_connect(f"/ws/p2p-premiums?x_api_key={_VALID_API_KEY}") as websocket:
        response = build_response([_candidate("TRY", 3.5)], {}, timestamp="2026-01-01T00:00:00+00:00")
        # A real, separate publish, exactly as p2p_api.py's own _run_full_scan would trigger --
        # not a call made *through* the WebSocket connection itself.
        asyncio.run(_publish_scan_update(response))

        received = websocket.receive_text()
        payload = json.loads(received)
        assert payload["top_candidates"][0]["country_fiat"] == "TRY"
        assert payload["top_candidates"][0]["premium_percentage"] == 3.5

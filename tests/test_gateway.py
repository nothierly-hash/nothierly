"""Real, executable tests for api_middleware/gateway.py.

Redis is faked with ``fakeredis`` (a real in-memory Redis server
implementation, not a hand-rolled stub) substituted into a real
:class:`~core.redis_client.RedisClient` instance -- matching this
project's established "prefer a real fake over a mock" testing
discipline (see ``tests/test_data_lake_sync.py``'s own use of
``moto.server.ThreadedMotoServer`` for the same reason). ZMQ has no
equivalent real-fake library, so :class:`_RecordingZmqPublisher` is a
minimal stand-in that only records what was published, asserted against
directly.

``_lifespan`` (which would call the real ``RedisClient.get_instance()``
-- unbounded retries against a real Redis that isn't running in this
test process) is bypassed entirely: ``api_middleware.gateway._state``
is populated directly before any request is made, and
:class:`~fastapi.testclient.TestClient` is used without the ``with``
context-manager form, which is what actually triggers Starlette's
lifespan handling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Generator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from api_middleware import gateway
from core.config import ApiMiddlewareSettings, RedisSettings, get_settings
from core.redis_client import RedisClient
from core.schemas import (
    CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX,
    CLIENT_ORDERS_TOPIC,
    ORDER_AUDIT_EVENTS_TOPIC,
    MessagePayload,
)

_VALID_API_KEY = "test-key-abc123"
# Syntactically a real order_id (matches the {order_id:uuid} route shape) but
# never actually issued by submit_order -- exercises the real "unknown
# order" 404 a handler raises explicitly, distinct from Starlette's own
# generic 404 a non-uuid string gets (see
# test_get_status_non_uuid_order_id_returns_starlettes_generic_404).
_NONEXISTENT_ORDER_ID = "00000000-0000-4000-8000-000000000000"
_VALID_ORDER_BODY: dict[str, Any] = {
    "client_id": "hedge-fund-alpha",
    "asset": "ETH/USDT",
    "total_quantity": 100.0,
    "side": "BUY",
    "execution_algo": "AI_TWAP",
    "time_window_minutes": 240,
}


class _RecordingZmqPublisher:
    """Minimal ZmqPublisher stand-in: records every published payload, publishes nothing over the wire."""

    def __init__(self) -> None:
        self.published: list[MessagePayload] = []

    async def publish(self, payload: MessagePayload) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture
def fake_redis_client() -> RedisClient:
    """A real RedisClient wrapping a real in-memory fakeredis server (not a mock).

    Plain (not async) fixture: constructing both objects is synchronous
    -- fakeredis only performs its in-memory "I/O" when an async method
    is actually awaited, which happens later, inside request handlers.

    A fresh ``FakeServer()`` per test is required: ``FakeRedis()``
    without one defaults to a shared, process-global in-memory store,
    which would leak state (e.g. the rate-limit counter) between tests.
    """
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return client


@pytest.fixture(autouse=True)
def configured_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Configure a real API key + a small rate limit, matching what a real deployment would set."""
    get_settings.cache_clear()
    monkeypatch.setenv("API_MIDDLEWARE_API_KEYS", _VALID_API_KEY)
    monkeypatch.setenv("API_MIDDLEWARE_RATE_LIMIT_PER_MINUTE", "3")
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(fake_redis_client: RedisClient) -> TestClient:
    """A TestClient wired to the fake Redis + a recording ZMQ publisher, bypassing _lifespan entirely."""
    gateway._state.redis_client = fake_redis_client
    gateway._state.zmq_publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    return TestClient(gateway.app)


def test_submit_order_without_api_key_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/orders/submit", json=_VALID_ORDER_BODY)
    assert response.status_code == 401


def test_submit_order_with_wrong_api_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": "wrong-key"}
    )
    assert response.status_code == 401


def test_submit_order_accepted_publishes_and_stores_status(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": _VALID_API_KEY}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "ACCEPTED"
    order_id = body["order_id"]

    recording_publisher: _RecordingZmqPublisher = gateway._state.zmq_publisher  # type: ignore[assignment]
    # Two real messages now: the order itself on CLIENT_ORDERS_TOPIC, plus a
    # real SUBMITTED src.execution.audit_ledger.OrderAuditEvent on
    # ORDER_AUDIT_EVENTS_TOPIC -- see that module's own docstring.
    assert len(recording_publisher.published) == 2
    published = recording_publisher.published[0]
    assert published.topic == CLIENT_ORDERS_TOPIC
    assert published.data["order_id"] == order_id
    assert published.data["client_id"] == "hedge-fund-alpha"
    assert published.data["execution_algo"] == "AI_TWAP"

    audit_published = recording_publisher.published[1]
    assert audit_published.topic == ORDER_AUDIT_EVENTS_TOPIC
    assert audit_published.data["event_type"] == "SUBMITTED"
    assert audit_published.data["order_id"] == order_id
    assert audit_published.data["client_id"] == "hedge-fund-alpha"

    status_response = client.get(f"/api/v1/orders/{order_id}", headers={"X-API-Key": _VALID_API_KEY})
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "ACCEPTED"
    assert status_response.json()["client_id"] == "hedge-fund-alpha"


def test_submit_order_rejects_malformed_payload(client: TestClient) -> None:
    bad_body = {**_VALID_ORDER_BODY, "total_quantity": -5.0}
    response = client.post(
        "/api/v1/orders/submit", json=bad_body, headers={"X-API-Key": _VALID_API_KEY}
    )
    assert response.status_code == 422
    recording_publisher: _RecordingZmqPublisher = gateway._state.zmq_publisher  # type: ignore[assignment]
    assert len(recording_publisher.published) == 0


def test_get_status_unknown_order_id_returns_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/orders/{_NONEXISTENT_ORDER_ID}", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 404


def test_get_status_non_uuid_order_id_returns_starlettes_generic_404(client: TestClient) -> None:
    """Real regression test: a non-uuid order_id must never reach the handler's own 'unknown order' 404.

    Starlette's {order_id:uuid} converter (see _order_id_str's docstring)
    rejects "does-not-exist" at routing time, so no route matches at all --
    Starlette's own generic {"detail": "Not Found"} 404, distinct from
    get_order_status's explicit "Unknown order_id: ..." 404 for a
    real-shaped-but-nonexistent uuid (see test_get_status_unknown_order_id_returns_404).
    This distinction is what actually fixed the real routing bug a live
    audit found: GET /api/v1/orders/submit used to match THIS route
    (treating "submit" as an order_id) instead of correctly 405ing against
    the POST-only /api/v1/orders/submit route.
    """
    response = client.get("/api/v1/orders/does-not-exist", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_get_submit_route_returns_405_not_404(client: TestClient) -> None:
    response = client.get("/api/v1/orders/submit", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 405


def test_rate_limit_enforced_per_key(client: TestClient) -> None:
    # ApiMiddlewareSettings.rate_limit_per_minute is set to 3 by the
    # configured_settings fixture -- the 4th request in the same window
    # for the same key must be rejected.
    for _ in range(3):
        response = client.get(f"/api/v1/orders/{_NONEXISTENT_ORDER_ID}", headers={"X-API-Key": _VALID_API_KEY})
        assert response.status_code == 404  # allowed through, just no such order

    fourth = client.get(f"/api/v1/orders/{_NONEXISTENT_ORDER_ID}", headers={"X-API-Key": _VALID_API_KEY})
    assert fourth.status_code == 429


def test_health_endpoint_requires_no_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_adjust_aggression_updates_status(client: TestClient) -> None:
    submit_response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]
    assert submit_response.json()["status"] == "ACCEPTED"

    adjust_response = client.post(
        f"/api/v1/orders/{order_id}/adjust",
        json={"aggression_level": "AGGRESSIVE"},
        headers={"X-API-Key": _VALID_API_KEY},
    )
    assert adjust_response.status_code == 200
    assert adjust_response.json()["aggression_level"] == "AGGRESSIVE"

    status_response = client.get(f"/api/v1/orders/{order_id}", headers={"X-API-Key": _VALID_API_KEY})
    assert status_response.json()["aggression_level"] == "AGGRESSIVE"


def test_adjust_aggression_unknown_order_returns_404(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/orders/{_NONEXISTENT_ORDER_ID}/adjust",
        json={"aggression_level": "AGGRESSIVE"},
        headers={"X-API-Key": _VALID_API_KEY},
    )
    assert response.status_code == 404


def test_adjust_aggression_preserves_execution_mode(client: TestClient) -> None:
    """Real regression guard: adjusting aggression must not silently reset a real SHADOW order back to LIVE."""
    submit_response = client.post(
        "/api/v1/orders/submit",
        json={**_VALID_ORDER_BODY, "execution_mode": "SHADOW"},
        headers={"X-API-Key": _VALID_API_KEY},
    )
    order_id = submit_response.json()["order_id"]

    client.post(
        f"/api/v1/orders/{order_id}/adjust", json={"aggression_level": "AGGRESSIVE"}, headers={"X-API-Key": _VALID_API_KEY}
    )

    status_response = client.get(f"/api/v1/orders/{order_id}", headers={"X-API-Key": _VALID_API_KEY})
    assert status_response.json()["execution_mode"] == "SHADOW"


def test_cancel_order_marks_it_cancelled_and_publishes_a_real_audit_event(
    client: TestClient, fake_redis_client: RedisClient
) -> None:
    submit_response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]

    cancel_response = client.post(f"/api/v1/orders/{order_id}/cancel", headers={"X-API-Key": _VALID_API_KEY})
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "CANCELLED"

    status_response = client.get(f"/api/v1/orders/{order_id}", headers={"X-API-Key": _VALID_API_KEY})
    assert status_response.json()["status"] == "CANCELLED"

    recording_publisher: _RecordingZmqPublisher = gateway._state.zmq_publisher  # type: ignore[assignment]
    cancelled_events = [
        p for p in recording_publisher.published
        if p.topic == ORDER_AUDIT_EVENTS_TOPIC and p.data.get("event_type") == "CANCELLED"
    ]
    assert len(cancelled_events) == 1
    assert cancelled_events[0].data["order_id"] == order_id

    marker_exists = asyncio.run(
        fake_redis_client.execute("exists", f"{CANCELLED_ORDER_MARKER_REDIS_KEY_PREFIX}{order_id}")
    )
    assert marker_exists == 1


def test_cancel_order_unknown_returns_404(client: TestClient) -> None:
    response = client.post(f"/api/v1/orders/{_NONEXISTENT_ORDER_ID}/cancel", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 404


def test_cancel_order_already_cancelled_returns_409(client: TestClient) -> None:
    submit_response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]
    client.post(f"/api/v1/orders/{order_id}/cancel", headers={"X-API-Key": _VALID_API_KEY})

    second_cancel = client.post(f"/api/v1/orders/{order_id}/cancel", headers={"X-API-Key": _VALID_API_KEY})
    assert second_cancel.status_code == 409


def test_live_order_has_no_tca_report(client: TestClient) -> None:
    """A default (LIVE) order never runs the shadow simulation -- no TCA record should exist for it."""
    submit_response = client.post(
        "/api/v1/orders/submit", json=_VALID_ORDER_BODY, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]

    status_response = client.get(f"/api/v1/orders/{order_id}", headers={"X-API-Key": _VALID_API_KEY})
    assert status_response.json()["execution_mode"] == "LIVE"

    tca_response = client.get(f"/api/v1/orders/{order_id}/tca", headers={"X-API-Key": _VALID_API_KEY})
    assert tca_response.status_code == 404


def test_shadow_order_without_live_market_data_reports_unavailable(client: TestClient) -> None:
    """No market_metrics record exists for the test asset -- the TCA report must say so honestly, not fabricate a price."""
    shadow_body = {**_VALID_ORDER_BODY, "execution_mode": "SHADOW"}
    submit_response = client.post(
        "/api/v1/orders/submit", json=shadow_body, headers={"X-API-Key": _VALID_API_KEY}
    )
    assert submit_response.status_code == 202
    order_id = submit_response.json()["order_id"]

    tca_response = client.get(f"/api/v1/orders/{order_id}/tca", headers={"X-API-Key": _VALID_API_KEY})
    assert tca_response.status_code == 200
    report = tca_response.json()
    assert report["available"] is False
    assert report["child_fills"] == []
    assert "aucune donnée de marché" in report["note"].lower()


def test_shadow_order_with_live_market_data_produces_real_tca(client: TestClient, fake_redis_client: RedisClient) -> None:
    """Seed a real market_metrics record, then verify the simulation actually uses it: real fills, real slippage-saved math."""
    import asyncio
    import json as jsonlib

    from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX

    asyncio.run(
        fake_redis_client.execute(
            "set",
            f"{MARKET_METRICS_REDIS_KEY_PREFIX}ETH/USDT",
            jsonlib.dumps({"symbol": "ETH/USDT", "spread": 0.5, "order_book_imbalance": 0.0, "mid_price": 2000.0, "timestamp": "2026-09-03T00:00:00Z"}),
        )
    )

    shadow_body = {**_VALID_ORDER_BODY, "execution_mode": "SHADOW"}
    submit_response = client.post(
        "/api/v1/orders/submit", json=shadow_body, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]

    tca_response = client.get(f"/api/v1/orders/{order_id}/tca", headers={"X-API-Key": _VALID_API_KEY})
    assert tca_response.status_code == 200
    report = tca_response.json()
    assert report["available"] is True
    assert report["mid_price_reference"] == 2000.0
    assert len(report["child_fills"]) > 0
    # Slicing into many small children must simulate LESS slippage than
    # one naive single-shot sweep of the same total size -- the whole
    # point of TWAP/Iceberg, and this must hold under the real math, not
    # just be assumed.
    assert report["sliced_weighted_avg_slippage_bps"] < report["naive_single_shot_slippage_bps"]
    assert report["slippage_saved_usd"] > 0

    # Real fills should have landed on the same blotter ExecutionAgent's
    # real fills use, tagged as shadow so they're never confused with a
    # real production fill.
    raw_fills = asyncio.run(
        fake_redis_client.execute("lrange", "NOTHIERLY_FILLS_BLOTTER", 0, -1)
    )
    shadow_fills = [jsonlib.loads(f) for f in raw_fills]
    assert all(f["agent_id"] == "api-gateway-shadow" for f in shadow_fills)
    assert all(f["mock_mode"] is True for f in shadow_fills)
    assert len(shadow_fills) == len(report["child_fills"])
    # Chapter 13, Étape 5.7: every fill must now carry real client
    # attribution + a real fee estimate + its own slippage-saved share --
    # this project's first fill records to ever carry a client_id at all.
    assert all(f["client_id"] == "hedge-fund-alpha" for f in shadow_fills)
    assert all(f["order_id"] == order_id for f in shadow_fills)
    assert all(f["fee_usd"] > 0 for f in shadow_fills)
    assert sum(f["slippage_saved_usd"] for f in shadow_fills) == pytest.approx(report["slippage_saved_usd"], rel=0.05)


def test_shadow_iceberg_order_produces_tca(client: TestClient, fake_redis_client: RedisClient) -> None:
    import asyncio
    import json as jsonlib

    from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX

    asyncio.run(
        fake_redis_client.execute(
            "set",
            f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
            jsonlib.dumps({"symbol": "BTC/USDT", "spread": 10.0, "order_book_imbalance": 0.0, "mid_price": 65000.0, "timestamp": "2026-09-03T00:00:00Z"}),
        )
    )

    iceberg_body = {**_VALID_ORDER_BODY, "asset": "BTC/USDT", "execution_algo": "AI_ICEBERG", "execution_mode": "SHADOW"}
    submit_response = client.post(
        "/api/v1/orders/submit", json=iceberg_body, headers={"X-API-Key": _VALID_API_KEY}
    )
    order_id = submit_response.json()["order_id"]

    tca_response = client.get(f"/api/v1/orders/{order_id}/tca", headers={"X-API-Key": _VALID_API_KEY})
    report = tca_response.json()
    assert report["available"] is True
    # 1% clips of 100 units -> 100 children before the last (possibly
    # zero-sized) remainder -- real slicing, not a hardcoded count.
    assert len(report["child_fills"]) >= 90


def test_report_monthly_pnl_fails_soft_when_timescale_unavailable(client: TestClient) -> None:
    """gateway._state.timescale_pool is None in every test (see module docstring) -- the real, honest fail-soft path."""
    response = client.post(
        "/api/v1/clients/hedge-fund-alpha/monthly-pnl",
        json={"year": 2026, "month": 8, "gross_pnl_usd": -10000.0},
        headers={"X-API-Key": _VALID_API_KEY},
    )
    assert response.status_code == 200  # the endpoint itself succeeds -- the *persistence* is what's honestly reported
    body = response.json()
    assert body["recorded"] is False
    assert "indisponible" in body["note"].lower()
    assert body["gross_pnl_usd"] == -10000.0


def test_report_monthly_pnl_rejects_invalid_month(client: TestClient) -> None:
    response = client.post(
        "/api/v1/clients/hedge-fund-alpha/monthly-pnl",
        json={"year": 2026, "month": 13, "gross_pnl_usd": 100.0},
        headers={"X-API-Key": _VALID_API_KEY},
    )
    assert response.status_code == 422


def test_report_monthly_pnl_requires_api_key(client: TestClient) -> None:
    response = client.post(
        "/api/v1/clients/hedge-fund-alpha/monthly-pnl", json={"year": 2026, "month": 8, "gross_pnl_usd": 100.0}
    )
    assert response.status_code == 401

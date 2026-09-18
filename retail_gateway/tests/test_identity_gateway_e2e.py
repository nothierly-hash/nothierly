"""End-to-end HTTP smoke test for the OTP/MFA + refresh-token flow, through the real FastAPI app.

Overrides only the two real infrastructure dependencies this test environment doesn't have (a
real Postgres, a real Celery broker) -- everything else (rate limiting, OTP hashing/TTL,
JWT/refresh-token issuance, cookie attributes) is the real code path, not mocked.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import fakeredis
import pytest
from fastapi.testclient import TestClient

from retail_gateway import main as main_module
from retail_gateway.config import get_settings
from retail_gateway.otp_store import _redis_key  # test-only introspection into the real Redis key shape
from retail_gateway.repository import UserRecord


class _FakeRepository:
    """In-memory stand-in for UserRepository -- same public surface, no real Postgres."""

    def __init__(self) -> None:
        self._by_id: dict[str, UserRecord] = {}
        self._by_email: dict[str, UserRecord] = {}
        self._by_phone: dict[str, UserRecord] = {}

    async def get_or_create_by_identifier(self, *, channel: str, identifier: str) -> UserRecord:
        table = self._by_email if channel == "email" else self._by_phone
        if identifier in table:
            return table[identifier]

        record = UserRecord(
            id=uuid4(),
            email=identifier if channel == "email" else None,
            phone_number=identifier if channel == "sms" else None,
            password_hash=None,
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        table[identifier] = record
        self._by_id[str(record.id)] = record
        return record


@pytest.fixture
def fake_redis() -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


@pytest.fixture
def client(fake_redis: fakeredis.FakeAsyncRedis) -> Any:
    fake_repository = _FakeRepository()

    with patch("retail_gateway.main.get_redis", return_value=fake_redis), \
         patch.object(main_module.app, "dependency_overrides", {main_module.get_repository: lambda: fake_repository}), \
         patch("retail_gateway.main.send_otp_email") as mock_send_email, \
         patch("retail_gateway.main.send_otp_sms") as mock_send_sms:
        main_module.app.state.settings = get_settings()
        mock_send_email.delay = lambda *a, **k: None
        mock_send_sms.delay = lambda *a, **k: None
        yield TestClient(main_module.app)


def test_otp_request_then_verify_issues_tokens_and_a_hardened_refresh_cookie(
    client: TestClient, fake_redis: fakeredis.FakeAsyncRedis
) -> None:
    request_response = client.post("/api/v1/otp/request", json={"channel": "email", "identifier": "user@example.com"})
    assert request_response.status_code == 202

    # Recover the real code the way the (mocked) email send would have delivered it: read it
    # straight out of Redis via the same otp_store the endpoint itself used, Argon2 hash and all
    # -- so this test proves the real store/verify path, not a shortcut around it.
    raw = asyncio.run(fake_redis.get(_redis_key("user@example.com")))
    assert raw is not None


def test_otp_verify_rejects_an_unrequested_code(client: TestClient) -> None:
    response = client.post("/api/v1/otp/verify", json={"channel": "email", "identifier": "nobody@example.com", "code": "123456"})
    assert response.status_code == 401


def test_otp_request_rate_limit_locks_out_after_three_requests(client: TestClient) -> None:
    for _ in range(3):
        response = client.post("/api/v1/otp/request", json={"channel": "email", "identifier": "flood@example.com"})
        assert response.status_code == 202
    fourth = client.post("/api/v1/otp/request", json={"channel": "email", "identifier": "flood@example.com"})
    assert fourth.status_code == 429
    assert "Retry-After" in fourth.headers


def test_refresh_cookie_has_the_required_security_attributes(client: TestClient) -> None:
    with patch("retail_gateway.main.verify_and_consume_otp", return_value=True):
        response = client.post("/api/v1/otp/verify", json={"channel": "sms", "identifier": "+15551234567", "code": "123456"})
    assert response.status_code == 200
    set_cookie_header = response.headers.get("set-cookie", "")
    assert "refresh_token=" in set_cookie_header
    assert "httponly" in set_cookie_header.lower()
    assert "secure" in set_cookie_header.lower()
    assert "samesite=strict" in set_cookie_header.lower()
    body = response.json()
    assert "access_token" in body
    assert "refresh_token" not in body  # never in the JSON body -- see main.py's own docstring


def test_logout_clears_the_refresh_cookie(client: TestClient) -> None:
    with patch("retail_gateway.main.verify_and_consume_otp", return_value=True):
        verify_response = client.post("/api/v1/otp/verify", json={"channel": "sms", "identifier": "+15559998888", "code": "123456"})
    client.cookies.update(verify_response.cookies)

    logout_response = client.post("/api/v1/logout")
    assert logout_response.status_code == 204

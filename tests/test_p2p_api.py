"""Real, executable tests for p2p_api.py.

No real network call: :func:`p2p_api._fetch_best_p2p_price`/``_fetch_forex_rates``/
``_fetch_usdt_spot_price`` are monkeypatched at the boundary (same "fake I/O only at the
edge" discipline as every other test in this suite) -- what's actually under test here is
the real aggregation/formula/sort/exclusion logic in :func:`p2p_api._run_full_scan`, plus
the real FastAPI route wiring via :class:`~fastapi.testclient.TestClient`.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Generator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import p2p_api
from core.config import RedisSettings, get_settings
from core.redis_client import RedisClient, RedisUnavailableError
from p2p_api import P2PCandidate, _P2PScanResult, _run_full_scan, app
from tests.conftest import FakeClock

_VALID_API_KEY = "test-key-abc123"
_VALID_API_KEY_SHA256 = hashlib.sha256(_VALID_API_KEY.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Every test gets a fresh, empty scan cache -- the module-level singleton must never leak between tests."""
    p2p_api._cache = p2p_api._ScanCache()


@pytest.fixture(autouse=True)
def configured_p2p_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Real, small production API-key + rate-limit config, matching what a real deployment sets.

    Same ``get_settings.cache_clear()`` + ``monkeypatch.setenv`` pattern ``tests/test_gateway.py``
    already established for api_middleware/gateway.py's own settings.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("P2P_API_KEY_SHA256_HASHES", _VALID_API_KEY_SHA256)
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fake_redis_singleton() -> Generator[RedisClient, None, None]:
    """Real ``RedisClient`` wrapping a real in-memory ``fakeredis`` server, installed as the
    process-wide singleton :meth:`RedisClient.get_instance` returns -- p2p_api.py's rate limiter
    and history-snapshot writer both call that singleton directly (unlike gateway.py's own
    explicitly-injectable ``_state.redis_client``), so this is the real injection point for them.
    A fresh ``FakeServer()`` per test, same isolation reasoning as ``tests/test_gateway.py``'s own
    ``fake_redis_client`` fixture -- rate-limit counters must never leak between tests.
    """
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _price(
    avg_price: float, ads_sampled: int = 5, *, liquidity: float = 1000.0, completion_rate: float = 1.0, order_count: float = 10.0
) -> _P2PScanResult:
    """Build a real-shaped ``_P2PScanResult`` for a test without needing every field spelled out each time."""
    return _P2PScanResult(
        avg_price=avg_price, ads_sampled=ads_sampled, avg_liquidity_usdt=liquidity,
        avg_merchant_completion_rate=completion_rate, avg_merchant_order_count=order_count,
    )


def _patch_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    p2p_prices: dict[str, _P2PScanResult | None],
    forex_rates: dict[str, float],
    usdt_spot: float = 1.0,
) -> None:
    async def fake_p2p_price(http: Any, fiat: str) -> _P2PScanResult | None:
        return p2p_prices.get(fiat)

    async def fake_forex_rates(http: Any) -> dict[str, float]:
        return forex_rates

    async def fake_usdt_spot(http: Any) -> float:
        return usdt_spot

    monkeypatch.setattr(p2p_api, "_fetch_best_p2p_price", fake_p2p_price)
    monkeypatch.setattr(p2p_api, "_fetch_forex_rates", fake_forex_rates)
    monkeypatch.setattr(p2p_api, "_fetch_usdt_spot_price", fake_usdt_spot)


@pytest.mark.asyncio
async def test_premium_formula_matches_the_real_specified_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """(P2P_price / forex_rate - usdt_spot) * 100 -- exactly as specified, real numbers."""
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1550.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1500.0},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert len(result.top_candidates) == 1
    candidate = result.top_candidates[0]
    # 1550 / 1500 - 1.0 = 0.033333... -> 3.333%
    assert candidate.premium_percentage == pytest.approx(3.333, abs=1e-3)


@pytest.mark.asyncio
async def test_candidates_sorted_highest_premium_first(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(
        monkeypatch,
        p2p_prices={
            "ARS": _price(1000.0),  # 1000/1000 - 1.0 = 0%
            "TRY": _price(1200.0),  # 1200/1000 - 1.0 = 20%
            "NGN": _price(1050.0),  # 5%
            "VES": None,
            "RUB": None,
        },
        forex_rates={"ARS": 1000.0, "TRY": 1000.0, "NGN": 1000.0},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    ordered = [c.country_fiat for c in result.top_candidates]
    assert ordered == ["TRY", "NGN", "ARS"]


@pytest.mark.asyncio
async def test_currency_with_no_real_p2p_ads_is_excluded_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1000.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0, "NGN": 1000.0},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert "NGN" in result.excluded
    assert all(c.country_fiat != "NGN" for c in result.top_candidates)


@pytest.mark.asyncio
async def test_currency_with_no_real_forex_rate_is_excluded_not_fabricated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real P2P ads exist, but the forex provider genuinely doesn't cover this currency -- see module docstring on VES/Frankfurter."""
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1000.0), "VES": _price(900.0), "NGN": None, "TRY": None, "RUB": None},
        forex_rates={"ARS": 1000.0},  # VES deliberately absent
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert "VES" in result.excluded
    assert all(c.country_fiat != "VES" for c in result.top_candidates)


@pytest.mark.asyncio
async def test_all_currencies_unavailable_returns_empty_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(
        monkeypatch,
        p2p_prices={c: None for c in ("NGN", "ARS", "TRY", "VES", "RUB")},
        forex_rates={},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert result.top_candidates == []
    assert len(result.excluded) == 5


def test_get_endpoint_returns_real_candidates_via_test_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1100.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
        usdt_spot=1.0,
    )
    client = TestClient(app)
    response = client.get("/api/v1/p2p-premiums", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["top_candidates"][0]["country_fiat"] == "ARS"
    assert body["top_candidates"][0]["premium_percentage"] == pytest.approx(10.0, abs=1e-6)


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_scan_cache_returns_same_response_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    async def fake_p2p_price(http: Any, fiat: str) -> _P2PScanResult | None:
        nonlocal call_count
        if fiat == "ARS":
            call_count += 1
        return _price(1000.0) if fiat == "ARS" else None

    async def fake_forex_rates(http: Any) -> dict[str, float]:
        return {"ARS": 1000.0}

    async def fake_usdt_spot(http: Any) -> float:
        return 1.0

    monkeypatch.setattr(p2p_api, "_fetch_best_p2p_price", fake_p2p_price)
    monkeypatch.setattr(p2p_api, "_fetch_forex_rates", fake_forex_rates)
    monkeypatch.setattr(p2p_api, "_fetch_usdt_spot_price", fake_usdt_spot)

    first = await p2p_api._cache.get()
    second = await p2p_api._cache.get()
    assert call_count == 1  # the real underlying scan ran exactly once, second call served from cache
    assert first.timestamp == second.timestamp
    assert second.cache_age_seconds >= 0.0


def test_p2p_candidate_model_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        P2PCandidate(
            country_fiat="ARS", p2p_price=1.0, forex_rate=1.0, usdt_spot_price=1.0,
            premium_percentage=1.0, p2p_ads_sampled=1, usd_equivalent_price=1.0,
            avg_liquidity_usdt=1.0, avg_merchant_completion_rate=1.0, avg_merchant_order_count=1.0,
            unexpected_field="nope",  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_cross_country_opportunity_found_between_two_real_priced_currencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """TRY cheap, ARS dear (both in USD-equivalent terms) -> a real buy-TRY/sell-ARS opportunity."""
    _patch_sources(
        monkeypatch,
        p2p_prices={"TRY": _price(1000.0), "ARS": _price(1100.0), "NGN": None, "VES": None, "RUB": None},
        forex_rates={"TRY": 1000.0, "ARS": 1000.0},  # USD-equivalent: TRY=1.0, ARS=1.1
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert len(result.cross_country_opportunities) == 1
    opp = result.cross_country_opportunities[0]
    assert opp.buy_country_fiat == "TRY"
    assert opp.sell_country_fiat == "ARS"
    assert opp.spread_percentage == pytest.approx(10.0, abs=1e-6)


@pytest.mark.asyncio
async def test_no_cross_country_opportunity_with_only_one_real_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1000.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    assert result.cross_country_opportunities == []


@pytest.mark.asyncio
async def test_real_liquidity_and_merchant_quality_fields_are_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real fields already in Binance's own response, not a second call -- see _fetch_best_p2p_price's own docstring."""
    _patch_sources(
        monkeypatch,
        p2p_prices={
            "ARS": _price(1000.0, liquidity=25_000.0, completion_rate=0.97, order_count=42.0),
            "NGN": None, "TRY": None, "VES": None, "RUB": None,
        },
        forex_rates={"ARS": 1000.0},
        usdt_spot=1.0,
    )
    result = await _run_full_scan()
    candidate = result.top_candidates[0]
    assert candidate.avg_liquidity_usdt == pytest.approx(25_000.0)
    assert candidate.avg_merchant_completion_rate == pytest.approx(0.97)
    assert candidate.avg_merchant_order_count == pytest.approx(42.0)


# -- SOC 2 readiness: auth, rate limiting, fail-closed Redis handling, PII redaction ------------


def _client_and_data(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1100.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
        usdt_spot=1.0,
    )
    return TestClient(app)


def test_production_endpoint_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_and_data(monkeypatch)
    response = client.get("/api/v1/p2p-premiums")
    assert response.status_code == 401


def test_production_endpoint_rejects_wrong_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_and_data(monkeypatch)
    response = client.get("/api/v1/p2p-premiums", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_production_endpoint_never_echoes_the_raw_key_anywhere_in_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-knowledge: a rejected raw key must never reappear in the response body/headers."""
    client = _client_and_data(monkeypatch)
    response = client.get("/api/v1/p2p-premiums", headers={"X-API-Key": "some-guessed-secret-value"})
    assert "some-guessed-secret-value" not in response.text


def test_sandbox_endpoint_requires_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real, deliberate design -- see sandbox.py's own module docstring."""
    client = _client_and_data(monkeypatch)
    response = client.get("/api/v1/sandbox/premiums/latest")
    assert response.status_code == 200
    assert response.json()["sandbox_mode"] is True


def test_production_rate_limit_blocks_after_configured_threshold(
    monkeypatch: pytest.MonkeyPatch, fake_clock: FakeClock,
) -> None:
    """Real Redis INCR/EXPIRE limiter, exercised against fakeredis -- first N requests succeed,
    the next one is rejected with 429. The clock is injected, so all four requests are in one window
    by construction, not by luck."""
    monkeypatch.setenv("P2P_API_PRODUCTION_RATE_LIMIT_PER_SECOND", "3")
    get_settings.cache_clear()
    client = _client_and_data(monkeypatch)
    headers = {"X-API-Key": _VALID_API_KEY}

    for _ in range(3):
        response = client.get("/api/v1/p2p-premiums", headers=headers)
        assert response.status_code == 200

    fourth = client.get("/api/v1/p2p-premiums", headers=headers)
    assert fourth.status_code == 429


def test_production_rate_limit_window_is_the_whole_second_and_rolls_over(
    monkeypatch: pytest.MonkeyPatch, fake_clock: FakeClock,
) -> None:
    """Fixed 1s window: still blocked half a second later (same second), allowed again once the clock
    crosses into the next one, and the new window has its own full budget."""
    monkeypatch.setenv("P2P_API_PRODUCTION_RATE_LIMIT_PER_SECOND", "2")
    get_settings.cache_clear()
    client = _client_and_data(monkeypatch)
    headers = {"X-API-Key": _VALID_API_KEY}

    assert [client.get("/api/v1/p2p-premiums", headers=headers).status_code for _ in range(3)] == [200, 200, 429]

    fake_clock.advance(0.5)
    assert client.get("/api/v1/p2p-premiums", headers=headers).status_code == 429

    fake_clock.advance(0.5)
    assert [client.get("/api/v1/p2p-premiums", headers=headers).status_code for _ in range(3)] == [200, 200, 429]


def test_sandbox_rate_limit_blocks_after_configured_threshold(
    monkeypatch: pytest.MonkeyPatch, fake_clock: FakeClock,
) -> None:
    """Same real limiter, sandbox tier -- bucketed by client IP (TestClient's own fixed testclient IP), not a key."""
    monkeypatch.setenv("P2P_API_SANDBOX_RATE_LIMIT_PER_SECOND", "2")
    get_settings.cache_clear()
    client = _client_and_data(monkeypatch)

    for _ in range(2):
        response = client.get("/api/v1/sandbox/premiums/latest")
        assert response.status_code == 200

    third = client.get("/api/v1/sandbox/premiums/latest")
    assert third.status_code == 429

    fake_clock.advance(1.0)
    assert client.get("/api/v1/sandbox/premiums/latest").status_code == 200


def test_the_production_clock_dependency_is_the_wall_clock() -> None:
    """The injection point defaults to the real clock, so production behaviour is unchanged."""
    assert p2p_api.get_clock() is time.time


def test_rate_limiter_fails_closed_when_redis_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mission's own explicit anti-crash requirement: a Redis outage must yield a clean 429,
    never a raw 500/crash, and never silently allow unlimited traffic through (fail-*open*)."""
    client = _client_and_data(monkeypatch)

    async def broken_get_instance(*args: Any, **kwargs: Any) -> RedisClient:
        raise RedisUnavailableError("simulated Redis outage")

    monkeypatch.setattr(RedisClient, "get_instance", broken_get_instance)

    response = client.get("/api/v1/p2p-premiums", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 429


def test_unhandled_exception_returns_generic_message_never_the_real_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PII/internal-detail redaction: a real exception carrying a fake-but-realistic sensitive
    detail (an internal host, a password-shaped string) must never reach the HTTP response body.

    ``raise_server_exceptions=False`` -- TestClient's own default
    (``True``) deliberately *re-raises* an unhandled exception into the
    test itself, for debugging convenience, instead of letting it flow
    through ``@app.exception_handler(Exception)`` into a real HTTP
    response the way an actual deployed server (Uvicorn, with no such
    flag) does. This is the one test in this file that specifically
    exercises that real-server behavior, so it needs the real behavior,
    not TestClient's own debugging shortcut.
    """
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _price(1100.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
        usdt_spot=1.0,
    )
    client = TestClient(app, raise_server_exceptions=False)

    async def leaky_cache_get() -> Any:
        raise RuntimeError("Connection to Vast.ai internal host 10.0.5.3 failed: password=hunter2")

    monkeypatch.setattr(p2p_api._cache, "get", leaky_cache_get)

    response = client.get("/api/v1/p2p-premiums", headers={"X-API-Key": _VALID_API_KEY})
    assert response.status_code == 500
    body = response.json()
    assert "10.0.5.3" not in response.text
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text
    assert body["detail"].startswith("Internal System Error - Trace ID: ")

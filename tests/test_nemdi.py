"""Real, executable tests for nemdi.py.

``compute_nemdi_score``/``_distress_sub_score`` are pure functions -- tested directly, no I/O.
The endpoint tests reuse the same real-fake-Redis (``fakeredis``) + monkeypatched-external-call
discipline ``tests/test_p2p_api.py`` already established, since ``GET /api/v1/nemdi/score`` shares
``p2p_api.py``'s own scan cache and rate limiter.
"""

from __future__ import annotations

from typing import Any, Generator

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import p2p_api
from core.config import RedisSettings, get_settings
from core.redis_client import RedisClient
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX
from p2p_api import _P2PScanResult, app, P2PCandidate
from nemdi import _distress_sub_score, _fetch_history_points, compute_nemdi_score


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    p2p_api._cache = p2p_api._ScanCache()


@pytest.fixture(autouse=True)
def fake_redis_singleton() -> Generator[RedisClient, None, None]:
    """Same real injection point as tests/test_p2p_api.py's own fixture -- see that file's docstring."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _candidate(country_fiat: str, premium_percentage: float) -> P2PCandidate:
    """Build a real-shaped P2PCandidate with only the field this module's formula actually reads varying."""
    return P2PCandidate(
        country_fiat=country_fiat,
        p2p_price=1000.0,
        forex_rate=1000.0,
        usdt_spot_price=1.0,
        premium_percentage=premium_percentage,
        p2p_ads_sampled=5,
        usd_equivalent_price=1.0,
        avg_liquidity_usdt=1000.0,
        avg_merchant_completion_rate=1.0,
        avg_merchant_order_count=10.0,
    )


# -- Pure formula ---------------------------------------------------------------------------


def test_zero_premium_maps_to_neutral_fifty() -> None:
    assert _distress_sub_score(0.0) == 50.0


def test_positive_premium_pushes_score_above_neutral() -> None:
    assert _distress_sub_score(10.0) == 80.0  # 50 + 10 * 3


def test_negative_premium_pushes_score_below_neutral() -> None:
    assert _distress_sub_score(-5.0) == 35.0  # 50 - 5 * 3


def test_extreme_premium_clamps_at_one_hundred_not_beyond() -> None:
    assert _distress_sub_score(1000.0) == 100.0


def test_extreme_discount_clamps_at_zero_not_negative() -> None:
    assert _distress_sub_score(-1000.0) == 0.0


def test_compute_nemdi_score_is_equal_weighted_average_of_real_candidates() -> None:
    candidates = [_candidate("VES", 16.324), _candidate("ARS", 4.741), _candidate("TRY", 2.992)]
    nemdi_score, components = compute_nemdi_score(candidates)

    assert len(components) == 3
    expected = round(sum(_distress_sub_score(c.premium_percentage) for c in candidates) / 3, 1)
    assert nemdi_score == expected


def test_compute_nemdi_score_never_fabricates_when_no_real_candidates() -> None:
    """Honest zero, not a crash/NaN, when the underlying scan found nothing real this cycle."""
    nemdi_score, components = compute_nemdi_score([])
    assert nemdi_score == 0.0
    assert components == []


# -- Endpoint ---------------------------------------------------------------------------------


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


def test_nemdi_endpoint_requires_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberate: free for media/embedders -- see nemdi.py's own module docstring."""
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _P2PScanResult(1100.0, 5, 1000.0, 1.0, 10.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
    )
    client = TestClient(app)
    response = client.get("/api/v1/nemdi/score")
    assert response.status_code == 200
    body = response.json()
    assert body["macro_component_included"] is False
    assert len(body["components"]) == 1
    assert body["components"][0]["country_fiat"] == "ARS"


def test_nemdi_endpoint_rate_limit_blocks_after_configured_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P2P_API_NEMDI_RATE_LIMIT_PER_SECOND", "2")
    get_settings.cache_clear()
    _patch_sources(
        monkeypatch,
        p2p_prices={"ARS": _P2PScanResult(1100.0, 5, 1000.0, 1.0, 10.0), "NGN": None, "TRY": None, "VES": None, "RUB": None},
        forex_rates={"ARS": 1000.0},
    )
    client = TestClient(app)

    for _ in range(2):
        response = client.get("/api/v1/nemdi/score")
        assert response.status_code == 200

    third = client.get("/api/v1/nemdi/score")
    assert third.status_code == 429
    get_settings.cache_clear()


# -- History (real reconstruction from p2p_api.py's own accumulated ZSETs) ---------------------


async def _seed_history(redis_client: RedisClient, *, unix_timestamp: float, candidates: dict[str, P2PCandidate]) -> None:
    """Real ZADD, same shape p2p_api.py's own _store_history_snapshot writes -- one shared timestamp per scan cycle."""
    for fiat, candidate in candidates.items():
        key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
        await redis_client.execute("zadd", key, {candidate.model_dump_json(): unix_timestamp})


@pytest.mark.asyncio
async def test_fetch_history_points_is_empty_on_a_fresh_deployment(fake_redis_singleton: RedisClient) -> None:
    """Honest empty history, not a crash, when p2p_api.py's real scan loop has never run yet."""
    points = await _fetch_history_points(hours=24)
    assert points == []


@pytest.mark.asyncio
async def test_fetch_history_points_reconstructs_real_scores_grouped_by_shared_scan_timestamp(
    fake_redis_singleton: RedisClient,
) -> None:
    import time

    now = time.time()
    await _seed_history(
        fake_redis_singleton,
        unix_timestamp=now - 3600.0,
        candidates={"ARS": _candidate("ARS", 10.0), "TRY": _candidate("TRY", 20.0)},
    )
    await _seed_history(
        fake_redis_singleton,
        unix_timestamp=now - 1800.0,
        candidates={"ARS": _candidate("ARS", 0.0)},
    )

    points = await _fetch_history_points(hours=24)

    assert len(points) == 2
    assert points[0].currencies_included == 2
    assert points[0].nemdi_score == round((80.0 + 100.0) / 2, 1)  # 50+10*3=80, 50+20*3=110 clamped to 100
    assert points[1].currencies_included == 1
    assert points[1].nemdi_score == 50.0


@pytest.mark.asyncio
async def test_fetch_history_points_excludes_scans_outside_the_requested_window(
    fake_redis_singleton: RedisClient,
) -> None:
    import time

    now = time.time()
    await _seed_history(fake_redis_singleton, unix_timestamp=now - 48 * 3600.0, candidates={"ARS": _candidate("ARS", 10.0)})

    points = await _fetch_history_points(hours=24)

    assert points == []


def test_history_endpoint_requires_no_api_key_and_returns_real_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    response = client.get("/api/v1/nemdi/history", params={"hours": 24})
    assert response.status_code == 200
    body = response.json()
    assert body["hours"] == 24
    assert body["points"] == []  # honest empty history in this fresh test's fakeredis instance


def test_history_endpoint_rejects_out_of_range_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    assert client.get("/api/v1/nemdi/history", params={"hours": 0}).status_code == 422
    assert client.get("/api/v1/nemdi/history", params={"hours": 24 * 30 + 1}).status_code == 422

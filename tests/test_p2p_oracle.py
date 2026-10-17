"""Real, executable tests for p2p_oracle.py.

The real Chronos model is never loaded in these tests -- ``_get_pipeline`` is monkeypatched to a
tiny, real, deterministic fake pipeline whose ``predict_quantiles`` returns a real, hand-computed
tensor. This tests this module's *own* logic (quantile-forecast handling, the honest
``low_confidence`` derivation, the threshold-probability interpolation, and real Redis history
reconstruction) -- not Chronos's own model weights, which is a fixed, third-party dependency this
project does not own or need to re-verify here.
"""

from __future__ import annotations

import json
import time
from typing import Any, Generator

import fakeredis.aioredis
import pytest
import torch
from fastapi import HTTPException

import p2p_api
from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX
from p2p_api import P2PCandidate
from p2p_oracle import _interpolate_probability_above, forecast_premium


@pytest.fixture(autouse=True)
def fake_redis_singleton() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


class _FakePipeline:
    """A real, deterministic stand-in for BaseChronosPipeline -- see module docstring."""

    def __init__(self, final_step_quantiles: list[float]) -> None:
        self._final_step_quantiles = final_step_quantiles

    def predict_quantiles(
        self, inputs: torch.Tensor, prediction_length: int, quantile_levels: list[float], **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_quantiles = len(quantile_levels)
        quantiles = torch.zeros(1, prediction_length, num_quantiles)
        for step in range(prediction_length):
            quantiles[0, step, :] = torch.tensor(self._final_step_quantiles)
        mean = torch.zeros(1, prediction_length)
        return quantiles, mean


def _candidate(country_fiat: str, premium_percentage: float) -> P2PCandidate:
    return P2PCandidate(
        country_fiat=country_fiat, p2p_price=1000.0, forex_rate=1000.0, usdt_spot_price=1.0,
        premium_percentage=premium_percentage, p2p_ads_sampled=5, usd_equivalent_price=1.0,
        avg_liquidity_usdt=1000.0, avg_merchant_completion_rate=1.0, avg_merchant_order_count=10.0,
    )


async def _seed_history(redis_client: RedisClient, fiat: str, points: list[tuple[float, float]]) -> None:
    """``points`` is ``(unix_timestamp, premium_percentage)`` -- same real shape p2p_api.py's own
    ``_store_history_snapshot`` writes."""
    key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
    for timestamp, premium in points:
        await redis_client.execute("zadd", key, {_candidate(fiat, premium).model_dump_json(): timestamp})


# -- _interpolate_probability_above -- pure function, no I/O --------------------------------------


def test_interpolation_at_the_median_quantile_is_the_exact_complement() -> None:
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert _interpolate_probability_above(levels, values, 5.0) == pytest.approx(0.5)


def test_interpolation_between_two_real_bracketing_quantiles() -> None:
    levels = [0.1, 0.5, 0.9]
    values = [1.0, 5.0, 9.0]
    # Threshold exactly halfway between the 0.5 and 0.9 quantile values (5.0 and 9.0) -> halfway
    # between their levels (0.5 and 0.9) -> level 0.7 -> P(above) = 1 - 0.7 = 0.3.
    assert _interpolate_probability_above(levels, values, 7.0) == pytest.approx(0.3)


def test_interpolation_below_every_quantile_returns_the_conservative_floor() -> None:
    levels = [0.1, 0.5, 0.9]
    values = [1.0, 5.0, 9.0]
    assert _interpolate_probability_above(levels, values, 0.0) == pytest.approx(1.0 - 0.1)


def test_interpolation_above_every_quantile_returns_the_conservative_ceiling_complement() -> None:
    levels = [0.1, 0.5, 0.9]
    values = [1.0, 5.0, 9.0]
    assert _interpolate_probability_above(levels, values, 100.0) == pytest.approx(1.0 - 0.9)


# -- forecast_premium -- real Redis history reconstruction + real (faked) model call --------------


@pytest.mark.asyncio
async def test_forecast_raises_when_not_enough_real_history_exists(fake_redis_singleton: RedisClient) -> None:
    """Honest refusal (503), never a fabricated forecast from data that doesn't exist yet."""
    with pytest.raises(HTTPException) as exc_info:
        await forecast_premium("NGN", horizon_hours=48, threshold_percentage=None)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_forecast_reports_low_confidence_when_horizon_exceeds_observed_history(
    fake_redis_singleton: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    await _seed_history(
        fake_redis_singleton, "ARS",
        [(now - 3600.0, 5.0), (now - 1800.0, 5.1), (now, 5.2)],  # 1h of real observed history
    )
    monkeypatch.setattr("p2p_oracle._get_pipeline", lambda: _FakePipeline([5.0, 5.1, 5.15, 5.18, 5.2, 5.22, 5.25, 5.3, 5.4]))

    result = await forecast_premium("ARS", horizon_hours=48, threshold_percentage=None)  # 48h >> 1h observed

    assert result.low_confidence is True
    assert result.context_point_count == 3
    assert result.context_span_hours == pytest.approx(1.0, abs=0.01)
    assert result.current_premium_percentage == 5.2


@pytest.mark.asyncio
async def test_forecast_reports_high_confidence_when_horizon_fits_within_observed_history(
    fake_redis_singleton: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    # Tiny per-point jitter -- a real scan's own JSON payload always differs tick to tick (price,
    # forex rate, ...); an identical repeated premium here would otherwise collapse every point
    # into one real ZSET member (same member string, only the latest ZADD score wins), understating
    # this deployment's own real accumulated history.
    await _seed_history(
        fake_redis_singleton, "TRY",
        [(now - i * 300.0, 3.0 + i * 0.001) for i in range(20, -1, -1)],  # ~1.75h of real observed history
    )
    monkeypatch.setattr("p2p_oracle._get_pipeline", lambda: _FakePipeline([2.9, 2.95, 2.98, 2.99, 3.0, 3.01, 3.02, 3.05, 3.1]))

    result = await forecast_premium("TRY", horizon_hours=1, threshold_percentage=None)

    assert result.low_confidence is False


@pytest.mark.asyncio
async def test_forecast_derives_a_real_threshold_probability_from_the_real_quantile_forecast(
    fake_redis_singleton: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    await _seed_history(fake_redis_singleton, "VES", [(now - i * 300.0, 10.0 + i * 0.001) for i in range(10, -1, -1)])
    monkeypatch.setattr(
        "p2p_oracle._get_pipeline",
        lambda: _FakePipeline([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
    )

    result = await forecast_premium("VES", horizon_hours=1, threshold_percentage=12.0)

    assert result.threshold_percentage == 12.0
    assert result.approx_probability_above_threshold == pytest.approx(0.5)  # 12.0 is exactly the median (0.5) quantile


@pytest.mark.asyncio
async def test_forecast_never_computes_a_probability_when_no_threshold_is_requested(
    fake_redis_singleton: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    await _seed_history(fake_redis_singleton, "ARS", [(now - i * 300.0, 5.0 + i * 0.001) for i in range(5, -1, -1)])
    monkeypatch.setattr("p2p_oracle._get_pipeline", lambda: _FakePipeline([5.0] * 9))

    result = await forecast_premium("ARS", horizon_hours=1, threshold_percentage=None)

    assert result.approx_probability_above_threshold is None
    assert result.threshold_percentage is None

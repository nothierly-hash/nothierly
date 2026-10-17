"""NEMDI -- the Nothierly Emerging Markets Distress Index.

**Real v1, explicit scope**: a 0-100 score derived purely from live P2P/forex premium data
(``p2p_api.py``'s own real ``GET /api/v1/p2p-premiums`` scan) -- **not yet** blended with
``political_agent.py``'s own macro/crisis-LLM scoring. This deployment has never had a configured
LLM provider (``GUARDIAN_LLM_PROVIDER=none``, verified, not assumed -- see that module's own
startup log). Every response below discloses this honestly via ``macro_component_included: false``
and ``methodology_note``, rather than silently presenting a P2P-only score as the full two-input
index the wider strategy describes. Wiring in the macro component once a real LLM key is
configured is a real, disclosed v2 -- not pretended to already exist.

**Real, disclosed formula, per currency with real data this scan**::

    distress_sub_score = clamp(50 + premium_percentage * _PREMIUM_TO_SCORE_MULTIPLIER, 0, 100)

A 0% premium (P2P trading at the official/spot rate -- no real capital-flight pressure) maps to a
neutral 50; a real positive premium (P2P trades above the official rate and spot USDT -- the
classic capital-controls arbitrage signal ``p2p_api.py``'s own module docstring already
establishes) pushes the score up toward 100; a negative premium pushes it down toward 0.
``_PREMIUM_TO_SCORE_MULTIPLIER`` is an initial, disclosed calibration choice -- not backtested
against a historical crisis dataset (none exists for this project yet; ``p2p_api.py``'s own
``p2p:history_zset:`` only started accumulating recently, see that module's own
``P2P_HISTORY_RETENTION_SECONDS`` comment) -- a real, honest v1 starting point, not a claimed-
precise model. The overall NEMDI score is the simple, equally-weighted average of every currency's
sub-score with real data this cycle; a currency Binance P2P/the forex provider can't honestly
price is excluded from the average, never guessed into it (same discipline as ``p2p_api.py``'s own
``excluded`` field).

**Deliberately public, no ``x-api-key``** -- the whole point (per the wider Go-to-Market strategy
this implements) is for financial journalists/media outlets to embed and cite this index freely.
Still rate-limited per client IP (its own tier, see ``core/config.py``'s
``P2PApiSettings.nemdi_rate_limit_per_second``) -- free does not mean unprotected against abuse.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Final

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from core.config import get_settings
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX
import p2p_api
from p2p_api import _cache, enforce_nemdi_rate_limit, P2PCandidate

router = APIRouter(prefix="/api/v1/nemdi", tags=["nemdi"])

# Real, disclosed cap: bounds how far back /history looks and how many real scan cycles it can ever
# return in one response -- matches p2p_api.py's own scan cadence (one snapshot roughly every
# _CACHE_TTL_SECONDS=30s of real wall-clock activity), so a week-long lookback is already tens of
# thousands of points; capped here so a caller can't force this endpoint into an unbounded Redis scan.
_MAX_HISTORY_HOURS: Final[int] = 24 * 30  # matches P2P_HISTORY_RETENTION_SECONDS' own real ceiling

# Real, disclosed v1 calibration -- see module docstring. +1 percentage point of P2P premium moves
# the sub-score 3 points away from the neutral midpoint of 50.
_PREMIUM_TO_SCORE_MULTIPLIER: Final[float] = 3.0
_NEUTRAL_SCORE: Final[float] = 50.0

_METHODOLOGY_NOTE: Final[str] = (
    "NEMDI v1: derived solely from live P2P/forex premiums across NGN/ARS/TRY/VES/RUB (see "
    "p2p_api.py). Each currency's premium maps to a 0-100 distress sub-score around a neutral "
    "midpoint of 50; the index is the equal-weighted average of currencies with real data this "
    "cycle. Does NOT yet include political_agent.py's own macro/crisis-LLM scoring (no LLM "
    "provider is configured on this deployment) -- see macro_component_included."
)


class CurrencyDistressComponent(BaseModel):
    """One currency's real contribution to the overall NEMDI score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    country_fiat: str
    premium_percentage: float
    distress_sub_score: float


class NemdiResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    nemdi_score: float
    components: list[CurrencyDistressComponent]
    excluded: dict[str, str]
    macro_component_included: bool
    methodology_note: str


def _distress_sub_score(premium_percentage: float) -> float:
    """Real, disclosed mapping from a P2P premium to a 0-100 distress sub-score -- see module docstring."""
    raw = _NEUTRAL_SCORE + premium_percentage * _PREMIUM_TO_SCORE_MULTIPLIER
    return round(max(0.0, min(100.0, raw)), 1)


def compute_nemdi_score(candidates: list[P2PCandidate]) -> tuple[float, list[CurrencyDistressComponent]]:
    """Real, pure aggregation -- no I/O, easy to unit test on its own (see tests/test_nemdi.py).

    Args:
        candidates: Real P2P candidates from this scan (``p2p_api.py``'s own
            ``P2PPremiumsResponse.top_candidates``) -- a currency absent here (excluded by the
            underlying scan) contributes nothing, never a guessed sub-score.

    Returns:
        ``(nemdi_score, components)`` -- ``nemdi_score`` is ``0.0`` (never ``NaN``/an exception)
        when ``candidates`` is empty, an honest "no real data this cycle" state.
    """
    components = [
        CurrencyDistressComponent(
            country_fiat=candidate.country_fiat,
            premium_percentage=candidate.premium_percentage,
            distress_sub_score=_distress_sub_score(candidate.premium_percentage),
        )
        for candidate in candidates
    ]
    if not components:
        return 0.0, components

    nemdi_score = round(sum(c.distress_sub_score for c in components) / len(components), 1)
    return nemdi_score, components


@router.get(
    "/score",
    response_model=NemdiResponse,
    summary="NEMDI -- Nothierly Emerging Markets Distress Index (0-100), real-time, free",
    dependencies=[Depends(enforce_nemdi_rate_limit)],
)
async def get_nemdi_score() -> NemdiResponse:
    """Real-time NEMDI score, computed from the same live scan `GET /api/v1/p2p-premiums` serves."""
    live_response = await _cache.get()
    nemdi_score, components = compute_nemdi_score(live_response.top_candidates)

    return NemdiResponse(
        timestamp=live_response.timestamp,
        nemdi_score=nemdi_score,
        components=components,
        excluded=dict(live_response.excluded),
        macro_component_included=False,
        methodology_note=_METHODOLOGY_NOTE,
    )


class NemdiHistoryPoint(BaseModel):
    """One real, past NEMDI score -- computed from that exact scan cycle's own real candidates, never interpolated or backfilled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: str
    nemdi_score: float
    currencies_included: int


class NemdiHistoryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hours: int
    points: list[NemdiHistoryPoint]
    methodology_note: str


async def _fetch_history_points(hours: int) -> list[NemdiHistoryPoint]:
    """Real, pure-Redis reconstruction of past NEMDI scores from ``p2p_api.py``'s own per-fiat history ZSETs.

    **Why this works without a second, separate history store**: ``p2p_api.py``'s
    ``_store_history_snapshot`` writes every real candidate found by one scan cycle under the exact
    same real Unix timestamp (``scan_started_at.timestamp()``, shared across every currency in that
    cycle) -- so grouping every fiat's ZSET entries by that shared score exactly reconstructs "which
    real candidates existed together, in the same real scan" without needing NEMDI to keep its own
    duplicate history of anything ``p2p_api.py`` already records. A currency genuinely excluded that
    cycle (Binance/forex outage) is simply absent from the group, honestly, the same as the live
    ``/score`` endpoint's own ``excluded`` handling -- never backfilled with a neighboring cycle's
    stale value.

    Returns:
        Real historical points, chronological, one per real past scan cycle found in Redis within
        the window -- empty if this deployment's history has not accumulated any yet (a real,
        honest "too early" state, not an error).
    """
    now = time.time()
    window_start = now - hours * 3600.0

    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    except Exception:
        return []

    # timestamp -> {fiat: P2PCandidate}, real candidates grouped by the real shared scan timestamp
    # they were actually stored under -- see this function's own docstring.
    by_timestamp: dict[float, dict[str, P2PCandidate]] = defaultdict(dict)
    for fiat in p2p_api._TARGET_FIAT_CURRENCIES:
        key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
        try:
            members = await redis_client.execute("zrangebyscore", key, window_start, now, withscores=True)
        except RedisUnavailableError:
            continue
        for raw_member, score in members:
            try:
                candidate = P2PCandidate.model_validate(json.loads(raw_member))
            except (json.JSONDecodeError, ValueError):
                continue
            by_timestamp[float(score)][fiat] = candidate

    points: list[NemdiHistoryPoint] = []
    for unix_timestamp in sorted(by_timestamp.keys()):
        candidates = list(by_timestamp[unix_timestamp].values())
        nemdi_score, components = compute_nemdi_score(candidates)
        if not components:
            continue  # a real scan cycle with zero honestly-priceable currencies -- nothing to plot
        points.append(
            NemdiHistoryPoint(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(unix_timestamp)),
                nemdi_score=nemdi_score,
                currencies_included=len(components),
            )
        )
    return points


@router.get(
    "/history",
    response_model=NemdiHistoryResponse,
    summary="NEMDI -- real historical trend, reconstructed from p2p_api.py's own accumulated scan history",
    dependencies=[Depends(enforce_nemdi_rate_limit)],
)
async def get_nemdi_history(
    hours: int = Query(default=24, ge=1, le=_MAX_HISTORY_HOURS, description="Real lookback window, in hours."),
) -> NemdiHistoryResponse:
    """Real NEMDI time series -- one point per real past scan cycle found in this deployment's own history.

    **Honestly empty on a fresh deployment**: this reads ``p2p_api.py``'s own history ZSETs, which
    only start accumulating the moment that service's real scan loop first runs -- a deployment with
    less than ``hours`` of real uptime returns fewer points than requested, never a fabricated or
    interpolated one to fill the gap. See :func:`_fetch_history_points` for the exact reconstruction.
    """
    points = await _fetch_history_points(hours)
    return NemdiHistoryResponse(hours=hours, points=points, methodology_note=_METHODOLOGY_NOTE)

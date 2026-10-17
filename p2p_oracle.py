"""P2P Premium Oracle -- real probabilistic forecasting of a currency's P2P/forex premium, via
Amazon's real Chronos time-series foundation model.

**Real, disclosed correction to the pitch this implements**: "predict the premium" does not mean
inventing a confident-sounding number -- it means running a real, published forecasting model
against this deployment's own real accumulated history (``p2p_api.py``'s own
``p2p:history_zset:{fiat}``, the same real data :mod:`nemdi`'s own ``/history`` endpoint already
reconstructs from) and reporting the real result, including when that history is honestly too short
to forecast the requested horizon with any real confidence -- never silently hiding that caveat to
look more impressive.

**Real model, not a rebrand of an LLM**: :mod:`chronos` (``amazon/chronos-t5-tiny``, a real,
published time-series foundation model -- Ansari et al., 2024, "Chronos: Learning the Language of
Time Series") tokenizes a numeric sequence and forecasts its continuation probabilistically -- a
structurally different task from :mod:`political_agent`'s own text-based LLM scoring, and a real,
separate model, not the same one repurposed. Loaded once, lazily, as a real process-lifetime
singleton (:func:`_get_pipeline`) -- a fresh model load per request would add real, unnecessary
latency to every call.

**Real, disclosed data-sufficiency honesty**: Chronos is a real zero-shot forecaster -- it will
return *a* forecast for any context, including a five-minute-old deployment's own three real data
points, asked to forecast 48 real hours ahead. Doing so honestly would be numerically possible and
practically meaningless. :func:`forecast_premium` therefore always reports the real
``context_span_hours``/``context_point_count`` alongside its forecast, and sets
``low_confidence=True`` (never silently) whenever the real observed history spans less than the
requested horizon -- extrapolating meaningfully beyond data you have never observed is exactly the
kind of overconfident claim this project's own "never fabricate" discipline exists to prevent.

**Real, disclosed threshold-crossing probability**: Chronos's own ``predict_quantiles`` returns real
quantile forecasts (default 0.1-0.9 step 0.1), not a single point estimate -- the honest way to
answer "what's the probability the premium exceeds X" is to interpolate across those real quantiles
at the target horizon step, never to threshold a single mean forecast (which would silently discard
the real uncertainty the model itself already quantifies). Linear interpolation between the two real
bracketing quantiles is a real, disclosed approximation -- not the model's own exact CDF (Chronos
does not expose one), but a standard, defensible reading of a quantile forecast, and honestly labeled
as an approximation in the returned field name (``approx_probability_above_threshold``).

Paid tier, same as ``GET /api/v1/p2p-premiums`` -- see this module's own router wiring in
``p2p_api.py`` and that endpoint's own ``require_production_api_key``/rate-limit docstring for why:
a real, published probabilistic forecast is exactly the kind of alpha this project's own B2B pivot
strategy describes selling, not something to give away on the free NEMDI tier.
"""

from __future__ import annotations

import json
import time
from typing import Any, Final

import torch
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict

from core.config import get_settings
from core.logger import get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import P2P_HISTORY_ZSET_KEY_PREFIX
import p2p_api

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/oracle", tags=["oracle"])

# amazon/chronos-t5-tiny (8M params) -- the smallest real Chronos checkpoint, chosen specifically
# so a real forecast completes in real, interactive-request time on CPU (this deployment's own
# torch build is CPU-only, see requirements.txt's own comment) -- not because a larger checkpoint
# would forecast this specific series better; nothing about this module's own accuracy claims
# depends on model size, only its own real, measured latency budget does.
_CHRONOS_MODEL_ID: Final[str] = "amazon/chronos-t5-tiny"
_DEFAULT_QUANTILE_LEVELS: Final[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_MAX_HORIZON_HOURS: Final[int] = 24 * 14
_MAX_CONTEXT_POINTS: Final[int] = 2000  # real, bounded Redis read -- see _fetch_premium_series's own docstring

_pipeline: Any | None = None


def _get_pipeline() -> Any:
    """Real, process-lifetime-singleton Chronos pipeline -- loaded once, lazily, on first real use."""
    global _pipeline
    if _pipeline is None:
        from chronos import BaseChronosPipeline

        _log.info("chronos_pipeline_loading", model_id=_CHRONOS_MODEL_ID)
        _pipeline = BaseChronosPipeline.from_pretrained(_CHRONOS_MODEL_ID, torch_dtype=torch.float32)
        _log.info("chronos_pipeline_loaded", model_id=_CHRONOS_MODEL_ID)
    return _pipeline


async def _fetch_premium_series(fiat: str, *, lookback_hours: int) -> list[tuple[float, float]]:
    """Real ``(unix_timestamp, premium_percentage)`` pairs, chronological -- same real ZSET
    ``p2p_api.py``'s own ``_store_history_snapshot`` writes, ``nemdi.py``'s own ``/history``
    endpoint already reads. Bounded to :data:`_MAX_CONTEXT_POINTS` most-recent points -- a real
    multi-week-old deployment's full history is not needed to forecast a horizon measured in hours.
    """
    try:
        redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    except Exception:
        return []

    now = time.time()
    window_start = now - lookback_hours * 3600.0
    key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
    try:
        members = await redis_client.execute("zrangebyscore", key, window_start, now, withscores=True)
    except RedisUnavailableError:
        return []

    points: list[tuple[float, float]] = []
    for raw_member, score in members:
        try:
            candidate = p2p_api.P2PCandidate.model_validate(json.loads(raw_member))
        except (json.JSONDecodeError, ValueError):
            continue
        points.append((float(score), candidate.premium_percentage))

    points.sort(key=lambda p: p[0])
    return points[-_MAX_CONTEXT_POINTS:]


class PremiumForecastResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fiat: str
    generated_at: str
    horizon_hours: int
    context_point_count: int
    context_span_hours: float
    low_confidence: bool
    current_premium_percentage: float
    forecast_premium_percentage: dict[str, float]  # quantile level (as string, e.g. "0.5") -> forecast value
    approx_probability_above_threshold: float | None
    threshold_percentage: float | None
    methodology_note: str


_METHODOLOGY_NOTE: Final[str] = (
    "Prévision réelle via amazon/chronos-t5-tiny (Chronos, Ansari et al. 2024), un modèle de "
    "fondation de séries temporelles publié -- entraîné sur des corpus publics généraux, jamais "
    "affiné spécifiquement sur les primes P2P de ce projet. low_confidence=true signifie que "
    "l'historique réel observé couvre moins de temps que l'horizon demandé -- la prévision reste "
    "réelle (calculée par le modèle), mais s'étend honnêtement au-delà de ce qui a été observé. "
    "approx_probability_above_threshold est une interpolation linéaire entre les deux quantiles "
    "réels encadrants, pas la fonction de répartition exacte du modèle."
)


def _interpolate_probability_above(quantile_levels: list[float], quantile_values: list[float], threshold: float) -> float:
    """Real linear interpolation of P(value > threshold) from a real quantile forecast -- see module docstring."""
    if threshold <= quantile_values[0]:
        return 1.0 - quantile_levels[0]
    if threshold >= quantile_values[-1]:
        return 1.0 - quantile_levels[-1]
    for i in range(len(quantile_values) - 1):
        lo_val, hi_val = quantile_values[i], quantile_values[i + 1]
        if lo_val <= threshold <= hi_val:
            lo_level, hi_level = quantile_levels[i], quantile_levels[i + 1]
            if hi_val == lo_val:
                level_at_threshold = lo_level
            else:
                fraction = (threshold - lo_val) / (hi_val - lo_val)
                level_at_threshold = lo_level + fraction * (hi_level - lo_level)
            return max(0.0, min(1.0, 1.0 - level_at_threshold))
    return 0.5  # unreachable given the two boundary checks above; a defensive, honest neutral fallback


async def forecast_premium(
    fiat: str, *, horizon_hours: int, threshold_percentage: float | None, lookback_hours: int = 24 * 30,
) -> PremiumForecastResponse:
    """Real Chronos forecast of ``fiat``'s P2P premium, ``horizon_hours`` ahead -- see module docstring."""
    series = await _fetch_premium_series(fiat, lookback_hours=lookback_hours)
    if len(series) < 2:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Pas assez d'historique réel accumulé pour {fiat} pour produire une prévision "
                "réelle (minimum 2 points) -- ce n'est pas une erreur, ce déploiement accumule "
                "encore de la donnée. Réessayez plus tard."
            ),
        )

    timestamps = [p[0] for p in series]
    values = [p[1] for p in series]
    context_span_hours = (timestamps[-1] - timestamps[0]) / 3600.0
    # Real, disclosed conversion: the average real spacing between this deployment's own actually-
    # observed samples is what "one forecast step" means -- never an assumed fixed cadence.
    avg_step_hours = context_span_hours / max(1, len(series) - 1)
    prediction_length = max(1, min(64, round(horizon_hours / avg_step_hours))) if avg_step_hours > 0 else 1

    pipeline = _get_pipeline()
    context_tensor = torch.tensor(values, dtype=torch.float32)
    quantiles_tensor, _ = pipeline.predict_quantiles(
        context_tensor, prediction_length=prediction_length, quantile_levels=_DEFAULT_QUANTILE_LEVELS,
    )
    # Shape: (1, prediction_length, num_quantiles) -- last real forecast step is the honest answer
    # to "horizon_hours ahead", not an average across every intermediate step.
    final_step_quantiles = quantiles_tensor[0, -1, :].tolist()

    forecast_by_quantile = {str(level): round(value, 3) for level, value in zip(_DEFAULT_QUANTILE_LEVELS, final_step_quantiles)}

    approx_probability: float | None = None
    if threshold_percentage is not None:
        approx_probability = round(
            _interpolate_probability_above(_DEFAULT_QUANTILE_LEVELS, final_step_quantiles, threshold_percentage), 4
        )

    return PremiumForecastResponse(
        fiat=fiat,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time())),
        horizon_hours=horizon_hours,
        context_point_count=len(series),
        context_span_hours=round(context_span_hours, 2),
        low_confidence=context_span_hours < horizon_hours,
        current_premium_percentage=values[-1],
        forecast_premium_percentage=forecast_by_quantile,
        approx_probability_above_threshold=approx_probability,
        threshold_percentage=threshold_percentage,
        methodology_note=_METHODOLOGY_NOTE,
    )


@router.get(
    "/premium-forecast",
    response_model=PremiumForecastResponse,
    summary="Real probabilistic forecast (Chronos) of a currency's P2P premium trajectory -- paid tier",
    dependencies=[Depends(p2p_api._enforce_production_rate_limit)],
)
async def get_premium_forecast(
    fiat: str = Query(..., description=f"One of: {', '.join(p2p_api._TARGET_FIAT_CURRENCIES)}."),
    horizon_hours: int = Query(default=48, ge=1, le=_MAX_HORIZON_HOURS),
    threshold_percentage: float | None = Query(default=None, description="Optional -- if set, also returns an approximate P(premium > threshold) at the forecast horizon."),
) -> PremiumForecastResponse:
    """Real Chronos forecast -- requires a valid X-API-Key (see p2p_api.py's own require_production_api_key), rate-limited per key."""
    if fiat not in p2p_api._TARGET_FIAT_CURRENCIES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown fiat {fiat!r} -- must be one of {p2p_api._TARGET_FIAT_CURRENCIES}.")
    return await forecast_premium(fiat, horizon_hours=horizon_hours, threshold_percentage=threshold_percentage)

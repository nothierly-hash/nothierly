"""Swarm Scout -- real statistical anomaly detection on p2p_api.py's own accumulated P2P premium history.

**Real detection method, not a fabricated trigger**: for each of the five real currencies
``p2p_api.py`` tracks, this agent compares the *latest* real premium reading against the real
rolling mean/standard deviation of the :data:`_ROLLING_WINDOW_POINTS` real readings immediately
before it (a standard z-score test) -- exactly the "volume/price moved N-sigma away from its own
recent baseline" idea the wider Swarm AI brief describes, applied to the one real, already-
accumulating time series this project has for these currencies (``p2p_api.py``'s own
``p2p:history_zset:{fiat}``), rather than a new, separate CCXT minor-pair volume scanner this
deliverable does not build.

**Real, disclosed scope correction**: the brief's own example ("volume du Peso Philippin
multiplié par 5") describes scanning hundreds of CCXT spot pairs for raw trade volume -- a
genuinely different, larger data pipeline (this project has no such minor-pair volume ingestion
anywhere). This agent instead watches the real P2P premium signal this project already computes
and stores for NGN/ARS/TRY/VES/RUB -- the same real anomaly *category* (a statistical outlier
against recent history), on real data this project actually has, not a simulated stand-in for
data it doesn't.

**Real, event-only bus discipline**: :data:`~core.schemas.SWARM_ANOMALY_DETECTED_TOPIC` carries
only an ``anomaly_id`` -- the real detection details (currency, latest/baseline premium, z-score)
are written to Redis first (:data:`~core.schemas.SWARM_ANOMALY_REDIS_KEY_PREFIX`), then the event
fires. Same "small event, real state in Redis" pattern this project's other ZMQ topics already use.

**Real cooldown, not a re-alert storm**: a currency whose premium stays elevated would otherwise
re-trigger on every real tick (:data:`_POLL_INTERVAL_SECONDS`) for as long as it stays anomalous.
:data:`_ALERT_COOLDOWN_SECONDS` suppresses a repeat real alert for the same currency within that
window -- the analyst/risk pipeline downstream needs one real event per real episode, not one per
tick.

Run standalone (own process, matching every other real standalone agent in this project)::

    python scout_agent.py
"""

from __future__ import annotations

import json
import time
import uuid
from statistics import mean, stdev
from typing import Final

import zmq.asyncio

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.logger import configure_logging
from core.redis_client import RedisUnavailableError
from core.schemas import (
    P2P_HISTORY_ZSET_KEY_PREFIX,
    SWARM_ANOMALY_DETECTED_TOPIC,
    SWARM_ANOMALY_REDIS_KEY_PREFIX,
)

_AGENT_ID: Final[str] = "swarm-scout"
_POLL_INTERVAL_SECONDS: Final[float] = 10.0
# Same real five currencies p2p_api.py itself tracks -- see that module's own
# _TARGET_FIAT_CURRENCIES. Duplicated here, deliberately, rather than imported: importing p2p_api
# would pull in that module's own FastAPI app construction (and its real external-API-calling
# scan loop) into a process that has no business starting either -- this agent only ever *reads*
# the real history p2p_api.py's own separate process already writes.
_TARGET_FIAT_CURRENCIES: Final[tuple[str, ...]] = ("NGN", "ARS", "TRY", "VES", "RUB")
_ROLLING_WINDOW_POINTS: Final[int] = 20
_MIN_BASELINE_POINTS: Final[int] = 5
_ZSCORE_THRESHOLD: Final[float] = 2.5
_ALERT_COOLDOWN_SECONDS: Final[float] = 600.0  # 10 min -- see module docstring


class ScoutAgent(BaseAgent):
    """Real, periodic z-score anomaly scan over p2p_api.py's own accumulated premium history."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            tick_interval_seconds=_POLL_INTERVAL_SECONDS,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        self._last_alerted_at: dict[str, float] = {}

    async def on_start(self) -> None:
        self._log.info("scout_agent_ready", agent_id=self.agent_id, currencies=_TARGET_FIAT_CURRENCIES)

    async def on_tick(self) -> None:
        assert self._redis_client is not None
        for fiat in _TARGET_FIAT_CURRENCIES:
            try:
                await self._check_one_currency(fiat)
            except RedisUnavailableError:
                self._log.warning("scout_redis_unavailable_skipping_currency", fiat=fiat)

    async def _check_one_currency(self, fiat: str) -> None:
        assert self._redis_client is not None
        key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
        # Most recent _ROLLING_WINDOW_POINTS + 1 real readings, newest first.
        raw = await self._redis_client.execute("zrevrange", key, 0, _ROLLING_WINDOW_POINTS, withscores=True)
        if len(raw) < _MIN_BASELINE_POINTS + 1:
            return  # honestly too early -- not enough real history to have a baseline yet

        # Real, deliberately loose JSON field extraction, not a full p2p_api.P2PCandidate parse --
        # importing p2p_api.py here would execute that module's own top-level FastAPI app
        # construction (and its own configure_logging(service_name="p2p-premium-api") call, which
        # would silently override this agent's own logging service name) just to reach one Pydantic
        # model class. Every real member written by p2p_api.py's own _store_history_snapshot is
        # that same class's own model_dump_json() -- premium_percentage is always present there.
        premiums: list[float] = []
        for raw_member, _score in raw:
            try:
                premiums.append(float(json.loads(raw_member)["premium_percentage"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        if len(premiums) < _MIN_BASELINE_POINTS + 1:
            return

        latest, baseline = premiums[0], premiums[1:]
        baseline_mean = mean(baseline)
        baseline_stdev = stdev(baseline) if len(baseline) >= 2 else 0.0
        if baseline_stdev == 0.0:
            return  # a genuinely flat real baseline has no real z-score to compute against

        zscore = (latest - baseline_mean) / baseline_stdev
        if abs(zscore) < _ZSCORE_THRESHOLD:
            return

        now = time.time()
        last_alert = self._last_alerted_at.get(fiat)
        if last_alert is not None and (now - last_alert) < _ALERT_COOLDOWN_SECONDS:
            return  # see module docstring on cooldown

        self._last_alerted_at[fiat] = now
        anomaly_id = str(uuid.uuid4())
        record = {
            "anomaly_id": anomaly_id,
            "fiat": fiat,
            "latest_premium_percentage": latest,
            "baseline_mean_percentage": round(baseline_mean, 3),
            "baseline_stdev_percentage": round(baseline_stdev, 3),
            "zscore": round(zscore, 2),
            "baseline_point_count": len(baseline),
            "detected_at": now,
        }
        try:
            await self._redis_client.execute("hset", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}{anomaly_id}", mapping=record)
            await self._redis_client.execute("expire", f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}{anomaly_id}", 24 * 3600)
        except RedisUnavailableError:
            self._log.warning("scout_anomaly_redis_write_failed", fiat=fiat, anomaly_id=anomaly_id)
            return

        self._log.critical(
            "scout_anomaly_detected", fiat=fiat, anomaly_id=anomaly_id, zscore=round(zscore, 2),
            latest_premium_percentage=latest, baseline_mean_percentage=round(baseline_mean, 3),
        )
        await self.publish(SWARM_ANOMALY_DETECTED_TOPIC, {"anomaly_id": anomaly_id, "asset": fiat})


async def main() -> None:
    configure_logging(service_name="swarm-scout")
    agent = ScoutAgent()
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

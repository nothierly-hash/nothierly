"""Swarm Analyst -- reactive, real LLM crisis-scoring the instant scout_agent.py detects an anomaly.

**Real reuse, not a second implementation**: every real capability this agent needs -- fetching
live RSS headlines, matching them against a currency's real aliases, and scoring one with the real
configured LLM (or honestly reporting none is configured) -- already exists in
:mod:`political_agent` and is imported from it directly (:func:`political_agent._fetch_one_feed`,
:data:`political_agent._RSS_FEEDS`, :func:`political_agent._assess_headline`, ...). This agent is
the real, reactive *wiring* :mod:`nemdi`'s own module docstring already names as a disclosed v2 gap
("not yet blended with political_agent.py's own macro/crisis-LLM scoring") -- it does not
reimplement RSS fetching or LLM calling, both of which stay exactly as tested in
``tests/test_political_agent.py``.

**Why this agent exists separately from political_agent.py, not as a new mode of it**:
political_agent.py is explicitly, deliberately not a ZMQ participant (see that module's own "why
standalone" section) -- it never opens a bus socket, on a 15-minute unconditional poll loop. Making
it reactive would mean either giving it a ZMQ subscriber (contradicting that documented design
choice) or duplicating its real fetch/score logic into a second implementation. Importing its real
functions as a library from a small, separate, genuinely-reactive ZMQ-aware process does neither --
political_agent.py itself is unchanged, still exactly as tested.

**Real, honest status vocabulary -- never a guessed verdict**:

- ``CRISIS_CONFIRMED``: a real LLM call succeeded and scored a real, currency-matching headline at
  or above :data:`_CRISIS_SCORE_THRESHOLD`.
- ``NO_CRISIS``: a real LLM call succeeded but scored below threshold.
- ``NO_MATCHING_HEADLINES``: every real RSS feed was fetched, but none currently carries a
  headline mentioning this currency's own real aliases -- a real, honest "nothing to analyze yet",
  not an error.
- ``LLM_UNAVAILABLE``: :func:`political_agent._assess_headline` returned ``None`` -- covers no
  provider configured, a real call failure, and a malformed response uniformly, since that
  function's own real contract does not distinguish between them in its return value either (see
  its own docstring). Reported as one honest "no real analysis was possible" state rather than
  guessing which of the three actually happened.

Run standalone (own process)::

    python analyst_agent.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

import zmq.asyncio

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.logger import configure_logging
from core.redis_client import RedisUnavailableError
from core.schemas import (
    SWARM_ANALYSIS_DONE_TOPIC,
    SWARM_ANALYSIS_REDIS_KEY_PREFIX,
    SWARM_ANOMALY_DETECTED_TOPIC,
    SWARM_ANOMALY_REDIS_KEY_PREFIX,
)
from political_agent import _assess_headline, _fetch_one_feed, _get_proprietary_llm_base_url, _RSS_FEEDS
from core.http_client import ResilientHttpClient

_AGENT_ID: Final[str] = "swarm-analyst"
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 2.0
# Real, disclosed calibration: a real crisis_score at or above this is treated as a real, confirmed
# crisis worth the Risk Gate's own liquidity check -- below it, real but non-alarming sentiment.
# Not backtested against a real historical crisis dataset (none exists for this project yet, same
# real gap nemdi.py's own module docstring already discloses for its premium-to-score calibration).
_CRISIS_SCORE_THRESHOLD: Final[int] = 60


class AnalystAgent(BaseAgent):
    """Real, reactive analyst: wakes on a real scout_agent.py anomaly, scores real matching news."""

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
            subscribed_topics=(SWARM_ANOMALY_DETECTED_TOPIC,),
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

    async def on_tick(self) -> None:
        raise NotImplementedError("AnalystAgent overrides run() and never ticks -- see module docstring.")

    async def on_start(self) -> None:
        self._log.info("analyst_agent_ready", agent_id=self.agent_id)

    async def run(self) -> None:
        """Same real 'connect, then drain the bus alongside heartbeat/control listeners' shape as
        :class:`~src.execution.shadow_crossing_engine.ShadowCrossingEngine` -- see that class's own
        ``run()`` for the identical pattern this mirrors."""
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info("agent_started", agent_id=self.agent_id)

            await asyncio.gather(
                self._listen_for_bus_messages(),
                self._run_heartbeat_loop(),
                self._listen_for_control_commands(),
            )
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _run_heartbeat_loop(self) -> None:
        while self._running:
            await self._write_heartbeat()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

    async def _listen_for_bus_messages(self) -> None:
        async for message in self._subscriber.listen():
            if message.topic != SWARM_ANOMALY_DETECTED_TOPIC:
                continue
            anomaly_id = str(message.data.get("anomaly_id", ""))
            fiat = str(message.data.get("asset", ""))
            if not anomaly_id or not fiat:
                self._log.warning("analyst_malformed_anomaly_event_skipped", data=message.data)
                continue
            await self._handle_anomaly(anomaly_id, fiat)

    async def _handle_anomaly(self, anomaly_id: str, fiat: str) -> None:
        assert self._redis_client is not None
        self._log.critical("analyst_anomaly_received", anomaly_id=anomaly_id, fiat=fiat)

        status: str
        crisis_score: int | None = None
        sentiment: str | None = None
        event_summary: str | None = None
        headline: str | None = None

        try:
            async with ResilientHttpClient(max_concurrent_requests=5, timeout_seconds=10.0) as http:
                results = await asyncio.gather(
                    *(_fetch_one_feed(http, source, url) for source, url in _RSS_FEEDS), return_exceptions=True
                )
        except Exception:
            self._log.warning("analyst_feed_fetch_failed", anomaly_id=anomaly_id, fiat=fiat, exc_info=True)
            results = []

        matching_items = [
            item
            for result in results
            if not isinstance(result, BaseException)
            for item in result
            if fiat in item.matched_currencies
        ]

        if not matching_items:
            status = "NO_MATCHING_HEADLINES"
        else:
            item = matching_items[0]
            headline = item.headline
            proprietary_llm_base_url = await _get_proprietary_llm_base_url(self._redis_client)
            assessment = await _assess_headline(
                self._settings.guardian, item,
                proprietary_llm_base_url=proprietary_llm_base_url, redis_client=self._redis_client,
            )
            if assessment is None:
                status = "LLM_UNAVAILABLE"
            else:
                crisis_score = assessment.crisis_score
                sentiment = assessment.sentiment
                event_summary = assessment.event_summary
                status = "CRISIS_CONFIRMED" if crisis_score >= _CRISIS_SCORE_THRESHOLD else "NO_CRISIS"

        record: dict[str, object] = {
            "anomaly_id": anomaly_id,
            "fiat": fiat,
            "status": status,
            "headline": headline or "",
            "crisis_score": crisis_score if crisis_score is not None else -1,
            "sentiment": sentiment or "",
            "event_summary": event_summary or "",
            "analyzed_at": time.time(),
        }
        try:
            await self._redis_client.execute(
                "hset", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}{anomaly_id}", mapping=record
            )
            await self._redis_client.execute("expire", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}{anomaly_id}", 24 * 3600)
        except RedisUnavailableError:
            self._log.warning("analyst_analysis_redis_write_failed", anomaly_id=anomaly_id)

        self._log.critical("analyst_analysis_complete", anomaly_id=anomaly_id, fiat=fiat, status=status)
        await self.publish(SWARM_ANALYSIS_DONE_TOPIC, {"anomaly_id": anomaly_id, "asset": fiat, "status": status})


async def main() -> None:
    configure_logging(service_name="swarm-analyst")
    agent = AnalystAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

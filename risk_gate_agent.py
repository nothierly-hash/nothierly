"""Swarm Risk Gate -- the real final quantitative filter before a Swarm opportunity is surfaced.

**Real, two-condition gate, both checked against real data, never assumed**:

1. ``analyst_agent.py``'s own real status must be ``CRISIS_CONFIRMED`` -- a real LLM call actually
   ran and scored a real, currency-matching headline at or above its own configured threshold (see
   that module's own docstring). ``NO_CRISIS``/``NO_MATCHING_HEADLINES``/``LLM_UNAVAILABLE`` all
   honestly fail this gate -- there is no real confirmed crisis to react to.
2. The currency's real, current P2P liquidity (``p2p_api.py``'s own already-computed
   ``avg_liquidity_usdt`` field, read from that currency's own most recent real history entry --
   no new external call) must clear :data:`_MIN_LIQUIDITY_USD`.

**Real, disclosed scope, stated as plainly as this project's other real financial-gate modules
(e.g. ``core.crossing_network``'s own MiFID II correction)**: this gate approves *visibility* of a
real opportunity to a human operator -- it does not, and structurally cannot, place a real trade.
This project holds no real broker/execution capability for P2P/fiat-rail arbitrage (no adapter
exists for "buy NGN P2P from a Nigerian bank account"), so a real autonomous "Orchestrator" that
deploys a trading subprocess on approval would be fiction, not a missing feature. What this agent
really does instead: persist the real approval to a real, bounded Redis list
(:data:`~core.schemas.SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY`) a human-facing dashboard section can
read -- a real, actionable alert, not a fabricated auto-trade.

Run standalone (own process)::

    python risk_gate_agent.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Final

import zmq.asyncio

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.logger import configure_logging
from core.redis_client import RedisUnavailableError
from core.schemas import (
    P2P_HISTORY_ZSET_KEY_PREFIX,
    SWARM_ANALYSIS_DONE_TOPIC,
    SWARM_ANALYSIS_REDIS_KEY_PREFIX,
    SWARM_APPROVED_OPPORTUNITIES_MAX_ENTRIES,
    SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY,
    SWARM_ARBITRAGE_APPROVED_TOPIC,
)

_AGENT_ID: Final[str] = "swarm-risk-gate"
_HEARTBEAT_INTERVAL_SECONDS: Final[float] = 2.0
# Real, disclosed floor -- matches the wider Swarm AI brief's own example ("liquidité suffisante
# > 100 000 $"). A real, tunable business decision, not a physical constant.
_MIN_LIQUIDITY_USD: Final[float] = 100_000.0
_CONFIRMED_STATUS: Final[str] = "CRISIS_CONFIRMED"


class RiskGateAgent(BaseAgent):
    """Real, reactive risk gate: confirmed-crisis status + real liquidity floor, or an honest rejection."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
        min_liquidity_usd: float = _MIN_LIQUIDITY_USD,
    ) -> None:
        super().__init__(
            _AGENT_ID,
            subscribed_topics=(SWARM_ANALYSIS_DONE_TOPIC,),
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )
        self._min_liquidity_usd = min_liquidity_usd

    async def on_tick(self) -> None:
        raise NotImplementedError("RiskGateAgent overrides run() and never ticks -- see module docstring.")

    async def on_start(self) -> None:
        self._log.info("risk_gate_agent_ready", agent_id=self.agent_id, min_liquidity_usd=self._min_liquidity_usd)

    async def run(self) -> None:
        """Same real reactive shape as :class:`analyst_agent.AnalystAgent`'s own ``run()`` -- see that module."""
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
            if message.topic != SWARM_ANALYSIS_DONE_TOPIC:
                continue
            anomaly_id = str(message.data.get("anomaly_id", ""))
            fiat = str(message.data.get("asset", ""))
            if not anomaly_id or not fiat:
                self._log.warning("risk_gate_malformed_analysis_event_skipped", data=message.data)
                continue
            await self._evaluate(anomaly_id, fiat)

    async def _current_liquidity_usd(self, fiat: str) -> float | None:
        """Real, already-computed liquidity from this currency's own most recent real scan --
        no new external call. ``None`` if genuinely no real history exists yet for this currency."""
        assert self._redis_client is not None
        key = f"{P2P_HISTORY_ZSET_KEY_PREFIX}{fiat}"
        try:
            latest = await self._redis_client.execute("zrevrange", key, 0, 0)
        except RedisUnavailableError:
            return None
        if not latest:
            return None
        try:
            return float(json.loads(latest[0])["avg_liquidity_usdt"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    async def _evaluate(self, anomaly_id: str, fiat: str) -> None:
        assert self._redis_client is not None
        try:
            analysis = await self._redis_client.execute("hgetall", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}{anomaly_id}")
        except RedisUnavailableError:
            self._log.warning("risk_gate_analysis_read_failed", anomaly_id=anomaly_id)
            return

        # Real, defensive decode -- found live, not theorized: this project's own pinned fakeredis
        # (2.37.1) does not always honor decode_responses=True for HGETALL, returning bytes keys
        # and values even when every other command on the same client correctly returns str (the
        # same real quirk core/message_integrity.py's own sign()/verify() already defend against
        # for GET). Real production Redis with decode_responses=True (this project's own real
        # config, see core/redis_client.py) always returns str here -- this normalization is a
        # no-op against it, and a real, necessary one against this dev/test substitute.
        analysis = {
            (k.decode("utf-8") if isinstance(k, bytes) else k): (v.decode("utf-8") if isinstance(v, bytes) else v)
            for k, v in analysis.items()
        } if analysis else analysis

        status = analysis.get("status") if analysis else None
        if status != _CONFIRMED_STATUS:
            self._log.info("risk_gate_rejected_no_confirmed_crisis", anomaly_id=anomaly_id, fiat=fiat, status=status)
            return

        liquidity_usd = await self._current_liquidity_usd(fiat)
        if liquidity_usd is None or liquidity_usd < self._min_liquidity_usd:
            self._log.info(
                "risk_gate_rejected_insufficient_liquidity",
                anomaly_id=anomaly_id, fiat=fiat, liquidity_usd=liquidity_usd, min_liquidity_usd=self._min_liquidity_usd,
            )
            return

        self._log.critical(
            "risk_gate_approved", anomaly_id=anomaly_id, fiat=fiat, liquidity_usd=liquidity_usd,
        )
        entry = {
            "anomaly_id": anomaly_id,
            "fiat": fiat,
            "liquidity_usd": liquidity_usd,
            "crisis_score": analysis.get("crisis_score"),
            "sentiment": analysis.get("sentiment"),
            "event_summary": analysis.get("event_summary"),
            "headline": analysis.get("headline"),
            "approved_at": time.time(),
        }
        try:
            await self._redis_client.execute("lpush", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, json.dumps(entry))
            await self._redis_client.execute(
                "ltrim", SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 0, SWARM_APPROVED_OPPORTUNITIES_MAX_ENTRIES - 1
            )
        except RedisUnavailableError:
            self._log.warning("risk_gate_approval_persist_failed", anomaly_id=anomaly_id)

        await self.publish(
            SWARM_ARBITRAGE_APPROVED_TOPIC, {"anomaly_id": anomaly_id, "asset": fiat, "liquidity_usd": liquidity_usd}
        )


async def main() -> None:
    configure_logging(service_name="swarm-risk-gate")
    agent = RiskGateAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

"""Macro-economic / on-chain data ingestion micro-agent.

:class:`MacroAgent` polls a handful of low-frequency macro indicators
(US 10-year treasury yield, BTC exchange net flow, ...) once per long
interval, validates each reading through :class:`core.schemas.
MacroSnapshot`, persists it to Redis with a long TTL, and publishes it on
the ZMQ bus.

Unlike :class:`~src.agents.market_data.agent.MarketDataAgent`'s
push-driven WebSocket streams, macro data is pull-driven and changes
slowly, so this agent overrides :meth:`~core.base_agent.BaseAgent.run`
with a simple fetch-then-sleep polling loop instead of a tick loop or
long-lived stream consumers.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import zmq.asyncio
from pydantic import SecretStr

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.http_client import ResilientHttpClient
from core.redis_client import RedisUnavailableError
from core.schemas import MacroSnapshot

_MACRO_DATA_TOPIC = "MACRO_DATA_UPDATE"

_FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
_GLASSNODE_EXCHANGE_FLOW_URL = (
    "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net"
)


@dataclass(frozen=True)
class MacroSourceConfig:
    """One indicator this agent fetches on every polling cycle.

    Attributes:
        indicator: ``UPPER_SNAKE_CASE`` indicator name, matching
            :attr:`~core.schemas.MacroSnapshot.indicator`'s pattern.
        source_name: Human-readable provider name, used as
            :attr:`~core.schemas.MacroSnapshot.source`.
        fetch: Zero-argument async callable returning the latest reading
            as a ``float``. May be backed by a real HTTP call or a
            simulated/dummy generator (see
            :meth:`MacroAgent._build_us_10y_yield_source` and
            :meth:`MacroAgent._build_btc_exchange_flow_source`).
    """

    indicator: str
    source_name: str
    fetch: Callable[[], Awaitable[float]]


class MacroAgent(BaseAgent):
    """Polls low-frequency macro/on-chain indicators and publishes them.

    Each configured indicator falls back to a simulated reading when its
    provider's API key isn't configured (see :class:`~core.config.
    MacroApiSettings`), so this agent runs end-to-end without requiring
    real credentials -- useful for local development and for chapters of
    this system that don't yet need live macro data.
    """

    def __init__(
        self,
        *,
        poll_interval_seconds: float = 3600.0,
        redis_ttl_seconds: int = 7200,
        fred_observations_url: str = _FRED_OBSERVATIONS_URL,
        glassnode_exchange_flow_url: str = _GLASSNODE_EXCHANGE_FLOW_URL,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Configure the agent and build its list of indicator sources.

        Args:
            poll_interval_seconds: Time between the start of one polling
                cycle and the next (e.g. ``3600.0`` for hourly updates).
            redis_ttl_seconds: TTL applied to each ``MacroSnapshot`` key
                written to Redis. Deliberately long relative to
                :class:`~src.agents.market_data.agent.MarketDataAgent`'s:
                macro indicators change slowly, so a stale-but-recent
                reading is still useful, unlike stale order book data.
            fred_observations_url: FRED API endpoint. Overridable for
                testing against a local mock server.
            glassnode_exchange_flow_url: Glassnode API endpoint.
                Overridable for testing against a local mock server.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
        """
        super().__init__(
            "macro-agent",
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._poll_interval_seconds = poll_interval_seconds
        self._redis_ttl_seconds = redis_ttl_seconds
        self._fred_observations_url = fred_observations_url
        self._glassnode_exchange_flow_url = glassnode_exchange_flow_url
        self._http_client = ResilientHttpClient(max_concurrent_requests=5, timeout_seconds=15.0)

        self._fred_api_key: SecretStr | None = self._settings.macro.fred_api_key
        self._glassnode_api_key: SecretStr | None = self._settings.macro.glassnode_api_key

        self._sources: list[MacroSourceConfig] = [
            self._build_us_10y_yield_source(),
            self._build_btc_exchange_flow_source(),
        ]

    # -- Source configuration ---------------------------------------------

    def _build_us_10y_yield_source(self) -> MacroSourceConfig:
        """Build the US 10Y treasury yield source (real FRED or simulated)."""
        if self._fred_api_key is not None:
            return MacroSourceConfig("US_10Y_YIELD", "FRED", self._fetch_us_10y_yield_from_fred)
        self._log.warning(
            "macro_source_using_simulated_data",
            agent_id=self.agent_id,
            indicator="US_10Y_YIELD",
            reason="MACRO_FRED_API_KEY not configured",
        )
        return MacroSourceConfig("US_10Y_YIELD", "SIMULATED", self._fetch_simulated_us_10y_yield)

    def _build_btc_exchange_flow_source(self) -> MacroSourceConfig:
        """Build the BTC exchange net flow source (real Glassnode or simulated)."""
        if self._glassnode_api_key is not None:
            return MacroSourceConfig(
                "BTC_EXCHANGE_FLOW", "GLASSNODE", self._fetch_btc_exchange_flow_from_glassnode
            )
        self._log.warning(
            "macro_source_using_simulated_data",
            agent_id=self.agent_id,
            indicator="BTC_EXCHANGE_FLOW",
            reason="MACRO_GLASSNODE_API_KEY not configured",
        )
        return MacroSourceConfig(
            "BTC_EXCHANGE_FLOW", "SIMULATED", self._fetch_simulated_btc_exchange_flow
        )

    # -- Real fetchers -----------------------------------------------------

    async def _fetch_us_10y_yield_from_fred(self) -> float:
        """Fetch the latest 10-Year Treasury Constant Maturity Rate (DGS10).

        Returns:
            The latest published yield, as a percentage (e.g. ``4.25``).

        Raises:
            ValueError: If FRED returns no observations, or its latest
                observation is the ``"."`` missing-value marker.
        """
        assert self._fred_api_key is not None
        data = await self._http_client.get_json(
            self._fred_observations_url,
            params={
                "series_id": "DGS10",
                "api_key": self._fred_api_key.get_secret_value(),
                "file_type": "json",
                "sort_order": "desc",
                "limit": "1",
            },
        )
        observations = data.get("observations", [])
        if not observations:
            raise ValueError("FRED returned no observations for series DGS10.")
        raw_value = observations[0]["value"]
        if raw_value == ".":
            raise ValueError("FRED's latest DGS10 observation is a missing-value marker ('.').")
        return float(raw_value)

    async def _fetch_btc_exchange_flow_from_glassnode(self) -> float:
        """Fetch the latest net BTC volume flowing into/out of exchanges.

        Returns:
            The latest net flow reading (positive = net inflow to
            exchanges, negative = net outflow), in BTC.

        Raises:
            ValueError: If Glassnode returns no data points.
        """
        assert self._glassnode_api_key is not None
        data = await self._http_client.get_json(
            self._glassnode_exchange_flow_url,
            params={
                "a": "BTC",
                "i": "1h",
                "api_key": self._glassnode_api_key.get_secret_value(),
            },
        )
        if not data:
            raise ValueError("Glassnode returned no data points for exchange net flow.")
        return float(data[-1]["v"])

    # -- Simulated fetchers (used when no API key is configured) -----------

    async def _fetch_simulated_us_10y_yield(self) -> float:
        """Simulated US 10Y yield reading, centered around a plausible value."""
        return round(4.0 + random.uniform(-0.15, 0.15), 3)

    async def _fetch_simulated_btc_exchange_flow(self) -> float:
        """Simulated BTC exchange net flow reading, in BTC."""
        return round(random.uniform(-500.0, 500.0), 2)

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent polls on its own schedule inside :meth:`run`.

        :meth:`run` is fully overridden with a fetch-then-sleep loop
        instead of :class:`~core.base_agent.BaseAgent`'s tick loop, so
        this is never called. Implemented (rather than omitted) only to
        satisfy the abstract base class, and raises loudly if it is ever
        invoked by mistake.
        """
        raise NotImplementedError("MacroAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Open the underlying ``aiohttp`` session."""
        await self._http_client.connect()
        self._log.info(
            "macro_agent_ready", agent_id=self.agent_id, source_count=len(self._sources)
        )

    async def on_shutdown(self) -> None:
        """Close the underlying ``aiohttp`` session.

        Runs as part of :class:`~core.base_agent.BaseAgent`'s inherited
        graceful-shutdown sequence, before the ZMQ sockets/context and
        the Redis connection are torn down.
        """
        await self._http_client.close()

    async def run(self) -> None:
        """Connect dependencies, then poll all sources on a fixed interval.

        Mirrors :meth:`BaseAgent.run`'s lifecycle (signal handling, Redis
        connection, cancellation-safe shutdown) but replaces the
        per-tick loop with a fetch-then-sleep polling loop, since macro
        data updates far too slowly to warrant a tick-driven design.

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info(
                "agent_started",
                agent_id=self.agent_id,
                poll_interval_seconds=self._poll_interval_seconds,
            )

            while self._running:
                cycle_started_at = time.monotonic()
                await self._run_fetch_cycle()

                elapsed = time.monotonic() - cycle_started_at
                remaining = max(0.0, self._poll_interval_seconds - elapsed)
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    # -- Fetch cycle --------------------------------------------------

    async def _run_fetch_cycle(self) -> None:
        """Concurrently fetch, validate, persist, and publish every source.

        Each source is handled independently (see
        :meth:`_fetch_and_publish`): one provider failing never prevents
        the others from succeeding.
        """
        self._log.info(
            "macro_fetch_cycle_started", agent_id=self.agent_id, source_count=len(self._sources)
        )
        await asyncio.gather(*(self._fetch_and_publish(source) for source in self._sources))

    async def _fetch_and_publish(self, source: MacroSourceConfig) -> None:
        """Fetch one source, validate it, persist to Redis, publish to ZMQ.

        Args:
            source: The indicator source to fetch this cycle.
        """
        try:
            value = await source.fetch()
            snapshot = MacroSnapshot(
                indicator=source.indicator, source=source.source_name, value=value
            )
        except Exception:
            self._log.exception(
                "macro_source_fetch_failed",
                agent_id=self.agent_id,
                indicator=source.indicator,
                source=source.source_name,
            )
            return

        if self._redis_client is not None:
            try:
                await self._redis_client.execute(
                    "set",
                    f"macro_snapshot:{snapshot.indicator}",
                    snapshot.model_dump_json(),
                    ex=self._redis_ttl_seconds,
                )
            except RedisUnavailableError:
                self._log.warning(
                    "macro_redis_write_failed",
                    agent_id=self.agent_id,
                    indicator=snapshot.indicator,
                )

        await self.publish(_MACRO_DATA_TOPIC, snapshot.model_dump(mode="json"))
        self._log.info(
            "macro_snapshot_published",
            agent_id=self.agent_id,
            indicator=snapshot.indicator,
            source=snapshot.source,
            value=snapshot.value,
        )

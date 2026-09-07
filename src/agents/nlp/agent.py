"""NLP sentiment ingestion micro-agent: fetch headlines, infer, publish.

:class:`NlpAgent` polls a financial news source on a fixed interval and,
for each headline, runs :class:`~src.agents.nlp.sentiment_engine.
SentimentEngine` (FinBERT) inference. The inference itself is CPU-bound
and synchronous by design (see that module's docstring); this agent is
where the async/sync boundary is actually crossed, via
``loop.run_in_executor`` against a dedicated ``ThreadPoolExecutor`` --
the mechanism that keeps every other agent's I/O responsive while a
sentiment call is computing.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import zmq.asyncio
from pydantic import SecretStr

from core.base_agent import BaseAgent
from core.config import AppSettings
from core.http_client import ResilientHttpClient
from core.redis_client import RedisUnavailableError
from core.schemas import SentimentSignal
from src.agents.nlp.sentiment_engine import SentimentEngine

_NLP_SIGNAL_TOPIC = "NLP_SIGNAL"
_NEWS_API_URL: Final[str] = "https://newsapi.org/v2/top-headlines"

# Used when no NEWS_API_KEY is configured, so this agent runs end-to-end
# without requiring real credentials. Each cycle samples a random subset,
# simulating the turnover of a live feed.
_SIMULATED_HEADLINES: Final[tuple[tuple[str, str], ...]] = (
    ("The Federal Reserve signaled a pause in rate hikes amid cooling inflation.", "Reuters"),
    ("Tech stocks rally as earnings beat analyst expectations across the board.", "Bloomberg"),
    ("Oil prices tumble on oversupply concerns and weak demand forecasts.", "CNBC"),
    ("Major bank reports unexpected quarterly loss, shares plunge in early trading.", "Reuters"),
    ("Central bank governor warns of persistent inflationary pressures ahead.", "Bloomberg"),
    ("Markets shrug off geopolitical tensions, indices close flat for the session.", "CNBC"),
    ("Regulators approve merger, clearing the way for industry consolidation.", "Reuters"),
    ("Consumer confidence index falls to lowest level in eighteen months.", "Bloomberg"),
)


@dataclass(frozen=True)
class RawHeadline:
    """One unanalyzed headline pulled from the news source.

    Attributes:
        headline: The headline text itself.
        source: Publisher name (e.g. ``"Reuters"``).
    """

    headline: str
    source: str


class NlpAgent(BaseAgent):
    """Polls financial news and publishes FinBERT sentiment signals.

    Falls back to a simulated headline feed when no
    :class:`~core.config.NewsApiSettings` API key is configured, so this
    agent runs end-to-end without requiring real credentials.
    """

    def __init__(
        self,
        *,
        poll_interval_seconds: float = 300.0,
        headlines_per_cycle: int = 5,
        max_inference_workers: int = 2,
        redis_ttl_seconds: int = 60,
        news_api_url: str = _NEWS_API_URL,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Configure the agent, load the sentiment model, and size its pool.

        Args:
            poll_interval_seconds: Time between the start of one polling
                cycle and the next.
            headlines_per_cycle: Max number of headlines fetched/analyzed
                per cycle.
            max_inference_workers: Size of the dedicated
                ``ThreadPoolExecutor`` FinBERT inference is dispatched to.
                Kept small: CPU-bound inference gains little from more
                worker threads than physical cores can actually run
                concurrently, and a large pool just wastes memory (each
                thread's call still shares the one loaded model).
            redis_ttl_seconds: TTL applied to each ``SentimentSignal`` key
                written to Redis. Deliberately short: a news headline's
                market impact is short-lived, unlike a macro indicator's.
            news_api_url: News API endpoint. Overridable for testing
                against a local mock server.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
        """
        super().__init__(
            "nlp-agent",
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._poll_interval_seconds = poll_interval_seconds
        self._headlines_per_cycle = headlines_per_cycle
        self._redis_ttl_seconds = redis_ttl_seconds
        self._news_api_url = news_api_url

        self._http_client = ResilientHttpClient(max_concurrent_requests=5, timeout_seconds=15.0)
        # Loaded once here, synchronously, before the event loop is even
        # running (agent construction happens before asyncio.run()) --
        # not inside an async method, so this cannot block the loop. See
        # SentimentEngine's docstring for why get_instance() is safe to
        # call repeatedly across agents in the same process.
        self._sentiment_engine = SentimentEngine.get_instance()
        self._executor = ThreadPoolExecutor(
            max_workers=max_inference_workers, thread_name_prefix="nlp-inference"
        )

        self._news_api_key: SecretStr | None = self._settings.news.api_key
        if self._news_api_key is None:
            self._log.warning(
                "nlp_news_source_using_simulated_data",
                agent_id=self.agent_id,
                reason="NEWS_API_KEY not configured",
            )

    # -- News source ---------------------------------------------------

    async def _fetch_headlines(self) -> list[RawHeadline]:
        """Fetch this cycle's headlines from the real API or the simulated feed."""
        if self._news_api_key is not None:
            return await self._fetch_headlines_from_news_api()
        return self._fetch_simulated_headlines()

    async def _fetch_headlines_from_news_api(self) -> list[RawHeadline]:
        """Fetch top business headlines from the configured news API.

        Returns:
            Up to ``headlines_per_cycle`` headlines, skipping any article
            with a missing title.
        """
        assert self._news_api_key is not None
        data = await self._http_client.get_json(
            self._news_api_url,
            params={
                "category": "business",
                "language": "en",
                "pageSize": str(self._headlines_per_cycle),
                "apiKey": self._news_api_key.get_secret_value(),
            },
        )
        headlines: list[RawHeadline] = []
        for article in data.get("articles", []):
            title = article.get("title")
            if not title:
                continue
            source_info = article.get("source") or {}
            source_name = source_info.get("name") or "UNKNOWN"
            headlines.append(RawHeadline(headline=title, source=source_name))
        return headlines

    def _fetch_simulated_headlines(self) -> list[RawHeadline]:
        """Sample a random subset of the built-in simulated headline pool."""
        sample_size = min(self._headlines_per_cycle, len(_SIMULATED_HEADLINES))
        chosen = random.sample(_SIMULATED_HEADLINES, sample_size)
        return [RawHeadline(headline=headline, source=source) for headline, source in chosen]

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent polls on its own schedule inside :meth:`run`.

        :meth:`run` is fully overridden with a fetch-then-sleep loop
        instead of :class:`~core.base_agent.BaseAgent`'s tick loop, so
        this is never called. Implemented (rather than omitted) only to
        satisfy the abstract base class, and raises loudly if it is ever
        invoked by mistake.
        """
        raise NotImplementedError("NlpAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Open the underlying ``aiohttp`` session."""
        await self._http_client.connect()
        self._log.info("nlp_agent_ready", agent_id=self.agent_id)

    async def on_shutdown(self) -> None:
        """Close the ``aiohttp`` session and drain the inference thread pool.

        ``executor.shutdown(wait=True)`` blocks until every in-flight
        inference call finishes -- correct for a clean shutdown, but a
        genuinely blocking call. It's dispatched via ``asyncio.to_thread``
        (onto asyncio's own default executor, not the one being shut
        down) so waiting for it still doesn't block the event loop.
        """
        await self._http_client.close()
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    async def run(self) -> None:
        """Connect dependencies, then poll for headlines on a fixed interval.

        Mirrors :meth:`BaseAgent.run`'s lifecycle (signal handling, Redis
        connection, cancellation-safe shutdown) but replaces the
        per-tick loop with a fetch-then-sleep polling loop.

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
        """Fetch this cycle's headlines, then analyze/persist/publish each.

        Every headline is analyzed concurrently: each
        :meth:`_process_headline` call submits its inference to the same
        bounded ``ThreadPoolExecutor``, which is what actually caps real
        concurrency -- the ``asyncio.gather`` here just avoids waiting
        for one headline's inference before dispatching the next.
        """
        try:
            headlines = await self._fetch_headlines()
        except Exception:
            self._log.exception("nlp_headline_fetch_failed", agent_id=self.agent_id)
            return

        self._log.info(
            "nlp_fetch_cycle_started", agent_id=self.agent_id, headline_count=len(headlines)
        )
        await asyncio.gather(*(self._process_headline(headline) for headline in headlines))

    async def _process_headline(self, headline: RawHeadline) -> None:
        """Run inference on one headline, then validate, persist, and publish it.

        Args:
            headline: The raw headline to analyze.
        """
        try:
            signal = await self._analyze_headline(headline)
        except Exception:
            self._log.exception(
                "nlp_analysis_failed", agent_id=self.agent_id, headline=headline.headline
            )
            return

        if self._redis_client is not None:
            digest = hashlib.sha256(signal.headline.encode("utf-8")).hexdigest()[:16]
            try:
                await self._redis_client.execute(
                    "set",
                    f"nlp_signal:{digest}",
                    signal.model_dump_json(),
                    ex=self._redis_ttl_seconds,
                )
            except RedisUnavailableError:
                self._log.warning("nlp_redis_write_failed", agent_id=self.agent_id)

        await self.publish(_NLP_SIGNAL_TOPIC, signal.model_dump(mode="json"))
        self._log.info(
            "nlp_signal_published",
            agent_id=self.agent_id,
            headline=signal.headline,
            source=signal.source,
            polarity=signal.polarity.value,
            score=signal.score,
        )

    async def _analyze_headline(self, headline: RawHeadline) -> SentimentSignal:
        """Run FinBERT inference on ``headline`` off the event loop thread.

        This is the non-blocking mechanic the whole agent exists to
        demonstrate: ``analyze_text`` is a synchronous, CPU-bound call,
        so it is dispatched to :attr:`_executor` via
        ``loop.run_in_executor`` rather than awaited directly -- the
        event loop remains free to service every other agent's I/O for
        the full duration of the inference call.

        Args:
            headline: The raw headline to analyze.

        Returns:
            The validated :class:`~core.schemas.SentimentSignal`.
        """
        loop = asyncio.get_running_loop()
        result: dict[str, Any] = await loop.run_in_executor(
            self._executor, self._sentiment_engine.analyze_text, headline.headline
        )
        return SentimentSignal(headline=headline.headline, source=headline.source, **result)

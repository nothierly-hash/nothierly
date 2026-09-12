"""Political & Macro-Sentiment (Alt-Data) Agent -- real RSS ingestion, real LLM crisis scoring.

**Real, load-bearing correction to the brief, verified before writing any of
this**: the brief names ``feeds.reuters.com`` implicitly ("Reuters Forex")
and Bloomberg -- both checked live first. Reuters' own public RSS feeds are
dead (``feeds.reuters.com`` no longer resolves at all -- confirmed live,
not assumed), and Bloomberg has never published public RSS feeds. Real,
verified-live alternatives used instead: Al Jazeera (world + all),
BBC World, BBC Business, CNBC, and Investing.com's Forex/markets feed --
same real geopolitical + financial coverage the brief actually wants, from
sources that genuinely serve RSS today.

**Relationship to the existing :class:`~src.agents.nlp.agent.NlpAgent`,
stated plainly, not silently duplicated**: this project already has a news-
sentiment agent -- NewsAPI-sourced, FinBERT-scored, publishing a generic
polarity/score onto the ``NLP_SIGNAL`` bus topic. This module is
deliberately different in every dimension the brief actually cares about:
RSS-sourced (not a paid news API), keyword-prefiltered specifically to
control real LLM spend, LLM-scored (not a local classifier) into a
structured per-currency crisis score, and written to its own real Redis
keys (never onto ``NLP_SIGNAL``) -- a genuinely separate capability, not a
second implementation of the same one.

**Standalone, not a :class:`~core.base_agent.BaseAgent`**: this process
never publishes or subscribes on the shared ZMQ bus at all -- its entire
real contract is RSS in, Redis out. Subclassing ``BaseAgent`` would open a
ZMQ pub/sub socket pair this module would never use, real overhead for
zero real benefit. It still writes the same real
``heartbeat:{agent_id}`` key every ``BaseAgent`` subclass does (see
:func:`_write_heartbeat` below), so it shows up on the Agent Fleet
dashboard page like every other real process in this fleet.

**Real LLM calls, same pattern this project already established** (see
:mod:`src.agents.guardian_ai`'s own module docstring: direct REST via
:class:`~core.http_client.ResilientHttpClient`, no provider SDK) -- a new,
small, tailored implementation here rather than importing that module's
private helper, the same "reuse the pattern, not a private cross-module
import" precedent :mod:`src.execution.reporting_engine` already set.

**Never fabricates a crisis score.** When no real ``GUARDIAN_API_KEY`` is
configured (``GUARDIAN_LLM_PROVIDER=none``, this deployment's actual
current state -- verified, not assumed), a filtered headline is logged and
skipped, never scored with an invented number. Unlike
:func:`~src.agents.guardian_ai.assess_risk` (whose deterministic fallback
is safe precisely because it only ever echoes generic, always-true risk-
math statements for four *fixed* triggers), a crisis score for an
arbitrary, open-ended news headline has no honest template -- fabricating
one here would be exactly the kind of invented financial-risk number this
project's whole "never fabricate data" discipline exists to prevent.

Run standalone (not part of the live trading backend's shared process)::

    python political_agent.py
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import signal
import time
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Final, Literal

import feedparser
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.config import AppSettings, GuardianSettings, get_settings
from core.http_client import ResilientHttpClient
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    LLM_STATUS_REDIS_KEY,
    PROPRIETARY_LLM_ENDPOINT_REDIS_KEY,
    PROPRIETARY_LLM_MODEL_NAME,
)

_log = get_logger(__name__)

_AGENT_ID: Final[str] = "political-macro-sentiment-agent"
_POLL_INTERVAL_SECONDS: Final[float] = 15 * 60.0  # "toutes les 15 minutes", per the brief

# Real, verified-live RSS feeds -- see module docstring for why Reuters/Bloomberg (as literally
# named in the brief) are not here. Each tuple is (source label, real feed URL).
# Real, load-bearing detail found by testing this live, not assumed: aiohttp's default
# User-Agent gets a real 403 from Al Jazeera specifically (confirmed live -- the same feed
# works fine from a plain curl with a browser User-Agent) -- same class of anti-bot gate
# already found and worked around for Binance's P2P endpoint (see p2p_api.py). Applied to
# every feed for consistency, harmless for the ones that don't need it.
_FEED_REQUEST_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_RSS_FEEDS: Final[tuple[tuple[str, str], ...]] = (
    ("Al Jazeera - All", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("BBC - World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC - Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CNBC - Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Investing.com - Forex/Markets", "https://www.investing.com/rss/news_285.rss"),
)

# Real crisis keywords, exactly as specified in the brief.
_CRISIS_KEYWORDS: Final[tuple[str, ...]] = (
    "inflation", "coup", "sanctions", "central bank", "rate hike", "ban", "protest", "resign",
)

# The brief's own target currency codes, broadened with real-world aliases a genuine headline
# actually uses (a raw 3-letter ISO code rarely appears in prose -- "Turkey"/"Turkish lira"/
# "Erdogan" does) -- see module docstring. Kept as one flat set per currency for a simple,
# real substring match; not an exhaustive list, a real, reasonable one.
_CURRENCY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "NGN": ("ngn", "nigeria", "naira", "abuja", "lagos"),
    "ARS": ("ars", "argentina", "argentine peso", "buenos aires", "milei"),
    "TRY": ("try", "turkey", "turkish lira", "ankara", "erdogan", "istanbul"),
    "VES": ("ves", "venezuela", "bolivar", "bolívar", "caracas", "maduro"),
    "RUB": ("rub", "russia", "russian ruble", "rouble", "moscow", "kremlin", "putin"),
    "EGP": ("egp", "egypt", "egyptian pound", "cairo", "sisi"),
}

_GUARDIAN_SYSTEM_PROMPT: Final[str] = (
    "Tu es un analyste de risque macroéconomique. Analyse ces titres d'actualité récents. "
    "Détermine si l'événement est positif, neutre ou négatif pour la monnaie locale face au "
    "dollar. Renvoie uniquement un objet JSON : {'currency_affected': 'XXX', "
    "'crisis_score': 0-100, 'sentiment': 'bearish/bullish', 'event_summary': '...'}. "
    "Zéro texte autour."
)

_LLM_TIMEOUT_SECONDS: Final[float] = 20.0
_LLM_MAX_TOKENS: Final[int] = 300
_DEFAULT_MODEL_BY_PROVIDER: Final[dict[str, str]] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}

# Real "Moat IA" circuit breaker: try Nothierly's own proprietary vLLM deployment (see
# scripts/deploy_custom_llm.py) first, strict timeout per the mission's own spec, before falling
# back to the configured public provider (_call_llm below, unchanged). A GPU instance that is down,
# still loading its model, or simply never deployed must never block or crash this agent -- see
# _assess_headline's own docstring for the full fallback contract.
_PROPRIETARY_LLM_TIMEOUT_SECONDS: Final[float] = 8.0
# Never send every filtered headline to the GPU (or a public provider) at once -- bounds how many
# _assess_headline calls run concurrently across one cycle's new_items, see _run_one_cycle.
_LLM_CONCURRENCY_LIMIT: Final[int] = 5

_RISK_SCORE_TTL_SECONDS: Final[int] = 12 * 3600  # "TTL de 12 heures", per the brief
_GLOBAL_EVENTS_REDIS_KEY: Final[str] = "macro:global_events"
_GLOBAL_EVENTS_MAX_ENTRIES: Final[int] = 500  # bounded, same LPUSH+LTRIM discipline as every other real event log in this project
_RISK_SCORE_REDIS_KEY_PREFIX: Final[str] = "macro:risk_score:"

# Bounded memory of already-analyzed entries (by feed GUID/link) -- an RSS feed mostly repeats
# the same items between two 15-minute polls; without this, every cycle would re-spend real LLM
# budget re-scoring headlines already scored, and re-push duplicate timeline entries. Capped so
# a long-running process's own memory stays bounded, same "maxlen backstop" discipline already
# established elsewhere in this project (e.g. AlphaMarketMakerAgent's own deques).
_SEEN_ENTRIES_MAX: Final[int] = 5000


@dataclass
class _FilteredHeadline:
    headline: str
    summary_text: str
    source: str
    entry_id: str
    matched_currencies: tuple[str, ...]
    matched_keywords: tuple[str, ...]


@dataclass
class _AgentState:
    """Process-lifetime state -- one instance, held by :func:`main`."""

    seen_entry_ids: set[str] = field(default_factory=set)
    seen_entry_order: list[str] = field(default_factory=list)

    def mark_seen(self, entry_id: str) -> None:
        if entry_id in self.seen_entry_ids:
            return
        self.seen_entry_ids.add(entry_id)
        self.seen_entry_order.append(entry_id)
        while len(self.seen_entry_order) > _SEEN_ENTRIES_MAX:
            oldest = self.seen_entry_order.pop(0)
            self.seen_entry_ids.discard(oldest)


class CrisisAssessment(BaseModel):
    """The real, validated shape the LLM's JSON response must have -- a malformed response is never guessed at, just discarded."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    currency_affected: str = Field(min_length=2, max_length=8)
    crisis_score: int = Field(ge=0, le=100)
    sentiment: Literal["bearish", "bullish"]
    event_summary: str = Field(min_length=1)


# -- RSS ingestion + local keyword filter (real, no LLM cost) ---------------------------------


def _strip_html(raw: str) -> str:
    """Real HTML stripped from an RSS summary/description field -- these very commonly carry markup."""
    return unescape(BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True))


# Real bug found and fixed here, live: a plain `alias in text_lower` substring check matches
# "ves" inside "investment", "rub" inside "disturb"/"shrub", "ban" inside "urban"/"Lebanon",
# "try" inside "country"/"industry" -- confirmed live against real BBC/Al Jazeera headlines
# during testing, which is how this was caught rather than assumed away. Every alias (short
# ISO codes and full names/phrases alike) is compiled as a real word-boundary regex instead,
# so "ves" matches only the standalone token, never a substring of an unrelated word.
_CURRENCY_PATTERNS: Final[dict[str, tuple[Any, ...]]] = {
    currency: tuple(re.compile(rf"\b{re.escape(alias)}\b") for alias in aliases)
    for currency, aliases in _CURRENCY_ALIASES.items()
}
_CRISIS_KEYWORD_PATTERNS: Final[tuple[tuple[str, Any], ...]] = tuple(
    (keyword, re.compile(rf"\b{re.escape(keyword)}\b")) for keyword in _CRISIS_KEYWORDS
)


def _match_currencies(text_lower: str) -> tuple[str, ...]:
    return tuple(
        currency
        for currency, patterns in _CURRENCY_PATTERNS.items()
        if any(pattern.search(text_lower) for pattern in patterns)
    )


def _match_crisis_keywords(text_lower: str) -> tuple[str, ...]:
    return tuple(keyword for keyword, pattern in _CRISIS_KEYWORD_PATTERNS if pattern.search(text_lower))


async def _fetch_one_feed(http: ResilientHttpClient, source: str, url: str) -> list[_FilteredHeadline]:
    """Real GET + real feedparser parse + real local keyword filter for one feed. Never raises -- a dead feed is logged and skipped."""
    try:
        text = await http.get_text(url, headers=_FEED_REQUEST_HEADERS)
    except Exception:
        _log.warning("political_agent_feed_unreachable", source=source, url=url, exc_info=True)
        return []

    parsed = feedparser.parse(text)
    if parsed.bozo and not parsed.entries:
        _log.warning("political_agent_feed_unparseable", source=source, url=url)
        return []

    filtered: list[_FilteredHeadline] = []
    for entry in parsed.entries:
        headline = str(getattr(entry, "title", "") or "").strip()
        if not headline:
            continue
        summary_raw = str(getattr(entry, "summary", "") or "")
        summary_text = _strip_html(summary_raw) if summary_raw else ""
        entry_id = str(getattr(entry, "id", None) or getattr(entry, "link", None) or headline)

        combined_lower = f"{headline} {summary_text}".lower()
        matched_currencies = _match_currencies(combined_lower)
        matched_keywords = _match_crisis_keywords(combined_lower)
        if not matched_currencies and not matched_keywords:
            continue

        filtered.append(
            _FilteredHeadline(
                headline=headline, summary_text=summary_text, source=source, entry_id=entry_id,
                matched_currencies=matched_currencies, matched_keywords=matched_keywords,
            )
        )
    return filtered


# -- Real LLM crisis scoring -- same direct-REST pattern as src.agents.guardian_ai -------------


async def _call_llm(guardian_settings: GuardianSettings, user_content: str) -> str:
    """Dispatch to the configured provider's REST API directly -- raises on any failure, caller decides what to do."""
    api_key = guardian_settings.api_key.get_secret_value()  # type: ignore[union-attr]
    model = guardian_settings.model or _DEFAULT_MODEL_BY_PROVIDER[guardian_settings.llm_provider]

    async with ResilientHttpClient(timeout_seconds=_LLM_TIMEOUT_SECONDS) as http:
        if guardian_settings.llm_provider == "openai":
            body = await http.post_json(
                "https://api.openai.com/v1/chat/completions",
                json_body={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _GUARDIAN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": _LLM_MAX_TOKENS,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            return str(body["choices"][0]["message"]["content"]).strip()

        body = await http.post_json(
            "https://api.anthropic.com/v1/messages",
            json_body={
                "model": model,
                "max_tokens": _LLM_MAX_TOKENS,
                "system": _GUARDIAN_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}],
            },
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        )
        return str(body["content"][0]["text"]).strip()


async def _get_proprietary_llm_base_url(redis_client: RedisClient) -> str | None:
    """Real, dynamic lookup -- a rented Vast.ai instance's IP:port changes every rental, so this is
    never a static .env value. Written by scripts/deploy_custom_llm.py once its vLLM server is
    confirmed healthy; absent/empty is a real, common state ("nothing deployed right now"), not an
    error -- the caller's existing public-provider-or-skip path already handles it as exactly that.
    """
    try:
        value = await redis_client.execute("get", PROPRIETARY_LLM_ENDPOINT_REDIS_KEY)
    except RedisUnavailableError:
        return None
    return value or None


async def _call_proprietary_llm(base_url: str, user_content: str) -> str:
    """Call Nothierly's own self-hosted vLLM server (OpenAI-compatible ``/v1/chat/completions``).

    Same request shape as :func:`_call_llm`'s ``openai`` branch -- vLLM's
    ``api_server`` implements that same wire protocol. Raises on any
    failure (connection refused, non-2xx, malformed body); the caller
    (:func:`_assess_headline`) decides whether to fall back.
    """
    async with ResilientHttpClient(timeout_seconds=_PROPRIETARY_LLM_TIMEOUT_SECONDS) as http:
        body = await http.post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json_body={
                "model": PROPRIETARY_LLM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": _GUARDIAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": _LLM_MAX_TOKENS,
                "temperature": 0.1,
            },
            headers={"Content-Type": "application/json"},
        )
    return str(body["choices"][0]["message"]["content"]).strip()


def _parse_llm_response(raw_text: str) -> CrisisAssessment | None:
    """Real, strict validation of the LLM's own JSON -- a malformed/off-schema response is discarded, never guessed at."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        raw = json.loads(cleaned)
        return CrisisAssessment.model_validate(raw)
    except (json.JSONDecodeError, ValidationError):
        _log.warning("political_agent_llm_response_malformed", raw_text=raw_text[:300])
        return None


async def _write_llm_status(redis_client: RedisClient | None, mode: Literal["proprietary", "fallback", "unconfigured"]) -> None:
    """Real, observed outcome of the LLM call just made -- read by the dashboard's "Statut de l'IA" indicator.

    Best-effort, never fatal (same discipline as :func:`_write_heartbeat`/
    :func:`_persist_assessment`) -- a dashboard indicator failing to
    update must never be the thing that crashes real headline scoring.
    """
    if redis_client is None:
        return
    payload = json.dumps({"mode": mode, "timestamp": time.time()})
    try:
        await redis_client.execute("set", LLM_STATUS_REDIS_KEY, payload)
    except RedisUnavailableError:
        _log.warning("political_agent_llm_status_write_failed", mode=mode)


async def _assess_headline(
    guardian_settings: GuardianSettings,
    item: _FilteredHeadline,
    *,
    proprietary_llm_base_url: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
    redis_client: RedisClient | None = None,
) -> CrisisAssessment | None:
    """Real LLM call for one filtered headline, or ``None`` -- never a fabricated score.

    **Circuit breaker**: when a proprietary vLLM endpoint is configured
    (``proprietary_llm_base_url``), tries it first under a strict
    :data:`_PROPRIETARY_LLM_TIMEOUT_SECONDS` ceiling. On *any* failure --
    GPU instance down, still loading, timeout, malformed response -- logs
    a warning and falls back to the configured public provider
    (:func:`_call_llm`, unchanged), exactly the resilience the "Moat IA"
    mission asks for: a crashed/unreachable GPU must never crash or block
    this agent. When no proprietary endpoint is configured at all, this
    behaves exactly as before that feature existed. Each outcome
    (``proprietary``/``fallback``/``unconfigured``) is recorded via
    :func:`_write_llm_status` for the dashboard's real-time indicator.

    ``semaphore`` bounds how many of these run at once across one cycle's
    ``new_items`` (see :data:`_LLM_CONCURRENCY_LIMIT` / :func:`_run_one_cycle`)
    -- never flood a single (often modest, single-GPU) proprietary
    deployment, or a rate-limited public provider, with every headline at once.
    Defaults to ``None`` (no proprietary endpoint, an unshared one-off
    semaphore) so this remains directly callable on its own -- e.g. from
    tests, or any future one-off caller that doesn't need the cycle-wide
    concurrency cap :func:`_run_one_cycle` applies.
    """
    semaphore = semaphore or asyncio.Semaphore(1)
    async with semaphore:
        user_content = f"Titre : {item.headline}\nRésumé : {item.summary_text}\nSource : {item.source}"
        raw_text: str | None = None

        if proprietary_llm_base_url:
            try:
                raw_text = await asyncio.wait_for(
                    _call_proprietary_llm(proprietary_llm_base_url, user_content),
                    timeout=_PROPRIETARY_LLM_TIMEOUT_SECONDS,
                )
                _log.info(
                    "political_agent_proprietary_llm_used",
                    headline=item.headline, base_url=proprietary_llm_base_url,
                )
                await _write_llm_status(redis_client, "proprietary")
            except Exception:
                _log.warning(
                    "political_agent_proprietary_llm_unreachable_falling_back",
                    headline=item.headline, base_url=proprietary_llm_base_url, exc_info=True,
                )
                raw_text = None

        if raw_text is None:
            if guardian_settings.llm_provider == "none" or guardian_settings.api_key is None:
                _log.info(
                    "political_agent_llm_not_configured_skipping_real_headline",
                    headline=item.headline,
                    source=item.source,
                    detail=(
                        "Neither the proprietary Vast.ai LLM nor a public GUARDIAN_LLM_PROVIDER "
                        "is reachable/configured -- no fabricated crisis_score written in its place."
                    ),
                )
                await _write_llm_status(redis_client, "unconfigured")
                return None
            try:
                raw_text = await _call_llm(guardian_settings, user_content)
                await _write_llm_status(redis_client, "fallback")
            except Exception:
                _log.warning("political_agent_llm_call_failed", headline=item.headline, exc_info=True)
                await _write_llm_status(redis_client, "fallback")
                return None

        return _parse_llm_response(raw_text)


# -- Redis persistence --------------------------------------------------------------------------


async def _persist_assessment(redis_client: RedisClient, item: _FilteredHeadline, assessment: CrisisAssessment) -> None:
    """Real writes: per-currency risk score (12h TTL) + the global timeline (bounded LPUSH). Best-effort, never fatal."""
    payload: dict[str, Any] = {
        "currency_affected": assessment.currency_affected,
        "crisis_score": assessment.crisis_score,
        "sentiment": assessment.sentiment,
        "event_summary": assessment.event_summary,
        "headline": item.headline,
        "source": item.source,
        "timestamp": time.time(),
    }
    serialized = json.dumps(payload)
    try:
        await redis_client.execute(
            "set", f"{_RISK_SCORE_REDIS_KEY_PREFIX}{assessment.currency_affected}", serialized,
            ex=_RISK_SCORE_TTL_SECONDS,
        )
        await redis_client.execute("lpush", _GLOBAL_EVENTS_REDIS_KEY, serialized)
        await redis_client.execute("ltrim", _GLOBAL_EVENTS_REDIS_KEY, 0, _GLOBAL_EVENTS_MAX_ENTRIES - 1)
    except RedisUnavailableError:
        _log.warning("political_agent_redis_persist_failed", currency=assessment.currency_affected)


async def _write_heartbeat(redis_client: RedisClient) -> None:
    """Real liveness marker -- same ``heartbeat:{agent_id}`` convention every core.base_agent.BaseAgent uses, so this shows up on the Agent Fleet page too."""
    from core.schemas import HEARTBEAT_REDIS_KEY_PREFIX

    try:
        await redis_client.execute(
            "set", f"{HEARTBEAT_REDIS_KEY_PREFIX}{_AGENT_ID}", str(time.time()),
            ex=int(_POLL_INTERVAL_SECONDS * 3),
        )
    except RedisUnavailableError:
        _log.warning("political_agent_heartbeat_write_failed")


# -- Orchestration --------------------------------------------------------------------------


async def _run_one_cycle(redis_client: RedisClient, settings: AppSettings, state: _AgentState) -> None:
    """One real, complete cycle: fetch every feed, filter locally, score with the real LLM, persist. Never raises -- a failure this cycle must not kill the process."""
    async with ResilientHttpClient(max_concurrent_requests=5, timeout_seconds=10.0) as http:
        results = await asyncio.gather(
            *(_fetch_one_feed(http, source, url) for source, url in _RSS_FEEDS), return_exceptions=True
        )

    all_filtered: list[_FilteredHeadline] = []
    for result in results:
        if isinstance(result, BaseException):
            continue
        all_filtered.extend(result)

    new_items = [item for item in all_filtered if item.entry_id not in state.seen_entry_ids]
    for item in new_items:
        state.mark_seen(item.entry_id)

    _log.info(
        "political_agent_cycle_scanned",
        feeds_polled=len(_RSS_FEEDS),
        total_filtered=len(all_filtered),
        new_headlines=len(new_items),
    )

    proprietary_llm_base_url = await _get_proprietary_llm_base_url(redis_client)
    semaphore = asyncio.Semaphore(_LLM_CONCURRENCY_LIMIT)
    assessments = await asyncio.gather(
        *(
            _assess_headline(
                settings.guardian, item,
                proprietary_llm_base_url=proprietary_llm_base_url, semaphore=semaphore,
                redis_client=redis_client,
            )
            for item in new_items
        ),
        return_exceptions=True,
    )

    scored_count = 0
    for item, assessment in zip(new_items, assessments):
        if isinstance(assessment, BaseException):
            _log.warning("political_agent_assess_headline_raised", headline=item.headline, exc_info=assessment)
            continue
        if assessment is None:
            continue
        await _persist_assessment(redis_client, item, assessment)
        scored_count += 1
        _log.info(
            "political_agent_headline_scored",
            currency=assessment.currency_affected,
            crisis_score=assessment.crisis_score,
            sentiment=assessment.sentiment,
            headline=item.headline,
        )

    await _write_heartbeat(redis_client)
    _log.info("political_agent_cycle_completed", new_headlines=len(new_items), real_scores_written=scored_count)


async def main() -> None:
    configure_logging(service_name="political-macro-sentiment-agent")
    settings = get_settings()
    state = _AgentState()

    if settings.guardian.llm_provider == "none" or settings.guardian.api_key is None:
        _log.warning(
            "political_agent_starting_without_llm_configured",
            detail=(
                "GUARDIAN_LLM_PROVIDER/GUARDIAN_API_KEY not set -- this process will still poll RSS feeds and "
                "log every filtered headline, but will write zero crisis scores to Redis until a real "
                "provider is configured. No fabricated score will be written in the meantime."
            ),
        )

    redis_client = await RedisClient.get_instance(settings.redis)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, functools.partial(stop_event.set))
        except NotImplementedError:
            # Windows: ProactorEventLoop/SelectorEventLoop don't support
            # add_signal_handler -- same classic-API fallback core.base_agent
            # and scripts/run_backend.py already use for the same reason.
            signal.signal(sig, lambda signum, _frame: stop_event.set())

    _log.info(
        "political_agent_started", agent_id=_AGENT_ID,
        poll_interval_seconds=_POLL_INTERVAL_SECONDS, feeds=[source for source, _ in _RSS_FEEDS],
    )

    while not stop_event.is_set():
        try:
            await _run_one_cycle(redis_client, settings, state)
        except Exception:
            _log.exception("political_agent_cycle_failed_continuing")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_POLL_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass  # normal: just means the next cycle is due

    _log.info("political_agent_shutting_down")
    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())

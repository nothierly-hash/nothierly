"""Real, executable tests for analyst_agent.py.

Reuses political_agent.py's own real functions unchanged (``_fetch_one_feed``, ``_assess_headline``)
-- these tests exercise ``AnalystAgent._handle_anomaly``'s own real wiring/status-derivation logic,
monkeypatching only the real network boundary (``_fetch_one_feed``, ``political_agent._call_llm``),
the exact same "fake I/O only at the edge" discipline ``tests/test_political_agent.py`` already
established for the module this reuses.
"""

from __future__ import annotations

import json
from typing import Any, Generator

import fakeredis.aioredis
import pytest

import analyst_agent
import political_agent
from analyst_agent import AnalystAgent
from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import SWARM_ANALYSIS_REDIS_KEY_PREFIX
from political_agent import _FilteredHeadline


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


def _agent(redis_client: RedisClient, *, guardian: GuardianSettings | None = None) -> AnalystAgent:
    settings = AppSettings(guardian=guardian) if guardian is not None else None
    instance = AnalystAgent(settings=settings, manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    return instance


def _decode(raw: dict[Any, Any]) -> dict[str, str]:
    """Same real, defensive decode as risk_gate_agent.py's own -- see that module's comment."""
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else k): (v.decode("utf-8") if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


def _headline(fiat: str, text: str = "Central bank announcement") -> _FilteredHeadline:
    return _FilteredHeadline(
        headline=text, summary_text="", source="Test", entry_id="1", matched_currencies=(fiat,), matched_keywords=(),
    )


@pytest.mark.asyncio
async def test_no_matching_headlines_when_every_feed_is_empty(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def empty_feed(*args: Any, **kwargs: Any) -> list[_FilteredHeadline]:
        return []

    monkeypatch.setattr(analyst_agent, "_fetch_one_feed", empty_feed)
    agent = _agent(redis_client)

    await agent._handle_anomaly("anomaly-1", "ARS")

    raw = await redis_client.execute("hgetall", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}anomaly-1")
    assert _decode(raw)["status"] == "NO_MATCHING_HEADLINES"


@pytest.mark.asyncio
async def test_llm_unavailable_when_no_real_provider_is_configured(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This deployment's own real, current state (GUARDIAN_LLM_PROVIDER=none) -- the honest default,
    not mocked away."""
    async def one_real_match(http: Any, source: str, url: str) -> list[_FilteredHeadline]:
        return [_headline("TRY")] if source == "feed-a" else []

    monkeypatch.setattr(analyst_agent, "_fetch_one_feed", one_real_match)
    monkeypatch.setattr(analyst_agent, "_RSS_FEEDS", (("feed-a", "http://example.test/a"),))
    agent = _agent(redis_client, guardian=GuardianSettings(llm_provider="none"))

    await agent._handle_anomaly("anomaly-2", "TRY")

    raw = await redis_client.execute("hgetall", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}anomaly-2")
    assert _decode(raw)["status"] == "LLM_UNAVAILABLE"


@pytest.mark.asyncio
async def test_crisis_confirmed_when_real_llm_score_clears_the_threshold(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def one_real_match(http: Any, source: str, url: str) -> list[_FilteredHeadline]:
        return [_headline("VES", "Venezuela bolivar collapses")] if source == "feed-a" else []

    async def fake_llm_call(guardian_settings: Any, user_content: str) -> str:
        return '{"currency_affected": "VES", "crisis_score": 85, "sentiment": "bearish", "event_summary": "Real collapse."}'

    monkeypatch.setattr(analyst_agent, "_fetch_one_feed", one_real_match)
    monkeypatch.setattr(analyst_agent, "_RSS_FEEDS", (("feed-a", "http://example.test/a"),))
    monkeypatch.setattr(political_agent, "_call_llm", fake_llm_call)
    agent = _agent(redis_client, guardian=GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test"))  # type: ignore[arg-type]

    await agent._handle_anomaly("anomaly-3", "VES")

    raw = _decode(await redis_client.execute("hgetall", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}anomaly-3"))
    assert raw["status"] == "CRISIS_CONFIRMED"
    assert int(raw["crisis_score"]) == 85
    assert raw["sentiment"] == "bearish"


@pytest.mark.asyncio
async def test_no_crisis_when_real_llm_score_stays_below_the_threshold(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def one_real_match(http: Any, source: str, url: str) -> list[_FilteredHeadline]:
        return [_headline("ARS")] if source == "feed-a" else []

    async def fake_llm_call(guardian_settings: Any, user_content: str) -> str:
        return '{"currency_affected": "ARS", "crisis_score": 20, "sentiment": "bullish", "event_summary": "Minor news."}'

    monkeypatch.setattr(analyst_agent, "_fetch_one_feed", one_real_match)
    monkeypatch.setattr(analyst_agent, "_RSS_FEEDS", (("feed-a", "http://example.test/a"),))
    monkeypatch.setattr(political_agent, "_call_llm", fake_llm_call)
    agent = _agent(redis_client, guardian=GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test"))  # type: ignore[arg-type]

    await agent._handle_anomaly("anomaly-4", "ARS")

    raw = _decode(await redis_client.execute("hgetall", f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}anomaly-4"))
    assert raw["status"] == "NO_CRISIS"
    assert int(raw["crisis_score"]) == 20

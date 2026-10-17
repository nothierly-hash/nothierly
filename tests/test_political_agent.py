"""Real, executable tests for political_agent.py.

No real network/LLM call: external boundaries (:func:`political_agent._call_llm`,
:class:`~core.redis_client.RedisClient`) are faked, same "fake I/O only at the edge"
discipline as every other test in this suite. The word-boundary regression tests below lock
in a real bug found live during development (see :data:`political_agent._CURRENCY_PATTERNS`'s
own comment) -- a naive substring filter matched "ves" inside "investment" and "ban" inside
"urban", the opposite of the brief's own "Anti-Spam" intent.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.config import GuardianSettings
from political_agent import (
    _PROPRIETARY_LLM_TIMEOUT_SECONDS,
    CrisisAssessment,
    _AgentState,
    _assess_headline,
    _FilteredHeadline,
    _match_crisis_keywords,
    _match_currencies,
    _parse_llm_response,
    _persist_assessment,
    _strip_html,
)


# -- Word-boundary filter regression -------------------------------------------------------


def test_ves_does_not_false_match_inside_investment() -> None:
    """Real bug found live: a naive 'ves' in text substring check matched 'investment'."""
    assert _match_currencies("google announces a major investment in finland") == ()


def test_ban_does_not_false_match_inside_urban_or_lebanon() -> None:
    assert _match_crisis_keywords("urban planning reform in the city") == ()
    assert _match_crisis_keywords("tensions rise across lebanon this week") == ()


def test_try_does_not_false_match_inside_country_or_industry() -> None:
    assert _match_currencies("the country's industry is growing fast") == ()


def test_rub_does_not_false_match_inside_shrub_or_disturb() -> None:
    assert _match_currencies("gardeners trim every shrub without disturbing the birds") == ()


def test_ves_matches_as_a_real_standalone_currency_reference() -> None:
    assert "VES" in _match_currencies("venezuela's economy remains in crisis")
    assert "VES" in _match_currencies("maduro addressed the nation on state television")


def test_ban_matches_as_a_real_standalone_crisis_keyword() -> None:
    assert "ban" in _match_crisis_keywords("the government announced a new import ban today")


def test_full_country_and_leader_names_match_correctly() -> None:
    assert _match_currencies("erdogan spoke about turkish lira volatility") == ("TRY",)
    assert _match_currencies("the central bank of nigeria raised rates") == ("NGN",)


def test_multiple_currencies_can_match_one_headline() -> None:
    matched = _match_currencies("both russia and turkey face new sanctions this week")
    assert set(matched) == {"RUB", "TRY"}


def test_no_match_returns_empty_tuple_not_none() -> None:
    assert _match_currencies("apple unveils new iphone at annual event") == ()
    assert _match_crisis_keywords("apple unveils new iphone at annual event") == ()


# -- HTML stripping -----------------------------------------------------------------------


def test_strip_html_removes_real_markup_and_unescapes_entities() -> None:
    raw = "<p>Turkey&#8217;s central bank <b>raised</b> rates</p>"
    assert _strip_html(raw) == "Turkey’s central bank raised rates"


# -- LLM response parsing (real validation, never guessed at) ------------------------------


def test_parse_llm_response_accepts_a_real_well_formed_json() -> None:
    raw = '{"currency_affected": "TRY", "crisis_score": 72, "sentiment": "bearish", "event_summary": "Rate hike surprise."}'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result.currency_affected == "TRY"
    assert result.crisis_score == 72
    assert result.sentiment == "bearish"


def test_parse_llm_response_strips_markdown_code_fences() -> None:
    raw = '```json\n{"currency_affected": "ARS", "crisis_score": 50, "sentiment": "bullish", "event_summary": "x"}\n```'
    result = _parse_llm_response(raw)
    assert result is not None
    assert result.currency_affected == "ARS"


def test_parse_llm_response_rejects_malformed_json_returns_none_not_a_guess() -> None:
    assert _parse_llm_response("this is not json at all") is None


def test_parse_llm_response_rejects_out_of_range_crisis_score() -> None:
    raw = '{"currency_affected": "TRY", "crisis_score": 150, "sentiment": "bearish", "event_summary": "x"}'
    assert _parse_llm_response(raw) is None


def test_parse_llm_response_rejects_invalid_sentiment_value() -> None:
    raw = '{"currency_affected": "TRY", "crisis_score": 50, "sentiment": "neutral", "event_summary": "x"}'
    assert _parse_llm_response(raw) is None  # only bearish/bullish are valid, exactly per the brief's own schema


# -- Never fabricate when no LLM is configured ----------------------------------------------


@pytest.mark.asyncio
async def test_assess_headline_returns_none_when_no_llm_configured() -> None:
    unconfigured = GuardianSettings(llm_provider="none")
    item = _FilteredHeadline(
        headline="Turkey central bank raises rates", summary_text="", source="Test",
        entry_id="1", matched_currencies=("TRY",), matched_keywords=("central bank",),
    )
    result = await _assess_headline(unconfigured, item)
    assert result is None


@pytest.mark.asyncio
async def test_assess_headline_returns_none_on_llm_failure_never_a_fabricated_score(monkeypatch: pytest.MonkeyPatch) -> None:
    import political_agent

    async def failing_call(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(political_agent, "_call_llm", failing_call)
    configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
    item = _FilteredHeadline(
        headline="Turkey central bank raises rates", summary_text="", source="Test",
        entry_id="1", matched_currencies=("TRY",), matched_keywords=("central bank",),
    )
    result = await _assess_headline(configured, item)
    assert result is None


@pytest.mark.asyncio
async def test_assess_headline_returns_real_parsed_assessment_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import political_agent

    async def fake_call(guardian_settings: Any, user_content: str) -> str:
        assert "Turkey" in user_content
        return '{"currency_affected": "TRY", "crisis_score": 80, "sentiment": "bearish", "event_summary": "Surprise hike."}'

    monkeypatch.setattr(political_agent, "_call_llm", fake_call)
    configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
    item = _FilteredHeadline(
        headline="Turkey central bank raises rates", summary_text="", source="Test",
        entry_id="1", matched_currencies=("TRY",), matched_keywords=("central bank",),
    )
    result = await _assess_headline(configured, item)
    assert result is not None
    assert result.currency_affected == "TRY"
    assert result.crisis_score == 80


# -- "Moat IA" circuit breaker: proprietary Vast.ai LLM -> public fallback -------------------
#
# Real, permanent regression coverage for the mission's own task 5 validation requirement
# ("force un plantage virtuel de l'IP Vast.ai, vérifie que le try/except redirige bien vers
# l'API de secours sans crasher l'agent") -- a deterministic monkeypatch of
# _call_proprietary_llm, not a real network call to an unreachable IP (which would make this
# suite flaky/slow depending on the sandbox's own network stack behavior on an unroutable
# address). Both real, distinct failure modes the mission's own pseudocode names are covered:
# an immediate refusal (ConnectionRefusedError, "GPU instance not running") and a timeout
# (the GPU is up but unresponsive/still loading) -- see each test's own docstring.


@pytest.mark.asyncio
async def test_circuit_breaker_falls_back_when_proprietary_gpu_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real simulation of a crashed/down Vast.ai instance (immediate refusal) -- the proprietary
    call must never crash the agent, and the public fallback must still produce a real, correctly
    parsed assessment."""
    import political_agent

    async def crashed_gpu(base_url: str, user_content: str) -> str:
        raise ConnectionRefusedError("simulated: Vast.ai instance unreachable")

    async def working_fallback(guardian_settings: Any, user_content: str) -> str:
        return '{"currency_affected": "ARS", "crisis_score": 55, "sentiment": "bearish", "event_summary": "Fallback used."}'

    monkeypatch.setattr(political_agent, "_call_proprietary_llm", crashed_gpu)
    monkeypatch.setattr(political_agent, "_call_llm", working_fallback)
    configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
    item = _FilteredHeadline(
        headline="Argentina peso slides on inflation data", summary_text="", source="Test",
        entry_id="1", matched_currencies=("ARS",), matched_keywords=("inflation",),
    )

    result = await _assess_headline(
        configured, item, proprietary_llm_base_url="http://10.255.255.1:8000",
    )

    assert result is not None
    assert result.currency_affected == "ARS"
    assert result.crisis_score == 55


@pytest.mark.asyncio
async def test_circuit_breaker_respects_strict_proprietary_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real simulation of a GPU that is up but never responds (still loading the model, hung
    process, ...) -- proves the strict 8s ceiling (mission's own spec) actually bounds the wait,
    rather than silently inheriting the much longer public-provider timeout on top of a hang."""
    import asyncio
    import time

    import political_agent

    async def hung_gpu(base_url: str, user_content: str) -> str:
        await asyncio.sleep(_PROPRIETARY_LLM_TIMEOUT_SECONDS + 30.0)
        return "unreachable"  # never reached

    async def working_fallback(guardian_settings: Any, user_content: str) -> str:
        return '{"currency_affected": "VES", "crisis_score": 70, "sentiment": "bearish", "event_summary": "Fallback used."}'

    monkeypatch.setattr(political_agent, "_call_proprietary_llm", hung_gpu)
    monkeypatch.setattr(political_agent, "_call_llm", working_fallback)
    configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
    item = _FilteredHeadline(
        headline="Venezuela bolivar collapses amid sanctions", summary_text="", source="Test",
        entry_id="1", matched_currencies=("VES",), matched_keywords=("sanctions",),
    )

    started = time.monotonic()
    result = await _assess_headline(
        configured, item, proprietary_llm_base_url="http://10.255.255.1:8000",
    )
    elapsed = time.monotonic() - started

    assert result is not None
    assert result.currency_affected == "VES"
    # Real, generous margin above the strict ceiling for scheduler jitter -- must not be anywhere
    # close to the 30s the hung call would otherwise have blocked for.
    assert elapsed < _PROPRIETARY_LLM_TIMEOUT_SECONDS + 5.0


@pytest.mark.asyncio
async def test_circuit_breaker_never_calls_proprietary_when_no_endpoint_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Vast.ai deployment right now (this project's real, current state) -- must behave exactly
    like the pre-"Moat IA" code path, never attempting a call to an empty/``None`` base URL."""
    import political_agent

    async def must_not_be_called(base_url: str, user_content: str) -> str:
        raise AssertionError("proprietary LLM must never be called when no endpoint is configured")

    async def working_fallback(guardian_settings: Any, user_content: str) -> str:
        return '{"currency_affected": "TRY", "crisis_score": 30, "sentiment": "bullish", "event_summary": "ok"}'

    monkeypatch.setattr(political_agent, "_call_proprietary_llm", must_not_be_called)
    monkeypatch.setattr(political_agent, "_call_llm", working_fallback)
    configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
    item = _FilteredHeadline(
        headline="Turkey central bank raises rates", summary_text="", source="Test",
        entry_id="1", matched_currencies=("TRY",), matched_keywords=("central bank",),
    )

    result = await _assess_headline(configured, item, proprietary_llm_base_url=None)

    assert result is not None
    assert result.currency_affected == "TRY"


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrent_llm_calls() -> None:
    """Real proof of task 3's own requirement: never send every filtered headline to the LLM at
    once. Runs 12 concurrent _assess_headline calls through a real asyncio.Semaphore(3) and
    asserts the observed peak concurrency never exceeds it."""
    import asyncio

    import political_agent

    peak_concurrency = 0
    current_concurrency = 0
    concurrency_limit = 3

    async def slow_fallback(guardian_settings: Any, user_content: str) -> str:
        nonlocal peak_concurrency, current_concurrency
        current_concurrency += 1
        peak_concurrency = max(peak_concurrency, current_concurrency)
        await asyncio.sleep(0.05)
        current_concurrency -= 1
        return '{"currency_affected": "TRY", "crisis_score": 10, "sentiment": "bullish", "event_summary": "ok"}'

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(political_agent, "_call_llm", slow_fallback)
    try:
        configured = GuardianSettings(llm_provider="anthropic", api_key="fake-key-for-test")  # type: ignore[arg-type]
        semaphore = asyncio.Semaphore(concurrency_limit)
        items = [
            _FilteredHeadline(
                headline=f"Headline {i}", summary_text="", source="Test",
                entry_id=str(i), matched_currencies=("TRY",), matched_keywords=(),
            )
            for i in range(12)
        ]

        results = await asyncio.gather(
            *(_assess_headline(configured, item, semaphore=semaphore) for item in items)
        )
    finally:
        monkeypatch.undo()

    assert all(result is not None for result in results)
    assert peak_concurrency <= concurrency_limit


# -- Dedup / bounded seen-entries state -----------------------------------------------------


def test_agent_state_deduplicates_seen_entries() -> None:
    state = _AgentState()
    state.mark_seen("entry-1")
    state.mark_seen("entry-1")
    assert len(state.seen_entry_order) == 1
    assert "entry-1" in state.seen_entry_ids


def test_agent_state_prunes_beyond_max_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _AgentState()
    import political_agent

    # monkeypatch.setattr (not a plain assignment) -- _SEEN_ENTRIES_MAX is a real, deliberate
    # Final module constant; setattr sidesteps that at the type level the same way it already
    # does for every other module-level constant this test suite patches, and restores the
    # original value automatically, no manual try/finally bookkeeping needed.
    monkeypatch.setattr(political_agent, "_SEEN_ENTRIES_MAX", 3)
    for i in range(5):
        state.mark_seen(f"entry-{i}")
    assert len(state.seen_entry_order) == 3
    assert "entry-0" not in state.seen_entry_ids  # oldest pruned first
    assert "entry-4" in state.seen_entry_ids


# -- Redis persistence (fake client, real key/TTL contract) --------------------------------


class _FakeRedisClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def execute(self, command: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((command, args, kwargs))
        return None


@pytest.mark.asyncio
async def test_persist_assessment_writes_real_keys_with_correct_ttl_and_bounds() -> None:
    fake_redis = _FakeRedisClient()
    item = _FilteredHeadline(
        headline="Turkey central bank raises rates", summary_text="", source="Test",
        entry_id="1", matched_currencies=("TRY",), matched_keywords=("central bank",),
    )
    assessment = CrisisAssessment(
        currency_affected="TRY", crisis_score=80, sentiment="bearish", event_summary="Surprise hike."
    )
    await _persist_assessment(fake_redis, item, assessment)  # type: ignore[arg-type]

    commands = [call[0] for call in fake_redis.calls]
    assert commands == ["set", "lpush", "ltrim"]

    set_call = fake_redis.calls[0]
    assert set_call[1][0] == "macro:risk_score:TRY"
    assert set_call[2]["ex"] == 12 * 3600

    lpush_call = fake_redis.calls[1]
    assert lpush_call[1][0] == "macro:global_events"

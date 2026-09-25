"""Real, executable tests for src/agents/alert_agent.py.

Redis is faked with ``fakeredis`` (same discipline as
``tests/test_gateway.py``/``tests/test_quant_copilot.py`` -- a real
in-memory Redis implementation, fresh ``FakeServer()`` per test for
isolation). ZMQ publishing is intercepted with the same
``_RecordingZmqPublisher`` stand-in ``test_gateway.py`` uses (no
real-fake library exists for ZMQ). Twilio's SDK is monkeypatched at
``alert_agent.TwilioClient`` -- these tests must never place a real
network call to Twilio (there are no real credentials in this
environment, and even if there were, a test run must never send a real
SMS).
"""

from __future__ import annotations

import json
import time
from typing import Any, Generator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import TwilioSettings
from core.redis_client import RedisClient
from core.schemas import HEARTBEAT_REDIS_KEY_PREFIX, MARKET_METRICS_REDIS_KEY_PREFIX
from core.security import STATUS_PANIC, SYSTEM_STATUS_KEY
from src.agents import alert_agent as alert_agent_module
from src.agents.alert_agent import AlertAgent, send_sms_alert


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


class _RecordingTwilioClient:
    """Stand-in for twilio.rest.Client -- records calls, never touches the network."""

    calls: list[dict[str, str]] = []

    def __init__(self, account_sid: str, auth_token: str) -> None:
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.messages = self

    def create(self, *, body: str, from_: str, to: str) -> None:
        _RecordingTwilioClient.calls.append({"body": body, "from_": from_, "to": to})


@pytest.fixture(autouse=True)
def reset_twilio_recording() -> None:
    _RecordingTwilioClient.calls = []


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """A real RedisClient over a real in-memory fakeredis server, ALSO installed as the process singleton.

    core.security.trip_kill_switch/is_system_safe deliberately always
    go through RedisClient.get_instance() (the real, correct design --
    the Kill Switch must be one true process-wide flag, not per-agent
    state) -- bypassing that singleton would leave those calls silently
    talking to whatever *real* Redis happens to be configured in this
    environment instead of this test's isolated data. Installing this
    fake client as the singleton (and resetting it after) is what makes
    kill-switch assertions in these tests actually observe what the
    test itself set up.
    """
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture
def agent(redis_client: RedisClient) -> Generator[AlertAgent, None, None]:
    """Closes the REAL publisher/subscriber/context BaseAgent.__init__ opened -- see
    tests/test_execution_agent_instrument_lock.py's identical fixture for why this is not optional."""
    instance = AlertAgent(manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    real_publisher = instance._publisher
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    # Real Twilio settings (shape-only, never touches the network -- see
    # TwilioClient monkeypatching in each test that needs it) so
    # send_sms_alert's own "unconfigured, skip" short-circuit doesn't
    # mask whether _raise_alert actually fired.
    instance._settings.twilio = _configured_twilio_settings()
    yield instance
    real_publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def _redis(agent: AlertAgent) -> RedisClient:
    """Type-narrowing helper: agent._redis_client is RedisClient | None on BaseAgent; the fixture always sets it."""
    assert agent._redis_client is not None
    return agent._redis_client


def _configured_twilio_settings() -> TwilioSettings:
    from pydantic import SecretStr

    return TwilioSettings(
        account_sid="ACtest", auth_token=SecretStr("test-token"), phone_number="+15550000000", admin_phone_number="+15551111111"
    )


@pytest.mark.asyncio
async def test_send_sms_alert_skips_when_unconfigured() -> None:
    sent = await send_sms_alert(TwilioSettings(), message="test")
    assert sent is False
    assert _RecordingTwilioClient.calls == []


@pytest.mark.asyncio
async def test_send_sms_alert_calls_real_twilio_client_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    sent = await send_sms_alert(_configured_twilio_settings(), message="[URGENT] test message")
    assert sent is True
    assert len(_RecordingTwilioClient.calls) == 1
    assert _RecordingTwilioClient.calls[0]["to"] == "+15551111111"
    assert _RecordingTwilioClient.calls[0]["from_"] == "+15550000000"
    assert _RecordingTwilioClient.calls[0]["body"] == "[URGENT] test message"


@pytest.mark.asyncio
async def test_microstructure_check_ignores_normal_obi(agent: AlertAgent) -> None:
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.1, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )
    await agent._check_microstructure_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_microstructure_check_alerts_on_catastrophic_obi(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.95, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )
    await agent._check_microstructure_danger()

    published = agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].topic == "EMERGENCY_HALT"
    assert published[0].data["category"] == "MICROSTRUCTURE"

    system_status = await _redis(agent).execute("get", SYSTEM_STATUS_KEY)
    # This fakeredis version doesn't fully honor decode_responses=True
    # for every command (a fakeredis quirk, not a real-Redis behavior --
    # real redis-py correctly returns str here) -- decode defensively
    # rather than asserting on a specific str/bytes shape.
    if isinstance(system_status, bytes):
        system_status = system_status.decode()
    assert system_status == STATUS_PANIC
    assert len(_RecordingTwilioClient.calls) == 1


def _decoded(value: object) -> str | None:
    """Same real fakeredis quirk already documented above -- see test_microstructure_check_alerts_on_catastrophic_obi."""
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


@pytest.mark.asyncio
async def test_off_suppresses_sms_kill_switch_and_emergency_halt(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    agent._control_state = AgentControlState.OFF
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.95, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )

    await agent._check_microstructure_danger()

    assert agent._publisher.published == []  # type: ignore[attr-defined]
    assert _RecordingTwilioClient.calls == []
    assert _decoded(await _redis(agent).execute("get", SYSTEM_STATUS_KEY)) != STATUS_PANIC


@pytest.mark.asyncio
async def test_off_does_not_consume_the_debounce(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """A danger suppressed by OFF must still be able to alert for real once the operator switches back to LIVE."""
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.95, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )

    agent._control_state = AgentControlState.OFF
    await agent._check_microstructure_danger()
    assert _RecordingTwilioClient.calls == []

    agent._control_state = AgentControlState.LIVE
    await agent._check_microstructure_danger()

    assert len(_RecordingTwilioClient.calls) == 1
    assert _decoded(await _redis(agent).execute("get", SYSTEM_STATUS_KEY)) == STATUS_PANIC


@pytest.mark.asyncio
async def test_shadow_logs_the_alert_but_sends_no_sms_or_kill_switch(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    agent._control_state = AgentControlState.SHADOW
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.95, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )

    await agent._check_microstructure_danger()

    assert agent._publisher.published == []  # type: ignore[attr-defined]
    assert _RecordingTwilioClient.calls == []
    assert _decoded(await _redis(agent).execute("get", SYSTEM_STATUS_KEY)) != STATUS_PANIC

    history = await _redis(agent).execute("lrange", "RED_PHONE_ALERT_LOG", 0, -1)
    assert len(history) == 1
    entry = json.loads(history[0])
    assert entry["category"] == "MICROSTRUCTURE"
    assert entry["sms_sent"] is False


@pytest.mark.asyncio
async def test_shadow_still_consumes_the_debounce(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike OFF, SHADOW performs a real action (the alert-history log) -- so it must still be rate-limited."""
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    agent._control_state = AgentControlState.SHADOW
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"order_book_imbalance": 0.95, "spread": 0.5, "mid_price": 65000.0, "timestamp": "2026-09-02T20:00:00Z"}),
    )

    await agent._check_microstructure_danger()
    await agent._check_microstructure_danger()

    history = await _redis(agent).execute("lrange", "RED_PHONE_ALERT_LOG", 0, -1)
    assert len(history) == 1  # the second call's alert was debounced, not double-logged


@pytest.mark.asyncio
async def test_economic_check_reacts_only_to_drawdown_trigger(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await _redis(agent).execute(
        "lpush", "NOTHIERLY_GUARDIAN_ALERTS",
        json.dumps({"timestamp": time.time(), "trigger": "SIGNAL_COLLISION", "summary": "not a drawdown", "verdict": "x", "source": "fallback"}),
    )
    await agent._check_economic_danger()
    assert agent._publisher.published == []  # type: ignore[attr-defined]

    await _redis(agent).execute(
        "lpush", "NOTHIERLY_GUARDIAN_ALERTS",
        json.dumps({"timestamp": time.time(), "trigger": "DRAWDOWN", "summary": "capital -6% en 24h", "verdict": "x", "source": "fallback"}),
    )
    await agent._check_economic_danger()
    published = agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) == 1
    assert published[0].data["category"] == "ECONOMIC"


@pytest.mark.asyncio
async def test_economic_check_never_reacts_twice_to_the_same_entry(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await _redis(agent).execute(
        "lpush", "NOTHIERLY_GUARDIAN_ALERTS",
        json.dumps({"timestamp": time.time(), "trigger": "DRAWDOWN", "summary": "capital -6%", "verdict": "x", "source": "fallback"}),
    )
    await agent._check_economic_danger()
    await agent._check_economic_danger()  # same entry, second tick
    assert len(_RecordingTwilioClient.calls) == 1  # only the first tick alerted


@pytest.mark.asyncio
async def test_infrastructure_check_requires_consecutive_misses(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await agent._check_infrastructure_danger()  # 1st miss -- no alert yet
    assert agent._publisher.published == []  # type: ignore[attr-defined]
    await agent._check_infrastructure_danger()  # 2nd consecutive miss -- alerts
    published = agent._publisher.published  # type: ignore[attr-defined]
    assert len(published) >= 1
    assert all(p.data["category"] == "INFRASTRUCTURE" for p in published)


@pytest.mark.asyncio
async def test_infrastructure_check_resets_streak_on_real_heartbeat(agent: AlertAgent) -> None:
    await agent._check_infrastructure_danger()  # 1st miss
    await _redis(agent).execute("set", f"{HEARTBEAT_REDIS_KEY_PREFIX}supervisor-agent", str(time.time()))
    await agent._check_infrastructure_danger()  # real heartbeat resets that agent's streak
    assert agent._infra_missing_streaks["supervisor-agent"] == 0


@pytest.mark.asyncio
async def test_debounce_suppresses_repeated_alerts_within_window(agent: AlertAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alert_agent_module, "TwilioClient", _RecordingTwilioClient)
    await agent._raise_alert(category="MICROSTRUCTURE", detail="first")
    await agent._raise_alert(category="MICROSTRUCTURE", detail="second, same category, should be debounced")
    assert len(_RecordingTwilioClient.calls) == 1
    assert len(agent._publisher.published) == 1  # type: ignore[attr-defined]

    # A different category is never debounced by another category's window.
    await agent._raise_alert(category="ECONOMIC", detail="different category")
    assert len(_RecordingTwilioClient.calls) == 2

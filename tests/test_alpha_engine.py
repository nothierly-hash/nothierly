"""Real, executable tests for src/agents/alpha_engine/agent.py.

Same disciplines as this project's other agent tests: Redis is faked
with ``fakeredis`` (fresh ``FakeServer()`` per test, see
``tests/test_alert_agent.py``'s ``redis_client`` fixture, reused
verbatim), ZMQ publishing is intercepted with a real recording stand-in
(no mocking framework -- see ``tests/test_gateway.py``), and
:class:`_FakeBroker` is a real, minimal :class:`~src.execution.
broker_gateway.BaseBroker` implementation (not a mock) -- its
``place_market_order``/``place_limit_order`` are genuinely
``@requires_system_active``-decorated the same way the real
``AsyncExchangeClient`` methods they stand in for actually are, so the
Kill Switch tests below exercise the real decorator, same discipline as
``tests/test_smart_order_router.py``'s ``_FakeVenue``.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Generator, Literal

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import RedisSettings
from core.nothierly_ontology import AssetClass, MarginSnapshot, UniversalAsset, UniversalPosition
from core.notifier import TelegramNotifier
from core.redis_client import RedisClient
from core.session_manager import MarketSessionManager
from core.schemas import (
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX,
    MARKET_METRICS_REDIS_KEY_PREFIX,
)
from core.security import requires_system_active, trip_kill_switch
from execution.predator_honeypot import DefensiveExecutionMode
from src.agents.alpha_engine.agent import (
    AlphaMarketMakerAgent,
    _apply_fill,
    _clamp_quote_sizes,
    _defensive_half_spread_bps,
    _skewed_quote_prices,
)
from network.resilience_mesh import RateLimitMesh
from src.execution.broker_gateway import BaseBroker

_ASSET = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)


# =============================================================================
# _skewed_quote_prices
# =============================================================================


def test_skewed_quote_prices_symmetric_at_zero_inventory() -> None:
    bid, ask = _skewed_quote_prices(
        mid_price=Decimal("65000"), half_spread_bps=Decimal("5"),
        inventory=Decimal(0), max_inventory=Decimal("0.5"), skew_bps=Decimal("15"),
    )
    mid = (bid + ask) / 2
    assert mid == Decimal("65000")
    assert ask > bid


def test_skewed_quote_prices_long_inventory_leans_both_quotes_down() -> None:
    bid_flat, ask_flat = _skewed_quote_prices(
        mid_price=Decimal("65000"), half_spread_bps=Decimal("5"),
        inventory=Decimal(0), max_inventory=Decimal("0.5"), skew_bps=Decimal("15"),
    )
    bid_long, ask_long = _skewed_quote_prices(
        mid_price=Decimal("65000"), half_spread_bps=Decimal("5"),
        inventory=Decimal("0.5"), max_inventory=Decimal("0.5"), skew_bps=Decimal("15"),
    )
    assert bid_long < bid_flat
    assert ask_long < ask_flat


# =============================================================================
# _clamp_quote_sizes
# =============================================================================


def test_clamp_quote_sizes_both_sides_open_when_flat() -> None:
    buy, sell = _clamp_quote_sizes(base_size=Decimal("0.01"), inventory=Decimal(0), max_inventory=Decimal("0.5"))
    assert buy == Decimal("0.01")
    assert sell == Decimal("0.01")


def test_clamp_quote_sizes_buy_side_closes_at_cap() -> None:
    buy, sell = _clamp_quote_sizes(base_size=Decimal("0.01"), inventory=Decimal("0.5"), max_inventory=Decimal("0.5"))
    assert buy == Decimal(0)
    assert sell == Decimal("0.01")


def test_clamp_quote_sizes_sell_side_closes_at_negative_cap() -> None:
    buy, sell = _clamp_quote_sizes(base_size=Decimal("0.01"), inventory=Decimal("-0.5"), max_inventory=Decimal("0.5"))
    assert buy == Decimal("0.01")
    assert sell == Decimal(0)


# =============================================================================
# _defensive_half_spread_bps
# =============================================================================


def test_defensive_half_spread_normal_mode_unchanged() -> None:
    assert _defensive_half_spread_bps(
        Decimal("5"), mode=DefensiveExecutionMode.NORMAL, cautious_multiplier=Decimal("2"),
    ) == Decimal("5")


def test_defensive_half_spread_cautious_mode_widened_by_multiplier() -> None:
    assert _defensive_half_spread_bps(
        Decimal("5"), mode=DefensiveExecutionMode.CAUTIOUS, cautious_multiplier=Decimal("2"),
    ) == Decimal("10")


def test_defensive_half_spread_post_only_mode_also_widened() -> None:
    """Not reachable via this engine's own wiring today (see module docstring), but the function
    itself states one total rule rather than a NORMAL-vs-CAUTIOUS-only special case."""
    assert _defensive_half_spread_bps(
        Decimal("5"), mode=DefensiveExecutionMode.POST_ONLY_ONLY, cautious_multiplier=Decimal("3"),
    ) == Decimal("15")


# =============================================================================
# _apply_fill
# =============================================================================


def test_apply_fill_opens_from_flat() -> None:
    qty, avg_entry, realized = _apply_fill(
        previous_qty=Decimal(0), previous_avg_entry_price=None, fill_qty=Decimal("0.01"), fill_price=Decimal("65000"),
    )
    assert qty == Decimal("0.01")
    assert avg_entry == Decimal("65000")
    assert realized == Decimal(0)


def test_apply_fill_adds_same_direction_blends_cost_basis() -> None:
    qty, avg_entry, realized = _apply_fill(
        previous_qty=Decimal("0.01"), previous_avg_entry_price=Decimal("65000"),
        fill_qty=Decimal("0.01"), fill_price=Decimal("66000"),
    )
    assert qty == Decimal("0.02")
    assert avg_entry == Decimal("65500")
    assert realized == Decimal(0)


def test_apply_fill_partial_close_realizes_pnl_keeps_cost_basis() -> None:
    qty, avg_entry, realized = _apply_fill(
        previous_qty=Decimal("0.02"), previous_avg_entry_price=Decimal("65000"),
        fill_qty=Decimal("-0.01"), fill_price=Decimal("66000"),
    )
    assert qty == Decimal("0.01")
    assert avg_entry == Decimal("65000")  # unchanged -- remaining lot keeps its original cost basis
    assert realized == Decimal("10")  # 0.01 * (66000 - 65000)


def test_apply_fill_exact_close_clears_cost_basis() -> None:
    qty, avg_entry, realized = _apply_fill(
        previous_qty=Decimal("0.01"), previous_avg_entry_price=Decimal("65000"),
        fill_qty=Decimal("-0.01"), fill_price=Decimal("64000"),
    )
    assert qty == Decimal(0)
    assert avg_entry is None
    assert realized == Decimal("-10")  # 0.01 * (64000 - 65000), a real loss


def test_apply_fill_flips_through_flat_opens_new_lot() -> None:
    qty, avg_entry, realized = _apply_fill(
        previous_qty=Decimal("0.01"), previous_avg_entry_price=Decimal("65000"),
        fill_qty=Decimal("-0.03"), fill_price=Decimal("64000"),
    )
    assert qty == Decimal("-0.02")
    assert avg_entry == Decimal("64000")  # the new short lot's own entry price, not blended with the old long
    assert realized == Decimal("-10")  # only the 0.01 that actually closed the old long realizes anything


# =============================================================================
# Agent-level tests -- real fakeredis + a real, minimal BaseBroker
# =============================================================================


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


class _FakeBroker(BaseBroker):
    """A real, minimal BaseBroker -- see module docstring."""

    def __init__(self, *, position_qty: Decimal = Decimal(0), supports_cancel_all: bool = True) -> None:
        self.broker_id = "fake"
        self.position_qty = position_qty
        self.limit_orders: list[tuple[str, Literal["buy", "sell"], float, float]] = []
        self.market_orders: list[tuple[str, Literal["buy", "sell"], float]] = []
        self.cancelled_order_ids: list[str] = []
        self.supports_cancel_all = supports_cancel_all
        self.orphaned_orders_to_report: list[dict[str, Any]] = []
        self.cancel_all_open_orders_calls: list[str | None] = []

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def cancel_all_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if not self.supports_cancel_all:
            raise NotImplementedError("this fake broker doesn't support it, for this one test")
        self.cancel_all_open_orders_calls.append(symbol)
        return self.orphaned_orders_to_report

    # The decorator's own Awaitable[...] wrapper type vs. BaseBroker's Coroutine[...] abstract signature
    # below trips mypy's LSP override check -- not a real violation (same real, deliberate decoration
    # AsyncExchangeClient.create_market_order carries).
    @requires_system_active
    async def place_market_order(  # type: ignore[override]
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        self.market_orders.append((symbol, side, quantity))
        return {"id": client_order_id, "status": "closed", "filled": quantity}

    async def emergency_flatten_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        """Real override -- NOT @requires_system_active-decorated, mirroring CcxtBrokerAdapter's real one."""
        self.market_orders.append((symbol, side, quantity))
        return {"id": client_order_id, "status": "closed", "filled": quantity}

    # Same note as place_market_order above.
    @requires_system_active
    async def place_limit_order(  # type: ignore[override]
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, price: float,
        *, client_order_id: str | None = None,
    ) -> dict[str, Any]:
        self.limit_orders.append((symbol, side, quantity, price))
        return {"id": client_order_id, "status": "open"}

    async def cancel_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        self.cancelled_order_ids.append(order_id)
        return {"id": order_id, "status": "cancelled"}

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> Any:
        raise NotImplementedError

    async def get_current_price(self, symbol: str) -> float:
        return 65000.0

    async def get_account_balance(self) -> float:
        return 100_000.0

    async def get_margin_snapshot(self) -> MarginSnapshot:
        return MarginSnapshot(broker_id=self.broker_id, account_balance_usd=Decimal(100_000))

    async def get_open_positions(self) -> list[UniversalPosition]:
        if self.position_qty == 0:
            return []
        return [
            UniversalPosition(
                broker_id=self.broker_id, asset=_ASSET, quantity=self.position_qty,
                avg_entry_price=Decimal("65000"), mark_price=Decimal("65000"), unrealized_pnl_usd=Decimal(0),
            )
        ]


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """Real fakeredis-backed RedisClient, also installed as the process singleton -- see tests/test_alert_agent.py."""
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


# BaseAgent.__init__ opens real ZMQ sockets/contexts; unclosed ones make pytest hang at exit inside
# zmq.Context.__del__ -> term() (the recurring fixture-leak pattern of this repo's agent tests).
_LEAKED_ZMQ: list[tuple[Any, Any, Any]] = []


@pytest.fixture(autouse=True)
def _close_real_zmq_sockets() -> Generator[None, None, None]:
    yield
    while _LEAKED_ZMQ:
        publisher, subscriber, context = _LEAKED_ZMQ.pop()
        publisher.close()
        subscriber.close()
        context.term()


def _make_agent(redis_client: RedisClient, **kwargs: Any) -> AlphaMarketMakerAgent:
    """Shared construction, real fakeredis + a real minimal broker -- used by the `agent` fixture and any test that needs non-default constructor args (e.g. max_daily_loss_usd)."""
    defaults: dict[str, Any] = dict(
        exchange_id="fake", manage_own_signals=False, manage_own_redis=False,
        max_inventory_units=Decimal("0.5"), quote_size_units=Decimal("0.01"),
    )
    defaults.update(kwargs)
    instance = AlphaMarketMakerAgent("BTC/USDT", **defaults)
    instance._redis_client = redis_client
    _LEAKED_ZMQ.append((instance._publisher, instance._subscriber, instance._zmq_context))
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._settings.mock_mode = False
    instance._broker = _FakeBroker()
    # A real RateLimitMesh, registered generously (1000/s) so no test is ever actually delayed by
    # real pacing -- these tests exercise order-placement logic, not RateLimitMesh's own throttle
    # timing (see tests/test_resilience_mesh.py for that).
    instance._rate_limit_mesh = RateLimitMesh()
    instance._rate_limit_mesh.register_broker("fake", requests_per_second=1000.0)
    # Unconfigured (no bot_token/chat_id) -- .send() logs and returns False, never touches the network,
    # same discipline as tests/test_alert_agent.py's send_sms_alert-skips-when-unconfigured test.
    instance._notifier = TelegramNotifier(instance._settings.telegram)
    return instance


@pytest.fixture
def agent(redis_client: RedisClient) -> AlphaMarketMakerAgent:
    return _make_agent(redis_client)


def _redis(agent: AlphaMarketMakerAgent) -> RedisClient:
    assert agent._redis_client is not None
    return agent._redis_client


def _broker(agent: AlphaMarketMakerAgent) -> _FakeBroker:
    assert isinstance(agent._broker, _FakeBroker)
    return agent._broker


async def _seed_fresh_market_metrics(
    agent: AlphaMarketMakerAgent, *, age_seconds: float = 0.0, spread: float = 1.0,
) -> None:
    timestamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"symbol": "BTC/USDT", "timestamp": timestamp.isoformat(), "spread": spread, "order_book_imbalance": 0.0, "mid_price": 65000.0}),
    )


@pytest.mark.asyncio
async def test_read_fresh_market_metrics_missing_key(agent: AlphaMarketMakerAgent) -> None:
    metrics, age = await agent._read_fresh_market_metrics()
    assert metrics is None
    assert age is None


@pytest.mark.asyncio
async def test_read_fresh_market_metrics_fresh(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent, age_seconds=0.5)
    metrics, age = await agent._read_fresh_market_metrics()
    assert metrics is not None
    assert age is not None and age < agent._max_market_data_age_seconds


@pytest.mark.asyncio
async def test_read_fresh_market_metrics_stale_is_rejected(agent: AlphaMarketMakerAgent) -> None:
    """The exact real gap this engine's own multi-agent-coordination audit found missing everywhere else."""
    await _seed_fresh_market_metrics(agent, age_seconds=agent._max_market_data_age_seconds + 1.0)
    metrics, age = await agent._read_fresh_market_metrics()
    assert metrics is None
    assert age is not None and age > agent._max_market_data_age_seconds


@pytest.mark.asyncio
async def test_read_fresh_market_metrics_unparseable_logs_the_real_reason_not_a_blind_preview(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-16 fix: found live analyzing a real 51h shadow run where 34k+ of these warnings
    carried only a 120-char raw_preview (always cut off before the field that actually failed) and
    no exception message at all -- undiagnosable after the fact. This must now carry the real
    Pydantic error and the full (capped) raw value."""
    logged: list[tuple[str, dict[str, object]]] = []
    original_warning = agent._log.warning

    def _recording_warning(event: str, **kwargs: object) -> None:
        logged.append((event, kwargs))
        original_warning(event, **kwargs)

    monkeypatch.setattr(agent._log, "warning", _recording_warning)

    # order_book_imbalance out of its real [-1, 1] range -- a genuine schema violation, not
    # malformed JSON, exercising the same `except Exception` path either shape hits.
    await _redis(agent).execute(
        "set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT",
        json.dumps({"symbol": "BTC/USDT", "timestamp": datetime.now(timezone.utc).isoformat(), "spread": 1.0, "order_book_imbalance": 1.5, "mid_price": 65000.0}),
    )

    metrics, age = await agent._read_fresh_market_metrics()

    assert metrics is None and age is None
    warnings = [kwargs for event, kwargs in logged if event == "alpha_engine_market_metrics_unparseable"]
    assert len(warnings) == 1
    assert warnings[0]["error_type"] == "ValidationError"
    assert "order_book_imbalance" in str(warnings[0]["error"])
    assert '"order_book_imbalance": 1.5' in str(warnings[0]["raw_value"])


@pytest.mark.asyncio
async def test_read_fresh_market_metrics_unparseable_caps_an_oversized_raw_value(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real safety half of the same fix: logging the full raw value must stay bounded -- a
    corrupted or adversarial value (see core/message_integrity.py's own docstring on who can write
    this key) must never turn one warning into an unbounded log line."""
    logged: list[tuple[str, dict[str, object]]] = []
    original_warning = agent._log.warning

    def _recording_warning(event: str, **kwargs: object) -> None:
        logged.append((event, kwargs))
        original_warning(event, **kwargs)

    monkeypatch.setattr(agent._log, "warning", _recording_warning)

    oversized_garbage = "{" + "x" * 5000
    await _redis(agent).execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}BTC/USDT", oversized_garbage)

    await agent._read_fresh_market_metrics()

    warnings = [kwargs for event, kwargs in logged if event == "alpha_engine_market_metrics_unparseable"]
    assert len(warnings) == 1
    raw_value = str(warnings[0]["raw_value"])
    # Real fakeredis quirk this codebase already documents (core/message_integrity.py's own
    # docstring on `sign`): decode_responses=True is not always honored, so `raw` may come back as
    # `bytes` here -- str(bytes) adds its own "b'...'" wrapper, so the exact total isn't pinned down
    # to len(oversized_garbage); what actually matters is that it stayed capped and says so honestly.
    assert len(raw_value) < 2100  # well below the raw 5000+-char garbage, whatever representation
    assert "...<truncated, " in raw_value and raw_value.endswith(" chars total>")


@pytest.mark.asyncio
async def test_off_control_state_places_no_orders(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent)
    agent._control_state = AgentControlState.OFF
    await agent._run_cycle()
    fake = _broker(agent)
    assert fake.limit_orders == []
    assert fake.market_orders == []


@pytest.mark.asyncio
async def test_live_cycle_places_two_sided_quote_when_flat(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    fake = _broker(agent)
    assert len(fake.limit_orders) == 2
    sides = {order[1] for order in fake.limit_orders}
    assert sides == {"buy", "sell"}


# =============================================================================
# VPIN-driven adverse-selection defense -- see module docstring's 2026-09-19 addendum
# =============================================================================


@pytest.mark.asyncio
async def test_live_cycle_widens_spread_and_shrinks_size_under_cautious_vpin(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real regression coverage for the fix: a toxic tape must not get the same spread/size as a calm one."""
    await _seed_fresh_market_metrics(agent)

    async def _fake_high_vpin() -> float:
        return 0.9  # >= AdverseFlowDefenseAgent's external_toxicity_threshold (0.7) -> CAUTIOUS

    monkeypatch.setattr(agent, "_fetch_current_vpin", _fake_high_vpin)

    await agent._run_cycle()

    fake = _broker(agent)
    assert len(fake.limit_orders) == 2
    for _symbol, _side, quantity, _price in fake.limit_orders:
        assert quantity == pytest.approx(0.003)  # 0.01 base size * 0.3 CAUTIOUS multiplier
    prices = sorted(price for *_rest, price in fake.limit_orders)
    # Widened half-spread (10 bps = 2x the 5 bps default) around a flat 65000 mid, zero skew (flat inventory).
    assert prices == pytest.approx([64935.0, 65065.0])


@pytest.mark.asyncio
async def test_live_cycle_normal_spread_when_vpin_below_threshold(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_fresh_market_metrics(agent)

    async def _fake_low_vpin() -> float:
        return 0.1

    monkeypatch.setattr(agent, "_fetch_current_vpin", _fake_low_vpin)

    await agent._run_cycle()

    fake = _broker(agent)
    assert len(fake.limit_orders) == 2
    for _symbol, _side, quantity, _price in fake.limit_orders:
        assert quantity == pytest.approx(0.01)  # unchanged -- posture stayed NORMAL


@pytest.mark.asyncio
async def test_telemetry_publishes_current_vpin_and_defensive_mode(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_fresh_market_metrics(agent)

    async def _fake_high_vpin() -> float:
        return 0.85

    monkeypatch.setattr(agent, "_fetch_current_vpin", _fake_high_vpin)
    await agent._run_cycle()

    raw = await _redis(agent).execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent.agent_id}")
    payload = json.loads(raw)
    assert payload["vpin"] == pytest.approx(0.85)
    assert payload["defensive_mode"] == "CAUTIOUS"


@pytest.mark.asyncio
async def test_missing_vpin_reading_keeps_last_known_score_not_a_reset_to_normal(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same disclosed conservative-by-omission behavior as MacroOrderExecutionAgent's own
    _fetch_current_vpin: a missing reading this cycle must not silently revert posture to NORMAL."""
    readings = iter([0.9, None])

    async def _fake_vpin() -> float | None:
        return next(readings)

    monkeypatch.setattr(agent, "_fetch_current_vpin", _fake_vpin)

    first = await agent._current_defensive_posture()
    second = await agent._current_defensive_posture()

    assert first.mode == DefensiveExecutionMode.CAUTIOUS
    assert second.mode == DefensiveExecutionMode.CAUTIOUS  # stale VPIN retained, not reset
    assert agent._last_vpin == pytest.approx(0.9)  # last known real value, not overwritten by None


@pytest.mark.asyncio
async def test_redis_client_none_reconnects_when_singleton_still_healthy(agent: AlphaMarketMakerAgent) -> None:
    """Real bug regression, found live via a dry run (2026-09-08): BaseAgent._write_heartbeat
    (core/base_agent.py:478) nulls agent._redis_client the instant one heartbeat write fails --
    but the process-wide RedisClient singleton it came from is untouched. A bare
    `assert self._redis_client is not None` in _run_cycle used to treat that as impossible and crash
    every cycle forever afterwards. It must instead re-fetch the singleton and resume normally, same
    cycle, once it is proven reachable again."""
    await _seed_fresh_market_metrics(agent)
    healthy_singleton = agent._redis_client
    agent._redis_client = None  # exactly what _write_heartbeat does on a transient Redis blip

    await agent._run_cycle()

    assert agent._redis_client is healthy_singleton
    fake = _broker(agent)
    assert len(fake.limit_orders) == 2  # normal quoting resumed this same cycle, not just the next one


@pytest.mark.asyncio
async def test_redis_unavailable_fails_safe_without_crashing(agent: AlphaMarketMakerAgent) -> None:
    """Other half of the same regression: when Redis is genuinely gone (not just this agent's stale
    reference), _run_cycle must fail safe -- cancel resting orders, skip the cycle, never crash and
    never place a fresh order blind to a real Master Toggle/staleness channel."""
    await _seed_fresh_market_metrics(agent)
    fake = _broker(agent)
    agent._resting_order_ids = {"buy": "order-1", "sell": "order-2"}
    agent._redis_client = None
    RedisClient._instance = None  # Redis itself is gone process-wide, not just this agent's reference
    # Port 1 is never a real Redis (or anything) in this test -- fails fast and deterministically,
    # unlike pointing at settings.redis's real configured host/port, which could accidentally
    # resolve to a real dev Redis running alongside this suite.
    agent._settings.redis = RedisSettings(host="127.0.0.1", port=1)

    await agent._run_cycle()  # must not raise

    assert agent._redis_client is None
    assert fake.cancelled_order_ids == ["order-1", "order-2"]
    assert fake.limit_orders == []


@pytest.mark.asyncio
async def test_inventory_breach_hedges_instead_of_quoting(agent: AlphaMarketMakerAgent) -> None:
    """Real capital-preservation path: inventory over the cap gets a real flatten order, not a fresh quote."""
    await _seed_fresh_market_metrics(agent)
    fake = _broker(agent)
    fake.position_qty = Decimal("0.6")  # over the 0.5 cap, baseline was 0 at construction time in this test
    agent._baseline_qty = Decimal(0)
    agent._last_known_total_qty = Decimal(0)

    await agent._run_cycle()

    assert fake.limit_orders == []  # no fresh quotes while over the cap
    assert len(fake.market_orders) == 1
    symbol, side, quantity = fake.market_orders[0]
    assert side == "sell"  # long and over cap -> sell to reduce
    assert quantity == pytest.approx(0.1)  # 0.6 - 0.5


@pytest.mark.asyncio
async def test_emergency_liquidate_flattens_position_and_forces_off(agent: AlphaMarketMakerAgent) -> None:
    fake = _broker(agent)
    fake.position_qty = Decimal("0.2")
    agent._baseline_qty = Decimal(0)

    await agent._execute_emergency_liquidate()

    assert len(fake.market_orders) == 1
    symbol, side, quantity = fake.market_orders[0]
    assert side == "sell"
    assert quantity == pytest.approx(0.2)
    assert agent.control_state == AgentControlState.OFF

    persisted = await _redis(agent).execute("get", f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent.agent_id}")
    # This installed fakeredis version returns raw bytes here despite decode_responses=True -- same real,
    # separately-documented quirk as core/base_agent.py's own control-listener pubsub branch.
    if isinstance(persisted, bytes):
        persisted = persisted.decode()
    assert persisted == AgentControlState.OFF.value


@pytest.mark.asyncio
async def test_emergency_liquidate_bypasses_a_tripped_global_kill_switch(agent: AlphaMarketMakerAgent) -> None:
    """The real break-glass property: Emergency Liquidate still flattens even during a global PANIC.

    A routine hedge/quote would be refused here (see
    test_inventory_breach_hedges_instead_of_quoting's sibling scenario) --
    this is the one, deliberate, disclosed exception.
    """
    await trip_kill_switch("test: simulating an unrelated global panic")

    fake = _broker(agent)
    fake.position_qty = Decimal("0.2")
    agent._baseline_qty = Decimal(0)

    await agent._execute_emergency_liquidate()

    assert len(fake.market_orders) == 1
    symbol, side, quantity = fake.market_orders[0]
    assert side == "sell"
    assert quantity == pytest.approx(0.2)
    assert agent.control_state == AgentControlState.OFF


@pytest.mark.asyncio
async def test_routine_hedge_is_still_blocked_by_a_tripped_kill_switch(agent: AlphaMarketMakerAgent) -> None:
    """Confirms the bypass is scoped to Emergency Liquidate only -- the routine auto-hedge stays gated."""
    await trip_kill_switch("test: simulating an unrelated global panic")
    await _seed_fresh_market_metrics(agent)
    fake = _broker(agent)
    fake.position_qty = Decimal("0.6")
    agent._baseline_qty = Decimal(0)
    agent._last_known_total_qty = Decimal(0)

    await agent._run_cycle()

    assert fake.market_orders == []  # blocked -- routine hedge never bypasses the Kill Switch


@pytest.mark.asyncio
async def test_telemetry_is_published_every_cycle(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    raw = await _redis(agent).execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent.agent_id}")
    assert raw is not None
    telemetry = json.loads(raw)
    assert telemetry["status"] == "LIVE"
    assert telemetry["symbol"] == "BTC/USDT"


# =============================================================================
# Circuit breakers -- module docstring points 8/9/10
# =============================================================================


def test_order_rate_limit_allows_normal_quoting_rate(agent: AlphaMarketMakerAgent) -> None:
    """Routine two-sided quoting (2 calls/cycle) must never trip this -- see module docstring point 8."""
    agent._max_order_rate_count = 10
    agent._max_order_rate_window_seconds = 10.0
    for _ in range(6):  # 3 cycles worth, well under the threshold
        assert agent._check_order_rate_limit() is True


def test_order_rate_limit_trips_on_a_real_runaway_burst(agent: AlphaMarketMakerAgent) -> None:
    agent._max_order_rate_count = 5
    agent._max_order_rate_window_seconds = 10.0
    results = [agent._check_order_rate_limit() for _ in range(8)]
    assert results[:5] == [True, True, True, True, True]
    assert results[5:] == [False, False, False]


@pytest.mark.asyncio
async def test_order_rate_limit_breach_forces_engine_off(agent: AlphaMarketMakerAgent) -> None:
    agent._max_order_rate_count = 1
    fake = _broker(agent)
    await _seed_fresh_market_metrics(agent)

    await agent._run_cycle()  # first requote: 2 real order placements, already over the count=1 cap

    assert agent.control_state == AgentControlState.OFF


def test_fat_finger_check_allows_normal_size(agent: AlphaMarketMakerAgent) -> None:
    assert agent._check_fat_finger(Decimal("0.01")) is True


def test_fat_finger_check_blocks_oversized_order(agent: AlphaMarketMakerAgent) -> None:
    """max_single_order_units defaults to 2x max_inventory_units (0.5 here) -- see __init__'s own docstring."""
    assert agent._max_single_order_units == Decimal("1.0")
    assert agent._check_fat_finger(Decimal("1.5")) is False


@pytest.mark.asyncio
async def test_fat_finger_blocked_order_never_reaches_the_broker_and_forces_off(
    agent: AlphaMarketMakerAgent,
) -> None:
    fake = _broker(agent)
    await agent._place_resting_order("buy", Decimal("100"), Decimal("65000"))  # absurdly oversized
    assert fake.limit_orders == []
    assert agent.control_state == AgentControlState.OFF


@pytest.mark.asyncio
async def test_fat_finger_breach_trips_the_global_kill_switch(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One of the two systemic triggers of the GLOBAL Kill Switch (2026-09-20 "Protocole UNBLOCK")."""
    trips: list[str] = []

    async def fake_trip(reason: str) -> None:
        trips.append(reason)

    monkeypatch.setattr("src.agents.alpha_engine.agent.trip_kill_switch", fake_trip)
    await agent._place_resting_order("buy", Decimal("100"), Decimal("65000"))
    assert len(trips) == 1 and "fat-finger" in trips[0]


@pytest.mark.asyncio
async def test_a_brief_feed_blip_does_not_trip_the_market_data_breaker(agent: AlphaMarketMakerAgent) -> None:
    """2026-09-20 shadow run: two stale ticks in a 60s window (one ~4s blip) forced the engine OFF for
    good at the old 5% rate. 50% of the window failing is a lost feed."""
    breaker = agent._market_data_circuit_breaker
    for _ in range(28):
        breaker.record_success()
    for _ in range(2):
        breaker.record_failure()
    assert not breaker.tripped
    for _ in range(40):
        breaker.record_failure()
    assert breaker.tripped


@pytest.mark.asyncio
async def test_max_daily_loss_forces_engine_off(redis_client: RedisClient) -> None:
    agent = _make_agent(redis_client, max_daily_loss_usd=Decimal("-50"))
    agent._baseline_qty = Decimal(0)
    agent._last_known_total_qty = Decimal("0.01")
    agent._own_avg_entry_price = Decimal("70000")  # bought high

    # A fill that closes the position at a much lower price -- a real, large realized loss.
    await agent._record_fill(Decimal("-0.01"), fill_price=Decimal("60000"))

    assert agent._daily_realized_pnl_usd == Decimal("-100")  # 0.01 * (60000 - 70000)
    assert agent.control_state == AgentControlState.OFF


@pytest.mark.asyncio
async def test_record_fill_persists_real_pnl_history_entry(redis_client: RedisClient) -> None:
    """The src/dashboard/ems_pages/bilan.py equity curve's real data source: one LPUSH per real fill."""
    agent = _make_agent(redis_client)
    agent._baseline_qty = Decimal(0)
    agent._last_known_total_qty = Decimal("0.01")
    agent._own_avg_entry_price = Decimal("70000")

    await agent._record_fill(Decimal("-0.01"), fill_price=Decimal("71000"))  # +$10 realized

    key = f"{ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX}{agent.agent_id}"
    raw_entries = await _redis(agent).execute("lrange", key, 0, -1)
    assert len(raw_entries) == 1
    entry = json.loads(raw_entries[0])
    assert entry["symbol"] == agent._symbol
    assert entry["fill_qty"] == "-0.01"
    assert entry["fill_price"] == "71000"
    assert Decimal(entry["realized_pnl_delta_usd"]) == Decimal("10")
    assert Decimal(entry["session_realized_pnl_usd"]) == Decimal("10")


@pytest.mark.asyncio
async def test_daily_pnl_survives_below_threshold_engine_stays_live(redis_client: RedisClient) -> None:
    agent = _make_agent(redis_client, max_daily_loss_usd=Decimal("-500"))
    agent._baseline_qty = Decimal(0)
    agent._last_known_total_qty = Decimal("0.01")
    agent._own_avg_entry_price = Decimal("70000")

    await agent._record_fill(Decimal("-0.01"), fill_price=Decimal("60000"))  # -$100, under the -$500 ceiling

    assert agent._daily_realized_pnl_usd == Decimal("-100")
    assert agent.control_state == AgentControlState.LIVE


def test_daily_pnl_resets_on_a_new_calendar_day(agent: AlphaMarketMakerAgent) -> None:
    agent._daily_realized_pnl_usd = Decimal("-42")
    agent._daily_pnl_date = date(2020, 1, 1)  # a real past date, guaranteed different from "today"

    agent._roll_over_daily_pnl_if_new_day()

    assert agent._daily_realized_pnl_usd == Decimal(0)
    assert agent._daily_pnl_date == datetime.now(timezone.utc).date()


def test_trade_timestamps_deque_has_a_maxlen_backstop(agent: AlphaMarketMakerAgent) -> None:
    assert agent._trade_timestamps.maxlen == 100_000


def test_order_placement_timestamps_deque_has_a_maxlen_backstop(agent: AlphaMarketMakerAgent) -> None:
    assert agent._order_placement_timestamps.maxlen == 10_000


# =============================================================================
# _bootstrap_cold_start_sanitizer -- module docstring point 11
# =============================================================================


@pytest.mark.asyncio
async def test_bootstrap_cancels_real_orphaned_orders(agent: AlphaMarketMakerAgent) -> None:
    fake = _broker(agent)
    fake.orphaned_orders_to_report = [{"id": "orphan-1"}, {"id": "orphan-2"}]

    await agent._bootstrap_cold_start_sanitizer()

    assert fake.cancel_all_open_orders_calls == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_bootstrap_establishes_baseline_on_a_real_first_run(agent: AlphaMarketMakerAgent) -> None:
    fake = _broker(agent)
    fake.position_qty = Decimal("0.2")  # e.g. the operator's own initial sub-account funding

    await agent._bootstrap_cold_start_sanitizer()

    assert agent._baseline_qty == Decimal("0.2")
    persisted = await _redis(agent).execute("get", f"{ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX}{agent.agent_id}")
    if isinstance(persisted, bytes):
        persisted = persisted.decode()
    assert persisted == "0.2"


@pytest.mark.asyncio
async def test_bootstrap_restores_persisted_baseline_instead_of_recomputing(agent: AlphaMarketMakerAgent) -> None:
    """The real bug this fixes: a restart must not 'forget' this engine's own prior real inventory."""
    fake = _broker(agent)
    # A previous real session persisted a baseline of 0.1 (its own starting point back then)...
    await _redis(agent).execute("set", f"{ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX}{agent.agent_id}", "0.1")
    # ...and had legitimately accumulated up to 0.3 real BTC by the time it crashed.
    fake.position_qty = Decimal("0.3")

    await agent._bootstrap_cold_start_sanitizer()

    assert agent._baseline_qty == Decimal("0.1")  # restored, NOT recomputed as "0.3, so 0 owned"
    own_inventory = fake.position_qty - agent._baseline_qty
    assert own_inventory == Decimal("0.2")  # the real, remembered exposure this engine is responsible for


@pytest.mark.asyncio
async def test_bootstrap_refuses_to_start_when_already_over_cap(agent: AlphaMarketMakerAgent) -> None:
    fake = _broker(agent)
    await _redis(agent).execute("set", f"{ALPHA_ENGINE_BASELINE_QTY_REDIS_KEY_PREFIX}{agent.agent_id}", "0")
    fake.position_qty = Decimal("0.9")  # own inventory 0.9 > the 0.5 cap, before any cycle ever ran

    with pytest.raises(RuntimeError, match="already exceeds"):
        await agent._bootstrap_cold_start_sanitizer()


@pytest.mark.asyncio
async def test_bootstrap_tolerates_a_broker_without_cancel_all_support(agent: AlphaMarketMakerAgent) -> None:
    fake = _broker(agent)
    fake.supports_cancel_all = False

    await agent._bootstrap_cold_start_sanitizer()  # must not raise -- logs and proceeds to baseline handling

    assert agent._baseline_qty == Decimal(0)


# =============================================================================
# _last_api_latency_ms -- real, measured, never fabricated (dashboard's "Latence API")
# =============================================================================


def test_api_latency_starts_as_none_before_any_real_cycle(agent: AlphaMarketMakerAgent) -> None:
    assert agent._last_api_latency_ms is None


@pytest.mark.asyncio
async def test_read_total_inventory_measures_a_real_non_negative_latency(agent: AlphaMarketMakerAgent) -> None:
    await agent._read_total_inventory()

    assert agent._last_api_latency_ms is not None
    assert agent._last_api_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_telemetry_includes_the_real_measured_latency(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()  # a real LIVE cycle calls _read_total_inventory internally

    raw = await _redis(agent).execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent.agent_id}")
    telemetry = json.loads(raw)
    assert telemetry["api_latency_ms"] is not None
    assert telemetry["api_latency_ms"] >= 0.0


# =============================================================================
# Cross-asset session/spread gating (module docstring point 14 / Missions 2 & 3).
# BTC/USDT's real registry entry (core/asset_universe.py) has trading_hours=None -- is_unwind_period
# is unconditionally False for it, so the Friday Unwind tests below monkeypatch
# MarketSessionManager.is_unwind_period directly rather than depending on the real wall clock ever
# landing in a real Friday-21:00-UTC-through-Sunday-22:00-UTC window (see tests/test_session_manager.py
# for that logic's own real-boundary parametrized coverage -- not re-tested here).
# =============================================================================


@pytest.mark.asyncio
async def test_spread_within_ceiling_places_quotes_normally(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent, spread=1.0)  # BTC/USDT's real ceiling is $5.00
    await agent._run_cycle()
    assert _broker(agent).limit_orders != []


@pytest.mark.asyncio
async def test_spread_wider_than_ceiling_is_rejected_silently(agent: AlphaMarketMakerAgent) -> None:
    """Mission 3's own real scenario: a broker-widened spread rejects the cycle before any quote price is computed."""
    await _seed_fresh_market_metrics(agent, spread=10.0)  # over BTC/USDT's real $5.00 ceiling
    await agent._run_cycle()
    fake = _broker(agent)
    assert fake.limit_orders == []
    assert fake.market_orders == []


@pytest.mark.asyncio
async def test_spread_rejected_cycle_publishes_spread_rejected_status(agent: AlphaMarketMakerAgent) -> None:
    await _seed_fresh_market_metrics(agent, spread=10.0)
    await agent._run_cycle()
    raw = await _redis(agent).execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent.agent_id}")
    assert json.loads(raw)["status"] == "SPREAD_REJECTED"


@pytest.mark.asyncio
async def test_friday_unwind_cancels_quotes_and_blocks_new_entries(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mission 2's own explicit scenario: force liquidation, block new entries -- real, end-to-end through _run_cycle."""
    monkeypatch.setattr(
        MarketSessionManager, "is_unwind_period", staticmethod(lambda symbol, *, now=None: True),
    )
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    assert _broker(agent).limit_orders == []


@pytest.mark.asyncio
async def test_friday_unwind_publishes_weekend_unwind_status(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MarketSessionManager, "is_unwind_period", staticmethod(lambda symbol, *, now=None: True),
    )
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    raw = await _redis(agent).execute("get", f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent.agent_id}")
    assert json.loads(raw)["status"] == "WEEKEND_UNWIND"


@pytest.mark.asyncio
async def test_friday_unwind_flattens_existing_inventory_to_zero_not_just_under_the_normal_cap(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real distinction from the normal hedge: cap=Decimal(0), so even inventory well under the
    normal 0.5 cap (0.2) is fully flattened -- a scheduled full unwind, not a partial cap-breach hedge."""
    agent = _make_agent(redis_client)
    agent._broker = _FakeBroker(position_qty=Decimal("0.2"))
    monkeypatch.setattr(
        MarketSessionManager, "is_unwind_period", staticmethod(lambda symbol, *, now=None: True),
    )
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    fake = _broker(agent)
    assert fake.market_orders == [("BTC/USDT", "sell", 0.2)]


@pytest.mark.asyncio
async def test_friday_unwind_does_nothing_when_already_flat(
    agent: AlphaMarketMakerAgent, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MarketSessionManager, "is_unwind_period", staticmethod(lambda symbol, *, now=None: True),
    )
    await _seed_fresh_market_metrics(agent)
    await agent._run_cycle()
    assert _broker(agent).market_orders == []


def test_unregistered_symbol_refuses_construction() -> None:
    """A real configuration gap (module docstring point 14) -- fails at construction, not deep inside a live cycle."""
    with pytest.raises(KeyError, match="not registered"):
        AlphaMarketMakerAgent("DOGE/USDT", exchange_id="fake", manage_own_signals=False, manage_own_redis=False)


# =============================================================================
# System-level crash prevention (core.exception_manager / core.watchdog) wiring
# =============================================================================


@pytest.mark.asyncio
async def test_cancel_all_orders_for_shutdown_calls_the_real_broker_api(agent: AlphaMarketMakerAgent) -> None:
    await agent.cancel_all_orders_for_shutdown()
    fake = _broker(agent)
    assert fake.cancel_all_open_orders_calls == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_cancel_all_orders_for_shutdown_is_a_noop_without_a_broker(agent: AlphaMarketMakerAgent) -> None:
    agent._broker = None
    await agent.cancel_all_orders_for_shutdown()  # must not raise


@pytest.mark.asyncio
async def test_cancel_all_orders_for_shutdown_tolerates_a_broker_with_no_real_cancel_all(
    agent: AlphaMarketMakerAgent,
) -> None:
    agent._broker = _FakeBroker(supports_cancel_all=False)
    await agent.cancel_all_orders_for_shutdown()  # logs a warning, must not raise


@pytest.mark.asyncio
async def test_ping_exchange_calls_the_real_broker_api(agent: AlphaMarketMakerAgent) -> None:
    await agent.ping_exchange()  # _FakeBroker.get_current_price returns a plain float -- must not raise


@pytest.mark.asyncio
async def test_on_start_installs_the_exception_handler_and_starts_the_watchdog_under_mock_mode(
    redis_client: RedisClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real end-to-end through on_start() itself -- MOCK_MODE skips BrokerFactory.create() entirely,
    so this needs no real exchange/network dependency to exercise the actual wiring code path."""
    installed: list[dict[str, Any]] = []
    started: list[_FakeSystemWatchdog] = []

    def fake_install_handler(loop: Any, shutdown_callback: Any, *, sender_id: str, publisher: Any = None, notifier: Any = None, **kwargs: Any) -> None:
        installed.append({"sender_id": sender_id, "publisher": publisher, "notifier": notifier})

    monkeypatch.setattr("src.agents.alpha_engine.agent.install_global_exception_handler", fake_install_handler)
    monkeypatch.setattr("src.agents.alpha_engine.agent.SystemWatchdog", lambda **kwargs: _record_watchdog(started, **kwargs))

    instance = AlphaMarketMakerAgent("BTC/USDT", exchange_id="fake", manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    instance._publisher = _RecordingZmqPublisher()  # type: ignore[assignment]
    instance._settings.mock_mode = True
    # Real fakeredis pubsub.listen() does not reliably honor task.cancel() the way a real Redis
    # server's connection does -- irrelevant to what this test is actually verifying (the exception
    # handler/watchdog wiring), so the emergency listener is replaced with an instantly-completing
    # no-op rather than fighting that unrelated, pre-existing fakeredis limitation in cleanup below.
    instance._listen_for_emergency_liquidate = _noop_coroutine  # type: ignore[method-assign]

    try:
        await instance.on_start()

        assert len(installed) == 1
        assert installed[0]["sender_id"] == instance.agent_id
        assert installed[0]["publisher"] is instance._publisher
        assert installed[0]["notifier"] is instance._notifier

        assert len(started) == 1
        assert started[0].started is True
        assert started[0].kwargs["redis_client"] is redis_client
        assert started[0].kwargs["exchange_ping"] is None  # no real broker under MOCK_MODE
    finally:
        # The real, full BaseAgent teardown sequence (cancels the emergency listener, calls
        # on_shutdown() -- which stops self._watchdog -- closes the real ZMQ publisher/subscriber
        # sockets this constructor opened, and terminates this instance's own owned ZMQ context).
        # Skipping this (an earlier draft only cancelled the emergency listener task by hand) left a
        # real, unclosed ZMQ context behind -- harmless for this one test's own assertions, but it
        # left pytest's process unable to exit cleanly afterward once nothing else joined it.
        await instance._shutdown()


async def _noop_coroutine() -> None:
    pass


class _FakeSystemWatchdog:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


def _record_watchdog(sink: list[_FakeSystemWatchdog], **kwargs: Any) -> _FakeSystemWatchdog:
    watchdog = _FakeSystemWatchdog(**kwargs)
    sink.append(watchdog)
    return watchdog

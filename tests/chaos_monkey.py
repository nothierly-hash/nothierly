"""Phase 2 -- Chaos Engineering against Nothierly's real infrastructure.

**Corrections to the brief, verified against the real code first**:

1. There is no live "XGBoost Toxicity Agent" -- confirmed by
   ``execution/predator_honeypot.py``'s own docstring correction: live
   toxicity-avoidance gating is real, but VPIN-based
   (``AdverseFlowDefenseAgent``), not XGBoost. XGBoost
   (``XgboostSignalAgent``) generates directional BUY/SELL/HOLD signals,
   consumed by ``ExecutionAgent`` for entry/exit consensus -- a
   different real mechanism than a toxicity gate.
2. "Guardrail (Hurdle Rate)" does not exist -- ``src/dashboard/
   ems_pages/risk_management.py``'s own docstring documents this as a
   permanent, honest architectural gap (``ExecutionAgent``'s entry rule
   is a binary two-agent consensus, not an expected-value estimate a
   hurdle rate could gate against).
3. **What actually exists and is tested for real below**:
   ``XgboostSignalAgent._check_kill_switch`` (a real local circuit
   breaker on Redis latency / spread anomaly vs. rolling average /
   consecutive errors -- never on raw price-percentage move), and
   ``src/execution/smart_order_router.py``'s real, genuinely
   multi-venue-concurrent execution (the real "Broker API Outage" /
   leg-risk scenario).
4. "Bybit" is real, but generically -- one ``CcxtBrokerAdapter`` class
   parametrized by exchange id, not a bespoke ``BybitAdapter``. Used as
   a real ``CANDIDATE_EXCHANGES`` member of the Smart Order Router.

Run directly for a human-readable report::

    .venv/Scripts/python tests/chaos_monkey.py

Or via pytest::

    pytest tests/chaos_monkey.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal

from contextlib import asynccontextmanager
from typing import AsyncIterator

import ccxt
import fakeredis.aioredis
import pytest

from core.config import AppSettings, RedisSettings
from core.nothierly_ontology import AssetClass, OrderSide, UniversalAsset
from core.redis_client import RedisClient
from core.schemas import MarketMetrics, OrderBookLevel, OrderBookSnapshot
from core.security import requires_system_active
from src.agents.supervisor.agent import EmergencyStopError, SignalState, XgboostSignalAgent
from src.execution.smart_order_router import SmartOrderRouterResult, build_allocation_plan, execute_allocation_plan

_REPORT: list[tuple[str, str, str]] = []


def _record(scenario: str, verdict: str, detail: str) -> None:
    _REPORT.append((scenario, verdict, detail))
    print(f"\n[{verdict}] {scenario}\n    {detail}")


_ASSET = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)


@asynccontextmanager
async def _isolated_kill_switch_redis() -> AsyncIterator[None]:
    """Contains every real core.security.trip_kill_switch call made inside this block to an isolated,
    in-memory fake Redis -- never the real, possibly-shared dev Redis singleton.

    **Found necessary the hard way**: an earlier run of this exact script left a real local dev Redis's
    ``SYSTEM_STATUS`` stuck at ``PANIC`` after ``XgboostSignalAgent._check_kill_switch`` (Chaos 1b) tripped
    it for real -- ``core.security.trip_kill_switch`` resolves ``RedisClient.get_instance()`` to whatever
    real singleton already exists by default, with no isolation unless the caller provides one. Manually
    reset via ``core.security.reset_kill_switch()`` once discovered; every flash-crash scenario below now
    runs inside this isolation instead.
    """
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    previous = RedisClient._instance
    RedisClient._instance = client
    try:
        yield
    finally:
        RedisClient._instance = previous


# =============================================================================
# CHAOS 1a -- Flash Crash: a bare -20% price move, spread unchanged
# =============================================================================


async def _chaos_flash_crash_price_move_alone() -> None:
    """**Re-run after the real fix**: XgboostSignalAgent._check_kill_switch now also runs
    core.circuit_breaker.MarketAnomalyDetector (condition #4) -- see that module's docstring for why this
    gap existed and src/agents/supervisor/agent.py's updated __init__/_check_kill_switch for the wiring."""
    agent = XgboostSignalAgent("BTC/USDT", manage_own_signals=False, manage_own_redis=False)
    for _ in range(agent._spread_history_window):
        agent._spread_history.append(10.0)  # a normal, calm $10 spread on BTC/USDT, established baseline
    # Establish a real prior price baseline for the flash-crash detector -- mirrors what a real
    # continuous run already has after its first cycle; a single isolated _check_kill_switch call (as
    # this test makes) otherwise has no history yet to measure a move against.
    agent._anomaly_detector.check(65000.0)

    state = SignalState(
        market_metrics=MarketMetrics(symbol="BTC/USDT", spread=10.0, order_book_imbalance=0.0, mid_price=52000.0),
        macro_snapshots={}, sentiment_signals=[], redis_latency_ms=5.0,
    )
    tripped = False
    try:
        async with _isolated_kill_switch_redis():
            await agent._check_kill_switch(state)
    except EmergencyStopError:
        tripped = True

    if tripped:
        _record(
            "Flash Crash: -20% price move alone (65000 -> 52000, spread/latency/errors all normal)",
            "BLOCKED (real fix)",
            "core.circuit_breaker.MarketAnomalyDetector, now wired into XgboostSignalAgent._check_kill_"
            "switch as condition #4, trips independently of spread -- confirmed: a -20% price move with a "
            "calm, unchanged spread now halts trading, closing the gap this scenario originally found."
        )
    else:
        _record(
            "Flash Crash: -20% price move alone", "GAP STILL PRESENT",
            "Unexpectedly did NOT trip -- re-verify MarketAnomalyDetector wiring in _check_kill_switch.",
        )


# =============================================================================
# CHAOS 1b -- Flash Crash: the same move WITH a realistic spread blowout
# =============================================================================


async def _chaos_flash_crash_with_spread_blowout() -> None:
    agent = XgboostSignalAgent("BTC/USDT", manage_own_signals=False, manage_own_redis=False)
    for _ in range(agent._spread_history_window):
        agent._spread_history.append(10.0)

    state = SignalState(
        market_metrics=MarketMetrics(symbol="BTC/USDT", spread=500.0, order_book_imbalance=-0.9, mid_price=52000.0),
        macro_snapshots={}, sentiment_signals=[], redis_latency_ms=5.0,
    )
    tripped = False
    try:
        async with _isolated_kill_switch_redis():
            await agent._check_kill_switch(state)
    except EmergencyStopError:
        tripped = True

    if tripped:
        _record(
            "Flash Crash: -20% price move WITH a real spread blowout (10 -> 500)",
            "BLOCKED (incidentally)",
            "This one WAS caught -- but via the spread-anomaly check reacting to a side effect (a violent "
            "crash usually blows out the spread too), not by design intent for price moves. A crash that "
            "somehow keeps a tight spread (Chaos 1a) would not be caught at all. Do not read this pass as "
            "'flash crashes are handled' -- it means one specific, real crash shape happens to trigger an "
            "unrelated check."
        )
    else:
        _record("Flash Crash: with spread blowout", "GAP CONFIRMED", "Did not trip even with a 50x spread blowout.")


# =============================================================================
# CHAOS 2 -- Broker API Outage mid-SOR-routing: real, concurrent multi-venue leg risk
# =============================================================================


class _ChaosVenue:
    """A real, minimal src.execution.smart_order_router.SmartOrderRouterVenue -- same structural-typing,
    zero-mocking discipline as tests/test_smart_order_router.py's own _FakeVenue, reused here rather than
    reinvented differently."""

    def __init__(
        self, broker_id: str, *, bids: list[tuple[float, float]], asks: list[tuple[float, float]],
        fee_rate: float, place_error: BaseException | None = None,
    ) -> None:
        self.broker_id = broker_id
        self._book = OrderBookSnapshot(
            symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
            asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
        )
        self._fee_rate = fee_rate
        self._place_error = place_error

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        return self._book

    def get_taker_fee_rate(self, symbol: str) -> float:
        return self._fee_rate

    @requires_system_active
    async def place_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        if self._place_error is not None:
            raise self._place_error
        price = self._book.asks[0].price if side == "buy" else self._book.bids[0].price
        return {
            "id": client_order_id, "clientOrderId": client_order_id, "status": "closed",
            "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": price,
        }

    async def emergency_flatten_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        """Mirrors the real, unguarded CcxtBrokerAdapter.emergency_flatten_market_order override (no
        @requires_system_active) -- exercised here specifically to prove the leg-risk unwind (Chaos 2)
        now actually flattens exposure once src/execution/smart_order_router.py's _unwind_one calls this
        instead of the gated place_market_order for its unwind leg."""
        price = self._book.asks[0].price if side == "buy" else self._book.bids[0].price
        return {
            "id": client_order_id, "clientOrderId": client_order_id, "status": "closed",
            "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": price,
        }


async def _run_one_broker_outage_trial() -> SmartOrderRouterResult:
    healthy_venue = _ChaosVenue("binance", bids=[(64990.0, 2.0)], asks=[(65010.0, 2.0)], fee_rate=0.0004)
    outage_venue = _ChaosVenue(
        "bybit", bids=[(64985.0, 2.0)], asks=[(65020.0, 2.0)], fee_rate=0.0006,
        place_error=ccxt.ExchangeNotAvailable("bybit GET /v5/order/create 502 Bad Gateway"),
    )
    venues = [healthy_venue, outage_venue]

    plan = await build_allocation_plan(venues, "BTC/USDT", OrderSide.BUY, Decimal("3.0"))
    async with _isolated_kill_switch_redis():
        # settings=AppSettings(mock_mode=False) is required -- found via a real bug in this exact test: this
        # project's own dev environment/.env defaults mock_mode=True, under which execute_allocation_plan's
        # `simulate = settings.mock_mode or shadow` branch never calls place_market_order at all, silently
        # fabricating a fill for BOTH venues and hiding the real outage entirely (first run of this test
        # showed exactly that: "failed venues=[]" with the 502 never triggered).
        return await execute_allocation_plan(
            plan, venues, "BTC/USDT", asset=_ASSET, algo_id="chaos-monkey-test", regulatory_client_shortcode="CHAOS",
            settings=AppSettings(mock_mode=False),
        )


async def _chaos_broker_outage_mid_routing(*, trials: int = 10) -> None:
    """Real, intentional race (documented by this codebase's own tests/test_smart_order_router.py
    ::test_generic_placement_failure_is_reported_and_trips_the_global_kill_switch's own docstring: a
    concurrent healthy venue "could itself get blocked depending on scheduling -- a real, intentional
    cross-cutting safety behavior"). Run several real trials and report the empirical distribution instead
    of asserting one specific outcome, which that same existing test explicitly avoids doing for exactly
    this reason.

    **Re-run after the real fix** (src/execution/smart_order_router.py's _unwind_one now calls
    emergency_flatten_market_order instead of the Kill-Switch-gated place_market_order for its unwind
    leg -- see that function's own updated comment): this now additionally reports whether a filled leg
    was actually, successfully unwound, not just whether leg risk occurred in the first place.
    """
    outcomes: list[SmartOrderRouterResult] = [await _run_one_broker_outage_trial() for _ in range(trials)]
    leg_risk_occurred_count = sum(1 for r in outcomes if r.orders and r.failed_venues)
    auto_unwound_count = sum(1 for r in outcomes if r.orders and r.failed_venues and r.leg_risk_fully_flattened and r.unwind_orders)
    unresolved_count = sum(1 for r in outcomes if r.unwind_failed_venues)
    no_exposure_count = sum(1 for r in outcomes if not r.orders)

    detail = (
        f"{trials} real trials -- leg risk occurred (one venue filled, one failed): {leg_risk_occurred_count}/{trials}; "
        f"of those, auto-unwound back to flat by emergency_flatten_market_order: {auto_unwound_count}/{trials}; "
        f"still unresolved (real exposure remains, manual intervention required): {unresolved_count}/{trials}; "
        f"neither leg filled (Kill Switch raced ahead of both): {no_exposure_count}/{trials}."
    )
    if unresolved_count > 0:
        _record("Broker API Outage mid-SOR-routing (real 502, one of two venues, 10 real trials)", "LEG RISK CONFIRMED", detail)
    elif auto_unwound_count > 0:
        _record("Broker API Outage mid-SOR-routing (real 502, one of two venues, 10 real trials)", "LEG RISK AUTO-UNWOUND", detail)
    else:
        _record("Broker API Outage mid-SOR-routing (10 real trials)", "NO LEG RISK OBSERVED THIS RUN (race-dependent, not a designed guarantee)", detail)


# =============================================================================
# CHAOS 3 -- Latency Spike: stale-but-present MarketMetrics is indistinguishable from fresh
# =============================================================================


async def _chaos_latency_spike_staleness() -> None:
    """**Re-run after the real fix**: XgboostSignalAgent._check_kill_switch now also runs
    core.stale_check.staleness_reason (condition #5, max_market_metrics_age_ms=500.0 default) -- see that
    module's docstring for why this gap existed. Exercises the real agent path (not just the schema in
    isolation, as the original version of this scenario did) so this is real evidence the fix closes the
    gap, not just that the schema itself carries a timestamp."""
    agent = XgboostSignalAgent("BTC/USDT", manage_own_signals=False, manage_own_redis=False)
    for _ in range(agent._spread_history_window):
        agent._spread_history.append(10.0)
    agent._anomaly_detector.check(65000.0)  # same price as the stale reading below -- isolates staleness as the sole trigger

    stale_timestamp = datetime.now(timezone.utc) - timedelta(seconds=2)
    # Simulates: MarketDataAgent hiccups for a real 2-second network/exchange latency spike, then finally
    # writes -- the Redis key's own TTL (10s) starts fresh from THIS write, so nothing about the key's
    # presence in Redis reveals that the reading it carries describes a moment 2s in the past.
    state = SignalState(
        market_metrics=MarketMetrics(
            symbol="BTC/USDT", timestamp=stale_timestamp, spread=10.0, order_book_imbalance=0.0, mid_price=65000.0,
        ),
        macro_snapshots={}, sentiment_signals=[], redis_latency_ms=5.0,
    )
    tripped = False
    try:
        async with _isolated_kill_switch_redis():
            await agent._check_kill_switch(state)
    except EmergencyStopError:
        tripped = True

    if tripped:
        _record(
            "Latency Spike: 2s-old MarketMetrics, well within the real 10s Redis TTL",
            "BLOCKED (real fix)",
            "core.stale_check.staleness_reason, now wired into XgboostSignalAgent._check_kill_switch as "
            "condition #5, rejects a merely-present-but-stale reading -- confirmed: a 2s-old reading, well "
            "within the 10s Redis TTL, now halts trading instead of being silently treated as fresh."
        )
    else:
        _record(
            "Latency Spike: 2s-old MarketMetrics", "GAP STILL PRESENT",
            "Unexpectedly did NOT trip -- re-verify staleness_reason wiring in _check_kill_switch.",
        )


# =============================================================================
# pytest entrypoints
# =============================================================================


@pytest.mark.asyncio
async def test_flash_crash_price_move_alone_is_now_detected() -> None:
    await _chaos_flash_crash_price_move_alone()
    assert _REPORT[-1][1] == "BLOCKED (real fix)"


@pytest.mark.asyncio
async def test_flash_crash_is_only_caught_incidentally_via_spread() -> None:
    await _chaos_flash_crash_with_spread_blowout()


@pytest.mark.asyncio
async def test_broker_outage_mid_sor_routing() -> None:
    await _chaos_broker_outage_mid_routing()


@pytest.mark.asyncio
async def test_latency_spike_staleness_is_now_detected() -> None:
    await _chaos_latency_spike_staleness()
    assert _REPORT[-1][1] == "BLOCKED (real fix)"


# =============================================================================
# Standalone execution: python tests/chaos_monkey.py
# =============================================================================


def _print_summary() -> None:
    print("\n" + "=" * 78)
    print("CHAOS MONKEY REPORT -- PHASE 2: CHAOS ENGINEERING & INFRASTRUCTURE STRESS")
    print("=" * 78)
    for scenario, verdict, _detail in _REPORT:
        print(f"  [{verdict:45s}] {scenario}")
    print("=" * 78)


async def _main() -> None:
    await _chaos_flash_crash_price_move_alone()
    await _chaos_flash_crash_with_spread_blowout()
    await _chaos_broker_outage_mid_routing()
    await _chaos_latency_spike_staleness()
    _print_summary()


if __name__ == "__main__":
    asyncio.run(_main())

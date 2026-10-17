"""Réconciliation d'état au démarrage : le moteur ADOPTE les positions réelles de l'exchange (au lieu de repartir à plat), le pont retient tout d'ici là."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from typing import Any

import pytest

from core.heartbeat import STATE_ACTIVE, STATE_RECONCILING, STATE_STARTING
from core.redis_client import RedisUnavailableError
from src.agents.execution import reconciliation as rec
from src.agents.execution.reconciliation import ExchangeReport, ReconcilerConfig, ReportedPosition
from src.bridge.platform_bridge import HOLD_STATES, BridgeConfig, EnginePosition, EngineView, PlatformBridge, _parse_positions, desired_target
from tests.test_metalabel_pipeline import agent_factory  # noqa: F401 -- fixture partagée
from workers.reconciliation import reporter as rp

D = Decimal
NOW_MS = 1_760_000_000_000
INSTRUMENT = "BTC/USDT"


def _report(venue: str = "HYPERLIQUID", *, ts: int = NOW_MS, equity: str = "10000", positions: list[Any] | None = None) -> dict[str, Any]:
    return {"venue": venue, "ts_ms": ts, "equity": equity, "positions": positions if positions is not None else [{"asset": "BTC", "size": "0.1", "price": "50000"}]}


def _parsed(venue: str = "HYPERLIQUID", **kw: Any) -> ExchangeReport:
    report = rec.parse_report(json.dumps(_report(venue, **kw)))
    assert report is not None
    return report


# --- Analyse des rapports ----------------------------------------------------------------------


def test_a_valid_report_is_parsed_with_exact_decimals() -> None:
    report = _parsed(positions=[{"asset": "BTC", "size": "-0.125", "price": "50000.5"}, {"asset": "ETH", "size": "0", "price": "3000"}])
    assert (report.venue, report.ts_ms, report.equity) == ("HYPERLIQUID", NOW_MS, D("10000"))
    assert report.positions == (ReportedPosition("BTC", D("-0.125"), D("50000.5")),)  # la taille nulle est omise


@pytest.mark.parametrize("raw", [
    None, "", b"", "pas du json", "[]", "null",
    json.dumps({**_report(), "venue": ""}), json.dumps({**_report(), "ts_ms": 0}), json.dumps({**_report(), "ts_ms": True}),
    json.dumps({**_report(), "equity": "0"}), json.dumps({**_report(), "equity": "-5"}), json.dumps({**_report(), "equity": "NaN"}),
    json.dumps({**_report(), "positions": "x"}), json.dumps(_report(positions=[{"asset": "BTC", "size": "1", "price": "0"}])),
    json.dumps(_report(positions=[{"asset": "BTC", "size": "abc", "price": "1"}])), json.dumps(_report(positions=[{"asset": "", "size": "1", "price": "1"}])),
    json.dumps(_report(positions=["BTC"])),
])
def test_a_doubtful_report_is_refused_whole(raw: Any) -> None:
    assert rec.parse_report(raw) is None  # jamais une adoption partielle : une seule ligne douteuse invalide tout le rapport


def test_one_bad_position_invalidates_the_whole_report() -> None:
    raw = json.dumps(_report(positions=[{"asset": "BTC", "size": "1", "price": "50000"}, {"asset": "ETH", "size": "1", "price": "-1"}]))
    assert rec.parse_report(raw) is None


def test_freshness_bounds_including_clock_skew() -> None:
    report = _parsed()
    assert rec.is_fresh(report, NOW_MS + 119_000, 120.0) and not rec.is_fresh(report, NOW_MS + 121_000, 120.0)  # périmé
    assert rec.is_fresh(report, NOW_MS - 59_000, 120.0) and not rec.is_fresh(report, NOW_MS - 61_000, 120.0)  # trop dans le futur


def test_the_reference_is_the_first_fresh_venue_in_priority_order() -> None:
    hl, by = _parsed("HYPERLIQUID"), _parsed("BYBIT")
    reports = {"HYPERLIQUID": hl, "BYBIT": by}
    assert rec.reference_report(reports, ["hyperliquid", "bybit"], NOW_MS, 120.0) is hl
    stale_hl = _parsed("HYPERLIQUID", ts=NOW_MS - 600_000)
    assert rec.reference_report({"HYPERLIQUID": stale_hl, "BYBIT": by}, ["HYPERLIQUID", "BYBIT"], NOW_MS, 120.0) is by  # le périmé est sauté
    assert rec.reference_report({"HYPERLIQUID": stale_hl}, ["HYPERLIQUID", "BYBIT"], NOW_MS, 120.0) is None


@pytest.mark.parametrize("asset,expected", [("BTC", INSTRUMENT), ("BTCUSDT", INSTRUMENT), ("btcusdt", INSTRUMENT), ("DOGE", None), ("BTCB", None), ("", None)])
def test_native_assets_map_to_engine_instruments(asset: str, expected: str | None) -> None:
    assert rec.instrument_for(asset, [INSTRUMENT, "ETH/USDT"]) == expected


# --- Conversion --------------------------------------------------------------------------------


def test_the_exposure_is_preserved_whatever_the_two_equities() -> None:
    """Compte d'exchange 10 000 $, 0,1 BTC à 50 000 $ = 50 % ; moteur à 20 000 $ -> 0,2 BTC (50 % de 20 000 $ à 50 000 $)."""
    plan = rec.plan_adoption(_parsed(), [INSTRUMENT], 20_000.0)
    (adoption,) = plan.adoptions
    assert (adoption.instrument, adoption.direction, adoption.price) == (INSTRUMENT, 1, 50_000.0)
    assert adoption.units == pytest.approx(0.2) and adoption.exposure_pct == pytest.approx(50.0)


def test_a_short_is_adopted_as_a_short() -> None:
    plan = rec.plan_adoption(_parsed(positions=[{"asset": "BTCUSDT", "size": "-0.05", "price": "50000"}]), [INSTRUMENT], 10_000.0)
    (adoption,) = plan.adoptions
    assert adoption.direction == -1 and adoption.units == pytest.approx(0.05) and adoption.exposure_pct == pytest.approx(-25.0)


def test_untracked_assets_and_dust_are_left_alone_and_reported() -> None:
    plan = rec.plan_adoption(
        _parsed(positions=[{"asset": "DOGE", "size": "1000", "price": "0.1"}, {"asset": "BTC", "size": "0.0000001", "price": "50000"}]), [INSTRUMENT], 10_000.0,
    )
    assert plan.adoptions == () and len(plan.ignored) == 2 and "non suivi" in plan.ignored[0] and "poussière" in plan.ignored[1]


def test_an_unknown_engine_equity_refuses_to_guess() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="équité"):
            rec.plan_adoption(_parsed(), [INSTRUMENT], bad)


def test_a_notable_gap_between_venues_is_reported_never_averaged() -> None:
    hl = _parsed("HYPERLIQUID")  # +50 %
    by = _parsed("BYBIT", equity="5000", positions=[{"asset": "BTCUSDT", "size": "0.01", "price": "50000"}])  # +10 %
    plan = rec.plan_adoption(hl, [INSTRUMENT], 10_000.0, others=[by])
    assert len(plan.adoptions) == 1 and plan.adoptions[0].exposure_pct == pytest.approx(50.0)  # on adopte UN compte réel
    assert any("BYBIT" in d and "HYPERLIQUID" in d for d in plan.divergences)
    flat = _parsed("BYBIT", positions=[])
    assert any("à plat" in d for d in rec.plan_adoption(hl, [INSTRUMENT], 10_000.0, others=[flat]).divergences)
    close = _parsed("BYBIT", equity="10000", positions=[{"asset": "BTCUSDT", "size": "0.1", "price": "50000"}])
    assert rec.plan_adoption(hl, [INSTRUMENT], 10_000.0, others=[close]).divergences == ()


# --- Configuration -----------------------------------------------------------------------------


def test_the_reconciliation_is_off_unless_venues_are_configured() -> None:
    assert ReconcilerConfig.from_env({}) is None and ReconcilerConfig.from_env({rec.ENV_VENUES: "  "}) is None
    cfg = ReconcilerConfig.from_env({rec.ENV_VENUES: "hyperliquid, bybit", rec.ENV_WAIT_S: "15"})
    assert cfg is not None and cfg.venues == ("HYPERLIQUID", "BYBIT") and cfg.wait_s == 15.0
    for bad in ({"venues": ()}, {"venues": ("A",), "poll_s": 0}, {"venues": ("A",), "max_age_s": 0}, {"venues": ("A",), "wait_s": -1}):
        with pytest.raises(ValueError):
            ReconcilerConfig(**bad)


def test_the_key_prefix_is_shared_by_the_reporter_and_the_engine() -> None:
    assert rp.KEY_PREFIX == rec.REPORT_KEY_PREFIX and rp.redis_key("bybit") == rec.report_key("BYBIT")


def test_read_reports_skips_absent_and_garbled_venues() -> None:
    store = {rec.report_key("HYPERLIQUID"): json.dumps(_report()), rec.report_key("BYBIT"): "{pas du json"}

    async def execute(command: str, key: str) -> Any:
        assert command == "get"
        return store.get(key)

    reports = asyncio.run(rec.read_reports(execute, ["HYPERLIQUID", "BYBIT", "BINANCE"]))
    assert list(reports) == ["HYPERLIQUID"]


# --- Moteur : adoption -------------------------------------------------------------------------


async def _put(agent: Any, report: dict[str, Any]) -> None:
    assert agent._redis_client is not None
    await agent._redis_client.execute("set", rec.report_key(report["venue"]), json.dumps(report))


def _cfg(*venues: str, wait_s: float = 0.3) -> ReconcilerConfig:
    return ReconcilerConfig(tuple(venues), wait_s=wait_s, poll_s=0.01, max_age_s=120.0)


def _now_report(venue: str = "HYPERLIQUID", **kw: Any) -> dict[str, Any]:
    return _report(venue, ts=int(time.time() * 1000), **kw)


@pytest.mark.asyncio
async def test_the_engine_adopts_the_real_position_and_the_bridge_asks_for_exactly_the_same_exposure(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)  # capital du moteur : 10 000 $
    await _put(agent, _now_report(positions=[{"asset": "BTC", "size": "0.03", "price": "50000"}]))  # 0,03 BTC sur un compte de 10 000 $ = 15 %
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID")) is True
    position = agent._positions[INSTRUMENT]
    assert position.status == "LONG" and position.entry_price == 50_000.0 and position.units_held == pytest.approx(0.03)
    assert position.highest_seen_price == 50_000.0 and position.entered_as_simulated and position.trailing_activated is False
    assert agent._current_exposure == pytest.approx(1500.0)
    published = json.loads(await agent._redis_client.execute("get", "NOTHIERLY_ACTIVE_POSITIONS"))
    assert [(p["instrument"], p["status"]) for p in published] == [(INSTRUMENT, "LONG")]
    # La propriété qui compte : ce que le pont publiera à partir de l'état adopté est l'exposition qu'a déjà le compte -> aucun ordre à passer.
    (engine_position,) = _parse_positions(json.dumps(published))
    target = desired_target(engine_position, D(str(agent._mock_balance)), BridgeConfig(size_bucket_pct=None))
    assert target is not None
    direction, pct = target
    assert direction.value == "LONG" and pct == D("15.00")


@pytest.mark.asyncio
async def test_a_position_above_the_bridge_cap_is_adopted_as_is_and_the_bridge_trims_it_to_the_cap(agent_factory: Any) -> None:  # noqa: F811
    """Adopter ne contourne aucun plafond : le moteur garde la taille réelle, le pont (plafond de 20 %) ramène l'exposition à 20 % au lieu de tout fermer."""
    agent = agent_factory(None)
    await _put(agent, _now_report(positions=[{"asset": "BTC", "size": "0.05", "price": "50000"}]))  # 25 %
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID")) is True
    assert agent._positions[INSTRUMENT].units_held == pytest.approx(0.05)
    published = json.loads(await agent._redis_client.execute("get", "NOTHIERLY_ACTIVE_POSITIONS"))
    (engine_position,) = _parse_positions(json.dumps(published))
    target = desired_target(engine_position, D(str(agent._mock_balance)), BridgeConfig(size_bucket_pct=None))
    assert target is not None and target[1] == D("20.00")  # réduit au plafond, pas fermé


@pytest.mark.asyncio
async def test_different_account_sizes_still_give_the_same_exposure(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)  # moteur à 10 000 $ virtuels, compte réel de 2 000 $ avec 0,01 BTC = 25 %
    await _put(agent, _now_report(equity="2000", positions=[{"asset": "BTC", "size": "-0.01", "price": "50000"}]))
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID")) is True
    position = agent._positions[INSTRUMENT]
    assert position.status == "SHORT" and position.units_held == pytest.approx(-0.05) and position.lowest_seen_price == 50_000.0


@pytest.mark.asyncio
async def test_a_flat_account_is_a_valid_answer(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)
    await _put(agent, _now_report(positions=[]))
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID")) is True
    assert all(p.status == "FLAT" for p in agent._positions.values())


@pytest.mark.asyncio
async def test_without_a_fresh_report_the_engine_does_not_pretend_to_be_flat(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID", wait_s=0.05)) is False  # aucun rapport
    await _put(agent, _report("HYPERLIQUID", ts=int(time.time() * 1000) - 600_000))  # un rapport vieux de 10 minutes
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID", wait_s=0.05)) is False
    assert agent._positions == {}  # rien d'adopté, rien d'inventé


@pytest.mark.asyncio
async def test_redis_down_is_a_blocked_reconciliation_not_a_crash(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)

    async def down(*args: Any, **kwargs: Any) -> Any:
        raise RedisUnavailableError("redis injoignable")

    agent._redis_client.execute = down
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID", wait_s=0.05)) is False


@pytest.mark.asyncio
async def test_it_retries_until_the_reporter_comes_back_then_adopts(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)

    async def reporter_comes_back() -> None:
        await asyncio.sleep(0.25)
        await _put(agent, _now_report(positions=[{"asset": "BTC", "size": "0.02", "price": "50000"}]))

    task = asyncio.create_task(reporter_comes_back())
    await asyncio.wait_for(agent._reconcile_until_done(_cfg("HYPERLIQUID", wait_s=0.05)), 5.0)  # plusieurs tentatives vides, puis l'adoption
    await task
    assert agent._positions[INSTRUMENT].status == "LONG"


@pytest.mark.asyncio
async def test_the_priority_venue_wins_and_a_missing_one_falls_back_after_the_wait(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)
    await _put(agent, _now_report("BYBIT", equity="10000", positions=[{"asset": "BTCUSDT", "size": "-0.02", "price": "50000"}]))
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID", "BYBIT", wait_s=0.05)) is True  # HYPERLIQUID n'a jamais répondu
    assert agent._positions[INSTRUMENT].status == "SHORT"
    other = agent_factory(None)
    await _put(other, _now_report("HYPERLIQUID", positions=[{"asset": "BTC", "size": "0.02", "price": "50000"}]))
    await _put(other, _now_report("BYBIT", positions=[{"asset": "BTCUSDT", "size": "-0.02", "price": "50000"}]))
    assert await other._reconcile_positions(_cfg("HYPERLIQUID", "BYBIT")) is True
    assert other._positions[INSTRUMENT].status == "LONG"  # la priorité décide, pas l'ordre d'arrivée


@pytest.mark.asyncio
async def test_an_engine_that_already_holds_a_position_is_left_alone(agent_factory: Any) -> None:  # noqa: F811
    agent = agent_factory(None)
    position = agent._position_for(INSTRUMENT)
    position.status, position.entry_price, position.units_held = "LONG", 49_000.0, 0.5
    await _put(agent, _now_report(positions=[{"asset": "BTC", "size": "-0.05", "price": "50000"}]))
    assert await agent._reconcile_positions(_cfg("HYPERLIQUID")) is True
    assert position.units_held == 0.5 and position.entry_price == 49_000.0


@pytest.mark.asyncio
async def test_the_engine_passes_through_reconciling_before_active_and_only_after_adopting(agent_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    agent = agent_factory(None)
    states: list[tuple[str, int]] = []

    class SpyHeartbeat:
        def start(self) -> None: ...

        def stop(self) -> None: ...

        def set_state(self, state: str) -> None:
            states.append((state, sum(1 for p in agent._positions.values() if p.status != "FLAT")))

    async def nothing() -> None:
        raise asyncio.CancelledError

    async def connect() -> None: ...

    async def on_start() -> None: ...

    async def no_shutdown() -> None: ...

    agent._engine_heartbeat = SpyHeartbeat()
    agent._reconciler = _cfg("HYPERLIQUID")
    monkeypatch.setattr(agent, "_connect_redis", connect)
    monkeypatch.setattr(agent, "on_start", on_start)
    monkeypatch.setattr(agent, "_listen_for_bus_messages", nothing)
    monkeypatch.setattr(agent, "_listen_for_manual_trade_requests", nothing)
    monkeypatch.setattr(agent, "_listen_for_control_commands", nothing)
    monkeypatch.setattr(agent, "_shutdown", no_shutdown)
    await _put(agent, _now_report(positions=[{"asset": "BTC", "size": "0.05", "price": "50000"}]))
    await agent.run()
    assert states == [(STATE_RECONCILING, 0), (STATE_ACTIVE, 1)]  # ACTIVE n'est annoncé qu'une fois la position adoptée


# --- Pont : rien n'est publié pendant la réconciliation ----------------------------------------


class _Reader:
    def __init__(self, view: EngineView) -> None:
        self.view = view

    async def read(self) -> EngineView:
        return self.view


class _Sink:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def publish(self, signal: Any) -> bool:
        self.signals.append(signal)
        return True


class _Clock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


def _bridge(view: EngineView, **cfg: Any) -> tuple[PlatformBridge, _Reader, _Sink, _Clock]:
    reader, sink, clock = _Reader(view), _Sink(), _Clock()
    cfg.setdefault("size_bucket_pct", None)
    cfg.setdefault("min_hold_s", 0.0)
    return PlatformBridge(reader, sink, BridgeConfig(**cfg), clock=clock), reader, sink, clock


def test_reconciling_is_a_hold_state_like_starting() -> None:
    assert HOLD_STATES == {STATE_STARTING, STATE_RECONCILING}


def test_while_the_engine_reconciles_the_bridge_publishes_nothing_so_no_flat_liquidates_the_real_position() -> None:
    """Le scénario du redémarrage : moteur vivant, positions vides en mémoire. Sans RECONCILING, le pont publierait FLAT et fermerait la position réelle."""
    view = EngineView((), D("10000"), True, True, STATE_RECONCILING)
    bridge, reader, sink, clock = _bridge(view, instruments=("BTC/USDT",))
    for _ in range(5):
        assert asyncio.run(bridge.step()) == []
        clock.now += 10
    assert sink.signals == []
    reader.view = EngineView((EnginePosition("BTC/USDT", "LONG", D("0.03"), D("50000")),), D("10000"), True, True, STATE_ACTIVE)
    asyncio.run(bridge.step())
    assert [(s.direction.value, float(s.position_size_pct)) for s in sink.signals] == [("LONG", 15.0)]  # la position adoptée est republiée telle quelle


def test_a_reconciliation_that_never_finishes_ends_in_the_known_conservative_fallback() -> None:
    view = EngineView((), D("10000"), True, True, STATE_RECONCILING)
    bridge, _, sink, clock = _bridge(view, instruments=("BTC/USDT",), engine_startup_grace_s=60.0, engine_grace_s=30.0)
    asyncio.run(bridge.step())
    clock.now += 61
    asyncio.run(bridge.step())  # la tolérance est écoulée : le silence redevient une panne
    clock.now += 31
    asyncio.run(bridge.step())
    assert sink.signals and sink.signals[-1].direction.value == "FLAT"


# --- Reporter ----------------------------------------------------------------------------------


def test_the_report_format_is_what_the_engine_parses() -> None:
    built = rp.build_report("bybit", D("10234.5"), {"BTCUSDT": (D("0.012"), D("50000")), "ETHUSDT": (D(0), D("3000"))}, NOW_MS)
    parsed = rec.parse_report(json.dumps(built))
    assert parsed is not None and parsed.venue == "BYBIT" and parsed.equity == D("10234.5")
    assert parsed.positions == (ReportedPosition("BTCUSDT", D("0.012"), D("50000")),)


class _Source:
    def __init__(self, venue: str, *, fail: Exception | None = None) -> None:
        from workers.common.ratelimit import RateLimitGate

        self.venue = venue
        self.rate_gate = RateLimitGate()
        self.fail = fail
        self.reads = 0

    def read(self) -> tuple[Decimal, dict[str, tuple[Decimal, Decimal]]]:
        self.reads += 1
        if self.fail is not None:
            raise self.fail
        return D("1000"), {"BTC": (D("0.01"), D("50000"))}


class _Sink2:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any], float]] = []

    async def publish(self, venue: str, report: Any, ttl_s: float) -> None:
        self.published.append((venue, dict(report), ttl_s))

    async def close(self) -> None: ...


class _Alerts:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, Any]] = []

    def alert(self, text: str, *, key: str, severity: Any = None) -> bool:
        self.alerts.append((key, severity))
        return True


def test_each_cycle_publishes_one_report_per_venue_with_a_ttl_of_three_intervals() -> None:
    sink = _Sink2()
    reporter = rp.PositionReporter([_Source("HYPERLIQUID"), _Source("BYBIT")], sink, interval_s=30.0, clock_ms=lambda: NOW_MS)

    async def one_cycle() -> None:
        await asyncio.gather(*(reporter.report_once(s) for s in reporter._sources))

    asyncio.run(one_cycle())
    assert sorted(v for v, _, _ in sink.published) == ["BYBIT", "HYPERLIQUID"] and all(ttl == 90.0 for _, _, ttl in sink.published)


def test_a_failing_venue_never_stops_the_others_and_alerts_once_then_recovers() -> None:
    sink, alerts = _Sink2(), _Alerts()
    bad, good = _Source("BYBIT", fail=ConnectionError("down")), _Source("HYPERLIQUID")
    reporter = rp.PositionReporter([bad, good], sink, interval_s=30.0, alerter=alerts, alert_after=3, clock_ms=lambda: NOW_MS)
    for _ in range(5):
        assert asyncio.run(reporter.report_once(bad)) is False
        assert asyncio.run(reporter.report_once(good)) is True
    assert [k for k, _ in alerts.alerts] == ["reporter:BYBIT"]  # une seule alerte pour l'épisode
    bad.fail = None
    assert asyncio.run(reporter.report_once(bad)) is True
    assert [k for k, _ in alerts.alerts] == ["reporter:BYBIT", "end:reporter:BYBIT"] and len(sink.published) == 6


def test_a_rate_limit_pause_is_slept_exactly_and_never_hammered() -> None:
    from workers.common.ratelimit import RateLimitError

    source = _Source("BYBIT", fail=RateLimitError(429, 5.0))
    reporter = rp.PositionReporter([source], _Sink2(), interval_s=30.0)
    assert asyncio.run(reporter.report_once(source)) is False
    source.fail = None
    source.rate_gate.block(429, "1")
    t0 = time.monotonic()
    assert asyncio.run(reporter.report_once(source)) is True
    assert time.monotonic() - t0 >= 0.9  # la pause demandée par l'exchange est respectée avant la lecture


def test_the_ttl_must_cover_two_intervals() -> None:
    with pytest.raises(ValueError, match="ttl_s"):
        rp.PositionReporter([_Source("BYBIT")], _Sink2(), interval_s=30.0, ttl_s=45.0)
    with pytest.raises(ValueError):
        rp.PositionReporter([], _Sink2())


def test_the_hyperliquid_source_pairs_each_position_with_its_mid() -> None:
    from workers.hyperliquid.client import AccountSnapshot

    class Client:
        rate_gate = _Source("X").rate_gate

        def snapshot(self) -> AccountSnapshot:
            return AccountSnapshot(D("5000"), {"BTC": D("0.1"), "ETH": D("0"), "DOGE": D("10")})

        def mids(self) -> dict[str, Decimal]:
            return {"BTC": D("50000"), "ETH": D("3000")}  # pas de prix pour DOGE : on ne l'invente pas

    equity, positions = rp.HyperliquidSource(Client()).read()
    assert equity == D("5000") and positions == {"BTC": (D("0.1"), D("50000"))}


def test_missing_credentials_are_a_startup_refusal_not_a_silently_skipped_venue() -> None:
    sources, problems = rp.build_sources(["hyperliquid", "bybit", "binance"], "testnet", {})
    assert sources == [] and len(problems) == 3 and any("READONLY" in p for p in problems) and any("HL_ACCOUNT_ADDRESS" in p for p in problems)


def test_the_readonly_keys_are_used_and_the_trading_keys_are_not_even_looked_at() -> None:
    env = {"BYBIT_READONLY_API_KEY": "k", "BYBIT_READONLY_API_SECRET": "s", "BYBIT_API_KEY": "TRADING", "BYBIT_API_SECRET": "TRADING"}
    sources, problems = rp.build_sources(["bybit"], "testnet", env)
    assert problems == [] and [s.venue for s in sources] == ["BYBIT"]
    assert rp.build_sources(["bybit"], "testnet", {"BYBIT_API_KEY": "TRADING", "BYBIT_API_SECRET": "TRADING"})[0] == []  # sans clé READONLY : refusé


def test_the_command_line_refuses_to_start_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HL_ACCOUNT_ADDRESS", "BYBIT_READONLY_API_KEY", "BYBIT_READONLY_API_SECRET", "WEBHOOK_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rp, "configure_logging", lambda *a, **k: None)
    assert rp.main(["--venues", "bybit"]) == 2
    with pytest.raises(SystemExit):
        rp.main(["--interval-s", "0"])

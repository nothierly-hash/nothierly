"""Tests du worker Hyperliquid (``workers/hyperliquid``). Aucun réseau, aucune vraie clé.

La planification est vérifiée avec des chiffres calculés à la main (equity 10 000 USD, BTC à 50 000 USD,
5 % de cible -> 500 USD -> 0,01 BTC). Le worker est testé avec un faux client qui met à jour sa position
quand un ordre est « rempli », ce qui permet de prouver qu'un signal republié ne double pas la position.
L'adaptateur SDK est testé avec de faux modules : le câblage des appels, pas le service réel.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SecureSignalPublisher, WeakSecretError, build_envelope
from workers.hyperliquid import worker as worker_module
from workers.hyperliquid.client import (
    AccountSnapshot,
    Network,
    OrderOutcome,
    OrderRequest,
    SdkHyperliquidClient,
    parse_order_response,
)
from workers.hyperliquid.planner import (
    OrderPlan,
    RiskLimits,
    Skip,
    hl_asset,
    make_cloid,
    plan_order,
    round_price,
    size_for_pct,
)
from workers.hyperliquid.state import HlState
from workers.hyperliquid.worker import HyperliquidWorker, HyperliquidWorkerConfig

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
OTHER_SECRET = "another-secret-9876543210fedcba-9876543210fedcba"
T0 = 1_760_000_000_000  # instant fixe : jamais de dépendance à l'horloge ni au minuit UTC
D = Decimal


def _signal(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.HYPERLIQUID,
        asset_class=AssetClass.CRYPTO,
        symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG,
        position_size_pct=5.0,
        order_type=OrderType.MARKET,
        timestamp_ms=T0,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _plan(signal: PlatformSignal | None = None, **overrides: Any) -> OrderPlan | Skip:
    sig = signal or _signal()
    kw: dict[str, Any] = dict(
        asset="BTC", target_abs_size=size_for_pct(D(10000), D(50000), D(str(sig.position_size_pct)), 5),
        equity=D(10000), position=D(0), mid=D(50000), sz_decimals=5, others_notional=D(0),
        limits=RiskLimits(), halted=False,
    )
    kw.update(overrides)
    return plan_order(sig, **kw)


# --- Planification : chiffres calculés à la main ---------------------------------------------


def test_size_for_pct_matches_the_hand_computation() -> None:
    assert size_for_pct(D(10000), D(50000), D(5), 5) == D("0.01")  # 500 USD / 50 000
    assert size_for_pct(D(10000), D(30000), D(5), 5) == D("0.01666")  # 0.016666... arrondi VERS ZÉRO, jamais au-dessus
    assert size_for_pct(D(10000), D(3000), D(5), 4) == D("0.1666")
    assert size_for_pct(D(10000), D(50000), D(0), 5) == 0
    assert size_for_pct(D(0), D(50000), D(5), 5) == 0 and size_for_pct(D(10000), D(0), D(5), 5) == 0


def test_long_market_order_is_an_ioc_buy_capped_by_slippage() -> None:
    plan = _plan()
    assert isinstance(plan, OrderPlan)
    assert (plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only) == (True, D("0.01"), D(50250), "Ioc", False)  # 50 000 x 1,005


def test_short_market_order_is_an_ioc_sell_floored_by_slippage() -> None:
    plan = _plan(_signal(direction=SignalDirection.SHORT))
    assert isinstance(plan, OrderPlan)
    assert (plan.is_buy, plan.size, plan.limit_px, plan.target_size) == (False, D("0.01"), D(49750), D("-0.01"))  # 50 000 x 0,995


def test_republishing_the_same_target_is_a_no_op() -> None:
    assert _plan(position=D("0.01")) == Skip("on_target")


def test_flat_closes_with_a_reduce_only_order_and_is_a_no_op_when_already_flat() -> None:
    flat = _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)
    closing = _plan(flat, position=D("0.01"), target_abs_size=D(0))
    assert isinstance(closing, OrderPlan)
    assert (closing.is_buy, closing.size, closing.reduce_only) == (False, D("0.01"), True)
    assert _plan(flat, position=D(0), target_abs_size=D(0)) == Skip("on_target")


def test_flip_long_to_short_is_one_order_for_twice_the_size_and_not_reduce_only() -> None:
    plan = _plan(_signal(direction=SignalDirection.SHORT), position=D("0.01"))
    assert isinstance(plan, OrderPlan)
    assert (plan.is_buy, plan.size, plan.reduce_only) == (False, D("0.02"), False)


def test_partial_reduction_is_reduce_only() -> None:
    plan = _plan(position=D("0.02"))  # cible 0,01 : on vend 0,01
    assert isinstance(plan, OrderPlan)
    assert (plan.is_buy, plan.size, plan.reduce_only) == (False, D("0.01"), True)


def test_small_drift_inside_the_tolerance_is_ignored() -> None:
    assert _plan(position=D("0.0099")) == Skip("within_tolerance")  # écart 0,0001 BTC = 5 USD < 0,5 % de 10 000


def test_orders_below_the_minimum_notional_are_skipped() -> None:
    sig = _signal()
    target = size_for_pct(D(100), D(50000), D(5), 5)  # 5 USD
    assert _plan(sig, equity=D(100), target_abs_size=target) == Skip("below_min_notional")


def test_position_cap_and_gross_cap() -> None:
    assert _plan(_signal(position_size_pct=25.0)) == Skip("exceeds_position_cap")  # plafond par défaut : 20 %
    wide = RiskLimits(max_position_pct=D(100))
    big = _signal(position_size_pct=60.0)  # 6 000 USD
    target = size_for_pct(D(10000), D(50000), D(60), 5)
    assert _plan(big, limits=wide, target_abs_size=target, others_notional=D(5000)) == Skip("exceeds_gross_cap")  # 11 000 > 10 000
    assert isinstance(_plan(big, limits=wide, target_abs_size=target, others_notional=D(4000)), OrderPlan)  # 10 000 : à la limite


def test_halted_mode_only_allows_reductions() -> None:
    assert _plan(halted=True) == Skip("halted_reduce_only")
    assert isinstance(_plan(_signal(direction=SignalDirection.FLAT, position_size_pct=0.0), position=D("0.01"), target_abs_size=D(0), halted=True), OrderPlan)
    assert isinstance(_plan(position=D("0.02"), halted=True), OrderPlan)  # réduction partielle autorisée


def test_limit_order_uses_the_metadata_price_and_refuses_a_fat_finger() -> None:
    ok = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 49900.0}))
    assert isinstance(ok, OrderPlan) and (ok.limit_px, ok.tif) == (D(49900), "Ioc")  # IOC par défaut : rien ne reste au carnet
    far = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 60000.0}))  # +2 000 bps du mid
    assert far == Skip("limit_far_from_mid")
    gtc = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 49900.0}), limits=RiskLimits(limit_tif="Gtc"))
    assert isinstance(gtc, OrderPlan) and gtc.tif == "Gtc"


def test_prediction_only_never_trades() -> None:
    assert _plan(_signal(order_type=OrderType.PREDICTION_ONLY, position_size_pct=0.0)) == Skip("prediction_only")


@pytest.mark.parametrize(
    ("price", "sz_decimals", "expected"),
    [
        ("50250", 5, "50250"),  # entier : toujours permis
        ("50250.5", 5, "50251"),  # 5 chiffres significatifs -> 0 décimale
        ("123456.7", 5, "123457"),  # plus de 5 chiffres, mais entier : permis
        ("0.123456", 0, "0.12346"),  # 5 chiffres significatifs
        ("0.123456", 3, "0.123"),  # plafond 6 - 3 = 3 décimales
        ("1.234567", 2, "1.2346"),  # 4 décimales (5 chiffres) <= 6 - 2
        ("1.234567", 4, "1.23"),  # plafond 6 - 4 = 2 décimales
    ],
)
def test_round_price_follows_the_exchange_rules(price: str, sz_decimals: int, expected: str) -> None:
    assert round_price(D(price), sz_decimals) == D(expected)


def test_round_price_rejects_non_positive_prices() -> None:
    for bad in ("0", "-1"):
        with pytest.raises(ValueError):
            round_price(D(bad), 5)


def test_asset_mapping() -> None:
    assert hl_asset("CRYPTO_BTC_USD") == "BTC" and hl_asset("CRYPTO_ETH_USDC") == "ETH"
    assert hl_asset("CRYPTO_BTC") is None and hl_asset("CRYPTO_BTC_EUR") is None and hl_asset("EQUITY_US_AAPL") is None
    assert hl_asset("CRYPTO_PEPE_USD", {"CRYPTO_PEPE_USD": "kPEPE"}) == "kPEPE"


def test_cloid_is_a_deterministic_128_bit_hex_per_signal() -> None:
    a, b = _signal(), _signal()
    assert make_cloid(a) == make_cloid(a) and make_cloid(a) != make_cloid(b)
    assert len(make_cloid(a)) == 34 and make_cloid(a).startswith("0x")
    int(make_cloid(a), 16)


# --- Worker : faux client --------------------------------------------------------------------


class FakeHl:
    def __init__(self, equity: str = "10000", positions: Mapping[str, str] | None = None) -> None:
        self.equity = D(equity)
        self.positions: dict[str, Decimal] = {k: D(v) for k, v in (positions or {}).items()}
        self.mid = {"BTC": D(50000), "ETH": D(3000)}
        self.decimals = {"BTC": 5, "ETH": 4}
        self.orders: list[OrderRequest] = []
        self.place_calls = 0
        self.fail_read = False
        self.fail_place = False
        self.reject = False

    def asset_decimals(self) -> Mapping[str, int]:
        return self.decimals

    def snapshot(self) -> AccountSnapshot:
        if self.fail_read:
            raise ConnectionError("lecture impossible")
        return AccountSnapshot(self.equity, dict(self.positions))

    def mids(self) -> Mapping[str, Decimal]:
        return self.mid

    def place_order(self, request: OrderRequest) -> OrderOutcome:
        self.place_calls += 1
        if self.fail_place:
            raise TimeoutError("réponse perdue")
        if self.reject:
            return OrderOutcome("rejected", D(0), None, None, "Insufficient margin")
        self.orders.append(request)
        delta = request.size if request.is_buy else -request.size
        self.positions[request.asset] = self.positions.get(request.asset, D(0)) + delta
        return OrderOutcome("filled", request.size, request.limit_px, 7, "")


def _worker(tmp_path: Path, client: FakeHl, **overrides: Any) -> HyperliquidWorker:
    cfg: dict[str, Any] = dict(address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "hl.sqlite3", dry_run=False)
    cfg.update(overrides)
    return HyperliquidWorker(HyperliquidWorkerConfig(**cfg), client)


def _run(worker: HyperliquidWorker, signal: PlatformSignal, asset: str = "BTC", now_ms: int | None = None) -> str:
    return asyncio.run(worker.execute_signal(signal, asset, now_ms=T0 + 100 if now_ms is None else now_ms))


def _frames(signal: PlatformSignal, secret: str = SECRET) -> list[bytes]:
    return [DEFAULT_TOPIC, json.dumps(build_envelope(signal, secret)).encode("utf-8")]


def test_long_signal_places_the_expected_order_and_journals_it(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client)
    sig = _signal()
    assert _run(w, sig) == "filled"
    (req,) = client.orders
    assert (req.asset, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only, req.cloid) == (
        "BTC", True, D("0.01"), D(50250), "Ioc", False, make_cloid(sig))
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.asset, rec.signal_id) == ("filled", "BUY", "BTC", str(sig.signal_id))


def test_the_same_target_republished_does_not_stack_positions(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "filled"
    for _ in range(5):  # le moteur republie la même cible à chaque tick
        assert _run(w, _signal()) == "on_target"
    assert len(client.orders) == 1 and client.positions["BTC"] == D("0.01")


def test_flat_signal_closes_the_position_with_a_reduce_only_order(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client)
    assert _run(w, _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)) == "filled"
    (req,) = client.orders
    assert (req.is_buy, req.reduce_only, req.size) == (False, True, D("0.01"))
    assert client.positions["BTC"] == 0


def test_dry_run_sends_nothing_but_journals_what_it_would_do(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client, dry_run=True)
    assert _run(w, _signal()) == "dry_run"
    assert client.place_calls == 0
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.size) == ("dry_run", "BUY", "0.01000")


def test_stale_or_expired_signals_are_refused_at_execution_time(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client, max_age_ms=5_000)
    assert _run(w, _signal(), now_ms=T0 + 5_001) == "stale_at_execution"
    assert _run(w, _signal(ttl_ms=1_000), now_ms=T0 + 1_001) == "stale_at_execution"
    assert client.place_calls == 0


def test_unknown_asset_and_read_failures_do_not_order(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client)
    assert _run(w, _signal(symbol="CRYPTO_DOGE_USD"), asset="DOGE") == "unknown_asset"
    client.fail_read = True
    assert _run(w, _signal()) == "read_error"
    assert client.place_calls == 0


def test_an_ambiguous_order_error_is_never_retried_and_a_rejection_is_journaled(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client)
    client.fail_place = True
    assert _run(w, _signal()) == "error"
    assert client.place_calls == 1  # un seul essai : le prochain signal relit la position réelle
    client.fail_place, client.reject = False, True
    assert _run(w, _signal()) == "rejected"
    assert [r.status for r in w.state.executions()] == ["error", "rejected"]


def test_drawdown_breaker_switches_to_reduce_only_persistently_until_cleared(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "on_target"  # fixe la base du jour : 10 000 USD
    client.equity = D("9400")  # -6 % >= 5 %
    assert _run(w, _signal(direction=SignalDirection.SHORT)) == "halted_reduce_only"
    assert w.halted
    w.state.close()
    reborn = _worker(tmp_path, client)  # même fichier : le coupe-circuit survit au redémarrage
    assert reborn.halted
    assert _run(reborn, _signal(direction=SignalDirection.SHORT)) == "halted_reduce_only"
    assert _run(reborn, _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)) == "filled"  # sortir reste possible
    reborn.clear_halt()
    assert not reborn.halted


def test_calculate_order_size_queries_the_account_when_not_given_a_snapshot(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeHl())
    assert asyncio.run(w._calculate_order_size("BTC", 5.0)) == D("0.01")
    assert asyncio.run(w._calculate_order_size("ETH", 5.0)) == D("0.1666")  # 500 / 3 000 arrondi vers zéro à 4 décimales
    with pytest.raises(KeyError):
        asyncio.run(w._calculate_order_size("DOGE", 5.0))


def test_a_weak_bus_secret_stops_the_worker_at_startup(tmp_path: Path) -> None:
    with pytest.raises(WeakSecretError):
        _worker(tmp_path, FakeHl(), secret_key="court")


# --- Réception -------------------------------------------------------------------------------


def test_only_valid_hyperliquid_signals_are_queued(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeHl())
    now = T0 + 100
    assert w.handle_frames(_frames(_signal()), now_ms=now) == "queued"
    assert w.handle_frames(_frames(_signal(target_venue=TargetVenue.ALL, symbol="CRYPTO_ETH_USD")), now_ms=now) == "queued"
    assert w.handle_frames(_frames(_signal(), secret=OTHER_SECRET), now_ms=now) == "bad_signature"
    forex = PlatformSignal(
        target_venue=TargetVenue.DARWINEX_ZERO, asset_class=AssetClass.FOREX, symbol="FOREX_EUR_USD",
        direction=SignalDirection.SHORT, position_size_pct=2.0, order_type=OrderType.MARKET,
        metadata={"stop_loss_price": 1.12}, timestamp_ms=T0,
    )
    assert w.handle_frames(_frames(forex), now_ms=now) == "wrong_venue"
    assert w.handle_frames(_frames(_signal(symbol="CRYPTO_BTC_EUR")), now_ms=now) == "unmappable_symbol"
    assert w.handle_frames([b"AUTRE", b"{}"], now_ms=now) == "bad_frames"
    assert sorted(w._pending) == ["BTC", "ETH"]


def test_too_old_and_expired_signals_are_not_queued(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeHl(), max_age_ms=5_000)
    assert w.handle_frames(_frames(_signal()), now_ms=T0 + 5_001) == "too_old"
    assert w.handle_frames(_frames(_signal(ttl_ms=500)), now_ms=T0 + 501) == "expired"
    assert w._pending == {}


def test_replay_is_rejected_even_after_a_restart(tmp_path: Path) -> None:
    sig = _signal(timestamp_ms=int(time.time() * 1000))
    first = _worker(tmp_path, FakeHl())
    assert first.handle_frames(_frames(sig)) == "queued"
    assert first.handle_frames(_frames(sig)) == "replay"
    first.state.close()
    assert _worker(tmp_path, FakeHl()).handle_frames(_frames(sig)) == "replay"


def test_only_the_latest_signal_per_asset_is_kept(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeHl())
    older = _signal(timestamp_ms=T0, position_size_pct=2.0)
    newer = _signal(timestamp_ms=T0 + 50, position_size_pct=4.0)
    now = T0 + 100
    w.handle_frames(_frames(newer), now_ms=now)
    w.handle_frames(_frames(older), now_ms=now)  # arrive après : ne remplace pas
    assert w._pending["BTC"].position_size_pct == 4.0


# --- Boucle ZMQ réelle -----------------------------------------------------------------------


def test_run_loop_executes_once_and_ignores_republished_targets(tmp_path: Path) -> None:
    client = FakeHl()

    async def scenario() -> tuple[int, list[str]]:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
            worker = _worker(tmp_path, client, address=pub.endpoint, poll_ms=20)
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            first = _signal(timestamp_ms=int(time.time() * 1000))
            deadline = time.monotonic() + 8.0
            while not client.orders and time.monotonic() < deadline:
                pub.publish(first)  # même signal_id répété : le ReplayGuard n'en laisse passer qu'un
                await asyncio.sleep(0.05)
            for _ in range(5):  # nouveaux signaux, même cible : doivent tomber en on_target
                pub.publish(_signal(timestamp_ms=int(time.time() * 1000)))
                await asyncio.sleep(0.05)
            deadline = time.monotonic() + 5.0
            while len(worker.state.executions()) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return len(client.orders), [r.status for r in worker.state.executions()]

    orders, statuses = asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop)
    assert orders == 1 and statuses[0] == "filled" and set(statuses[1:]) == {"skipped:on_target"}


# --- Adaptateur SDK : câblage avec de faux modules -------------------------------------------

ACCOUNT = "0x" + "11" * 20
VAULT = "0x" + "22" * 20
FAKE_KEY = "0x" + "ab" * 32  # valeur factice de test, sans lien avec une vraie clé


class _FakeInfo:
    def __init__(self, base_url: str, skip_ws: bool = False, timeout: float | None = None) -> None:
        self.base_url, self.skip_ws, self.queried = base_url, skip_ws, []

    def meta(self) -> dict[str, Any]:
        return {"universe": [{"name": "BTC", "szDecimals": 5}, {"name": "ETH", "szDecimals": 4}]}

    def user_state(self, address: str) -> dict[str, Any]:
        self.queried.append(address)
        return {
            "marginSummary": {"accountValue": "12345.67"},
            "assetPositions": [{"type": "oneWay", "position": {"coin": "BTC", "szi": "-0.5"}}],
        }

    def all_mids(self) -> dict[str, str]:
        return {"BTC": "50000.5", "ETH": "3000.25"}


class _FakeExchange:
    last: _FakeExchange | None = None

    def __init__(self, wallet: Any, base_url: str, **kwargs: Any) -> None:
        self.wallet, self.base_url, self.kwargs, self.calls = wallet, base_url, kwargs, []
        _FakeExchange.last = self

    def order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((args, kwargs))
        return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": "0.01", "avgPx": "50010.0", "oid": 99}}]}}}


class _FakeCloid:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    @classmethod
    def from_str(cls, raw: str) -> _FakeCloid:
        return cls(raw)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("hyperliquid")
    module("hyperliquid.utils")
    module("hyperliquid.exchange", Exchange=_FakeExchange)
    module("hyperliquid.info", Info=_FakeInfo)
    module("hyperliquid.utils.types", Cloid=_FakeCloid)
    module("eth_account", Account=types.SimpleNamespace(from_key=lambda key: f"wallet-for-{len(key)}-chars"))
    _FakeExchange.last = None


def test_read_only_client_never_builds_an_exchange_and_refuses_orders(fake_sdk: None) -> None:
    client = SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT)
    assert client.read_only and _FakeExchange.last is None
    with pytest.raises(RuntimeError, match="lecture seule"):
        client.place_order(OrderRequest("BTC", True, D("0.01"), D(50250), "Ioc", False, "0x" + "00" * 16))


def test_client_parses_the_account_state_and_prices(fake_sdk: None) -> None:
    client = SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT)
    snap = client.snapshot()
    assert snap.equity == D("12345.67") and snap.positions == {"BTC": D("-0.5")}
    assert client.mids() == {"BTC": D("50000.5"), "ETH": D("3000.25")}
    assert client.asset_decimals() == {"BTC": 5, "ETH": 4}


def test_client_queries_the_vault_when_trading_for_one_and_uses_the_right_network(fake_sdk: None) -> None:
    client = SdkHyperliquidClient(network=Network.MAINNET, account_address=ACCOUNT, vault_address=VAULT, agent_secret=FAKE_KEY)
    client.snapshot()
    assert client._info.queried == [VAULT]
    assert client._info.base_url == "https://api.hyperliquid.xyz" and client._info.skip_ws is True
    assert _FakeExchange.last is not None and _FakeExchange.last.kwargs["vault_address"] == VAULT
    testnet = SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT)
    assert testnet._info.base_url == "https://api.hyperliquid-testnet.xyz"


def test_client_sends_the_documented_order_call(fake_sdk: None) -> None:
    client = SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT, agent_secret=FAKE_KEY)
    cloid = "0x" + "ab" * 16
    outcome = client.place_order(OrderRequest("BTC", True, D("0.01"), D("50250"), "Ioc", False, cloid))
    assert (outcome.status, outcome.filled_size, outcome.avg_px, outcome.oid) == ("filled", D("0.01"), D("50010.0"), 99)
    assert _FakeExchange.last is not None
    ((args, kwargs),) = _FakeExchange.last.calls
    assert args == ("BTC", True, 0.01, 50250.0, {"limit": {"tif": "Ioc"}})
    assert kwargs["reduce_only"] is False and kwargs["cloid"].raw == cloid


def test_a_bad_private_key_is_refused_without_echoing_it(fake_sdk: None) -> None:
    bad = "not-a-key-" + "x" * 40
    with pytest.raises(ValueError) as exc:
        SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT, agent_secret=bad)
    assert bad not in str(exc.value)
    with pytest.raises(ValueError):
        SdkHyperliquidClient(network=Network.TESTNET, account_address="0x123")


def test_the_agent_key_never_appears_in_the_client_repr(fake_sdk: None) -> None:
    client = SdkHyperliquidClient(network=Network.TESTNET, account_address=ACCOUNT, agent_secret=FAKE_KEY)
    assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(vars(client))


def test_order_response_parsing() -> None:
    filled = {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": "1.5", "avgPx": "10", "oid": 1}}]}}}
    resting = {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 5}}]}}}
    errored = {"status": "ok", "response": {"data": {"statuses": [{"error": "Order must have minimum value of $10."}]}}}
    assert parse_order_response(filled).status == "filled" and parse_order_response(filled).filled_size == D("1.5")
    assert parse_order_response(resting).status == "resting" and parse_order_response(resting).oid == 5
    assert parse_order_response(errored).status == "rejected" and "minimum value" in parse_order_response(errored).detail
    for garbage in ({"status": "err", "response": "boom"}, None, "x", {"status": "ok"}, {"status": "ok", "response": {"data": {"statuses": []}}}):
        assert parse_order_response(garbage).status == "rejected"


# --- Garde-fous de la ligne de commande ------------------------------------------------------


def _cli_env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key in ("HL_ACCOUNT_ADDRESS", "HL_AGENT_SECRET", "HL_VAULT_ADDRESS", "NOTHIERLY_ZMQ_ADDRESS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_mainnet_live_requires_the_explicit_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_env(monkeypatch, HL_ACCOUNT_ADDRESS=ACCOUNT, HL_AGENT_SECRET=FAKE_KEY)

    def forbidden(**_: Any) -> None:
        raise AssertionError("aucun client ne doit être créé")

    monkeypatch.setattr(worker_module, "SdkHyperliquidClient", forbidden)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://x:1", "--live", "--network", "mainnet"])


def test_live_requires_an_agent_key_and_dry_run_never_receives_one(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def recorder(**kwargs: Any) -> None:
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(worker_module, "SdkHyperliquidClient", recorder)
    _cli_env(monkeypatch, HL_ACCOUNT_ADDRESS=ACCOUNT)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://x:1", "--live"])  # pas de HL_AGENT_SECRET
    _cli_env(monkeypatch, HL_ACCOUNT_ADDRESS=ACCOUNT, HL_AGENT_SECRET=FAKE_KEY)
    with pytest.raises(RuntimeError, match="stop"):
        worker_module.main(["--address", "tcp://x:1"])  # dry-run : la clé présente dans l'environnement est ignorée
    assert seen["agent_secret"] is None and seen["network"] is Network.TESTNET


# --- Homme mort : sortie d'urgence -----------------------------------------------------------


def _flatten(worker: HyperliquidWorker, asset: str = "BTC", attempt: int = 1) -> str:
    return asyncio.run(worker.emergency_flatten(asset, attempt=attempt, now_ms=T0))


def test_emergency_flatten_sells_a_long_with_a_reduce_only_ioc_below_the_mid(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client)
    assert _flatten(w) == "filled"
    (req,) = client.orders
    assert (req.asset, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only) == ("BTC", False, D("0.01"), D(48500), "Ioc", True)  # 50 000 x (1 - 3 %)
    assert client.positions["BTC"] == 0
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.signal_id) == ("dead_man_flatten:filled", "SELL", "dead-man:BTC")


def test_emergency_flatten_buys_back_a_short_above_the_mid(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "-0.02"})
    assert _flatten(_worker(tmp_path, client)) == "filled"
    (req,) = client.orders
    assert (req.is_buy, req.size, req.limit_px, req.reduce_only) == (True, D("0.02"), D(51500), True)  # 50 000 x (1 + 3 %)


def test_emergency_slippage_widens_with_each_attempt(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client)
    client.reject = True  # l'ordre n'est jamais accepté : on regarde seulement le prix demandé
    sent = []
    for attempt in (1, 2, 3, 4, 5):
        client.reject = False
        client.positions["BTC"] = D("0.01")
        _flatten(w, attempt=attempt)
        sent.append(client.orders[-1].limit_px)
    assert sent == [D(48500), D(47000), D(44000), D(40000), D(40000)]  # 3 %, 6 %, 12 %, puis plafond 20 %


def test_emergency_flatten_does_nothing_on_a_flat_account_and_reports_failures(tmp_path: Path) -> None:
    client = FakeHl()
    w = _worker(tmp_path, client)
    assert _flatten(w) == "no_position" and client.place_calls == 0
    client.positions["BTC"] = D("0.01")
    assert _flatten(w, asset="DOGE") == "unknown_asset"
    client.fail_read = True
    assert _flatten(w) == "read_error"
    client.fail_read, client.fail_place = False, True
    assert _flatten(w) == "error"
    client.fail_place, client.reject = False, True
    assert _flatten(w) == "rejected"
    assert client.positions["BTC"] == D("0.01")  # rien n'a fermé : le worker réessaiera


def test_emergency_flatten_in_dry_run_sends_nothing(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client, dry_run=True)
    assert _flatten(w) == "dry_run" and client.place_calls == 0
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.size, rec.px) == ("dead_man_flatten:dry_run", "SELL", "0.01", "48500")


def test_a_silent_engine_closes_the_hyperliquid_position_then_locks_the_worker(tmp_path: Path) -> None:
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client, dead_man_timeout_s=300.0, dead_man_startup_grace_s=0.0)
    w._started_ms = T0 - 10_000_000
    assert w.handle_frames(_frames(_signal()), now_ms=T0 + 100) == "queued"
    assert asyncio.run(w.check_dead_man(T0 + 100 + 300_000)) == "ok"
    assert asyncio.run(w.check_dead_man(T0 + 100 + 300_001)) == "tripped"
    assert client.positions["BTC"] == 0
    assert (client.orders[0].is_buy, client.orders[0].reduce_only) == (False, True)  # vente en sortie seule
    assert w.handle_frames(_frames(_signal(timestamp_ms=T0 + 400_000)), now_ms=T0 + 400_100) == "dead_man_halted"
    assert asyncio.run(w.check_dead_man(T0 + 100 + 300_002)) == "halted_flat"


def test_hyperliquid_supports_the_dead_man_switch_option(tmp_path: Path) -> None:
    assert HyperliquidWorker.supports_dead_man
    assert _worker(tmp_path, FakeHl(), dead_man_timeout_s=300.0).dead_man_tripped is False


def test_cli_clear_halt_needs_no_sdk_no_key_and_no_bus_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _cli_env(monkeypatch)  # aucune variable : ni compte, ni clé, ni secret de bus

    def forbidden(**_: Any) -> None:
        raise AssertionError("aucun client réseau ne doit être créé pour lever un verrou")

    monkeypatch.setattr(worker_module, "SdkHyperliquidClient", forbidden)
    db = tmp_path / "locked.sqlite3"
    state = HlState(db)
    state.set("halted", "1")
    state.set("dead_man_tripped", "123")
    state.set("dead_man_pending", '["BTC"]')
    state.set("last_signal:BTC", "1")
    state.close()
    worker_module.main(["--clear-halt", "--db", str(db)])
    after = HlState(db)
    assert after.get("halted") is None and after.get("dead_man_tripped") is None and after.get("dead_man_pending") is None
    assert int(after.get("last_signal:BTC") or 0) > 1  # l'horloge repart de maintenant


def test_cli_dead_man_timeout_defaults_to_300_and_zero_is_refused_on_mainnet_live(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class _Stop(Exception):
        pass

    def worker_factory(config: Any, client: Any) -> None:
        seen.append(config.dead_man_timeout_s)
        raise _Stop

    monkeypatch.setattr(worker_module, "SdkHyperliquidClient", lambda **_: object())
    monkeypatch.setattr(worker_module, "HyperliquidWorker", worker_factory)
    for args, expected in ((["--address", "tcp://x:1"], 300.0), (["--address", "tcp://x:1", "--dead-man-timeout", "45"], 45.0),
                           (["--address", "tcp://x:1", "--dead-man-timeout", "0"], None)):
        _cli_env(monkeypatch, HL_ACCOUNT_ADDRESS=ACCOUNT, NOTHIERLY_ZMQ_SECRET=SECRET)
        with pytest.raises(_Stop):
            worker_module.main(args)
        assert seen[-1] == expected
    _cli_env(monkeypatch, HL_ACCOUNT_ADDRESS=ACCOUNT, HL_AGENT_SECRET=FAKE_KEY, NOTHIERLY_ZMQ_SECRET=SECRET)
    with pytest.raises(SystemExit):  # mainnet live sans homme mort : refusé
        worker_module.main(["--address", "tcp://x:1", "--live", "--network", "mainnet", "--i-understand-real-funds", "--dead-man-timeout", "0"])
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://x:1", "--dead-man-timeout", "-5"])

"""Tests du worker Bybit (``workers/bybit``) et du socle commun (``workers/common``). Aucun réseau, aucune clé.

Chiffres calculés à la main : equity 10 000 USD, BTCUSDT à 50 000, cible 5 % -> 500 USD -> 0,010 BTC
(pas de quantité 0,001, tick 0,1). La signature HMAC est recalculée dans le test d'après la formule de la doc
Bybit V5 (``timestamp + clé API + recv_window + (query | corps JSON)``), indépendamment du code testé.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SecureSignalPublisher, build_envelope
from workers.bybit import worker as worker_module
from workers.bybit.client import (
    BybitApiError,
    BybitNetwork,
    BybitOrderOutcome,
    BybitOrderRequest,
    BybitPosition,
    BybitRestClient,
    BybitSnapshot,
    HedgeModeError,
    KeyInfo,
    check_key_info,
)
from workers.bybit.planner import (
    InstrumentSpec,
    OrderPlan,
    RiskLimits,
    Skip,
    bybit_symbol,
    make_order_link_id,
    plan_order,
    round_price,
)
from workers.bybit.worker import BybitWorker, BybitWorkerConfig
from workers.common.exposure import floor_to_step, size_for_pct_step

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
OTHER_SECRET = "another-secret-9876543210fedcba-9876543210fedcba"
T0 = 1_760_000_000_000
D = Decimal

BTC = InstrumentSpec("BTCUSDT", qty_step=D("0.001"), min_qty=D("0.001"), max_qty=D("1000"), min_notional=D("5"), tick_size=D("0.1"), copy_trading="both")
ETH = InstrumentSpec("ETHUSDT", qty_step=D("0.01"), min_qty=D("0.01"), max_qty=D("5000"), min_notional=D("5"), tick_size=D("0.01"), copy_trading="both")
DOGE = InstrumentSpec("DOGEUSDT", qty_step=D("1"), min_qty=D("1"), max_qty=None, min_notional=D("5"), tick_size=D("0.00001"), copy_trading="none")


def _signal(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.BYBIT, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG, position_size_pct=5.0, order_type=OrderType.MARKET, timestamp_ms=T0,
    )
    base.update(overrides)
    return PlatformSignal(**base)


# --- Briques communes ------------------------------------------------------------------------


def test_floor_to_step_and_step_sizing() -> None:
    assert floor_to_step(D("0.016666"), D("0.001")) == D("0.016")
    assert floor_to_step(D("0.02"), D("0.001")) == D("0.02")  # déjà un multiple : inchangé
    assert floor_to_step(D("0.0009"), D("0.001")) == 0
    with pytest.raises(ValueError):
        floor_to_step(D("1"), D("0"))
    assert size_for_pct_step(D(10000), D(50000), D(5), D("0.001")) == D("0.01")  # 500 / 50 000
    assert size_for_pct_step(D(10000), D(30000), D(5), D("0.001")) == D("0.016")  # 0,016666 arrondi vers zéro
    assert size_for_pct_step(D(0), D(50000), D(5), D("0.001")) == 0


# --- Planification ---------------------------------------------------------------------------


def _plan(signal: PlatformSignal | None = None, **overrides: Any) -> OrderPlan | Skip:
    sig = signal or _signal()
    kw: dict[str, Any] = dict(
        symbol="BTCUSDT", target_abs_size=size_for_pct_step(D(10000), D(50000), D(str(sig.position_size_pct)), BTC.qty_step),
        equity=D(10000), position=D(0), mid=D(50000), spec=BTC, others_notional=D(0), limits=RiskLimits(), halted=False,
    )
    kw.update(overrides)
    return plan_order(sig, **kw)


def test_symbol_mapping_only_accepts_usdt_linear_perps() -> None:
    assert bybit_symbol("CRYPTO_BTC_USD") == "BTCUSDT" and bybit_symbol("CRYPTO_ETH_USDT") == "ETHUSDT"
    assert bybit_symbol("CRYPTO_BTC_USDC") is None and bybit_symbol("CRYPTO_BTC") is None and bybit_symbol("EQUITY_US_AAPL") is None
    assert bybit_symbol("CRYPTO_PEPE_USD", {"CRYPTO_PEPE_USD": "1000PEPEUSDT"}) == "1000PEPEUSDT"


def test_order_link_id_is_deterministic_and_within_bybit_limits() -> None:
    a, b = _signal(), _signal()
    assert make_order_link_id(a) == make_order_link_id(a) != make_order_link_id(b)
    assert len(make_order_link_id(a)) == 33 <= 36 and make_order_link_id(a).isalnum()


def test_round_price_directions() -> None:
    assert round_price(D("50374.06725"), D("0.1"), up=True) == D("50374.1")
    assert round_price(D("50374.06725"), D("0.1"), up=False) == D("50374.0")
    assert round_price(D("50250.0"), D("0.1"), up=True) == D("50250.0")  # déjà un multiple du tick : inchangé
    with pytest.raises(ValueError):
        round_price(D(0), D("0.1"), up=True)


def test_long_market_is_an_ioc_limit_capped_by_slippage() -> None:
    plan = _plan()
    assert isinstance(plan, OrderPlan)
    assert (plan.symbol, plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only) == (
        "BTCUSDT", True, D("0.01"), D("50250.0"), "IOC", False)  # 50 000 x 1,005


def test_short_market_sells_down_to_the_slippage_floor() -> None:
    plan = _plan(_signal(direction=SignalDirection.SHORT))
    assert isinstance(plan, OrderPlan)
    assert (plan.is_buy, plan.size, plan.limit_px, plan.target_size) == (False, D("0.01"), D("49750.0"), D("-0.01"))  # 50 000 x 0,995


def test_a_buy_rounds_up_and_a_sell_rounds_down_so_the_order_always_crosses() -> None:
    buy = _plan(mid=D("50123.45"))
    sell = _plan(_signal(direction=SignalDirection.SHORT), mid=D("50123.45"))
    assert isinstance(buy, OrderPlan) and isinstance(sell, OrderPlan)
    assert buy.limit_px == D("50374.1")  # 50 374,06725 arrondi au tick supérieur
    assert sell.limit_px == D("49872.8")  # 49 872,83275 arrondi au tick inférieur


def test_same_target_is_a_no_op_and_flat_closes_reduce_only() -> None:
    assert _plan(position=D("0.01")) == Skip("on_target")
    closing = _plan(_signal(direction=SignalDirection.FLAT, position_size_pct=0.0), position=D("0.01"), target_abs_size=D(0))
    assert isinstance(closing, OrderPlan) and (closing.is_buy, closing.size, closing.reduce_only) == (False, D("0.01"), True)


def test_flip_is_not_reduce_only_and_partial_reduction_is() -> None:
    flip = _plan(_signal(direction=SignalDirection.SHORT), position=D("0.01"))
    assert isinstance(flip, OrderPlan) and (flip.size, flip.reduce_only) == (D("0.02"), False)
    partial = _plan(position=D("0.02"))
    assert isinstance(partial, OrderPlan) and (partial.is_buy, partial.size, partial.reduce_only) == (False, D("0.01"), True)


def test_exchange_rules_quantity_and_notional_limits() -> None:
    assert _plan(target_abs_size=D("0.005"), spec=InstrumentSpec("BTCUSDT", D("0.001"), D("0.01"), D("1000"), D("5"), D("0.1"), "both")) == Skip("below_min_qty")
    assert _plan(spec=InstrumentSpec("BTCUSDT", D("0.001"), D("0.001"), D("0.005"), D("5"), D("0.1"), "both")) == Skip("exceeds_max_order_qty")
    assert _plan(spec=InstrumentSpec("BTCUSDT", D("0.001"), D("0.001"), D("1000"), D("600"), D("0.1"), "both")) == Skip("below_min_notional")  # 500 < 600 de la plateforme
    closing = _plan(_signal(direction=SignalDirection.FLAT, position_size_pct=0.0), position=D("0.02"), target_abs_size=D(0),
                    spec=InstrumentSpec("BTCUSDT", D("0.001"), D("0.05"), D("1000"), D("5"), D("0.1"), "both"))
    assert isinstance(closing, OrderPlan)  # une sortie n'est pas bloquée par la quantité minimale


def test_limit_orders_round_toward_the_less_aggressive_side_and_refuse_fat_fingers() -> None:
    buy = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 49999.97}))
    sell = _plan(_signal(direction=SignalDirection.SHORT, order_type=OrderType.LIMIT, metadata={"limit_price": 50000.03}))
    assert isinstance(buy, OrderPlan) and isinstance(sell, OrderPlan)
    assert buy.limit_px == D("49999.9") and sell.limit_px == D("50000.1")  # jamais plus cher / moins cher qu'annoncé
    assert _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 60000.0})) == Skip("limit_far_from_mid")
    gtc = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 49900.0}), limits=RiskLimits(limit_tif="Gtc"))
    assert isinstance(gtc, OrderPlan) and gtc.tif == "GTC"
    with pytest.raises(ValueError, match="limit_tif"):
        _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 49900.0}), limits=RiskLimits(limit_tif="Fok"))


def test_shared_risk_rules_apply_here_too() -> None:
    assert _plan(_signal(position_size_pct=25.0)) == Skip("exceeds_position_cap")
    assert _plan(halted=True) == Skip("halted_reduce_only")
    assert _plan(position=D("0.0099")) == Skip("within_tolerance")
    assert _plan(_signal(order_type=OrderType.PREDICTION_ONLY, position_size_pct=0.0)) == Skip("prediction_only")


# --- Client REST : signature et décodage -----------------------------------------------------

KEY, API_SECRET, CLOCK = "KEY123", "SECRET456", 1_700_000_000_000


class FakeTransport:
    """Transport sans réseau : répond selon le chemin et garde chaque appel pour inspection."""

    def __init__(self) -> None:
        self.routes: dict[str, list[tuple[int, Any]]] = {}
        self.calls: list[dict[str, Any]] = []

    def add(self, path: str, payload: Any, status: int = 200) -> None:
        self.routes.setdefault(path, []).append((status, payload))

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        path = url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body, "path": "/" + path})
        queue = self.routes["/" + path]
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()


def _ok(result: Any) -> dict[str, Any]:
    return {"retCode": 0, "retMsg": "OK", "result": result}


def _client(transport: FakeTransport, network: BybitNetwork = BybitNetwork.TESTNET) -> BybitRestClient:
    return BybitRestClient(api_key=KEY, api_secret=API_SECRET, network=network, transport=transport, clock_ms=lambda: CLOCK)


def _expected_sign(payload: str) -> str:
    return hmac.new(API_SECRET.encode(), f"{CLOCK}{KEY}5000{payload}".encode(), hashlib.sha256).hexdigest()


def test_get_signature_follows_the_documented_formula() -> None:
    t = FakeTransport()
    t.add("/v5/user/query-api", _ok({"readOnly": 0, "permissions": {}}))
    _client(t).key_info()
    (call,) = t.calls
    assert call["headers"] == {
        "X-BAPI-API-KEY": KEY, "X-BAPI-TIMESTAMP": str(CLOCK), "X-BAPI-RECV-WINDOW": "5000", "X-BAPI-SIGN": _expected_sign(""),
    }
    t.add("/v5/market/tickers", _ok({"list": []}))
    _client(t).mid("BTCUSDT")
    query = t.calls[1]["url"].split("?", 1)[1]
    assert query == "category=linear&symbol=BTCUSDT"
    assert t.calls[1]["headers"]["X-BAPI-SIGN"] == _expected_sign(query)  # la query signée est celle de l'URL


def test_post_signature_covers_the_exact_body_that_is_sent() -> None:
    t = FakeTransport()
    t.add("/v5/order/create", _ok({"orderId": "123", "orderLinkId": "x"}))
    req = BybitOrderRequest("BTCUSDT", True, D("0.010"), D("50250.0"), "IOC", False, "n" + "ab" * 16)
    outcome = _client(t).place_order(req)
    (call,) = t.calls
    assert call["method"] == "POST" and call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["X-BAPI-SIGN"] == _expected_sign(call["body"].decode())
    assert json.loads(call["body"]) == {
        "category": "linear", "symbol": "BTCUSDT", "side": "Buy", "orderType": "Limit", "qty": "0.01", "price": "50250",
        "timeInForce": "IOC", "reduceOnly": False, "orderLinkId": "n" + "ab" * 16, "positionIdx": 0,
    }
    assert (outcome.status, outcome.order_id) == ("accepted", "123")


def test_a_rejection_is_an_outcome_but_an_unreadable_answer_is_an_exception() -> None:
    t = FakeTransport()
    t.add("/v5/order/create", {"retCode": 110007, "retMsg": "ab not enough for new order", "result": {}})
    req = BybitOrderRequest("BTCUSDT", False, D("0.01"), D("49750"), "IOC", True, "nxx")
    rejected = _client(t).place_order(req)
    assert rejected.status == "rejected" and "110007" in rejected.detail
    t2 = FakeTransport()
    t2.add("/v5/order/create", b"<html>502 Bad Gateway</html>", status=502)
    with pytest.raises(BybitApiError):  # résultat ambigu : le worker ne le traitera pas comme un refus
        _client(t2).place_order(req)


def test_read_errors_raise_with_the_bybit_code() -> None:
    t = FakeTransport()
    t.add("/v5/account/wallet-balance", {"retCode": 10002, "retMsg": "invalid request, please check your server timestamp", "result": {}})
    with pytest.raises(BybitApiError) as exc:
        _client(t).snapshot()
    assert exc.value.code == 10002


def test_instruments_are_paginated_filtered_and_decoded() -> None:
    t = FakeTransport()
    page1 = {"nextPageCursor": "CUR1", "list": [
        {"symbol": "BTCUSDT", "settleCoin": "USDT", "contractType": "LinearPerpetual", "copyTrading": "both",
         "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001", "maxOrderQty": "1190", "minNotionalValue": "5"},
         "priceFilter": {"tickSize": "0.10"}},
        {"symbol": "BTCPERP", "settleCoin": "USDC", "contractType": "LinearPerpetual", "lotSizeFilter": {"qtyStep": "0.001"}, "priceFilter": {"tickSize": "0.1"}},
    ]}
    page2 = {"nextPageCursor": "", "list": [
        {"symbol": "ETHUSDT", "settleCoin": "USDT", "contractType": "LinearPerpetual",
         "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01", "maxOrderQty": "0", "minNotionalValue": "5"}, "priceFilter": {"tickSize": "0.01"}},
        {"symbol": "BTC-26DEC25", "settleCoin": "USDT", "contractType": "LinearFutures", "lotSizeFilter": {"qtyStep": "0.001"}, "priceFilter": {"tickSize": "0.1"}},
    ]}
    t.add("/v5/market/instruments-info", _ok(page1))
    t.add("/v5/market/instruments-info", _ok(page2))
    specs = _client(t).instruments()
    assert sorted(specs) == ["BTCUSDT", "ETHUSDT"]  # ni USDC ni contrat à terme
    assert specs["BTCUSDT"] == InstrumentSpec("BTCUSDT", D("0.001"), D("0.001"), D("1190"), D("5"), D("0.10"), "both")
    assert specs["ETHUSDT"].max_qty is None and specs["ETHUSDT"].copy_trading == "unknown"
    assert "cursor=CUR1" in t.calls[1]["url"]


def test_pagination_cannot_loop_forever() -> None:
    t = FakeTransport()
    t.add("/v5/market/instruments-info", _ok({"nextPageCursor": "SAME", "list": []}))
    with pytest.raises(BybitApiError, match="pagination"):
        _client(t).instruments()


def test_snapshot_decodes_equity_and_signed_positions() -> None:
    t = FakeTransport()
    t.add("/v5/account/wallet-balance", _ok({"list": [{"accountType": "UNIFIED", "totalEquity": "12345.67"}]}))
    t.add("/v5/position/list", _ok({"nextPageCursor": "", "list": [
        {"symbol": "BTCUSDT", "side": "Buy", "size": "0.5", "positionIdx": 0, "markPrice": "50000"},
        {"symbol": "ETHUSDT", "side": "Sell", "size": "2", "positionIdx": 0, "markPrice": "3000"},
        {"symbol": "SOLUSDT", "side": "", "size": "0", "positionIdx": 0, "markPrice": "150"},
    ]}))
    snap = _client(t).snapshot()
    assert snap.equity == D("12345.67")
    assert snap.positions == {"BTCUSDT": BybitPosition(D("0.5"), D("50000")), "ETHUSDT": BybitPosition(D("-2"), D("3000"))}
    assert t.calls[0]["url"].endswith("accountType=UNIFIED")


def test_snapshot_refuses_hedge_mode_and_tolerates_an_empty_wallet() -> None:
    t = FakeTransport()
    t.add("/v5/account/wallet-balance", _ok({"list": []}))
    t.add("/v5/position/list", _ok({"list": [{"symbol": "BTCUSDT", "side": "Buy", "size": "1", "positionIdx": 1, "markPrice": "5"}]}))
    with pytest.raises(HedgeModeError):
        _client(t).snapshot()
    t2 = FakeTransport()
    t2.add("/v5/account/wallet-balance", _ok({"list": []}))
    t2.add("/v5/position/list", _ok({"list": []}))
    assert _client(t2).snapshot().equity == 0


def test_mid_uses_the_book_then_the_last_price() -> None:
    book, last, empty = FakeTransport(), FakeTransport(), FakeTransport()
    book.add("/v5/market/tickers", _ok({"list": [{"bid1Price": "49999.9", "ask1Price": "50000.1", "lastPrice": "1"}]}))
    last.add("/v5/market/tickers", _ok({"list": [{"bid1Price": "", "ask1Price": "", "lastPrice": "49000"}]}))
    empty.add("/v5/market/tickers", _ok({"list": []}))
    assert _client(book).mid("BTCUSDT") == D("50000.0")
    assert _client(last).mid("BTCUSDT") == D("49000")
    assert _client(empty).mid("NOPE") is None


def test_key_info_and_the_withdraw_guard() -> None:
    t = FakeTransport()
    t.add("/v5/user/query-api", _ok({"readOnly": 0, "permissions": {"ContractTrade": ["Order", "Position"], "Wallet": ["AccountTransfer"]}}))
    good = _client(t).key_info()
    assert not good.can_withdraw and good.can_trade_contracts and not good.read_only
    assert check_key_info(good, live=True) == []
    risky = KeyInfo(read_only=False, permissions={"ContractTrade": ["Order", "Position"], "Wallet": ["Withdraw"]})
    assert any("retrait" in p for p in check_key_info(risky, live=False))  # refusé même en dry-run
    ro = KeyInfo(read_only=True, permissions={"ContractTrade": ["Order", "Position"], "Wallet": []})
    assert check_key_info(ro, live=False) == [] and any("lecture seule" in p for p in check_key_info(ro, live=True))
    no_perm = KeyInfo(read_only=False, permissions={"ContractTrade": [], "Wallet": []})
    assert any("Orders & Positions" in p for p in check_key_info(no_perm, live=True))


def test_client_uses_the_right_host_and_never_exposes_secrets() -> None:
    t = FakeTransport()
    t.add("/v5/user/query-api", _ok({"readOnly": 1, "permissions": {}}))
    _client(t, BybitNetwork.MAINNET).key_info()
    _client(t, BybitNetwork.TESTNET).key_info()
    assert t.calls[0]["url"].startswith("https://api.bybit.com/") and t.calls[1]["url"].startswith("https://api-testnet.bybit.com/")
    assert API_SECRET not in repr(_client(t)) and KEY not in repr(_client(t))
    with pytest.raises(ValueError):
        BybitRestClient(api_key="", api_secret="x", network=BybitNetwork.TESTNET)


# --- Worker : faux client --------------------------------------------------------------------


class FakeBybit:
    def __init__(self, equity: str = "10000", positions: Mapping[str, str] | None = None) -> None:
        self.equity = D(equity)
        self.positions = {k: D(v) for k, v in (positions or {}).items()}
        self.specs = {"BTCUSDT": BTC, "ETHUSDT": ETH, "DOGEUSDT": DOGE}
        self.prices = {"BTCUSDT": D(50000), "ETHUSDT": D(3000), "DOGEUSDT": D("0.1")}
        self.orders: list[BybitOrderRequest] = []
        self.place_calls = 0
        self.fail_read = False
        self.hedge = False
        self.fail_place = False
        self.reject = False

    def instruments(self) -> Mapping[str, InstrumentSpec]:
        return self.specs

    def snapshot(self) -> BybitSnapshot:
        if self.hedge:
            raise HedgeModeError("position hedge")
        if self.fail_read:
            raise ConnectionError("lecture impossible")
        return BybitSnapshot(self.equity, {s: BybitPosition(q, self.prices[s]) for s, q in self.positions.items() if q != 0})

    def mid(self, symbol: str) -> Decimal | None:
        return self.prices.get(symbol)

    def key_info(self) -> KeyInfo:
        return KeyInfo(read_only=False, permissions={"ContractTrade": ["Order", "Position"], "Wallet": []})

    def place_order(self, request: BybitOrderRequest) -> BybitOrderOutcome:
        self.place_calls += 1
        if self.fail_place:
            raise TimeoutError("réponse perdue")
        if self.reject:
            return BybitOrderOutcome("rejected", None, "110007: insufficient balance")
        self.orders.append(request)
        self.positions[request.symbol] = self.positions.get(request.symbol, D(0)) + (request.size if request.is_buy else -request.size)
        return BybitOrderOutcome("accepted", "oid-1", "")


def _worker(tmp_path: Path, client: FakeBybit, **overrides: Any) -> BybitWorker:
    cfg: dict[str, Any] = dict(address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "by.sqlite3", dry_run=False)
    cfg.update(overrides)
    return BybitWorker(BybitWorkerConfig(**cfg), client)


def _run(worker: BybitWorker, signal: PlatformSignal, asset: str = "BTCUSDT", now_ms: int | None = None) -> str:
    return asyncio.run(worker.execute_signal(signal, asset, now_ms=T0 + 100 if now_ms is None else now_ms))


def _frames(signal: PlatformSignal, secret: str = SECRET) -> list[bytes]:
    return [DEFAULT_TOPIC, json.dumps(build_envelope(signal, secret)).encode("utf-8")]


def test_long_signal_places_the_expected_ioc_order_and_journals_it(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    sig = _signal()
    assert _run(w, sig) == "accepted"
    (req,) = client.orders
    assert (req.symbol, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only, req.order_link_id) == (
        "BTCUSDT", True, D("0.01"), D("50250.0"), "IOC", False, make_order_link_id(sig))
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.asset) == ("accepted", "BUY", "BTCUSDT")


def test_the_same_target_republished_never_stacks_positions(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "accepted"
    assert [_run(w, _signal()) for _ in range(5)] == ["on_target"] * 5
    assert len(client.orders) == 1 and client.positions["BTCUSDT"] == D("0.01")


def test_flat_closes_with_a_reduce_only_order(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01"})
    w = _worker(tmp_path, client)
    assert _run(w, _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)) == "accepted"
    (req,) = client.orders
    assert (req.is_buy, req.reduce_only, req.size) == (False, True, D("0.01")) and client.positions["BTCUSDT"] == 0


def test_dry_run_sends_nothing_but_journals_the_plan(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client, dry_run=True)
    assert _run(w, _signal()) == "dry_run" and client.place_calls == 0
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.size, rec.px) == ("dry_run", "BUY", "0.01", "50250.0")


def test_refusals_stale_unknown_not_copy_tradable_no_price(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    assert _run(w, _signal(), now_ms=T0 + 5_001) == "stale_at_execution"
    assert _run(w, _signal(symbol="CRYPTO_XRP_USD"), asset="XRPUSDT") == "unknown_symbol"
    assert _run(w, _signal(symbol="CRYPTO_DOGE_USD"), asset="DOGEUSDT") == "not_copy_tradable"
    del client.prices["BTCUSDT"]
    assert _run(w, _signal()) == "no_price"
    assert client.place_calls == 0
    relaxed = _worker(tmp_path, FakeBybit(), db_path=tmp_path / "other.sqlite3", require_copy_trading=False, dry_run=True)
    assert _run(relaxed, _signal(symbol="CRYPTO_DOGE_USD"), asset="DOGEUSDT") == "dry_run"


def test_read_failures_and_hedge_mode_stop_the_order(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    client.fail_read = True
    assert _run(w, _signal()) == "read_error"
    client.fail_read, client.hedge = False, True
    assert _run(w, _signal()) == "hedge_mode_unsupported"
    assert client.place_calls == 0


def test_an_ambiguous_error_is_not_retried_and_a_rejection_is_journaled(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    client.fail_place = True
    assert _run(w, _signal()) == "error" and client.place_calls == 1
    client.fail_place, client.reject = False, True
    assert _run(w, _signal()) == "rejected"
    assert [r.status for r in w.state.executions()] == ["error", "rejected"]


def test_max_equity_caps_the_sizing_but_not_the_drawdown_measure(tmp_path: Path) -> None:
    client = FakeBybit(equity="100000")  # le compte lu est dix fois plus gros que prévu
    w = _worker(tmp_path, client, max_equity_usd=D(10000))
    assert _run(w, _signal()) == "accepted"
    assert client.orders[0].size == D("0.01")  # 5 % de 10 000, pas de 100 000
    assert asyncio.run(w._calculate_order_size("BTCUSDT", 5.0)) == D("0.01")


def test_the_shared_drawdown_breaker_works_for_bybit_and_persists(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01"})
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "on_target"  # base du jour : 10 000
    client.equity = D("9400")  # -6 %
    assert _run(w, _signal(direction=SignalDirection.SHORT)) == "halted_reduce_only" and w.halted
    w.state.close()
    reborn = _worker(tmp_path, client)
    assert reborn.halted
    assert _run(reborn, _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)) == "accepted"
    reborn.clear_halt()
    assert not reborn.halted


def test_calculate_order_size_queries_the_account(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeBybit())
    assert asyncio.run(w._calculate_order_size("BTCUSDT", 5.0)) == D("0.01")
    assert asyncio.run(w._calculate_order_size("ETHUSDT", 5.0)) == D("0.16")  # 500 / 3 000 = 0,1666 arrondi au pas de 0,01
    with pytest.raises(KeyError):
        asyncio.run(w._calculate_order_size("XRPUSDT", 5.0))


def test_only_bybit_signals_are_queued(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeBybit())
    now = T0 + 100
    assert w.handle_frames(_frames(_signal()), now_ms=now) == "queued"
    assert w.handle_frames(_frames(_signal(target_venue=TargetVenue.ALL, symbol="CRYPTO_ETH_USD")), now_ms=now) == "queued"
    assert w.handle_frames(_frames(_signal(target_venue=TargetVenue.HYPERLIQUID)), now_ms=now) == "wrong_venue"
    assert w.handle_frames(_frames(_signal(), secret=OTHER_SECRET), now_ms=now) == "bad_signature"
    assert w.handle_frames(_frames(_signal(symbol="CRYPTO_BTC_USDC")), now_ms=now) == "unmappable_symbol"
    assert sorted(w._pending) == ["BTCUSDT", "ETHUSDT"]


def test_run_loop_executes_once_over_a_real_zmq_socket(tmp_path: Path) -> None:
    client = FakeBybit()

    async def scenario() -> tuple[int, list[str]]:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
            worker = _worker(tmp_path, client, address=pub.endpoint, poll_ms=20)
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            first = _signal(timestamp_ms=int(time.time() * 1000))
            deadline = time.monotonic() + 8.0
            while not client.orders and time.monotonic() < deadline:
                pub.publish(first)
                await asyncio.sleep(0.05)
            for _ in range(3):
                pub.publish(_signal(timestamp_ms=int(time.time() * 1000)))
                await asyncio.sleep(0.05)
            deadline = time.monotonic() + 5.0
            while len(worker.state.executions()) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return len(client.orders), [r.status for r in worker.state.executions()]

    orders, statuses = asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop)
    assert orders == 1 and statuses[0] == "accepted" and set(statuses[1:]) == {"skipped:on_target"}


# --- Ligne de commande -----------------------------------------------------------------------


def _env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "NOTHIERLY_ZMQ_ADDRESS", "NOTHIERLY_ZMQ_SECRET"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


class _KeyOnly:
    def __init__(self, info: KeyInfo) -> None:
        self._info = info

    def key_info(self) -> KeyInfo:
        return self._info


def test_cli_refuses_mainnet_live_without_the_acknowledgement_and_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, BYBIT_API_KEY="k", BYBIT_API_SECRET="s")

    def forbidden(**_: Any) -> None:
        raise AssertionError("aucun client ne doit être créé")

    monkeypatch.setattr(worker_module, "BybitRestClient", forbidden)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--live", "--network", "mainnet"])
    _env(monkeypatch)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1"])  # clés absentes


def test_cli_refuses_a_key_that_can_withdraw_and_never_starts_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, BYBIT_API_KEY="k", BYBIT_API_SECRET="s", NOTHIERLY_ZMQ_SECRET=SECRET)
    risky = KeyInfo(read_only=False, permissions={"ContractTrade": ["Order", "Position"], "Wallet": ["Withdraw"]})
    monkeypatch.setattr(worker_module, "BybitRestClient", lambda **_: _KeyOnly(risky))

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("le worker ne doit pas démarrer")

    monkeypatch.setattr(worker_module, "BybitWorker", forbidden)
    assert worker_module.main(["--address", "tcp://127.0.0.1:1"]) == 2
    assert worker_module.main(["--address", "tcp://127.0.0.1:1", "--live"]) == 2


def test_cli_defaults_to_dry_run_testnet_with_copy_trading_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, BYBIT_API_KEY="k", BYBIT_API_SECRET="s", NOTHIERLY_ZMQ_SECRET=SECRET)
    seen: dict[str, Any] = {}
    ok = KeyInfo(read_only=True, permissions={"ContractTrade": ["Order", "Position"], "Wallet": []})

    def client_factory(**kwargs: Any) -> _KeyOnly:
        seen["network"] = kwargs["network"]
        return _KeyOnly(ok)

    def worker_factory(config: BybitWorkerConfig, client: Any) -> None:
        seen["config"] = config
        raise RuntimeError("stop")

    monkeypatch.setattr(worker_module, "BybitRestClient", client_factory)
    monkeypatch.setattr(worker_module, "BybitWorker", worker_factory)
    with pytest.raises(RuntimeError, match="stop"):
        worker_module.main(["--address", "tcp://127.0.0.1:1"])
    assert seen["network"] is BybitNetwork.TESTNET
    assert seen["config"].dry_run is True and seen["config"].require_copy_trading is True and seen["config"].max_equity_usd is None


# --- Homme mort : sortie d'urgence -----------------------------------------------------------


def _flatten(worker: BybitWorker, asset: str = "BTCUSDT", attempt: int = 1) -> str:
    return asyncio.run(worker.emergency_flatten(asset, attempt=attempt, now_ms=T0))


def test_emergency_flatten_sells_a_long_with_a_reduce_only_ioc_rounded_down_to_the_tick(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01"})
    w = _worker(tmp_path, client)
    assert _flatten(w) == "accepted"
    (req,) = client.orders
    assert (req.symbol, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only) == ("BTCUSDT", False, D("0.01"), D("48500.0"), "IOC", True)  # 50 000 x (1 - 3 %)
    assert client.positions["BTCUSDT"] == 0 and len(req.order_link_id) == 33
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.signal_id) == ("dead_man_flatten:accepted", "SELL", "dead-man:BTCUSDT")


def test_emergency_flatten_buys_back_a_short_rounding_up_so_the_order_always_crosses(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "-0.02"})
    client.prices["BTCUSDT"] = D("50123.45")
    assert _flatten(_worker(tmp_path, client)) == "accepted"
    (req,) = client.orders
    assert (req.is_buy, req.size, req.limit_px, req.reduce_only) == (True, D("0.02"), D("51627.2"), True)  # 50 123,45 x 1,03 = 51 627,1535, arrondi au tick supérieur
    client2 = FakeBybit(positions={"BTCUSDT": "0.01"})
    client2.prices["BTCUSDT"] = D("50123.45")
    _flatten(_worker(tmp_path, client2, db_path=tmp_path / "b.sqlite3"))
    assert client2.orders[0].limit_px == D("48619.7")  # 48 619,7465 arrondi au tick inférieur


def test_emergency_slippage_widens_and_the_link_id_changes_with_each_attempt(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01"})
    w = _worker(tmp_path, client)
    for attempt in (1, 2, 3, 4):
        client.positions["BTCUSDT"] = D("0.01")
        _flatten(w, attempt=attempt)
    assert [o.limit_px for o in client.orders] == [D("48500.0"), D("47000.0"), D("44000.0"), D("40000.0")]
    assert len({o.order_link_id for o in client.orders}) == 4  # un identifiant distinct par tentative


def test_emergency_flatten_does_nothing_on_a_flat_account_and_reports_failures(tmp_path: Path) -> None:
    client = FakeBybit()
    w = _worker(tmp_path, client)
    assert _flatten(w) == "no_position" and client.place_calls == 0
    client.positions["BTCUSDT"] = D("0.01")
    assert _flatten(w, asset="XRPUSDT") == "unknown_symbol"
    client.fail_read = True
    assert _flatten(w) == "read_error"
    client.fail_read, client.hedge = False, True
    assert _flatten(w) == "hedge_mode_unsupported" and client.place_calls == 0  # ne s'envoie pas : on le dit fort, on réessaie
    client.hedge, client.fail_place = False, True
    assert _flatten(w) == "error"
    client.fail_place, client.reject = False, True
    assert _flatten(w) == "rejected"
    assert client.positions["BTCUSDT"] == D("0.01")


def test_emergency_flatten_in_dry_run_sends_nothing(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01"})
    w = _worker(tmp_path, client, dry_run=True)
    assert _flatten(w) == "dry_run" and client.place_calls == 0
    (rec,) = w.state.executions()
    assert (rec.status, rec.px) == ("dead_man_flatten:dry_run", "48500.0")


def test_a_silent_engine_closes_every_bybit_position_then_locks_the_worker(tmp_path: Path) -> None:
    client = FakeBybit(positions={"BTCUSDT": "0.01", "ETHUSDT": "-2"})
    w = _worker(tmp_path, client, dead_man_timeout_s=300.0, dead_man_startup_grace_s=0.0)
    w._started_ms = T0 - 10_000_000
    assert w.handle_frames(_frames(_signal()), now_ms=T0 + 100) == "queued"
    assert w.handle_frames(_frames(_signal(symbol="CRYPTO_ETH_USD", timestamp_ms=T0 + 280_000)), now_ms=T0 + 280_100) == "queued"
    assert asyncio.run(w.check_dead_man(T0 + 100 + 300_001)) == "tripped"
    assert client.positions == {"BTCUSDT": D(0), "ETHUSDT": D(0)}  # ETH parlait encore, mais plus aucun signal ne sera traité
    assert sorted((o.symbol, o.is_buy, o.reduce_only) for o in client.orders) == [("BTCUSDT", False, True), ("ETHUSDT", True, True)]
    assert w.handle_frames(_frames(_signal(timestamp_ms=T0 + 500_000)), now_ms=T0 + 500_100) == "dead_man_halted"


def test_bybit_supports_the_dead_man_switch_option(tmp_path: Path) -> None:
    assert BybitWorker.supports_dead_man


def test_cli_clear_halt_needs_no_key_no_secret_and_no_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env(monkeypatch)  # aucune variable

    def forbidden(**_: Any) -> None:
        raise AssertionError("aucun client réseau ne doit être créé pour lever un verrou")

    monkeypatch.setattr(worker_module, "BybitRestClient", forbidden)
    from workers.common.state import ExecutionState

    db = tmp_path / "locked.sqlite3"
    state = ExecutionState(db)
    state.set("halted", "1")
    state.set("dead_man_tripped", "123")
    state.set("dead_man_pending", '["BTCUSDT"]')
    state.set("last_signal:BTCUSDT", "1")
    state.close()
    assert worker_module.main(["--clear-halt", "--db", str(db)]) == 0
    after = ExecutionState(db)
    assert after.get("halted") is None and after.get("dead_man_tripped") is None and after.get("dead_man_pending") is None
    assert int(after.get("last_signal:BTCUSDT") or 0) > 1


def test_cli_dead_man_timeout_defaults_to_300_and_zero_is_refused_on_mainnet_live(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    ok = KeyInfo(read_only=False, permissions={"ContractTrade": ["Order", "Position"], "Wallet": []})

    class _Stop(Exception):
        pass

    def worker_factory(config: BybitWorkerConfig, client: Any) -> None:
        seen.append(config.dead_man_timeout_s)
        raise _Stop

    monkeypatch.setattr(worker_module, "BybitRestClient", lambda **_: _KeyOnly(ok))
    monkeypatch.setattr(worker_module, "BybitWorker", worker_factory)
    for args, expected in ((["--address", "tcp://127.0.0.1:1"], 300.0), (["--address", "tcp://127.0.0.1:1", "--dead-man-timeout", "45"], 45.0),
                           (["--address", "tcp://127.0.0.1:1", "--dead-man-timeout", "0"], None)):
        _env(monkeypatch, BYBIT_API_KEY="k", BYBIT_API_SECRET="s", NOTHIERLY_ZMQ_SECRET=SECRET)
        with pytest.raises(_Stop):
            worker_module.main(args)
        assert seen[-1] == expected
    _env(monkeypatch, BYBIT_API_KEY="k", BYBIT_API_SECRET="s", NOTHIERLY_ZMQ_SECRET=SECRET)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--live", "--network", "mainnet", "--i-understand-real-funds", "--dead-man-timeout", "0"])
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--dead-man-timeout", "-5"])

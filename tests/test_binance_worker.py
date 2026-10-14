"""Tests du worker Binance USDⓈ-M (``workers/binance/``) et de son flux de fills (``BinanceFillFeed``).

Aucun appel réseau, aucune clé : transport HTTP simulé, faux client pour le worker, faux flux WebSocket. Les réponses
simulées reprennent les formes de la documentation officielle (``account`` V3, ``positionSide/dual``, ``apiRestrictions``,
erreurs ``{"code": -2019, "msg": ...}``) et, pour ``exchangeInfo``, ``bookTicker`` et ``ORDER_TRADE_UPDATE``, le schéma
connu de l'API (la page consultée ne les détaillait pas) : ces tests prouvent que le code lit CES formes, pas que Binance
les envoie encore telles quelles.

Preuves recherchées :

- la signature est celle de la formule documentée (HMAC-SHA256 de la chaîne de requête, ``signature`` en dernier) ;
- une clé avec droit de retrait, un compte en hedge, une clé illisible en mainnet : refus propre au démarrage (code 2) ;
- les règles de précision (pas, tick, notionnel minimal) sont appliquées ; un refus de Binance est ``rejected``, un résultat
  inconnu lève (jamais de nouvel essai à l'aveugle) ;
- chaque exécution (partielle ou finale) du flux privé est rapprochée de l'intention et enregistrée avec son slippage.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import urllib.parse
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, build_envelope
from workers.binance import worker as worker_module
from workers.binance.client import (
    BinanceApiError,
    BinanceNetwork,
    BinanceOrderOutcome,
    BinanceOrderRequest,
    BinancePosition,
    BinanceRestClient,
    BinanceSnapshot,
    HedgeModeError,
    KeyPermissions,
    check_key_permissions,
)
from workers.binance.planner import InstrumentSpec, OrderPlan, Skip, binance_symbol, make_client_order_id, plan_order, round_price
from workers.binance.worker import BinanceWorker, BinanceWorkerConfig
from workers.common.exposure import RiskLimits, size_for_pct_step
from workers.common.fills_db import FillsDatabase
from workers.common.ws_tracker import BinanceFillFeed, FillTracker

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
API_KEY = "KEY-abc123"
API_SECRET = "api-secret-DO-NOT-LEAK"
T0 = 1_760_000_000_000
D = Decimal

BTC = InstrumentSpec("BTCUSDT", qty_step=D("0.001"), min_qty=D("0.001"), max_qty=D("1000"), min_notional=D("100"), tick_size=D("0.1"))
ETH = InstrumentSpec("ETHUSDT", qty_step=D("0.01"), min_qty=D("0.01"), max_qty=D("10000"), min_notional=D("20"), tick_size=D("0.01"))


# --- Transport simulé ------------------------------------------------------------------------


class FakeTransport:
    """Enregistre les requêtes ; ``routes`` associe ``(méthode, chemin)`` à une réponse ``(statut, corps)`` ou à une liste jouée dans l'ordre."""

    def __init__(self, routes: Mapping[tuple[str, str], Any] | None = None) -> None:
        self.routes: dict[tuple[str, str], Any] = dict(routes or {})
        self.calls: list[tuple[str, str, dict[str, str], dict[str, str], bytes | None]] = []  # méthode, chemin, paramètres, en-têtes, corps
        self.hosts: list[str] = []

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        parts = urllib.parse.urlsplit(url)
        params = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        self.calls.append((method, parts.path, params, dict(headers), body))
        self.hosts.append(f"{parts.scheme}://{parts.netloc}")
        route = self.routes[(method, parts.path)]
        if isinstance(route, list):
            route = route.pop(0) if len(route) > 1 else route[0]
        if isinstance(route, Exception):
            raise route
        status, payload = route
        return status, payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def last(self, path: str) -> tuple[str, dict[str, str], dict[str, str]]:
        for method, p, params, headers, _ in reversed(self.calls):
            if p == path:
                return method, params, headers
        raise AssertionError(f"aucun appel à {path}")


def _client(transport: FakeTransport, network: BinanceNetwork = BinanceNetwork.MAINNET, **kwargs: Any) -> BinanceRestClient:
    return BinanceRestClient(api_key=API_KEY, api_secret=API_SECRET, network=network, transport=transport, clock_ms=lambda: T0, **kwargs)


def _signature(query_without_signature: str) -> str:
    return hmac.new(API_SECRET.encode(), query_without_signature.encode(), hashlib.sha256).hexdigest()


EXCHANGE_INFO = {"symbols": [
    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "marginAsset": "USDT", "filters": [
        {"filterType": "PRICE_FILTER", "minPrice": "556.80", "maxPrice": "4529764", "tickSize": "0.10"},
        {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "120", "stepSize": "0.001"},
        {"filterType": "MIN_NOTIONAL", "notional": "100"}]},
    {"symbol": "ETHUSDC", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDC", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"}]},
    {"symbol": "BTCUSDT_261225", "status": "TRADING", "contractType": "CURRENT_QUARTER", "quoteAsset": "USDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.1"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100"}]},
    {"symbol": "OLDUSDT", "status": "BREAK", "contractType": "PERPETUAL", "quoteAsset": "USDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.1"}, {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1", "maxQty": "100"}]},
    {"symbol": "NOLOTUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}]},
]}

ACCOUNT_V3 = {"totalWalletBalance": "10050.00", "totalMarginBalance": "10100.50", "availableBalance": "9000", "positions": [
    {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "-0.020", "notional": "-1000.0", "unrealizedProfit": "5", "updateTime": 1},
    {"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt": "0.000", "notional": "0", "unrealizedProfit": "0", "updateTime": 1},
]}


# --- Client : signature et lectures ----------------------------------------------------------


def test_a_signed_request_carries_the_documented_signature_last_and_the_key_in_the_header() -> None:
    transport = FakeTransport({("GET", "/fapi/v3/account"): (200, ACCOUNT_V3)})
    _client(transport).snapshot()
    ((method, path, params, headers, body),) = transport.calls
    assert (method, path, body) == ("GET", "/fapi/v3/account", None) and headers == {"X-MBX-APIKEY": API_KEY}
    assert params["timestamp"] == str(T0) and params["recvWindow"] == "5000"
    raw_query = urllib.parse.urlsplit("?" + "&".join(f"{k}={v}" for k, v in params.items())).query
    assert list(params)[-1] == "signature"  # en dernier, comme exigé
    unsigned = "&".join(f"{k}={v}" for k, v in params.items() if k != "signature")
    assert params["signature"] == _signature(unsigned) and raw_query


def test_the_secret_never_appears_in_the_request_the_repr_or_an_error_message() -> None:
    transport = FakeTransport({("GET", "/fapi/v3/account"): (400, {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action."})})
    client = _client(transport)
    with pytest.raises(BinanceApiError) as exc:
        client.snapshot()
    assert API_SECRET not in repr(client) and API_SECRET not in str(exc.value) and exc.value.code == -2015
    assert all(API_SECRET not in str(call) for call in transport.calls)
    with pytest.raises(ValueError):
        BinanceRestClient(api_key="", api_secret="x", network=BinanceNetwork.TESTNET)


def test_a_clock_skew_error_resynchronises_once_and_retries_with_the_corrected_timestamp() -> None:
    transport = FakeTransport({
        ("GET", "/fapi/v3/account"): [(400, {"code": -1021, "msg": "Timestamp for this request is outside of the recvWindow."}), (200, ACCOUNT_V3)],
        ("GET", "/fapi/v1/time"): (200, {"serverTime": T0 + 3000}),
    })
    _client(transport).snapshot()
    account_calls = [c for c in transport.calls if c[1] == "/fapi/v3/account"]
    assert len(account_calls) == 2 and account_calls[0][2]["timestamp"] == str(T0) and account_calls[1][2]["timestamp"] == str(T0 + 3000)
    always = FakeTransport({("GET", "/fapi/v3/account"): (400, {"code": -1021, "msg": "x"}), ("GET", "/fapi/v1/time"): (200, {"serverTime": T0})})
    with pytest.raises(BinanceApiError):  # un seul nouvel essai : jamais de boucle
        _client(always).snapshot()
    assert len([c for c in always.calls if c[1] == "/fapi/v3/account"]) == 2


def test_exchange_info_keeps_only_trading_usdt_perpetuals_with_the_filters_needed_to_round() -> None:
    specs = _client(FakeTransport({("GET", "/fapi/v1/exchangeInfo"): (200, EXCHANGE_INFO)})).instruments()
    assert list(specs) == ["BTCUSDT"]  # ni USDC, ni échéance, ni symbole hors cotation, ni symbole sans LOT_SIZE
    assert specs["BTCUSDT"] == InstrumentSpec("BTCUSDT", D("0.001"), D("0.001"), D("1000"), D("100"), D("0.10"))


def test_the_account_snapshot_reads_equity_signed_positions_and_the_mark_price() -> None:
    snapshot = _client(FakeTransport({("GET", "/fapi/v3/account"): (200, ACCOUNT_V3)})).snapshot()
    assert snapshot.equity == D("10100.50") and set(snapshot.positions) == {"BTCUSDT"}  # la position nulle est ignorée
    assert snapshot.positions["BTCUSDT"] == BinancePosition(D("-0.020"), D("50000"))  # |notional / quantité|


def test_a_hedge_position_raises_instead_of_being_sized_blindly() -> None:
    hedge = {"totalMarginBalance": "1", "positions": [{"symbol": "BTCUSDT", "positionSide": "LONG", "positionAmt": "0.1", "notional": "5000"}]}
    with pytest.raises(HedgeModeError, match="BTCUSDT"):
        _client(FakeTransport({("GET", "/fapi/v3/account"): (200, hedge)})).snapshot()


def test_mid_is_the_book_midpoint_with_a_last_price_fallback() -> None:
    book = FakeTransport({("GET", "/fapi/v1/ticker/bookTicker"): (200, {"symbol": "BTCUSDT", "bidPrice": "49999.9", "askPrice": "50000.1"})})
    assert _client(book).mid("BTCUSDT") == D("50000.0")
    empty = FakeTransport({("GET", "/fapi/v1/ticker/bookTicker"): (200, {"symbol": "X", "bidPrice": "0", "askPrice": "0"}),
                           ("GET", "/fapi/v1/ticker/price"): (200, {"symbol": "X", "price": "123.4"})})
    assert _client(empty).mid("X") == D("123.4")
    none = FakeTransport({("GET", "/fapi/v1/ticker/bookTicker"): (200, {"bidPrice": "0", "askPrice": "0"}), ("GET", "/fapi/v1/ticker/price"): (200, {"price": "0"})})
    assert _client(none).mid("X") is None


def test_the_position_mode_endpoint_is_read_for_hedge_detection() -> None:
    hedge = FakeTransport({("GET", "/fapi/v1/positionSide/dual"): (200, {"dualSidePosition": True})})
    one_way = FakeTransport({("GET", "/fapi/v1/positionSide/dual"): (200, {"dualSidePosition": False})})
    assert _client(hedge).is_hedge_mode() is True and _client(one_way).is_hedge_mode() is False


# --- Client : droits de la clé ---------------------------------------------------------------


def test_key_permissions_come_from_the_spot_host_api_restrictions_endpoint_not_from_trading_status() -> None:
    transport = FakeTransport({("GET", "/sapi/v1/account/apiRestrictions"): (200, {"ipRestrict": True, "enableReading": True, "enableWithdrawals": True, "enableFutures": True})})
    info = _client(transport).key_permissions()
    assert info == KeyPermissions(True, can_withdraw=True, can_read=True, can_trade_futures=True, ip_restricted=True)
    assert transport.hosts == ["https://api.binance.com"] and all(c[1] != "/fapi/v1/apiTradingStatus" for c in transport.calls)
    assert "signature" in transport.last("/sapi/v1/account/apiRestrictions")[1]


def test_unreadable_permissions_and_the_test_network_are_reported_as_unverified_not_as_safe() -> None:
    refused = FakeTransport({("GET", "/sapi/v1/account/apiRestrictions"): (400, {"code": -2015, "msg": "Invalid API-key"})})
    assert _client(refused).key_permissions().verified is False
    test_network = FakeTransport()
    info = _client(test_network, network=BinanceNetwork.TESTNET).key_permissions()
    assert info.verified is False and "réseau de test" in info.reason and test_network.calls == []  # aucun appel : l'endpoint n'existe pas là


def test_the_key_checks_refuse_withdrawal_rights_and_unprovable_mainnet_keys() -> None:
    safe = KeyPermissions(True, can_withdraw=False, can_read=True, can_trade_futures=True)
    assert check_key_permissions(safe, live=True, mainnet=True) == []
    assert any("retrait" in p for p in check_key_permissions(KeyPermissions(True, can_withdraw=True, can_trade_futures=True), live=False, mainnet=True))  # refusé même en dry-run
    assert any("prouvée" in p for p in check_key_permissions(KeyPermissions(False, reason="BinanceApiError"), live=False, mainnet=True))
    assert check_key_permissions(KeyPermissions(False, reason="réseau de test"), live=True, mainnet=False) == []  # testnet : non vérifiable, pas bloquant
    assert any("Enable Futures" in p for p in check_key_permissions(KeyPermissions(True, can_withdraw=False, can_trade_futures=False), live=True, mainnet=True))
    assert check_key_permissions(KeyPermissions(True, can_withdraw=False, can_trade_futures=False), live=False, mainnet=True) == []


# --- Client : ordre --------------------------------------------------------------------------


def _request(**overrides: Any) -> BinanceOrderRequest:
    base: dict[str, Any] = dict(symbol="BTCUSDT", is_buy=True, size=D("0.010"), limit_px=D("50250.0"), tif="IOC", reduce_only=False, client_order_id="nabc")
    base.update(overrides)
    return BinanceOrderRequest(**base)


def test_the_order_is_a_signed_limit_ioc_with_normalised_numbers_and_a_result_response() -> None:
    transport = FakeTransport({("POST", "/fapi/v1/order"): (200, {"orderId": 77, "status": "FILLED", "executedQty": "0.010", "clientOrderId": "nabc"})})
    outcome = _client(transport).place_order(_request())
    method, params, headers = transport.last("/fapi/v1/order")
    assert method == "POST" and transport.calls[0][4] == b"" and headers == {"X-MBX-APIKEY": API_KEY}
    assert {k: v for k, v in params.items() if k not in ("signature", "timestamp", "recvWindow")} == {
        "symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "IOC", "quantity": "0.01", "price": "50250",
        "newClientOrderId": "nabc", "newOrderRespType": "RESULT"}  # pas de reduceOnly, nombres sans zéros inutiles ni notation scientifique
    assert outcome == BinanceOrderOutcome("filled", "77", D("0.010"), "")
    sell = FakeTransport({("POST", "/fapi/v1/order"): (200, {"orderId": 1, "status": "FILLED", "executedQty": "1"})})
    _client(sell).place_order(_request(is_buy=False, reduce_only=True, size=D("1E+2"), limit_px=D("0.00001230")))
    _, sell_params, _ = sell.last("/fapi/v1/order")
    assert sell_params["side"] == "SELL" and sell_params["reduceOnly"] == "true" and sell_params["quantity"] == "100" and sell_params["price"] == "0.0000123"


def test_every_order_result_maps_to_a_distinct_status() -> None:
    def outcome(response: dict[str, Any]) -> BinanceOrderOutcome:
        return _client(FakeTransport({("POST", "/fapi/v1/order"): (200, response)})).place_order(_request())

    assert outcome({"orderId": 1, "status": "EXPIRED", "executedQty": "0"}).status == "unfilled"  # IOC sans exécution
    partial = outcome({"orderId": 2, "status": "EXPIRED", "executedQty": "0.004"})  # IOC exécuté en partie, le reste expire
    assert (partial.status, partial.executed_qty) == ("partial", D("0.004"))
    assert outcome({"orderId": 3, "status": "FILLED", "executedQty": "0.01"}).status == "filled"


def test_a_binance_refusal_is_rejected_but_an_unknown_outcome_raises_so_nothing_is_retried_blindly() -> None:
    refused = FakeTransport({("POST", "/fapi/v1/order"): (400, {"code": -2019, "msg": "Margin is insufficient."})})
    outcome = _client(refused).place_order(_request())
    assert outcome.status == "rejected" and "-2019" in outcome.detail and outcome.order_id is None
    unknown = FakeTransport({("POST", "/fapi/v1/order"): (503, {"code": -1007, "msg": "Timeout waiting for response from backend server. Send status unknown; execution status unknown."})})
    with pytest.raises(BinanceApiError):
        _client(unknown).place_order(_request())
    garbage = FakeTransport({("POST", "/fapi/v1/order"): (502, b"<html>Bad Gateway</html>")})
    with pytest.raises(BinanceApiError):
        _client(garbage).place_order(_request())
    broken = FakeTransport({("POST", "/fapi/v1/order"): TimeoutError("délai dépassé")})
    with pytest.raises(TimeoutError):
        _client(broken).place_order(_request())


def test_the_user_stream_key_endpoints_need_only_the_api_key_header() -> None:
    transport = FakeTransport({("POST", "/fapi/v1/listenKey"): (200, {"listenKey": "LK-123"}), ("PUT", "/fapi/v1/listenKey"): (200, {})})
    client = _client(transport)
    assert client.start_user_stream() == "LK-123"
    client.keepalive_user_stream()
    for method, path, params, headers, body in transport.calls:
        assert path == "/fapi/v1/listenKey" and params == {} and headers == {"X-MBX-APIKEY": API_KEY} and body == b""  # ni signature ni horodatage


def test_the_test_network_url_is_the_documented_one_and_can_be_overridden() -> None:
    transport = FakeTransport({("GET", "/fapi/v1/time"): (200, {"serverTime": T0})})
    _client(transport, network=BinanceNetwork.TESTNET).sync_time()
    assert transport.hosts == ["https://demo-fapi.binance.com"]
    custom = FakeTransport({("GET", "/fapi/v1/time"): (200, {"serverTime": T0})})
    _client(custom, network=BinanceNetwork.TESTNET, base_url="https://testnet.binancefuture.com/").sync_time()
    assert custom.hosts == ["https://testnet.binancefuture.com"]


# --- Planification ---------------------------------------------------------------------------


def _signal(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.BINANCE, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD", direction=SignalDirection.LONG,
        position_size_pct=5.0, order_type=OrderType.MARKET, timestamp_ms=T0,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _plan(signal: PlatformSignal | None = None, **overrides: Any) -> OrderPlan | Skip:
    sig = signal or _signal()
    kw: dict[str, Any] = dict(
        symbol="BTCUSDT", target_abs_size=size_for_pct_step(D(10000), D(50000), D(str(sig.position_size_pct)), BTC.qty_step), equity=D(10000),
        position=D(0), mid=D(50000), spec=BTC, others_notional=D(0), limits=RiskLimits(), halted=False,
    )
    kw.update(overrides)
    return plan_order(sig, **kw)


def test_symbols_are_translated_to_usdt_perpetuals_only() -> None:
    assert binance_symbol("CRYPTO_BTC_USD") == binance_symbol("CRYPTO_BTC_USDT") == "BTCUSDT"
    assert binance_symbol("CRYPTO_BTC_USDC") is None and binance_symbol("FOREX_EUR_USD") is None and binance_symbol("btc") is None
    assert binance_symbol("CRYPTO_PEPE_USD", {"CRYPTO_PEPE_USD": "1000PEPEUSDT"}) == "1000PEPEUSDT"


def test_the_client_order_id_is_deterministic_and_matches_the_documented_pattern() -> None:
    signal = _signal()
    assert make_client_order_id(signal) == make_client_order_id(signal) != make_client_order_id(_signal())
    assert re.fullmatch(r"[\.A-Z\:/a-z0-9_-]{1,36}", make_client_order_id(signal))


def test_a_long_market_signal_becomes_an_ioc_buy_rounded_to_step_and_tick() -> None:
    plan = _plan()
    assert isinstance(plan, OrderPlan)
    # 5 % de 10 000 = 500 USDT / 50 000 = 0,01 BTC (pas 0,001) ; limite = 50 000 x 1,005 = 50 250 (tick 0,1)
    assert (plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only) == (True, D("0.010"), D("50250.0"), "IOC", False)
    sell = _plan(_signal(direction=SignalDirection.SHORT))
    assert isinstance(sell, OrderPlan) and (sell.is_buy, sell.limit_px) == (False, D("49750.0"))  # 50 000 x 0,995


def test_the_size_is_floored_never_rounded_up_and_small_orders_are_skipped_by_the_exchange_rules() -> None:
    assert size_for_pct_step(D(10000), D(30000), D(5), D("0.001")) == D("0.016")  # 0,016666... arrondi VERS ZÉRO
    tiny = _signal(position_size_pct=0.5)  # 50 USDT : sous le notionnel minimal de 100
    assert _plan(tiny, target_abs_size=size_for_pct_step(D(10000), D(50000), D("0.5"), BTC.qty_step)) == Skip("below_min_notional")
    below_qty = _signal(position_size_pct=0.5)
    low_qty_spec = InstrumentSpec("BTCUSDT", D("0.001"), D("0.01"), D("1000"), D("0"), D("0.1"))  # pas de notionnel minimal, quantité minimale 0,01
    assert _plan(below_qty, spec=low_qty_spec, target_abs_size=D("0.005")) == Skip("below_min_qty")
    huge = _plan(_signal(), spec=InstrumentSpec("BTCUSDT", D("0.001"), D("0.001"), D("0.005"), D("100"), D("0.1")))
    assert huge == Skip("exceeds_max_order_qty")  # on ne fractionne pas en silence


def test_a_close_is_reduce_only_and_the_same_target_is_a_no_op() -> None:
    flat = _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)
    closing = _plan(flat, position=D("0.010"), target_abs_size=D(0))
    assert isinstance(closing, OrderPlan) and (closing.is_buy, closing.size, closing.reduce_only) == (False, D("0.010"), True)
    assert _plan(position=D("0.010")) == Skip("on_target")


def test_round_price_goes_to_the_next_tick_in_the_requested_direction() -> None:
    assert round_price(D("50250.07"), D("0.1"), up=True) == D("50250.1") and round_price(D("50250.07"), D("0.1"), up=False) == D("50250.0")
    assert round_price(D("50250.1"), D("0.1"), up=True) == D("50250.1")
    with pytest.raises(ValueError):
        round_price(D(0), D("0.1"), up=True)


# --- Worker : faux client --------------------------------------------------------------------


class FakeBinance:
    def __init__(self, equity: str = "10000", positions: Mapping[str, str] | None = None) -> None:
        self.equity = D(equity)
        self.positions = {k: D(v) for k, v in (positions or {}).items()}
        self.specs = {"BTCUSDT": BTC, "ETHUSDT": ETH}
        self.prices = {"BTCUSDT": D(50000), "ETHUSDT": D(3000)}
        self.orders: list[BinanceOrderRequest] = []
        self.place_calls = 0
        self.fail_read = False
        self.hedge = False
        self.fail_place = False
        self.reject = False
        self.status = "filled"

    def instruments(self) -> Mapping[str, InstrumentSpec]:
        return self.specs

    def snapshot(self) -> BinanceSnapshot:
        if self.hedge:
            raise HedgeModeError("position hedge")
        if self.fail_read:
            raise ConnectionError("lecture impossible")
        return BinanceSnapshot(self.equity, {s: BinancePosition(q, self.prices[s]) for s, q in self.positions.items() if q != 0})

    def mid(self, symbol: str) -> Decimal | None:
        return self.prices.get(symbol)

    def place_order(self, request: BinanceOrderRequest) -> BinanceOrderOutcome:
        self.place_calls += 1
        if self.fail_place:
            raise TimeoutError("réponse perdue")
        if self.reject:
            return BinanceOrderOutcome("rejected", None, D(0), "-2019: Margin is insufficient.")
        self.orders.append(request)
        self.positions[request.symbol] = self.positions.get(request.symbol, D(0)) + (request.size if request.is_buy else -request.size)
        return BinanceOrderOutcome(self.status, "ord-9", request.size, "")


def _worker(tmp_path: Path, client: FakeBinance, **overrides: Any) -> BinanceWorker:
    cfg: dict[str, Any] = dict(address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "bn.sqlite3", dry_run=False)
    cfg.update(overrides)
    return BinanceWorker(BinanceWorkerConfig(**cfg), client)


def _run(worker: BinanceWorker, signal: PlatformSignal, asset: str = "BTCUSDT", now_ms: int | None = None) -> str:
    return asyncio.run(worker.execute_signal(signal, asset, now_ms=T0 + 100 if now_ms is None else now_ms))


def test_the_worker_inherits_the_shared_base_and_targets_binance() -> None:
    from workers.common.base import ExposureWorkerBase

    assert issubclass(BinanceWorker, ExposureWorkerBase) and BinanceWorker.venue is TargetVenue.BINANCE and BinanceWorker.supports_dead_man


def test_a_long_signal_places_the_expected_ioc_order_and_journals_it(tmp_path: Path) -> None:
    client = FakeBinance()
    w = _worker(tmp_path, client)
    sig = _signal()
    assert _run(w, sig) == "filled"
    (req,) = client.orders
    assert (req.symbol, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only, req.client_order_id) == (
        "BTCUSDT", True, D("0.010"), D("50250.0"), "IOC", False, make_client_order_id(sig))
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.asset, rec.signal_id) == ("filled", "BUY", "BTCUSDT", str(sig.signal_id))


def test_the_same_target_republished_does_not_stack_positions(tmp_path: Path) -> None:
    client = FakeBinance()
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "filled"
    assert _run(w, _signal()) == "on_target" and len(client.orders) == 1


def test_a_flat_signal_closes_with_a_reduce_only_order(tmp_path: Path) -> None:
    client = FakeBinance(positions={"BTCUSDT": "0.010"})
    w = _worker(tmp_path, client)
    assert _run(w, _signal(direction=SignalDirection.FLAT, position_size_pct=0.0)) == "filled"
    (req,) = client.orders
    assert (req.is_buy, req.size, req.reduce_only) == (False, D("0.010"), True)


def test_dry_run_reads_the_account_but_never_sends(tmp_path: Path) -> None:
    client = FakeBinance()
    w = _worker(tmp_path, client, dry_run=True)
    assert _run(w, _signal()) == "dry_run" and client.place_calls == 0


def test_the_equity_cap_limits_the_sizing_but_not_the_drawdown_measure(tmp_path: Path) -> None:
    client = FakeBinance(equity="100000")
    w = _worker(tmp_path, client, max_equity_usd=D("10000"))
    assert _run(w, _signal()) == "filled" and client.orders[0].size == D("0.010")  # 5 % de 10 000, pas de 100 000


def test_stale_unknown_unreadable_and_unpriced_signals_never_order(tmp_path: Path) -> None:
    client = FakeBinance()
    w = _worker(tmp_path, client)
    assert _run(w, _signal(), now_ms=T0 + 60_000) == "stale_at_execution"
    assert _run(w, _signal(symbol="CRYPTO_DOGE_USD"), asset="DOGEUSDT") == "unknown_symbol"
    client.fail_read = True
    assert _run(w, _signal()) == "read_error"
    client.fail_read = False
    client.prices.pop("BTCUSDT")
    client.prices["BTCUSDT"] = D(50000)
    client.mid = lambda symbol: None  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    assert _run(w, _signal()) == "no_price" and client.place_calls == 0


def test_a_hedge_position_stops_the_order_with_a_clear_status(tmp_path: Path) -> None:
    client = FakeBinance()
    client.hedge = True
    assert _run(_worker(tmp_path, client), _signal()) == "hedge_mode_unsupported" and client.place_calls == 0


def test_a_rejection_is_journaled_as_rejected_and_an_ambiguous_failure_is_an_error_without_retry(tmp_path: Path) -> None:
    rejected = FakeBinance()
    rejected.reject = True
    w = _worker(tmp_path, rejected)
    assert _run(w, _signal()) == "rejected" and rejected.place_calls == 1
    ambiguous = FakeBinance()
    ambiguous.fail_place = True
    other = tmp_path / "other"
    other.mkdir()
    w2 = _worker(other, ambiguous)
    assert _run(w2, _signal()) == "error" and ambiguous.place_calls == 1  # pas de nouvel essai : le signal suivant relit la position


def test_signed_frames_are_accepted_only_for_binance_or_all(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeBinance())
    ok = [DEFAULT_TOPIC, json.dumps(build_envelope(_signal(), SECRET)).encode()]
    assert w.handle_frames(ok, now_ms=T0 + 100) == "queued"
    other = [DEFAULT_TOPIC, json.dumps(build_envelope(_signal(target_venue=TargetVenue.BYBIT), SECRET)).encode()]
    assert w.handle_frames(other, now_ms=T0 + 100) != "queued"
    all_venues = [DEFAULT_TOPIC, json.dumps(build_envelope(_signal(target_venue=TargetVenue.ALL), SECRET)).encode()]
    assert w.handle_frames(all_venues, now_ms=T0 + 100) == "queued"


def test_emergency_flatten_sells_a_long_with_a_reduce_only_ioc_three_percent_below_mid(tmp_path: Path) -> None:
    client = FakeBinance(positions={"BTCUSDT": "0.010"})
    w = _worker(tmp_path, client)
    assert asyncio.run(w.emergency_flatten("BTCUSDT", attempt=1, now_ms=T0)) == "filled"
    (req,) = client.orders
    assert (req.symbol, req.is_buy, req.size, req.limit_px, req.tif, req.reduce_only) == ("BTCUSDT", False, D("0.010"), D("48500.0"), "IOC", True)  # 50 000 x (1 - 3 %)
    assert asyncio.run(w.emergency_flatten("BTCUSDT", attempt=2, now_ms=T0)) == "no_position"  # déjà plat


def test_emergency_flatten_buys_back_a_short_and_widens_with_each_attempt(tmp_path: Path) -> None:
    client = FakeBinance(positions={"BTCUSDT": "-0.010"})
    w = _worker(tmp_path, client)
    asyncio.run(w.emergency_flatten("BTCUSDT", attempt=2, now_ms=T0))
    assert client.orders[0].is_buy is True and client.orders[0].limit_px == D("53000.0")  # 50 000 x (1 + 6 %)


def test_emergency_flatten_failures_are_reported_not_hidden(tmp_path: Path) -> None:
    client = FakeBinance(positions={"BTCUSDT": "0.010"})
    client.fail_read = True
    w = _worker(tmp_path, client)
    assert asyncio.run(w.emergency_flatten("BTCUSDT", attempt=1, now_ms=T0)) == "read_error"
    client.fail_read = False
    client.fail_place = True
    assert asyncio.run(w.emergency_flatten("BTCUSDT", attempt=1, now_ms=T0)) == "error"


# --- Mesure du slippage : intention et flux privé --------------------------------------------


def test_the_worker_records_the_intent_before_sending_and_attaches_the_order_id(tmp_path: Path) -> None:
    db = FillsDatabase()
    client = FakeBinance()
    seen: list[Any] = []
    original = client.place_order

    def spy(request: BinanceOrderRequest) -> BinanceOrderOutcome:
        seen.append(db.find_intent("BINANCE", client_id=request.client_order_id))
        return original(request)

    client.place_order = spy  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    w = _worker(tmp_path, client, fills_db=db)
    assert _run(w, _signal()) == "filled"
    assert seen[0] is not None and seen[0].order_id is None  # écrite AVANT l'envoi
    intent = db.find_intent("BINANCE", client_id=client.orders[0].client_order_id)
    assert intent is not None and intent.order_id == "ord-9" and intent.intended_price == D(50000) and intent.symbol == "BTCUSDT" and intent.kind == "signal"
    dry = FakeBinance()
    dry_db = FillsDatabase()
    other = tmp_path / "dry"
    other.mkdir()
    _run(_worker(other, dry, fills_db=dry_db, dry_run=True), _signal())
    with dry_db._lock:  # noqa: SLF001
        assert dry_db._db.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == 0


ORDER_UPDATE: dict[str, Any] = {"e": "ORDER_TRADE_UPDATE", "E": T0, "T": T0, "o": {
    "s": "BTCUSDT", "c": "nabc", "S": "BUY", "o": "LIMIT", "f": "IOC", "q": "0.010", "p": "50250", "ap": "50010.5", "sp": "0",
    "x": "TRADE", "X": "FILLED", "i": 8886774, "l": "0.010", "z": "0.010", "L": "50010.5", "N": "USDT", "n": "0.2500525", "T": T0, "t": 555,
    "b": "0", "a": "0", "m": False, "R": False, "wt": "CONTRACT_PRICE", "ot": "LIMIT", "ps": "BOTH", "cp": False, "rp": "0", "pP": False, "si": 0, "ss": 0}}


class FakeUserStream:
    def __init__(self) -> None:
        self.started = 0
        self.kept = 0
        self.fail_keepalive = False

    def start_user_stream(self) -> str:
        self.started += 1
        return f"LK{self.started}"

    def keepalive_user_stream(self) -> None:
        self.kept += 1
        if self.fail_keepalive:
            raise BinanceApiError(-1125, "This listenKey does not exist.")


def _update(**order: Any) -> str:
    return json.dumps({**ORDER_UPDATE, "o": {**ORDER_UPDATE["o"], **order}})


def test_a_trade_event_becomes_a_fill_matched_by_client_order_id_and_the_account_events_are_ignored() -> None:
    feed = BinanceFillFeed(api=FakeUserStream())
    (event,) = feed.parse(_update())
    assert (event.venue, event.symbol, event.is_buy, event.price, event.qty, event.time_ms) == ("BINANCE", "BTCUSDT", True, D("50010.5"), D("0.010"), T0)
    assert (event.fill_id, event.order_id, event.client_id, event.fee, event.fee_currency) == ("BTCUSDT:555", "8886774", "nabc", D("0.2500525"), "USDT")
    assert feed.parse(_update(x="NEW", X="NEW", l="0")) == [] and feed.parse(_update(x="CANCELED", X="CANCELED")) == []
    assert feed.parse(_update(l="0")) == []  # un événement TRADE sans quantité exécutée n'est pas un fill
    assert feed.parse(json.dumps({"e": "ACCOUNT_UPDATE", "a": {}})) == [] and feed.parse(json.dumps({"e": "listenKeyExpired"})) == []
    (sell,) = feed.parse(_update(S="SELL", c="", t=556))
    assert sell.is_buy is False and sell.client_id is None  # ordre sans identifiant client (passé à la main) : apparié par i seulement
    with pytest.raises(ValueError):
        feed.parse(_update(S="???"))
    with pytest.raises(ValueError):
        feed.parse(_update(L="pas un nombre"))


def test_partial_fills_are_each_a_fill_because_only_final_events_would_lose_the_partial_executions() -> None:
    feed = BinanceFillFeed(api=FakeUserStream())
    first = feed.parse(_update(X="PARTIALLY_FILLED", l="0.004", z="0.004", L="50000", t=1))
    second = feed.parse(_update(X="FILLED", l="0.006", z="0.010", L="50020", t=2))
    assert [(e.qty, e.price, e.fill_id) for e in first + second] == [(D("0.004"), D("50000"), "BTCUSDT:1"), (D("0.006"), D("50020"), "BTCUSDT:2")]


def test_the_feed_url_uses_the_documented_private_scheme_with_a_fresh_key_and_hides_it_in_repr() -> None:
    api = FakeUserStream()
    feed = BinanceFillFeed(api=api, ws_base_url="wss://fstream.binance.com/")
    assert feed.url == "wss://fstream.binance.com/private/ws"
    assert asyncio.run(feed.prepare()) == "wss://fstream.binance.com/private/ws?listenKey=LK1&events=ORDER_TRADE_UPDATE"
    assert asyncio.run(feed.prepare()).endswith("listenKey=LK2&events=ORDER_TRADE_UPDATE") and "LK" not in repr(feed)
    asyncio.run(feed.keepalive())
    assert api.kept == 1 and feed.heartbeat_message() is None


def _tracker_harness(feed: Any, db: FillsDatabase) -> Any:
    from tests.test_fill_tracking import Harness

    return Harness(feed, db)


def test_the_tracker_prepares_a_fresh_key_for_every_connection_and_records_a_matched_fill_with_its_slippage() -> None:
    from tests.test_fill_tracking import _intent

    api = FakeUserStream()
    db = FillsDatabase()
    db.register_intent(_intent("nabc", venue="BINANCE", price="50000"))
    h = _tracker_harness(BinanceFillFeed(api=api), db)
    urls: list[str] = []
    original = h._connect

    async def spy(url: str) -> Any:
        urls.append(url)
        return await original(url)

    h.tracker._connect = spy
    h.scripts = [[_update()]]
    asyncio.run(h.run_until(lambda: len(db.executions()) == 1))
    (row,) = db.executions()
    assert urls and urls[0].endswith("listenKey=LK1&events=ORDER_TRADE_UPDATE")
    assert (row.venue, row.symbol, row.direction, row.kind, row.intended_price, row.fill_price) == ("BINANCE", "BTCUSDT", "BUY", "signal", 50000.0, 50010.5)
    assert row.slippage_bps == pytest.approx(2.1)  # (50010,5 - 50000) / 50000 x 10 000
    assert (row.fee, row.fee_currency, row.order_id, row.fill_id) == (0.2500525, "USDT", "8886774", "BTCUSDT:555")


def test_partial_fills_of_one_order_are_all_recorded_and_replays_are_not_double_counted() -> None:
    from tests.test_fill_tracking import _intent

    db = FillsDatabase()
    db.register_intent(_intent("nabc", venue="BINANCE", price="50000"))
    h = _tracker_harness(BinanceFillFeed(api=FakeUserStream()), db)
    h.scripts = [[_update(X="PARTIALLY_FILLED", l="0.004", L="50000", t=1), _update(X="FILLED", l="0.006", L="50020", t=2), _update(X="FILLED", l="0.006", L="50020", t=2)]]
    asyncio.run(h.run_until(lambda: len(db.executions()) == 2 and h.tracker.stats["duplicates"] == 1))
    assert [(e.fill_qty, e.fill_price) for e in db.executions()] == [(0.004, 50000.0), (0.006, 50020.0)]


def test_a_failing_keepalive_forces_a_reconnection_with_a_new_key() -> None:
    from tests.test_fill_tracking import _intent

    class FastKeepalive(BinanceFillFeed):
        keepalive_interval_s = 0.05

    api = FakeUserStream()
    api.fail_keepalive = True
    db = FillsDatabase()
    db.register_intent(_intent("nabc", venue="BINANCE", price="50000"))
    h = _tracker_harness(FastKeepalive(api=api), db)
    h.scripts = [[], [_update()]]  # 1re connexion : vide, jusqu'à l'échec du keepalive ; 2e : le fill
    asyncio.run(h.run_until(lambda: len(db.executions()) == 1))
    assert api.kept >= 1 and api.started >= 2 and h.tracker.stats["reconnects"] >= 1  # clé neuve à la reconnexion


def test_a_working_keepalive_extends_the_key_without_reconnecting() -> None:
    class FastKeepalive(BinanceFillFeed):
        keepalive_interval_s = 0.05

    api = FakeUserStream()
    h = _tracker_harness(FastKeepalive(api=api), FillsDatabase())
    h.scripts = [[]]
    asyncio.run(h.run_until(lambda: api.kept >= 2))
    assert api.kept >= 2 and api.started == 1 and h.tracker.stats["reconnects"] == 0


# --- Ligne de commande -----------------------------------------------------------------------


class _CliClient:
    def __init__(self, *, permissions: KeyPermissions, hedge: bool = False, boom: Exception | None = None, **_: Any) -> None:
        self._permissions, self._hedge, self._boom = permissions, hedge, boom

    def key_permissions(self) -> KeyPermissions:
        if self._boom:
            raise self._boom
        return self._permissions

    def is_hedge_mode(self) -> bool:
        return self._hedge

    def start_user_stream(self) -> str:
        return "LK"

    def keepalive_user_stream(self) -> None:
        return None


def _cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str, permissions: KeyPermissions | None = None, hedge: bool = False, boom: Exception | None = None) -> tuple[int | None, dict[str, Any]]:
    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def worker_factory(config: Any, client: Any) -> None:
        seen["config"] = config
        raise _Stop

    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", SECRET)
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    perms = permissions or KeyPermissions(True, can_withdraw=False, can_read=True, can_trade_futures=True)
    monkeypatch.setattr(worker_module, "BinanceRestClient", lambda **kw: _CliClient(permissions=perms, hedge=hedge, boom=boom, **kw))
    monkeypatch.setattr(worker_module, "BinanceWorker", worker_factory)
    monkeypatch.setattr(worker_module, "configure_logging", lambda *a, **k: None)
    try:
        code = worker_module.main(["--address", "tcp://127.0.0.1:1", "--db", str(tmp_path / "bn.sqlite3"), *args])
    except _Stop:
        return None, seen
    return code, seen


def test_cli_refuses_a_key_that_can_withdraw_and_never_starts_the_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    code, seen = _cli(monkeypatch, tmp_path, "--network", "mainnet", permissions=KeyPermissions(True, can_withdraw=True, can_read=True, can_trade_futures=True))
    assert code == 2 and "config" not in seen and "retrait" in caplog.text


def test_cli_refuses_an_account_in_hedge_mode_with_a_clean_exit_and_no_traceback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    code, seen = _cli(monkeypatch, tmp_path, hedge=True)
    assert code == 2 and "config" not in seen and "hedge" in caplog.text and "Traceback" not in caplog.text


def test_cli_refuses_a_mainnet_key_whose_rights_cannot_be_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, seen = _cli(monkeypatch, tmp_path, "--network", "mainnet", permissions=KeyPermissions(False, reason="BinanceApiError"))
    assert code == 2 and "config" not in seen


def test_cli_stops_cleanly_when_the_account_cannot_be_queried_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    code, seen = _cli(monkeypatch, tmp_path, boom=BinanceApiError(-2015, "Invalid API-key"))
    assert code == 2 and "config" not in seen and "BinanceApiError" in caplog.text


def test_cli_defaults_to_dry_run_on_the_test_network_without_fill_tracking(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, seen = _cli(monkeypatch, tmp_path)
    cfg = seen["config"]
    assert code is None and cfg.dry_run is True and cfg.fill_tracker is None and cfg.fills_db is None and cfg.dead_man_timeout_s == 300.0


def test_cli_tracks_fills_only_when_live_and_puts_the_fills_database_next_to_the_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _, seen = _cli(monkeypatch, tmp_path, "--live")
    cfg = seen["config"]
    assert cfg.dry_run is False and isinstance(cfg.fill_tracker, FillTracker) and cfg.fills_db is not None and (tmp_path / "binance_fills.sqlite3").exists()
    _, off = _cli(monkeypatch, tmp_path, "--live", "--no-fill-tracking")
    assert off["config"].fill_tracker is None and off["config"].fills_db is None


def test_cli_requires_the_explicit_acknowledgement_and_the_dead_man_for_mainnet_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--live", "--network", "mainnet")
    with pytest.raises(SystemExit):
        _cli(monkeypatch, tmp_path, "--live", "--network", "mainnet", "--i-understand-real-funds", "--dead-man-timeout", "0")
    ok_code, ok = _cli(monkeypatch, tmp_path, "--live", "--network", "mainnet", "--i-understand-real-funds")
    assert ok_code is None and ok["config"].dry_run is False


def test_cli_needs_the_api_credentials_and_clear_halt_needs_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", SECRET)
    monkeypatch.setattr(worker_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(worker_module, "BinanceRestClient", lambda **_: pytest.fail("aucun client pour --clear-halt ni sans clés"))
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--db", str(tmp_path / "x.sqlite3")])
    assert worker_module.main(["--clear-halt", "--db", str(tmp_path / "x.sqlite3")]) == 0

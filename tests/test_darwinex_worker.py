"""Tests du worker Darwinex Zero (``workers/darwinex``). Aucun terminal MT5, aucun compte, aucun réseau.

Chiffres calculés à la main : EURUSD, bid 1,0999 / ask 1,1001 (mid 1,1000), 1 lot = 110 000 USD, equity 100 000 USD,
cible 5 % -> 5 000 USD -> 0,0454 lot -> 0,04 lot au pas de 0,01. Le client est testé avec un faux module ``MetaTrader5``
dont les constantes sont volontairement des nombres arbitraires : le code doit les utiliser par leur NOM.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, build_envelope
from workers.common.exposure import RiskLimits, Skip
from workers.common.state import ExecutionState
from workers.darwinex import worker as worker_module
from workers.darwinex.client import (
    AccountInfo,
    MetaTrader5Client,
    Mt5Error,
    Mt5OrderOutcome,
    Mt5OrderRequest,
    Mt5Snapshot,
    MultiplePositionsError,
    RealAccountRefused,
    WrongAccountError,
    interpret_result,
)
from workers.darwinex.planner import (
    Mt5OrderPlan,
    Mt5Position,
    Quote,
    SymbolSpec,
    choose_filling,
    deviation_points,
    lots_for_pct,
    make_comment,
    mt5_symbol,
    plan_order,
)
from workers.darwinex.worker import DarwinexWorker, DarwinexWorkerConfig

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
T0 = 1_760_000_000_000
D = Decimal
LOGIN = 5001234

EURUSD = SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100"), D("0.01"), 10, "FULL", "INSTANT", 3)
QUOTE = Quote(D("1.0999"), D("1.1001"))
LOT_VALUE = D("110000")


def _signal(**overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.DARWINEX_ZERO, asset_class=AssetClass.FOREX, symbol="FOREX_EUR_USD",
        direction=SignalDirection.LONG, position_size_pct=5.0, order_type=OrderType.MARKET,
        metadata={"stop_loss_price": 1.095}, timestamp_ms=T0,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _short(**overrides: Any) -> PlatformSignal:
    fields: dict[str, Any] = {"direction": SignalDirection.SHORT, "metadata": {"stop_loss_price": 1.105}}
    fields.update(overrides)
    return _signal(**fields)


def _flat(**overrides: Any) -> PlatformSignal:
    return _signal(direction=SignalDirection.FLAT, position_size_pct=0.0, metadata={}, **overrides)


# --- Planificateur ---------------------------------------------------------------------------


def _plan(signal: PlatformSignal | None = None, **overrides: Any) -> Mt5OrderPlan | Skip:
    sig = signal or _signal()
    kw: dict[str, Any] = dict(
        symbol="EURUSD", target_abs_lots=lots_for_pct(D(100000), LOT_VALUE, D(str(sig.position_size_pct)), EURUSD), equity=D(100000),
        position=None, quote=QUOTE, spec=EURUSD, notional_per_lot=LOT_VALUE, others_notional=D(0), limits=RiskLimits(),
        halted=False, max_spread_bps=D(20),
    )
    kw.update(overrides)
    return plan_order(sig, **kw)


def test_symbol_mapping() -> None:
    assert mt5_symbol("FOREX_EUR_USD") == "EURUSD" and mt5_symbol("FOREX_XAU_USD") == "XAUUSD"
    assert mt5_symbol("FOREX_EUR_USD", ".r") == "EURUSD.r"
    assert mt5_symbol("FOREX_EUR") is None and mt5_symbol("CRYPTO_BTC_USD") is None and mt5_symbol("FOREX_E_USD") is None
    assert mt5_symbol("FOREX_US30_USD", overrides={"FOREX_US30_USD": "US30.cash"}) == "US30.cash"


def test_lots_are_floored_to_the_volume_step() -> None:
    assert lots_for_pct(D(100000), LOT_VALUE, D(5), EURUSD) == D("0.04")  # 5 000 / 110 000 = 0,04545
    assert lots_for_pct(D(10000), LOT_VALUE, D(5), EURUSD) == 0  # 500 USD < un lot minimal
    assert lots_for_pct(D(0), LOT_VALUE, D(5), EURUSD) == 0 and lots_for_pct(D(100000), D(0), D(5), EURUSD) == 0


def test_a_long_opens_at_the_ask_with_a_server_side_stop_and_a_slippage_cap() -> None:
    sig = _signal()
    plan = _plan(sig)
    assert isinstance(plan, Mt5OrderPlan)
    assert (plan.symbol, plan.is_buy, plan.volume, plan.price, plan.sl, plan.filling) == ("EURUSD", True, D("0.04"), D("1.10010"), D("1.09500"), "IOC")
    assert plan.deviation_points == 551  # 1,1001 x 0,5 % = 0,0055005 = 550,05 points -> 551
    assert (plan.reduce_only, plan.position_ticket) == (False, None)
    assert plan.comment == make_comment(sig.signal_id) and len(plan.comment) == 17


def test_a_short_opens_at_the_bid() -> None:
    plan = _plan(_short())
    assert isinstance(plan, Mt5OrderPlan) and (plan.is_buy, plan.volume, plan.price, plan.sl) == (False, D("0.04"), D("1.09990"), D("1.10500"))


def test_the_stop_must_respect_the_brokers_minimum_distance_and_the_right_side() -> None:
    # stops_level 10 points = 0,0001 : un achat veut sl <= bid - 0,0001 = 1,0998
    assert isinstance(_plan(_signal(metadata={"stop_loss_price": 1.0998})), Mt5OrderPlan)
    assert _plan(_signal(metadata={"stop_loss_price": 1.09981})) == Skip("stop_too_close_or_wrong_side")
    assert _plan(_signal(metadata={"stop_loss_price": 1.2})) == Skip("stop_too_close_or_wrong_side")  # du mauvais côté
    # une vente veut sl >= ask + 0,0001 = 1,1002
    assert isinstance(_plan(_short(metadata={"stop_loss_price": 1.1002})), Mt5OrderPlan)
    assert _plan(_signal(direction=SignalDirection.SHORT, metadata={"stop_loss_price": 1.10019})) == Skip("stop_too_close_or_wrong_side")


def test_an_account_too_small_for_one_minimal_lot_says_so_instead_of_pretending_to_be_on_target() -> None:
    small = _plan(equity=D(10000), target_abs_lots=lots_for_pct(D(10000), LOT_VALUE, D(5), EURUSD))
    assert small == Skip("below_min_volume")


def test_same_target_is_a_no_op_and_flat_closes_with_the_position_ticket() -> None:
    assert _plan(position=Mt5Position(77, D("0.04"), D("1.1"))) == Skip("on_target")
    close = _plan(_flat(), position=Mt5Position(77, D("0.04"), D("1.1")), target_abs_lots=D(0))
    assert isinstance(close, Mt5OrderPlan)
    assert (close.is_buy, close.volume, close.reduce_only, close.position_ticket, close.sl, close.price) == (False, D("0.04"), True, 77, None, D("1.09990"))


def test_a_reversal_only_closes_so_a_hedging_account_never_ends_up_with_both_sides() -> None:
    plan = _plan(_short(), position=Mt5Position(77, D("0.04"), D("1.1")))
    assert isinstance(plan, Mt5OrderPlan)
    assert (plan.is_buy, plan.volume, plan.reduce_only, plan.position_ticket, plan.sl) == (False, D("0.04"), True, 77, None)


def test_a_partial_reduction_closes_the_excess_with_the_ticket() -> None:
    plan = _plan(position=Mt5Position(77, D("0.08"), D("1.1")))
    assert isinstance(plan, Mt5OrderPlan)
    assert (plan.is_buy, plan.volume, plan.reduce_only, plan.position_ticket) == (False, D("0.04"), True, 77)


def test_trade_modes_block_opens_but_never_closes() -> None:
    def spec(mode: str) -> SymbolSpec:
        return SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100"), D("0.01"), 10, mode, "INSTANT", 3)

    assert _plan(spec=spec("CLOSEONLY")) == Skip("close_only")
    assert _plan(_short(), spec=spec("LONGONLY")) == Skip("long_only")
    assert _plan(spec=spec("SHORTONLY")) == Skip("short_only")
    assert _plan(spec=spec("DISABLED")) == Skip("symbol_trading_disabled")
    close = _plan(_flat(), spec=spec("CLOSEONLY"), position=Mt5Position(5, D("0.04"), D("1.1")), target_abs_lots=D(0))
    assert isinstance(close, Mt5OrderPlan) and close.reduce_only


def test_a_wide_spread_blocks_an_open_but_not_a_close() -> None:
    wide = Quote(D("1.0999"), D("1.1100"))  # ~100 points de base
    assert _plan(quote=wide) == Skip("spread_too_wide")
    close = _plan(_flat(), quote=wide, position=Mt5Position(5, D("0.04"), D("1.1")), target_abs_lots=D(0))
    assert isinstance(close, Mt5OrderPlan)


def test_volume_limits() -> None:
    capped = SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("0.03"), D("0.01"), 10, "FULL", "INSTANT", 3)
    assert _plan(spec=capped) == Skip("exceeds_max_volume")  # on ne fractionne pas une ouverture
    close = _plan(_flat(), spec=capped, position=Mt5Position(5, D("0.10"), D("1.1")), target_abs_lots=D(0))
    assert isinstance(close, Mt5OrderPlan) and close.volume == D("0.03")  # une fermeture trop grosse se fait en plusieurs tours


@pytest.mark.parametrize(
    ("execution", "flags", "expected"),
    [("INSTANT", 0, "IOC"), ("REQUEST", 0, "IOC"), ("MARKET", 2, "IOC"), ("MARKET", 1, "FOK"), ("MARKET", 3, "IOC"),
     ("MARKET", 0, None), ("EXCHANGE", 0, "RETURN"), ("EXCHANGE", 1, "FOK")],
)
def test_filling_policy_follows_the_documented_table(execution: str, flags: int, expected: str | None) -> None:
    spec = SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100"), D("0.01"), 10, "FULL", execution, flags)
    assert choose_filling(spec) == expected
    if expected is None:
        assert _plan(spec=spec) == Skip("no_valid_filling")


def test_limit_orders_round_to_the_less_aggressive_side_and_refuse_fat_fingers() -> None:
    buy = _plan(_signal(order_type=OrderType.LIMIT, metadata={"stop_loss_price": 1.095, "limit_price": 1.09996}))
    assert isinstance(buy, Mt5OrderPlan) and (buy.price, buy.deviation_points) == (D("1.09996"), 0)
    assert _plan(_signal(order_type=OrderType.LIMIT, metadata={"stop_loss_price": 1.095, "limit_price": 1.2})) == Skip("limit_far_from_mid")


def test_deviation_points_and_comment() -> None:
    assert deviation_points(D("1.1001"), D(50), EURUSD) == 551 and deviation_points(D("1.0999"), D(300), EURUSD) == 3300
    assert make_comment("x") == make_comment("x") != make_comment("y") and len(make_comment("x")) == 17


def test_shared_risk_rules_apply_here_too() -> None:
    assert _plan(_signal(position_size_pct=25.0)) == Skip("exceeds_position_cap")
    assert _plan(halted=True) == Skip("halted_reduce_only")
    assert _plan(_signal(order_type=OrderType.PREDICTION_ONLY, position_size_pct=0.0, metadata={})) == Skip("prediction_only")


# --- Client : faux module MetaTrader5 ---------------------------------------------------------


class FakeMt5:
    """Faux module ``MetaTrader5`` : constantes arbitraires (le code doit les utiliser par leur nom)."""

    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 100, 101
    TRADE_ACTION_DEAL, ORDER_TIME_GTC = 200, 201
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 10, 11, 12
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 300, 301
    ACCOUNT_TRADE_MODE_DEMO, ACCOUNT_TRADE_MODE_CONTEST, ACCOUNT_TRADE_MODE_REAL = 20, 21, 22
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING, ACCOUNT_MARGIN_MODE_EXCHANGE, ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 30, 31, 32
    SYMBOL_TRADE_MODE_DISABLED, SYMBOL_TRADE_MODE_LONGONLY, SYMBOL_TRADE_MODE_SHORTONLY = 40, 41, 42
    SYMBOL_TRADE_MODE_CLOSEONLY, SYMBOL_TRADE_MODE_FULL = 43, 44
    SYMBOL_TRADE_EXECUTION_REQUEST, SYMBOL_TRADE_EXECUTION_INSTANT = 50, 51
    SYMBOL_TRADE_EXECUTION_MARKET, SYMBOL_TRADE_EXECUTION_EXCHANGE = 52, 53

    def __init__(self) -> None:
        self.connected = False
        self.initialize_ok = True
        self.init_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.login = LOGIN
        self.account_mode = self.ACCOUNT_TRADE_MODE_DEMO
        self.positions: list[Any] | None = []
        self.last_error_value: tuple[int, str] = (1, "Success")
        self.sent: list[dict[str, Any]] = []
        self.order_result: Any = SimpleNamespace(retcode=10009, comment="Request executed")
        self.contract_size = 100000.0
        self.conversion = 0.9  # devise de cotation -> devise du compte
        self.tick: Any = SimpleNamespace(bid=1.0999, ask=1.1001)
        self.active = 0
        self.max_active = 0
        self.slow = 0.0
        self.selected: list[str] = []
        self.shutdowns = 0

    def _enter(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.slow:
            time.sleep(self.slow)
        self.active -= 1

    def terminal_info(self) -> Any:
        return SimpleNamespace(connected=True) if self.connected else None

    def initialize(self, *args: Any, **kwargs: Any) -> bool:
        self.init_calls.append((args, kwargs))
        self.connected = self.initialize_ok
        return self.initialize_ok

    def shutdown(self) -> None:
        self.shutdowns += 1

    def last_error(self) -> tuple[int, str]:
        return self.last_error_value

    def account_info(self) -> Any:
        return SimpleNamespace(login=self.login, trade_mode=self.account_mode, margin_mode=self.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING,
                               currency="USD", server="Darwinex-Demo", trade_allowed=True, equity=100000.0)

    def symbol_select(self, symbol: str, enable: bool) -> bool:
        self.selected.append(symbol)
        return True

    def symbol_info(self, symbol: str) -> Any:
        if symbol != "EURUSD":
            return None
        return SimpleNamespace(digits=5, point=0.00001, volume_min=0.01, volume_max=100.0, volume_step=0.01, trade_stops_level=10,
                               trade_mode=self.SYMBOL_TRADE_MODE_FULL, trade_exemode=self.SYMBOL_TRADE_EXECUTION_INSTANT, filling_mode=3)

    def symbol_info_tick(self, symbol: str) -> Any:
        self._enter()
        return self.tick

    def positions_get(self) -> Any:
        return self.positions

    def order_calc_profit(self, action: int, symbol: str, volume: float, price_open: float, price_close: float) -> float | None:
        return volume * self.contract_size * (price_close - price_open) * self.conversion

    def order_send(self, request: dict[str, Any]) -> Any:
        self.sent.append(request)
        return self.order_result


def _position(symbol: str = "EURUSD", ticket: int = 77, volume: float = 0.04, kind: int = FakeMt5.POSITION_TYPE_BUY, price: float = 1.1) -> Any:
    return SimpleNamespace(symbol=symbol, ticket=ticket, volume=volume, type=kind, price_current=price)


def _client(mt5: FakeMt5, **overrides: Any) -> MetaTrader5Client:
    kw: dict[str, Any] = dict(expected_login=LOGIN, magic=424242, mt5_module=mt5)
    kw.update(overrides)
    return MetaTrader5Client(**kw)


def test_account_check_pins_the_login_and_connects_only_when_needed() -> None:
    mt5 = FakeMt5()
    client = _client(mt5, path="C:/mt5/terminal64.exe", login=LOGIN, password="pwd-secret", server="Darwinex-Demo")
    info = client.account_check()
    assert info == AccountInfo(LOGIN, "DEMO", "HEDGING", "USD", "Darwinex-Demo", True)
    assert mt5.init_calls == [(("C:/mt5/terminal64.exe",), {"timeout": 60000, "login": LOGIN, "server": "Darwinex-Demo", "password": "pwd-secret"})]
    client.account_check()
    assert len(mt5.init_calls) == 1  # déjà connecté : pas de réinitialisation


def test_a_terminal_that_cannot_start_is_a_clear_error() -> None:
    mt5 = FakeMt5()
    mt5.initialize_ok = False
    mt5.last_error_value = (-10003, "IPC initialize failed")
    with pytest.raises(Mt5Error, match="IPC initialize failed"):
        _client(mt5).account_check()


def test_the_wrong_account_is_refused_on_every_read_and_before_every_order() -> None:
    mt5 = FakeMt5()
    client = _client(mt5)
    client.account_check()
    mt5.login = 9999999  # quelqu'un se connecte à un autre compte dans le terminal
    with pytest.raises(WrongAccountError, match="9999999"):
        client.snapshot()
    req = Mt5OrderRequest("EURUSD", True, D("0.04"), D("1.1001"), 56, D("1.095"), "IOC", None, "c")
    with pytest.raises(WrongAccountError):
        client.send(req)
    assert mt5.sent == []  # aucun ordre n'est parti vers le mauvais compte


def test_a_real_account_is_refused_unless_explicitly_allowed() -> None:
    mt5 = FakeMt5()
    mt5.account_mode = mt5.ACCOUNT_TRADE_MODE_REAL
    with pytest.raises(RealAccountRefused):
        _client(mt5).account_check()
    assert _client(mt5, allow_real_account=True).account_check().trade_mode == "REAL"


def test_symbol_spec_and_quote_are_decoded() -> None:
    mt5 = FakeMt5()
    client = _client(mt5)
    spec = client.symbol_spec("EURUSD")
    assert spec == SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100.0"), D("0.01"), 10, "FULL", "INSTANT", 3)
    assert mt5.selected == ["EURUSD"] and client.symbol_spec("NOPE") is None
    assert client.quote("EURUSD") == Quote(D("1.0999"), D("1.1001"))
    mt5.tick = SimpleNamespace(bid=0.0, ask=1.1)
    assert client.quote("EURUSD") is None
    mt5.tick = None
    assert client.quote("EURUSD") is None


def test_snapshot_returns_signed_positions_and_the_equity() -> None:
    mt5 = FakeMt5()
    mt5.positions = [_position(), _position("GBPUSD", 78, 0.10, FakeMt5.POSITION_TYPE_SELL, 1.3)]
    snap = _client(mt5).snapshot()
    assert snap == Mt5Snapshot(D("100000.0"), {"EURUSD": Mt5Position(77, D("0.04"), D("1.1")), "GBPUSD": Mt5Position(78, D("-0.10"), D("1.3"))})


def test_two_positions_on_one_symbol_are_refused_and_an_empty_account_is_understood() -> None:
    mt5 = FakeMt5()
    mt5.positions = [_position(ticket=1), _position(ticket=2, kind=FakeMt5.POSITION_TYPE_SELL)]
    with pytest.raises(MultiplePositionsError):
        _client(mt5).snapshot()
    mt5.positions = None  # None + « Success » = aucune position ; None + vraie erreur = échec de lecture
    assert _client(mt5).snapshot().positions == {}
    mt5.last_error_value = (-1, "Terminal: Call failed")
    with pytest.raises(Mt5Error, match="positions_get"):
        _client(mt5).snapshot()


def test_the_value_of_a_lot_is_measured_in_the_account_currency() -> None:
    mt5 = FakeMt5()
    client = _client(mt5)
    value = client.notional_per_lot("EURUSD", D("1.1"))
    assert value is not None and abs(value - D("100000") * D("1.1") * D("0.9")) < D("0.0001")  # contrat x prix x conversion
    mt5.contract_size = 0.0  # order_calc_profit incohérent : indéterminable plutôt qu'inventé
    assert client.notional_per_lot("EURUSD", D("1.1")) is None


def test_open_and_close_requests_use_the_constants_by_name() -> None:
    mt5 = FakeMt5()
    client = _client(mt5)
    open_req = Mt5OrderRequest("EURUSD", True, D("0.04"), D("1.1001"), 56, D("1.095"), "IOC", None, "nabc")
    assert client.send(open_req).status == "filled"
    close_req = Mt5OrderRequest("EURUSD", False, D("0.04"), None, 3300, None, "FOK", 77, "ndef")
    client.send(close_req)
    opened, closed = mt5.sent
    assert opened == {"action": 200, "symbol": "EURUSD", "volume": 0.04, "type": 100, "deviation": 56, "magic": 424242,
                      "comment": "nabc", "type_time": 201, "type_filling": 11, "price": 1.1001, "sl": 1.095}
    assert closed == {"action": 200, "symbol": "EURUSD", "volume": 0.04, "type": 101, "deviation": 3300, "magic": 424242,
                      "comment": "ndef", "type_time": 201, "type_filling": 10, "position": 77}  # ni prix (exécution Market) ni sl


@pytest.mark.parametrize(
    ("retcode", "status"),
    [(10009, "filled"), (10010, "partial"), (10008, "placed"), (10004, "rejected"), (10006, "rejected"), (10018, "rejected"),
     (10019, "rejected"), (10016, "rejected"), (10012, "error"), (10031, "error"), (99999, "rejected")],
)
def test_return_codes_follow_the_official_table(retcode: int, status: str) -> None:
    outcome = interpret_result(SimpleNamespace(retcode=retcode, comment="c"))
    assert (outcome.status, outcome.retcode) == (status, retcode)


def test_an_unknown_order_result_is_ambiguous_and_the_detail_names_the_code() -> None:
    assert interpret_result(None).status == "error"
    assert "10018 MARKET_CLOSED" in interpret_result(SimpleNamespace(retcode=10018, comment="Market closed")).detail


def test_every_terminal_call_is_serialized_by_one_lock() -> None:
    mt5 = FakeMt5()
    mt5.slow = 0.02
    client = _client(mt5)
    threads = [threading.Thread(target=client.quote, args=("EURUSD",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mt5.max_active == 1  # l'API MT5 n'est jamais appelée en parallèle


def test_the_client_never_exposes_credentials() -> None:
    client = _client(FakeMt5(), login=LOGIN, password="pwd-secret", server="Darwinex-Demo")
    assert "pwd-secret" not in repr(client) and "Darwinex" not in repr(client)
    client.close()


# --- Worker : faux client --------------------------------------------------------------------


class FakeDwx:
    def __init__(self, equity: str = "100000") -> None:
        self.equity = D(equity)
        self.positions: dict[str, Mt5Position] = {}
        self.spec = EURUSD
        self.quote_value: Quote | None = QUOTE
        self.lot_value: Decimal | None = LOT_VALUE
        self.sent: list[Mt5OrderRequest] = []
        self.next_ticket = 100
        self.raise_on_read: Exception | None = None
        self.raise_on_send: Exception | None = None
        self.outcome: Mt5OrderOutcome | None = None
        self.unknown_value_for: set[str] = set()

    def account_check(self) -> AccountInfo:
        return AccountInfo(LOGIN, "DEMO", "HEDGING", "USD", "Darwinex-Demo", True)

    def symbol_spec(self, symbol: str) -> SymbolSpec | None:
        return self.spec if symbol == "EURUSD" else None

    def quote(self, symbol: str) -> Quote | None:
        return self.quote_value

    def snapshot(self) -> Mt5Snapshot:
        if self.raise_on_read is not None:
            raise self.raise_on_read
        return Mt5Snapshot(self.equity, dict(self.positions))

    def notional_per_lot(self, symbol: str, price: Decimal) -> Decimal | None:
        return None if symbol in self.unknown_value_for else self.lot_value

    def send(self, request: Mt5OrderRequest) -> Mt5OrderOutcome:
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append(request)
        if self.outcome is not None:
            return self.outcome
        current = self.positions.get(request.symbol)
        signed = request.volume if request.is_buy else -request.volume
        if request.position_ticket is not None and current is not None:  # fermeture par ticket
            remaining = current.volume + signed
            if remaining == 0:
                del self.positions[request.symbol]
            else:
                self.positions[request.symbol] = Mt5Position(current.ticket, remaining, current.price_current)
        elif current is None:
            self.next_ticket += 1
            self.positions[request.symbol] = Mt5Position(self.next_ticket, signed, request.price or D("1.1"))
        else:
            self.positions[request.symbol] = Mt5Position(current.ticket, current.volume + signed, current.price_current)
        return Mt5OrderOutcome("filled", 10009, "10009 DONE: Request executed")


def _worker(tmp_path: Path, client: FakeDwx, **overrides: Any) -> DarwinexWorker:
    cfg: dict[str, Any] = dict(address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "dwx.sqlite3", dry_run=False)
    cfg.update(overrides)
    return DarwinexWorker(DarwinexWorkerConfig(**cfg), client)


def _run(worker: DarwinexWorker, signal: PlatformSignal, asset: str = "EURUSD", now_ms: int | None = None) -> str:
    return asyncio.run(worker.execute_signal(signal, asset, now_ms=T0 + 100 if now_ms is None else now_ms))


def _frames(signal: PlatformSignal) -> list[bytes]:
    return [DEFAULT_TOPIC, json.dumps(build_envelope(signal, SECRET)).encode("utf-8")]


def test_a_long_signal_opens_with_a_stop_loss_and_is_journaled(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    sig = _signal()
    assert _run(w, sig) == "filled"
    (req,) = client.sent
    assert (req.symbol, req.is_buy, req.volume, req.price, req.deviation, req.sl, req.filling, req.position_ticket, req.comment) == (
        "EURUSD", True, D("0.04"), D("1.10010"), 551, D("1.09500"), "IOC", None, make_comment(sig.signal_id))
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.size, rec.px) == ("filled", "BUY", "0.04", "1.10010")


def test_the_same_target_republished_never_stacks_lots(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "filled"
    assert [_run(w, _signal()) for _ in range(5)] == ["on_target"] * 5
    assert len(client.sent) == 1 and client.positions["EURUSD"].volume == D("0.04")


def test_flat_closes_with_the_position_ticket(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    w = _worker(tmp_path, client)
    assert _run(w, _flat()) == "filled"
    (req,) = client.sent
    assert (req.is_buy, req.volume, req.position_ticket, req.sl) == (False, D("0.04"), 77, None) and client.positions == {}


def test_a_reversal_closes_first_and_the_next_signal_opens_the_other_side(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    w = _worker(tmp_path, client)
    assert _run(w, _short()) == "filled"  # étape 1 : fermeture
    assert client.sent[0].position_ticket == 77 and client.positions == {}  # jamais de position inverse tant que l'autre existe
    assert _run(w, _short()) == "filled"  # étape 2 : ouverture
    assert (client.sent[1].is_buy, client.sent[1].position_ticket, client.sent[1].sl) == (False, None, D("1.10500"))
    assert next(iter(client.positions.values())).volume == D("-0.04")


def test_market_execution_symbols_send_no_price(tmp_path: Path) -> None:
    client = FakeDwx()
    client.spec = SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100"), D("0.01"), 10, "FULL", "MARKET", 2)
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "filled" and client.sent[0].price is None


def test_dry_run_sends_nothing(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client, dry_run=True)
    assert _run(w, _signal()) == "dry_run" and client.sent == []
    (rec,) = w.state.executions()
    assert (rec.status, rec.side, rec.size) == ("dry_run", "BUY", "0.04")


def test_refusals_leave_a_journal_and_send_nothing(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    assert _run(w, _signal(), now_ms=T0 + 5_001) == "stale_at_execution"
    assert _run(w, _signal(symbol="FOREX_GBP_USD"), asset="GBPUSD") == "unknown_symbol"
    client.quote_value = None
    assert _run(w, _signal()) == "no_quote"
    client.quote_value, client.lot_value = QUOTE, None
    assert _run(w, _signal()) == "no_valuation"
    client.lot_value = LOT_VALUE
    for error, code in ((WrongAccountError("autre compte"), "wrong_account"), (RealAccountRefused("réel"), "real_account_refused"),
                        (MultiplePositionsError("deux"), "hedge_positions_unsupported"), (ConnectionError("x"), "read_error")):
        client.raise_on_read = error
        assert _run(w, _signal()) == code
    assert client.sent == []


def test_a_position_on_another_symbol_whose_value_is_unknown_stops_the_order(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["GBPUSD"] = Mt5Position(9, D("0.10"), D("1.3"))
    client.unknown_value_for = {"GBPUSD"}
    assert _run(_worker(tmp_path, client), _signal()) == "no_valuation" and client.sent == []


def test_the_gross_exposure_cap_counts_the_other_positions(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["GBPUSD"] = Mt5Position(9, D("1.0"), D("1.3"))  # 1 lot = 110 000 : 110 % de l'equity, au-dessus du plafond de 100 %
    assert _run(_worker(tmp_path, client), _signal()) == "exceeds_gross_cap" and client.sent == []


def test_ambiguous_errors_are_never_retried_and_rejections_are_journaled(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    client.raise_on_send = TimeoutError("perdu")
    assert _run(w, _signal()) == "error" and client.sent == []
    client.raise_on_send = None
    client.outcome = Mt5OrderOutcome("error", 10012, "10012 TIMEOUT: x")
    assert _run(w, _signal()) == "error" and len(client.sent) == 1
    client.outcome = Mt5OrderOutcome("rejected", 10018, "10018 MARKET_CLOSED: Market closed")
    assert _run(w, _signal()) == "rejected"
    assert [r.status for r in w.state.executions()] == ["error", "error", "rejected"]
    assert "MARKET_CLOSED" in w.state.executions()[-1].detail


def test_the_account_is_rechecked_when_sending(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    client.raise_on_send = WrongAccountError("changé en cours de route")
    assert _run(w, _signal()) == "wrong_account"


def test_the_shared_drawdown_breaker_works_here_too(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    w = _worker(tmp_path, client)
    assert _run(w, _signal()) == "on_target"
    client.equity = D("94000")  # -6 %
    assert _run(w, _short()) == "halted_reduce_only" and w.halted
    assert _run(w, _flat()) == "filled"  # sortir reste possible


def test_only_darwinex_forex_signals_are_queued(tmp_path: Path) -> None:
    w = _worker(tmp_path, FakeDwx())
    now = T0 + 100
    assert w.handle_frames(_frames(_signal()), now_ms=now) == "queued"
    assert w.handle_frames(_frames(_signal(target_venue=TargetVenue.ALL)), now_ms=now) == "queued"
    crypto = PlatformSignal(
        target_venue=TargetVenue.HYPERLIQUID, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG, position_size_pct=5.0, order_type=OrderType.MARKET, timestamp_ms=T0,
    )
    assert w.handle_frames(_frames(crypto), now_ms=now) == "wrong_venue"
    assert w.handle_frames(_frames(_signal(symbol="FOREX_EUR")), now_ms=now) == "unmappable_symbol"
    assert list(w._pending) == ["EURUSD"]


def test_the_symbol_suffix_is_applied(tmp_path: Path) -> None:
    assert _worker(tmp_path, FakeDwx(), symbol_suffix=".r").map_symbol("FOREX_EUR_USD") == "EURUSD.r"


# --- Homme mort ------------------------------------------------------------------------------


def _flatten(worker: DarwinexWorker, attempt: int = 1) -> str:
    return asyncio.run(worker.emergency_flatten("EURUSD", attempt=attempt, now_ms=T0))


def test_the_dead_man_flatten_closes_with_the_ticket_and_a_widening_deviation(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    w = _worker(tmp_path, client)
    assert _flatten(w) == "filled"
    (req,) = client.sent
    assert (req.is_buy, req.volume, req.position_ticket, req.sl, req.deviation, req.price) == (False, D("0.04"), 77, None, 3300, D("1.0999"))
    deviations = []
    for attempt in (2, 3):
        client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
        _flatten(w, attempt)
        deviations.append(client.sent[-1].deviation)
    assert deviations == [6600, 13199]  # 6 % puis 12 % de 1,0999, en points


def test_the_dead_man_flatten_buys_back_a_short_at_the_ask(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(78, D("-0.05"), D("1.1"))
    assert _flatten(_worker(tmp_path, client)) == "filled"
    req = client.sent[0]
    assert (req.is_buy, req.volume, req.position_ticket, req.price) == (True, D("0.05"), 78, D("1.1001"))


def test_the_dead_man_flatten_reports_instead_of_hiding_failures(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    assert _flatten(w) == "no_position" and client.sent == []
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    client.outcome = Mt5OrderOutcome("rejected", 10018, "10018 MARKET_CLOSED: x")
    assert _flatten(w) == "rejected"  # le week-end : refusé, et réessayé au cycle suivant
    client.outcome = None
    client.raise_on_send = TimeoutError("perdu")
    assert _flatten(w) == "error"
    client.raise_on_send, client.raise_on_read = None, WrongAccountError("autre compte")
    assert _flatten(w) == "wrong_account"
    client.spec = SymbolSpec("EURUSD", 5, D("0.00001"), D("0.01"), D("100"), D("0.01"), 10, "FULL", "MARKET", 0)
    client.raise_on_read = None
    assert _flatten(w) == "no_valid_filling"
    assert client.positions["EURUSD"].volume == D("0.04")  # rien n'a fermé


def test_the_dead_man_flatten_in_dry_run_sends_nothing(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    assert _flatten(_worker(tmp_path, client, dry_run=True)) == "dry_run" and client.sent == []


def test_a_silent_engine_closes_the_mt5_position_then_locks_the_worker(tmp_path: Path) -> None:
    client = FakeDwx()
    client.positions["EURUSD"] = Mt5Position(77, D("0.04"), D("1.1"))
    w = _worker(tmp_path, client, dead_man_timeout_s=300.0, dead_man_startup_grace_s=0.0)
    w._started_ms = T0 - 10_000_000
    assert DarwinexWorker.supports_dead_man
    assert w.handle_frames(_frames(_signal()), now_ms=T0 + 100) == "queued"
    assert asyncio.run(w.check_dead_man(T0 + 100 + 300_001)) == "tripped"
    assert client.positions == {} and client.sent[0].position_ticket == 77
    assert w.handle_frames(_frames(_signal(timestamp_ms=T0 + 400_000)), now_ms=T0 + 400_100) == "dead_man_halted"


# --- Ligne de commande -----------------------------------------------------------------------


def _env(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for key in ("DWZ_ACCOUNT_LOGIN", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "MT5_PATH", "NOTHIERLY_ZMQ_ADDRESS", "NOTHIERLY_ZMQ_SECRET",
                "NOTHIERLY_CURVE_SERVER_PUBLIC", "NOTHIERLY_CURVE_CLIENT_PUBLIC", "NOTHIERLY_CURVE_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_cli_clear_halt_needs_no_terminal_no_account_and_no_secret(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _env(monkeypatch)

    def forbidden(**_: Any) -> None:
        raise AssertionError("aucun client MT5 ne doit être créé pour lever un verrou")

    monkeypatch.setattr(worker_module, "MetaTrader5Client", forbidden)
    db = tmp_path / "locked.sqlite3"
    state = ExecutionState(db)
    state.set("halted", "1")
    state.set("dead_man_tripped", "123")
    state.set("last_signal:EURUSD", "1")
    state.close()
    assert worker_module.main(["--clear-halt", "--db", str(db)]) == 0
    after = ExecutionState(db)
    assert after.get("halted") is None and after.get("dead_man_tripped") is None and int(after.get("last_signal:EURUSD") or 0) > 1


def test_cli_requires_a_pinned_account_and_refuses_unsafe_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_module, "MetaTrader5Client", lambda **_: pytest.fail("client créé malgré une erreur de configuration"))
    _env(monkeypatch, NOTHIERLY_ZMQ_SECRET=SECRET)
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1"])  # pas de compte épinglé
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://10.0.0.5:1", "--account-login", str(LOGIN)])  # distant et en clair
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN), "--live", "--allow-real-account", "--dead-man-timeout", "0"])
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN), "--dead-man-timeout", "-1"])


def test_cli_stops_when_the_terminal_is_logged_into_the_wrong_account(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Wrong:
        def account_check(self) -> AccountInfo:
            raise WrongAccountError("compte 123 au lieu de 5001234")

    _env(monkeypatch, NOTHIERLY_ZMQ_SECRET=SECRET)
    monkeypatch.setattr(worker_module, "MetaTrader5Client", lambda **_: _Wrong())
    monkeypatch.setattr(worker_module, "DarwinexWorker", lambda *a, **k: pytest.fail("le worker ne doit pas démarrer"))
    assert worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN)]) == 2


def test_cli_defaults_are_dry_run_demo_with_the_dead_man_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    class _Ok:
        def account_check(self) -> AccountInfo:
            return AccountInfo(LOGIN, "DEMO", "HEDGING", "USD", "Darwinex-Demo", True)

        def close(self) -> None:
            seen["closed"] = True

    def client_factory(**kwargs: Any) -> _Ok:
        seen["client_kwargs"] = kwargs
        return _Ok()

    def worker_factory(config: DarwinexWorkerConfig, client: Any) -> None:
        seen["config"] = config
        raise _Stop

    _env(monkeypatch, NOTHIERLY_ZMQ_SECRET=SECRET, MT5_LOGIN=str(LOGIN), MT5_PASSWORD="pwd", MT5_SERVER="Darwinex-Demo")
    monkeypatch.setattr(worker_module, "MetaTrader5Client", client_factory)
    monkeypatch.setattr(worker_module, "DarwinexWorker", worker_factory)
    with pytest.raises(_Stop):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN), "--symbol-suffix", ".r"])
    assert seen["config"].dry_run is True and seen["config"].dead_man_timeout_s == 300.0 and seen["config"].symbol_suffix == ".r"
    assert seen["client_kwargs"]["expected_login"] == LOGIN and seen["client_kwargs"]["allow_real_account"] is False
    assert (seen["client_kwargs"]["login"], seen["client_kwargs"]["server"]) == (LOGIN, "Darwinex-Demo")


# --- Stop-loss par défaut --------------------------------------------------------------------


def test_the_worker_adds_a_default_stop_at_one_percent_of_the_entry_price() -> None:
    buy = _plan(_signal(metadata={}), default_stop_loss_bps=D(100))
    assert isinstance(buy, Mt5OrderPlan) and buy.sl_defaulted
    assert buy.sl == D("1.08909")  # ask 1,1001 x 0,99 = 1,089099, arrondi au point INFÉRIEUR (plus loin = plus protecteur)
    sell = _plan(_short(metadata={}), default_stop_loss_bps=D(100))
    assert isinstance(sell, Mt5OrderPlan) and sell.sl == D("1.11090") and sell.sl_defaulted  # bid 1,0999 x 1,01 = 1,110899 -> point supérieur


def test_the_default_stop_of_a_limit_order_is_measured_from_its_own_entry_price() -> None:
    limit = _plan(_signal(order_type=OrderType.LIMIT, metadata={"limit_price": 1.095}), default_stop_loss_bps=D(100))
    assert isinstance(limit, Mt5OrderPlan) and limit.price == D("1.09500")
    assert limit.sl == D("1.08405")  # 1,095 x 0,99 : sous l'ENTRÉE, pas sous le marché (sinon le stop serait au-dessus de l'entrée)


def test_a_default_stop_is_pushed_out_to_the_brokers_minimum_distance() -> None:
    buy = _plan(_signal(metadata={}), default_stop_loss_bps=D(1))  # 1 pb = 0,00011 : plus près que bid - 10 points
    sell = _plan(_short(metadata={}), default_stop_loss_bps=D(1))
    assert isinstance(buy, Mt5OrderPlan) and isinstance(sell, Mt5OrderPlan)
    assert buy.sl == D("1.09980") and sell.sl == D("1.10020")  # bid - 0,0001 et ask + 0,0001


def test_a_stop_given_by_the_signal_is_never_silently_replaced() -> None:
    bad = _signal(metadata={"stop_loss_price": 1.0999})  # trop près du bid
    assert _plan(bad, default_stop_loss_bps=D(100)) == Skip("stop_too_close_or_wrong_side")
    kept = _plan(_signal(metadata={"stop_loss_price": 1.095}), default_stop_loss_bps=D(100))
    assert isinstance(kept, Mt5OrderPlan) and kept.sl == D("1.09500") and not kept.sl_defaulted


def test_without_a_default_a_missing_stop_refuses_the_order_and_a_close_never_needs_one() -> None:
    assert _plan(_signal(metadata={})) == Skip("missing_stop_loss")
    close = _plan(_flat(), position=Mt5Position(77, D("0.04"), D("1.1")), target_abs_lots=D(0), default_stop_loss_bps=D(100))
    assert isinstance(close, Mt5OrderPlan) and close.sl is None and not close.sl_defaulted


def test_the_worker_sends_the_default_stop_to_mt5_and_journals_that_it_chose_it(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client)
    assert w._dwx_cfg.default_stop_loss_bps == D("100")  # armé par défaut
    assert _run(w, _signal(metadata={})) == "filled"
    assert client.sent[0].sl == D("1.08909")
    assert "stop par défaut" in w.state.executions()[0].detail


def test_with_the_default_disabled_the_worker_refuses_a_signal_without_a_stop(tmp_path: Path) -> None:
    client = FakeDwx()
    w = _worker(tmp_path, client, default_stop_loss_bps=None)
    assert _run(w, _signal(metadata={})) == "missing_stop_loss" and client.sent == []
    assert _run(w, _signal(metadata={"stop_loss_price": 1.095})) == "filled"  # un stop explicite passe toujours


def test_cli_default_stop_loss_option(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class _Stop(Exception):
        pass

    class _Ok:
        def account_check(self) -> AccountInfo:
            return AccountInfo(LOGIN, "DEMO", "HEDGING", "USD", "x", True)

        def close(self) -> None:
            pass

    def worker_factory(config: DarwinexWorkerConfig, client: Any) -> None:
        seen.append(config.default_stop_loss_bps)
        raise _Stop

    monkeypatch.setattr(worker_module, "MetaTrader5Client", lambda **_: _Ok())
    monkeypatch.setattr(worker_module, "DarwinexWorker", worker_factory)
    for extra, expected in (([], D("100")), (["--default-stop-loss-bps", "50"], D("50")), (["--default-stop-loss-bps", "0"], None)):
        _env(monkeypatch, NOTHIERLY_ZMQ_SECRET=SECRET)
        with pytest.raises(_Stop):
            worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN), *extra])
        assert seen[-1] == expected
    with pytest.raises(SystemExit):
        worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN), "--default-stop-loss-bps", "-5"])


# --- Vérification d'OS (CI Linux) ------------------------------------------------------------


def test_the_client_refuses_to_start_off_windows_but_a_mocked_module_works_on_any_os(monkeypatch: pytest.MonkeyPatch) -> None:
    from workers.darwinex import client as client_module

    monkeypatch.setattr(client_module, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(client_module.UnsupportedPlatformError, match="Windows"):
        MetaTrader5Client(expected_login=LOGIN, magic=1)
    # Avec un module simulé, aucune exigence d'OS : c'est ce qui permet aux tests de tourner sur un pipeline CI Linux.
    assert MetaTrader5Client(expected_login=LOGIN, magic=1, mt5_module=FakeMt5()).account_check().login == LOGIN


def test_a_missing_metatrader5_package_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from workers.darwinex import client as client_module

    monkeypatch.setattr(client_module, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)  # `import MetaTrader5` lève ImportError
    with pytest.raises(client_module.Mt5NotInstalledError, match="pip install MetaTrader5"):
        MetaTrader5Client(expected_login=LOGIN, magic=1)


def test_the_command_line_stops_with_a_message_not_a_traceback_off_windows(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from workers.darwinex import client as client_module

    _env(monkeypatch, NOTHIERLY_ZMQ_SECRET=SECRET)
    monkeypatch.setattr(client_module, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(worker_module, "DarwinexWorker", lambda *a, **k: pytest.fail("le worker ne doit pas démarrer"))
    assert worker_module.main(["--address", "tcp://127.0.0.1:1", "--account-login", str(LOGIN)]) == 2
    assert "Windows" in capsys.readouterr().err

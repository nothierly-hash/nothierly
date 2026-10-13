"""Tests du pont moteur -> workers (``src/bridge/platform_bridge.py``). Aucun Redis réel, aucun réseau.

Chiffres calculés à la main : equity moteur 10 000 USD, BTC/USDT long de 0,1 BTC à 50 000 USD -> 5 000 USD -> 50 %,
plafonné à 20 % ; 0,01 BTC -> 500 USD -> 5 %.
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from typing import Any

import pytest

from contracts.platform_signal import PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import open_envelope
from src.bridge.platform_bridge import (
    ACTIVE_POSITIONS_KEY,
    ENGINE_HEARTBEAT_KEY,
    EQUITY_CURVE_KEY,
    BridgeConfig,
    EnginePosition,
    EngineView,
    PlatformBridge,
    RedisEngineReader,
    desired_target,
    quantize_size,
    to_internal_symbol,
)

D = Decimal


def _pos(instrument: str = "BTC/USDT", status: str = "LONG", units: str = "0.01", price: str = "50000") -> EnginePosition:
    return EnginePosition(instrument, status, D(units), D(price))


def _view(*positions: EnginePosition, equity: str | None = "10000", alive: bool = True, safe: bool = True) -> EngineView:
    return EngineView(tuple(positions), None if equity is None else D(equity), alive, safe)


class FakeReader:
    def __init__(self, view: EngineView | None = None) -> None:
        self.view = view
        self.error: Exception | None = None

    async def read(self) -> EngineView:
        if self.error is not None:
            raise self.error
        assert self.view is not None
        return self.view


class FakeSink:
    def __init__(self) -> None:
        self.signals: list[PlatformSignal] = []
        self.accept = True

    def publish(self, signal: PlatformSignal) -> bool:
        if not self.accept:
            return False
        self.signals.append(signal)
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _bridge(view: EngineView | None = None, **cfg: Any) -> tuple[PlatformBridge, FakeReader, FakeSink, Clock]:
    reader, sink, clock = FakeReader(view), FakeSink(), Clock()
    cfg.setdefault("size_bucket_pct", None)  # tailles exactes, sauf dans les tests de quantification
    return PlatformBridge(reader, sink, BridgeConfig(**cfg), clock=clock), reader, sink, clock


def _step(bridge: PlatformBridge) -> list[str]:
    return asyncio.run(bridge.step())


# --- Traduction ------------------------------------------------------------------------------


def test_symbol_translation() -> None:
    assert to_internal_symbol("BTC/USDT") == "CRYPTO_BTC_USD" and to_internal_symbol("ETH/USD") == "CRYPTO_ETH_USD"
    assert to_internal_symbol("EUR_USD") is None and to_internal_symbol("BTC/EUR") is None and to_internal_symbol("btc/usdt") is None
    assert to_internal_symbol("PEPE/USDT", {"PEPE/USDT": "CRYPTO_1000PEPE_USD"}) == "CRYPTO_1000PEPE_USD"


def test_target_exposure_is_the_engines_own_sizing_as_a_percentage_of_its_equity() -> None:
    cfg = BridgeConfig(size_bucket_pct=None)
    assert desired_target(_pos(units="0.01"), D(10000), cfg) == (SignalDirection.LONG, D("5.00"))  # 500 / 10 000
    assert desired_target(_pos(status="SHORT", units="0.01"), D(10000), cfg) == (SignalDirection.SHORT, D("5.00"))
    assert desired_target(_pos(units="0.1"), D(10000), cfg) == (SignalDirection.LONG, D("20.00"))  # 50 % plafonné à 20 %
    assert desired_target(None, D(10000), cfg) == (SignalDirection.FLAT, D(0))


def test_a_dust_position_becomes_flat_and_an_unknown_equity_is_never_guessed() -> None:
    cfg = BridgeConfig(size_bucket_pct=None)
    assert desired_target(_pos(units="0.000001"), D(10000), cfg) == (SignalDirection.FLAT, D(0))
    assert desired_target(_pos(), None, cfg) is None and desired_target(_pos(), D(0), cfg) is None
    assert desired_target(None, None, cfg) == (SignalDirection.FLAT, D(0))  # pas de position : flat n'a pas besoin d'equity


# --- Quantification de la taille --------------------------------------------------------------


def test_quantize_rounds_half_up_to_the_nearest_bucket_with_decimals() -> None:
    q = lambda raw: quantize_size(D(raw), D("2.5"), D("20"))  # noqa: E731
    assert q("5.00") == D("5.0") and q("6.24") == D("5.0") and q("6.25") == D("7.5")  # 2,5 x 2,5 = 6,25 : moitié vers le HAUT
    assert q("3.75") == D("5.0") and q("3.74") == D("2.5")  # 6,25 / 2,5 = 2,5 : round() du banquier donnerait 2 -> 5,0 ; ici 3 -> 7,5
    assert q("18.80") == D("20.0") and q("19.9") == D("20.0")


def test_quantize_never_returns_zero_and_never_exceeds_the_cap() -> None:
    assert quantize_size(D("0.01"), D("2.5"), D("20")) == D("2.5")  # une position ouverte n'est jamais un signal de taille 0
    assert quantize_size(D("1.24"), D("2.5"), D("20")) == D("2.5")
    assert quantize_size(D("20"), D("2.5"), D("19")) == D("19")  # le plafond prime sur la grille
    assert quantize_size(D("100"), D("2.5"), D("20")) == D("20")


def test_quantize_hysteresis_keeps_the_published_bucket_near_a_boundary() -> None:
    # frontière entre 5,0 et 7,5 = 6,25 ; sans hystérésis 6,24 -> 5,0 et 6,26 -> 7,5 : le signal basculerait à chaque cycle
    assert quantize_size(D("6.26"), D("2.5"), D("20"), previous=D("5.0")) == D("5.0")
    assert quantize_size(D("6.87"), D("2.5"), D("20"), previous=D("5.0")) == D("5.0")  # |écart| 1,87 < 0,75 x 2,5
    assert quantize_size(D("6.88"), D("2.5"), D("20"), previous=D("5.0")) == D("7.5")  # 1,88 >= 1,875 : on bascule
    assert quantize_size(D("3.13"), D("2.5"), D("20"), previous=D("5.0")) == D("5.0")  # 1,87 < 1,875 : on garde
    assert quantize_size(D("3.12"), D("2.5"), D("20"), previous=D("5.0")) == D("2.5")  # 1,88 vers le bas : bascule


def test_the_bridge_publishes_buckets_not_the_engines_exact_size() -> None:
    bridge, reader, sink, clock = _bridge(_view(_pos(units="0.0113")), instruments=("BTC/USDT",), size_bucket_pct=D("2.5"))
    assert _step(bridge) == ["BTC/USDT:LONG:5.00:engine_state"]  # 5,65 % exact -> 5,0
    reader.view = _view(_pos(units="0.0125"))  # 6,25 % : à la frontière, mais à 1,25 < 1,875 du palier publié : hystérésis
    clock.now += 1
    assert _step(bridge) == []
    reader.view = _view(_pos(units="0.0140"))  # 7,00 % : écart 2,0 >= 1,875 -> 7,5
    assert _step(bridge) == ["BTC/USDT:LONG:7.50:engine_state"]
    assert [s.position_size_pct for s in sink.signals] == [5.0, 7.5]


def test_a_tiny_position_is_published_at_one_bucket_and_flat_and_direction_changes_reset_the_grid() -> None:
    bridge, reader, sink, clock = _bridge(_view(_pos(units="0.0004")), instruments=("BTC/USDT",), size_bucket_pct=D("2.5"))
    assert _step(bridge) == ["BTC/USDT:LONG:2.50:engine_state"]  # 0,2 % exact : un palier, pas 0
    reader.view = _view(_pos(status="SHORT", units="0.0004"))
    assert _step(bridge) == ["BTC/USDT:SHORT:2.50:engine_state"]
    reader.view = _view()
    assert _step(bridge) == ["BTC/USDT:FLAT:0:engine_state"] and sink.signals[-1].position_size_pct == 0.0


def test_quantized_signals_are_always_valid_contract_signals() -> None:
    bridge, reader, sink, _ = _bridge(instruments=("BTC/USDT",), size_bucket_pct=D("2.5"))
    for units in ("0.0001", "0.0099", "0.0151", "0.05", "0.5"):
        reader.view = _view(_pos(units=units))
        bridge._last.clear()
        _step(bridge)
    assert all(0 < s.position_size_pct <= 20 and (D(str(s.position_size_pct)) % D("2.5")) == 0 for s in sink.signals)


# --- Publication -----------------------------------------------------------------------------


def test_an_open_engine_position_is_published_as_a_valid_signal_without_confidence() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT",))
    assert _step(bridge) == ["BTC/USDT:LONG:5.00:engine_state"]
    (signal,) = sink.signals
    assert (signal.symbol, signal.direction, signal.position_size_pct, signal.target_venue, signal.ttl_ms) == (
        "CRYPTO_BTC_USD", SignalDirection.LONG, 5.0, TargetVenue.ALL, 10_000)
    assert signal.confidence_score is None  # interdit pour toute plateforme crypto


def test_every_tracked_instrument_gets_a_signal_flat_when_the_engine_holds_nothing() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos()))
    _step(bridge)
    by_symbol = {s.symbol: (s.direction, s.position_size_pct) for s in sink.signals}
    assert by_symbol == {"CRYPTO_BTC_USD": (SignalDirection.LONG, 5.0), "CRYPTO_ETH_USD": (SignalDirection.FLAT, 0.0)}


def test_unchanged_state_is_not_republished_until_the_refresh_interval() -> None:
    bridge, _, sink, clock = _bridge(_view(_pos()), instruments=("BTC/USDT",), refresh_s=10.0)
    assert len(_step(bridge)) == 1
    clock.now += 5
    assert _step(bridge) == [] and len(sink.signals) == 1
    clock.now += 5  # 10 s écoulées : republication de rattrapage
    assert len(_step(bridge)) == 1 and len(sink.signals) == 2


def test_a_change_of_direction_or_a_big_enough_size_change_publishes_immediately() -> None:
    bridge, reader, sink, clock = _bridge(_view(_pos(units="0.01")), instruments=("BTC/USDT",))
    _step(bridge)
    reader.view = _view(_pos(units="0.0104"))  # 5,20 % : +0,20 point, sous le seuil de 0,25
    clock.now += 1
    assert _step(bridge) == []
    reader.view = _view(_pos(units="0.0106"))  # 5,30 % : +0,30 point depuis la dernière publication
    assert len(_step(bridge)) == 1
    reader.view = _view(_pos(status="SHORT", units="0.0106"))
    assert _step(bridge)[0].startswith("BTC/USDT:SHORT")
    reader.view = _view()  # le moteur a fermé : FLAT part tout de suite
    assert _step(bridge) == ["BTC/USDT:FLAT:0:engine_state"] and sink.signals[-1].position_size_pct == 0.0


def test_a_dropped_message_is_retried_on_the_next_cycle() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT",))
    sink.accept = False
    assert _step(bridge) == []
    sink.accept = True
    assert len(_step(bridge)) == 1  # rien n'a été mémorisé comme publié


def test_unsupported_instruments_are_ignored_not_guessed() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos("EUR_USD", units="1000", price="1.1")), instruments=("EUR_USD", "BTC/USDT"))
    _step(bridge)
    assert [s.symbol for s in sink.signals] == ["CRYPTO_BTC_USD"]


def test_an_open_position_with_unknown_equity_publishes_nothing_for_it() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos(), equity=None), instruments=("BTC/USDT", "ETH/USDT"))
    _step(bridge)
    assert [(s.symbol, s.direction) for s in sink.signals] == [("CRYPTO_ETH_USD", SignalDirection.FLAT)]


def test_the_venue_can_target_a_single_platform() -> None:
    bridge, _, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT",), venue=TargetVenue.BYBIT)
    _step(bridge)
    assert sink.signals[0].target_venue is TargetVenue.BYBIT


# --- Sécurité de repli -----------------------------------------------------------------------


def test_the_global_kill_switch_flattens_everything_immediately() -> None:
    bridge, reader, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT", "ETH/USDT"))
    _step(bridge)
    reader.view = _view(_pos(), safe=False)  # PANIC : même si le moteur croit encore détenir la position
    assert _step(bridge) == ["BTC/USDT:FLAT:0:kill_switch"]  # ETH était déjà FLAT : rien de plus à dire
    assert (sink.signals[-1].symbol, sink.signals[-1].direction, sink.signals[-1].position_size_pct) == ("CRYPTO_BTC_USD", SignalDirection.FLAT, 0.0)


def test_flatten_on_halt_can_be_disabled() -> None:
    bridge, reader, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT",), flatten_on_halt=False)
    _step(bridge)
    reader.view = _view(_pos(), safe=False)
    assert _step(bridge) == []  # halte ignorée : l'état du moteur (LONG, inchangé) reste la cible
    assert sink.signals[-1].direction is SignalDirection.LONG


def test_a_silent_engine_is_flattened_only_after_the_grace_period() -> None:
    bridge, reader, sink, clock = _bridge(_view(_pos()), instruments=("BTC/USDT",), engine_grace_s=30.0)
    _step(bridge)
    reader.view = _view(_pos(), alive=False)
    clock.now += 10  # première observation du silence : le délai de grâce part d'ici
    assert _step(bridge) == []  # dans le délai de grâce : on ne bouge pas
    clock.now += 19  # 19 s de silence observé
    assert _step(bridge) == []
    clock.now += 12  # 31 s de silence observé
    assert _step(bridge) == ["BTC/USDT:FLAT:0:engine_down"]
    reader.view = _view(_pos(), alive=True)  # le moteur revient : la cible redevient celle de son état
    clock.now += 1
    assert _step(bridge)[0].startswith("BTC/USDT:LONG")


def test_unreadable_state_counts_as_engine_down_after_the_grace_period() -> None:
    bridge, reader, sink, clock = _bridge(_view(_pos()), instruments=("BTC/USDT",), engine_grace_s=30.0)
    _step(bridge)
    reader.error = ConnectionError("redis injoignable")
    clock.now += 20  # première lecture ratée : le délai de grâce part d'ici
    assert _step(bridge) == []
    clock.now += 31  # 31 s d'état illisible
    assert _step(bridge) == ["BTC/USDT:FLAT:0:engine_down"]
    assert sink.signals[-1].direction is SignalDirection.FLAT


def test_a_short_blip_in_the_heartbeat_resets_the_grace_clock() -> None:
    bridge, reader, _, clock = _bridge(_view(_pos()), instruments=("BTC/USDT",), engine_grace_s=30.0)
    _step(bridge)
    reader.view = _view(_pos(), alive=False)
    clock.now += 25
    _step(bridge)
    reader.view = _view(_pos(), alive=True)
    clock.now += 1
    _step(bridge)  # sain : l'horloge de silence repart de zéro
    reader.view = _view(_pos(), alive=False)
    clock.now += 25
    assert _step(bridge) == []  # 25 s depuis le NOUVEAU silence, pas 51 s cumulées


# --- Lecture Redis ---------------------------------------------------------------------------


class FakeRedis:
    def __init__(self, store: dict[str, Any]) -> None:
        self.store = store

    async def execute(self, command: str, *args: Any) -> Any:
        if command == "get":
            return self.store.get(args[0])
        if command == "lindex":
            return self.store.get(args[0])
        if command == "exists":
            return 1 if args[0] in self.store else 0
        raise AssertionError(command)


def test_redis_reader_decodes_the_engines_real_formats() -> None:
    positions = [
        {"instrument": "BTC/USDT", "status": "LONG", "entry_price": 49000.0, "current_price": 50000.0, "units": 0.01, "pnl": 10.0},
        {"instrument": "ETH/USDT", "status": "SHORT", "entry_price": 3000.0, "current_price": None, "units": -2.0, "pnl": 0.0},
    ]
    store = {ACTIVE_POSITIONS_KEY: json.dumps(positions), EQUITY_CURVE_KEY: json.dumps({"timestamp": 1.0, "balance": 10234.5}), ENGINE_HEARTBEAT_KEY: "1700000000.0"}

    async def safe() -> bool:
        return True

    view = asyncio.run(RedisEngineReader(FakeRedis(store), safe).read())
    assert view.equity == D("10234.5") and view.engine_alive and view.system_safe
    assert view.positions == (
        EnginePosition("BTC/USDT", "LONG", D("0.01"), D("50000.0")),
        EnginePosition("ETH/USDT", "SHORT", D("2.0"), D("3000.0")),  # unités signées -> valeur absolue ; prix d'entrée si pas de prix courant
    )


def test_redis_reader_handles_an_empty_engine_and_a_halted_system() -> None:
    async def halted() -> bool:
        return False

    view = asyncio.run(RedisEngineReader(FakeRedis({}), halted).read())
    assert view == EngineView((), None, False, False)


def test_a_bridged_signal_is_accepted_by_a_real_worker_check() -> None:
    """Bout en bout du contrat : ce que le pont publie traverse la vérification d'un worker Hyperliquid."""
    from contracts.secure_signal_bus import build_envelope

    bridge, _, sink, _ = _bridge(_view(_pos()), instruments=("BTC/USDT",))
    _step(bridge)
    secret = "test-secret-0123456789abcdef-0123456789abcdef"
    for venue in (TargetVenue.HYPERLIQUID, TargetVenue.BYBIT):
        verdict = open_envelope(build_envelope(sink.signals[0], secret), secret, venue=venue)
        assert verdict.accepted, venue
    assert open_envelope(build_envelope(sink.signals[0], secret), secret, venue=TargetVenue.NUMERAI_SIGNALS).reject_reason == "wrong_venue"

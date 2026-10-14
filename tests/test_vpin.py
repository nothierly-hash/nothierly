"""Tests du VPIN : horloge de volume, démon de flux publics, écriture Redis, lecture par la Porte 3 de l'``ExecutionAgent``.

Preuves recherchées :

- l'horloge de volume est exacte (chiffres faits à la main) : chaque seau fermé vaut EXACTEMENT ``bucket_size`` même quand un gros
  trade en traverse plusieurs, aucun VPIN n'est annoncé avant que la fenêtre soit pleine, et le résultat ne dépend pas de la
  convention achat/vente ;
- le sens vient de la plateforme (``m`` de Binance, ``side`` d'Hyperliquid) ;
- le démon survit aux coupures, aux messages illisibles et à un Redis en panne ; un trou de flux remet la fenêtre à zéro ;
- le moteur lit ``market:vpin:<SYMBOLE>``, ne fabrique jamais une valeur, et un modèle qui déclare ``vpin`` répond neutre sans lecture.

Aucun service externe : faux flux en mémoire, serveur WebSocket sur ``127.0.0.1``, fakeredis.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import fakeredis
import fakeredis.aioredis
import numpy as np
import pytest

from risk.meta_labeling import FEATURES_V2, FEATURES_V3, MetaLabelModel
from workers.common.alerts import Severity
from workers.common.ws_tracker import WsConnection
from workers.market_data import vpin_tracker as vt
from tests.test_metalabel_pipeline import agent_factory  # noqa: F401 -- fixture partagée : construit un ExecutionAgent isolé
from workers.market_data.vpin import VolumeBucketVpin
from workers.market_data.vpin_tracker import (
    BinanceAggTradeFeed,
    HyperliquidTradesFeed,
    RedisVpinSink,
    VpinTracker,
    redis_key,
)

D = Decimal
T0 = 1_760_000_000_000


# --- Horloge de volume (chiffres faits à la main) --------------------------------------------


def _calc(bucket: str = "100", window: int = 2) -> VolumeBucketVpin:
    return VolumeBucketVpin(D(bucket), window)


def test_a_bucket_closes_at_exactly_the_bucket_size_and_the_vpin_waits_for_a_full_window() -> None:
    calc = _calc()
    assert calc.add_trade(D(1), D(60), True) == [] and calc.partial_volume == D(60) and calc.vpin is None
    (first,) = calc.add_trade(D(1), D(40), False, T0)
    assert (first.buy_volume, first.sell_volume, first.imbalance, first.closed_at_ms) == (D(60), D(40), D(20), T0)
    assert first.vpin is None and first.buckets_in_window == 1 and calc.warmup == 1  # 1 seau sur 2 : pas de valeur
    (second,) = calc.add_trade(D(1), D(100), True)
    assert second.imbalance == D(100) and second.vpin == D("0.6") and calc.warmup == 0  # (20 + 100) / (2 x 100)
    (third,) = calc.add_trade(D(1), D(100), False)
    assert third.vpin == D(1)  # le plus ancien seau sort : (100 + 100) / 200


def test_a_trade_larger_than_a_bucket_is_split_so_every_closed_bucket_is_exact() -> None:
    calc = _calc()
    closed = calc.add_trade(D(1), D(250), True)
    assert [(r.buy_volume, r.sell_volume, r.imbalance) for r in closed] == [(D(100), D(0), D(100))] * 2
    assert calc.partial_volume == D(50) and calc.buckets_closed == 2 and closed[-1].vpin == D(1)
    crossing = _calc()
    crossing.add_trade(D(1), D(80), True)
    (bucket,) = crossing.add_trade(D(1), D(50), False)  # 20 finissent le seau (80 achat, 20 vente), 30 ouvrent le suivant
    assert (bucket.buy_volume, bucket.sell_volume, bucket.imbalance) == (D(80), D(20), D(60))
    assert crossing.partial_volume == D(30)


def test_the_notional_is_price_times_quantity() -> None:
    calc = _calc()
    (reading,) = calc.add_trade(D(50000), D("0.002"), True)  # 50 000 x 0,002 = 100 $
    assert reading.buy_volume == D(100) and calc.partial_volume == D(0)


def test_balanced_flow_gives_zero_and_one_sided_flow_gives_one() -> None:
    balanced, one_sided = _calc(window=3), _calc(window=3)
    for _ in range(3):
        balanced.add_trade(D(1), D(50), True)
        balanced.add_trade(D(1), D(50), False)
        one_sided.add_trade(D(1), D(100), True)
    assert balanced.vpin == D(0) and one_sided.vpin == D(1)


def test_every_closed_bucket_is_exact_and_the_vpin_stays_in_zero_one_on_random_flow() -> None:
    rng = random.Random(42)
    calc = VolumeBucketVpin(D("1000"), 5)
    readings = []
    for _ in range(2000):
        readings += calc.add_trade(D(str(round(rng.uniform(1, 200), 2))), D(str(round(rng.uniform(0.001, 30), 4))), rng.random() < 0.5)
    assert len(readings) > 50
    assert all(r.buy_volume + r.sell_volume == D(1000) for r in readings)  # jamais un seau qui déborde
    assert all(r.vpin is None or D(0) <= r.vpin <= D(1) for r in readings) and all(r.vpin is not None for r in readings[4:])
    assert all(r.vpin is None for r in readings[:4])


def test_swapping_buy_and_sell_labels_never_changes_the_vpin() -> None:
    """Pourquoi une convention de sens douteuse (Hyperliquid) est sans effet : seul |achat - vente| compte."""
    rng = random.Random(7)
    flow = [(D(str(round(rng.uniform(10, 90), 2))), D(str(round(rng.uniform(0.1, 5), 3))), rng.random() < 0.6) for _ in range(800)]
    a, b = VolumeBucketVpin(D(500), 4), VolumeBucketVpin(D(500), 4)
    for price, qty, is_buy in flow:
        assert [r.vpin for r in a.add_trade(price, qty, is_buy)] == [r.vpin for r in b.add_trade(price, qty, not is_buy)]


def test_reset_clears_the_window_and_the_partial_bucket() -> None:
    calc = _calc()
    calc.add_trade(D(1), D(250), True)
    calc.reset()
    assert (calc.vpin, calc.partial_volume, calc.buckets_closed, calc.warmup) == (None, D(0), 0, 2)


@pytest.mark.parametrize("price, qty", [(D(0), D(1)), (D(1), D(0)), (D(-1), D(1)), (D(1), D(-5)), (D("NaN"), D(1)), (D(1), D("Infinity"))])
def test_a_doubtful_trade_is_refused_not_half_counted(price: Decimal, qty: Decimal) -> None:
    calc = _calc()
    with pytest.raises(ValueError):
        calc.add_trade(price, qty, True)
    assert calc.partial_volume == D(0)


@pytest.mark.parametrize("bucket, window", [("0", 5), ("-1", 5), ("NaN", 5), ("100", 0)])
def test_invalid_parameters_are_refused(bucket: str, window: int) -> None:
    with pytest.raises(ValueError):
        VolumeBucketVpin(D(bucket), window)


# --- Flux publics ----------------------------------------------------------------------------


def _agg(symbol: str = "BTCUSDT", *, agg_id: int = 1, price: str = "50000", qty: str = "0.002", maker: bool = False, wrapped: bool = True) -> str:
    event = {"e": "aggTrade", "E": T0, "s": symbol, "a": agg_id, "p": price, "q": qty, "f": agg_id, "l": agg_id, "T": T0, "m": maker}
    return json.dumps({"stream": f"{symbol.lower()}@aggTrade", "data": event} if wrapped else event)


def test_binance_aggressor_side_follows_the_buyer_is_maker_flag() -> None:
    feed = BinanceAggTradeFeed(["BTCUSDT"])
    (sell,) = feed.parse(_agg(maker=True))  # l'acheteur est le teneur de marché : le preneur a VENDU
    (buy,) = feed.parse(_agg(maker=False, agg_id=2, wrapped=False))  # message non enveloppé accepté
    assert (sell.is_buy, buy.is_buy) == (False, True)
    assert (buy.symbol, buy.price, buy.qty, buy.time_ms, buy.trade_id) == ("BTCUSDT", D("50000"), D("0.002"), T0, 2)
    assert feed.parse(json.dumps({"result": None, "id": 1})) == [] and feed.parse(json.dumps({"data": {"e": "markPriceUpdate"}})) == []
    with pytest.raises(ValueError):
        feed.parse(_agg(price="pas un nombre"))
    with pytest.raises(KeyError):
        feed.parse(json.dumps({"e": "aggTrade", "s": "BTCUSDT"}))


def test_binance_url_uses_the_documented_market_stream_scheme_with_lowercase_symbols() -> None:
    assert BinanceAggTradeFeed(["BTCUSDT", "ETHUSDT"]).url == "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade"
    assert BinanceAggTradeFeed(["BTCUSDT"], "wss://demo-fstream.binance.com/").url == "wss://demo-fstream.binance.com/market/stream?streams=btcusdt@aggTrade"


HL_TRADE = {"coin": "BTC", "side": "B", "px": "50000.5", "sz": "0.01", "time": T0, "hash": "0xabc", "tid": 99, "users": ["0xb", "0xs"]}


def _hl(*trades: dict[str, Any]) -> str:
    return json.dumps({"channel": "trades", "data": list(trades)})


def test_hyperliquid_trades_are_parsed_and_mapped_to_the_engine_symbol() -> None:
    feed = HyperliquidTradesFeed(["BTCUSDT", "ETHUSDT"], "testnet")
    buy, sell = feed.parse(_hl(HL_TRADE, {**HL_TRADE, "side": "A", "tid": 100, "coin": "ETH"}))
    assert (buy.symbol, buy.is_buy, buy.price, buy.qty, buy.trade_id) == ("BTCUSDT", True, D("50000.5"), D("0.01"), 99)
    assert (sell.symbol, sell.is_buy) == ("ETHUSDT", False)
    assert feed.parse(_hl({**HL_TRADE, "coin": "DOGE"})) == []  # pièce non suivie
    assert feed.parse(json.dumps({"channel": "subscriptionResponse", "data": {}})) == [] and feed.parse(json.dumps({"channel": "pong"})) == []
    with pytest.raises(ValueError):
        feed.parse(_hl({**HL_TRADE, "side": "?"}))
    assert feed.url == "wss://api.hyperliquid-testnet.xyz/ws" and HyperliquidTradesFeed(["BTCUSDT"], "mainnet").url == "wss://api.hyperliquid.xyz/ws"


def test_hyperliquid_subscribes_to_each_coin_and_pings() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, text: str) -> None:
            self.sent.append(text)

    feed = HyperliquidTradesFeed(["BTCUSDT", "ETHUSDT"])
    rec = Recorder()
    asyncio.run(feed.handshake(rec))  # type: ignore[arg-type]  # le faux ne fournit que send(), seule méthode appelée par le handshake
    assert [json.loads(m)["subscription"] for m in rec.sent] == [{"type": "trades", "coin": "BTC"}, {"type": "trades", "coin": "ETH"}]
    assert json.loads(feed.heartbeat_message() or "{}") == {"method": "ping"}


# --- Démon sur faux flux ---------------------------------------------------------------------


class ScriptConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.inbox: asyncio.Queue[str | Exception] = asyncio.Queue()
        self.closed = False

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def recv(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


class RecordingSink:
    def __init__(self) -> None:
        self.published: list[tuple[str, Decimal, float]] = []
        self.cleared: list[str] = []
        self.fail = False

    async def publish(self, symbol: str, vpin: Decimal, ttl_s: float) -> None:
        if self.fail:
            raise ConnectionError("redis en panne")
        self.published.append((symbol, vpin, ttl_s))

    async def clear(self, symbol: str) -> None:
        self.cleared.append(symbol)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Alerts:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, str, Severity]] = []

    def alert(self, text: str, *, key: str, severity: Severity = Severity.CRITICAL) -> bool:
        self.alerts.append((key, text, severity))
        return True


class Rig:
    """Un ``VpinTracker`` réel sur des connexions simulées ; ``scripts`` = les messages de chaque connexion successive."""

    def __init__(self, feed: Any = None, *, bucket: str = "100", window: int = 2, symbols: tuple[str, ...] = ("BTCUSDT",), **kwargs: Any) -> None:
        self.sink = RecordingSink()
        self.clock = Clock()
        self.scripts: list[list[str | Exception]] = []
        self.connect_errors: list[Exception] = []
        self.connections: list[ScriptConnection] = []
        self.on_connect: list[Any] = []
        defaults: dict[str, Any] = dict(recv_timeout_s=0.01, reconnect_min_s=0.01, reconnect_max_s=0.02, refresh_s=1000.0, ttl_s=2000.0, clock=self.clock)
        defaults.update(kwargs)
        self.tracker = VpinTracker(
            feed or BinanceAggTradeFeed(list(symbols)), {s: VolumeBucketVpin(D(bucket), window) for s in symbols}, self.sink, connect=self._connect, **defaults,
        )

    async def _connect(self, url: str) -> WsConnection:
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        for hook in self.on_connect:
            hook(len(self.connections))
        conn = ScriptConnection()
        for item in self.scripts.pop(0) if self.scripts else []:
            conn.inbox.put_nowait(item)
        self.connections.append(conn)
        return conn

    async def run_until(self, condition: Any, timeout_s: float = 3.0) -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(self.tracker.run(stop))
        deadline = time.monotonic() + timeout_s
        while not condition() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 3.0)


def test_the_tracker_publishes_the_vpin_only_once_the_window_is_full_with_the_ttl() -> None:
    # seau de 100 $, fenêtre de 2 ; 50 000 $ x 0,002 = 100 $ par trade : un seau par trade
    rig = Rig(ttl_s=120.0)
    rig.scripts = [[_agg(agg_id=1, maker=False), _agg(agg_id=2, maker=True), _agg(agg_id=3, maker=False)]]  # achat, vente, achat
    asyncio.run(rig.run_until(lambda: len(rig.sink.published) == 2))
    # après 2 seaux (imbalances 100 et 100) : 200 / 200 = 1 ; après le 3e : idem. Rien n'est publié au 1er seau.
    assert [(s, v, t) for s, v, t in rig.sink.published] == [("BTCUSDT", D(1), 120.0), ("BTCUSDT", D(1), 120.0)]
    assert rig.tracker.stats["buckets"] == 3 and rig.tracker.snapshot() == {"BTCUSDT": "1.000000"}


def test_balanced_buckets_give_a_low_vpin_written_with_six_decimals() -> None:
    rig = Rig(bucket="100", window=2)
    # seau 1 : 60 achat + 40 vente (imbalance 20) ; seau 2 : 50 / 50 (imbalance 0) -> (20 + 0) / 200 = 0,1
    rig.scripts = [[_agg(agg_id=1, price="1", qty="60", maker=False), _agg(agg_id=2, price="1", qty="40", maker=True),
                    _agg(agg_id=3, price="1", qty="50", maker=False), _agg(agg_id=4, price="1", qty="50", maker=True)]]
    asyncio.run(rig.run_until(lambda: bool(rig.sink.published)))
    assert [v for _, v, _ in rig.sink.published] == [D("0.1")] and rig.tracker.snapshot() == {"BTCUSDT": "0.100000"}


def test_replayed_malformed_and_unknown_symbol_messages_never_count() -> None:
    rig = Rig()
    rig.scripts = [["pas du json {", _agg(agg_id=5), _agg(agg_id=5), _agg(agg_id=4), _agg("ETHUSDT", agg_id=1), _agg(agg_id=6, price="x"), _agg(agg_id=7)]]
    asyncio.run(rig.run_until(lambda: rig.tracker.stats["trades"] == 2 and rig.tracker.stats["malformed"] == 2))
    assert rig.tracker.stats["duplicates"] == 2 and rig.tracker.stats["trades"] == 2  # 5 et 7 comptés ; 5 rejoué et 4 ancien écartés


def test_the_last_value_is_refreshed_while_connected_so_a_quiet_market_keeps_its_key() -> None:
    rig = Rig(refresh_s=0.05, ttl_s=1.0, clock=time.monotonic)  # horloge réelle : la ré-écriture dépend du temps qui passe
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)]]
    asyncio.run(rig.run_until(lambda: False, timeout_s=0.3))  # aucun nouveau trade : seules les ré-écritures publient
    values = {v for _, v, _ in rig.sink.published}
    assert len(rig.sink.published) >= 3 and values == {D(1)}  # la même valeur, ré-écrite


def test_a_short_outage_keeps_the_window_and_a_long_one_resets_it_and_deletes_the_key() -> None:
    short = Rig(max_gap_s=30.0)
    short.scripts = [[_agg(agg_id=1), _agg(agg_id=2), ConnectionResetError("coupée")], [_agg(agg_id=3, maker=True)]]
    short.on_connect = [lambda n: setattr(short.clock, "now", short.clock.now + 5.0) if n == 1 else None]  # 5 s de coupure
    asyncio.run(short.run_until(lambda: len(short.connections) == 2 and short.tracker.stats["buckets"] == 3))
    assert short.tracker.stats["resets"] == 0 and short.sink.cleared == []
    long_gap = Rig(max_gap_s=30.0)
    long_gap.scripts = [[_agg(agg_id=1), _agg(agg_id=2), ConnectionResetError("coupée")], [_agg(agg_id=3, maker=True)]]
    long_gap.on_connect = [lambda n: setattr(long_gap.clock, "now", long_gap.clock.now + 100.0) if n == 1 else None]  # 100 s de coupure
    asyncio.run(long_gap.run_until(lambda: len(long_gap.connections) == 2 and long_gap.tracker.stats["trades"] == 3))
    assert long_gap.tracker.stats["resets"] == 1 and long_gap.sink.cleared == ["BTCUSDT"]
    assert long_gap.tracker.snapshot() == {"BTCUSDT": None}  # fenêtre repartie de zéro : 1 seau sur 2, pas de valeur
    # les identifiants ont été oubliés avec la fenêtre : le trade 3 (id plus petit que 2 ? non, 3 > 2) est de toute façon compté


def test_a_failing_redis_never_stops_the_calculation_and_the_stream_keeps_being_read() -> None:
    rig = Rig()
    rig.sink.fail = True
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True), _agg(agg_id=3)]]
    asyncio.run(rig.run_until(lambda: rig.tracker.stats["trades"] == 3 and rig.tracker.stats["sink_errors"] >= 1))
    assert rig.tracker.stats["trades"] == 3 and rig.tracker.snapshot()["BTCUSDT"] is not None  # le calcul continue


def test_a_long_feed_outage_alerts_once_and_the_recovery_is_announced() -> None:
    alerts = Alerts()
    rig = Rig(alerter=alerts, alert_after_failures=3)
    rig.connect_errors = [ConnectionError("x")] * 5
    asyncio.run(rig.run_until(lambda: bool(rig.connections) and len(alerts.alerts) >= 2))
    assert [k for k, _, _ in alerts.alerts] == ["vpin_feed_down", "vpin_feed_up"] and alerts.alerts[1][2] is Severity.INFO


def test_the_heartbeat_is_sent_at_the_feeds_rhythm() -> None:
    class FastPing(HyperliquidTradesFeed):
        heartbeat_interval_s = 0.05

    rig = Rig(FastPing(["BTCUSDT"]), clock=time.monotonic)
    asyncio.run(rig.run_until(lambda: bool(rig.connections) and sum('"ping"' in m for m in rig.connections[0].sent) >= 2))
    assert sum('"ping"' in m for m in rig.connections[0].sent) >= 2 and any('"trades"' in m for m in rig.connections[0].sent)


# --- Serveur WebSocket local (client réel) ---------------------------------------------------


def test_the_real_websocket_client_reads_a_binance_like_stream_and_computes_the_expected_vpin() -> None:
    pytest.importorskip("websockets")
    from websockets.asyncio.server import serve

    from workers.common.ws_tracker import websockets_connect

    async def handler(ws: Any) -> None:
        for i, maker in enumerate((False, True, False, False), start=1):  # achat, vente, achat, achat ; 100 $ chacun
            await ws.send(_agg(agg_id=i, maker=maker))
        await asyncio.sleep(1)

    async def scenario() -> tuple[VpinTracker, RecordingSink]:
        async with serve(handler, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            sink = RecordingSink()
            feed = BinanceAggTradeFeed(["BTCUSDT"], f"ws://127.0.0.1:{port}")
            tracker = VpinTracker(feed, {"BTCUSDT": VolumeBucketVpin(D(100), 2)}, sink, connect=websockets_connect, recv_timeout_s=0.05, refresh_s=1000.0, ttl_s=2000.0)
            stop = asyncio.Event()
            task = asyncio.create_task(tracker.run(stop))
            for _ in range(200):
                if tracker.stats["buckets"] == 4:
                    break
                await asyncio.sleep(0.02)
            stop.set()
            await asyncio.wait_for(task, 5.0)
            return tracker, sink

    tracker, sink = asyncio.run(scenario())
    assert tracker.stats["buckets"] == 4 and tracker.stats["malformed"] == 0
    assert [v for _, v, _ in sink.published] == [D(1)] * 3  # imbalance 100 à chaque seau : (100 + 100) / 200 dès le 2e


# --- Redis -----------------------------------------------------------------------------------


def test_the_redis_sink_writes_the_documented_key_with_an_expiry_and_clears_it() -> None:
    async def scenario() -> tuple[str | None, int, str | None]:
        sink = RedisVpinSink(host="127.0.0.1", port=1, password="topsecret")
        assert "topsecret" not in repr(sink)
        fake = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
        sink._redis = fake  # noqa: SLF001
        await sink.publish("BTCUSDT", D("0.65342177"), 120.0)
        raw = await fake.get("market:vpin:BTCUSDT")
        value, ttl_ms = (raw.decode() if isinstance(raw, bytes) else raw), await fake.pttl("market:vpin:BTCUSDT")
        await sink.clear("BTCUSDT")
        return value, ttl_ms, await fake.get("market:vpin:BTCUSDT")

    value, ttl_ms, after = asyncio.run(scenario())
    assert value == "0.653422" and 0 < ttl_ms <= 120_000 and after is None  # six décimales, durée de vie, supprimée
    assert redis_key("ETHUSDT") == "market:vpin:ETHUSDT"


# --- Ligne de commande -----------------------------------------------------------------------


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run(coro: Any, **kwargs: Any) -> None:
        coro.close()  # on ne lance pas la boucle : on vérifie ce que la CLI a construit
        seen["ran"] = True

    original = vt.VpinTracker.__init__

    def capture(self: Any, feed: Any, calcs: Any, sink: Any, **kw: Any) -> None:
        seen.update(feed=feed, calcs=calcs, sink=sink, kwargs=kw)
        original(self, feed, calcs, sink, **kw)

    monkeypatch.setattr(asyncio, "run", fake_run)
    monkeypatch.setattr(vt.VpinTracker, "__init__", capture)
    monkeypatch.setattr(vt, "configure_logging", lambda *a, **k: None)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    assert vt.main(list(args)) == 0
    return seen


def test_the_command_line_builds_the_documented_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_HOST", "redis-broker")
    seen = _run_cli(monkeypatch, "--dry-run")
    assert isinstance(seen["feed"], BinanceAggTradeFeed) and isinstance(seen["sink"], vt.LogOnlySink) and seen["ran"]
    assert sorted(seen["calcs"]) == ["BTCUSDT", "ETHUSDT"]
    calc = seen["calcs"]["BTCUSDT"]
    assert (calc.bucket_size, calc.window_size) == (D(100_000), 50)  # VOLUME_BUCKET_SIZE et WINDOW_SIZE par défaut
    custom = _run_cli(monkeypatch, "--dry-run", "--source", "hyperliquid", "--symbols", "btcusdt", "--bucket-size", "5000", "--window", "20")
    assert isinstance(custom["feed"], HyperliquidTradesFeed) and custom["calcs"]["BTCUSDT"].window_size == 20 and custom["calcs"]["BTCUSDT"].bucket_size == D(5000)


def test_the_command_line_writes_to_redis_unless_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REDIS_HOST", "redis-broker")
    monkeypatch.setenv("REDIS_PASSWORD", "pw")
    monkeypatch.setenv("NOTHIERLY_DATA_DIR", str(tmp_path))  # l'archive SQLite s'ouvre au démarrage : jamais dans le dossier du dépôt
    seen = _run_cli(monkeypatch)
    assert isinstance(seen["sink"], RedisVpinSink)
    seen["kwargs"]["archive"].close()


@pytest.mark.parametrize("args", [["--bucket-size", "0"], ["--window", "0"], ["--ttl-s", "10", "--refresh-s", "30"], ["--symbols", "btc/usdt"], ["--symbols", "bad symbol"]])
def test_the_command_line_refuses_invalid_settings(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(vt, "configure_logging", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        vt.main(["--dry-run", *args])


# --- Moteur : lecture par la Porte 3 ---------------------------------------------------------


def _vpin_artifact(tmp_path: Path, features: tuple[str, ...]) -> Path:
    """Un modèle dont le résultat dépend UNIQUEMENT du VPIN (dernière variable) : le gain arrive quand le flux est peu toxique."""
    from tests.test_metalabel_pipeline import tm

    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=(500, len(features)))
    y = (x[:, features.index("vpin")] < 0.5).astype(np.int64)
    out = tmp_path / "metalabel_weights.json"
    tm.save_artifact(tm.fit_model(x, y), out, {"version": 3, "feature_names": list(features), "payoff_ratio": 1.0})
    return out


async def _set_vpin(agent: Any, symbol: str, value: str) -> None:
    await agent._redis_client.execute("set", f"market:vpin:{symbol}", value)


@pytest.mark.asyncio
async def test_the_gate_reads_market_vpin_for_crypto_instruments_only(agent_factory: Any) -> None:
    agent = agent_factory(None)
    await _set_vpin(agent, "BTCUSDT", "0.42")
    assert await agent._read_vpin("BTC/USDT") == pytest.approx(0.42)
    assert await agent._read_vpin("ETH/USDT") is None  # aucune clé
    assert await agent._read_vpin("EUR_USD") is None  # pas un instrument crypto : pas de lecture


@pytest.mark.asyncio
async def test_a_missing_key_falls_back_to_the_last_value_read_for_five_minutes_then_to_no_reading(agent_factory: Any) -> None:
    agent = agent_factory(None)
    await _set_vpin(agent, "BTCUSDT", "0.30")
    assert await agent._read_vpin("BTC/USDT") == pytest.approx(0.30)
    await agent._redis_client.execute("delete", "market:vpin:BTCUSDT")  # clé expirée : le démon est tombé
    assert await agent._read_vpin("BTC/USDT") == pytest.approx(0.30)  # la dernière valeur lue, encore récente
    value, _ = agent._vpin_cache["BTC/USDT"]
    agent._vpin_cache["BTC/USDT"] = (value, time.monotonic() - 301.0)
    assert await agent._read_vpin("BTC/USDT") is None  # trop ancienne : plus de lecture, jamais une valeur inventée


@pytest.mark.asyncio
@pytest.mark.parametrize("garbage", ["abc", "nan", "inf", "1.5", "-0.1", ""])
async def test_an_unusable_value_is_ignored_and_never_replaces_a_good_cached_one(agent_factory: Any, garbage: str) -> None:
    agent = agent_factory(None)
    await _set_vpin(agent, "BTCUSDT", "0.25")
    assert await agent._read_vpin("BTC/USDT") == pytest.approx(0.25)
    await _set_vpin(agent, "BTCUSDT", garbage)
    assert await agent._read_vpin("BTC/USDT") == pytest.approx(0.25)  # retombe sur la valeur en cache, pas sur la valeur douteuse
    fresh = agent_factory(None)
    await _set_vpin(fresh, "BTCUSDT", garbage)
    assert await fresh._read_vpin("BTC/USDT") is None


@pytest.mark.asyncio
async def test_a_model_that_declares_vpin_uses_it_and_answers_neutrally_without_a_reading(agent_factory: Any, tmp_path: Path) -> None:
    from tests.test_metalabel_pipeline import _feed_market, _signals

    agent = agent_factory(_vpin_artifact(tmp_path, FEATURES_V3))
    _feed_market(agent)
    calm = agent._meta_label_assess("BTC/USDT", _signals(), vpin=0.1)
    toxic = agent._meta_label_assess("BTC/USDT", _signals(), vpin=0.9)
    assert calm.fitted and calm.features["vpin"] == 0.1 and calm.confidence > 0.8 > 0.2 > toxic.confidence
    assert calm.multiplier > 1.0 > toxic.multiplier
    blind = agent._meta_label_assess("BTC/USDT", _signals(), vpin=None)
    assert (blind.multiplier, blind.confidence) == (1.0, 0.5) and "vpin" not in blind.features  # pas de 0,5 inventé : réponse neutre


@pytest.mark.asyncio
async def test_a_model_without_vpin_in_its_features_ignores_it_so_older_artifacts_keep_working(agent_factory: Any, tmp_path: Path) -> None:
    from tests.test_metalabel_pipeline import _feed_market, _signals, _train_artifact

    agent = agent_factory(_train_artifact(tmp_path))  # modèle v2
    _feed_market(agent)
    with_vpin = agent._meta_label_assess("BTC/USDT", _signals(ppo=0.95), vpin=0.9)
    without = agent._meta_label_assess("BTC/USDT", _signals(ppo=0.95), vpin=None)
    assert with_vpin.confidence == without.confidence and with_vpin.multiplier == without.multiplier


@pytest.mark.asyncio
async def test_gate_3_reads_the_key_from_redis_sizes_with_it_and_logs_it_in_the_decision(agent_factory: Any, tmp_path: Path) -> None:
    from tests.test_metalabel_pipeline import _capture_publish, _feed_market, _signals

    agent = agent_factory(_vpin_artifact(tmp_path, FEATURES_V3))
    sent = _capture_publish(agent)
    _feed_market(agent)
    await _set_vpin(agent, "BTCUSDT", "0.05")
    result, units = await agent._gate_3_sizing("BTC/USDT", 50_000.0, 49_500.0, 10_000.0, _signals())
    base = await agent.calculate_position_size("BTC/USDT", 50_000.0, 49_500.0, 10_000.0)
    assert result.status == "PASSED" and units > base  # flux peu toxique : taille majorée
    assert sent[-1][1]["features"]["vpin"] == pytest.approx(0.05)
    await _set_vpin(agent, "BTCUSDT", "0.95")
    result2, units2 = await agent._gate_3_sizing("BTC/USDT", 50_000.0, 49_500.0, 10_000.0, _signals())
    assert result2.status == "FAILED" or units2 < base  # flux très toxique : taille réduite ou veto


@pytest.mark.asyncio
async def test_a_manual_override_reads_no_vpin(agent_factory: Any) -> None:
    agent = agent_factory(None)

    async def forbidden(*a: Any, **k: Any) -> None:
        raise AssertionError("aucune lecture VPIN pour un ordre manuel")

    agent._read_vpin = forbidden
    result, _ = await agent._gate_3_sizing("BTC/USDT", 50_000.0, 49_500.0, 10_000.0, None)
    assert result.status == "PASSED"


# --- Entraînement ----------------------------------------------------------------------------


def test_the_training_script_can_learn_the_vpin_when_decisions_carry_it_and_drops_older_ones_otherwise(tmp_path: Path) -> None:
    from tests.test_metalabel_pipeline import tm, _fill

    rng = np.random.default_rng(11)
    decisions, fills = [], []
    for i in range(600):
        entry = 1_760_000_000 + i * 1000
        vpin = float(rng.uniform(0, 1))
        features = {n: float(rng.uniform(0, 1)) for n in FEATURES_V2}
        if i % 3:  # un tiers des décisions n'a pas de VPIN (écrites avant son arrivée)
            features["vpin"] = vpin
        win = vpin + rng.normal(0, 0.08) < 0.5
        decisions.append({"decision_time": entry - 10.0, "instrument": "BTC/USDT", "direction": 1, "features": features})
        fills += [_fill(entry * 1000, "BUY", 1, 100.0, fid=f"b{i}"), _fill(entry * 1000 + 60_000, "SELL", 1, 101.0 if win else 99.0, fid=f"s{i}")]
    code, report, details = tm.run(decisions, fills, out=tmp_path / "w.json", feature_names=FEATURES_V3, min_trades=300, min_test=50)
    assert code == 0 and report.rows_missing_features == 200 and details["n_labeled"] == 400
    assert details["metrics"]["auc"] > 0.85
    model = MetaLabelModel.load_or_skeleton(tmp_path / "w.json")
    assert model.feature_names == FEATURES_V3 and model.is_fitted
    assert FEATURES_V3 == (*FEATURES_V2, "vpin") and sys.modules  # le v2 reste inchangé

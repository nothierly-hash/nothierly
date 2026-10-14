"""Tests de la mesure du slippage : ``workers/common/fills_db.py``, ``ws_tracker.py`` et leur intégration aux workers.

Preuves recherchées :

- le calcul est juste (chiffres faits à la main) et son signe est sans ambiguïté (positif = coût) ;
- un fill n'est jamais compté deux fois (reconnexion, instantané, message rejoué) et jamais apparié à tort ;
- un fill qui arrive AVANT la réponse de l'API (Hyperliquid n'a pas de ``cloid`` dans ses fills) est quand même apparié ;
- la boucle ``asyncio`` ne s'arrête pas pendant que la base écrit, et une panne du flux ou de la base ne touche pas l'exécution ;
- le client WebSocket réel parle correctement à un serveur local qui vérifie la signature Bybit.

Aucun service externe : faux flux en mémoire, serveur WebSocket sur ``127.0.0.1``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from collections.abc import Callable, Coroutine
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from workers.common.alerts import Severity
from workers.common.base import ExposureWorkerBase
from workers.common.fills_db import (
    Execution,
    FillsDatabase,
    OrderIntent,
    slippage_bps,
    summarize,
)
from workers.common.ratelimit import RateLimitGate
from workers.common.ws_tracker import (
    BybitFillFeed,
    FeedError,
    FillEvent,
    FillTracker,
    HyperliquidFillFeed,
    WsConnection,
    websockets_connect,
)

D = Decimal
T0 = 1_760_000_000_000


# --- Calcul ----------------------------------------------------------------------------------


def test_slippage_is_positive_when_the_fill_is_worse_than_intended_for_both_sides() -> None:
    assert slippage_bps(True, D(100), D("100.05")) == D(5)  # achat 5 bps plus cher que voulu : coût
    assert slippage_bps(False, D(100), D("99.95")) == D(5)  # vente 5 bps moins chère que voulu : coût
    assert slippage_bps(True, D(100), D("99.90")) == D(-10)  # achat moins cher : amélioration
    assert slippage_bps(False, D(100), D("100.10")) == D(-10)
    assert slippage_bps(True, D(50000), D(50000)) == D(0)


def test_slippage_is_undefined_without_a_positive_intended_price() -> None:
    assert slippage_bps(True, D(0), D(100)) is None and slippage_bps(False, D(-1), D(100)) is None


def _exec(slip: float | None, price: float = 100.0, qty: float = 1.0, fill_id: str = "f", venue: str = "BYBIT", symbol: str = "BTCUSDT") -> Execution:
    return Execution(T0, venue, symbol, "BUY", None if slip is None else 100.0, price, qty, slip, "o", fill_id, "signal" if slip is not None else "unmatched", None, None)


def test_summary_statistics_match_a_hand_computation() -> None:
    rows = [_exec(5.0, fill_id="1"), _exec(-2.0, fill_id="2"), _exec(10.0, fill_id="3"), _exec(3.0, fill_id="4")]
    s = summarize(rows, unmatched=2)
    assert (s.fills, s.unmatched, s.notional) == (4, 2, 400.0)
    assert s.mean_bps == 4.0 and s.median_bps == 4.0 and s.worst_bps == 10.0 and s.p95_bps == 10.0  # tri : -2, 3, 5, 10
    assert s.qty_weighted_mean_bps == 4.0  # notionnels égaux : même valeur que la moyenne simple


def test_the_weighted_mean_gives_big_fills_more_weight_and_an_empty_summary_has_no_numbers() -> None:
    s = summarize([_exec(10.0, qty=9.0, fill_id="1"), _exec(0.0, qty=1.0, fill_id="2")])
    assert s.mean_bps == 5.0 and s.qty_weighted_mean_bps == 9.0  # (10 x 900 + 0 x 100) / 1000
    empty = summarize([], unmatched=3)
    assert (empty.fills, empty.unmatched, empty.mean_bps, empty.worst_bps) == (0, 3, None, None)


# --- Base ------------------------------------------------------------------------------------


def _intent(client_id: str = "c1", order_id: str | None = None, *, is_buy: bool = True, price: str = "100", venue: str = "BYBIT", kind: str = "signal", at: int = T0) -> OrderIntent:
    return OrderIntent(venue, client_id, order_id, "BTCUSDT", is_buy, D(price), D("0.5"), kind, at)


def test_an_intent_is_found_by_client_id_first_then_by_the_platform_order_id() -> None:
    db = FillsDatabase()
    db.register_intent(_intent("c1"))
    assert db.find_intent("BYBIT", client_id="c1") == _intent("c1")
    assert db.find_intent("BYBIT", client_id="c1", order_id="nope") is not None
    assert db.find_intent("BYBIT", order_id="X1") is None  # pas encore rattaché
    assert db.attach_order_id("BYBIT", "c1", "X1") is True
    found = db.find_intent("BYBIT", order_id="X1")
    assert found is not None and found.client_id == "c1" and found.order_id == "X1"
    assert db.find_intent("HYPERLIQUID", client_id="c1") is None  # jamais d'appariement entre plateformes
    assert db.attach_order_id("BYBIT", "inconnu", "Y") is False


def test_a_fill_row_is_inserted_once_and_the_intent_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "f.sqlite3"
    first = FillsDatabase(path)
    first.register_intent(_intent("c1", "X1"))
    row = _exec(5.0, fill_id="fill-1")
    assert first.insert_execution(row) is True and first.insert_execution(row) is False  # idempotent
    first.close()
    second = FillsDatabase(path)  # redémarrage du worker
    assert second.find_intent("BYBIT", order_id="X1") is not None and second.has_fill("BYBIT", "fill-1")
    assert len(second.executions()) == 1
    second.close()


def test_old_intents_can_be_pruned_and_queries_filter_by_venue_symbol_and_time() -> None:
    db = FillsDatabase()
    db.register_intent(_intent("old", at=T0 - 10_000))
    db.register_intent(_intent("new", at=T0))
    assert db.prune_intents_before(T0 - 1) == 1 and db.find_intent("BYBIT", client_id="old") is None
    db.insert_execution(_exec(1.0, fill_id="a", venue="BYBIT", symbol="BTCUSDT"))
    db.insert_execution(_exec(2.0, fill_id="b", venue="HYPERLIQUID", symbol="ETH"))
    assert [e.fill_id for e in db.executions(venue="HYPERLIQUID")] == ["b"]
    assert [e.fill_id for e in db.executions(symbol="BTCUSDT")] == ["a"]
    assert db.executions(since_ms=T0 + 1) == []
    assert db.summary(venue="BYBIT").fills == 1


# --- Parseurs --------------------------------------------------------------------------------

HL_FILL = {"coin": "BTC", "px": "50025.5", "sz": "0.01", "side": "B", "time": T0, "startPosition": "0", "dir": "Open Long", "closedPnl": "0",
           "hash": "0xabc", "oid": 777, "crossed": True, "fee": "0.17", "tid": 4242, "feeToken": "USDC"}


def _hl_message(*fills: dict[str, Any], snapshot: bool = False) -> str:
    return json.dumps({"channel": "userFills", "data": {"isSnapshot": snapshot, "user": "0xabc", "fills": list(fills)}})


def test_hyperliquid_fills_are_parsed_and_other_channels_ignored() -> None:
    feed = HyperliquidFillFeed(user="0x" + "1" * 40, network="testnet")
    (event,) = feed.parse(_hl_message(HL_FILL))
    assert (event.symbol, event.is_buy, event.price, event.qty, event.time_ms) == ("BTC", True, D("50025.5"), D("0.01"), T0)
    assert (event.fill_id, event.order_id, event.client_id, event.fee, event.fee_currency, event.snapshot) == ("4242", "777", None, D("0.17"), "USDC", False)
    (sell,) = feed.parse(_hl_message({**HL_FILL, "side": "A", "tid": 1}, snapshot=True))
    assert sell.is_buy is False and sell.snapshot is True
    assert feed.parse(json.dumps({"channel": "pong"})) == [] and feed.parse(json.dumps({"channel": "subscriptionResponse", "data": {}})) == []
    with pytest.raises(ValueError):
        feed.parse(_hl_message({**HL_FILL, "side": "?"}))
    with pytest.raises(ValueError):
        feed.parse(_hl_message({**HL_FILL, "px": "pas un nombre"}))


def test_the_hyperliquid_urls_and_subscription() -> None:
    assert HyperliquidFillFeed(user="u", network="mainnet").url == "wss://api.hyperliquid.xyz/ws"
    assert HyperliquidFillFeed(user="u", network="testnet").url == "wss://api.hyperliquid-testnet.xyz/ws"

    class Recorder:
        sent: list[str]

        def __init__(self) -> None:
            self.sent = []

        async def send(self, text: str) -> None:
            self.sent.append(text)

    rec = Recorder()
    asyncio.run(HyperliquidFillFeed(user="0xUSER").handshake(rec))  # type: ignore[arg-type]  # le faux ne fournit que send(), seule méthode appelée par le handshake
    assert json.loads(rec.sent[0]) == {"method": "subscribe", "subscription": {"type": "userFills", "user": "0xUSER"}}


BY_TRADE = {"category": "linear", "symbol": "BTCUSDT", "orderId": "ord-1", "orderLinkId": "link-1", "side": "Buy", "execId": "e-1",
            "execPrice": "50010.5", "execQty": "0.01", "execTime": str(T0), "execType": "Trade", "isMaker": False, "execFee": "0.275", "feeCurrency": "USDT"}


def _by_message(*items: dict[str, Any]) -> str:
    return json.dumps({"id": "x", "topic": "execution.linear", "creationTime": T0, "data": list(items)})


def test_bybit_trades_are_parsed_and_funding_and_other_exec_types_are_not_fills() -> None:
    feed = BybitFillFeed(api_key="k", api_secret="s", network="testnet")
    (event,) = feed.parse(_by_message(BY_TRADE, {**BY_TRADE, "execId": "e-2", "execType": "Funding"}, {**BY_TRADE, "execId": "e-3", "execType": "BustTrade"}))
    assert (event.symbol, event.is_buy, event.price, event.qty, event.time_ms, event.fill_id) == ("BTCUSDT", True, D("50010.5"), D("0.01"), T0, "e-1")
    assert (event.order_id, event.client_id, event.fee, event.fee_currency) == ("ord-1", "link-1", D("0.275"), "USDT")
    (manual,) = feed.parse(_by_message({**BY_TRADE, "orderLinkId": "", "side": "Sell"}))
    assert manual.client_id is None and manual.is_buy is False  # ordre sans identifiant client : apparié par orderId seulement
    assert feed.parse(json.dumps({"op": "pong"})) == [] and feed.parse(json.dumps({"topic": "order", "data": []})) == []
    assert feed.url == "wss://stream-testnet.bybit.com/v5/private" and BybitFillFeed(api_key="k", api_secret="s", network="mainnet").url == "wss://stream.bybit.com/v5/private"


class ScriptConnection:
    """Faux flux : ``push`` met un message en file, ``recv`` les rend dans l'ordre ; ``close_with`` fait échouer la lecture."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.inbox: asyncio.Queue[str | Exception] = asyncio.Queue()
        self.closed = False

    def push(self, message: str | Exception | dict[str, Any]) -> None:
        self.inbox.put_nowait(message if isinstance(message, (str, Exception)) else json.dumps(message))

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def recv(self) -> str:
        item = await self.inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


def test_bybit_authentication_signs_get_realtime_with_the_expiry_and_never_shows_the_secret() -> None:
    feed = BybitFillFeed(api_key="KEY", api_secret="topsecret-value", network="testnet", clock_ms=lambda: 1_000_000)
    conn = ScriptConnection()
    conn.push({"op": "auth", "success": True})
    conn.push({"op": "subscribe", "success": True})
    asyncio.run(feed.handshake(conn))
    auth = json.loads(conn.sent[0])
    expires = 1_010_000
    expected = hmac.new(b"topsecret-value", f"GET/realtime{expires}".encode(), hashlib.sha256).hexdigest()
    assert auth == {"op": "auth", "args": ["KEY", expires, expected]}
    assert json.loads(conn.sent[1]) == {"op": "subscribe", "args": ["execution.linear"]}
    assert "topsecret-value" not in repr(feed) and "KEY" not in repr(feed)


def test_a_refused_authentication_or_a_silent_server_fails_the_handshake_without_leaking_the_secret() -> None:
    feed = BybitFillFeed(api_key="KEY", api_secret="topsecret-value", ack_timeout_s=0.2)
    refused = ScriptConnection()
    refused.push({"op": "auth", "success": False, "ret_msg": "invalid signature"})
    with pytest.raises(FeedError) as exc:
        asyncio.run(feed.handshake(refused))
    assert "topsecret-value" not in str(exc.value) and "auth" in str(exc.value)
    with pytest.raises((FeedError, asyncio.TimeoutError)):
        asyncio.run(feed.handshake(ScriptConnection()))  # rien ne répond


# --- Tracker sur faux flux -------------------------------------------------------------------


class Harness:
    """Lance un ``FillTracker`` réel sur une suite de connexions simulées ; ``clock`` est maîtrisable pour le délai de grâce."""

    def __init__(self, feed: Any, db: FillsDatabase | None = None, **tracker_kwargs: Any) -> None:
        self.db = db or FillsDatabase()
        self.connections: list[ScriptConnection] = []
        self.connect_errors: list[Exception] = []
        self.scripts: list[list[Any]] = []
        self.now = T0
        defaults: dict[str, Any] = dict(recv_timeout_s=0.01, reconnect_min_s=0.01, reconnect_max_s=0.02, match_grace_s=0.5, clock_ms=lambda: self.now)
        defaults.update(tracker_kwargs)
        self.tracker = FillTracker(feed, self.db, connect=self._connect, **defaults)

    async def _connect(self, url: str) -> WsConnection:
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        conn = ScriptConnection()
        for scripted in self.scripts.pop(0) if self.scripts else []:  # un script par connexion, vide ensuite
            conn.push(scripted)
        self.connections.append(conn)
        return conn

    def script(self, *messages: str | dict[str, Any]) -> None:
        """Messages de la PREMIÈRE connexion ; les suivantes (après une coupure) démarrent vides."""
        self.scripts = [list(messages)]

    async def run_until(self, condition: Callable[[], bool], timeout_s: float = 3.0) -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(self.tracker.run(stop))
        deadline = time.monotonic() + timeout_s
        while not condition() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 3.0)


def _go(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def test_a_bybit_fill_is_matched_by_its_link_id_and_written_with_its_slippage() -> None:
    h = Harness(BybitFillFeed(api_key="k", api_secret="s"), )
    h.db.register_intent(_intent("link-1", price="50000"))
    h.script({"op": "auth", "success": True}, {"op": "subscribe", "success": True}, _by_message(BY_TRADE))
    _go(h.run_until(lambda: len(h.db.executions()) == 1))
    (row,) = h.db.executions()
    assert (row.venue, row.symbol, row.direction, row.kind, row.order_id, row.fill_id) == ("BYBIT", "BTCUSDT", "BUY", "signal", "ord-1", "e-1")
    assert (row.intended_price, row.fill_price, row.fill_qty) == (50000.0, 50010.5, 0.01)
    assert row.slippage_bps == pytest.approx(2.1)  # (50010,5 - 50000) / 50000 x 10 000
    assert (row.fee, row.fee_currency) == (0.275, "USDT") and h.connections[0].closed


def test_a_hyperliquid_fill_that_arrives_before_the_order_id_is_known_is_matched_once_it_is() -> None:
    """Le cas réel : le fill passe sur le flux avant que la réponse de l'API ait donné l'``oid``."""
    h = Harness(HyperliquidFillFeed(user="0xabc"), match_grace_s=5.0)
    h.db.register_intent(_intent("cloid-1", venue="HYPERLIQUID", price="50000"))  # avant l'envoi, sans oid
    h.script(_hl_message(HL_FILL))

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(h.tracker.run(stop))
        await asyncio.sleep(0.2)
        assert h.db.executions() == [] and h.tracker.stats["received"] == 1  # reçu, en attente : pas encore d'intention rattachée
        h.db.attach_order_id("HYPERLIQUID", "cloid-1", "777")  # la réponse de l'API arrive
        for _ in range(100):
            if h.db.executions():
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(task, 3.0)

    _go(scenario())
    (row,) = h.db.executions()
    assert row.kind == "signal" and row.intended_price == 50000.0 and row.slippage_bps == pytest.approx(5.1)  # (50025,5 - 50000) / 50000 x 10 000 = 5,1
    assert h.tracker.stats["matched"] == 1 and h.tracker.stats["unmatched"] == 0


def test_a_fill_with_no_known_intent_is_recorded_as_unmatched_after_the_grace_never_as_a_zero() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"), match_grace_s=0.5)
    h.script(_hl_message(HL_FILL))

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(h.tracker.run(stop))
        await asyncio.sleep(0.2)
        assert h.db.executions() == []  # encore dans le délai de grâce
        h.now += 1000  # le délai de grâce (0,5 s) est écoulé
        for _ in range(100):
            if h.db.executions():
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(task, 3.0)

    _go(scenario())
    (row,) = h.db.executions()
    assert (row.kind, row.intended_price, row.slippage_bps) == ("unmatched", None, None)
    assert h.db.summary().fills == 0 and h.db.summary().unmatched == 1  # compté à part, jamais moyenné


def test_replayed_fills_are_counted_once() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    h.script(_hl_message(HL_FILL), _hl_message(HL_FILL), _hl_message(HL_FILL, snapshot=True))
    _go(h.run_until(lambda: h.tracker.stats["received"] >= 3 and h.tracker.stats["duplicates"] >= 2))
    assert len(h.db.executions()) == 1 and h.tracker.stats["duplicates"] == 2


def test_a_snapshot_recovers_missed_fills_with_a_known_intent_and_ignores_foreign_history() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    foreign = {**HL_FILL, "oid": 1, "tid": 1}  # un ancien fill qui n'est pas à nous
    h.script(_hl_message(foreign, HL_FILL, snapshot=True))
    _go(h.run_until(lambda: len(h.db.executions()) == 1 and h.tracker.stats["received"] == 2))
    assert [e.fill_id for e in h.db.executions()] == ["4242"] and h.tracker.stats["unmatched"] == 0


def test_a_malformed_message_is_counted_and_the_next_one_is_still_recorded() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    h.script("pas du json {", _hl_message({**HL_FILL, "px": "x"}), _hl_message(HL_FILL))
    _go(h.run_until(lambda: len(h.db.executions()) == 1))
    assert h.tracker.stats["malformed"] == 2 and len(h.db.executions()) == 1


def test_the_tracker_reconnects_with_backoff_after_a_failure_and_after_a_drop() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    h.connect_errors = [ConnectionError("refusée"), TimeoutError("lente")]
    h.script(_hl_message(HL_FILL))
    _go(h.run_until(lambda: len(h.db.executions()) == 1))
    assert h.tracker.stats["reconnects"] == 2 and len(h.connections) == 1
    drop = Harness(HyperliquidFillFeed(user="0xabc"))
    drop.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    drop.scripts = [[ConnectionResetError("coupée")], [_hl_message(HL_FILL)]]  # 1re connexion : coupée ; 2e : le fill
    _go(drop.run_until(lambda: len(drop.db.executions()) == 1))
    assert len(drop.connections) >= 2 and drop.tracker.stats["reconnects"] >= 1  # reconnecté, fill de la 2e connexion enregistré


class RecordingAlerter:
    def __init__(self) -> None:
        self.alerts: list[tuple[str, str, Severity]] = []

    def alert(self, text: str, *, key: str, severity: Severity = Severity.CRITICAL) -> bool:
        self.alerts.append((key, text, severity))
        return True


def test_the_tracker_prunes_week_old_intents_at_startup_and_keeps_recent_ones() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("old", at=T0 - 8 * 24 * 3600 * 1000))
    h.db.register_intent(_intent("recent", at=T0 - 3600 * 1000))
    h.script()
    _go(h.run_until(lambda: bool(h.connections)))
    assert h.db.find_intent("BYBIT", client_id="old") is None and h.db.find_intent("BYBIT", client_id="recent") is not None


def test_a_long_outage_alerts_once_and_the_recovery_is_announced() -> None:
    rec = RecordingAlerter()
    h = Harness(HyperliquidFillFeed(user="0xabc"), alerter=rec, alert_after_failures=3)
    h.connect_errors = [ConnectionError("x")] * 5
    h.script()
    _go(h.run_until(lambda: bool(h.connections) and len(rec.alerts) >= 2))
    assert [k for k, _, _ in rec.alerts] == ["fill_tracker_down", "fill_tracker_up"]  # une alerte pour 5 échecs, puis le retour
    assert "n'est pas touchée" in rec.alerts[0][1] and rec.alerts[1][2] is Severity.INFO


def test_the_heartbeat_is_sent_at_the_feeds_rhythm() -> None:
    class FastFeed(HyperliquidFillFeed):
        heartbeat_interval_s = 0.05

    h = Harness(FastFeed(user="0xabc"))
    h.script()
    _go(h.run_until(lambda: bool(h.connections) and sum('"ping"' in m for m in h.connections[0].sent) >= 2))
    assert sum('"ping"' in m for m in h.connections[0].sent) >= 2


def test_the_event_loop_keeps_running_while_the_database_is_slow() -> None:
    """Preuve de « ne bloque pas la boucle » : une écriture de 0,4 s ne gèle pas une tâche concurrente qui bat toutes les 10 ms."""
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    original = h.db.insert_execution

    def slow_insert(execution: Execution, *, now_ms: int | None = None) -> bool:
        time.sleep(0.4)
        return original(execution, now_ms=now_ms)

    h.db.insert_execution = slow_insert  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    h.script(_hl_message(HL_FILL))
    gaps: list[float] = []

    async def scenario() -> None:
        async def ticker(stop: asyncio.Event) -> None:
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(0.01)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        stop = asyncio.Event()
        t1 = asyncio.create_task(h.tracker.run(stop))
        t2 = asyncio.create_task(ticker(stop))
        for _ in range(300):
            if h.db.executions():
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.gather(t1, t2)

    _go(scenario())
    assert len(h.db.executions()) == 1
    assert max(gaps) < 0.2  # sans thread, l'insertion de 0,4 s aurait produit un trou d'au moins 0,4 s


def test_a_failing_database_does_not_stop_the_listener() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    h.db.register_intent(_intent("c", "777", venue="HYPERLIQUID", price="50000"))
    calls = {"n": 0}
    original = h.db.insert_execution

    def flaky(execution: Execution, *, now_ms: int | None = None) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disque plein")
        return original(execution, now_ms=now_ms)

    h.db.insert_execution = flaky  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    h.script(_hl_message(HL_FILL), _hl_message({**HL_FILL, "tid": 99}))
    _go(h.run_until(lambda: len(h.db.executions()) == 1))
    assert [e.fill_id for e in h.db.executions()] == ["99"]  # le premier a échoué, le suivant est passé


# --- Serveur WebSocket local (client réel) ---------------------------------------------------


def test_the_real_websocket_client_authenticates_against_a_local_bybit_like_server_and_records_a_fill() -> None:
    pytest.importorskip("websockets")
    from websockets.asyncio.server import serve

    secret = "local-secret"
    seen: dict[str, Any] = {"auth_ok": False, "pings": 0}

    async def handler(ws: Any) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            if msg["op"] == "auth":
                key, expires, signature = msg["args"]
                expected = hmac.new(secret.encode(), f"GET/realtime{expires}".encode(), hashlib.sha256).hexdigest()
                seen["auth_ok"] = key == "KEY" and hmac.compare_digest(signature, expected)
                await ws.send(json.dumps({"op": "auth", "success": seen["auth_ok"]}))
            elif msg["op"] == "subscribe":
                await ws.send(json.dumps({"op": "subscribe", "success": True}))
                await ws.send(_by_message(BY_TRADE))
            elif msg["op"] == "ping":
                seen["pings"] += 1

    async def scenario() -> list[Execution]:
        async with serve(handler, "127.0.0.1", 0) as server:
            port = list(server.sockets)[0].getsockname()[1]
            db = FillsDatabase()
            db.register_intent(_intent("link-1", price="50000", at=int(time.time() * 1000)))  # horloge réelle : le tracker purge les intentions de plus de 7 jours
            feed = BybitFillFeed(api_key="KEY", api_secret=secret, url=f"ws://127.0.0.1:{port}")
            tracker = FillTracker(feed, db, connect=websockets_connect, recv_timeout_s=0.05)
            stop = asyncio.Event()
            task = asyncio.create_task(tracker.run(stop))
            for _ in range(200):
                if db.executions():
                    break
                await asyncio.sleep(0.02)
            stop.set()
            await asyncio.wait_for(task, 5.0)
            return db.executions()

    rows = asyncio.run(scenario())
    assert seen["auth_ok"] is True and len(rows) == 1 and rows[0].slippage_bps == pytest.approx(2.1)


# --- Intégration aux workers -----------------------------------------------------------------


def test_the_hyperliquid_worker_records_the_intent_before_sending_and_attaches_the_oid(tmp_path: Path) -> None:
    from tests.test_hyperliquid_worker import FakeHl, _run, _signal, _worker

    db = FillsDatabase()
    client = FakeHl()
    seen_before_send: list[OrderIntent | None] = []
    original = client.place_order

    def spy(request: Any) -> Any:
        seen_before_send.append(db.find_intent("HYPERLIQUID", client_id=request.cloid))  # l'intention doit déjà exister ici
        return original(request)

    client.place_order = spy  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    w = _worker(tmp_path, client, fills_db=db)
    assert _run(w, _signal()) == "filled"
    (before,) = seen_before_send
    assert before is not None and before.order_id is None  # écrite AVANT l'envoi, sans oid
    (req,) = client.orders
    after = db.find_intent("HYPERLIQUID", client_id=req.cloid)
    assert after is not None and after.order_id == "7"  # complétée à la réponse
    assert (after.symbol, after.is_buy, after.intended_price, after.intended_qty, after.kind) == ("BTC", True, D(50000), D("0.01"), "signal")


def test_dry_run_and_a_missing_database_record_nothing_and_a_broken_database_never_blocks_the_order(tmp_path: Path) -> None:
    from tests.test_hyperliquid_worker import FakeHl, _run, _signal, _worker

    db = FillsDatabase()
    dry = _worker(tmp_path, FakeHl(), fills_db=db, dry_run=True)
    assert _run(dry, _signal()) == "dry_run" and db.find_intent("HYPERLIQUID", client_id="x") is None
    with db._lock:  # noqa: SLF001 -- aucune ligne d'intention après un dry-run
        assert db._db.execute("SELECT COUNT(*) FROM order_intents").fetchone()[0] == 0
    plain = tmp_path / "a"
    plain.mkdir()
    assert _run(_worker(plain, FakeHl()), _signal()) == "filled"  # pas de base : rien à signaler
    broken = FillsDatabase()
    broken.close()  # toute écriture lèvera
    other = tmp_path / "b"
    other.mkdir()
    client2 = FakeHl()
    assert _run(_worker(other, client2, fills_db=broken), _signal()) == "filled" and len(client2.orders) == 1  # l'ordre part malgré tout


def test_the_hyperliquid_emergency_flatten_is_tracked_as_an_emergency_intent(tmp_path: Path) -> None:
    import asyncio as aio

    from tests.test_hyperliquid_worker import FakeHl, _worker

    db = FillsDatabase()
    client = FakeHl(positions={"BTC": "0.01"})
    w = _worker(tmp_path, client, fills_db=db)
    assert aio.run(w.emergency_flatten("BTC", attempt=1, now_ms=T0)) == "filled"
    (req,) = client.orders
    intent = db.find_intent("HYPERLIQUID", client_id=req.cloid)
    assert intent is not None and intent.kind == "emergency" and intent.is_buy is False and intent.intended_price == D(50000)


def test_the_bybit_worker_records_the_intent_under_its_order_link_id(tmp_path: Path) -> None:
    from tests.test_bybit_worker import FakeBybit, _run, _signal, _worker

    db = FillsDatabase()
    client = FakeBybit()
    w = _worker(tmp_path, client, fills_db=db)
    assert _run(w, _signal()) == "accepted"
    (req,) = client.orders
    intent = db.find_intent("BYBIT", client_id=req.order_link_id)
    assert intent is not None and intent.order_id == "oid-1" and intent.symbol == "BTCUSDT" and intent.intended_price == D(50000) and intent.is_buy is True
    assert db.find_intent("BYBIT", order_id="oid-1") == intent  # un fill sans orderLinkId se rapproche par orderId


class _Tracker:
    def __init__(self, *, crash: bool = False) -> None:
        self.started = False
        self.stopped = False
        self._crash = crash

    async def run(self, stop: asyncio.Event) -> None:
        self.started = True
        if self._crash:
            raise RuntimeError("bug du tracker")
        await stop.wait()
        self.stopped = True


def _run_worker_for(worker: ExposureWorkerBase, seconds: float) -> None:
    async def go() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        await asyncio.sleep(seconds)
        stop.set()
        await asyncio.wait_for(task, 5.0)

    asyncio.run(go(), loop_factory=asyncio.SelectorEventLoop)


def test_the_worker_runs_its_tracker_beside_the_zmq_loop_and_stops_it_with_the_worker(tmp_path: Path) -> None:
    from tests.test_hyperliquid_worker import FakeHl, _worker

    tracker = _Tracker()
    _run_worker_for(_worker(tmp_path, FakeHl(), fill_tracker=tracker, dead_man_timeout_s=None), 0.4)
    assert tracker.started and tracker.stopped


def test_a_crashing_tracker_alerts_but_never_stops_the_worker(tmp_path: Path) -> None:
    from tests.test_hyperliquid_worker import FakeHl, _worker

    rec = RecordingAlerter()
    tracker = _Tracker(crash=True)
    w = _worker(tmp_path, FakeHl(), fill_tracker=tracker, alerter=rec, dead_man_timeout_s=None)
    _run_worker_for(w, 0.4)  # le worker a continué jusqu'à l'arrêt demandé
    assert tracker.started and [k for k, _, _ in rec.alerts] == ["fill_tracker_crashed"]


# --- Ligne de commande -----------------------------------------------------------------------


def _hl_cli_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *args: str) -> Any:
    from workers.hyperliquid import worker as hl

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def factory(config: Any, client: Any) -> None:
        seen["config"] = config
        raise _Stop

    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", "test-secret-0123456789abcdef-0123456789abcdef")
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setenv("HL_AGENT_SECRET", "a" * 64)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.setattr(hl, "SdkHyperliquidClient", lambda **_: SimpleNamespace(rate_gate=RateLimitGate()))
    monkeypatch.setattr(hl, "HyperliquidWorker", factory)
    monkeypatch.setattr(hl, "configure_logging", lambda *a, **k: None)
    with pytest.raises(_Stop):
        hl.main(["--address", "tcp://127.0.0.1:1", "--db", str(tmp_path / "hl.sqlite3"), *args])
    return seen["config"]


def test_hyperliquid_tracks_fills_only_when_live_and_puts_the_database_next_to_the_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dry = _hl_cli_config(monkeypatch, tmp_path)
    assert dry.fills_db is None and dry.fill_tracker is None  # dry-run : aucun ordre, rien à mesurer
    live = _hl_cli_config(monkeypatch, tmp_path, "--live")
    assert isinstance(live.fill_tracker, FillTracker) and live.fills_db is not None
    assert (tmp_path / "hyperliquid_fills.sqlite3").exists()
    off = _hl_cli_config(monkeypatch, tmp_path, "--live", "--no-fill-tracking")
    assert off.fills_db is None and off.fill_tracker is None
    custom = _hl_cli_config(monkeypatch, tmp_path, "--live", "--fills-db", str(tmp_path / "mine.sqlite3"))
    assert custom.fills_db is not None and (tmp_path / "mine.sqlite3").exists()

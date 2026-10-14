"""Écoute des fills sur les flux WebSocket privés (Hyperliquid, Bybit) et mesure du slippage.

Chaque fill reçu est rapproché de l'**intention** de l'ordre qui l'a produit (``fills_db.OrderIntent``, enregistrée par le
worker avant l'envoi), le slippage est calculé et la ligne insérée dans SQLite (``fills_db.FillsDatabase``). Voir ce module
pour la convention de signe et pour ce que ce slippage mesure.

**Ne bloque rien.** Le tracker est une tâche ``asyncio`` à part, lancée par le socle du worker à côté de la boucle ZMQ.
Lecture du socket : ``await`` pur. Accès SQLite : dans un thread (``asyncio.to_thread``). Une panne du flux (coupure, jeton
refusé, message illisible) est capturée ici : elle ne remonte jamais au worker, qui continue d'exécuter ses signaux.

**Rapprochement fill -> intention.**

- Bybit : le fill porte ``orderLinkId`` (l'identifiant client choisi par le worker) et ``orderId``.
- Hyperliquid : le fill ne porte **pas** de ``cloid`` (vérifié dans la doc des abonnements) : seul ``oid`` rapproche, et
  ``oid`` n'est connu qu'à la réponse de l'API. Un fill peut arriver sur le flux AVANT cette réponse ; il attend donc dans
  une file (``match_grace_s``) et le rapprochement est retenté à chaque tour. Passé ce délai : ligne ``unmatched``.

**Reconnexion.** Retour exponentiel (1 s à 60 s). Hyperliquid renvoie un instantané des derniers fills à chaque
abonnement : les fills manqués pendant une coupure sont récupérés (insertion idempotente), mais seulement s'ils se
rapprochent d'une intention connue ; un fill d'instantané sans intention n'est pas enregistré (ce serait de l'historique
étranger). **Bybit ne rejoue rien** : les fills survenus pendant une coupure du flux sont perdus pour cette base (une
réconciliation par l'API REST reste à écrire).

**Flux dont l'adresse se prépare** (Binance) : un flux peut définir ``async prepare() -> str`` (renvoie l'adresse à
ouvrir, appelée avant CHAQUE connexion, par exemple pour obtenir une clé de session) et ``async keepalive()`` avec
``keepalive_interval_s`` (appelée dans la boucle d'écoute ; si elle échoue, la connexion est refaite avec une clé neuve).

**Non vérifié contre les vrais services.** Les formats de messages viennent de la documentation officielle (abonnement
``userFills`` d'Hyperliquid ; authentification, ``ping`` et sujet ``execution`` du flux privé Bybit V5). Les adresses
Hyperliquid et le rythme de ping de 50 s viennent de la documentation et de l'usage courant, pas de la page consultée.
Aucune connexion réelle n'a été faite ici ; la logique est testée sur de faux flux et sur un serveur WebSocket local.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Protocol

from workers.common.alerts import Alerter, Severity
from workers.common.fills_db import BUY, SELL, Execution, FillsDatabase, OrderIntent, slippage_bps

_log = logging.getLogger(__name__)

HYPERLIQUID_WS: Final[dict[str, str]] = {
    "mainnet": "wss://api.hyperliquid.xyz/ws",
    "testnet": "wss://api.hyperliquid-testnet.xyz/ws",
}
BINANCE_PRIVATE_PATH: Final[str] = "/private/ws"  # schéma /public /market /private ; l'ancien /ws/<listenKey> est arrêté (doc Binance)
BYBIT_PRIVATE_WS: Final[dict[str, str]] = {
    "mainnet": "wss://stream.bybit.com/v5/private",
    "testnet": "wss://stream-testnet.bybit.com/v5/private",
}
_ZERO: Final[Decimal] = Decimal(0)


class FeedError(RuntimeError):
    """Le flux a refusé l'authentification ou l'abonnement. Le message ne contient jamais de clé."""


class WsConnection(Protocol):
    async def send(self, text: str) -> None: ...

    async def recv(self) -> str:
        """Renvoie le prochain message ; lève quand la connexion se ferme."""
        ...

    async def close(self) -> None: ...


ConnectFn = Callable[[str], Awaitable[WsConnection]]


@dataclass(frozen=True)
class FillEvent:
    venue: str
    symbol: str
    is_buy: bool
    price: Decimal
    qty: Decimal
    time_ms: int
    fill_id: str
    order_id: str | None
    client_id: str | None
    fee: Decimal | None
    fee_currency: str | None
    snapshot: bool = False


class FillFeed(Protocol):
    venue: str
    url: str
    heartbeat_interval_s: float

    async def handshake(self, conn: WsConnection) -> None: ...

    def heartbeat_message(self) -> str | None: ...

    def parse(self, raw: str) -> list[FillEvent]:
        """Extrait les fills d'un message ; ``[]`` pour tout le reste (accusés, pong). Lève sur un fill malformé."""
        ...


def _decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("nombre illisible") from exc
    if not number.is_finite():
        raise ValueError("nombre non fini")
    return number


# --- Hyperliquid -----------------------------------------------------------------------------


class HyperliquidFillFeed:
    """Abonnement ``userFills`` pour ``user`` (le vault s'il y en a un, sinon le compte principal)."""

    venue = "HYPERLIQUID"
    heartbeat_interval_s = 50.0

    def __init__(self, *, user: str, network: str = "testnet", url: str | None = None) -> None:
        self.url = url or HYPERLIQUID_WS[network]
        self._user = user

    async def handshake(self, conn: WsConnection) -> None:
        await conn.send(json.dumps({"method": "subscribe", "subscription": {"type": "userFills", "user": self._user}}))

    def heartbeat_message(self) -> str | None:
        return json.dumps({"method": "ping"})

    def parse(self, raw: str) -> list[FillEvent]:
        message = json.loads(raw)
        if not isinstance(message, dict) or message.get("channel") != "userFills":
            return []
        data = message.get("data") or {}
        snapshot = bool(data.get("isSnapshot", False))
        events = []
        for fill in data.get("fills", []):
            side = str(fill["side"]).upper()
            if side in ("B", "BUY"):
                is_buy = True
            elif side in ("A", "S", "SELL"):
                is_buy = False
            else:
                raise ValueError(f"côté de fill inconnu : {side[:10]!r}")
            fee = fill.get("fee")
            events.append(FillEvent(
                venue=self.venue, symbol=str(fill["coin"]), is_buy=is_buy, price=_decimal(fill["px"]), qty=_decimal(fill["sz"]),
                time_ms=int(fill["time"]), fill_id=str(fill["tid"]), order_id=str(fill["oid"]), client_id=None,
                fee=None if fee is None else _decimal(fee), fee_currency=fill.get("feeToken"), snapshot=snapshot,
            ))
        return events


# --- Bybit -----------------------------------------------------------------------------------


class BybitFillFeed:
    """Flux privé V5, sujet ``execution.linear`` (perpétuels USDT). Seuls les ``execType == "Trade"`` sont des fills : le
    même flux porte aussi les paiements de financement (``Funding``) et les liquidations, qui ne sont pas nos ordres."""

    venue = "BYBIT"
    heartbeat_interval_s = 20.0

    def __init__(self, *, api_key: str, api_secret: str, network: str = "testnet", url: str | None = None,
                 clock_ms: Callable[[], int] = lambda: int(time.time() * 1000), ack_timeout_s: float = 10.0) -> None:
        self.url = url or BYBIT_PRIVATE_WS[network]
        self._key = api_key
        self._secret = api_secret
        self._clock_ms = clock_ms
        self._ack_timeout_s = ack_timeout_s

    def __repr__(self) -> str:
        return f"BybitFillFeed(url={self.url!r})"  # jamais la clé ni le secret

    def _auth_message(self) -> str:
        expires = self._clock_ms() + 10_000
        signature = hmac.new(self._secret.encode(), f"GET/realtime{expires}".encode(), hashlib.sha256).hexdigest()
        return json.dumps({"op": "auth", "args": [self._key, expires, signature]})

    async def _expect(self, conn: WsConnection, op: str) -> None:
        deadline = time.monotonic() + self._ack_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FeedError(f"pas de réponse « {op} » du flux Bybit")
            message = json.loads(await asyncio.wait_for(conn.recv(), remaining))
            if isinstance(message, dict) and message.get("op") == op:
                if message.get("success") is True:
                    return
                raise FeedError(f"le flux Bybit a refusé « {op} » (ret_msg : {str(message.get('ret_msg', ''))[:80]!r})")

    async def handshake(self, conn: WsConnection) -> None:
        await conn.send(self._auth_message())
        await self._expect(conn, "auth")
        await conn.send(json.dumps({"op": "subscribe", "args": ["execution.linear"]}))
        await self._expect(conn, "subscribe")

    def heartbeat_message(self) -> str | None:
        return json.dumps({"op": "ping"})

    def parse(self, raw: str) -> list[FillEvent]:
        message = json.loads(raw)
        if not isinstance(message, dict) or not str(message.get("topic", "")).startswith("execution"):
            return []
        events = []
        for item in message.get("data", []):
            if item.get("execType") != "Trade":
                continue
            side = str(item["side"])
            if side not in ("Buy", "Sell"):
                raise ValueError(f"côté de fill inconnu : {side[:10]!r}")
            fee = item.get("execFee")
            events.append(FillEvent(
                venue=self.venue, symbol=str(item["symbol"]), is_buy=side == "Buy", price=_decimal(item["execPrice"]),
                qty=_decimal(item["execQty"]), time_ms=int(item["execTime"]), fill_id=str(item["execId"]),
                order_id=str(item["orderId"]) or None, client_id=str(item.get("orderLinkId") or "") or None,
                fee=None if fee in (None, "") else _decimal(fee), fee_currency=item.get("feeCurrency") or None,
            ))
        return events


# --- Binance ---------------------------------------------------------------------------------


class UserStreamApi(Protocol):
    """Ce que le flux Binance demande à l'API REST : obtenir et prolonger la clé du flux privé (appels bloquants)."""

    def start_user_stream(self) -> str: ...

    def keepalive_user_stream(self) -> None: ...


class BinanceFillFeed:
    """Flux privé USDⓈ-M : événements ``ORDER_TRADE_UPDATE``.

    L'adresse porte la clé de session : ``<base>/private/ws?listenKey=<clé>&events=ORDER_TRADE_UPDATE`` (schéma officiel ;
    l'ancien ``/ws/<clé>`` ne délivre plus les données privées). La clé vit 60 minutes : elle est prolongée toutes les 30 et
    redemandée à chaque reconnexion ; Binance coupe de toute façon une connexion après 24 h.

    **Un fill = un événement ``x == "TRADE"``** (dernière exécution ``l`` à ``L``), qu'il soit partiel
    (``X == PARTIALLY_FILLED``) ou final (``FILLED``) : ne retenir que ``FILLED`` perdrait les exécutions partielles et
    fausserait le prix moyen. Le fill est rapproché par l'identifiant client ``c`` (notre ``newClientOrderId``) puis par ``i``.

    Schéma de l'événement tiré de la connaissance de l'API (stable depuis des années) : la page officielle consultée ne le
    détaillait pas. Les champs requis qui manqueraient font rejeter le message (compté ``malformed``), jamais inventer un prix.
    """

    venue = "BINANCE"
    heartbeat_interval_s = 0.0  # Binance envoie des pings de protocole, auxquels la bibliothèque répond seule
    keepalive_interval_s = 30 * 60.0

    def __init__(self, *, api: UserStreamApi, ws_base_url: str = "wss://fstream.binance.com") -> None:
        self._api = api
        self._base = ws_base_url.rstrip("/")
        self.url = f"{self._base}{BINANCE_PRIVATE_PATH}"

    def __repr__(self) -> str:
        return f"BinanceFillFeed(base={self._base!r})"  # l'adresse complète contient la clé de session

    async def prepare(self) -> str:
        listen_key = await asyncio.to_thread(self._api.start_user_stream)
        self.url = f"{self._base}{BINANCE_PRIVATE_PATH}?listenKey={listen_key}&events=ORDER_TRADE_UPDATE"
        return self.url

    async def keepalive(self) -> None:
        await asyncio.to_thread(self._api.keepalive_user_stream)

    async def handshake(self, conn: WsConnection) -> None:
        """Rien à envoyer : la clé et les événements voulus sont dans l'adresse."""

    def heartbeat_message(self) -> str | None:
        return None

    def parse(self, raw: str) -> list[FillEvent]:
        message = json.loads(raw)
        if not isinstance(message, dict) or message.get("e") != "ORDER_TRADE_UPDATE":
            return []
        order = message["o"]
        if order.get("x") != "TRADE":
            return []
        qty = _decimal(order["l"])
        if qty <= 0:
            return []
        side = str(order["S"]).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"côté de fill inconnu : {side[:10]!r}")
        commission = order.get("n")
        symbol = str(order["s"])
        return [FillEvent(
            venue=self.venue, symbol=symbol, is_buy=side == "BUY", price=_decimal(order["L"]), qty=qty, time_ms=int(order["T"]),
            fill_id=f"{symbol}:{order['t']}", order_id=str(order["i"]), client_id=str(order.get("c") or "") or None,
            fee=None if commission in (None, "") else _decimal(commission), fee_currency=order.get("N") or None,
        )]


# --- Transport -------------------------------------------------------------------------------


class _WebsocketsConnection:
    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send(self, text: str) -> None:
        await self._ws.send(text)

    async def recv(self) -> str:
        message = await self._ws.recv()
        return message if isinstance(message, str) else bytes(message).decode("utf-8", "replace")

    async def close(self) -> None:
        await self._ws.close()


async def websockets_connect(url: str) -> WsConnection:
    """Connexion réelle (paquet ``websockets``, importé à l'usage). Le ping de protocole est coupé : les deux plateformes
    veulent leur propre message de ping applicatif, que le tracker envoie."""
    try:
        import websockets  # type: ignore[import-not-found,unused-ignore]
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise RuntimeError("le paquet websockets n'est pas installé : pip install websockets") from exc
    ws = await websockets.connect(url, open_timeout=10, ping_interval=None, max_size=4 * 1024 * 1024)
    return _WebsocketsConnection(ws)


# --- Tracker ---------------------------------------------------------------------------------


class FillTracker:
    def __init__(
        self,
        feed: FillFeed,
        db: FillsDatabase,
        *,
        connect: ConnectFn = websockets_connect,
        match_grace_s: float = 10.0,
        recv_timeout_s: float = 0.5,
        reconnect_min_s: float = 1.0,
        reconnect_max_s: float = 60.0,
        alerter: Alerter | None = None,
        alert_after_failures: int = 5,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        intent_retention_ms: int = 7 * 24 * 3600 * 1000,
    ) -> None:
        self._feed = feed
        self._db = db
        self._connect = connect
        self._grace_ms = int(match_grace_s * 1000)
        self._recv_timeout_s = recv_timeout_s
        self._backoff_min = reconnect_min_s
        self._backoff_max = reconnect_max_s
        self._alerter = alerter
        self._alert_after = alert_after_failures
        self._clock_ms = clock_ms
        self._retention_ms = intent_retention_ms
        self._pending: dict[str, tuple[FillEvent, int]] = {}  # fill_id -> (fill, instant de première réception)
        self._alerted = False
        self.connected = False
        self.stats: dict[str, int] = {"received": 0, "matched": 0, "unmatched": 0, "duplicates": 0, "malformed": 0, "reconnects": 0}

    # --- boucle -----------------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """Se connecte, écoute, se reconnecte, jusqu'à ``stop``. Ne lève jamais (sauf annulation)."""
        backoff = self._backoff_min
        failures = 0
        await self._safe(asyncio.to_thread(self._db.prune_intents_before, self._clock_ms() - self._retention_ms))
        while not stop.is_set():
            conn: WsConnection | None = None
            try:
                prepare = getattr(self._feed, "prepare", None)
                url = await prepare() if prepare is not None else self._feed.url
                conn = await self._connect(url)
                await self._feed.handshake(conn)
                failures, backoff = 0, self._backoff_min
                self.connected = True
                if self._alerted:
                    self._alerted = False
                    self._alert("fill_tracker_up", "le flux de fills est rétabli.", Severity.INFO)
                _log.info("fill_tracker_connected venue=%s", self._feed.venue)
                await self._pump(conn, stop)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- une panne du flux ne doit jamais atteindre le worker
                failures += 1
                self.stats["reconnects"] += 1
                _log.warning("fill_tracker_disconnected venue=%s error=%s failures=%d", self._feed.venue, type(exc).__name__, failures)
                if failures == self._alert_after and not self._alerted:
                    self._alerted = True
                    self._alert(
                        "fill_tracker_down",
                        f"le flux de fills {self._feed.venue} est coupé depuis {failures} tentatives ({type(exc).__name__}) : "
                        "le slippage n'est plus mesuré (l'exécution des ordres n'est pas touchée).",
                    )
            finally:
                self.connected = False
                if conn is not None:
                    await self._safe(conn.close())
            if stop.is_set():
                break
            try:
                await asyncio.wait_for(stop.wait(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, self._backoff_max)
        await self._flush_pending(force=True)

    async def _pump(self, conn: WsConnection, stop: asyncio.Event) -> None:
        interval = self._feed.heartbeat_interval_s
        keepalive = getattr(self._feed, "keepalive", None)
        keepalive_interval = float(getattr(self._feed, "keepalive_interval_s", 0.0))
        last_heartbeat = last_keepalive = time.monotonic()
        while not stop.is_set():
            try:
                raw: str | None = await asyncio.wait_for(conn.recv(), self._recv_timeout_s)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                await self._ingest(raw)
            now = time.monotonic()
            if interval and now - last_heartbeat >= interval:
                heartbeat = self._feed.heartbeat_message()
                if heartbeat is not None:
                    await conn.send(heartbeat)
                last_heartbeat = now
            if keepalive is not None and keepalive_interval and now - last_keepalive >= keepalive_interval:
                await keepalive()  # échec : remonte, la connexion est refaite avec une clé neuve
                last_keepalive = now
            if self._pending:
                await self._flush_pending()

    # --- traitement -------------------------------------------------------------------------

    async def _ingest(self, raw: str) -> None:
        try:
            events = self._feed.parse(raw)
        except Exception as exc:  # noqa: BLE001 -- un message illisible n'arrête pas l'écoute
            self.stats["malformed"] += 1
            _log.warning("fill_tracker_malformed venue=%s error=%s", self._feed.venue, type(exc).__name__)
            return
        if not events:
            return
        self.stats["received"] += len(events)
        now = self._clock_ms()
        try:
            for event in await asyncio.to_thread(self._store, events, now):
                self._pending.setdefault(event.fill_id, (event, now))
        except Exception:  # noqa: BLE001 -- la base ne doit pas arrêter l'écoute
            _log.exception("fill_tracker_store_failed venue=%s", self._feed.venue)

    async def _flush_pending(self, *, force: bool = False) -> None:
        if not self._pending:
            return
        now = self._clock_ms()
        batch = [event for event, _ in self._pending.values()]
        expired = {fid for fid, (_, first) in self._pending.items() if force or now - first >= self._grace_ms}
        try:
            still = await asyncio.to_thread(self._store, batch, now, expired)
        except Exception:  # noqa: BLE001
            _log.exception("fill_tracker_store_failed venue=%s", self._feed.venue)
            return
        keep = {e.fill_id for e in still}
        self._pending = {fid: item for fid, item in self._pending.items() if fid in keep}

    def _store(self, events: list[FillEvent], now_ms: int, expired: set[str] | None = None) -> list[FillEvent]:
        """Thread de travail. Enregistre ce qui peut l'être ; renvoie les fills encore sans intention (à réessayer)."""
        waiting: list[FillEvent] = []
        for event in events:
            if self._db.has_fill(event.venue, event.fill_id):
                self.stats["duplicates"] += 1
                continue
            intent = self._db.find_intent(event.venue, client_id=event.client_id, order_id=event.order_id)
            if intent is not None:
                self._insert(event, intent)
                self.stats["matched"] += 1
            elif event.snapshot:
                continue  # historique sans intention connue : pas à nous
            elif expired is not None and event.fill_id in expired:
                self._insert(event, None)
                self.stats["unmatched"] += 1
            else:
                waiting.append(event)
        return waiting

    def _insert(self, event: FillEvent, intent: OrderIntent | None) -> None:
        slippage = None if intent is None else slippage_bps(event.is_buy, intent.intended_price, event.price)
        self._db.insert_execution(Execution(
            timestamp=event.time_ms, venue=event.venue, symbol=event.symbol, direction=BUY if event.is_buy else SELL,
            intended_price=None if intent is None else float(intent.intended_price), fill_price=float(event.price),
            fill_qty=float(event.qty), slippage_bps=None if slippage is None else float(slippage), order_id=event.order_id,
            fill_id=event.fill_id, kind="unmatched" if intent is None else intent.kind,
            fee=None if event.fee is None else float(event.fee), fee_currency=event.fee_currency,
        ))

    # --- utilitaires ------------------------------------------------------------------------

    def _alert(self, key: str, text: str, severity: Severity = Severity.CRITICAL) -> None:
        if self._alerter is None:
            return
        try:
            self._alerter.alert(text, key=key, severity=severity)
        except Exception:  # noqa: BLE001
            _log.exception("fill_tracker_alert_failed key=%s", key)

    @staticmethod
    async def _safe(awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- fermeture et nettoyage : on n'en fait pas une erreur
            _log.debug("fill_tracker_cleanup_failed", exc_info=True)

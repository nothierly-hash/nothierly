"""Démon asynchrone : VPIN en temps réel à partir des trades publics, écrit dans Redis pour le moteur.

    python -m workers.market_data.vpin_tracker --source binance --symbols BTCUSDT ETHUSDT --bucket-size 100000 --window 50

Aucune authentification : les flux de trades sont publics. Le démon n'a ni clé d'exchange ni droit de trading ; il lit un flux
public et écrit une clé Redis (``market:vpin:<SYMBOLE>``, par exemple ``market:vpin:BTCUSDT`` = ``0.653421``). Il a besoin d'un
accès sortant vers l'exchange et d'un accès à Redis, rien d'autre : c'est le conteneur « market-data » avec egress de
l'architecture, pour que le moteur lui-même n'ait aucune sortie internet.

**Sources (une par démon).**

- ``binance`` : ``<symbole>@aggTrade`` du flux ``/market`` de ``wss://fstream.binance.com`` (schéma ``/public`` ``/market``
  ``/private`` : l'ancien ``/ws`` ne délivre plus les flux de marché). ``m == true`` = l'acheteur est le teneur de marché = le
  preneur a VENDU ; ``m == false`` = le preneur a acheté.
- ``hyperliquid`` : abonnement ``trades`` par pièce. La doc ne définit pas ``side`` ; on lit ``B`` = achat, ``A`` = vente. Sans effet sur
  le VPIN (voir ``vpin.py`` : seul ``|achat - vente|`` compte), mais à ne pas réutiliser ailleurs sans l'avoir vérifié.

Ne pas lancer les deux sources pour le même symbole : elles écriraient la même clé avec des flux différents.

**Clé et fraîcheur.** La valeur est écrite avec une durée de vie (``--ttl-s``, 120 s) à chaque seau fermé et ré-écrite toutes
les ``--refresh-s`` (30 s) TANT QUE la connexion vit : un marché calme qui ne ferme pas de seau garde donc sa clé, alors qu'un démon
mort ou déconnecté la laisse expirer. Le moteur lit « clé absente » comme « pas de lecture », jamais comme une valeur. Pas de
valeur du tout avant que la fenêtre soit pleine (voir ``vpin.py``).

**Flux muet (connexion semi-ouverte).** Un lien TCP mort sans fermeture propre ne lève aucune erreur : ``recv`` attend sans fin, et la ré-écriture
périodique garderait vivante une valeur gelée que le moteur lirait comme fraîche. Deux défenses : le keepalive TCP du système (activé par
``websockets_connect``, voir ``ws_tracker.enable_tcp_keepalive``) et un watchdog de fraîcheur ici (``--stale-after-s``, 60 s) : aucun message depuis
ce délai -> la clé Redis est SUPPRIMÉE (le moteur lit « pas de lecture », il se sait aveugle), une alerte part, la socket est fermée et la reconnexion
suit son backoff. La clé n'est ré-écrite qu'à partir du moment où un message est reçu de nouveau. Marché très calme (altcoin peu liquide) : relever
le délai, car 60 s sans le moindre trade y est normal.

**Trou dans le flux.** Après une coupure de plus de ``--max-gap-s`` (30 s), les trades manqués fausseraient les seaux en cours :
la fenêtre est vidée et la clé supprimée, la mesure repart de zéro. Les identifiants de trade déjà vus sont ignorés.

**Archive continue (SQLite).** Redis ne garde que la dernière valeur, avec expiration. Chaque seau fermé est donc aussi écrit dans
``<NOTHIERLY_DATA_DIR>/vpin_history.sqlite3`` (``--archive-db``, ``--no-archive`` pour couper), par un thread dédié : la boucle WebSocket ne
touche jamais au disque (voir ``vpin_archive.py``). L'archive ne dépend pas de Redis (Redis en panne : rien n'est perdu) et ne l'arrête pas
(archive en panne : Redis continue, et une alerte part). ``--dry-run`` n'écrit nulle part, archive comprise.

**Ce que ce VPIN n'est pas.** Ce n'est PAS le VPIN que le moteur lit déjà (``research/alpha_features.compute_vpin`` et la lecture live de
``macro_order_execution`` : seaux en unités de base de 1,0, fenêtre de 10, trade non découpé). Autres unités, autre fenêtre : les deux
valeurs ne se comparent pas et ne se mélangent pas. Celle-ci n'alimente qu'une variable du modèle de méta-labeling.

**Non vérifié contre les vrais flux.** Les formes de messages viennent de la documentation (Hyperliquid : abonnement et champs de
``WsTrade``) et de la connaissance de l'API (Binance ``aggTrade`` : la page consultée ne la détaillait pas). Testé sur de faux flux
et sur un serveur WebSocket local.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Protocol

from workers.common.alerts import AlertConfigError, Alerter, Severity, WebhookAlerter
from workers.common.logging_setup import configure_logging
from workers.common.ws_tracker import ConnectFn, FeedStaleError, WsConnection, websockets_connect
from workers.market_data.vpin import VolumeBucketVpin
from workers.market_data.vpin_archive import BucketRecord, VpinArchive, default_archive_path

_log = logging.getLogger(__name__)

KEY_PREFIX: Final[str] = "market:vpin:"
BINANCE_MARKET_WS: Final[str] = "wss://fstream.binance.com"
HYPERLIQUID_WS: Final[Mapping[str, str]] = {"mainnet": "wss://api.hyperliquid.xyz/ws", "testnet": "wss://api.hyperliquid-testnet.xyz/ws"}
_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9]{3,20}$")


def redis_key(symbol: str) -> str:
    return f"{KEY_PREFIX}{symbol}"


@dataclass(frozen=True)
class Trade:
    symbol: str  # ex. BTCUSDT
    price: Decimal
    qty: Decimal
    is_buy: bool  # le PRENEUR a acheté
    time_ms: int
    trade_id: int | None


class TradeFeed(Protocol):
    url: str
    heartbeat_interval_s: float

    async def handshake(self, conn: WsConnection) -> None: ...

    def heartbeat_message(self) -> str | None: ...

    def parse(self, raw: str) -> list[Trade]:
        """Trades d'un message ; ``[]`` pour tout le reste (accusés, pong). Lève sur un trade malformé."""
        ...


def _number(value: Any) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("nombre illisible") from exc
    if not number.is_finite():
        raise ValueError("nombre non fini")
    return number


class BinanceAggTradeFeed:
    """``<symbole>@aggTrade`` en mode ``stream`` (plusieurs symboles sur une connexion, messages enveloppés)."""

    heartbeat_interval_s = 0.0  # Binance ping, la bibliothèque répond

    def __init__(self, symbols: Sequence[str], ws_base_url: str = BINANCE_MARKET_WS) -> None:
        streams = "/".join(f"{s.lower()}@aggTrade" for s in symbols)
        self.url = f"{ws_base_url.rstrip('/')}/market/stream?streams={streams}"

    async def handshake(self, conn: WsConnection) -> None:
        """Rien à envoyer : les flux sont dans l'adresse."""

    def heartbeat_message(self) -> str | None:
        return None

    def parse(self, raw: str) -> list[Trade]:
        message = json.loads(raw)
        event = message.get("data", message) if isinstance(message, dict) else None
        if not isinstance(event, dict) or event.get("e") != "aggTrade":
            return []
        return [Trade(
            symbol=str(event["s"]).upper(), price=_number(event["p"]), qty=_number(event["q"]), is_buy=not bool(event["m"]),
            time_ms=int(event["T"]), trade_id=int(event["a"]),
        )]


class HyperliquidTradesFeed:
    """Abonnement ``trades`` par pièce ; les pièces s'écrivent ``BTC``, les clés Redis ``BTCUSDT`` (même nom que côté moteur)."""

    heartbeat_interval_s = 50.0

    def __init__(self, symbols: Sequence[str], network: str = "mainnet", url: str | None = None) -> None:
        self.url = url or HYPERLIQUID_WS[network]
        self._coins = {s.removesuffix("USDT"): s for s in symbols}

    async def handshake(self, conn: WsConnection) -> None:
        for coin in self._coins:
            await conn.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))

    def heartbeat_message(self) -> str | None:
        return json.dumps({"method": "ping"})

    def parse(self, raw: str) -> list[Trade]:
        message = json.loads(raw)
        if not isinstance(message, dict) or message.get("channel") != "trades":
            return []
        trades = []
        for item in message.get("data", []):
            symbol = self._coins.get(str(item["coin"]))
            if symbol is None:
                continue
            side = str(item["side"]).upper()
            if side not in ("A", "B"):
                raise ValueError(f"côté de trade inconnu : {side[:10]!r}")
            trades.append(Trade(symbol, _number(item["px"]), _number(item["sz"]), side == "B", int(item["time"]), int(item["tid"])))
        return trades


# --- Redis -----------------------------------------------------------------------------------


class VpinSink(Protocol):
    async def publish(self, symbol: str, vpin: Decimal, ttl_s: float) -> None: ...

    async def clear(self, symbol: str) -> None: ...


class RedisVpinSink:
    """``SET market:vpin:<SYMBOLE> <valeur> EX <ttl>`` (``redis.asyncio`` ; pas de ``core``, les workers n'en importent jamais)."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 6379, password: str | None = None, db: int = 0) -> None:
        from redis.asyncio import Redis

        self._redis: Any = Redis(host=host, port=port, password=password, db=db, decode_responses=True, socket_timeout=5.0, socket_connect_timeout=5.0)

    def __repr__(self) -> str:
        return "RedisVpinSink()"  # jamais le mot de passe

    async def publish(self, symbol: str, vpin: Decimal, ttl_s: float) -> None:
        await self._redis.set(redis_key(symbol), format(vpin, ".6f"), px=max(1, int(ttl_s * 1000)))

    async def clear(self, symbol: str) -> None:
        await self._redis.delete(redis_key(symbol))

    async def close(self) -> None:
        await self._redis.aclose()


class LogOnlySink:
    """``--dry-run`` : journalise, n'écrit nulle part."""

    async def publish(self, symbol: str, vpin: Decimal, ttl_s: float) -> None:
        _log.info("vpin_dry_run %s=%s", redis_key(symbol), format(vpin, ".6f"))

    async def clear(self, symbol: str) -> None:
        _log.info("vpin_dry_run_clear %s", redis_key(symbol))


# --- Démon -----------------------------------------------------------------------------------


class VpinTracker:
    def __init__(
        self,
        feed: TradeFeed,
        calculators: Mapping[str, VolumeBucketVpin],
        sink: VpinSink,
        *,
        connect: ConnectFn = websockets_connect,
        ttl_s: float = 120.0,
        refresh_s: float = 30.0,
        max_gap_s: float = 30.0,
        recv_timeout_s: float = 0.5,
        reconnect_min_s: float = 1.0,
        reconnect_max_s: float = 60.0,
        alerter: Alerter | None = None,
        alert_after_failures: int = 5,
        clock: Callable[[], float] = time.monotonic,
        archive: VpinArchive | None = None,
        stale_after_s: float | None = 60.0,
    ) -> None:
        self._feed = feed
        self._calcs = dict(calculators)
        self._sink = sink
        self._connect = connect
        self._ttl_s = ttl_s
        self._refresh_s = refresh_s
        self._max_gap_s = max_gap_s
        self._recv_timeout_s = recv_timeout_s
        self._backoff_min = reconnect_min_s
        self._backoff_max = reconnect_max_s
        self._alerter = alerter
        self._alert_after = alert_after_failures
        self._clock = clock
        self._archive = archive
        self._stale_after_s = stale_after_s if stale_after_s else None  # 0 ou None : watchdog coupé
        self._last_activity = clock()  # dernier message reçu, ou début de la connexion tant qu'aucun n'est arrivé
        self._data_seen = False  # un message est-il arrivé depuis la dernière connexion ? (sinon rien n'est ré-écrit dans Redis)
        self._stale_alerted = False
        self._archive_errors_seen = 0
        self._archive_dropped_seen = 0
        self._archive_alerted = False
        self._last_ids: dict[str, int] = {}
        self._disconnected_at: float | None = None
        self._alerted = False
        self.connected = False
        self.stats: dict[str, int] = {"trades": 0, "buckets": 0, "duplicates": 0, "malformed": 0, "reconnects": 0, "resets": 0, "sink_errors": 0, "stale": 0}

    # --- boucle -----------------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """Se connecte, écoute, se reconnecte, jusqu'à ``stop``. Ne lève jamais (sauf annulation)."""
        backoff = self._backoff_min
        failures = 0
        while not stop.is_set():
            conn: WsConnection | None = None
            try:
                conn = await self._connect(self._feed.url)
                await self._feed.handshake(conn)
                failures, backoff = 0, self._backoff_min
                await self._on_connected()
                await self._pump(conn, stop)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- une panne du flux ne doit jamais arrêter le démon
                failures += 1
                self.stats["reconnects"] += 1
                _log.warning("vpin_feed_disconnected error=%s failures=%d", type(exc).__name__, failures)
                if failures == self._alert_after and not self._alerted:
                    self._alerted = True
                    self._alert("vpin_feed_down", f"le flux de trades est coupé depuis {failures} tentatives ({type(exc).__name__}) : le VPIN n'est plus mis à jour.")
            finally:
                if self.connected:
                    self._disconnected_at = self._last_activity  # le trou commence au dernier message reçu, pas à la fermeture
                self.connected = False
                if conn is not None:
                    try:
                        await conn.close()
                    except Exception:  # noqa: BLE001 -- nettoyage
                        _log.debug("vpin_close_failed", exc_info=True)
            if stop.is_set():
                break
            try:
                await asyncio.wait_for(stop.wait(), backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, self._backoff_max)

    async def _on_connected(self) -> None:
        self.connected = True
        gap = None if self._disconnected_at is None else self._clock() - self._disconnected_at
        if gap is not None and gap > self._max_gap_s:
            await self._reset(f"trou de {gap:.0f}s dans le flux")
        self._disconnected_at = None
        if self._alerted:
            self._alerted = False
            self._alert("vpin_feed_up", "le flux de trades est rétabli.", Severity.INFO)
        _log.info("vpin_feed_connected symbols=%s", sorted(self._calcs))

    async def _reset(self, reason: str) -> None:
        self.stats["resets"] += 1
        _log.warning("vpin_reset reason=%s", reason)
        for symbol, calc in self._calcs.items():
            calc.reset()
            self._last_ids.pop(symbol, None)
            await self._safe_sink(self._sink.clear(symbol))

    async def _pump(self, conn: WsConnection, stop: asyncio.Event) -> None:
        interval = self._feed.heartbeat_interval_s
        last_heartbeat = last_refresh = self._last_activity = self._clock()
        self._data_seen = False
        while not stop.is_set():
            try:
                raw: str | None = await asyncio.wait_for(conn.recv(), self._recv_timeout_s)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                self._last_activity = self._clock()
                self._data_seen = True
                if self._stale_alerted:
                    self._stale_alerted = False
                    self._alert("vpin_feed_stale_end", "le flux de trades reçoit de nouveau des messages.", Severity.INFO)
                await self._ingest(raw)
            now = self._clock()
            if self._stale_after_s is not None and now - self._last_activity > self._stale_after_s:
                await self._on_stale(now - self._last_activity)
            if interval and now - last_heartbeat >= interval:
                heartbeat = self._feed.heartbeat_message()
                if heartbeat is not None:
                    await conn.send(heartbeat)
                last_heartbeat = now
            if now - last_refresh >= self._refresh_s:
                last_refresh = now
                await self._refresh()

    async def _on_stale(self, silent_s: float) -> None:
        """Aveugle : supprime les clés (le moteur ne lit plus rien), alerte une fois, puis force la reconnexion."""
        self.stats["stale"] += 1
        _log.warning("vpin_feed_stale silent_s=%.0f : clés supprimées, reconnexion", silent_s)
        for symbol in self._calcs:
            await self._safe_sink(self._sink.clear(symbol))
        if not self._stale_alerted:
            self._stale_alerted = True
            self._alert("vpin_feed_stale", f"le flux de trades est muet depuis {silent_s:.0f}s (connexion semi-ouverte probable) : le VPIN est retiré de Redis, reconnexion forcée.")
        raise FeedStaleError(f"aucun message depuis {silent_s:.0f}s")

    async def _ingest(self, raw: str) -> None:
        try:
            trades = self._feed.parse(raw)
        except Exception as exc:  # noqa: BLE001 -- un message illisible n'arrête pas l'écoute
            self.stats["malformed"] += 1
            _log.warning("vpin_malformed error=%s", type(exc).__name__)
            return
        for trade in trades:
            calc = self._calcs.get(trade.symbol)
            if calc is None:
                continue
            if trade.trade_id is not None:
                if trade.trade_id <= self._last_ids.get(trade.symbol, -1):
                    self.stats["duplicates"] += 1
                    continue
                self._last_ids[trade.symbol] = trade.trade_id
            try:
                closed = calc.add_trade(trade.price, trade.qty, trade.is_buy, trade.time_ms)
            except ValueError:
                self.stats["malformed"] += 1
                continue
            self.stats["trades"] += 1
            if closed:
                self.stats["buckets"] += len(closed)
                if self._archive is not None:  # d'abord : l'archive ne dépend ni de Redis ni de sa disponibilité ; ``submit`` ne bloque jamais
                    for reading in closed:
                        self._archive.submit(BucketRecord(
                            trade.symbol, reading.closed_at_ms, reading.vpin, reading.buy_volume, reading.sell_volume, calc.bucket_size, calc.window_size,
                        ))
                latest = closed[-1].vpin
                if latest is not None:
                    await self._safe_sink(self._sink.publish(trade.symbol, latest, self._ttl_s))

    def _check_archive(self) -> None:
        """Alerte (une fois) quand l'archive échoue ou abandonne des lignes ; annonce le retour à la normale."""
        archive = self._archive
        if archive is None:
            return
        failing = archive.errors > self._archive_errors_seen or archive.dropped > self._archive_dropped_seen  # nouveau depuis le dernier contrôle
        self._archive_errors_seen, self._archive_dropped_seen = archive.errors, archive.dropped
        if failing and not self._archive_alerted:
            self._archive_alerted = True
            self._alert(
                "vpin_archive_failing",
                f"l'archive VPIN n'écrit plus (erreurs={archive.errors}, lignes abandonnées={archive.dropped}, en attente={archive.backlog}, "
                f"dernière erreur : {archive.last_error}) : l'historique de ce marché est en train de se perdre.",
            )
        elif self._archive_alerted and not failing and archive.backlog == 0:
            self._archive_alerted = False
            self._alert("end:vpin_archive_failing", "l'archive VPIN écrit de nouveau.", Severity.INFO)

    async def _refresh(self) -> None:
        """Ré-écrit la dernière valeur pour prolonger sa durée de vie : un marché calme ne ferme pas de seau."""
        self._check_archive()
        if not self._data_seen:
            return  # rien reçu depuis la (re)connexion : on ne ré-écrit jamais une valeur dont on n'a plus de nouvelles
        for symbol, calc in self._calcs.items():
            value = calc.vpin
            if value is not None:
                await self._safe_sink(self._sink.publish(symbol, value, self._ttl_s))

    async def _safe_sink(self, awaitable: Any) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- Redis indisponible : on continue de calculer, la clé expirera
            self.stats["sink_errors"] += 1
            _log.warning("vpin_sink_failed error=%s", type(exc).__name__)

    def _alert(self, key: str, text: str, severity: Severity = Severity.CRITICAL) -> None:
        if self._alerter is None:
            return
        try:
            self._alerter.alert(text, key=key, severity=severity)
        except Exception:  # noqa: BLE001
            _log.exception("vpin_alert_failed key=%s", key)

    def snapshot(self) -> dict[str, str | None]:
        return {symbol: None if calc.vpin is None else format(calc.vpin, ".6f") for symbol, calc in self._calcs.items()}


# --- Ligne de commande -----------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VPIN en temps réel à partir des trades publics, écrit dans Redis (market:vpin:<SYMBOLE>).")
    parser.add_argument("--source", choices=["binance", "hyperliquid"], default="binance")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="noms de clé Redis, ex. BTCUSDT (Hyperliquid : BTCUSDT -> pièce BTC)")
    parser.add_argument("--bucket-size", type=Decimal, default=Decimal(100_000), help="notionnel d'un seau (VOLUME_BUCKET_SIZE)")
    parser.add_argument("--window", type=int, default=50, help="nombre de seaux de la moyenne glissante (WINDOW_SIZE)")
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="mainnet", help="Hyperliquid seulement")
    parser.add_argument("--ws-base-url", default=None, help="adresse WebSocket (Binance : base, sans /market ; Hyperliquid : adresse complète)")
    parser.add_argument("--ttl-s", type=float, default=120.0, help="durée de vie de la clé Redis")
    parser.add_argument("--refresh-s", type=float, default=30.0, help="ré-écriture de la dernière valeur tant que la connexion vit")
    parser.add_argument("--max-gap-s", type=float, default=30.0, help="coupure au-delà de laquelle la fenêtre est remise à zéro")
    parser.add_argument("--archive-db", default=None, help="base SQLite de l'historique des seaux (défaut : <NOTHIERLY_DATA_DIR ou data/>/vpin_history.sqlite3)")
    parser.add_argument("--no-archive", action="store_true", help="n'écrit pas l'historique SQLite (déconseillé : c'est la matière première de l'entraînement)")
    parser.add_argument("--stale-after-s", type=float, default=60.0, help="aucun message depuis ce délai : clé Redis supprimée, alerte, reconnexion (0 = coupé)")
    parser.add_argument("--dry-run", action="store_true", help="journalise les valeurs sans écrire dans Redis ni dans l'archive")
    parser.add_argument("--log-dir", default=None, help="dossier d'un fichier de journal tournant (10 Mo x 5) en plus de stdout")
    args = parser.parse_args(argv)
    if args.bucket_size <= 0 or not args.bucket_size.is_finite():
        parser.error("--bucket-size doit être positif")
    if args.window < 1:
        parser.error("--window doit être >= 1")
    if args.refresh_s <= 0 or args.ttl_s <= args.refresh_s:
        parser.error("--ttl-s doit être strictement supérieur à --refresh-s (sinon la clé expire entre deux ré-écritures)")
    if args.stale_after_s < 0:
        parser.error("--stale-after-s doit être >= 0")
    symbols = [s.upper() for s in args.symbols]
    bad = [s for s in symbols if not _SYMBOL_RE.match(s)]
    if bad:
        parser.error(f"symbole invalide : {bad[0]!r}")
    try:
        alerter = WebhookAlerter.from_env(source="vpin-tracker")
    except AlertConfigError as exc:
        parser.error(str(exc))
    configure_logging("vpin-tracker", log_dir=args.log_dir)
    feed: TradeFeed
    if args.source == "binance":
        feed = BinanceAggTradeFeed(symbols, args.ws_base_url or BINANCE_MARKET_WS)
    else:
        feed = HyperliquidTradesFeed(symbols, args.network, args.ws_base_url)
    sink: VpinSink = LogOnlySink() if args.dry_run else RedisVpinSink(
        host=os.environ.get("REDIS_HOST", "127.0.0.1"), port=int(os.environ.get("REDIS_PORT", "6379")),
        password=os.environ.get("REDIS_PASSWORD") or None, db=int(os.environ.get("REDIS_DB", "0")),
    )
    calculators = {s: VolumeBucketVpin(args.bucket_size, args.window) for s in symbols}
    archive: VpinArchive | None = None
    if not args.dry_run and not args.no_archive:
        try:
            archive = VpinArchive(args.archive_db or default_archive_path())  # échec rapide : pas de collecte sans archive, sauf --no-archive
        except (OSError, sqlite3.Error) as exc:
            parser.error(f"archive VPIN inutilisable ({type(exc).__name__}: {exc}) ; corriger NOTHIERLY_DATA_DIR / --archive-db, ou --no-archive")
    tracker = VpinTracker(feed, calculators, sink, ttl_s=args.ttl_s, refresh_s=args.refresh_s, max_gap_s=args.max_gap_s, alerter=alerter, archive=archive, stale_after_s=args.stale_after_s)
    _log.info(
        "vpin_tracker_start source=%s symbols=%s bucket=%s window=%d mode=%s archive=%s", args.source, symbols, args.bucket_size, args.window,
        "DRY-RUN" if args.dry_run else "REDIS", "off" if archive is None else archive.path,
    )

    async def serve() -> None:
        stop = asyncio.Event()
        if archive is not None:
            archive.start()
        try:
            await tracker.run(stop)
        finally:
            if archive is not None:
                left = archive.close()
                if left:
                    _log.error("vpin_archive_lost_on_shutdown rows=%d", left)
            closer = getattr(sink, "close", None)
            if closer is not None:
                await closer()

    try:
        asyncio.run(serve(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("vpin_tracker_stopped")
    finally:
        if alerter is not None:
            alerter.close(2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

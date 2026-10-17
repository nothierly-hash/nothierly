"""Reporter de positions : publie dans Redis les positions RÉELLES de chaque plateforme, pour que le moteur les adopte au redémarrage.

    python -m workers.reconciliation.reporter --venues hyperliquid bybit binance --network testnet --interval-s 30

**Pourquoi un service à part (et pas les workers).** Le moteur n'a ni clé ni accès sortant ; les workers ont les clés de TRADING et n'ont volontairement aucun
accès à Redis. Donner Redis aux workers mettrait un mot de passe d'infrastructure dans les conteneurs qui peuvent passer des ordres. Ce reporter ne passe
jamais d'ordre : il lit le compte avec des clés en LECTURE SEULE (Hyperliquid n'en demande aucune : les positions d'une adresse sont publiques) et écrit une
clé Redis. S'il est compromis, un attaquant lit des positions et peut écrire dans Redis (le même niveau de confiance que ``NOTHIERLY_ACTIVE_POSITIONS``) ;
il ne peut pas trader.

**Clés (variables d'environnement, jamais en argument).**

- Hyperliquid : ``HL_ACCOUNT_ADDRESS`` (et ``HL_VAULT_ADDRESS`` si le compte est un vault). Aucun secret.
- Bybit : ``BYBIT_READONLY_API_KEY`` / ``BYBIT_READONLY_API_SECRET`` ; Binance : ``BINANCE_READONLY_API_KEY`` / ``BINANCE_READONLY_API_SECRET``. Créer des clés
  SANS droit de trading ni de retrait ; réutiliser les clés de trading marcherait, mais ferait de ce service une cible aussi sensible que les workers.
- Redis : ``REDIS_HOST``, ``REDIS_PORT``, ``REDIS_PASSWORD``, ``REDIS_DB``.

**Format (clé ``exchange:positions:<PLATEFORME>``, durée de vie ``--ttl-s``, 3 intervalles par défaut)** : ``{"venue", "ts_ms", "equity", "positions":
[{"asset", "size", "price"}]}`` ; ``size`` signée en unités de base, ``price`` = prix de marque (Bybit, Binance) ou milieu (Hyperliquid). Voir
``src/agents/execution/reconciliation.py`` pour l'usage et ses limites.

**Pannes.** Un cycle raté (réseau, limite de débit, clé refusée) est journalisé et n'efface rien : la clé expire d'elle-même, et le moteur lit « pas de rapport
frais » comme tel. Une pause de débit (HTTP 429/418) est respectée (``ratelimit.py``) : le reporter dort exactement la durée demandée. Après
``--alert-after`` échecs consécutifs d'une plateforme, une alerte webhook part (``WEBHOOK_URL``) ; elle se rétablit à la première réussite.

**Non vérifié contre de vrais comptes** : les clients sont ceux des workers (testés sur fausses réponses), utilisés en lecture seule.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any, Final, Protocol

from workers.common.alerts import AlertConfigError, Alerter, Severity, WebhookAlerter
from workers.common.logging_setup import configure_logging
from workers.common.ratelimit import RateLimitError, RateLimitGate

_log = logging.getLogger(__name__)

KEY_PREFIX: Final[str] = "exchange:positions:"  # même préfixe que src/agents/execution/reconciliation.REPORT_KEY_PREFIX (un test vérifie l'accord)
VENUES: Final[tuple[str, ...]] = ("hyperliquid", "bybit", "binance")


def redis_key(venue: str) -> str:
    return f"{KEY_PREFIX}{venue.upper()}"


def build_report(venue: str, equity: Decimal, positions: Mapping[str, tuple[Decimal, Decimal]], now_ms: int) -> dict[str, Any]:
    """``positions`` : actif natif -> (taille signée, prix). Les tailles nulles sont omises ; tout est en chaînes (``Decimal``), jamais en flottant."""
    return {
        "venue": venue.upper(),
        "ts_ms": now_ms,
        "equity": format(equity, "f"),
        "positions": [{"asset": a, "size": format(size, "f"), "price": format(price, "f")} for a, (size, price) in sorted(positions.items()) if size != 0],
    }


class PositionSource(Protocol):
    """Ce qu'il faut d'une plateforme : l'équité, les positions (taille signée, prix de marque), et la porte de débit de son client."""

    venue: str
    rate_gate: RateLimitGate

    def read(self) -> tuple[Decimal, dict[str, tuple[Decimal, Decimal]]]: ...


class HyperliquidSource:
    venue = "HYPERLIQUID"

    def __init__(self, client: Any) -> None:
        self._client = client
        self.rate_gate: RateLimitGate = client.rate_gate

    def read(self) -> tuple[Decimal, dict[str, tuple[Decimal, Decimal]]]:
        snapshot = self._client.snapshot()
        mids = self._client.mids() if snapshot.positions else {}
        return snapshot.equity, {coin: (size, mids[coin]) for coin, size in snapshot.positions.items() if size != 0 and coin in mids}


class RestSource:
    """Bybit et Binance : ``snapshot()`` donne déjà ``(taille signée, prix de marque)`` par symbole."""

    def __init__(self, venue: str, client: Any) -> None:
        self.venue = venue.upper()
        self._client = client
        self.rate_gate: RateLimitGate = client.rate_gate

    def read(self) -> tuple[Decimal, dict[str, tuple[Decimal, Decimal]]]:
        snapshot = self._client.snapshot()
        return snapshot.equity, {sym: (p.size, p.mark_price) for sym, p in snapshot.positions.items() if p.size != 0}


class ReportSink(Protocol):
    async def publish(self, venue: str, report: Mapping[str, Any], ttl_s: float) -> None: ...

    async def close(self) -> None: ...


class RedisReportSink:
    """``SET exchange:positions:<PLATEFORME> <json> PX <ttl>`` (``redis.asyncio`` ; pas de ``core``)."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 6379, password: str | None = None, db: int = 0) -> None:
        from redis.asyncio import Redis

        self._redis: Any = Redis(host=host, port=port, password=password, db=db, decode_responses=True, socket_timeout=5.0, socket_connect_timeout=5.0)

    def __repr__(self) -> str:
        return "RedisReportSink()"  # jamais le mot de passe

    async def publish(self, venue: str, report: Mapping[str, Any], ttl_s: float) -> None:
        await self._redis.set(redis_key(venue), json.dumps(report, separators=(",", ":")), px=max(1, int(ttl_s * 1000)))

    async def close(self) -> None:
        await self._redis.aclose()


class PositionReporter:
    def __init__(
        self,
        sources: Sequence[PositionSource],
        sink: ReportSink,
        *,
        interval_s: float = 30.0,
        ttl_s: float | None = None,
        alerter: Alerter | None = None,
        alert_after: int = 5,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        if interval_s <= 0 or not sources:
            raise ValueError("interval_s > 0 et au moins une source sont requis")
        self._sources = list(sources)
        self._sink = sink
        self._interval_s = interval_s
        self._ttl_s = ttl_s if ttl_s is not None else 3 * interval_s
        if self._ttl_s < 2 * interval_s:
            raise ValueError("ttl_s doit couvrir au moins deux intervalles (sinon un seul cycle raté efface le rapport)")
        self._alerter = alerter
        self._alert_after = alert_after
        self._clock_ms = clock_ms
        self._failures: dict[str, int] = {s.venue: 0 for s in self._sources}
        self._alerted: set[str] = set()
        self.published: dict[str, int] = {s.venue: 0 for s in self._sources}

    async def report_once(self, source: PositionSource) -> bool:
        """Un cycle pour une plateforme. ``False`` si lecture ou écriture a échoué (journalisé, jamais levé)."""
        try:
            await source.rate_gate.wait()  # pause de débit en cours : on dort la durée exacte, sans requête
            equity, positions = await asyncio.to_thread(source.read)
            await self._sink.publish(source.venue, build_report(source.venue, equity, positions, self._clock_ms()), self._ttl_s)
        except asyncio.CancelledError:
            raise
        except RateLimitError as exc:
            _log.warning("reporter_rate_limited venue=%s status=%s pause_s=%.0f", source.venue, exc.status, exc.retry_after_s)
            return self._failed(source.venue, "rate_limited")
        except Exception as exc:  # noqa: BLE001 -- une plateforme en panne ne doit jamais arrêter les autres ni le service
            _log.warning("reporter_cycle_failed venue=%s error=%s", source.venue, type(exc).__name__)
            return self._failed(source.venue, type(exc).__name__)
        self.published[source.venue] += 1
        if self._failures[source.venue] >= self._alert_after and source.venue in self._alerted:
            self._alerted.discard(source.venue)
            self._alert(f"end:reporter:{source.venue}", f"le reporter de positions {source.venue} fonctionne de nouveau.", Severity.INFO)
        self._failures[source.venue] = 0
        return True

    def _failed(self, venue: str, reason: str) -> bool:
        self._failures[venue] += 1
        if self._failures[venue] == self._alert_after and venue not in self._alerted:
            self._alerted.add(venue)
            self._alert(
                f"reporter:{venue}",
                f"le reporter de positions {venue} échoue depuis {self._failures[venue]} cycles ({reason}) : au prochain redémarrage le moteur ne pourra pas "
                "adopter les positions réelles de cette plateforme.",
            )
        return False

    def _alert(self, key: str, text: str, severity: Severity = Severity.CRITICAL) -> None:
        if self._alerter is None:
            return
        try:
            self._alerter.alert(text, key=key, severity=severity)
        except Exception:  # noqa: BLE001
            _log.exception("reporter_alert_failed key=%s", key)

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await asyncio.gather(*(self.report_once(s) for s in self._sources))
            try:
                await asyncio.wait_for(stop.wait(), self._interval_s)
            except asyncio.TimeoutError:
                pass


# --- Ligne de commande -----------------------------------------------------------------------


def build_sources(venues: Sequence[str], network: str, environ: Mapping[str, str]) -> tuple[list[PositionSource], list[str]]:
    """Sources lisibles avec l'environnement donné, et les raisons pour lesquelles les autres ne le sont pas (échec rapide : jamais de plateforme ignorée en silence)."""
    sources: list[PositionSource] = []
    problems: list[str] = []
    for venue in venues:
        if venue == "hyperliquid":
            account = environ.get("HL_ACCOUNT_ADDRESS")
            if not account:
                problems.append("hyperliquid : HL_ACCOUNT_ADDRESS manquant")
                continue
            from workers.hyperliquid.client import Network, SdkHyperliquidClient

            hl_client = SdkHyperliquidClient(network=Network(network), account_address=account, agent_secret=None, vault_address=environ.get("HL_VAULT_ADDRESS"))
            sources.append(HyperliquidSource(hl_client))
        elif venue == "bybit":
            key, secret = environ.get("BYBIT_READONLY_API_KEY"), environ.get("BYBIT_READONLY_API_SECRET")
            if not key or not secret:
                problems.append("bybit : BYBIT_READONLY_API_KEY / BYBIT_READONLY_API_SECRET manquants (clés SANS droit de trading)")
                continue
            from workers.bybit.client import BybitNetwork, BybitRestClient

            sources.append(RestSource("bybit", BybitRestClient(api_key=key, api_secret=secret, network=BybitNetwork(network))))
        elif venue == "binance":
            key, secret = environ.get("BINANCE_READONLY_API_KEY"), environ.get("BINANCE_READONLY_API_SECRET")
            if not key or not secret:
                problems.append("binance : BINANCE_READONLY_API_KEY / BINANCE_READONLY_API_SECRET manquants (clés SANS droit de trading)")
                continue
            from workers.binance.client import BinanceNetwork, BinanceRestClient

            sources.append(RestSource("binance", BinanceRestClient(api_key=key, api_secret=secret, network=BinanceNetwork(network))))
    return sources, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publie les positions réelles des plateformes dans Redis (exchange:positions:<PLATEFORME>) pour la réconciliation du moteur.")
    parser.add_argument("--venues", nargs="+", choices=VENUES, default=list(VENUES))
    parser.add_argument("--network", choices=["testnet", "mainnet"], default="testnet")
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--ttl-s", type=float, default=None, help="durée de vie de la clé (défaut : 3 intervalles)")
    parser.add_argument("--alert-after", type=int, default=5)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)
    if args.interval_s <= 0 or args.alert_after < 1:
        parser.error("--interval-s doit être > 0 et --alert-after >= 1")
    try:
        alerter = WebhookAlerter.from_env(source="position-reporter")
    except AlertConfigError as exc:
        parser.error(str(exc))
    configure_logging("position-reporter", log_dir=args.log_dir)
    sources, problems = build_sources(args.venues, args.network, os.environ)
    if problems:
        for problem in problems:
            _log.error("reporter_startup_refused %s", problem)
        return 2
    sink: ReportSink = RedisReportSink(
        host=os.environ.get("REDIS_HOST", "127.0.0.1"), port=int(os.environ.get("REDIS_PORT", "6379")),
        password=os.environ.get("REDIS_PASSWORD") or None, db=int(os.environ.get("REDIS_DB", "0")),
    )
    reporter = PositionReporter(sources, sink, interval_s=args.interval_s, ttl_s=args.ttl_s, alerter=alerter, alert_after=args.alert_after)
    _log.info("position_reporter_start venues=%s network=%s interval_s=%g", [s.venue for s in sources], args.network, args.interval_s)

    async def serve() -> None:
        try:
            await reporter.run(asyncio.Event())
        finally:
            await sink.close()

    try:
        asyncio.run(serve(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("position_reporter_stopped")
    finally:
        if alerter is not None:
            alerter.close(2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

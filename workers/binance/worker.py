"""Worker Binance USDⓈ-M Futures : amène la position réelle vers l'exposition cible des ``PlatformSignal`` signés.

Réception signée, file « dernier signal par actif », coupe-circuit de drawdown, homme mort, alertes, mesure du slippage et
boucle ZMQ : voir ``workers.common.base``. Ici : lecture du compte Binance, règles de cotation (``exchangeInfo``),
planification (``planner``) et envoi.

Particularités Binance :

- **Perpétuels USDT uniquement** (``BTCUSDT``) ; les signaux ciblent ``TargetVenue.BINANCE`` (ou ``ALL``).
- **Mode one-way seulement.** Au démarrage la CLI interroge ``positionSide/dual`` et s'arrête proprement (code 2, sans trace)
  si le mode hedge est actif ; en marche, une position ``LONG``/``SHORT`` hedge arrête l'ordre (``hedge_mode_unsupported``).
- **Clé API.** Au démarrage la CLI refuse toute clé avec droit de retrait (``enableWithdrawals``, lu sur
  ``/sapi/v1/account/apiRestrictions``), et en mainnet toute clé dont les droits ne peuvent pas être lus.
- **Réponse d'ordre synchrone** (``newOrderRespType=RESULT``) : le statut journalisé est ``filled``, ``partial``, ``unfilled`` ou
  ``rejected`` ; les fills eux-mêmes viennent du flux privé (``ws_tracker.BinanceFillFeed``).
- ``max_equity_usd`` (optionnel) plafonne l'equity utilisée pour dimensionner.

Dry-run et réseau de test par défaut ; mainnet live exige ``--live`` + ``--network mainnet`` + ``--i-understand-real-funds``.
Jamais exécuté contre Binance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

from contracts.curve_transport import CurveClientConfig, CurveConfigError, transport_refusal
from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import load_secret_from_env
from workers.binance.client import (
    WS_URLS,
    BinanceApiError,
    BinanceClient,
    BinanceNetwork,
    BinanceOrderRequest,
    BinanceRestClient,
    BinanceSnapshot,
    HedgeModeError,
    check_key_permissions,
)
from workers.binance.planner import ZERO, OrderPlan, binance_symbol, plan_order, round_price, size_for_pct_step
from workers.common.paths import default_db_path, ensure_parent
from workers.common.alerts import AlertConfigError, WebhookAlerter
from workers.common.base import EmergencyOrder, ExposureWorkerBase, ExposureWorkerConfig, clear_execution_halt, dead_man_timeout_problem
from workers.common.exposure import emergency_slippage_bps
from workers.common.fills_db import FillsDatabase
from workers.common.logging_setup import configure_logging
from workers.common.state import ExecutionState
from workers.common.ws_tracker import BinanceFillFeed, FillTracker

_log = logging.getLogger(__name__)

__all__ = ["BinanceWorker", "BinanceWorkerConfig", "main"]


@dataclass(frozen=True)
class BinanceWorkerConfig(ExposureWorkerConfig):
    db_path: str | Path = "binance_worker.sqlite3"
    max_equity_usd: Decimal | None = None


class BinanceWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.BINANCE
    _tag: ClassVar[str] = "binance"
    supports_dead_man: ClassVar[bool] = True

    def __init__(self, config: BinanceWorkerConfig, client: BinanceClient, state: ExecutionState | None = None) -> None:
        super().__init__(config, state)
        self._client = client
        self._binance_cfg = config

    def map_symbol(self, symbol: str) -> str | None:
        return binance_symbol(symbol, self._cfg.symbol_overrides)

    def _sizing_equity(self, equity: Decimal) -> Decimal:
        cap = self._binance_cfg.max_equity_usd
        return equity if cap is None else min(equity, cap)

    async def _calculate_order_size(
        self, symbol: str, pct: float, *, snapshot: BinanceSnapshot | None = None, mid: Decimal | None = None
    ) -> Decimal:
        """Taille absolue (unités de base) pour ``pct`` % de l'equity, arrondie vers zéro à ``stepSize``.

        Interroge le compte et le prix si ``snapshot``/``mid`` ne sont pas fournis. Lève ``KeyError`` pour un symbole inconnu.
        """
        spec = (await asyncio.to_thread(self._client.instruments))[symbol]
        if snapshot is None:
            snapshot = await asyncio.to_thread(self._client.snapshot)
        if mid is None:
            fetched = await asyncio.to_thread(self._client.mid, symbol)
            if fetched is None:
                raise KeyError(f"pas de prix pour {symbol}")
            mid = fetched
        return size_for_pct_step(self._sizing_equity(snapshot.equity), mid, Decimal(str(pct)), spec.qty_step)

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        """Lit le compte, planifie et (hors dry-run) envoie l'ordre. Renvoie un code court, toujours journalisé."""
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if signal.is_expired(now) or now - signal.timestamp_ms > self._cfg.max_age_ms:
            self._record(signal, asset, "stale_at_execution", now, None, "")
            return "stale_at_execution"
        try:
            specs = await asyncio.to_thread(self._client.instruments)
            spec = specs.get(asset)
            if spec is None:
                self._record(signal, asset, "unknown_symbol", now, None, "")
                return "unknown_symbol"
            snapshot, mid = await asyncio.gather(
                asyncio.to_thread(self._client.snapshot), asyncio.to_thread(self._client.mid, asset)
            )
        except HedgeModeError as exc:
            _log.error("binance_hedge_mode_unsupported %s", exc)
            self._record(signal, asset, "hedge_mode_unsupported", now, None, str(exc)[:200])
            return "hedge_mode_unsupported"
        except Exception as exc:  # noqa: BLE001 -- lecture réseau : on n'ordonne pas à l'aveugle
            _log.exception("binance_read_failed")
            self._record(signal, asset, "read_error", now, None, repr(exc)[:200])
            return "read_error"
        if mid is None:
            self._record(signal, asset, "no_price", now, None, "")
            return "no_price"

        halted = self._update_halt(snapshot.equity, now)  # le drawdown se mesure sur l'equity réelle, non plafonnée
        equity = self._sizing_equity(snapshot.equity)
        target_abs = await self._calculate_order_size(asset, signal.position_size_pct, snapshot=snapshot, mid=mid)
        position = snapshot.positions.get(asset)
        others = sum((abs(p.size) * p.mark_price for sym, p in snapshot.positions.items() if sym != asset), ZERO)
        plan = plan_order(
            signal,
            symbol=asset,
            target_abs_size=target_abs,
            equity=equity,
            position=position.size if position else ZERO,
            mid=mid,
            spec=spec,
            others_notional=others,
            limits=self._cfg.limits,
            halted=halted,
        )
        if not isinstance(plan, OrderPlan):
            self._record(signal, asset, f"skipped:{plan.reason}", now, None, "")
            return plan.reason
        if self._cfg.dry_run:
            self._record(signal, asset, "dry_run", now, plan, "")
            _log.info("binance_dry_run %s %s qty=%s px=%s %s reduce_only=%s", asset, "BUY" if plan.is_buy else "SELL", plan.size, plan.limit_px, plan.tif, plan.reduce_only)
            return "dry_run"
        request = BinanceOrderRequest(plan.symbol, plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only, plan.client_order_id)
        self._intent_before_send(plan.client_order_id, plan.symbol, plan.is_buy, mid, plan.size)
        try:
            outcome = await asyncio.to_thread(self._client.place_order, request)
        except Exception as exc:  # noqa: BLE001 -- ambigu : pas de nouvel essai, le prochain signal relit la position
            _log.exception("binance_place_order_failed")
            self._record(signal, asset, "error", now, plan, repr(exc)[:200])
            return "error"
        self._intent_order_id(plan.client_order_id, outcome.order_id)
        self._record(signal, asset, outcome.status, now, plan, outcome.detail or f"orderId={outcome.order_id} executed={outcome.executed_qty}")
        return outcome.status

    async def emergency_flatten(self, asset: str, *, attempt: int, now_ms: int) -> str:
        """Ferme la position de ``asset`` : limite IOC ``reduceOnly`` à ``mid`` +/- un écart qui double à chaque tentative."""
        try:
            specs = await asyncio.to_thread(self._client.instruments)
            spec = specs.get(asset)
            if spec is None:
                self._record_emergency(asset, "unknown_symbol", now_ms, None)
                return "unknown_symbol"
            snapshot, mid = await asyncio.gather(
                asyncio.to_thread(self._client.snapshot), asyncio.to_thread(self._client.mid, asset)
            )
        except HedgeModeError as exc:
            _log.error("binance_dead_man_hedge_mode %s", exc)
            self._record_emergency(asset, "hedge_mode_unsupported", now_ms, None, str(exc)[:200])
            return "hedge_mode_unsupported"
        except Exception as exc:  # noqa: BLE001 -- lecture réseau : on réessaiera au prochain cycle
            _log.exception("binance_dead_man_read_failed")
            self._record_emergency(asset, "read_error", now_ms, None, repr(exc)[:200])
            return "read_error"
        position = snapshot.positions.get(asset)
        if position is None or position.size == 0:
            return "no_position"
        if mid is None:
            self._record_emergency(asset, "no_price", now_ms, None)
            return "no_price"
        slippage = emergency_slippage_bps(self._cfg.emergency_slippage_bps, attempt) / Decimal(10_000)
        is_buy = position.size < 0  # une position courte se ferme en achetant
        raw = mid * (1 + slippage) if is_buy else mid * (1 - slippage)
        order = EmergencyOrder(is_buy, abs(position.size), round_price(raw, spec.tick_size, up=is_buy))
        if self._cfg.dry_run:
            self._record_emergency(asset, "dry_run", now_ms, order)
            _log.warning("binance_dead_man_dry_run %s %s qty=%s px=%s (aucun ordre envoyé)", asset, "BUY" if is_buy else "SELL", order.size, order.limit_px)
            return "dry_run"
        client_id = "n" + hashlib.sha256(f"deadman:{asset}:{now_ms}:{attempt}".encode()).hexdigest()[:32]
        request = BinanceOrderRequest(asset, is_buy, order.size, order.limit_px, "IOC", True, client_id)
        self._intent_before_send(client_id, asset, is_buy, mid, order.size, "emergency")
        try:
            outcome = await asyncio.to_thread(self._client.place_order, request)
        except Exception as exc:  # noqa: BLE001 -- ambigu : la position est relue au prochain cycle, qui décide
            _log.exception("binance_dead_man_place_order_failed")
            self._record_emergency(asset, "error", now_ms, order, repr(exc)[:200])
            return "error"
        self._intent_order_id(client_id, outcome.order_id)
        self._record_emergency(asset, outcome.status, now_ms, order, outcome.detail or f"orderId={outcome.order_id}")
        return outcome.status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Worker Binance USDⓈ-M Futures (dry-run sur le réseau de test par défaut).")
    parser.add_argument("--address", default=os.environ.get("NOTHIERLY_ZMQ_ADDRESS"), help="endpoint ZMQ du publisher")
    parser.add_argument("--db", default=str(default_db_path("binance_worker.sqlite3")), help="base SQLite d'état (défaut : $NOTHIERLY_DATA_DIR ou data/)")
    parser.add_argument("--network", choices=[n.value for n in BinanceNetwork], default=BinanceNetwork.TESTNET.value)
    parser.add_argument("--base-url", default=None, help="surcharge l'adresse REST futures (le réseau de test a changé d'adresse dans la doc)")
    parser.add_argument("--ws-base-url", default=None, help="surcharge l'adresse WebSocket (ex. wss://fstream.binance.com)")
    parser.add_argument("--live", action="store_true", help="envoie réellement les ordres ; sinon dry-run")
    parser.add_argument("--i-understand-real-funds", action="store_true", help="requis pour --live sur mainnet")
    parser.add_argument("--clear-halt", action="store_true", help="lève les verrous (drawdown et homme mort) puis quitte")
    parser.add_argument("--allow-plaintext-remote", action="store_true", help="autorise un endpoint non local SANS chiffrement CURVE")
    parser.add_argument("--fills-db", default=None, help="base SQLite des fills et du slippage (défaut : binance_fills.sqlite3 à côté de --db)")
    parser.add_argument("--no-fill-tracking", action="store_true", help="désactive l'écoute des fills (activée avec --live)")
    parser.add_argument("--log-dir", default=None, help="dossier d'un fichier de journal tournant (10 Mo x 5) en plus de stdout ; défaut : NOTHIERLY_LOG_DIR")
    parser.add_argument("--dead-man-timeout", type=float, default=300.0, metavar="SECONDES",
                        help="ferme toutes les positions puis se verrouille après ce délai sans signal valide (0 = désactivé)")
    parser.add_argument("--max-equity-usd", type=Decimal, default=None, help="plafond d'equity utilisée pour dimensionner")
    args = parser.parse_args(argv)
    ensure_parent(args.db)
    if args.clear_halt:  # ne touche que la base locale : ni réseau, ni clé, ni secret du bus
        configure_logging("binance-worker", log_dir=args.log_dir)
        clear_execution_halt(ExecutionState(args.db))
        _log.info("binance_halt_cleared")
        return 0
    if args.dead_man_timeout < 0:
        parser.error("--dead-man-timeout doit être >= 0")
    if (timeout_problem := dead_man_timeout_problem(args.dead_man_timeout)) is not None:
        parser.error(timeout_problem)
    if not args.address:
        parser.error("--address ou NOTHIERLY_ZMQ_ADDRESS requis")
    try:
        curve = CurveClientConfig.from_env()
    except CurveConfigError as exc:
        parser.error(str(exc))
    refusal = transport_refusal(args.address, curve_enabled=curve is not None, allow_plaintext_remote=args.allow_plaintext_remote)
    if refusal:
        parser.error(refusal)
    network = BinanceNetwork(args.network)
    mainnet = network is BinanceNetwork.MAINNET
    if args.live and mainnet and not args.i_understand_real_funds:
        parser.error("--live sur mainnet exige --i-understand-real-funds")
    if args.live and mainnet and args.dead_man_timeout == 0:
        parser.error("--live sur mainnet exige l'homme mort : --dead-man-timeout doit être > 0")
    key, secret = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
    if not key or not secret:
        parser.error("BINANCE_API_KEY et BINANCE_API_SECRET requis (même en dry-run : solde et positions sont privés ; clé en lecture seule conseillée)")
    try:
        alerter = WebhookAlerter.from_env(source="binance-worker")
    except AlertConfigError as exc:
        parser.error(str(exc))
    configure_logging("binance-worker", log_dir=args.log_dir)
    client = BinanceRestClient(api_key=key, api_secret=secret, network=network, base_url=args.base_url)
    try:
        permissions = client.key_permissions()
        if not permissions.verified:
            _log.warning("binance_key_permissions_unverified reason=%s", permissions.reason)
        problems = check_key_permissions(permissions, live=args.live, mainnet=mainnet)
        if not problems and client.is_hedge_mode():
            problems.append("le compte est en mode hedge (dualSidePosition=true) : ce worker ne gère que le mode one-way, à changer dans Binance")
    except (BinanceApiError, OSError, ValueError) as exc:
        problems = [f"le compte Binance n'a pas pu être interrogé au démarrage ({type(exc).__name__}) : arrêt, rien n'est supposé"]
    if problems:
        for problem in problems:
            _log.error("binance_startup_refused %s", problem)
        return 2
    fills_db: FillsDatabase | None = None
    tracker: FillTracker | None = None
    if args.live and not args.no_fill_tracking:  # en dry-run aucun ordre ne part : rien à mesurer
        fills_db = FillsDatabase(args.fills_db or Path(args.db).with_name("binance_fills.sqlite3"))
        feed = BinanceFillFeed(api=client, ws_base_url=args.ws_base_url or WS_URLS[network])
        tracker = FillTracker(feed, fills_db, alerter=alerter)
    config = BinanceWorkerConfig(
        address=args.address,
        secret_key=load_secret_from_env(),
        db_path=args.db,
        dry_run=not args.live,
        max_equity_usd=args.max_equity_usd,
        dead_man_timeout_s=args.dead_man_timeout or None,
        curve=curve,
        alerter=alerter,
        fills_db=fills_db,
        fill_tracker=tracker,
    )
    worker = BinanceWorker(config, client)
    _log.info("binance_worker_start network=%s mode=%s", network.value, "LIVE" if args.live else "DRY-RUN")
    try:
        # Boucle Selector explicite : sous Windows, la boucle Proactor par défaut est incompatible avec zmq.asyncio.
        asyncio.run(worker.run(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("binance_worker_stopped")
    finally:
        if alerter is not None:
            alerter.close(2.0)
        if fills_db is not None:
            fills_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

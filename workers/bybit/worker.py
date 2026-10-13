"""Worker Bybit : amène la position réelle vers l'exposition cible des ``PlatformSignal`` signés.

Réception signée, file « dernier signal par actif », coupe-circuit de drawdown et boucle ZMQ : voir
``workers.common.base``. Ici : lecture du compte Bybit, règles de cotation, planification (``planner``) et envoi.

Particularités Bybit :

- **Perpétuels USDT uniquement** pour le copy trading ; ``require_copy_trading`` (défaut : oui) refuse un symbole
  dont le champ ``copyTrading`` de ``instruments-info`` vaut ``none``.
- **Réponse d'ordre asynchrone** : Bybit n'accuse que la réception. Le statut journalisé est ``accepted`` ; le
  remplissage réel se lit dans la position au signal suivant (c'est la source de vérité de la logique d'exposition).
- **Mode one-way seulement** : une position hedge arrête l'ordre (``hedge_mode_unsupported``).
- **Clé API** : au démarrage la CLI refuse toute clé avec droit de retrait, et en mode live toute clé sans droit de trading.
- ``max_equity_usd`` (optionnel) plafonne l'equity utilisée pour dimensionner : garde-fou tant que le compte
  qui détient les fonds d'un master trader n'est pas confirmé (voir ``client``).

Dry-run et testnet par défaut ; mainnet live exige ``--live`` + ``--network mainnet`` + ``--i-understand-real-funds``.
Jamais exécuté contre Bybit.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import load_secret_from_env
from workers.bybit.client import (
    BybitClient,
    BybitNetwork,
    BybitOrderRequest,
    BybitRestClient,
    BybitSnapshot,
    HedgeModeError,
    check_key_info,
)
from workers.bybit.planner import ZERO, OrderPlan, bybit_symbol, plan_order, size_for_pct_step
from workers.common.base import ExposureWorkerBase, ExposureWorkerConfig
from workers.common.state import ExecutionState

_log = logging.getLogger(__name__)

__all__ = ["BybitWorker", "BybitWorkerConfig", "main"]


@dataclass(frozen=True)
class BybitWorkerConfig(ExposureWorkerConfig):
    db_path: str | Path = "bybit_worker.sqlite3"
    require_copy_trading: bool = True
    max_equity_usd: Decimal | None = None


class BybitWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.BYBIT
    _tag: ClassVar[str] = "bybit"

    def __init__(self, config: BybitWorkerConfig, client: BybitClient, state: ExecutionState | None = None) -> None:
        super().__init__(config, state)
        self._client = client
        self._bybit_cfg = config

    def map_symbol(self, symbol: str) -> str | None:
        return bybit_symbol(symbol, self._cfg.symbol_overrides)

    async def _calculate_order_size(
        self, symbol: str, pct: float, *, snapshot: BybitSnapshot | None = None, mid: Decimal | None = None
    ) -> Decimal:
        """Taille absolue (unités de base) pour ``pct`` % de l'equity, arrondie vers zéro à ``qtyStep``.

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

    def _sizing_equity(self, equity: Decimal) -> Decimal:
        cap = self._bybit_cfg.max_equity_usd
        return equity if cap is None else min(equity, cap)

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
            if self._bybit_cfg.require_copy_trading and spec.copy_trading == "none":
                self._record(signal, asset, "not_copy_tradable", now, None, "")
                return "not_copy_tradable"
            snapshot, mid = await asyncio.gather(
                asyncio.to_thread(self._client.snapshot), asyncio.to_thread(self._client.mid, asset)
            )
        except HedgeModeError as exc:
            _log.error("bybit_hedge_mode_unsupported %s", exc)
            self._record(signal, asset, "hedge_mode_unsupported", now, None, str(exc)[:200])
            return "hedge_mode_unsupported"
        except Exception as exc:  # noqa: BLE001 -- lecture réseau : on n'ordonne pas à l'aveugle
            _log.exception("bybit_read_failed")
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
            _log.info("bybit_dry_run %s %s qty=%s px=%s %s reduce_only=%s", asset, "BUY" if plan.is_buy else "SELL", plan.size, plan.limit_px, plan.tif, plan.reduce_only)
            return "dry_run"
        request = BybitOrderRequest(plan.symbol, plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only, plan.order_link_id)
        try:
            outcome = await asyncio.to_thread(self._client.place_order, request)
        except Exception as exc:  # noqa: BLE001 -- ambigu : pas de nouvel essai, le prochain signal relit la position
            _log.exception("bybit_place_order_failed")
            self._record(signal, asset, "error", now, plan, repr(exc)[:200])
            return "error"
        self._record(signal, asset, outcome.status, now, plan, outcome.detail or f"orderId={outcome.order_id}")
        return outcome.status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Worker Bybit copy trading (dry-run testnet par défaut).")
    parser.add_argument("--address", default=os.environ.get("NOTHIERLY_ZMQ_ADDRESS"), help="endpoint ZMQ du publisher")
    parser.add_argument("--db", default="bybit_worker.sqlite3")
    parser.add_argument("--network", choices=[n.value for n in BybitNetwork], default=BybitNetwork.TESTNET.value)
    parser.add_argument("--live", action="store_true", help="envoie réellement les ordres ; sinon dry-run")
    parser.add_argument("--i-understand-real-funds", action="store_true", help="requis pour --live sur mainnet")
    parser.add_argument("--clear-halt", action="store_true", help="lève le mode réduction seule puis quitte")
    parser.add_argument("--allow-non-copy-symbols", action="store_true", help="n'exige pas copyTrading != none")
    parser.add_argument("--max-equity-usd", type=Decimal, default=None, help="plafond d'equity utilisée pour dimensionner")
    args = parser.parse_args(argv)
    if not args.address:
        parser.error("--address ou NOTHIERLY_ZMQ_ADDRESS requis")
    network = BybitNetwork(args.network)
    if args.live and network is BybitNetwork.MAINNET and not args.i_understand_real_funds:
        parser.error("--live sur mainnet exige --i-understand-real-funds")
    key, secret = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
    if not key or not secret:
        parser.error("BYBIT_API_KEY et BYBIT_API_SECRET requis (même en dry-run : solde et positions sont privés ; clé en lecture seule conseillée)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = BybitRestClient(api_key=key, api_secret=secret, network=network)
    problems = check_key_info(client.key_info(), live=args.live)
    if problems:
        for problem in problems:
            _log.error("bybit_key_refused %s", problem)
        return 2
    config = BybitWorkerConfig(
        address=args.address,
        secret_key=load_secret_from_env(),
        db_path=args.db,
        dry_run=not args.live,
        require_copy_trading=not args.allow_non_copy_symbols,
        max_equity_usd=args.max_equity_usd,
    )
    worker = BybitWorker(config, client)
    if args.clear_halt:
        worker.clear_halt()
        _log.info("bybit_halt_cleared")
        return 0
    _log.info("bybit_worker_start network=%s mode=%s", network.value, "LIVE" if args.live else "DRY-RUN")
    try:
        # Boucle Selector explicite : sous Windows, la boucle Proactor par défaut est incompatible avec zmq.asyncio.
        asyncio.run(worker.run(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("bybit_worker_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Worker Hyperliquid : amène la position réelle vers l'exposition cible des ``PlatformSignal`` signés.

La réception signée, la file « dernier signal par actif », le coupe-circuit de drawdown et la boucle ZMQ
sont dans ``workers.common.base`` (voir ses choix de conception). Ce module ne contient que ce qui est propre
à Hyperliquid : traduction des symboles, lecture du compte, planification (``planner``) et envoi de l'ordre.

Sûr par défaut : dry-run et testnet, sauf demande explicite. Le mainnet exige ``--live`` ET
``--network mainnet`` ET ``--i-understand-real-funds``.

Ce worker n'a jamais été exécuté contre Hyperliquid (ni testnet ni mainnet) : voir ``client``.
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
from workers.common.base import DEFAULT_MAX_AGE_MS, ExposureWorkerBase, ExposureWorkerConfig
from workers.hyperliquid.client import AccountSnapshot, HyperliquidClient, Network, OrderRequest, SdkHyperliquidClient
from workers.hyperliquid.planner import ZERO, OrderPlan, hl_asset, plan_order, size_for_pct
from workers.hyperliquid.state import HlState

_log = logging.getLogger(__name__)

__all__ = ["DEFAULT_MAX_AGE_MS", "HyperliquidWorker", "HyperliquidWorkerConfig", "main"]


@dataclass(frozen=True)
class HyperliquidWorkerConfig(ExposureWorkerConfig):
    db_path: str | Path = "hyperliquid_worker.sqlite3"


class HyperliquidWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.HYPERLIQUID
    _tag: ClassVar[str] = "hl"

    def __init__(self, config: HyperliquidWorkerConfig, client: HyperliquidClient, state: HlState | None = None) -> None:
        super().__init__(config, state)
        self._client = client

    def map_symbol(self, symbol: str) -> str | None:
        return hl_asset(symbol, self._cfg.symbol_overrides)

    async def _calculate_order_size(
        self, asset: str, pct: float, *, snapshot: AccountSnapshot | None = None, mid: Decimal | None = None
    ) -> Decimal:
        """Taille absolue (unités de ``asset``) correspondant à ``pct`` % de l'equity, arrondie vers zéro à ``szDecimals``.

        Interroge le compte et le prix si ``snapshot``/``mid`` ne sont pas fournis. Lève ``KeyError`` pour un actif inconnu.
        """
        decimals = (await asyncio.to_thread(self._client.asset_decimals))[asset]
        if snapshot is None:
            snapshot = await asyncio.to_thread(self._client.snapshot)
        if mid is None:
            mid = (await asyncio.to_thread(self._client.mids))[asset]
        return size_for_pct(snapshot.equity, mid, Decimal(str(pct)), decimals)

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        """Lit le compte, planifie et (hors dry-run) envoie l'ordre. Renvoie un code court, toujours journalisé."""
        now = int(time.time() * 1000) if now_ms is None else now_ms
        if signal.is_expired(now) or now - signal.timestamp_ms > self._cfg.max_age_ms:
            self._record(signal, asset, "stale_at_execution", now, None, "")
            return "stale_at_execution"
        try:
            decimals_map = await asyncio.to_thread(self._client.asset_decimals)
            if asset not in decimals_map:
                self._record(signal, asset, "unknown_asset", now, None, "")
                return "unknown_asset"
            snapshot, mids = await asyncio.gather(
                asyncio.to_thread(self._client.snapshot), asyncio.to_thread(self._client.mids)
            )
        except Exception as exc:  # noqa: BLE001 -- lecture réseau : on n'ordonne pas à l'aveugle
            _log.exception("hl_read_failed")
            self._record(signal, asset, "read_error", now, None, repr(exc)[:200])
            return "read_error"
        mid = mids.get(asset)
        if mid is None:
            self._record(signal, asset, "no_price", now, None, "")
            return "no_price"

        halted = self._update_halt(snapshot.equity, now)
        target_abs = await self._calculate_order_size(asset, signal.position_size_pct, snapshot=snapshot, mid=mid)
        others = sum(
            (abs(size) * mids.get(coin, ZERO) for coin, size in snapshot.positions.items() if coin != asset), ZERO
        )
        plan = plan_order(
            signal,
            asset=asset,
            target_abs_size=target_abs,
            equity=snapshot.equity,
            position=snapshot.positions.get(asset, ZERO),
            mid=mid,
            sz_decimals=decimals_map[asset],
            others_notional=others,
            limits=self._cfg.limits,
            halted=halted,
        )
        if not isinstance(plan, OrderPlan):
            self._record(signal, asset, f"skipped:{plan.reason}", now, None, "")
            return plan.reason
        if self._cfg.dry_run:
            self._record(signal, asset, "dry_run", now, plan, "")
            _log.info("hl_dry_run %s %s sz=%s px=%s reduce_only=%s", asset, "BUY" if plan.is_buy else "SELL", plan.size, plan.limit_px, plan.reduce_only)
            return "dry_run"
        request = OrderRequest(plan.asset, plan.is_buy, plan.size, plan.limit_px, plan.tif, plan.reduce_only, plan.cloid)
        try:
            outcome = await asyncio.to_thread(self._client.place_order, request)
        except Exception as exc:  # noqa: BLE001 -- ambigu : pas de nouvel essai, le prochain signal relit la position
            _log.exception("hl_place_order_failed")
            self._record(signal, asset, "error", now, plan, repr(exc)[:200])
            return "error"
        self._record(signal, asset, outcome.status, now, plan, outcome.detail or f"oid={outcome.oid} filled={outcome.filled_size}")
        return outcome.status


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Worker Hyperliquid (dry-run testnet par défaut).")
    parser.add_argument("--address", default=os.environ.get("NOTHIERLY_ZMQ_ADDRESS"), help="endpoint ZMQ du publisher")
    parser.add_argument("--db", default="hyperliquid_worker.sqlite3")
    parser.add_argument("--network", choices=[n.value for n in Network], default=Network.TESTNET.value)
    parser.add_argument("--live", action="store_true", help="envoie réellement les ordres ; sinon dry-run (lecture seule)")
    parser.add_argument("--i-understand-real-funds", action="store_true", help="requis pour --live sur mainnet")
    parser.add_argument("--clear-halt", action="store_true", help="lève le mode réduction seule puis quitte")
    args = parser.parse_args(argv)
    if not args.address:
        parser.error("--address ou NOTHIERLY_ZMQ_ADDRESS requis")
    network = Network(args.network)
    if args.live and network is Network.MAINNET and not args.i_understand_real_funds:
        parser.error("--live sur mainnet exige --i-understand-real-funds")
    account = os.environ.get("HL_ACCOUNT_ADDRESS")
    if not account:
        parser.error("HL_ACCOUNT_ADDRESS requis (compte principal dont l'equity sert de base)")
    secret = os.environ.get("HL_AGENT_SECRET") if args.live else None
    if args.live and not secret:
        parser.error("--live exige HL_AGENT_SECRET (clé d'un API wallet sans droit de retrait)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = SdkHyperliquidClient(
        network=network, account_address=account, agent_secret=secret, vault_address=os.environ.get("HL_VAULT_ADDRESS")
    )
    config = HyperliquidWorkerConfig(
        address=args.address, secret_key=load_secret_from_env(), db_path=args.db, dry_run=not args.live
    )
    worker = HyperliquidWorker(config, client)
    if args.clear_halt:
        worker.clear_halt()
        _log.info("hl_halt_cleared")
        return
    _log.info("hl_worker_start network=%s mode=%s", network.value, "LIVE" if args.live else "DRY-RUN")
    try:
        # Boucle Selector explicite : sous Windows, la boucle Proactor par défaut est incompatible avec zmq.asyncio.
        asyncio.run(worker.run(), loop_factory=asyncio.SelectorEventLoop)
    except KeyboardInterrupt:
        _log.info("hl_worker_stopped")


if __name__ == "__main__":
    main()

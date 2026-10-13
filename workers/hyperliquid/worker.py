"""Worker Hyperliquid : écoute les ``PlatformSignal`` signés et amène la position réelle vers l'exposition cible.

Flux : bus ZMQ -> ``open_envelope`` (signature, plateforme, âge, rejeu) -> file « dernier signal par
actif » -> exécuteur : lecture du compte (equity, positions, prix) -> ``plan_order`` -> ordre IOC.

Choix de conception à connaître :

1. **Exposition cible, pas ordre.** ``position_size_pct`` est la part d'equity visée ; le worker envoie
   l'écart avec la position réelle. Republier le même signal ne fait rien (``on_target``). Le moteur
   publiant à chaque tick, c'est ce qui empêche les positions de s'empiler. Voir ``planner``.
2. **Pas de nouvelle tentative automatique d'un ordre.** Après une erreur réseau ambiguë, on ne sait pas
   si l'ordre est passé. Rejouer risquerait un doublon ; le prochain signal recalcule l'écart depuis la
   position réelle, ce qui est exact dans les deux cas.
3. **Dernier signal par actif.** Pendant qu'un ordre part, les signaux suivants s'accumulent sans bloquer
   la réception ; pour un même actif, seul le plus récent est exécuté (les anciens sont périmés par
   construction avec une cible). Fraîcheur revérifiée juste avant l'envoi.
4. **Coupe-circuit local.** Si l'equity du compte perd ``max_daily_drawdown_pct`` depuis la base du jour,
   le worker passe en réduction seule, de façon persistante (``clear_halt`` pour le lever). Un retrait ou un
   dépôt dans la journée fausse cette mesure : elle se base sur l'equity brute du compte.
5. **Sûr par défaut.** Dry-run (aucun ordre envoyé) et testnet, sauf demande explicite. Le mainnet exige
   ``--live`` ET ``--network mainnet`` ET ``--i-understand-real-funds``.

Ce worker n'a jamais été exécuté contre Hyperliquid (ni testnet ni mainnet) : voir ``client``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import zmq
import zmq.asyncio

from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, ReplayGuard, load_secret_from_env, open_envelope, verify_signature
from workers.hyperliquid.client import AccountSnapshot, HyperliquidClient, Network, OrderRequest, SdkHyperliquidClient
from workers.hyperliquid.planner import ZERO, OrderPlan, RiskLimits, hl_asset, plan_order, size_for_pct
from workers.hyperliquid.state import ExecutionRecord, HlState

_log = logging.getLogger(__name__)

DEFAULT_MAX_AGE_MS = 5_000
_REPLAY_WINDOW_MS = 60 * 60 * 1000


@dataclass(frozen=True)
class HyperliquidWorkerConfig:
    address: str  # endpoint ZMQ du publisher
    secret_key: str | bytes  # clé HMAC du bus (NOTHIERLY_ZMQ_SECRET)
    topic: bytes = DEFAULT_TOPIC
    db_path: str | Path = "hyperliquid_worker.sqlite3"
    limits: RiskLimits = RiskLimits()
    max_age_ms: int = DEFAULT_MAX_AGE_MS
    symbol_overrides: Mapping[str, str] = field(default_factory=dict)
    dry_run: bool = True
    poll_ms: int = 200


class HyperliquidWorker:
    def __init__(self, config: HyperliquidWorkerConfig, client: HyperliquidClient, state: HlState | None = None) -> None:
        verify_signature({}, config.secret_key)  # lève WeakSecretError dès le démarrage si la clé est faible
        self._cfg = config
        self._client = client
        self._state = state if state is not None else HlState(config.db_path)
        self._replay_guard = ReplayGuard()
        self._pending: dict[str, PlatformSignal] = {}
        now_ms = int(time.time() * 1000)
        for signal_id in self._state.recent_signal_ids(now_ms - _REPLAY_WINDOW_MS):
            self._replay_guard.check_and_add(uuid.UUID(signal_id))  # un redémarrage n'ouvre pas de fenêtre de rejeu
        self._state.prune_seen_before(now_ms - 2 * _REPLAY_WINDOW_MS)

    @property
    def state(self) -> HlState:
        return self._state

    @property
    def halted(self) -> bool:
        return self._state.get("halted") == "1"

    def clear_halt(self) -> None:
        """Lève le mode réduction seule. Action d'opérateur, jamais automatique."""
        self._state.delete("halted")

    # --- réception --------------------------------------------------------------------------

    def handle_frames(self, frames: Sequence[bytes], *, now_ms: int | None = None) -> str:
        """Valide un message ``[topic, enveloppe]`` et met le signal en attente. ``queued`` ou la raison du rejet."""
        if len(frames) != 2 or frames[0] != self._cfg.topic:
            return "bad_frames"
        now = int(time.time() * 1000) if now_ms is None else now_ms
        verdict = open_envelope(
            frames[1],
            self._cfg.secret_key,
            venue=TargetVenue.HYPERLIQUID,
            max_age_ms=self._cfg.max_age_ms,
            replay_guard=self._replay_guard,
            now_ms=now,
        )
        signal = verdict.signal
        if signal is None:
            return verdict.reject_reason or "rejected"
        self._state.remember_signal(str(signal.signal_id), now)
        asset = hl_asset(signal.symbol, self._cfg.symbol_overrides)
        if asset is None:
            _log.warning("hl_unmappable_symbol symbol=%s", signal.symbol)
            return "unmappable_symbol"
        current = self._pending.get(asset)
        if current is None or signal.timestamp_ms >= current.timestamp_ms:
            self._pending[asset] = signal  # le plus récent remplace : avec une cible, l'ancien est caduc
        return "queued"

    # --- exécution --------------------------------------------------------------------------

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

    def _update_halt(self, equity: Decimal, now_ms: int) -> bool:
        day = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime("%Y%m%d")
        if self._state.get("dd_day") != day:
            self._state.set("dd_day", day)
            self._state.set("dd_start_equity", str(equity))
        if self.halted:
            return True
        start = Decimal(self._state.get("dd_start_equity") or equity)
        if start > 0 and (start - equity) / start * 100 >= self._cfg.limits.max_daily_drawdown_pct:
            self._state.set("halted", "1")
            _log.critical("hl_drawdown_halt start=%s equity=%s : réduction seule jusqu'à intervention", start, equity)
            return True
        return False

    def _record(self, signal: PlatformSignal, asset: str, status: str, now_ms: int, plan: OrderPlan | None, detail: str) -> None:
        self._state.log_execution(
            ExecutionRecord(
                ts_ms=now_ms,
                signal_id=str(signal.signal_id),
                asset=asset,
                status=status,
                side="" if plan is None else ("BUY" if plan.is_buy else "SELL"),
                size="" if plan is None else str(plan.size),
                px="" if plan is None else str(plan.limit_px),
                detail=detail,
            )
        )

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

    # --- boucle -----------------------------------------------------------------------------

    async def _executor(self, stop: asyncio.Event, wake: asyncio.Event) -> None:
        while not stop.is_set():
            await wake.wait()
            wake.clear()
            while self._pending and not stop.is_set():
                asset = next(iter(self._pending))
                signal = self._pending.pop(asset)
                try:
                    await self.execute_signal(signal, asset)
                except Exception:  # noqa: BLE001 -- un signal ne doit jamais arrêter l'exécuteur
                    _log.exception("hl_execute_failed asset=%s", asset)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Écoute le bus sans bloquer ; l'exécution réseau tourne dans une tâche séparée.

        Sous Windows, à lancer sur une ``asyncio.SelectorEventLoop`` (``zmq.asyncio`` ne fonctionne pas avec la
        boucle Proactor par défaut, sauf si ``tornado`` est installé).
        """
        stop = stop if stop is not None else asyncio.Event()
        wake = asyncio.Event()
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, self._cfg.topic)
        socket.connect(self._cfg.address)
        executor = asyncio.create_task(self._executor(stop, wake))
        try:
            while not stop.is_set():
                if await socket.poll(self._cfg.poll_ms):
                    frames = await socket.recv_multipart()
                    try:
                        if self.handle_frames(frames) == "queued":
                            wake.set()
                    except Exception:  # noqa: BLE001 -- un message ne doit jamais arrêter le worker
                        _log.exception("hl_handle_frames_failed")
        finally:
            stop.set()
            wake.set()
            await asyncio.gather(executor, return_exceptions=True)
            socket.close(linger=0)
            context.term()


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

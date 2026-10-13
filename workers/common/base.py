"""Socle commun des workers d'exécution temps réel : réception signée, file « dernier signal par actif »,
coupe-circuit de drawdown, journal d'audit et boucle ZMQ.

Une plateforme fournit seulement : ``venue``, ``map_symbol`` (symbole interne -> symbole de la plateforme)
et ``execute_signal`` (lire le compte, planifier, ordonner). Tout le reste est ici, une seule fois.

Choix de conception (communs à tous les workers temps réel) :

1. **Exposition cible, pas ordre** : voir ``workers/common/exposure.py``.
2. **Pas de nouvelle tentative automatique d'un ordre.** Après une erreur réseau ambiguë on ne sait pas si
   l'ordre est passé ; rejouer risquerait un doublon. Le prochain signal recalcule l'écart depuis la
   position réelle, ce qui est exact dans les deux cas.
3. **Dernier signal par actif.** Pendant qu'un ordre part, les signaux suivants s'accumulent sans bloquer
   la réception ; pour un même actif, seul le plus récent est exécuté. Fraîcheur revérifiée avant l'envoi.
4. **Coupe-circuit local.** Si l'equity du compte perd ``max_daily_drawdown_pct`` depuis la base du jour, le
   worker passe en réduction seule, de façon persistante (``clear_halt`` pour le lever). Un retrait ou un
   dépôt dans la journée fausse cette mesure : elle se base sur l'equity brute du compte.
5. **Sûr par défaut** : dry-run (aucun ordre envoyé) tant que ``dry_run`` n'est pas explicitement désactivé.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Protocol

import zmq
import zmq.asyncio

from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, ReplayGuard, open_envelope, verify_signature
from workers.common.exposure import RiskLimits
from workers.common.state import ExecutionRecord, ExecutionState

_log = logging.getLogger(__name__)

DEFAULT_MAX_AGE_MS = 5_000
_REPLAY_WINDOW_MS = 60 * 60 * 1000


class PlannedOrder(Protocol):
    @property
    def is_buy(self) -> bool: ...

    @property
    def size(self) -> Decimal: ...

    @property
    def limit_px(self) -> Decimal: ...


@dataclass(frozen=True)
class ExposureWorkerConfig:
    address: str  # endpoint ZMQ du publisher
    secret_key: str | bytes  # clé HMAC du bus (NOTHIERLY_ZMQ_SECRET)
    topic: bytes = DEFAULT_TOPIC
    db_path: str | Path = "execution_worker.sqlite3"
    limits: RiskLimits = RiskLimits()
    max_age_ms: int = DEFAULT_MAX_AGE_MS
    symbol_overrides: Mapping[str, str] = field(default_factory=dict)
    dry_run: bool = True
    poll_ms: int = 200


class ExposureWorkerBase(ABC):
    venue: ClassVar[TargetVenue]
    _tag: ClassVar[str] = "worker"  # préfixe des événements de journal

    def __init__(self, config: ExposureWorkerConfig, state: ExecutionState | None = None) -> None:
        verify_signature({}, config.secret_key)  # lève WeakSecretError dès le démarrage si la clé est faible
        self._cfg = config
        self._state = state if state is not None else ExecutionState(config.db_path)
        self._replay_guard = ReplayGuard()
        self._pending: dict[str, PlatformSignal] = {}
        now_ms = int(time.time() * 1000)
        for signal_id in self._state.recent_signal_ids(now_ms - _REPLAY_WINDOW_MS):
            self._replay_guard.check_and_add(uuid.UUID(signal_id))  # un redémarrage n'ouvre pas de fenêtre de rejeu
        self._state.prune_seen_before(now_ms - 2 * _REPLAY_WINDOW_MS)

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def halted(self) -> bool:
        return self._state.get("halted") == "1"

    def clear_halt(self) -> None:
        """Lève le mode réduction seule. Action d'opérateur, jamais automatique."""
        self._state.delete("halted")

    @abstractmethod
    def map_symbol(self, symbol: str) -> str | None:
        """Symbole interne (``CRYPTO_BTC_USD``) -> symbole de la plateforme, ou ``None`` si intraduisible."""

    @abstractmethod
    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        """Lit le compte, planifie et (hors dry-run) envoie l'ordre. Renvoie un code court, toujours journalisé."""

    # --- réception --------------------------------------------------------------------------

    def handle_frames(self, frames: Sequence[bytes], *, now_ms: int | None = None) -> str:
        """Valide un message ``[topic, enveloppe]`` et met le signal en attente. ``queued`` ou la raison du rejet."""
        if len(frames) != 2 or frames[0] != self._cfg.topic:
            return "bad_frames"
        now = int(time.time() * 1000) if now_ms is None else now_ms
        verdict = open_envelope(
            frames[1],
            self._cfg.secret_key,
            venue=self.venue,
            max_age_ms=self._cfg.max_age_ms,
            replay_guard=self._replay_guard,
            now_ms=now,
        )
        signal = verdict.signal
        if signal is None:
            return verdict.reject_reason or "rejected"
        self._state.remember_signal(str(signal.signal_id), now)
        asset = self.map_symbol(signal.symbol)
        if asset is None:
            _log.warning("%s_unmappable_symbol symbol=%s", self._tag, signal.symbol)
            return "unmappable_symbol"
        current = self._pending.get(asset)
        if current is None or signal.timestamp_ms >= current.timestamp_ms:
            self._pending[asset] = signal  # le plus récent remplace : avec une cible, l'ancien est caduc
        return "queued"

    # --- risque et audit --------------------------------------------------------------------

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
            _log.critical("%s_drawdown_halt start=%s equity=%s : réduction seule jusqu'à intervention", self._tag, start, equity)
            return True
        return False

    def _record(
        self, signal: PlatformSignal, asset: str, status: str, now_ms: int, plan: PlannedOrder | None, detail: str
    ) -> None:
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
                    _log.exception("%s_execute_failed asset=%s", self._tag, asset)

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
                        _log.exception("%s_handle_frames_failed", self._tag)
        finally:
            stop.set()
            wake.set()
            await asyncio.gather(executor, return_exceptions=True)
            socket.close(linger=0)
            context.term()

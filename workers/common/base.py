"""Socle commun des workers d'exécution temps réel : réception signée, file « dernier signal par actif »,
coupe-circuits (drawdown, homme mort), journal d'audit et boucle ZMQ.

Une plateforme fournit seulement : ``venue``, ``map_symbol`` (symbole interne -> symbole de la plateforme),
``execute_signal`` (lire le compte, planifier, ordonner) et, si elle supporte l'homme mort, ``emergency_flatten``.
Tout le reste est ici, une seule fois.

Choix de conception (communs à tous les workers temps réel) :

1. **Exposition cible, pas ordre** : voir ``workers/common/exposure.py``.
2. **Pas de nouvelle tentative automatique d'un ordre.** Après une erreur réseau ambiguë on ne sait pas si
   l'ordre est passé ; rejouer risquerait un doublon. Le prochain signal recalcule l'écart depuis la
   position réelle, ce qui est exact dans les deux cas.
3. **Dernier signal par actif.** Pendant qu'un ordre part, les signaux suivants s'accumulent sans bloquer
   la réception ; pour un même actif, seul le plus récent est exécuté. Fraîcheur revérifiée avant l'envoi.
4. **Coupe-circuit de drawdown.** Si l'equity du compte perd ``max_daily_drawdown_pct`` depuis la base du jour, le
   worker passe en réduction seule, de façon persistante. Un retrait ou un dépôt dans la journée fausse cette
   mesure : elle se base sur l'equity brute du compte.
5. **Homme mort (``dead_man_timeout_s``).** Si aucun signal valide n'arrive pour un actif pendant plus de N secondes
   alors qu'une position est ouverte, le worker ferme TOUTES ses positions puis se verrouille (``HALTED``). Détail
   ci-dessous.
6. **Sûr par défaut** : dry-run (aucun ordre envoyé) tant que ``dry_run`` n'est pas explicitement désactivé.

**Homme mort.** Le moteur ou le réseau peut tomber, ou Windows redémarrer : un worker qui garde sa dernière
position sans plus rien entendre reste exposé à l'aveugle.

- ``last_signal`` de chaque actif est écrit dans SQLite à chaque signal valide (un ``FLAT`` compte : c'est la preuve
  que le moteur est vivant). Il survit donc à un redémarrage du worker.
- Chaque ``dead_man_check_interval_s`` le worker compare. Passé le délai (strictement), il lit le compte : actif à
  plat -> rien à faire (et on ne le relit pas avant ``max(N, 60)`` s) ; position ouverte -> **déclenchement**.
- Déclenchement : état ``dead_man_tripped`` persistant ; la file de signaux est vidée ; TOUS les actifs suivis sont
  fermés (une fois verrouillé, plus aucun signal n'est traité : une position laissée serait aveugle).
- Fermeture : ordre ``reduce_only`` IOC agressif (``emergency_slippage_bps`` autour du mid, doublé à chaque
  tentative jusqu'à 20 %), **répété à chaque intervalle jusqu'à lecture d'une position nulle**. Un IOC qui ne se
  remplit pas ne laisse pas la position ouverte.
- ``HALTED`` : tout signal reçu ensuite est refusé sans traitement (``dead_man_halted``, journal de niveau debug).
  Sortie : ``--clear-halt`` (depuis un autre processus, le worker en marche le voit) ou redémarrage après
  ``--clear-halt``. Le réarmement remet les horloges à l'instant présent : si le moteur est toujours muet, il
  redéclenche après N secondes.
- Délai de grâce au démarrage (``dead_man_startup_grace_s``) : un worker relancé ne se déclenche pas avant que le
  moteur ait pu lui parler, même si son dernier signal persisté est ancien.
- Les horloges sont murales : un saut d'horloge vers l'avant supérieur à N déclenche (sens sûr : on réduit).
- Capacité par worker : ``supports_dead_man``. Un worker sans positions (Numerai) n'hérite pas de ce socle.
"""

from __future__ import annotations

import asyncio
import json
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
_LAST_SIGNAL_PREFIX = "last_signal:"
_DM_TRIPPED = "dead_man_tripped"
_DM_PENDING = "dead_man_pending"
_FLAT_RECHECK_MIN_S = 60.0
# Résultats d'``emergency_flatten`` qui terminent le travail sur un actif (les autres sont réessayés).
_FLATTEN_DONE = frozenset({"no_position", "dry_run"})


class PlannedOrder(Protocol):
    @property
    def is_buy(self) -> bool: ...

    @property
    def size(self) -> Decimal: ...

    @property
    def limit_px(self) -> Decimal: ...


@dataclass(frozen=True)
class EmergencyOrder:
    """Ordre de sortie d'urgence tel qu'il est journalisé (satisfait ``PlannedOrder``)."""

    is_buy: bool
    size: Decimal
    limit_px: Decimal


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
    # Homme mort : ``None`` = désactivé. Voir la docstring du module.
    dead_man_timeout_s: float | None = None
    dead_man_check_interval_s: float = 5.0
    dead_man_startup_grace_s: float = 30.0
    emergency_slippage_bps: Decimal = Decimal("300")


class ExposureWorkerBase(ABC):
    venue: ClassVar[TargetVenue]
    _tag: ClassVar[str] = "worker"  # préfixe des événements de journal
    supports_dead_man: ClassVar[bool] = False  # un worker l'active en implémentant ``emergency_flatten``

    def __init__(self, config: ExposureWorkerConfig, state: ExecutionState | None = None) -> None:
        verify_signature({}, config.secret_key)  # lève WeakSecretError dès le démarrage si la clé est faible
        self._validate_dead_man(config)
        self._cfg = config
        self._state = state if state is not None else ExecutionState(config.db_path)
        self._replay_guard = ReplayGuard()
        self._pending: dict[str, PlatformSignal] = {}
        self._order_lock = asyncio.Lock()  # signal en cours d'exécution et sortie d'urgence ne se croisent jamais
        self._started_ms = int(time.time() * 1000)
        self._flat_until_ms: dict[str, int] = {}
        self._flatten_attempts: dict[str, int] = {}
        for signal_id in self._state.recent_signal_ids(self._started_ms - _REPLAY_WINDOW_MS):
            self._replay_guard.check_and_add(uuid.UUID(signal_id))  # un redémarrage n'ouvre pas de fenêtre de rejeu
        self._state.prune_seen_before(self._started_ms - 2 * _REPLAY_WINDOW_MS)

    @classmethod
    def _validate_dead_man(cls, config: ExposureWorkerConfig) -> None:
        if config.dead_man_timeout_s is None:
            return
        if not cls.supports_dead_man:
            raise ValueError(f"{cls.__name__} ne supporte pas l'homme mort : dead_man_timeout_s doit rester None.")
        if config.dead_man_timeout_s <= 0 or config.dead_man_check_interval_s <= 0:
            raise ValueError("dead_man_timeout_s et dead_man_check_interval_s doivent être strictement positifs.")
        if config.dead_man_startup_grace_s < 0 or config.emergency_slippage_bps <= 0:
            raise ValueError("dead_man_startup_grace_s doit être >= 0 et emergency_slippage_bps > 0.")

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def halted(self) -> bool:
        """Mode réduction seule du coupe-circuit de drawdown (distinct de l'homme mort, voir ``dead_man_tripped``)."""
        return self._state.get("halted") == "1"

    @property
    def dead_man_tripped(self) -> bool:
        """Lu dans SQLite à chaque appel : un ``--clear-halt`` lancé depuis un autre processus est vu sans redémarrage."""
        return self._state.get(_DM_TRIPPED) is not None

    def clear_halt(self) -> None:
        """Lève les deux verrous (drawdown et homme mort). Action d'opérateur, jamais automatique."""
        clear_execution_halt(self._state)
        self._flat_until_ms.clear()
        self._flatten_attempts.clear()

    @abstractmethod
    def map_symbol(self, symbol: str) -> str | None:
        """Symbole interne (``CRYPTO_BTC_USD``) -> symbole de la plateforme, ou ``None`` si intraduisible."""

    @abstractmethod
    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        """Lit le compte, planifie et (hors dry-run) envoie l'ordre. Renvoie un code court, toujours journalisé."""

    async def emergency_flatten(self, asset: str, *, attempt: int, now_ms: int) -> str:
        """Lit la position de ``asset`` et envoie un ordre ``reduce_only`` IOC agressif pour la fermer.

        Renvoie ``no_position`` (rien à fermer), ``dry_run`` (simulé, jamais envoyé), ou un autre code (ordre envoyé
        ou échec : l'actif sera réessayé au prochain intervalle). À implémenter par tout worker qui définit
        ``supports_dead_man = True``.
        """
        raise NotImplementedError(f"{type(self).__name__} n'implémente pas emergency_flatten.")

    # --- réception --------------------------------------------------------------------------

    def handle_frames(self, frames: Sequence[bytes], *, now_ms: int | None = None) -> str:
        """Valide un message ``[topic, enveloppe]`` et met le signal en attente. ``queued`` ou la raison du rejet."""
        if self.dead_man_tripped:
            _log.debug("%s_signal_ignored_dead_man_halted", self._tag)
            return "dead_man_halted"  # HALTED : rien n'est vérifié ni traité, en silence
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
        if self._cfg.dead_man_timeout_s is not None:
            self._state.set(f"{_LAST_SIGNAL_PREFIX}{asset}", str(now))  # persisté : survit à un redémarrage
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

    def _record_emergency(self, asset: str, status: str, now_ms: int, order: PlannedOrder | None, detail: str = "") -> None:
        """Journalise une sortie d'urgence (pas de signal d'origine : l'identifiant est ``dead-man:<actif>``)."""
        self._state.log_execution(
            ExecutionRecord(
                ts_ms=now_ms,
                signal_id=f"dead-man:{asset}",
                asset=asset,
                status=f"dead_man_flatten:{status}",
                side="" if order is None else ("BUY" if order.is_buy else "SELL"),
                size="" if order is None else str(order.size),
                px="" if order is None else str(order.limit_px),
                detail=detail,
            )
        )

    # --- homme mort -------------------------------------------------------------------------

    def last_signal_ms(self, asset: str) -> int | None:
        raw = self._state.get(f"{_LAST_SIGNAL_PREFIX}{asset}")
        return None if raw is None else int(raw)

    def _tracked_assets(self) -> dict[str, int]:
        return {k[len(_LAST_SIGNAL_PREFIX):]: int(v) for k, v in self._state.items_with_prefix(_LAST_SIGNAL_PREFIX).items()}

    def _expired_assets(self, now_ms: int) -> list[str]:
        assert self._cfg.dead_man_timeout_s is not None
        timeout_ms = int(self._cfg.dead_man_timeout_s * 1000)
        grace_until = self._started_ms + int(self._cfg.dead_man_startup_grace_s * 1000)
        expired = []
        for asset, last in self._tracked_assets().items():
            if now_ms < grace_until or now_ms - last <= timeout_ms:
                continue  # dans le délai de grâce du démarrage, ou signal assez récent (le dépassement est strict)
            if self._flat_until_ms.get(asset, 0) > now_ms:
                continue  # déjà vu à plat récemment : pas de relecture du compte à chaque intervalle
            expired.append(asset)
        return expired

    async def check_dead_man(self, now_ms: int | None = None) -> str:
        """Un cycle de surveillance. ``disabled``, ``ok``, ``tripped``, ``halted_flattening`` ou ``halted_flat``."""
        timeout = self._cfg.dead_man_timeout_s
        if timeout is None:
            return "disabled"
        now = int(time.time() * 1000) if now_ms is None else now_ms
        async with self._order_lock:
            if self.dead_man_tripped:
                return await self._dead_man_flatten_pending(now, skip=frozenset())
            for asset in self._expired_assets(now):
                self._flatten_attempts[asset] = self._flatten_attempts.get(asset, 0) + 1
                result = await self.emergency_flatten(asset, attempt=self._flatten_attempts[asset], now_ms=now)
                if result == "no_position":
                    self._flatten_attempts.pop(asset, None)
                    self._flat_until_ms[asset] = now + int(max(timeout, _FLAT_RECHECK_MIN_S) * 1000)
                    continue
                self._trip(asset, now, silent_for_ms=now - (self.last_signal_ms(asset) or now))
                # ``asset`` vient de recevoir son ordre : on le revérifie au cycle suivant, on ferme les autres tout de suite.
                await self._dead_man_flatten_pending(now, skip=frozenset({asset}))
                return "tripped"
            return "ok"

    def _trip(self, asset: str, now_ms: int, *, silent_for_ms: int) -> None:
        """Verrouille le worker et programme la fermeture de TOUS les actifs suivis."""
        self._state.set(_DM_TRIPPED, str(now_ms))
        self._state.set(_DM_PENDING, json.dumps(sorted(self._tracked_assets())))
        self._pending.clear()  # aucun signal en attente ne sera exécuté
        _log.critical(
            "%s_dead_man_tripped asset=%s silent_for_s=%.1f timeout_s=%s : fermeture de toutes les positions puis HALTED",
            self._tag, asset, silent_for_ms / 1000, self._cfg.dead_man_timeout_s,
        )

    async def _dead_man_flatten_pending(self, now_ms: int, *, skip: frozenset[str]) -> str:
        pending = [str(a) for a in json.loads(self._state.get(_DM_PENDING) or "[]")]
        remaining: list[str] = []
        for asset in pending:
            if asset in skip:
                remaining.append(asset)
                continue
            self._flatten_attempts[asset] = self._flatten_attempts.get(asset, 0) + 1
            result = await self.emergency_flatten(asset, attempt=self._flatten_attempts[asset], now_ms=now_ms)
            if result in _FLATTEN_DONE:
                self._flatten_attempts.pop(asset, None)
            else:
                remaining.append(asset)
                _log.error("%s_dead_man_flatten_incomplete asset=%s result=%s attempt=%d : nouvel essai au prochain cycle",
                           self._tag, asset, result, self._flatten_attempts[asset])
        self._state.set(_DM_PENDING, json.dumps(remaining))
        return "halted_flattening" if remaining else "halted_flat"

    # --- boucle -----------------------------------------------------------------------------

    async def _executor(self, stop: asyncio.Event, wake: asyncio.Event) -> None:
        while not stop.is_set():
            await wake.wait()
            wake.clear()
            while self._pending and not stop.is_set() and not self.dead_man_tripped:
                asset = next(iter(self._pending))
                signal = self._pending.pop(asset)
                try:
                    async with self._order_lock:
                        if self.dead_man_tripped:  # déclenché pendant l'attente du verrou
                            break
                        await self.execute_signal(signal, asset)
                except Exception:  # noqa: BLE001 -- un signal ne doit jamais arrêter l'exécuteur
                    _log.exception("%s_execute_failed asset=%s", self._tag, asset)

    async def _dead_man_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.check_dead_man()
            except Exception:  # noqa: BLE001 -- une surveillance qui plante ne doit pas disparaître en silence
                _log.exception("%s_dead_man_check_failed", self._tag)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._cfg.dead_man_check_interval_s)
            except asyncio.TimeoutError:
                pass

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Écoute le bus sans bloquer ; l'exécution réseau et la surveillance tournent dans des tâches séparées.

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
        tasks = [asyncio.create_task(self._executor(stop, wake))]
        if self._cfg.dead_man_timeout_s is not None:
            tasks.append(asyncio.create_task(self._dead_man_loop(stop)))
            _log.info("%s_dead_man_armed timeout_s=%s grace_s=%s", self._tag, self._cfg.dead_man_timeout_s, self._cfg.dead_man_startup_grace_s)
        else:
            _log.warning("%s_dead_man_disabled : sans signal du moteur, les positions restent ouvertes à l'aveugle", self._tag)
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
            await asyncio.gather(*tasks, return_exceptions=True)
            socket.close(linger=0)
            context.term()


def clear_execution_halt(state: ExecutionState, now_ms: int | None = None) -> None:
    """Lève les verrous d'un worker (drawdown et homme mort) directement dans sa base, sans client réseau ni clé.

    Les horloges de ``last_signal`` repartent de l'instant présent : si le moteur est toujours muet, l'homme mort se
    redéclenchera après N secondes plutôt que de rester désarmé. Utilisable depuis un autre processus que le worker.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    state.delete("halted")
    state.delete(_DM_TRIPPED)
    state.delete(_DM_PENDING)
    for key in state.items_with_prefix(_LAST_SIGNAL_PREFIX):
        state.set(key, str(now))

"""Battement de cœur du moteur, sur un thread dédié, avec l'état de démarrage (``STARTING`` puis ``ACTIVE``).

**Pourquoi un thread et pas une tâche ``asyncio``.** Tous les agents du moteur partagent une seule boucle d'événements
(``scripts/run_backend.py``). Une tâche ``asyncio`` qui écrit le heartbeat n'est « découplée » que sur le papier : tant qu'un calcul
bloque la boucle (inférence XGBoost ou PPO, estimation de Lyapunov, dimensionnement), AUCUNE tâche ne tourne, heartbeat compris. Avec un
TTL de quelques secondes, quelques secondes de calcul suffisent à faire expirer la clé, et le pont conclut « moteur muet ». Le thread, lui,
continue pendant les calculs qui relâchent le GIL (numpy, XGBoost, PyTorch) et reprend la main toutes les quelques millisecondes pendant
ceux qui le gardent.

**Ce que ce module garantit.**

- Le thread a son PROPRE client Redis synchrone, avec des délais de connexion et de lecture bornés : un Redis lent ne le fige pas pour
  toujours, et il ne partage ni connexion ni état avec le reste du moteur.
- Un échec d'écriture est compté et journalisé (au plus une fois par ``log_every_s``), le client est jeté et recréé au battement suivant.
  Il ne lève jamais et ne touche à aucun autre composant. (Le heartbeat de ``BaseAgent`` met ``self._redis_client = None`` au premier
  incident Redis : un seul accroc coupait le heartbeat pour toujours, et le reste de l'agent avec lui.)
- Le TTL vaut au moins ``min_ratio`` (3) fois l'intervalle, vérifié à la construction : un battement perdu ou en retard ne fait jamais
  expirer la clé.

**Séquence d'armement.** ``start()`` écrit tout de suite, avant de rendre la main : le processus se déclare vivant à l'état ``STARTING``
AVANT le chargement lourd des modèles. L'état passe à ``ACTIVE`` (``set_state``) quand le moteur est réellement prêt à décider. On n'écrit
pas ``ACTIVE`` d'emblée : un moteur qui charge encore ses modèles ne peut pas décider, et le pont qui lit cet état ferait confiance à des
positions que personne ne tient encore.

Clés écrites à chaque battement, avec le même TTL : ``heartbeat:<agent_id>`` (l'horodatage, lu par le pont et le dashboard) et
``engine:state`` (JSON ``{"state", "since", "pid"}``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from core.config import RedisSettings

_log = logging.getLogger(__name__)

ENGINE_STATE_KEY: Final[str] = "engine:state"
STATE_STARTING: Final[str] = "STARTING"
STATE_RECONCILING: Final[str] = "RECONCILING"  # modèles chargés, positions réelles de l'exchange en cours d'adoption : le pont ne publie encore rien
STATE_ACTIVE: Final[str] = "ACTIVE"
DEFAULT_INTERVAL_S: Final[float] = 3.0
DEFAULT_TTL_S: Final[float] = 10.0
MIN_TTL_RATIO: Final[float] = 3.0
_CLIENT_TIMEOUT_S: Final[float] = 2.0


class HeartbeatClient(Protocol):
    """Ce que le thread demande à Redis : écrire une clé avec expiration, et se fermer."""

    def set(self, name: str, value: str, *, px: int | None = None) -> object: ...

    def close(self) -> None: ...


ClientFactory = Callable[[], HeartbeatClient]


@dataclass(frozen=True)
class HeartbeatConfig:
    key: str
    interval_s: float = DEFAULT_INTERVAL_S
    ttl_s: float = DEFAULT_TTL_S
    state_key: str | None = ENGINE_STATE_KEY
    min_ratio: float = MIN_TTL_RATIO
    log_every_s: float = 30.0

    def __post_init__(self) -> None:
        if self.interval_s <= 0 or self.ttl_s <= 0:
            raise ValueError("interval_s et ttl_s doivent être strictement positifs.")
        if self.ttl_s < self.min_ratio * self.interval_s:
            raise ValueError(
                f"TTL {self.ttl_s}s < {self.min_ratio} x intervalle {self.interval_s}s : un seul battement perdu ou retardé ferait expirer la clé."
            )


def redis_client_factory(settings: RedisSettings) -> ClientFactory:
    """Client Redis SYNCHRONE dédié (jamais celui de la boucle asyncio), délais bornés à 2 s."""

    def build() -> HeartbeatClient:
        import redis

        client: HeartbeatClient = redis.Redis(
            host=settings.host, port=settings.port, db=settings.db,
            password=None if settings.password is None else settings.password.get_secret_value(),
            ssl=settings.use_tls, socket_timeout=_CLIENT_TIMEOUT_S, socket_connect_timeout=_CLIENT_TIMEOUT_S, decode_responses=True,
        )
        return client

    return build


class ThreadedHeartbeat:
    def __init__(
        self,
        config: HeartbeatConfig,
        client_factory: ClientFactory,
        *,
        state: str = STATE_STARTING,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._factory = client_factory
        self._clock = wall_clock
        self._state = state
        self._since = wall_clock()
        self._state_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: HeartbeatClient | None = None
        self._last_logged = float("-inf")
        self.beats = 0
        self.failures = 0
        self.last_success_monotonic: float | None = None

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Premier battement SYNCHRONE (l'état de démarrage est écrit avant le retour), puis le thread. Idempotent."""
        if self._thread is not None:
            return
        self._beat()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat:{self._config.key}", daemon=True)
        self._thread.start()

    def set_state(self, state: str) -> None:
        """Change l'état publié ; un battement part immédiatement (le thread est réveillé), sans attendre l'intervalle."""
        with self._state_lock:
            if state == self._state:
                return
            self._state, self._since = state, self._clock()
        self._wake.set()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout_s)
        self._drop_client()

    # --- thread ----------------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._config.interval_s)  # sort à l'intervalle, ou plus tôt si l'état change ou si on arrête
            self._wake.clear()
            if self._stop.is_set():
                return
            self._beat()

    def _beat(self) -> None:
        try:
            client = self._client if self._client is not None else self._factory()
            self._client = client
            ttl_ms = max(1, int(self._config.ttl_s * 1000))
            client.set(self._config.key, repr(self._clock()), px=ttl_ms)
            if self._config.state_key is not None:
                with self._state_lock:
                    payload = json.dumps({"state": self._state, "since": self._since, "pid": os.getpid()})
                client.set(self._config.state_key, payload, px=ttl_ms)
        except Exception as exc:  # noqa: BLE001 -- un battement raté ne doit jamais sortir de ce thread ni toucher le moteur
            self.failures += 1
            self._drop_client()
            now = time.monotonic()
            if now - self._last_logged >= self._config.log_every_s:
                self._last_logged = now
                _log.warning("engine_heartbeat_write_failed key=%s error=%s failures=%d", self._config.key, type(exc).__name__, self.failures)
            return
        self.beats += 1
        self.last_success_monotonic = time.monotonic()

    def _drop_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 -- nettoyage
                _log.debug("engine_heartbeat_close_failed", exc_info=True)

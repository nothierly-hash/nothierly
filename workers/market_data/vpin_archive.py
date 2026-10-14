"""Archive SQLite continue des seaux VPIN : l'historique de microstructure pour la recherche et l'entraînement.

**Pourquoi.** Le VPIN écrit dans Redis (``market:vpin:<SYMBOLE>``) est la DERNIÈRE valeur, avec une durée de vie : aucun historique.
Il n'arrivait dans l'archive du moteur que recopié dans ``META_LABEL_DECISION.features`` quand le moteur prenait une décision de
consensus, donc jamais tant qu'aucun modèle n'est entraîné. Ce module écrit chaque seau fermé, 24 h/24, dans
``<NOTHIERLY_DATA_DIR>/vpin_history.sqlite3`` (mode WAL), indépendamment de Redis : un Redis en panne ne perd rien, et une archive en
panne ne coupe pas Redis.

**Table ``vpin_buckets``** (une ligne par seau fermé) :

- ``timestamp`` (UTC, ISO 8601) et ``timestamp_ms`` : heure du trade qui a fermé le seau (heure de l'exchange, pas l'heure locale) ;
- ``symbol`` ; ``vpin_value`` : la valeur à la fermeture, ``NULL`` tant que la fenêtre n'est pas pleine (les ``window_size - 1``
  premiers seaux après un démarrage ou une remise à zéro) ;
- ``bucket_buy_vol`` / ``bucket_sell_vol`` : notionnel acheteur / vendeur du seau (leur somme vaut exactement ``bucket_size``) ;
- ``bucket_size`` / ``window_size`` : les paramètres du calcul, pour ne jamais mélanger deux configurations dans une série ;
- ``recorded_ms`` : heure d'écriture (l'écart avec ``timestamp_ms`` mesure le retard de l'archive).

Les seaux de chauffe sont archivés eux aussi (``vpin_value`` vide) : avec ``bucket_buy_vol`` et ``bucket_sell_vol`` on peut recalculer
le VPIN hors ligne avec une autre fenêtre. Une coupure du flux de plus de ``--max-gap-s`` remet le calcul à zéro : on le voit dans la
série comme un trou de ``timestamp`` suivi de valeurs vides.

**Écriture sans bloquer la boucle WebSocket.** ``submit`` ne fait que mettre la ligne dans une file bornée (``put_nowait``) : aucun
accès disque, aucune attente, jamais. Un thread dédié, seul propriétaire de la connexion SQLite, vide la file par lots (une transaction
par lot, ``--flush-every-s`` au plus d'attente). File pleine (disque bloqué trop longtemps) : la ligne est abandonnée et COMPTÉE
(``dropped``) ; elle n'est jamais attendue.

**Échecs.** Une écriture qui échoue (disque plein, verrou) garde le lot en mémoire (borné) et le retente ; ``errors`` compte les
échecs et ``last_error`` garde le dernier : le démon les surveille et alerte, car une archive muette pendant un mois est exactement la
panne qu'on ne veut pas découvrir à l'entraînement. Le dossier de la base est créé et la base ouverte à la construction : un chemin
inutilisable empêche le démarrage (échec rapide), il ne dégrade pas en silence.

Stdlib pure (``sqlite3``, ``threading``, ``queue``) : les workers n'importent jamais ``core``.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final

from workers.common.paths import default_db_path, ensure_parent

_log = logging.getLogger(__name__)

DEFAULT_FILENAME: Final[str] = "vpin_history.sqlite3"
TABLE: Final[str] = "vpin_buckets"

SCHEMA: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    vpin_value REAL,
    bucket_buy_vol REAL NOT NULL,
    bucket_sell_vol REAL NOT NULL,
    bucket_size REAL NOT NULL,
    window_size INTEGER NOT NULL,
    recorded_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{TABLE}_symbol_time ON {TABLE}(symbol, timestamp_ms);
"""

_INSERT: Final[str] = (
    f"INSERT INTO {TABLE} (timestamp, timestamp_ms, symbol, vpin_value, bucket_buy_vol, bucket_sell_vol, bucket_size, window_size, recorded_ms)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_Row = tuple[str, int, str, float | None, float, float, float, int, int]


def default_archive_path() -> Path:
    return default_db_path(DEFAULT_FILENAME)


def iso_utc(time_ms: int) -> str:
    return datetime.fromtimestamp(time_ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class BucketRecord:
    symbol: str
    closed_at_ms: int
    vpin: Decimal | None
    buy_volume: Decimal
    sell_volume: Decimal
    bucket_size: Decimal
    window_size: int


class VpinArchive:
    def __init__(
        self,
        path: str | Path,
        *,
        queue_size: int = 50_000,
        batch_max: int = 500,
        flush_every_s: float = 1.0,
        busy_timeout_s: float = 2.0,
        retry_every_s: float = 1.0,
    ) -> None:
        if queue_size < 1 or batch_max < 1 or flush_every_s <= 0:
            raise ValueError("queue_size, batch_max et flush_every_s doivent être positifs.")
        self.path = ensure_parent(path)
        self._capacity = queue_size
        self._batch_max = batch_max
        self._flush_every_s = flush_every_s
        self._retry_every_s = retry_every_s
        self._queue: queue.Queue[_Row | None] = queue.Queue(maxsize=queue_size)  # None = ordre d'arrêt
        self._pending: list[_Row] = []  # lots dont l'écriture a échoué, retentés avant tout le reste
        self._thread: threading.Thread | None = None
        # Ouverte ICI (échec rapide si le chemin est inutilisable), puis utilisée par le seul thread d'écriture.
        self._db = sqlite3.connect(str(self.path), timeout=busy_timeout_s, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self.last_error: str | None = None

    # --- côté boucle asyncio : ne bloque jamais -----------------------------------------------

    def submit(self, record: BucketRecord) -> bool:
        """Met un seau en file. ``False`` (et ``dropped`` + 1) si la file est pleine ; ne touche jamais au disque."""
        row: _Row = (
            iso_utc(record.closed_at_ms), record.closed_at_ms, record.symbol,
            None if record.vpin is None else float(record.vpin),
            float(record.buy_volume), float(record.sell_volume), float(record.bucket_size), record.window_size, int(time.time() * 1000),
        )
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1
            return False
        return True

    @property
    def backlog(self) -> int:
        return self._queue.qsize() + len(self._pending)

    # --- cycle de vie ------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="vpin-archive", daemon=True)
            self._thread.start()

    def close(self, timeout_s: float = 10.0) -> int:
        """Vide la file, arrête le thread, ferme la base. Renvoie le nombre de lignes NON écrites (0 = tout est sur disque)."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self._queue.put(None, timeout=timeout_s)
            except queue.Full:  # file saturée et thread bloqué : on ne peut pas faire mieux que le dire
                pass
            thread.join(timeout_s)
        left = self.backlog
        if thread is None or not thread.is_alive():
            if thread is None and left:  # jamais démarré (tests, dry-run) : on vide dans le thread courant
                self._drain_inline()
                left = self.backlog
            self._db.close()
        if left:
            _log.error("vpin_archive_unflushed rows=%d", left)
        return left

    # --- thread d'écriture -------------------------------------------------------------------

    def _run(self) -> None:
        stopping = False
        while not stopping:
            batch = self._take_pending()
            try:
                stopping = self._collect(self._queue.get(timeout=self._flush_every_s), batch)
            except queue.Empty:
                pass
            while not stopping and len(batch) < self._batch_max:
                try:
                    stopping = self._collect(self._queue.get_nowait(), batch)
                except queue.Empty:
                    break
            if batch and not self._write(batch):
                self._retain(batch)
                if not stopping:
                    time.sleep(self._retry_every_s)
        self._drain_inline()  # arrêt : dernier essai sur ce qui reste

    def _take_pending(self) -> list[_Row]:
        batch, self._pending = self._pending, []
        return batch

    @staticmethod
    def _collect(item: _Row | None, batch: list[_Row]) -> bool:
        """Ajoute ``item`` au lot ; ``True`` si c'était l'ordre d'arrêt."""
        if item is None:
            return True
        batch.append(item)
        return False

    def _retain(self, batch: list[_Row]) -> None:
        keep = batch[-self._capacity:]
        self.dropped += len(batch) - len(keep)
        self._pending = keep

    def _write(self, batch: list[_Row]) -> bool:
        try:
            with self._db:  # une transaction par lot : tout ou rien
                self._db.executemany(_INSERT, batch)
        except sqlite3.Error as exc:
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            _log.warning("vpin_archive_write_failed rows=%d error=%s", len(batch), self.last_error)
            return False
        self.written += len(batch)
        return True

    def _drain_inline(self) -> None:
        rows = self._take_pending()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._collect(item, rows)
        if rows and not self._write(rows):
            self._retain(rows)

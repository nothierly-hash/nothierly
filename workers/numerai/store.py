"""Stockage local durable du worker Numerai (SQLite, stdlib).

SQLite plutôt que Redis : le worker est isolé du moteur par conception et n'a pas accès à son Redis ;
un fichier local sur son propre volume survit aux redémarrages sans infrastructure en plus.

Trois tables :

- ``scores`` : dernier score par ticker, en attente de soumission. Un score plus récent écrase le
  précédent ; un score plus ancien ou de même horodatage est ignoré, ce qui neutralise le rejeu d'un
  vieux message authentique (qui écraserait sinon le score du jour).
- ``seen_signals`` : identifiants déjà acceptés, rechargés au démarrage dans le ``ReplayGuard``
  pour qu'un redémarrage n'ouvre pas de fenêtre de rejeu.
- ``submissions`` : une ligne par jour UTC (statut, tentatives), pour ne soumettre qu'une fois par
  jour même après un redémarrage, et limiter les nouvelles tentatives.

Les accès sont sérialisés par un verrou : le worker appelle ``submit_if_due`` dans un thread
(appel réseau bloquant) pendant que la boucle asynchrone écrit les scores.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    numerai_ticker TEXT PRIMARY KEY,
    signal         REAL NOT NULL,
    signal_id      TEXT NOT NULL,
    timestamp_ms   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS seen_signals (
    signal_id TEXT PRIMARY KEY,
    seen_ms   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    day             TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL,
    last_attempt_ms INTEGER NOT NULL,
    n_rows          INTEGER NOT NULL,
    detail          TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ScoreRow:
    numerai_ticker: str
    signal: float
    signal_id: str
    timestamp_ms: int


@dataclass(frozen=True)
class SubmissionRecord:
    day: str
    status: str
    attempts: int
    last_attempt_ms: int
    n_rows: int
    detail: str


class SignalStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- scores -----------------------------------------------------------------------------

    def upsert_score(self, numerai_ticker: str, signal: float, signal_id: str, timestamp_ms: int) -> bool:
        """Mémorise le score s'il est strictement plus récent que celui en place. ``False`` sinon."""
        with self._lock:
            cur = self._db.execute(
                """INSERT INTO scores(numerai_ticker, signal, signal_id, timestamp_ms) VALUES (?, ?, ?, ?)
                   ON CONFLICT(numerai_ticker) DO UPDATE SET
                       signal=excluded.signal, signal_id=excluded.signal_id, timestamp_ms=excluded.timestamp_ms
                   WHERE excluded.timestamp_ms > scores.timestamp_ms""",
                (numerai_ticker, signal, signal_id, timestamp_ms),
            )
            return cur.rowcount == 1

    def scores_since(self, min_timestamp_ms: int) -> list[ScoreRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT numerai_ticker, signal, signal_id, timestamp_ms FROM scores "
                "WHERE timestamp_ms >= ? ORDER BY numerai_ticker",
                (min_timestamp_ms,),
            ).fetchall()
        return [ScoreRow(*row) for row in rows]

    def count_scores(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM scores").fetchone()[0])

    def delete_scores_up_to(self, timestamp_ms: int) -> int:
        """Supprime les scores d'horodatage <= ``timestamp_ms`` (ceux qui viennent d'être soumis)."""
        with self._lock:
            return self._db.execute("DELETE FROM scores WHERE timestamp_ms <= ?", (timestamp_ms,)).rowcount

    # --- anti-rejeu -------------------------------------------------------------------------

    def remember_signal(self, signal_id: str, seen_ms: int) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO seen_signals(signal_id, seen_ms) VALUES (?, ?)", (signal_id, seen_ms))

    def recent_signal_ids(self, since_ms: int) -> list[str]:
        """Identifiants vus depuis ``since_ms``, du plus ancien au plus récent."""
        with self._lock:
            rows = self._db.execute(
                "SELECT signal_id FROM seen_signals WHERE seen_ms >= ? ORDER BY seen_ms", (since_ms,)
            ).fetchall()
        return [row[0] for row in rows]

    def prune_seen_before(self, before_ms: int) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM seen_signals WHERE seen_ms < ?", (before_ms,)).rowcount

    # --- soumissions ------------------------------------------------------------------------

    def submission(self, day: str) -> SubmissionRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT day, status, attempts, last_attempt_ms, n_rows, detail FROM submissions WHERE day = ?", (day,)
            ).fetchone()
        return None if row is None else SubmissionRecord(*row)

    def record_submission(self, day: str, status: str, attempts: int, now_ms: int, n_rows: int, detail: str) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO submissions(day, status, attempts, last_attempt_ms, n_rows, detail)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(day) DO UPDATE SET status=excluded.status, attempts=excluded.attempts,
                       last_attempt_ms=excluded.last_attempt_ms, n_rows=excluded.n_rows, detail=excluded.detail""",
                (day, status, attempts, now_ms, n_rows, detail),
            )

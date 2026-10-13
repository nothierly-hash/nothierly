"""État local durable d'un worker d'exécution (SQLite, stdlib).

- ``seen_signals`` : identifiants acceptés, rechargés au démarrage dans le ``ReplayGuard``.
- ``kv`` : drapeau ``halted`` (mode réduction seule) et base de drawdown du jour. Persistés : un
  redémarrage ne doit ni lever un coupe-circuit ni remettre la base de drawdown à zéro.
- ``executions`` : journal d'audit de chaque décision (ordre envoyé, ignoré, rejeté, simulé).

Accès sérialisés par un verrou (l'exécution réseau tourne dans des threads).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_signals (signal_id TEXT PRIMARY KEY, seen_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    signal_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    status TEXT NOT NULL,
    side TEXT NOT NULL,
    size TEXT NOT NULL,
    px TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class ExecutionRecord:
    ts_ms: int
    signal_id: str
    asset: str
    status: str
    side: str
    size: str
    px: str
    detail: str


class ExecutionState:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def remember_signal(self, signal_id: str, seen_ms: int) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO seen_signals(signal_id, seen_ms) VALUES (?, ?)", (signal_id, seen_ms))

    def recent_signal_ids(self, since_ms: int) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT signal_id FROM seen_signals WHERE seen_ms >= ? ORDER BY seen_ms", (since_ms,)
            ).fetchall()
        return [row[0] for row in rows]

    def prune_seen_before(self, before_ms: int) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM seen_signals WHERE seen_ms < ?", (before_ms,)).rowcount

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
            )

    def delete(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM kv WHERE key = ?", (key,))

    def items_with_prefix(self, prefix: str) -> dict[str, str]:
        """Toutes les paires ``kv`` dont la clé commence par ``prefix`` (comparaison exacte, sans joker SQL)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM kv WHERE substr(key, 1, ?) = ?", (len(prefix), prefix)
            ).fetchall()
        return {str(k): str(v) for k, v in rows}

    def log_execution(self, record: ExecutionRecord) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO executions(ts_ms, signal_id, asset, status, side, size, px, detail) VALUES (?,?,?,?,?,?,?,?)",
                (record.ts_ms, record.signal_id, record.asset, record.status, record.side, record.size, record.px, record.detail),
            )

    def executions(self) -> list[ExecutionRecord]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts_ms, signal_id, asset, status, side, size, px, detail FROM executions ORDER BY id"
            ).fetchall()
        return [ExecutionRecord(*row) for row in rows]

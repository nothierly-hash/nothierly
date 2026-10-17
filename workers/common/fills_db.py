"""Base SQLite des exécutions réelles et du slippage : ce que le worker a voulu, ce que la plateforme a donné.

**Où est ce module, et pourquoi pas ``core/analytics/db.py``.** Les workers n'importent jamais ``core`` ni ``src`` (frontière
d'IP, vérifiée par un test AST) et ce module est utilisé par eux : il vit donc ici, en stdlib pure. Le moteur qui voudrait
lire les résultats peut ouvrir le fichier SQLite en lecture, ou importer ce module comme n'importe quel lecteur externe.

**Deux tables.**

- ``order_intents`` : l'**intention** d'un ordre, écrite AVANT son envoi (clé : identifiant client de l'ordre, ``cloid`` /
  ``orderLinkId``), puis complétée par l'identifiant d'ordre de la plateforme dès que la réponse arrive. Le prix voulu
  (``intended_price``) est le prix de référence du worker au moment où l'ordre part (le ``mid`` qui a servi à le dimensionner).
  Persistée : un fill qui arrive après un redémarrage retrouve son intention.
- ``executions`` : un **fill** par ligne (un ordre exécuté en trois fois donne trois lignes), idempotent sur
  ``(venue, fill_id)`` : un fill rejoué par une reconnexion ou un instantané n'est jamais compté deux fois.

**Convention du slippage (``slippage_bps``).** Positif = **coût** (exécuté moins bien que voulu), négatif = amélioration :

- achat : ``(fill - voulu) / voulu x 10 000`` ; vente : ``(voulu - fill) / voulu x 10 000``.

**Ce que ce slippage mesure, et ce qu'il ne mesure pas.** Écart entre le prix de référence du worker à l'envoi et le prix
d'exécution : le coût de l'exécution (traversée du carnet, mouvement pendant l'aller-retour réseau). Il ne contient PAS la
dérive entre la décision du moteur et l'arrivée du signal au worker (le contrat ne porte pas de prix de décision) : ce
délai-là reste à mesurer séparément. Un fill sans intention connue (ordre passé à la main, intention perdue) est enregistré
avec ``intended_price`` et ``slippage_bps`` vides et ``kind='unmatched'`` : jamais un faux zéro.

Prix et quantités sont stockés en ``REAL`` pour pouvoir faire des agrégats en SQL ; le calcul du slippage se fait en
``Decimal`` avant la conversion.
"""

from __future__ import annotations

import sqlite3
import statistics
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS order_intents (
    venue TEXT NOT NULL,
    client_id TEXT NOT NULL,
    order_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    intended_price REAL NOT NULL,
    intended_qty REAL NOT NULL,
    kind TEXT NOT NULL,
    sent_at_ms INTEGER NOT NULL,
    PRIMARY KEY (venue, client_id)
);
CREATE INDEX IF NOT EXISTS idx_intents_order ON order_intents(venue, order_id);
CREATE INDEX IF NOT EXISTS idx_intents_sent ON order_intents(sent_at_ms);
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    venue TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    intended_price REAL,
    fill_price REAL NOT NULL,
    fill_qty REAL NOT NULL,
    slippage_bps REAL,
    order_id TEXT,
    fill_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    fee REAL,
    fee_currency TEXT,
    recorded_ms INTEGER NOT NULL,
    UNIQUE (venue, fill_id)
);
CREATE INDEX IF NOT EXISTS idx_exec_time ON executions(timestamp);
CREATE INDEX IF NOT EXISTS idx_exec_order ON executions(venue, order_id);
"""

BUY: Final[str] = "BUY"
SELL: Final[str] = "SELL"


def slippage_bps(is_buy: bool, intended: Decimal, fill: Decimal) -> Decimal | None:
    """Slippage en points de base, positif = coût. ``None`` si le prix voulu n'est pas strictement positif."""
    if intended <= 0:
        return None
    ratio = (fill - intended) / intended if is_buy else (intended - fill) / intended
    return ratio * Decimal(10_000)


@dataclass(frozen=True)
class OrderIntent:
    venue: str
    client_id: str
    order_id: str | None
    symbol: str
    is_buy: bool
    intended_price: Decimal
    intended_qty: Decimal
    kind: str  # ``signal`` ou ``emergency``
    sent_at_ms: int


@dataclass(frozen=True)
class Execution:
    timestamp: int
    venue: str
    symbol: str
    direction: str
    intended_price: float | None
    fill_price: float
    fill_qty: float
    slippage_bps: float | None
    order_id: str | None
    fill_id: str
    kind: str
    fee: float | None
    fee_currency: str | None


@dataclass(frozen=True)
class SlippageSummary:
    """Agrégat des fills APPARIÉS (intention connue) ; les autres sont comptés à part, pas moyennés."""

    fills: int
    unmatched: int
    notional: float
    mean_bps: float | None
    qty_weighted_mean_bps: float | None
    median_bps: float | None
    p95_bps: float | None
    worst_bps: float | None


class FillsDatabase:
    """Accès SQLite sûr entre threads (un verrou, ``check_same_thread=False``) : le tracker écrit depuis un thread de travail,
    le worker enregistre ses intentions depuis la boucle principale, sans jamais attendre l'un l'autre plus que quelques ms."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- intentions -------------------------------------------------------------------------

    def register_intent(self, intent: OrderIntent) -> None:
        """À appeler AVANT l'envoi : un fill peut arriver sur le flux avant la réponse de l'API."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO order_intents(venue, client_id, order_id, symbol, direction, intended_price, intended_qty, kind, sent_at_ms)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (intent.venue, intent.client_id, intent.order_id, intent.symbol, BUY if intent.is_buy else SELL,
                 float(intent.intended_price), float(intent.intended_qty), intent.kind, intent.sent_at_ms),
            )

    def attach_order_id(self, venue: str, client_id: str, order_id: str) -> bool:
        with self._lock:
            return self._db.execute(
                "UPDATE order_intents SET order_id = ? WHERE venue = ? AND client_id = ?", (str(order_id), venue, client_id)
            ).rowcount == 1

    def find_intent(self, venue: str, *, client_id: str | None = None, order_id: str | None = None) -> OrderIntent | None:
        """Cherche par identifiant client (prioritaire), sinon par identifiant d'ordre de la plateforme."""
        with self._lock:
            row = None
            if client_id:
                row = self._db.execute(
                    "SELECT venue, client_id, order_id, symbol, direction, intended_price, intended_qty, kind, sent_at_ms"
                    " FROM order_intents WHERE venue = ? AND client_id = ?", (venue, client_id)).fetchone()
            if row is None and order_id:
                row = self._db.execute(
                    "SELECT venue, client_id, order_id, symbol, direction, intended_price, intended_qty, kind, sent_at_ms"
                    " FROM order_intents WHERE venue = ? AND order_id = ? ORDER BY sent_at_ms DESC LIMIT 1", (venue, str(order_id))).fetchone()
        if row is None:
            return None
        return OrderIntent(row[0], row[1], row[2], row[3], row[4] == BUY, Decimal(repr(row[5])), Decimal(repr(row[6])), row[7], int(row[8]))

    def prune_intents_before(self, before_ms: int) -> int:
        with self._lock:
            return self._db.execute("DELETE FROM order_intents WHERE sent_at_ms < ?", (before_ms,)).rowcount

    # --- exécutions -------------------------------------------------------------------------

    def insert_execution(self, execution: Execution, *, now_ms: int | None = None) -> bool:
        """``True`` si la ligne est nouvelle, ``False`` si ce fill était déjà enregistré (idempotent)."""
        recorded = int(time.time() * 1000) if now_ms is None else now_ms
        with self._lock:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO executions(timestamp, venue, symbol, direction, intended_price, fill_price, fill_qty, slippage_bps,"
                " order_id, fill_id, kind, fee, fee_currency, recorded_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (execution.timestamp, execution.venue, execution.symbol, execution.direction, execution.intended_price,
                 execution.fill_price, execution.fill_qty, execution.slippage_bps, execution.order_id, execution.fill_id,
                 execution.kind, execution.fee, execution.fee_currency, recorded),
            )
            return cursor.rowcount == 1

    def has_fill(self, venue: str, fill_id: str) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1 FROM executions WHERE venue = ? AND fill_id = ?", (venue, fill_id)).fetchone() is not None

    def executions(self, *, venue: str | None = None, symbol: str | None = None, since_ms: int = 0, limit: int = 1000) -> list[Execution]:
        clauses = ["timestamp >= ?"]
        params: list[int | str] = [since_ms]
        if venue:
            clauses.append("venue = ?")
            params.append(venue)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        with self._lock:
            rows = self._db.execute(
                "SELECT timestamp, venue, symbol, direction, intended_price, fill_price, fill_qty, slippage_bps, order_id, fill_id, kind, fee, fee_currency"
                f" FROM executions WHERE {' AND '.join(clauses)} ORDER BY timestamp, id LIMIT ?", (*params, limit)).fetchall()
        return [Execution(*row) for row in rows]

    def summary(self, *, venue: str | None = None, symbol: str | None = None, since_ms: int = 0) -> SlippageSummary:
        rows = self.executions(venue=venue, symbol=symbol, since_ms=since_ms, limit=1_000_000)
        matched = [r for r in rows if r.slippage_bps is not None]
        return summarize(matched, unmatched=len(rows) - len(matched))


def summarize(matched: Sequence[Execution], *, unmatched: int = 0) -> SlippageSummary:
    values = sorted(r.slippage_bps for r in matched if r.slippage_bps is not None)
    if not values:
        return SlippageSummary(0, unmatched, 0.0, None, None, None, None, None)
    notional = sum(r.fill_price * r.fill_qty for r in matched)
    weighted = sum(r.slippage_bps * r.fill_price * r.fill_qty for r in matched if r.slippage_bps is not None)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]  # rang supérieur : prudent sur de petits échantillons
    return SlippageSummary(
        fills=len(values), unmatched=unmatched, notional=notional, mean_bps=statistics.fmean(values),
        qty_weighted_mean_bps=weighted / notional if notional else None, median_bps=statistics.median(values),
        p95_bps=p95, worst_bps=values[-1],
    )

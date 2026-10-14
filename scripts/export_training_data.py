"""Exporte l'historique du Shadow Run en CSV propres, pour ``train_model.py``, ``train_ppo.py`` et ``train_metalabel.py``.

    python scripts/export_training_data.py --symbol BTCUSDT
    python scripts/export_training_data.py --symbol BTCUSDT --since-days 14 --source fills
    python scripts/export_training_data.py --symbol ETHUSDT --fills-db /data/bybit_fills.sqlite3 --out-dir /tmp/exports

**Où sont réellement les données (et ce que ce script lit).** Il n'existe pas de ``market_data.db`` ni de ``executions.db`` :

- *Fills et slippage* : un fichier SQLite par plateforme, ``<venue>_fills.sqlite3`` (tables ``executions`` et ``order_intents``,
  voir ``workers/common/fills_db.py``), dans ``$NOTHIERLY_DATA_DIR`` (défaut ``data/``). Découverts automatiquement, ou donnés par
  ``--fills-db``. Ouverts en LECTURE SEULE : le script ne modifie jamais une base de production.
- *Carnet, décisions du méta-label (dont le VPIN et les autres features), exécutions du moteur* : l'archive TimescaleDB
  (table ``bus_messages``, écrite par l'ArchivistAgent) : c'est ce que lit ``train_model.py``. Lue avec les réglages ``TIMESCALE_*``
  de l'environnement (``--source bus`` ou ``all``).
- *VPIN* : la série continue, un seau par ligne, écrite 24 h/24 par ``workers/market_data/vpin_tracker.py`` dans
  ``<NOTHIERLY_DATA_DIR>/vpin_history.sqlite3`` (table ``vpin_buckets``, voir ``vpin_archive.py``) ; ``--vpin-db`` pour un autre chemin.
  La clé Redis ``market:vpin:<SYMBOLE>`` n'est que la dernière valeur, avec expiration : ce n'est PAS une source d'historique.
  ``vpin_value`` est vide pour les seaux de chauffe (fenêtre pas encore pleine) ; ``bucket_buy_vol`` et ``bucket_sell_vol`` permettent de
  recalculer le VPIN avec une autre fenêtre.

Sortie : ``research/exports/<table>_<SYMBOLE>_<horodatage UTC>.csv`` (``--out-dir`` pour changer). Jamais d'écrasement : l'horodatage
fait partie du nom. Un export vide n'est pas écrit, et le script sort en erreur : mieux vaut le savoir que lancer un entraînement
sur un fichier vide.

Symboles : comparés sans casse ni séparateurs (``BTC/USDT`` = ``BTCUSDT``), et un symbole nu d'une plateforme (``BTC`` sur
Hyperliquid) correspond à ``BTCUSDT`` / ``BTCUSD`` / ``BTCUSDC``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

# Même contrat que ``workers/common/paths.py`` (un test vérifie l'accord). Recopié ici et non importé : l'image du moteur, où cet outil
# tourne à côté de l'archive, ne contient pas le paquet ``workers``.
ENV_DATA_DIR: Final[str] = "NOTHIERLY_DATA_DIR"
DEFAULT_DATA_DIR: Final[str] = "data/"
DEFAULT_OUT_DIR: Final[str] = "research/exports"
VPIN_FILENAME: Final[str] = "vpin_history.sqlite3"  # même nom que workers/market_data/vpin_archive.DEFAULT_FILENAME (un test vérifie l'accord)
FILLS_GLOB: Final[str] = "*_fills.sqlite3"
_QUOTES: Final[tuple[str, ...]] = ("USDT", "USDC", "USD", "PERP")
BUS_TOPICS: Final[tuple[str, ...]] = ("MARKET_DATA_L2", "META_LABEL_DECISION", "TRADE_EXECUTED")
_BUS_SYMBOL_KEYS: Final[tuple[str, ...]] = ("symbol", "instrument")
_TABLES: Final[dict[str, str]] = {"executions": "timestamp", "order_intents": "sent_at_ms"}  # table -> colonne d'horodatage (ms)


def data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA_DIR) or DEFAULT_DATA_DIR)


def normalize_symbol(symbol: str) -> str:
    """Majuscules, sans séparateur (même règle que ``scripts/train_model.py``)."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def symbol_matches(row_symbol: str, wanted: str) -> bool:
    """``True`` si ``row_symbol`` désigne le même instrument que ``wanted`` (voir l'en-tête du module)."""
    row, want = normalize_symbol(row_symbol), normalize_symbol(wanted)
    if not row or not want:
        return False
    if row == want:
        return True
    return any(want == row + quote or row == want + quote for quote in _QUOTES)


# --- Fills (SQLite) --------------------------------------------------------------------------


def find_fills_databases(directory: Path) -> list[Path]:
    return sorted(directory.glob(FILLS_GLOB))


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"base introuvable : {path}")
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            f"{path} : ouverture en lecture seule impossible ({exc}). Une base WAL sur un volume monté en lecture seule ne s'ouvre "
            "pas : monter le volume en lecture-écriture pour l'export (le script n'écrit rien)."
        ) from exc
    return conn


def read_table(path: Path, table: str, *, since_ms: int, symbol: str) -> list[dict[str, Any]]:
    """Lignes de ``table`` plus récentes que ``since_ms`` pour ``symbol``, par ordre chronologique."""
    time_column = _TABLES.get(table)
    if time_column is None:
        raise ValueError(f"table inconnue : {table!r}")  # ``table`` et ``time_column`` viennent d'une liste blanche interne
    conn = _open_read_only(path)
    try:
        present = {str(r["name"]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in present:
            raise RuntimeError(f"{path} : pas de table {table!r} (tables présentes : {sorted(present)}) ; ce n'est pas une base de fills.")
        cursor = conn.execute(f"SELECT * FROM {table} WHERE {time_column} >= ? ORDER BY {time_column}, rowid", (since_ms,))  # noqa: S608
        return [dict(r) for r in cursor if symbol_matches(str(r["symbol"]), symbol)]
    finally:
        conn.close()


def read_vpin(path: Path, *, since_ms: int, symbol: str) -> list[dict[str, Any]]:
    """Seaux VPIN de ``symbol`` plus récents que ``since_ms``, par ordre chronologique (``id`` départage les seaux fermés par un même trade)."""
    conn = _open_read_only(path)
    try:
        present = {str(r["name"]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "vpin_buckets" not in present:
            raise RuntimeError(f"{path} : pas de table 'vpin_buckets' (tables présentes : {sorted(present)}) ; ce n'est pas une archive VPIN.")
        cursor = conn.execute("SELECT * FROM vpin_buckets WHERE timestamp_ms >= ? ORDER BY timestamp_ms, id", (since_ms,))
        return [dict(r) for r in cursor if symbol_matches(str(r["symbol"]), symbol)]
    finally:
        conn.close()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Écrit ``rows`` ; refuse d'écraser. Renvoie le nombre de lignes (0 : rien n'est écrit). Colonnes : union ordonnée des clés."""
    if not rows:
        return 0
    header = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:  # "x" : jamais d'écrasement
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# --- Archive du bus (TimescaleDB) ------------------------------------------------------------


def flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Aplatit un JSON imbriqué : ``{"features": {"vpin": 0.6}}`` -> ``("features.vpin", 0.6)``. Les listes restent du JSON."""
    if isinstance(value, Mapping):
        for key, inner in value.items():
            yield from flatten(inner, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, json.dumps(value) if isinstance(value, list) else value


def bus_row_to_record(time_: datetime, sender_id: str, topic: str, data: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"time": time_.astimezone(timezone.utc).isoformat(), "topic": topic, "sender_id": sender_id}
    record.update(flatten(data))
    return record


def bus_symbol(data: Mapping[str, Any]) -> str | None:
    for key in _BUS_SYMBOL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def select_bus_rows(rows: Iterable[tuple[datetime, str, str, Mapping[str, Any]]], symbol: str) -> list[dict[str, Any]]:
    """Garde les messages du symbole demandé. Un message sans symbole identifiable est écarté (jamais deviné)."""
    out: list[dict[str, Any]] = []
    for time_, sender_id, topic, data in rows:
        found = bus_symbol(data)
        if found is not None and symbol_matches(found, symbol):
            out.append(bus_row_to_record(time_, sender_id, topic, data))
    return out


async def fetch_bus_rows(start: datetime, end: datetime, topics: Sequence[str]) -> list[tuple[datetime, str, str, dict[str, Any]]]:
    """Lit ``bus_messages`` avec les réglages ``TIMESCALE_*`` du moteur (imports tardifs : ``--source fills`` n'exige ni asyncpg ni .env)."""
    import asyncpg

    from core.config import get_settings

    ts = get_settings().timescale
    pool = await asyncpg.create_pool(
        host=ts.host, port=ts.port, database=ts.database, user=ts.user,
        password=ts.password.get_secret_value() if ts.password is not None else None, min_size=1, max_size=2, timeout=10,
    )
    try:
        records = await pool.fetch(
            "SELECT time, sender_id, topic, data FROM bus_messages WHERE topic = ANY($1::text[]) AND time >= $2 AND time < $3 ORDER BY time",
            list(topics), start, end,
        )
    finally:
        await pool.close()
    return [(r["time"], r["sender_id"], r["topic"], json.loads(r["data"]) if isinstance(r["data"], str) else dict(r["data"])) for r in records]


# --- CLI -------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="symbole à exporter, ex. BTCUSDT ou BTC/USDT (obligatoire : un jeu de données = un symbole)")
    parser.add_argument("--since-days", type=float, default=30.0, help="profondeur de l'historique, en jours (défaut 30)")
    parser.add_argument("--source", choices=("fills", "bus", "vpin", "all"), default="all", help="fills = bases SQLite ; bus = archive TimescaleDB ; vpin = série VPIN SQLite ; all = tout")
    parser.add_argument("--fills-db", nargs="*", default=None, help="bases de fills explicites (défaut : <NOTHIERLY_DATA_DIR>/*_fills.sqlite3)")
    parser.add_argument("--vpin-db", default=None, help=f"archive VPIN explicite (défaut : <NOTHIERLY_DATA_DIR>/{VPIN_FILENAME})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"dossier de sortie (défaut {DEFAULT_OUT_DIR}/)")
    args = parser.parse_args(argv)
    if args.since_days <= 0:
        parser.error("--since-days doit être > 0")
    return args


def run(args: argparse.Namespace, *, now: datetime | None = None) -> dict[str, int]:
    """Exporte et renvoie ``{fichier: nombre de lignes}``. Lève ``RuntimeError`` si rien n'a pu être exporté."""
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=args.since_days)
    since_ms = int(start.timestamp() * 1000)
    out_dir = Path(args.out_dir)
    tag = f"{normalize_symbol(args.symbol)}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    written: dict[str, int] = {}

    if args.source in ("fills", "all"):
        paths = [Path(p) for p in args.fills_db] if args.fills_db else find_fills_databases(data_dir())
        if not paths:
            if args.source == "fills":
                raise RuntimeError(f"aucune base de fills dans {data_dir()} (motif {FILLS_GLOB}) ; donner --fills-db ou NOTHIERLY_DATA_DIR")
            print(f"[export] aucune base de fills dans {data_dir()} : fills ignorés", file=sys.stderr)
        for table, time_column in _TABLES.items():
            rows = [row for path in paths for row in read_table(path, table, since_ms=since_ms, symbol=args.symbol)]
            rows.sort(key=lambda r: r[time_column])
            target = out_dir / f"{table}_{tag}.csv"
            if write_csv(target, rows):
                written[str(target)] = len(rows)

    if args.source in ("vpin", "all"):
        vpin_path = Path(args.vpin_db) if args.vpin_db else data_dir() / VPIN_FILENAME
        if not vpin_path.is_file() and args.source == "all":
            print(f"[export] pas d'archive VPIN ({vpin_path}) : VPIN ignoré", file=sys.stderr)
        else:  # source explicite : une archive absente est une erreur (FileNotFoundError), pas un export vide
            vpin_rows = read_vpin(vpin_path, since_ms=since_ms, symbol=args.symbol)
            target = out_dir / f"vpin_{tag}.csv"
            if write_csv(target, vpin_rows):
                written[str(target)] = len(vpin_rows)

    if args.source in ("bus", "all"):
        try:
            records = asyncio.run(fetch_bus_rows(start, now, BUS_TOPICS))
        except Exception as exc:  # noqa: BLE001 -- asyncpg, pydantic (réglages) et réseau lèvent des familles sans parent commun
            raise RuntimeError(f"archive TimescaleDB illisible ({type(exc).__name__}: {exc}) ; --source fills pour n'exporter que les fills") from exc
        by_topic: dict[str, list[dict[str, Any]]] = {}
        for record in select_bus_rows(records, args.symbol):
            by_topic.setdefault(str(record["topic"]), []).append(record)
        for topic, rows in sorted(by_topic.items()):
            target = out_dir / f"{topic.lower()}_{tag}.csv"
            if write_csv(target, rows):
                written[str(target)] = len(rows)

    if not written:
        raise RuntimeError(f"rien à exporter pour {args.symbol!r} sur {args.since_days:g} jours (source : {args.source}) ; aucun fichier écrit")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    t0 = time.monotonic()
    try:
        written = run(args)
    except (RuntimeError, OSError, ValueError) as exc:  # FileNotFoundError est un OSError
        print(f"[export] ÉCHEC : {exc}", file=sys.stderr)
        return 1
    for path, count in written.items():
        print(f"[export] {count:>8} lignes -> {path}")
    print(f"[export] terminé en {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

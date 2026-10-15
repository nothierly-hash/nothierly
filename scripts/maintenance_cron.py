"""Démon de maintenance : purge des bases SQLite, ``VACUUM``, rétention et compression de TimescaleDB. Tourne en tâche de fond (Docker).

    python scripts/maintenance_cron.py                       # boucle : un cycle au démarrage, puis toutes les 24 h
    python scripts/maintenance_cron.py --once --dry-run      # un seul cycle, sans rien modifier : dit ce qui serait purgé
    python scripts/maintenance_cron.py --once --vpin-retention-days 90

**Ce qu'il fait à chaque cycle** (``--interval-hours``, 24 par défaut ; un cycle part ``--startup-delay-s`` après le démarrage) :

1. ``<NOTHIERLY_DATA_DIR>/vpin_history.sqlite3`` : supprime les seaux plus vieux que ``--vpin-retention-days`` (180 ; 0 = ne jamais purger).
2. ``*_fills.sqlite3`` : supprime les intentions d'ordre (``order_intents``) de plus de ``--intents-retention-days`` (7). Les fills
   (``executions``) sont des données d'entraînement : gardés (``--fills-retention-days 0``) sauf demande contraire.
3. ``*_worker.sqlite3`` : supprime les identifiants de signaux déjà vus (``seen_signals``) de plus de ``--seen-retention-days`` (7), très
   au-delà de la fenêtre de rejeu d'une heure. Les workers ne purgent cette table qu'à leur démarrage.
4. ``VACUUM`` d'une base seulement si au moins ``--vacuum-min-free-ratio`` (25 %) de son fichier est libre ET au moins
   ``--vacuum-min-free-mb`` (100 Mo) : un ``VACUUM`` réécrit le fichier et bloque les écrivains pendant ce temps.
5. TimescaleDB, table ``bus_messages`` : compression des morceaux de plus de ``--timescale-compress-after-days`` (7) et suppression de ceux de
   plus de ``--timescale-retention-days`` (120). Les politiques sont (ré)appliquées à chaque cycle, de façon idempotente ; c'est TimescaleDB
   lui-même qui les exécute ensuite. ``--skip-timescale`` pour les ignorer.

**La rétention doit dépasser la fenêtre d'entraînement.** ``train_model.py`` lit ``--lookback-days`` (30 par défaut) de ``bus_messages`` ; la
rétention TimescaleDB (120) et celle du VPIN (180) laissent de la marge. Exporter (``export_training_data.py``) AVANT de raccourcir une rétention : ce
qui est purgé ne revient pas.

**Pourquoi la suppression se fait par lots.** Un seul ``DELETE`` de millions de lignes tiendrait le verrou d'écriture pendant toute sa durée, et
le suiveur VPIN (qui écrit en continu dans ce fichier) accumulerait ses lignes en mémoire. Ici : lots de ``--batch-size`` lignes, un ``COMMIT``
entre chaque, une courte pause : l'écrivain s'intercale. Le suiveur garde jusqu'à 50 000 lignes en attente si l'accès disque est bloqué
(voir ``vpin_archive.py``) : un ``VACUUM`` long reste donc dans sa marge, sans l'élargir.

**Échec.** Une étape qui échoue est journalisée, n'arrête pas les autres, et déclenche une alerte webhook (``WEBHOOK_URL``) avec le détail ; la boucle
continue au cycle suivant. Une base verrouillée trop longtemps (``busy_timeout``) est simplement reportée au cycle suivant. ``--dry-run`` ne
modifie rien (``SELECT COUNT`` seulement). Un fichier témoin (``--heartbeat-file``) est touché chaque minute pour la sonde de santé du conteneur.

Non exécuté contre un vrai TimescaleDB : les instructions SQL des politiques sont construites par ``timescale_statements`` (testé) et suivent la
documentation de TimescaleDB 2.x ; à valider sur le serveur avec ``--once --dry-run`` puis ``--once``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# `python scripts/maintenance_cron.py` met scripts/ (pas la racine du projet) dans sys.path : sans cette ligne, `from workers.common.alerts`
# (plus bas) lève ModuleNotFoundError dans le conteneur, dont le WORKDIR n'est pas sur le chemin d'import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_log = logging.getLogger("maintenance")

ENV_DATA_DIR: Final[str] = "NOTHIERLY_DATA_DIR"  # même contrat que workers/common/paths.py
DEFAULT_DATA_DIR: Final[str] = "data/"
VPIN_FILENAME: Final[str] = "vpin_history.sqlite3"
DAY_MS: Final[int] = 86_400_000
_BUSY_TIMEOUT_S: Final[float] = 30.0
_HYPERTABLE: Final[str] = "bus_messages"
# table -> colonne d'horodatage en millisecondes (liste blanche : ces noms entrent dans du SQL)
_PRUNABLE: Final[dict[str, str]] = {"vpin_buckets": "timestamp_ms", "order_intents": "sent_at_ms", "executions": "timestamp", "seen_signals": "seen_ms"}


@dataclass(frozen=True)
class MaintenanceConfig:
    data_dir: Path
    vpin_retention_days: float = 180.0
    intents_retention_days: float = 7.0
    fills_retention_days: float = 0.0
    seen_retention_days: float = 7.0
    vacuum_min_free_ratio: float = 0.25
    vacuum_min_free_mb: float = 100.0
    batch_size: int = 10_000
    batch_pause_s: float = 0.05
    timescale_retention_days: int = 120
    timescale_compress_after_days: int = 7
    skip_timescale: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        if min(self.vpin_retention_days, self.intents_retention_days, self.fills_retention_days, self.seen_retention_days) < 0:
            raise ValueError("une rétention négative n'a pas de sens (0 = ne jamais purger)")
        if self.batch_size < 1:
            raise ValueError("batch_size doit être >= 1")
        if not 0 < self.vacuum_min_free_ratio <= 1:
            raise ValueError("vacuum_min_free_ratio doit être dans ]0, 1]")
        if self.timescale_compress_after_days < 1 or self.timescale_retention_days <= self.timescale_compress_after_days:
            raise ValueError("TimescaleDB : la rétention doit dépasser le délai de compression (compression > 0 jour)")


@dataclass
class StepReport:
    target: str
    deleted: int = 0
    vacuumed: bool = False
    bytes_before: int = 0
    bytes_after: int = 0
    skipped: str | None = None
    error: str | None = None


@dataclass
class CycleReport:
    steps: list[StepReport] = field(default_factory=list)

    @property
    def errors(self) -> list[StepReport]:
        return [s for s in self.steps if s.error is not None]


def data_dir() -> Path:
    return Path(os.environ.get(ENV_DATA_DIR) or DEFAULT_DATA_DIR)


# --- SQLite ----------------------------------------------------------------------------------


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_S, isolation_level=None)  # autocommit : transactions explicites, courtes
    conn.execute("PRAGMA busy_timeout = %d" % int(_BUSY_TIMEOUT_S * 1000))
    return conn


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def prune_table(path: Path, table: str, cutoff_ms: int, *, batch_size: int, pause_s: float = 0.0, dry_run: bool = False) -> int:
    """Supprime les lignes de ``table`` antérieures à ``cutoff_ms`` par lots (un ``COMMIT`` entre chaque). Renvoie le nombre de lignes (supprimées, ou
    qui le seraient en ``dry_run``). ``table`` doit figurer dans la liste blanche interne."""
    column = _PRUNABLE.get(table)
    if column is None:
        raise ValueError(f"table non purgeable : {table!r}")
    conn = _connect(path)
    try:
        if not _has_table(conn, table):
            return 0
        if dry_run:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} < ?", (cutoff_ms,)).fetchone()[0])  # noqa: S608
        total = 0
        while True:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} WHERE {column} < ? LIMIT ?)",  # noqa: S608
                (cutoff_ms, batch_size),
            )
            if cursor.rowcount <= 0:
                return total
            total += cursor.rowcount
            if pause_s > 0:
                time.sleep(pause_s)  # le suiveur VPIN place son écriture entre deux lots
    finally:
        conn.close()


def ensure_time_index(path: Path) -> None:
    """Index sur ``timestamp_ms`` seul (les archives anciennes n'ont que ``(symbol, timestamp_ms)``) : sans lui, chaque lot de purge balaie la table."""
    conn = _connect(path)
    try:
        if _has_table(conn, "vpin_buckets"):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vpin_buckets_time ON vpin_buckets(timestamp_ms)")  # même nom que vpin_archive.SCHEMA
    finally:
        conn.close()


def free_bytes(path: Path) -> tuple[int, int]:
    """``(octets libres dans le fichier, taille du fichier)`` d'après ``PRAGMA freelist_count`` / ``page_count``."""
    conn = _connect(path)
    try:
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        free = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        return free * page_size, pages * page_size
    finally:
        conn.close()


def maybe_vacuum(path: Path, cfg: MaintenanceConfig) -> bool:
    """``VACUUM`` seulement si le fichier a assez de pages libres (ratio ET volume). Renvoie ``True`` si un ``VACUUM`` a eu lieu."""
    free, size = free_bytes(path)
    if size == 0 or free / size < cfg.vacuum_min_free_ratio or free < cfg.vacuum_min_free_mb * 1024 * 1024:
        return False
    if cfg.dry_run:
        return True  # signale ce qui serait fait
    conn = _connect(path)
    try:
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return True


def _checkpoint(path: Path) -> None:
    conn = _connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # replie le fichier WAL (l'auto-checkpoint ne le raccourcit jamais)
    finally:
        conn.close()


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def maintain_database(path: Path, tables: Sequence[tuple[str, float]], cfg: MaintenanceConfig, now_ms: int) -> StepReport:
    """Purge ``tables`` (``(table, jours de rétention)`` ; 0 jour = ne rien purger) puis ``VACUUM`` si utile. Ne lève jamais : l'erreur va dans le rapport."""
    report = StepReport(target=path.name, bytes_before=_size(path))
    try:
        if any(table == "vpin_buckets" and days > 0 for table, days in tables) and not cfg.dry_run:
            ensure_time_index(path)
        for table, days in tables:
            if days <= 0:
                continue
            report.deleted += prune_table(path, table, now_ms - int(days * DAY_MS), batch_size=cfg.batch_size, pause_s=cfg.batch_pause_s, dry_run=cfg.dry_run)
        report.vacuumed = maybe_vacuum(path, cfg)
        if not cfg.dry_run:
            _checkpoint(path)
    except (sqlite3.Error, OSError) as exc:  # base verrouillée trop longtemps, disque, fichier qui n'en est pas une : reporté, jamais fatal
        report.error = f"{type(exc).__name__}: {exc}"
    report.bytes_after = _size(path)
    return report


def maintain_sqlite(cfg: MaintenanceConfig, now_ms: int) -> list[StepReport]:
    reports: list[StepReport] = []
    vpin = cfg.data_dir / VPIN_FILENAME
    if vpin.is_file():
        reports.append(maintain_database(vpin, [("vpin_buckets", cfg.vpin_retention_days)], cfg, now_ms))
    else:
        reports.append(StepReport(target=VPIN_FILENAME, skipped="absente"))
    for path in sorted(cfg.data_dir.glob("*_fills.sqlite3")):
        reports.append(maintain_database(path, [("order_intents", cfg.intents_retention_days), ("executions", cfg.fills_retention_days)], cfg, now_ms))
    for path in sorted(cfg.data_dir.glob("*_worker.sqlite3")):
        reports.append(maintain_database(path, [("seen_signals", cfg.seen_retention_days)], cfg, now_ms))
    return reports


# --- TimescaleDB -----------------------------------------------------------------------------


def timescale_statements(retention_days: int, compress_after_days: int) -> list[str]:
    """Instructions SQL idempotentes : compression, puis rétention. Les politiques existantes sont retirées avant d'être recréées, pour que la
    valeur demandée gagne toujours (``add_*_policy(if_not_exists => true)`` laisse l'ancienne configuration en place). Les durées sont des entiers validés."""
    if compress_after_days < 1 or retention_days <= compress_after_days:
        raise ValueError("la rétention doit dépasser le délai de compression")
    table = _HYPERTABLE
    return [
        f"ALTER TABLE {table} SET (timescaledb.compress, timescaledb.compress_segmentby = 'topic', timescaledb.compress_orderby = 'time DESC')",
        f"SELECT remove_compression_policy('{table}', if_exists => true)",
        f"SELECT add_compression_policy('{table}', INTERVAL '{int(compress_after_days)} days')",
        f"SELECT remove_retention_policy('{table}', if_exists => true)",
        f"SELECT add_retention_policy('{table}', INTERVAL '{int(retention_days)} days')",
    ]


async def apply_timescale_policies(cfg: MaintenanceConfig) -> StepReport:
    report = StepReport(target="timescaledb.bus_messages")
    statements = timescale_statements(cfg.timescale_retention_days, cfg.timescale_compress_after_days)
    if cfg.dry_run:
        report.skipped = "dry-run : " + " ; ".join(statements)
        return report
    try:
        import asyncpg  # import tardif : les modes SQLite seuls n'en ont pas besoin

        from core.config import get_settings

        ts = get_settings().timescale
        conn = await asyncpg.connect(
            host=ts.host, port=ts.port, database=ts.database, user=ts.user,
            password=ts.password.get_secret_value() if ts.password is not None else None, timeout=10,
        )
        try:
            found = await conn.fetchval("SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = $1", _HYPERTABLE)
            if found is None:
                report.skipped = f"{_HYPERTABLE} n'est pas encore une hypertable (l'archiviste n'a rien écrit) : reporté"
                return report
            for statement in statements:
                await conn.execute(statement)
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001 -- asyncpg, pydantic (réglages) et réseau lèvent des familles sans parent commun ; reporté, jamais fatal
        report.error = f"{type(exc).__name__}: {exc}"
    return report


# --- Cycle et boucle -------------------------------------------------------------------------


def run_cycle(cfg: MaintenanceConfig, *, now_ms: int | None = None) -> CycleReport:
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    report = CycleReport(steps=maintain_sqlite(cfg, now))
    if not cfg.skip_timescale:
        report.steps.append(asyncio.run(apply_timescale_policies(cfg)))
    for step in report.steps:
        if step.error is not None:
            _log.error("maintenance_step_failed target=%s error=%s", step.target, step.error)
        else:
            _log.info(
                "maintenance_step target=%s deleted=%d vacuumed=%s bytes=%d->%d%s",
                step.target, step.deleted, step.vacuumed, step.bytes_before, step.bytes_after, f" skipped={step.skipped}" if step.skipped else "",
            )
    return report


def summarize_errors(report: CycleReport) -> str:
    return " ; ".join(f"{s.target} : {s.error}" for s in report.errors)


def _touch(path: Path | None) -> None:
    if path is not None:
        try:
            path.touch()
        except OSError:
            _log.debug("heartbeat_file_unwritable", exc_info=True)


def _sleep(total_s: float, heartbeat: Path | None) -> None:
    """Dort ``total_s`` par tranches d'une minute en touchant le fichier témoin (la sonde de santé ne voit pas un conteneur « mort » de 24 h)."""
    end = time.monotonic() + total_s
    while (left := end - time.monotonic()) > 0:
        _touch(heartbeat)
        time.sleep(min(60.0, left))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="un seul cycle puis quitter")
    parser.add_argument("--dry-run", action="store_true", help="ne modifie rien : compte ce qui serait purgé et affiche les instructions TimescaleDB")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    parser.add_argument("--startup-delay-s", type=float, default=120.0, help="attente avant le premier cycle (laisse les services démarrer)")
    parser.add_argument("--data-dir", default=None, help="dossier des bases SQLite (défaut : $NOTHIERLY_DATA_DIR ou data/)")
    parser.add_argument("--vpin-retention-days", type=float, default=180.0, help="0 = ne jamais purger le VPIN")
    parser.add_argument("--intents-retention-days", type=float, default=7.0)
    parser.add_argument("--fills-retention-days", type=float, default=0.0, help="0 = garder tous les fills (données d'entraînement)")
    parser.add_argument("--seen-retention-days", type=float, default=7.0)
    parser.add_argument("--vacuum-min-free-ratio", type=float, default=0.25)
    parser.add_argument("--vacuum-min-free-mb", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--timescale-retention-days", type=int, default=120)
    parser.add_argument("--timescale-compress-after-days", type=int, default=7)
    parser.add_argument("--skip-timescale", action="store_true")
    parser.add_argument("--heartbeat-file", default=None, help="fichier touché chaque minute (sonde de santé Docker)")
    args = parser.parse_args(argv)
    if args.interval_hours <= 0 or args.startup_delay_s < 0:
        parser.error("--interval-hours doit être > 0 et --startup-delay-s >= 0")
    return args


def _config(args: argparse.Namespace) -> MaintenanceConfig:
    return MaintenanceConfig(
        data_dir=Path(args.data_dir) if args.data_dir else data_dir(), vpin_retention_days=args.vpin_retention_days,
        intents_retention_days=args.intents_retention_days, fills_retention_days=args.fills_retention_days, seen_retention_days=args.seen_retention_days,
        vacuum_min_free_ratio=args.vacuum_min_free_ratio, vacuum_min_free_mb=args.vacuum_min_free_mb, batch_size=args.batch_size,
        timescale_retention_days=args.timescale_retention_days, timescale_compress_after_days=args.timescale_compress_after_days,
        skip_timescale=args.skip_timescale, dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    try:
        cfg = _config(args)
    except ValueError as exc:
        print(f"[maintenance] configuration refusée : {exc}", file=sys.stderr)
        return 2
    heartbeat = Path(args.heartbeat_file) if args.heartbeat_file else None

    from workers.common.alerts import AlertConfigError, Severity, WebhookAlerter  # image du moteur : workers/common y est copié (Dockerfile)

    try:
        alerter = WebhookAlerter.from_env(source="maintenance")
    except AlertConfigError as exc:
        print(f"[maintenance] {exc}", file=sys.stderr)
        return 2
    _log.info("maintenance_start data_dir=%s dry_run=%s once=%s interval_h=%g", cfg.data_dir, cfg.dry_run, args.once, args.interval_hours)
    failing = False
    try:
        _sleep(0.0 if args.once else args.startup_delay_s, heartbeat)
        while True:
            _touch(heartbeat)
            report = run_cycle(cfg)
            if alerter is not None:
                if report.errors:
                    failing = True
                    alerter.alert(f"maintenance : {len(report.errors)} étape(s) en échec : {summarize_errors(report)}", key="maintenance_failed", severity=Severity.CRITICAL)
                elif failing:
                    failing = False
                    alerter.alert("maintenance : le dernier cycle s'est terminé sans erreur.", key="end:maintenance_failed", severity=Severity.INFO)
            if args.once:
                return 1 if report.errors else 0
            _sleep(args.interval_hours * 3600.0, heartbeat)
    except KeyboardInterrupt:
        _log.info("maintenance_stopped")
        return 0
    finally:
        if alerter is not None:
            alerter.close(2.0)


if __name__ == "__main__":
    raise SystemExit(main())

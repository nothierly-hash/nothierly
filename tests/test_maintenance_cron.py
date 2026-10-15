"""Tests du démon de maintenance (``scripts/maintenance_cron.py``) : purge par lots, VACUUM conditionnel, politiques TimescaleDB, pannes."""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
import types
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts import maintenance_cron as mc
from workers.common.fills_db import FillsDatabase
from workers.market_data.vpin_archive import BucketRecord, VpinArchive

NOW_MS = 1_760_000_000_000
DAY = mc.DAY_MS


def _cfg(tmp_path: Path, **over: Any) -> mc.MaintenanceConfig:
    base: dict[str, Any] = dict(data_dir=tmp_path, skip_timescale=True, batch_pause_s=0.0, batch_size=100)
    base.update(over)
    return mc.MaintenanceConfig(**base)


def _vpin_db(path: Path, ages_days: list[float]) -> Path:
    archive = VpinArchive(path)
    for i, age in enumerate(ages_days):
        archive.submit(BucketRecord("BTCUSDT", NOW_MS - int(age * DAY) + i, Decimal("0.5"), Decimal(60), Decimal(40), Decimal(100), 50))
    assert archive.close() == 0
    return path


def _count(path: Path, table: str, where: str = "1") -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0])
    finally:
        conn.close()


# --- Purge --------------------------------------------------------------------------------------


def test_prune_removes_only_the_old_rows_in_batches(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "vpin_history.sqlite3", [200.0] * 250 + [10.0] * 40)
    deleted = mc.prune_table(db, "vpin_buckets", NOW_MS - 180 * DAY, batch_size=100)
    assert deleted == 250 and _count(db, "vpin_buckets") == 40  # 3 lots (100 + 100 + 50) puis un lot vide


def test_dry_run_counts_and_deletes_nothing(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "vpin_history.sqlite3", [200.0] * 5 + [1.0])
    assert mc.prune_table(db, "vpin_buckets", NOW_MS - 180 * DAY, batch_size=100, dry_run=True) == 5
    assert _count(db, "vpin_buckets") == 6


def test_only_whitelisted_tables_can_be_pruned_and_a_missing_table_is_a_no_op(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "v.sqlite3", [1.0])
    with pytest.raises(ValueError, match="non purgeable"):
        mc.prune_table(db, "vpin_buckets; DROP TABLE vpin_buckets", 0, batch_size=1)
    assert mc.prune_table(db, "seen_signals", NOW_MS, batch_size=10) == 0  # table absente de cette base


def test_a_retention_of_zero_never_purges(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "vpin_history.sqlite3", [900.0] * 10)
    (report,) = [r for r in mc.maintain_sqlite(_cfg(tmp_path, vpin_retention_days=0.0), NOW_MS) if r.target == "vpin_history.sqlite3"]
    assert report.deleted == 0 and report.error is None and _count(db, "vpin_buckets") == 10


def test_every_kind_of_database_gets_its_own_retention(tmp_path: Path) -> None:
    _vpin_db(tmp_path / "vpin_history.sqlite3", [400.0, 1.0])
    fills = FillsDatabase(tmp_path / "bybit_fills.sqlite3")
    fills.close()
    conn = sqlite3.connect(tmp_path / "bybit_fills.sqlite3")
    conn.executemany(
        "INSERT INTO order_intents VALUES ('BYBIT', ?, NULL, 'BTCUSDT', 'BUY', 1.0, 1.0, 'signal', ?)", [("old", NOW_MS - 20 * DAY), ("new", NOW_MS - DAY)],
    )
    conn.executemany(
        "INSERT INTO executions (timestamp, venue, symbol, direction, fill_price, fill_qty, order_id, fill_id, kind, recorded_ms) VALUES (?, 'BYBIT', 'BTCUSDT', 'BUY', 1.0, 1.0, 'o', ?, 'signal', 0)",
        [(NOW_MS - 400 * DAY, "e-old"), (NOW_MS - DAY, "e-new")],
    )
    conn.commit()
    conn.close()
    worker = sqlite3.connect(tmp_path / "bybit_worker.sqlite3")
    worker.execute("CREATE TABLE seen_signals (signal_id TEXT PRIMARY KEY, seen_ms INTEGER NOT NULL)")
    worker.executemany("INSERT INTO seen_signals VALUES (?, ?)", [("old", NOW_MS - 30 * DAY), ("new", NOW_MS - 3600_000)])
    worker.commit()
    worker.close()

    reports = {r.target: r for r in mc.maintain_sqlite(_cfg(tmp_path), NOW_MS)}
    assert all(r.error is None for r in reports.values())
    assert _count(tmp_path / "vpin_history.sqlite3", "vpin_buckets") == 1
    assert _count(tmp_path / "bybit_fills.sqlite3", "order_intents") == 1
    assert _count(tmp_path / "bybit_fills.sqlite3", "executions") == 2  # les fills sont des données d'entraînement : gardés par défaut
    assert _count(tmp_path / "bybit_worker.sqlite3", "seen_signals") == 1
    mc.maintain_sqlite(_cfg(tmp_path, fills_retention_days=365.0), NOW_MS)
    assert _count(tmp_path / "bybit_fills.sqlite3", "executions") == 1


def test_an_absent_vpin_archive_is_reported_as_skipped_not_failed(tmp_path: Path) -> None:
    (report,) = mc.maintain_sqlite(_cfg(tmp_path), NOW_MS)
    assert report.skipped == "absente" and report.error is None


# --- VACUUM -------------------------------------------------------------------------------------


def test_vacuum_runs_only_when_enough_of_the_file_is_free_and_it_shrinks_the_file(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "vpin_history.sqlite3", [200.0] * 4000 + [1.0] * 10)
    strict = _cfg(tmp_path)  # seuils par défaut : 25 % ET 100 Mo -> pas de VACUUM pour un si petit fichier
    assert mc.maybe_vacuum(db, strict) is False
    mc.prune_table(db, "vpin_buckets", NOW_MS - 180 * DAY, batch_size=500)
    before = db.stat().st_size
    free, size = mc.free_bytes(db)
    assert free / size > 0.5  # le fichier est surtout vide
    assert mc.maybe_vacuum(db, strict) is False  # ratio atteint mais volume sous le seuil : on ne réécrit pas pour quelques centaines de Ko
    loose = _cfg(tmp_path, vacuum_min_free_mb=0.01, vacuum_min_free_ratio=0.1)
    assert mc.maybe_vacuum(db, loose) is True
    assert db.stat().st_size < before and _count(db, "vpin_buckets") == 10


def test_a_dry_run_never_vacuums_but_says_it_would(tmp_path: Path) -> None:
    db = _vpin_db(tmp_path / "vpin_history.sqlite3", [200.0] * 4000)
    mc.prune_table(db, "vpin_buckets", NOW_MS, batch_size=1000)
    size = db.stat().st_size
    assert mc.maybe_vacuum(db, _cfg(tmp_path, vacuum_min_free_mb=0.01, vacuum_min_free_ratio=0.1, dry_run=True)) is True
    assert db.stat().st_size == size


# --- Cohabitation avec l'écrivain VPIN ---------------------------------------------------------


def test_the_vpin_writer_keeps_writing_while_the_purge_runs_in_batches(tmp_path: Path) -> None:
    path = tmp_path / "vpin_history.sqlite3"
    archive = VpinArchive(path, flush_every_s=0.01)
    archive.start()
    for i in range(3000):
        archive.submit(BucketRecord("BTCUSDT", NOW_MS - 400 * DAY + i, Decimal("0.5"), Decimal(60), Decimal(40), Decimal(100), 50))
    deadline = time.monotonic() + 5
    while archive.written < 3000 and time.monotonic() < deadline:
        time.sleep(0.01)
    new_rows = 600
    pruned: list[int] = []

    def purge() -> None:
        pruned.append(mc.prune_table(path, "vpin_buckets", NOW_MS - 180 * DAY, batch_size=100, pause_s=0.002))

    thread = threading.Thread(target=purge)
    thread.start()
    for i in range(new_rows):  # l'écrivain reste actif pendant toute la purge
        archive.submit(BucketRecord("BTCUSDT", NOW_MS - DAY + i, Decimal("0.5"), Decimal(60), Decimal(40), Decimal(100), 50))
        time.sleep(0.001)
    thread.join(30)
    assert archive.close() == 0
    assert pruned == [3000] and archive.dropped == 0
    assert _count(path, "vpin_buckets") == new_rows  # aucune ligne neuve perdue, aucune ancienne restante


# --- Pannes -------------------------------------------------------------------------------------


def test_a_corrupt_database_is_reported_and_the_other_steps_still_run(tmp_path: Path) -> None:
    (tmp_path / "vpin_history.sqlite3").write_bytes(b"ceci n'est pas une base SQLite " * 100)
    fills = FillsDatabase(tmp_path / "bybit_fills.sqlite3")
    fills.close()
    reports = {r.target: r for r in mc.maintain_sqlite(_cfg(tmp_path), NOW_MS)}
    assert reports["vpin_history.sqlite3"].error is not None
    assert reports["bybit_fills.sqlite3"].error is None  # la panne d'une base n'arrête pas les autres


def test_a_database_locked_for_too_long_is_deferred_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _vpin_db(tmp_path / "vpin_history.sqlite3", [200.0] * 10)
    monkeypatch.setattr(mc, "_BUSY_TIMEOUT_S", 0.05)
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        (report,) = mc.maintain_sqlite(_cfg(tmp_path), NOW_MS)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert report.error is not None and "locked" in report.error
    assert _count(path, "vpin_buckets") == 10  # rien n'a été supprimé à moitié


def test_invalid_configurations_are_refused(tmp_path: Path) -> None:
    for over in ({"vpin_retention_days": -1.0}, {"batch_size": 0}, {"vacuum_min_free_ratio": 0.0}, {"timescale_retention_days": 7}, {"timescale_compress_after_days": 0}):
        with pytest.raises(ValueError):
            _cfg(tmp_path, **over)


# --- TimescaleDB --------------------------------------------------------------------------------


def test_timescale_statements_apply_compression_then_retention_idempotently() -> None:
    statements = mc.timescale_statements(120, 7)
    assert statements[0].startswith("ALTER TABLE bus_messages SET (timescaledb.compress") and "compress_segmentby = 'topic'" in statements[0]
    assert statements[1:3] == ["SELECT remove_compression_policy('bus_messages', if_exists => true)", "SELECT add_compression_policy('bus_messages', INTERVAL '7 days')"]
    assert statements[3:] == ["SELECT remove_retention_policy('bus_messages', if_exists => true)", "SELECT add_retention_policy('bus_messages', INTERVAL '120 days')"]
    with pytest.raises(ValueError):
        mc.timescale_statements(7, 7)  # la rétention doit dépasser la compression


class _FakeConn:
    def __init__(self, hypertable: bool) -> None:
        self.hypertable = hypertable
        self.executed: list[str] = []
        self.closed = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        return 1 if self.hypertable else None

    async def execute(self, statement: str) -> None:
        self.executed.append(statement)

    async def close(self) -> None:
        self.closed = True


def _fake_timescale(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn | Exception) -> None:
    async def connect(**kwargs: Any) -> _FakeConn:
        if isinstance(conn, Exception):
            raise conn
        return conn

    monkeypatch.setitem(sys.modules, "asyncpg", types.SimpleNamespace(connect=connect))
    settings = types.SimpleNamespace(timescale=types.SimpleNamespace(host="h", port=5432, database="d", user="u", password=None))
    monkeypatch.setattr("core.config.get_settings", lambda: settings)


def test_the_policies_are_applied_to_an_existing_hypertable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(hypertable=True)
    _fake_timescale(monkeypatch, conn)
    report = mc.run_cycle(_cfg(tmp_path, skip_timescale=False), now_ms=NOW_MS).steps[-1]
    assert report.error is None and conn.executed == mc.timescale_statements(120, 7) and conn.closed


def test_a_table_not_yet_created_is_deferred_to_the_next_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(hypertable=False)
    _fake_timescale(monkeypatch, conn)
    report = mc.run_cycle(_cfg(tmp_path, skip_timescale=False), now_ms=NOW_MS).steps[-1]
    assert report.error is None and report.skipped is not None and "reporté" in report.skipped and conn.executed == [] and conn.closed


def test_an_unreachable_timescale_is_a_reported_error_not_a_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_timescale(monkeypatch, ConnectionRefusedError("timescale down"))
    cycle = mc.run_cycle(_cfg(tmp_path, skip_timescale=False), now_ms=NOW_MS)
    assert cycle.errors
    error = cycle.errors[0].error
    assert error is not None and "ConnectionRefusedError" in error and "timescaledb" in cycle.errors[0].target


def test_a_dry_run_only_prints_the_timescale_statements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(hypertable=True)
    _fake_timescale(monkeypatch, conn)
    report = mc.run_cycle(_cfg(tmp_path, skip_timescale=False, dry_run=True), now_ms=NOW_MS).steps[-1]
    assert report.skipped is not None and "add_retention_policy" in report.skipped and conn.executed == []


# --- Ligne de commande --------------------------------------------------------------------------


def test_once_dry_run_exits_zero_and_touches_the_heartbeat_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    _vpin_db(tmp_path / "vpin_history.sqlite3", [400.0, 1.0])
    beat = tmp_path / "beat"
    code = mc.main(["--once", "--dry-run", "--skip-timescale", "--data-dir", str(tmp_path), "--heartbeat-file", str(beat)])
    assert code == 0 and beat.exists() and _count(tmp_path / "vpin_history.sqlite3", "vpin_buckets") == 2


def test_once_exits_with_an_error_code_when_a_step_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    (tmp_path / "vpin_history.sqlite3").write_bytes(b"x" * 4096)
    assert mc.main(["--once", "--skip-timescale", "--data-dir", str(tmp_path)]) == 1


def test_the_command_line_refuses_an_invalid_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    assert mc.main(["--once", "--data-dir", str(tmp_path), "--timescale-retention-days", "3"]) == 2
    with pytest.raises(SystemExit):
        mc.main(["--interval-hours", "0"])


def test_the_data_dir_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTHIERLY_DATA_DIR", str(tmp_path))
    assert mc.data_dir() == tmp_path
    monkeypatch.delenv("NOTHIERLY_DATA_DIR")
    assert mc.data_dir() == Path("data/")


def test_the_script_runs_as_a_standalone_file_outside_the_project_root(tmp_path: Path) -> None:
    """Comme dans le conteneur : `python scripts/maintenance_cron.py` (sys.path[0] = scripts/, ni PYTHONPATH ni répertoire courant du projet)."""
    import subprocess

    script = Path(mc.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--once", "--dry-run", "--skip-timescale", "--startup-delay-s", "0", "--data-dir", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, env={"WEBHOOK_URL": "", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "ModuleNotFoundError" not in result.stderr

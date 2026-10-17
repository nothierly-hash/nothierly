"""Tests de l'archive VPIN continue (``workers/market_data/vpin_archive.py``), de son branchement au démon et de son export."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sqlite3
import threading
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts import export_training_data as ex
from tests.test_vpin import Alerts, Rig, _agg
from workers.common.paths import ENV_DATA_DIR
from workers.market_data import vpin_archive as va
from workers.market_data import vpin_tracker as vt
from workers.market_data.vpin_archive import BucketRecord, VpinArchive

D = Decimal
T0_MS = 1_760_000_000_000


def _record(i: int = 0, *, symbol: str = "BTCUSDT", vpin: str | None = "0.4") -> BucketRecord:
    return BucketRecord(symbol, T0_MS + i * 1000, None if vpin is None else D(vpin), D("60"), D("40"), D("100"), 2)


def _rows(path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM vpin_buckets ORDER BY id"))
    finally:
        conn.close()


def _wait(predicate: Any, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# --- Archive : schéma, WAL, écriture ----------------------------------------------------------


def test_the_archive_is_a_wal_database_with_the_documented_columns(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "data" / "vpin_history.sqlite3")  # le dossier est créé
    archive.close()
    conn = sqlite3.connect(tmp_path / "data" / "vpin_history.sqlite3")
    assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(vpin_buckets)")]
    conn.close()
    assert columns[:6] == ["id", "timestamp", "timestamp_ms", "symbol", "vpin_value", "bucket_buy_vol"] and "bucket_sell_vol" in columns


def test_a_closed_bucket_is_written_with_utc_time_values_and_parameters(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3", flush_every_s=0.05)
    archive.start()
    archive.submit(_record(0))
    archive.submit(_record(1, vpin=None))  # seau de chauffe : valeur vide, volumes présents
    assert _wait(lambda: archive.written == 2)
    assert archive.close() == 0
    first, second = _rows(tmp_path / "v.sqlite3")
    assert first["timestamp"] == datetime.fromtimestamp(T0_MS / 1000, tz=timezone.utc).isoformat(timespec="milliseconds")
    assert first["timestamp"].endswith("+00:00") and first["timestamp_ms"] == T0_MS
    assert (first["symbol"], first["vpin_value"], first["bucket_buy_vol"], first["bucket_sell_vol"]) == ("BTCUSDT", 0.4, 60.0, 40.0)
    assert (first["bucket_size"], first["window_size"]) == (100.0, 2) and first["recorded_ms"] > 0
    assert second["vpin_value"] is None and second["bucket_buy_vol"] == 60.0


def test_close_flushes_everything_still_queued(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3", flush_every_s=30.0)  # le délai de lot ne doit pas retarder l'arrêt
    archive.start()
    for i in range(250):
        archive.submit(_record(i))
    assert archive.close() == 0 and len(_rows(tmp_path / "v.sqlite3")) == 250


def test_close_drains_even_if_the_writer_thread_was_never_started(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3")
    archive.submit(_record(0))
    assert archive.close() == 0 and len(_rows(tmp_path / "v.sqlite3")) == 1


# --- La boucle ne bloque jamais ----------------------------------------------------------------


def test_submit_never_waits_for_a_locked_database_and_nothing_is_lost_once_it_is_free(tmp_path: Path) -> None:
    path = tmp_path / "v.sqlite3"
    archive = VpinArchive(path, flush_every_s=0.02, busy_timeout_s=0.05, retry_every_s=0.02)
    archive.start()
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")  # un autre processus tient le verrou d'écriture
    try:
        t0 = time.perf_counter()
        for i in range(500):
            archive.submit(_record(i))
        assert time.perf_counter() - t0 < 0.5  # 500 soumissions pendant que le disque est bloqué : aucune attente
        assert _wait(lambda: archive.errors >= 1) and archive.last_error is not None and "locked" in archive.last_error
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert _wait(lambda: archive.written == 500)  # le lot retenu est réécrit, dans l'ordre
    assert archive.close() == 0 and archive.dropped == 0
    assert [r["timestamp_ms"] for r in _rows(path)] == [T0_MS + i * 1000 for i in range(500)]


def test_a_full_queue_drops_and_counts_instead_of_blocking(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3", queue_size=3)  # thread non démarré : rien ne vide la file
    results = [archive.submit(_record(i)) for i in range(5)]
    assert results == [True, True, True, False, False] and archive.dropped == 2
    assert archive.close() == 0 and len(_rows(tmp_path / "v.sqlite3")) == 3


def test_a_retained_backlog_is_bounded(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3", queue_size=4)
    rows: list[tuple[str, int, str, float | None, float, float, float, int, int]] = [("t", i, "BTCUSDT", None, 1.0, 1.0, 2.0, 2, i) for i in range(10)]
    archive._retain(rows)
    assert len(archive._pending) == 4 and archive.dropped == 6 and [r[1] for r in archive._pending] == [6, 7, 8, 9]  # les plus récentes
    archive._pending.clear()
    archive.close()


def test_an_unusable_path_fails_at_construction(tmp_path: Path) -> None:
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    with pytest.raises(OSError):
        VpinArchive(blocker / "sous" / "v.sqlite3")
    with pytest.raises(ValueError):
        VpinArchive(tmp_path / "w.sqlite3", queue_size=0)


# --- Branchement au démon ----------------------------------------------------------------------


def test_every_closed_bucket_is_archived_even_while_redis_is_down(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3", flush_every_s=0.02)
    archive.start()
    rig = Rig(archive=archive)  # seau de 100 $, fenêtre de 2 : un seau par trade (voir test_vpin.py)
    rig.sink.fail = True  # Redis en panne
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True), _agg(agg_id=3)]]
    asyncio.run(rig.run_until(lambda: rig.tracker.stats["buckets"] == 3))
    assert archive.close() == 0
    rows = _rows(tmp_path / "v.sqlite3")
    assert [r["vpin_value"] for r in rows] == [None, 1.0, 1.0]  # le 1er seau est de chauffe : pas de valeur, mais il est gardé
    assert all(r["symbol"] == "BTCUSDT" and r["bucket_size"] == 100.0 and r["window_size"] == 2 for r in rows)
    assert [r["bucket_buy_vol"] + r["bucket_sell_vol"] for r in rows] == [100.0, 100.0, 100.0]  # chaque seau vaut exactement bucket_size
    assert rig.tracker.stats["sink_errors"] >= 1


def test_no_archive_means_no_file_and_no_change_in_behaviour() -> None:
    rig = Rig()
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)]]
    asyncio.run(rig.run_until(lambda: rig.tracker.stats["buckets"] == 2))
    assert rig.tracker.stats["buckets"] == 2


class FailingArchive:
    """Archive qui échoue : le démon doit alerter, une seule fois, puis annoncer le retour à la normale."""

    path = Path("x")

    def __init__(self) -> None:
        self.errors = 0
        self.dropped = 0
        self.last_error: str | None = None
        self.backlog = 0
        self.submitted = 0

    def submit(self, record: BucketRecord) -> bool:
        self.submitted += 1
        return True


def test_a_failing_archive_alerts_once_and_the_recovery_is_announced() -> None:
    alerts = Alerts()
    fake = FailingArchive()
    rig = Rig(alerter=alerts, archive=fake)
    tracker = rig.tracker
    tracker._check_archive()
    assert alerts.alerts == []  # rien d'anormal : pas d'alerte
    fake.errors, fake.last_error, fake.backlog = 1, "OperationalError: database or disk is full", 40
    tracker._check_archive()
    tracker._check_archive()  # même état : pas de second message
    assert [k for k, _, _ in alerts.alerts] == ["vpin_archive_failing"] and "disk is full" in alerts.alerts[0][1]
    fake.backlog = 0
    tracker._check_archive()
    assert [k for k, _, _ in alerts.alerts] == ["vpin_archive_failing", "end:vpin_archive_failing"]


def test_dropped_rows_also_raise_the_alert() -> None:
    alerts = Alerts()
    fake = FailingArchive()
    rig = Rig(alerter=alerts, archive=fake)
    fake.dropped = 3
    rig.tracker._check_archive()
    assert [k for k, _, _ in alerts.alerts] == ["vpin_archive_failing"]


# --- Ligne de commande -------------------------------------------------------------------------


def test_the_command_line_archives_under_the_data_dir_unless_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tests.test_vpin import _run_cli

    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "vol"))
    monkeypatch.setenv("REDIS_HOST", "redis-broker")
    seen = _run_cli(monkeypatch)
    archive = seen["kwargs"]["archive"]
    assert isinstance(archive, VpinArchive) and archive.path == tmp_path / "vol" / "vpin_history.sqlite3" and archive.path.is_file()
    archive.close()
    assert _run_cli(monkeypatch, "--dry-run")["kwargs"].get("archive") is None  # dry-run n'écrit nulle part
    assert _run_cli(monkeypatch, "--no-archive")["kwargs"].get("archive") is None
    custom = _run_cli(monkeypatch, "--archive-db", str(tmp_path / "autre.sqlite3"))["kwargs"]["archive"]
    assert custom.path == tmp_path / "autre.sqlite3"
    custom.close()


def test_the_command_line_refuses_an_unusable_archive_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    monkeypatch.setattr(vt, "configure_logging", lambda *a, **k: None)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with pytest.raises(SystemExit):
        vt.main(["--archive-db", str(blocker / "sous" / "v.sqlite3")])  # pas de collecte silencieusement sans archive


def test_the_exporter_and_the_archive_agree_on_the_file_name() -> None:
    assert ex.VPIN_FILENAME == va.DEFAULT_FILENAME


# --- Extracteur --------------------------------------------------------------------------------


def _archive_with(path: Path, records: list[BucketRecord]) -> None:
    archive = VpinArchive(path)
    for record in records:
        archive.submit(record)
    assert archive.close() == 0


def _args(tmp_path: Path, **over: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(symbol="BTCUSDT", since_days=30.0, source="vpin", fills_db=None, vpin_db=None, out_dir=str(tmp_path / "out"))
    base.update(over)
    return argparse.Namespace(**base)


def test_the_exporter_writes_the_vpin_series_for_one_symbol_in_order(tmp_path: Path) -> None:
    now = datetime.fromtimestamp(T0_MS / 1000 + 3600, tz=timezone.utc)
    db = tmp_path / "vpin_history.sqlite3"
    _archive_with(db, [_record(2), _record(0, vpin=None), _record(1), _record(1, symbol="ETHUSDT"), BucketRecord("BTCUSDT", T0_MS - 40 * 86_400_000, D("0.9"), D(1), D(1), D(2), 2)])
    ((path, count),) = ex.run(_args(tmp_path, vpin_db=str(db)), now=now).items()
    assert count == 3 and Path(path).name.startswith("vpin_BTCUSDT_")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["timestamp_ms"] for r in rows] == [str(T0_MS), str(T0_MS + 1000), str(T0_MS + 2000)]  # trié ; ETH et le seau de 40 jours écartés
    assert rows[0]["vpin_value"] == "" and rows[1]["vpin_value"] == "0.4"  # chauffe : vide
    assert list(rows[0])[:6] == ["id", "timestamp", "timestamp_ms", "symbol", "vpin_value", "bucket_buy_vol"]


def test_the_vpin_archive_is_found_in_the_data_dir_and_read_while_the_writer_is_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    archive = VpinArchive(tmp_path / "vpin_history.sqlite3", flush_every_s=0.02)  # le démon tourne encore : WAL, lecture concurrente
    archive.start()
    archive.submit(_record(0))
    assert _wait(lambda: archive.written == 1)
    now = datetime.fromtimestamp(T0_MS / 1000 + 60, tz=timezone.utc)
    try:
        assert sum(ex.run(_args(tmp_path), now=now).values()) == 1
    finally:
        archive.close()


def test_a_missing_or_foreign_vpin_archive_is_an_error_for_the_vpin_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ex.run(_args(tmp_path, vpin_db=str(tmp_path / "absente.sqlite3")))
    foreign = tmp_path / "autre.sqlite3"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE kv (k TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="pas de table 'vpin_buckets'"):
        ex.run(_args(tmp_path, vpin_db=str(foreign)))


def test_source_all_skips_a_missing_vpin_archive_but_a_symbol_without_data_still_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "vide"))

    async def no_bus(*_: Any) -> AsyncIterator[Any]:
        return
        yield  # pragma: no cover -- fait de ``no_bus`` un générateur asynchrone vide

    monkeypatch.setattr(ex, "iter_bus_rows", no_bus)
    with pytest.raises(RuntimeError, match="rien à exporter"):
        ex.run(_args(tmp_path, source="all"))


def test_the_writer_thread_is_a_daemon_and_does_not_outlive_close(tmp_path: Path) -> None:
    archive = VpinArchive(tmp_path / "v.sqlite3")
    archive.start()
    assert any(t.name == "vpin-archive" and t.daemon for t in threading.enumerate())
    archive.close()
    assert not any(t.name == "vpin-archive" for t in threading.enumerate())

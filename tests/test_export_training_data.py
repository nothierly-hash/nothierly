"""Tests de l'extracteur du Shadow Run (``scripts/export_training_data.py``) et du dossier de données (``workers/common/paths.py``)."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from scripts import export_training_data as ex
from workers.common.fills_db import FillsDatabase
from workers.common import paths
from workers.common.paths import DEFAULT_DATA_DIR, ENV_DATA_DIR, data_dir, default_db_path, ensure_parent

NOW = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)
DAY_MS = 86_400_000


def _fills_db(path: Path, rows: list[tuple[int, str, str]]) -> Path:
    """Une vraie base de fills (schéma de production) avec ``(timestamp_ms, symbole, fill_id)``."""
    FillsDatabase(path).close()
    conn = sqlite3.connect(path)
    for ts, symbol, fill_id in rows:
        conn.execute(
            "INSERT INTO executions (timestamp, venue, symbol, direction, intended_price, fill_price, fill_qty, slippage_bps, order_id, fill_id, kind, fee, fee_currency, recorded_ms)"
            " VALUES (?, 'BYBIT', ?, 'BUY', 100.0, 100.1, 1.0, 10.0, 'o1', ?, 'signal', 0.01, 'USDT', ?)", (ts, symbol, fill_id, ts),
        )
    conn.commit()
    conn.close()
    return path


def _args(tmp_path: Path, **over: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(symbol="BTCUSDT", since_days=30.0, source="fills", fills_db=None, out_dir=str(tmp_path / "out"))
    base.update(over)
    return argparse.Namespace(**base)


def _fake_bus(rows: list[tuple[datetime, str, str, dict[str, Any]]]) -> Any:
    async def source(start: datetime, end: datetime, topics: Sequence[str]) -> AsyncIterator[tuple[datetime, str, str, dict[str, Any]]]:
        for row in rows:
            if row[2] in topics:
                yield row

    return source


def _ids(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [r["fill_id"] for r in csv.DictReader(handle)]


# --- Symboles ---------------------------------------------------------------------------------


@pytest.mark.parametrize("row,wanted,expected", [
    ("BTCUSDT", "BTC/USDT", True), ("btc_usdt", "BTCUSDT", True), ("BTC", "BTCUSDT", True), ("BTCUSDT", "BTC", True),
    ("ETHUSDT", "BTCUSDT", False), ("BTCB", "BTC", False), ("", "BTC", False), ("BTC", "", False), ("BTC-PERP", "BTC", True),
])
def test_symbol_matching(row: str, wanted: str, expected: bool) -> None:
    assert ex.symbol_matches(row, wanted) is expected


# --- Fills ------------------------------------------------------------------------------------


def test_exports_only_the_requested_symbol_and_window_in_chronological_order(tmp_path: Path) -> None:
    db = _fills_db(tmp_path / "bybit_fills.sqlite3", [
        (NOW_MS - 2 * DAY_MS, "BTCUSDT", "b"), (NOW_MS - 3 * DAY_MS, "BTCUSDT", "a"), (NOW_MS - 2 * DAY_MS, "ETHUSDT", "e"),
        (NOW_MS - 40 * DAY_MS, "BTCUSDT", "old"),
    ])
    written = ex.run(_args(tmp_path, fills_db=[str(db)]), now=NOW)
    ((path, count),) = written.items()  # order_intents est vide : pas de fichier
    assert count == 2 and Path(path).name == "executions_BTCUSDT_20261014T120000Z.csv"
    assert _ids(path) == ["a", "b"]


def test_fills_from_several_venues_are_merged_and_a_bare_symbol_matches(tmp_path: Path) -> None:
    a = _fills_db(tmp_path / "bybit_fills.sqlite3", [(NOW_MS - 1000, "BTCUSDT", "x")])
    b = _fills_db(tmp_path / "hyperliquid_fills.sqlite3", [(NOW_MS - 500, "BTC", "y")])
    ((path, count),) = ex.run(_args(tmp_path, fills_db=[str(a), str(b)]), now=NOW).items()
    assert count == 2 and _ids(path) == ["x", "y"]


def test_databases_are_discovered_in_the_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fills_db(tmp_path / "binance_fills.sqlite3", [(NOW_MS - 1000, "BTCUSDT", "z")])
    (tmp_path / "binance_worker.sqlite3").write_bytes(b"pas une base de fills")  # ne correspond pas au motif
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path))
    assert ex.find_fills_databases(data_dir()) == [tmp_path / "binance_fills.sqlite3"]
    assert sum(ex.run(_args(tmp_path), now=NOW).values()) == 1


def test_the_export_never_modifies_the_source_database(tmp_path: Path) -> None:
    db = _fills_db(tmp_path / "bybit_fills.sqlite3", [(NOW_MS - 1000, "BTCUSDT", "x")])
    conn = ex._open_read_only(db)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM executions")
    conn.close()
    check = sqlite3.connect(db)
    assert check.execute("SELECT COUNT(*) FROM executions").fetchone() == (1,)
    check.close()


# --- Échecs : bruyants, jamais un fichier vide ------------------------------------------------


def test_an_empty_export_fails_and_writes_nothing(tmp_path: Path) -> None:
    db = _fills_db(tmp_path / "bybit_fills.sqlite3", [(NOW_MS - 1000, "ETHUSDT", "e")])
    with pytest.raises(RuntimeError, match="rien à exporter"):
        ex.run(_args(tmp_path, fills_db=[str(db)]), now=NOW)
    assert not (tmp_path / "out").exists()


def test_an_existing_export_is_never_overwritten(tmp_path: Path) -> None:
    db = _fills_db(tmp_path / "bybit_fills.sqlite3", [(NOW_MS - 1000, "BTCUSDT", "x")])
    ex.run(_args(tmp_path, fills_db=[str(db)]), now=NOW)
    with pytest.raises(FileExistsError):
        ex.run(_args(tmp_path, fills_db=[str(db)]), now=NOW)  # même horodatage, même fichier


def test_a_missing_or_foreign_database_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ex.run(_args(tmp_path, fills_db=[str(tmp_path / "absente.sqlite3")]), now=NOW)
    foreign = tmp_path / "autre_fills.sqlite3"
    conn = sqlite3.connect(foreign)
    conn.execute("CREATE TABLE kv (k TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="pas de table 'executions'"):
        ex.run(_args(tmp_path, fills_db=[str(foreign)]), now=NOW)


def test_no_database_at_all_is_an_error_for_fills_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "vide"))
    with pytest.raises(RuntimeError, match="aucune base de fills"):
        ex.run(_args(tmp_path), now=NOW)


def test_main_reports_the_failure_with_exit_code_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = ex.main(["--symbol", "BTCUSDT", "--source", "fills", "--fills-db", str(tmp_path / "x.sqlite3"), "--out-dir", str(tmp_path)])
    assert code == 1 and "ÉCHEC" in capsys.readouterr().err


def test_the_symbol_is_mandatory_and_the_window_positive() -> None:
    with pytest.raises(SystemExit):
        ex.main([])
    with pytest.raises(SystemExit):
        ex.main(["--symbol", "BTCUSDT", "--since-days", "0"])


def test_an_unreachable_archive_is_reported_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*_: Any) -> AsyncIterator[Any]:
        raise ConnectionRefusedError("timescale down")
        yield  # pragma: no cover -- fait de ``boom`` un générateur asynchrone

    monkeypatch.setattr(ex, "iter_bus_rows", boom)
    with pytest.raises(RuntimeError, match="TimescaleDB illisible"):
        ex.run(_args(tmp_path, source="bus"), now=NOW)


# --- Archive du bus ---------------------------------------------------------------------------


def test_bus_rows_are_filtered_by_symbol_and_flattened() -> None:
    rows: list[tuple[datetime, str, str, dict[str, Any]]] = [
        (NOW, "market-data", "MARKET_DATA_L2", {"symbol": "BTC/USDT", "spread": 0.5}),
        (NOW, "market-data", "MARKET_DATA_L2", {"symbol": "ETH/USDT", "spread": 0.1}),
        (NOW, "execution-agent", "META_LABEL_DECISION", {"instrument": "BTC/USDT", "features": {"vpin": 0.61, "x": [1, 2]}, "vetoed": False}),
        (NOW, "market-data", "MARKET_DATA_L2", {"spread": 9.9}),  # sans symbole : écarté, pas deviné
    ]
    out = ex.select_bus_rows(rows, "BTCUSDT")
    assert len(out) == 2
    decision = out[1]
    assert decision["features.vpin"] == 0.61 and decision["features.x"] == "[1, 2]" and decision["topic"] == "META_LABEL_DECISION"
    assert decision["time"] == "2026-10-14T12:00:00+00:00"


def test_bus_export_writes_one_file_per_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "iter_bus_rows", _fake_bus([
        (NOW, "m", "MARKET_DATA_L2", {"symbol": "BTCUSDT", "spread": 1.0}), (NOW, "e", "META_LABEL_DECISION", {"instrument": "BTCUSDT", "vetoed": True}),
    ]))
    written = ex.run(_args(tmp_path, source="bus"), now=NOW)
    assert sorted(Path(p).name for p in written) == ["market_data_l2_BTCUSDT_20261014T120000Z.csv", "meta_label_decision_BTCUSDT_20261014T120000Z.csv"]


# --- Dossier de données -----------------------------------------------------------------------


def test_data_dir_follows_the_environment_and_defaults_to_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_DATA_DIR, raising=False)
    assert data_dir() == Path(DEFAULT_DATA_DIR) and default_db_path("a.sqlite3") == Path("data") / "a.sqlite3"
    monkeypatch.setenv(ENV_DATA_DIR, "")
    assert data_dir() == Path(DEFAULT_DATA_DIR)  # une variable vide ne signifie pas « dossier courant »
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "vol"))
    assert default_db_path("a.sqlite3") == tmp_path / "vol" / "a.sqlite3"


def test_default_db_path_is_pure_and_ensure_parent_creates_the_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "vol"))
    target = default_db_path("a.sqlite3")
    assert not (tmp_path / "vol").exists()  # construire le parseur (--help) ne touche pas au disque
    assert ensure_parent(target) == target and (tmp_path / "vol").is_dir()


def test_ensure_parent_fails_fast_when_the_directory_is_impossible(tmp_path: Path) -> None:
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    with pytest.raises(OSError):
        ensure_parent(blocker / "sous" / "a.sqlite3")


def test_the_exporter_and_the_workers_agree_on_the_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert (ex.ENV_DATA_DIR, ex.DEFAULT_DATA_DIR) == (paths.ENV_DATA_DIR, paths.DEFAULT_DATA_DIR)
    for value in (None, "", str(tmp_path / "v")):
        if value is None:
            monkeypatch.delenv(ENV_DATA_DIR, raising=False)
        else:
            monkeypatch.setenv(ENV_DATA_DIR, value)
        assert ex.data_dir() == paths.data_dir()


# --- Flux : mémoire constante, fichiers jamais tronqués ----------------------------------------


def _bulk_fills(path: Path, n: int) -> None:
    FillsDatabase(path).close()
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO executions (timestamp, venue, symbol, direction, intended_price, fill_price, fill_qty, slippage_bps, order_id, fill_id, kind, fee, fee_currency, recorded_ms)"
        " VALUES (?, 'BYBIT', 'BTCUSDT', 'BUY', 100.0, 100.1, 1.0, 10.0, 'o1', ?, 'signal', 0.01, 'USDT', ?)",
        ((NOW_MS - n + i, f"f{i}", NOW_MS) for i in range(n)),
    )
    conn.commit()
    conn.close()


def test_memory_stays_flat_whatever_the_number_of_rows(tmp_path: Path) -> None:
    import tracemalloc

    peaks = {}
    for n in (20_000, 200_000):
        db = tmp_path / f"b{n}_fills.sqlite3"
        _bulk_fills(db, n)
        tracemalloc.start()
        ex.run(_args(tmp_path, fills_db=[str(db)], out_dir=str(tmp_path / f"out{n}")), now=NOW)
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    assert peaks[200_000] < 3 * peaks[20_000] + 2_000_000  # 10 fois plus de lignes, pas 10 fois plus de mémoire
    assert peaks[200_000] < 20_000_000  # une liste de 200 000 dictionnaires dépasserait largement 100 Mo
    out = next((tmp_path / "out200000").glob("executions_*.csv"))
    with open(out, newline="", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 200_001  # en-tête + toutes les lignes


def test_several_databases_are_merged_in_time_order_without_loading_them(tmp_path: Path) -> None:
    a = _fills_db(tmp_path / "a_fills.sqlite3", [(NOW_MS - 5000, "BTCUSDT", "a1"), (NOW_MS - 3000, "BTCUSDT", "a2"), (NOW_MS - 1000, "BTCUSDT", "a3")])
    b = _fills_db(tmp_path / "b_fills.sqlite3", [(NOW_MS - 4000, "BTCUSDT", "b1"), (NOW_MS - 2000, "BTCUSDT", "b2")])
    ((path, count),) = ex.run(_args(tmp_path, fills_db=[str(a), str(b)]), now=NOW).items()
    assert count == 5 and _ids(path) == ["a1", "b1", "a2", "b2", "a3"]


def test_a_failure_midway_leaves_no_file_and_no_part_file(tmp_path: Path) -> None:
    def rows() -> Any:
        yield {"a": 1}
        raise RuntimeError("disque plein")

    target = tmp_path / "x.csv"
    with pytest.raises(RuntimeError, match="disque plein"):
        ex.write_csv_stream(target, ["a"], rows())
    assert not target.exists() and not (tmp_path / "x.csv.part").exists()  # jamais un CSV tronqué sous un nom plausible


def test_a_completed_export_leaves_only_the_final_file(tmp_path: Path) -> None:
    target = tmp_path / "x.csv"
    assert ex.write_csv_stream(target, ["a", "b"], [{"a": 1, "b": 2}, {"a": 3}]) == 2
    assert target.read_text(encoding="utf-8").splitlines() == ["a,b", "1,2", "3,"]  # colonne manquante : vide
    assert not (tmp_path / "x.csv.part").exists()
    assert ex.write_csv_stream(tmp_path / "vide.csv", ["a"], []) == 0 and not (tmp_path / "vide.csv").exists()


def test_the_bus_export_uses_every_column_seen_even_when_messages_differ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "iter_bus_rows", _fake_bus([
        (NOW, "e", "META_LABEL_DECISION", {"instrument": "BTCUSDT", "features": {"vpin": 0.6}}),
        (NOW, "e", "META_LABEL_DECISION", {"instrument": "BTCUSDT", "features": {"vpin": 0.7, "rsi": 55}, "vetoed": True}),  # colonnes apparues plus tard
    ]))
    ((path, count),) = ex.run(_args(tmp_path, source="bus"), now=NOW).items()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert count == 2 and rows[0]["features.rsi"] == "" and rows[1]["features.rsi"] == "55" and rows[1]["vetoed"] == "True"


def test_a_bus_export_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "iter_bus_rows", _fake_bus([(NOW, "m", "MARKET_DATA_L2", {"symbol": "BTCUSDT", "spread": 1.0})]))
    ex.run(_args(tmp_path, source="bus"), now=NOW)
    with pytest.raises(FileExistsError):
        ex.run(_args(tmp_path, source="bus"), now=NOW)

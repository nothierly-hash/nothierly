"""Reconstitution historique : même VPIN que le tracker, barres exactes, trous et rejeux traités comme en direct, téléchargement vérifié et explicite."""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import urllib.request
import zipfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import backfill_vpin as bf
from src.rl import historical_features as hf
from workers.market_data.vpin import VolumeBucketVpin
from workers.market_data.vpin_archive import TABLE, VpinArchive

T0 = 1_760_000_040_000
HEADER = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def _rows(n: int, *, start_id: int = 1, start_ms: int = T0, step_ms: int = 1_000, price: float = 100.0, qty: float = 600.0) -> list[list[str]]:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        maker = "true" if rng.random() < 0.45 else "false"
        rows.append([str(start_id + i), f"{price + 0.01 * (i % 50):.2f}", f"{qty:.3f}", str(start_id + i), str(start_id + i), str(start_ms + i * step_ms), maker])
    return rows


def _zip(path: Path, rows: list[list[str]], *, header: bool = True) -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if header:
        writer.writerow(HEADER)
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", buffer.getvalue())
    return path


def _archive_rows(db: Path) -> list[tuple[int, float | None]]:
    with sqlite3.connect(db) as conn:
        return [(int(a), b) for a, b in conn.execute(f"SELECT timestamp_ms, vpin_value FROM {TABLE} ORDER BY timestamp_ms, id")]


def test_parse_row_handles_headers_sides_and_garbage() -> None:
    assert bf.parse_row(HEADER) is None
    assert bf.parse_row(["5", "100.5", "2", "5", "5", "1760000000000", "false"]) == (5, Decimal("100.5"), Decimal("2"), 1_760_000_000_000, True)  # preneur = acheteur
    sell = bf.parse_row(["6", "100.5", "2", "6", "6", "1760000000001", "true"])
    assert sell is not None and sell[4] is False
    with pytest.raises(ValueError):
        bf.parse_row(["1", "2"])
    with pytest.raises(ValueError):
        bf.parse_row(["1", "abc", "2", "1", "1", "1760000000000", "true"])
    with pytest.raises(ValueError):
        bf.parse_row(["1", "1", "2", "1", "1", "1760000000000", "maybe"])


def test_the_backfill_produces_exactly_what_the_live_tracker_computation_would(tmp_path: Path) -> None:
    rows = _rows(400)
    archive = VpinArchive(tmp_path / "a.sqlite3")
    counters = bf.backfill(rows, symbol="BTCUSDT", archive=archive, bucket_size=Decimal(5_000), window=5)
    assert archive.close() == 0

    reference = VolumeBucketVpin(Decimal(5_000), 5)  # la classe du tracker, appelée directement
    expected = []
    for row in rows:
        parsed = bf.parse_row(row)
        assert parsed is not None
        for reading in reference.add_trade(parsed[1], parsed[2], parsed[4], parsed[3]):
            expected.append((reading.closed_at_ms, None if reading.vpin is None else float(reading.vpin)))
    got = _archive_rows(tmp_path / "a.sqlite3")
    assert len(got) == len(expected) == counters["buckets"] and len(got) > 20
    assert [t for t, _ in got] == [t for t, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        assert (a is None) == (b is None) and (a is None or b is None or abs(a - b) < 1e-9)
    assert got[0][1] is None and got[-1][1] is not None  # pas de VPIN avant que la fenêtre soit pleine


def test_replayed_ids_are_ignored_and_a_gap_resets_the_window(tmp_path: Path) -> None:
    first = _rows(200)
    replay = first[100:150]  # identifiants déjà vus : ignorés comme dans le tracker
    after_gap = _rows(200, start_id=1_000, start_ms=T0 + 200_000 + 120_000)  # trou de 2 min > 30 s
    archive = VpinArchive(tmp_path / "a.sqlite3")
    c = bf.backfill([*first, *replay, *after_gap], symbol="BTCUSDT", archive=archive, bucket_size=Decimal(5_000), window=5)
    archive.close()
    assert c["skipped_ids"] == 50 and c["gaps"] == 1 and c["trades"] == 400
    rows = _archive_rows(tmp_path / "a.sqlite3")
    first_after_gap = next(i for i, (t, _) in enumerate(rows) if t > T0 + 200_000 + 119_000)
    assert all(v is None for _, v in rows[first_after_gap : first_after_gap + 4])  # la fenêtre repart de zéro après le trou


def test_streaming_bars_equal_the_batch_aggregation() -> None:
    parsed = [p for p in (bf.parse_row(r) for r in _rows(500, step_ms=700)) if p is not None]
    bars = bf.StreamingBars()
    for p in parsed:
        bars.add(p[3], p[1], p[2], p[4])
    streamed = bars.result()
    batch = hf.build_time_bars(
        np.array([p[3] for p in parsed], dtype=np.int64), np.array([float(p[1]) for p in parsed]), np.array([float(p[2]) for p in parsed]),
        np.array([p[4] for p in parsed]),
    )
    for key in batch:
        assert np.allclose(streamed[key], batch[key]), key


def test_run_treats_several_files_as_one_continuous_stream_and_writes_both_outputs(tmp_path: Path) -> None:
    day1 = _zip(tmp_path / "BTCUSDT-aggTrades-2026-10-01.zip", _rows(300, start_id=1))
    day2 = _zip(tmp_path / "BTCUSDT-aggTrades-2026-10-02.zip", _rows(300, start_id=301, start_ms=T0 + 300_000), header=False)  # sans en-tête, juste après
    out = bf.run([day1, day2], symbol="BTCUSDT", out_db=tmp_path / "v.sqlite3", out_bars=tmp_path / "bars.npz", bucket_size=Decimal(5_000), window=5, max_gap_s=30.0)
    assert out["trades"] == 600 and out["gaps"] == 0 and out["bars"] > 5  # aucun trou : le seau et la fenêtre passent d'un jour au suivant
    loaded = np.load(tmp_path / "bars.npz")
    assert str(loaded["symbol"]) == "BTCUSDT" and len(loaded["close"]) == out["bars"] and np.all(np.diff(loaded["bar_end_s"]) > 0)
    assert len(_archive_rows(tmp_path / "v.sqlite3")) == out["buckets"]
    with pytest.raises(FileNotFoundError):
        bf.run([tmp_path / "absent.zip"], symbol="BTCUSDT", out_db=tmp_path / "x.sqlite3", out_bars=tmp_path / "x.npz", bucket_size=Decimal(5_000), window=5, max_gap_s=30.0)


def test_download_is_explicit_and_listing_downloads_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(bf, "remote_size", lambda url: 123_000_000)
    monkeypatch.setattr(bf, "download_verified", lambda *a, **k: pytest.fail("aucun téléchargement attendu"))
    assert bf.main(["--symbol", "btcusdt", "--start", "2026-10-01", "--end", "2026-10-02", "--list"]) == 0
    out = capsys.readouterr().out
    assert "BTCUSDT-aggTrades-2026-10-01.zip" in out and "246.0 Mo" in out and "rien n'a été téléchargé" in out
    assert bf.main(["--symbol", "BTCUSDT", "--start", "2026-10-01", "--end", "2026-10-02", "--download-dir", str(tmp_path)]) == 2  # sans --yes-download
    assert bf.day_url("BTCUSDT", date(2026, 10, 1)) == "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-10-01.zip"
    with pytest.raises(ValueError):
        bf.days(date(2026, 10, 2), date(2026, 10, 1))


def test_a_download_with_a_wrong_checksum_is_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    payload = b"contenu du fichier"
    checksum = [hashlib.sha256(payload).hexdigest()]

    def fake_urlopen(url: str, timeout: float = 0) -> Response:
        return Response(f"{checksum[0]}  fichier.zip\n".encode() if url.endswith(".CHECKSUM") else payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    ok = tmp_path / "ok.zip"
    bf.download_verified("https://x/fichier.zip", ok)
    assert ok.read_bytes() == payload
    checksum[0] = "0" * 64
    bad = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="somme de contrôle"):
        bf.download_verified("https://x/fichier.zip", bad)
    assert not bad.exists()  # un fichier non vérifié n'est jamais conservé


def test_a_fast_pass_never_loses_buckets_even_with_a_tiny_writer_queue(tmp_path: Path) -> None:
    """Régression : la file de l'archive (50 000 par défaut) abandonne en silence quand elle est pleine ; l'historique doit attendre, pas perdre."""
    rows = _rows(600, qty=60.0)  # un seau par ligne environ (6 000 USDT) : des centaines de seaux, bien plus que la file de test
    archive = VpinArchive(tmp_path / "a.sqlite3", queue_size=4)
    counters = bf.backfill(rows, symbol="BTCUSDT", archive=archive, bucket_size=Decimal(6_000), window=5)
    assert archive.close() == 0 and archive.dropped == 0
    assert counters["buckets"] > 200 and len(_archive_rows(tmp_path / "a.sqlite3")) == counters["buckets"]

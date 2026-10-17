"""Rapport en lecture seule de l'archive VPIN : chiffres exacts sur une archive connue, et aucune écriture."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from workers.market_data import shadow_report
from workers.market_data.vpin_archive import BucketRecord, VpinArchive

T0 = 1_760_000_000_000


def _fill(path: Path) -> None:
    archive = VpinArchive(path)
    for i in range(10):  # BTCUSDT : un seau par minute, 2 sans VPIN (fenêtre pas encore pleine), VPIN 0,1 à 0,8 ensuite
        vpin = None if i < 2 else Decimal(str(round(0.1 * (i - 1), 1)))
        archive.submit(BucketRecord("BTCUSDT", T0 + i * 60_000, vpin, Decimal(60), Decimal(40), Decimal(100), 50))
    archive.submit(BucketRecord("ETHUSDT", T0, Decimal("0.95"), Decimal(60), Decimal(40), Decimal(100), 50))
    archive.submit(BucketRecord("ETHUSDT", T0 + 3_600_000, Decimal("0.2"), Decimal(60), Decimal(40), Decimal(100), 50))  # un trou d'une heure
    assert archive.close() == 0


def test_the_report_gives_exact_counts_ranges_gaps_and_toxic_shares(tmp_path: Path) -> None:
    db = tmp_path / "v.sqlite3"
    _fill(db)
    text = shadow_report.report(db, 0.7)
    assert "== BTCUSDT" in text and "seaux : 10 dont 8 avec un VPIN" in text
    assert "max 0.800" in text and "> 0.7 : 12.5 %" in text  # 1 seau sur 8 dépasse 0,7
    assert "== ETHUSDT" in text and "trous > 600 s : 1" in text and "plus long intervalle entre deux seaux : 3600 s" in text
    assert "aucune rentabilité" in text


def test_the_report_is_read_only_and_refuses_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "v.sqlite3"
    _fill(db)
    before = db.read_bytes()
    assert shadow_report.main(["--db", str(db)]) == 0
    assert db.read_bytes() == before  # aucune écriture
    assert "BTCUSDT" in capsys.readouterr().out
    assert shadow_report.main(["--db", str(tmp_path / "absente.sqlite3")]) == 1


def test_an_empty_archive_says_nothing_was_collected(tmp_path: Path) -> None:
    db = tmp_path / "v.sqlite3"
    VpinArchive(db).close()
    assert "rien n'a encore été collecté" in shadow_report.report(db, 0.7)


def test_the_window_keeps_only_recent_buckets_shows_hourly_counts_and_flags_empty_hours(tmp_path: Path) -> None:
    db = tmp_path / "w.sqlite3"
    archive = VpinArchive(db)
    base = 1_760_000_000_000 // 3_600_000 * 3_600_000  # début d'une heure UTC
    for i in range(30):  # heure H : 30 seaux
        archive.submit(BucketRecord("BTCUSDT", base + i * 60_000, Decimal("0.4"), Decimal(60), Decimal(40), Decimal(5_000_000), 50))
    for i in range(30):  # H+3 : 30 seaux ; H+1 et H+2 : aucun
        archive.submit(BucketRecord("BTCUSDT", base + 3 * 3_600_000 + i * 60_000, Decimal("0.4"), Decimal(60), Decimal(40), Decimal(5_000_000), 50))
    archive.submit(BucketRecord("BTCUSDT", base - 40 * 3_600_000, Decimal("0.9"), Decimal(60), Decimal(40), Decimal(5_000_000), 50))  # hors fenêtre de 24 h
    assert archive.close() == 0
    text = shadow_report.report(db, 0.7, hours=24, now_ms=base + 3 * 3_600_000 + 40 * 60_000)
    assert "les 24 dernières heures" in text and "seaux : 60 dont 60" in text and "max 0.400" in text  # le seau à 0,9 d'il y a 40 h est exclu
    assert "seaux par heure UTC" in text and "h:30" in text and "h:0" in text and "2 heure(s) sans aucun seau" in text
    assert "taille de seau : 5,000,000" in text and "SATURÉ" not in text and "premier seau de la fenêtre" in text  # 20 h de fenêtre sans aucun seau avant la collecte
    empty = shadow_report.report(db, 0.7, hours=1, now_ms=base + 100 * 3_600_000)
    assert "rien n'a encore été collecté" in empty


def test_a_saturated_vpin_is_flagged_with_its_bucket_size(tmp_path: Path) -> None:
    db = tmp_path / "s.sqlite3"
    archive = VpinArchive(db)
    for i in range(40):
        archive.submit(BucketRecord("BTCUSDT", 1_760_000_000_000 + i * 1000, Decimal("0.95"), Decimal(60), Decimal(40), Decimal(100_000), 50))
    assert archive.close() == 0
    text = shadow_report.report(db, 0.7)
    assert "VPIN SATURÉ" in text and "100 %" in text and "100,000" in text


def test_invalid_hours_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        shadow_report.main(["--db", str(tmp_path / "x.sqlite3"), "--hours", "0"])

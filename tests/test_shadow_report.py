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

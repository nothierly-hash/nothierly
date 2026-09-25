"""Tests for scripts/dukascopy_bi5.py.

Fixtures are built with the format's own layout (20-byte big-endian records, LZMA-compressed). Price and
volume values are ones a float32 stores exactly. The real-data validation (27,256 of 27,256 bids equal to a
website export) is recorded in TESTING.md.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import lzma
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from core.schemas import MarketMetrics


def _load(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bi5 = _load("dukascopy_bi5", "dukascopy_bi5.py")
zb = _load("zscore_backtest_for_bi5", "zscore_backtest.py")
ingester = _load("historical_ingester_for_bi5", "historical_ingester.py")

_HOUR = datetime(2016, 10, 6, 23, tzinfo=timezone.utc)


def _file(ticks: list[tuple[int, int, int, float, float]]) -> bytes:
    """(offset_ms, ask, bid, ask_volume, bid_volume) -> a compressed .bi5 payload."""
    return lzma.compress(b"".join(struct.pack(">IIIff", *tick) for tick in ticks))


_TICKS = [(149, 123681, 123668, 0.75, 1.5), (347, 123681, 123668, 0.75, 1.5), (398, 123682, 123669, 1.5, 0.75)]


def test_decodes_offsets_prices_and_volumes() -> None:
    ticks = list(bi5.decode_bi5(_file(_TICKS), _HOUR, 100_000))

    assert [t.timestamp.strftime("%H:%M:%S.%f")[:-3] for t in ticks] == ["23:00:00.149", "23:00:00.347", "23:00:00.398"]
    assert (ticks[0].bid, ticks[0].ask) == (1.23668, 1.23681)
    assert (ticks[2].bid, ticks[2].ask) == (1.23669, 1.23682)
    assert (ticks[0].ask_volume, ticks[0].bid_volume) == (0.75, 1.5)


def test_the_ask_is_read_before_the_bid_as_in_the_validated_layout() -> None:
    (tick,) = bi5.decode_bi5(_file([(0, 200, 100, 0.0, 0.0)]), _HOUR, 100)

    assert tick.ask == 2.0 and tick.bid == 1.0


def test_an_empty_file_is_an_hour_without_ticks() -> None:
    assert list(bi5.decode_bi5(b"", _HOUR, 100_000)) == []


def test_malformed_input_fails_loudly() -> None:
    with pytest.raises(ValueError, match="LZMA"):
        list(bi5.decode_bi5(b"not lzma at all", _HOUR, 100_000))
    with pytest.raises(ValueError, match="multiple"):
        list(bi5.decode_bi5(lzma.compress(b"x" * 25), _HOUR, 100_000))
    with pytest.raises(ValueError, match="timezone"):
        list(bi5.decode_bi5(_file(_TICKS), datetime(2016, 10, 6, 23), 100_000))


def test_an_impossible_quote_points_at_the_price_scale() -> None:
    with pytest.raises(ValueError, match="price-scale"):
        list(bi5.decode_bi5(_file([(0, 100, 200, 0.0, 0.0)]), _HOUR, 100_000))  # ask below bid
    with pytest.raises(ValueError, match="price-scale"):
        list(bi5.decode_bi5(_file([(0, 0, 0, 0.0, 0.0)]), _HOUR, 100_000))  # not positive


def test_the_price_scale_must_be_a_power_of_ten() -> None:
    assert bi5.decimals_for(100_000) == 5 and bi5.decimals_for(1_000) == 3 and bi5.decimals_for(1) == 0
    for bad in (0, -10, 250, 99_999):
        with pytest.raises(ValueError, match="power of ten"):
            bi5.decimals_for(bad)


def test_consecutive_hours_are_offset_and_an_empty_hour_still_counts(tmp_path: Path) -> None:
    first, empty, third = tmp_path / "a.bi5", tmp_path / "b.bi5", tmp_path / "c.bi5"
    first.write_bytes(_file([(5000, 123681, 123668, 1.0, 1.0)]))
    empty.write_bytes(b"")
    third.write_bytes(_file([(10, 123690, 123677, 1.0, 1.0)]))

    ticks = list(bi5.decode_files([first, empty, third], _HOUR, 100_000))

    assert [t.timestamp.isoformat(timespec="milliseconds") for t in ticks] == [
        "2016-10-06T23:00:05.000+00:00", "2016-10-07T01:00:00.010+00:00",
    ]


def test_an_error_names_the_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.bi5"
    broken.write_bytes(b"garbage")

    with pytest.raises(ValueError, match="broken.bi5"):
        list(bi5.decode_files([broken], _HOUR, 100_000))


def test_csv_has_the_documented_columns_and_formatting(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"

    written = bi5.write_csv(bi5.decode_bi5(_file(_TICKS), _HOUR, 100_000), output, 100_000)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert written == 3 and list(rows[0].keys()) == list(bi5.CSV_COLUMNS)
    assert rows[0]["timestamp_ms"] == str(1475794800149) and rows[0]["time_utc"] == "2016-10-06T23:00:00.149+00:00"
    assert (rows[0]["bid"], rows[0]["ask"]) == ("1.23668", "1.23681")
    assert (rows[0]["bid_volume"], rows[0]["ask_volume"]) == ("1.5", "0.75")


def test_the_csv_is_read_directly_by_the_backtest_and_the_ingester(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    bi5.write_csv(bi5.decode_bi5(_file(_TICKS), _HOUR, 100_000), output, 100_000)

    quotes = list(zb.iter_quotes(output, fmt="csv", price_is="bid", spread=0.0, timestamp_column="timestamp_ms",
                                 price_column="bid", timestamp_unit="ms", ask_column="ask"))
    assert [(q.bid, q.ask) for q in quotes] == [(1.23668, 1.23681), (1.23668, 1.23681), (1.23669, 1.23682)]
    assert quotes[0].timestamp == pytest.approx(1475794800.149)

    class _Recorder:
        def __init__(self) -> None:
            self.payloads: list[Any] = []

        async def publish(self, payload: Any) -> None:
            self.payloads.append(payload)

    recorder = _Recorder()
    columns = ingester.ColumnMap(timestamp="timestamp_ms", bid="bid", ask="ask", bid_volume="bid_volume", ask_volume="ask_volume")
    stats = asyncio.run(ingester.replay_csv(output, recorder, default_symbol="GBP_USD", speed=0, columns=columns, timestamp_unit="ms"))
    metrics = MarketMetrics.model_validate(recorder.payloads[0].data)
    assert stats.sent == 3 and stats.assumed_spread_used is False
    assert metrics.spread == pytest.approx(0.00013) and metrics.order_book_imbalance == pytest.approx((1.5 - 0.75) / 2.25)


def test_cli_writes_the_file_and_reports_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "23h.bi5", tmp_path / "out.csv"
    source.write_bytes(_file(_TICKS))

    monkeypatch.setattr(sys, "argv", ["dukascopy_bi5.py", str(source), "--start", "2016-10-06T23:00:00Z", "--price-scale", "100000", "--output", str(output)])
    assert bi5.main() == 0
    assert "wrote 3 ticks" in capsys.readouterr().out and output.exists()

    monkeypatch.setattr(sys, "argv", ["dukascopy_bi5.py", str(source), "--start", "2016-10-06T23:30:00Z", "--price-scale", "100000", "--output", str(output)])
    assert bi5.main() == 2 and "whole UTC hour" in capsys.readouterr().out

    empty = tmp_path / "empty.bi5"
    empty.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["dukascopy_bi5.py", str(empty), "--start", "2016-10-06T23:00:00Z", "--price-scale", "100000", "--output", str(tmp_path / "e.csv")])
    assert bi5.main() == 1 and "no ticks" in capsys.readouterr().out

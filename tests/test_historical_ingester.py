"""Tests for scripts/historical_ingester.py.

One test does a real ZMQ round trip (in-process transport, real ``ZmqPublisher``, sockets closed and
the context terminated in ``finally`` -- see TESTING.md). The rest use a recording stand-in for the
publisher so they exercise parsing, mapping, rejection and pacing without any sockets.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import zmq
import zmq.asyncio

from core.schemas import MarketMetrics, MessagePayload
from core.zmq_bus import ZmqPublisher

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "historical_ingester.py"
_spec = importlib.util.spec_from_file_location("historical_ingester", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
historical_ingester = importlib.util.module_from_spec(_spec)
sys.modules["historical_ingester"] = historical_ingester
_spec.loader.exec_module(historical_ingester)


class _RecordingPublisher:
    def __init__(self) -> None:
        self.payloads: list[MessagePayload] = []

    async def publish(self, payload: MessagePayload) -> None:
        self.payloads.append(payload)


def _write_csv(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# =============================================================================
# Pure mapping
# =============================================================================


def test_tick_to_metrics_computes_spread_mid_imbalance_and_liquidity() -> None:
    tick = historical_ingester.parse_row({"bid": "1.1000", "ask": "1.1002", "bid_vol": "300", "ask_vol": "100"})

    metrics = historical_ingester.tick_to_metrics(tick, symbol="EUR_USD")

    assert metrics.spread == pytest.approx(0.0002)
    assert metrics.mid_price == pytest.approx(1.1001)
    assert metrics.order_book_imbalance == pytest.approx(0.5)
    assert metrics.top_of_book_liquidity == pytest.approx(400.0)


def test_missing_volumes_mean_neutral_imbalance_and_unknown_liquidity() -> None:
    tick = historical_ingester.parse_row({"bid": "1.1000", "ask": "1.1002", "bid_vol": "", "ask_vol": ""})

    metrics = historical_ingester.tick_to_metrics(tick, symbol="EUR_USD")

    assert metrics.order_book_imbalance == 0.0
    assert metrics.top_of_book_liquidity == 0.0


def test_a_crossed_book_is_rejected_by_the_schema() -> None:
    tick = historical_ingester.parse_row({"bid": "1.1005", "ask": "1.1004"})

    with pytest.raises(ValueError):
        historical_ingester.tick_to_metrics(tick, symbol="EUR_USD")


def test_timestamps_accept_iso_8601_and_epoch_seconds() -> None:
    iso = historical_ingester.parse_row({"bid": "1", "ask": "1", "timestamp": "2024-01-05T13:30:00Z"})
    epoch = historical_ingester.parse_row({"bid": "1", "ask": "1", "timestamp": "1704461400"})

    assert iso.timestamp_seconds == pytest.approx(epoch.timestamp_seconds)


# =============================================================================
# replay_csv against a recording publisher
# =============================================================================


@pytest.mark.asyncio
async def test_bad_rows_are_counted_and_reported_and_do_not_stop_the_replay(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", [
        "Timestamp,BID,Ask,bid_vol,ask_vol",
        "2024-01-05T13:30:00Z,1.10000,1.10020,300,100",
        "2024-01-05T13:30:01Z,1.10050,1.10040,100,100",
        "2024-01-05T13:30:02Z,not-a-number,1.1,1,1",
        "2024-01-05T13:30:03Z,1.10020,1.10040,,",
    ])
    publisher = _RecordingPublisher()

    stats = await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=0)

    assert stats.sent == 2
    assert stats.rejected == 2
    assert [reason.split(":")[0] for reason in stats.first_rejections] == ["line 3", "line 4"]


@pytest.mark.asyncio
async def test_published_ticks_are_stamped_now_not_with_the_csv_timestamp(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["timestamp,bid,ask", "2019-01-01T00:00:00Z,1.1,1.2"])
    publisher = _RecordingPublisher()

    await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=0)

    metrics = MarketMetrics.model_validate(publisher.payloads[0].data)
    assert abs((datetime.now(timezone.utc) - metrics.timestamp).total_seconds()) < 60


@pytest.mark.asyncio
async def test_the_row_symbol_wins_over_the_default(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["symbol,bid,ask", "GBP_USD,1.3,1.31", ",1.1,1.2"])
    publisher = _RecordingPublisher()

    await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=0)

    assert [p.data["symbol"] for p in publisher.payloads] == ["GBP_USD", "EUR_USD"]


@pytest.mark.asyncio
async def test_a_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["bid,mid", "1.1,1.15"])

    with pytest.raises(ValueError, match="ask"):
        await historical_ingester.replay_csv(csv_path, _RecordingPublisher(), default_symbol="EUR_USD", speed=0)


@pytest.mark.asyncio
async def test_pacing_needs_timestamps(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["bid,ask", "1.1,1.2"])
    publisher = _RecordingPublisher()

    stats = await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=1.0)

    assert stats.sent == 0
    assert "timestamp is required" in stats.first_rejections[0]


@pytest.mark.asyncio
async def test_speed_scales_the_gap_between_ticks(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["timestamp,bid,ask", "0,1.1,1.2", "0.2,1.1,1.2", "0.4,1.1,1.2"])
    publisher = _RecordingPublisher()

    started = time.perf_counter()
    await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=4.0)
    elapsed = time.perf_counter() - started

    assert len(publisher.payloads) == 3
    assert 0.08 <= elapsed < 0.35  # 0.4s of recorded time at 4x is about 0.1s


# =============================================================================
# Bid-only files (e.g. a Dukascopy BID export) and custom column names
# =============================================================================

def _dukascopy_columns() -> Any:
    return historical_ingester.ColumnMap(timestamp="Etc/UTC", bid="Close", ask=None)


@pytest.mark.asyncio
async def test_bid_only_file_uses_the_explicit_assumed_spread(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bid.csv", [
        "Etc/UTC,Open,High,Low,Close,Volume",
        "2024-06-07T12:00:01+00:00,1.08984,1.08984,1.08984,1.08984,900000",
        "2024-06-07T12:00:02+00:00,1.08983,1.08983,1.08983,1.08983,450000",
    ])
    publisher = _RecordingPublisher()

    stats = await historical_ingester.replay_csv(
        csv_path, publisher, default_symbol="EUR_USD", speed=0,
        columns=_dukascopy_columns(), assumed_spread=0.0001,
    )

    assert stats.sent == 2
    assert stats.assumed_spread_used is True
    first = MarketMetrics.model_validate(publisher.payloads[0].data)
    assert first.spread == pytest.approx(0.0001)
    assert first.mid_price == pytest.approx(1.08984 + 0.00005)
    # The one-sided Volume column is not turned into an imbalance or a liquidity figure.
    assert first.order_book_imbalance == 0.0
    assert first.top_of_book_liquidity == 0.0


@pytest.mark.asyncio
async def test_bid_only_file_without_an_assumed_spread_fails_loudly(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bid.csv", ["Etc/UTC,Close", "2024-06-07T12:00:01+00:00,1.08984"])

    with pytest.raises(ValueError, match="assumed-spread"):
        await historical_ingester.replay_csv(
            csv_path, _RecordingPublisher(), default_symbol="EUR_USD", speed=0, columns=_dukascopy_columns(),
        )


@pytest.mark.asyncio
async def test_a_real_ask_column_wins_over_the_assumed_spread(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", ["bid,ask", "1.1000,1.1003"])
    publisher = _RecordingPublisher()

    stats = await historical_ingester.replay_csv(
        csv_path, publisher, default_symbol="EUR_USD", speed=0, assumed_spread=0.0001,
    )

    assert stats.assumed_spread_used is False
    assert MarketMetrics.model_validate(publisher.payloads[0].data).spread == pytest.approx(0.0003)


@pytest.mark.asyncio
async def test_custom_column_names_are_case_insensitive_and_pace_by_the_named_timestamp(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "bid.csv", [
        "Etc/UTC,Close",
        "2024-06-07T12:00:00+00:00,1.1",
        "2024-06-07T12:00:00.4+00:00,1.1001",
    ])
    publisher = _RecordingPublisher()

    started = time.perf_counter()
    stats = await historical_ingester.replay_csv(
        csv_path, publisher, default_symbol="EUR_USD", speed=4.0,
        columns=historical_ingester.ColumnMap(timestamp="ETC/utc", bid="CLOSE", ask=None), assumed_spread=0.0001,
    )
    elapsed = time.perf_counter() - started

    assert stats.sent == 2
    assert 0.08 <= elapsed < 0.35  # 0.4s of recorded time at 4x is about 0.1s


# =============================================================================
# Real bid/ask books (e.g. a Binance bookTicker export): units, zip, quantities
# =============================================================================


@pytest.mark.asyncio
async def test_millisecond_timestamps_pace_correctly_only_with_the_unit_declared(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "book.csv", ["time_ms,bid,ask", "1685577600000,1.10,1.12", "1685577600200,1.10,1.12", "1685577600400,1.10,1.12"])
    publisher = _RecordingPublisher()

    started = time.perf_counter()
    stats = await historical_ingester.replay_csv(
        csv_path, publisher, default_symbol="X", speed=4.0, timestamp_unit="ms",
        columns=historical_ingester.ColumnMap(timestamp="time_ms"),
    )
    elapsed = time.perf_counter() - started

    assert stats.sent == 3
    assert 0.08 <= elapsed < 0.35  # 0.4 s of recorded time at 4x is about 0.1 s


@pytest.mark.asyncio
async def test_a_zip_is_read_directly(tmp_path: Path) -> None:
    archive = tmp_path / "book.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("book.csv", "bid,ask\n1.10,1.12\n1.11,1.13\n")
    publisher = _RecordingPublisher()

    stats = await historical_ingester.replay_csv(archive, publisher, default_symbol="X", speed=0)

    assert stats.sent == 2 and len(publisher.payloads) == 2


@pytest.mark.asyncio
async def test_real_ask_and_quantities_give_the_files_own_spread_imbalance_and_liquidity(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "book.csv", [
        "best_bid_price,best_bid_qty,best_ask_price,best_ask_qty",
        "0.2600,300,0.2604,100",
    ])
    publisher = _RecordingPublisher()
    columns = historical_ingester.ColumnMap(bid="best_bid_price", ask="best_ask_price",
                                            bid_volume="best_bid_qty", ask_volume="best_ask_qty")

    stats = await historical_ingester.replay_csv(csv_path, publisher, default_symbol="X", speed=0, columns=columns)

    metrics = MarketMetrics.model_validate(publisher.payloads[0].data)
    assert stats.assumed_spread_used is False
    assert metrics.spread == pytest.approx(0.0004) and metrics.mid_price == pytest.approx(0.2602)
    assert metrics.order_book_imbalance == pytest.approx(0.5) and metrics.top_of_book_liquidity == pytest.approx(400.0)


# =============================================================================
# Real ZMQ round trip
# =============================================================================


@pytest.mark.asyncio
async def test_real_zmq_round_trip_uses_the_bus_wire_format(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "ticks.csv", [
        "bid,ask,bid_vol,ask_vol",
        "1.10000,1.10020,300,100",
        "1.10010,1.10030,200,200",
    ])
    address = "inproc://historical-ingester-round-trip"
    context = zmq.asyncio.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.bind(address)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"MARKET_DATA_L2")
    publisher = ZmqPublisher(address, context=context, bind=False)

    async def _receive(count: int) -> list[list[bytes]]:
        return [await subscriber.recv_multipart() for _ in range(count)]

    # The receive must be pending before publishing: an asyncio ZMQ socket only services the
    # in-process connection while a recv is awaited, so an idle subscriber would miss everything.
    receiver = asyncio.ensure_future(_receive(2))
    try:
        await asyncio.sleep(0.2)  # slow-joiner settle
        stats = await historical_ingester.replay_csv(csv_path, publisher, default_symbol="EUR_USD", speed=0)
        frames = await asyncio.wait_for(receiver, timeout=2.0)
    finally:
        receiver.cancel()
        publisher.close()
        subscriber.close(linger=0)
        context.term()

    assert stats.sent == 2
    assert all(topic == b"MARKET_DATA_L2" for topic, _ in frames)
    first = MarketMetrics.model_validate(MessagePayload.deserialize(frames[0][1]).data)
    assert first.symbol == "EUR_USD"
    assert first.order_book_imbalance == pytest.approx(0.5)

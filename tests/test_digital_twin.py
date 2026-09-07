"""Real, executable tests for simulation/digital_twin.py.

Most tests here are fast, pure-logic tests (CSV parsing, timestamp
parsing, the Mock Guardian). One real integration test
(``test_backtest_against_real_historical_data``) runs the full
pipeline against the *actual* Tardis-format sample files in
``research/sample_data/`` -- not a fake or synthetic book -- verifying
this module's central claim (real historical replay) against real data,
not just its own internal logic. That test is slower (loading real
528MB/414MB CSVs) by necessity; kept to a single run, not repeated per
assertion, to keep the suite fast.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from core.nothierly_ontology import OrderSide
from simulation.digital_twin import (
    COVERAGE_END_UTC,
    COVERAGE_START_UTC,
    DigitalTwinError,
    MockGuardianAgent,
    _parse_timestamp,
    build_sample_client_trades_csv,
    parse_client_trades_csv,
    run_digital_twin_backtest,
)

# -- _parse_timestamp --------------------------------


def test_parse_timestamp_iso8601_with_z_suffix() -> None:
    result = _parse_timestamp("2020-01-01T00:00:10Z")
    assert result == datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_parse_timestamp_iso8601_with_explicit_offset() -> None:
    result = _parse_timestamp("2020-01-01T00:00:10+00:00")
    assert result == datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_parse_timestamp_unix_seconds() -> None:
    result = _parse_timestamp("1577836810")
    assert result == datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_parse_timestamp_unix_milliseconds() -> None:
    result = _parse_timestamp("1577836810000")
    assert result == datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_parse_timestamp_unix_microseconds() -> None:
    result = _parse_timestamp("1577836810000000")
    assert result == datetime(2020, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_parse_timestamp_garbage_raises() -> None:
    with pytest.raises(DigitalTwinError):
        _parse_timestamp("not-a-real-timestamp")


# -- parse_client_trades_csv --------------------------------


def test_parse_csv_real_brief_columns(tmp_path: Path) -> None:
    """Exactly the brief's own literal column set: timestamp, size, side, price -- no symbol."""
    csv_path = tmp_path / "past_trades.csv"
    csv_path.write_text("timestamp,size,side,price\n1577836810,1.5,BUY,7196.00\n1577836900,0.5,SELL,7185.00\n")
    trades = parse_client_trades_csv(csv_path)
    assert len(trades) == 2
    assert trades[0].side == OrderSide.BUY
    assert trades[0].quantity == Decimal("1.5")
    assert trades[0].symbol == "BTCUSDT"  # real default applied, no symbol column given
    assert trades[1].side == OrderSide.SELL


def test_parse_csv_with_optional_symbol_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "past_trades.csv"
    csv_path.write_text("timestamp,symbol,quantity,side,price\n1577836810,ETHUSDT,10,buy,130.5\n")
    trades = parse_client_trades_csv(csv_path)
    assert trades[0].symbol == "ETHUSDT"
    assert trades[0].quantity == Decimal("10")


def test_parse_csv_missing_required_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,size,price\n1577836810,1.5,7196.00\n")  # no side column
    with pytest.raises(DigitalTwinError):
        parse_client_trades_csv(csv_path)


def test_parse_csv_unrecognized_side_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,size,side,price\n1577836810,1.5,SIDEWAYS,7196.00\n")
    with pytest.raises(DigitalTwinError):
        parse_client_trades_csv(csv_path)


def test_parse_csv_empty_file_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    with pytest.raises(DigitalTwinError):
        parse_client_trades_csv(csv_path)


# -- build_sample_client_trades_csv --------------------------------


def test_build_sample_client_trades_csv_unknown_symbol_raises() -> None:
    with pytest.raises(DigitalTwinError):
        build_sample_client_trades_csv("DOGEUSDT")


def test_build_sample_client_trades_csv_returns_real_in_coverage_rows(tmp_path: Path) -> None:
    """Every row must be a genuinely parseable, in-coverage BTCUSDT trade -- not a fabricated one."""
    csv_text = build_sample_client_trades_csv("BTCUSDT", n_trades=8)
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(csv_text)

    trades = parse_client_trades_csv(csv_path)
    assert len(trades) == 8
    for trade in trades:
        assert trade.symbol == "BTCUSDT"
        assert COVERAGE_START_UTC <= trade.timestamp <= COVERAGE_END_UTC
        assert trade.price > 0
        assert trade.quantity > 0


def test_build_sample_client_trades_csv_is_genuinely_replayable(tmp_path: Path) -> None:
    """Real integration proof: the generated sample must actually replay against the real historical book,
    not just parse -- otherwise the dashboard's own 'try it now' button would silently hand visitors a CSV
    that fails the exact backtest it exists to demonstrate."""
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(build_sample_client_trades_csv("BTCUSDT", n_trades=5))

    report = run_digital_twin_backtest(csv_path)
    assert report.trades_total == 5
    assert report.trades_replayed > 0


# -- MockGuardianAgent --------------------------------


def test_mock_guardian_approves_small_slice() -> None:
    guardian = MockGuardianAgent(max_book_depth_fraction=Decimal("0.5"))
    verdict = guardian.assess(child_quantity=Decimal("1"), top_of_book_depth=Decimal("100"))
    assert verdict.approved is True


def test_mock_guardian_blocks_large_slice() -> None:
    guardian = MockGuardianAgent(max_book_depth_fraction=Decimal("0.5"))
    verdict = guardian.assess(child_quantity=Decimal("80"), top_of_book_depth=Decimal("100"))
    assert verdict.approved is False
    assert "80%" in verdict.reason


def test_mock_guardian_zero_depth_is_never_approved() -> None:
    guardian = MockGuardianAgent()
    verdict = guardian.assess(child_quantity=Decimal("1"), top_of_book_depth=Decimal("0"))
    assert verdict.approved is False


# -- run_digital_twin_backtest: coverage honesty (no real data touched) --------------------------------


def test_out_of_coverage_trade_is_reported_not_fabricated(tmp_path: Path) -> None:
    """A trade dated outside the real 2020-01-01 sample window must be honestly marked unavailable."""
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text("timestamp,size,side,price\n2024-06-01T12:00:00Z,1.0,BUY,50000\n")
    report = run_digital_twin_backtest(csv_path)
    assert report.trades_total == 1
    assert report.trades_replayed == 0
    assert report.rows[0].replay_available is False
    assert report.rows[0].unavailable_reason is not None
    assert report.total_naive_cost_usd == Decimal(0)
    assert report.total_ai_simulated_cost_usd == Decimal(0)


def test_unknown_symbol_trade_is_reported_not_fabricated(tmp_path: Path) -> None:
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text("timestamp,symbol,size,side,price\n1577836810,DOGEUSDT,1.0,BUY,0.01\n")
    report = run_digital_twin_backtest(csv_path)
    assert report.trades_replayed == 0
    assert report.rows[0].replay_available is False


# -- Real integration test against the actual historical sample data --------------------------------


def test_backtest_against_real_historical_data(tmp_path: Path) -> None:
    """The real thing: replay real trades against the real Tardis BTCUSDT sample book for 2020-01-01."""
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text(
        "timestamp,symbol,size,side,price\n"
        "1577836810000000,BTCUSDT,0.5,BUY,7200.00\n"   # real in-coverage BUY
        "1577837000000000,BTCUSDT,0.3,SELL,7190.00\n"  # real in-coverage SELL
        "1577836000000000,BTCUSDT,1.0,BUY,7196.00\n"   # before real coverage starts (00:00:05Z)
    )
    report = run_digital_twin_backtest(csv_path, execution_window_minutes=1, num_slices=3)

    assert report.trades_total == 3
    # At least the two real in-coverage trades must have been genuinely attempted (may or may not fully
    # fill depending on real captured depth/Mock Guardian -- but must not be silently skipped).
    in_coverage_rows = [r for r in report.rows if r.replay_available]
    assert len(in_coverage_rows) == 2
    out_of_coverage_rows = [r for r in report.rows if not r.replay_available]
    assert len(out_of_coverage_rows) == 1
    assert out_of_coverage_rows[0].quantity == Decimal("1.0")

    for row in in_coverage_rows:
        assert row.rl_signal in (-1, 0, 1)
        assert row.rl_confidence is not None
        assert 0.0 <= row.rl_confidence <= 1.0
        assert row.guardian_verdict is not None
        assert row.naive_price > 0
        # A real quantity-weighted average fill price from the real book, when the simulation produced one.
        if row.ai_simulated_price is not None:
            assert row.ai_simulated_price > 0

    assert "real historical L2 data" in report.coverage_note


def test_oversized_order_blocked_by_guardian_gets_a_precise_reason(tmp_path: Path) -> None:
    """Real regression test: found via live testing that an oversized order relative to real depth needs a
    specific, actionable reason, not a generic 'Guardian or depth' message covering three different real
    failure modes at once. 10 BTC (2.0 BTC per child, num_slices=5) at this real historical moment is
    empirically >> real top-of-book depth on either side (ask 0.219, bid 0.053) -- verified directly against
    research/sample_data before writing this test, on the *correct* side (BUY walks asks, not bids -- an
    earlier version of this test asserted against the wrong side and had to be corrected)."""
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text("timestamp,symbol,size,side,price\n1577836900000000,BTCUSDT,10.0,BUY,7185.00\n")
    report = run_digital_twin_backtest(csv_path)

    assert report.trades_replayed == 0  # the trade was attempted (replay_available=True) but never filled
    row = report.rows[0]
    assert row.replay_available is True
    assert row.ai_simulated_price is None
    assert row.unavailable_reason is not None
    assert "Mock Guardian blocked" in row.unavailable_reason
    assert "too large relative to real top-of-book depth" in row.unavailable_reason

"""Tests for scripts/zscore_backtest.py and scripts/generate_tear_sheet.py.

The fill scenarios use a hand-derivable price series (window 5, entry z 1.5, exit z 0.5):

    idx:    0    1   2    3    4    5    6    7    8    9    10
    price: 100  101  99  100  100  120  120  120  120  120  120

At idx 5 the window is [101, 99, 100, 100, 120]: mean 104, z = 1.99 -> SELL signal. At idx 8 the window
is [100, 120, 120, 120, 120]: mean 116, std 8, z = 0.5 exactly -> CLOSE signal (the band is inclusive).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _load(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


zb = _load("zscore_backtest", "zscore_backtest.py")
ts_mod = _load("generate_tear_sheet", "generate_tear_sheet.py")

_PRICES = [100.0, 101.0, 99.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0]
_FRACTIONAL_TS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.000, 5.002, 5.006, 5.010, 5.014, 5.016]
_WHOLE_SECOND_TS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0]


def _quotes(prices: list[float], timestamps: list[float], spread: float = 0.02) -> list[Any]:
    return [zb.make_quote(p, t, price_is="mid", spread=spread) for p, t in zip(prices, timestamps)]


def _run(quotes: list[Any], *, latency_ms: float = 5.0, size: float = 10.0, lot: float = 10.0) -> tuple[Any, list[Any]]:
    trades: list[Any] = []
    stats = zb.run_backtest(
        quotes, zb.ZScoreStrategy(window=5, entry_z=1.5, exit_z=0.5),
        zb.FillConfig(latency_ms / 1000.0, size, lot, 2.0, 1.0), symbol="TEST", on_trade=trades.append,
    )
    return stats, trades


# =============================================================================
# Z-score and decision rule
# =============================================================================


def test_zscore_matches_numpy_and_includes_the_current_value() -> None:
    values = list(np.random.default_rng(1).normal(100, 2, 100))
    z = zb.RollingZScore(100)
    result = None
    for v in values:
        result = z.update(v)
    assert result == pytest.approx((values[-1] - np.mean(values)) / np.std(values))


def test_zscore_is_none_until_the_window_is_full_and_for_a_constant_window() -> None:
    z = zb.RollingZScore(5)
    assert [z.update(100.0) for _ in range(4)] == [None] * 4
    assert z.update(100.0) is None  # full but constant: no defined z-score
    assert z.update(81_000.1) is not None


def test_decide_enters_only_when_flat_and_exits_inside_the_band() -> None:
    s = zb.ZScoreStrategy(window=5, entry_z=3.0, exit_z=0.5)
    assert s.decide(3.1, 0) == "SELL" and s.decide(-3.1, 0) == "BUY"
    assert s.decide(2.9, 0) is None and s.decide(None, 0) is None
    assert s.decide(-3.5, 1) is None and s.decide(0.6, -1) is None  # no flip, not yet inside the band
    assert s.decide(0.5, -1) == "CLOSE" and s.decide(-0.5, 1) == "CLOSE"


def test_strategy_rejects_an_exit_band_wider_than_the_entry() -> None:
    with pytest.raises(ValueError):
        zb.ZScoreStrategy(entry_z=0.5, exit_z=0.5)


# =============================================================================
# Fill model, in data time
# =============================================================================


def test_fill_happens_on_the_first_later_quote_after_the_latency_and_uses_bid_ask_and_commission() -> None:
    stats, trades = _run(_quotes(_PRICES, _FRACTIONAL_TS))

    assert stats.signals == 2 and stats.trades == 1 and stats.dropped_orders == 0
    trade = trades[0]
    assert trade.side == "SELL" and trade.exit_reason == "SIGNAL"
    # signal at idx 5 (t=5.000); idx 6 (5.002) is too early; filled at idx 7 (5.006) on the bid.
    assert trade.entry_ts == 5.006 and trade.entry_price == pytest.approx(119.99)
    assert trade.entry_latency_ms == pytest.approx(6.0, abs=1e-6)
    # close signal at idx 8 (5.010); idx 9 (5.014) too early; filled at idx 10 (5.016) on the ask.
    assert trade.exit_ts == 5.016 and trade.exit_price == pytest.approx(120.01)
    assert trade.exit_latency_ms == pytest.approx(6.0, abs=1e-6)
    assert trade.gross_pnl == pytest.approx(-0.02 * 10)          # short: -(120.01 - 119.99) x size
    assert trade.commission_usd == pytest.approx(4.0)            # 2 fills x 2.0 per lot x 1 lot
    assert trade.net_pnl == pytest.approx(-4.2)


def test_a_signal_is_never_filled_on_the_quote_that_produced_it_even_with_zero_latency() -> None:
    _, trades = _run(_quotes(_PRICES, _FRACTIONAL_TS), latency_ms=0.0)

    assert trades[0].entry_ts == 5.002  # idx 6, not idx 5


def test_whole_second_timestamps_push_the_fill_to_a_later_second_and_the_delay_is_recorded() -> None:
    stats, trades = _run(_quotes(_PRICES, _WHOLE_SECOND_TS))

    assert stats.fractional_second_quotes == 0
    assert trades[0].entry_latency_ms == pytest.approx(1000.0) and trades[0].exit_latency_ms == pytest.approx(1000.0)


def test_a_position_open_at_the_end_is_closed_on_the_last_quote_and_flagged() -> None:
    _, trades = _run(_quotes(_PRICES[:8], _FRACTIONAL_TS[:8]))

    assert len(trades) == 1 and trades[0].exit_reason == "END_OF_DATA"
    assert trades[0].exit_ts == 5.006 and trades[0].exit_latency_ms == 0.0


def test_a_signal_on_the_last_quote_is_dropped_not_filled() -> None:
    stats, trades = _run(_quotes(_PRICES[:6], _FRACTIONAL_TS[:6]))

    assert stats.signals == 1 and stats.dropped_orders == 1 and trades == []


# =============================================================================
# Readers
# =============================================================================


def test_csv_reader_uses_named_columns_and_iso_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "bid.csv"
    path.write_text("Etc/UTC,Open,High,Low,Close\n2024-06-07T12:00:01+00:00,1,1,1,1.08984\n", encoding="utf-8")

    (quote,) = list(zb.iter_quotes(path, fmt="csv", price_is="bid", spread=0.0001,
                                   timestamp_column="Etc/UTC", price_column="Close"))

    assert quote.bid == 1.08984 and quote.ask == pytest.approx(1.08994)
    assert quote.timestamp == pytest.approx(1717761601.0)


def test_binance_trades_are_read_headerless_from_a_zip_with_millisecond_timestamps(tmp_path: Path) -> None:
    archive = tmp_path / "BTCUSDT-trades-x.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("BTCUSDT-trades-x.csv",
                   "0,4261.48000000,0.1,426.1,1502942428322,True,True\n1,4261.50000000,1.6,6818.4,1502942432285,True,True\n")

    quotes = list(zb.iter_quotes(archive, fmt="binance-trades", price_is="mid", spread=1.0))

    assert [q.timestamp for q in quotes] == [pytest.approx(1502942428.322), pytest.approx(1502942432.285)]
    assert quotes[0].bid == pytest.approx(4260.98) and quotes[0].ask == pytest.approx(4261.98)


def test_a_missing_column_and_a_bad_row_fail_loudly(tmp_path: Path) -> None:
    no_price = tmp_path / "a.csv"
    no_price.write_text("timestamp,x\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="price"):
        list(zb.iter_quotes(no_price, fmt="csv", price_is="bid", spread=0.0, timestamp_unit="s"))

    bad_row = tmp_path / "b.csv"
    bad_row.write_text("timestamp,price\n1,100\n2,oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":3:"):
        list(zb.iter_quotes(bad_row, fmt="csv", price_is="bid", spread=0.0, timestamp_unit="s"))


def test_a_real_ask_column_gives_the_files_own_spread_and_millisecond_time(tmp_path: Path) -> None:
    path = tmp_path / "book.csv"
    path.write_text("event_time,best_bid,best_ask\n1685577600123,0.2600,0.2604\n1685577600456,0.2601,0.2603\n", encoding="utf-8")

    quotes = list(zb.iter_quotes(path, fmt="csv", price_is="bid", spread=0.0, timestamp_column="event_time",
                                 price_column="best_bid", timestamp_unit="ms", ask_column="best_ask"))

    assert [(q.bid, q.ask) for q in quotes] == [(0.2600, 0.2604), (0.2601, 0.2603)]
    assert quotes[0].timestamp == pytest.approx(1685577600.123)
    assert quotes[0].mid == pytest.approx(0.2602)


def test_a_crossed_or_missing_real_ask_fails_loudly_with_the_line(tmp_path: Path) -> None:
    crossed = tmp_path / "crossed.csv"
    crossed.write_text("t,bid,ask\n1,10.0,10.1\n2,10.2,10.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":3:.*crossed"):
        list(zb.iter_quotes(crossed, fmt="csv", price_is="bid", spread=0.0, timestamp_column="t",
                            price_column="bid", timestamp_unit="s", ask_column="ask"))
    no_ask = tmp_path / "noask.csv"
    no_ask.write_text("t,bid\n1,10.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ask"):
        list(zb.iter_quotes(no_ask, fmt="csv", price_is="bid", spread=0.0, timestamp_column="t",
                            price_column="bid", timestamp_unit="s", ask_column="ask"))


# =============================================================================
# Executions log
# =============================================================================


def _meta(run_id: str) -> dict[str, str]:
    return {"Run_ID": run_id, "Data_File": "f.csv", "Assumed_Spread": "0.02", "Price_Is": "mid",
            "Latency_Model_ms": "5.0", "Commission_Per_Lot": "2.0", "Quote_To_USD": "1.0"}


def test_logger_writes_the_required_columns_once_and_appends_across_runs(tmp_path: Path) -> None:
    _, trades = _run(_quotes(_PRICES, _FRACTIONAL_TS))
    path = tmp_path / "executions.csv"
    for run_id in ("run-a", "run-b"):
        logger = zb.TradeLogger(path, run_meta=_meta(run_id))
        logger.log(trades[0])
        logger.close()
        assert not logger._thread.is_alive()

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(zb.CSV_COLUMNS) and len(rows) == 2
    required = ["Timestamp_Entry", "Timestamp_Exit", "Symbol", "Side", "Size", "Entry_Price", "Exit_Price", "Net_PnL", "Latency_Slippage_ms"]
    assert all(column in rows[0] for column in required)
    assert {r["Run_ID"] for r in rows} == {"run-a", "run-b"} and {r["Source"] for r in rows} == {"SIMULATED_BACKTEST"}
    assert rows[0]["Side"] == "SELL" and float(rows[0]["Net_PnL"]) == pytest.approx(-4.2)
    assert path.read_text(encoding="utf-8").count("Timestamp_Entry") == 1


def test_logger_refuses_to_append_to_a_file_with_a_different_header(tmp_path: Path) -> None:
    path = tmp_path / "executions.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different header"):
        zb.TradeLogger(path, run_meta=_meta("r"))


# =============================================================================
# Tear sheet
# =============================================================================


def _row(exit_ts: str, pnl: float, *, run: str = "r1", source: str = "SIMULATED_BACKTEST") -> dict[str, str]:
    row = {c: "x" for c in zb.CSV_COLUMNS}
    row.update({"Timestamp_Entry": exit_ts, "Timestamp_Exit": exit_ts, "Net_PnL": repr(pnl), "Run_ID": run,
                "Source": source, "Symbol": "TEST", "Side": "BUY"})
    return row


def test_tear_sheet_refuses_unlabeled_rows_and_pooled_runs() -> None:
    with pytest.raises(ValueError, match="Source"):
        ts_mod.select_run([_row("2024-01-01T10:00:00+00:00", 1.0, source="LIVE")], None)
    mixed = [_row("2024-01-01T10:00:00+00:00", 1.0, run="a"), _row("2024-01-02T10:00:00+00:00", 1.0, run="b")]
    with pytest.raises(ValueError, match="--run-id"):
        ts_mod.select_run(mixed, None)
    assert len(ts_mod.select_run(mixed, "b")) == 1


def test_returns_are_daily_pnl_over_start_of_day_equity_not_a_pct_change_of_pnl() -> None:
    rows = [_row("2024-01-01T10:00:00+00:00", 10.0), _row("2024-01-01T15:00:00+00:00", 5.0),
            _row("2024-01-03T10:00:00+00:00", -20.0)]

    report = ts_mod.compute_report(rows, 1000.0)

    assert report.trades == 3 and report.wins == 2
    assert report.total_net_pnl == pytest.approx(-5.0) and report.total_return == pytest.approx(-0.005)
    assert report.daily_observations == 3  # Jan 1, Jan 2 (no trades, zero return), Jan 3
    assert report.first_day == date(2024, 1, 1) and report.last_day == date(2024, 1, 3)
    assert report.profit_factor == pytest.approx(15.0 / 20.0)
    assert report.max_drawdown == pytest.approx((995.0 - 1015.0) / 1015.0)
    assert report.sharpe is None and report.cumulative_pnl == [10.0, 15.0, -5.0]


def test_sharpe_needs_enough_daily_observations() -> None:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]
    rows = [_row(f"{day.isoformat()}T10:00:00+00:00", 5.0 if i % 3 else -3.0) for i, day in enumerate(days)]

    report = ts_mod.compute_report(rows, 10_000.0)

    assert report.daily_observations == 40 and report.sharpe is not None
    assert ts_mod.compute_report(rows[:29], 10_000.0).sharpe is None  # 29 daily observations: below the minimum


def test_report_refuses_a_starting_capital_the_trades_wipe_out() -> None:
    with pytest.raises(ValueError, match="zero or below"):
        ts_mod.compute_report([_row("2024-01-01T10:00:00+00:00", -500.0)], 100.0)


def test_html_carries_the_banner_and_the_assumptions() -> None:
    rows = [_row("2024-01-01T10:00:00+00:00", 10.0), _row("2024-01-02T10:00:00+00:00", -4.0)]
    report = ts_mod.compute_report(rows, 1000.0)

    page = ts_mod.render_html(rows, report, 1000.0)

    assert "SIMULATED BACKTEST - NOT LIVE OR PAPER-TRADING PERFORMANCE" in page
    assert "Assumed_Spread" in page and "n/a (fewer than 30 daily observations)" in page

"""Tick-level z-score mean-reversion backtest with a simulated fill model and an executions log.

This is an offline SIMULATION over historical data. It does not touch the live agents or the ZMQ bus,
and its output is not paper-trading or live performance: every row it writes is labeled
``SIMULATED_BACKTEST`` together with the assumptions that produced it.

Why it reads the data file directly instead of subscribing to the bus: the ingester
(``scripts/historical_ingester.py``) stamps published ticks with the current time, so the original tick
times are lost, and fill latency has to be modeled in the data's own time.

Strategy (a toy, not a claim of edge): z-score of the mid price over the last ``--window`` quotes
(population standard deviation, the current quote included). Flat and z > +entry -> SELL; flat and
z < -entry -> BUY; in a position and |z| <= exit -> CLOSE. One order in flight at a time; no flipping.

Fill model, all in data time:

- A signal computed on quote i is filled on the first later quote whose timestamp is at least
  ``signal_ts + --latency-ms``. A signal is never filled on the quote that produced it.
- BUY fills at that quote's ask and SELL at its bid. With a real ask column (``--ask-column``, price column
  = real bid) the spread is the file's own. Otherwise: with ``--price-is bid`` (a bid-only file) the ask is
  ``bid + --assumed-spread``; with ``--price-is mid`` (e.g. trade prints) the bid and ask are
  ``price -/+ spread / 2``, and that spread is an assumption, never measured.
- Commission: ``--commission-per-lot`` for every fill (entry and exit) per ``--lot-size`` units traded.
- ``Net_PnL`` = gross PnL x ``--quote-to-usd`` - commissions. It is in USD only if the quote currency is
  USD or a rate is supplied.
- A position still open at the end of the data is closed on the last quote and flagged END_OF_DATA.

Timestamp resolution limits the latency model. A file with whole-second timestamps cannot resolve 5 ms:
a signal is then filled on the first quote of a later second, so the real modeled delay can approach a
second. ``Latency_Slippage_ms`` records the modeled delay actually applied, and the run prints a warning.

Input formats: ``--format csv`` (header row; ``--timestamp-column``, ``--price-column``,
``--timestamp-unit iso|s|ms|us``) and ``--format binance-trades`` (headerless Binance export, millisecond
timestamps; a ``.zip`` is read directly).

Usage::

    python scripts/zscore_backtest.py EURUSD_bid.csv --symbol EUR_USD --timestamp-column Etc/UTC \\
        --price-column Close --assumed-spread 0.0001 --size 100000 --lot-size 100000
    python scripts/zscore_backtest.py BTCUSDT-trades-2020-03-25.zip --format binance-trades \\
        --symbol BTC_USDT --assumed-spread 1.0 --size 1 --lot-size 1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import io
import math
import queue
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Literal, TextIO

Action = Literal["BUY", "SELL", "CLOSE"]

SOURCE_LABEL: Final[str] = "SIMULATED_BACKTEST"
CSV_COLUMNS: Final[tuple[str, ...]] = (
    "Timestamp_Entry", "Timestamp_Exit", "Symbol", "Side", "Size", "Entry_Price", "Exit_Price", "Net_PnL",
    "Latency_Slippage_ms", "Exit_Latency_ms", "Gross_PnL", "Commission_USD", "Exit_Reason", "Source", "Run_ID",
    "Data_File", "Assumed_Spread", "Price_Is", "Latency_Model_ms", "Commission_Per_Lot", "Quote_To_USD",
)
_MIN_RELATIVE_STD: Final[float] = 1e-12
_TIME_UNIT_DIVISORS: Final[dict[str, float]] = {"s": 1.0, "ms": 1e3, "us": 1e6}
_SUBSECOND_TOLERANCE: Final[float] = 1e-6


@dataclass(frozen=True)
class Quote:
    timestamp: float
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def parse_timestamp(raw: str, unit: str) -> float:
    """Epoch seconds from an ISO-8601 string or a numeric epoch in seconds, milliseconds or microseconds."""
    if unit == "iso":
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00")).timestamp()
    return float(raw) / _TIME_UNIT_DIVISORS[unit]


def make_quote(price: float, timestamp: float, *, price_is: str, spread: float) -> Quote:
    if price <= 0 or spread < 0:
        raise ValueError(f"invalid price {price} or spread {spread}")
    if price_is == "bid":
        return Quote(timestamp, price, price + spread)
    return Quote(timestamp, price - spread / 2.0, price + spread / 2.0)


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8", newline="")
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle


def iter_quotes(
    path: Path,
    *,
    fmt: str,
    price_is: str,
    spread: float,
    timestamp_column: str = "timestamp",
    price_column: str = "price",
    timestamp_unit: str = "iso",
    ask_column: str | None = None,
) -> Iterator[Quote]:
    """Stream quotes from a file, in file order. Raises ``ValueError`` with the line number on a bad row."""
    with _open_text(path) as handle:
        if fmt == "binance-trades":
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if not row:
                    continue
                try:
                    yield make_quote(float(row[1]), parse_timestamp(row[4], "ms"), price_is=price_is, spread=spread)
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            return
        reader = csv.DictReader(handle)
        header = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        ts_key, price_key = timestamp_column.strip().lower(), price_column.strip().lower()
        ask_key = ask_column.strip().lower() if ask_column else None
        missing = [name for name in (ts_key, price_key, ask_key) if name and name not in header]
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = parse_timestamp(row[header[ts_key]], timestamp_unit)
                if ask_key is None:
                    yield make_quote(float(row[header[price_key]]), timestamp, price_is=price_is, spread=spread)
                else:
                    bid, ask = float(row[header[price_key]]), float(row[header[ask_key]])
                    if bid <= 0 or ask < bid:
                        raise ValueError(f"invalid or crossed quote (bid {bid}, ask {ask})")
                    yield Quote(timestamp, bid, ask)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


class RollingZScore:
    """Z-score of the latest value against the last ``window`` values (current included, ddof=0)."""

    def __init__(self, window: int = 100) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self._window = window
        self._buffer: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> float | None:
        """``None`` until the window is full, or when the window is (numerically) constant."""
        buffer = self._buffer
        buffer.append(value)
        count = len(buffer)
        if count < self._window:
            return None
        mean = sum(buffer) / count
        variance = sum((item - mean) ** 2 for item in buffer) / count
        if variance <= (_MIN_RELATIVE_STD * abs(mean)) ** 2:
            return None
        return (value - mean) / math.sqrt(variance)


class ZScoreStrategy:
    def __init__(self, *, window: int = 100, entry_z: float = 3.0, exit_z: float = 0.5) -> None:
        if not entry_z > exit_z >= 0:
            raise ValueError("need entry_z > exit_z >= 0")
        self._zscore = RollingZScore(window)
        self._entry_z = entry_z
        self._exit_z = exit_z

    def update(self, mid: float) -> float | None:
        return self._zscore.update(mid)

    def decide(self, z: float | None, position: int) -> Action | None:
        """``position`` is the executed net position: +1 long, -1 short, 0 flat."""
        if z is None:
            return None
        if position == 0:
            if z > self._entry_z:
                return "SELL"
            if z < -self._entry_z:
                return "BUY"
            return None
        return "CLOSE" if abs(z) <= self._exit_z else None


@dataclass(frozen=True)
class FillConfig:
    latency_seconds: float = 0.005
    size: float = 1.0
    lot_size: float = 1.0
    commission_per_lot: float = 2.0
    quote_to_usd: float = 1.0


@dataclass(frozen=True)
class TradeRecord:
    entry_ts: float
    exit_ts: float
    symbol: str
    side: Literal["BUY", "SELL"]
    size: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission_usd: float
    net_pnl: float
    entry_latency_ms: float
    exit_latency_ms: float
    exit_reason: Literal["SIGNAL", "END_OF_DATA"]


@dataclass
class BacktestStats:
    quotes: int = 0
    signals: int = 0
    trades: int = 0
    dropped_orders: int = 0
    net_pnl: float = 0.0
    fractional_second_quotes: int = 0
    alpha_ns_total: int = 0
    alpha_ns_max: int = 0


@dataclass
class _Pending:
    action: Action
    signal_ts: float
    signal_index: int
    due_ts: float


@dataclass
class _Open:
    side: Literal["BUY", "SELL"]
    entry_price: float
    entry_ts: float
    entry_latency_ms: float


def run_backtest(
    quotes: Iterable[Quote],
    strategy: ZScoreStrategy,
    config: FillConfig,
    *,
    symbol: str,
    on_trade: Callable[[TradeRecord], None],
) -> BacktestStats:
    """Run the strategy over ``quotes`` in order, emitting each closed trade through ``on_trade``."""
    stats = BacktestStats()
    pending: _Pending | None = None
    position: _Open | None = None
    last_quote: Quote | None = None

    def close_trade(open_position: _Open, exit_price: float, exit_ts: float, exit_latency_ms: float, reason: str) -> None:
        direction = 1.0 if open_position.side == "BUY" else -1.0
        gross = direction * (exit_price - open_position.entry_price) * config.size
        commission = 2.0 * config.commission_per_lot * config.size / config.lot_size
        net = gross * config.quote_to_usd - commission
        stats.trades += 1
        stats.net_pnl += net
        on_trade(TradeRecord(
            open_position.entry_ts, exit_ts, symbol, open_position.side, config.size, open_position.entry_price,
            exit_price, gross, commission, net, open_position.entry_latency_ms, exit_latency_ms,
            "SIGNAL" if reason == "SIGNAL" else "END_OF_DATA",
        ))

    for index, quote in enumerate(quotes):
        stats.quotes += 1
        last_quote = quote
        if abs(quote.timestamp - round(quote.timestamp)) > _SUBSECOND_TOLERANCE:
            stats.fractional_second_quotes += 1

        if pending is not None and index > pending.signal_index and quote.timestamp >= pending.due_ts:
            latency_ms = (quote.timestamp - pending.signal_ts) * 1000.0
            if pending.action in ("BUY", "SELL"):
                price = quote.ask if pending.action == "BUY" else quote.bid
                position = _Open(pending.action, price, quote.timestamp, latency_ms)
            elif position is not None:
                price = quote.bid if position.side == "BUY" else quote.ask
                close_trade(position, price, quote.timestamp, latency_ms, "SIGNAL")
                position = None
            pending = None

        started = time.perf_counter_ns()
        z = strategy.update(quote.mid)
        action = None
        if pending is None:
            action = strategy.decide(z, 0 if position is None else (1 if position.side == "BUY" else -1))
        elapsed = time.perf_counter_ns() - started
        stats.alpha_ns_total += elapsed
        stats.alpha_ns_max = max(stats.alpha_ns_max, elapsed)

        if action is not None:
            stats.signals += 1
            pending = _Pending(action, quote.timestamp, index, quote.timestamp + config.latency_seconds)

    if pending is not None:
        stats.dropped_orders += 1
    if position is not None and last_quote is not None:
        price = last_quote.bid if position.side == "BUY" else last_quote.ask
        close_trade(position, price, last_quote.timestamp, 0.0, "END_OF_DATA")
    return stats


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="milliseconds")


class TradeLogger:
    """Appends closed trades to a CSV from a dedicated thread, so file I/O never blocks the run loop."""

    def __init__(self, path: Path, *, run_meta: dict[str, str]) -> None:
        self._path = path
        self._meta = run_meta
        self._new_file = not path.exists() or path.stat().st_size == 0
        if not self._new_file:
            with path.open("r", encoding="utf-8", newline="") as handle:
                existing = next(csv.reader(handle), [])
            if tuple(existing) != CSV_COLUMNS:
                raise ValueError(f"{path} exists with a different header; refusing to append")
        self._queue: queue.Queue[dict[str, str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="trade-logger")
        self._thread.start()

    def log(self, trade: TradeRecord) -> None:
        self._queue.put({
            "Timestamp_Entry": _iso(trade.entry_ts), "Timestamp_Exit": _iso(trade.exit_ts), "Symbol": trade.symbol,
            "Side": trade.side, "Size": repr(trade.size), "Entry_Price": repr(trade.entry_price),
            "Exit_Price": repr(trade.exit_price), "Net_PnL": f"{trade.net_pnl:.6f}",
            "Latency_Slippage_ms": f"{trade.entry_latency_ms:.3f}", "Exit_Latency_ms": f"{trade.exit_latency_ms:.3f}",
            "Gross_PnL": f"{trade.gross_pnl:.6f}", "Commission_USD": f"{trade.commission_usd:.6f}",
            "Exit_Reason": trade.exit_reason, "Source": SOURCE_LABEL, **self._meta,
        })

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _writer(self) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if self._new_file:
                writer.writeheader()
            while True:
                row = self._queue.get()
                if row is None:
                    break
                writer.writerow(row)
            handle.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="Tick file (.csv or .zip).")
    parser.add_argument("--format", choices=("csv", "binance-trades"), default="csv")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--price-column", default="price")
    parser.add_argument("--timestamp-unit", choices=("iso", "s", "ms", "us"), default="iso")
    parser.add_argument("--price-is", choices=("bid", "mid"), default=None, help="What the price column is (default: bid for csv, mid for binance-trades).")
    parser.add_argument("--ask-column", default=None, help="csv format: a real ask column. The price column is then the real bid, and no spread is assumed.")
    parser.add_argument("--assumed-spread", type=float, default=None, help="Assumed spread in price units (required unless --ask-column is given). An assumption, not a measurement.")
    parser.add_argument("--size", type=float, default=1.0, help="Units per trade (default 1.0).")
    parser.add_argument("--lot-size", type=float, default=1.0, help="Units in a standard lot for commissions (default 1.0; FX: 100000).")
    parser.add_argument("--latency-ms", type=float, default=5.0)
    parser.add_argument("--commission-per-lot", type=float, default=2.0, help="USD per lot for each fill (default 2.0).")
    parser.add_argument("--quote-to-usd", type=float, default=1.0, help="Rate converting the quote currency to USD (default 1.0).")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--entry-z", type=float, default=3.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("executions.csv"), help="Executions CSV, appended to (default: executions.csv).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.data.exists():
        print(f"data file not found: {args.data}")
        return 2
    real_book = args.ask_column is not None
    if args.format == "binance-trades" and real_book:
        print("--ask-column applies to --format csv only")
        return 2
    if not real_book and args.assumed_spread is None:
        print("--assumed-spread is required unless the file has a real ask column (--ask-column)")
        return 2
    price_is = "bid_ask" if real_book else (args.price_is or ("mid" if args.format == "binance-trades" else "bid"))
    timestamp_unit = "ms" if args.format == "binance-trades" else args.timestamp_unit
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "Run_ID": run_id, "Data_File": args.data.name,
        "Assumed_Spread": "none (real bid/ask)" if real_book else repr(args.assumed_spread), "Price_Is": price_is,
        "Latency_Model_ms": repr(args.latency_ms), "Commission_Per_Lot": repr(args.commission_per_lot),
        "Quote_To_USD": repr(args.quote_to_usd),
    }
    quotes = iter_quotes(
        args.data, fmt=args.format, price_is=price_is, spread=args.assumed_spread or 0.0,
        timestamp_column=args.timestamp_column, price_column=args.price_column, timestamp_unit=timestamp_unit,
        ask_column=args.ask_column,
    )
    logger = TradeLogger(args.output, run_meta=meta)
    try:
        stats = run_backtest(
            quotes,
            ZScoreStrategy(window=args.window, entry_z=args.entry_z, exit_z=args.exit_z),
            FillConfig(args.latency_ms / 1000.0, args.size, args.lot_size, args.commission_per_lot, args.quote_to_usd),
            symbol=args.symbol,
            on_trade=logger.log,
        )
    finally:
        logger.close()

    print(f"{SOURCE_LABEL} run {run_id}: {stats.quotes} quotes, {stats.signals} signals, {stats.trades} closed trades, "
          f"{stats.dropped_orders} order dropped at end of data; net PnL {stats.net_pnl:.2f} (USD only if the quote currency is USD or --quote-to-usd is set).")
    if stats.quotes:
        print(f"alpha step per quote on this machine: mean {stats.alpha_ns_total / stats.quotes / 1000:.1f} us, max {stats.alpha_ns_max / 1000:.1f} us.")
        if stats.fractional_second_quotes / stats.quotes < 0.01:
            print(f"WARNING: timestamps are (nearly) whole seconds, so the {args.latency_ms:g} ms latency cannot be resolved; "
                  "fills happen on the first quote of a later second. See Latency_Slippage_ms for the delay actually applied.")
    spread_note = "spread from the file's real bid/ask" if real_book else f"spread {args.assumed_spread} ({price_is} prices)"
    print(f"assumptions: {spread_note}, latency {args.latency_ms:g} ms, {args.commission_per_lot} USD/lot/fill, "
          f"size {args.size} of lot {args.lot_size}. Appended to {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

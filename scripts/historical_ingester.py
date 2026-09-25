"""Replay a CSV of historical top-of-book ticks onto the Nothierly ZMQ bus.

Each CSV row becomes one :class:`~core.schemas.MarketMetrics` (spread, mid price, and order-book
imbalance / liquidity when volumes are present), wrapped in a :class:`~core.schemas.MessagePayload` and
published on ``MARKET_DATA_L2`` -- the same topic and wire format :class:`~src.agents.market_data.agent.
MarketDataAgent` uses. Like every standalone launcher here, the publisher CONNECTS to the broker
frontend (``ZMQ_BROKER_FRONTEND_ADDRESS``), so a broker must already be running.

What it is not:

- **Top-of-book only.** A bid/ask/volume tick file carries no depth, so this is not an L2 replay.
- **Original timestamps only pace the replay.** Published ticks are stamped with the current time,
  because consumers reject stale ticks against the wall clock (:mod:`core.stale_check`).
- **``sent`` means handed to the PUB socket.** ZMQ PUB/SUB gives no delivery confirmation and drops
  at the high-water mark, so at ``--speed 0`` a slow consumer can miss ticks silently.
- **Which checks fire depends on what is subscribed.** This script only feeds the bus.

CSV columns (case-insensitive; rename with ``--timestamp-column`` / ``--bid-column`` / ``--ask-column`` /
``--bid-volume-column`` / ``--ask-volume-column``): ``bid`` and ``ask``, ``bid_vol`` and ``ask_vol``
(optional), ``symbol`` (optional, else ``--symbol``, which must match the consumer's configured
instrument), and ``timestamp`` (required only when ``--speed`` > 0; ISO-8601 or epoch seconds by default,
or force the unit with ``--timestamp-unit iso|s|ms|us``, e.g. ``ms`` for Binance exports). A ``.zip`` is
read directly (its first member). With a real ask and both quantities, spread, imbalance and liquidity
are the file's own, not assumed.

Bid-only files (e.g. a Dukascopy BID export) have no ask, hence no real spread or mid. Pass
``--assumed-spread``: an explicit operator assumption applied to every tick (ask = bid + spread, mid =
bid + spread / 2), echoed in the run summary. A real ask column always wins over it. Volume columns from
such exports are one-sided and are not used: imbalance is 0.0 and liquidity is 0.0 (the schema's
"unknown"), never a made-up figure.

Usage::

    # Terminal 1 -- the broker (needs ZMQ_BROKER_FRONTEND_ADDRESS/_BACKEND_ADDRESS, see .env):
    python scripts/run_zmq_broker.py

    # Terminal 2 -- whichever agents should consume MARKET_DATA_L2.

    # Terminal 3 -- replay in real time, or as fast as possible with --speed 0:
    python scripts/historical_ingester.py ticks.csv --symbol EUR_USD
    python scripts/historical_ingester.py bid_only.csv --timestamp-column Etc/UTC --bid-column Close \\
        --assumed-spread 0.0001 --speed 60
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import csv
import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, TextIO

import zmq.asyncio

# Must precede any event loop on Windows: core.zmq_bus sets the selector loop policy pyzmq needs.
import core.zmq_bus  # noqa: F401
from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.schemas import MarketMetrics, MessagePayload
from core.zmq_bus import ZmqPublisher

_log = get_logger(__name__)

MARKET_DATA_L2_TOPIC: Final[str] = "MARKET_DATA_L2"
_SENDER_ID: Final[str] = "historical-ingester"
_WARMUP_SECONDS: Final[float] = 1.0
_MAX_RECORDED_REJECTIONS: Final[int] = 5


@dataclass(frozen=True)
class ColumnMap:
    """CSV column names (compared case-insensitively). ``ask=None`` means the file has no ask column."""

    timestamp: str = "timestamp"
    bid: str = "bid"
    ask: str | None = "ask"
    bid_volume: str = "bid_vol"
    ask_volume: str = "ask_vol"
    symbol: str = "symbol"

    def lowered(self) -> ColumnMap:
        return ColumnMap(
            timestamp=self.timestamp.strip().lower(),
            bid=self.bid.strip().lower(),
            ask=self.ask.strip().lower() if self.ask else None,
            bid_volume=self.bid_volume.strip().lower(),
            ask_volume=self.ask_volume.strip().lower(),
            symbol=self.symbol.strip().lower(),
        )


@dataclass(frozen=True)
class Tick:
    symbol: str | None
    timestamp_seconds: float | None
    bid: float
    ask: float
    bid_volume: float | None
    ask_volume: float | None


@dataclass
class ReplayStats:
    sent: int = 0
    rejected: int = 0
    assumed_spread_used: bool = False
    first_rejections: list[str] = field(default_factory=list)

    def reject(self, line_number: int, reason: str) -> None:
        self.rejected += 1
        if len(self.first_rejections) < _MAX_RECORDED_REJECTIONS:
            self.first_rejections.append(f"line {line_number}: {reason}")


_TIME_UNIT_DIVISORS: Final[dict[str, float]] = {"s": 1.0, "ms": 1e3, "us": 1e6}


def _parse_timestamp(raw: str, unit: str = "auto") -> float:
    """Epoch seconds. ``auto`` = a number is seconds, anything else is ISO-8601; or force ``iso``/``s``/``ms``/``us``."""
    if unit in _TIME_UNIT_DIVISORS:
        return float(raw) / _TIME_UNIT_DIVISORS[unit]
    if unit == "auto":
        try:
            return float(raw)
        except ValueError:
            pass
    return datetime.fromisoformat(raw.strip().replace("Z", "+00:00")).timestamp()


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    """The CSV as text; a ``.zip`` is read directly (its first member)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as raw:
                yield io.TextIOWrapper(raw, encoding="utf-8", newline="")
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            yield handle


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    return float(raw) if raw else None


def parse_row(
    row: dict[str, str],
    columns: ColumnMap = ColumnMap(),
    assumed_spread: float | None = None,
    timestamp_unit: str = "auto",
) -> Tick:
    """Parse one CSV row (keys and ``columns`` already lowercased). Raises ``ValueError`` on anything unusable.

    A real ask column wins; ``assumed_spread`` only fills in when the file has no ask column at all.
    """
    raw_timestamp = (row.get(columns.timestamp) or "").strip()
    bid = float(row[columns.bid])
    if columns.ask is not None and columns.ask in row:
        ask = float(row[columns.ask])
    elif assumed_spread is not None:
        ask = bid + assumed_spread
    else:
        raise ValueError("no ask column and no --assumed-spread")
    return Tick(
        symbol=(row.get(columns.symbol) or "").strip() or None,
        timestamp_seconds=_parse_timestamp(raw_timestamp, timestamp_unit) if raw_timestamp else None,
        bid=bid,
        ask=ask,
        bid_volume=_optional_float(row, columns.bid_volume),
        ask_volume=_optional_float(row, columns.ask_volume),
    )


def tick_to_metrics(tick: Tick, *, symbol: str) -> MarketMetrics:
    """Build the bus message body for one tick, stamped with the current time.

    Missing volumes mean a neutral imbalance (0.0) and unknown liquidity (0.0, the schema's own
    "unknown"). A crossed book or a non-positive mid is rejected by ``MarketMetrics`` itself.
    """
    imbalance = 0.0
    liquidity = 0.0
    if tick.bid_volume is not None and tick.ask_volume is not None:
        total = tick.bid_volume + tick.ask_volume
        if total > 0:
            imbalance = (tick.bid_volume - tick.ask_volume) / total
            liquidity = total
    return MarketMetrics(
        symbol=symbol,
        spread=tick.ask - tick.bid,
        order_book_imbalance=imbalance,
        mid_price=(tick.bid + tick.ask) / 2.0,
        top_of_book_liquidity=liquidity,
    )


async def replay_csv(
    path: Path,
    publisher: ZmqPublisher,
    *,
    default_symbol: str,
    speed: float,
    columns: ColumnMap = ColumnMap(),
    assumed_spread: float | None = None,
    timestamp_unit: str = "auto",
) -> ReplayStats:
    """Publish every valid row of ``path``. ``speed`` 0 = no pacing, N > 0 = N x real time."""
    stats = ReplayStats()
    previous_timestamp: float | None = None
    mapping = columns.lowered()
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        header = {name.strip().lower() for name in (reader.fieldnames or [])}
        if mapping.bid not in header:
            raise ValueError(f"{path}: missing required column: {mapping.bid}")
        if (mapping.ask is None or mapping.ask not in header) and assumed_spread is None:
            raise ValueError(
                f"{path}: missing ask column ({mapping.ask or 'none configured'}); "
                "for bid-only data pass --assumed-spread"
            )
        stats.assumed_spread_used = assumed_spread is not None and (mapping.ask is None or mapping.ask not in header)

        for line_number, raw_row in enumerate(reader, start=2):
            row = {key.strip().lower(): value for key, value in raw_row.items() if key}
            try:
                tick = parse_row(row, mapping, assumed_spread, timestamp_unit)
                if speed > 0 and tick.timestamp_seconds is None:
                    raise ValueError("a timestamp is required when --speed > 0")
            except (ValueError, KeyError) as exc:
                stats.reject(line_number, str(exc))
                continue

            if speed > 0 and tick.timestamp_seconds is not None:
                if previous_timestamp is not None:
                    delay = (tick.timestamp_seconds - previous_timestamp) / speed
                    if delay > 0:
                        await asyncio.sleep(delay)
                previous_timestamp = tick.timestamp_seconds

            try:
                metrics = tick_to_metrics(tick, symbol=tick.symbol or default_symbol)
            except ValueError as exc:
                stats.reject(line_number, str(exc).splitlines()[0])
                continue

            await publisher.publish(
                MessagePayload(
                    topic=MARKET_DATA_L2_TOPIC,
                    sender_id=_SENDER_ID,
                    data=metrics.model_dump(mode="json"),
                )
            )
            stats.sent += 1
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="CSV of historical ticks (see module docstring for columns).")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier; 0 = as fast as possible (default: 1.0, real time).")
    parser.add_argument("--symbol", default="EUR_USD", help="Symbol for rows without a symbol column (default: EUR_USD).")
    parser.add_argument("--address", default=None, help="Broker frontend address (default: ZMQ_BROKER_FRONTEND_ADDRESS).")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name (default: timestamp).")
    parser.add_argument("--bid-column", default="bid", help="Bid price column name (default: bid).")
    parser.add_argument("--ask-column", default="ask", help="Ask price column name (default: ask).")
    parser.add_argument("--bid-volume-column", default="bid_vol", help="Best-bid quantity column (default: bid_vol). With both quantities present, imbalance and liquidity are real.")
    parser.add_argument("--ask-volume-column", default="ask_vol", help="Best-ask quantity column (default: ask_vol).")
    parser.add_argument("--timestamp-unit", choices=("auto", "iso", "s", "ms", "us"), default="auto", help="Timestamp format: auto = a number is seconds, else ISO-8601 (default); or force iso, s, ms (e.g. Binance) or us.")
    parser.add_argument("--assumed-spread", type=float, default=None, help="Bid-only files: assumed spread in price units (e.g. 0.0001 = 1 pip on EUR/USD), applied to every tick.")
    parser.add_argument("--high-water-mark", type=int, default=100_000, help="PUB send high-water mark (default: 100000).")
    parser.add_argument("--flush-ms", type=int, default=5000, help="At exit, wait up to this long for queued ticks to flush before discarding the rest (default: 5000).")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    configure_logging(log_level="INFO", log_dir=None, service_name="historical-ingester")
    if not args.csv_path.exists():
        print(f"CSV not found: {args.csv_path}")
        return 2
    address = args.address or get_settings().zmq.broker_frontend_address
    if address is None:
        print("No broker frontend address: set ZMQ_BROKER_FRONTEND_ADDRESS or pass --address, and start scripts/run_zmq_broker.py first.")
        return 2

    columns = ColumnMap(
        timestamp=args.timestamp_column, bid=args.bid_column, ask=args.ask_column,
        bid_volume=args.bid_volume_column, ask_volume=args.ask_volume_column,
    )
    context = zmq.asyncio.Context()
    publisher = ZmqPublisher(address, high_water_mark=args.high_water_mark, context=context, bind=False)
    try:
        await asyncio.sleep(_WARMUP_SECONDS)  # PUB/SUB slow-joiner settle
        try:
            stats = await replay_csv(
                args.csv_path, publisher, default_symbol=args.symbol, speed=args.speed,
                columns=columns, assumed_spread=args.assumed_spread, timestamp_unit=args.timestamp_unit,
            )
        except ValueError as exc:
            print(f"Cannot replay: {exc}")
            return 2
    finally:
        # ZMQ discards queued-but-unsent messages when a socket closes at linger 0 (the right call for live
        # agents and tests, where a stale message is worse than a dropped one). A replay exists to deliver
        # its file, so wait a bounded time for the queue to flush. Not observed to matter for a 500-tick
        # burst to a local broker (500/500 delivered even with --flush-ms 0); a larger burst or a remote
        # broker may differ.
        publisher.close(linger_ms=args.flush_ms)
        await asyncio.to_thread(context.term)

    print(f"published {stats.sent} tick(s) to {address}; rejected {stats.rejected}. "
          "'published' = handed to the PUB socket, delivery is not confirmed.")
    if stats.assumed_spread_used:
        print(f"ASSUMPTION: no real ask in this data; spread fixed at {args.assumed_spread} on every tick "
              "(ask = bid + spread, mid = bid + spread / 2). Imbalance 0.0 and liquidity 0.0 (unknown).")
    for reason in stats.first_rejections:
        print(f"  rejected {reason}")
    return 0 if stats.sent else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run(_parse_args())))

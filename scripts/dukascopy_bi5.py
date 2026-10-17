"""Decode Dukascopy raw tick files (``.bi5``) into a CSV the ingester and the backtest read directly.

The public Dukascopy datafeed stores one LZMA-compressed file per instrument and UTC hour. Each tick is
20 bytes, big-endian: a uint32 millisecond offset within the hour, a uint32 ask, a uint32 bid (both integers
in units of ``1 / price_scale``), then a float32 ask volume and a float32 bid volume. Unlike a CSV exported
from the Dukascopy website, the raw timestamps are not shifted by a second and no tick is dropped.

Validated: on GBP/USD, 2016-10-07 12:00-13:00 UTC, every one of the 27,256 bids of a website export equals the
decoded raw bid, in order (``TESTING.md``). Not validated: the volume columns, and any price scale other than
1e5. Quoting conventions differ by instrument (JPY pairs, metals, indices), so ``--price-scale`` is required
and never guessed; the decoder refuses a tick whose ask is below its bid or whose price is not positive, which
is what a wrong scale tends to produce. Check the first printed price against a known level.

Files are not downloaded here. The server throttles (HTTP 503, connection resets), so fetch them yourself,
politely, and check the site's terms.

Output columns: ``timestamp_ms`` (epoch, UTC), ``time_utc``, ``bid``, ``ask``, ``bid_volume``, ``ask_volume``.
Pass several files as consecutive UTC hours, starting at ``--start``; an empty file (an hour without ticks)
contributes nothing but still counts as an hour.

Usage::

    python scripts/dukascopy_bi5.py 23h_ticks.bi5 --start 2016-10-06T23:00:00Z --price-scale 100000 --output gbpusd.csv

    python scripts/historical_ingester.py gbpusd.csv --symbol GBP_USD --timestamp-column timestamp_ms \\
        --timestamp-unit ms --bid-column bid --ask-column ask --bid-volume-column bid_volume \\
        --ask-volume-column ask_volume --speed 0
    python scripts/zscore_backtest.py gbpusd.csv --symbol GBP_USD --timestamp-column timestamp_ms \\
        --timestamp-unit ms --price-column bid --ask-column ask --size 100000 --lot-size 100000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import lzma
import math
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

_RECORD: Final[struct.Struct] = struct.Struct(">IIIff")
CSV_COLUMNS: Final[tuple[str, ...]] = ("timestamp_ms", "time_utc", "bid", "ask", "bid_volume", "ask_volume")


@dataclass(frozen=True)
class Tick:
    timestamp: datetime
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float


def decimals_for(price_scale: int) -> int:
    """Number of decimals for a power-of-ten ``price_scale`` (1e5 -> 5). Raises ``ValueError`` otherwise."""
    if price_scale < 1 or 10 ** round(math.log10(price_scale)) != price_scale:
        raise ValueError(f"price_scale must be a power of ten (for example 100000), got {price_scale}")
    return round(math.log10(price_scale))


def decode_bi5(data: bytes, hour_start: datetime, price_scale: int) -> Iterator[Tick]:
    """Decode one hourly file. An empty file yields nothing. Raises ``ValueError`` on anything malformed."""
    decimals_for(price_scale)
    if hour_start.tzinfo is None:
        raise ValueError("hour_start must be timezone-aware")
    if not data:
        return
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError as exc:
        raise ValueError(f"not a valid LZMA stream: {exc}") from exc
    if len(raw) % _RECORD.size:
        raise ValueError(f"{len(raw)} decompressed bytes is not a multiple of the {_RECORD.size}-byte record")
    for index in range(len(raw) // _RECORD.size):
        offset_ms, ask, bid, ask_volume, bid_volume = _RECORD.unpack_from(raw, index * _RECORD.size)
        if bid <= 0 or ask < bid:
            raise ValueError(f"tick {index}: bid {bid} and ask {ask} are not a valid quote (wrong --price-scale?)")
        yield Tick(hour_start + timedelta(milliseconds=offset_ms), bid / price_scale, ask / price_scale, bid_volume, ask_volume)


def decode_files(paths: Sequence[Path], start: datetime, price_scale: int) -> Iterator[Tick]:
    """Decode consecutive UTC hours, the first file starting at ``start``."""
    for hour, path in enumerate(paths):
        try:
            yield from decode_bi5(path.read_bytes(), start + timedelta(hours=hour), price_scale)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc


def write_csv(ticks: Iterator[Tick], output: Path, price_scale: int) -> int:
    """Write ticks as CSV and return how many were written."""
    decimals = decimals_for(price_scale)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for tick in ticks:
            writer.writerow([
                round(tick.timestamp.timestamp() * 1000), tick.timestamp.isoformat(timespec="milliseconds"),
                f"{tick.bid:.{decimals}f}", f"{tick.ask:.{decimals}f}", repr(tick.bid_volume), repr(tick.ask_volume),
            ])
            count += 1
    return count


def _parse_start(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("--start must be on a whole UTC hour")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="Hourly .bi5 files, consecutive UTC hours.")
    parser.add_argument("--start", required=True, help="UTC hour of the first file, e.g. 2016-10-06T23:00:00Z.")
    parser.add_argument("--price-scale", type=int, required=True, help="Price units per 1.0, a power of ten (e.g. 100000). Never guessed.")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write.")
    args = parser.parse_args()
    try:
        start = _parse_start(args.start)
        ticks = list(decode_files(args.files, start, args.price_scale))
        written = write_csv(iter(ticks), args.output, args.price_scale)
    except (ValueError, OSError) as exc:
        print(f"Cannot decode: {exc}")
        return 2
    if not ticks:
        print(f"no ticks in {len(args.files)} file(s); wrote only the header to {args.output}")
        return 1
    decimals = decimals_for(args.price_scale)
    print(f"wrote {written:,} ticks to {args.output}: {ticks[0].timestamp.isoformat(timespec='milliseconds')} -> "
          f"{ticks[-1].timestamp.isoformat(timespec='milliseconds')}; first bid/ask {ticks[0].bid:.{decimals}f}/{ticks[0].ask:.{decimals}f} "
          "(check it against a known price: the scale is yours, not guessed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

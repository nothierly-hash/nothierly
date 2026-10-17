"""Rapport de slippage à partir d'une base de fills (``workers/common/fills_db.py``).

Usage ::

    python scripts/fills_report.py hyperliquid_fills.sqlite3 [--venue HYPERLIQUID] [--symbol BTC] [--since-hours 24]

Le slippage est en points de base, positif = coût (voir ``workers.common.fills_db``). Seuls les fills dont l'intention est
connue sont moyennés ; les autres (ordres passés à la main, intention perdue) sont comptés à part. N'écrit aucun fill ; la base
est en WAL, la lire pendant que le worker tourne est sans danger.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.common.fills_db import FillsDatabase, SlippageSummary  # noqa: E402


def _fmt(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:+.2f}{unit}"


def render(title: str, summary: SlippageSummary) -> str:
    return (
        f"{title}\n"
        f"  fills appariés : {summary.fills}   non appariés : {summary.unmatched}   notionnel : {summary.notional:,.2f}\n"
        f"  slippage moyen : {_fmt(summary.mean_bps, ' bps')}   pondéré par le notionnel : {_fmt(summary.qty_weighted_mean_bps, ' bps')}\n"
        f"  médiane : {_fmt(summary.median_bps, ' bps')}   p95 : {_fmt(summary.p95_bps, ' bps')}   pire : {_fmt(summary.worst_bps, ' bps')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rapport de slippage (bps, positif = coût).")
    parser.add_argument("db", help="fichier SQLite des fills")
    parser.add_argument("--venue")
    parser.add_argument("--symbol")
    parser.add_argument("--since-hours", type=float, default=0.0, help="0 = tout l'historique")
    args = parser.parse_args(argv)
    path = Path(args.db)
    if not path.is_file():
        print(f"Base introuvable : {path}", file=sys.stderr)
        return 2
    since = 0 if args.since_hours <= 0 else int(time.time() * 1000 - args.since_hours * 3_600_000)
    try:
        db = FillsDatabase(path)
        print(render("Tous les fills", db.summary(venue=args.venue, symbol=args.symbol, since_ms=since)))
        pairs = {(e.venue, e.symbol) for e in db.executions(venue=args.venue, symbol=args.symbol, since_ms=since, limit=1_000_000)}
        for execution_venue, symbol in sorted(pairs):
            print(render(f"{execution_venue} {symbol}", db.summary(venue=execution_venue, symbol=symbol, since_ms=since)))
        db.close()
    except sqlite3.Error as exc:
        print(f"Base illisible : {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

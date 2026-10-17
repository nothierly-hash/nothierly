"""HTML summary of a simulated-backtest executions file (see ``scripts/zscore_backtest.py``).

Guardrails, because a tear sheet reads like a track record:

- Every row must carry ``Source == SIMULATED_BACKTEST``; anything else is refused. This system has no
  live or paper-trading fills, so there is nothing else this report may honestly describe.
- One run at a time. ``executions.csv`` is appended to across runs; pass ``--run-id`` when the file holds
  more than one, so different data files, symbols and assumptions are never pooled.
- ``--starting-capital`` is required and shown as an illustrative parameter, not a real account.
- Returns are daily PnL divided by the equity at the start of that day (calendar days, zero-PnL days
  included). They are not a percentage change of the PnL series, which has no meaning.
- Sharpe is reported only from at least 30 daily observations; below that it is "n/a".
- The report opens with a banner and lists the assumptions that produced the numbers.

Usage::

    python scripts/generate_tear_sheet.py executions.csv --starting-capital 100000 --output tear_sheet.html
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import html
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

import numpy as np

from core.quant_math import max_drawdown, sharpe_ratio
from scripts.zscore_backtest import CSV_COLUMNS, SOURCE_LABEL

MIN_DAILY_OBSERVATIONS: Final[int] = 30
_BANNER: Final[str] = (
    "SIMULATED BACKTEST - NOT LIVE OR PAPER-TRADING PERFORMANCE. Every figure comes from a historical replay "
    "with the assumptions listed below. The starting capital is an illustrative parameter, not a real account."
)
_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "Data_File", "Symbol", "Assumed_Spread", "Price_Is", "Latency_Model_ms", "Commission_Per_Lot", "Quote_To_USD",
)
_TRADE_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "Timestamp_Entry", "Timestamp_Exit", "Symbol", "Side", "Size", "Entry_Price", "Exit_Price", "Net_PnL", "Latency_Slippage_ms",
)
_TRADE_TABLE_LIMIT: Final[int] = 100


@dataclass(frozen=True)
class Report:
    trades: int
    wins: int
    total_net_pnl: float
    total_return: float
    average_trade: float
    profit_factor: float | None
    max_drawdown: float
    sharpe: float | None
    daily_observations: int
    first_day: date
    last_day: date
    cumulative_pnl: list[float]


def load_executions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in CSV_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s): {', '.join(missing)}")
        return list(reader)


def select_run(rows: list[dict[str, str]], run_id: str | None) -> list[dict[str, str]]:
    """Refuse unlabeled rows, and refuse to pool several runs."""
    if not rows:
        raise ValueError("no executions in the file")
    unlabeled = {row["Source"] for row in rows} - {SOURCE_LABEL}
    if unlabeled:
        raise ValueError(f"refusing rows whose Source is not {SOURCE_LABEL}: {sorted(unlabeled)}")
    runs = sorted({row["Run_ID"] for row in rows})
    if run_id is None:
        if len(runs) > 1:
            raise ValueError(f"the file holds {len(runs)} runs; pass --run-id, one of: {', '.join(runs)}")
        return rows
    selected = [row for row in rows if row["Run_ID"] == run_id]
    if not selected:
        raise ValueError(f"no rows for run {run_id}; runs in the file: {', '.join(runs)}")
    return selected


def compute_report(rows: list[dict[str, str]], starting_capital: float, *, periods_per_year: int = 252) -> Report:
    if starting_capital <= 0:
        raise ValueError("starting capital must be positive")
    trades = sorted(rows, key=lambda row: row["Timestamp_Exit"])
    pnls = [float(row["Net_PnL"]) for row in trades]
    exit_days = [datetime.fromisoformat(row["Timestamp_Exit"]).date() for row in trades]

    equity = starting_capital
    curve = [equity]
    for pnl in pnls:
        equity += pnl
        curve.append(equity)
    if min(curve) <= 0:
        raise ValueError("equity reached zero or below; the starting capital is too small for these trades")

    first_day, last_day = exit_days[0], exit_days[-1]
    pnl_by_day: dict[date, float] = {}
    for day, pnl in zip(exit_days, pnls):
        pnl_by_day[day] = pnl_by_day.get(day, 0.0) + pnl
    daily_returns: list[float] = []
    day_equity = starting_capital
    day = first_day
    while day <= last_day:
        day_pnl = pnl_by_day.get(day, 0.0)
        daily_returns.append(day_pnl / day_equity)
        day_equity += day_pnl
        day += timedelta(days=1)

    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    sharpe: float | None = None
    if len(daily_returns) >= MIN_DAILY_OBSERVATIONS:
        try:
            sharpe = sharpe_ratio(np.array(daily_returns, dtype=np.float64), periods_per_year=periods_per_year)
        except ValueError:
            sharpe = None
    total = sum(pnls)
    running = np.cumsum(pnls).tolist()
    return Report(
        trades=len(pnls), wins=len(wins), total_net_pnl=total, total_return=total / starting_capital,
        average_trade=total / len(pnls), profit_factor=(sum(wins) / -sum(losses)) if losses else None,
        max_drawdown=max_drawdown(np.array(curve, dtype=np.float64)), sharpe=sharpe,
        daily_observations=len(daily_returns), first_day=first_day, last_day=last_day, cumulative_pnl=running,
    )


def _svg_curve(values: list[float]) -> str:
    if len(values) < 2:
        return "<p>Fewer than 2 trades: no curve.</p>"
    width, height, pad = 720, 220, 24
    low, high = min(0.0, min(values)), max(0.0, max(values))
    span = (high - low) or 1.0
    points = " ".join(
        f"{pad + i * (width - 2 * pad) / (len(values) - 1):.1f},{height - pad - (v - low) / span * (height - 2 * pad):.1f}"
        for i, v in enumerate(values)
    )
    zero_y = height - pad - (0.0 - low) / span * (height - 2 * pad)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Cumulative net PnL by trade">'
        f'<line x1="{pad}" x2="{width - pad}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#999" stroke-dasharray="4"/>'
        f'<polyline fill="none" stroke="#1f5fbf" stroke-width="1.5" points="{points}"/></svg>'
    )


def render_html(rows: list[dict[str, str]], report: Report, starting_capital: float) -> str:
    esc = html.escape
    provenance = "".join(
        f"<tr><th>{esc(column)}</th><td>{esc(', '.join(sorted({row[column] for row in rows})))}</td></tr>"
        for column in _PROVENANCE_COLUMNS
    )
    sharpe = f"{report.sharpe:.2f}" if report.sharpe is not None else f"n/a (fewer than {MIN_DAILY_OBSERVATIONS} daily observations)"
    factor = f"{report.profit_factor:.2f}" if report.profit_factor is not None else "n/a (no losing trades)"
    metrics = [
        ("Closed trades", str(report.trades)), ("Win rate", f"{report.wins / report.trades:.1%}"),
        ("Total net PnL", f"{report.total_net_pnl:,.2f}"), ("Return on starting capital", f"{report.total_return:.4%}"),
        ("Average trade", f"{report.average_trade:,.2f}"), ("Profit factor", factor),
        ("Max drawdown (trade-level equity)", f"{report.max_drawdown:.4%}"),
        ("Sharpe (annualized, daily returns)", sharpe),
        ("Calendar days spanned", f"{report.daily_observations} ({report.first_day} to {report.last_day})"),
    ]
    metric_rows = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in metrics)
    shown = sorted(rows, key=lambda row: row["Timestamp_Exit"])[:_TRADE_TABLE_LIMIT]
    head = "".join(f"<th>{esc(c)}</th>" for c in _TRADE_TABLE_COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{esc(row[c])}</td>" for c in _TRADE_TABLE_COLUMNS) + "</tr>" for row in shown)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Simulated backtest summary</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:880px;margin:24px auto;padding:0 16px;color:#111}}
.banner{{background:#b00020;color:#fff;padding:12px 16px;font-weight:600;border-radius:4px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:13px}}
th{{background:#f3f3f3}}</style></head><body>
<div class="banner">{esc(_BANNER)}</div>
<h1>Simulated backtest summary</h1>
<p>Starting capital (illustrative): {starting_capital:,.2f}. Net PnL is in USD only if the quote currency is USD or a conversion rate was supplied (Quote_To_USD below).</p>
<h2>Assumptions and data</h2><table>{provenance}</table>
<h2>Metrics</h2><table>{metric_rows}</table>
<h2>Cumulative net PnL by trade</h2>{_svg_curve(report.cumulative_pnl)}
<h2>Trades (first {min(len(rows), _TRADE_TABLE_LIMIT)} of {len(rows)})</h2><table><tr>{head}</tr>{body}</table>
<p>A toy strategy replayed over a few historical sessions, with an assumed spread and commission. It is not evidence of an edge.</p>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("executions", type=Path)
    parser.add_argument("--starting-capital", type=float, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--periods-per-year", type=int, default=252)
    parser.add_argument("--output", type=Path, default=Path("tear_sheet.html"))
    args = parser.parse_args()
    try:
        rows = select_run(load_executions(args.executions), args.run_id)
        report = compute_report(rows, args.starting_capital, periods_per_year=args.periods_per_year)
    except ValueError as exc:
        print(f"Cannot build the report: {exc}")
        return 2
    args.output.write_text(render_html(rows, report, args.starting_capital), encoding="utf-8")
    print(f"wrote {args.output}: {report.trades} trades, total net PnL {report.total_net_pnl:,.2f}; "
          f"Sharpe {'n/a' if report.sharpe is None else f'{report.sharpe:.2f}'} ({report.daily_observations} daily observations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

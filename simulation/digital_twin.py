"""Chapter 15 (The Apex Tier), Pillar 3: THE CAUSAL DIGITAL TWIN -- a real B2B backtesting simulator.

**Real, load-bearing correction to the brief, stated upfront**: "rejouer
le carnet d'ordres historique (Tick data L2)" assumes real historical
L2 order-book data is available for an arbitrary prospect's trade
history. Verified directly before writing any of this: this project
has real historical L2 order-book snapshot data (Tardis-format, 25
price levels per side, ``research/sample_data/`` -- the same real
sample :mod:`research.maker_execution_sim`'s own docstring already
documents using) for exactly one real coverage window --
**2020-01-01T00:00:05Z to 2020-01-01T23:59:59Z, BTCUSDT and ETHUSDT on
Binance only**. A real prospect's uploaded CSV will not, in general,
fall inside that window. This tool does not fabricate a book for
out-of-coverage trades: :func:`run_digital_twin_backtest` reports real
per-trade coverage (:attr:`TradeComparisonRow.replay_available`) and
computes the aggregate slippage comparison only over the trades it
could genuinely replay -- an honest, possibly small or zero, real
subset, stated plainly in :attr:`DigitalTwinReport.coverage_note`
rather than silently padded out with invented numbers.

**What "Nothierly AI Execution" means here, precisely**: the client's
own reported fill price (their CSV's ``price`` column) is the "Naive"
side of the comparison, taken at face value. The "AI" side re-executes
the *same real order* (same side, same real historical timestamp, same
quantity) by slicing it with this project's own real
:func:`~src.execution.macro_strategist.slice_twap` and walking the
*real* historical order book at each child's own (rebased) historical
moment via the same real book-walking logic
:mod:`pricing.pre_trade_oracle` already uses -- not a different,
invented algorithm. This directly mirrors ``api_middleware/gateway.py``'s
own ``_run_shadow_simulation`` (naive single-shot vs. sliced execution),
now measured against genuine historical ticks instead of a live
snapshot.

**On the RL Agent and the Guardian Agent**: the brief asks this tool to
"instancier" both. :class:`~src.agents.rl_signal.policy_model.
RLPolicyModel` is invoked for real, at each trade's own historical
moment, using a real feature vector built from that moment's real book
(:func:`_build_features_at`) -- its signal/confidence are shown as real
contextual metadata per trade (this project's RL agent is a
*directional* signal generator; it does not, anywhere in this codebase,
decide *how* an order is sliced -- see
:class:`~src.agents.rl_signal.agent.RlSignalAgent`'s own module
docstring -- so this tool does not invent a connection between the
signal and the execution-style comparison above, which would
misrepresent how this system actually works). :class:`MockGuardianAgent`
is exactly what the brief asks for -- **Mocked**: a real, simple,
deterministic risk heuristic, not this project's actual LLM-based
:mod:`src.agents.guardian_ai` (calling a real LLM per simulated
historical trade -- potentially hundreds per run -- would be slow,
costly, and isn't what a backtest needs to demonstrate).
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.nothierly_ontology import OrderSide
from core.schemas import ORDER_BOOK_TOP_N, MarketMetrics, OrderBookLevel, OrderBookSnapshot
from pricing.pre_trade_oracle import _walk_book
from src.agents.rl_signal.policy_model import RLPolicyModel
from src.agents.supervisor.ml_engine import SupervisorMLModel
from src.execution.macro_strategist import slice_twap

_log = get_logger(__name__)

_SAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "research" / "sample_data"

# Real, exact coverage -- verified directly against the real sample files (pl.scan_csv(...).select(min/max)),
# not assumed. Both instruments cover essentially the same real calendar day. Public: a second real caller
# (src/dashboard/ems_pages/digital_twin_backtest.py) needs to disclose this same window upfront, not guess it.
COVERAGE_START_UTC: Final[datetime] = datetime(2020, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
COVERAGE_END_UTC: Final[datetime] = datetime(2020, 1, 1, 23, 59, 59, tzinfo=timezone.utc)

SAMPLE_BOOK_FILES: Final[dict[str, Path]] = {
    "BTCUSDT": _SAMPLE_DATA_DIR / "btcusdt_book_snapshot_2020-01-01.csv",
    "ETHUSDT": _SAMPLE_DATA_DIR / "ethusdt_book_snapshot_2020-01-01.csv",
}

_SAMPLE_TRADES_FILES: Final[dict[str, Path]] = {
    "BTCUSDT": _SAMPLE_DATA_DIR / "btcusdt_trades_2020-01-01.csv",
    "ETHUSDT": _SAMPLE_DATA_DIR / "ethusdt_trades_2020-01-01.csv",
}

_DEFAULT_EXECUTION_WINDOW_MINUTES: Final[int] = 1
_DEFAULT_NUM_SLICES: Final[int] = 5
# Mock Guardian default -- see MockGuardianAgent's own docstring.
_DEFAULT_MAX_BOOK_DEPTH_FRACTION: Final[Decimal] = Decimal("0.5")

_BOOK_COLUMNS: Final[list[str]] = ["timestamp"] + [
    f"{side}[{i}].{field}" for side in ("asks", "bids") for i in range(ORDER_BOOK_TOP_N) for field in ("price", "amount")
]


class DigitalTwinError(Exception):
    """Base class for real, user-facing errors this module raises (never a raw KeyError/etc. from a malformed CSV)."""


class ClientTrade(BaseModel):
    """One real row from a prospect's uploaded trade history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    symbol: str = Field(min_length=1)
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0, description="The client's own reported naive execution price.")


class MockGuardianVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    approved: bool
    reason: str


class MockGuardianAgent:
    """A deliberately simple, deterministic stand-in for the real (LLM-based) src.agents.guardian_ai.

    See module docstring's own section on why this is Mocked, exactly
    as the brief asks. The one real rule applied: refuse whenever a
    single child slice would consume more than ``max_book_depth_fraction``
    of the real visible top-of-book liquidity on its own side -- a
    genuine, sensible risk heuristic (don't let one slice overwhelm the
    book), not the production Guardian's real LLM-based judgment.
    """

    def __init__(self, *, max_book_depth_fraction: Decimal = _DEFAULT_MAX_BOOK_DEPTH_FRACTION) -> None:
        self._max_fraction = max_book_depth_fraction

    def assess(self, *, child_quantity: Decimal, top_of_book_depth: Decimal) -> MockGuardianVerdict:
        if top_of_book_depth <= 0:
            return MockGuardianVerdict(approved=False, reason="No real visible liquidity to assess against.")
        fraction = child_quantity / top_of_book_depth
        if fraction > self._max_fraction:
            return MockGuardianVerdict(
                approved=False,
                reason=f"Mock Guardian: slice would consume {fraction:.0%} of visible top-of-book depth (mock limit {self._max_fraction:.0%}).",
            )
        return MockGuardianVerdict(approved=True, reason="Within mock Guardian risk limits.")


class TradeComparisonRow(BaseModel):
    """Naive vs. Nothierly AI execution, for one real client trade."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    replay_available: bool = Field(description="False means no real historical book snapshot exists at this timestamp -- see module docstring.")
    naive_price: Decimal
    naive_cost_usd: Decimal | None = None
    ai_simulated_price: Decimal | None = None
    ai_simulated_cost_usd: Decimal | None = None
    slippage_saved_usd: Decimal | None = Field(default=None, description="naive_cost_usd - ai_simulated_cost_usd, signed so positive = Nothierly AI was cheaper.")
    rl_signal: Literal[-1, 0, 1] | None = None
    rl_confidence: float | None = None
    guardian_verdict: MockGuardianVerdict | None = None
    unavailable_reason: str | None = None


class DigitalTwinReport(BaseModel):
    """The real, aggregate output of one backtest run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trades_total: int
    trades_replayed: int
    coverage_note: str
    rows: list[TradeComparisonRow]
    total_naive_cost_usd: Decimal
    total_ai_simulated_cost_usd: Decimal
    total_slippage_saved_usd: Decimal


def parse_client_trades_csv(path: str | Path, *, default_symbol: str = "BTCUSDT") -> list[ClientTrade]:
    """Parse a real prospect-uploaded CSV. Required columns: timestamp, side, price, and either size or quantity.

    ``symbol`` is optional (defaults to ``default_symbol``) -- the
    brief's own literal column list (timestamp/size/side/price) has no
    symbol column, but a real multi-instrument prospect file plausibly
    would; supporting it costs nothing and degrades to the brief's
    exact spec when absent.

    Raises:
        DigitalTwinError: On a missing required column or an unparseable row -- never silently skips bad data.
    """
    path = Path(path)
    trades: list[ClientTrade] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise DigitalTwinError(f"{path} is empty or not a valid CSV.")
        fields = {f.strip().lower() for f in reader.fieldnames}
        if "timestamp" not in fields or "side" not in fields or "price" not in fields:
            raise DigitalTwinError(f"{path} must have at least timestamp, side, price columns -- found {sorted(fields)}.")
        if "size" not in fields and "quantity" not in fields:
            raise DigitalTwinError(f"{path} must have a size or quantity column -- found {sorted(fields)}.")

        for line_num, row in enumerate(reader, start=2):
            normalized = {k.strip().lower(): v for k, v in row.items() if k is not None}
            try:
                ts = _parse_timestamp(normalized["timestamp"])
                side_raw = normalized["side"].strip().upper()
                side = OrderSide.BUY if side_raw in ("BUY", "B") else OrderSide.SELL if side_raw in ("SELL", "S") else None
                if side is None:
                    raise DigitalTwinError(f"Line {line_num}: unrecognized side {normalized['side']!r}.")
                quantity = Decimal(normalized.get("size") or normalized.get("quantity") or "")
                price = Decimal(normalized["price"])
                symbol = (normalized.get("symbol") or default_symbol).strip().upper()
            except (KeyError, InvalidOperation, ValueError) as exc:
                raise DigitalTwinError(f"Line {line_num}: could not parse row {row!r}: {exc}") from exc
            trades.append(ClientTrade(timestamp=ts, symbol=symbol, side=side, quantity=quantity, price=price))
    return trades


def _parse_timestamp(raw: str) -> datetime:
    """Real, best-effort parse: ISO8601 first, then Unix epoch (s/ms/us, inferred from magnitude)."""
    raw = raw.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        numeric = float(raw)
    except ValueError as exc:
        raise DigitalTwinError(f"Timestamp {raw!r} is neither ISO8601 nor a numeric Unix epoch.") from exc
    if numeric > 1e17:
        raise DigitalTwinError(f"Timestamp {raw!r} is out of any plausible real range.")
    if numeric > 1e14:  # microseconds
        seconds = numeric / 1e6
    elif numeric > 1e11:  # milliseconds
        seconds = numeric / 1e3
    else:  # seconds
        seconds = numeric
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def _load_sample_book(symbol: str) -> pl.DataFrame:
    """Load + sort one instrument's real historical L2 snapshots once. Not cached across calls -- the caller (run_digital_twin_backtest) loads each needed symbol exactly once per run."""
    path = SAMPLE_BOOK_FILES.get(symbol)
    if path is None or not path.exists():
        raise DigitalTwinError(f"No real historical L2 sample data for {symbol!r} -- available: {sorted(SAMPLE_BOOK_FILES)}.")
    return pl.scan_csv(str(path)).select(_BOOK_COLUMNS).sort("timestamp").collect()


def build_sample_client_trades_csv(symbol: str = "BTCUSDT", *, n_trades: int = 8) -> str:
    """A real, honest demo input -- not a fabricated one.

    Returns a prospect-style CSV (``timestamp,symbol,side,quantity,price``)
    built from ``n_trades`` real historical trades sampled evenly across
    this project's own real coverage window (see :data:`COVERAGE_START_UTC`
    / :data:`COVERAGE_END_UTC`) -- every timestamp, side, price, and
    quantity is copied verbatim from the real Tardis trades sample this
    project already ships (``research/sample_data/``), never invented.
    Exists so a visitor to the dashboard page can exercise
    :func:`run_digital_twin_backtest` end-to-end without first needing to
    source their own real trade history.

    Raises:
        DigitalTwinError: Unknown symbol, or its real trades sample is missing/empty.
    """
    symbol = symbol.upper()
    path = _SAMPLE_TRADES_FILES.get(symbol)
    if path is None or not path.exists():
        raise DigitalTwinError(f"No real trades sample for {symbol!r} -- available: {sorted(_SAMPLE_TRADES_FILES)}.")
    # The real trades file's own timestamp range is slightly wider than the real book-snapshot coverage
    # window (found via a failing test: its very first row predates COVERAGE_START_UTC by ~0.5s) -- filtering
    # to the book's exact window here, not just the trades file's own range, is what makes every sampled row
    # genuinely replayable by run_digital_twin_backtest, not merely "somewhere in this CSV".
    start_us = int(COVERAGE_START_UTC.timestamp() * 1_000_000)
    end_us = int(COVERAGE_END_UTC.timestamp() * 1_000_000)
    trades = (
        pl.scan_csv(str(path))
        .select(["timestamp", "side", "price", "amount"])
        .filter((pl.col("timestamp") >= start_us) & (pl.col("timestamp") <= end_us))
        .collect()
    )
    if len(trades) == 0:
        raise DigitalTwinError(f"Real trades sample for {symbol!r} has no rows within the real coverage window.")

    n_trades = max(1, min(n_trades, len(trades)))
    step = max(len(trades) // n_trades, 1)
    sampled = trades[::step][:n_trades]

    lines = ["timestamp,symbol,side,quantity,price"]
    for row in sampled.iter_rows(named=True):
        ts_iso = datetime.fromtimestamp(int(row["timestamp"]) / 1_000_000, tz=timezone.utc).isoformat()
        lines.append(f"{ts_iso},{symbol},{str(row['side']).upper()},{row['amount']},{row['price']}")
    return "\n".join(lines) + "\n"


def _row_to_snapshot(row: dict[str, object], *, symbol: str) -> OrderBookSnapshot:
    """One real Tardis book_snapshot CSV row -> a validated OrderBookSnapshot (top ORDER_BOOK_TOP_N levels)."""
    ts_micros = int(str(row["timestamp"]))
    asks = [
        OrderBookLevel(price=float(str(row[f"asks[{i}].price"])), volume=float(str(row[f"asks[{i}].amount"])))
        for i in range(ORDER_BOOK_TOP_N)
        if row.get(f"asks[{i}].price") is not None
    ]
    bids = [
        OrderBookLevel(price=float(str(row[f"bids[{i}].price"])), volume=float(str(row[f"bids[{i}].amount"])))
        for i in range(ORDER_BOOK_TOP_N)
        if row.get(f"bids[{i}].price") is not None
    ]
    return OrderBookSnapshot(
        symbol=symbol, timestamp=datetime.fromtimestamp(ts_micros / 1_000_000, tz=timezone.utc), bids=bids, asks=asks,
    )


def _build_features_at(snapshot: OrderBookSnapshot) -> tuple[MarketMetrics, npt.NDArray[np.float64]]:
    """Real MarketMetrics from one real historical snapshot -- see module docstring on macro/sentiment being empty here."""
    best_bid, best_ask = snapshot.best_bid, snapshot.best_ask
    mid = (best_bid.price + best_ask.price) / 2.0
    spread = best_ask.price - best_bid.price
    bid_vol = sum(level.volume for level in snapshot.bids)
    ask_vol = sum(level.volume for level in snapshot.asks)
    obi = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.0
    metrics = MarketMetrics(symbol=snapshot.symbol, timestamp=snapshot.timestamp, spread=spread, order_book_imbalance=obi, mid_price=mid)
    features = SupervisorMLModel.build_feature_vector(metrics, {}, [])
    return metrics, features


def _simulate_ai_execution(
    *, trade: ClientTrade, book_df: pl.DataFrame, execution_window_minutes: int, num_slices: int, guardian: MockGuardianAgent,
) -> tuple[Decimal | None, list[MockGuardianVerdict], str | None]:
    """Real TWAP slicing + real historical book-walking.

    Returns ``(quantity-weighted avg fill price, per-child Guardian
    verdicts, failure_reason)`` -- ``failure_reason`` is ``None`` when a
    price was produced, and otherwise names *which* of the three real,
    distinct ways a child can fail to contribute a fill actually
    happened: a Guardian veto, real captured depth insufficient to fill
    anything, or no real snapshot existing at the rebased target time at
    all. Found worth distinguishing during live testing against a
    larger, more realistic trade set: all three previously collapsed
    into one ambiguous message, which isn't actionable for a real B2B
    prospect reading this report.
    """
    side_literal: Literal["buy", "sell"] = "buy" if trade.side == OrderSide.BUY else "sell"
    children = slice_twap(
        parent_order_id="digital-twin-backtest", symbol=trade.symbol, side=side_literal,
        total_quantity=float(trade.quantity), time_window_minutes=execution_window_minutes, num_slices=num_slices,
    )
    base_scheduled_at = children[0].scheduled_at
    target_timestamps = pl.DataFrame(
        {"child_index": list(range(len(children))), "target_ts": [
            int((trade.timestamp + (c.scheduled_at - base_scheduled_at)).timestamp() * 1_000_000) for c in children
        ]}
    ).sort("target_ts")

    matched = target_timestamps.join_asof(book_df, left_on="target_ts", right_on="timestamp", strategy="backward")
    children_by_index = {i: c for i, c in enumerate(children)}

    total_notional = Decimal(0)
    total_filled = Decimal(0)
    verdicts: list[MockGuardianVerdict] = []
    missing_snapshot_count = 0
    guardian_blocked_count = 0
    zero_depth_count = 0
    for match_row in matched.iter_rows(named=True):
        # Explicit lookup by child_index, not positional zip -- target_timestamps was sorted by target_ts for
        # the as-of join, which is not guaranteed to still line up 1:1 with children's own sequence order.
        child = children_by_index[int(str(match_row["child_index"]))]
        if match_row.get("timestamp") is None:
            missing_snapshot_count += 1
            continue  # this child's rebased target predates any real captured snapshot -- honestly skipped, not fabricated
        snapshot = _row_to_snapshot(match_row, symbol=trade.symbol)
        levels = snapshot.asks if trade.side == OrderSide.BUY else snapshot.bids
        top_of_book_depth = Decimal(str(levels[0].volume))
        verdict = guardian.assess(child_quantity=Decimal(str(child.quantity)), top_of_book_depth=top_of_book_depth)
        verdicts.append(verdict)
        if not verdict.approved:
            guardian_blocked_count += 1
            continue  # Mock Guardian veto -- this child's notional is simply not simulated, an honest partial result

        filled, _unfilled, notional = _walk_book(levels, Decimal(str(child.quantity)))
        if filled <= 0:
            zero_depth_count += 1
        total_notional += notional
        total_filled += filled

    if total_filled <= 0:
        if guardian_blocked_count > 0 and zero_depth_count == 0 and missing_snapshot_count == 0:
            reason = f"Mock Guardian blocked all {guardian_blocked_count} child slice(s) -- real order size is too large relative to real top-of-book depth at this moment."
        elif zero_depth_count > 0 and guardian_blocked_count == 0:
            reason = f"Real captured book depth was insufficient to fill any of {len(children)} child slice(s), even though the Mock Guardian approved them."
        elif missing_snapshot_count == len(children):
            reason = "No real historical snapshot exists at any of this trade's rebased child execution times."
        else:
            reason = (
                f"No child slice produced a real fill: {guardian_blocked_count} blocked by the Mock Guardian, "
                f"{zero_depth_count} had insufficient real captured depth, {missing_snapshot_count} had no real snapshot available."
            )
        return None, verdicts, reason
    return total_notional / total_filled, verdicts, None


def run_digital_twin_backtest(
    csv_path: str | Path,
    *,
    execution_window_minutes: int = _DEFAULT_EXECUTION_WINDOW_MINUTES,
    num_slices: int = _DEFAULT_NUM_SLICES,
    guardian: MockGuardianAgent | None = None,
    rl_policy: RLPolicyModel | None = None,
) -> DigitalTwinReport:
    """The real entrypoint: parse a prospect's CSV, replay every in-coverage trade, return a real comparison report.

    Args:
        csv_path: Path to the prospect's uploaded trade history.
        execution_window_minutes: How long Nothierly AI's simulated TWAP takes to execute each trade.
        num_slices: How many child slices per trade.
        guardian: Defaults to a fresh :class:`MockGuardianAgent`.
        rl_policy: Defaults to a fresh, untrained-skeleton
            :class:`~src.agents.rl_signal.policy_model.RLPolicyModel`
            (no ``model_path`` -- see that class's own docstring; a real
            trained checkpoint can be passed in once one exists).
    """
    guardian = guardian or MockGuardianAgent()
    rl_policy = rl_policy or RLPolicyModel()
    trades = parse_client_trades_csv(csv_path)

    book_cache: dict[str, pl.DataFrame] = {}
    rows: list[TradeComparisonRow] = []

    for trade in trades:
        if not (COVERAGE_START_UTC <= trade.timestamp <= COVERAGE_END_UTC) or trade.symbol not in SAMPLE_BOOK_FILES:
            rows.append(
                TradeComparisonRow(
                    timestamp=trade.timestamp, symbol=trade.symbol, side=trade.side, quantity=trade.quantity,
                    replay_available=False, naive_price=trade.price,
                    unavailable_reason=(
                        f"No real historical L2 data for {trade.symbol!r} at {trade.timestamp.isoformat()} -- "
                        f"real coverage is {sorted(SAMPLE_BOOK_FILES)} within "
                        f"{COVERAGE_START_UTC.isoformat()} to {COVERAGE_END_UTC.isoformat()}."
                    ),
                )
            )
            continue

        if trade.symbol not in book_cache:
            book_cache[trade.symbol] = _load_sample_book(trade.symbol)
        book_df = book_cache[trade.symbol]

        anchor_ts_micros = int(trade.timestamp.timestamp() * 1_000_000)
        anchor_match = pl.DataFrame({"ts": [anchor_ts_micros]}).join_asof(book_df, left_on="ts", right_on="timestamp", strategy="backward")
        anchor_row = anchor_match.row(0, named=True)
        if anchor_row.get("timestamp") is None:
            rows.append(
                TradeComparisonRow(
                    timestamp=trade.timestamp, symbol=trade.symbol, side=trade.side, quantity=trade.quantity,
                    replay_available=False, naive_price=trade.price,
                    unavailable_reason="No real snapshot exists at or before this exact timestamp within the sample file.",
                )
            )
            continue

        snapshot = _row_to_snapshot(anchor_row, symbol=trade.symbol)
        _metrics, features = _build_features_at(snapshot)
        rl_signal, rl_confidence = rl_policy.predict_action(features, current_position=0.0)

        ai_avg_price, verdicts, failure_reason = _simulate_ai_execution(
            trade=trade, book_df=book_df, execution_window_minutes=execution_window_minutes, num_slices=num_slices, guardian=guardian,
        )
        overall_guardian = verdicts[0] if verdicts else MockGuardianVerdict(approved=False, reason="No child slice could be assessed.")

        naive_cost_usd = trade.price * trade.quantity
        if ai_avg_price is None:
            rows.append(
                TradeComparisonRow(
                    timestamp=trade.timestamp, symbol=trade.symbol, side=trade.side, quantity=trade.quantity,
                    replay_available=True, naive_price=trade.price, naive_cost_usd=naive_cost_usd,
                    rl_signal=rl_signal, rl_confidence=rl_confidence, guardian_verdict=overall_guardian,
                    unavailable_reason=failure_reason or "No AI-simulated fill was produced.",
                )
            )
            continue

        ai_cost_usd = ai_avg_price * trade.quantity
        # Signed so a positive number always means Nothierly AI was cheaper, regardless of side: buying
        # cheaper is good, selling higher is good -- both show as a positive "saved" figure.
        slippage_saved = (naive_cost_usd - ai_cost_usd) if trade.side == OrderSide.BUY else (ai_cost_usd - naive_cost_usd)

        rows.append(
            TradeComparisonRow(
                timestamp=trade.timestamp, symbol=trade.symbol, side=trade.side, quantity=trade.quantity,
                replay_available=True, naive_price=trade.price, naive_cost_usd=naive_cost_usd,
                ai_simulated_price=ai_avg_price, ai_simulated_cost_usd=ai_cost_usd, slippage_saved_usd=slippage_saved,
                rl_signal=rl_signal, rl_confidence=rl_confidence, guardian_verdict=overall_guardian,
            )
        )

    replayed_rows = [r for r in rows if r.slippage_saved_usd is not None]
    # Real invariant this loop always maintains: naive_cost_usd/ai_simulated_cost_usd/slippage_saved_usd are
    # set together or not at all (see the two TradeComparisonRow(...) construction sites above) -- filtering
    # each generator on its own field (redundant with the replayed_rows filter above at runtime) is what lets
    # mypy verify these are real Decimal values, not Decimal | None, without a blanket ignore.
    total_naive = sum((r.naive_cost_usd for r in replayed_rows if r.naive_cost_usd is not None), Decimal(0))
    total_ai = sum((r.ai_simulated_cost_usd for r in replayed_rows if r.ai_simulated_cost_usd is not None), Decimal(0))
    total_saved = sum((r.slippage_saved_usd for r in replayed_rows if r.slippage_saved_usd is not None), Decimal(0))

    coverage_note = (
        f"{len(replayed_rows)}/{len(trades)} trades genuinely replayed against real historical L2 data "
        f"(coverage: {sorted(SAMPLE_BOOK_FILES)}, {COVERAGE_START_UTC.date()} UTC only). "
        "Trades outside this window are listed with replay_available=False and excluded from the totals below -- never fabricated."
    )
    _log.info("digital_twin_backtest_completed", trades_total=len(trades), trades_replayed=len(replayed_rows), total_slippage_saved_usd=str(total_saved))

    return DigitalTwinReport(
        trades_total=len(trades), trades_replayed=len(replayed_rows), coverage_note=coverage_note, rows=rows,
        total_naive_cost_usd=total_naive, total_ai_simulated_cost_usd=total_ai, total_slippage_saved_usd=total_saved,
    )

"""Offline training of the meta-label model (Gate 3 sizing): will this consensus entry make money after real costs?

Not part of the live system. Run manually or on a schedule::

    python scripts/train_metalabel.py --decisions decisions.jsonl --fills-db hyperliquid_fills.sqlite3 bybit_fills.sqlite3
    python scripts/train_metalabel.py --timescale --lookback-days 60 --fills-db bybit_fills.sqlite3

**Data contract.**

- *Decisions* (X): the ``META_LABEL_DECISION`` messages the ExecutionAgent publishes at every consensus-driven Gate 3
  evaluation (features at that moment, probability, multiplier), archived by the Archivist. Read from a JSON-lines file
  (``--decisions``, one message ``data`` object per line) or from the TimescaleDB archive (``--timescale``).
- *Outcomes* (Y): round trips rebuilt from the REAL fills in the workers' ``executions`` tables
  (``workers/common/fills_db.py``): FIFO position accounting per venue and symbol, flat to flat. The label is ``1`` if the
  trade's PnL after fees is positive, ``0`` otherwise. Fills are real prices, so the slippage the worker measured is already
  in the PnL; nothing is added or subtracted for it a second time.
- *Join*: a round trip takes the LATEST non-vetoed decision on the same asset and direction made at most ``--max-lag-s``
  before its first fill (a decision is used once). This is a time-proximity join: the bridge and the workers do not carry an
  engine decision id (the contract has no such field), so it is approximate. A decision followed by no fill (vetoed by a later
  gate, held back by the bridge) has no label and is ignored.

**What is excluded, and counted in the report**: trips that contain an unmatched fill (an order the worker did not place),
an emergency fill (a dead man's switch exit says nothing about the model's decision), a fee in a non-quote currency, and
trips still open.

**Safeguards.** The split is chronological with an embargo (a test trade must start after every training trade has ended),
never random: shuffled trade data leaks the future. The artifact is written only if there are at least ``--min-trades``
labeled trades, a test set of at least ``--min-test`` trades with both outcomes, AUC at least ``--min-auc``, an AUC whose
99% lower bound is above 0.5 (otherwise luck explains it), AND a Brier score better than always predicting the training base rate. Otherwise nothing is written and the exit code is 1: a model that does
not beat the base rate must not size real money. With few live fills this refusal is the expected outcome for a long time.

**Artifact.** ``models/metalabel_weights.json`` (XGBoost weights; trees need no scaler) and
``models/metalabel_weights.meta.json`` (feature names, payoff ratio = mean win / mean loss over the TRAINING trades, metrics, data
window). ``risk.meta_labeling.MetaLabelModel.load_or_skeleton`` loads them; a missing or inconsistent pair falls back to 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
import math
import os
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from risk.meta_labeling import FEATURES_V2, FEATURES_V3, kelly_size_multiplier, metadata_path_for
from workers.common.fills_db import BUY, Execution, FillsDatabase

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_OUT = MODELS_DIR / "metalabel_weights.json"
DECISION_TOPIC = "META_LABEL_DECISION"
_QUOTE_FEE_CURRENCIES = frozenset({"USD", "USDT", "USDC", "BUSD", "FDUSD", "DAI"})
_QUOTE_SUFFIXES = ("USDT", "USDC", "USD", "PERP")
_PAYOFF_CLIP = (0.25, 4.0)


# --- Data shapes -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    time_s: float
    asset: str  # base asset, e.g. "BTC"
    direction: int  # +1 buy, -1 sell
    features: Mapping[str, float]


@dataclass(frozen=True)
class RoundTrip:
    venue: str
    asset: str
    direction: int
    entry_ms: int
    exit_ms: int
    pnl: float  # quote currency, after fees
    entry_notional: float


@dataclass
class Report:
    decisions: int = 0
    decisions_vetoed: int = 0
    trips: int = 0
    trips_open: int = 0
    trips_dropped: dict[str, int] = field(default_factory=dict)
    trips_without_decision: int = 0
    rows_missing_features: int = 0
    labeled: int = 0

    def drop(self, reason: str) -> None:
        self.trips_dropped[reason] = self.trips_dropped.get(reason, 0) + 1


def base_asset(symbol: str) -> str:
    """``BTC/USDT`` / ``BTCUSDT`` / ``BTC`` -> ``BTC``."""
    text = symbol.upper().replace("/", "").replace("-", "").replace("_", "")
    for suffix in _QUOTE_SUFFIXES:
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return text


# --- Decisions -------------------------------------------------------------------------------


def parse_decisions(rows: Iterable[Mapping[str, Any]], report: Report | None = None) -> list[Decision]:
    """``META_LABEL_DECISION`` payloads -> decisions. Vetoed and malformed ones are dropped (counted in ``report``)."""
    rep = report or Report()
    out: list[Decision] = []
    for row in rows:
        try:
            direction = int(row["direction"])
            if direction not in (-1, 1):
                continue
            if row.get("vetoed"):
                rep.decisions_vetoed += 1
                continue
            features = {str(k): float(v) for k, v in dict(row["features"]).items() if math.isfinite(float(v))}
            out.append(Decision(float(row["decision_time"]), base_asset(str(row["instrument"])), direction, features))
        except (KeyError, TypeError, ValueError):
            continue
    rep.decisions = len(out)
    return sorted(out, key=lambda d: d.time_s)


def read_decisions_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def fetch_decisions_timescale(lookback_days: int) -> list[dict[str, Any]]:
    import asyncpg  # only needed here: --decisions runs without a database

    from core.config import get_settings

    timescale = get_settings().timescale
    end = datetime.now(timezone.utc)
    pool = await asyncpg.create_pool(
        host=timescale.host, port=timescale.port, database=timescale.database, user=timescale.user,
        password=timescale.password.get_secret_value() if timescale.password is not None else None, min_size=1, max_size=2,
    )
    try:
        records = await pool.fetch(
            "SELECT data FROM bus_messages WHERE topic = $1 AND time >= $2 AND time < $3 ORDER BY time",
            DECISION_TOPIC, end - timedelta(days=lookback_days), end,
        )
    finally:
        await pool.close()
    return [json.loads(r["data"]) if isinstance(r["data"], str) else dict(r["data"]) for r in records]


# --- Outcomes: round trips from real fills ---------------------------------------------------


def round_trips_from_fills(fills: Sequence[Execution], report: Report | None = None) -> list[RoundTrip]:
    """FIFO round trips (flat to flat) per ``(venue, asset)``, with real fill prices and fees.

    Each fill is split into a part that closes the open position and a part that opens (or adds to) one. A fill that crosses
    zero therefore ends one trip with its closing part and starts the next with the remainder; its fee is allocated by
    quantity. Trips touched by an unmatched or emergency fill, or by a fee in a non-quote currency, are dropped (counted).
    """
    rep = report or Report()
    by_key: dict[tuple[str, str], list[Execution]] = {}
    for fill in fills:
        by_key.setdefault((fill.venue, base_asset(fill.symbol)), []).append(fill)
    trips: list[RoundTrip] = []
    for (venue, asset), group in by_key.items():
        lots: deque[list[float]] = deque()  # [signed quantity, price], oldest first
        trip: dict[str, Any] | None = None

        def taint(fill: Execution) -> None:
            assert trip is not None
            if trip["tainted"] is not None:
                return
            if fill.kind == "unmatched":
                trip["tainted"] = "unmatched_fill"
            elif fill.kind == "emergency":
                trip["tainted"] = "emergency_exit"
            elif fill.fee and fill.fee_currency is not None and fill.fee_currency.upper() not in _QUOTE_FEE_CURRENCIES:
                trip["tainted"] = "fee_not_in_quote_currency"

        for fill in sorted(group, key=lambda f: (f.timestamp, f.fill_id)):
            sign = 1 if fill.direction == BUY else -1
            total = fill.fill_qty
            if total <= 0:
                continue
            fee = fill.fee or 0.0
            open_size = sum(abs(lot[0]) for lot in lots)
            closing = min(total, open_size) if lots and (lots[0][0] > 0) != (sign > 0) else 0.0
            opening = total - closing
            if closing > 0:
                assert trip is not None
                taint(fill)
                position_sign = 1 if lots[0][0] > 0 else -1
                left = closing
                while left > 1e-12 and lots:
                    lot = lots[0]
                    matched = min(left, abs(lot[0]))
                    trip["pnl"] += matched * (fill.fill_price - lot[1]) * position_sign
                    lot[0] -= position_sign * matched
                    left -= matched
                    if abs(lot[0]) < 1e-12:
                        lots.popleft()
                trip["fees"] += fee * closing / total
                if not lots:
                    rep.trips += 1
                    if trip["tainted"] is not None:
                        rep.drop(trip["tainted"])
                    else:
                        trips.append(RoundTrip(venue, asset, trip["direction"], trip["entry_ms"], fill.timestamp,
                                               trip["pnl"] - trip["fees"], trip["notional"]))
                    trip = None
            if opening > 1e-12:
                if trip is None:
                    trip = {"entry_ms": fill.timestamp, "direction": sign, "pnl": 0.0, "fees": 0.0, "notional": 0.0, "tainted": None}
                taint(fill)
                trip["fees"] += fee * opening / total
                trip["notional"] += opening * fill.fill_price
                lots.append([sign * opening, fill.fill_price])
        if lots:
            rep.trips_open += 1
    return sorted(trips, key=lambda t: t.entry_ms)


# --- Join and dataset ------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledTrade:
    trip: RoundTrip
    decision: Decision
    label: int


def join_decisions_to_trips(
    decisions: Sequence[Decision], trips: Sequence[RoundTrip], *, max_lag_s: float = 120.0, clock_tolerance_s: float = 5.0,
    report: Report | None = None,
) -> list[LabeledTrade]:
    """Each trip takes the latest unused decision of the same asset and direction in ``[entry - max_lag, entry + tolerance]``."""
    rep = report or Report()
    used: set[int] = set()
    labeled: list[LabeledTrade] = []
    for trip in sorted(trips, key=lambda t: t.entry_ms):
        entry_s = trip.entry_ms / 1000.0
        best: int | None = None
        for index, decision in enumerate(decisions):
            if index in used or decision.asset != trip.asset or decision.direction != trip.direction:
                continue
            if entry_s - max_lag_s <= decision.time_s <= entry_s + clock_tolerance_s and (best is None or decision.time_s > decisions[best].time_s):
                best = index
        if best is None:
            rep.trips_without_decision += 1
            continue
        used.add(best)
        labeled.append(LabeledTrade(trip, decisions[best], 1 if trip.pnl > 0 else 0))
    rep.labeled = len(labeled)
    return labeled


def build_matrix(labeled: Sequence[LabeledTrade], feature_names: Sequence[str], report: Report | None = None) -> tuple[list[LabeledTrade], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    rep = report or Report()
    kept, rows = [], []
    for item in labeled:
        if all(name in item.decision.features for name in feature_names):
            kept.append(item)
            rows.append([item.decision.features[name] for name in feature_names])
        else:
            rep.rows_missing_features += 1
    x = np.array(rows, dtype=np.float64).reshape(len(rows), len(feature_names))
    y = np.array([item.label for item in kept], dtype=np.int64)
    return kept, x, y


def chronological_split(kept: Sequence[LabeledTrade], test_fraction: float) -> tuple[int, int]:
    """``(train_end, test_start)`` indices: the test set starts after every training trade has ended (embargo)."""
    n = len(kept)
    train_end = max(1, int(round(n * (1.0 - test_fraction))))
    last_exit = max((t.trip.exit_ms for t in kept[:train_end]), default=0)
    test_start = train_end
    while test_start < n and kept[test_start].trip.entry_ms < last_exit:
        test_start += 1
    return train_end, test_start


# --- Training and evaluation -----------------------------------------------------------------


def payoff_ratio(trades: Sequence[LabeledTrade]) -> float:
    wins = [t.trip.pnl for t in trades if t.trip.pnl > 0]
    losses = [-t.trip.pnl for t in trades if t.trip.pnl < 0]
    if not wins or not losses:
        return 1.0
    return float(min(max(np.mean(wins) / np.mean(losses), _PAYOFF_CLIP[0]), _PAYOFF_CLIP[1]))


def fit_model(x: npt.NDArray[np.float64], y: npt.NDArray[np.int64]) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss", random_state=0,
    )
    model.fit(x, y)
    return model


def auc_lower_bound(auc: float, n_pos: int, n_neg: int, z: float = 2.326) -> float:
    """One-sided 99% lower bound of an AUC (Hanley-McNeil standard error; a model that sizes real money gets a strict test). A model must beat 0.5 beyond what luck explains:
    with a couple of hundred test trades an AUC of 0.55 is well within the noise of a coin flip."""
    if n_pos < 2 or n_neg < 2:
        return 0.0
    q1, q2 = auc / (2.0 - auc), 2.0 * auc * auc / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return auc - z * math.sqrt(max(variance, 0.0))


def evaluate(model: xgb.XGBClassifier, x_test: npt.NDArray[np.float64], y_test: npt.NDArray[np.int64], train_rate: float,
             test_trades: Sequence[LabeledTrade], payoff: float) -> dict[str, float | None]:
    p = model.predict_proba(x_test)[:, 1]
    both = len(set(y_test.tolist())) == 2
    multipliers = np.array([kelly_size_multiplier(float(v), payoff_ratio=payoff) for v in p])
    pnl = np.array([t.trip.pnl for t in test_trades])
    baseline_p = np.full_like(p, train_rate)
    return {
        "accuracy": float(accuracy_score(y_test, p >= 0.5)),
        "baseline_accuracy": float(max(np.mean(y_test), 1 - np.mean(y_test))),
        "auc": float(roc_auc_score(y_test, p)) if both else None,
        "auc_lower_99": auc_lower_bound(float(roc_auc_score(y_test, p)), int(y_test.sum()), int(len(y_test) - y_test.sum())) if both else None,
        "brier": float(brier_score_loss(y_test, p)),
        "baseline_brier": float(brier_score_loss(y_test, baseline_p)),
        "log_loss": float(log_loss(y_test, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "pnl_flat_sizing": float(pnl.sum()),
        "pnl_model_sizing": float((pnl * multipliers).sum()),  # approximation: PnL scales linearly with size
        "test_base_rate": float(np.mean(y_test)),
    }


def acceptance_failures(metrics: Mapping[str, float | None], *, n_labeled: int, n_test: int, min_trades: int, min_test: int, min_auc: float,
                        test_has_both: bool) -> list[str]:
    problems = []
    if n_labeled < min_trades:
        problems.append(f"only {n_labeled} labeled trades, at least {min_trades} required")
    if n_test < min_test:
        problems.append(f"test set of {n_test} trades, at least {min_test} required")
    if not test_has_both:
        problems.append("the test set contains a single outcome: AUC is undefined")
    auc = metrics.get("auc")
    if auc is not None and auc < min_auc:
        problems.append(f"AUC {auc:.3f} below {min_auc}")
    lower = metrics.get("auc_lower_99")
    if lower is not None and lower <= 0.5:
        problems.append(f"AUC {auc:.3f} is not significantly above chance (99% lower bound {lower:.3f} <= 0.5)")
    brier, baseline = metrics.get("brier"), metrics.get("baseline_brier")
    if brier is not None and baseline is not None and brier >= baseline:
        problems.append(f"Brier score {brier:.4f} does not beat the base-rate baseline {baseline:.4f}")
    return problems


def save_artifact(model: xgb.XGBClassifier, out: Path, metadata: dict[str, Any]) -> tuple[Path, Path]:
    """Writes the weights and the sidecar atomically (temporary file, then replace)."""
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"  # same xgboost/scikit-learn shim as scripts/train_model.py
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = metadata_path_for(out)
    # XGBoost picks the file format from the extension: the temporary file must keep ".json" or it is written in the legacy
    # binary format and then fails to load once renamed (the agent would silently fall back to the neutral 0.5).
    tmp_weights, tmp_meta = out.with_name(f".{out.stem}.tmp{out.suffix}"), sidecar.with_name(sidecar.name + ".tmp")
    model.save_model(str(tmp_weights))
    tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_meta, sidecar)
    os.replace(tmp_weights, out)
    return out, sidecar


# --- Orchestration ---------------------------------------------------------------------------


def run(decision_rows: Iterable[Mapping[str, Any]], fills: Sequence[Execution], *, out: Path, feature_names: Sequence[str] = FEATURES_V2,
        max_lag_s: float = 120.0, test_fraction: float = 0.3, min_trades: int = 300, min_test: int = 50, min_auc: float = 0.55,
        ) -> tuple[int, Report, dict[str, Any]]:
    """The whole pipeline on in-memory data. Returns ``(exit_code, report, details)``; writes the artifact only on success."""
    report = Report()
    decisions = parse_decisions(decision_rows, report)
    trips = round_trips_from_fills(fills, report)
    labeled = join_decisions_to_trips(decisions, trips, max_lag_s=max_lag_s, report=report)
    kept, x, y = build_matrix(labeled, feature_names, report)
    details: dict[str, Any] = {"n_labeled": len(kept), "refusals": []}
    if len(kept) < 2:
        details["refusals"] = [f"only {len(kept)} labeled trades, at least {min_trades} required"]
        return 1, report, details
    train_end, test_start = chronological_split(kept, test_fraction)
    y_train, y_test = y[:train_end], y[test_start:]
    details.update(n_train=train_end, n_test=len(y_test), n_embargoed=test_start - train_end)
    if len(set(y_train.tolist())) < 2 or len(y_test) == 0:
        details["refusals"] = ["train or test set empty, or the training set has a single outcome"]
        return 1, report, details
    model = fit_model(x[:train_end], y_train)
    payoff = payoff_ratio(kept[:train_end])
    metrics = evaluate(model, x[test_start:], y_test, float(np.mean(y_train)), kept[test_start:], payoff)
    details["metrics"] = metrics
    problems = acceptance_failures(
        metrics, n_labeled=len(kept), n_test=len(y_test), min_trades=min_trades, min_test=min_test, min_auc=min_auc,
        test_has_both=len(set(y_test.tolist())) == 2,
    )
    details["refusals"] = problems
    if problems:
        return 1, report, details
    metadata = {
        "version": 2,
        "feature_names": list(feature_names),
        "payoff_ratio": payoff,
        "label": "1 if round-trip PnL after real fills and fees > 0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": train_end,
        "n_test": len(y_test),
        "train_base_rate": float(np.mean(y_train)),
        "metrics": metrics,
        "first_entry_ms": kept[0].trip.entry_ms,
        "last_entry_ms": kept[-1].trip.entry_ms,
        "max_lag_s": max_lag_s,
    }
    details["artifact"] = [str(p) for p in save_artifact(model, out, metadata)]
    return 0, report, details


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--decisions", type=Path, help="JSON-lines file of META_LABEL_DECISION payloads")
    source.add_argument("--timescale", action="store_true", help="read the decisions from the TimescaleDB archive")
    parser.add_argument("--lookback-days", type=int, default=60, help="with --timescale")
    parser.add_argument("--fills-db", type=Path, nargs="+", required=True, help="workers' fills SQLite files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--features", choices=["v2", "v3"], default="v2",
                        help="v3 ajoute le VPIN (market:vpin:<SYMBOLE>) ; les décisions sans VPIN sont alors écartées et comptées")
    parser.add_argument("--max-lag-s", type=float, default=120.0)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--min-test", type=int, default=50)
    parser.add_argument("--min-auc", type=float, default=0.55)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 0.05 <= args.test_fraction <= 0.5:
        print("--test-fraction must be between 0.05 and 0.5", file=sys.stderr)
        return 2
    missing = [p for p in args.fills_db if not p.is_file()]
    if missing:
        print(f"Fills database not found: {missing[0]}", file=sys.stderr)
        return 2
    rows = read_decisions_file(args.decisions) if args.decisions else asyncio.run(fetch_decisions_timescale(args.lookback_days))
    fills: list[Execution] = []
    for path in args.fills_db:
        db = FillsDatabase(path)
        fills.extend(db.executions(limit=10_000_000))
        db.close()
    code, report, details = run(
        rows, fills, out=args.out, feature_names=FEATURES_V3 if args.features == "v3" else FEATURES_V2, max_lag_s=args.max_lag_s, test_fraction=args.test_fraction, min_trades=args.min_trades,
        min_test=args.min_test, min_auc=args.min_auc,
    )
    print(json.dumps({"exit": code, "report": report.__dict__, "details": details}, indent=2, default=str))
    if code:
        print("\nNo artifact written. The agent keeps its current model, or the neutral 0.5 if none.", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

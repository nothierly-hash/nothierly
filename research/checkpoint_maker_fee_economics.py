"""Chapter 12, Étape 3.9: Piste 3 -- does maker-side execution fix the economics?

Two honest, separate real-data questions, not one:

**Part 1 -- does a smaller fee make the SAME features/labels clear the
60% checkpoint?** Re-runs Étape 3.8's exact 13-feature architecture, but
with the Triple Barrier's round-trip cost halved (0.0004 instead of
0.0008) -- the real economics of entering **passively as a maker**
(assumed 0% fee) and exiting **decisively as a taker** (0.04%), the
standard, realistic asymmetric setup (you can choose to wait for a good
entry; you generally can't guarantee a passive fill the moment you need
to be OUT on a stop-loss).

**Part 2 -- is that maker entry actually achievable, and what does it
really cost?** A resting limit order does not fill for free just
because the label says "PT" was reachable from that price -- it fills
only once real trade flow has consumed the real queue ahead of it (see
:mod:`research.maker_execution_sim`'s module docstring for exactly what
is and isn't modeled). This part reports the REAL fill rate and fill
delay from the queue-position simulation, and checks for **adverse
selection**: is the forward outcome for orders that got filled
systematically worse than for orders that didn't -- the classic hidden
cost of passive market-making (you get filled disproportionately when
the market is about to move against you, since that's when aggressor
flow is heavy enough to reach your queue position)?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.alpha_features import (
    attach_cross_momentum,
    attach_trade_flow_features,
    attach_vpin_feature,
    compute_micro_price,
    compute_obi,
    compute_rolling_volatility,
    compute_vpin,
    label_triple_barrier,
    load_tardis_book_snapshot,
    load_tardis_trades,
    rolling_zscore,
)
from research.checkpoint_xgboost_orthogonal import (
    BASE_FEATURE_COLUMNS,
    ETH_BOOK_PATH,
    FEATURE_COLUMNS,
    PT_MULTIPLIER,
    SL_MULTIPLIER,
    VERTICAL_BARRIER_TICKS,
    _RAW_ORTHOGONAL_COLUMNS,
    _LABEL_TO_XGB_CLASS,
    _XGB_CLASS_NAMES,
)
from research.maker_execution_sim import simulate_maker_fill_times

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
BTC_BOOK_PATH = SAMPLE_DIR / "btcusdt_book_snapshot_2020-01-01.csv"
BTC_TRADES_PATH = SAMPLE_DIR / "btcusdt_trades_2020-01-01.csv"

_TAKER_FEE_PER_LEG = 0.0004
# Maker-entry (assumed 0% fee) + taker-exit (0.04%): the round-trip cost
# label_triple_barrier charges is 2 * taker_fee, so passing half the
# per-leg fee here reproduces a single taker-equivalent round trip.
_MAKER_ENTRY_TAKER_EXIT_FEE = _TAKER_FEE_PER_LEG / 2.0

_MAKER_FILL_SAMPLE_EVERY = 10  # real posting-opportunity density -- see maker_execution_sim's docstring
_MAKER_FILL_MAX_WAIT_SECONDS = 30.0
# Forward-outcome window after a real fill, to check for adverse
# selection -- same scale as this chapter's already-validated horizon.
_ADVERSE_SELECTION_HORIZON_TICKS = 500


def build_feature_matrix(book: pl.DataFrame, trades: pl.DataFrame, eth_book: pl.DataFrame, *, taker_fee: float) -> pl.DataFrame:
    """Identical to checkpoint_xgboost_orthogonal.py's, except the Triple Barrier's round-trip fee is parameterized."""
    book = compute_obi(book, depth=1).rename({"obi": "obi_depth1"})
    book = compute_obi(book, depth=5).rename({"obi": "obi_depth5"})
    book = compute_obi(book, depth=10).rename({"obi": "obi_depth10"})
    book = compute_micro_price(book)
    book = book.with_columns(
        (pl.col("asks[0].price") - pl.col("bids[0].price")).alias("spread"),
        (pl.col("micro_price") - pl.col("micro_price").shift(5)).alias("micro_price_momentum_5"),
        (pl.col("micro_price") - pl.col("micro_price").shift(20)).alias("micro_price_momentum_20"),
    )
    book = attach_trade_flow_features(book, trades)
    vpin_result = compute_vpin(trades)
    book = attach_vpin_feature(book, vpin_result)
    book = attach_cross_momentum(book, eth_book, asset_b_label="eth")
    for raw_column in _RAW_ORTHOGONAL_COLUMNS:
        book = rolling_zscore(book, raw_column)
    return label_triple_barrier(
        book, vertical_barrier_ticks=VERTICAL_BARRIER_TICKS, pt_multiplier=PT_MULTIPLIER,
        sl_multiplier=SL_MULTIPLIER, taker_fee=taker_fee,
    )


def run_checkpoint(clean: pl.DataFrame, *, label: str) -> dict[str, float]:
    """Train + evaluate, identical methodology to checkpoint_xgboost_orthogonal.py."""
    X = clean.select(FEATURE_COLUMNS).to_numpy()
    y = np.array([_LABEL_TO_XGB_CLASS[v] for v in clean["label"].to_numpy()], dtype=np.int64)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    predictions = model.predict(X_test_scaled)

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(classification_report(y_test, predictions, labels=[0, 1, 2], target_names=_XGB_CLASS_NAMES, zero_division=0))
    print("Confusion matrix [SL, noise, PT]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    predicted_totals = {c: int((predictions == c).sum()) for c in (0, 1, 2)}
    precision = {
        c: (float(((predictions == c) & (y_test == c)).sum()) / predicted_totals[c] if predicted_totals[c] else float("nan"))
        for c in (0, 2)
    }
    print(f"SL precision: {precision[0]:.2%} ({predicted_totals[0]} calls)")
    print(f"PT precision: {precision[2]:.2%} ({predicted_totals[2]} calls)")
    verdict = "PASSED" if precision[0] >= 0.60 and precision[2] >= 0.60 else "NOT MET"
    print(f"CHECKPOINT: {verdict}")
    return {"sl_precision": precision[0], "pt_precision": precision[2]}


def part1_reduced_fee_checkpoint(btc_book: pl.DataFrame, btc_trades: pl.DataFrame, eth_book: pl.DataFrame) -> None:
    print("\n" + "#" * 70)
    print("# PART 1: Does halving the round-trip fee clear the checkpoint?")
    print("#" * 70)

    featured_taker = build_feature_matrix(btc_book, btc_trades, eth_book, taker_fee=_TAKER_FEE_PER_LEG)
    clean_taker = featured_taker.drop_nulls(subset=[*FEATURE_COLUMNS, "label"])
    print(f"\n[Baseline: full taker round-trip, 0.08%] usable rows: {clean_taker.height}")
    for row in clean_taker["label"].value_counts().sort("label").iter_rows(named=True):
        print(f"  label={row['label']}: {row['count']} ({row['count'] / clean_taker.height:.2%})")
    baseline = run_checkpoint(clean_taker, label="BASELINE: full taker fee (0.08% round trip) -- Étape 3.8 reproduction")

    featured_maker = build_feature_matrix(btc_book, btc_trades, eth_book, taker_fee=_MAKER_ENTRY_TAKER_EXIT_FEE)
    clean_maker = featured_maker.drop_nulls(subset=[*FEATURE_COLUMNS, "label"])
    print(f"\n[Maker-entry + taker-exit, 0.04%] usable rows: {clean_maker.height}")
    for row in clean_maker["label"].value_counts().sort("label").iter_rows(named=True):
        print(f"  label={row['label']}: {row['count']} ({row['count'] / clean_maker.height:.2%})")
    reduced = run_checkpoint(clean_maker, label="MAKER-ENTRY + TAKER-EXIT: half round-trip fee (0.04%)")

    print(f"\nComparison: PT precision {baseline['pt_precision']:.2%} (full fee) -> {reduced['pt_precision']:.2%} (half fee)")


def part2_real_fill_simulation(btc_book: pl.DataFrame, btc_trades: pl.DataFrame) -> None:
    print("\n" + "#" * 70)
    print("# PART 2: Is the maker entry actually achievable -- real queue simulation")
    print("#" * 70)

    print(
        f"\nSimulating real queue-position BUY-side maker fills "
        f"(sample_every={_MAKER_FILL_SAMPLE_EVERY}, max_wait={_MAKER_FILL_MAX_WAIT_SECONDS}s)..."
    )
    fills = simulate_maker_fill_times(
        btc_book, btc_trades, side="buy", max_wait_seconds=_MAKER_FILL_MAX_WAIT_SECONDS, sample_every=_MAKER_FILL_SAMPLE_EVERY
    )
    fill_rate = float(fills["filled"].mean())
    print(f"Real fill rate: {fill_rate:.1%} of {fills.height} simulated postings (within {_MAKER_FILL_MAX_WAIT_SECONDS}s)")
    filled_only = fills.filter(pl.col("filled"))
    if filled_only.height > 0:
        print(f"Real mean fill delay (when filled): {float(filled_only['fill_delay_seconds'].mean()):.2f}s, median: {float(filled_only['fill_delay_seconds'].median()):.2f}s")

    print("\nChecking for adverse selection: real forward price outcome, filled vs. never-filled postings...")
    book_with_vol = compute_rolling_volatility(compute_micro_price(btc_book))
    labeled = label_triple_barrier(
        book_with_vol, vertical_barrier_ticks=_ADVERSE_SELECTION_HORIZON_TICKS,
        taker_fee=_MAKER_ENTRY_TAKER_EXIT_FEE, pt_multiplier=PT_MULTIPLIER, sl_multiplier=SL_MULTIPLIER,
    ).select(["timestamp", "label"])

    fills_with_outcome = fills.join_asof(
        labeled.sort("timestamp"), left_on="post_timestamp", right_on="timestamp", strategy="backward"
    )
    for group_name, group_df in (("FILLED", fills_with_outcome.filter(pl.col("filled"))), ("NEVER FILLED", fills_with_outcome.filter(~pl.col("filled")))):
        clean_group = group_df.drop_nulls("label")
        if clean_group.height == 0:
            print(f"  {group_name}: no labeled rows.")
            continue
        counts = clean_group["label"].value_counts().sort("label")
        total = clean_group.height
        parts = {row["label"]: row["count"] for row in counts.iter_rows(named=True)}
        pt_rate = parts.get(1.0, 0) / total
        sl_rate = parts.get(-1.0, 0) / total
        print(f"  {group_name} ({total} rows): PT rate={pt_rate:.2%}, SL rate={sl_rate:.2%}, noise={1 - pt_rate - sl_rate:.2%}")


def main() -> None:
    for path in (BTC_BOOK_PATH, BTC_TRADES_PATH, ETH_BOOK_PATH):
        if not path.exists():
            print(f"Real sample data not found at {path}.", file=sys.stderr)
            sys.exit(1)

    print("Loading real BTC/ETH data...")
    btc_book = load_tardis_book_snapshot(BTC_BOOK_PATH)
    btc_trades = load_tardis_trades(BTC_TRADES_PATH)
    eth_book = load_tardis_book_snapshot(ETH_BOOK_PATH)
    print(f"  BTC: {btc_book.height} snapshots, {btc_trades.height} trades. ETH: {eth_book.height} snapshots.")

    part1_reduced_fee_checkpoint(btc_book, btc_trades, eth_book)
    part2_real_fill_simulation(btc_book, btc_trades)


if __name__ == "__main__":
    main()

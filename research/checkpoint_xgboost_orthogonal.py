"""Chapter 12, Étape 3.8: Orthogonal Feature Expansion -- The Tape & Cross-Asset.

Diagnosis this responds to (Étape 3.7, real, measured): Triple Barrier
labeling, properly calibrated (horizon=1000 ticks, pt_multiplier=0.25),
gave a well-powered 3-class problem, but the 6 original microstructure
features (OBI x3 depths, spread, 2 momentum windows) only reached 5.04%
precision on the PT (win) class -- a real, statistically significant
edge over the 4.07% base rate (z=9.53), but ~12x short of the 60%
checkpoint. The diagnosis: not enough independent information in the
feature set (Information Coefficient too low), not a labeling problem.

This script adds real, orthogonal signal sources on top of the same
Triple Barrier labels (unchanged from Étape 3.7):

**The Tape** (real trades, already downloaded, no new data needed):
:func:`research.alpha_features.attach_trade_flow_features` (TFI at
1s/5s/30s -- real wall-clock rolling aggressor-signed volume) and
:func:`research.alpha_features.attach_vpin_feature` (the VPIN function
built in Étape 1 but never actually wired into a model's feature set
until now).

**Cross-asset** (real, newly downloaded): Tardis's free ETHUSDT sample
for the same real day, same exchange, same mechanism as the existing
BTCUSDT data (downloaded with your explicit confirmation given the file
size) -- :func:`research.alpha_features.attach_cross_momentum` computes
ETH's own trailing return, causally as-of joined onto BTC's timeline.

**NLP sentiment -- deliberately NOT included**: a real, free, historical
source was found (alternative.me's Fear & Greed Index, no auth needed;
2020-01-01 = 37/100, "Fear"), but it is a *daily* index -- one value for
this entire single real trading day. Added to a single-day dataset it
would be a constant column (and ``OBI_x_Sentiment`` would just be ``OBI
* constant``, not new information) -- real data, but not real *signal*
for this dataset. Skipped rather than fabricating a fake intraday
series, per this research chapter's real-data-only rule.

All new features get a trailing (causal) rolling Z-Score
(:func:`research.alpha_features.rolling_zscore`), per the brief -- TFI/
VPIN/cross-momentum are intraday non-stationary (activity intensity
varies a lot over the day) in a way the 6 original features' single
global ``StandardScaler`` (fit once on the training split, unchanged
here) doesn't correct for.
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
    compute_vpin,
    label_triple_barrier,
    load_tardis_book_snapshot,
    load_tardis_trades,
    rolling_zscore,
)

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
BTC_BOOK_PATH = SAMPLE_DIR / "btcusdt_book_snapshot_2020-01-01.csv"
BTC_TRADES_PATH = SAMPLE_DIR / "btcusdt_trades_2020-01-01.csv"
ETH_BOOK_PATH = SAMPLE_DIR / "ethusdt_book_snapshot_2020-01-01.csv"

# Same calibration as Étape 3.7's real parameter sweep -- see
# checkpoint_xgboost_triple_barrier.py's module docstring for why.
VERTICAL_BARRIER_TICKS = 1000
PT_MULTIPLIER = 0.25
SL_MULTIPLIER = 1.0

BASE_FEATURE_COLUMNS = [
    "obi_depth1", "obi_depth5", "obi_depth10", "spread",
    "micro_price_momentum_5", "micro_price_momentum_20",
]
_RAW_ORTHOGONAL_COLUMNS = [
    "tfi_1s", "tfi_5s", "tfi_30s", "vpin",
    "cross_momentum_eth_1s", "cross_momentum_eth_5s", "cross_momentum_eth_30s",
]
ORTHOGONAL_FEATURE_COLUMNS = [f"{col}_zscore" for col in _RAW_ORTHOGONAL_COLUMNS]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ORTHOGONAL_FEATURE_COLUMNS

_LABEL_TO_XGB_CLASS: dict[float, int] = {-1.0: 0, 0.0: 1, 1.0: 2}
_XGB_CLASS_NAMES: list[str] = ["SL (loss, -1)", "vertical (noise, 0)", "PT (win, +1)"]


def build_feature_matrix(book: pl.DataFrame, trades: pl.DataFrame, eth_book: pl.DataFrame) -> pl.DataFrame:
    """The 6 real microstructure features (Étape 3) + 7 real orthogonal features (Étape 3.8) + Triple Barrier labels."""
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
        book, vertical_barrier_ticks=VERTICAL_BARRIER_TICKS, pt_multiplier=PT_MULTIPLIER, sl_multiplier=SL_MULTIPLIER
    )


def main() -> None:
    for path in (BTC_BOOK_PATH, BTC_TRADES_PATH, ETH_BOOK_PATH):
        if not path.exists():
            print(f"Real sample data not found at {path}.", file=sys.stderr)
            sys.exit(1)

    print(f"Loading real BTC order book: {BTC_BOOK_PATH.name}")
    btc_book = load_tardis_book_snapshot(BTC_BOOK_PATH)
    print(f"  {btc_book.height} real snapshots")
    print(f"Loading real BTC trades: {BTC_TRADES_PATH.name}")
    btc_trades = load_tardis_trades(BTC_TRADES_PATH)
    print(f"  {btc_trades.height} real trades")
    print(f"Loading real ETH order book: {ETH_BOOK_PATH.name}")
    eth_book = load_tardis_book_snapshot(ETH_BOOK_PATH)
    print(f"  {eth_book.height} real snapshots")

    print(
        f"\nBuilding {len(FEATURE_COLUMNS)} features ({len(BASE_FEATURE_COLUMNS)} microstructure + "
        f"{len(ORTHOGONAL_FEATURE_COLUMNS)} orthogonal) + Triple Barrier labels "
        f"(horizon={VERTICAL_BARRIER_TICKS}, pt_mult={PT_MULTIPLIER}, sl_mult={SL_MULTIPLIER})..."
    )
    featured = build_feature_matrix(btc_book, btc_trades, eth_book)
    clean = featured.drop_nulls(subset=[*FEATURE_COLUMNS, "label"])
    print(f"Rows usable after dropping all warm-up/join/barrier nulls: {clean.height} / {btc_book.height}")

    print("\nReal label distribution:")
    label_names = {-1.0: "SL (loss, -1)", 0.0: "vertical (noise, 0)", 1.0: "PT (win, +1)"}
    for row in clean["label"].value_counts().sort("label").iter_rows(named=True):
        print(f"  {label_names[row['label']]}: {row['count']} ({row['count'] / clean.height:.2%})")

    X = clean.select(FEATURE_COLUMNS).to_numpy()
    y = np.array([_LABEL_TO_XGB_CLASS[v] for v in clean["label"].to_numpy()], dtype=np.int64)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"\nTrain: {len(X_train)} rows (first 80%) | Test: {len(X_test)} rows (last 20%, never seen)")

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

    print(f"\n{'=' * 70}")
    print("ORTHOGONAL FEATURE EXPANSION -- CHECKPOINT RESULT")
    print(f"{'=' * 70}")
    print(classification_report(y_test, predictions, labels=[0, 1, 2], target_names=_XGB_CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted), order = [SL, noise, PT]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    xgb_class_of = {0: "SL", 1: "noise", 2: "PT"}
    predicted_totals = {c: int((predictions == c).sum()) for c in (0, 1, 2)}
    per_class_precision = {
        c: (float(((predictions == c) & (y_test == c)).sum()) / predicted_totals[c] if predicted_totals[c] else float("nan"))
        for c in (0, 1, 2)
    }
    print("\nPer-class precision (the metric the 60% checkpoint refers to):")
    for c in (0, 2):
        n_calls = predicted_totals[c]
        print(f"  {xgb_class_of[c]}: {per_class_precision[c]:.2%} ({n_calls} calls, {'PASSES' if per_class_precision[c] >= 0.60 else 'fails'} 60%)")

    true_pt_count = int((y_test == 2).sum())
    pt_calls = predicted_totals[2]
    expected_by_chance = pt_calls * true_pt_count / len(y_test) if len(y_test) else 0.0
    actual_pt_correct = int(((predictions == 2) & (y_test == 2)).sum())
    std_by_chance = (pt_calls * (true_pt_count / len(y_test)) * (1 - true_pt_count / len(y_test))) ** 0.5 if len(y_test) else 0.0
    z_score = (actual_pt_correct - expected_by_chance) / std_by_chance if std_by_chance > 0 else float("nan")
    print(
        f"\nPT base rate in test set: {true_pt_count}/{len(y_test)} = {true_pt_count / len(y_test):.3%}. "
        f"At {pt_calls} PT calls, chance expects ~{expected_by_chance:.1f} correct (std={std_by_chance:.1f}); "
        f"model got {actual_pt_correct} (z={z_score:.2f})."
    )

    # Étape 3.7 baseline for direct comparison (6 features, same barrier calibration).
    print(f"\nÉtape 3.7 baseline (6 microstructure features only): SL precision=24.03%, PT precision=5.04%, z=9.53")

    if per_class_precision[0] >= 0.60 and per_class_precision[2] >= 0.60:
        print("\nCHECKPOINT PASSED: both SL and PT precision >= 60%.")
    else:
        print("\nCHECKPOINT NOT MET: at least one of SL/PT precision is below 60%.")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    print("\nFeature importances (highest first):")
    for name, importance in sorted(importances.items(), key=lambda kv: -kv[1]):
        marker = " [orthogonal]" if name in ORTHOGONAL_FEATURE_COLUMNS else ""
        print(f"  {name}: {importance:.3f}{marker}")

    print(
        "\nCaveat: single real day, one market regime -- a genuine first real data point, "
        "not a statistically reliable verdict on real-world edge either way."
    )


if __name__ == "__main__":
    main()

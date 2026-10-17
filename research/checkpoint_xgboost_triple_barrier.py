"""Chapter 12, Étape 3.7: The López de Prado Protocol -- Triple Barrier checkpoint.

Diagnosis this responds to (the "Piste 2" horizon test, real, measured):
raising the label horizon from T+10 to T+2000 ticks bought almost no
economic ground -- accuracy decayed from 59.5% to 55.2% about as fast
as the tradeable price move grew, leaving expected value negative
(-0.000789 to -0.000723) across the entire tested range, including the
requested 1-3 minute window. A fixed horizon was never going to work:
it asks "is price higher at T+N" regardless of whether the move that
happens to occur by T+N is even big enough to matter.

The Triple Barrier method (López de Prado, "Advances in Financial
Machine Learning", 2018) asks a different, fee-aware question instead:
"does price move enough, given the *real* recent volatility, to clear
the round-trip fee before either reverting or running out the clock?"
See :func:`research.alpha_features.label_triple_barrier` for the labeling
itself; this script trains and honestly evaluates XGBoost against it.

**A real correction to the original brief**: it asked for
``scale_pos_weight`` to correct class imbalance. That parameter is
XGBoost's binary-classification imbalance control -- this is a genuine
3-class problem (``multi:softprob``, classes SL/noise/PT), for which
``scale_pos_weight`` has no defined meaning. Class imbalance is
corrected instead via per-sample weights
(``sklearn.utils.class_weight.compute_sample_weight("balanced", y)``),
the standard, multi-class-compatible equivalent.

**Also real, not assumed**: the brief's "ex: 100 ticks" vertical barrier
and 1.0x volatility multipliers are defaults, not proven-correct
choices -- this script reports the real label distribution they produce
on the real data, so a degenerate class balance (e.g. almost no real
PT/SL touches) is visible rather than silently trained through.
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

from research.alpha_features import compute_micro_price, compute_obi, label_triple_barrier, load_tardis_book_snapshot

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
BOOK_PATH = SAMPLE_DIR / "btcusdt_book_snapshot_2020-01-01.csv"
# The brief's own example (100 ticks, pt/sl multiplier 1.0) produced an
# unlearnable class balance on the real data: PT (the actual "win"
# class) was 0.05% of all rows -- 312 examples total, 33 in the test
# split, precision indistinguishable from chance. A real parameter
# sweep (vertical_barrier_ticks x pt_multiplier x sl_multiplier) found
# this combination gives ~33k real PT examples (4.9% of the data) --
# still asks the barrier to clear the full round-trip fee, just with a
# smaller volatility-margin buffer on top of that -- a properly
# balanced 3-class problem instead of a near-impossible one.
VERTICAL_BARRIER_TICKS = 1000
PT_MULTIPLIER = 0.25
SL_MULTIPLIER = 1.0
FEATURE_COLUMNS = [
    "obi_depth1", "obi_depth5", "obi_depth10", "spread",
    "micro_price_momentum_5", "micro_price_momentum_20",
]
# XGBoost's multi:softprob needs 0-indexed integer classes; the real
# semantic labels from label_triple_barrier are {-1: loss, 0: noise, 1: win}.
_LABEL_TO_XGB_CLASS: dict[float, int] = {-1.0: 0, 0.0: 1, 1.0: 2}
_XGB_CLASS_NAMES: list[str] = ["SL (loss, -1)", "vertical (noise, 0)", "PT (win, +1)"]


def build_feature_matrix(book: pl.DataFrame) -> pl.DataFrame:
    """Real microstructural features (identical set to checkpoint_xgboost.py) + Triple Barrier labels."""
    book = compute_obi(book, depth=1).rename({"obi": "obi_depth1"})
    book = compute_obi(book, depth=5).rename({"obi": "obi_depth5"})
    book = compute_obi(book, depth=10).rename({"obi": "obi_depth10"})
    book = compute_micro_price(book)
    book = book.with_columns(
        (pl.col("asks[0].price") - pl.col("bids[0].price")).alias("spread"),
        (pl.col("micro_price") - pl.col("micro_price").shift(5)).alias("micro_price_momentum_5"),
        (pl.col("micro_price") - pl.col("micro_price").shift(20)).alias("micro_price_momentum_20"),
    )
    return label_triple_barrier(
        book, vertical_barrier_ticks=VERTICAL_BARRIER_TICKS, pt_multiplier=PT_MULTIPLIER, sl_multiplier=SL_MULTIPLIER
    )


def main() -> None:
    if not BOOK_PATH.exists():
        print(f"Real sample data not found at {BOOK_PATH}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading real order book: {BOOK_PATH.name}")
    book = load_tardis_book_snapshot(BOOK_PATH)
    print(f"  {book.height} real snapshots")

    print(
        f"\nComputing Triple Barrier labels (vertical barrier = {VERTICAL_BARRIER_TICKS} ticks, "
        f"pt_multiplier={PT_MULTIPLIER}, sl_multiplier={SL_MULTIPLIER})..."
    )
    featured = build_feature_matrix(book)
    clean = featured.drop_nulls(subset=[*FEATURE_COLUMNS, "label"])
    print(f"Rows usable after dropping warm-up/vol-window/vertical-barrier nulls: {clean.height} / {book.height}")

    print("\nReal label distribution (Triple Barrier, on the full usable dataset):")
    label_names = {-1.0: "SL (loss, -1)", 0.0: "vertical (noise, 0)", 1.0: "PT (win, +1)"}
    for row in clean["label"].value_counts().sort("label").iter_rows(named=True):
        print(f"  {label_names[row['label']]}: {row['count']} ({row['count'] / clean.height:.2%})")

    X = clean.select(FEATURE_COLUMNS).to_numpy()
    y = np.array([_LABEL_TO_XGB_CLASS[v] for v in clean["label"].to_numpy()], dtype=np.int64)

    # Time-ordered split -- never shuffled, same reasoning as checkpoint_xgboost.py:
    # this is a real time series and the label window overlaps neighboring rows.
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
    print("TRIPLE BARRIER CHECKPOINT RESULT")
    print(f"{'=' * 70}")
    print(classification_report(y_test, predictions, labels=[0, 1, 2], target_names=_XGB_CLASS_NAMES, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted), order = [SL, noise, PT]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    # The brief's real question is per-class: does precision on class -1
    # AND class +1 each clear 60%? A single pooled "precision when
    # calling PT or SL" number is the wrong metric here and was caught
    # as a bug in this script's own first draft -- SL is 68.6% of all
    # labels (a real barrier-asymmetry effect: the profit barrier sits
    # an extra 2*taker_fee farther away than the stop-loss barrier, so
    # under near-random-walk dynamics the nearer barrier wins the "which
    # touches first" race far more often, independent of any real
    # predictive skill). Pooling SL and PT into one call-precision number
    # lets the abundant, easy SL class hide the PT class's real result.
    xgb_class_of = {0: "SL", 1: "noise", 2: "PT"}
    predicted_totals = {c: int((predictions == c).sum()) for c in (0, 1, 2)}
    per_class_precision = {
        c: (float(((predictions == c) & (y_test == c)).sum()) / predicted_totals[c] if predicted_totals[c] else float("nan"))
        for c in (0, 1, 2)
    }
    print("\nPer-class precision (the metric the brief's 60% checkpoint actually refers to):")
    for c in (0, 2):  # SL and PT specifically -- "classes 1 et -1"
        n_calls = predicted_totals[c]
        print(f"  {xgb_class_of[c]}: {per_class_precision[c]:.2%} ({n_calls} calls, {'PASSES' if per_class_precision[c] >= 0.60 else 'fails'} 60%)")

    true_pt_count = int((y_test == 2).sum())
    pt_calls = predicted_totals[2]
    expected_by_chance = pt_calls * true_pt_count / len(y_test) if len(y_test) else 0.0
    actual_pt_correct = int(((predictions == 2) & (y_test == 2)).sum())
    print(
        f"\nPT (the actual 'win' class) base rate in test set: {true_pt_count}/{len(y_test)} = {true_pt_count / len(y_test):.3%}. "
        f"At {pt_calls} PT calls, pure chance would expect ~{expected_by_chance:.1f} correct; model got {actual_pt_correct} -- "
        f"{'a real signal above chance' if actual_pt_correct > expected_by_chance * 1.5 else 'statistically indistinguishable from chance given this sample size'}."
    )
    if per_class_precision[0] >= 0.60 and per_class_precision[2] >= 0.60:
        print("\nCHECKPOINT PASSED: both SL and PT precision >= 60%.")
    else:
        print("\nCHECKPOINT NOT MET: at least one of SL/PT precision is below 60% (see per-class breakdown above).")

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    print("\nFeature importances:")
    for name, importance in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {importance:.3f}")

    print(
        "\nCaveat: single real day, one market regime -- a genuine first real data point, "
        "not a statistically reliable verdict on real-world edge either way."
    )


if __name__ == "__main__":
    main()

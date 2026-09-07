"""Offline training pipeline for the Supervisor's decision model.

Not part of the live trading system -- run manually or on a schedule,
completely independent of every running agent::

    python scripts/train_model.py --lookback-days 30 --horizon-minutes 15

Pipeline: TimescaleDB (``bus_messages``, written by
:class:`~src.agents.archivist.agent.ArchivistAgent`) -> an aligned
feature/label frame -> an XGBoost classifier + its fitted feature scaler
-> ``models/supervisor_model_<timestamp>.json`` /
``models/supervisor_scaler_<timestamp>.pkl``, loadable by
:class:`~src.agents.supervisor.ml_engine.SupervisorMLModel` on the
Supervisor's next restart.

Design notes:
    - Feature order matches :attr:`SupervisorMLModel.FEATURE_NAMES`
      exactly, so a model trained here is directly usable without any
      adapter code.
    - Every data-shaping step (:func:`build_macro_feature_frame`,
      :func:`build_sentiment_feature_frame`, :func:`label_future_returns`,
      :func:`build_training_dataset`) is a pure function over already-
      fetched rows/DataFrames, independent of the database connection --
      each is independently unit-testable with synthetic data, and this
      is what "modular" means for this script in practice.
    - Labels are built with an *as-of* (backward-looking) join for
      features and a *forward* lookup for the future price used in the
      label, so no feature ever leaks information from after its own
      timestamp (look-ahead bias).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/train_model.py` adds this file's own directory
# (scripts/) to sys.path, not the project root -- `core`/`src` live at
# the root, so it must be added explicitly before those imports below,
# regardless of the working directory the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.config import get_settings
from core.logger import configure_logging, get_logger
from src.agents.supervisor.ml_engine import SupervisorMLModel

_log = get_logger(__name__)

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"

_MARKET_TOPIC = "MARKET_DATA_L2"
_MACRO_TOPIC = "MACRO_DATA_UPDATE"
_NLP_TOPIC = "NLP_SIGNAL"


# -- Database access (thin; everything below is pure and independently testable) --


async def fetch_topic_rows(
    pool: asyncpg.Pool, topic: str, start: datetime, end: datetime
) -> list[asyncpg.Record]:
    """Fetch every archived ``bus_messages`` row for ``topic`` in ``[start, end)``.

    Args:
        pool: An open connection pool to the TimescaleDB archive.
        topic: The ZMQ topic to filter on (e.g. ``"MARKET_DATA_L2"``).
        start: Inclusive start of the time window.
        end: Exclusive end of the time window.

    Returns:
        Rows ordered by time, each with ``time``/``sender_id``/``data``.
    """
    rows: list[asyncpg.Record] = await pool.fetch(
        "SELECT time, sender_id, data FROM bus_messages "
        "WHERE topic = $1 AND time >= $2 AND time < $3 ORDER BY time",
        topic,
        start,
        end,
    )
    return rows


# -- Pure data-shaping functions (unit-testable without a database) --------


def _records_to_frame(records: list[Any]) -> pd.DataFrame:
    """Flatten archived ``(time, sender_id, data)`` rows into a flat DataFrame.

    Args:
        records: Rows as returned by :func:`fetch_topic_rows` (or any
            object supporting ``record["time"]``/``record["data"]``
            item access, e.g. a plain ``dict`` -- used directly by this
            script's own unit tests).

    Returns:
        One row per record, one column per top-level key of its
        ``data`` JSON payload, plus a ``time`` column. Empty (with just
        a ``time`` column) if ``records`` is empty.
    """
    rows = []
    for record in records:
        payload = record["data"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        # asyncpg returns already-timezone-aware datetimes for TIMESTAMPTZ
        # columns; pd.Timestamp(..., tz="UTC") only accepts *naive* input
        # and raises on an aware one, so aware/naive are normalized
        # separately rather than passing tz= unconditionally.
        time_value = pd.Timestamp(record["time"])
        time_value = (
            time_value.tz_localize("UTC") if time_value.tzinfo is None else time_value.tz_convert("UTC")
        )
        rows.append({"time": time_value, **payload})
    if not rows:
        return pd.DataFrame(columns=["time"])
    return pd.DataFrame.from_records(rows).sort_values("time").reset_index(drop=True)


def build_market_feature_frame(market_records: list[Any]) -> pd.DataFrame:
    """Extract ``time``, ``spread``, ``order_book_imbalance``, ``mid_price``.

    Args:
        market_records: Raw ``MARKET_DATA_L2`` rows (``MarketMetrics``
            payloads).

    Returns:
        A DataFrame with those four columns, time-sorted.

    Raises:
        ValueError: If any row is missing ``mid_price`` (an older
            archive predating that field being added to
            ``MarketMetrics`` -- see ``core/schemas.py``).
    """
    frame = _records_to_frame(market_records)
    if frame.empty:
        return pd.DataFrame(columns=["time", "spread", "order_book_imbalance", "mid_price"])
    if "mid_price" not in frame.columns:
        raise ValueError(
            "Archived MARKET_DATA_L2 rows have no 'mid_price' field; cannot compute "
            "future-return labels without it. Was this data archived before mid_price "
            "was added to MarketMetrics?"
        )
    return frame[["time", "spread", "order_book_imbalance", "mid_price"]]


def build_macro_feature_frame(macro_records: list[Any]) -> pd.DataFrame:
    """Pivot raw ``MacroSnapshot`` rows into one column per indicator.

    Args:
        macro_records: Raw ``MACRO_DATA_UPDATE`` rows.

    Returns:
        A DataFrame with a ``time`` column and one column per indicator
        name seen in the data (e.g. ``US_10Y_YIELD``,
        ``BTC_EXCHANGE_FLOW``), each holding that indicator's value as
        of that timestamp.
    """
    frame = _records_to_frame(macro_records)
    if frame.empty or "indicator" not in frame.columns:
        # See build_sentiment_feature_frame's identical fix: an object-
        # dtype empty time column here breaks the merge_asof against
        # market_frame's real datetime64 column in
        # align_market_macro_sentiment.
        return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]")})
    pivoted = frame.pivot_table(index="time", columns="indicator", values="value", aggfunc="last")
    return pivoted.reset_index().sort_values("time")


def build_sentiment_feature_frame(
    nlp_records: list[Any], *, rolling_window: str = "1h"
) -> pd.DataFrame:
    """Turn raw ``SentimentSignal`` rows into a rolling signed-sentiment series.

    Args:
        nlp_records: Raw ``NLP_SIGNAL`` rows.
        rolling_window: Pandas time-offset string for the rolling mean
            (e.g. ``"1h"``): at each headline's timestamp, average the
            signed score of every headline in the preceding window.

    Returns:
        A DataFrame with ``time`` and ``sentiment_score_signed`` columns.
        ``POSITIVE`` contributes ``+score``, ``NEGATIVE`` contributes
        ``-score``, ``NEUTRAL`` contributes ``0`` -- the same convention
        as :meth:`SupervisorMLModel._aggregate_sentiment`.
    """
    frame = _records_to_frame(nlp_records)
    if frame.empty or "polarity" not in frame.columns:
        # dtype="datetime64[ns, UTC]" explicitly, not the default object
        # dtype an empty-columns DataFrame would otherwise get: this
        # frame is merge_asof'd against market_frame's real datetime64
        # column in align_market_macro_sentiment, which raises
        # MergeError on an object/datetime64 dtype mismatch -- a real
        # gap only reachable when nlp_records is genuinely empty (no
        # sentiment provider configured/reachable), which every prior
        # exercise of this function happened not to hit.
        return pd.DataFrame(
            {"time": pd.Series(dtype="datetime64[ns, UTC]"), "sentiment_score_signed": pd.Series(dtype=float)}
        )

    sign = frame["polarity"].map({"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0})
    frame = frame.assign(signed_score=sign * frame["score"]).set_index("time")
    rolling_mean = frame["signed_score"].rolling(rolling_window).mean()
    return (
        rolling_mean.rename("sentiment_score_signed")
        .reset_index()
        .sort_values("time")
    )


def label_future_returns(
    market_frame: pd.DataFrame, horizon: pd.Timedelta, up_threshold: float = 0.0
) -> pd.DataFrame:
    """Attach a binary "will price be higher `horizon` from now" label.

    For each row, looks up the first known mid-price at or after
    ``time + horizon`` (a forward-looking lookup applied *only* to the
    label, never to any feature -- see the module docstring on
    look-ahead bias) and computes the percentage return to it.

    Args:
        market_frame: Output of :func:`build_market_feature_frame`.
        horizon: How far into the future the label looks.
        up_threshold: Minimum future return to label a row "up" (``1``);
            rows with a smaller (or negative) return are labeled ``0``.

    Returns:
        ``market_frame`` with two extra columns: ``future_return`` and
        ``label`` (``0``/``1``, ``pd.NA`` where no future price exists
        within the queried window -- typically the last ``horizon``'s
        worth of rows -- so callers should drop those before training).
    """
    if market_frame.empty:
        return market_frame.assign(future_return=pd.Series(dtype=float), label=pd.Series(dtype="Int64"))

    lookup = pd.merge_asof(
        market_frame[["time"]].assign(lookup_time=market_frame["time"] + horizon),
        market_frame[["time", "mid_price"]].rename(
            columns={"time": "future_time", "mid_price": "future_mid_price"}
        ),
        left_on="lookup_time",
        right_on="future_time",
        direction="forward",
    )
    result = market_frame.copy()
    result["future_return"] = (
        lookup["future_mid_price"].to_numpy() - result["mid_price"].to_numpy()
    ) / result["mid_price"].to_numpy()
    result["label"] = (result["future_return"] > up_threshold).astype("Int64")
    result.loc[result["future_return"].isna(), "label"] = pd.NA
    return result


def align_market_macro_sentiment(
    market_frame: pd.DataFrame, macro_frame: pd.DataFrame, sentiment_frame: pd.DataFrame
) -> pd.DataFrame:
    """As-of (backward-looking) join of macro/sentiment onto market timestamps.

    The shared alignment step behind both this chapter's XGBoost pipeline
    (:func:`build_training_dataset`) and chapter 7's PPO pipeline
    (:mod:`scripts.train_ppo`): at each market-data timestamp, attach the
    most recently known macro/sentiment reading -- never a future one, to
    avoid look-ahead bias -- with columns renamed to match
    :attr:`~src.agents.supervisor.ml_engine.SupervisorMLModel.
    FEATURE_NAMES` exactly.

    Args:
        market_frame: Output of :func:`build_market_feature_frame` (must
            have a ``time`` column plus ``mid_price``, at minimum).
        macro_frame: Output of :func:`build_macro_feature_frame`.
        sentiment_frame: Output of :func:`build_sentiment_feature_frame`.

    Returns:
        ``market_frame`` with ``macro_us_10y_yield``,
        ``macro_btc_exchange_flow``, and ``sentiment_score_signed``
        columns attached (missing/unmatched values filled with ``0.0``,
        i.e. "neutral" -- consistent with
        :meth:`SupervisorMLModel.build_feature_vector`'s own handling of
        a temporarily-unavailable provider).
    """
    merged = pd.merge_asof(
        market_frame.sort_values("time"), macro_frame.sort_values("time"), on="time", direction="backward"
    )
    merged = pd.merge_asof(
        merged.sort_values("time"), sentiment_frame.sort_values("time"), on="time", direction="backward"
    )

    for column in ("US_10Y_YIELD", "BTC_EXCHANGE_FLOW", "sentiment_score_signed"):
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = merged[column].fillna(0.0)

    return merged.rename(
        columns={"US_10Y_YIELD": "macro_us_10y_yield", "BTC_EXCHANGE_FLOW": "macro_btc_exchange_flow"}
    )


def build_training_dataset(
    market_records: list[Any],
    macro_records: list[Any],
    nlp_records: list[Any],
    *,
    horizon: pd.Timedelta,
    up_threshold: float = 0.0,
    sentiment_rolling_window: str = "1h",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """End-to-end: raw archived rows -> aligned ``(X, y)`` ready for training.

    Features are assembled via :func:`align_market_macro_sentiment` (an
    as-of join, so no feature ever uses information from after its own
    row's time). Labels come from :func:`label_future_returns`.

    Args:
        market_records: Raw ``MARKET_DATA_L2`` rows.
        macro_records: Raw ``MACRO_DATA_UPDATE`` rows.
        nlp_records: Raw ``NLP_SIGNAL`` rows.
        horizon: Label look-ahead window (e.g. ``pd.Timedelta(minutes=15)``).
        up_threshold: Passed through to :func:`label_future_returns`.
        sentiment_rolling_window: Passed through to
            :func:`build_sentiment_feature_frame`.

    Returns:
        ``(X, y)``: a feature matrix of shape ``(n, len(SupervisorMLModel.
        FEATURE_NAMES))`` in that exact column order, and a matching
        binary label vector of shape ``(n,)``. Rows with no valid future
        price (nearest the end of the queried window) are dropped.

    Raises:
        ValueError: If there is no usable market data in the queried
            window (see :func:`build_market_feature_frame`).
    """
    market = build_market_feature_frame(market_records)
    if market.empty:
        raise ValueError("No market data available for the requested window.")

    macro = build_macro_feature_frame(macro_records)
    sentiment = build_sentiment_feature_frame(nlp_records, rolling_window=sentiment_rolling_window)
    labeled = label_future_returns(market, horizon, up_threshold)

    merged = align_market_macro_sentiment(labeled, macro, sentiment)
    merged = merged.dropna(subset=["label"])

    X = merged[list(SupervisorMLModel.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = merged["label"].to_numpy(dtype=np.int64)
    return X, y


# -- Training / persistence -------------------------------------------------


def train_xgboost_classifier(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, StandardScaler, dict[str, float]]:
    """Fit a feature scaler + XGBoost classifier, and report holdout metrics.

    Args:
        X: Feature matrix, shape ``(n, len(SupervisorMLModel.FEATURE_NAMES))``.
        y: Binary labels, shape ``(n,)``.
        test_size: Fraction of rows held out for evaluation.
        random_state: Seed for the train/test split (reproducibility).

    Returns:
        ``(model, scaler, metrics)`` -- the fitted classifier, its paired
        fitted scaler, and a dict of holdout ``accuracy``/``roc_auc``.

    Raises:
        ValueError: If ``X``/``y`` contain fewer than 10 rows (too little
            data to meaningfully split and train on).
    """
    if len(X) < 10:
        raise ValueError(f"Need at least 10 labeled rows to train, got {len(X)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) > 1 else float("nan"),
        "train_rows": float(len(X_train)),
        "test_rows": float(len(X_test)),
    }
    return model, scaler, metrics


def save_model(
    model: xgb.XGBClassifier, scaler: StandardScaler, models_dir: Path = MODELS_DIR
) -> tuple[Path, Path]:
    """Save the model (XGBoost native format) and scaler (joblib), timestamped.

    Uses XGBoost's own ``save_model``/``load_model`` serialization
    (not pickle) for the model itself, since that is exactly what
    :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel` already
    loads via ``XGBClassifier.load_model`` -- a ``.pkl`` here would
    silently fail to round-trip through that existing loader.

    Args:
        model: The fitted classifier.
        scaler: Its paired fitted scaler.
        models_dir: Directory to save into (created if missing).

    Returns:
        ``(model_path, scaler_path)``.
    """
    # Compatibility shim: XGBClassifier.save_model() (as of xgboost 2.1.x)
    # asserts self._estimator_type is set, which it expects scikit-learn's
    # ClassifierMixin to have provided as a class attribute -- but recent
    # scikit-learn (this project pins scikit-learn<2.0.0 and xgboost<3.0.0
    # independently, and 1.9.0 + 2.1.4, both within those ranges, hit this)
    # has been migrating that mechanism to __sklearn_tags__(), dropping the
    # old attribute. "classifier" is exactly the value ClassifierMixin used
    # to supply, so this sets the objectively correct value XGBoost's own
    # save path needs, rather than routing around the check.
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"

    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model_path = models_dir / f"supervisor_model_{timestamp}.json"
    scaler_path = models_dir / f"supervisor_scaler_{timestamp}.pkl"

    model.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path


# -- Orchestration ------------------------------------------------------


async def run_training_pipeline(
    *, lookback_days: int, horizon_minutes: int, up_threshold: float, models_dir: Path = MODELS_DIR
) -> None:
    """Fetch, align, train, evaluate, and save -- the full offline pipeline.

    Args:
        lookback_days: How many days of archived history to query.
        horizon_minutes: Label look-ahead window, in minutes.
        up_threshold: Minimum future return to label a row "up".
        models_dir: Where to save the trained artifacts.

    Raises:
        ValueError: If there isn't enough archived data to train on (see
            :func:`build_training_dataset`, :func:`train_xgboost_classifier`).
    """
    settings = get_settings()
    timescale = settings.timescale

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    _log.info(
        "training_pipeline_started",
        lookback_days=lookback_days,
        horizon_minutes=horizon_minutes,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    pool = await asyncpg.create_pool(
        host=timescale.host,
        port=timescale.port,
        database=timescale.database,
        user=timescale.user,
        password=timescale.password.get_secret_value() if timescale.password is not None else None,
        min_size=1,
        max_size=2,
    )
    try:
        market_records = await fetch_topic_rows(pool, _MARKET_TOPIC, start, end)
        macro_records = await fetch_topic_rows(pool, _MACRO_TOPIC, start, end)
        nlp_records = await fetch_topic_rows(pool, _NLP_TOPIC, start, end)
    finally:
        await pool.close()

    _log.info(
        "training_data_fetched",
        market_rows=len(market_records),
        macro_rows=len(macro_records),
        nlp_rows=len(nlp_records),
    )

    X, y = build_training_dataset(
        market_records,
        macro_records,
        nlp_records,
        horizon=pd.Timedelta(minutes=horizon_minutes),
        up_threshold=up_threshold,
    )
    _log.info("training_dataset_built", rows=len(X), positive_rate=float(np.mean(y)) if len(y) else float("nan"))

    model, scaler, metrics = train_xgboost_classifier(X, y)
    _log.info("model_trained", **metrics)

    model_path, scaler_path = save_model(model, scaler, models_dir)
    _log.info("model_saved", model_path=str(model_path), scaler_path=str(scaler_path))

    print(f"Model saved to:  {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Holdout metrics: {metrics}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of archived history to train on.")
    parser.add_argument("--horizon-minutes", type=int, default=15, help="Future-return label horizon, in minutes.")
    parser.add_argument(
        "--up-threshold",
        type=float,
        default=0.0,
        help="Minimum future return to label a row 'up' (default: any positive return).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging(log_level="INFO", log_dir=None)
    args = _parse_args()
    asyncio.run(
        run_training_pipeline(
            lookback_days=args.lookback_days,
            horizon_minutes=args.horizon_minutes,
            up_threshold=args.up_threshold,
        )
    )


if __name__ == "__main__":
    main()

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

Evaluation (rewritten 2026-10-14 after an audit; the previous version used a
shuffled ``train_test_split``, which on overlapping time-series labels measures
memorization of neighbouring rows, not prediction):
    - **Chronological split with an embargo.** The first ``1 - test_fraction`` of
      the rows (sorted by time) train, the last ones test, never shuffled. A block of
      ``--embargo-minutes`` (default 24 h, never shorter than the label horizon or
      the 1 h sentiment window) is removed between them: a training row's label looks
      ``horizon`` into the future, i.e. into the test period, and the rolling
      features carry memory across the boundary.
    - **One symbol per dataset.** ``MARKET_DATA_L2`` archives every crypto symbol
      under one topic; the label looks up the "first mid price at or after t + horizon"
      across the whole frame, so mixing BTC and ETH rows makes the return a jump
      between two price levels. ``--symbol`` selects one; an archive that mixes several
      symbols and no ``--symbol`` is refused.
    - **Hanley-McNeil, one-sided 99 %, on an independent sample.** Test rows a few
      seconds apart share almost their whole label window: they are one observation, not
      thousands. The significance test runs on test rows spaced at least one horizon
      apart. If the AUC's 99 % lower bound does not exceed 0.5, nothing is saved.
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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from core.config import get_settings
from core.logger import configure_logging, get_logger
from risk.model_validation import AucSignificance, evaluate_auc_significance, split_by_time, split_chronological
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


def normalize_symbol(symbol: str) -> str:
    """``BTC/USDT``, ``btcusdt`` et ``BTC-USDT`` désignent le même instrument : comparés sans séparateurs ni casse."""
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def build_market_feature_frame(market_records: list[Any], symbol: str | None = None) -> pd.DataFrame:
    """Extract ``time``, ``spread``, ``order_book_imbalance``, ``mid_price`` for ONE symbol.

    Args:
        market_records: Raw ``MARKET_DATA_L2`` rows (``MarketMetrics``
            payloads).
        symbol: The symbol to keep (e.g. ``"BTC/USDT"`` or ``"BTCUSDT"``: compared
            ignoring case and separators, see :func:`normalize_symbol`). ``None`` is
            only accepted when the archive holds a single symbol.

    Returns:
        A DataFrame with those four columns, time-sorted.

    Raises:
        ValueError: If any row is missing ``mid_price`` (an older
            archive predating that field being added to
            ``MarketMetrics`` -- see ``core/schemas.py``), if ``symbol`` is
            absent from the archive, or if the archive mixes several symbols
            and none was selected: the forward-looking label would compare
            prices of different instruments.
    """
    frame = _records_to_frame(market_records)
    if frame.empty:
        return pd.DataFrame(columns=["time", "spread", "order_book_imbalance", "mid_price"])
    if "symbol" in frame.columns:
        present = sorted(frame["symbol"].dropna().astype(str).unique())
        if symbol is not None:
            wanted = normalize_symbol(symbol)
            if wanted not in {normalize_symbol(name) for name in present}:
                raise ValueError(f"Symbol {symbol!r} not found in the archive; symbols present: {present}.")
            frame = frame[frame["symbol"].astype(str).map(normalize_symbol) == wanted].reset_index(drop=True)
        elif len(present) > 1:
            raise ValueError(
                f"MARKET_DATA_L2 mixes {len(present)} symbols {present}: the future-return label would compare prices of "
                "different instruments. Select one with symbol=... (--symbol)."
            )
    elif symbol is not None:
        raise ValueError("Archived MARKET_DATA_L2 rows have no 'symbol' field; cannot select one.")
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


@dataclass(frozen=True)
class TrainingSet:
    """Rows sorted by time: features, binary labels, and each row's timestamp (ns since the epoch, UTC)."""

    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    times_ns: npt.NDArray[np.int64]


def build_training_set(
    market_records: list[Any],
    macro_records: list[Any],
    nlp_records: list[Any],
    *,
    horizon: pd.Timedelta,
    up_threshold: float = 0.0,
    sentiment_rolling_window: str = "1h",
    symbol: str | None = None,
) -> TrainingSet:
    """End-to-end: raw archived rows -> aligned, time-sorted ``(X, y, times)`` ready for a chronological split.

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
        symbol: Passed through to :func:`build_market_feature_frame`.

    Returns:
        The :class:`TrainingSet`; ``X`` has shape ``(n, len(SupervisorMLModel.
        FEATURE_NAMES))`` in that exact column order. Rows with no valid future
        price (nearest the end of the queried window) are dropped.

    Raises:
        ValueError: If there is no usable market data in the queried
            window (see :func:`build_market_feature_frame`).
    """
    market = build_market_feature_frame(market_records, symbol)
    if market.empty:
        raise ValueError("No market data available for the requested window.")

    macro = build_macro_feature_frame(macro_records)
    sentiment = build_sentiment_feature_frame(nlp_records, rolling_window=sentiment_rolling_window)
    labeled = label_future_returns(market, horizon, up_threshold)

    merged = align_market_macro_sentiment(labeled, macro, sentiment)
    merged = merged.dropna(subset=["label"]).sort_values("time", kind="stable").reset_index(drop=True)

    X = merged[list(SupervisorMLModel.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = merged["label"].to_numpy(dtype=np.int64)
    times_ns = merged["time"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return TrainingSet(X, y, times_ns)


def build_training_dataset(
    market_records: list[Any],
    macro_records: list[Any],
    nlp_records: list[Any],
    *,
    horizon: pd.Timedelta,
    up_threshold: float = 0.0,
    sentiment_rolling_window: str = "1h",
    symbol: str | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """``(X, y)`` of :func:`build_training_set`, in chronological order (kept for existing callers; no timestamps)."""
    training = build_training_set(
        market_records, macro_records, nlp_records, horizon=horizon, up_threshold=up_threshold,
        sentiment_rolling_window=sentiment_rolling_window, symbol=symbol,
    )
    return training.X, training.y


# -- Training / persistence -------------------------------------------------


def train_xgboost_classifier(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    *,
    test_size: float = 0.2,
    embargo_rows: int = 0,
    times_ns: npt.NDArray[np.int64] | None = None,
    embargo_ns: int = 0,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, StandardScaler, dict[str, float]]:
    """Fit a feature scaler + XGBoost classifier on the EARLY rows, report metrics on the LATE rows.

    **The rows must be sorted by time**: the split is chronological (the first
    ``1 - test_size`` train, the last ones test), never shuffled. An embargo block
    is removed between the two, given as a number of rows (``embargo_rows``) or, when
    ``times_ns`` is provided, as a duration (``embargo_ns``: the test starts at the first row
    whose time reaches the end of the training set plus the embargo). Callers that do not pass
    timestamps and leave ``embargo_rows`` at 0 get no embargo and must say so in what they report.

    Args:
        X: Feature matrix, shape ``(n, len(SupervisorMLModel.FEATURE_NAMES))``, time-ordered.
        y: Binary labels, shape ``(n,)``.
        test_size: Fraction of the rows at the end that is held out (before the embargo).
        embargo_rows: Rows removed between train and test (ignored when ``times_ns`` is given).
        times_ns: Row timestamps (ns), non-decreasing, for a time-based embargo.
        embargo_ns: Embargo duration in ns (with ``times_ns``).
        random_state: XGBoost seed (reproducibility).

    Returns:
        ``(model, scaler, metrics)`` -- the fitted classifier, its paired
        fitted scaler, and ``accuracy``/``roc_auc`` on the test rows plus
        ``train_rows``, ``embargoed_rows``, ``test_rows``. ``accuracy`` and ``roc_auc``
        here use every test row, which are strongly autocorrelated: the significance
        test is :func:`fit_and_validate`'s job, not these.

    Raises:
        ValueError: If there are fewer than 10 rows, the split leaves no test row, or the
            training rows hold a single outcome.
    """
    if len(X) < 10:
        raise ValueError(f"Need at least 10 labeled rows to train, got {len(X)}.")
    if times_ns is not None:
        train_end, test_start = split_by_time(times_ns, test_size, embargo_ns)
    else:
        train_end, test_start = split_chronological(len(X), test_size, embargo_rows)
    X_train, y_train, X_test, y_test = X[:train_end], y[:train_end], X[test_start:], y[test_start:]
    if len(set(y_train.tolist())) < 2:
        raise ValueError("The training rows contain a single outcome: nothing to learn from.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if len(set(y_test.tolist())) > 1 else float("nan"),
        "train_rows": float(len(X_train)),
        "embargoed_rows": float(test_start - train_end),
        "test_rows": float(len(X_test)),
    }
    return model, scaler, metrics


@dataclass(frozen=True)
class ValidationOutcome:
    model: xgb.XGBClassifier
    scaler: StandardScaler
    metrics: dict[str, float]
    significance: AucSignificance
    refusals: list[str]

    @property
    def accepted(self) -> bool:
        return not self.refusals


def fit_and_validate(
    training: TrainingSet,
    *,
    horizon: pd.Timedelta,
    embargo: pd.Timedelta,
    test_fraction: float = 0.2,
    min_independent_test: int = 50,
    sentiment_rolling_window: pd.Timedelta = pd.Timedelta("1h"),
) -> ValidationOutcome:
    """Chronological fit with an embargo, then the one-sided 99 % Hanley-McNeil test on an independent test sample.

    ``refusals`` lists every reason NOT to save the weights; an empty list means the model beats chance with 99 % one-sided
    confidence on rows spaced at least one ``horizon`` apart. Multiple testing is not corrected: re-running this on ten
    variants and keeping the best invalidates the threshold.

    Raises:
        ValueError: If the embargo is shorter than the label horizon or the sentiment window (the split would leak).
    """
    if embargo < max(horizon, sentiment_rolling_window):
        raise ValueError(
            f"Embargo {embargo} shorter than max(label horizon {horizon}, sentiment window {sentiment_rolling_window}): "
            "the end of the training set would share information with the start of the test set."
        )
    embargo_ns, horizon_ns = int(embargo.value), int(horizon.value)
    model, scaler, metrics = train_xgboost_classifier(
        training.X, training.y, test_size=test_fraction, times_ns=training.times_ns, embargo_ns=embargo_ns
    )
    _, test_start = split_by_time(training.times_ns, test_fraction, embargo_ns)
    test_x, test_y, test_times = training.X[test_start:], training.y[test_start:], training.times_ns[test_start:]
    scores = model.predict_proba(scaler.transform(test_x))[:, 1]
    significance = evaluate_auc_significance(test_y, scores.astype(np.float64), test_times, horizon_ns=horizon_ns)
    refusals: list[str] = []
    if significance.n_independent < min_independent_test:
        refusals.append(
            f"only {significance.n_independent} independent test observations (rows >= {horizon} apart), at least {min_independent_test} required"
        )
    if significance.auc_independent is None:
        refusals.append("the independent test sample has fewer than 2 observations of one outcome: AUC undefined")
    elif not significance.significant:
        refusals.append(
            f"AUC {significance.auc_independent:.3f} is not significantly above chance "
            f"(one-sided 99% lower bound {significance.lower_bound:.3f} <= 0.5, n={significance.n_independent})"
        )
    return ValidationOutcome(model, scaler, metrics, significance, refusals)


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
    *,
    lookback_days: int,
    horizon_minutes: int,
    up_threshold: float,
    symbol: str | None = None,
    test_fraction: float = 0.2,
    embargo_minutes: int = 24 * 60,
    min_independent_test: int = 50,
    models_dir: Path = MODELS_DIR,
) -> bool:
    """Fetch, align, train chronologically, test significance, and save ONLY if the model beats chance.

    Args:
        lookback_days: How many days of archived history to query.
        horizon_minutes: Label look-ahead window, in minutes.
        up_threshold: Minimum future return to label a row "up".
        symbol: The symbol to train on (required when the archive mixes several).
        test_fraction: Share of the (time-ordered) rows held out at the end.
        embargo_minutes: Block removed between train and test (>= horizon and >= the 1 h sentiment window).
        min_independent_test: Minimum number of independent test observations for the significance test.
        models_dir: Where to save the trained artifacts.

    Returns:
        ``True`` if the weights were saved, ``False`` if they were refused (nothing is written).

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

    horizon = pd.Timedelta(minutes=horizon_minutes)
    training = build_training_set(
        market_records, macro_records, nlp_records, horizon=horizon, up_threshold=up_threshold, symbol=symbol
    )
    _log.info("training_dataset_built", rows=len(training.y), positive_rate=float(np.mean(training.y)) if len(training.y) else float("nan"))

    outcome = fit_and_validate(
        training, horizon=horizon, embargo=pd.Timedelta(minutes=embargo_minutes), test_fraction=test_fraction,
        min_independent_test=min_independent_test,
    )
    _log.info("model_trained", **outcome.metrics)
    significance = outcome.significance
    print(f"Chronological split: train {int(outcome.metrics['train_rows'])} rows, embargo {int(outcome.metrics['embargoed_rows'])}, test {int(outcome.metrics['test_rows'])}")
    print(
        f"Independent test sample: n={significance.n_independent} (positives {significance.n_positive}, negatives {significance.n_negative}), "
        f"AUC={significance.auc_independent}, one-sided 99% lower bound={significance.lower_bound:.4f}, "
        f"AUC on all (autocorrelated) test rows={significance.auc_full}"
    )
    if not outcome.accepted:
        for reason in outcome.refusals:
            _log.warning("model_refused", reason=reason)
            print(f"REFUSED: {reason}")
        print("No weights written: the existing model stays in place.")
        return False

    model_path, scaler_path = save_model(outcome.model, outcome.scaler, models_dir)
    _log.info("model_saved", model_path=str(model_path), scaler_path=str(scaler_path))
    print(f"Model saved to:  {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Holdout metrics: {outcome.metrics}")
    return True


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
    parser.add_argument("--symbol", default=None, help="Symbol to train on, e.g. BTC/USDT (required if the archive mixes symbols).")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Share of the latest rows held out for testing.")
    parser.add_argument(
        "--embargo-minutes", type=int, default=24 * 60,
        help="Block removed between train and test (default 24 h; must cover the label horizon and the 1 h sentiment window).",
    )
    parser.add_argument("--min-independent-test", type=int, default=50, help="Minimum independent test observations for the significance test.")
    args = parser.parse_args()
    if not 0.05 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.05 and 0.5")
    if args.embargo_minutes < max(args.horizon_minutes, 60):
        parser.error(f"--embargo-minutes must be >= max(horizon {args.horizon_minutes} min, 60 min sentiment window)")
    return args


def main() -> int:
    """CLI entry point. Exit code 0 if the weights were saved, 1 if the model was refused (nothing written)."""
    configure_logging(log_level="INFO", log_dir=None)
    args = _parse_args()
    saved = asyncio.run(
        run_training_pipeline(
            lookback_days=args.lookback_days,
            horizon_minutes=args.horizon_minutes,
            up_threshold=args.up_threshold,
            symbol=args.symbol,
            test_fraction=args.test_fraction,
            embargo_minutes=args.embargo_minutes,
            min_independent_test=args.min_independent_test,
        )
    )
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())

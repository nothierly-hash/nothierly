"""Vectorized technical-indicator computation over historical OHLCV.

Every computation here operates on whole ``pandas.Series``/``DataFrame``
columns at once (``.ewm``/``.rolling``/vectorized arithmetic) -- never a
Python-level ``for`` loop over rows, per the explicit requirement this
module was built against. ``pandas`` is the right tool specifically here
(unlike the rest of this project's live, per-message path -- see
``requirements.txt``'s existing note on why the live path stays pandas-
free): this module's whole job is *batch* computation over a historical
window pulled from TimescaleDB, which is exactly pandas' designed use
case, not a per-tick calculation where its overhead would cost latency
for no benefit.

Four indicators, computed in :meth:`FeatureEngine.compute_features`:

- **EMA** (fast/slow exponential moving average): trend direction/momentum.
- **RSI** (Relative Strength Index): overbought/oversold oscillator, 0-100.
- **Bollinger Bands** (upper/lower): volatility-relative price envelope.
- **ATR** (Average True Range): absolute volatility magnitude.

**Normalization** (so RL's neural network sees inputs in a stable,
comparable range -- unbounded raw price levels would otherwise dominate
gradient updates): two different strategies, chosen per indicator's own
mathematical shape rather than one blanket transform applied everywhere:

- RSI already has a fixed, known theoretical range (``[0, 100]``) --
  :func:`normalize_bounded_oscillator` is a direct linear rescale to
  ``[-1, 1]``. Using a rolling window here (as if RSI's range were
  unknown) would artificially stretch small fluctuations during calm
  periods to fill the whole output range, manufacturing false signal.
- Every other indicator (EMA fast/slow, the Bollinger bands, ATR) is a
  raw *price-scale* or unbounded-volatility value with no fixed range --
  :func:`normalize_rolling_zscore` computes a rolling Z-score (how many
  standard deviations from its own recent rolling mean) and clips to
  ``[-1, 1]``, the standard way to make a trending, unbounded series
  comparable across instruments/regimes without knowing its bounds in
  advance.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

EMA_FAST_PERIOD: Final[int] = 12
EMA_SLOW_PERIOD: Final[int] = 26
RSI_PERIOD: Final[int] = 14
BOLLINGER_PERIOD: Final[int] = 20
BOLLINGER_STD_MULTIPLIER: Final[float] = 2.0
ATR_PERIOD: Final[int] = 14
_ZSCORE_WINDOW: Final[int] = 100

# Column order every FeatureVector published by
# src.data_pipeline.feature_store must agree with -- see that module.
FEATURE_COLUMNS: Final[tuple[str, ...]] = ("ema_fast", "ema_slow", "rsi", "bb_upper", "bb_lower", "atr")


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, vectorized via pandas' own EWM implementation."""
    return close.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index, vectorized (Wilder's smoothing via ``ewm(alpha=1/period)``).

    Returns:
        A series in ``[0, 100]``. ``avg_loss == 0`` is handled
        explicitly rather than left to fall out of the division: a
        smoothing window with zero losses is a real, valid market
        condition (a pure uptrend), and its correct limiting RSI is
        ``100`` (maximally overbought) -- collapsing it to a neutral
        ``50``, as a naive ``x / 0 -> NaN -> fillna(50)`` chain would,
        is wrong precisely when the signal is strongest. ``50`` is only
        used for the genuine no-signal case: zero gains *and* zero
        losses (a perfectly flat price, including the start of a
        constant series where ``diff()`` is undefined).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        relative_strength = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    # The series' very first row has no prior value for .diff() to
    # compare against (delta/gain/loss are all NaN there), which no
    # branch above resolves -- fall back to neutral for that one
    # genuinely-undefined position, same as the flat-series case.
    return rsi.fillna(50.0)


def compute_bollinger_bands(
    close: pd.Series, period: int = BOLLINGER_PERIOD, num_std: float = BOLLINGER_STD_MULTIPLIER
) -> tuple[pd.Series, pd.Series]:
    """Upper/lower Bollinger Bands: a rolling mean +/- ``num_std`` rolling standard deviations."""
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return upper, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range: Wilder-smoothed max(high-low, |high-prev_close|, |low-prev_close|)."""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def normalize_bounded_oscillator(series: pd.Series, *, lower_bound: float = 0.0, upper_bound: float = 100.0) -> pd.Series:
    """Linearly rescale a series with a known fixed range (e.g. RSI) to ``[-1, 1]``."""
    return (series - lower_bound) / (upper_bound - lower_bound) * 2.0 - 1.0


def normalize_rolling_zscore(series: pd.Series, *, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling Z-score of an unbounded series, clipped to ``[-1, 1]``.

    ``min_periods=1`` so the very start of a series (fewer than
    ``window`` prior points) still produces a value instead of ``NaN``
    -- a wider, less confident estimate early on, which is honest (there
    genuinely isn't a full window of history yet) rather than a gap the
    caller has to special-case.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().replace(0.0, np.nan)
    z_score = (series - rolling_mean) / rolling_std
    return z_score.clip(-1.0, 1.0).fillna(0.0)


class FeatureEngine:
    """Computes a normalized, fixed-shape feature matrix from raw OHLCV -- fully vectorized."""

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute every indicator in :data:`FEATURE_COLUMNS`, normalized to ``[-1, 1]``.

        Args:
            ohlcv: Ascending-time-order DataFrame with ``open``/``high``/
                ``low``/``close``/``volume`` columns (as returned by a
                query against ``ohlcv_candles`` -- see
                :mod:`~src.data_pipeline.schema`). Index is preserved
                verbatim onto the result.

        Returns:
            A DataFrame, same index as ``ohlcv``, with exactly the
            columns in :data:`FEATURE_COLUMNS`, each in ``[-1, 1]``.
        """
        close = ohlcv["close"]
        high = ohlcv["high"]
        low = ohlcv["low"]

        bb_upper, bb_lower = compute_bollinger_bands(close)
        raw = pd.DataFrame(
            {
                "ema_fast": compute_ema(close, EMA_FAST_PERIOD),
                "ema_slow": compute_ema(close, EMA_SLOW_PERIOD),
                "rsi": compute_rsi(close),
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "atr": compute_atr(high, low, close),
            },
            index=ohlcv.index,
        )

        normalized = pd.DataFrame(index=ohlcv.index)
        normalized["rsi"] = normalize_bounded_oscillator(raw["rsi"])
        for column in FEATURE_COLUMNS:
            if column == "rsi":
                continue
            normalized[column] = normalize_rolling_zscore(raw[column])

        return normalized[list(FEATURE_COLUMNS)]

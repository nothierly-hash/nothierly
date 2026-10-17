"""Vectorized historical backtest engine (Phase 2: "La Machine à Voyager dans le Temps").

Turns a precomputed directional signal series into a full performance
report over historical closing prices -- entirely via ``pandas``/``numpy``
vectorized operations, never a Python-level ``for`` loop over rows, so a
multi-year daily backtest runs in milliseconds rather than seconds (see
this module's own test for a concrete 5-year timing assertion). Same
"batch computation over a historical window is exactly pandas' designed
use case" reasoning as :mod:`~src.data_pipeline.feature_engine`'s own
vectorization discipline -- this module borrows that same discipline for
the same reason.

Deliberately strategy-agnostic: :func:`run_vectorized_backtest` takes an
already-decided ``signal`` series (``-1``/``0``/``1`` per period) rather
than owning any prediction logic itself -- the same signal a live
:class:`~src.agents.supervisor.agent.XgboostSignalAgent`/:class:`~src.
agents.rl_signal.agent.RlSignalAgent` would have produced, or a simple
rule-based strategy for research, are equally valid inputs. This keeps
"what decides to trade" and "how good would that have been historically"
as two separate, independently testable concerns.

**Look-ahead bias, avoided by construction**: a signal computed from bar
``t``'s data is only knowable once bar ``t`` has closed, so the resulting
position can only be *held* starting bar ``t+1`` -- enforced by
``signal.shift(1)`` before anything else touches it. Getting this wrong
(using bar ``t``'s signal to trade bar ``t``'s own return) is the single
most common way a backtest silently overstates a strategy's real-world
performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from core.quant_math import max_drawdown, sharpe_ratio

_DEFAULT_FEE_PCT: Final[float] = 0.0005
_DEFAULT_PERIODS_PER_YEAR: Final[int] = 252
_VALID_SIGNAL_VALUES: Final[frozenset[int]] = frozenset({-1, 0, 1})


@dataclass(frozen=True)
class BacktestResult:
    """One backtest run's full performance report.

    Attributes:
        total_return_pct: Total return over the whole period, as a
            percentage (e.g. ``23.4`` for +23.4%).
        sharpe_ratio: Annualized Sharpe ratio of the strategy's net
            (after-fee) periodic returns -- see :func:`core.quant_math.
            sharpe_ratio`.
        max_drawdown_pct: Maximum peak-to-trough decline of the equity
            curve, as a negative percentage (e.g. ``-18.2`` for a 18.2%
            drawdown) -- see :func:`core.quant_math.max_drawdown`.
        num_trades: Number of periods where the position actually
            changed (opened, closed, or flipped direction). A flip from
            short to long counts once here (one rebalancing event), even
            though it crosses two units of position and is charged
            ``fee_pct`` twice in :attr:`equity_curve`.
        win_rate: Fraction of periods with a nonzero net return that
            were positive, in ``[0, 1]``. ``0.0`` if the strategy never
            held a position (every return was exactly zero).
        equity_curve: Portfolio value at every period, same index as the
            input ``close``/``signal`` series, starting at the run's
            ``initial_capital``.
    """

    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    equity_curve: pd.Series


def run_vectorized_backtest(
    close: pd.Series,
    signal: pd.Series,
    *,
    initial_capital: float = 10_000.0,
    fee_pct: float = _DEFAULT_FEE_PCT,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _DEFAULT_PERIODS_PER_YEAR,
) -> BacktestResult:
    """Simulate holding ``signal``'s direction against ``close``, fully vectorized.

    Args:
        close: Closing price per period, ascending chronological order.
        signal: Directional call per period (``-1``/``0``/``1``), same
            index as ``close``. Interpreted as "decided using this
            period's data" -- see the module docstring on why it is
            shifted forward one period before being traded.
        initial_capital: Starting portfolio value.
        fee_pct: Proportional transaction cost charged on the notional
            crossed by every position change (e.g. ``0.0005`` = 5 basis
            points per unit of position opened/closed/flipped).
        risk_free_rate: Per-period risk-free rate, passed through to
            :func:`core.quant_math.sharpe_ratio`.
        periods_per_year: Periods per year for Sharpe annualization
            (``252`` for daily trading-day data -- the default).

    Returns:
        The full :class:`BacktestResult`.

    Raises:
        ValueError: If ``close``/``signal`` don't share an index, either
            has fewer than 2 periods, ``signal`` contains a value other
            than ``-1``/``0``/``1``, or ``initial_capital`` is not
            positive.
    """
    if not close.index.equals(signal.index):
        raise ValueError("close and signal must share the same index.")
    if len(close) < 2:
        raise ValueError("close must have at least 2 periods.")
    if not signal.isin(_VALID_SIGNAL_VALUES).all():
        raise ValueError("signal must only contain -1, 0, or 1.")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")

    # See module docstring: a signal is only actionable from the *next*
    # period onward.
    position = signal.shift(1).fillna(0.0)

    period_returns = close.pct_change().fillna(0.0)
    gross_returns = position * period_returns

    # Fee charged on every unit of position crossed -- .abs() means a
    # direct short<->long flip (a diff of 2) is correctly charged 2x a
    # simple open/close (a diff of 1).
    position_changes = position.diff().fillna(position.iloc[0]).abs()
    net_returns = gross_returns - position_changes * fee_pct

    equity_curve = initial_capital * (1.0 + net_returns).cumprod()
    equity_curve.name = "equity"

    total_return_pct = float(equity_curve.iloc[-1] / initial_capital - 1.0) * 100.0
    sharpe = sharpe_ratio(
        net_returns.to_numpy(), risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    mdd_pct = max_drawdown(equity_curve.to_numpy()) * 100.0
    num_trades = int((position_changes > 0).sum())

    nonzero_returns = net_returns[net_returns != 0.0]
    win_rate = float((nonzero_returns > 0).mean()) if len(nonzero_returns) > 0 else 0.0

    return BacktestResult(
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown_pct=mdd_pct,
        num_trades=num_trades,
        win_rate=win_rate,
        equity_curve=equity_curve,
    )

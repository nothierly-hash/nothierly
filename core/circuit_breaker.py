"""Price-move-based flash-crash detection, independent of spread.

Real gap this closes (found by ``tests/chaos_monkey.py``'s Chaos 1a):
:meth:`~src.agents.supervisor.agent.XgboostSignalAgent._check_kill_switch` only watches
Redis latency, spread-vs-rolling-baseline, and consecutive errors -- no code path anywhere
in this system computed a bare price-percentage move and halted on it, so a flash crash
that happens to keep a tight spread (a real, plausible market condition, not a contrived
one) slipped through entirely. :class:`MarketAnomalyDetector` adds exactly that one missing
check, deliberately as a *separate* signal from the spread-anomaly check it sits beside --
Chaos 1b already showed relying on a crash's side effects (a spread blowout) to trip an
unrelated check is not the same as actually detecting the price move itself.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Deque


class MarketAnomalyDetector:
    """Rolling-window flash-crash detector.

    Trips when the absolute fractional price move within
    ``window_seconds`` exceeds ``max_move_fraction`` -- independent of
    spread, order-book imbalance, or any other signal. Stateful per
    instrument: one instance per traded symbol, owned for the same
    lifetime as the agent that holds it (mirrors ``XgboostSignalAgent
    ._spread_history``'s own per-agent, per-symbol scoping -- see that
    class for the intended composition point).
    """

    def __init__(self, *, window_seconds: float = 60.0, max_move_fraction: float = 0.05) -> None:
        """
        Args:
            window_seconds: Lookback horizon a price move is measured
                over (e.g. ``60.0`` -- "did price move >X% in the last
                minute", not tick-to-tick, which would false-positive on
                ordinary intra-second volatility).
            max_move_fraction: Absolute fractional move (e.g. ``0.05`` =
                5%) within ``window_seconds`` that trips the detector.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if max_move_fraction <= 0:
            raise ValueError("max_move_fraction must be positive.")
        self._window_seconds = window_seconds
        self._max_move_fraction = max_move_fraction
        self._history: Deque[tuple[datetime, float]] = deque()

    def check(self, mid_price: float, *, now: datetime | None = None) -> str | None:
        """Record ``mid_price`` and return a reason string if a flash-crash move is detected, else ``None``.

        Args:
            mid_price: Latest real mid-price reading (e.g.
                ``MarketMetrics.mid_price``).
            now: Injectable clock for tests. Defaults to
                ``datetime.now(timezone.utc)``.
        """
        now = now or datetime.now(timezone.utc)
        self._history.append((now, mid_price))

        cutoff = now.timestamp() - self._window_seconds
        while self._history and self._history[0][0].timestamp() < cutoff:
            self._history.popleft()

        if len(self._history) < 2:
            return None  # nothing old enough in the window yet to measure a move against

        oldest_price = self._history[0][1]
        if oldest_price <= 0:
            return None

        move_fraction = abs(mid_price - oldest_price) / oldest_price
        if move_fraction > self._max_move_fraction:
            return (
                f"price moved {move_fraction:.2%} within the last {self._window_seconds:.0f}s "
                f"({oldest_price:.6f} -> {mid_price:.6f}), exceeding the "
                f"{self._max_move_fraction:.2%} flash-crash threshold -- independent of spread"
            )
        return None

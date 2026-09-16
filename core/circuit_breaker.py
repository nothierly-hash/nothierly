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

import time
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


class IngestionCircuitBreaker:
    """Rolling-window circuit breaker over a stream of good/bad data-ingestion outcomes.

    Real gap this closes ("Protocole AEGIS" post-mortem, 2026-09-16): a
    34,637-event ``alpha_engine_market_metrics_unparseable`` /
    36,211-event ``alpha_engine_stale_data_detected`` spam window over a
    real 51.5h shadow run (see ``src/agents/alpha_engine/agent.py``'s
    ``_read_fresh_market_metrics``/``_run_cycle``) was already correctly
    logged and correctly fail-closed (no trade was ever placed on bad
    data) -- but nothing in this codebase ever *counted* those repeated
    failures toward a live decision to stop trying and force a safe
    state. This class is that counter, deliberately decoupled from the
    consequence: it only tracks state and reports :attr:`tripped`, the
    same "detector returns a signal, the caller decides what to do"
    shape as :class:`MarketAnomalyDetector` above and
    ``AlphaMarketMakerAgent._check_order_rate_limit``'s identical
    rolling-window idiom -- not a new generic ``OPEN``/``HALF_OPEN``/
    ``CLOSED`` state-machine abstraction this codebase has never used.

    Trips (latches ``tripped = True`` until :meth:`reset`) when either:

    - :attr:`_max_consecutive_failures` failures have landed back-to-back
      with no intervening success, or
    - the failure rate within :attr:`_window_seconds` exceeds
      :attr:`_max_failure_rate`, once at least :attr:`_min_samples_for_rate`
      outcomes have landed in that window (a "rate" computed from a
      single sample is not a rate -- avoids a false trip on e.g. the very
      first outcome ever recorded happening to be a failure).
    """

    def __init__(
        self,
        *,
        max_consecutive_failures: int = 50,
        max_failure_rate: float = 0.05,
        window_seconds: float = 60.0,
        min_samples_for_rate: int = 20,
    ) -> None:
        """
        Args:
            max_consecutive_failures: Back-to-back failures (no success in
                between) that trip the breaker on their own, regardless of
                rate -- the brief's own "50 erreurs consécutives" example.
            max_failure_rate: Fraction of failures within
                ``window_seconds`` that trips the breaker (e.g. ``0.05`` =
                5%) -- the brief's own "> 5% sur une minute" example.
            window_seconds: Lookback horizon the failure rate is measured
                over.
            min_samples_for_rate: Minimum outcomes recorded within
                ``window_seconds`` before the rate condition is even
                evaluated.
        """
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive.")
        if not (0.0 < max_failure_rate <= 1.0):
            raise ValueError("max_failure_rate must be in (0, 1].")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if min_samples_for_rate <= 0:
            raise ValueError("min_samples_for_rate must be positive.")
        self._max_consecutive_failures = max_consecutive_failures
        self._max_failure_rate = max_failure_rate
        self._window_seconds = window_seconds
        self._min_samples_for_rate = min_samples_for_rate
        self._consecutive_failures = 0
        self._history: Deque[tuple[float, bool]] = deque()  # (monotonic_ts, was_failure)
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """``True`` from the moment a trip condition is met until :meth:`reset` is called."""
        return self._tripped

    @property
    def consecutive_failures(self) -> int:
        """Current back-to-back failure streak -- for telemetry/log fields, not decision logic."""
        return self._consecutive_failures

    def record_success(self, *, now: float | None = None) -> None:
        """Record one good outcome -- resets the consecutive-failure streak; never un-trips on its own."""
        resolved_now = now if now is not None else time.monotonic()
        self._consecutive_failures = 0
        self._record(resolved_now, was_failure=False)

    def record_failure(self, *, now: float | None = None) -> bool:
        """Record one bad outcome. Returns ``True`` exactly on the call that newly trips the breaker (an edge, not a level)."""
        resolved_now = now if now is not None else time.monotonic()
        self._consecutive_failures += 1
        self._record(resolved_now, was_failure=True)
        if not self._tripped and self._check_trip_conditions():
            self._tripped = True
            return True
        return False

    def _record(self, now: float, *, was_failure: bool) -> None:
        self._history.append((now, was_failure))
        cutoff = now - self._window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _check_trip_conditions(self) -> bool:
        if self._consecutive_failures >= self._max_consecutive_failures:
            return True
        if len(self._history) >= self._min_samples_for_rate:
            failure_rate = sum(1 for _, was_failure in self._history if was_failure) / len(self._history)
            if failure_rate > self._max_failure_rate:
                return True
        return False

    def reset(self) -> None:
        """Real re-arm after remediation (a clean reconnect completed, an operator reactivated the engine, ...)."""
        self._consecutive_failures = 0
        self._history.clear()
        self._tripped = False

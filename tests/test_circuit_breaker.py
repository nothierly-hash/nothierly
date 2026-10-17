"""Real, executable tests for core/circuit_breaker.py's IngestionCircuitBreaker.

"Protocole AEGIS" (2026-09-16) -- ground-truth tests for the new rolling-window circuit breaker
that closes the real gap found in a 51.5h shadow-run window (34,637 alpha_engine_market_metrics_
unparseable + 36,211 alpha_engine_stale_data_detected warnings, correctly logged and correctly
fail-closed, but never counted toward any live decision to stop retrying). See
src/agents/alpha_engine/agent.py's/src/agents/market_data/agent.py's own wiring for the real
callers; this file only exercises the breaker's own state machine in isolation.
"""

from __future__ import annotations

import pytest

from core.circuit_breaker import IngestionCircuitBreaker


def test_constructor_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        IngestionCircuitBreaker(max_consecutive_failures=0)
    with pytest.raises(ValueError):
        IngestionCircuitBreaker(max_failure_rate=0.0)
    with pytest.raises(ValueError):
        IngestionCircuitBreaker(max_failure_rate=1.5)
    with pytest.raises(ValueError):
        IngestionCircuitBreaker(window_seconds=0.0)
    with pytest.raises(ValueError):
        IngestionCircuitBreaker(min_samples_for_rate=0)


def test_starts_untripped() -> None:
    breaker = IngestionCircuitBreaker()
    assert breaker.tripped is False
    assert breaker.consecutive_failures == 0


def test_trips_on_consecutive_failures() -> None:
    # max_failure_rate=1.0 (the maximum allowed) means the rate condition can never trip on its own
    # (a rate can never exceed 1.0) -- isolates this test to the consecutive-failure condition only.
    breaker = IngestionCircuitBreaker(max_consecutive_failures=50, max_failure_rate=1.0, window_seconds=3600.0)
    now = 0.0
    tripped_edge_count = 0
    for i in range(49):
        just_tripped = breaker.record_failure(now=now + i)
        if just_tripped:
            tripped_edge_count += 1
        assert breaker.tripped is False  # not yet -- fewer than 50 consecutive failures so far

    # The 50th consecutive failure trips it.
    just_tripped = breaker.record_failure(now=now + 49)
    assert just_tripped is True
    assert breaker.tripped is True
    assert breaker.consecutive_failures == 50
    assert tripped_edge_count == 0  # never tripped early


def test_a_success_resets_the_consecutive_streak_and_prevents_a_trip() -> None:
    breaker = IngestionCircuitBreaker(max_consecutive_failures=10, max_failure_rate=0.99, window_seconds=3600.0)
    for i in range(9):
        breaker.record_failure(now=float(i))
    assert breaker.consecutive_failures == 9
    assert breaker.tripped is False

    breaker.record_success(now=9.0)
    assert breaker.consecutive_failures == 0

    # Nine more failures after the reset still shouldn't trip -- streak restarted, not accumulated.
    for i in range(9):
        breaker.record_failure(now=10.0 + i)
    assert breaker.tripped is False


def test_trips_on_failure_rate_within_window_once_enough_samples_exist() -> None:
    breaker = IngestionCircuitBreaker(
        max_consecutive_failures=1000,  # effectively disabled -- isolate the rate condition
        max_failure_rate=0.05,
        window_seconds=60.0,
        min_samples_for_rate=20,
    )
    now = 0.0
    # 19 successes, interleaved with failures below min_samples_for_rate, should never trip: rate
    # condition is only even evaluated once len(history) >= min_samples_for_rate.
    for i in range(19):
        breaker.record_success(now=now + i)
    assert breaker.tripped is False

    # The 20th sample is a failure: 1/20 = 5% exactly -- not STRICTLY greater than 5%, must not trip.
    just_tripped = breaker.record_failure(now=now + 19)
    assert just_tripped is False
    assert breaker.tripped is False

    # A 21st sample, also a failure: 2/21 ~= 9.5% > 5% -- trips now.
    just_tripped = breaker.record_failure(now=now + 20)
    assert just_tripped is True
    assert breaker.tripped is True


def test_old_outcomes_fall_out_of_the_window_and_stop_counting_toward_the_rate() -> None:
    breaker = IngestionCircuitBreaker(
        max_consecutive_failures=1000, max_failure_rate=0.05, window_seconds=60.0, min_samples_for_rate=5,
    )
    for i in range(5):
        breaker.record_failure(now=float(i))  # 5 failures, all within the first 5 seconds
    assert breaker.tripped is True  # 5/5 = 100% failure rate, well above 5%

    # A fresh breaker, well past the window, should not immediately trip on old history it never saw.
    fresh = IngestionCircuitBreaker(
        max_consecutive_failures=1000, max_failure_rate=0.05, window_seconds=60.0, min_samples_for_rate=5,
    )
    for i in range(5):
        fresh.record_success(now=float(i))
    assert fresh.tripped is False
    # 100 seconds later (past the 60s window), old successes have aged out -- only 4 fresh samples
    # exist, below min_samples_for_rate, so a single new failure must not trip on a too-small sample.
    just_tripped = fresh.record_failure(now=100.0)
    assert just_tripped is False


def test_trip_latches_until_reset_even_if_conditions_later_improve() -> None:
    breaker = IngestionCircuitBreaker(max_consecutive_failures=3, max_failure_rate=0.99, window_seconds=3600.0)
    for i in range(3):
        breaker.record_failure(now=float(i))
    assert breaker.tripped is True

    breaker.record_success(now=3.0)
    assert breaker.tripped is True  # a single later success does not silently un-trip

    breaker.reset()
    assert breaker.tripped is False
    assert breaker.consecutive_failures == 0


def test_defaults_trip_on_50_consecutive_failures_even_spread_far_apart_in_time() -> None:
    # Real defaults: max_consecutive_failures=50, window_seconds=60.0. Spacing failures 1000s apart
    # keeps each one's own 60s rate-window containing only itself (every earlier sample has already
    # aged out by the time the next one lands) -- isolates the consecutive-failure counter (which is
    # a pure sequential count, unaffected by elapsed time) from the rate condition.
    breaker = IngestionCircuitBreaker()
    for i in range(49):
        just_tripped = breaker.record_failure(now=float(i) * 1000.0)
        assert just_tripped is False
    assert breaker.tripped is False
    assert breaker.record_failure(now=49.0 * 1000.0) is True
    assert breaker.tripped is True


def test_defaults_trip_on_5_percent_failure_rate_within_60_seconds() -> None:
    # Real defaults: max_failure_rate=0.05, window_seconds=60.0, min_samples_for_rate=20. 19 clean
    # reads, then a run of failures -- crosses 5% well before 50 consecutive failures.
    breaker = IngestionCircuitBreaker()
    now = 0.0
    for i in range(19):
        breaker.record_success(now=now + i)
    assert breaker.tripped is False
    # 20th sample (a failure): 1/20 = 5%, not strictly greater -- must not trip yet.
    assert breaker.record_failure(now=now + 19) is False
    # 21st sample (also a failure): 2/21 ~= 9.5% > 5% -- trips.
    assert breaker.record_failure(now=now + 20) is True

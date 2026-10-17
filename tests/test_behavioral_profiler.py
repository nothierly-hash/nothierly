"""Real, executable tests for risk/behavioral_profiler.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from risk.behavioral_profiler import (
    ActionType,
    BehavioralProfiler,
    ClientAction,
)

_T0 = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def _fill(*, at: datetime, instrument: str = "BTC/USDT", quantity: float, pnl: float | None = None) -> ClientAction:
    return ClientAction(client_id="c1", timestamp=at, action_type=ActionType.ORDER_FILLED, instrument=instrument, quantity=quantity, realized_pnl_usd=pnl)


def _submission(*, at: datetime, instrument: str = "BTC/USDT") -> ClientAction:
    return ClientAction(client_id="c1", timestamp=at, action_type=ActionType.ORDER_SUBMITTED, instrument=instrument, quantity=1.0)


def _stop_move(*, at: datetime, price: float, instrument: str = "EUR_USD") -> ClientAction:
    return ClientAction(client_id="c1", timestamp=at, action_type=ActionType.STOP_LOSS_MOVED, instrument=instrument, quantity=1.0, stop_loss_price=price)


# -- record_action / client_id isolation --------------------------------


def test_record_action_rejects_wrong_client_id() -> None:
    profiler = BehavioralProfiler("c1")
    with pytest.raises(ValueError):
        profiler.record_action(_fill(at=_T0, quantity=1.0).model_copy(update={"client_id": "c2"}))


def test_history_is_capped_by_max_history() -> None:
    profiler = BehavioralProfiler("c1", max_history=3)
    for i in range(5):
        profiler.record_action(_submission(at=_T0 + timedelta(seconds=i)))
    assert len(profiler.history) == 3


# -- detect_revenge_trading --------------------------------


def test_revenge_trading_insufficient_data_with_no_pnl() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0))
    result = profiler.detect_revenge_trading()
    assert result.insufficient_data is True
    assert result.detected is False


def test_revenge_trading_detects_real_martingale_pattern() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0, pnl=-500.0))
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=30), quantity=2.0, pnl=None))
    result = profiler.detect_revenge_trading()
    assert result.detected is True
    assert result.confidence > 0
    assert len(result.evidence) == 1


def test_revenge_trading_ignores_outside_time_window() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0, pnl=-500.0))
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=120), quantity=5.0, pnl=None))  # too late
    result = profiler.detect_revenge_trading()
    assert result.detected is False


def test_revenge_trading_ignores_smaller_next_order() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0, pnl=-500.0))
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=10), quantity=1.1, pnl=None))  # not >= 1.5x
    result = profiler.detect_revenge_trading()
    assert result.detected is False


def test_revenge_trading_ignores_a_winning_trade() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0, pnl=500.0))  # a real win, not a loss
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=10), quantity=5.0, pnl=None))
    result = profiler.detect_revenge_trading()
    assert result.detected is False
    assert result.insufficient_data is False  # real pnl data existed, just no loss


# -- detect_stop_loss_creep --------------------------------


def test_stop_loss_creep_insufficient_data_with_no_moves() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0))
    result = profiler.detect_stop_loss_creep()
    assert result.insufficient_data is True


def test_stop_loss_creep_detects_repeated_downward_moves() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_stop_move(at=_T0, price=1.0800))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=1), price=1.0750))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=2), price=1.0700))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=3), price=1.0650))
    result = profiler.detect_stop_loss_creep()
    assert result.detected is True
    assert len(result.evidence) == 3  # 3 downward moves > threshold of 2


def test_stop_loss_creep_does_not_flag_two_moves() -> None:
    """The brief says '>2 fois' -- exactly 2 moves must NOT trigger."""
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_stop_move(at=_T0, price=1.0800))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=1), price=1.0750))
    result = profiler.detect_stop_loss_creep()
    assert result.detected is False


def test_stop_loss_creep_ignores_upward_moves() -> None:
    """Moving a stop UP on a long position is real risk management, not creep."""
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_stop_move(at=_T0, price=1.0700))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=1), price=1.0750))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=2), price=1.0800))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=3), price=1.0850))
    result = profiler.detect_stop_loss_creep()
    assert result.detected is False


def test_stop_loss_creep_tracks_instruments_independently() -> None:
    profiler = BehavioralProfiler("c1")
    # Two downward moves each on two different instruments -- neither alone exceeds the threshold.
    profiler.record_action(_stop_move(at=_T0, price=1.08, instrument="EUR_USD"))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=1), price=1.07, instrument="EUR_USD"))
    profiler.record_action(_stop_move(at=_T0, price=1.30, instrument="GBP_USD"))
    profiler.record_action(_stop_move(at=_T0 + timedelta(minutes=1), price=1.29, instrument="GBP_USD"))
    result = profiler.detect_stop_loss_creep()
    assert result.detected is False


# -- detect_overtrading --------------------------------


def test_overtrading_insufficient_data_with_few_orders() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_submission(at=_T0))
    result = profiler.detect_overtrading(now=_T0)
    assert result.insufficient_data is True


def test_overtrading_detects_real_statistical_spike() -> None:
    profiler = BehavioralProfiler("c1", max_history=1000)
    today = _T0 + timedelta(days=10)
    # A quiet, consistent baseline: exactly 2 orders/day for 9 days.
    for day in range(9):
        day_start = _T0 + timedelta(days=day)
        profiler.record_action(_submission(at=day_start))
        profiler.record_action(_submission(at=day_start + timedelta(hours=1)))
    # Today: a real spike.
    for i in range(40):
        profiler.record_action(_submission(at=today + timedelta(minutes=i)))

    # `now` must be after every "today" timestamp above, or the <= now filter
    # would drop the very submissions meant to represent today's spike.
    now = today + timedelta(hours=1)
    result = profiler.detect_overtrading(now=now)
    assert result.detected is True
    assert result.confidence > 0


def test_overtrading_does_not_flag_a_normal_day() -> None:
    profiler = BehavioralProfiler("c1", max_history=1000)
    today = _T0 + timedelta(days=10)
    for day in range(10):
        day_start = _T0 + timedelta(days=day)
        profiler.record_action(_submission(at=day_start))
        profiler.record_action(_submission(at=day_start + timedelta(hours=1)))
    now = today + timedelta(hours=2)
    result = profiler.detect_overtrading(now=now)
    assert result.detected is False


# -- generate_report / tilt score --------------------------------


def test_generate_report_no_action_when_nothing_detected() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0))
    report = profiler.generate_report(now=_T0)
    assert report.recommended_action == "NO_ACTION"
    assert report.tilt_score == 0.0


def test_generate_report_recommends_lock_when_severe() -> None:
    profiler = BehavioralProfiler("c1")
    profiler.record_action(_fill(at=_T0, quantity=1.0, pnl=-1000.0))
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=5), quantity=10.0, pnl=-2000.0))
    profiler.record_action(_fill(at=_T0 + timedelta(seconds=10), quantity=50.0, pnl=None))
    report = profiler.generate_report(now=_T0 + timedelta(seconds=10))
    assert report.tilt_score > 0
    assert any(d.detected for d in report.detections)


def test_generate_report_now_param_propagates_to_overtrading_detector() -> None:
    """Real regression test: generate_report's `now` must actually reach detect_overtrading, not just default to real time."""
    profiler = BehavioralProfiler("c1", max_history=1000)
    today = _T0 + timedelta(days=10)
    for day in range(9):
        day_start = _T0 + timedelta(days=day)
        profiler.record_action(_submission(at=day_start))
    for i in range(30):
        profiler.record_action(_submission(at=today + timedelta(minutes=i)))

    fixed_now = today + timedelta(hours=1)  # after every "today" timestamp above
    report = profiler.generate_report(now=fixed_now)
    overtrading = next(d for d in report.detections if d.bias_name == "overtrading")
    assert overtrading.detected is True

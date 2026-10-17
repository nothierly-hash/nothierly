"""Real, executable tests for execution/predator_honeypot.py.

Every test drives the real detector/agent with real, deterministic
timestamps -- no mocking of the detection logic itself. These tests
are the actual evidence behind this module's real claims: the pattern
detector fires on genuine spoofing-shaped flow and stays quiet on
legitimate flow, the time window really ages out old evidence, and the
defensive agent never does anything but recommend a posture.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

import pytest

from core.nothierly_ontology import OrderSide
from execution.predator_honeypot import (
    AdverseFlowDefenseAgent,
    DefensiveExecutionMode,
    OrderEvent,
    SpoofingPatternDetector,
    ToxicityLevel,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _evt(
    event_type: Literal["ADD", "CANCEL", "TRADE"], order_id: str, quantity: str, t: datetime, *, instrument: str = "BTC/USDT"
) -> OrderEvent:
    return OrderEvent(
        event_type=event_type, order_id=order_id, instrument=instrument, side=OrderSide.SELL,
        price=Decimal("50000"), quantity=Decimal(quantity), timestamp=t,
    )


# =============================================================================
# SpoofingPatternDetector -- real positive detection
# =============================================================================


def test_repeated_fleeting_large_cancels_escalate_to_toxic() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"), pattern_count_threshold=3)
    levels = []
    for i in range(3):
        detector.on_order_added(_evt("ADD", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 100)))
        levels.append(detector.on_order_cancelled(_evt("CANCEL", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 100 + 5))))

    assert levels == [ToxicityLevel.ELEVATED, ToxicityLevel.ELEVATED, ToxicityLevel.TOXIC]
    assert detector.current_level("BTC/USDT") == ToxicityLevel.TOXIC
    assert detector.fleeting_cancel_count("BTC/USDT") == 3


def test_untouched_instrument_stays_normal() -> None:
    detector = SpoofingPatternDetector()
    assert detector.current_level("ETH/USDT") == ToxicityLevel.NORMAL


def test_pattern_is_tracked_independently_per_instrument() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"), pattern_count_threshold=3)
    for i in range(3):
        detector.on_order_added(_evt("ADD", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10), instrument="BTC/USDT"))
        detector.on_order_cancelled(_evt("CANCEL", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10 + 5), instrument="BTC/USDT"))

    assert detector.current_level("BTC/USDT") == ToxicityLevel.TOXIC
    assert detector.current_level("ETH/USDT") == ToxicityLevel.NORMAL  # a different instrument's flow is unaffected


# =============================================================================
# Real, deliberate non-detection -- must not false-positive
# =============================================================================


def test_a_real_fill_never_counts_as_a_fleeting_cancel() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"))
    detector.on_order_added(_evt("ADD", "real-1", "10.0", _T0))
    detector.on_order_traded(_evt("TRADE", "real-1", "10.0", _T0 + timedelta(milliseconds=5)))
    assert detector.current_level("BTC/USDT") == ToxicityLevel.NORMAL


def test_slow_cancel_of_a_large_order_is_not_fleeting() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"))
    detector.on_order_added(_evt("ADD", "slow-1", "10.0", _T0))
    level = detector.on_order_cancelled(_evt("CANCEL", "slow-1", "10.0", _T0 + timedelta(milliseconds=500)))
    assert level == ToxicityLevel.NORMAL


def test_fast_cancel_of_a_small_order_is_not_large_enough() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"))
    detector.on_order_added(_evt("ADD", "small-1", "0.5", _T0))
    level = detector.on_order_cancelled(_evt("CANCEL", "small-1", "0.5", _T0 + timedelta(milliseconds=5)))
    assert level == ToxicityLevel.NORMAL


def test_cancel_of_an_order_never_seen_added_is_ignored_not_crashed_on() -> None:
    detector = SpoofingPatternDetector()
    level = detector.on_order_cancelled(_evt("CANCEL", "unknown-order", "10.0", _T0))
    assert level == ToxicityLevel.NORMAL


def test_wrong_event_type_raises_a_clean_error() -> None:
    detector = SpoofingPatternDetector()
    with pytest.raises(ValueError, match="ADD"):
        detector.on_order_added(_evt("CANCEL", "x", "1.0", _T0))
    with pytest.raises(ValueError, match="CANCEL"):
        detector.on_order_cancelled(_evt("ADD", "x", "1.0", _T0))
    with pytest.raises(ValueError, match="TRADE"):
        detector.on_order_traded(_evt("ADD", "x", "1.0", _T0))


# =============================================================================
# Real time-window aging
# =============================================================================


def test_old_fleeting_cancels_age_out_of_the_pattern_window() -> None:
    detector = SpoofingPatternDetector(
        fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"), pattern_window_seconds=1.0, pattern_count_threshold=3
    )
    for i in range(2):
        detector.on_order_added(_evt("ADD", f"old-{i}", "10.0", _T0 + timedelta(seconds=i * 0.1)))
        detector.on_order_cancelled(_evt("CANCEL", f"old-{i}", "10.0", _T0 + timedelta(seconds=i * 0.1, milliseconds=5)))
    assert detector.current_level("BTC/USDT") == ToxicityLevel.ELEVATED

    # A 3rd fleeting cancel 2s later -- well outside the 1s window, so the first two must have aged out
    detector.on_order_added(_evt("ADD", "late-1", "10.0", _T0 + timedelta(seconds=2.0)))
    level = detector.on_order_cancelled(_evt("CANCEL", "late-1", "10.0", _T0 + timedelta(seconds=2.0, milliseconds=5)))
    assert level == ToxicityLevel.ELEVATED  # not TOXIC -- would be TOXIC if the old evidence hadn't aged out
    assert detector.fleeting_cancel_count("BTC/USDT") == 1


# =============================================================================
# AdverseFlowDefenseAgent -- real posture, never places an order itself
# =============================================================================


def test_toxic_level_produces_post_only_only_posture_with_zero_size() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"), pattern_count_threshold=3)
    for i in range(3):
        detector.on_order_added(_evt("ADD", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10)))
        detector.on_order_cancelled(_evt("CANCEL", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10 + 5)))

    agent = AdverseFlowDefenseAgent(detector)
    posture = agent.posture_for("BTC/USDT")

    assert posture.mode == DefensiveExecutionMode.POST_ONLY_ONLY
    assert posture.max_order_size_multiplier == 0.0
    assert posture.require_post_only is True
    assert "BTC/USDT" in posture.reason


def test_normal_level_produces_normal_posture() -> None:
    detector = SpoofingPatternDetector()
    agent = AdverseFlowDefenseAgent(detector)
    posture = agent.posture_for("ETH/USDT")

    assert posture.mode == DefensiveExecutionMode.NORMAL
    assert posture.max_order_size_multiplier == 1.0
    assert posture.require_post_only is False


def test_external_toxicity_score_alone_can_escalate_to_cautious() -> None:
    """Real corroborating evidence from a second, independent signal (e.g. real OBI) matters even with a quiet detector."""
    detector = SpoofingPatternDetector()  # no order-flow evidence at all
    agent = AdverseFlowDefenseAgent(detector, external_toxicity_score=lambda instrument: 0.9, external_toxicity_threshold=0.7)

    posture = agent.posture_for("BTC/USDT")

    assert posture.mode == DefensiveExecutionMode.CAUTIOUS
    assert posture.require_post_only is True
    assert "score de toxicité externe" in posture.reason


def test_external_toxicity_score_below_threshold_does_not_escalate() -> None:
    detector = SpoofingPatternDetector()
    agent = AdverseFlowDefenseAgent(detector, external_toxicity_score=lambda instrument: 0.2, external_toxicity_threshold=0.7)

    posture = agent.posture_for("BTC/USDT")

    assert posture.mode == DefensiveExecutionMode.NORMAL


def test_toxic_detector_level_dominates_regardless_of_external_score() -> None:
    detector = SpoofingPatternDetector(fleeting_threshold_ms=10.0, large_size_threshold=Decimal("5.0"), pattern_count_threshold=3)
    for i in range(3):
        detector.on_order_added(_evt("ADD", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10)))
        detector.on_order_cancelled(_evt("CANCEL", f"o{i}", "10.0", _T0 + timedelta(milliseconds=i * 10 + 5)))

    agent = AdverseFlowDefenseAgent(detector, external_toxicity_score=lambda instrument: 0.0)  # external says "fine"
    posture = agent.posture_for("BTC/USDT")

    assert posture.mode == DefensiveExecutionMode.POST_ONLY_ONLY  # real detector evidence still wins

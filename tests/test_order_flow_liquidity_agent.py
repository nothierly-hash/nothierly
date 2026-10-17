"""Real, executable tests for execution/order_flow_liquidity_agent.py.

Every test drives the real agent with real, deterministic timestamps
and real, signed order-book-imbalance readings -- no mocking of the
detection math itself, same discipline as
``tests/test_predator_honeypot.py``.
"""

from __future__ import annotations

import pytest

from execution.order_flow_liquidity_agent import OrderFlowLiquidityAgent

_INSTRUMENT = "BTC/USDT"
_T0 = 1_800_000_000.0


def test_no_ticks_yet_is_neutral() -> None:
    agent = OrderFlowLiquidityAgent()
    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0)
    assert rec.direction == "NEUTRAL"
    assert rec.confidence == 0.0
    assert rec.size_multiplier == 1.0
    assert rec.dominance == 0.0


def test_sustained_one_sided_buy_pressure_recommends_buy() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=3, dominance_threshold=0.3, max_size_multiplier=2.0)
    for i in range(10):
        agent.on_tick(_INSTRUMENT, 0.8, timestamp=_T0 + i * 0.5)

    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0 + 4.5)
    assert rec.direction == "BUY"
    assert rec.dominance > 0.3
    assert rec.confidence == pytest.approx(rec.dominance)
    assert 1.0 < rec.size_multiplier <= 2.0
    assert rec.buy_intensity > rec.sell_intensity
    assert "BUY" in rec.reason


def test_sustained_one_sided_sell_pressure_recommends_sell() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=3, dominance_threshold=0.3, max_size_multiplier=2.0)
    for i in range(10):
        agent.on_tick(_INSTRUMENT, -0.8, timestamp=_T0 + i * 0.5)

    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0 + 4.5)
    assert rec.direction == "SELL"
    assert rec.dominance < -0.3
    assert rec.sell_intensity > rec.buy_intensity


def test_balanced_alternating_flow_stays_neutral() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=3, dominance_threshold=0.3)
    for i in range(10):
        imbalance = 0.6 if i % 2 == 0 else -0.6
        agent.on_tick(_INSTRUMENT, imbalance, timestamp=_T0 + i * 0.5)

    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0 + 4.5)
    assert rec.direction == "NEUTRAL"
    assert abs(rec.dominance) < 0.3
    assert rec.size_multiplier == 1.0


def test_zero_imbalance_tick_contributes_to_neither_side() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=1, dominance_threshold=0.01)
    agent.on_tick(_INSTRUMENT, 0.0, timestamp=_T0)
    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0)
    assert rec.direction == "NEUTRAL"
    assert rec.buy_intensity == rec.sell_intensity  # both still exactly baseline_intensity


def test_too_few_events_stays_neutral_even_with_extreme_imbalance() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=5, dominance_threshold=0.1)
    agent.on_tick(_INSTRUMENT, 1.0, timestamp=_T0)
    rec = agent.recommendation_for(_INSTRUMENT, at_time=_T0)
    assert rec.direction == "NEUTRAL"


def test_old_flow_decays_out_over_time() -> None:
    """A burst of real buy pressure long ago must decay back toward neutral -- not stay 'BUY' forever."""
    agent = OrderFlowLiquidityAgent(min_events_per_side=3, dominance_threshold=0.3, decay_rate=0.5)
    for i in range(10):
        agent.on_tick(_INSTRUMENT, 0.9, timestamp=_T0 + i * 0.1)

    rec_immediately_after = agent.recommendation_for(_INSTRUMENT, at_time=_T0 + 1.0)
    rec_much_later = agent.recommendation_for(_INSTRUMENT, at_time=_T0 + 500.0)

    assert rec_immediately_after.direction == "BUY"
    assert rec_much_later.dominance < rec_immediately_after.dominance
    assert rec_much_later.direction == "NEUTRAL"


def test_different_instruments_are_tracked_independently() -> None:
    agent = OrderFlowLiquidityAgent(min_events_per_side=3, dominance_threshold=0.3)
    for i in range(10):
        agent.on_tick("BTC/USDT", 0.8, timestamp=_T0 + i * 0.5)
    rec_untouched = agent.recommendation_for("ETH/USDT", at_time=_T0 + 4.5)
    assert rec_untouched.direction == "NEUTRAL"


def test_invalid_order_book_imbalance_rejected() -> None:
    agent = OrderFlowLiquidityAgent()
    with pytest.raises(ValueError, match="order_book_imbalance"):
        agent.on_tick(_INSTRUMENT, 1.5)
    with pytest.raises(ValueError, match="order_book_imbalance"):
        agent.on_tick(_INSTRUMENT, -1.5)


def test_invalid_constructor_args_rejected() -> None:
    with pytest.raises(ValueError, match="dominance_threshold"):
        OrderFlowLiquidityAgent(dominance_threshold=0.0)
    with pytest.raises(ValueError, match="dominance_threshold"):
        OrderFlowLiquidityAgent(dominance_threshold=1.0)
    with pytest.raises(ValueError, match="max_size_multiplier"):
        OrderFlowLiquidityAgent(max_size_multiplier=0.5)

"""Real, executable tests for src/execution/data_normalizer.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.execution.data_normalizer import normalize_l2_update


def _raw_book(*, timestamp_ms: int | None = 1_700_000_000_000) -> dict[str, object]:
    return {
        "bids": [[100.0, 2.0], [99.5, 1.0]],
        "asks": [[100.5, 1.5], [101.0, 3.0]],
        "timestamp": timestamp_ms,
    }


def test_normalize_l2_update_produces_valid_snapshot_and_metrics() -> None:
    result = normalize_l2_update(_raw_book(), symbol="BTC/USDT")

    assert result.snapshot.symbol == "BTC/USDT"
    assert result.snapshot.best_bid.price == 100.0
    assert result.snapshot.best_ask.price == 100.5
    assert result.metrics.spread == pytest.approx(0.5)
    assert result.metrics.mid_price == pytest.approx(100.25)
    # bid_volume=3.0, ask_volume=4.5 -> (3.0-4.5)/7.5 = -0.2
    assert result.metrics.order_book_imbalance == pytest.approx(-0.2)


def test_normalize_l2_update_falls_back_to_now_when_timestamp_missing() -> None:
    result = normalize_l2_update(_raw_book(timestamp_ms=None), symbol="ETH/USDT")
    assert result.snapshot.timestamp is not None


def test_normalize_l2_update_rejects_crossed_book() -> None:
    crossed = {"bids": [[101.0, 1.0]], "asks": [[100.0, 1.0]], "timestamp": 1_700_000_000_000}
    with pytest.raises(ValidationError):
        normalize_l2_update(crossed, symbol="BTC/USDT")


def test_normalize_l2_update_truncates_to_top_n_levels() -> None:
    deep_book = {
        "bids": [[100.0 - i, 1.0] for i in range(10)],
        "asks": [[101.0 + i, 1.0] for i in range(10)],
        "timestamp": 1_700_000_000_000,
    }
    result = normalize_l2_update(deep_book, symbol="BTC/USDT")
    assert len(result.snapshot.bids) == 5
    assert len(result.snapshot.asks) == 5

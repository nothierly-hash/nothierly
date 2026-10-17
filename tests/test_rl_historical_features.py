"""Mesures « prix + flux » : exactitude sur des cas calculés à la main, causalité (aucun regard vers le futur), refus des trous."""

from __future__ import annotations

import numpy as np
import pytest

from src.rl import historical_features as hf


def _trades(n_bars: int, per_bar: int = 3, *, start_ms: int = 1_760_000_040_000, price0: float = 100.0, drift: float = 0.01) -> dict[str, np.ndarray]:
    times, prices, qtys, buys = [], [], [], []
    for b in range(n_bars):
        for k in range(per_bar):
            times.append(start_ms + b * 60_000 + 10_000 * k)
            prices.append(price0 + drift * (b * per_bar + k))
            qtys.append(1.0 + k)
            buys.append(k % 2 == 0)
    return {"times_ms": np.array(times, dtype=np.int64), "prices": np.array(prices), "quantities": np.array(qtys), "is_buy": np.array(buys)}


def _bars(n_bars: int, **kwargs: float) -> dict[str, np.ndarray]:
    t = _trades(n_bars, **kwargs)  # type: ignore[arg-type]
    return hf.build_time_bars(t["times_ms"], t["prices"], t["quantities"], t["is_buy"])


def test_bars_aggregate_notional_by_taker_side_and_keep_the_last_price() -> None:
    t = _trades(2, per_bar=3)
    bars = hf.build_time_bars(t["times_ms"], t["prices"], t["quantities"], t["is_buy"])
    assert len(bars["close"]) == 2
    first = [(t["prices"][i], t["quantities"][i], t["is_buy"][i]) for i in range(3)]
    assert bars["buy_notional"][0] == pytest.approx(sum(p * q for p, q, b in first if b))
    assert bars["sell_notional"][0] == pytest.approx(sum(p * q for p, q, b in first if not b))
    assert bars["close"][0] == t["prices"][2]
    assert bars["bar_end_s"][1] - bars["bar_end_s"][0] == 60.0


def test_empty_unsorted_or_invalid_trades_are_refused() -> None:
    assert len(hf.build_time_bars(np.empty(0, np.int64), np.empty(0), np.empty(0), np.empty(0, bool))["close"]) == 0
    t = _trades(2)
    with pytest.raises(ValueError, match="triés"):
        hf.build_time_bars(t["times_ms"][::-1], t["prices"], t["quantities"], t["is_buy"])
    with pytest.raises(ValueError, match="positifs"):
        hf.build_time_bars(t["times_ms"], t["prices"] * 0.0, t["quantities"], t["is_buy"])
    with pytest.raises(ValueError, match="longueur"):
        hf.build_time_bars(t["times_ms"][:-1], t["prices"], t["quantities"], t["is_buy"])


def test_features_match_hand_computed_values() -> None:
    close = 100.0 * np.exp(np.cumsum(np.full(60, 0.001)))  # +0,1 % par barre
    buy, sell = np.full(60, 3.0), np.full(60, 1.0)
    features = hf.price_flow_features(close, buy, sell)
    assert features.shape == (60, 5)
    assert np.isnan(features[:29, 3]).all() and np.isfinite(features[30:]).all()  # vol_30 connue dès la barre 30
    row = features[40]
    assert row[0] == pytest.approx(0.001) and row[1] == pytest.approx(0.005) and row[2] == pytest.approx(0.015)
    assert row[3] == pytest.approx(0.0, abs=1e-12) and row[4] == pytest.approx(0.5)  # (3-1)/(3+1)


def test_features_are_causal_a_later_bar_never_changes_an_earlier_row() -> None:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, 120)))
    buy, sell = rng.uniform(1, 5, 120), rng.uniform(1, 5, 120)
    full = hf.price_flow_features(close, buy, sell)
    truncated = hf.price_flow_features(close[:80], buy[:80], sell[:80])
    assert np.allclose(full[:80], truncated, equal_nan=True)  # aucune ligne ne dépend de ce qui vient après
    altered_future = close.copy()
    altered_future[100:] *= 3.0
    assert np.allclose(hf.price_flow_features(altered_future, buy, sell)[:100], full[:100], equal_nan=True)


def test_a_gap_splits_the_data_and_features_never_span_it() -> None:
    a = _bars(80)
    b = _bars(80, start_ms=1_760_000_040_000 + 200 * 60_000, price0=500.0)  # trou de 120 minutes, prix très différent
    merged = {k: np.concatenate((a[k], b[k])) for k in a}
    runs = hf.contiguous_runs(merged["bar_end_s"])
    assert [(r.start, r.stop) for r in runs] == [(0, 80), (80, 160)]
    out = hf.featurize_runs(merged, min_rows=10)
    assert len(out) == 2 and all(np.isfinite(r["features"]).all() for r in out)
    assert max(abs(r["features"][:, 0]).max() for r in out) < 0.01  # aucun rendement géant à cheval sur le saut 100 -> 500


def test_short_runs_without_enough_usable_rows_are_dropped() -> None:
    bars = _bars(40)  # 40 barres : seules 10 lignes ont leurs mesures (avertissement de 30)
    assert len(hf.featurize_runs(bars, min_rows=11)) == 0
    kept = hf.featurize_runs(bars, min_rows=10)
    assert len(kept) == 1 and len(kept[0]["close"]) == 10 and kept[0]["features"].shape == (10, 5)

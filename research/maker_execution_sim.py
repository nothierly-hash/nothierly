"""Chapter 12, Étape 3.9: Piste 3 -- real queue-position maker-fill simulation.

Diagnosis this responds to (Étape 3.7/3.8, real, measured): the real,
statistically significant directional edge found so far (PT precision
5.04-5.95%, vs. a ~4% base rate) is consistently ~10x short of
profitability against the 0.08% round-trip **taker** fee -- three
independent labeling/feature methodologies all converge on the same
economic wall. Piste 3 attacks the other side of the inequality: instead
of trying to grow the edge further (diminishing returns, per Étape 3.8),
shrink the fee by executing entries as a passive **maker** order instead
of a fee-crossing taker order.

**This is not a free lunch, and is simulated honestly**: a resting limit
order does not fill just because price touches it -- it fills only once
the real volume *ahead of it in the queue* at that price level has been
consumed by real aggressor flow. Assuming instant/certain maker fills
(the common naive mistake) silently erases the real cost of maker
execution: waiting, and sometimes never filling at all, or only filling
when the fill itself is a bad sign (adverse selection -- you get hit
disproportionately when the market is about to move against you, since
that's exactly when aggressor flow is heavy enough to reach your queue
position).

**Real data, real limitation, stated plainly**: this project has L2
order book snapshots (top 25 levels) and trade-by-trade data with a
real exchange-labeled aggressor side -- not L3 (individual order)
data, which the free Tardis sample doesn't include. True queue
position (exact rank among individual resting orders) can't be
reconstructed from this. The standard, defensible academic
approximation used here instead: assume the simulated order joins at
the **back** of the real resting volume already visible at that price
level in the book snapshot (a conservative, worst-case assumption --
real queue position is likely no worse than this), and it is filled
once *real* trades at that exact price, on the consuming side, have
traded a cumulative volume greater than or equal to that starting
queue depth. This also ignores one real effect: if a *better* price
gets posted by someone else after ours, real exchanges give that new,
better-priced order priority over ours even at the same original price
level in some circumstances -- this simulation does not model that, a
real (documented, not hidden) source of optimism in the fill-rate
estimate below.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

_DEFAULT_MAX_WAIT_SECONDS: Final[float] = 60.0


def simulate_maker_fill_times(
    book: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    side: Literal["buy", "sell"],
    max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
    sample_every: int = 1,
) -> pl.DataFrame:
    """Real queue-position fill simulation: for each sampled book row, when (if ever) would a resting order fill?

    Args:
        book: Order book snapshots with ``timestamp`` (microsecond
            epoch), ``bids[0].price``/``bids[0].amount``,
            ``asks[0].price``/``asks[0].amount`` columns.
        trades: Real trade-by-trade data with ``side``, ``price``,
            ``amount``, ``timestamp`` columns (Tardis schema).
        side: ``"buy"`` simulates resting a buy limit order at the
            current best bid (filled by real aggressive **sell**
            trades at that price); ``"sell"`` rests at the best ask
            (filled by real aggressive **buy** trades).
        max_wait_seconds: An order not filled within this real elapsed
            time is treated as never filled (cancelled) -- real market
            makers don't leave orders resting indefinitely as the
            market moves.
        sample_every: Only simulate posting a new order every this-many
            book rows (default every row). No real strategy re-quotes
            at every single L2 update; this also keeps the simulation's
            runtime bounded without changing what's being measured --
            just how densely it's sampled.

    Returns:
        One row per sampled book row, with ``post_timestamp``,
        ``post_price``, ``queue_ahead`` (real resting volume at that
        price level at post time), ``filled`` (bool), and
        ``fill_timestamp``/``fill_delay_seconds`` (``null`` if never
        filled within ``max_wait_seconds``).

    Raises:
        ValueError: If ``side`` is invalid, or required columns are missing.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'.")
    if sample_every < 1:
        raise ValueError("sample_every must be >= 1.")

    price_col, amount_col = ("bids[0].price", "bids[0].amount") if side == "buy" else ("asks[0].price", "asks[0].amount")
    consuming_side = "sell" if side == "buy" else "buy"

    required_book = ["timestamp", price_col, amount_col]
    missing_book = [c for c in required_book if c not in book.columns]
    if missing_book:
        raise ValueError(f"Missing expected book columns: {missing_book}")
    required_trades = ["side", "price", "amount", "timestamp"]
    missing_trades = [c for c in required_trades if c not in trades.columns]
    if missing_trades:
        raise ValueError(f"Missing expected trades columns: {missing_trades}")

    sampled = book.select(["timestamp", price_col, amount_col]).rename(
        {price_col: "post_price", amount_col: "queue_ahead"}
    )[::sample_every]

    # Real, per-price-level cumulative consumed volume, in chronological
    # order -- monotonically non-decreasing within each price group, so
    # a first-crossing search per group is a plain sorted search, not a
    # per-row linear scan.
    consuming_trades = (
        trades.filter(pl.col("side") == consuming_side)
        .sort("timestamp")
        .with_columns(pl.col("amount").cum_sum().over("price").alias("_cum_volume"))
    )

    post_prices: npt.NDArray[np.float64] = sampled["post_price"].to_numpy()
    post_times: npt.NDArray[np.int64] = sampled["timestamp"].to_numpy()
    queue_ahead: npt.NDArray[np.float64] = sampled["queue_ahead"].to_numpy()
    n = len(post_prices)

    fill_timestamp: npt.NDArray[np.float64] = np.full(n, np.nan, dtype=np.float64)
    max_wait_us = max_wait_seconds * 1_000_000.0

    for price_value, group in consuming_trades.group_by("price"):
        price_value = price_value[0] if isinstance(price_value, tuple) else price_value
        row_mask = post_prices == price_value
        if not row_mask.any():
            continue

        g_times: npt.NDArray[np.int64] = group["timestamp"].to_numpy()
        g_cum: npt.NDArray[np.float64] = group["_cum_volume"].to_numpy()
        if len(g_times) == 0:
            continue

        row_indices = np.nonzero(row_mask)[0]
        for idx in row_indices:
            t0 = post_times[idx]
            q = queue_ahead[idx]
            if q <= 0:
                # Nothing resting ahead -- filled by the very next real
                # consuming trade at this price, if any arrives in time.
                base_cum = 0.0
                target = q if q > 0 else np.nextafter(0.0, 1.0)
            else:
                base_position = int(np.searchsorted(g_times, t0, side="right")) - 1
                base_cum = g_cum[base_position] if base_position >= 0 else 0.0
                target = base_cum + q

            fill_position = int(np.searchsorted(g_cum, target, side="left"))
            if fill_position < len(g_cum):
                candidate_time = g_times[fill_position]
                if t0 <= candidate_time <= t0 + max_wait_us:
                    fill_timestamp[idx] = float(candidate_time)

    filled = ~np.isnan(fill_timestamp)
    fill_delay_seconds = np.where(filled, (fill_timestamp - post_times) / 1_000_000.0, np.nan)
    fill_timestamp_or_null: list[int | None] = [int(v) if f else None for v, f in zip(fill_timestamp, filled)]

    return sampled.with_columns(
        pl.Series("filled", filled),
        pl.Series("fill_timestamp", fill_timestamp_or_null, dtype=pl.Int64),
        pl.Series("fill_delay_seconds", fill_delay_seconds),
    ).rename({"timestamp": "post_timestamp"})

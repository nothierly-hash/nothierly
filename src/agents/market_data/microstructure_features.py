"""Real, live VPIN/TFI computation from :class:`~src.agents.market_data.agent.MarketDataAgent`'s real trade tape.

**Closes a real, previously-disclosed gap**: ``src/dashboard/ems_pages/
microstructure.py``'s own module docstring said VPIN and Trade Flow
Imbalance were "not computed anywhere in this project's live ``src/``
pipeline -- both exist only in ``research/alpha_features.py`` (offline
research code)." That real research code is reused here, unchanged,
against the real, now-persisted trade tape (see
:data:`~core.schemas.TRADE_TAPE_REDIS_KEY_PREFIX`) -- not
reimplemented, not approximated.

**Bucket/window sizing, deliberately not the research defaults**:
:func:`research.alpha_features.compute_vpin`'s own default
(``bucket_volume=50.0``) was calibrated for offline historical
datasets spanning far more cumulative volume than a live, capped
~5,000-trade rolling tape will ever hold for most instruments --
using it verbatim here would mean ``vpin`` is null forever in
practice, a "works in the demo, useless in the real dashboard"
gap. This module exposes both real sizing knobs to the caller
(the dashboard's own inputs) instead of silently overriding them
with a second guessed default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

import polars as pl

from core.redis_client import RedisClient
from core.schemas import TRADE_TAPE_REDIS_KEY_PREFIX
from research.alpha_features import compute_tfi, compute_vpin

_DEFAULT_TFI_WINDOWS: Final[tuple[str, ...]] = ("1s", "5s", "30s")


@dataclass(frozen=True)
class LiveMicrostructureReading:
    """One real, current VPIN/TFI reading -- or an honest reason there isn't one yet."""

    symbol: str
    trade_count: int
    vpin: float | None
    tfi_by_window: dict[str, float | None]
    insufficient_data_reason: str | None


async def fetch_real_trade_tape(redis_client: RedisClient, symbol: str, *, limit: int = 5_000) -> list[dict[str, Any]]:
    """Every real trade record currently on :data:`~core.schemas.TRADE_TAPE_REDIS_KEY_PREFIX`'s tape for ``symbol``, newest first."""
    raw_entries = await redis_client.execute("lrange", f"{TRADE_TAPE_REDIS_KEY_PREFIX}{symbol}", 0, limit - 1)
    return [json.loads(entry) for entry in raw_entries]


def compute_live_microstructure_reading(
    trade_tape: list[dict[str, Any]],
    *,
    symbol: str,
    vpin_bucket_volume: float,
    vpin_rolling_buckets: int = 10,
    tfi_windows: tuple[str, ...] = _DEFAULT_TFI_WINDOWS,
) -> LiveMicrostructureReading:
    """Pure: real VPIN/TFI from an already-fetched real trade tape (no I/O here -- see :func:`fetch_real_trade_tape`).

    Args:
        trade_tape: Real trade records, as returned by
            :func:`fetch_real_trade_tape` (each with real
            ``timestamp``/``side``/``amount`` keys).
        symbol: For the returned reading's own label only.
        vpin_bucket_volume: Real traded volume per VPIN bucket -- see
            module docstring on why this must be sized to the real
            instrument's live volume, not the research default.
        vpin_rolling_buckets: Trailing buckets VPIN averages over.
        tfi_windows: Real wall-clock rolling windows for TFI.

    Returns:
        A :class:`LiveMicrostructureReading`. ``vpin``/``tfi_by_window``
        entries are ``None`` (with :attr:`~LiveMicrostructureReading.
        insufficient_data_reason` explaining why) when the real tape
        doesn't yet hold enough data for even one real reading --
        never a fabricated placeholder value.
    """
    if not trade_tape:
        return LiveMicrostructureReading(
            symbol=symbol, trade_count=0, vpin=None, tfi_by_window=dict.fromkeys(tfi_windows),
            insufficient_data_reason="Aucun trade réel accumulé pour l'instant sur ce symbole.",
        )

    trades_df = pl.DataFrame(
        {
            # ccxt's real trade timestamp is epoch milliseconds;
            # research.alpha_features' functions expect epoch
            # microseconds -- a real, deliberate unit conversion, not a
            # research-code change.
            "timestamp": [int(trade["timestamp"]) * 1000 for trade in trade_tape],
            "side": [trade["side"] for trade in trade_tape],
            "amount": [float(trade["amount"]) for trade in trade_tape],
        }
    )

    tfi_result = compute_tfi(trades_df, windows=tfi_windows)
    latest_tfi_row = tfi_result.sort("timestamp").tail(1)
    tfi_by_window: dict[str, float | None] = {
        window: (float(value) if (value := latest_tfi_row[f"tfi_{window}"][0]) is not None else None)
        for window in tfi_windows
    }

    vpin_value: float | None = None
    insufficient_data_reason: str | None = None
    try:
        vpin_result = compute_vpin(trades_df, bucket_volume=vpin_bucket_volume, rolling_buckets=vpin_rolling_buckets)
        non_null_vpin = vpin_result.filter(pl.col("vpin").is_not_null())
        if non_null_vpin.height > 0:
            vpin_value = float(non_null_vpin.sort("bucket_end_timestamp").tail(1)["vpin"][0])
        else:
            insufficient_data_reason = (
                f"Volume réel insuffisant pour {vpin_rolling_buckets} buckets de {vpin_bucket_volume:g} -- "
                "réduisez la taille de bucket ou attendez davantage de trades réels."
            )
    except ValueError as exc:
        insufficient_data_reason = str(exc)

    return LiveMicrostructureReading(
        symbol=symbol, trade_count=len(trade_tape), vpin=vpin_value, tfi_by_window=tfi_by_window,
        insufficient_data_reason=insufficient_data_reason,
    )

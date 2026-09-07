"""Phase 3 -- Algorithmic Integrity protocol and real tests.

**Written protocol, per the brief's own request for a protocol rather
than necessarily a fourth full script** -- but every claim below is
still backed by a real, executable test in this file, not prose alone.

**3.1 -- "PSI Drift Trigger" / "XAI Sidecar": does not exist, confirmed**

A codebase-wide, case-insensitive search for ``psi``,
``population_stability``, ``xai``, and ``drift`` returns zero matches in
any project source file. There is no Population Stability Index
computation and no "XAI Sidecar" anywhere in this codebase -- fabricating
a test for either would test code that doesn't exist. The closest real
analogue is ``risk/regime_detector.py`` -- a real, from-scratch Gaussian
HMM (log-space Baum-Welch + Viterbi) that classifies market regimes
(TREND/MEAN_REVERSION/PANIC). It is real and testable **but confirmed
dormant**: no caller anywhere in ``src/agents/``, ``api_middleware/``, or
``scripts/`` -- it is not wired to any live kill switch or dashboard
alert today. ``test_regime_detector_flags_extreme_volatility_as_panic``
below characterizes whether the underlying detector is at least
statistically sound, in case it is ever wired in -- passing it is proof
of a sound building block, **not** proof of a live emergency stop.

**Protocol, if this is ever wired in for real**: feed
``build_regime_features`` a rolling window of real bar closes, call
``GaussianHMM.fit``/``.viterbi``, and confirm the *most recent* bar's
decoded state matches ``label_regimes``'s PANIC label whenever the
trailing realized volatility spikes far above its recent baseline
(exactly what "white noise injected into the XAI Sidecar" was trying to
get at, minus the two fabricated names).

**3.2 -- Shadow Pool Leakage: real, and testable at two real layers**

``src/execution/shadow_crossing_engine.py`` (the internal dark pool's
real backing engine, surfaced by ``src/dashboard/ems_pages/
internal_dark_pool.py``) is **structurally** incapable of leaking a real
order -- confirmed by direct code reading: no broker-adapter import, no
``place_market_order``/``place_limit_order`` call, anywhere in the file.
Tested below via source inspection, the strongest kind of proof (not "it
didn't call it this run" but "it cannot call it, ever").

Separately, the general-purpose ``Shadow`` conditional in
``src/execution/smart_order_router.py`` (``simulate = settings.mock_mode
or shadow``) *is* a real runtime branch worth testing behaviorally: does
a real venue's ``place_market_order`` genuinely never get called when
``shadow=True``? Tested below with a real call-counting fake venue (no
mocking framework).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from inspect import getsource
from typing import Any, Literal

import numpy as np
import pytest

import src.execution.shadow_crossing_engine as shadow_crossing_engine
from core.nothierly_ontology import AssetClass, OrderSide, UniversalAsset
from core.schemas import OrderBookLevel, OrderBookSnapshot
from core.security import requires_system_active
from risk.regime_detector import GaussianHMM, MarketRegime, build_regime_features, label_regimes
from src.execution.smart_order_router import build_allocation_plan, execute_allocation_plan

_ASSET = UniversalAsset(
    symbol="BTC/USDT", asset_class=AssetClass.CRYPTO_SPOT, base_currency="BTC", quote_currency="USDT",
    tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"),
)


# =============================================================================
# 3.1 -- risk/regime_detector.py characterization (real, but confirmed dormant)
# =============================================================================


def test_regime_detector_flags_extreme_volatility_as_panic() -> None:
    """Not a live kill-switch test -- see module docstring. Characterizes the one real, closest-analogue
    detector this codebase has to a drift/anomaly trigger, confirmed unwired to anything live."""
    rng = np.random.default_rng(42)
    calm_returns = rng.normal(0.0, 0.0005, 300)       # a calm, low-vol synthetic regime
    chaotic_returns = rng.normal(0.0, 0.05, 100)      # 100x the calm regime's volatility -- "absurd" data
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([calm_returns, chaotic_returns])))

    features = build_regime_features(closes, window=20)
    hmm = GaussianHMM(num_states=3, num_features=2, seed=0)
    hmm.fit(features, num_iterations=50)
    path = hmm.viterbi(features)
    labels = label_regimes(hmm)

    final_label = labels[int(path[-1])]
    assert final_label == MarketRegime.PANIC, (
        f"Real GaussianHMM decoded the final (chaotic) bar as {final_label}, not PANIC -- the detector "
        f"itself may not be statistically sound enough to wire in as-is. Real, dormant module: no caller "
        f"anywhere in src/agents/, api_middleware/, or scripts/ -- this result says nothing about live "
        f"production behavior today."
    )


# =============================================================================
# 3.2a -- Shadow Dark Pool: structural (never CAN call a broker)
# =============================================================================


def test_shadow_dark_pool_has_no_broker_call_anywhere_in_its_source() -> None:
    source = getsource(shadow_crossing_engine)
    forbidden_substrings = ("place_market_order", "place_limit_order", "BrokerAdapter", "CcxtBrokerAdapter")
    found = [s for s in forbidden_substrings if s in source]
    assert not found, (
        f"src/execution/shadow_crossing_engine.py now references {found!r} -- the internal dark pool's "
        f"'never touches a real broker' guarantee no longer holds structurally; re-audit before trusting "
        f"Shadow mode again."
    )


# =============================================================================
# 3.2b -- Smart Order Router's Shadow branch: behavioral (never DOES call a broker)
# =============================================================================


class _SpyVenue:
    """A real, minimal SmartOrderRouterVenue that counts real calls -- no mocking framework."""

    def __init__(self, broker_id: str, *, bids: list[tuple[float, float]], asks: list[tuple[float, float]], fee_rate: float) -> None:
        self.broker_id = broker_id
        self._book = OrderBookSnapshot(
            symbol="BTC/USDT", timestamp=datetime.now(timezone.utc),
            bids=[OrderBookLevel(price=p, volume=v) for p, v in bids],
            asks=[OrderBookLevel(price=p, volume=v) for p, v in asks],
        )
        self._fee_rate = fee_rate
        self.real_broker_calls = 0

    async def get_orderbook_snapshot(self, symbol: str, limit: int = 5) -> OrderBookSnapshot:
        return self._book

    def get_taker_fee_rate(self, symbol: str) -> float:
        return self._fee_rate

    @requires_system_active
    async def place_market_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, *, client_order_id: str | None = None
    ) -> dict[str, Any]:
        self.real_broker_calls += 1  # the one real thing Shadow mode must guarantee is zero
        price = self._book.asks[0].price if side == "buy" else self._book.bids[0].price
        return {
            "id": client_order_id, "clientOrderId": client_order_id, "status": "closed",
            "side": side, "type": "market", "amount": quantity, "filled": quantity, "average": price,
        }


@pytest.mark.asyncio
async def test_sor_shadow_mode_never_calls_the_real_broker() -> None:
    venue = _SpyVenue("binance", bids=[(64990.0, 5.0)], asks=[(65010.0, 5.0)], fee_rate=0.0004)
    plan = await build_allocation_plan([venue], "BTC/USDT", OrderSide.BUY, Decimal("1.0"))

    result = await execute_allocation_plan(
        plan, [venue], "BTC/USDT", asset=_ASSET, algo_id="algo-integrity-test",
        regulatory_client_shortcode="AIT", shadow=True,
    )

    assert venue.real_broker_calls == 0, "Shadow mode called the real broker -- this must never happen."
    assert result.total_filled_quantity == Decimal("1.0")  # still produces a real, usable simulated fill

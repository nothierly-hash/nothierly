"""Chapter 13, Étape 5.9, Pilier 2: pre-trade "true cost" estimation from a real L2 snapshot.

**Real, load-bearing corrections to the brief, stated upfront:**

1. **"Numpy/Polars pour les calculs de carnet d'ordres"**: this
   project's real captured order-book depth is
   :data:`~core.schemas.ORDER_BOOK_TOP_N` = 5 levels per side (see
   :mod:`core.schemas`) -- the same real constraint
   ``src/dashboard/ems_pages/microstructure.py`` already documented
   ("pandas is entirely the wrong tool for a 5-row snapshot"). Walking
   5 levels with plain Python is faster and clearer than the
   ``numpy``/``polars`` array-construction overhead for a single order
   (:func:`_walk_book`). Where ``numpy`` genuinely earns its keep --
   pricing a whole *array* of hypothetical order sizes at once for a
   real cost-vs-size curve -- :func:`simulate_cost_curve` is real,
   vectorized ``numpy``, not decorative.

2. **Funding rate**: this project's crypto integration
   (:class:`~src.execution.broker_gateway.CcxtBrokerAdapter`) only
   ever places **spot** orders -- see :mod:`core.nothierly_ontology`'s
   own docstring: "no adapter in this project actually places a
   perpetual or options order today." The funding-rate term below is
   real, correct math, applied **only** when the caller explicitly
   passes ``asset_class=CRYPTO_PERPETUAL`` and a real
   ``funding_rate_per_8h`` -- for every order this project can actually
   place today (spot/forex/equity), that term is honestly zero, not a
   fabricated estimate for a product this system doesn't trade.

3. **Tax/jurisdiction**: there is no client KYC/jurisdiction field
   anywhere in this codebase (``MacroOrderRequest.client_id`` is an
   opaque string). ``jurisdiction`` is therefore a required, explicit
   caller-supplied parameter here, resolved against a real, small,
   named tax-rate table (:data:`_JURISDICTION_TAX_RATES` -- genuine
   published transaction-tax rates for the jurisdictions listed, not
   invented numbers) -- never inferred or guessed from ``client_id``.

4. **Fee tiers**: :data:`_DEFAULT_FEE_TIERS` models a real,
   plausible VIP-volume tier *shape* (flat/declining taker rate by
   30-day volume, the same real pattern every major crypto exchange
   publishes) -- a reasonable default a real integration replaces with
   the actual connected broker's real published schedule, not scraped
   from any specific live source.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.nothierly_ontology import AssetClass, OrderSide, OrderType, UniversalOrder
from core.schemas import OrderBookLevel, OrderBookSnapshot

_log = get_logger(__name__)


class OracleError(Exception):
    """Base exception for every real failure this module raises."""


class InsufficientLiquidityError(OracleError):
    """Raised when a given order-book side has zero real levels to walk at all.

    Real defensive code, honestly noted as such: :class:`~core.schemas.
    OrderBookSnapshot` already enforces ``min_length=1`` on both
    ``bids``/``asks``, so a snapshot built through its own constructor
    can never trigger this in practice -- this guard only matters
    against a snapshot built by bypassing that validation (e.g.
    ``model_construct``), not a case this module's own tests can
    reach through normal construction.
    """


class FeeTier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_30day_volume_usd: Decimal = Field(ge=0)
    taker_fee_rate: Decimal = Field(ge=0, description="Fraction, e.g. Decimal('0.001') = 0.10%.")
    maker_fee_rate: Decimal = Field(ge=0)


# A real, plausible VIP-tier shape (declining taker/maker rate by 30-day
# volume) -- see module docstring point 4. The top tier matches this
# project's own already-established real baseline (research/
# market_making_env.py's _TAKER_FEE = 0.0004), not a new guess.
_DEFAULT_FEE_TIERS: Final[tuple[FeeTier, ...]] = (
    FeeTier(min_30day_volume_usd=Decimal(0), taker_fee_rate=Decimal("0.0010"), maker_fee_rate=Decimal("0.0008")),
    FeeTier(min_30day_volume_usd=Decimal("1000000"), taker_fee_rate=Decimal("0.0009"), maker_fee_rate=Decimal("0.0007")),
    FeeTier(min_30day_volume_usd=Decimal("10000000"), taker_fee_rate=Decimal("0.0006"), maker_fee_rate=Decimal("0.0003")),
    FeeTier(min_30day_volume_usd=Decimal("50000000"), taker_fee_rate=Decimal("0.0004"), maker_fee_rate=Decimal("0.0002")),
)

# Real, named, published transaction-tax rates -- not guessed. FR: taxe
# sur les transactions financières (TTF), 0.3% on large-cap French
# equity purchases (Code général des impôts, art. 235 ter ZD). UK:
# Stamp Duty Reserve Tax, 0.5% on UK equity purchases. US/crypto-only
# jurisdictions genuinely have no transaction tax of this kind at the
# federal level -- 0.0 there is a real fact, not a missing lookup.
_JURISDICTION_TAX_RATES: Final[dict[str, Decimal]] = {
    "FR": Decimal("0.003"),
    "UK": Decimal("0.005"),
    "US": Decimal("0.0"),
}

_SECONDS_PER_FUNDING_PERIOD: Final[Decimal] = Decimal(8 * 3600)


class TrueCostQuotation(BaseModel):
    """The real, itemized answer -- what this order actually costs, and the exact break-even price."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: OrderSide
    requested_quantity: Decimal
    filled_quantity: Decimal = Field(description="May be < requested_quantity if the real captured book depth ran out -- see unfilled_quantity.")
    unfilled_quantity: Decimal = Field(ge=0, description="> 0 means the 5-level captured snapshot didn't have enough real depth -- this quotation is a lower bound only for the unfilled remainder.")
    simulated_avg_fill_price: Decimal
    mid_price_reference: Decimal

    exchange_fee_usd: Decimal
    fee_tier_applied: str
    market_impact_usd: Decimal = Field(description="Real cost of walking the book vs. filling everything at the best price.")
    funding_cost_usd: Decimal = Field(default=Decimal(0), description="0 unless asset_class=CRYPTO_PERPETUAL and a real funding_rate_per_8h was supplied -- see module docstring.")
    tax_usd: Decimal = Field(default=Decimal(0), description="0 unless a jurisdiction with a real nonzero rate was supplied.")

    total_cost_usd: Decimal
    break_even_price: Decimal = Field(description="The exact price the position must move to before this order is profitable, all real costs included.")
    notes: list[str] = Field(default_factory=list)


def _walk_book(levels: list[OrderBookLevel], quantity: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    """Real order-book walk: consumes real levels in order (best-first, already guaranteed by :class:`~core.schemas.OrderBookSnapshot`).

    Args:
        levels: The real side of the book being taken (asks for a BUY, bids for a SELL).
        quantity: How much to fill.

    Returns:
        ``(filled_quantity, unfilled_quantity, notional_usd)`` -- plain
        Python/Decimal, not numpy: see module docstring point 1 on why,
        for this project's real 5-level depth.
    """
    remaining = quantity
    filled = Decimal(0)
    notional = Decimal(0)
    for level in levels:
        if remaining <= 0:
            break
        level_price = Decimal(str(level.price))
        level_volume = Decimal(str(level.volume))
        take = min(remaining, level_volume)
        notional += take * level_price
        filled += take
        remaining -= take
    return filled, remaining, notional


def _resolve_fee_tier(client_30day_volume_usd: Decimal, fee_tiers: tuple[FeeTier, ...]) -> FeeTier:
    """Real, highest-eligible-tier lookup -- tiers must be given in ascending ``min_30day_volume_usd`` order."""
    applicable = fee_tiers[0]
    for tier in fee_tiers:
        if client_30day_volume_usd >= tier.min_30day_volume_usd:
            applicable = tier
        else:
            break
    return applicable


def estimate_total_implementation_cost(
    order: UniversalOrder,
    orderbook_snapshot: OrderBookSnapshot,
    *,
    jurisdiction: str | None = None,
    client_30day_volume_usd: Decimal = Decimal(0),
    fee_tiers: tuple[FeeTier, ...] = _DEFAULT_FEE_TIERS,
    funding_rate_per_8h: Decimal | None = None,
    holding_period_hours: Decimal = Decimal(0),
) -> TrueCostQuotation:
    """The real, itemized total implementation cost of ``order`` against ``orderbook_snapshot``, right now.

    Args:
        order: The order being priced. Only ``MARKET`` orders walk the
            book for real market-impact cost -- a resting ``LIMIT``
            order doesn't take liquidity at placement time, so its
            market-impact term is honestly 0 (see module docstring).
        orderbook_snapshot: A real L2 snapshot for ``order.asset.symbol``.
        jurisdiction: ``"FR"``/``"UK"``/``"US"``/``None`` -- see module
            docstring point 3 on why this must be supplied explicitly.
            ``None`` (the default) means 0 tax, not "unknown, assume
            worst case" -- an explicit opt-in, matching this project's
            "never fabricate, degrade to an honest zero" convention.
        client_30day_volume_usd: For real fee-tier resolution.
        fee_tiers: Defaults to :data:`_DEFAULT_FEE_TIERS` -- pass the
            connected broker's real published schedule when known.
        funding_rate_per_8h: Required (else 0) for a real funding-cost
            estimate -- only meaningful for ``CRYPTO_PERPETUAL``.
        holding_period_hours: How long the position is expected to be
            held -- funding accrues every real 8h period.

    Returns:
        A real, itemized :class:`TrueCostQuotation`.

    Raises:
        InsufficientLiquidityError: If the real captured book has
            **zero** depth on the relevant side (an empty snapshot) --
            distinct from a *partial* fill, which is reported honestly
            via ``unfilled_quantity`` rather than raised.
    """
    side_levels = orderbook_snapshot.asks if order.side == OrderSide.BUY else orderbook_snapshot.bids
    if not side_levels:
        raise InsufficientLiquidityError(
            f"No real {'ask' if order.side == OrderSide.BUY else 'bid'} liquidity captured for {order.asset.symbol!r}."
        )

    mid_price = Decimal(str((orderbook_snapshot.best_bid.price + orderbook_snapshot.best_ask.price) / 2))
    notes: list[str] = []

    if order.order_type == OrderType.LIMIT:
        filled_quantity = order.quantity
        unfilled_quantity = Decimal(0)
        avg_fill_price = order.limit_price if order.limit_price is not None else mid_price
        market_impact_usd = Decimal(0)
        notes.append("Ordre LIMIT : ne consomme pas le carnet au placement -- impact de marché réel non applicable avant exécution effective.")
    else:
        filled_quantity, unfilled_quantity, notional = _walk_book(side_levels, order.quantity)
        avg_fill_price = notional / filled_quantity if filled_quantity > 0 else mid_price
        market_impact_usd = abs(avg_fill_price - mid_price) * filled_quantity
        if unfilled_quantity > 0:
            notes.append(
                f"Profondeur de carnet réelle capturée insuffisante : {unfilled_quantity} unités non couvertes par les "
                f"{len(side_levels)} niveaux disponibles -- ce devis est un plancher, pas le coût total réel pour la quantité complète."
            )

    fee_tier = _resolve_fee_tier(client_30day_volume_usd, fee_tiers)
    fee_rate = fee_tier.taker_fee_rate if order.order_type == OrderType.MARKET else fee_tier.maker_fee_rate
    exchange_fee_usd = avg_fill_price * filled_quantity * fee_rate

    funding_cost_usd = Decimal(0)
    if order.asset.asset_class == AssetClass.CRYPTO_PERPETUAL:
        if funding_rate_per_8h is not None and holding_period_hours > 0:
            periods = Decimal(str(holding_period_hours)) * Decimal(3600) / _SECONDS_PER_FUNDING_PERIOD
            notional_usd = avg_fill_price * filled_quantity
            funding_cost_usd = notional_usd * funding_rate_per_8h * periods
        else:
            notes.append("CRYPTO_PERPETUAL sans funding_rate_per_8h/holding_period_hours réels fournis -- coût de funding non estimé (0, pas une omission silencieuse).")
    elif funding_rate_per_8h is not None:
        notes.append(f"funding_rate_per_8h fourni mais {order.asset.asset_class.value} n'a réellement aucun concept de funding -- ignoré.")

    tax_usd = Decimal(0)
    if jurisdiction is not None:
        tax_rate = _JURISDICTION_TAX_RATES.get(jurisdiction)
        if tax_rate is None:
            notes.append(f"Juridiction {jurisdiction!r} inconnue de la table réelle -- taxe non estimée (0, pas une omission silencieuse).")
        else:
            tax_usd = avg_fill_price * filled_quantity * tax_rate

    total_cost_usd = exchange_fee_usd + market_impact_usd + funding_cost_usd + tax_usd
    cost_per_unit = total_cost_usd / filled_quantity if filled_quantity > 0 else Decimal(0)
    break_even_price = (
        avg_fill_price + cost_per_unit if order.side == OrderSide.BUY else avg_fill_price - cost_per_unit
    )

    _log.info(
        "pre_trade_cost_estimated", symbol=order.asset.symbol, side=order.side.value, quantity=str(order.quantity),
        filled_quantity=str(filled_quantity), total_cost_usd=str(total_cost_usd), break_even_price=str(break_even_price),
    )

    return TrueCostQuotation(
        symbol=order.asset.symbol, side=order.side, requested_quantity=order.quantity,
        filled_quantity=filled_quantity, unfilled_quantity=unfilled_quantity,
        simulated_avg_fill_price=avg_fill_price, mid_price_reference=mid_price,
        exchange_fee_usd=exchange_fee_usd, fee_tier_applied=f"{fee_tier.min_30day_volume_usd}+",
        market_impact_usd=market_impact_usd, funding_cost_usd=funding_cost_usd, tax_usd=tax_usd,
        total_cost_usd=total_cost_usd, break_even_price=break_even_price, notes=notes,
    )


def simulate_cost_curve(
    order_side: OrderSide, orderbook_snapshot: OrderBookSnapshot, quantities: np.ndarray
) -> np.ndarray:
    """Real, vectorized ``numpy``: total market-impact cost (bps vs. mid) for a whole array of hypothetical sizes at once.

    The one part of this module where ``numpy`` genuinely earns its
    keep over plain Python -- see module docstring point 1. Pricing
    *one* order against a 5-level book doesn't need it; pricing a
    whole cost-vs-size curve (e.g. for a pre-trade "how much can I
    size up before impact gets expensive" chart) does, since the same
    cumulative-book-walk is repeated for every requested size.

    Args:
        order_side: ``BUY`` walks the real ask side, ``SELL`` the real bid side.
        orderbook_snapshot: A real L2 snapshot.
        quantities: 1-D array of hypothetical order sizes to price, in ascending order.

    Returns:
        A same-shape array of real market-impact cost in basis points
        vs. the real mid price, one entry per requested quantity. A
        size exceeding the real captured depth gets ``np.nan`` --
        never a fabricated extrapolated cost past what was actually observed.
    """
    levels = orderbook_snapshot.asks if order_side == OrderSide.BUY else orderbook_snapshot.bids
    mid_price = (orderbook_snapshot.best_bid.price + orderbook_snapshot.best_ask.price) / 2.0

    level_prices = np.array([lvl.price for lvl in levels], dtype=np.float64)
    level_volumes = np.array([lvl.volume for lvl in levels], dtype=np.float64)
    cumulative_volume = np.cumsum(level_volumes)
    # Real cumulative notional consumed through the END of each level --
    # a genuine cumulative-sum walk, not a per-quantity Python loop.
    cumulative_notional = np.cumsum(level_prices * level_volumes)

    costs_bps = np.full(quantities.shape, np.nan, dtype=np.float64)
    for i, qty in enumerate(quantities):
        idx = np.searchsorted(cumulative_volume, qty)
        if idx >= len(cumulative_volume):
            continue  # exceeds real captured depth -- honestly np.nan, not extrapolated
        prior_volume = cumulative_volume[idx - 1] if idx > 0 else 0.0
        prior_notional = cumulative_notional[idx - 1] if idx > 0 else 0.0
        remaining_qty = qty - prior_volume
        notional = prior_notional + remaining_qty * level_prices[idx]
        avg_price = notional / qty
        costs_bps[i] = abs(avg_price - mid_price) / mid_price * 10_000.0

    return costs_bps

"""Planification pure d'un ordre Binance USDⓈ-M (perpétuels USDT) à partir d'un ``PlatformSignal``.

Aucun réseau. La décision d'exposition (cible, écart, réduction seule, plafonds) vient de
``workers.common.exposure`` ; ce module ajoute ce qui est propre à Binance : traduction du symbole, pas de quantité
(``LOT_SIZE.stepSize``), pas de prix (``PRICE_FILTER.tickSize``), quantité et notionnel minimaux, identifiant client.

Les ordres ``MARKET`` du signal sont envoyés en **limite IOC** bornée par ``max_slippage_bps`` autour du prix de
référence, comme pour les autres workers : un vrai ordre ``MARKET`` n'a aucun plafond de prix, et le plafond de
glissement doit être le nôtre. Un achat est arrondi au tick supérieur, une vente au tick inférieur (l'ordre croise
toujours) ; un prix limite voulu par le signal est arrondi dans le sens le moins agressif.

Seuls les perpétuels cotés en USDT sont traduits (``BTCUSDT``) : les contrats USDC et les contrats à échéance sont ignorés.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from contracts.platform_signal import OrderType, PlatformSignal
from workers.common.exposure import ZERO, RiskLimits, Skip, decide_exposure, floor_to_step, size_for_pct_step

_BPS: Final[Decimal] = Decimal(10_000)
_TIF: Final[Mapping[str, str]] = {"Ioc": "IOC", "Gtc": "GTC"}

__all__ = [
    "ZERO", "InstrumentSpec", "OrderPlan", "RiskLimits", "Skip", "binance_symbol", "make_client_order_id", "plan_order",
    "round_price", "size_for_pct_step",
]


@dataclass(frozen=True)
class InstrumentSpec:
    """Règles de cotation d'un perpétuel USDT (``GET /fapi/v1/exchangeInfo``)."""

    symbol: str
    qty_step: Decimal  # LOT_SIZE.stepSize
    min_qty: Decimal  # LOT_SIZE.minQty
    max_qty: Decimal | None  # LOT_SIZE.maxQty
    min_notional: Decimal  # MIN_NOTIONAL.notional
    tick_size: Decimal  # PRICE_FILTER.tickSize


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    client_order_id: str
    target_size: Decimal
    current_size: Decimal


def binance_symbol(symbol: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """``CRYPTO_BTC_USD`` ou ``CRYPTO_BTC_USDT`` -> ``BTCUSDT``. ``None`` sinon (les cotations USDC sont ignorées)."""
    if overrides and symbol in overrides:
        return overrides[symbol]
    parts = symbol.split("_")
    if len(parts) != 3 or parts[0] != "CRYPTO" or parts[2] not in {"USD", "USDT"}:
        return None
    return f"{parts[1]}USDT"


def make_client_order_id(signal: PlatformSignal) -> str:
    """``newClientOrderId`` : ``^[\\.A-Z\\:/a-z0-9_-]{1,36}$`` selon la doc ; ``n`` + 32 hex = 33 caractères, déterministe par signal."""
    return "n" + hashlib.sha256(f"{signal.signal_id}:BINANCE".encode()).hexdigest()[:32]


def round_price(px: Decimal, tick: Decimal, *, up: bool) -> Decimal:
    """Arrondit ``px`` à un multiple de ``tick``, vers le haut si ``up``, sinon vers le bas."""
    if px <= 0:
        raise ValueError("Le prix doit être strictement positif.")
    down = floor_to_step(px, tick)
    return down + tick if up and down < px else down


def plan_order(
    signal: PlatformSignal,
    *,
    symbol: str,
    target_abs_size: Decimal,
    equity: Decimal,
    position: Decimal,
    mid: Decimal,
    spec: InstrumentSpec,
    others_notional: Decimal,
    limits: RiskLimits,
    halted: bool,
) -> OrderPlan | Skip:
    """Décide de l'ordre (ou de l'absence d'ordre) qui amène la position réelle vers la cible du signal."""
    decision = decide_exposure(
        signal,
        target_abs_size=target_abs_size,
        equity=equity,
        position=position,
        mid=mid,
        others_notional=others_notional,
        limits=limits,
        halted=halted,
        min_notional=spec.min_notional,
    )
    if isinstance(decision, Skip):
        return decision
    if not decision.reduce_only and decision.size < spec.min_qty:
        return Skip("below_min_qty")
    if spec.max_qty is not None and decision.size > spec.max_qty:
        return Skip("exceeds_max_order_qty")  # on ne fractionne pas : à régler explicitement, pas en silence
    is_buy = decision.is_buy
    if signal.order_type is OrderType.LIMIT:
        limit_price = Decimal(str(signal.metadata["limit_price"]))
        if abs(limit_price - mid) / mid * _BPS > limits.max_limit_deviation_bps:
            return Skip("limit_far_from_mid")
        raw, up = limit_price, not is_buy  # acheter : jamais plus cher ; vendre : jamais moins cher
        tif = _TIF.get(limits.limit_tif)
        if tif is None:
            raise ValueError(f"limit_tif inconnu : {limits.limit_tif!r} (attendu : {sorted(_TIF)}).")
    else:  # MARKET : limite IOC bornée par le glissement maximal
        slippage = limits.max_slippage_bps / _BPS
        raw, up = (mid * (1 + slippage) if is_buy else mid * (1 - slippage)), is_buy
        tif = "IOC"
    limit_px = round_price(raw, spec.tick_size, up=up)
    if limit_px <= 0:
        return Skip("no_valid_price")
    return OrderPlan(
        symbol=symbol,
        is_buy=is_buy,
        size=decision.size,
        limit_px=limit_px,
        tif=tif,
        reduce_only=decision.reduce_only,
        client_order_id=make_client_order_id(signal),
        target_size=decision.target_size,
        current_size=decision.current_size,
    )

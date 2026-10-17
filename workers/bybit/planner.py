"""Planification pure d'un ordre Bybit (perpétuels linéaires USDT) à partir d'un ``PlatformSignal``.

Aucun réseau. La décision d'exposition (cible, écart, réduction seule, plafonds) vient de
``workers.common.exposure`` ; ce module ajoute ce qui est propre à Bybit : traduction du symbole, pas de
quantité (``qtyStep``), pas de prix (``tickSize``), quantité et notionnel minimaux, identifiant d'ordre.

Les ordres ``MARKET`` du signal sont envoyés en **limite IOC** bornée par ``max_slippage_bps`` autour du
prix de référence : le plafond de glissement est le nôtre, pas celui, interne, de Bybit. Un prix d'achat
est arrondi au tick supérieur et un prix de vente au tick inférieur (l'ordre croise toujours) ; un prix
limite voulu par le signal est arrondi dans le sens le moins agressif (jamais plus cher qu'annoncé).
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
    "ZERO", "InstrumentSpec", "OrderPlan", "RiskLimits", "Skip", "bybit_symbol", "make_order_link_id", "plan_order",
    "round_price", "size_for_pct_step",
]


@dataclass(frozen=True)
class InstrumentSpec:
    """Règles de cotation d'un perpétuel linéaire USDT (``GET /v5/market/instruments-info``)."""

    symbol: str
    qty_step: Decimal
    min_qty: Decimal
    max_qty: Decimal | None
    min_notional: Decimal
    tick_size: Decimal
    copy_trading: str  # "none" = non éligible au copy trading ; autres valeurs : voir la doc Bybit


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    order_link_id: str
    target_size: Decimal
    current_size: Decimal


def bybit_symbol(symbol: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """``CRYPTO_BTC_USD`` ou ``CRYPTO_BTC_USDT`` -> ``BTCUSDT``. ``None`` sinon.

    Seuls les perpétuels USDT sont éligibles au copy trading (doc Bybit V5) : les cotations USDC sont refusées.
    Les symboles particuliers (``1000PEPEUSDT``...) passent par ``overrides``.
    """
    if overrides and symbol in overrides:
        return overrides[symbol]
    parts = symbol.split("_")
    if len(parts) != 3 or parts[0] != "CRYPTO" or parts[2] not in {"USD", "USDT"}:
        return None
    return f"{parts[1]}USDT"


def make_order_link_id(signal: PlatformSignal) -> str:
    """``orderLinkId`` (36 caractères max, alphanumérique), déterministe pour un même signal : ``n`` + 32 hex."""
    return "n" + hashlib.sha256(f"{signal.signal_id}:BYBIT".encode()).hexdigest()[:32]


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
        order_link_id=make_order_link_id(signal),
        target_size=decision.target_size,
        current_size=decision.current_size,
    )


"""Planification pure d'un ordre Hyperliquid à partir d'un ``PlatformSignal`` et de l'état réel du compte.

Aucun réseau ici : tout est déterministe et testable avec des chiffres calculés à la main.

Sémantique : ``position_size_pct`` est une **exposition cible** (en % de l'equity), pas un ordre.
Le planificateur compare la cible à la position réelle et n'émet que l'écart. Conséquence voulue :
rejouer ou republier le même signal à chaque tick ne fait rien (``on_target``), au lieu d'empiler des
positions. ``FLAT`` vise une position nulle ; une réduction pure est toujours ``reduce_only`` (un
ordre de sortie ne peut donc jamais ouvrir de position inverse par accident).

Règles d'Hyperliquid appliquées (doc officielle, perps) : prix à 5 chiffres significatifs au plus et
au plus ``6 - szDecimals`` décimales, les prix entiers étant toujours permis ; tailles multiples de
``10^-szDecimals``. Les tailles sont arrondies vers zéro : on n'envoie jamais plus que la cible.

``min_order_notional_usd`` (10 USD) est une valeur connue de la plateforme que je n'ai pas pu
reconfirmer dans la doc consultée ; elle est paramétrable et, de toute façon, l'échange rejette.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Final

from contracts.platform_signal import OrderType, PlatformSignal, SignalDirection

ZERO: Final[Decimal] = Decimal(0)
_MAX_DECIMALS_PERP: Final[int] = 6
_QUOTE_SUFFIXES: Final[frozenset[str]] = frozenset({"USD", "USDC", "USDT"})
_BPS: Final[Decimal] = Decimal(10_000)


@dataclass(frozen=True)
class RiskLimits:
    """Limites locales du worker : il juge sur l'état réel du compte, jamais sur ce que dit le moteur."""

    max_position_pct: Decimal = Decimal("20")  # |exposition cible| / equity, par actif
    max_gross_exposure_pct: Decimal = Decimal("100")  # somme des |expositions| / equity
    rebalance_tolerance_pct: Decimal = Decimal("0.5")  # écart minimal (en % d'equity) pour réajuster
    min_order_notional_usd: Decimal = Decimal("10")
    max_slippage_bps: Decimal = Decimal("50")  # ordres MARKET : IOC limité à mid +/- ce plafond
    max_limit_deviation_bps: Decimal = Decimal("500")  # un LIMIT trop loin du mid est une faute de frappe
    limit_tif: str = "Ioc"  # "Gtc" laisserait des ordres au carnet si le worker s'arrête : opt-in
    max_daily_drawdown_pct: Decimal = Decimal("5")  # au-delà : mode réduction seule


@dataclass(frozen=True)
class OrderPlan:
    asset: str
    is_buy: bool
    size: Decimal
    limit_px: Decimal
    tif: str
    reduce_only: bool
    cloid: str
    target_size: Decimal
    current_size: Decimal


@dataclass(frozen=True)
class Skip:
    reason: str


def hl_asset(symbol: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """``CRYPTO_BTC_USD`` -> ``BTC``. ``None`` si le symbole n'est pas de la forme ``CRYPTO_<BASE>_<USD|USDC|USDT>``.

    Les noms particuliers d'Hyperliquid (``kPEPE``...) passent par ``overrides``.
    """
    if overrides and symbol in overrides:
        return overrides[symbol]
    parts = symbol.split("_")
    if len(parts) != 3 or parts[0] != "CRYPTO" or parts[2] not in _QUOTE_SUFFIXES:
        return None
    return parts[1]


def make_cloid(signal: PlatformSignal) -> str:
    """Identifiant d'ordre client de 128 bits (``0x`` + 32 hex), déterministe pour un même signal."""
    return "0x" + hashlib.sha256(f"{signal.signal_id}:HYPERLIQUID".encode()).hexdigest()[:32]


def size_for_pct(equity: Decimal, mid: Decimal, pct: Decimal, sz_decimals: int) -> Decimal:
    """Taille absolue (unités de l'actif) pour ``pct`` % de l'equity, arrondie vers zéro à ``szDecimals``."""
    if equity <= 0 or mid <= 0 or pct < 0:
        return ZERO
    return (equity * pct / Decimal(100) / mid).quantize(Decimal(1).scaleb(-sz_decimals), rounding=ROUND_DOWN)


def round_price(px: Decimal, sz_decimals: int) -> Decimal:
    """Arrondit ``px`` à un prix valide : 5 chiffres significatifs, au plus ``6 - szDecimals`` décimales,
    prix entiers toujours permis."""
    if px <= 0:
        raise ValueError("Le prix doit être strictement positif.")
    if px == px.to_integral_value():
        return px.quantize(Decimal(1))
    decimals = min(_MAX_DECIMALS_PERP - sz_decimals, 4 - px.adjusted())  # 4 - exposant = 5 chiffres significatifs
    return px.quantize(Decimal(1).scaleb(-max(decimals, 0)), rounding=ROUND_HALF_UP)


def _is_pure_reduction(position: Decimal, target: Decimal) -> bool:
    return (position > 0 and 0 <= target < position) or (position < 0 and position < target <= 0)


def plan_order(
    signal: PlatformSignal,
    *,
    asset: str,
    target_abs_size: Decimal,
    equity: Decimal,
    position: Decimal,
    mid: Decimal,
    sz_decimals: int,
    others_notional: Decimal,
    limits: RiskLimits,
    halted: bool,
) -> OrderPlan | Skip:
    """Décide de l'ordre (ou de l'absence d'ordre) qui amène la position réelle vers la cible du signal.

    Args:
        target_abs_size: taille absolue visée (``size_for_pct``) ; le signe vient de ``direction``.
        position: taille signée réelle de l'actif (positif = long).
        others_notional: exposition absolue des AUTRES actifs, en USD (pour le plafond brut).
        halted: mode réduction seule (coupe-circuit de drawdown) : seules les réductions passent.
    """
    if signal.order_type is OrderType.PREDICTION_ONLY:
        return Skip("prediction_only")
    if equity <= 0:
        return Skip("no_equity")
    if mid <= 0:
        return Skip("no_price")
    if Decimal(str(signal.position_size_pct)) > limits.max_position_pct:
        return Skip("exceeds_position_cap")

    sign = {SignalDirection.LONG: 1, SignalDirection.SHORT: -1, SignalDirection.FLAT: 0}[signal.direction]
    target = target_abs_size * sign
    delta = target - position
    if delta == 0:
        return Skip("on_target")

    reduction = _is_pure_reduction(position, target)
    delta_notional = abs(delta) * mid
    if target != 0 and delta_notional < limits.rebalance_tolerance_pct / Decimal(100) * equity:
        return Skip("within_tolerance")
    if halted and not reduction:
        return Skip("halted_reduce_only")
    if not reduction:
        if delta_notional < limits.min_order_notional_usd:
            return Skip("below_min_notional")
        if others_notional + abs(target) * mid > limits.max_gross_exposure_pct / Decimal(100) * equity:
            return Skip("exceeds_gross_cap")

    is_buy = delta > 0
    if signal.order_type is OrderType.LIMIT:
        limit_price = Decimal(str(signal.metadata["limit_price"]))
        if abs(limit_price - mid) / mid * _BPS > limits.max_limit_deviation_bps:
            return Skip("limit_far_from_mid")
        raw, tif = limit_price, limits.limit_tif
    else:  # MARKET : IOC borné par le slippage maximal, jamais un ordre sans prix
        slippage = limits.max_slippage_bps / _BPS
        raw, tif = (mid * (1 + slippage) if is_buy else mid * (1 - slippage)), "Ioc"
    return OrderPlan(
        asset=asset,
        is_buy=is_buy,
        size=abs(delta),
        limit_px=round_price(raw, sz_decimals),
        tif=tif,
        reduce_only=reduction,
        cloid=make_cloid(signal),
        target_size=target,
        current_size=position,
    )

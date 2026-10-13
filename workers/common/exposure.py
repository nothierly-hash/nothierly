"""Décision d'exposition cible, indépendante de la plateforme.

``position_size_pct`` d'un ``PlatformSignal`` est une exposition **cible** (en % de l'equity), pas un
ordre à ajouter : on compare la cible à la position réelle et on n'émet que l'écart. Republier le même
signal ne fait donc rien (``on_target``) au lieu d'empiler des positions. ``FLAT`` vise une position
nulle ; une réduction pure est toujours ``reduce_only`` (un ordre de sortie ne peut jamais ouvrir de
position inverse par accident).

Ce module ne connaît ni prix limites, ni pas de cotation, ni identifiants d'ordre : cela dépend de la
plateforme (voir ``workers/hyperliquid/planner.py`` et ``workers/bybit/planner.py``). Il porte la
logique de risque commune, pour qu'elle n'existe qu'à un seul endroit : plafonds d'exposition, tolérance
de rééquilibrage, notionnel minimal, mode réduction seule.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Final

from contracts.platform_signal import OrderType, PlatformSignal, SignalDirection

ZERO: Final[Decimal] = Decimal(0)


@dataclass(frozen=True)
class RiskLimits:
    """Limites locales d'un worker : il juge sur l'état réel du compte, jamais sur ce que dit le moteur."""

    max_position_pct: Decimal = Decimal("20")  # |exposition cible| / equity, par actif
    max_gross_exposure_pct: Decimal = Decimal("100")  # somme des |expositions| / equity
    rebalance_tolerance_pct: Decimal = Decimal("0.5")  # écart minimal (en % d'equity) pour réajuster
    min_order_notional_usd: Decimal = Decimal("10")
    max_slippage_bps: Decimal = Decimal("50")  # ordres MARKET : IOC limité à mid +/- ce plafond
    max_limit_deviation_bps: Decimal = Decimal("500")  # un LIMIT trop loin du mid est une faute de frappe
    limit_tif: str = "Ioc"  # "Gtc" laisserait des ordres au carnet si le worker s'arrête : opt-in
    max_daily_drawdown_pct: Decimal = Decimal("5")  # au-delà : mode réduction seule


@dataclass(frozen=True)
class Skip:
    reason: str


@dataclass(frozen=True)
class ExposureDecision:
    is_buy: bool
    size: Decimal  # valeur absolue de l'écart à combler
    reduce_only: bool
    target_size: Decimal
    current_size: Decimal


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Arrondit ``value`` vers zéro au multiple de ``step`` : on n'envoie jamais plus que la cible."""
    if step <= 0:
        raise ValueError("step doit être strictement positif.")
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def size_for_pct_step(equity: Decimal, mid: Decimal, pct: Decimal, qty_step: Decimal) -> Decimal:
    """Taille absolue pour ``pct`` % de l'equity, arrondie vers zéro au pas de quantité ``qty_step``."""
    if equity <= 0 or mid <= 0 or pct < 0:
        return ZERO
    return floor_to_step(equity * pct / Decimal(100) / mid, qty_step)


def _is_pure_reduction(position: Decimal, target: Decimal) -> bool:
    return (position > 0 and 0 <= target < position) or (position < 0 and position < target <= 0)


def decide_exposure(
    signal: PlatformSignal,
    *,
    target_abs_size: Decimal,
    equity: Decimal,
    position: Decimal,
    mid: Decimal,
    others_notional: Decimal,
    limits: RiskLimits,
    halted: bool,
    min_notional: Decimal | None = None,
) -> ExposureDecision | Skip:
    """Décide s'il faut ordonner, et de combien, pour amener la position réelle vers la cible du signal.

    Args:
        target_abs_size: taille absolue visée (déjà arrondie au pas de la plateforme) ; le signe vient de ``direction``.
        position: taille signée réelle de l'actif (positif = long).
        others_notional: exposition absolue des AUTRES actifs, en USD (pour le plafond brut).
        halted: mode réduction seule (coupe-circuit de drawdown) : seules les réductions passent.
        min_notional: notionnel minimal propre à la plateforme ; sinon ``limits.min_order_notional_usd``.
            Le plus grand des deux s'applique.
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
        floor = max(limits.min_order_notional_usd, min_notional or ZERO)
        if delta_notional < floor:
            return Skip("below_min_notional")
        if others_notional + abs(target) * mid > limits.max_gross_exposure_pct / Decimal(100) * equity:
            return Skip("exceeds_gross_cap")
    return ExposureDecision(
        is_buy=delta > 0, size=abs(delta), reduce_only=reduction, target_size=target, current_size=position
    )

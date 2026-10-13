"""Planification pure d'un ordre MetaTrader 5 (Darwinex Zero) à partir d'un ``PlatformSignal``.

Aucun réseau ni terminal ici. La décision d'exposition (cible, écart, réduction seule, plafonds) vient de
``workers.common.exposure`` ; ce module ajoute ce qui est propre à MT5 : lots, pas de volume, stop-loss serveur,
niveaux de stop du courtier, déviation en points, politique de remplissage, spread.

**Valeur d'un lot.** Un lot Forex n'a pas la valeur « prix x quantité » : elle dépend de la taille de contrat et de la
conversion de la devise de cotation vers celle du compte. Le client la mesure (``notional_per_lot``, via
``order_calc_profit``) et ce module l'utilise comme « valeur d'une unité » dans ``decide_exposure`` : un lot vaut
``notional_per_lot`` unités de compte.

**Pas d'inversion en un ordre.** Sur un compte hedging, un ordre opposé plus gros que la position OUVRE une position
inverse sans fermer la première. Une inversion LONG -> SHORT est donc planifiée comme la fermeture seule (avec le ticket
de la position) ; le signal suivant, état-pilote, ouvre le reste.

**Stop-loss.** Le score de risque Darwinex impose un stop sur toute position. Tout ordre d'ouverture part avec un stop
côté serveur (si le worker ou le pont meurt, la position n'est pas nue). Le ``stop_loss_price`` du signal est utilisé
s'il existe ; il est refusé (``stop_too_close_or_wrong_side``) s'il est du mauvais côté du prix ou plus près que
``trade_stops_level``, **jamais remplacé en silence**. S'il manque, le worker pose un stop par défaut à
``default_stop_loss_bps`` du prix de l'ordre (prix limite pour un ``LIMIT``, sinon ask/bid), reculé au besoin pour
respecter la distance minimale du courtier ; sans ``default_stop_loss_bps`` l'ordre est refusé (``missing_stop_loss``).
Un stop n'est posé qu'à l'ouverture : un changement ultérieur sur une position déjà à la cible n'est pas appliqué.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Final

from contracts.platform_signal import OrderType, PlatformSignal, SignalDirection
from workers.common.exposure import ZERO, RiskLimits, Skip, decide_exposure, floor_to_step

_BPS: Final[Decimal] = Decimal(10_000)
_FILL_FOK: Final[int] = 1  # bits de SYMBOL_FILLING_MODE (doc MQL5)
_FILL_IOC: Final[int] = 2

__all__ = [
    "ZERO", "Mt5OrderPlan", "Mt5Position", "Quote", "RiskLimits", "Skip", "SymbolSpec", "choose_filling",
    "default_stop_loss", "deviation_points", "lots_for_pct", "make_comment", "mt5_symbol", "plan_order",
]


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    digits: int
    point: Decimal
    volume_min: Decimal
    volume_max: Decimal
    volume_step: Decimal
    stops_level: int  # distance minimale d'un stop, en points
    trade_mode: str  # FULL | LONGONLY | SHORTONLY | CLOSEONLY | DISABLED
    execution_mode: str  # REQUEST | INSTANT | MARKET | EXCHANGE | UNKNOWN
    filling_flags: int  # bits SYMBOL_FILLING_* (FOK = 1, IOC = 2)


@dataclass(frozen=True)
class Quote:
    bid: Decimal
    ask: Decimal


@dataclass(frozen=True)
class Mt5Position:
    ticket: int
    volume: Decimal  # signé : positif = achat
    price_current: Decimal


@dataclass(frozen=True)
class Mt5OrderPlan:
    symbol: str
    is_buy: bool
    volume: Decimal
    price: Decimal  # prix de référence (ask pour un achat, bid pour une vente), ou prix limite
    deviation_points: int
    sl: Decimal | None
    filling: str  # IOC | FOK | RETURN
    reduce_only: bool
    position_ticket: int | None
    comment: str
    target_lots: Decimal
    current_lots: Decimal
    sl_defaulted: bool = False  # le stop vient du worker, pas du signal

    @property
    def size(self) -> Decimal:  # satisfait PlannedOrder (journal)
        return self.volume

    @property
    def limit_px(self) -> Decimal:
        return self.price


def mt5_symbol(symbol: str, suffix: str = "", overrides: Mapping[str, str] | None = None) -> str | None:
    """``FOREX_EUR_USD`` -> ``EURUSD`` (+ suffixe du courtier). ``None`` si ce n'est pas une paire ``FOREX_XXX_YYY``."""
    if overrides and symbol in overrides:
        return overrides[symbol]
    parts = symbol.split("_")
    if len(parts) != 3 or parts[0] != "FOREX" or not all(p.isalnum() and 2 <= len(p) <= 5 for p in parts[1:]):
        return None
    return f"{parts[1]}{parts[2]}{suffix}"


def choose_filling(spec: SymbolSpec) -> str | None:
    """Politique de remplissage valide pour ce symbole (table de la doc MQL5), ou ``None`` si aucune ne convient.

    Exécution Instant/Request : IOC et FOK toujours permis. Market : selon les drapeaux du symbole, RETURN interdit.
    Exchange : IOC/FOK selon les drapeaux, RETURN toujours permis.
    """
    if spec.execution_mode in ("INSTANT", "REQUEST"):
        return "IOC"
    if spec.filling_flags & _FILL_IOC:
        return "IOC"
    if spec.filling_flags & _FILL_FOK:
        return "FOK"
    return None if spec.execution_mode == "MARKET" else "RETURN"


def lots_for_pct(equity: Decimal, notional_per_lot: Decimal, pct: Decimal, spec: SymbolSpec) -> Decimal:
    """Lots pour ``pct`` % de l'equity, arrondis vers zéro au pas de volume (jamais plus que la cible)."""
    if equity <= 0 or notional_per_lot <= 0 or pct < 0:
        return ZERO
    return floor_to_step(equity * pct / Decimal(100) / notional_per_lot, spec.volume_step)


def deviation_points(price: Decimal, slippage_bps: Decimal, spec: SymbolSpec) -> int:
    """Déviation maximale en points pour tolérer ``slippage_bps`` autour de ``price`` (arrondie au point supérieur)."""
    return max(1, math.ceil(price * slippage_bps / _BPS / spec.point))


def make_comment(signal_id: object) -> str:
    """Commentaire d'ordre (limite MT5 prudente : 17 caractères) : ``n`` + 16 hex, déterministe pour un signal."""
    return "n" + hashlib.sha256(f"{signal_id}:DARWINEX".encode()).hexdigest()[:16]


def default_stop_loss(is_buy: bool, ref_price: Decimal, quote: Quote, spec: SymbolSpec, bps: Decimal) -> Decimal:
    """Stop par défaut à ``bps`` du prix de l'ordre, jamais plus près que la distance minimale du courtier.

    Achat : sous le prix, au plus ``bid - distance`` ; vente : au-dessus, au moins ``ask + distance``. L'arrondi au point
    va toujours dans le sens qui éloigne le stop (plus protecteur, et ne peut pas violer la distance minimale).
    """
    distance = spec.stops_level * spec.point
    if is_buy:
        return _round_to_digits(min(ref_price * (1 - bps / _BPS), quote.bid - distance), spec, up=False)
    return _round_to_digits(max(ref_price * (1 + bps / _BPS), quote.ask + distance), spec, up=True)


def _round_to_digits(value: Decimal, spec: SymbolSpec, *, up: bool | None = None) -> Decimal:
    quantum = Decimal(1).scaleb(-spec.digits)
    if up is None:
        return value.quantize(quantum, rounding=ROUND_HALF_UP)
    down = value.quantize(quantum, rounding=ROUND_DOWN)
    return down + quantum if up and down < value else down


def plan_order(
    signal: PlatformSignal,
    *,
    symbol: str,
    target_abs_lots: Decimal,
    equity: Decimal,
    position: Mt5Position | None,
    quote: Quote,
    spec: SymbolSpec,
    notional_per_lot: Decimal,
    others_notional: Decimal,
    limits: RiskLimits,
    halted: bool,
    max_spread_bps: Decimal,
    default_stop_loss_bps: Decimal | None = None,
) -> Mt5OrderPlan | Skip:
    """Décide de l'ordre (ou de l'absence d'ordre) qui amène la position réelle vers la cible du signal."""
    current = position.volume if position is not None else ZERO
    sign = {SignalDirection.LONG: 1, SignalDirection.SHORT: -1, SignalDirection.FLAT: 0}[signal.direction]
    if spec.trade_mode == "DISABLED":
        return Skip("symbol_trading_disabled")
    filling = choose_filling(spec)
    if filling is None:
        return Skip("no_valid_filling")
    if sign != 0 and current == 0 and signal.order_type is not OrderType.PREDICTION_ONLY and target_abs_lots < spec.volume_min:
        return Skip("below_min_volume")  # le compte est trop petit pour détenir un lot minimal à ce pourcentage

    decision = decide_exposure(
        signal, target_abs_size=target_abs_lots, equity=equity, position=current, mid=notional_per_lot,
        others_notional=others_notional, limits=limits, halted=halted,
    )
    if isinstance(decision, Skip):
        return decision

    is_buy, size, reduce_only = decision.is_buy, decision.size, decision.reduce_only
    if current != 0 and decision.target_size * current < 0:  # inversion : on ferme seulement (voir la docstring du module)
        is_buy, size, reduce_only = current < 0, abs(current), True
    mid = (quote.bid + quote.ask) / 2

    if not reduce_only:
        if spec.trade_mode == "CLOSEONLY":
            return Skip("close_only")
        if spec.trade_mode == "LONGONLY" and not is_buy:
            return Skip("long_only")
        if spec.trade_mode == "SHORTONLY" and is_buy:
            return Skip("short_only")
        if size < spec.volume_min:
            return Skip("below_min_volume")
        if size > spec.volume_max:
            return Skip("exceeds_max_volume")  # on ne fractionne pas une ouverture
        if (quote.ask - quote.bid) / mid * _BPS > max_spread_bps:
            return Skip("spread_too_wide")
    else:
        size = min(size, spec.volume_max)  # une fermeture trop grosse se fait en plusieurs tours

    if signal.order_type is OrderType.LIMIT and not reduce_only:
        limit_price = Decimal(str(signal.metadata["limit_price"]))
        if abs(limit_price - mid) / mid * _BPS > limits.max_limit_deviation_bps:
            return Skip("limit_far_from_mid")
        price, deviation = _round_to_digits(limit_price, spec, up=not is_buy), 0  # acheter : jamais plus cher
    else:
        price = _round_to_digits(quote.ask if is_buy else quote.bid, spec)
        deviation = deviation_points(price, limits.max_slippage_bps, spec)

    sl: Decimal | None = None
    sl_defaulted = False
    if not reduce_only:
        raw_sl = signal.metadata.get("stop_loss_price")
        if raw_sl is not None:
            sl = _round_to_digits(Decimal(str(raw_sl)), spec)
            distance = spec.stops_level * spec.point
            if (is_buy and sl > quote.bid - distance) or (not is_buy and sl < quote.ask + distance):
                return Skip("stop_too_close_or_wrong_side")  # le stop du signal n'est jamais remplacé en silence
        elif default_stop_loss_bps is None:
            return Skip("missing_stop_loss")
        else:
            sl, sl_defaulted = default_stop_loss(is_buy, price, quote, spec, default_stop_loss_bps), True
    return Mt5OrderPlan(
        symbol=symbol, is_buy=is_buy, volume=size, price=price, deviation_points=deviation, sl=sl, filling=filling,
        reduce_only=reduce_only, position_ticket=position.ticket if (reduce_only and position is not None) else None,
        comment=make_comment(signal.signal_id), target_lots=decision.target_size, current_lots=current,
        sl_defaulted=sl_defaulted,
    )

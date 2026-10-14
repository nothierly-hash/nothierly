"""Horloge de volume et VPIN : le calcul pur, sans réseau, sans Redis, sans horloge murale.

**VPIN** (Easley, López de Prado, O'Hara, 2012) mesure la toxicité du flux d'ordres : à quel point les volumes agressifs
sont déséquilibrés d'un côté. Le temps est mesuré en **volume**, pas en secondes : un seau se ferme quand ``bucket_size``
(notionnel, par exemple 100 000 $) a été échangé, vite en marché actif, lentement en marché calme.

    VPIN = somme des |achat - vente| des ``window_size`` derniers seaux / (``window_size`` x ``bucket_size``)

La valeur est dans ``[0, 1]`` : 0 = flux équilibré, 1 = tous les seaux sont à sens unique.

**Deux précisions par rapport à la version naïve (seau fermé dès que le volume est atteint ou dépassé) :**

- **Un trade qui déborde est découpé.** Un gros trade peut remplir la fin d'un seau et en commencer, voire en remplir,
  d'autres. Sans découpage un seau dépasse ``bucket_size``, ce qui fausse le dénominateur (``window_size x bucket_size`` suppose
  des seaux exacts) et attribue à un seul seau un déséquilibre qui en vaut plusieurs. Ici chaque seau fermé vaut EXACTEMENT
  ``bucket_size``.
- **Pas de valeur avant d'avoir une fenêtre complète.** Divisé par ``window_size x bucket_size`` avec moins de seaux, le
  résultat est sous-estimé (3 seaux sur 50 donneraient un VPIN près de 0 par construction). ``vpin`` vaut ``None`` tant que
  la fenêtre n'est pas pleine ; ``warmup`` dit combien il manque.

**Sens du trade.** Il vient de la plateforme (``is_buy`` = le preneur a acheté), pas d'une règle de tick. Le VPIN ne dépend
que de ``|achat - vente|`` : échanger les deux étiquettes ne change pas le résultat. Une erreur de convention sur le sens (ce qui
est le cas possible pour Hyperliquid, dont la doc ne définit pas ``side``) est donc sans effet, pourvu qu'elle soit constante.

Arithmétique en ``Decimal`` : des seaux exacts, pas de dérive d'arrondi flottant.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

ZERO: Final[Decimal] = Decimal(0)


@dataclass(frozen=True)
class VpinReading:
    """Un seau vient de se fermer."""

    vpin: Decimal | None  # ``None`` tant que la fenêtre n'est pas pleine
    imbalance: Decimal  # |achat - vente| de CE seau
    buy_volume: Decimal
    sell_volume: Decimal
    closed_at_ms: int  # horodatage du trade qui a fermé le seau
    buckets_in_window: int


class VolumeBucketVpin:
    def __init__(self, bucket_size: Decimal, window_size: int) -> None:
        if not bucket_size.is_finite() or bucket_size <= 0:
            raise ValueError("bucket_size doit être un nombre positif.")
        if window_size < 1:
            raise ValueError("window_size doit être >= 1.")
        self._bucket_size = bucket_size
        self._window = window_size
        self._imbalances: deque[Decimal] = deque(maxlen=window_size)
        self._buy = ZERO
        self._sell = ZERO
        self.buckets_closed = 0

    @property
    def bucket_size(self) -> Decimal:
        return self._bucket_size

    @property
    def window_size(self) -> int:
        return self._window

    @property
    def partial_volume(self) -> Decimal:
        """Volume déjà accumulé dans le seau en cours (toujours < ``bucket_size``)."""
        return self._buy + self._sell

    @property
    def warmup(self) -> int:
        """Nombre de seaux qui manquent avant la première valeur (0 = fenêtre pleine)."""
        return self._window - len(self._imbalances)

    @property
    def vpin(self) -> Decimal | None:
        if len(self._imbalances) < self._window:
            return None
        return sum(self._imbalances, ZERO) / (self._window * self._bucket_size)

    def reset(self) -> None:
        """Repart de zéro (un trou dans le flux rend la fenêtre en cours trompeuse)."""
        self._imbalances.clear()
        self._buy = self._sell = ZERO
        self.buckets_closed = 0

    def add_trade(self, price: Decimal, qty: Decimal, is_buy: bool, time_ms: int = 0) -> list[VpinReading]:
        """Ajoute un trade (notionnel = prix x quantité) ; renvoie les seaux qu'il ferme (zéro, un, ou plusieurs).

        Lève ``ValueError`` pour un prix ou une quantité qui n'est pas strictement positif et fini : un trade douteux
        n'est jamais compté à moitié.
        """
        if not (price.is_finite() and qty.is_finite()) or price <= 0 or qty <= 0:
            raise ValueError("prix et quantité doivent être strictement positifs et finis.")
        remaining = price * qty
        closed: list[VpinReading] = []
        while remaining > 0:
            take = min(remaining, self._bucket_size - self.partial_volume)
            if is_buy:
                self._buy += take
            else:
                self._sell += take
            remaining -= take
            if self.partial_volume >= self._bucket_size:
                imbalance = abs(self._buy - self._sell)
                self._imbalances.append(imbalance)
                self.buckets_closed += 1
                closed.append(VpinReading(self.vpin, imbalance, self._buy, self._sell, time_ms, len(self._imbalances)))
                self._buy = self._sell = ZERO
        return closed

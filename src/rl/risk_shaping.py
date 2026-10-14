"""Aversion au risque de l'agent PPO face aux régimes toxiques : fonctions pures, sans PyTorch ni Gymnasium.

Trois leviers, tous NEUTRES par défaut (une configuration vide reproduit exactement l'environnement d'avant) :

1. **Mise en forme de la récompense.**

   - *Taxe de toxicité* : ``lambda x max(0, VPIN - seuil) x |position|`` retirée à chaque pas passé en position, même si le trade est gagnant à cet instant. On utilise
     bien l'EXCÈS au-dessus du seuil (la formule), pas le VPIN brut : sous le seuil le coût est nul, sinon l'agent serait taxé en permanence, y compris en marché sain.
   - *Pénalité de stop non linéaire* : une perte de ``L`` bps touchée au stop reçoit en plus ``w x (L (L / stop)^(alpha - 1) - L)``. **Pourquoi pas ``|L|^alpha`` brut** : la
     récompense de l'environnement est un rendement logarithmique (de l'ordre de 1e-3). Une puissance 1,5 d'un nombre inférieur à 1 donne un résultat PLUS PETIT
     (``0,004^1,5 = 2,5e-4 < 0,004``) : la pénalité adoucirait la perte au lieu de la durcir. La forme normalisée vaut exactement ``L`` à ``L = stop`` (alpha neutre au
     stop nominal) et ``L x (L/stop)^(alpha-1)`` au-delà : x1,41 à deux fois le stop avec alpha = 1,5 ; c'est le glissement extrême qu'elle punit.
   - *Bonus de refuge* : récompense ``safe_harbor_bonus`` quand l'agent est à plat alors que le VPIN dépasse le seuil.

2. **Représentation de l'état** : ``vpin_derivatives`` fournit le niveau, la vitesse (VPIN(t) - VPIN(t - w)) et l'accélération (VPIN(t) - 2 VPIN(t - w) + VPIN(t - 2w)). Tout est
   CAUSAL : chaque valeur vient du dernier relevé daté d'avant ou à l'instant ``t`` (jamais d'un relevé futur), et un relevé plus vieux que ``max_age_s`` est NaN, jamais
   présenté comme frais (un flux gelé ne doit pas passer pour un VPIN stable).

3. **Masque d'actions déterministe** : au-delà du VPIN de coupe-circuit, ``BUY`` est interdit ; si la position est ouverte ``HOLD`` l'est aussi (seule la sortie reste permise),
   à plat on peut rester à plat. ``SELL`` n'est JAMAIS masqué : dans cet environnement long seulement, ``SELL`` est la SORTIE vers le plat ; la masquer piégerait l'agent dans
   sa position pendant la toxicité, exactement l'inverse du but.

Non fourni : l'OBI profond (volumes limites à 50 bps du spread). Le schéma de marché archivé (``MarketMetrics``) ne contient qu'un déséquilibre sur les premiers niveaux
du carnet ; la profondeur à 50 bps n'est mesurée nulle part, donc aucune fonction ne la fabrique ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

ACTION_HOLD: Final[int] = 0
ACTION_BUY: Final[int] = 1
ACTION_SELL: Final[int] = 2
N_ACTIONS: Final[int] = 3
BPS: Final[float] = 10_000.0

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class RiskShapingConfig:
    """Réglages d'aversion au risque. Les poids sont dans l'unité de la récompense de l'environnement (rendement logarithmique par pas)."""

    vpin_threshold: float = 0.7
    toxicity_penalty_weight: float = 0.0        # lambda ; 0 = pas de taxe
    stop_loss_bps: float | None = None          # None = pas de stop dans l'environnement
    tail_risk_weight: float = 0.0               # w ; 0 = pas de pénalité supplémentaire sur les stops
    tail_alpha: float = 1.5
    safe_harbor_bonus: float = 0.0
    kill_switch_vpin: float | None = None       # None = pas de masque

    def __post_init__(self) -> None:
        if not 0 <= self.vpin_threshold <= 1:
            raise ValueError("vpin_threshold doit être dans [0, 1]")
        if min(self.toxicity_penalty_weight, self.tail_risk_weight, self.safe_harbor_bonus) < 0:
            raise ValueError("les poids de pénalité et le bonus ne peuvent pas être négatifs")
        if self.stop_loss_bps is not None and self.stop_loss_bps <= 0:
            raise ValueError("stop_loss_bps doit être > 0")
        if self.tail_alpha < 1:
            raise ValueError("tail_alpha doit être >= 1 (sous 1 la pénalité diminue quand la perte grandit)")
        if self.tail_risk_weight > 0 and self.stop_loss_bps is None:
            raise ValueError("tail_risk_weight > 0 exige stop_loss_bps (sans stop, aucun événement à pénaliser)")
        if self.kill_switch_vpin is not None and not 0 < self.kill_switch_vpin <= 1:
            raise ValueError("kill_switch_vpin doit être dans ]0, 1]")

    @property
    def needs_vpin(self) -> bool:
        """Vrai si une partie de la récompense ou du masque dépend du VPIN : l'environnement doit alors recevoir la série."""
        return self.toxicity_penalty_weight > 0 or self.safe_harbor_bonus > 0 or self.kill_switch_vpin is not None


def toxicity_tax(vpin: float, position: float, config: RiskShapingConfig) -> float:
    """``lambda x max(0, VPIN - seuil) x |position|`` : coût de rester exposé dans un marché toxique (toujours >= 0)."""
    return float(config.toxicity_penalty_weight * max(0.0, vpin - config.vpin_threshold) * abs(position))


def stop_penalty(loss_bps: float, config: RiskShapingConfig) -> float:
    """Pénalité SUPPLÉMENTAIRE d'un stop dont la perte réalisée est ``loss_bps`` (>= stop) ; nulle sans stop ou sans poids.

    ``w x (L (L / stop)^(alpha - 1) - L) / 1e4`` : voir l'en-tête du module pour la normalisation. Toujours >= 0 pour ``L >= stop``.
    """
    if config.stop_loss_bps is None or config.tail_risk_weight <= 0 or loss_bps <= 0:
        return 0.0
    ratio = max(loss_bps / config.stop_loss_bps, 1.0)
    return float(config.tail_risk_weight * loss_bps * (float(ratio) ** (config.tail_alpha - 1.0) - 1.0) / BPS)


def safe_harbor_reward(action: int, position_after: int, vpin: float, config: RiskShapingConfig) -> float:
    """Bonus quand l'agent est à plat après son action alors que le VPIN dépasse le seuil."""
    if config.safe_harbor_bonus > 0 and position_after == 0 and vpin > config.vpin_threshold:
        return config.safe_harbor_bonus
    return 0.0


def compute_action_mask(position: float, vpin: float | None, kill_switch_vpin: float | None) -> BoolArray:
    """Actions permises ``[HOLD, BUY, SELL]``. Tout est permis sans coupe-circuit, sans VPIN connu... sauf qu'un VPIN INCONNU n'est pas un VPIN sain : voir ci-dessous.

    VPIN connu au-dessus du coupe-circuit : ``BUY`` interdit ; ``HOLD`` interdit si on est en position (sortie forcée) ; ``SELL`` toujours permis.
    VPIN inconnu (``None`` / NaN) avec coupe-circuit configuré : on ne sait pas, donc on n'ouvre pas (``BUY`` interdit) mais on ne force pas une sortie.
    """
    mask = np.ones(N_ACTIONS, dtype=np.bool_)
    if kill_switch_vpin is None:
        return mask
    unknown = vpin is None or not np.isfinite(vpin)
    if unknown:
        mask[ACTION_BUY] = False
        return mask
    assert vpin is not None
    if vpin > kill_switch_vpin:
        mask[ACTION_BUY] = False
        if position > 0:
            mask[ACTION_HOLD] = False
    return mask


def safe_action(position: float) -> int:
    """L'action de repli quand une action demandée est masquée : sortir si on est en position, sinon rester à plat."""
    return ACTION_SELL if position > 0 else ACTION_HOLD


def vpin_derivatives(
    query_times_s: FloatArray, vpin_times_s: FloatArray, vpin_values: FloatArray, *, window_s: float = 60.0, max_age_s: float = 600.0,
) -> FloatArray:
    """``(T, 3)`` : niveau, vitesse, accélération du VPIN à chaque instant de ``query_times_s``. NaN là où une valeur n'est pas disponible.

    ``vpin_times_s`` doit être trié croissant. Causal : la valeur à l'instant ``q`` est celle du dernier relevé ``<= q`` ; la vitesse compare ``q`` à ``q - window_s`` et l'accélération
    à ``q - 2 window_s`` (même règle). Un relevé de plus de ``max_age_s`` avant l'instant demandé est considéré absent (NaN).
    """
    if window_s <= 0 or max_age_s <= 0:
        raise ValueError("window_s et max_age_s doivent être > 0")
    times = np.asarray(vpin_times_s, dtype=np.float64)
    values = np.asarray(vpin_values, dtype=np.float64)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("vpin_times_s et vpin_values doivent avoir la même forme (N,)")
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError("vpin_times_s doit être trié par ordre croissant")
    query = np.asarray(query_times_s, dtype=np.float64)

    def value_at(instants: FloatArray) -> FloatArray:
        out = np.full(instants.shape, np.nan)
        if times.size == 0:
            return out
        idx = np.searchsorted(times, instants, side="right") - 1
        valid = (idx >= 0) & (instants - times[np.clip(idx, 0, None)] <= max_age_s)
        out[valid] = values[idx[valid]]
        return out

    level = value_at(query)
    previous = value_at(query - window_s)
    before_previous = value_at(query - 2.0 * window_s)
    return np.column_stack([level, level - previous, level - 2.0 * previous + before_previous])

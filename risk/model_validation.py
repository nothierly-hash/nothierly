"""Validation statistique des modèles : découpage chronologique avec embargo, échantillon indépendant, borne de Hanley-McNeil.

Trois outils, partagés par ``scripts/train_model.py`` (XGBoost primaire) et ``scripts/train_metalabel.py`` (méta-modèle) pour que les
deux appliquent exactement les mêmes règles :

**1. Découpage chronologique avec embargo (anti-anticipation).** Les lignes sont triées par temps ; les premières ``1 - test_fraction``
servent à l'entraînement, les dernières au test, jamais mélangées. Entre les deux, un bloc d'**embargo** est retiré : il doit couvrir au
moins l'horizon de l'étiquette (l'étiquette d'une ligne d'entraînement proche de la coupure regarde ``horizon`` dans le futur, donc
dans le test) et la fenêtre des variables à mémoire (moyenne glissante de sentiment). Sans lui, la fin de l'entraînement et le début
du test partagent de l'information et l'évaluation est optimiste.

**2. Échantillon indépendant.** Des lignes espacées de quelques secondes ont des étiquettes qui se recouvrent presque entièrement
(horizon de 15 minutes) : 100 000 lignes de test valent en réalité quelques centaines d'observations. Le test statistique ci-dessous
suppose des observations indépendantes ; on le fait donc sur un sous-échantillon dont les lignes sont espacées d'au moins un horizon
(``independent_indices``). Sans cette correction, la variance de l'AUC est sous-estimée d'un facteur énorme et n'importe quel AUC
légèrement supérieur à 0,5 paraît « significatif ».

**3. Borne inférieure de l'AUC (Hanley-McNeil, unilatérale à 99 %).** L'erreur-type de Hanley et McNeil (1982) donne une borne
``AUC - z x SE`` (z = 2,326). Le modèle ne bat le hasard que si cette borne dépasse 0,5 ET si le test nul de Mann-Whitney
(``null_z_score``) dépasse aussi z : l'erreur-type de Hanley-McNeil s'annule quand l'AUC vaut 1, ce qui rendrait « significatif »
un échantillon minuscule parfaitement séparé. La formule suppose l'indépendance (point 2)
et une AUC calculée hors échantillon (point 1) ; elle ne corrige PAS les essais multiples : tester dix variantes de modèle et garder la
meilleure invalide le seuil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

Z_99_ONE_SIDED = 2.326


def auc_lower_bound(auc: float, n_pos: int, n_neg: int, z: float = Z_99_ONE_SIDED) -> float:
    """Borne inférieure unilatérale (99 % par défaut) d'une AUC, par l'erreur-type de Hanley-McNeil.

    ``0.0`` si l'un des deux effectifs est inférieur à 2 : aucune conclusion possible.
    """
    if n_pos < 2 or n_neg < 2:
        return 0.0
    q1, q2 = auc / (2.0 - auc), 2.0 * auc * auc / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return auc - z * math.sqrt(max(variance, 0.0))


def split_chronological(n: int, test_fraction: float, embargo_rows: int = 0) -> tuple[int, int]:
    """``(train_end, test_start)`` pour ``n`` lignes déjà triées : entraînement ``[0, train_end)``, embargo ``[train_end, test_start)``, test ``[test_start, n)``."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction doit être strictement entre 0 et 1.")
    if embargo_rows < 0:
        raise ValueError("embargo_rows doit être >= 0.")
    train_end = max(1, int(n * (1.0 - test_fraction)))
    test_start = train_end + embargo_rows
    if test_start >= n:
        raise ValueError(f"Aucune ligne de test : {n} lignes, entraînement {train_end}, embargo {embargo_rows}.")
    return train_end, test_start


def split_by_time(times_ns: npt.NDArray[np.int64], test_fraction: float, embargo_ns: int) -> tuple[int, int]:
    """Comme :func:`split_chronological`, mais l'embargo est une DURÉE : le test commence à la première ligne dont le temps atteint la fin de l'entraînement + ``embargo_ns``.

    Lève ``ValueError`` si ``times_ns`` n'est pas trié par ordre croissant (un découpage chronologique d'une série mélangée serait faux
    en silence).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction doit être strictement entre 0 et 1.")
    if embargo_ns < 0:
        raise ValueError("embargo_ns doit être >= 0.")
    if len(times_ns) < 2 or bool(np.any(np.diff(times_ns) < 0)):
        raise ValueError("Les temps doivent être triés par ordre croissant (au moins 2 lignes).")
    train_end = max(1, int(len(times_ns) * (1.0 - test_fraction)))
    boundary = int(times_ns[train_end - 1]) + embargo_ns
    test_start = int(np.searchsorted(times_ns, boundary, side="left"))
    test_start = max(test_start, train_end)
    if test_start >= len(times_ns):
        raise ValueError("Aucune ligne de test : l'embargo couvre tout le reste des données.")
    return train_end, test_start


def independent_indices(times_ns: npt.NDArray[np.int64], min_gap_ns: int) -> npt.NDArray[np.int64]:
    """Indices d'un sous-échantillon dont les lignes sont espacées d'au moins ``min_gap_ns`` (la première est retenue, puis la suivante assez éloignée, etc.)."""
    if min_gap_ns <= 0:
        return np.arange(len(times_ns), dtype=np.int64)
    kept: list[int] = []
    last: int | None = None
    for index, moment in enumerate(times_ns.tolist()):
        if last is None or moment - last >= min_gap_ns:
            kept.append(index)
            last = moment
    return np.array(kept, dtype=np.int64)


def split_purged_intervals(starts_ns: npt.NDArray[np.int64], ends_ns: npt.NDArray[np.int64], test_fraction: float) -> tuple[int, int]:
    """``(train_end, test_start)`` pour des TRADES triés par ouverture : le test commence au premier trade qui s'ouvre après la fermeture du dernier trade d'entraînement.

    Purge par intervalles : un trade d'entraînement encore ouvert quand un trade de test s'ouvre partage ses conditions de marché. Plus
    stricte qu'un embargo de durée fixe pour des trades de durées très inégales. ``test_start`` peut valoir ``len(starts_ns)`` (aucun
    trade de test) : à l'appelant de le traiter comme un refus.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction doit être strictement entre 0 et 1.")
    if len(starts_ns) != len(ends_ns) or len(starts_ns) < 2:
        raise ValueError("Il faut au moins 2 trades, avec autant de fermetures que d'ouvertures.")
    if bool(np.any(np.diff(starts_ns) < 0)):
        raise ValueError("Les trades doivent être triés par ordre d'ouverture croissant.")
    train_end = max(1, int(round(len(starts_ns) * (1.0 - test_fraction))))
    last_exit = int(np.max(ends_ns[:train_end]))
    test_start = train_end
    while test_start < len(starts_ns) and int(starts_ns[test_start]) < last_exit:
        test_start += 1
    return train_end, test_start


def non_overlapping_indices(starts_ns: npt.NDArray[np.int64], ends_ns: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Indices (dans l'ordre d'origine) d'un sous-ensemble de trades dont les durées ne se chevauchent pas.

    Pour des TRADES, deux observations sont indépendantes quand leurs durées de détention ne se recouvrent pas (un trade qui s'ouvre
    pendant qu'un autre est ouvert subit les mêmes conditions de marché). Glouton par ordre d'ouverture : on retient un trade s'il
    s'ouvre à l'instant où le dernier retenu est fermé ou après.
    """
    if len(starts_ns) != len(ends_ns):
        raise ValueError("starts_ns et ends_ns doivent avoir la même longueur.")
    kept: list[int] = []
    last_end: int | None = None
    for index in np.argsort(starts_ns, kind="stable").tolist():
        if last_end is None or int(starts_ns[index]) >= last_end:
            kept.append(index)
            last_end = int(ends_ns[index])
    return np.array(sorted(kept), dtype=np.int64)


@dataclass(frozen=True)
class AucSignificance:
    n_total: int
    n_independent: int
    n_positive: int
    n_negative: int
    auc_full: float | None  # sur toutes les lignes de test (indicatif : lignes corrélées)
    auc_independent: float | None  # sur le sous-échantillon indépendant : celle du test
    lower_bound: float  # borne inférieure à 99 % de ``auc_independent`` (0.0 si indéfinie)
    null_z: float = 0.0  # (AUC - 0,5) / écart-type sous l'hypothèse nulle (Mann-Whitney) ; 0.0 si indéfini
    z: float = Z_99_ONE_SIDED

    @property
    def significant(self) -> bool:
        """Bat le hasard à 99 % unilatéral : la borne de Hanley-McNeil dépasse 0,5 ET le test nul de Mann-Whitney le confirme.

        Le second test existe parce que l'erreur-type de Hanley-McNeil, faite pour un intervalle AUTOUR d'une AUC estimée, tombe à zéro
        quand l'AUC vaut 1 : deux positifs et deux négatifs parfaitement séparés donneraient une borne de 1,0. L'écart-type sous
        l'hypothèse nulle, lui, dépend seulement des effectifs et reste large sur un petit échantillon.
        """
        return self.lower_bound > 0.5 and self.null_z > self.z


def null_z_score(auc: float, n_pos: int, n_neg: int) -> float:
    """``(AUC - 0,5) / SE0`` avec ``SE0 = sqrt((n_pos + n_neg + 1) / (12 n_pos n_neg))`` : l'écart-type de l'AUC si le modèle n'a aucun pouvoir (Mann-Whitney)."""
    if n_pos < 1 or n_neg < 1:
        return 0.0
    return (auc - 0.5) / math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


def evaluate_auc_significance(
    y_true: npt.NDArray[np.int64],
    scores: npt.NDArray[np.float64],
    times_ns: npt.NDArray[np.int64],
    *,
    horizon_ns: int = 0,
    z: float = Z_99_ONE_SIDED,
    keep: npt.NDArray[np.int64] | None = None,
) -> AucSignificance:
    """AUC hors échantillon et sa borne de Hanley-McNeil, calculées sur des lignes espacées d'au moins ``horizon_ns``.

    ``keep`` (indices) remplace le critère d'espacement par un sous-échantillon choisi par l'appelant, par exemple les trades qui ne
    se chevauchent pas (:func:`non_overlapping_indices`).
    """
    if not (len(y_true) == len(scores) == len(times_ns)):
        raise ValueError("y_true, scores et times_ns doivent avoir la même longueur.")
    full = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None
    keep = independent_indices(times_ns, horizon_ns) if keep is None else keep
    y_kept, score_kept = y_true[keep], scores[keep]
    n_pos = int(y_kept.sum())
    n_neg = int(len(y_kept) - n_pos)
    if n_pos < 2 or n_neg < 2:
        return AucSignificance(len(y_true), len(keep), n_pos, n_neg, full, None, 0.0)
    auc = float(roc_auc_score(y_kept, score_kept))
    return AucSignificance(len(y_true), len(keep), n_pos, n_neg, full, auc, auc_lower_bound(auc, n_pos, n_neg, z), null_z_score(auc, n_pos, n_neg), z)

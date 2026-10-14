"""Tests de ``risk/model_validation.py`` : découpage chronologique avec embargo, échantillon indépendant, borne de Hanley-McNeil.

Chiffres faits à la main pour les découpages ; jeux synthétiques pour la significativité. L'idée centrale testée : sur des lignes
espacées de quelques secondes dont les étiquettes se recouvrent, le test sur toutes les lignes déclare significatif un AUC qui n'est que du
bruit, alors que le test sur un échantillon indépendant ne le fait pas.
"""

from __future__ import annotations

import numpy as np
import pytest

from risk.model_validation import (
    auc_lower_bound,
    evaluate_auc_significance,
    independent_indices,
    split_by_time,
    split_chronological,
)

SECOND = 1_000_000_000


# --- Découpage par lignes --------------------------------------------------------------------


def test_the_split_keeps_time_order_with_a_gap_of_exactly_the_embargo() -> None:
    assert split_chronological(100, 0.2, embargo_rows=0) == (80, 80)
    assert split_chronological(100, 0.2, embargo_rows=5) == (80, 85)  # entraînement [0, 80), embargo [80, 85), test [85, 100)
    assert split_chronological(10, 0.2, embargo_rows=1) == (8, 9)


@pytest.mark.parametrize("n, fraction, embargo", [(100, 0.0, 0), (100, 1.0, 0), (100, 0.2, -1), (100, 0.2, 20), (100, 0.2, 99)])
def test_invalid_splits_are_refused_instead_of_producing_an_empty_or_leaking_test_set(n: int, fraction: float, embargo: int) -> None:
    with pytest.raises(ValueError):
        split_chronological(n, fraction, embargo)


# --- Découpage par durée ---------------------------------------------------------------------


def test_a_time_based_embargo_skips_rows_by_duration_not_by_count() -> None:
    times = (np.arange(100, dtype=np.int64) * SECOND)  # une ligne par seconde
    assert split_by_time(times, 0.2, 10 * SECOND) == (80, 89)  # fin d'entraînement à t = 79 s ; le test commence à t >= 89 s
    irregular = np.concatenate([np.arange(80, dtype=np.int64) * SECOND, np.array([85, 100, 130, 131, 140, 141, 150, 151, 160, 161, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179], dtype=np.int64) * SECOND])
    train_end, test_start = split_by_time(irregular, 0.2, 10 * SECOND)
    assert train_end == 80
    assert test_start == 81  # fin d'entraînement 79 s + 10 s = 89 s : la ligne de 85 s (indice 80) est dans l'embargo, la première >= 89 s est celle de 100 s
    assert irregular[test_start] >= irregular[train_end - 1] + 10 * SECOND > irregular[test_start - 1]


def test_no_embargo_means_the_test_starts_right_after_the_training_rows() -> None:
    times = np.arange(50, dtype=np.int64) * SECOND
    assert split_by_time(times, 0.2, 0) == (40, 40)


def test_an_unsorted_series_or_an_embargo_that_swallows_the_test_set_is_refused() -> None:
    shuffled = np.array([3, 1, 2, 5, 4, 7, 6, 9, 8, 10], dtype=np.int64) * SECOND
    with pytest.raises(ValueError, match="triés"):
        split_by_time(shuffled, 0.2, 0)  # un « découpage chronologique » d'une série mélangée serait faux en silence
    with pytest.raises(ValueError, match="Aucune ligne de test"):
        split_by_time(np.arange(100, dtype=np.int64) * SECOND, 0.2, 60 * SECOND)
    with pytest.raises(ValueError):
        split_by_time(np.array([1], dtype=np.int64), 0.2, 0)
    with pytest.raises(ValueError):
        split_by_time(np.arange(10, dtype=np.int64), 0.2, -1)


# --- Échantillon indépendant -----------------------------------------------------------------


def test_independent_rows_are_spaced_by_at_least_the_gap() -> None:
    times = np.array([0, 1, 2, 10, 11, 25], dtype=np.int64) * SECOND
    assert independent_indices(times, 10 * SECOND).tolist() == [0, 3, 5]  # 0 s, puis 10 s, puis 25 s (>= 20 s)
    assert independent_indices(times, 0).tolist() == [0, 1, 2, 3, 4, 5]
    assert independent_indices(np.array([], dtype=np.int64), SECOND).tolist() == []


# --- Hanley-McNeil ---------------------------------------------------------------------------


def test_the_lower_bound_is_tight_for_big_samples_wide_for_small_ones_and_zero_when_undefined() -> None:
    assert auc_lower_bound(0.5, 100, 100) < 0.5
    assert auc_lower_bound(0.9, 1000, 1000) > 0.88
    assert auc_lower_bound(0.6, 20, 20) < auc_lower_bound(0.6, 2000, 2000)
    assert auc_lower_bound(0.9, 1, 50) == 0.0 and auc_lower_bound(0.9, 50, 0) == 0.0
    assert auc_lower_bound(0.7, 100, 100, z=1.645) > auc_lower_bound(0.7, 100, 100, z=2.326)  # plus de confiance = borne plus basse


def test_a_real_signal_is_significant_and_pure_noise_is_not() -> None:
    rng = np.random.default_rng(0)
    n = 600
    times = np.arange(n, dtype=np.int64) * 900 * SECOND  # une observation par horizon : déjà indépendantes
    y = rng.integers(0, 2, n).astype(np.int64)
    signal = evaluate_auc_significance(y, y + rng.normal(0, 0.6, n), times, horizon_ns=900 * SECOND)
    assert signal.n_independent == n and signal.significant and signal.auc_independent is not None and signal.auc_independent > 0.8
    noise = evaluate_auc_significance(y, rng.normal(size=n), times, horizon_ns=900 * SECOND)
    assert not noise.significant and noise.auc_independent is not None and noise.auc_independent < 0.58


def test_overlapping_rows_make_noise_look_significant_unless_the_sample_is_thinned() -> None:
    """Pur bruit, mais lent : 20 000 lignes à 1 s d'écart, étiquette ET score constants sur 15 minutes (variables à mémoire).

    Il n'y a qu'environ 22 observations indépendantes. Le test classique sur les 20 000 lignes conclut « significatif » dans près de la moitié
    des tirages de bruit ; le test sur un échantillon espacé d'un horizon n'y est presque jamais pris.
    """
    n, horizon = 20_000, 900
    times = np.arange(n, dtype=np.int64) * SECOND
    blocks = np.arange(n) // horizon
    naive_false_positives = independent_false_positives = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n // horizon + 1)[blocks].astype(np.int64)
        scores = rng.normal(size=n // horizon + 1)[blocks] + rng.normal(0, 0.01, n)  # indépendant de y : aucun pouvoir prédictif
        result = evaluate_auc_significance(y, scores, times, horizon_ns=horizon * SECOND)
        assert result.auc_full is not None and result.n_independent <= n // horizon + 1
        naive_false_positives += auc_lower_bound(result.auc_full, int(y.sum()), int(n - y.sum())) > 0.5
        independent_false_positives += result.significant
    assert naive_false_positives >= 15  # observé : 24 sur 60
    assert independent_false_positives <= 2  # un test à 1 % : environ 0,6 attendu sur 60 ; observé : 0


def test_degenerate_samples_report_an_undefined_auc_rather_than_a_number() -> None:
    times = np.arange(10, dtype=np.int64) * SECOND
    one_class = evaluate_auc_significance(np.zeros(10, dtype=np.int64), np.linspace(0, 1, 10), times, horizon_ns=SECOND)
    assert one_class.auc_full is None and one_class.auc_independent is None and one_class.lower_bound == 0.0 and not one_class.significant
    tiny = evaluate_auc_significance(np.array([0, 1, 0, 1], dtype=np.int64), np.array([0.1, 0.9, 0.2, 0.8]), np.arange(4, dtype=np.int64) * SECOND, horizon_ns=SECOND)
    assert tiny.auc_independent == 1.0 and tiny.lower_bound == 1.0  # Hanley-McNeil s'annule à AUC = 1 : la borne seule dirait « parfait »
    assert tiny.null_z < tiny.z and not tiny.significant  # le test nul (Mann-Whitney) exige de vraies observations : 2 + 2 ne suffisent pas


def test_mismatched_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        evaluate_auc_significance(np.array([0, 1], dtype=np.int64), np.array([0.1]), np.array([1, 2], dtype=np.int64), horizon_ns=1)


# --- Trades : indépendance par intervalles et purge --------------------------------------------


def test_overlapping_trades_are_thinned_by_interval_not_by_a_fixed_gap() -> None:
    from risk.model_validation import non_overlapping_indices

    starts = np.array([0, 5, 12, 12, 30, 31], dtype=np.int64) * SECOND
    ends = np.array([10, 8, 20, 15, 40, 35], dtype=np.int64) * SECOND
    # 0-10 retenu ; 5-8 chevauche ; 12-20 retenu (12 >= 10) ; 12-15 chevauche ; 30-40 retenu ; 31-35 chevauche
    assert non_overlapping_indices(starts, ends).tolist() == [0, 2, 4]
    touching = non_overlapping_indices(np.array([0, 10], dtype=np.int64), np.array([10, 20], dtype=np.int64))
    assert touching.tolist() == [0, 1]  # un trade qui s'ouvre à l'instant où l'autre se ferme est indépendant
    unsorted = non_overlapping_indices(np.array([30, 0, 12], dtype=np.int64), np.array([40, 10, 20], dtype=np.int64))
    assert unsorted.tolist() == [0, 1, 2]  # indices rendus dans l'ordre d'origine
    with pytest.raises(ValueError):
        non_overlapping_indices(np.array([0, 1], dtype=np.int64), np.array([5], dtype=np.int64))


def test_the_purged_split_starts_the_test_after_the_last_training_trade_has_closed() -> None:
    from risk.model_validation import split_purged_intervals

    starts = np.array([0, 5, 20, 30, 60, 80], dtype=np.int64)
    ends = np.array([10, 50, 25, 40, 70, 90], dtype=np.int64)
    # entraînement = 3 premiers (fermetures 10, 50, 25 : la dernière est 50) ; le 4e s'ouvre à 30 < 50 : purgé ; le 5e s'ouvre à 60 >= 50
    assert split_purged_intervals(starts, ends, 0.5) == (3, 4)
    with pytest.raises(ValueError, match="triés"):
        split_purged_intervals(starts[::-1].copy(), ends, 0.5)
    with pytest.raises(ValueError):
        split_purged_intervals(starts[:1], ends[:1], 0.5)
    with pytest.raises(ValueError):
        split_purged_intervals(starts, ends, 1.0)


def test_a_caller_chosen_sample_replaces_the_spacing_rule() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300).astype(np.int64)
    times = np.arange(300, dtype=np.int64) * SECOND
    keep = np.arange(0, 300, 3, dtype=np.int64)
    result = evaluate_auc_significance(y, y + rng.normal(0, 0.5, 300), times, keep=keep)
    assert result.n_independent == 100 and result.n_total == 300 and result.significant

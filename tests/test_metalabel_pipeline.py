"""Tests du pipeline de méta-labeling : sizing de Kelly, chargement sûr du modèle, entraînement, intégration à la Porte 3.

Preuves recherchées :

- le multiplicateur de Kelly vaut exactement 1,0 au point neutre (le squelette non entraîné ne change rien), diminue sous 0,5,
  croît au-dessus, est plafonné et vétoïque sous le seuil ;
- un artefact absent, corrompu ou incohérent ne peut jamais empêcher l'agent de démarrer : retour à 0,5 ;
- les allers-retours reconstruits depuis les fills sont justes (chiffres faits à la main : FIFO, frais, inversions) ;
- l'entraînement apprend un vrai signal, REFUSE le bruit et les échantillons trop petits, et n'écrit alors aucun fichier ;
- la séparation est chronologique avec embargo : aucune fuite du futur ;
- la Porte 3 de l'``ExecutionAgent`` construit les entrées par nom, publie chaque décision et retombe sur le neutre quand
  une entrée manque ;
- toute taille publiée par le pont tombe sur la grille de quantification, quel que soit le multiplicateur.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import deque
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import fakeredis
import fakeredis.aioredis
import numpy as np
import pytest

from core.config import AppSettings, GuardianSettings, RedisSettings
from core.redis_client import RedisClient
from core.schemas import TradingSignal
from risk.meta_labeling import (
    FEATURES_V2,
    MetaLabelModel,
    kelly_size_multiplier,
    metadata_path_for,
)
from src.agents.execution.agent import ExecutionAgent
from workers.common.fills_db import Execution

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_metalabel.py"
T0 = 1_760_000_000  # secondes


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_metalabel_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_metalabel_under_test"] = module
    spec.loader.exec_module(module)
    return module


tm = _load_script()


# --- Sizing de Kelly -------------------------------------------------------------------------


def test_the_neutral_probability_gives_exactly_one_for_every_payoff_ratio_and_fraction() -> None:
    for payoff in (0.25, 1.0, 2.0, 4.0):
        for fraction in (0.25, 0.5, 1.0):
            assert kelly_size_multiplier(0.5, payoff_ratio=payoff, kelly_fraction=fraction) == 1.0


def test_kelly_multiplier_values_match_a_hand_computation() -> None:
    # b = 1, moitié de Kelly : pente = 0,5 x (1 + 1/1) = 1  ->  m = 1 + (p - 0,5)
    assert kelly_size_multiplier(0.7) == pytest.approx(1.2)
    assert kelly_size_multiplier(0.4) == pytest.approx(0.9)
    assert kelly_size_multiplier(1.0) == pytest.approx(1.5)  # exposition maximale proche de 1
    # b = 2 : pente = 0,5 x (1 + 0,5) = 0,75  ->  m = 1 + 0,75 (p - 0,5)
    assert kelly_size_multiplier(0.7, payoff_ratio=2.0) == pytest.approx(1.15)
    assert kelly_size_multiplier(0.4, payoff_ratio=2.0) == pytest.approx(0.925)


def test_the_multiplier_shrinks_below_neutral_grows_above_is_monotone_and_capped() -> None:
    values = [kelly_size_multiplier(p / 100) for p in range(35, 101)]
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))
    assert kelly_size_multiplier(0.45) < 1.0 < kelly_size_multiplier(0.55)
    assert max(values) == 1.5 and kelly_size_multiplier(1.0, max_multiplier=1.2) == 1.2


def test_a_probability_below_the_veto_threshold_is_a_hard_veto_and_the_boundary_is_not() -> None:
    assert kelly_size_multiplier(0.3499) == 0.0
    assert kelly_size_multiplier(0.35) == pytest.approx(0.85)
    assert kelly_size_multiplier(0.2, veto_below=0.1) == pytest.approx(0.7)


@pytest.mark.parametrize("kwargs", [
    {"probability": 1.1}, {"probability": -0.1}, {"probability": 0.5, "payoff_ratio": 0.0}, {"probability": 0.5, "payoff_ratio": float("inf")},
    {"probability": 0.5, "kelly_fraction": 0.0}, {"probability": 0.5, "kelly_fraction": 1.5}, {"probability": 0.5, "veto_below": 0.5},
    {"probability": 0.5, "max_multiplier": 0.9},
])
def test_invalid_kelly_arguments_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        kelly_size_multiplier(**kwargs)


# --- Modèle : chargement sûr -----------------------------------------------------------------


def _train_artifact(tmp_path: Path, *, features: tuple[str, ...] = FEATURES_V2, n: int = 400, seed: int = 0) -> Path:
    """Un vrai artefact entraîné : le gain dépend de ``ppo_confidence`` (la première variable)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, len(features)))
    y = (x[:, 0] + rng.normal(0, 0.1, n) > 0.5).astype(np.int64)
    model = tm.fit_model(x, y)
    out = tmp_path / "metalabel_weights.json"
    tm.save_artifact(model, out, {"version": 2, "feature_names": list(features), "payoff_ratio": 2.0})
    return out


def test_a_trained_artifact_loads_with_its_feature_names_and_payoff_ratio_and_discriminates(tmp_path: Path) -> None:
    model = MetaLabelModel.load_or_skeleton(_train_artifact(tmp_path))
    assert model.is_fitted and model.feature_names == FEATURES_V2 and model.payoff_ratio == 2.0
    high = model.vectorize({"ppo_confidence": 0.95, "xgb_confidence": 0.5, "confidence_gap": 0.1, "spread": 0.5, "volatility": 0.5, "imbalance_aligned": 0.5})
    low = model.vectorize({"ppo_confidence": 0.05, "xgb_confidence": 0.5, "confidence_gap": 0.1, "spread": 0.5, "volatility": 0.5, "imbalance_aligned": 0.5})
    assert high is not None and low is not None
    assert model.predict_confidence(high) > 0.8 > 0.2 > model.predict_confidence(low)


def test_a_missing_corrupt_or_inconsistent_artifact_falls_back_to_the_neutral_skeleton(tmp_path: Path) -> None:
    neutral = MetaLabelModel.load_or_skeleton(None)
    assert not neutral.is_fitted
    assert neutral.predict_confidence(MetaLabelModel.build_feature_vector(min_confidence=0.9, confidence_spread=0.0, spread=1.0, recent_conflict_count=0)) == pytest.approx(0.5)
    assert not MetaLabelModel.load_or_skeleton(tmp_path / "absent.json").is_fitted  # fichier absent : pas d'exception
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ ceci n'est pas un modèle", encoding="utf-8")
    assert not MetaLabelModel.load_or_skeleton(corrupt).is_fitted
    good = _train_artifact(tmp_path)
    metadata_path_for(good).write_text(json.dumps({"feature_names": ["a", "b"], "payoff_ratio": 1.0}), encoding="utf-8")  # 2 noms pour 6 variables apprises
    assert not MetaLabelModel.load_or_skeleton(good).is_fitted
    metadata_path_for(good).write_text("pas du json", encoding="utf-8")
    assert not MetaLabelModel.load_or_skeleton(good).is_fitted
    metadata_path_for(good).write_text(json.dumps({"feature_names": list(FEATURES_V2), "payoff_ratio": -3}), encoding="utf-8")
    assert not MetaLabelModel.load_or_skeleton(good).is_fitted  # rapport de gain invalide


def test_the_neutral_skeleton_maps_to_a_multiplier_of_exactly_one() -> None:
    skeleton = MetaLabelModel.load_or_skeleton(None)
    features = skeleton.vectorize({"min_confidence": 0.99, "confidence_spread": 0.3, "spread": 2.0, "recent_conflict_count": 4})
    assert features is not None
    assert kelly_size_multiplier(skeleton.predict_confidence(features), payoff_ratio=skeleton.payoff_ratio) == 1.0


def test_vectorize_never_invents_a_missing_or_non_finite_input() -> None:
    model = MetaLabelModel(feature_names=FEATURES_V2)
    full = {name: 0.5 for name in FEATURES_V2}
    assert model.vectorize(full) is not None
    assert model.vectorize({k: v for k, v in full.items() if k != "volatility"}) is None
    assert model.vectorize({**full, "spread": float("nan")}) is None
    assert model.vectorize({**full, "spread": "abc"}) is None  # type: ignore[dict-item]  # valeur volontairement non numérique : le test vérifie qu'elle est refusée


# --- Allers-retours (chiffres faits à la main) -----------------------------------------------


def _fill(ts: int, side: str, qty: float, price: float, *, fee: float | None = None, kind: str = "signal", fid: str | None = None,
          venue: str = "BYBIT", symbol: str = "BTCUSDT", fee_currency: str | None = "USDT") -> Execution:
    return Execution(ts, venue, symbol, side, price, price, qty, 0.0, "o", fid or f"{ts}-{side}-{qty}-{price}", kind, fee, fee_currency if fee is not None else None)


def test_a_simple_long_round_trip_subtracts_the_fees() -> None:
    trips = tm.round_trips_from_fills([_fill(1000, "BUY", 1, 100, fee=0.1), _fill(2000, "SELL", 1, 110, fee=0.11)])
    (trip,) = trips
    assert (trip.direction, trip.entry_ms, trip.exit_ms) == (1, 1000, 2000)
    assert trip.pnl == pytest.approx(9.79)  # (110 - 100) x 1 - 0,1 - 0,11


def test_a_short_round_trip_and_a_flat_result_are_not_a_win() -> None:
    (short,) = tm.round_trips_from_fills([_fill(1000, "SELL", 2, 50), _fill(2000, "BUY", 2, 48)])
    assert short.direction == -1 and short.pnl == pytest.approx(4.0)  # (50 - 48) x 2
    (flat,) = tm.round_trips_from_fills([_fill(1000, "BUY", 2, 100), _fill(2000, "SELL", 1, 105), _fill(3000, "SELL", 1, 95)])
    assert flat.pnl == pytest.approx(0.0)  # +5 puis -5
    decision = tm.Decision(0.5, "BTC", 1, {})
    assert tm.join_decisions_to_trips([decision], [flat])[0].label == 0  # un PnL nul n'est pas un gain


def test_fifo_matches_the_oldest_lot_first() -> None:
    (trip,) = tm.round_trips_from_fills([_fill(1, "BUY", 1, 100), _fill(2, "BUY", 1, 110), _fill(3, "SELL", 2, 120)])
    assert trip.pnl == pytest.approx(30.0)  # (120 - 100) + (120 - 110)
    (partial,) = tm.round_trips_from_fills([_fill(1, "BUY", 1, 100), _fill(2, "BUY", 1, 110), _fill(3, "SELL", 1, 120), _fill(4, "SELL", 1, 90)])
    assert partial.pnl == pytest.approx(20.0 + (90 - 110))  # le premier SELL ferme le lot à 100, le second celui à 110


def test_a_fill_that_flips_the_position_ends_one_trip_and_starts_the_next_with_its_fee_split_by_quantity() -> None:
    trips = tm.round_trips_from_fills([
        _fill(1, "BUY", 1, 100),
        _fill(2, "SELL", 3, 110, fee=0.3),  # ferme 1 (+10) et ouvre une vente de 2 à 110
        _fill(3, "BUY", 2, 105, fee=0.2),   # ferme la vente : 2 x (110 - 105) = +10
    ])
    first, second = trips
    assert (first.direction, first.pnl) == (1, pytest.approx(10.0 - 0.1))  # 1/3 des 0,3 de frais
    assert (second.direction, second.pnl) == (-1, pytest.approx(10.0 - 0.2 - 0.2))  # 2/3 des 0,3, plus les 0,2 de la sortie
    assert second.entry_ms == 2 and first.exit_ms == 2


def test_trips_are_kept_apart_per_venue_and_symbol_names_are_normalised() -> None:
    fills = [
        _fill(1, "BUY", 1, 100, venue="BYBIT", symbol="BTCUSDT"), _fill(2, "SELL", 1, 101, venue="BYBIT", symbol="BTCUSDT"),
        _fill(3, "BUY", 1, 200, venue="HYPERLIQUID", symbol="BTC", fee_currency="USDC"), _fill(4, "SELL", 1, 198, venue="HYPERLIQUID", symbol="BTC"),
    ]
    trips = tm.round_trips_from_fills(fills)
    assert sorted((t.venue, t.asset, round(t.pnl, 6)) for t in trips) == [("BYBIT", "BTC", 1.0), ("HYPERLIQUID", "BTC", -2.0)]
    assert tm.base_asset("BTC/USDT") == tm.base_asset("BTCUSDT") == tm.base_asset("BTC") == "BTC"


def test_trips_touched_by_a_foreign_forced_or_unpriceable_fill_are_dropped_and_counted_and_open_ones_ignored() -> None:
    report = tm.Report()
    fills = [
        _fill(1, "BUY", 1, 100, kind="unmatched", symbol="ETHUSDT"), _fill(2, "SELL", 1, 101, symbol="ETHUSDT"),
        _fill(10, "BUY", 1, 100, symbol="SOLUSDT"), _fill(11, "SELL", 1, 90, kind="emergency", symbol="SOLUSDT"),
        _fill(20, "BUY", 1, 100, fee=0.1, fee_currency="BTC", symbol="XRPUSDT"), _fill(21, "SELL", 1, 101, symbol="XRPUSDT"),
        _fill(30, "BUY", 1, 100, symbol="DOGEUSDT"),
    ]
    assert tm.round_trips_from_fills(fills, report) == []
    assert report.trips_dropped == {"unmatched_fill": 1, "emergency_exit": 1, "fee_not_in_quote_currency": 1} and report.trips_open == 1


# --- Jointure et séparation ------------------------------------------------------------------


def _decision(t: float, asset: str = "BTC", direction: int = 1, **features: float) -> Any:
    return tm.Decision(t, asset, direction, features)


def _trip(entry_s: float, pnl: float = 1.0, asset: str = "BTC", direction: int = 1) -> Any:
    return tm.RoundTrip("BYBIT", asset, direction, int(entry_s * 1000), int(entry_s * 1000) + 60_000, pnl, 100.0)


def test_a_trip_takes_the_latest_matching_decision_before_its_entry_and_a_decision_is_used_once() -> None:
    decisions = [_decision(100.0, x=1.0), _decision(150.0, x=2.0), _decision(900.0, x=9.0)]
    labeled = tm.join_decisions_to_trips(decisions, [_trip(160.0), _trip(165.0)], max_lag_s=120.0)
    assert len(labeled) == 2
    assert labeled[0].decision.features["x"] == 2.0 and labeled[1].decision.features["x"] == 1.0  # la 2e prend ce qui reste, dans la fenêtre


def test_decisions_of_another_asset_or_direction_or_outside_the_window_are_never_joined() -> None:
    report = tm.Report()
    decisions = [_decision(100.0, "ETH"), _decision(100.0, direction=-1), _decision(10.0), _decision(500.0)]  # 10 s : 190 s avant l'entrée ; 500 s : après
    assert tm.join_decisions_to_trips(decisions, [_trip(200.0)], max_lag_s=120.0, report=report) == []
    assert report.trips_without_decision == 1
    near = tm.join_decisions_to_trips([_decision(203.0)], [_trip(200.0)], clock_tolerance_s=5.0)  # horloges décalées de 3 s : toléré
    assert len(near) == 1


def test_vetoed_and_malformed_decisions_are_dropped_when_parsing() -> None:
    report = tm.Report()
    rows = [
        {"decision_time": 1.0, "instrument": "BTC/USDT", "direction": 1, "features": {"a": 1.0}, "vetoed": False},
        {"decision_time": 2.0, "instrument": "BTC/USDT", "direction": 1, "features": {"a": 1.0}, "vetoed": True},
        {"decision_time": 3.0, "instrument": "BTC/USDT", "direction": 0, "features": {"a": 1.0}},
        {"decision_time": "x", "instrument": "BTC/USDT", "direction": 1, "features": {}},
        {"instrument": "BTC/USDT"},
    ]
    assert [d.time_s for d in tm.parse_decisions(rows, report)] == [1.0] and report.decisions_vetoed == 1


def test_the_split_is_chronological_with_an_embargo_so_no_test_trade_starts_before_a_training_trade_ends() -> None:
    def trade(entry: int, exit_: int) -> Any:
        return tm.LabeledTrade(tm.RoundTrip("V", "BTC", 1, entry, exit_, 1.0, 1.0), _decision(0.0), 1)

    kept = [trade(0, 10), trade(5, 50), trade(20, 25), trade(30, 40), trade(60, 70), trade(80, 90)]
    train_end, test_start = tm.chronological_split(kept, test_fraction=0.5)
    # entraînement = les 3 premiers (fins 10, 50, 25 : la dernière fin est 50) ; le 4e commence à 30 < 50 : écarté ; le 5e commence à 60 : premier du test
    assert train_end == 3 and test_start == 4
    assert all(t.trip.entry_ms >= max(k.trip.exit_ms for k in kept[:train_end]) for t in kept[test_start:])


# --- Entraînement de bout en bout ------------------------------------------------------------


def _synthetic_world(n: int, *, signal: bool, seed: int = 1) -> tuple[list[dict[str, Any]], list[Execution]]:
    """``n`` décisions et leurs allers-retours. Avec ``signal``, le gain dépend de ``ppo_confidence`` ; sinon, d'une pièce à pile ou face."""
    rng = np.random.default_rng(seed)
    decisions: list[dict[str, Any]] = []
    fills: list[Execution] = []
    for i in range(n):
        entry = T0 + i * 1000
        ppo = float(rng.uniform(0.3, 1.0))
        features = {"ppo_confidence": ppo, "xgb_confidence": float(rng.uniform(0.3, 1.0)), "confidence_gap": float(rng.uniform(0, 0.4)),
                    "spread": float(rng.uniform(0.1, 1.0)), "volatility": float(rng.uniform(0.0, 0.01)), "imbalance_aligned": float(rng.uniform(-1, 1))}
        win = (ppo + rng.normal(0, 0.08) > 0.65) if signal else bool(rng.integers(0, 2))
        decisions.append({"decision_time": entry - 10.0, "instrument": "BTC/USDT", "direction": 1, "features": features, "vetoed": False})
        exit_price = 101.0 if win else 99.0
        fills.append(_fill(entry * 1000, "BUY", 1, 100.0, fee=0.01, fid=f"b{i}"))
        fills.append(_fill(entry * 1000 + 60_000, "SELL", 1, exit_price, fee=0.01, fid=f"s{i}"))
    return decisions, fills


def test_training_learns_a_real_signal_writes_the_artifact_and_the_agent_can_load_it(tmp_path: Path) -> None:
    decisions, fills = _synthetic_world(600, signal=True)
    out = tmp_path / "models" / "metalabel_weights.json"
    code, report, details = tm.run(decisions, fills, out=out, min_trades=300, min_test=50)
    assert code == 0 and out.is_file() and metadata_path_for(out).is_file() and report.labeled == 600
    assert details["metrics"]["auc"] > 0.85 and details["metrics"]["brier"] < details["metrics"]["baseline_brier"]
    assert details["n_train"] + details["n_embargoed"] + details["n_test"] == 600
    meta = json.loads(metadata_path_for(out).read_text(encoding="utf-8"))
    assert meta["feature_names"] == list(FEATURES_V2) and meta["version"] == 2 and meta["payoff_ratio"] == pytest.approx(1.0, abs=0.1)
    model = MetaLabelModel.load_or_skeleton(out)
    assert model.is_fitted
    strong = model.vectorize({**decisions[0]["features"], "ppo_confidence": 0.95})
    weak = model.vectorize({**decisions[0]["features"], "ppo_confidence": 0.35})
    assert strong is not None and weak is not None and model.predict_confidence(strong) > 0.7 > 0.3 > model.predict_confidence(weak)


def test_training_refuses_pure_noise_and_writes_nothing(tmp_path: Path) -> None:
    decisions, fills = _synthetic_world(600, signal=False, seed=7)
    out = tmp_path / "metalabel_weights.json"
    code, _, details = tm.run(decisions, fills, out=out, min_trades=300, min_test=50)
    assert code == 1 and not out.exists() and not metadata_path_for(out).exists()
    assert any("AUC" in r or "Brier" in r for r in details["refusals"])  # un modèle qui ne bat pas le taux de base ne dimensionne rien


def test_the_acceptance_gate_rejects_luck_across_many_noise_datasets_and_accepts_a_real_signal(tmp_path: Path) -> None:
    """Sans signal, l'AUC dépasse 0,55 par hasard avec ~170 trades de test : le seuil seul ne suffit pas, d'où la borne inférieure."""
    accepted = 0
    for seed in range(12):
        decisions, fills = _synthetic_world(500, signal=False, seed=100 + seed)
        code, _, _ = tm.run(decisions, fills, out=tmp_path / f"noise{seed}.json", min_trades=300, min_test=50)
        accepted += code == 0
    assert accepted == 0  # borne unilatérale à 99 % (jeux de données figés : résultat déterministe ; à 95 %, 1 jeu sur 12 passait par hasard)
    decisions, fills = _synthetic_world(500, signal=True, seed=5)
    assert tm.run(decisions, fills, out=tmp_path / "signal.json", min_trades=300, min_test=50)[0] == 0


def test_the_auc_lower_bound_is_tight_for_big_samples_wide_for_small_ones_and_zero_when_undefined() -> None:
    assert tm.auc_lower_bound(0.5, 100, 100) < 0.5
    assert tm.auc_lower_bound(0.9, 1000, 1000) > 0.88
    assert tm.auc_lower_bound(0.6, 20, 20) < tm.auc_lower_bound(0.6, 2000, 2000)
    assert tm.auc_lower_bound(0.9, 1, 50) == 0.0


def test_training_refuses_too_few_trades_and_a_single_outcome(tmp_path: Path) -> None:
    decisions, fills = _synthetic_world(40, signal=True)
    out = tmp_path / "metalabel_weights.json"
    code, _, details = tm.run(decisions, fills, out=out, min_trades=300, min_test=50)
    assert code == 1 and not out.exists() and any("labeled trades" in r for r in details["refusals"])
    code, _, details = tm.run([], [], out=out)
    assert code == 1 and not out.exists()
    # un seul résultat (tout gagnant) : l'entraînement refuse, il ne produit pas un modèle trivial
    wins = [_fill(T0 * 1000 + i * 1_000_000, "BUY", 1, 100.0, fid=f"b{i}") for i in range(10)]
    wins += [_fill(T0 * 1000 + i * 1_000_000 + 60_000, "SELL", 1, 101.0, fid=f"s{i}") for i in range(10)]
    rows = [{"decision_time": T0 + i * 1000 - 5.0, "instrument": "BTC/USDT", "direction": 1, "features": {n: 0.5 for n in FEATURES_V2}} for i in range(10)]
    code, _, _ = tm.run(rows, wins, out=out, min_trades=5, min_test=1)
    assert code == 1 and not out.exists()


def test_missing_features_drop_the_row_and_are_counted(tmp_path: Path) -> None:
    decisions, fills = _synthetic_world(10, signal=True)
    del decisions[0]["features"]["volatility"]
    report = tm.Report()
    labeled = tm.join_decisions_to_trips(tm.parse_decisions(decisions), tm.round_trips_from_fills(fills), report=report)
    kept, x, y = tm.build_matrix(labeled, FEATURES_V2, report)
    assert len(kept) == 9 and x.shape == (9, 6) and report.rows_missing_features == 1 and y.shape == (9,)


def test_the_command_line_refuses_missing_inputs_and_reports_a_refusal_with_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tm.main(["--decisions", str(tmp_path / "d.jsonl"), "--fills-db", str(tmp_path / "absent.sqlite3")]) == 2
    from workers.common.fills_db import FillsDatabase

    db_path = tmp_path / "fills.sqlite3"
    db = FillsDatabase(db_path)
    for fill in _synthetic_world(20, signal=True)[1]:
        db.insert_execution(fill)
    db.close()
    decisions_path = tmp_path / "d.jsonl"
    decisions_path.write_text("\n".join(json.dumps(r) for r in _synthetic_world(20, signal=True)[0]), encoding="utf-8")
    out = tmp_path / "w.json"
    assert tm.main(["--decisions", str(decisions_path), "--fills-db", str(db_path), "--out", str(out)]) == 1  # 20 trades : très en dessous du minimum
    assert not out.exists() and "No artifact written" in capsys.readouterr().err
    assert tm.main(["--decisions", str(decisions_path), "--fills-db", str(db_path), "--test-fraction", "0.9"]) == 2


# --- Porte 3 de l'ExecutionAgent -------------------------------------------------------------

_INSTRUMENT = "BTC/USDT"


@pytest.fixture
def agent_factory(tmp_path: Path) -> Iterator[Any]:
    agents: list[ExecutionAgent] = []

    def make(model_path: Path | None = None) -> ExecutionAgent:
        client = RedisClient(RedisSettings())
        client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
        instance = ExecutionAgent(
            {_INSTRUMENT}, capital=10_000.0, settings=AppSettings(mock_mode=True, guardian=GuardianSettings(llm_provider="none")),
            manage_own_signals=False, manage_own_redis=False, meta_label_model_path=model_path,
        )
        instance._redis_client = client
        agents.append(instance)
        return instance

    yield make
    for instance in agents:
        instance._publisher.close()
        instance._subscriber.close()
        instance._zmq_context.term()


def _signals(ppo: float = 0.9, xgb: float = 0.8, direction: Literal[-1, 1] = 1) -> dict[str, TradingSignal]:
    now = time.time()
    return {
        "rl-signal-agent": TradingSignal(timestamp=now, agent_id="rl-signal-agent", instrument=_INSTRUMENT, signal=direction, confidence=ppo),
        "supervisor-agent": TradingSignal(timestamp=now, agent_id="supervisor-agent", instrument=_INSTRUMENT, signal=direction, confidence=xgb),
    }


def _capture_publish(agent: ExecutionAgent) -> list[tuple[str, dict[str, Any]]]:
    sent: list[tuple[str, dict[str, Any]]] = []

    async def fake(topic: str, data: dict[str, Any]) -> None:
        sent.append((topic, data))

    agent.publish = fake  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne
    return sent


def _feed_market(agent: ExecutionAgent, ticks: int = 40, imbalance: float = 0.4) -> None:
    price = 50_000.0
    for i in range(ticks):
        price *= 1.0 + 0.0005 * (1 if i % 2 else -1)
        agent._recent_mids.setdefault(_INSTRUMENT, deque(maxlen=120)).append(price)
    agent._last_spreads[_INSTRUMENT] = 0.5
    agent._last_imbalances[_INSTRUMENT] = imbalance


@pytest.mark.asyncio
async def test_without_a_trained_model_gate_3_sizes_exactly_as_before_and_still_logs_the_decision(agent_factory: Any) -> None:
    agent = agent_factory(None)
    sent = _capture_publish(agent)
    _feed_market(agent)
    result, units = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, _signals())
    assessment = agent._meta_label_assess(_INSTRUMENT, _signals())
    assert result.status == "PASSED" and (assessment.multiplier, assessment.confidence, assessment.fitted) == (1.0, 0.5, False)
    base = await agent.calculate_position_size(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0)
    assert units == pytest.approx(base)  # multiplicateur exactement 1
    (topic, data), = sent
    assert topic == "META_LABEL_DECISION" and data["direction"] == 1 and data["instrument"] == _INSTRUMENT and data["model_fitted"] is False and data["vetoed"] is False
    assert {"ppo_confidence", "xgb_confidence", "confidence_gap", "spread", "volatility", "imbalance_aligned"} <= set(data["features"])


@pytest.mark.asyncio
async def test_a_trained_model_resizes_the_entry_with_kelly_and_receives_its_inputs_by_name(agent_factory: Any, tmp_path: Path) -> None:
    agent = agent_factory(_train_artifact(tmp_path))
    sent = _capture_publish(agent)
    _feed_market(agent, imbalance=0.4)
    strong = agent._meta_label_assess(_INSTRUMENT, _signals(ppo=0.95))
    weak = agent._meta_label_assess(_INSTRUMENT, _signals(ppo=0.05))
    assert strong.fitted and strong.confidence > 0.8 > 0.2 > weak.confidence
    assert strong.multiplier == pytest.approx(kelly_size_multiplier(strong.confidence, payoff_ratio=2.0))
    assert strong.multiplier > 1.0 > weak.multiplier
    f = strong.features
    assert f["ppo_confidence"] == 0.95 and f["xgb_confidence"] == 0.8 and f["confidence_gap"] == pytest.approx(0.15)
    assert f["imbalance_aligned"] == pytest.approx(0.4)  # achat : l'imbalance est comptée telle quelle
    assert agent._meta_label_assess(_INSTRUMENT, _signals(direction=-1)).features["imbalance_aligned"] == pytest.approx(-0.4)  # vente : de signe opposé
    assert f["volatility"] > 0
    result, units = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, _signals(ppo=0.95))
    base = await agent.calculate_position_size(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0)
    assert result.status == "PASSED" and units == pytest.approx(base * strong.multiplier)
    assert sent[-1][1]["model_fitted"] is True and sent[-1][1]["multiplier"] == pytest.approx(strong.multiplier)


@pytest.mark.asyncio
async def test_a_veto_fails_gate_3_and_is_still_logged_as_vetoed(agent_factory: Any, tmp_path: Path) -> None:
    agent = agent_factory(_train_artifact(tmp_path))
    sent = _capture_publish(agent)
    _feed_market(agent)
    result, units = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, _signals(ppo=0.02))
    assert result.status == "FAILED" and "Veto meta-label" in result.detail and units == 0.0
    assert sent[-1][1]["vetoed"] is True and sent[-1][1]["multiplier"] == 0.0


@pytest.mark.asyncio
async def test_an_input_the_model_needs_but_has_no_reading_for_gives_the_neutral_answer_not_a_made_up_value(agent_factory: Any, tmp_path: Path) -> None:
    agent = agent_factory(_train_artifact(tmp_path))
    sent = _capture_publish(agent)  # aucun tick : ni volatilité ni imbalance
    assessment = agent._meta_label_assess(_INSTRUMENT, _signals(ppo=0.95))
    assert (assessment.multiplier, assessment.confidence) == (1.0, 0.5) and "volatility" not in assessment.features
    result, _ = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, _signals(ppo=0.95))
    assert result.status == "PASSED" and sent[-1][1]["multiplier"] == 1.0


@pytest.mark.asyncio
async def test_a_manual_override_has_no_meta_label_and_publishes_no_decision(agent_factory: Any) -> None:
    agent = agent_factory(None)
    sent = _capture_publish(agent)
    result, units = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, None)
    base = await agent.calculate_position_size(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0)
    assert result.status == "PASSED" and units == pytest.approx(base) and sent == []


@pytest.mark.asyncio
async def test_a_failing_decision_log_never_changes_or_delays_the_entry(agent_factory: Any, tmp_path: Path) -> None:
    agent = agent_factory(_train_artifact(tmp_path))
    _feed_market(agent)

    async def broken(topic: str, data: dict[str, Any]) -> None:
        raise ConnectionError("bus indisponible")

    agent.publish = broken
    result, units = await agent._gate_3_sizing(_INSTRUMENT, 50_000.0, 49_500.0, 10_000.0, _signals(ppo=0.95))
    assert result.status == "PASSED" and units > 0


def test_an_agent_given_a_missing_or_corrupt_model_file_starts_with_the_neutral_skeleton(agent_factory: Any, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("pas un modèle", encoding="utf-8")
    assert not agent_factory(corrupt)._meta_label_model.is_fitted
    assert not agent_factory(tmp_path / "absent.json")._meta_label_model.is_fitted


def test_the_backend_loads_the_trained_model_only_when_the_file_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("run_backend_under_test", SCRIPT.parent / "run_backend.py")
    assert spec is not None and spec.loader is not None
    run_backend = importlib.util.module_from_spec(spec)
    sys.modules["run_backend_under_test"] = run_backend
    spec.loader.exec_module(run_backend)
    monkeypatch.setattr(run_backend, "_META_LABEL_MODEL_PATH", tmp_path / "metalabel_weights.json")
    assert run_backend._resolve_meta_label_model_path() is None
    (tmp_path / "metalabel_weights.json").write_text("{}", encoding="utf-8")
    assert run_backend._resolve_meta_label_model_path() == tmp_path / "metalabel_weights.json"


# --- Quantification : la conviction dynamique ne sort jamais telle quelle --------------------


def test_every_size_the_bridge_publishes_lands_on_the_quantization_grid_whatever_the_multiplier() -> None:
    from src.bridge.platform_bridge import BridgeConfig, EnginePosition, desired_target

    cfg = BridgeConfig()  # défaut : paliers de 2,5 %
    equity = Decimal(10_000)
    sizes = set()
    for multiplier in np.linspace(0.7, 1.5, 33):
        units = Decimal(str(round(0.012 * float(multiplier), 6)))  # taille du moteur ~6 % de l'equity, modulée par le multiplicateur
        target = desired_target(EnginePosition("BTC/USDT", "LONG", units, Decimal(50_000)), equity, cfg)
        assert target is not None
        pct = target[1]
        assert pct % Decimal("2.5") == 0 and 0 < pct <= cfg.max_position_pct
        sizes.add(pct)
    assert 1 < len(sizes) <= 4  # des dizaines de multiplicateurs distincts se réduisent à quelques paliers


def test_publishing_exact_sizes_needs_an_explicit_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("run_platform_bridge_for_quant", SCRIPT.parent / "run_platform_bridge.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_platform_bridge_for_quant"] = module
    spec.loader.exec_module(module)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with pytest.raises(SystemExit):
        module.main(["--dry-run", "--size-bucket-pct", "0"])
    assert "--allow-exact-size" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module.main(["--dry-run", "--size-bucket-pct", "-1"])


# --- Validation partagée : trades sans chevauchement et garde de Mann-Whitney -------------------


def _overlapping_world(n: int, seed: int = 2) -> tuple[list[dict[str, Any]], list[Any]]:
    """Trois actifs en rotation, un trade tous les 100 s qui dure 250 s : les trades d'actifs voisins se chevauchent (jamais ceux du même actif)."""
    rng = np.random.default_rng(seed)
    decisions: list[dict[str, Any]] = []
    fills: list[Any] = []
    for i in range(n):
        asset = ("BTC", "ETH", "SOL")[i % 3]
        entry = T0 + i * 100
        ppo = float(rng.uniform(0.3, 1.0))
        features = {name: float(rng.uniform(0, 1)) for name in FEATURES_V2}
        features["ppo_confidence"] = ppo
        win = ppo + rng.normal(0, 0.08) > 0.65
        decisions.append({"decision_time": entry - 10.0, "instrument": f"{asset}/USDT", "direction": 1, "features": features, "vetoed": False})
        fills.append(_fill(entry * 1000, "BUY", 1, 100.0, fid=f"b{i}", symbol=f"{asset}USDT"))
        fills.append(_fill((entry + 250) * 1000, "SELL", 1, 101.0 if win else 99.0, fid=f"s{i}", symbol=f"{asset}USDT"))
    return decisions, fills


def test_the_significance_test_runs_on_non_overlapping_trades_only(tmp_path: Path) -> None:
    decisions, fills = _overlapping_world(900)
    code, _, details = tm.run(decisions, fills, out=tmp_path / "w.json", min_trades=300, min_test=20)
    metrics = details["metrics"]
    assert details["n_test"] > 60 and 20 <= metrics["n_independent"] < details["n_test"] * 0.5  # environ un trade sur trois est indépendant
    assert metrics["auc_all_test_trades"] is not None and metrics["null_z"] > 2.326 and metrics["significant"] == 1.0
    assert code == 0 and (tmp_path / "w.json").is_file()


def test_too_few_independent_trades_refuse_even_when_the_test_set_looks_large(tmp_path: Path) -> None:
    decisions, fills = _overlapping_world(900)
    code, _, details = tm.run(decisions, fills, out=tmp_path / "w.json", min_trades=300, min_test=10_000)
    assert code == 1 and not (tmp_path / "w.json").exists()
    assert any("independent (non-overlapping) test trades" in reason for reason in details["refusals"])


def test_the_mann_whitney_guard_blocks_a_bound_that_hanley_mcneil_alone_would_accept() -> None:
    base = {"auc": 1.0, "auc_lower_99": 1.0, "n_independent": 60.0, "brier": 0.1, "baseline_brier": 0.25}
    assert tm.acceptance_failures({**base, "significant": 1.0, "null_z": 9.0}, n_labeled=500, n_test=60, min_trades=300, min_test=50, min_auc=0.55, test_has_both=True) == []
    problems = tm.acceptance_failures({**base, "significant": 0.0, "null_z": 1.5}, n_labeled=500, n_test=60, min_trades=300, min_test=50, min_auc=0.55, test_has_both=True)
    assert len(problems) == 1 and "Mann-Whitney" in problems[0] and "Hanley-McNeil" in problems[0]

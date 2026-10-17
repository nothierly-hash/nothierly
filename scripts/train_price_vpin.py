"""Entraîne un PPO « prix + flux + VPIN » sur un historique reconstitué (``scripts/backfill_vpin.py``), puis le juge HORS ÉCHANTILLON, frais compris, face à des références.

    python scripts/train_price_vpin.py --bars data/bars_BTCUSDT.npz --vpin-db data/vpin_backfill.sqlite3 --symbol BTCUSDT [--export]

**Ce que fait la chaîne, dans l'ordre**
1. Mesures ``price_flow_v1`` par barre de 60 s (``src/rl/historical_features.py``) + VPIN, vitesse et accélération du VPIN alignés CAUSALEMENT sur chaque barre
   (``train_ppo.attach_vpin_features``) ; on garde la plus longue plage continue où tout est connu (jamais de mesure à cheval sur un trou).
2. Découpage CHRONOLOGIQUE entrainement / validation / test, séparés par une marge (``--embargo-bars``, 90 barres) : les mesures glissantes et la fenêtre du VPIN
   d'une barre de test ne contiennent alors aucune information de l'entraînement. Aucun mélange, aucun tirage aléatoire des lignes.
3. Normalisation par la moyenne et l'écart-type de l'ENTRAÎNEMENT seulement (jamais ceux de la validation ou du test) ; ils sont enregistrés avec la politique.
4. Entraînement PPO (perte de Huber, gamma 0,97, taxe de toxicité et coupe-circuit VPIN optionnels : mêmes réglages que ``train_ppo.py``).
5. Évaluation sur entraînement, validation et test, face à : rester à plat, acheter et garder, long tant que le VPIN reste bas. Le Sharpe est donné avec son erreur-type.

**Lecture honnête du résultat.** Le test ne sert qu'UNE fois : si on retouche les réglages d'après ses résultats, il devient de la validation et il faut un nouveau test.
Un résultat sur quelques semaines d'un seul actif ne prouve rien d'un avantage durable. Ce modèle n'a PAS le contrat du modèle de production (pas de spread,
de carnet, de macro ni de sentiment) : ``feature_set = "price_flow_v1"`` est écrit dans ses métadonnées et ne doit jamais être servi avec les mesures de production.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import train_ppo as tp  # noqa: E402
from src.rl import evaluation as ev  # noqa: E402
from src.rl import historical_features as hf  # noqa: E402
from src.rl.networks import ActorCritic  # noqa: E402
from src.rl.risk_shaping import RiskShapingConfig  # noqa: E402

BAR_S = hf.DEFAULT_BAR_S
DEFAULT_COST = 0.0005  # 5 points de base par changement de position (frais + glissement, hypothèse prudente à valider sur tes fills)


@dataclass(frozen=True)
class Dataset:
    """Une plage continue où les huit mesures (5 prix/flux + 3 VPIN) sont connues, alignées ligne à ligne."""

    times_s: npt.NDArray[np.float64]
    prices: npt.NDArray[np.float64]
    features: npt.NDArray[np.float64]  # (T, 5)
    vpin: npt.NDArray[np.float64]  # (T, 3) : niveau, vitesse, accélération


def prepare_dataset(bars: dict[str, npt.NDArray[np.float64]], vpin_times_s: npt.NDArray[np.float64], vpin_values: npt.NDArray[np.float64]) -> Dataset:
    """Plus longue plage continue où prix/flux ET VPIN sont connus ; lève ``ValueError`` s'il n'y en a pas d'assez longue."""
    best: Dataset | None = None
    for part in hf.featurize_runs(bars, bar_s=BAR_S, min_rows=tp.MIN_TRAINING_ROWS):
        try:
            vpin, run = tp.attach_vpin_features(part["bar_end_s"], vpin_times_s, vpin_values)
        except ValueError:
            continue
        candidate = Dataset(part["bar_end_s"][run], part["close"][run], part["features"][run], vpin[run])
        if best is None or len(candidate.prices) > len(best.prices):
            best = candidate
    if best is None:
        raise ValueError(f"aucune plage continue d'au moins {tp.MIN_TRAINING_ROWS} barres avec les prix/flux ET le VPIN connus")
    return best


def chronological_split(n: int, fractions: tuple[float, float, float], embargo: int) -> tuple[slice, slice, slice]:
    """Trois tranches contiguës dans l'ordre du temps, séparées par ``embargo`` lignes jetées."""
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError("les fractions doivent être positives et sommer à 1")
    if embargo < 0:
        raise ValueError("embargo doit être >= 0")
    usable = n - 2 * embargo
    if usable < 3 * 3:
        raise ValueError("pas assez de lignes pour trois tranches avec cette marge")
    n_train = int(usable * fractions[0])
    n_val = int(usable * fractions[1])
    train = slice(0, n_train)
    val = slice(n_train + embargo, n_train + embargo + n_val)
    test = slice(n_train + n_val + 2 * embargo, n)
    return train, val, test


def normalization(data: Dataset, train: slice) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Moyenne et écart-type des huit colonnes, calculés sur l'ENTRAÎNEMENT seulement."""
    block = np.concatenate((data.features[train], data.vpin[train]), axis=1)
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def apply(data: Dataset, part: slice, mean: npt.NDArray[np.float64], std: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    block = (np.concatenate((data.features[part], data.vpin[part]), axis=1) - mean) / std
    return block[:, :5], block[:, 5:], data.prices[part]


def build_metadata(
    *, symbol: str, mean: npt.NDArray[np.float64], std: npt.NDArray[np.float64], shaping: RiskShapingConfig, config: tp.PPOConfig, data: Dataset, splits: dict[str, tuple[int, int]],
    cost: float,
) -> dict[str, Any]:
    meta = tp.policy_metadata(9, True, shaping, config)
    meta.update({
        "feature_set": hf.FEATURE_SET_NAME, "feature_names": list(hf.FEATURE_NAMES), "bar_s": BAR_S, "symbol": symbol,
        "normalization": {"mean": mean.tolist(), "std": std.tolist(), "columns": [*hf.FEATURE_NAMES, "vpin", "dvpin", "d2vpin"]},
        "period_utc_s": [float(data.times_s[0]), float(data.times_s[-1])], "splits_rows": {k: list(v) for k, v in splits.items()}, "transaction_cost_pct": cost,
    })
    return meta


def run(args: argparse.Namespace) -> int:
    loaded = np.load(args.bars)
    bars = {k: loaded[k] for k in ("bar_end_s", "close", "buy_notional", "sell_notional")}
    symbol = args.symbol or str(loaded["symbol"])
    times, values = tp.load_vpin_series(
        args.vpin_db, symbol, float(bars["bar_end_s"][0]) - 7200.0, float(bars["bar_end_s"][-1]) + 60.0, args.vpin_bucket_size
    )
    data = prepare_dataset(bars, times, values)
    train, val, test = chronological_split(len(data.prices), (args.train_frac, args.val_frac, args.test_frac), args.embargo_bars)
    mean, std = normalization(data, train)
    print(f"{symbol} : {len(data.prices)} barres continues de {BAR_S:.0f} s ; train {train.stop - train.start}, validation {val.stop - val.start}, test {test.stop - test.start} (marge {args.embargo_bars}).")

    shaping = RiskShapingConfig(
        toxicity_penalty_weight=args.toxicity_weight, kill_switch_vpin=args.kill_switch_vpin,
        vpin_threshold=args.vpin_threshold, safe_harbor_bonus=0.0,
    )
    config = tp.PPOConfig(
        total_timesteps=args.total_timesteps, rollout_length=args.rollout_length, gamma=args.gamma, value_loss="huber", minibatch_size=min(64, args.rollout_length),
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    f_train, v_train, p_train = apply(data, train, mean, std)
    model = tp.run_ppo_training(f_train, p_train, config, vpin_features=v_train, risk_shaping=shaping)

    reports: dict[str, dict[str, ev.Metrics]] = {}
    for name, part in (("entraînement", train), ("validation", val), ("test", test)):
        f, v, p = apply(data, part, mean, std)
        raw_vpin = data.vpin[part][:, 0]
        results = {"politique": ev.evaluate_policy(model, f, p, v, transaction_cost_pct=args.cost, bar_s=BAR_S, kill_switch_vpin=args.kill_switch_vpin)}
        results.update(ev.baselines(p, raw_vpin, transaction_cost_pct=args.cost, bar_s=BAR_S, vpin_threshold=args.vpin_threshold))
        reports[name] = results
        print(f"\n== {name}\n{ev.format_table(results)}")

    test_results = reports["test"]
    best_baseline = max((m for k, m in test_results.items() if k != "politique"), key=lambda m: m.total_log_return)
    policy = test_results["politique"]
    beats = policy.total_log_return > best_baseline.total_log_return
    print(
        f"\nVerdict sur le TEST : la politique {'bat' if beats else 'ne bat PAS'} la meilleure référence "
        f"({policy.total_log_return:+.4f} contre {best_baseline.total_log_return:+.4f} en log-rendement net de frais)."
    )
    if policy.sharpe is not None and policy.sharpe_std_error is not None and abs(policy.sharpe) < 2 * policy.sharpe_std_error:
        print("Le Sharpe de la politique est dans l'erreur statistique (|Sharpe| < 2 erreurs-types) : ce résultat n'est pas distinguable de zéro.")
    print("Rappel : le test ne sert qu'une fois ; retoucher les réglages d'après lui en fait de la validation.")

    if args.export:
        splits = {"train": (train.start, train.stop), "val": (val.start, val.stop), "test": (test.start, test.stop)}
        meta = build_metadata(symbol=symbol, mean=mean, std=std, shaping=shaping, config=config, data=data, splits=splits, cost=args.cost)
        path = tp.export_model(model, args.models_dir, meta)
        print(f"politique exportée : {path} (feature_set = {hf.FEATURE_SET_NAME} : ne pas la servir avec les mesures de production)")
    if args.report_json is not None:
        args.report_json.write_text(json.dumps({k: {n: m.__dict__ for n, m in v.items()} for k, v in reports.items()}, indent=2), encoding="utf-8")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO prix + flux + VPIN sur historique reconstitué, évalué hors échantillon.")
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--vpin-db", type=Path, required=True)
    parser.add_argument("--symbol", default=None, help="défaut : celui du fichier de barres")
    parser.add_argument("--vpin-bucket-size", type=float, default=None, help="taille de seau (USDT) à utiliser ; obligatoire si l'archive en mélange plusieurs")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--embargo-bars", type=int, default=90)
    parser.add_argument("--cost", type=float, default=DEFAULT_COST, help="coût par changement de position, en log-rendement")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-length", type=int, default=1_024)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--toxicity-weight", type=float, default=0.01)
    parser.add_argument("--vpin-threshold", type=float, default=0.7)
    parser.add_argument("--kill-switch-vpin", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", action="store_true", help="écrit la politique et ses métadonnées (feature_set, normalisation) dans --models-dir")
    parser.add_argument("--models-dir", type=Path, default=tp.MODELS_DIR)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cost < 0:
        parser.error("--cost doit être >= 0")
    return args


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))

"""Évaluation hors échantillon d'une politique, face à des références simples, AVEC frais : la seule mesure qui dise si l'entraînement a servi à quelque chose.

Ce module ne mesure pas une performance future : il compare, sur une période que la politique n'a jamais vue, ce qu'elle aurait fait à ce que feraient des
règles triviales (rester à plat, acheter et garder, ne rien ouvrir quand le VPIN est haut). Une politique qui ne bat pas la meilleure de ces références n'a rien
appris d'utile, quel que soit le résultat de l'entraînement.

Conventions (identiques à ``MarketEnv``) : la décision à la ligne ``t`` détient sa position entre le prix ``t`` et le prix ``t+1`` ; le rendement est le log-rendement
de ce pas ; chaque changement de position coûte ``transaction_cost_pct`` (par côté, en log-rendement, comme l'environnement). Politique longue seule : position 0 ou 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import torch

from src.rl.networks import ActorCritic, masked_probabilities
from src.rl.risk_shaping import RiskShapingConfig
from src.rl.trading_env import MarketEnv

SECONDS_PER_YEAR: Final[float] = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class Metrics:
    n_steps: int
    total_log_return: float
    sharpe: float | None  # annualisé ; None si la variance est nulle
    sharpe_std_error: float | None  # erreur-type approchée du Sharpe annualisé : un Sharpe petit devant elle n'est pas distinguable de zéro
    max_drawdown: float  # <= 0, sur la courbe de capital exp(somme des log-rendements)
    n_trades: int  # changements de position
    exposure: float  # part du temps en position
    hit_rate: float | None  # part des pas EN POSITION qui gagnent (hors frais) ; None si jamais en position


def metrics(net_log_returns: npt.NDArray[np.float64], positions: npt.NDArray[np.float64], *, bar_s: float) -> Metrics:
    """``net_log_returns`` : par pas, frais déduits ; ``positions`` : la position détenue pendant ce pas (0 ou 1)."""
    if len(net_log_returns) != len(positions) or len(net_log_returns) == 0:
        raise ValueError("net_log_returns et positions doivent avoir la même longueur non nulle.")
    n = len(net_log_returns)
    mean, std = float(np.mean(net_log_returns)), float(np.std(net_log_returns, ddof=1)) if n > 1 else 0.0
    bars_per_year = SECONDS_PER_YEAR / bar_s
    sharpe = mean / std * float(np.sqrt(bars_per_year)) if std > 0 else None
    # erreur-type du Sharpe par pas (approximation usuelle, rendements ~ i.i.d.), annualisée
    std_error = float(np.sqrt((1.0 + 0.5 * (mean / std) ** 2) / n) * np.sqrt(bars_per_year)) if std > 0 else None
    equity = np.exp(np.cumsum(net_log_returns))
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    in_position = positions > 0
    changes = int(np.count_nonzero(np.diff(np.concatenate(([0.0], positions)))))
    gross = net_log_returns[in_position]
    return Metrics(
        n_steps=n, total_log_return=float(np.sum(net_log_returns)), sharpe=sharpe, sharpe_std_error=std_error,
        max_drawdown=float(np.min(equity / peak - 1.0)), n_trades=changes, exposure=float(np.mean(in_position)),
        hit_rate=float(np.mean(gross > 0)) if in_position.any() else None,
    )


def _net_returns(positions: npt.NDArray[np.float64], prices: npt.NDArray[np.float64], cost: float) -> npt.NDArray[np.float64]:
    log_returns = np.diff(np.log(prices))  # pas t : prix t -> prix t+1
    previous = np.concatenate(([0.0], positions[:-1]))
    result: npt.NDArray[np.float64] = positions * log_returns - cost * np.abs(positions - previous)
    return result


def run_positions(positions: npt.NDArray[np.float64], prices: npt.NDArray[np.float64], *, transaction_cost_pct: float, bar_s: float) -> Metrics:
    """Métriques d'une série de positions décidées pour les pas ``0 .. T-2`` (``positions`` a ``T-1`` éléments)."""
    if len(positions) != len(prices) - 1:
        raise ValueError("positions doit avoir un élément de moins que prices (un par pas).")
    return metrics(_net_returns(positions, prices, transaction_cost_pct), positions, bar_s=bar_s)


def baselines(
    prices: npt.NDArray[np.float64], vpin_level: npt.NDArray[np.float64], *, transaction_cost_pct: float, bar_s: float, vpin_threshold: float = 0.7
) -> dict[str, Metrics]:
    """Trois références : à plat, acheter et garder, long tant que le VPIN reste sous le seuil (décision prise sur le VPIN connu AU pas t)."""
    steps = len(prices) - 1
    flat = np.zeros(steps)
    hold = np.ones(steps)
    rule = (vpin_level[:steps] <= vpin_threshold).astype(np.float64)
    return {
        "à plat": run_positions(flat, prices, transaction_cost_pct=transaction_cost_pct, bar_s=bar_s),
        "acheter et garder": run_positions(hold, prices, transaction_cost_pct=transaction_cost_pct, bar_s=bar_s),
        f"long si VPIN <= {vpin_threshold}": run_positions(rule, prices, transaction_cost_pct=transaction_cost_pct, bar_s=bar_s),
    }


def evaluate_policy(
    model: ActorCritic,
    features: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    vpin_features: npt.NDArray[np.float64],
    *,
    transaction_cost_pct: float,
    bar_s: float,
    kill_switch_vpin: float | None = None,
) -> Metrics:
    """Rejoue la politique en mode déterministe (action la plus probable, masque du coupe-circuit appliqué) sur ``features``/``prices``, du début à la fin."""
    shaping = RiskShapingConfig(kill_switch_vpin=kill_switch_vpin)
    env = MarketEnv(features, prices, transaction_cost_pct=transaction_cost_pct, vpin_features=vpin_features, risk_shaping=shaping)
    observation, info = env.reset(seed=0)
    positions: list[float] = []
    model.eval()
    with torch.no_grad():
        while True:
            probabilities, _ = model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
            mask = torch.as_tensor(info["action_mask"]).unsqueeze(0)
            action = int(torch.argmax(masked_probabilities(probabilities, mask), dim=1).item())
            observation, _reward, _terminated, truncated, info = env.step(action)
            positions.append(float(info["position"]))
            if truncated:
                break
    held = np.array(positions, dtype=np.float64)
    return run_positions(held, prices[: len(held) + 1], transaction_cost_pct=transaction_cost_pct, bar_s=bar_s)


def format_table(results: dict[str, Metrics]) -> str:
    """Tableau lisible ; le Sharpe est suivi de son erreur-type, pour ne jamais lire un petit Sharpe comme un résultat."""
    lines = [f"{'stratégie':<26}{'pas':>8}{'rend. log':>11}{'Sharpe ± err.':>18}{'max DD':>9}{'trades':>8}{'expo.':>7}{'hit':>7}"]
    for name, m in results.items():
        sharpe = "n/a" if m.sharpe is None or m.sharpe_std_error is None else f"{m.sharpe:+.1f} ± {m.sharpe_std_error:.1f}"
        hit = "n/a" if m.hit_rate is None else f"{m.hit_rate:.0%}"
        lines.append(f"{name:<26}{m.n_steps:>8}{m.total_log_return:>+11.4f}{sharpe:>18}{m.max_drawdown:>9.1%}{m.n_trades:>8}{m.exposure:>7.0%}{hit:>7}")
    return "\n".join(lines)

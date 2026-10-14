"""Simulation Monte Carlo du PnL net de Nothierly sur ~45 jours, à partir d'HYPOTHÈSES que vous fixez ci-dessous.

    python scripts/simulate_pnl_montecarlo.py                       # 10 000 chemins, 45 jours, frais Binance
    python scripts/simulate_pnl_montecarlo.py --venue hyperliquid --win-rate 0.55 --rr 1.2 --stop-bps 50
    python scripts/simulate_pnl_montecarlo.py --paths 50000 --seed 7 --no-compound

**Ce que ce script est, et n'est pas.** C'est un calculateur de conséquences : « SI le taux de réussite est X, le rapport gain/perte Y et les frais Z, ALORS la
distribution du PnL à 45 jours est celle-ci ». Il ne prédit pas le PnL de Nothierly : aucun modèle n'est entraîné aujourd'hui et aucune donnée réelle n'a encore été
mesurée. Les valeurs par défaut sont illustratives (un ordre de grandeur plausible pour du scalping ML), pas des résultats. Le seul chiffre honnête sur le futur réel sera
le taux de réussite mesuré HORS ÉCHANTILLON sur les données du Shadow Run, à reporter dans ``WIN_RATE_MEAN``.

**Comment ajuster les probabilités PPO / XGBoost.** Le script ne contient pas les modèles : il n'a besoin que de leur effet net, un taux de réussite par trade.
- ``WIN_RATE_MEAN`` : la proportion de trades gagnants OBSERVÉE hors échantillon, après vos filtres (seuil de confiance du consensus PPO/XGBoost, méta-label qui
  veto). Ce n'est pas l'AUC ni la précision brute d'XGBoost : un modèle à 60 % de précision directionnelle peut donner 52 % de trades gagnants une fois les frais, les
  stops et les entrées manquées pris en compte. Pour un chiffre prudent, utilisez la borne basse de l'intervalle de confiance (par exemple celle du test de
  significativité de ``risk/model_validation.py``), pas l'estimation centrale.
- ``WIN_RATE_STD`` : votre incertitude sur ce taux (le vrai taux futur n'est pas celui du passé). 0,02 signifie « le vrai taux est probablement entre 52 % et 60 % ».
  C'est le paramètre qui élargit le plus l'éventail des résultats ; ne le mettez pas à 0.
- ``DAILY_WIN_RATE_STD`` : les trades d'une même journée ne sont pas indépendants (régimes de marché) ; ce bruit commun fait grossir les queues de distribution.
- ``RR`` (gain visé / perte au stop) et ``STOP_BPS`` : la combinaison décide si le taux de réussite suffit à payer les frais. Le script affiche le taux de réussite de
  RENTABILITÉ pour vos réglages : s'il est au-dessus de votre ``WIN_RATE_MEAN``, l'espérance par trade est négative et plus de trades ne fait que perdre plus vite.

**Modèle (vectorisé : chemins x jours, sans boucle sur les trades).** Par chemin : un taux de réussite tiré une fois (incertitude du modèle), un nombre moyen de trades par
jour tiré une fois dans [TRADES_MIN, TRADES_MAX]. Chaque jour : nombre de trades ~ Poisson, taux du jour = taux du chemin + bruit commun, gagnants ~ Binomiale. Chaque
trade gagne ``RR x STOP_BPS`` ou perd ``STOP_BPS`` (en points de base du notionnel), moins les frais et le glissement ; la dispersion des gains et pertes réalisés et du
glissement est ajoutée par approximation normale de la somme journalière. La taille de position vise ``RISK_PER_TRADE_PCT`` de l'équité par stop (levier plafonné). Une perte
journalière est plafonnée à ``DAILY_LOSS_LIMIT_PCT`` (le coupe-circuit du moteur). Capitalisation journalière (``--no-compound`` : tailles fixées sur le capital initial).

**Limites connues (non modélisées).** Impact de marché et capacité (les tailles sont supposées sans effet sur les prix), funding, latence et exécutions manquées, ordres
partiellement remplis, changement de régime durable, pannes. Drawdown mesuré sur la courbe journalière : il sous-estime le creux intra-journalier. Les frais par défaut sont
des barèmes publics de base (sans palier VIP ni remise) à VÉRIFIER sur votre compte avant de s'y fier.

**Intervalles.** « Fourchette à 95 % » = percentiles 2,5 et 97,5 de la distribution des chemins (où tombe un chemin simulé). L'intervalle de confiance de la MOYENNE
(moyenne +/- 1,96 écart-type / racine du nombre de chemins) ne mesure que l'erreur de la simulation, pas l'incertitude sur la stratégie.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# --- Hypothèses (modifiables ici, ou par la ligne de commande) -------------------------------------------------------------------------------------------------
INITIAL_CAPITAL: Final[float] = 10_000.0
DAYS: Final[int] = 45
N_PATHS: Final[int] = 10_000
SEED: Final[int] = 42
TRADES_PER_DAY_MIN: Final[float] = 50.0
TRADES_PER_DAY_MAX: Final[float] = 500.0
WIN_RATE_MEAN: Final[float] = 0.56          # taux de trades gagnants attendu (réaliste pour du ML : 0,54 à 0,58)
WIN_RATE_STD: Final[float] = 0.015          # incertitude sur le VRAI taux (écart-type, tiré une fois par chemin)
DAILY_WIN_RATE_STD: Final[float] = 0.01     # bruit commun aux trades d'une même journée (régimes)
RR: Final[float] = 1.2                      # gain visé / perte au stop
STOP_BPS: Final[float] = 40.0               # distance du stop, en points de base du prix d'entrée
RISK_PER_TRADE_PCT: Final[float] = 0.10     # % de l'équité risqué par trade (perte si le stop est touché)
MAX_LEVERAGE: Final[float] = 3.0
DAILY_LOSS_LIMIT_PCT: Final[float] = 3.0    # coupe-circuit journalier (perte maximale d'une journée, en % de l'équité)
ENTRY_MAKER_FRACTION: Final[float] = 0.5    # part des entrées exécutées en maker
TP_MAKER_FRACTION: Final[float] = 0.7       # part des sorties « objectif atteint » exécutées en maker (un stop est toujours taker)
SLIPPAGE_BPS: Final[float] = 1.0            # glissement moyen par exécution taker
SLIPPAGE_STD_BPS: Final[float] = 1.0        # dispersion du glissement par exécution
PAYOFF_DISPERSION: Final[float] = 0.25      # écart-type relatif du gain ou de la perte réalisés autour de la cible (fills, gaps)
ANNUALIZATION_DAYS: Final[int] = 365        # crypto : marché ouvert 7 jours sur 7
RUIN_EQUITY_FRACTION: Final[float] = 0.5    # « ruine » = équité finale sous cette fraction du capital initial

# Barèmes publics de base, en points de base du notionnel (maker, taker). À VÉRIFIER sur la grille de frais actuelle de votre compte.
VENUE_FEES_BPS: Final[dict[str, tuple[float, float]]] = {
    "binance": (2.0, 5.0),
    "hyperliquid": (1.5, 4.5),
    "bybit": (2.0, 5.5),
}


@dataclass(frozen=True)
class SimConfig:
    capital: float = INITIAL_CAPITAL
    days: int = DAYS
    paths: int = N_PATHS
    seed: int = SEED
    trades_min: float = TRADES_PER_DAY_MIN
    trades_max: float = TRADES_PER_DAY_MAX
    win_rate: float = WIN_RATE_MEAN
    win_rate_std: float = WIN_RATE_STD
    daily_win_rate_std: float = DAILY_WIN_RATE_STD
    rr: float = RR
    stop_bps: float = STOP_BPS
    risk_pct: float = RISK_PER_TRADE_PCT
    max_leverage: float = MAX_LEVERAGE
    daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT
    maker_bps: float = VENUE_FEES_BPS["binance"][0]
    taker_bps: float = VENUE_FEES_BPS["binance"][1]
    entry_maker_fraction: float = ENTRY_MAKER_FRACTION
    tp_maker_fraction: float = TP_MAKER_FRACTION
    slippage_bps: float = SLIPPAGE_BPS
    slippage_std_bps: float = SLIPPAGE_STD_BPS
    payoff_dispersion: float = PAYOFF_DISPERSION
    compound: bool = True

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.capital <= 0:
            problems.append("capital doit être > 0")
        if self.days < 1 or self.paths < 2:
            problems.append("days >= 1 et paths >= 2 sont requis")
        if not 0 < self.trades_min <= self.trades_max:
            problems.append("0 < trades_min <= trades_max est requis")
        if not 0 < self.win_rate < 1:
            problems.append("win_rate doit être dans ]0, 1[")
        if self.win_rate_std < 0 or self.daily_win_rate_std < 0 or self.slippage_std_bps < 0 or self.payoff_dispersion < 0:
            problems.append("les écarts-types ne peuvent pas être négatifs")
        if self.rr <= 0 or self.stop_bps <= 0:
            problems.append("rr et stop_bps doivent être > 0")
        if not 0 < self.risk_pct <= 100 or self.max_leverage <= 0:
            problems.append("risk_pct dans ]0, 100] et max_leverage > 0 sont requis")
        if not 0 < self.daily_loss_limit_pct <= 100:
            problems.append("daily_loss_limit_pct doit être dans ]0, 100]")
        if min(self.maker_bps, self.taker_bps, self.slippage_bps) < 0:
            problems.append("frais et glissement ne peuvent pas être négatifs")
        if not (0 <= self.entry_maker_fraction <= 1 and 0 <= self.tp_maker_fraction <= 1):
            problems.append("les parts maker doivent être dans [0, 1]")
        if problems:
            raise ValueError("configuration invalide : " + " ; ".join(problems))

    @property
    def take_profit_bps(self) -> float:
        return self.rr * self.stop_bps

    @property
    def leverage(self) -> float:
        """Notionnel / équité : le risque visé divisé par la distance du stop, plafonné."""
        return min((self.risk_pct / 100.0) / (self.stop_bps / 10_000.0), self.max_leverage)


@dataclass(frozen=True)
class TradeEconomics:
    """Ce que rapporte UN trade moyen, en points de base du notionnel, une fois les frais payés."""

    net_win_bps: float
    net_loss_bps: float  # négatif
    cost_win_bps: float
    cost_loss_bps: float
    breakeven_win_rate: float
    expectancy_bps: float  # espérance par trade au taux de réussite configuré


def trade_economics(cfg: SimConfig) -> TradeEconomics:
    entry_fee = cfg.entry_maker_fraction * cfg.maker_bps + (1 - cfg.entry_maker_fraction) * cfg.taker_bps
    entry_taker = 1 - cfg.entry_maker_fraction
    exit_fee_win = cfg.tp_maker_fraction * cfg.maker_bps + (1 - cfg.tp_maker_fraction) * cfg.taker_bps
    exit_taker_win = 1 - cfg.tp_maker_fraction
    cost_win = entry_fee + exit_fee_win + cfg.slippage_bps * (entry_taker + exit_taker_win)
    cost_loss = entry_fee + cfg.taker_bps + cfg.slippage_bps * (entry_taker + 1.0)  # un stop est un ordre au marché
    net_win = cfg.take_profit_bps - cost_win
    net_loss = -(cfg.stop_bps + cost_loss)
    breakeven = -net_loss / (net_win - net_loss) if net_win - net_loss > 0 else float("nan")
    expectancy = cfg.win_rate * net_win + (1 - cfg.win_rate) * net_loss
    return TradeEconomics(net_win, net_loss, cost_win, cost_loss, breakeven, expectancy)


@dataclass(frozen=True)
class SimResult:
    equity: FloatArray          # (chemins, jours + 1), capital initial en colonne 0
    daily_returns: FloatArray   # (chemins, jours), fraction du capital du début de journée
    trades: FloatArray          # (chemins, jours)


def simulate(cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    econ = trade_economics(cfg)
    lev = cfg.leverage
    n_paths = cfg.paths
    path_win_rate = np.clip(rng.normal(cfg.win_rate, cfg.win_rate_std, n_paths), 0.01, 0.99)  # incertitude du modèle : un taux par chemin
    path_trades = rng.uniform(cfg.trades_min, cfg.trades_max, n_paths)                          # un régime d'activité par chemin
    loss_limit = cfg.daily_loss_limit_pct / 100.0
    equity = np.empty((n_paths, cfg.days + 1))
    equity[:, 0] = cfg.capital
    returns = np.empty((n_paths, cfg.days))
    trades = np.empty((n_paths, cfg.days))
    tp, stop = cfg.take_profit_bps, cfg.stop_bps
    for day in range(cfg.days):
        n = rng.poisson(path_trades).astype(np.float64)
        p_day = np.clip(path_win_rate + rng.normal(0.0, cfg.daily_win_rate_std, n_paths), 0.01, 0.99)
        wins = rng.binomial(n.astype(np.int64), p_day).astype(np.float64)
        losses = n - wins
        mean_bps = wins * econ.net_win_bps + losses * econ.net_loss_bps
        var_bps = (wins * (tp * cfg.payoff_dispersion) ** 2 + losses * (stop * cfg.payoff_dispersion) ** 2
                   + n * 2.0 * cfg.slippage_std_bps ** 2)
        day_bps = mean_bps + rng.standard_normal(n_paths) * np.sqrt(var_bps)  # somme de n trades : approximation normale
        raw = lev * day_bps / 10_000.0
        # Coupe-circuit = arrêt EN COURS de journée : une journée qui touche -limite s'arrête là, même si elle aurait fini plus haut. Plafonner seulement le résultat
        # final serait optimiste (biais positif). Pont brownien : sachant le résultat final, la probabilité d'avoir touché -limite est exp(-2 L (L + final) / variance
        # du jour), indépendamment de la dérive.
        # variance TOTALE de la journée : dispersion des fills + variance binomiale du nombre de gagnants (l'oublier sous-estime les arrêts et laisse un biais positif)
        day_var = (lev / 10_000.0) ** 2 * (var_bps + n * p_day * (1.0 - p_day) * (econ.net_win_bps - econ.net_loss_bps) ** 2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            touched = np.where(day_var > 0, np.exp(-2.0 * loss_limit * (loss_limit + raw) / day_var), 0.0)
        stopped = (raw <= -loss_limit) | (rng.random(n_paths) < np.where(raw > -loss_limit, touched, 1.0))
        ret = np.where(stopped, -loss_limit, raw)
        base = equity[:, day] if cfg.compound else np.full(n_paths, cfg.capital)
        pnl = ret * base
        equity[:, day + 1] = np.maximum(equity[:, day] + pnl, 0.0)  # pas de capital négatif
        returns[:, day] = np.divide(pnl, equity[:, day], out=np.zeros(n_paths), where=equity[:, day] > 0)
        trades[:, day] = n
    return SimResult(equity, returns, trades)


@dataclass(frozen=True)
class Summary:
    pnl_pct: FloatArray         # PnL net par chemin, en % du capital initial
    pnl_usd: FloatArray
    max_drawdown_pct: FloatArray
    sharpe: FloatArray
    ruin_probability: float
    loss_probability: float
    paths: int


def summarize(cfg: SimConfig, result: SimResult) -> Summary:
    final = result.equity[:, -1]
    pnl_usd = final - cfg.capital
    peak = np.maximum.accumulate(result.equity, axis=1)
    drawdown = np.where(peak > 0, (peak - result.equity) / peak, 0.0)
    std = result.daily_returns.std(axis=1, ddof=1) if cfg.days > 1 else np.zeros(cfg.paths)
    mean = result.daily_returns.mean(axis=1)
    sharpe = np.divide(mean, std, out=np.full(cfg.paths, np.nan), where=std > 0) * math.sqrt(ANNUALIZATION_DAYS)
    return Summary(
        pnl_pct=pnl_usd / cfg.capital * 100.0, pnl_usd=pnl_usd, max_drawdown_pct=drawdown.max(axis=1) * 100.0, sharpe=sharpe,
        ruin_probability=float(np.mean(final < RUIN_EQUITY_FRACTION * cfg.capital)), loss_probability=float(np.mean(final < cfg.capital)), paths=cfg.paths,
    )


def _q(values: FloatArray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def mean_confidence_interval(values: FloatArray) -> tuple[float, float]:
    """Moyenne +/- 1,96 erreur-type : l'erreur de la SIMULATION, pas l'incertitude sur la stratégie."""
    finite = values[np.isfinite(values)]
    mean = float(finite.mean())
    half = 1.96 * float(finite.std(ddof=1)) / math.sqrt(finite.size) if finite.size > 1 else float("nan")
    return mean - half, mean + half


def render(cfg: SimConfig, econ: TradeEconomics, summary: Summary, venue: str) -> str:
    pnl, usd = summary.pnl_pct, summary.pnl_usd
    lo, hi = mean_confidence_interval(pnl)
    lines = [
        "=" * 78,
        f"SIMULATION MONTE CARLO : {cfg.days} jours, {summary.paths} chemins, capital {cfg.capital:,.0f} $, frais {venue}, "
        f"{'capitalisé' if cfg.compound else 'taille fixe'}",
        "=" * 78,
        "HYPOTHÈSES (aucune n'est une mesure : voir l'en-tête du script)",
        f"  taux de réussite {cfg.win_rate:.1%} (incertitude +/- {cfg.win_rate_std:.1%}) | gain/perte {cfg.rr:g} | stop {cfg.stop_bps:g} bps, objectif {cfg.take_profit_bps:g} bps",
        f"  trades/jour {cfg.trades_min:g} à {cfg.trades_max:g} | risque/trade {cfg.risk_pct:g} % de l'équité (levier {cfg.leverage:.2f}x) | coupe-circuit {cfg.daily_loss_limit_pct:g} %/jour",
        f"  frais maker/taker {cfg.maker_bps:g}/{cfg.taker_bps:g} bps | glissement {cfg.slippage_bps:g} bps par exécution taker",
        "ÉCONOMIE D'UN TRADE (bps du notionnel, frais payés)",
        f"  gagnant {econ.net_win_bps:+.1f} (frais {econ.cost_win_bps:.1f}) | perdant {econ.net_loss_bps:+.1f} (frais {econ.cost_loss_bps:.1f})",
        f"  taux de réussite de RENTABILITÉ : {econ.breakeven_win_rate:.1%} | espérance par trade à {cfg.win_rate:.1%} : {econ.expectancy_bps:+.2f} bps",
    ]
    if econ.expectancy_bps <= 0:
        lines.append("  >>> espérance par trade NÉGATIVE : le résultat moyen est négatif quelle que soit la variance ; plus de trades perd plus vite.")
    lines += [
        "PNL NET SUR LA PÉRIODE (en % du capital, puis en $)",
        f"  pire scénario    (P2,5)  : {_q(pnl, 2.5):+8.2f} %  ({_q(usd, 2.5):+12,.0f} $)",
        f"  scénario moyen           : {float(np.mean(pnl)):+8.2f} %  ({float(np.mean(usd)):+12,.0f} $)   IC95 % de la moyenne [{lo:+.2f} ; {hi:+.2f}] %",
        f"  scénario médian  (P50)   : {_q(pnl, 50):+8.2f} %  ({_q(usd, 50):+12,.0f} $)",
        f"  meilleur scénario (P97,5): {_q(pnl, 97.5):+8.2f} %  ({_q(usd, 97.5):+12,.0f} $)",
        f"  extrêmes simulés         : {float(pnl.min()):+.2f} % ... {float(pnl.max()):+.2f} %",
        f"  fourchette à 95 % des chemins : [{_q(pnl, 2.5):+.2f} % ; {_q(pnl, 97.5):+.2f} %]",
        f"  probabilité de finir en perte : {summary.loss_probability:.1%} | de finir sous {RUIN_EQUITY_FRACTION:.0%} du capital : {summary.ruin_probability:.2%}",
        "RISQUE",
        f"  drawdown maximal (courbe journalière) : médian {_q(summary.max_drawdown_pct, 50):.2f} % | P95 {_q(summary.max_drawdown_pct, 95):.2f} % | pire {float(summary.max_drawdown_pct.max()):.2f} %",
        f"  Sharpe annualisé ({ANNUALIZATION_DAYS} j) : médian {_q(summary.sharpe, 50):+.2f} | fourchette à 95 % [{_q(summary.sharpe, 2.5):+.2f} ; {_q(summary.sharpe, 97.5):+.2f}]",
    ]
    return "\n".join(lines)


def sensitivity_table(cfg: SimConfig, win_rates: Sequence[float], stops: Sequence[float], paths: int = 3000) -> str:
    """PnL médian (%) selon le taux de réussite (lignes) et la distance du stop (colonnes) : montre à quel point les frais décident."""
    head = "SENSIBILITÉ : PnL médian en % (lignes : taux de réussite ; colonnes : stop en bps, objectif = rr x stop)\n" + " " * 10 + "".join(f"{s:>10g}" for s in stops)
    rows = []
    for p in win_rates:
        cells = []
        for s in stops:
            variant = replace(cfg, win_rate=p, stop_bps=s, paths=paths)
            cells.append(f"{_q(summarize(variant, simulate(variant)).pnl_pct, 50):>+10.1f}")
        rows.append(f"{p:>9.0%} " + "".join(cells))
    return head + "\n" + "\n".join(rows)


def _parse(argv: Sequence[str] | None) -> tuple[SimConfig, str, bool]:
    parser = argparse.ArgumentParser(description="Monte Carlo du PnL net de Nothierly sous hypothèses explicites.")
    parser.add_argument("--venue", choices=sorted(VENUE_FEES_BPS), default="binance")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--win-rate", type=float, default=WIN_RATE_MEAN)
    parser.add_argument("--win-rate-std", type=float, default=WIN_RATE_STD)
    parser.add_argument("--rr", type=float, default=RR)
    parser.add_argument("--stop-bps", type=float, default=STOP_BPS)
    parser.add_argument("--risk-pct", type=float, default=RISK_PER_TRADE_PCT)
    parser.add_argument("--trades-min", type=float, default=TRADES_PER_DAY_MIN)
    parser.add_argument("--trades-max", type=float, default=TRADES_PER_DAY_MAX)
    parser.add_argument("--no-compound", action="store_true", help="tailles fixées sur le capital initial (pas de capitalisation)")
    parser.add_argument("--no-sensitivity", action="store_true", help="n'affiche pas le tableau de sensibilité")
    args = parser.parse_args(argv)
    maker, taker = VENUE_FEES_BPS[args.venue]
    try:
        cfg = SimConfig(
            capital=args.capital, days=args.days, paths=args.paths, seed=args.seed, trades_min=args.trades_min, trades_max=args.trades_max,
            win_rate=args.win_rate, win_rate_std=args.win_rate_std, rr=args.rr, stop_bps=args.stop_bps, risk_pct=args.risk_pct,
            maker_bps=maker, taker_bps=taker, compound=not args.no_compound,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return cfg, args.venue, not args.no_sensitivity


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # console Windows en cp1252 : les accents du rapport doivent s'afficher
        reconfigure(encoding="utf-8")
    cfg, venue, sensitivity = _parse(argv)
    summary = summarize(cfg, simulate(cfg))
    print(render(cfg, trade_economics(cfg), summary, venue))
    if sensitivity:
        print()
        print(sensitivity_table(cfg, (0.50, 0.52, 0.54, 0.56, 0.58, 0.60), (20.0, 40.0, 60.0, 100.0)))
    print("\nRappel : ces chiffres découlent des hypothèses ci-dessus, pas d'une mesure de Nothierly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

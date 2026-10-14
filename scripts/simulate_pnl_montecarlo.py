"""Simulation Monte Carlo du PnL net de Nothierly sur ~45 jours, à partir d'HYPOTHÈSES que vous fixez ci-dessous.

    python scripts/simulate_pnl_montecarlo.py                       # 10 000 chemins, 45 jours, frais Binance, régimes + queues de stop + funding actifs
    python scripts/simulate_pnl_montecarlo.py --venue hyperliquid --win-rate 0.55 --rr 1.2 --stop-bps 50
    python scripts/simulate_pnl_montecarlo.py --adverse-selection-bps 2 --max-notional 250000
    python scripts/simulate_pnl_montecarlo.py --no-regimes --tail-prob 0 --funding-drag-bps 0   # modèle « propre » sans les briques 3, 4, 5

**Ce que ce script est, et n'est pas.** C'est un calculateur de conséquences : « SI le taux de réussite est X, le rapport gain/perte Y et les frais Z, ALORS la
distribution du PnL à 45 jours est celle-ci ». Il ne prédit pas le PnL de Nothierly : aucun modèle n'est entraîné aujourd'hui et aucune donnée réelle n'a encore été
mesurée. Les valeurs par défaut sont illustratives (un ordre de grandeur plausible pour du scalping ML), pas des résultats. Le seul chiffre honnête sur le futur réel sera
le taux de réussite mesuré HORS ÉCHANTILLON sur les données du Shadow Run, à reporter dans ``WIN_RATE_MEAN``.

**Comment ajuster les probabilités PPO / XGBoost.** Le script ne contient pas les modèles : il n'a besoin que de leur effet net, un taux de réussite par trade.
- ``WIN_RATE_MEAN`` : la proportion de trades gagnants OBSERVÉE hors échantillon, après vos filtres (seuil de confiance du consensus PPO/XGBoost, méta-label qui
  veto). Ce n'est pas l'AUC ni la précision brute d'XGBoost. Pour un chiffre prudent, utilisez la borne basse de l'intervalle de confiance (par exemple celle du test de
  significativité de ``risk/model_validation.py``), pas l'estimation centrale.
- ``WIN_RATE_STD`` : votre incertitude sur ce taux. C'est le paramètre qui élargit le plus l'éventail des résultats ; ne le mettez pas à 0.
- ``DAILY_WIN_RATE_STD`` : bruit INDÉPENDANT d'un jour à l'autre. Avec les régimes (brique 3) il est remplacé par la chaîne de Markov.
- ``RR`` et ``STOP_BPS`` : le script affiche le taux de réussite de RENTABILITÉ pour vos réglages ; s'il dépasse votre ``WIN_RATE_MEAN``, l'espérance par trade est négative.

**Modèle (vectorisé : chemins x jours, sans boucle sur les trades).** Par chemin : un taux de réussite tiré une fois (incertitude du modèle), un nombre moyen de
signaux par jour tiré une fois dans [TRADES_MIN, TRADES_MAX]. Chaque jour : signaux ~ Poisson, taux du jour, gagnants ~ Binomiale. Chaque trade exécuté gagne
``RR x STOP_BPS`` ou perd ``STOP_BPS`` (points de base du notionnel), moins frais, glissement, impact, portage ; la dispersion des gains et pertes réalisés est ajoutée par
approximation normale de la somme journalière. La taille vise ``RISK_PER_TRADE_PCT`` de l'équité par stop (levier plafonné). Le coupe-circuit journalier est un arrêt EN
COURS de journée (pont brownien). Capitalisation journalière (``--no-compound`` : tailles fixées sur le capital initial).

**Brique 1 : sélection adverse (ordres limites).** Les perdants sont TOUJOURS exécutés, une partie des gagnants part sans nous (« unfilled »). ``--adverse-selection-bps`` = le
profit espéré perdu PAR SIGNAL maker du fait de ces gagnants manqués ; il fixe ``q = bps / (taux de réussite x gain net maker)``, bornée à 1. Espérance par signal réduite
de ``part_maker x bps`` (formule exacte). Ne touche pas les entrées taker.

**Brique 2 : plafond de liquidité et impact de marché.** ``--max-notional`` plafonne le notionnel d'UN trade (le levier effectif baisse à mesure que l'équité croît).
``--adv-usd`` + ``--daily-vol-bps`` + ``--impact-coef`` : impact en racine carrée par exécution taker = coef x volatilité_jour x racine(notionnel / volume_jour), en bps.

**Brique 3 : régimes de marché (chaîne de Markov à deux états).** État 0 = tendance, état 1 = chop/toxique. Chaque chemin suit sa propre chaîne (matrice de transition
``--p-stay-trend`` / ``--p-stay-toxic``, départ au régime stationnaire) : les régimes durent, donc les mauvaises journées arrivent en GRAPPES. Dans le régime toxique, le taux de
réussite subit une pénalité (``--toxic-winrate-penalty``) et l'écart de la journée à son espérance est multiplié par ``--toxic-vol-mult`` (variance x mult au carré). Par défaut les décalages sont
CENTRÉS sur la loi stationnaire : ``--win-rate`` reste la moyenne de long terme, le régime ne fait que créer la grappe (``--regimes-lower-mean`` pour une pénalité qui baisse aussi la moyenne).

**Brique 4 : stops à queue épaisse.** Une part ``--tail-prob`` des trades PERDANTS ne sort pas au stop mais dans un vide de liquidité : la perte vaut M x STOP_BPS, avec M tiré dans une
loi de Pareto tronquée à ``[--tail-mult-low, --tail-mult-high]`` (indice ``--tail-alpha``). Ne touche pas les gagnants. Le coût moyen est inclus dans le taux de rentabilité.

**Brique 5 : coût de portage (funding).** ``--funding-drag-bps`` = coût en bps du notionnel PAR PÉRIODE DE 8 HEURES (positif = la position paie, négatif = elle reçoit ; 1 bps = 0,01 %,
le barème de base de Binance/Bybit). Un trade détenu ``--avg-hold-minutes`` croise en moyenne ``détention / intervalle`` échéances de funding (8 h Binance/Bybit, 1 h Hyperliquid) et paie à chacune
``drag x intervalle / 8`` : même coût moyen quel que soit l'intervalle (``drag x détention en heures / 8``), mais lumpy à 8 h et lissé à 1 h. Proportionnel à l'exposition (notionnel).
Pour un scalp de quelques minutes le portage est minime ; il compte dès que la détention se compte en heures.

**Valeurs par défaut.** La bibliothèque (``SimConfig()``) a des briques 3, 4, 5 NEUTRES ; la LIGNE DE COMMANDE les active (régimes, 5 % de queues de stop, funding de 1 bp / 8 h) pour
approcher le jumeau numérique. Désactivation : ``--no-regimes --tail-prob 0 --funding-drag-bps 0``.

**Limites connues (non modélisées).** Latence et exécutions manquées autres que la sélection adverse, ordres partiellement remplis, changement de régime durable (les régimes
reviennent à leur loi stationnaire), pannes, corrélation entre actifs, signe du funding corrélé à la direction du trade (un seul paramètre moyen), participation agrégée au volume (l'impact
suppose des ordres petits devant le volume). Drawdown mesuré sur la courbe journalière : il sous-estime le creux intra-journalier. Les frais par défaut sont des barèmes
publics de base (sans palier VIP ni remise) à VÉRIFIER sur votre compte avant de s'y fier.

**Intervalles.** « Fourchette à 95 % » = percentiles 2,5 et 97,5 de la distribution des chemins. L'intervalle de confiance de la MOYENNE (moyenne +/- 1,96 écart-type / racine du
nombre de chemins) ne mesure que l'erreur de la simulation, pas l'incertitude sur la stratégie.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# --- Hypothèses (modifiables ici, ou par la ligne de commande) -------------------------------------------------------------------------------------------------
INITIAL_CAPITAL: Final[float] = 10_000.0
DAYS: Final[int] = 45
N_PATHS: Final[int] = 10_000
SEED: Final[int] = 42
TRADES_PER_DAY_MIN: Final[float] = 50.0
TRADES_PER_DAY_MAX: Final[float] = 500.0
WIN_RATE_MEAN: Final[float] = 0.56          # taux de trades gagnants attendu (réaliste pour du ML : 0,54 à 0,58)
WIN_RATE_STD: Final[float] = 0.015          # incertitude sur le VRAI taux (écart-type, tiré une fois par chemin)
DAILY_WIN_RATE_STD: Final[float] = 0.01     # bruit indépendant d'un jour à l'autre (remplacé par les régimes s'ils sont actifs)
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
DEFAULT_DAILY_VOL_BPS: Final[float] = 250.0  # volatilité journalière de l'instrument pour la loi d'impact (BTC : de l'ordre de 2 à 3 %)
DEFAULT_IMPACT_COEF: Final[float] = 0.7     # coefficient sans dimension de la loi de la racine carrée (ordre de grandeur de la littérature : 0,5 à 1)
# Briques 3, 4, 5 : valeurs de la LIGNE DE COMMANDE (la bibliothèque, elle, est neutre par défaut).
DEFAULT_P_STAY_TREND: Final[float] = 0.90   # probabilité de rester en tendance le lendemain (épisode moyen : 10 jours)
DEFAULT_P_STAY_TOXIC: Final[float] = 0.75   # probabilité de rester en chop/toxique le lendemain (épisode moyen : 4 jours)
DEFAULT_TREND_BONUS: Final[float] = 0.01    # décalage du taux de réussite en tendance (avant centrage)
DEFAULT_TOXIC_PENALTY: Final[float] = 0.04  # pénalité du taux de réussite en régime toxique (avant centrage)
DEFAULT_TOXIC_VOL_MULT: Final[float] = 1.6  # multiplicateur de l'écart de la journée à son espérance en régime toxique
DEFAULT_TAIL_PROB: Final[float] = 0.05      # part des trades perdants victimes d'un stop à queue épaisse
DEFAULT_TAIL_MULT_LOW: Final[float] = 2.0
DEFAULT_TAIL_MULT_HIGH: Final[float] = 3.0
DEFAULT_TAIL_ALPHA: Final[float] = 2.5      # indice de la loi de Pareto (plus petit = queue plus épaisse)
DEFAULT_FUNDING_DRAG_BPS: Final[float] = 1.0  # bps du notionnel par période de 8 h ; positif = coût (stratégie momentum qui paie souvent le funding)
DEFAULT_AVG_HOLD_MINUTES: Final[float] = 15.0
FUNDING_QUOTE_HOURS: Final[float] = 8.0     # le paramètre ``drag`` est exprimé par période de 8 heures
_TINY: Final[float] = 1e-12

# Barèmes publics de base, en points de base du notionnel (maker, taker). À VÉRIFIER sur la grille de frais actuelle de votre compte.
VENUE_FEES_BPS: Final[dict[str, tuple[float, float]]] = {
    "binance": (2.0, 5.0),
    "hyperliquid": (1.5, 4.5),
    "bybit": (2.0, 5.5),
}
VENUE_FUNDING_HOURS: Final[dict[str, float]] = {"binance": 8.0, "hyperliquid": 1.0, "bybit": 8.0}


# --- Paramètres par brique -----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdverseSelectionParams:
    """Brique 1. ``bps`` = profit espéré perdu par signal maker (gagnants non exécutés). 0 = désactivée."""

    bps: float = 0.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("adverse_selection_bps ne peut pas être négatif")

    @property
    def enabled(self) -> bool:
        return self.bps > 0


@dataclass(frozen=True)
class LiquidityParams:
    """Brique 2. ``max_notional_usd`` : plafond du notionnel d'un trade. ``adv_usd`` : volume quotidien de l'instrument (active la loi d'impact en racine carrée)."""

    max_notional_usd: float | None = None
    adv_usd: float | None = None
    daily_vol_bps: float = DEFAULT_DAILY_VOL_BPS
    impact_coef: float = DEFAULT_IMPACT_COEF

    def __post_init__(self) -> None:
        if self.max_notional_usd is not None and self.max_notional_usd <= 0:
            raise ValueError("max_notional doit être > 0")
        if self.adv_usd is not None and self.adv_usd <= 0:
            raise ValueError("adv_usd doit être > 0")
        if self.daily_vol_bps <= 0 or self.impact_coef < 0:
            raise ValueError("daily_vol_bps doit être > 0 et impact_coef >= 0")

    @property
    def cap_enabled(self) -> bool:
        return self.max_notional_usd is not None

    @property
    def impact_enabled(self) -> bool:
        return self.adv_usd is not None and self.impact_coef > 0


@dataclass(frozen=True)
class RegimeParams:
    """Brique 3 : deux régimes de Markov, tendance (0) et chop/toxique (1). ``initial_probs=None`` : départ à la loi stationnaire."""

    enabled: bool = False
    transition: tuple[tuple[float, float], tuple[float, float]] = ((DEFAULT_P_STAY_TREND, 1 - DEFAULT_P_STAY_TREND), (1 - DEFAULT_P_STAY_TOXIC, DEFAULT_P_STAY_TOXIC))
    win_rate_shift: tuple[float, float] = (DEFAULT_TREND_BONUS, -DEFAULT_TOXIC_PENALTY)   # décalage du taux de réussite dans chaque régime
    volatility_multiplier: tuple[float, float] = (1.0, DEFAULT_TOXIC_VOL_MULT)           # multiplicateur de l'écart du jour à son espérance
    initial_probs: tuple[float, float] | None = None
    preserve_mean: bool = True                                                            # centre les décalages sur la loi stationnaire

    def __post_init__(self) -> None:
        rows = [*self.transition] + ([self.initial_probs] if self.initial_probs is not None else [])
        for row in rows:
            if min(row) < 0 or abs(sum(row) - 1.0) > 1e-9:
                raise ValueError("les lignes de transition et les probabilités initiales doivent être positives et sommer à 1")
        if min(self.volatility_multiplier) <= 0:
            raise ValueError("les multiplicateurs de volatilité doivent être > 0")
        if max(abs(x) for x in self.win_rate_shift) >= 0.5:
            raise ValueError("un décalage de taux de réussite de 50 points ou plus n'a pas de sens")

    def stationary(self) -> tuple[float, float]:
        """Loi stationnaire de la chaîne à deux états : ``(P10, P01) / (P01 + P10)``. Chaîne figée (aucune transition) : la loi initiale, ou (1, 0)."""
        p01, p10 = self.transition[0][1], self.transition[1][0]
        if p01 + p10 <= 0:
            return self.initial_probs if self.initial_probs is not None else (1.0, 0.0)
        return (p10 / (p01 + p10), p01 / (p01 + p10))

    def start_probs(self) -> tuple[float, float]:
        return self.initial_probs if self.initial_probs is not None else self.stationary()

    def effective_shifts(self) -> tuple[float, float]:
        """Décalages appliqués au taux de réussite ; centrés sur la loi stationnaire si ``preserve_mean`` (la moyenne de long terme reste ``win_rate``)."""
        if not self.preserve_mean:
            return self.win_rate_shift
        pi = self.stationary()
        mean = pi[0] * self.win_rate_shift[0] + pi[1] * self.win_rate_shift[1]
        return (self.win_rate_shift[0] - mean, self.win_rate_shift[1] - mean)

    def mean_toxic_spell_days(self) -> float:
        stay = self.transition[1][1]
        return math.inf if stay >= 1.0 else 1.0 / (1.0 - stay)


def truncated_pareto_moment(order: float, low: float, high: float, alpha: float) -> float:
    """``E[M^order]`` pour une loi de Pareto d'indice ``alpha`` tronquée à ``[low, high]`` (``high`` peut valoir l'infini ; moment infini si ``order >= alpha``)."""
    finite_high = math.isfinite(high)
    ratio = (low / high) ** alpha if finite_high else 0.0
    norm = 1.0 - ratio
    if abs(order - alpha) < 1e-12:
        return float(alpha * low ** alpha / norm * math.log(high / low)) if finite_high else math.inf
    if not finite_high:
        return math.inf if order > alpha else float(alpha * low ** order / (alpha - order))
    return float(alpha * low ** alpha / norm * (high ** (order - alpha) - low ** (order - alpha)) / (order - alpha))


def sample_stop_multiplier(u: FloatArray, low: float, high: float, alpha: float) -> FloatArray:
    """Inverse de la fonction de répartition de la Pareto tronquée : ``u`` uniforme sur [0, 1[ -> multiplicateur M dans ``[low, high]``."""
    ratio = (low / high) ** alpha if math.isfinite(high) else 0.0
    return np.asarray(low / (1.0 - u * (1.0 - ratio)) ** (1.0 / alpha), dtype=np.float64)


@dataclass(frozen=True)
class StopTailParams:
    """Brique 4 : une part ``tail_prob`` des trades PERDANTS sort à M x STOP_BPS, M ~ Pareto tronquée à ``[multiplier_low, multiplier_high]``, indice ``alpha``."""

    enabled: bool = False
    tail_prob: float = DEFAULT_TAIL_PROB
    multiplier_low: float = DEFAULT_TAIL_MULT_LOW
    multiplier_high: float = DEFAULT_TAIL_MULT_HIGH
    alpha: float = DEFAULT_TAIL_ALPHA

    def __post_init__(self) -> None:
        if not 0 <= self.tail_prob <= 1 or not 1 <= self.multiplier_low <= self.multiplier_high:
            raise ValueError("tail_prob dans [0, 1] et 1 <= multiplier_low <= multiplier_high sont requis")
        if self.alpha <= 0:
            raise ValueError("alpha doit être > 0")
        if math.isinf(self.multiplier_high) and self.alpha <= 2:
            raise ValueError("avec une borne haute infinie, alpha doit dépasser 2 (sinon la variance de la perte est infinie)")

    @property
    def active(self) -> bool:
        return self.enabled and self.tail_prob > 0

    def mean_multiplier(self) -> float:
        return truncated_pareto_moment(1.0, self.multiplier_low, self.multiplier_high, self.alpha)

    def mean_extra(self) -> float:
        """``E[M - 1]`` : glissement moyen au-delà du stop, en multiples de STOP_BPS, pour un trade victime d'un vide de liquidité."""
        return self.mean_multiplier() - 1.0

    def second_moment_extra(self) -> float:
        second = truncated_pareto_moment(2.0, self.multiplier_low, self.multiplier_high, self.alpha)
        return second - 2.0 * self.mean_multiplier() + 1.0


@dataclass(frozen=True)
class FundingParams:
    """Brique 5 : coût de portage des perpétuels. ``drag_bps`` : bps du notionnel par période de 8 h (positif = coût, négatif = reçu). ``interval_hours`` : 1 (Hyperliquid) ou 8 (Binance, Bybit)."""

    drag_bps: float = 0.0
    interval_hours: float = FUNDING_QUOTE_HOURS
    avg_hold_minutes: float = DEFAULT_AVG_HOLD_MINUTES

    def __post_init__(self) -> None:
        if self.interval_hours <= 0 or self.avg_hold_minutes < 0:
            raise ValueError("interval_hours > 0 et avg_hold_minutes >= 0 sont requis")

    @property
    def enabled(self) -> bool:
        return self.drag_bps != 0.0 and self.avg_hold_minutes > 0

    @property
    def hold_hours(self) -> float:
        return self.avg_hold_minutes / 60.0

    @property
    def payment_bps(self) -> float:
        """Montant payé à chaque échéance franchie : le coût de 8 h ramené à l'intervalle."""
        return self.drag_bps * self.interval_hours / FUNDING_QUOTE_HOURS

    @property
    def crossings_per_trade(self) -> float:
        """Nombre moyen d'échéances franchies par un trade : détention / intervalle (peut dépasser 1)."""
        return self.hold_hours / self.interval_hours

    @property
    def crossing_probability(self) -> float:
        """Probabilité qu'un trade franchisse au moins une échéance (borne à 1)."""
        return min(1.0, self.crossings_per_trade)

    @property
    def cost_per_trade_bps(self) -> float:
        """Coût MOYEN par trade : ``drag x détention en heures / 8``, identique quel que soit l'intervalle."""
        return self.drag_bps * self.hold_hours / FUNDING_QUOTE_HOURS if self.enabled else 0.0


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
    adverse: AdverseSelectionParams = field(default_factory=AdverseSelectionParams)
    liquidity: LiquidityParams = field(default_factory=LiquidityParams)
    regime: RegimeParams = field(default_factory=RegimeParams)
    stop_tail: StopTailParams = field(default_factory=StopTailParams)
    funding: FundingParams = field(default_factory=FundingParams)

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


# --- Économie d'un trade ------------------------------------------------------------------------------------------------------------------------------------


def impact_bps(notional_usd: float | FloatArray, adv_usd: float, daily_vol_bps: float, coef: float) -> float | FloatArray:
    """Loi d'impact en racine carrée, par exécution taker : ``coef x volatilité_jour x racine(notionnel / volume_jour)``, en bps."""
    return coef * daily_vol_bps * np.sqrt(np.maximum(notional_usd, 0.0) / adv_usd)


@dataclass(frozen=True)
class LegEconomics:
    """Gain et perte nets (bps du notionnel, frais et glissement payés, SANS impact, portage ni queue de stop) selon le type d'ENTRÉE ; la sortie suit ``tp_maker_fraction``."""

    net_win_maker_entry: float
    net_win_taker_entry: float
    net_loss_maker_entry: float
    net_loss_taker_entry: float
    exit_taker_share_win: float  # part des sorties gagnantes exécutées en taker (le stop, lui, est toujours taker)


def leg_economics(cfg: SimConfig) -> LegEconomics:
    exit_win_fee = cfg.tp_maker_fraction * cfg.maker_bps + (1 - cfg.tp_maker_fraction) * cfg.taker_bps
    exit_win_slip = cfg.slippage_bps * (1 - cfg.tp_maker_fraction)
    exit_loss = cfg.taker_bps + cfg.slippage_bps  # un stop est un ordre au marché
    entry_maker = cfg.maker_bps
    entry_taker = cfg.taker_bps + cfg.slippage_bps
    tp, stop = cfg.take_profit_bps, cfg.stop_bps
    return LegEconomics(
        net_win_maker_entry=tp - (entry_maker + exit_win_fee + exit_win_slip), net_win_taker_entry=tp - (entry_taker + exit_win_fee + exit_win_slip),
        net_loss_maker_entry=-(stop + entry_maker + exit_loss), net_loss_taker_entry=-(stop + entry_taker + exit_loss),
        exit_taker_share_win=1 - cfg.tp_maker_fraction,
    )


@dataclass(frozen=True)
class TradeEconomics:
    """Ce que rapporte UN signal moyen, en points de base du notionnel, portage et queues de stop compris (hors impact : il dépend de la taille, donc de l'équité)."""

    net_win_bps: float            # moyenne sur les types d'entrée, portage déduit
    net_loss_bps: float           # négatif ; queues de stop et portage compris
    cost_win_bps: float
    cost_loss_bps: float
    breakeven_win_rate: float     # taux de réussite des SIGNAUX qui annule l'espérance par signal (sélection adverse comprise)
    expectancy_bps: float         # espérance par SIGNAL au taux de réussite configuré (sélection adverse comprise)
    unfilled_winner_prob: float = 0.0   # probabilité qu'un gagnant à entrée maker ne soit pas exécuté
    executed_fraction: float = 1.0      # part des signaux réellement exécutés
    effective_win_rate: float = 0.0     # taux de réussite des trades EXÉCUTÉS
    tail_extra_bps_per_loss: float = 0.0   # coût moyen, par trade perdant, des stops à queue épaisse
    funding_bps_per_trade: float = 0.0     # coût moyen de portage par trade exécuté


def unfilled_winner_probability(adverse_bps: float, win_rate: float | FloatArray, net_win_maker_bps: float) -> float | FloatArray:
    """``q = bps / (p x gain net maker)``, bornée à [0, 1] : le profit espéré perdu par signal maker vaut ``p x q x gain net maker = bps``."""
    if adverse_bps <= 0 or net_win_maker_bps <= 0:
        return np.zeros_like(win_rate) if isinstance(win_rate, np.ndarray) else 0.0
    return np.clip(adverse_bps / (np.maximum(win_rate, _TINY) * net_win_maker_bps), 0.0, 1.0)


def trade_economics(cfg: SimConfig) -> TradeEconomics:
    legs = leg_economics(cfg)
    m = cfg.entry_maker_fraction
    funding = cfg.funding.cost_per_trade_bps
    tail_extra = cfg.stop_tail.tail_prob * cfg.stop_bps * cfg.stop_tail.mean_extra() if cfg.stop_tail.active else 0.0
    net_win = m * legs.net_win_maker_entry + (1 - m) * legs.net_win_taker_entry - funding
    net_loss = m * legs.net_loss_maker_entry + (1 - m) * legs.net_loss_taker_entry - funding - tail_extra
    p = cfg.win_rate
    q = float(unfilled_winner_probability(cfg.adverse.bps, p, legs.net_win_maker_entry))
    adverse_cost = m * p * q * legs.net_win_maker_entry  # = m x bps tant que q < 1 : profit espéré perdu par signal
    executed = 1.0 - m * p * q
    expectancy = p * net_win + (1 - p) * net_loss - adverse_cost
    spread = net_win - net_loss
    breakeven = (-net_loss + m * cfg.adverse.bps) / spread if spread > 0 else float("nan")
    effective_p = p * (1 - m * q) / executed if executed > 0 else float("nan")
    return TradeEconomics(
        net_win_bps=net_win, net_loss_bps=net_loss, cost_win_bps=cfg.take_profit_bps - net_win, cost_loss_bps=-net_loss - cfg.stop_bps,
        breakeven_win_rate=breakeven, expectancy_bps=expectancy, unfilled_winner_prob=q, executed_fraction=executed, effective_win_rate=effective_p,
        tail_extra_bps_per_loss=tail_extra, funding_bps_per_trade=funding,
    )


# --- Simulation ------------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    equity: FloatArray          # (chemins, jours + 1), capital initial en colonne 0
    daily_returns: FloatArray   # (chemins, jours), fraction du capital du début de journée
    trades: FloatArray          # (chemins, jours), trades EXÉCUTÉS
    signals: FloatArray | None = None        # signaux émis (exécutés + gagnants non exécutés)
    unfilled: FloatArray | None = None       # gagnants maker non exécutés (sélection adverse)
    leverage_used: FloatArray | None = None  # notionnel d'un trade / équité, après plafond
    impact_bps_per_fill: FloatArray | None = None  # impact moyen par exécution taker
    regimes: IntArray | None = None          # régime de chaque jour (0 tendance, 1 toxique), si les régimes sont actifs
    loser_trades: FloatArray | None = None   # trades perdants exécutés
    tail_events: FloatArray | None = None    # perdants victimes d'un stop à queue épaisse
    funding_bps: FloatArray | None = None    # coût de portage total du jour, en bps du notionnel (somme sur les trades)


def simulate(cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    legs = leg_economics(cfg)
    m = cfg.entry_maker_fraction
    n_paths = cfg.paths
    regime, tail, funding = cfg.regime, cfg.stop_tail, cfg.funding
    path_win_rate = np.clip(rng.normal(cfg.win_rate, cfg.win_rate_std, n_paths), 0.01, 0.99)  # incertitude du modèle : un taux par chemin
    path_trades = rng.uniform(cfg.trades_min, cfg.trades_max, n_paths)                          # un régime d'activité par chemin
    loss_limit = cfg.daily_loss_limit_pct / 100.0
    equity = np.empty((n_paths, cfg.days + 1))
    equity[:, 0] = cfg.capital
    returns = np.empty((n_paths, cfg.days))
    executed_grid = np.empty((n_paths, cfg.days))
    signals_grid = np.empty((n_paths, cfg.days))
    unfilled_grid = np.empty((n_paths, cfg.days))
    leverage_grid = np.empty((n_paths, cfg.days))
    impact_grid = np.zeros((n_paths, cfg.days))
    regime_grid = np.zeros((n_paths, cfg.days), dtype=np.int64)
    loser_grid = np.zeros((n_paths, cfg.days))
    tail_grid = np.zeros((n_paths, cfg.days))
    funding_grid = np.zeros((n_paths, cfg.days))
    tp, stop = cfg.take_profit_bps, cfg.stop_bps
    liquidity = cfg.liquidity
    # Brique 3 : un régime par chemin, tiré à la loi de départ ; il évolue chaque jour par la matrice de transition.
    shifts = np.array(regime.effective_shifts())
    vol_mult = np.array(regime.volatility_multiplier)
    to_toxic = np.array([regime.transition[0][1], regime.transition[1][1]])  # P(toxique demain | régime aujourd'hui)
    state = (rng.random(n_paths) < regime.start_probs()[1]).astype(np.int64) if regime.enabled else np.zeros(n_paths, dtype=np.int64)
    tail_extra_1, tail_extra_2 = (tail.mean_extra(), tail.second_moment_extra()) if tail.active else (0.0, 0.0)
    cross_base = math.floor(funding.crossings_per_trade)
    cross_frac = funding.crossings_per_trade - cross_base
    for day in range(cfg.days):
        base = equity[:, day] if cfg.compound else np.full(n_paths, cfg.capital)
        # --- Brique 2 : plafond de notionnel (le levier effectif baisse quand l'équité croît) et impact racine carrée -----------------------------------------------
        lev = np.full(n_paths, cfg.leverage)
        if liquidity.max_notional_usd is not None:
            lev = np.minimum(lev, liquidity.max_notional_usd / np.maximum(base, _TINY))
        notional = lev * base
        impact = np.zeros(n_paths)
        if liquidity.impact_enabled and liquidity.adv_usd is not None:
            impact = np.asarray(impact_bps(notional, liquidity.adv_usd, liquidity.daily_vol_bps, liquidity.impact_coef), dtype=np.float64)
        # --- Signaux et taux du jour : régime de Markov (brique 3) à la place du bruit indépendant -------------------------------------------------------------------
        n = rng.poisson(path_trades)
        if regime.enabled:
            p_day = np.clip(path_win_rate + shifts[state], 0.01, 0.99)
            day_vol = vol_mult[state]
        else:
            p_day = np.clip(path_win_rate + rng.normal(0.0, cfg.daily_win_rate_std, n_paths), 0.01, 0.99)
            day_vol = np.ones(n_paths)
        # --- Brique 1 : sélection adverse sur les entrées maker ------------------------------------------------------------------------------------------------------
        n_maker = rng.binomial(n, m)
        n_taker = n - n_maker
        wins_taker = rng.binomial(n_taker, p_day)
        wins_maker_signal = rng.binomial(n_maker, p_day)
        q = np.asarray(unfilled_winner_probability(cfg.adverse.bps, p_day, legs.net_win_maker_entry), dtype=np.float64)
        wins_maker = rng.binomial(wins_maker_signal, 1.0 - q) if cfg.adverse.enabled else wins_maker_signal
        unfilled = (wins_maker_signal - wins_maker).astype(np.float64)
        losses_taker = (n_taker - wins_taker).astype(np.float64)
        losses_maker = (n_maker - wins_maker_signal).astype(np.float64)  # un perdant maker est TOUJOURS exécuté (flux toxique)
        w_t, w_m = wins_taker.astype(np.float64), wins_maker.astype(np.float64)
        wins, losses = w_t + w_m, losses_taker + losses_maker
        executed = wins + losses
        # Gains et pertes nets par catégorie, impact compris (taker seulement : entrée taker, sortie gagnante taker, stop toujours taker).
        nw_t = legs.net_win_taker_entry - impact * (1.0 + legs.exit_taker_share_win)
        nw_m = legs.net_win_maker_entry - impact * legs.exit_taker_share_win
        nl_t = legs.net_loss_taker_entry - impact * 2.0
        nl_m = legs.net_loss_maker_entry - impact
        mean_bps = w_t * nw_t + w_m * nw_m + losses_taker * nl_t + losses_maker * nl_m
        # --- Brique 4 : stops à queue épaisse (perdants seulement) ---------------------------------------------------------------------------------------------------
        tail_bps = np.zeros(n_paths)
        tail_count = np.zeros(n_paths, dtype=np.int64)
        if tail.active:
            tail_count = rng.binomial(losses.astype(np.int64), tail.tail_prob)
            if int(tail_count.sum()) > 0:
                multipliers = sample_stop_multiplier(rng.random(int(tail_count.sum())), tail.multiplier_low, tail.multiplier_high, tail.alpha)
                tail_bps = np.bincount(np.repeat(np.arange(n_paths), tail_count), weights=stop * (multipliers - 1.0), minlength=n_paths)
        # --- Brique 5 : portage (funding), proportionnel au nombre de trades donc à l'exposition ---------------------------------------------------------------------
        funding_bps = np.zeros(n_paths)
        if funding.enabled:
            crossings = executed * cross_base + (rng.binomial(executed.astype(np.int64), cross_frac) if cross_frac > 0 else 0)
            funding_bps = crossings * funding.payment_bps
        var_bps = ((wins * (tp * cfg.payoff_dispersion) ** 2 + losses * (stop * cfg.payoff_dispersion) ** 2 + executed * 2.0 * cfg.slippage_std_bps ** 2))
        noise = rng.standard_normal(n_paths) * np.sqrt(var_bps)  # somme de n trades : approximation normale
        day_bps = mean_bps - tail_bps - funding_bps + noise
        if regime.enabled:
            # Régime : l'écart de la journée à son ESPÉRANCE (hasard binomial, fills, queues, portage) est multiplié par la volatilité du régime ; la moyenne ne bouge pas.
            expected_executed = n_taker + n_maker * (1.0 - p_day * q)
            expected_bps = (n_taker * (p_day * nw_t + (1.0 - p_day) * nl_t) + n_maker * (p_day * (1.0 - q) * nw_m + (1.0 - p_day) * nl_m)
                            - n * (1.0 - p_day) * tail.tail_prob * stop * tail_extra_1 * (1.0 if tail.active else 0.0) - expected_executed * funding.cost_per_trade_bps)
            day_bps = expected_bps + day_vol * (day_bps - expected_bps)
        raw = lev * day_bps / 10_000.0
        # Coupe-circuit = arrêt EN COURS de journée (pont brownien) : plafonner seulement le résultat final serait optimiste. La variance du jour inclut la variance
        # binomiale du nombre de gagnants exécutés, celle des queues de stop et celle du portage ; l'oublier sous-estime les arrêts et laisse un biais positif.
        exec_p = np.divide(wins, executed, out=np.full(n_paths, 0.5), where=executed > 0)
        avg_win = np.divide(w_t * nw_t + w_m * nw_m, wins, out=np.full(n_paths, legs.net_win_taker_entry), where=wins > 0)
        avg_loss = np.divide(losses_taker * nl_t + losses_maker * nl_m, losses, out=np.full(n_paths, legs.net_loss_taker_entry), where=losses > 0)
        tail_var = losses * (tail.tail_prob * tail_extra_2 - (tail.tail_prob * tail_extra_1) ** 2) * stop ** 2 if tail.active else 0.0
        funding_var = executed * cross_frac * (1.0 - cross_frac) * funding.payment_bps ** 2 if funding.enabled else 0.0
        day_var = (lev / 10_000.0) ** 2 * day_vol ** 2 * (var_bps + tail_var + funding_var + executed * exec_p * (1.0 - exec_p) * (avg_win - avg_loss) ** 2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            touched = np.where(day_var > 0, np.exp(-2.0 * loss_limit * (loss_limit + raw) / day_var), 0.0)
        stopped = (raw <= -loss_limit) | (rng.random(n_paths) < np.where(raw > -loss_limit, touched, 1.0))
        ret = np.where(stopped, -loss_limit, raw)
        pnl = ret * base
        equity[:, day + 1] = np.maximum(equity[:, day] + pnl, 0.0)  # pas de capital négatif
        returns[:, day] = np.divide(pnl, equity[:, day], out=np.zeros(n_paths), where=equity[:, day] > 0)
        executed_grid[:, day], signals_grid[:, day], unfilled_grid[:, day] = executed, n.astype(np.float64), unfilled
        leverage_grid[:, day], impact_grid[:, day] = lev, impact
        regime_grid[:, day], loser_grid[:, day], tail_grid[:, day], funding_grid[:, day] = state, losses, tail_count.astype(np.float64), funding_bps
        if regime.enabled:
            state = (rng.random(n_paths) < to_toxic[state]).astype(np.int64)
    return SimResult(
        equity, returns, executed_grid, signals_grid, unfilled_grid, leverage_grid, impact_grid, regime_grid if regime.enabled else None, loser_grid, tail_grid, funding_grid,
    )


# --- Statistiques et rapport ---------------------------------------------------------------------------------------------------------------------------------


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


def lag1_autocorrelation(values: FloatArray) -> float:
    """Corrélation entre le jour t et le jour t+1, sur tous les chemins : proche de 0 si les jours sont indépendants, positive si les bons et mauvais jours viennent en grappes."""
    if values.shape[1] < 3:
        return float("nan")
    x, y = values[:, :-1].ravel(), values[:, 1:].ravel()
    if x.std() <= 0 or y.std() <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def loss_day_persistence(returns: FloatArray) -> tuple[float, float]:
    """``(P(perte demain | perte aujourd'hui), P(perte un jour quelconque))`` : l'écart mesure les grappes de mauvaises journées."""
    loss = returns < 0
    after_loss = loss[:, :-1]
    conditional = float(loss[:, 1:][after_loss].mean()) if after_loss.any() else float("nan")
    return conditional, float(loss.mean())


def _q(values: FloatArray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def mean_confidence_interval(values: FloatArray) -> tuple[float, float]:
    """Moyenne +/- 1,96 erreur-type : l'erreur de la SIMULATION, pas l'incertitude sur la stratégie."""
    finite = values[np.isfinite(values)]
    mean = float(finite.mean())
    half = 1.96 * float(finite.std(ddof=1)) / math.sqrt(finite.size) if finite.size > 1 else float("nan")
    return mean - half, mean + half


def microstructure_lines(cfg: SimConfig, econ: TradeEconomics, result: SimResult | None) -> list[str]:
    """Lignes du rapport pour les briques 1 à 5, seulement si elles sont actives."""
    lines: list[str] = []
    if cfg.adverse.enabled:
        lines.append("SÉLECTION ADVERSE (entrées maker : gagnants parfois non exécutés, perdants toujours exécutés)")
        lines.append(
            f"  {cfg.adverse.bps:g} bps perdus par signal maker -> un gagnant maker est manqué avec probabilité {econ.unfilled_winner_prob:.1%} ; "
            f"{econ.executed_fraction:.1%} des signaux sont exécutés ; taux de réussite des trades exécutés {econ.effective_win_rate:.1%} (signaux : {cfg.win_rate:.1%})"
        )
        if result is not None and result.signals is not None and result.unfilled is not None:
            lines.append(f"  simulé : {float(result.unfilled.sum() / result.signals.sum()):.1%} des signaux non exécutés")
    if cfg.liquidity.cap_enabled or cfg.liquidity.impact_enabled:
        lines.append("LIQUIDITÉ")
        if cfg.liquidity.max_notional_usd is not None:
            binding = float(np.mean(result.leverage_used < cfg.leverage - 1e-9)) if result is not None and result.leverage_used is not None else float("nan")
            lines.append(f"  notionnel maximal par trade {cfg.liquidity.max_notional_usd:,.0f} $ : plafond actif sur {binding:.1%} des jours-chemins (levier visé {cfg.leverage:.2f}x)")
        if cfg.liquidity.impact_enabled and cfg.liquidity.adv_usd is not None:
            mean_impact = float(result.impact_bps_per_fill.mean()) if result is not None and result.impact_bps_per_fill is not None else float("nan")
            lines.append(
                f"  impact racine carrée : coef {cfg.liquidity.impact_coef:g}, volatilité {cfg.liquidity.daily_vol_bps:g} bps/jour, volume {cfg.liquidity.adv_usd:,.0f} $/jour "
                f"-> {mean_impact:.3f} bps par exécution taker en moyenne"
            )
    if cfg.regime.enabled:
        pi = cfg.regime.stationary()
        shifts = cfg.regime.effective_shifts()
        lines.append("RÉGIMES DE MARCHÉ (chaîne de Markov : tendance / chop-toxique)")
        lines.append(
            f"  rester en tendance {cfg.regime.transition[0][0]:.0%} / en toxique {cfg.regime.transition[1][1]:.0%} -> {pi[1]:.1%} du temps en régime toxique, "
            f"épisode toxique moyen {cfg.regime.mean_toxic_spell_days():.1f} j ; taux de réussite {cfg.win_rate + shifts[0]:.1%} (tendance) / {cfg.win_rate + shifts[1]:.1%} (toxique) ; "
            f"volatilité x{cfg.regime.volatility_multiplier[0]:g} / x{cfg.regime.volatility_multiplier[1]:g}"
            f"{' ; moyenne de long terme conservée' if cfg.regime.preserve_mean else ' ; la pénalité baisse aussi la moyenne'} ; bruit journalier indépendant remplacé"
        )
        if result is not None:
            suffix = f" ; part simulée en toxique {float((result.regimes == 1).mean()):.1%}" if result.regimes is not None else ""
            cond, uncond = loss_day_persistence(result.daily_returns)
            lines.append(
                f"  grappes : autocorrélation jour t / t+1 des rendements {lag1_autocorrelation(result.daily_returns):+.2f}, "
                f"P(perte demain | perte aujourd'hui) {cond:.1%} contre {uncond:.1%} sans mémoire{suffix}"
            )
    if cfg.stop_tail.active:
        tail = cfg.stop_tail
        high = "sans borne" if math.isinf(tail.multiplier_high) else f"x{tail.multiplier_high:g}"
        lines.append("QUEUES DE STOP (vide de liquidité : le stop n'est pas exécuté à sa limite)")
        lines.append(
            f"  {tail.tail_prob:.1%} des trades perdants sortent à x{tail.multiplier_low:g}..{high} la distance du stop (Pareto tronquée, indice {tail.alpha:g}), "
            f"multiplicateur moyen x{tail.mean_multiplier():.2f} -> +{econ.tail_extra_bps_per_loss:.2f} bps en moyenne par trade perdant"
        )
        if result is not None and result.tail_events is not None and result.loser_trades is not None and float(result.loser_trades.sum()) > 0:
            lines.append(f"  simulé : {float(result.tail_events.sum() / result.loser_trades.sum()):.2%} des trades perdants touchés")
    if cfg.funding.enabled:
        fund = cfg.funding
        lines.append("PORTAGE (funding des perpétuels)")
        lines.append(
            f"  {fund.drag_bps:+g} bps par 8 h ({'coût' if fund.drag_bps > 0 else 'reçu'}), échéance toutes les {fund.interval_hours:g} h, détention moyenne {fund.avg_hold_minutes:g} min -> "
            f"{fund.crossing_probability:.1%} des trades franchissent une échéance, {fund.cost_per_trade_bps:+.3f} bps par trade"
        )
    return lines


def render(cfg: SimConfig, econ: TradeEconomics, summary: Summary, venue: str, result: SimResult | None = None) -> str:
    pnl, usd = summary.pnl_pct, summary.pnl_usd
    lo, hi = mean_confidence_interval(pnl)
    lines = [
        "=" * 78,
        f"SIMULATION MONTE CARLO : {cfg.days} jours, {summary.paths} chemins, capital {cfg.capital:,.0f} $, frais {venue}, "
        f"{'capitalisé' if cfg.compound else 'taille fixe'}",
        "=" * 78,
        "HYPOTHÈSES (aucune n'est une mesure : voir l'en-tête du script)",
        f"  taux de réussite {cfg.win_rate:.1%} (incertitude +/- {cfg.win_rate_std:.1%}) | gain/perte {cfg.rr:g} | stop {cfg.stop_bps:g} bps, objectif {cfg.take_profit_bps:g} bps",
        f"  signaux/jour {cfg.trades_min:g} à {cfg.trades_max:g} | risque/trade {cfg.risk_pct:g} % de l'équité (levier {cfg.leverage:.2f}x) | coupe-circuit {cfg.daily_loss_limit_pct:g} %/jour",
        f"  frais maker/taker {cfg.maker_bps:g}/{cfg.taker_bps:g} bps | glissement {cfg.slippage_bps:g} bps par exécution taker",
        "ÉCONOMIE D'UN TRADE (bps du notionnel, frais payés, hors impact)",
        f"  gagnant {econ.net_win_bps:+.1f} (frais {econ.cost_win_bps:.1f}) | perdant {econ.net_loss_bps:+.1f} (frais {econ.cost_loss_bps:.1f})",
        f"  taux de réussite de RENTABILITÉ : {econ.breakeven_win_rate:.1%} | espérance par signal à {cfg.win_rate:.1%} : {econ.expectancy_bps:+.2f} bps",
    ]
    lines += microstructure_lines(cfg, econ, result)
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
    parser.add_argument("--adverse-selection-bps", type=float, default=0.0, help="brique 1 : profit espéré perdu par signal maker (gagnants non exécutés) ; 0 = désactivée")
    parser.add_argument("--max-notional", type=float, default=None, help="brique 2 : notionnel maximal d'UN trade, en $ (plafond de liquidité)")
    parser.add_argument("--adv-usd", type=float, default=None, help="brique 2 : volume quotidien de l'instrument, en $ (active l'impact en racine carrée)")
    parser.add_argument("--daily-vol-bps", type=float, default=DEFAULT_DAILY_VOL_BPS, help="brique 2 : volatilité journalière de l'instrument, en bps")
    parser.add_argument("--impact-coef", type=float, default=DEFAULT_IMPACT_COEF, help="brique 2 : coefficient de la loi de la racine carrée")
    parser.add_argument("--no-regimes", action="store_true", help="brique 3 : désactive les régimes de Markov (retour au bruit journalier indépendant)")
    parser.add_argument("--p-stay-trend", type=float, default=DEFAULT_P_STAY_TREND, help="brique 3 : probabilité de rester en régime tendance le lendemain")
    parser.add_argument("--p-stay-toxic", type=float, default=DEFAULT_P_STAY_TOXIC, help="brique 3 : probabilité de rester en régime chop/toxique le lendemain")
    parser.add_argument("--toxic-winrate-penalty", type=float, default=DEFAULT_TOXIC_PENALTY, help="brique 3 : pénalité du taux de réussite en régime toxique (0,04 = 4 points)")
    parser.add_argument("--toxic-vol-mult", type=float, default=DEFAULT_TOXIC_VOL_MULT, help="brique 3 : multiplicateur de volatilité en régime toxique")
    parser.add_argument("--regimes-lower-mean", action="store_true", help="brique 3 : la pénalité baisse aussi la moyenne de long terme (défaut : décalages centrés)")
    parser.add_argument("--tail-prob", type=float, default=DEFAULT_TAIL_PROB, help="brique 4 : part des trades perdants victimes d'un stop à queue épaisse ; 0 = désactivée")
    parser.add_argument("--tail-mult-low", type=float, default=DEFAULT_TAIL_MULT_LOW, help="brique 4 : multiplicateur minimal de la distance du stop")
    parser.add_argument("--tail-mult-high", type=float, default=DEFAULT_TAIL_MULT_HIGH, help="brique 4 : multiplicateur maximal (inf pour une Pareto non tronquée)")
    parser.add_argument("--tail-alpha", type=float, default=DEFAULT_TAIL_ALPHA, help="brique 4 : indice de la loi de Pareto (petit = queue épaisse)")
    parser.add_argument("--funding-drag-bps", type=float, default=DEFAULT_FUNDING_DRAG_BPS, help="brique 5 : coût en bps du notionnel par période de 8 h (positif = coût, négatif = reçu) ; 0 = désactivée")
    parser.add_argument("--funding-interval-hours", type=float, default=None, help="brique 5 : intervalle de funding (défaut : 8 h Binance/Bybit, 1 h Hyperliquid)")
    parser.add_argument("--avg-hold-minutes", type=float, default=DEFAULT_AVG_HOLD_MINUTES, help="brique 5 : durée moyenne de détention d'un trade, en minutes")
    parser.add_argument("--no-compound", action="store_true", help="tailles fixées sur le capital initial (pas de capitalisation)")
    parser.add_argument("--no-sensitivity", action="store_true", help="n'affiche pas le tableau de sensibilité")
    args = parser.parse_args(argv)
    maker, taker = VENUE_FEES_BPS[args.venue]
    try:
        cfg = SimConfig(
            capital=args.capital, days=args.days, paths=args.paths, seed=args.seed, trades_min=args.trades_min, trades_max=args.trades_max,
            win_rate=args.win_rate, win_rate_std=args.win_rate_std, rr=args.rr, stop_bps=args.stop_bps, risk_pct=args.risk_pct,
            maker_bps=maker, taker_bps=taker, compound=not args.no_compound,
            adverse=AdverseSelectionParams(bps=args.adverse_selection_bps),
            liquidity=LiquidityParams(max_notional_usd=args.max_notional, adv_usd=args.adv_usd, daily_vol_bps=args.daily_vol_bps, impact_coef=args.impact_coef),
            regime=RegimeParams(
                enabled=not args.no_regimes,
                transition=((args.p_stay_trend, 1 - args.p_stay_trend), (1 - args.p_stay_toxic, args.p_stay_toxic)),
                win_rate_shift=(DEFAULT_TREND_BONUS, -args.toxic_winrate_penalty), volatility_multiplier=(1.0, args.toxic_vol_mult), preserve_mean=not args.regimes_lower_mean,
            ),
            stop_tail=StopTailParams(
                enabled=args.tail_prob > 0, tail_prob=args.tail_prob, multiplier_low=args.tail_mult_low, multiplier_high=args.tail_mult_high, alpha=args.tail_alpha,
            ),
            funding=FundingParams(
                drag_bps=args.funding_drag_bps, interval_hours=args.funding_interval_hours if args.funding_interval_hours is not None else VENUE_FUNDING_HOURS[args.venue], avg_hold_minutes=args.avg_hold_minutes,
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))
    return cfg, args.venue, not args.no_sensitivity


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # console Windows en cp1252 : les accents du rapport doivent s'afficher
        reconfigure(encoding="utf-8")
    cfg, venue, sensitivity = _parse(argv)
    result = simulate(cfg)
    print(render(cfg, trade_economics(cfg), summarize(cfg, result), venue, result))
    if sensitivity:
        print()
        print(sensitivity_table(cfg, (0.50, 0.52, 0.54, 0.56, 0.58, 0.60), (20.0, 40.0, 60.0, 100.0)))
    print("\nRappel : ces chiffres découlent des hypothèses ci-dessus, pas d'une mesure de Nothierly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

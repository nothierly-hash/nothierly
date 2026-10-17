"""Mesures « prix + flux » calculées à partir des TRADES seuls : de quoi entraîner sur un historique long, sans carnet d'ordres, macro ni sentiment.

Pourquoi un second jeu de mesures. L'état du PPO de production a cinq mesures de base (spread, déséquilibre du carnet, rendement US 10 ans, flux des
échanges BTC, sentiment) qu'AUCUN historique de trades ne permet de reconstituer. Ce module construit cinq autres mesures, toutes calculables par un
agent en direct à partir du flux de trades qu'il voit déjà, et qui occupent les mêmes cinq colonnes que ``MarketEnv`` attend : l'environnement n'a donc pas
changé. Ce n'est PAS le même contrat que la politique de production : un modèle entraîné ici porte ``feature_set = "price_flow_v1"`` dans ses métadonnées,
et ne doit jamais être servi avec les mesures de production (ni l'inverse).

Mesures (par barre de ``bar_s`` secondes, causales : la barre ``t`` n'utilise que les barres ``<= t``) :

1. ``ret_1``  : log-rendement de la barre.
2. ``ret_5``  : log-rendement sur 5 barres.
3. ``ret_15`` : log-rendement sur 15 barres.
4. ``vol_30`` : écart-type des ``ret_1`` sur 30 barres.
5. ``flow``   : déséquilibre du flux de la barre, ``(achat - vente) / (achat + vente)`` en notionnel, dans [-1, 1].

Les 30 premières barres n'ont pas toutes leurs mesures (``NaN``) : à écarter avant l'entraînement (voir :func:`valid_rows`).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

FEATURE_SET_NAME: Final[str] = "price_flow_v1"
FEATURE_NAMES: Final[tuple[str, ...]] = ("ret_1", "ret_5", "ret_15", "vol_30", "flow")
WARMUP_BARS: Final[int] = 30
DEFAULT_BAR_S: Final[float] = 60.0


def build_time_bars(
    times_ms: npt.NDArray[np.int64],
    prices: npt.NDArray[np.float64],
    quantities: npt.NDArray[np.float64],
    is_buy: npt.NDArray[np.bool_],
    *,
    bar_s: float = DEFAULT_BAR_S,
) -> dict[str, npt.NDArray[np.float64]]:
    """Agrège des trades (triés par temps) en barres de ``bar_s`` secondes ; ne garde QUE les barres où il y a eu au moins un trade.

    Renvoie ``bar_end_s`` (fin de la barre, en secondes : la mesure n'est connue qu'à ce moment), ``close`` (prix du dernier trade), ``buy_notional`` et
    ``sell_notional`` (prix x quantité, côté du PRENEUR). Une barre sans trade n'existe pas : un trou de collecte se voit donc dans ``bar_end_s``
    (voir :func:`contiguous_runs`), il n'est jamais comblé.
    """
    n = len(times_ms)
    if not (len(prices) == len(quantities) == len(is_buy) == n):
        raise ValueError("times_ms, prices, quantities et is_buy doivent avoir la même longueur.")
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return {"bar_end_s": empty, "close": empty, "buy_notional": empty, "sell_notional": empty}
    if np.any(np.diff(times_ms) < 0):
        raise ValueError("les trades doivent être triés par temps croissant.")
    if not (np.isfinite(prices).all() and np.isfinite(quantities).all()) or np.any(prices <= 0) or np.any(quantities <= 0):
        raise ValueError("prix et quantités doivent être finis et strictement positifs.")

    bar_ms = int(round(bar_s * 1000))
    bar_index = times_ms // bar_ms
    boundaries = np.flatnonzero(np.diff(bar_index)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    notional = prices * quantities
    buy_notional = np.add.reduceat(np.where(is_buy, notional, 0.0), starts)
    sell_notional = np.add.reduceat(np.where(is_buy, 0.0, notional), starts)
    return {
        "bar_end_s": ((bar_index[starts] + 1) * bar_ms / 1000.0).astype(np.float64),
        "close": prices[ends - 1].astype(np.float64),
        "buy_notional": buy_notional.astype(np.float64),
        "sell_notional": sell_notional.astype(np.float64),
    }


def price_flow_features(
    close: npt.NDArray[np.float64], buy_notional: npt.NDArray[np.float64], sell_notional: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Matrice ``(T, 5)`` des mesures ``price_flow_v1`` ; ``NaN`` tant que 30 barres ne sont pas connues. Causale : aucune ligne n'utilise le futur."""
    n = len(close)
    if not (len(buy_notional) == len(sell_notional) == n):
        raise ValueError("close, buy_notional et sell_notional doivent avoir la même longueur.")
    if n and (not np.isfinite(close).all() or np.any(close <= 0)):
        raise ValueError("close doit être fini et strictement positif.")
    features = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    if n == 0:
        return features
    log_close = np.log(close)
    ret_1 = np.full(n, np.nan)
    ret_1[1:] = np.diff(log_close)
    for column, lag in ((1, 5), (2, 15)):
        values = np.full(n, np.nan)
        if n > lag:
            values[lag:] = log_close[lag:] - log_close[:-lag]
        features[:, column] = values
    features[:, 0] = ret_1
    vol = np.full(n, np.nan)
    for t in range(WARMUP_BARS, n):
        vol[t] = float(np.std(ret_1[t - WARMUP_BARS + 1 : t + 1], ddof=1))  # ret_1[t-29..t] : 30 rendements, tous connus à t
    features[:, 3] = vol
    total = buy_notional + sell_notional
    with np.errstate(divide="ignore", invalid="ignore"):
        features[:, 4] = np.where(total > 0, (buy_notional - sell_notional) / total, np.nan)
    return features


def valid_rows(features: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Lignes dont les 5 mesures sont connues."""
    mask: npt.NDArray[np.bool_] = np.isfinite(features).all(axis=1)
    return mask


def contiguous_runs(bar_end_s: npt.NDArray[np.float64], *, bar_s: float = DEFAULT_BAR_S, max_gap_bars: float = 1.5) -> list[slice]:
    """Découpe en plages SANS trou : une plage se rompt quand deux barres consécutives sont séparées de plus de ``max_gap_bars`` barres.

    Une mesure calculée à travers un trou (rendement sur 15 barres dont 10 manquent) serait fausse : on n'entraîne jamais à cheval sur un trou.
    """
    if len(bar_end_s) == 0:
        return []
    breaks = np.flatnonzero(np.diff(bar_end_s) > max_gap_bars * bar_s) + 1
    edges = np.concatenate(([0], breaks, [len(bar_end_s)]))
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def featurize_runs(
    bars: dict[str, npt.NDArray[np.float64]], *, bar_s: float = DEFAULT_BAR_S, min_rows: int = 1
) -> list[dict[str, npt.NDArray[np.float64]]]:
    """Calcule les mesures SÉPARÉMENT dans chaque plage sans trou, et ne garde que les lignes où les cinq sont connues.

    Chaque élément contient ``bar_end_s``, ``close`` et ``features`` (``(n, 5)``, sans ``NaN``), tous alignés et causaux. Les plages de moins de
    ``min_rows`` lignes utilisables sont écartées.
    """
    out: list[dict[str, npt.NDArray[np.float64]]] = []
    for run in contiguous_runs(bars["bar_end_s"], bar_s=bar_s):
        close = bars["close"][run]
        features = price_flow_features(close, bars["buy_notional"][run], bars["sell_notional"][run])
        ok = valid_rows(features)
        if int(ok.sum()) >= max(1, min_rows):
            out.append({"bar_end_s": bars["bar_end_s"][run][ok], "close": close[ok], "features": features[ok]})
    return out

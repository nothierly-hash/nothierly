"""Réconciliation d'état au démarrage : le moteur ADOPTE les positions réelles de l'exchange au lieu de repartir « à plat ».

**Le problème.** ``ExecutionAgent._positions`` vit en mémoire. Après un redémarrage (VPS, mise à jour du conteneur), le moteur se croit à plat ; le pont
publie alors ``FLAT`` pour chaque instrument suivi et les workers ferment au marché la position réelle : spread et frais taker payés pour une simple
opération de maintenance, et une décision de sortie que ni le PPO ni le XGBoost n'ont prise.

**La politique : adopter, puis laisser le modèle réévaluer.** Avant de passer ``ACTIVE``, le moteur lit les positions réelles que
``workers/reconciliation/reporter.py`` publie dans Redis (``exchange:positions:<PLATEFORME>``), les convertit dans SES unités et les inscrit dans
``_positions``. Ensuite la logique normale s'applique, sans code spécial : prochain signal de consensus, stop suiveur, arrêt de drawdown. Pendant tout ce
temps l'état publié est ``RECONCILING`` : le pont ne publie RIEN (ni cible, ni ``FLAT``) ; il n'agit donc jamais sur un état que le moteur n'a pas encore
vérifié.

**Conversion.** Le moteur raisonne en unités et en équité virtuelle (``_mock_balance``) ; chaque compte d'exchange a sa propre équité. L'exposition est donc
reportée en pourcentage : ``exposition = taille x prix / équité du compte``, puis ``unités = exposition x équité du moteur / prix``. Le pont refait le calcul
inverse (unités x prix / équité du moteur) et retrouve la même exposition : les workers ne voient aucun écart à corriger, donc n'envoient aucun ordre.

**Ce que l'adoption ne restitue PAS.** Le prix d'entrée réel, l'ancre du stop suiveur et la date d'ouverture ne sont pas rapportés par les clients : la position
adoptée est datée de maintenant, son prix d'entrée est le prix de marché du rapport (PnL latent, stop et suiveur repartent de là). Et le moteur ne sait que
ouvrir ou fermer (pas de redimensionnement) : « conserver, réduire ou fermer » se réduit à conserver ou fermer ; une taille différente de la cible est
corrigée par le pont (plafonds, seaux), pas par le moteur.

**Plusieurs plateformes.** Le pont publie la même cible à toutes : leurs expositions devraient coïncider. Le moteur adopte celle de la première plateforme
fraîche de la liste de priorité et journalise tout écart notable avec les autres (jamais une moyenne qui ne correspondrait à aucun compte réel).

**Rapports invalides ou périmés.** Un rapport illisible, périmé (``--max-age``), d'équité nulle ou de prix non positif est ignoré en entier, jamais adopté à
moitié. Sans aucun rapport frais le moteur NE passe PAS ``ACTIVE`` : il réessaie, et le pont, qui retient tout pendant ``RECONCILING``, finit par traiter ce
silence comme une panne (``engine_startup_grace_s``) ; c'est le repli prudent connu, pas un pari sur « plat ».

Aucune dépendance autre que la stdlib : testable sans Redis ni agent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Final, Literal

REPORT_KEY_PREFIX: Final[str] = "exchange:positions:"
ENV_VENUES: Final[str] = "NOTHIERLY_RECONCILE_VENUES"
ENV_WAIT_S: Final[str] = "NOTHIERLY_RECONCILE_WAIT_S"
DEFAULT_WAIT_S: Final[float] = 120.0
DEFAULT_POLL_S: Final[float] = 2.0
DEFAULT_MAX_AGE_S: Final[float] = 120.0
FUTURE_SKEW_S: Final[float] = 60.0  # un rapport daté d'au plus 60 s dans le futur (horloges légèrement décalées) est accepté
MIN_EXPOSURE_PCT: Final[Decimal] = Decimal("0.01")  # sous 0,01 % de l'équité : poussière, pas une position à adopter
DIVERGENCE_PCT: Final[Decimal] = Decimal(2)  # écart d'exposition entre plateformes journalisé au-delà de 2 points de pourcentage
_QUOTES: Final[tuple[str, ...]] = ("USDT", "USDC", "USD", "PERP")


def report_key(venue: str) -> str:
    return f"{REPORT_KEY_PREFIX}{venue.upper()}"


@dataclass(frozen=True)
class ReportedPosition:
    asset: str  # nom natif de la plateforme : « BTC » (Hyperliquid), « BTCUSDT » (Bybit, Binance)
    size: Decimal  # signée, en unités de base (positif = long)
    price: Decimal  # prix de marque ou milieu au moment du rapport


@dataclass(frozen=True)
class ExchangeReport:
    venue: str
    ts_ms: int
    equity: Decimal
    positions: tuple[ReportedPosition, ...]


def _decimal(value: Any) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def parse_report(raw: str | bytes | None) -> ExchangeReport | None:
    """Rapport JSON -> :class:`ExchangeReport`, ou ``None`` au moindre doute (jamais une adoption partielle)."""
    if raw is None or not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    venue, ts, equity_raw, items = data.get("venue"), data.get("ts_ms"), data.get("equity"), data.get("positions")
    equity = _decimal(equity_raw)
    if not isinstance(venue, str) or not venue or isinstance(ts, bool) or not isinstance(ts, int) or ts <= 0 or equity is None or equity <= 0 or not isinstance(items, list):
        return None
    positions: list[ReportedPosition] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        asset, size, price = item.get("asset"), _decimal(item.get("size")), _decimal(item.get("price"))
        if not isinstance(asset, str) or not asset or size is None or price is None or price <= 0:
            return None
        if size != 0:
            positions.append(ReportedPosition(asset, size, price))
    return ExchangeReport(venue.upper(), ts, equity, tuple(positions))


def is_fresh(report: ExchangeReport, now_ms: int, max_age_s: float) -> bool:
    age_s = (now_ms - report.ts_ms) / 1000.0
    return -FUTURE_SKEW_S <= age_s <= max_age_s


def reference_report(reports: Mapping[str, ExchangeReport], priority: Sequence[str], now_ms: int, max_age_s: float) -> ExchangeReport | None:
    """Le rapport frais de la première plateforme de ``priority`` qui en a un."""
    for venue in priority:
        report = reports.get(venue.upper())
        if report is not None and is_fresh(report, now_ms, max_age_s):
            return report
    return None


def _normalize(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper() if ch.isalnum())


def instrument_for(asset: str, instruments: Iterable[str]) -> str | None:
    """L'instrument du moteur (``BTC/USDT``) que désigne l'actif natif ``asset`` (``BTC`` ou ``BTCUSDT``), sinon ``None``."""
    want = _normalize(asset)
    for instrument in instruments:
        have = _normalize(instrument)
        if have == want or any(have == want + quote or want == have + quote for quote in _QUOTES):
            return instrument
    return None


@dataclass(frozen=True)
class Adoption:
    instrument: str
    direction: Literal[1, -1]
    units: float  # positif ; le sens est dans ``direction``
    price: float
    exposure_pct: float  # signée, en % de l'équité du compte d'exchange


@dataclass(frozen=True)
class AdoptionPlan:
    adoptions: tuple[Adoption, ...]
    ignored: tuple[str, ...]  # actifs non suivis par le moteur, ou poussière : laissés tels quels
    divergences: tuple[str, ...]  # écarts notables avec les autres plateformes (information, jamais bloquant)


def _exposure_pct(position: ReportedPosition, equity: Decimal) -> Decimal:
    return position.size * position.price / equity * 100


def plan_adoption(
    report: ExchangeReport,
    instruments: Iterable[str],
    engine_equity: float,
    *,
    others: Iterable[ExchangeReport] = (),
    min_exposure_pct: Decimal = MIN_EXPOSURE_PCT,
) -> AdoptionPlan:
    """Calcule ce que le moteur doit adopter (aucune écriture). Lève ``ValueError`` si l'équité du moteur n'est pas strictement positive."""
    if not engine_equity > 0:
        raise ValueError("l'équité du moteur doit être strictement positive pour convertir une exposition en unités")
    instrument_list = list(instruments)
    adoptions: list[Adoption] = []
    ignored: list[str] = []
    exposures: dict[str, Decimal] = {}
    for position in report.positions:
        instrument = instrument_for(position.asset, instrument_list)
        if instrument is None:
            ignored.append(f"{position.asset}: instrument non suivi par le moteur")
            continue
        exposure = _exposure_pct(position, report.equity)
        if abs(exposure) < min_exposure_pct:
            ignored.append(f"{position.asset}: poussière ({exposure:.4f} % de l'équité)")
            continue
        exposures[instrument] = exposure
        units = float(abs(exposure) / 100) * engine_equity / float(position.price)
        adoptions.append(Adoption(instrument, 1 if exposure > 0 else -1, units, float(position.price), float(exposure)))
    divergences: list[str] = []
    for other in others:
        if other.venue == report.venue:
            continue
        for position in other.positions:
            instrument = instrument_for(position.asset, instrument_list)
            if instrument is None:
                continue
            delta = _exposure_pct(position, other.equity) - exposures.get(instrument, Decimal(0))
            if abs(delta) > DIVERGENCE_PCT:
                divergences.append(f"{instrument}: {other.venue} {_exposure_pct(position, other.equity):+.2f} % contre {report.venue} {exposures.get(instrument, Decimal(0)):+.2f} %")
        seen = {instrument_for(p.asset, instrument_list) for p in other.positions}
        for instrument, exposure in exposures.items():
            if instrument not in seen and abs(exposure) > DIVERGENCE_PCT:
                divergences.append(f"{instrument}: {report.venue} {exposure:+.2f} % contre {other.venue} à plat")
    return AdoptionPlan(tuple(adoptions), tuple(ignored), tuple(divergences))


@dataclass(frozen=True)
class ReconcilerConfig:
    venues: tuple[str, ...]  # par ordre de priorité
    wait_s: float = DEFAULT_WAIT_S
    poll_s: float = DEFAULT_POLL_S
    max_age_s: float = DEFAULT_MAX_AGE_S

    def __post_init__(self) -> None:
        if not self.venues or any(not v.strip() for v in self.venues):
            raise ValueError("au moins une plateforme est requise")
        if self.wait_s < 0 or self.poll_s <= 0 or self.max_age_s <= 0:
            raise ValueError("wait_s >= 0, poll_s > 0 et max_age_s > 0 sont requis")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ReconcilerConfig | None:
        """``None`` (réconciliation coupée, comportement historique) tant que ``NOTHIERLY_RECONCILE_VENUES`` n'est pas défini : on n'attend jamais un
        rapport qui n'existera pas dans un déploiement sans workers."""
        env = os.environ if environ is None else environ
        raw = env.get(ENV_VENUES, "").strip()
        if not raw:
            return None
        venues = tuple(v.strip().upper() for v in raw.split(",") if v.strip())
        wait = env.get(ENV_WAIT_S)
        return cls(venues, wait_s=float(wait)) if wait else cls(venues)


Execute = Callable[..., Awaitable[Any]]


async def read_reports(execute: Execute, venues: Iterable[str]) -> dict[str, ExchangeReport]:
    """Lit et analyse les rapports présents dans Redis ; une plateforme absente ou illisible n'apparaît simplement pas dans le résultat."""
    reports: dict[str, ExchangeReport] = {}
    for venue in venues:
        report = parse_report(await execute("get", report_key(venue)))
        if report is not None:
            reports[venue.upper()] = report
    return reports

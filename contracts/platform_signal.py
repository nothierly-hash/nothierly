"""Contrat de signal multi-plateformes : le seul message qui sort du moteur vers les workers.

Le moteur ne parle jamais aux API des plateformes. Il valide un ``PlatformSignal``, le sérialise
en JSON (``to_zmq_payload``) et le publie sur le bus ZeroMQ ; chaque worker (Numerai, Hyperliquid,
Darwinex Zero) le relit avec ``from_zmq_payload``, qui rejoue exactement la même validation.
Un signal incohérent lève donc côté moteur, avant publication, et ne peut pas non plus être
accepté côté worker.

Ce n'est PAS ``core.schemas.TradingSignal`` : celui-là est une opinion d'agent interne (``-1/0/1`` +
confiance) consommée par l'``ExecutionAgent`` pour son consensus, et n'autorise rien. Celui-ci est
une décision déjà arbitrée, destinée à une plateforme externe : d'où le nom distinct.

``confidence_score`` n'est transmis qu'à Numerai (qui en a besoin pour son score). Pour Hyperliquid
(vault public on-chain) et Darwinex, il est interdit : la magnitude de la conviction permettrait de
rétro-ingénierer l'alpha depuis le flux d'ordres.

Frontière d'IP : ce module ne doit rien importer de ``core`` ni de ``src``. Les workers n'importent
que ``contracts``, jamais le code du moteur.

Les règles croisées (venue/classe d'actif, ``limit_price``, ``stop_loss_price``...) ne sont pas
exprimables en JSON Schema dérivé de Pydantic : ``platform_signal.schema.json`` décrit la
structure et les bornes, la validation Python reste la source de vérité.

Régénérer le schéma : ``python -m contracts.platform_signal``.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetVenue(str, Enum):
    """Les 8 plateformes cibles, par pôle, plus ``ALL``. Pas de 9e : ajouter une plateforme est un choix explicite."""

    # Pôle crypto (temps réel)
    HYPERLIQUID = "HYPERLIQUID"
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    ZIGNALY = "ZIGNALY"
    ENZYME = "ENZYME"
    # Pôle actions (batch)
    NUMERAI_SIGNALS = "NUMERAI_SIGNALS"
    # Pôle forex / CFD (pont MT4/MT5)
    AXI_SELECT = "AXI_SELECT"
    DARWINEX_ZERO = "DARWINEX_ZERO"
    ALL = "ALL"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    PREDICTION_ONLY = "PREDICTION_ONLY"


# Classe d'actif que traite chaque plateforme (une seule). ``target_venue=ALL`` signifie « toutes les
# plateformes de la classe d'actif du signal » : 5 pour CRYPTO, 2 pour FOREX, 1 pour EQUITY.
_ASSET_CLASS_BY_VENUE: Final[dict[TargetVenue, AssetClass]] = {
    TargetVenue.HYPERLIQUID: AssetClass.CRYPTO,
    TargetVenue.BINANCE: AssetClass.CRYPTO,
    TargetVenue.BYBIT: AssetClass.CRYPTO,
    TargetVenue.ZIGNALY: AssetClass.CRYPTO,
    TargetVenue.ENZYME: AssetClass.CRYPTO,
    TargetVenue.NUMERAI_SIGNALS: AssetClass.EQUITY,
    TargetVenue.AXI_SELECT: AssetClass.FOREX,
    TargetVenue.DARWINEX_ZERO: AssetClass.FOREX,
}


def venues_for(asset_class: AssetClass) -> frozenset[TargetVenue]:
    """Plateformes qui traitent ``asset_class`` (ce que couvre ``target_venue=ALL``)."""
    return frozenset(v for v, cls in _ASSET_CLASS_BY_VENUE.items() if cls is asset_class)

# EQUITY_US_AAPL, CRYPTO_BTC_USD, FOREX_EUR_USD : préfixe de classe, puis au moins un segment.
_SYMBOL_PATTERN: Final[str] = r"^[A-Z0-9]+(?:_[A-Z0-9]+)+$"
_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(_SYMBOL_PATTERN)


def _positive_finite(metadata: dict[str, Any], key: str) -> float:
    """Lit ``metadata[key]`` comme prix strictement positif et fini, sinon ``ValueError``.

    ``bool`` est exclu explicitement (``True`` est un ``int`` en Python et passerait pour 1.0), de
    même que les chaînes, ``NaN`` et ``inf`` : un prix limite ou un stop doit être un vrai nombre.
    """
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata['{key}'] doit être un nombre, reçu {type(value).__name__}.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"metadata['{key}'] doit être un prix fini et strictement positif, reçu {value!r}.")
    return float(value)


class PlatformSignal(BaseModel):
    """Décision arbitrée à destination d'une plateforme externe. Immuable, champs inconnus refusés.

    Attributes:
        signal_id: UUID unique du signal (sert de base à l'idempotence côté worker).
        timestamp_ms: Instant d'émission, en millisecondes epoch (horloge murale).
        target_venue: Plateforme visée, ou ``ALL`` (= toutes les plateformes de la classe d'actif).
        asset_class: ``EQUITY``, ``CRYPTO`` ou ``FOREX``.
        symbol: Ticker interne générique, préfixé par la classe d'actif (ex. ``CRYPTO_BTC_USD``).
            Le mapping vers le ticker de la plateforme est fait par chaque worker.
        direction: ``LONG``, ``SHORT`` ou ``FLAT`` (cible : pas d'exposition).
        confidence_score: Confiance de l'alpha, dans ``[0, 1]``. **Obligatoire** pour
            ``NUMERAI_SIGNALS``, **interdit** (doit rester ``None``) pour toutes les autres
            plateformes (crypto et forex) : leur flux d'ordres est visible.
        position_size_pct: Exposition **cible**, en % du capital, dans ``[0, 100]`` (pas un ordre à
            ajouter : un worker compare la cible à sa position réelle et n'envoie que l'écart, donc
            republier le même signal ne change rien). Doit valoir 0 pour ``FLAT`` et dépasser 0 pour
            un ordre ``LONG``/``SHORT`` exécutable.
        order_type: ``MARKET``, ``LIMIT`` ou ``PREDICTION_ONLY`` (aucun ordre : score seul).
        metadata: Dictionnaire libre. Clés exigées selon le cas : ``limit_price`` (``LIMIT``),
            ``stop_loss_price`` (``DARWINEX_ZERO``, ordre exécutable).
        ttl_ms: Durée de validité optionnelle. ``None`` = pas d'expiration (cas Numerai, qui
            n'est pas temps réel). Un worker temps réel rejette un signal expiré (``is_expired``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000), gt=0)
    target_venue: TargetVenue
    asset_class: AssetClass
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    direction: SignalDirection
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    position_size_pct: float = Field(ge=0.0, le=100.0)
    order_type: OrderType
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> PlatformSignal:
        self._check_symbol_matches_asset_class()
        venues = self._effective_venues()
        self._check_confidence_disclosure(venues)
        self._check_sizing()
        self._check_limit_price()
        if TargetVenue.NUMERAI_SIGNALS in venues and self.order_type is not OrderType.PREDICTION_ONLY:
            raise ValueError("Numerai n'accepte que des signaux de type PREDICTION_ONLY.")
        if TargetVenue.DARWINEX_ZERO in venues and self.order_type is not OrderType.PREDICTION_ONLY:
            if "stop_loss_price" not in self.metadata:
                raise ValueError("Les workers Darwinex exigent 'stop_loss_price' dans metadata (gestion VaR).")
        self._check_stop_loss_side()
        return self

    def _check_symbol_matches_asset_class(self) -> None:
        if not self.symbol.startswith(f"{self.asset_class.value}_"):
            raise ValueError(
                f"symbol {self.symbol!r} doit commencer par '{self.asset_class.value}_' (asset_class={self.asset_class.value})."
            )

    def _effective_venues(self) -> frozenset[TargetVenue]:
        """Plateformes réellement concernées ; refuse une plateforme qui ne traite pas la classe."""
        if self.target_venue is TargetVenue.ALL:
            return venues_for(self.asset_class)
        expected = _ASSET_CLASS_BY_VENUE[self.target_venue]
        if expected is not self.asset_class:
            raise ValueError(
                f"{self.target_venue.value} ne traite pas la classe {self.asset_class.value} "
                f"(classe de cette plateforme : {expected.value})."
            )
        return frozenset({self.target_venue})

    def _check_confidence_disclosure(self, venues: frozenset[TargetVenue]) -> None:
        """``confidence_score`` : exigé pour Numerai, interdit pour toutes les plateformes à flux visible."""
        if TargetVenue.NUMERAI_SIGNALS in venues:
            if self.confidence_score is None:
                raise ValueError("Numerai exige confidence_score (il devient le score soumis).")
        elif self.confidence_score is not None:
            raise ValueError(
                f"confidence_score est interdit pour {', '.join(sorted(v.value for v in venues))} : "
                "la magnitude de la conviction fuiterait l'alpha via le flux d'ordres visible."
            )

    def _check_sizing(self) -> None:
        if self.direction is SignalDirection.FLAT and self.position_size_pct != 0.0:
            raise ValueError("Un signal FLAT doit avoir position_size_pct == 0.")
        executable = self.order_type is not OrderType.PREDICTION_ONLY
        if executable and self.direction is not SignalDirection.FLAT and self.position_size_pct == 0.0:
            raise ValueError("Un ordre LONG/SHORT exécutable doit avoir position_size_pct > 0.")

    def _check_limit_price(self) -> None:
        if self.order_type is OrderType.LIMIT:
            if "limit_price" not in self.metadata:
                raise ValueError("Un ordre LIMIT exige la clé 'limit_price' dans metadata.")
            _positive_finite(self.metadata, "limit_price")

    def _check_stop_loss_side(self) -> None:
        """Un stop doit être du bon côté du prix limite quand les deux sont connus."""
        if "stop_loss_price" not in self.metadata:
            return
        stop = _positive_finite(self.metadata, "stop_loss_price")
        if "limit_price" not in self.metadata or self.direction is SignalDirection.FLAT:
            return
        limit = _positive_finite(self.metadata, "limit_price")
        if self.direction is SignalDirection.LONG and stop >= limit:
            raise ValueError("LONG : stop_loss_price doit être inférieur à limit_price.")
        if self.direction is SignalDirection.SHORT and stop <= limit:
            raise ValueError("SHORT : stop_loss_price doit être supérieur à limit_price.")

    def targets(self, venue: TargetVenue) -> bool:
        """``True`` si ``venue`` doit traiter ce signal (``ALL`` couvre toutes les plateformes de la classe)."""
        return venue in self._effective_venues()

    def is_expired(self, now_ms: int | None = None) -> bool:
        """``True`` si ``ttl_ms`` est défini et dépassé. Sans ``ttl_ms``, jamais expiré."""
        if self.ttl_ms is None:
            return False
        current = int(time.time() * 1000) if now_ms is None else now_ms
        return current - self.timestamp_ms > self.ttl_ms

    def to_zmq_payload(self) -> str:
        """JSON du contrat validé, prêt à être publié sur ZeroMQ."""
        return self.model_dump_json()

    @classmethod
    def from_zmq_payload(cls, payload: str | bytes) -> PlatformSignal:
        """Relit et revalide un payload reçu du bus. Lève ``ValidationError`` si invalide."""
        return cls.model_validate_json(payload)


SCHEMA_PATH: Final[Path] = Path(__file__).with_name("platform_signal.schema.json")


def json_schema() -> dict[str, Any]:
    """JSON Schema (structure et bornes) du contrat, tel que publié pour les workers."""
    schema = PlatformSignal.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://nothierly.local/contracts/platform_signal.schema.json",
        **schema,
    }


if __name__ == "__main__":
    SCHEMA_PATH.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"schéma écrit : {SCHEMA_PATH}")

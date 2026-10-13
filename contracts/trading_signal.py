"""Contrat de signal multi-plateformes : le seul message qui sort du moteur vers les workers.

Le moteur ne parle jamais aux API des plateformes. Il valide un ``TradingSignal``, le sérialise
en JSON (``to_zmq_payload``) et le publie sur le bus ZeroMQ ; chaque worker (Numerai, Hyperliquid,
Darwinex Zero) le relit avec ``from_zmq_payload``, qui rejoue exactement la même validation.
Un signal incohérent lève donc côté moteur, avant publication, et ne peut pas non plus être
accepté côté worker.

Ce n'est PAS ``core.schemas.TradingSignal`` : celui-là est une opinion d'agent (``-1/0/1`` +
confiance) consommée par l'``ExecutionAgent`` pour son consensus, et n'autorise rien. Celui-ci est
une décision déjà arbitrée, destinée à une plateforme externe. Les deux coexistent ; importer
toujours par chemin complet (``from contracts.trading_signal import TradingSignal``).

Frontière d'IP : ce module ne doit rien importer de ``core`` ni de ``src``. Les workers n'importent
que ``contracts``, jamais le code du moteur.

Les règles croisées (venue/classe d'actif, ``limit_price``, ``stop_loss_price``...) ne sont pas
exprimables en JSON Schema dérivé de Pydantic : ``trading_signal.schema.json`` décrit la
structure et les bornes, la validation Python reste la source de vérité.

Régénérer le schéma : ``python -m contracts.trading_signal``.
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
    NUMERAI = "NUMERAI"
    HYPERLIQUID = "HYPERLIQUID"
    DARWINEX = "DARWINEX"
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


# Plateforme qui traite chaque classe d'actif dans cette architecture. Une seule par classe :
# c'est ce qui donne un sens à ``target_venue=ALL`` (le signal va à la plateforme de sa classe).
# À mettre à jour si une classe d'actif gagne une deuxième plateforme.
_VENUE_BY_ASSET_CLASS: Final[dict[AssetClass, TargetVenue]] = {
    AssetClass.EQUITY: TargetVenue.NUMERAI,
    AssetClass.CRYPTO: TargetVenue.HYPERLIQUID,
    AssetClass.FOREX: TargetVenue.DARWINEX,
}

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


class TradingSignal(BaseModel):
    """Décision arbitrée à destination d'une plateforme externe. Immuable, champs inconnus refusés.

    Attributes:
        signal_id: UUID unique du signal (sert de base à l'idempotence côté worker).
        timestamp_ms: Instant d'émission, en millisecondes epoch (horloge murale).
        target_venue: Plateforme visée, ou ``ALL`` (= la plateforme de la classe d'actif).
        asset_class: ``EQUITY``, ``CRYPTO`` ou ``FOREX``.
        symbol: Ticker interne générique, préfixé par la classe d'actif (ex. ``CRYPTO_BTC_USD``).
            Le mapping vers le ticker de la plateforme est fait par chaque worker.
        direction: ``LONG``, ``SHORT`` ou ``FLAT`` (cible : pas d'exposition).
        confidence_score: Confiance de l'alpha, dans ``[0, 1]``.
        position_size_pct: Pourcentage du capital alloué, dans ``[0, 100]``. Doit valoir 0 pour
            ``FLAT`` et dépasser 0 pour un ordre ``LONG``/``SHORT`` exécutable.
        order_type: ``MARKET``, ``LIMIT`` ou ``PREDICTION_ONLY`` (aucun ordre : score seul).
        metadata: Dictionnaire libre. Clés exigées selon le cas : ``limit_price`` (``LIMIT``),
            ``stop_loss_price`` (Darwinex, ordre exécutable).
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
    confidence_score: float = Field(ge=0.0, le=1.0)
    position_size_pct: float = Field(ge=0.0, le=100.0)
    order_type: OrderType
    metadata: dict[str, Any] = Field(default_factory=dict)
    ttl_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> TradingSignal:
        self._check_symbol_matches_asset_class()
        venues = self._effective_venues()
        self._check_sizing()
        self._check_limit_price()
        if TargetVenue.NUMERAI in venues and self.order_type is not OrderType.PREDICTION_ONLY:
            raise ValueError("Numerai n'accepte que des signaux de type PREDICTION_ONLY.")
        if TargetVenue.DARWINEX in venues and self.order_type is not OrderType.PREDICTION_ONLY:
            if "stop_loss_price" not in self.metadata:
                raise ValueError("Les workers Darwinex exigent 'stop_loss_price' dans metadata (gestion VaR).")
        self._check_stop_loss_side()
        return self

    def _check_symbol_matches_asset_class(self) -> None:
        if not self.symbol.startswith(f"{self.asset_class.value}_"):
            raise ValueError(
                f"symbol {self.symbol!r} doit commencer par '{self.asset_class.value}_' (asset_class={self.asset_class.value})."
            )

    def _effective_venues(self) -> set[TargetVenue]:
        """Plateformes réellement concernées ; refuse une plateforme qui ne traite pas la classe."""
        owner = _VENUE_BY_ASSET_CLASS[self.asset_class]
        if self.target_venue is not TargetVenue.ALL and self.target_venue is not owner:
            raise ValueError(
                f"{self.target_venue.value} ne traite pas la classe {self.asset_class.value} (plateforme attendue : {owner.value})."
            )
        return {owner}

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
    def from_zmq_payload(cls, payload: str | bytes) -> TradingSignal:
        """Relit et revalide un payload reçu du bus. Lève ``ValidationError`` si invalide."""
        return cls.model_validate_json(payload)


SCHEMA_PATH: Final[Path] = Path(__file__).with_name("trading_signal.schema.json")


def json_schema() -> dict[str, Any]:
    """JSON Schema (structure et bornes) du contrat, tel que publié pour les workers."""
    schema = TradingSignal.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://nothierly.local/contracts/trading_signal.schema.json",
        **schema,
    }


if __name__ == "__main__":
    SCHEMA_PATH.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"schéma écrit : {SCHEMA_PATH}")

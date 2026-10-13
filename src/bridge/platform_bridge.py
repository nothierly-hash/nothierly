"""Pont Cerveau -> plateformes externes : l'état réel des positions du moteur devient des ``PlatformSignal``.

**D'où vient le signal.** Les agents de signal (``XgboostSignalAgent``, ``RlSignalAgent``) sortent chacun une
opinion ``(signal ∈ {-1, 0, 1}, confiance ∈ [0, 1])`` (``core.schemas.TradingSignal``). L'``ExecutionAgent`` les
arbitre (consensus, 8 portes) et tient les positions qui en résultent. Le pont ne refait AUCUN arbitrage et ne lit
pas les opinions brutes : il lit l'**état arbitré** que l'``ExecutionAgent`` publie dans Redis
(``NOTHIERLY_ACTIVE_POSITIONS``, ``NOTHIERLY_EQUITY_CURVE``). Un état plutôt qu'un flux d'événements, parce que :

- les workers raisonnent en **exposition cible** : republier l'état courant est idempotent et se répare seul ;
- un événement de sortie manqué (par exemple une fermeture simulée en MOCK/SHADOW, qui ne publie pas toujours
  ``TRADE_EXECUTED``) laisserait une position externe ouverte ; un état, jamais.

**Traduction.** Position ``LONG``/``SHORT`` -> direction ; absence de position -> ``FLAT``. Taille :
``|unités| x prix / equity du moteur`` en %, donc le dimensionnement du moteur (risque, Kelly, plafonds) est
réutilisé tel quel ; plafonné par ``max_position_pct`` ; les workers l'appliquent ensuite à LEUR equity.
Le ``confidence_score`` n'est pas transmis : il est interdit pour toute plateforme crypto (le flux d'ordres y est
visible, la magnitude de la conviction fuiterait l'alpha). Voir ``contracts.platform_signal``.

**Sécurité de repli** (sens sûr = réduire) :

- kill switch global actif (``PANIC``/``LATCHED_DATA``, Redis injoignable, signature invalide) -> ``FLAT`` tout de suite ;
- moteur silencieux (heartbeat de l'``ExecutionAgent`` absent) ou état illisible plus de ``engine_grace_s`` -> ``FLAT`` ;
  en deçà, on ne publie rien (pas de bascule sur une coupure d'une seconde) ;
- equity inconnue ou nulle -> on ne publie rien pour une position ouverte (on ne devine pas une taille).

**Limites connues, à ne pas ignorer :**

- Seules les paires crypto ``BASE/USDT`` sont traduites. Le moteur n'a **aucun modèle d'actions** : rien ne peut
  alimenter ``NUMERAI_SIGNALS`` (qui exige 100+ scores par jour). Le forex (``EUR_USD``) exigerait un ``stop_loss_price``
  que le moteur ne fournit pas : ignoré, avec un avertissement.
- Si le moteur tourne en ``MOCK_MODE``/``SHADOW``, ses positions sont **simulées** : un worker en mode live
  tradera de l'argent réel d'après des décisions simulées. Garder les workers en dry-run/testnet/démo dans ce cas.
- Un worker ramène le compte à la cible pour CHAQUE symbole suivi, y compris ``FLAT`` : une position ouverte à la
  main sur ce symbole serait fermée. Utiliser des comptes dédiés.
- Si le pont meurt, ce sont les workers qui protègent : leur homme mort (``--dead-man-timeout``, 300 s par défaut)
  ferme toutes leurs positions après N secondes sans signal valide, puis les verrouille (``workers/common/base.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final, Protocol

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue

_log = logging.getLogger(__name__)

ACTIVE_POSITIONS_KEY: Final[str] = "NOTHIERLY_ACTIVE_POSITIONS"
EQUITY_CURVE_KEY: Final[str] = "NOTHIERLY_EQUITY_CURVE"
ENGINE_HEARTBEAT_KEY: Final[str] = "heartbeat:execution-agent"
_QUOTES: Final[frozenset[str]] = frozenset({"USDT", "USD"})
_MIN_PCT: Final[Decimal] = Decimal("0.01")


@dataclass(frozen=True)
class EnginePosition:
    instrument: str  # ex. "BTC/USDT"
    status: str  # "LONG" | "SHORT"
    units: Decimal  # valeur absolue
    price: Decimal  # prix courant (sinon prix d'entrée)


@dataclass(frozen=True)
class EngineView:
    positions: tuple[EnginePosition, ...]
    equity: Decimal | None  # equity du moteur en USD, ou None si inconnue
    engine_alive: bool  # heartbeat de l'ExecutionAgent présent
    system_safe: bool  # kill switch global : False = halte


class EngineStateReader(Protocol):
    async def read(self) -> EngineView:
        """Lit l'état du moteur. Lève si l'état est illisible (Redis injoignable, JSON invalide...)."""
        ...


class SignalSink(Protocol):
    def publish(self, signal: PlatformSignal) -> bool:
        """``True`` si le message est en file d'envoi, ``False`` s'il est abandonné."""
        ...


@dataclass(frozen=True)
class BridgeConfig:
    instruments: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    venue: TargetVenue = TargetVenue.ALL  # ALL = toutes les plateformes crypto
    max_position_pct: Decimal = Decimal("20")
    change_threshold_pct: Decimal = Decimal("0.25")  # variation de cible qui déclenche une republication
    refresh_s: float = 10.0  # republication périodique : les workers se recalent, un worker tardif rattrape
    engine_grace_s: float = 30.0
    ttl_ms: int = 10_000
    poll_s: float = 2.0
    flatten_on_halt: bool = True
    flatten_when_engine_down: bool = True
    symbol_overrides: Mapping[str, str] = field(default_factory=dict)


def to_internal_symbol(instrument: str, overrides: Mapping[str, str] | None = None) -> str | None:
    """``BTC/USDT`` -> ``CRYPTO_BTC_USD``. ``None`` pour tout ce qui n'est pas une paire crypto ``BASE/USDT|USD``."""
    if overrides and instrument in overrides:
        return overrides[instrument]
    base, sep, quote = instrument.partition("/")
    if not sep or quote not in _QUOTES or not base.isalnum() or not base.isupper():
        return None
    return f"CRYPTO_{base}_USD"


def desired_target(position: EnginePosition | None, equity: Decimal | None, cfg: BridgeConfig) -> tuple[SignalDirection, Decimal] | None:
    """``(direction, % d'equity)`` visé pour un instrument, ou ``None`` si la taille est indéterminable."""
    if position is None:
        return SignalDirection.FLAT, Decimal(0)
    if equity is None or equity <= 0:
        return None
    pct = min(position.units * position.price / equity * 100, cfg.max_position_pct)
    if pct < _MIN_PCT:
        return SignalDirection.FLAT, Decimal(0)  # trop petit pour être un signal LONG/SHORT valide
    direction = SignalDirection.LONG if position.status == "LONG" else SignalDirection.SHORT
    return direction, pct.quantize(Decimal("0.01"))


@dataclass
class _Published:
    direction: SignalDirection
    pct: Decimal
    at: float


class PlatformBridge:
    def __init__(
        self, reader: EngineStateReader, sink: SignalSink, config: BridgeConfig | None = None,
        *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reader = reader
        self._sink = sink
        self._cfg = config or BridgeConfig()
        self._clock = clock
        self._last: dict[str, _Published] = {}
        self._unhealthy_since: float | None = None
        self._warned: set[str] = set()
        self._symbols = {i: to_internal_symbol(i, self._cfg.symbol_overrides) for i in self._cfg.instruments}
        for instrument, symbol in self._symbols.items():
            if symbol is None:
                _log.warning("bridge_instrument_ignored instrument=%s : seules les paires crypto BASE/USDT sont traduites", instrument)

    def _build(self, symbol: str, direction: SignalDirection, pct: Decimal) -> PlatformSignal:
        return PlatformSignal(
            target_venue=self._cfg.venue,
            asset_class=AssetClass.CRYPTO,
            symbol=symbol,
            direction=direction,
            position_size_pct=float(pct),
            order_type=OrderType.MARKET,
            ttl_ms=self._cfg.ttl_ms,
        )

    def _should_publish(self, instrument: str, direction: SignalDirection, pct: Decimal, now: float) -> bool:
        last = self._last.get(instrument)
        if last is None or last.direction is not direction:
            return True
        return abs(pct - last.pct) >= self._cfg.change_threshold_pct or now - last.at >= self._cfg.refresh_s

    def _publish(self, instrument: str, direction: SignalDirection, pct: Decimal, reason: str, now: float) -> str | None:
        symbol = self._symbols[instrument]
        if symbol is None:
            return None
        signal = self._build(symbol, direction, pct)
        if not self._sink.publish(signal):
            _log.warning("bridge_signal_dropped instrument=%s", instrument)
            return None
        self._last[instrument] = _Published(direction, pct, now)
        _log.info("bridge_signal instrument=%s %s %s%% venue=%s reason=%s", instrument, direction.value, pct, self._cfg.venue.value, reason)
        return f"{instrument}:{direction.value}:{pct}:{reason}"

    async def step(self) -> list[str]:
        """Un cycle : lit le moteur, décide, publie ce qui doit l'être. Renvoie les publications faites (« instrument:dir:pct:raison »)."""
        now = self._clock()
        view: EngineView | None
        try:
            view = await self._reader.read()
        except Exception as exc:  # noqa: BLE001 -- état illisible : on ne devine pas, on laisse jouer le délai de grâce
            _log.warning("bridge_engine_state_unreadable %s: %s", type(exc).__name__, exc)
            view = None

        healthy = view is not None and view.engine_alive
        if healthy:
            self._unhealthy_since = None
        elif self._unhealthy_since is None:
            self._unhealthy_since = now
        down_for = 0.0 if self._unhealthy_since is None else now - self._unhealthy_since

        flatten_reason: str | None = None
        if view is not None and not view.system_safe and self._cfg.flatten_on_halt:
            flatten_reason = "kill_switch"
        elif not healthy and down_for >= self._cfg.engine_grace_s and self._cfg.flatten_when_engine_down:
            flatten_reason = "engine_down"
        elif not healthy:
            return []  # dans le délai de grâce : on garde la dernière cible, on ne publie rien

        published: list[str] = []
        by_instrument = {p.instrument: p for p in (view.positions if view is not None else ())}
        for instrument in self._cfg.instruments:
            if self._symbols[instrument] is None:
                continue
            if flatten_reason is not None:
                direction, pct, reason = SignalDirection.FLAT, Decimal(0), flatten_reason
            else:
                assert view is not None
                target = desired_target(by_instrument.get(instrument), view.equity, self._cfg)
                if target is None:
                    if instrument not in self._warned:
                        self._warned.add(instrument)
                        _log.warning("bridge_no_equity instrument=%s : position ouverte mais equity inconnue, rien n'est publié", instrument)
                    continue
                self._warned.discard(instrument)
                (direction, pct), reason = target, "engine_state"
            if self._should_publish(instrument, direction, pct, now):
                entry = self._publish(instrument, direction, pct, reason, now)
                if entry is not None:
                    published.append(entry)
        return published

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop if stop is not None else asyncio.Event()
        while not stop.is_set():
            try:
                await self.step()
            except Exception:  # noqa: BLE001 -- un cycle raté ne doit jamais arrêter le pont
                _log.exception("bridge_step_failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._cfg.poll_s)
            except asyncio.TimeoutError:
                pass


class RedisEngineReader:
    """Lit l'état du moteur dans Redis (``client.execute(commande, ...)`` : ``core.redis_client.RedisClient``)."""

    def __init__(self, client: Any, system_safe: Callable[[], Any]) -> None:
        self._client = client
        self._system_safe = system_safe  # coroutine ``core.security.is_system_safe``

    async def read(self) -> EngineView:
        raw_positions, raw_equity, heartbeat = await asyncio.gather(
            self._client.execute("get", ACTIVE_POSITIONS_KEY),
            self._client.execute("lindex", EQUITY_CURVE_KEY, 0),
            self._client.execute("exists", ENGINE_HEARTBEAT_KEY),
        )
        positions = tuple(_parse_positions(raw_positions))
        equity = _parse_equity(raw_equity)
        return EngineView(positions, equity, bool(heartbeat), bool(await self._system_safe()))


def _parse_positions(raw: Any) -> Sequence[EnginePosition]:
    if raw is None:
        return []
    out = []
    for item in json.loads(raw):
        status = str(item.get("status", ""))
        if status not in ("LONG", "SHORT"):
            continue
        price = Decimal(str(item.get("current_price") or item.get("entry_price") or 0))
        out.append(EnginePosition(str(item["instrument"]), status, abs(Decimal(str(item["units"]))), price))
    return out


def _parse_equity(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    balance = json.loads(raw).get("balance")
    return None if balance is None else Decimal(str(balance))

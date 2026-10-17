"""Limites de débit REST : HTTP 429 (trop de requêtes) et 418 (adresse IP bannie, Binance). Pause exacte, jamais de martèlement.

**Le risque.** Un exchange qui répond 429 demande de ralentir ; continuer à envoyer des requêtes escalade vers un bannissement d'IP (Binance : 418,
de quelques minutes à plusieurs jours). Or un worker relit le compte à chaque signal et ré-essaie à chaque republication du pont (10 s) : sans garde, une
limite atteinte devient un bannissement.

**Ce que fait ce module.**

- ``HttpRateLimited`` : levée par le transport HTTP (``urllib``) sur un 429/418, avec la valeur brute de l'en-tête ``Retry-After``.
- ``RateLimitGate`` : un verrou de pause PARTAGÉ par tous les appels d'un client (thread-safe : les appels REST tournent dans ``asyncio.to_thread``).
  ``block`` fixe l'échéance (``Retry-After`` si présent : secondes ou date HTTP ; sinon un défaut qui double à chaque 429 consécutif, plafonné à
  15 min), ``check`` refuse LOCALEMENT toute requête avant l'échéance (rien n'est envoyé), ``wait`` dort exactement jusqu'à l'échéance sans bloquer la boucle
  asyncio (``asyncio.wait_for`` sur l'événement d'arrêt : un arrêt du worker interrompt la pause).
- ``guard_transport`` : enveloppe un transport ``(méthode, url, en-têtes, corps, délai) -> (statut, octets)`` : vérifie la porte avant d'envoyer, convertit
  un 429/418 en ``RateLimitError`` et arme la porte.
- ``RateLimitError`` hérite d'``OSError`` : les gestionnaires existants (« erreur réseau, on ne prend aucune décision à l'aveugle ») la traitent déjà
  comme une panne de lecture ; les workers l'attrapent en amont pour un code d'état précis (``rate_limited``).

**429 = ordre non pris.** Une requête refusée pour débit n'a pas été traitée : renvoyer l'ordre après la pause ne peut pas le doubler (contrairement à un
délai d'attente réseau, ambigu). Le worker garde donc le signal en attente et l'exécute (le plus récent pour l'actif) à la fin de la pause.

**Non couvert.** Bybit signale aussi un dépassement par le code ``retCode`` du corps JSON ou un HTTP 403 (limite d'IP) ; seuls le statut HTTP 429 (et 418)
sont traités ici, conformément à la demande. Le comportement exact de ces autres signaux n'a pas été vérifié contre la documentation actuelle.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Final

RATE_LIMITED_STATUSES: Final[frozenset[int]] = frozenset({429, 418})
DEFAULT_429_PAUSE_S: Final[float] = 60.0
DEFAULT_418_PAUSE_S: Final[float] = 300.0
MAX_ESCALATED_DEFAULT_S: Final[float] = 900.0  # le défaut doublé ne dépasse jamais 15 min (un Retry-After explicite, lui, est respecté tel quel)
MAX_PAUSE_S: Final[float] = 3 * 86_400.0  # Binance : un bannissement peut durer jusqu'à 3 jours
MIN_PAUSE_S: Final[float] = 1.0  # « Retry-After: 0 » ne veut pas dire « tout de suite et sans fin »

Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


class RateLimitError(OSError):
    """Limite de débit atteinte (ou pause encore en cours : ``local=True``, aucune requête n'est partie). ``retry_after_s`` : pause restante."""

    def __init__(self, status: int, retry_after_s: float, *, local: bool = False) -> None:
        super().__init__(f"limite de débit (HTTP {status}) : pause de {retry_after_s:.0f}s" + (" (requête non envoyée)" if local else ""))
        self.status = status
        self.retry_after_s = retry_after_s
        self.local = local


class HttpRateLimited(Exception):
    """Levée par un transport HTTP sur un 429/418 ; ``retry_after`` est la valeur BRUTE de l'en-tête (ou ``None``)."""

    def __init__(self, status: int, retry_after: str | None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """``Retry-After`` : un nombre de secondes ou une date HTTP. ``None`` si absent ou illisible (jamais une pause inventée) ; une date déjà passée donne la pause minimale."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - (now or datetime.now(timezone.utc))).total_seconds()
    if not math.isfinite(seconds):
        return None
    return MIN_PAUSE_S if seconds < 0 else seconds  # une date déjà passée : pause minimale, pas de martèlement


def rate_limit_status(exc: BaseException) -> tuple[int, str | None] | None:
    """``(statut, Retry-After brut)`` si ``exc`` est une erreur 429/418 d'un SDK (``status_code`` + ``header``), sinon ``None``. Duck-typing : les SDK
    d'exchange sont optionnels et sans stubs."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or status not in RATE_LIMITED_STATUSES:
        return None
    headers = getattr(exc, "header", None) or getattr(exc, "headers", None)
    value: Any = None
    if headers is not None and hasattr(headers, "get"):
        value = headers.get("Retry-After") or headers.get("retry-after")
    return status, None if value is None else str(value)


class RateLimitGate:
    """Échéance de pause partagée. Thread-safe ; l'horloge est injectable (``time.monotonic`` par défaut)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, max_pause_s: float = MAX_PAUSE_S) -> None:
        self._clock = clock
        self._max = max_pause_s
        self._lock = threading.Lock()
        self._until = float("-inf")
        self._consecutive_defaults = 0
        self.last_status: int | None = None
        self.blocks = 0

    def block(self, status: int, retry_after: str | None = None) -> float:
        """Arme la pause ; renvoie sa durée. ``Retry-After`` valide : respecté tel quel (borné à 3 jours). Absent : défaut qui double à chaque fois."""
        parsed = parse_retry_after(retry_after)
        with self._lock:
            if parsed is not None:
                pause = parsed
                self._consecutive_defaults = 0
            else:
                base = DEFAULT_418_PAUSE_S if status == 418 else DEFAULT_429_PAUSE_S
                pause = min(base * (2 ** self._consecutive_defaults), MAX_ESCALATED_DEFAULT_S)
                self._consecutive_defaults += 1
            pause = min(max(pause, MIN_PAUSE_S), self._max)
            self._until = max(self._until, self._clock() + pause)
            self.last_status = status
            self.blocks += 1
            return pause

    def note_success(self) -> None:
        with self._lock:
            self._consecutive_defaults = 0

    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._until - self._clock())

    def check(self) -> None:
        """Lève ``RateLimitError(local=True)`` tant que la pause court : la requête n'est PAS envoyée."""
        left = self.remaining()
        if left > 0:
            raise RateLimitError(self.last_status or 429, left, local=True)

    async def wait(self, stop: asyncio.Event | None = None) -> float:
        """Dort (sans bloquer la boucle) exactement jusqu'à l'échéance, y compris si elle est repoussée pendant l'attente. ``stop`` interrompt la pause.
        Renvoie le temps attendu."""
        waited = 0.0
        while (left := self.remaining()) > 0:
            if stop is not None and stop.is_set():
                return waited
            start = time.monotonic()
            try:
                if stop is None:
                    await asyncio.sleep(left)
                else:
                    await asyncio.wait_for(stop.wait(), timeout=left)
            except asyncio.TimeoutError:
                pass
            waited += time.monotonic() - start
        return waited


def guard_transport(transport: Transport, gate: RateLimitGate) -> Transport:
    """Enveloppe ``transport`` : porte vérifiée avant l'envoi, 429/418 -> pause armée + ``RateLimitError``."""

    def guarded(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        gate.check()
        try:
            status, raw = transport(method, url, headers, body, timeout)
        except HttpRateLimited as exc:
            raise RateLimitError(exc.status, gate.block(exc.status, exc.retry_after)) from None
        if status in RATE_LIMITED_STATUSES:  # un transport qui renvoie (429, corps) au lieu de lever : pas d'en-tête, défaut escaladé
            raise RateLimitError(status, gate.block(status, None))
        gate.note_success()
        return status, raw

    return guarded

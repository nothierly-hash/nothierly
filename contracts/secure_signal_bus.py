"""Transport signé des ``PlatformSignal`` sur ZeroMQ : publisher du moteur, vérification des workers.

Menace couverte : un acteur sur le réseau (workers sur un VPS Windows, serveurs cloud différents)
qui injecte un faux signal sur le bus, ou rejoue un vrai signal capturé.

Format sur le fil (même découpage que ``core.zmq_bus`` : un message multipart) ::

    frame 0 : topic            (ex. b"EXECUTION.SIGNAL", sert au filtrage SUB)
    frame 1 : enveloppe JSON   {"signature": "<hex>", "payload": "<JSON du PlatformSignal>"}

``signature`` = HMAC-SHA256, clé ``NOTHIERLY_ZMQ_SECRET``, calculé sur les octets UTF-8 exacts de la
chaîne ``payload`` telle qu'envoyée. Le worker vérifie la signature AVANT de parser le payload et
ne re-sérialise jamais : aucune ambiguïté de canonicalisation JSON.

Ce que le HMAC ne fait pas à lui seul :

- **Rejeu.** Un message authentique capturé reste authentique. ``open_envelope`` ajoute donc un âge
  maximal, une tolérance sur l'horloge future, l'expiration ``ttl_ms`` du signal et une mémoire
  d'identifiants déjà vus (``ReplayGuard``).
- **Confidentialité.** Le payload n'est pas chiffré. Un lecteur du réseau voit les signaux ; il ne
  peut pas en forger. Chiffrer le canal (ZMQ CURVE) est une étape distincte, non couverte ici.
- **Aiguillage.** La signature ne couvre pas le topic (l'enveloppe est celle de la spécification).
  Le worker contrôle donc ``signal.routed_venue`` : un signal Hyperliquid rejoué vers le worker
  Darwinex est refusé même avec une signature valide.

Frontière d'IP : ce module n'importe que la stdlib, ``zmq`` et ``contracts.platform_signal``.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

import zmq
from pydantic import ValidationError

from contracts.platform_signal import PlatformSignal, TargetVenue

_log = logging.getLogger(__name__)

SECRET_ENV_VAR: Final[str] = "NOTHIERLY_ZMQ_SECRET"
DEFAULT_TOPIC: Final[bytes] = b"EXECUTION.SIGNAL"
# 32 octets = la taille de sortie de SHA-256 ; en dessous, la clé est plus faible que le MAC.
MIN_SECRET_LENGTH: Final[int] = 32
DEFAULT_MAX_AGE_MS: Final[int] = 60_000
DEFAULT_MAX_FUTURE_SKEW_MS: Final[int] = 5_000
DEFAULT_REPLAY_CAPACITY: Final[int] = 10_000


class WeakSecretError(ValueError):
    """Clé absente ou trop courte : erreur de configuration, volontairement bruyante."""


def _secret_bytes(secret_key: str | bytes) -> bytes:
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    if len(key) < MIN_SECRET_LENGTH:
        raise WeakSecretError(f"La clé HMAC doit faire au moins {MIN_SECRET_LENGTH} octets (reçu {len(key)}).")
    return key


def load_secret_from_env(environ: Mapping[str, str] | None = None) -> bytes:
    """Lit ``NOTHIERLY_ZMQ_SECRET`` ; lève ``WeakSecretError`` si absente ou trop courte."""
    raw = (os.environ if environ is None else environ).get(SECRET_ENV_VAR)
    if not raw:
        raise WeakSecretError(f"La variable d'environnement {SECRET_ENV_VAR} n'est pas définie.")
    return _secret_bytes(raw)


def _hmac_hex(key: bytes, message: str) -> str:
    return hmac.new(key, message.encode("utf-8"), sha256).hexdigest()


def sign_payload(payload: str, secret_key: str | bytes) -> str:
    """HMAC-SHA256 (hex minuscule) de la chaîne ``payload``."""
    return _hmac_hex(_secret_bytes(secret_key), payload)


def build_envelope(signal: PlatformSignal, secret_key: str | bytes) -> dict[str, str]:
    """``{"signature", "payload"}`` pour un signal ; le payload est ``model_dump_json()``."""
    payload = signal.to_zmq_payload()
    return {"signature": sign_payload(payload, secret_key), "payload": payload}


def _parse_envelope(envelope: Mapping[str, Any] | str | bytes) -> tuple[str, str] | None:
    """``(signature, payload)`` si l'enveloppe est bien formée, sinon ``None`` (jamais d'exception)."""
    data: Any = envelope
    if isinstance(envelope, (str, bytes, bytearray)):
        try:
            data = json.loads(envelope)
        except (ValueError, UnicodeDecodeError):
            return None
    if not isinstance(data, Mapping):
        return None
    signature, payload = data.get("signature"), data.get("payload")
    if not isinstance(signature, str) or not isinstance(payload, str):
        return None
    return signature, payload


def verify_signature(envelope: Mapping[str, Any] | str | bytes, secret_key: str | bytes) -> bool:
    """``True`` seulement si la signature de l'enveloppe correspond à son payload.

    Accepte le dictionnaire ou son JSON (``str``/``bytes``). Une enveloppe mal formée, altérée ou
    signée avec une autre clé donne ``False``, sans exception ni réponse à l'émetteur. Seule une clé
    de configuration invalide lève (``WeakSecretError``) : c'est une panne locale, pas un message.
    La comparaison se fait en temps constant.
    """
    key = _secret_bytes(secret_key)
    parsed = _parse_envelope(envelope)
    if parsed is None:
        return False
    signature, payload = parsed
    return hmac.compare_digest(_hmac_hex(key, payload).encode("ascii"), signature.encode("utf-8"))


class ReplayGuard:
    """Mémoire bornée des ``signal_id`` déjà acceptés. Un seul thread/boucle d'événements par instance."""

    def __init__(self, capacity: int = DEFAULT_REPLAY_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity doit être strictement positif.")
        self._capacity = capacity
        self._seen: OrderedDict[uuid.UUID, None] = OrderedDict()

    def check_and_add(self, signal_id: uuid.UUID) -> bool:
        """``True`` si l'identifiant est nouveau (et le mémorise), ``False`` si déjà vu."""
        if signal_id in self._seen:
            return False
        self._seen[signal_id] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)  # oublie le plus ancien ; l'âge maximal borne le reste
        return True


@dataclass(frozen=True)
class EnvelopeVerdict:
    """Résultat de ``open_envelope`` : le signal accepté, ou la raison (code court) du rejet."""

    signal: PlatformSignal | None
    reject_reason: str | None

    @property
    def accepted(self) -> bool:
        return self.signal is not None


def _reject(reason: str, **context: object) -> EnvelopeVerdict:
    # Rien du message n'est journalisé (ni payload ni signature) : seulement le code de rejet.
    _log.warning("signal_rejected reason=%s %s", reason, " ".join(f"{k}={v}" for k, v in context.items()))
    return EnvelopeVerdict(signal=None, reject_reason=reason)


def open_envelope(
    envelope: Mapping[str, Any] | str | bytes,
    secret_key: str | bytes,
    *,
    venue: TargetVenue,
    max_age_ms: int | None = DEFAULT_MAX_AGE_MS,
    max_future_skew_ms: int = DEFAULT_MAX_FUTURE_SKEW_MS,
    replay_guard: ReplayGuard | None = None,
    now_ms: int | None = None,
) -> EnvelopeVerdict:
    """Vérification complète côté worker. Rejette sans lever ni répondre ; renvoie un verdict.

    Ordre : forme de l'enveloppe, signature, validité du ``PlatformSignal``, plateforme visée,
    expiration ``ttl_ms``, âge maximal, horloge future, rejeu. Les codes de rejet :
    ``malformed_envelope``, ``bad_signature``, ``invalid_payload``, ``wrong_venue``, ``expired``,
    ``too_old``, ``from_future``, ``replay``.

    Args:
        venue: la plateforme de CE worker (jamais ``ALL``).
        max_age_ms: âge maximal accepté. ``None`` le désactive (worker Numerai, non temps réel).
        max_future_skew_ms: avance d'horloge tolérée entre moteur et worker.
        replay_guard: mémoire des identifiants déjà vus ; sans elle, pas de détection de rejeu.
    """
    if venue is TargetVenue.ALL:
        raise ValueError("venue doit être la plateforme du worker, pas ALL.")
    parsed = _parse_envelope(envelope)
    if parsed is None:
        return _reject("malformed_envelope")
    if not verify_signature(envelope, secret_key):
        return _reject("bad_signature")
    try:
        signal = PlatformSignal.from_zmq_payload(parsed[1])
    except ValidationError:
        # Signature valide mais contenu incohérent : le moteur a signé un message qu'il n'aurait pas dû.
        return _reject("invalid_payload")
    if signal.routed_venue is not venue:
        return _reject("wrong_venue", signal_id=signal.signal_id, expected=venue.value, got=signal.routed_venue.value)
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if signal.is_expired(now):
        return _reject("expired", signal_id=signal.signal_id)
    if max_age_ms is not None and now - signal.timestamp_ms > max_age_ms:
        return _reject("too_old", signal_id=signal.signal_id)
    if signal.timestamp_ms - now > max_future_skew_ms:
        return _reject("from_future", signal_id=signal.signal_id)
    if replay_guard is not None and not replay_guard.check_and_add(signal.signal_id):
        return _reject("replay", signal_id=signal.signal_id)
    return EnvelopeVerdict(signal=signal, reject_reason=None)


class SecureSignalPublisher:
    """Socket ``PUB`` qui signe chaque ``PlatformSignal`` avant de le publier.

    Même conventions que ``core.zmq_bus.ZmqPublisher`` (message multipart ``[topic, corps]``,
    ``SNDHWM`` borné, ``LINGER`` 0), mais envoi **non bloquant** : le moteur n'attend jamais un
    worker ou le réseau. Si la file est pleine, le signal est abandonné (``publish`` renvoie
    ``False``) plutôt que mis en attente : un signal périmé vaut moins qu'un signal perdu.
    """

    def __init__(
        self,
        address: str,
        secret_key: str | bytes,
        *,
        topic: bytes = DEFAULT_TOPIC,
        bind: bool = True,
        high_water_mark: int = 1000,
        context: zmq.Context[Any] | None = None,
    ) -> None:
        """Crée le socket PUB. ``bind=True`` : écoute sur ``address`` (``tcp://127.0.0.1:*`` accepté) ;
        ``bind=False`` : se connecte à ``address`` (frontend d'un broker XSUB).

        Raises:
            WeakSecretError: clé absente ou trop courte.
            ValueError: topic vide.
        """
        if not topic:
            raise ValueError("topic ne doit pas être vide.")
        self._key = _secret_bytes(secret_key)
        self._topic = topic
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, high_water_mark)
        self._socket.setsockopt(zmq.LINGER, 0)
        if bind:
            self._socket.bind(address)
        else:
            self._socket.connect(address)
        self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._closed = False

    @classmethod
    def from_env(cls, address: str, **kwargs: Any) -> SecureSignalPublisher:
        """Construit le publisher avec la clé de ``NOTHIERLY_ZMQ_SECRET``."""
        return cls(address, load_secret_from_env(), **kwargs)

    @property
    def endpoint(self) -> str:
        """Adresse réelle du socket (utile avec ``tcp://127.0.0.1:*``)."""
        return self._endpoint

    def publish(self, signal: PlatformSignal) -> bool:
        """Revalide, signe et publie. ``True`` si le message est en file d'envoi, ``False`` si la file est pleine.

        Un ``PlatformSignal`` construit en contournant la validation (``model_construct``,
        ``model_copy(update=...)``) est revalidé ici : il lève ``ValidationError`` avant tout envoi.

        Raises:
            RuntimeError: publisher fermé.
            TypeError: l'argument n'est pas un ``PlatformSignal``.
            ValidationError: signal incohérent.
            ValueError: signal déjà expiré.
        """
        if self._closed:
            raise RuntimeError("SecureSignalPublisher est fermé.")
        if not isinstance(signal, PlatformSignal):
            raise TypeError(f"PlatformSignal attendu, reçu {type(signal).__name__}.")
        payload = signal.to_zmq_payload()
        PlatformSignal.from_zmq_payload(payload)  # lève avant l'envoi si le signal est incohérent
        if signal.is_expired():
            raise ValueError(f"Signal {signal.signal_id} déjà expiré : non publié.")
        body = json.dumps(
            {"signature": _hmac_hex(self._key, payload), "payload": payload}, separators=(",", ":")
        ).encode("utf-8")
        try:
            self._socket.send_multipart([self._topic, body], flags=zmq.DONTWAIT)
        except zmq.Again:
            _log.warning("signal_dropped_queue_full signal_id=%s", signal.signal_id)
            return False
        return True

    def close(self) -> None:
        """Ferme le socket (sans attendre les messages en file) et le contexte s'il lui appartient."""
        if self._closed:
            return
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()
        self._closed = True

    def __enter__(self) -> SecureSignalPublisher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

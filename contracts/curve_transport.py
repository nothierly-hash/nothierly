"""Chiffrement et authentification du bus ZeroMQ (CURVE, Curve25519) : le canal devient opaque et authentifié.

Le HMAC de ``secure_signal_bus`` prouve qu'un message vient du moteur ; il ne le cache pas. Sur un réseau partagé
(internet, plusieurs VPS), un observateur lirait le payload, donc l'alpha. CURVE chiffre et authentifie le canal
lui-même, en complément du HMAC (qui reste : il protège aussi contre un message rejoué ou fabriqué par un pair déjà
autorisé).

**Qui authentifie qui.**

- Le **publisher** (le pont) est le serveur CURVE : il possède une clé secrète et n'accepte que les clés publiques
  des workers déclarés (liste blanche, vérifiée par ZAP). Sans liste blanche, le canal serait chiffré mais ouvert :
  la clé publique du serveur n'est pas secrète, n'importe qui pourrait s'abonner et lire. Une configuration serveur
  sans client autorisé est donc refusée.
- Chaque **worker** est un client CURVE : il possède sa paire de clés et **épingle la clé publique du serveur**. Un faux
  publisher qui n'a pas la clé secrète du serveur ne peut pas établir le canal : le worker ne lui parle jamais.
- Un client sans CURVE (mécanisme NULL) est refusé par un serveur CURVE : pas de repli en clair.

**Gestion des clés.** Chaînes Z85 de 40 caractères, lues dans l'environnement (voir ``from_env``). ``scripts/generate_curve_keys.py``
les génère dans des fichiers ``.env`` (ignorés par git) sans jamais afficher une clé secrète. Un serveur par contexte
ZeroMQ : le gestionnaire ZAP est lié au contexte.

**Ce que CURVE ne fait pas.** Il ne cache pas la taille ni le rythme des messages, ne protège pas contre un hôte
compromis (les clés sont en mémoire sur chaque machine), et n'apporte pas de secret persistant « avant » : la
rotation des clés se fait en les régénérant et en redéployant.

**Interrupteur ``NOTHIERLY_ZMQ_ENCRYPTION``** (préfixe ``NOTHIERLY_`` et non ``ZMQ_`` : ce dernier est celui de ``core.config.ZmqSettings``) :

- absent ou ``auto`` : CURVE si les clés sont dans l'environnement, sinon rien (comportement historique) ;
- ``true`` : CURVE **obligatoire**, des clés manquantes sont une erreur de démarrage (à mettre en conteneur : pas de
  démarrage en clair par oubli d'un fichier ``.env``) ;
- ``false`` : les clés éventuelles sont ignorées, canal en clair (tests locaux). Ne contourne pas ``transport_refusal`` :
  un endpoint non local reste refusé sans ``--allow-plaintext-remote``.

Ce module n'importe que la stdlib, ``zmq`` et ne dépend ni de ``core`` ni de ``src``.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final

import zmq
from zmq.utils import z85

_log = logging.getLogger(__name__)

ENV_SERVER_SECRET: Final[str] = "NOTHIERLY_CURVE_SERVER_SECRET"
ENV_CLIENT_KEYS: Final[str] = "NOTHIERLY_CURVE_CLIENT_KEYS"
ENV_SERVER_PUBLIC: Final[str] = "NOTHIERLY_CURVE_SERVER_PUBLIC"
ENV_CLIENT_PUBLIC: Final[str] = "NOTHIERLY_CURVE_CLIENT_PUBLIC"
ENV_CLIENT_SECRET: Final[str] = "NOTHIERLY_CURVE_CLIENT_SECRET"
ENV_ENCRYPTION: Final[str] = "NOTHIERLY_ZMQ_ENCRYPTION"

_LOCAL_PREFIXES: Final[tuple[str, ...]] = ("tcp://127.0.0.1:", "tcp://localhost:", "tcp://[::1]:", "inproc://", "ipc://")


class CurveConfigError(ValueError):
    """Configuration CURVE absente, partielle ou incohérente : erreur de déploiement, volontairement bruyante."""


_TRUE: Final[frozenset[str]] = frozenset({"true", "1", "yes", "on", "required"})
_FALSE: Final[frozenset[str]] = frozenset({"false", "0", "no", "off"})


def encryption_mode(environ: Mapping[str, str] | None = None) -> bool | None:
    """``True`` : CURVE obligatoire ; ``False`` : désactivé ; ``None`` : selon la présence des clés (``auto``)."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_ENCRYPTION, "").strip().lower()
    if raw in ("", "auto"):
        return None
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise CurveConfigError(f"{ENV_ENCRYPTION} doit valoir true, false ou auto (reçu : {raw!r}).")


def _check_key(name: str, value: str) -> str:
    """Valide une clé Z85 (40 caractères -> 32 octets) sans jamais l'inclure dans le message d'erreur."""
    try:
        if len(value) != 40 or len(z85.decode(value.encode("ascii"))) != 32:
            raise ValueError
    except (ValueError, KeyError, IndexError, UnicodeEncodeError, TypeError) as exc:  # KeyError : caractère hors alphabet Z85
        raise CurveConfigError(f"{name} n'est pas une clé CURVE valide (40 caractères Z85).") from exc
    return value


@dataclass(frozen=True)
class CurveKeyPair:
    public: str
    secret: str = field(repr=False)  # jamais dans un log ni un repr


def generate_keypair() -> CurveKeyPair:
    public, secret = zmq.curve_keypair()
    return CurveKeyPair(public.decode("ascii"), secret.decode("ascii"))


def public_from_secret(secret_key: str) -> str:
    return zmq.curve_public(_check_key("clé secrète", secret_key).encode("ascii")).decode("ascii")


@dataclass(frozen=True)
class CurveServerConfig:
    """Côté publisher : sa clé secrète et les clés publiques des clients autorisés à s'abonner."""

    secret_key: str = field(repr=False)
    authorized_clients: frozenset[str]

    def __post_init__(self) -> None:
        _check_key("la clé secrète du serveur", self.secret_key)
        if not self.authorized_clients:
            raise CurveConfigError("Aucun client autorisé : un serveur CURVE sans liste blanche laisse n'importe qui s'abonner.")
        for key in self.authorized_clients:
            _check_key("une clé client autorisée", key)

    @property
    def public_key(self) -> str:
        return public_from_secret(self.secret_key)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CurveServerConfig | None:
        """``None`` si aucune variable n'est définie (pas de CURVE) ou si ``NOTHIERLY_ZMQ_ENCRYPTION=false`` ; erreur si partielle."""
        env = os.environ if environ is None else environ
        mode = encryption_mode(env)
        if mode is False:
            return None
        secret, clients = env.get(ENV_SERVER_SECRET), env.get(ENV_CLIENT_KEYS)
        if not secret and not clients:
            if mode:
                raise CurveConfigError(f"{ENV_ENCRYPTION}=true mais {ENV_SERVER_SECRET} et {ENV_CLIENT_KEYS} ne sont pas définis.")
            return None
        if not secret or not clients:
            raise CurveConfigError(f"Configuration CURVE serveur partielle : {ENV_SERVER_SECRET} et {ENV_CLIENT_KEYS} vont ensemble.")
        return cls(secret, frozenset(k.strip() for k in clients.split(",") if k.strip()))


@dataclass(frozen=True)
class CurveClientConfig:
    """Côté worker : sa paire de clés et la clé publique du serveur qu'il épingle."""

    server_public_key: str
    public_key: str
    secret_key: str = field(repr=False)

    def __post_init__(self) -> None:
        _check_key("la clé publique du serveur", self.server_public_key)
        _check_key("la clé publique du client", self.public_key)
        if public_from_secret(self.secret_key) != self.public_key:
            raise CurveConfigError("La clé secrète du client ne correspond pas à sa clé publique.")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> CurveClientConfig | None:
        """``None`` si aucune variable n'est définie (pas de CURVE) ou si ``NOTHIERLY_ZMQ_ENCRYPTION=false`` ; erreur si partielle."""
        env = os.environ if environ is None else environ
        mode = encryption_mode(env)
        if mode is False:
            return None
        names = (ENV_SERVER_PUBLIC, ENV_CLIENT_PUBLIC, ENV_CLIENT_SECRET)
        values = [env.get(n) for n in names]
        if not any(values):
            if mode:
                raise CurveConfigError(f"{ENV_ENCRYPTION}=true mais {', '.join(names)} ne sont pas définis.")
            return None
        if not all(values):
            raise CurveConfigError(f"Configuration CURVE client partielle : {', '.join(names)} vont ensemble.")
        server_public, public, secret = values
        assert server_public and public and secret
        return cls(server_public, public, secret)


def apply_server(socket: zmq.Socket[Any], config: CurveServerConfig) -> None:
    """Fait de ``socket`` un serveur CURVE : le mécanisme NULL (en clair) est refusé."""
    socket.curve_server = True
    socket.curve_secretkey = config.secret_key.encode("ascii")


def apply_client(socket: zmq.Socket[Any], config: CurveClientConfig) -> None:
    """Fait de ``socket`` un client CURVE qui n'accepte que le serveur de clé publique ``server_public_key``."""
    socket.curve_serverkey = config.server_public_key.encode("ascii")
    socket.curve_publickey = config.public_key.encode("ascii")
    socket.curve_secretkey = config.secret_key.encode("ascii")


_ZAP_ENDPOINT: Final[str] = "inproc://zeromq.zap.01"
_ZAP_VERSION: Final[bytes] = b"1.0"
_ZAP_POLL_MS: Final[int] = 100


class ClientAuthenticator:
    """Gestionnaire ZAP (RFC 27) du serveur : n'autorise que les clés publiques de la liste blanche.

    À démarrer sur le contexte du socket serveur, avant son ``bind``, et à arrêter avant ``context.term()``.

    Écrit à la main plutôt que ``zmq.auth.thread.ThreadAuthenticator``, car celui-ci lance une boucle ``asyncio`` dans
    son thread : sous Windows (boucle Proactor par défaut) ce thread meurt, plus aucun client n'est jamais autorisé et
    ``stop()`` reste bloqué à jamais. Ici : un thread ordinaire, un socket ``REP``, aucune dépendance à asyncio, un
    arrêt borné. Tout mécanisme autre que CURVE est refusé (défense en profondeur : un socket serveur CURVE ne
    l'accepte déjà pas).
    """

    def __init__(self, context: zmq.Context[Any], config: CurveServerConfig) -> None:
        self._context = context
        self._allowed = config.authorized_clients
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: zmq.Socket[Any] | None = None

    def start(self) -> None:
        socket = self._context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        try:
            socket.bind(_ZAP_ENDPOINT)  # échoue ici, dans le thread appelant, si un autre gestionnaire ZAP existe déjà
        except zmq.ZMQError:
            socket.close(linger=0)
            raise
        self._socket = socket
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="curve-zap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():  # ne devrait jamais arriver : le thread sonde toutes les 100 ms
            _log.error("curve_zap_thread_did_not_stop")
        self._thread = None

    def _serve(self) -> None:
        socket = self._socket
        assert socket is not None
        try:
            while not self._stop.is_set():
                if not socket.poll(_ZAP_POLL_MS):
                    continue
                request = socket.recv_multipart()
                try:
                    reply = self._decide(request)
                except Exception:  # noqa: BLE001 -- une requête illisible est refusée, jamais autorisée ni fatale
                    _log.exception("curve_zap_request_failed")
                    reply = self._reply(request[1] if len(request) > 1 else b"", b"400", b"Internal error")
                socket.send_multipart(reply)
        finally:
            socket.close(linger=0)  # fermé par le thread qui l'utilise

    @staticmethod
    def _reply(request_id: bytes, status: bytes, text: bytes) -> list[bytes]:
        return [_ZAP_VERSION, request_id, status, text, b"nothierly-client" if status == b"200" else b"", b""]

    def _decide(self, request: list[bytes]) -> list[bytes]:
        if len(request) < 6:
            return self._reply(request[1] if len(request) > 1 else b"", b"400", b"Not enough frames")
        version, request_id, _domain, _address, _identity, mechanism = request[:6]
        credentials = request[6:]
        if version != _ZAP_VERSION:
            return self._reply(request_id, b"400", b"Invalid version")
        if mechanism != b"CURVE" or len(credentials) != 1:
            _log.warning("curve_zap_refused mechanism=%r", mechanism)
            return self._reply(request_id, b"400", b"CURVE only")
        key = z85.encode(credentials[0]).decode("ascii")
        if key not in self._allowed:
            _log.warning("curve_client_refused key=%s", key[:8] + "...")  # préfixe seulement
            return self._reply(request_id, b"400", b"Unknown key")
        return self._reply(request_id, b"200", b"OK")

    def __enter__(self) -> ClientAuthenticator:
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.stop()


def is_local_endpoint(address: str) -> bool:
    return address.startswith(_LOCAL_PREFIXES)


def transport_refusal(address: str, *, curve_enabled: bool, allow_plaintext_remote: bool) -> str | None:
    """Message de refus si ``address`` n'est pas local et qu'aucun chiffrement n'est configuré, sinon ``None``."""
    if is_local_endpoint(address) or curve_enabled or allow_plaintext_remote:
        return None
    return (
        f"Refusé : {address!r} n'est pas un endpoint local et aucun chiffrement CURVE n'est configuré "
        f"(variables NOTHIERLY_CURVE_*, voir scripts/generate_curve_keys.py). --allow-plaintext-remote pour passer outre, à vos risques."
    )

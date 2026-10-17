"""Alertes push par webhook pour les urgences opérationnelles d'un worker (homme mort, arrêt, pannes répétées).

**Pourquoi pas ``core.notifier``.** Ce module existe déjà côté moteur (``TelegramNotifier``, ``aiohttp``, ``core.config``) ;
les workers n'importent jamais ``core`` ni ``src`` (frontière d'IP, vérifiée par un test AST) et ne portent pas ses
dépendances. Celui-ci n'utilise que la stdlib.

**Garanties.**

- ``alert()`` ne bloque **jamais** l'appelant et ne lève **jamais** : le message part dans une file bornée, un thread
  démon fait la requête ``POST``. Webhook hors ligne, DNS lent, réponse 5xx, URL invalide : l'échec est journalisé (sans
  l'URL) et c'est tout. La boucle de trading n'attend rien.
- Délai d'attente de la requête : ``timeout_s`` (3 s) par opération réseau. Il ne borne pas la résolution DNS, qui peut
  durer plus longtemps ; ça n'a pas d'importance ici puisque cela se passe dans le thread d'envoi, pas dans le worker.
- Anti-spam : une même clé n'est envoyée qu'une fois par ``cooldown_s`` (5 min). La file est bornée : plein, le message
  est abandonné (une alerte perdue vaut mieux qu'un worker qui accumule).
- **Aucune garantie de livraison.** Une alerte est un confort pour l'opérateur ; aucune décision de trading ni de repli
  n'en dépend. Un worker dont le webhook est muet se protège exactement comme avant.

**Configuration.** ``WEBHOOK_URL`` dans l'environnement. L'URL contient souvent un jeton : elle n'est ni journalisée ni
affichée (seul l'hôte l'est). Elle doit être en ``https`` (``http`` n'est accepté que vers la boucle locale, pour les
tests) ; sinon ``from_env`` lève ``AlertConfigError`` au démarrage : une alerte mal configurée doit se voir tout de
suite, pas le jour de l'incident.

**Formats.** Discord (``content``), Telegram (``text``, le ``chat_id`` doit figurer dans la requête de l'URL), Slack
(``text``) ; hôte inconnu : ``content`` et ``text`` ensemble. Formes tirées de la documentation de chaque service, jamais
éprouvées contre le vrai service ici.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Final, Protocol

_log = logging.getLogger(__name__)

ENV_WEBHOOK_URL: Final[str] = "WEBHOOK_URL"
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
_DISCORD_LIMIT: Final[int] = 2000
_TELEGRAM_LIMIT: Final[int] = 4096


class AlertConfigError(ValueError):
    """``WEBHOOK_URL`` invalide : erreur de déploiement, volontairement bruyante (le message ne répète jamais l'URL)."""


class Severity(str, Enum):
    CRITICAL = "CRITIQUE"
    INFO = "INFO"


class Alerter(Protocol):
    """Ce que les workers utilisent : un envoi sans attente ni exception."""

    def alert(self, text: str, *, key: str, severity: Severity = Severity.CRITICAL) -> bool: ...


def validate_webhook_url(url: str) -> str:
    """Renvoie l'hôte (pour les journaux) ou lève ``AlertConfigError`` sans citer l'URL."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if not host:
        raise AlertConfigError(f"{ENV_WEBHOOK_URL} n'est pas une URL valide.")
    if parsed.scheme == "https":
        return host
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return host
    raise AlertConfigError(f"{ENV_WEBHOOK_URL} doit être en https (http n'est accepté que vers la boucle locale).")


def build_payload(host: str, text: str) -> dict[str, str]:
    """Corps JSON selon le service : l'hôte décide de la forme."""
    if host.endswith("discord.com") or host.endswith("discordapp.com"):
        return {"content": text[:_DISCORD_LIMIT]}
    if host == "api.telegram.org":
        return {"text": text[:_TELEGRAM_LIMIT]}
    if host.endswith("slack.com"):
        return {"text": text}
    return {"content": text[:_DISCORD_LIMIT], "text": text}


class WebhookAlerter:
    def __init__(
        self,
        url: str,
        *,
        source: str,
        timeout_s: float = 3.0,
        cooldown_s: float = 300.0,
        queue_size: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._host = validate_webhook_url(url)
        self._url = url  # jamais journalisée : elle porte le jeton
        self._source = source
        self._timeout_s = timeout_s
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.delivered = 0
        self.failed = 0
        self.dropped = 0

    @classmethod
    def from_env(cls, *, source: str, environ: Mapping[str, str] | None = None, timeout_s: float = 3.0, cooldown_s: float = 300.0) -> WebhookAlerter | None:
        """``None`` si ``WEBHOOK_URL`` est absente ou vide (alertes désactivées) ; ``AlertConfigError`` si elle est invalide."""
        url = (os.environ if environ is None else environ).get(ENV_WEBHOOK_URL, "").strip()
        return cls(url, source=source, timeout_s=timeout_s, cooldown_s=cooldown_s) if url else None

    def alert(self, text: str, *, key: str, severity: Severity = Severity.CRITICAL) -> bool:
        """Met l'alerte en file. ``True`` si elle sera envoyée ; ``False`` si étouffée (délai) ou abandonnée (file pleine)."""
        try:
            now = self._clock()
            with self._lock:
                last = self._last_sent.get(key)
                if last is not None and now - last < self._cooldown_s:
                    return False
                self._last_sent[key] = now
            message = f"[{severity.value}][{self._source}] {text}"
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                self.dropped += 1
                _log.warning("alert_dropped queue_full key=%s", key)
                return False
            self._ensure_thread()
            return True
        except Exception:  # noqa: BLE001 -- une alerte ne doit jamais faire échouer l'appelant
            _log.exception("alert_failed key=%s", key)
            return False

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="webhook-alerter", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            try:
                if message is None:
                    return
                self._deliver(message)
            finally:
                self._queue.task_done()

    def _deliver(self, message: str) -> None:
        body = json.dumps(build_payload(self._host, message)).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=body, method="POST", headers={"Content-Type": "application/json", "User-Agent": "nothierly-worker-alert"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:  # noqa: S310 -- schéma vérifié par validate_webhook_url
                response.read(256)
            self.delivered += 1
        except Exception as exc:  # noqa: BLE001 -- jamais d'exception hors de ce thread ; ni l'URL ni la réponse ne sont journalisées
            self.failed += 1
            _log.warning("alert_delivery_failed host=%s error=%s", self._host, type(exc).__name__)

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Attend que la file soit vidée et envoyée (arrêt propre, tests). ``False`` si le délai est dépassé."""
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout_s: float = 5.0) -> None:
        self.flush(timeout_s)
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                return
            thread.join(timeout=timeout_s)

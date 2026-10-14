"""Clients Numerai du worker : un client sans réseau (par défaut) et l'adaptateur ``numerapi``.

Le worker ne parle à Numerai que par ce protocole. Le client par défaut est ``DryRunClient`` : rien
n'est envoyé tant qu'un ``NumerapiClient`` n'est pas câblé explicitement.

``NumerapiClient`` suit l'API documentée de ``numerapi`` (``SignalsAPI(public_id, secret_key)``,
``upload_predictions(file_path, model_id=...)``, ``ticker_universe()``). Il n'a **pas** été exécuté
contre le vrai service : ``numerapi`` n'est pas installé dans cet environnement et aucun identifiant
Numerai n'a été utilisé. Les tests vérifient le câblage des appels avec un faux module.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

_log = logging.getLogger(__name__)


class NumeraiClient(Protocol):
    def ticker_universe(self) -> frozenset[str] | None:
        """Tickers acceptés aujourd'hui, ou ``None`` si inconnus (pas de filtrage possible)."""
        ...

    def submit(self, csv_path: Path) -> str:
        """Envoie le fichier ; renvoie un identifiant de soumission. Lève en cas d'échec."""
        ...


class DryRunClient:
    """N'envoie rien : journalise ce qui serait soumis. Univers inconnu, donc aucun filtrage."""

    def ticker_universe(self) -> frozenset[str] | None:
        return None

    def submit(self, csv_path: Path) -> str:
        _log.warning("numerai_dry_run_no_submission file=%s", csv_path)
        return f"dry-run:{csv_path.name}"


class NumerapiClient:
    """Adaptateur ``numerapi``. Import différé : ``numerapi`` n'est requis que pour une soumission réelle."""

    def __init__(self, public_id: str, secret_key: str, model_id: str) -> None:
        try:
            import numerapi  # paquet optionnel, sans stubs (voir pyproject : override mypy « numerapi.* »)
        except ImportError as exc:  # pragma: no cover - dépend de l'environnement
            raise RuntimeError("numerapi n'est pas installé : pip install numerapi") from exc
        if not (public_id and secret_key and model_id):
            raise ValueError("public_id, secret_key et model_id sont requis.")
        self._api: Any = numerapi.SignalsAPI(public_id, secret_key)
        self._model_id = model_id

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> NumerapiClient:
        env = os.environ if environ is None else environ
        missing = [k for k in ("NUMERAI_PUBLIC_ID", "NUMERAI_SECRET_KEY", "NUMERAI_MODEL_ID") if not env.get(k)]
        if missing:
            raise ValueError(f"Variables d'environnement manquantes : {', '.join(missing)}")
        return cls(env["NUMERAI_PUBLIC_ID"], env["NUMERAI_SECRET_KEY"], env["NUMERAI_MODEL_ID"])

    def ticker_universe(self) -> frozenset[str] | None:
        return frozenset(str(t) for t in self._api.ticker_universe())

    def submit(self, csv_path: Path) -> str:
        return str(self._api.upload_predictions(str(csv_path), model_id=self._model_id))

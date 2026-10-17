"""Dossier des données persistantes des workers (bases SQLite), piloté par ``NOTHIERLY_DATA_DIR``.

Dans Docker, ``NOTHIERLY_DATA_DIR=/data`` est un volume : c'est lui qui garantit que les fills et le slippage survivent à un
redémarrage ou à une mise à jour du conteneur. Hors Docker, le défaut est ``data/`` (dossier ignoré par git).

Stdlib pure : les workers n'importent jamais ``core`` ni ``src``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

ENV_DATA_DIR: Final[str] = "NOTHIERLY_DATA_DIR"
DEFAULT_DATA_DIR: Final[str] = "data/"


def data_dir() -> Path:
    """``$NOTHIERLY_DATA_DIR`` (ou ``data/`` si absent ou vide). Le dossier n'est PAS créé ici : voir :func:`ensure_parent`."""
    return Path(os.environ.get(ENV_DATA_DIR) or DEFAULT_DATA_DIR)


def default_db_path(filename: str) -> Path:
    """``<data_dir>/<filename>``. Pur : ne touche pas au disque (appelé à la construction du parseur, y compris pour ``--help``)."""
    return data_dir() / filename


def ensure_parent(path: str | Path) -> Path:
    """Crée le dossier parent de ``path``. Échoue vite (``OSError``) si on ne peut pas y écrire : mieux vaut ne pas démarrer que
    tourner sans persistance."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

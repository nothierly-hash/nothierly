"""Journalisation des workers et du pont : stdout, plus un fichier tournant si un dossier est configuré.

Un worker qui tourne seul sur un VPS écrit en continu ; un fichier de journal sans limite finit par remplir le disque
(incident déjà vécu sur ce projet, voir le commentaire de ``x-default-logging`` dans ``docker-compose.yml``). Le fichier est donc
un ``logging.handlers.RotatingFileHandler`` : 10 Mo par fichier, 5 sauvegardes, soit au plus ~60 Mo par processus.

Dans Docker, le plus simple reste de ne pas écrire de fichier et de laisser le pilote ``json-file`` du compose faire la
rotation de stdout (déjà plafonnée dans ``docker-compose.platform.yml``) ; ce module sert aux hôtes sans Docker (le worker
Darwinex sous Windows, un lancement direct sur le VPS).

Le moteur a son propre module (``core.logger``, JSON structuré, 50 Mo x 10) : celui-ci ne l'importe pas, les workers
n'importent jamais ``core``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Final

ENV_LOG_DIR: Final[str] = "NOTHIERLY_LOG_DIR"
DEFAULT_MAX_BYTES: Final[int] = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT: Final[int] = 5
_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"
_MARK: Final[str] = "_nothierly_worker_handler"


def configure_logging(
    name: str,
    *,
    level: int = logging.INFO,
    log_dir: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> Path | None:
    """Installe les gestionnaires du logger racine. Renvoie le chemin du fichier de journal, ou ``None`` (stdout seul).

    ``log_dir`` absent : la variable ``NOTHIERLY_LOG_DIR`` ; toujours absente : pas de fichier. Appel répété sans
    doublon (les gestionnaires posés par un appel précédent sont retirés). Un dossier impossible à créer n'empêche pas
    le démarrage : stdout continue, et l'erreur est journalisée.
    """
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MARK, False)]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    setattr(stream, _MARK, True)
    root.addHandler(stream)

    directory = log_dir if log_dir is not None else os.environ.get(ENV_LOG_DIR)
    if not directory:
        return None
    path = Path(directory) / f"{name}.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8", delay=True)
    except OSError as exc:
        logging.getLogger(__name__).error("log_file_unavailable dir=%s error=%s : stdout seul", directory, type(exc).__name__)
        return None
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MARK, True)
    root.addHandler(file_handler)
    return path

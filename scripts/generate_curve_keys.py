"""Génère les clés CURVE du bus ZeroMQ : une paire pour le serveur (le pont) et une par worker.

Usage ::

    python scripts/generate_curve_keys.py --clients hyperliquid bybit numerai darwinex --out-dir secrets/curve

Écrit, dans ``--out-dir`` :

- ``curve_server.env`` : à placer sur la machine du moteur / du pont. Contient la clé SECRÈTE du serveur et les clés
  PUBLIQUES des workers autorisés.
- ``curve_<client>.env`` : un par worker, à placer sur SA machine uniquement. Contient la clé secrète de ce worker, sa
  clé publique et la clé publique du serveur (qu'il épingle).

Ne jamais afficher ni commiter une clé secrète : ce script n'imprime que des chemins et des clés publiques, refuse
d'écraser un fichier existant (régénérer = invalider les clés déployées) et vérifie que git ignore les fichiers écrits.
Charger un fichier avant de lancer un processus : ``set -a; source curve_server.env`` (bash) ou ``Get-Content`` +
``$env:`` (PowerShell), ou le copier dans le ``.env`` du service.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# `python scripts/...` met scripts/ (et non la racine) dans sys.path : `contracts` vit à la racine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.curve_transport import (  # noqa: E402
    ENV_CLIENT_KEYS,
    ENV_CLIENT_PUBLIC,
    ENV_CLIENT_SECRET,
    ENV_SERVER_PUBLIC,
    ENV_SERVER_SECRET,
    generate_keypair,
)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _write_private(path: Path, text: str) -> None:
    """Crée ``path`` (jamais d'écrasement) avec des droits restreints au propriétaire quand le système le permet."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _git_ignores(path: Path) -> bool | None:
    """``True``/``False`` selon ``git check-ignore``, ``None`` si git n'est pas utilisable ici."""
    try:
        return subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Génère les clés CURVE du bus (serveur + workers).")
    parser.add_argument("--clients", nargs="+", required=True, metavar="NOM", help="un nom par worker (ex. hyperliquid bybit)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    bad = [name for name in args.clients if not _NAME_RE.match(name)]
    if bad or len(set(args.clients)) != len(args.clients):
        print("Noms de clients : minuscules/chiffres/-/_, 32 caractères max, sans doublon.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.out_dir / "curve_server.env", *(args.out_dir / f"curve_{name}.env" for name in args.clients)]
    existing = [t for t in targets if t.exists()]
    if existing:
        print("Refusé : fichier(s) existant(s), rien n'est écrasé :\n  " + "\n  ".join(map(str, existing)), file=sys.stderr)
        return 2

    server = generate_keypair()
    clients = {name: generate_keypair() for name in args.clients}
    _write_private(
        targets[0],
        f"{ENV_SERVER_SECRET}={server.secret}\n{ENV_CLIENT_KEYS}={','.join(c.public for c in clients.values())}\n",
    )
    for name, pair in clients.items():
        _write_private(
            args.out_dir / f"curve_{name}.env",
            f"{ENV_SERVER_PUBLIC}={server.public}\n{ENV_CLIENT_PUBLIC}={pair.public}\n{ENV_CLIENT_SECRET}={pair.secret}\n",
        )

    print(f"Serveur : {targets[0]}  (clé publique {server.public})")
    for name, pair in clients.items():
        print(f"Worker {name:<12}: {args.out_dir / f'curve_{name}.env'}  (clé publique {pair.public})")
    ignored = [_git_ignores(t) for t in targets]
    if any(flag is False for flag in ignored):
        print("ATTENTION : git n'ignore pas certains de ces fichiers. Ne les commitez pas ; ajoutez le dossier à .gitignore.", file=sys.stderr)
    print("Chaque fichier va sur SA machine uniquement. Les clés secrètes ne sont pas affichées.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone Vast.ai connectivity check -- validates the API key and lists the 3 best current GPU offers.

Never rents a machine. Real, live checks only, never fabricated output:

1. Confirms ``VASTAI_API_KEY`` is configured at all (fails fast,
   honestly, if not -- see the module docstring pattern established by
   ``src/dashboard/ems_pages/finops.py``'s own "NON CONFIGURÉ" badge).
2. Confirms the Vast.ai master switch (``VASTAI_ENABLED`` /
   :data:`~core.schemas.VAST_AI_ENABLED_REDIS_KEY`) is on -- :meth:`~src.
   mlops.vast_client.VastAIClient.search_offers` is gated by
   :func:`~src.mlops.vast_client.requires_vast_ai_enabled` the same as
   every other exposure-creating call, so a disabled module fails this
   script the same honest way it fails everything else, rather than a
   confusing generic error.
3. Calls the real ``GET /bundles/`` endpoint via
   :meth:`~src.mlops.vast_client.VastAIClient.search_offers` (reliability
   > 98%, cheapest first) and prints the 3 cheapest real offers -- an
   empty, genuinely-matching market prints "0 offres", never a
   placeholder row.

Run manually, offline::

    python scripts/test_vastai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same real, verified-not-assumed bug scripts/analyze_shadow_run.py already fixes: Windows' default
# console codepage (cp1252) cannot encode this script's own accented French output, corrupting it
# (never crashes, but garbles every é/à/û on a real Windows terminal).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import asyncio

from core.config import get_settings
from core.logger import configure_logging, get_logger
from src.mlops.vast_client import VastAIClient, VastAIDisabledError, VastAIError

_log = get_logger(__name__)


async def _run() -> int:
    settings = get_settings()
    vastai = settings.vastai

    print("=" * 60)
    print("Vast.ai -- vérification de liaison réelle (aucune location)")
    print("=" * 60)

    if vastai.api_key is None:
        print("[FAIL] VASTAI_API_KEY n'est pas configurée -- rien à tester.")
        print("       Ajoutez-la dans .env (voir core/config.py::VastAiSettings).")
        return 2
    print("[OK]   VASTAI_API_KEY est configurée.")

    async with VastAIClient(api_key=vastai.api_key.get_secret_value()) as client:
        try:
            offers = await client.search_offers(min_reliability=0.98, limit=3)
        except VastAIDisabledError:
            print(
                "[FAIL] Module Vast.ai désactivé (VASTAI_ENABLED=false et aucun "
                "override Redis actif) -- voir le Master Switch sur la page Cloud & FinOps du dashboard."
            )
            return 2
        except VastAIError as exc:
            print(f"[FAIL] Échec réel de l'appel à l'API Vast.ai : {exc}")
            return 2

    print(f"[OK]   Clé valide, {len(offers)} offre(s) réelle(s) trouvée(s) (fiabilité > 98%, prix croissant) :")
    if not offers:
        print("       0 offres -- marché réel actuellement vide pour ce filtre, pas une erreur.")
    for rank, offer in enumerate(offers, start=1):
        ram = f"{offer.cpu_ram_gb:.0f} GB RAM" if offer.cpu_ram_gb is not None else "RAM inconnue"
        print(
            f"       {rank}. ask={offer.ask_id:<10} {offer.gpu_name:<12} "
            f"${offer.dph_total:.4f}/h  fiabilité={offer.reliability:.3f}  {ram}"
        )

    print("=" * 60)
    print("Liaison Vast.ai opérationnelle.")
    return 0


def main() -> None:
    configure_logging(service_name="test-vastai")
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()

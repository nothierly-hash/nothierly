"""Lance le pont Cerveau -> plateformes : lit l'état arbitré du moteur dans Redis et publie des ``PlatformSignal`` signés.

Usage (depuis la racine du projet, avec le moteur déjà démarré et ``NOTHIERLY_ZMQ_SECRET`` défini) ::

    python scripts/run_platform_bridge.py --endpoint tcp://127.0.0.1:25570

Le pont **écoute** sur l'endpoint (``bind``) ; les workers s'y connectent (``--address`` du worker). Lire la
docstring de ``src/bridge/platform_bridge.py`` : ce qui est traduit, ce qui ne l'est pas, et les limites.

``--dry-run`` journalise les signaux sans rien publier (premier coup d'œil, sans worker).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/...` met scripts/ (et non la racine) dans sys.path : `core`, `src` et `contracts` vivent à la racine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import logging
from decimal import Decimal

from contracts.curve_transport import CurveClientConfig, CurveConfigError, CurveServerConfig, transport_refusal
from contracts.platform_signal import PlatformSignal, TargetVenue
from contracts.secure_signal_bus import SecureSignalPublisher, WeakSecretError, load_secret_from_env
from core.config import get_settings
from core.redis_client import RedisClient
from core.security import is_system_safe
from src.bridge.platform_bridge import BridgeConfig, PlatformBridge, RedisEngineReader

_log = logging.getLogger("platform_bridge")
_CRYPTO_VENUES = ("ALL", "HYPERLIQUID", "BINANCE", "BYBIT", "ENZYME")


class _LogOnlySink:
    def publish(self, signal: PlatformSignal) -> bool:
        _log.info("bridge_dry_run_signal %s", signal.model_dump_json())
        return True


def _curve_config(*, connect: bool) -> CurveServerConfig | CurveClientConfig | None:
    """Serveur CURVE quand le pont écoute, client quand il se connecte à un broker ; ``None`` si rien n'est configuré."""
    return CurveClientConfig.from_env() if connect else CurveServerConfig.from_env()


async def _amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.mock_mode:
        _log.warning(
            "MOCK_MODE actif : les positions du moteur sont SIMULÉES. Un worker en mode live tradera de l'argent réel "
            "d'après des décisions simulées : garder les workers en dry-run / testnet / démo."
        )
    client = await RedisClient.get_instance(settings.redis, base_delay_seconds=0.5, max_delay_seconds=5.0, max_attempts=3)
    config = BridgeConfig(
        instruments=tuple(args.instruments),
        venue=TargetVenue(args.venue),
        max_position_pct=Decimal(str(args.max_position_pct)),
        size_bucket_pct=Decimal(str(args.size_bucket_pct)) if args.size_bucket_pct > 0 else None,
        min_hold_s=args.min_hold_s,
        refresh_s=args.refresh_s,
        engine_grace_s=args.engine_grace_s,
        poll_s=args.poll_s,
    )
    reader = RedisEngineReader(client, is_system_safe)
    if args.dry_run:
        await PlatformBridge(reader, _LogOnlySink(), config).run()
        return 0
    curve = _curve_config(connect=args.connect)
    with SecureSignalPublisher.from_env(args.endpoint, bind=not args.connect, curve=curve) as publisher:
        _log.info("platform_bridge_started endpoint=%s venue=%s instruments=%s encrypted=%s", publisher.endpoint, config.venue.value, list(config.instruments), publisher.encrypted)
        await PlatformBridge(reader, publisher, config).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pont moteur -> workers de plateformes (PlatformSignal signés).")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:25570", help="endpoint où le pont écoute (les workers s'y connectent)")
    parser.add_argument("--connect", action="store_true", help="se connecte à l'endpoint (frontend d'un broker) au lieu d'écouter")
    parser.add_argument("--venue", choices=_CRYPTO_VENUES, default="ALL", help="ALL = toutes les plateformes crypto")
    parser.add_argument("--instruments", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--max-position-pct", type=float, default=20.0)
    parser.add_argument("--size-bucket-pct", type=float, default=2.5, help="pas de quantification de la taille publiée (masque la conviction du moteur) ; 0 = taille exacte")
    parser.add_argument("--min-hold-s", type=float, default=60.0, help="durée minimale de maintien d'une position publiée avant sa fermeture/inversion ; 0 = aucune")
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--refresh-s", type=float, default=10.0)
    parser.add_argument("--engine-grace-s", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="journalise les signaux sans les publier")
    parser.add_argument("--allow-plaintext-remote", action="store_true", help="autorise un endpoint non local SANS chiffrement CURVE")
    args = parser.parse_args(argv)
    if args.min_hold_s < 0:
        parser.error("--min-hold-s doit être >= 0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not args.dry_run:
        try:
            load_secret_from_env()
            curve = _curve_config(connect=args.connect)
        except (WeakSecretError, CurveConfigError) as exc:
            print(f"Erreur : {exc}", file=sys.stderr)
            return 2
        refusal = transport_refusal(args.endpoint, curve_enabled=curve is not None, allow_plaintext_remote=args.allow_plaintext_remote)
        if refusal:
            print(refusal, file=sys.stderr)
            return 2
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        _log.info("platform_bridge_stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

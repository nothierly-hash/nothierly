"""Injecte un ``PlatformSignal`` signé sur le bus ZMQ, pour tester un worker (dry-run) de bout en bout.

Usage (depuis la racine du projet) ::

    set NOTHIERLY_ZMQ_SECRET=<la même clé que le worker, 32 octets minimum>
    python scripts/inject_test_signal.py --endpoint tcp://127.0.0.1:25570

Le script **écoute** sur l'endpoint (``bind``) et le worker s'y **connecte** (``--address`` du worker) ;
lancer le worker d'abord, puis ce script.

Choix qui évitent les faux négatifs et les accidents :

- Un abonné ZMQ déjà lancé met un instant à se (re)connecter après le ``bind`` : publier une seule fois
  tout de suite perd le message. Le script attend ``--settle`` secondes puis publie le **même** signal
  ``--repeat`` fois ; le worker le dédoublonne par ``signal_id`` (les rejeux suivants apparaissent comme
  ``replay`` dans ses logs, c'est attendu).
- Le signal expire après ``--ttl-ms`` (10 s par défaut) : un message de test oublié ne sera pas exécuté plus tard.
- Endpoint non local refusé sans chiffrement CURVE (ou ``--allow-plaintext-remote``) : on ne pousse pas un signal de test vers un bus réel par mégarde.
- Aucun ordre n'est envoyé par ce script : il publie un signal. Ce que le worker en fait dépend de son mode
  (dry-run par défaut ; ``--live`` envoie de vrais ordres).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# `python scripts/...` met scripts/ (et non la racine) dans sys.path : `contracts` vit à la racine.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from contracts.curve_transport import CurveClientConfig, CurveConfigError, CurveServerConfig, transport_refusal  # noqa: E402
from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue  # noqa: E402
from contracts.secure_signal_bus import DEFAULT_TOPIC, SecureSignalPublisher, WeakSecretError, load_secret_from_env  # noqa: E402



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Injecte un PlatformSignal signé sur le bus ZMQ (test de worker).")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:25570", help="endpoint où ce script écoute (le worker s'y connecte)")
    parser.add_argument("--connect", action="store_true", help="se connecte à l'endpoint (frontend d'un broker) au lieu d'écouter")
    parser.add_argument("--venue", choices=[v.value for v in TargetVenue], default=TargetVenue.HYPERLIQUID.value)
    parser.add_argument("--symbol", default="CRYPTO_BTC_USD")
    parser.add_argument("--direction", choices=[d.value for d in SignalDirection], default=SignalDirection.LONG.value)
    parser.add_argument("--pct", type=float, default=None, help="exposition cible en %% (défaut : 5, ou 0 pour FLAT)")
    parser.add_argument("--stop-loss", type=float, default=None, help="stop_loss_price (exigé par le contrat pour ouvrir chez DARWINEX_ZERO)")
    parser.add_argument("--ttl-ms", type=int, default=10_000)
    parser.add_argument("--settle", type=float, default=2.0, help="secondes d'attente de la connexion de l'abonné")
    parser.add_argument("--repeat", type=int, default=3, help="nombre de publications du même signal")
    parser.add_argument("--allow-plaintext-remote", action="store_true", help="autorise un endpoint non local SANS chiffrement CURVE")
    args = parser.parse_args(argv)

    try:
        secret = load_secret_from_env()
        curve = CurveClientConfig.from_env() if args.connect else CurveServerConfig.from_env()
    except (WeakSecretError, CurveConfigError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    refusal = transport_refusal(args.endpoint, curve_enabled=curve is not None, allow_plaintext_remote=args.allow_plaintext_remote)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2

    direction = SignalDirection(args.direction)
    pct = args.pct if args.pct is not None else (0.0 if direction is SignalDirection.FLAT else 5.0)
    prefix = args.symbol.split("_", 1)[0]
    try:
        asset_class = AssetClass(prefix)
    except ValueError:
        print(f"Erreur : le préfixe {prefix!r} de {args.symbol!r} n'est pas une classe d'actif.", file=sys.stderr)
        return 2

    with SecureSignalPublisher(args.endpoint, secret, topic=DEFAULT_TOPIC, bind=not args.connect, curve=curve) as publisher:
        print(f"{'Connexion à' if args.connect else 'Écoute sur'} {publisher.endpoint} ; attente de l'abonné ({args.settle}s)...")
        time.sleep(args.settle)
        try:  # créé après l'attente : le signal part « frais » (le worker rejette un signal trop vieux)
            signal = PlatformSignal(
                target_venue=TargetVenue(args.venue),
                asset_class=asset_class,
                symbol=args.symbol,
                direction=direction,
                position_size_pct=pct,
                order_type=OrderType.MARKET,
                metadata={} if args.stop_loss is None else {"stop_loss_price": args.stop_loss},
                ttl_ms=args.ttl_ms,
            )
        except ValidationError as exc:
            print(f"Signal refusé par le contrat :\n{exc}", file=sys.stderr)
            return 2
        print(f"Signal {signal.signal_id} : {signal.direction.value} {signal.symbol} cible {signal.position_size_pct}% -> {signal.target_venue.value}")
        queued = [publisher.publish(signal) for _ in range(args.repeat)]
        time.sleep(0.5)  # laisse la file d'envoi se vider avant la fermeture (LINGER 0)
    print(f"Publié {sum(queued)}/{args.repeat} fois (les rejeux sont ignorés par le worker).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnostic global (pre-flight) de Nothierly : vérifie l'infrastructure réelle SANS lancer le trading.

    python scripts/system_health_check.py                       # réseau de test, dry-run
    python scripts/system_health_check.py --live --network mainnet --curve-dir secrets/curve

Ce que le script fait : des lectures (ping Redis, ouverture des bases SQLite avec une écriture annulée, lecture des permissions
des clés d'API et de l'état des comptes, une connexion aux flux publics de trades). Ce qu'il ne fait JAMAIS : passer un ordre,
modifier un compte, écrire une valeur dans Redis, publier un signal. Les clés viennent de l'environnement ou de fichiers ``.env``
(``platform.env``) ; aucune valeur secrète (clé, secret, URL de webhook) n'est imprimée.

Rapport : un tableau PASS / WARN / FAIL / SKIP par sous-système. Code de sortie 0 s'il n'y a aucun FAIL, 1 sinon, avec le détail
précis de chaque échec (``--strict`` : les WARN comptent aussi). ``WARN`` = fonctionne mais pas dans l'état attendu pour la
production (ex. aucun méta-modèle entraîné : le squelette neutre est actif) ; ``FAIL`` = à corriger avant de démarrer.

Sous-systèmes : dépendances importables, Redis, bases SQLite des fills, port du pont ZMQ, secret HMAC du bus, clés CURVE (cohérence
serveur/clients), webhook d'alerte, mode du moteur, modèles du cerveau, méta-modèle (Kelly + quantification sur un état factice
avec faux VPIN), VPIN dans Redis, flux publics, et par plateforme (Hyperliquid, Bybit, Binance) : clés, droits de retrait, mode
de position, lecture du compte.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import functools
import importlib
import importlib.util
import io
import json
import math
import os
import socket
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ALL_VENUES: Final[tuple[str, ...]] = ("hyperliquid", "bybit", "binance")
_ENCODING_FIX: Final[str] = "utf-8"


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    status: Status
    detail: str = ""


def passed(group: str, name: str, detail: str = "") -> Check:
    return Check(group, name, Status.PASS, detail)


def warned(group: str, name: str, detail: str) -> Check:
    return Check(group, name, Status.WARN, detail)


def failed(group: str, name: str, detail: str) -> Check:
    return Check(group, name, Status.FAIL, detail)


def skipped(group: str, name: str, detail: str) -> Check:
    return Check(group, name, Status.SKIP, detail)


# --- Environnement ---------------------------------------------------------------------------


def parse_env_file(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` (commentaires, ``export``, guillemets simples ou doubles). Ne lève pas pour un fichier absent."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def build_env(root: Path, extra_files: Sequence[Path] = (), environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``.env`` du moteur, puis ``platform.env``, puis les fichiers demandés, puis l'environnement du processus (prioritaire)."""
    merged: dict[str, str] = {}
    for path in (root / ".env", root / "platform.env", *extra_files):
        merged.update({k: v for k, v in parse_env_file(path).items() if v != ""})
    merged.update({k: v for k, v in (os.environ if environ is None else environ).items()})
    return merged


def redactor(env: Mapping[str, str]) -> Callable[[str], str]:
    """Remplace par ``***`` toute valeur secrète de l'environnement qui se retrouverait dans un message d'erreur."""
    secret_names = ("SECRET", "KEY", "TOKEN", "PASSWORD", "WEBHOOK")
    secrets = sorted({v for k, v in env.items() if v and len(v) >= 8 and any(s in k.upper() for s in secret_names)}, key=len, reverse=True)

    def redact(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, "***")
        return text

    return redact


# --- Infrastructure --------------------------------------------------------------------------

_REQUIRED_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("redis", "redis"), ("zmq", "pyzmq"), ("pydantic", "pydantic"), ("websockets", "websockets"), ("xgboost", "xgboost"),
    ("sklearn", "scikit-learn"), ("joblib", "joblib"), ("numpy", "numpy"), ("eth_account", "eth-account"),
    ("hyperliquid.info", "hyperliquid-python-sdk"),
)
_PROJECT_MODULES: Final[tuple[str, ...]] = (
    "contracts.platform_signal", "contracts.secure_signal_bus", "contracts.curve_transport", "risk.meta_labeling",
    "src.bridge.platform_bridge", "workers.common.base", "workers.common.fills_db", "workers.common.ws_tracker",
    "workers.hyperliquid.worker", "workers.bybit.worker", "workers.binance.worker", "workers.market_data.vpin_tracker",
)


def check_imports() -> list[Check]:
    checks = []
    missing = []
    for module, package in _REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 -- on veut l'erreur précise, quelle qu'elle soit
            missing.append(f"{package} ({type(exc).__name__})")
    checks.append(failed("Infrastructure", "Dépendances tierces", "manquantes : " + ", ".join(missing) + " -> pip install -r requirements.txt requirements-platform.txt") if missing else passed("Infrastructure", "Dépendances tierces", f"{len(_REQUIRED_MODULES)} paquets importables"))
    broken = []
    for module in _PROJECT_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{module}: {type(exc).__name__}: {str(exc)[:120]}")
    checks.append(failed("Infrastructure", "Modules du projet", "import impossible : " + " | ".join(broken)) if broken else passed("Infrastructure", "Modules du projet", f"{len(_PROJECT_MODULES)} modules importables"))
    return checks


def check_redis(env: Mapping[str, str], *, engine_expected: bool) -> list[Check]:
    group = "Infrastructure"
    host, port = env.get("REDIS_HOST", "127.0.0.1"), int(env.get("REDIS_PORT", "6379"))
    try:
        import redis

        client = redis.Redis(host=host, port=port, password=env.get("REDIS_PASSWORD") or None, db=int(env.get("REDIS_DB", "0")), socket_timeout=3.0, socket_connect_timeout=3.0)
        if not client.ping():
            return [failed(group, "Redis", f"{host}:{port} répond mais le PING n'est pas accepté")]
        checks = [passed(group, "Redis", f"PING ok sur {host}:{port}")]
        alive = bool(client.exists("heartbeat:execution-agent"))
        if alive:
            checks.append(passed(group, "Moteur (heartbeat)", "l'ExecutionAgent écrit son heartbeat"))
        elif engine_expected:
            checks.append(failed(group, "Moteur (heartbeat)", "aucun heartbeat:execution-agent : le moteur ne tourne pas"))
        else:
            checks.append(warned(group, "Moteur (heartbeat)", "moteur arrêté (normal avant le lancement : le lanceur le démarre en dernier)"))
        return checks
    except Exception as exc:  # noqa: BLE001
        return [failed(group, "Redis", f"{host}:{port} injoignable : {type(exc).__name__}: {str(exc)[:160]}")]


def check_sqlite(paths: Sequence[Path]) -> list[Check]:
    """Ouvre chaque base, vérifie ``executions`` et ``order_intents`` et PROUVE l'écriture : insertion annulée, rien ne reste."""
    from workers.common.fills_db import FillsDatabase

    checks = []
    for path in paths:
        name = f"SQLite {path.name}"
        existed = path.is_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            FillsDatabase(path).close()  # crée le schéma s'il manque
            conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
            try:
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                missing = {"executions", "order_intents"} - tables
                if missing:
                    checks.append(failed("Infrastructure", name, f"tables manquantes : {sorted(missing)}"))
                    continue
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO executions(timestamp, venue, symbol, direction, fill_price, fill_qty, fill_id, kind, recorded_ms)"
                    " VALUES (0, 'HEALTHCHECK', 'X', 'BUY', 1.0, 1.0, 'healthcheck-probe', 'probe', 0)")
                conn.execute("ROLLBACK")  # l'écriture a réussi ; on ne laisse aucune trace
                rows = conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]
            finally:
                conn.close()
            checks.append(passed("Infrastructure", name, f"écriture ok (annulée), {rows} fills enregistrés" + ("" if existed else ", base créée")))
        except Exception as exc:  # noqa: BLE001
            checks.append(failed("Infrastructure", name, f"{type(exc).__name__}: {str(exc)[:160]}"))
    if not checks:
        checks.append(warned("Infrastructure", "SQLite fills", "aucune base à vérifier"))
    return checks


def endpoint_port(endpoint: str) -> int:
    return int(endpoint.rsplit(":", 1)[1])


def check_bridge_port(endpoint: str, *, running_ok: bool) -> Check:
    group, name = "Infrastructure", f"Port du pont ({endpoint})"
    try:
        port = endpoint_port(endpoint)
    except (ValueError, IndexError):
        return failed(group, name, "endpoint illisible : attendu tcp://hôte:port")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            if running_ok:
                return warned(group, name, f"port {port} déjà occupé (pont déjà lancé ?) : {exc.strerror}")
            return failed(group, name, f"port {port} déjà occupé : un pont ou un autre processus l'utilise ({exc.strerror}) -> l'arrêter ou changer --endpoint")
    remote = not endpoint.startswith(("tcp://127.0.0.1", "tcp://localhost", "ipc://", "inproc://"))
    return passed(group, name, f"port {port} libre" + (" ; adresse non locale : CURVE obligatoire, ouvrir le port au pare-feu pour le nœud Windows (Darwinex)" if remote else ""))


# --- Sécurité --------------------------------------------------------------------------------


def check_bus_secret(env: Mapping[str, str]) -> Check:
    from contracts.secure_signal_bus import WeakSecretError, load_secret_from_env

    try:
        load_secret_from_env(env)
    except WeakSecretError as exc:
        return failed("Sécurité", "Secret HMAC du bus", str(exc))
    return passed("Sécurité", "Secret HMAC du bus", "NOTHIERLY_ZMQ_SECRET présent et assez long (valeur non affichée)")


def _load_curve_file(path: Path) -> dict[str, str]:
    return parse_env_file(path)


def check_curve(env: Mapping[str, str], curve_dir: Path | None, venues: Sequence[str]) -> list[Check]:
    """Cohérence serveur/clients : clés valides, la clé publique de chaque worker est dans la liste blanche du serveur, et chacun épingle la bonne clé serveur."""
    from contracts.curve_transport import CurveClientConfig, CurveConfigError, CurveServerConfig, public_from_secret

    group = "Sécurité"
    checks: list[Check] = []
    mode = env.get("NOTHIERLY_ZMQ_ENCRYPTION", "").strip().lower()
    if mode in ("false", "0", "no", "off"):
        checks.append(warned(group, "CURVE (interrupteur)", "NOTHIERLY_ZMQ_ENCRYPTION=false : bus en clair, à réserver aux tests locaux"))
    elif mode not in ("true", "1", "yes", "on", "required"):
        checks.append(warned(group, "CURVE (interrupteur)", "NOTHIERLY_ZMQ_ENCRYPTION n'est pas à « true » : sans clés le bus resterait en clair ; le mettre à true pour l'exiger"))
    else:
        checks.append(passed(group, "CURVE (interrupteur)", "NOTHIERLY_ZMQ_ENCRYPTION=true : le chiffrement est obligatoire"))
    server_env = {**env, **(_load_curve_file(curve_dir / "curve_server.env") if curve_dir else {})}
    try:
        server = CurveServerConfig.from_env({**server_env, "NOTHIERLY_ZMQ_ENCRYPTION": "true"})
    except CurveConfigError as exc:
        checks.append(failed(group, "CURVE serveur (pont)", f"{exc} -> python scripts/generate_curve_keys.py --clients {' '.join(venues)} --out-dir secrets/curve"))
        return checks
    assert server is not None
    checks.append(passed(group, "CURVE serveur (pont)", f"clé secrète valide, {len(server.authorized_clients)} client(s) autorisé(s)"))
    server_public = public_from_secret(server.secret_key)
    for venue in venues:
        name = f"CURVE client {venue}"
        client_env = {**env, **(_load_curve_file(curve_dir / f"curve_{venue}.env") if curve_dir else {})}
        try:
            client = CurveClientConfig.from_env({**client_env, "NOTHIERLY_ZMQ_ENCRYPTION": "true"})
        except CurveConfigError as exc:
            checks.append(failed(group, name, str(exc)))
            continue
        assert client is not None
        if client.server_public_key != server_public:
            checks.append(failed(group, name, "épingle une clé publique de serveur qui n'est pas celle du pont : régénérer les clés ensemble"))
        elif client.public_key not in server.authorized_clients:
            checks.append(failed(group, name, "sa clé publique n'est pas dans la liste blanche du pont (NOTHIERLY_CURVE_CLIENT_KEYS)"))
        else:
            checks.append(passed(group, name, "paire valide, serveur épinglé, autorisé par le pont"))
    return checks


def check_webhook(env: Mapping[str, str], *, send_test: bool) -> Check:
    from workers.common.alerts import AlertConfigError, Severity, WebhookAlerter, validate_webhook_url

    group, name = "Sécurité", "Webhook d'alerte"
    url = env.get("WEBHOOK_URL", "").strip()
    if not url:
        return failed(group, name, "WEBHOOK_URL absente de l'environnement : aucune alerte push ne partira (kill switch, homme mort, drawdown...)")
    try:
        host = validate_webhook_url(url)
    except AlertConfigError as exc:
        return failed(group, name, str(exc))
    if not send_test:
        return passed(group, name, f"URL valide ({host}) ; envoi non testé (--send-test-alert)")
    alerter = WebhookAlerter(url, source="health-check")
    alerter.alert("test du health check : le webhook fonctionne.", key="healthcheck", severity=Severity.INFO)
    alerter.flush(10.0)
    alerter.close(2.0)
    if alerter.delivered == 1:
        return passed(group, name, f"message de test envoyé à {host}")
    return failed(group, name, f"l'envoi à {host} a échoué (URL, jeton ou réseau)")


# --- Cerveau ---------------------------------------------------------------------------------


def check_engine_mode(env: Mapping[str, str], *, live: bool) -> Check:
    mock = env.get("MOCK_MODE", "true").strip().lower() not in ("false", "0", "no", "off")
    if mock and live:
        return failed("Cerveau", "Mode du moteur", "MOCK_MODE actif (défaut) : les positions du moteur sont SIMULÉES ; des workers --live ordonneraient de l'argent réel sur des décisions simulées. Mettre MOCK_MODE=false seulement si le moteur trade réellement")
    if mock:
        return warned("Cerveau", "Mode du moteur", "MOCK_MODE actif : positions simulées, workers en dry-run/testnet uniquement")
    return passed("Cerveau", "Mode du moteur", "MOCK_MODE désactivé : le moteur passe de vrais ordres (workers --live autorisés)")


def check_brain_models() -> list[Check]:
    group = "Cerveau"
    try:
        spec = importlib.util.spec_from_file_location("nothierly_run_backend_check", ROOT / "scripts" / "run_backend.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["nothierly_run_backend_check"] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return [failed(group, "Lanceur du moteur (run_backend)", f"import impossible : {type(exc).__name__}: {str(exc)[:200]}")]
    checks = [passed(group, "Lanceur du moteur (run_backend)", "importable : tous les agents se chargent")]
    model, scaler = module._SUPERVISOR_MODEL_PATH, module._SUPERVISOR_SCALER_PATH
    if model.is_file() and scaler.is_file():
        checks.append(passed(group, "XGBoost primaire (artefact)", f"{model.name} + {scaler.name}"))
    else:
        checks.append(warned(group, "XGBoost primaire (artefact)", f"fichier(s) absent(s) ({model.name}) : le squelette neutre tourne (jamais de signal XGBoost informatif)"))
    ppo = sorted((ROOT / "models").glob("ppo_policy_*.pt"))
    checks.append(passed(group, "PPO (politique)", f"{len(ppo)} politique(s) dans models/") if ppo else warned(group, "PPO (politique)", "aucun models/ppo_policy_*.pt : politique non entraînée"))
    return checks


def _fake_state(feature_names: Sequence[str], vpin: float) -> dict[str, float]:
    """Un état factice mais plausible, VPIN fictif inclus : seul le fait que l'équation s'exécute est testé, pas un résultat de marché."""
    base = {
        "ppo_confidence": 0.8, "xgb_confidence": 0.7, "confidence_gap": 0.1, "min_confidence": 0.7, "confidence_spread": 0.1,
        "spread": 0.5, "volatility": 0.002, "imbalance_aligned": 0.3, "recent_conflict_count": 0.0, "vpin": vpin,
    }
    return {name: base[name] for name in feature_names if name in base}


def check_meta_label(model_path: Path) -> list[Check]:
    """Charge le méta-modèle (ou signale le squelette), exécute Kelly sur un état factice avec un faux VPIN et vérifie la quantification."""
    from risk.meta_labeling import MetaLabelModel, kelly_size_multiplier
    from src.bridge.platform_bridge import BridgeConfig, EnginePosition, desired_target

    group = "Cerveau"
    checks: list[Check] = []
    if model_path.is_file():
        try:
            model = MetaLabelModel.from_artifact(model_path)  # strict : une erreur est une erreur, pas un repli
        except Exception as exc:  # noqa: BLE001
            return [failed(group, "Méta-modèle (chargement)", f"{model_path.name} illisible ou incohérent : {type(exc).__name__}: {str(exc)[:200]}")]
        checks.append(passed(group, "Méta-modèle (chargement)", f"{model_path.name} chargé, variables : {', '.join(model.feature_names)}"))
    else:
        model = MetaLabelModel.load_or_skeleton(None)
        checks.append(warned(group, "Méta-modèle (chargement)", f"{model_path} absent : squelette neutre actif (probabilité 0,5, multiplicateur 1,0). Normal tant qu'aucun entraînement n'a abouti (voir scripts/train_metalabel.py)"))
    multipliers = []
    try:
        for vpin in (0.05, 0.35, 0.95):
            features = model.vectorize(_fake_state(model.feature_names, vpin))
            if features is None:
                return checks + [failed(group, "Méta-modèle + Kelly (état factice)", f"le vecteur d'état factice ne couvre pas les variables du modèle {model.feature_names}")]
            probability = model.predict_confidence(features)
            multiplier = kelly_size_multiplier(probability, payoff_ratio=model.payoff_ratio)
            if not (math.isfinite(multiplier) and 0.0 <= multiplier <= 1.5 + 1e-9):
                return checks + [failed(group, "Méta-modèle + Kelly (état factice)", f"multiplicateur hors bornes : {multiplier} (probabilité {probability})")]
            multipliers.append(multiplier)
        if not model.is_fitted and any(m != 1.0 for m in multipliers):
            return checks + [failed(group, "Méta-modèle + Kelly (état factice)", f"le squelette devrait donner exactement 1,0, obtenu {multipliers}")]
    except Exception as exc:  # noqa: BLE001
        return checks + [failed(group, "Méta-modèle + Kelly (état factice)", f"{type(exc).__name__}: {str(exc)[:200]}")]
    checks.append(passed(group, "Méta-modèle + Kelly (état factice)", "multiplicateurs pour VPIN 0,05 / 0,35 / 0,95 : " + ", ".join(f"{m:.3f}" for m in multipliers)))
    cfg = BridgeConfig()
    grid, equity = set(), Decimal(10_000)
    for multiplier in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, *multipliers]:
        if multiplier <= 0:
            continue
        units = Decimal(str(round(0.012 * multiplier, 6)))
        target = desired_target(EnginePosition("BTC/USDT", "LONG", units, Decimal(50_000)), equity, cfg)
        if target is None or target[1] % Decimal("2.5") != 0 or not (0 < target[1] <= cfg.max_position_pct):
            return checks + [failed(group, "Quantification (paliers 2,5 %)", f"taille publiée hors grille pour le multiplicateur {multiplier} : {target}")]
        grid.add(target[1])
    checks.append(passed(group, "Quantification (paliers 2,5 %)", f"tailles publiées sur la grille : {sorted(str(g) for g in grid)} (plafond {cfg.max_position_pct} %)"))
    return checks


def check_vpin_key(env: Mapping[str, str], symbols: Sequence[str]) -> Check:
    group, name = "Données", "VPIN dans Redis"
    try:
        import redis

        client = redis.Redis(host=env.get("REDIS_HOST", "127.0.0.1"), port=int(env.get("REDIS_PORT", "6379")), password=env.get("REDIS_PASSWORD") or None,
                             decode_responses=True, socket_timeout=3.0, socket_connect_timeout=3.0)
        found = {s: client.get(f"market:vpin:{s}") for s in symbols}
    except Exception as exc:  # noqa: BLE001
        return failed(group, name, f"lecture impossible : {type(exc).__name__}: {str(exc)[:120]}")
    present = {s: v for s, v in found.items() if v is not None}
    if not present:
        return warned(group, name, "aucune clé market:vpin:* (démon VPIN arrêté ou fenêtre pas encore pleine : normal avant le lancement)")
    return passed(group, name, ", ".join(f"{s}={v}" for s, v in present.items()))


async def _read_trades(feed: Any, subscriptions: list[str], timeout_s: float) -> list[Any]:
    """Se connecte, s'abonne, lit jusqu'à recevoir un vrai trade (les accusés d'abonnement sont ignorés) ou jusqu'au délai."""
    from workers.common.ws_tracker import websockets_connect

    conn = await websockets_connect(feed.url)
    try:
        for message in subscriptions:
            await conn.send(message)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while (remaining := deadline - loop.time()) > 0:
            try:
                trades: list[Any] = feed.parse(await asyncio.wait_for(conn.recv(), remaining))
            except asyncio.TimeoutError:
                break  # flux muet : « aucun trade », pas une erreur sans message
            if trades:
                return trades
        return []
    finally:
        await conn.close()


def check_public_streams(source: str, symbols: Sequence[str], timeout_s: float = 20.0, *, feed: Any = None) -> Check:
    """Lit UN trade sur le flux public du VPIN (aucune authentification, aucun abonnement privé)."""
    from workers.market_data.vpin_tracker import BinanceAggTradeFeed, HyperliquidTradesFeed

    group, name = "Données", f"Flux public de trades ({source})"
    if feed is None:
        feed = BinanceAggTradeFeed(list(symbols)) if source == "binance" else HyperliquidTradesFeed(list(symbols))
    subscriptions: list[str] = [] if source == "binance" else [json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": c}}) for c in feed._coins]  # noqa: SLF001
    try:
        trades = asyncio.run(_read_trades(feed, subscriptions, timeout_s))
    except Exception as exc:  # noqa: BLE001
        return failed(group, name, f"{type(exc).__name__}: {str(exc)[:160]} ({feed.url})")
    if trades:
        return passed(group, name, f"trade reçu : {trades[0].symbol} {trades[0].qty} @ {trades[0].price}")
    return failed(group, name, f"connexion ouverte mais aucun trade en {timeout_s:.0f} s ({feed.url}) : flux muet ou format inattendu")


# --- Plateformes -----------------------------------------------------------------------------


def _need(env: Mapping[str, str], *names: str) -> list[str]:
    return [n for n in names if not env.get(n)]


def check_hyperliquid(env: Mapping[str, str], network: str, *, live: bool, client_factory: Callable[..., Any] | None = None) -> list[Check]:
    group = "Plateformes"
    missing = _need(env, "HL_ACCOUNT_ADDRESS")
    if missing:
        return [failed(group, "Hyperliquid (compte)", f"{', '.join(missing)} absente : adresse du compte principal dont l'equity sert de base")]
    checks: list[Check] = []
    secret = env.get("HL_AGENT_SECRET", "")
    if secret:
        try:
            from eth_account import Account

            agent = Account.from_key(secret).address
            if agent.lower() == env["HL_ACCOUNT_ADDRESS"].lower():
                checks.append(failed(group, "Hyperliquid (clé d'agent)", "la clé fournie est celle du COMPTE PRINCIPAL (droit de retrait) : créer un API wallet sans retrait et l'approuver"))
            else:
                checks.append(passed(group, "Hyperliquid (clé d'agent)", "adresse d'agent distincte du compte principal (un agent ne peut pas retirer) ; approbation côté Hyperliquid non vérifiée"))
        except Exception as exc:  # noqa: BLE001
            checks.append(failed(group, "Hyperliquid (clé d'agent)", f"clé illisible : {type(exc).__name__}"))
    elif live:
        checks.append(failed(group, "Hyperliquid (clé d'agent)", "HL_AGENT_SECRET absente : --live l'exige"))
    else:
        checks.append(warned(group, "Hyperliquid (clé d'agent)", "HL_AGENT_SECRET absente : suffisant pour le dry-run (lecture seule)"))
    try:
        from workers.hyperliquid.client import Network, SdkHyperliquidClient

        factory = client_factory or SdkHyperliquidClient
        client = factory(network=Network(network), account_address=env["HL_ACCOUNT_ADDRESS"], agent_secret=None, vault_address=env.get("HL_VAULT_ADDRESS"))
        snapshot = client.snapshot()
        universe = client.asset_decimals()
        checks.append(passed(group, "Hyperliquid (lecture du compte)", f"equity {snapshot.equity}, {len(snapshot.positions)} position(s), {len(universe)} actifs ({network})"))
    except Exception as exc:  # noqa: BLE001
        checks.append(failed(group, "Hyperliquid (lecture du compte)", f"{type(exc).__name__}: {str(exc)[:200]}"))
    return checks


def check_bybit(env: Mapping[str, str], network: str, *, live: bool, client_factory: Callable[..., Any] | None = None) -> list[Check]:
    from workers.bybit.client import BybitNetwork, BybitRestClient, HedgeModeError, check_key_info

    group = "Plateformes"
    missing = _need(env, "BYBIT_API_KEY", "BYBIT_API_SECRET")
    if missing:
        return [failed(group, "Bybit (clés)", f"{', '.join(missing)} absente(s)")]
    try:
        client = (client_factory or BybitRestClient)(api_key=env["BYBIT_API_KEY"], api_secret=env["BYBIT_API_SECRET"], network=BybitNetwork(network))
        problems = check_key_info(client.key_info(), live=live)
    except Exception as exc:  # noqa: BLE001
        return [failed(group, "Bybit (permissions)", f"{type(exc).__name__}: {str(exc)[:200]}")]
    checks = [failed(group, "Bybit (permissions)", " ; ".join(problems)) if problems else passed(group, "Bybit (permissions)", "aucun droit de retrait" + (", trading de contrats autorisé" if live else ""))]
    try:
        snapshot = client.snapshot()
        specs = client.instruments()
        checks.append(passed(group, "Bybit (compte, mode one-way)", f"equity {snapshot.equity}, {len(snapshot.positions)} position(s), {len(specs)} perpétuels USDT ({network})"))
    except HedgeModeError as exc:
        checks.append(failed(group, "Bybit (compte, mode one-way)", f"mode hedge détecté : {exc}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(failed(group, "Bybit (compte, mode one-way)", f"{type(exc).__name__}: {str(exc)[:200]}"))
    return checks


def check_binance(env: Mapping[str, str], network: str, *, live: bool, client_factory: Callable[..., Any] | None = None) -> list[Check]:
    from workers.binance.client import BinanceNetwork, BinanceRestClient, HedgeModeError, check_key_permissions

    group = "Plateformes"
    missing = _need(env, "BINANCE_API_KEY", "BINANCE_API_SECRET")
    if missing:
        return [failed(group, "Binance (clés)", f"{', '.join(missing)} absente(s)")]
    try:
        client = (client_factory or BinanceRestClient)(api_key=env["BINANCE_API_KEY"], api_secret=env["BINANCE_API_SECRET"], network=BinanceNetwork(network), base_url=env.get("BINANCE_BASE_URL") or None)
        permissions = client.key_permissions()
        problems = check_key_permissions(permissions, live=live, mainnet=network == "mainnet")
    except Exception as exc:  # noqa: BLE001
        return [failed(group, "Binance (permissions)", f"{type(exc).__name__}: {str(exc)[:200]}")]
    if problems:
        checks = [failed(group, "Binance (permissions)", " ; ".join(problems))]
    elif not permissions.verified:
        checks = [warned(group, "Binance (permissions)", f"droits de la clé non vérifiables ({permissions.reason}) : acceptable sur le réseau de test seulement")]
    else:
        checks = [passed(group, "Binance (permissions)", "aucun droit de retrait (apiRestrictions)" + ("" if permissions.ip_restricted else " ; ATTENTION : clé non restreinte par IP"))]
    try:
        if client.is_hedge_mode():
            checks.append(failed(group, "Binance (mode de position)", "mode hedge actif (dualSidePosition) : passer le compte en one-way"))
        else:
            snapshot = client.snapshot()
            checks.append(passed(group, "Binance (mode de position)", f"one-way ; equity {snapshot.equity}, {len(snapshot.positions)} position(s), {len(client.instruments())} perpétuels USDT ({network})"))
    except HedgeModeError as exc:
        checks.append(failed(group, "Binance (mode de position)", f"position hedge détectée : {exc}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(failed(group, "Binance (mode de position)", f"{type(exc).__name__}: {str(exc)[:200]}"))
    return checks


# --- Rapport ---------------------------------------------------------------------------------

_SYMBOL: Final[dict[Status, str]] = {Status.PASS: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.SKIP: "SKIP"}


def render(checks: Sequence[Check]) -> str:
    lines = ["", "=" * 100, " NOTHIERLY - DIAGNOSTIC GLOBAL (pre-flight, aucun ordre passé)", "=" * 100]
    group = None
    for check in checks:
        if check.group != group:
            group = check.group
            lines.append(f"\n[{group}]")
        first, *rest = (check.detail or "").split("\n")
        lines.append(f"  {_SYMBOL[check.status]:<4}  {check.name:<42} {first}")
        lines += [f"        {'':<42} {r}" for r in rest]
    counts = {s: sum(1 for c in checks if c.status is s) for s in Status}
    lines += ["", "-" * 100, f" PASS {counts[Status.PASS]}   WARN {counts[Status.WARN]}   FAIL {counts[Status.FAIL]}   SKIP {counts[Status.SKIP]}", "-" * 100]
    failures = [c for c in checks if c.status is Status.FAIL]
    if failures:
        lines.append("\nDETAIL DES ECHECS (à corriger avant le lancement) :")
        lines += [f"  - [{c.group}] {c.name} : {c.detail}" for c in failures]
    return "\n".join(lines)


def run_all(args: argparse.Namespace, env: Mapping[str, str]) -> list[Check]:
    redact = redactor(env)
    venues = [v for v in args.venues]
    checks: list[Check] = []

    def guard(group: str, name: str, fn: Callable[[], Check | list[Check]]) -> None:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 -- un sous-système qui plante est un échec, jamais un arrêt du diagnostic
            checks.append(failed(group, name, redact(f"{type(exc).__name__}: {str(exc)[:240]}")))
            return
        for check in ([result] if isinstance(result, Check) else result):
            checks.append(Check(check.group, check.name, check.status, redact(check.detail)))

    guard("Infrastructure", "Imports", check_imports)
    guard("Infrastructure", "Redis", lambda: check_redis(env, engine_expected=args.running))
    data_dir = Path(args.data_dir)
    sqlite_paths = [Path(p) for p in args.fills_db] if args.fills_db else [data_dir / f"{v}_fills.sqlite3" for v in venues]
    guard("Infrastructure", "SQLite", lambda: check_sqlite(sqlite_paths))
    guard("Infrastructure", "Port du pont", lambda: check_bridge_port(args.endpoint, running_ok=args.running))
    guard("Sécurité", "Secret HMAC", lambda: check_bus_secret(env))
    guard("Sécurité", "CURVE", lambda: check_curve(env, Path(args.curve_dir) if args.curve_dir else None, venues))
    guard("Sécurité", "Webhook", lambda: check_webhook(env, send_test=args.send_test_alert))
    guard("Cerveau", "Mode du moteur", lambda: check_engine_mode(env, live=args.live))
    guard("Cerveau", "Modèles", check_brain_models)
    guard("Cerveau", "Méta-modèle", lambda: check_meta_label(Path(args.meta_label_model)))
    guard("Données", "VPIN Redis", lambda: check_vpin_key(env, args.symbols))
    if args.no_network:
        checks.append(skipped("Données", "Flux public de trades", "--no-network"))
    else:
        guard("Données", "Flux public", lambda: check_public_streams(args.vpin_source, args.symbols))
    for venue in venues:
        if args.no_network:
            checks.append(skipped("Plateformes", venue, "--no-network : appels REST non effectués"))
            continue
        runner = {"hyperliquid": check_hyperliquid, "bybit": check_bybit, "binance": check_binance}[venue]
        guard("Plateformes", venue, functools.partial(runner, env, args.network, live=args.live))
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", choices=["testnet", "mainnet"], default="testnet", help="réseau des appels REST aux plateformes")
    parser.add_argument("--live", action="store_true", help="vérifie aussi les conditions du mode réel (clés de trading, MOCK_MODE désactivé)")
    parser.add_argument("--venues", nargs="+", choices=ALL_VENUES, default=list(ALL_VENUES))
    parser.add_argument("--curve-dir", default="secrets/curve", help="dossier de generate_curve_keys.py (vide pour lire l'environnement seul)")
    parser.add_argument("--endpoint", default="tcp://0.0.0.0:25570", help="adresse où le pont écoutera")
    parser.add_argument("--data-dir", default=".", help="dossier des bases SQLite des workers")
    parser.add_argument("--fills-db", nargs="*", default=[], help="chemins explicites des bases de fills")
    parser.add_argument("--meta-label-model", default=str(ROOT / "models" / "metalabel_weights.json"))
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--vpin-source", choices=["binance", "hyperliquid"], default="binance")
    parser.add_argument("--no-network", action="store_true", help="n'appelle ni plateformes ni flux publics")
    parser.add_argument("--running", action="store_true", help="le système tourne déjà : port occupé et moteur actif sont normaux")
    parser.add_argument("--send-test-alert", action="store_true", help="envoie un message de test au webhook (action sortante, off par défaut)")
    parser.add_argument("--strict", action="store_true", help="les WARN comptent comme des échecs")
    parser.add_argument("--env-file", action="append", default=[], type=Path, help="fichier .env supplémentaire")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):  # redirigé vers un fichier ou un tube : pas de reconfigure
            try:
                stream.reconfigure(encoding=_ENCODING_FIX, errors="replace")
            except OSError:
                pass
    args = parse_args(argv)
    env = build_env(ROOT, args.env_file)
    # les clés sont lues dans ce dictionnaire ; les bibliothèques du projet lisent os.environ : on y met ce que le diagnostic a trouvé
    for key, value in env.items():
        os.environ.setdefault(key, value)
    checks = run_all(args, env)
    print(render(checks))
    bad = [c for c in checks if c.status is Status.FAIL or (args.strict and c.status is Status.WARN)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

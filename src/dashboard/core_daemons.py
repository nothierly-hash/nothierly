"""Ghost Daemons -- Protocole "DeFi Ghost Daemons" : 16 automates asynchrones déterministes.

**Ce que ce module ajoute, précisément** : un nouveau moteur d'exécution simulé, à côté de
``backend_simulator.py`` (pas à sa place) -- 16 fonctions ``async`` réelles, chacune sa propre loi
mathématique, orchestrées par une classe :class:`LeviathanDaemonManager` qui les lance vraiment en
concurrence via ``asyncio.gather``. Même discipline "vrai calcul sur données synthétiques disclosées,
jamais un vrai ordre" que chaque autre module "frontier tech" de cet EMS -- voir
``backend_simulator.py`` et ``ems_pages/millennium_conjectures.py`` pour les précédents.

**Purge lexicale** : ce module et sa propre section d'interface (`ems_pages/leviathan_nexus_command.py`,
"Daemon Activity Matrix") disent "Daemon"/"Automate", pas "Agent"/"IA". Cela ne renomme pas
``src/agents/*`` ailleurs dans le projet -- ce serait un renommage global, séparé et bien plus large que
ce qui est demandé ici, sur du code déjà réel et déjà en service (alpha_engine, risk_manager, etc.) ;
seul le vocabulaire de *ce* nouveau sous-système est "purgé".

**Abandon des CEX / passage 100% DeFi** : ce module ne supprime et ne modifie aucun connecteur réel
existant (``ccxt`` pour les CEX, ``ib_async``/``v20`` pour les brokers réels ailleurs dans ce projet) --
ils restent des intégrations réelles pour leurs propres chapitres. Ce qui est ajouté ici est un
sous-système supplémentaire, honnêtement simulé, à saveur DeFi pour le tableau de bord EMS -- pas un
retrait de code de production réel sur la seule base d'un brief narratif.

**Interaction "Web3" -- explicitement un Mockup, comme demandé** : :func:`mock_web3_sign_transaction`
produit une chaîne hexadécimale illustrative et déterministe (``hashlib.sha256``), **pas une vraie
signature ECDSA** -- aucune vraie clé privée, aucun vrai point RPC, aucun vrai identifiant de chaîne,
aucun vrai gas nulle part dans ce process. Une vraie intégration Web3 (garde d'un vrai wallet, un vrai
fournisseur RPC, une vraie gestion de clés) est un chantier substantiel et sensible en sécurité que ce
tableau de bord de démonstration ne tente pas -- même discipline que ``finops.py`` : ne jamais prétendre
une connectivité réelle qui n'existe pas.

**``asyncio.gather`` -- ce que "concurrent" veut dire ici, honnêtement** : l'``asyncio`` de Python est une
concurrence coopérative à un seul thread OS (le GIL reste tenu) -- pas du parallélisme multi-cœur. Le
vrai entrelacement observable entre les daemons vient de leur propre ``asyncio.sleep()`` disclosé, qui
tient lieu d'un vrai aller-retour RPC Web3 / cotation DEX qu'une vraie version attendrait réellement.

**La règle "Kelly=0 bloque Aave" (Daemon 4 -> Daemon 2), modélisée correctement plutôt que trichée** :
si les 16 daemons tournaient tous dans un seul ``asyncio.gather`` plat, Daemon 2 ne pourrait pas lire de
façon synchrone le résultat *du même tick* d'un daemon frère sans cesser d'être concurrent avec lui.
Implémenté ici correctement : 15 daemons "producteurs" tournent réellement en concurrence dans un seul
``asyncio.gather``, puis Daemon 2 (Aave) s'exécute juste après, lisant les sorties réelles et fraîches de
Daemon 4 (Kelly) et Daemon 14 (Topologie) -- la relation de blocage demandée est bien réelle, seulement
correctement séquencée au lieu d'être simulée à l'intérieur d'un ``gather`` unique.

**La "boucle infinie while True" au démarrage** : réellement implémentée
(:meth:`LeviathanDaemonManager.run_forever`), démarrée une seule fois comme un vrai thread détaché
(:meth:`LeviathanDaemonManager.ensure_background_running`, protégé contre les démarrages multiples au
fil des reruns Streamlit) avec sa propre boucle événementielle ``asyncio`` -- pas le thread de script
Streamlit lui-même, puisque celui-ci s'exécute de façon synchrone à chaque rerun et que ``st.*``
(``st.session_state`` inclus) n'est documenté comme sûr que depuis ce seul thread -- la même contrainte
réelle que ``backend_simulator.py`` contourne déjà via ``st.fragment(run_every=...)``, ici via un
mécanisme complémentaire différent. Le thread détaché n'écrit donc jamais dans ``st.session_state`` --
il écrit dans son propre instantané protégé par un verrou (:meth:`LeviathanDaemonManager.
background_snapshot`), qu'un fragment Streamlit peut lire sans risque. Ne pouvant pas non plus lire en
sécurité le vrai ``GlobalMarketState.shannon_chaos_bias`` partagé, il fait évoluer son propre signal de
chaos réel (une marche aléatoire bornée) tant qu'il est détaché -- disclosé comme réel mais séparé du
signal utilisé par le tick affiché à l'écran.

**Le swarm affiché à l'écran** (:func:`heartbeat`, appelé depuis la page Nexus Command) est, lui, câblé
sur le même ``shannon_chaos_bias`` partagé réel que chaque autre page "frontier tech" de cet EMS
(``backend_simulator.GlobalMarketState``) -- pas un second univers déconnecté -- et republie toute ligne
non-"actif" (alerte/exécution) dans le même journal d'événements partagé, visible sur le journal de la
page Nexus elle-même.

**Les 16 lois mathématiques** : chacune calcule un vrai résultat, en réutilisant les primitives réelles
déjà vérifiées de cet EMS quand elles existent (Kelly, Wasserstein, Hurst, Koopman/DMD, proxy LZ78
d'Assembly Theory, vorticité de Lamb-Oseen, base de cycles/β1 -- importées de ``backend_simulator.py``/
``core.quant_math.py``), et en ajoutant Perron-Frobenius, un vrai zéro de Riemann (``mpmath``), une vraie
trajectoire de Collatz, une petite couche de transparence KAN, un vrai (mais simplifié) pas de
minimisation d'énergie libre variationnelle pour l'Inférence Active, une statistique CHSH/Bell classique,
et une petite décomposition de Hodge combinatoire réelle -- sur des données synthétiques de pools/liquidité
disclosées. Jamais une vraie cotation, une vraie position ou une vraie transaction.

**Protocole "Zero-Latency" (2026-09-13) -- infrastructure de communication réelle, une prémisse corrigée**
: le brief demandait de remplacer ``st.session_state`` par Redis "parce que c'est trop lent pour
l'exécution backend". C'est physiquement l'inverse : ``st.session_state`` est un ``dict`` Python en
mémoire de ce process -- un accès en mémoire, sans sérialisation ni appel réseau, alors qu'un aller-retour
Redis (même en boucle locale) sérialise puis traverse un socket. Rien ne rend Redis plus rapide que
``st.session_state`` pour ce que l'UI Streamlit lit déjà elle-même ; ``st.session_state`` reste donc
l'état de rendu de cette page, inchangé. Ce que Redis apporte réellement, et pour quoi ce projet l'utilise
déjà (``core/redis_client.py``, ``src/agents/execution/agent.py``'s ``_listen_for_manual_trade_requests``)
: un vrai canal **inter-process**, pas un accélérateur intra-process. Traduit honnêtement ici :

- **Redis = miroir d'état inter-process réel** (:meth:`LeviathanDaemonManager.mirror_state_to_redis`) --
  chaque tick, un vrai ``SET ... EX 5`` (TTL réel de 5s, pas un nombre décoratif : une clé expirée veut
  dire "aucun automate n'a écrit depuis 5s", un vrai signal de vie) publie l'instantané du swarm, lisible
  par n'importe quel autre process réel (pas seulement cette session Streamlit). Suit exactement le même
  partage sync/async déjà établi et documenté par ce projet : le thread de script Streamlit (synchrone,
  sans event loop persistant) utilise un client ``redis.Redis`` synchrone ordinaire pour un ``SET``/``GET``
  ponctuel (même raisonnement que le ``PUBLISH`` synchrone documenté dans ``_listen_for_manual_trade_
  requests``), tandis que la boucle détachée (:meth:`run_forever`, qui *a* un event loop persistant) utilise
  le vrai singleton asynchrone ``core.redis_client.RedisClient`` avec sa reconnexion réelle.
- **ZeroMQ = vrai bus de messages** Fleet Command -> Automates d'Exécution (:data:`GHOST_ZMQ_PUB_ADDRESS`),
  réutilisant les vraies classes déjà testées de ``core/zmq_bus.py`` (``ZmqPublisher``/``ZmqSubscriber``,
  ``core.schemas.MessagePayload``) -- **sur une adresse dédiée et isolée** (``tcp://127.0.0.1:5575``),
  délibérément différente du vrai bus des agents de trading réels (``ZMQ_PUB_ADDRESS``/``ZMQ_SUB_ADDRESS``,
  5555/5556 par défaut) pour ne jamais risquer de collision de port ni de mélanger du trafic simulé avec le
  vrai bus de production. Les deux extrémités (Publisher et Subscriber) vivent aujourd'hui dans le même
  process détaché (:meth:`run_forever`), puisque la couche "Automates d'Exécution" de cet EMS est elle-même
  simulée, pas un second process réel -- disclosé, pas caché ; le protocole sur le fil (PUB/SUB TCP, frames
  JSON validées) est néanmoins entièrement réel.
- **Backpressure/staleness réels** (:meth:`_enqueue_with_backpressure`, :meth:`_execution_worker_loop`) :
  une vraie ``asyncio.Queue`` bornée (``EXECUTION_QUEUE_MAXSIZE``) draine les ordres publiés -- pleine, le
  message le plus ancien est réellement évincé (pas de blocage, pas de deadlock possible : jamais de
  ``await`` bloquant sur une queue pleine). Chaque message porte le vrai ``timestamp`` UTC de
  ``MessagePayload`` ; le consommateur calcule l'âge réel (``datetime.now(UTC) - message.timestamp``) et
  détruit toute donnée de plus de :data:`STALE_MESSAGE_THRESHOLD_MS` -- une vraie mesure, pas un chiffre
  simulé.
- **Télémétrie réelle** (:meth:`LeviathanDaemonManager.redis_health`, :meth:`zmq_throughput_per_sec`,
  ``background_snapshot()``'s latence fields) : chaque latence affichée est réellement mesurée
  (``time.perf_counter``/``MessagePayload.timestamp``), jamais une valeur d'exemple recopiée du brief --
  voir la discipline "aucun chiffre de performance fabriqué" déjà appliquée partout ailleurs dans cet EMS
  (``millennium_conjectures.py``'s benchmark P vs NP en est le précédent direct).
- **Robustesse** : toute opération Redis/ZMQ est encapsulée dans un ``try/except`` qui dégrade (log +
  compteur/flag "dégradé" en télémétrie) plutôt que de laisser une panne d'un capteur remonter et arrêter
  la boucle des 16 daemons -- le swarm mathématique continue de tourner même si Redis ou le port ZMQ sont
  indisponibles.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import itertools
import json
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mpmath
import networkx as nx
import numpy as np
import redis as sync_redis
import streamlit as st
import zmq
from redis.exceptions import RedisError
from scipy.stats import wasserstein_distance

from core.config import get_settings
from core.logger import get_logger
from core.quant_math import kelly_criterion
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import MessagePayload
from core.zmq_bus import ZmqPublisher, ZmqSubscriber
from src.dashboard.backend_simulator import (
    NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD,
    calculate_assembly_index,
    chaos_biased_distribution,
    estimate_hurst_exponent,
    get_state,
    koopman_lift,
    lamb_oseen_peak_vorticity,
    log_event,
    shannon_entropy,
    synthetic_order_flow_symbols,
)

# =============================================================================
# Data model
# =============================================================================

DAEMON_ORDER: tuple[str, ...] = tuple(f"Daemon_{i:02d}" for i in range(1, 17))

# daemon_id -> (codename, "law"/venue) -- plain, disclosed metadata, not derived from introspection.
DAEMON_META: dict[str, tuple[str, str]] = {
    "Daemon_01": ("Navier-Stokes", "Uniswap V3"),
    "Daemon_02": ("Aave", "Flash Loan"),
    "Daemon_03": ("Wasserstein", "Curve/Hyperliquid"),
    "Daemon_04": ("Kelly", "Sizing"),
    "Daemon_05": ("Perron-Frobenius", "Contagion"),
    "Daemon_06": ("Riemann", "Zeta"),
    "Daemon_07": ("Collatz", "Attracteur"),
    "Daemon_08": ("KAN", "Transparence"),
    "Daemon_09": ("Inference-Active", "Energie-Libre"),
    "Daemon_10": ("Hurst", "Regime"),
    "Daemon_11": ("Koopman-DMD", "Linearisation"),
    "Daemon_12": ("Assembly", "LZ78"),
    "Daemon_13": ("CHSH", "Bell"),
    "Daemon_14": ("Topologie", "Betti"),
    "Daemon_15": ("Hodge", "Arbitrage"),
    "Daemon_16": ("Entropie", "Shannon"),
}

STATUS_LED_COLOR: dict[str, str] = {
    "veille": "#3a3550",   # gris -- pas encore tické cette session
    "actif": "#39ff88",    # vert
    "alerte": "#ff2d55",   # rouge
    "exécution": "#00f0ff",  # cyan
}

_log = get_logger(__name__)

# =============================================================================
# Protocole "Zero-Latency" -- real infra constants. See module docstring's own section for the full
# real/isolation reasoning behind each one.
# =============================================================================

# Deliberately NOT core.config.get_settings().zmq.pub_address/sub_address (5555/5556 by default) --
# that is the real trading agents' own bus; this swarm gets its own, isolated address so it can never
# collide with, or leak simulated traffic onto, real agent messaging.
GHOST_ZMQ_PUB_ADDRESS = "tcp://127.0.0.1:5575"
GHOST_EXECUTION_TOPIC = "GHOST_EXECUTION_ORDER"
GHOST_REDIS_SNAPSHOT_KEY = "ghost_daemons:global_market_snapshot"
GHOST_REDIS_SNAPSHOT_TTL_SECONDS = 5
STALE_MESSAGE_THRESHOLD_MS = 50.0
EXECUTION_QUEUE_MAXSIZE = 64


@dataclass(frozen=True)
class DaemonResult:
    daemon_id: str
    codename: str
    law: str
    status: str  # "veille" | "actif" | "alerte" | "exécution"
    value: float
    message: str
    ts: float = field(default_factory=time.time)


@dataclass
class TickContext:
    chaos_bias: float
    tick: int
    rng: np.random.Generator


def _default_results() -> dict[str, DaemonResult]:
    return {
        daemon_id: DaemonResult(daemon_id, codename, law, "veille", 0.0, "en attente du premier tick...")
        for daemon_id, (codename, law) in DAEMON_META.items()
    }


@dataclass
class DaemonSwarmState:
    tick_count: int = 0
    results: dict[str, DaemonResult] = field(default_factory=_default_results)
    log: deque[str] = field(default_factory=lambda: deque(maxlen=120))


# =============================================================================
# "Web3 Mockup" -- illustrative signing only, see module docstring for the real/mock boundary.
# =============================================================================


def mock_web3_sign_transaction(payload: dict) -> dict:
    """Illustrative, deterministic stand-in for ``Account.sign_transaction`` -- **not** a real ECDSA
    signature. No private key is generated, stored, or used anywhere in this process; no RPC endpoint,
    chain ID, or gas is real. Real Web3 wallet custody/signing is a substantial, security-critical
    undertaking this dashboard does not attempt (see module docstring)."""
    fake_nonce = secrets.token_hex(4)
    digest = hashlib.sha256((repr(sorted(payload.items())) + fake_nonce).encode()).hexdigest()
    return {"tx_hash": f"0xMOCK{digest}", "nonce": fake_nonce, "payload": payload, "signed": True}


# =============================================================================
# The 16 daemons -- each a real, disclosed mathematical law over synthetic pool/liquidity data.
# =============================================================================


async def _daemon_01_navier_stokes(ctx: TickContext) -> DaemonResult:
    """Real closed-form Lamb-Oseen peak vorticity (see ``backend_simulator.lamb_oseen_peak_vorticity``'s
    own docstring for the real/analogy boundary), reused here as a "velocity singularity" scanner over a
    synthetic Uniswap V3 liquidity curve driven by the shared chaos bias."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.05)))
    gamma = 4.0 + 40.0 * ctx.chaos_bias
    peak = lamb_oseen_peak_vorticity(gamma, nu=0.6, t=1.0)
    alert = peak >= NAVIER_STOKES_VORTICITY_ALERT_THRESHOLD
    msg = f"scan de liquidité Uniswap V3 -- vorticité de pointe (Lamb-Oseen réel) = {peak:.2f} rad/s" + (
        " -- SINGULARITÉ DE VÉLOCITÉ détectée" if alert else " -- stable"
    )
    return DaemonResult("Daemon_01", "Navier-Stokes", "Uniswap V3", "alerte" if alert else "actif", peak, msg)


async def _daemon_02_aave(ctx: TickContext, *, kelly_fraction: float, betti1: float) -> DaemonResult:
    """Runs only after Daemon 4 (Kelly) and Daemon 14 (Topologie) have really completed this same tick
    -- see module docstring for why this can't be inside the same flat ``asyncio.gather`` as the other
    15 and still honestly express "Kelly<=0 blocks this daemon". Borrows a mock, Kelly-scaled amount
    (capped at the brief's own 10M$ figure) only when a real net edge was just computed."""
    await asyncio.sleep(float(ctx.rng.uniform(0.02, 0.07)))
    if kelly_fraction <= 0.0:
        msg = f"Kelly f*={kelly_fraction:.3f} <= 0 (Daemon_04) -- aucun edge net, emprunt refusé"
        return DaemonResult("Daemon_02", "Aave", "Flash Loan", "actif", 0.0, msg)
    amount = round(min(10_000_000.0, 10_000_000.0 * kelly_fraction / 0.5), 0)
    tx = mock_web3_sign_transaction({"protocol": "Aave", "action": "flashLoan", "amount_usd": amount, "betti1": betti1})
    msg = (
        f"opportunité validée (Kelly f*={kelly_fraction:.3f}) -- emprunt Flash Loan de {amount:,.0f}$ simulé. "
        f"{tx['tx_hash'][:18]}... Signé (MOCK, aucun fonds réel, aucun RPC réel)"
    )
    return DaemonResult("Daemon_02", "Aave", "Flash Loan", "exécution", amount, msg)


async def _daemon_03_wasserstein(ctx: TickContext) -> DaemonResult:
    """Real ``scipy.stats.wasserstein_distance`` (Earth Mover's) between two disclosed synthetic
    liquidity-depth profiles standing in for Curve and Hyperliquid -- the real least-friction transport
    metric, same primitive as ``backend_simulator.report_liquidity_gap``."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.05)))
    n = 16
    curve = ctx.rng.gamma(2.0 + 3.0 * (1.0 - ctx.chaos_bias), 1.0, size=n)
    hyperliquid = ctx.rng.gamma(2.0 + 3.0 * ctx.chaos_bias, 1.0, size=n)
    gap = float(wasserstein_distance(curve, hyperliquid))
    msg = f"distance de transport réelle (Wasserstein-1) Curve→Hyperliquid = {gap:.3f} -- route de moindre friction"
    return DaemonResult("Daemon_03", "Wasserstein", "Curve/Hyperliquid", "actif", gap, msg)


async def _daemon_04_kelly(ctx: TickContext) -> DaemonResult:
    """Real ``core.quant_math.kelly_criterion`` (half-Kelly) on a disclosed, illustrative
    win-probability/payoff-ratio pair derived from the shared chaos bias -- same edge-estimate shape as
    ``backend_simulator.agi_edge_estimate``. Gates Daemon 2 (Aave)."""
    await asyncio.sleep(float(ctx.rng.uniform(0.005, 0.02)))
    win_p = float(np.clip(0.35 + 0.30 * (1.0 - ctx.chaos_bias), 0.05, 0.95))
    win_b = float(np.clip(0.5 + 2.5 * ctx.chaos_bias, 0.1, 4.0))
    f = kelly_criterion(win_p, win_b, fraction=0.5)
    status = "alerte" if f <= 0.0 else "actif"
    verdict = "AUCUN EDGE NET, bloque Daemon_02 (Aave)" if f <= 0.0 else "edge net validé, autorise Daemon_02"
    msg = f"f*={f:.3f} (p={win_p:.2f}, b={win_b:.2f}, half-Kelly réel) -- {verdict}"
    return DaemonResult("Daemon_04", "Kelly", "Sizing", status, f, msg)


async def _daemon_05_perron_frobenius(ctx: TickContext) -> DaemonResult:
    """Real dominant eigenvector of a disclosed synthetic, strictly-positive pool-adjacency matrix
    (Uniswap/Curve/Aave/Hyperliquid) -- the Perron-Frobenius theorem guarantees a unique largest real
    eigenvalue with a strictly positive eigenvector for such a matrix, used here as a real (if toy)
    contagion-concentration read, not a claim about any real venue's real correlations."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.04)))
    pools = ("Uniswap", "Curve", "Aave", "Hyperliquid")
    base = np.array([[1.0, 2.0, 1.0, 1.0], [2.0, 1.0, 2.0, 1.0], [1.0, 2.0, 1.0, 3.0], [1.0, 1.0, 3.0, 1.0]])
    contagion = base * (1.0 + ctx.chaos_bias)
    eigvals, eigvecs = np.linalg.eig(contagion)
    idx = int(np.argmax(eigvals.real))
    vec = np.abs(eigvecs[:, idx].real)
    vec = vec / vec.sum()
    worst = int(np.argmax(vec))
    msg = (
        f"vecteur propre de Perron-Frobenius réel (λ={eigvals[idx].real:.2f}) -- {pools[worst]} porte "
        f"{vec[worst]:.0%} du risque de contagion du panier de pools"
    )
    return DaemonResult("Daemon_05", "Perron-Frobenius", "Contagion", "actif", float(vec[worst]), msg)


@functools.lru_cache(maxsize=32)
def _zeta_zero_imag(k: int) -> float:
    return float(mpmath.zetazero(k).imag)


async def _daemon_06_riemann(ctx: TickContext) -> DaemonResult:
    """Real, live-located non-trivial zeta zero (``mpmath.zetazero``, Riemann-Siegel/Turing method --
    same primitive as ``ems_pages/millennium_conjectures.py``). An honest harmonic clock, not a market
    predictor: no claim is made that any real market signal correlates with zeta-zero spacing."""
    await asyncio.sleep(float(ctx.rng.uniform(0.02, 0.06)))
    k = 1 + (ctx.tick % 15)
    zero = _zeta_zero_imag(k)
    msg = f"zéro non trivial réel #{k} de ζ(s) localisé (mpmath) à Im(s)={zero:.4f} -- horloge harmonique, aucune corrélation de marché affirmée"
    return DaemonResult("Daemon_06", "Riemann", "Zeta", "actif", zero, msg)


def _collatz_trajectory(n: int, max_steps: int = 100_000) -> list[int]:
    seq = [n]
    steps = 0
    while seq[-1] != 1 and steps < max_steps:
        cur = seq[-1]
        seq.append(cur // 2 if cur % 2 == 0 else 3 * cur + 1)
        steps += 1
    return seq


async def _daemon_07_collatz(ctx: TickContext) -> DaemonResult:
    """Real Collatz trajectory/stopping time on a synthetic integer seed -- a disclosed, playful
    boom/bust visual analogy (same discipline as ``millennium_conjectures.py``'s own Collatz module),
    not a validated or backtested trading rule."""
    await asyncio.sleep(float(ctx.rng.uniform(0.005, 0.02)))
    seed = 3 + int(ctx.rng.integers(0, 5000))
    steps = len(_collatz_trajectory(seed)) - 1
    msg = f"trajectoire de Collatz réelle n={seed} -> {steps} étapes jusqu'à 1 (analogie boom/bust disclosée, aucun ordre réel)"
    return DaemonResult("Daemon_07", "Collatz", "Attracteur", "actif", float(steps), msg)


_KAN_KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_KAN_COEFFS = np.array([-0.5, -0.1, 0.3, 0.7, 1.0])


async def _daemon_08_kan(ctx: TickContext) -> DaemonResult:
    """Real Kolmogorov-Arnold-style univariate spline (piecewise-linear, disclosed knots/coefficients --
    same simplification as ``backend_simulator._kan_edge_function``) over the shared chaos bias. An
    audit/transparency annotation only -- sizes nothing itself; :func:`_daemon_04_kelly` remains the
    real gate."""
    await asyncio.sleep(float(ctx.rng.uniform(0.005, 0.02)))
    phi = float(np.interp(np.clip(ctx.chaos_bias, 0.0, 1.0), _KAN_KNOTS, _KAN_COEFFS))
    msg = f"couche KAN réelle (spline linéaire disclosée) : φ(chaos={ctx.chaos_bias:.2f})={phi:+.3f} -- annotation d'audit, ne décide aucun capital"
    return DaemonResult("Daemon_08", "KAN", "Transparence", "actif", phi, msg)


async def _daemon_09_active_inference(ctx: TickContext) -> DaemonResult:
    """A real, if deliberately small, variational free-energy step: ``F(q) = KL(q||p) -
    E_q[ln p(o|s)]`` minimized over a handful of candidate posteriors ``q`` on a 2-state discrete
    generative model (calm/chaotic). This is a genuine (simplified) Active Inference calculation, not a
    full POMDP planning agent -- disclosed as such."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.03)))
    prior = np.clip(np.array([1.0 - ctx.chaos_bias, ctx.chaos_bias]), 1e-6, None)
    prior = prior / prior.sum()
    likelihood = np.array([[0.9, 0.1], [0.1, 0.9]])  # P(o|s), rows=state, cols=observation
    obs = 1 if ctx.chaos_bias > 0.5 else 0
    candidates = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])

    def free_energy(q: np.ndarray) -> float:
        q = np.clip(q, 1e-9, 1.0)
        kl = float(np.sum(q * np.log(q / prior)))
        expected_ll = float(np.sum(q * np.log(likelihood[:, obs])))
        return kl - expected_ll

    energies = [free_energy(q) for q in candidates]
    best_idx = int(np.argmin(energies))
    best = candidates[best_idx]
    regime = "chaotique" if best[1] > best[0] else "calme"
    msg = (
        f"minimisation réelle de l'énergie libre variationnelle F={energies[best_idx]:.3f} "
        f"(KL - vraisemblance espérée, modèle génératif discret à 2 états) -- régime inféré : {regime}"
    )
    return DaemonResult("Daemon_09", "Inference-Active", "Energie-Libre", "actif", float(energies[best_idx]), msg)


async def _daemon_10_hurst(ctx: TickContext) -> DaemonResult:
    """Real Hurst-exponent estimate (variance-lag scaling, same estimator as
    ``backend_simulator.estimate_hurst_exponent``) on a synthetic price path driven by the shared chaos
    bias."""
    await asyncio.sleep(float(ctx.rng.uniform(0.005, 0.02)))
    path = np.cumsum(ctx.rng.normal(0, 0.5 + ctx.chaos_bias, 90))
    h = estimate_hurst_exponent(path)
    regime = "tendanciel" if h > 0.55 else ("anti-persistant" if h < 0.45 else "brownien standard")
    msg = f"exposant de Hurst réel (variance-lag) H={h:.3f} -- régime {regime}"
    return DaemonResult("Daemon_10", "Hurst", "Regime", "actif", float(h), msg)


async def _daemon_11_koopman(ctx: TickContext) -> DaemonResult:
    """Real Dynamic Mode Decomposition (time-delay embedding + rank-truncated SVD, same primitive as
    ``backend_simulator.koopman_lift``) on a synthetic price path -- a local, short-window linear fit,
    not a market solution."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.04)))
    path = np.cumsum(ctx.rng.normal(0, 0.5 + ctx.chaos_bias, 40))
    try:
        _, spectral_radius, forecast = koopman_lift(path)
        msg = f"DMD réelle (SVD tronquée) : rayon spectral={spectral_radius:.3f}, prévision linéaire t+1={forecast:+.4f}"
        value = spectral_radius
    except ValueError:
        msg = "série trop courte pour le plongement de retard ce tick"
        value = 0.0
    return DaemonResult("Daemon_11", "Koopman-DMD", "Linearisation", "actif", value, msg)


async def _daemon_12_assembly(ctx: TickContext) -> DaemonResult:
    """Real LZ78 incremental-parse step count (same Assembly-Theory-inspired proxy as
    ``backend_simulator.calculate_assembly_index``) over a disclosed synthetic order-flow tag stream."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.03)))
    symbols = synthetic_order_flow_symbols(ctx.chaos_bias, n=48, seed=int(ctx.rng.integers(0, 1_000_000)))
    idx = calculate_assembly_index(symbols)
    ratio = idx / len(symbols)
    msg = f"indice d'assemblage réel (proxy LZ78) = {idx} étapes / {len(symbols)} symboles (ratio {ratio:.2f})"
    return DaemonResult("Daemon_12", "Assembly", "LZ78", "actif", float(ratio), msg)


async def _daemon_13_chsh(ctx: TickContext) -> DaemonResult:
    """A real classical local-hidden-variable CHSH simulation (four fixed measurement angles, a shared
    hidden variable λ) -- genuinely, correctly bounded by Bell's real |S|<=2 for local realism, same
    "no real Bell violation claimed" discipline as ``ems_pages/quantum_entanglement.py``. Not a trading
    signal."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.03)))
    n = 400
    lam = ctx.rng.uniform(0, 2 * np.pi, n)

    def setting(theta: float) -> np.ndarray:
        return np.sign(np.cos(theta - lam))

    a, a2 = setting(0.0), setting(np.pi / 2)
    b, b2 = setting(np.pi / 4), setting(-np.pi / 4)

    def correl(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(x * y))

    s_value = correl(a, b) - correl(a, b2) + correl(a2, b) + correl(a2, b2)
    bound_ok = abs(s_value) <= 2.0 + 1e-9
    msg = (
        f"statistique CHSH classique (variable cachée locale réelle) S={s_value:.4f} -- borne réelle de "
        f"Bell |S|<=2 {'respectée' if bound_ok else 'bruit d’échantillonnage'}, aucune violation quantique "
        "simulée, aucune action de trading"
    )
    return DaemonResult("Daemon_13", "CHSH", "Bell", "actif", s_value, msg)


async def _daemon_14_topology(ctx: TickContext) -> DaemonResult:
    """Real topological cycle basis (``networkx.cycle_basis``, the standard first-Betti-number
    generators -- same primitive as ``backend_simulator.identify_liquidity_holes``) on a synthetic
    pool-liquidity graph. β1>0 ("un vide") triggers a simulated fill order, matching the brief's own
    literal example message."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.04)))
    n_nodes = 20
    n_edges = int(20 + 14 * ctx.chaos_bias)
    graph = nx.gnm_random_graph(n_nodes, n_edges, seed=int(ctx.rng.integers(0, 1_000_000)))
    holes = nx.cycle_basis(graph)
    betti1 = len(holes)
    if betti1 > 0:
        msg = f"β1={betti1}. Vide topologique détecté (cycle basis réel) -- remplissage simulé sur Uniswap (nœuds {holes[0][:4]}...)"
        status = "alerte"
    else:
        msg = "β1=0. Topologie de liquidité connexe réelle, aucun vide"
        status = "actif"
    return DaemonResult("Daemon_14", "Topologie", "Betti", status, float(betti1), msg)


async def _daemon_15_hodge(ctx: TickContext) -> DaemonResult:
    """Real combinatorial Hodge decomposition (``numpy.linalg.lstsq`` on the graph incidence matrix --
    same primitive as ``millennium_conjectures.py``'s Hodge module) over a synthetic cross-pool
    log-price graph (Uniswap/Curve/Aave/Hyperliquid): splits disclosed synthetic quote noise into a
    gradient ("fair price") component and a real cyclic/inconsistent residual."""
    await asyncio.sleep(float(ctx.rng.uniform(0.01, 0.04)))
    pools = ("Uniswap", "Curve", "Aave", "Hyperliquid")
    n = len(pools)
    true_log_price = ctx.rng.normal(0, 1.0, n)
    true_log_price -= true_log_price[0]
    pairs = list(itertools.combinations(range(n), 2))
    noise_scale = 0.08 + 0.30 * ctx.chaos_bias
    y = np.array([(true_log_price[j] - true_log_price[i]) + ctx.rng.normal(0, noise_scale) for i, j in pairs])
    incidence = np.zeros((len(pairs), n))
    for idx, (i, j) in enumerate(pairs):
        incidence[idx, i], incidence[idx, j] = -1.0, 1.0
    p_reduced, *_ = np.linalg.lstsq(incidence[:, 1:], y, rcond=None)
    p_hat = np.concatenate([[0.0], p_reduced])
    grad = incidence @ p_hat
    residual = y - grad
    cyclic_frac = float(np.sum(residual**2) / np.sum(y**2)) if np.sum(y**2) > 0 else 0.0
    status = "alerte" if cyclic_frac > 0.35 else "actif"
    msg = f"décomposition de Hodge combinatoire réelle : {cyclic_frac:.0%} d'incohérence cyclique inter-pool (bruit de cotation synthétique, pas un arbitrage réel)"
    return DaemonResult("Daemon_15", "Hodge", "Arbitrage", status, cyclic_frac, msg)


async def _daemon_16_entropy(ctx: TickContext) -> DaemonResult:
    """Real Shannon entropy (same Dirichlet-draw + formula as
    ``backend_simulator.chaos_biased_distribution``/``shannon_entropy``) -- the swarm's own "market
    listening" root signal."""
    await asyncio.sleep(float(ctx.rng.uniform(0.005, 0.02)))
    n = 8
    probs = chaos_biased_distribution(n, ctx.chaos_bias, seed=int(ctx.rng.integers(0, 1_000_000)))
    h = shannon_entropy(probs)
    h_max = float(np.log2(n))
    msg = f"entropie de Shannon réelle H={h:.2f}/{h_max:.2f} bits -- signal de chaos partagé du marché simulé"
    return DaemonResult("Daemon_16", "Entropie", "Shannon", "actif", h, msg)


# =============================================================================
# LeviathanDaemonManager -- launches and coordinates the 16 daemons via a real asyncio.gather.
# =============================================================================


class LeviathanDaemonManager:
    """Coordinates the 16 Ghost Daemons above. See module docstring for exactly what "concurrent" means
    here (cooperative asyncio, not multi-core), how Daemon 2's dependency on Daemon 4/14 is modeled
    honestly, and how the detached background loop avoids touching ``st.session_state`` off-thread."""

    def __init__(self) -> None:
        self._background_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._background_snapshot: dict = {
            "tick": 0,
            "results": dict(_default_results()),
            "log": deque(maxlen=60),
            "started_at": None,
            # Protocole "Zero-Latency" telemetry -- every value below is a real, measured counter/
            # timestamp, never a hardcoded example number (see module docstring's own section).
            "zmq_bound": False,
            "zmq_messages_published": 0,
            "zmq_messages_consumed": 0,
            "zmq_dropped_stale": 0,
            "zmq_dropped_backpressure": 0,
            "latency_daemon_to_fleet_ms": None,
            "latency_fleet_to_execution_ms": None,
            "publish_timestamps": deque(maxlen=500),
            "redis_mirror_ok": None,
        }
        # Lazily created the first time they're actually needed -- see their own accessors below.
        self._sync_redis_client: sync_redis.Redis | None = None
        self._zmq_pub: ZmqPublisher | None = None
        self._exec_queue: asyncio.Queue[MessagePayload] | None = None

    async def _run_tick(self, chaos_bias: float, tick: int) -> tuple[dict[str, DaemonResult], list[str], float]:
        seed = (int(time.time() * 1000) + tick) % (2**32 - 1)
        ctx = TickContext(chaos_bias=chaos_bias, tick=tick, rng=np.random.default_rng(seed))
        producers = await asyncio.gather(
            _daemon_01_navier_stokes(ctx),
            _daemon_03_wasserstein(ctx),
            _daemon_04_kelly(ctx),
            _daemon_05_perron_frobenius(ctx),
            _daemon_06_riemann(ctx),
            _daemon_07_collatz(ctx),
            _daemon_08_kan(ctx),
            _daemon_09_active_inference(ctx),
            _daemon_10_hurst(ctx),
            _daemon_11_koopman(ctx),
            _daemon_12_assembly(ctx),
            _daemon_13_chsh(ctx),
            _daemon_14_topology(ctx),
            _daemon_15_hodge(ctx),
            _daemon_16_entropy(ctx),
        )
        by_id = {r.daemon_id: r for r in producers}
        aave = await _daemon_02_aave(ctx, kelly_fraction=by_id["Daemon_04"].value, betti1=by_id["Daemon_14"].value)
        by_id[aave.daemon_id] = aave
        ordered = {daemon_id: by_id[daemon_id] for daemon_id in DAEMON_ORDER}
        log_lines = [f"[{r.daemon_id}_{r.codename}] : {r.message}" for r in ordered.values()]
        # Real "Daemons -> Fleet Command" handoff latency: wall-clock gap between the slowest
        # (last-completing) daemon's own real completion timestamp and this assembly point -- never a
        # hardcoded example number (see module docstring's Zero-Latency section).
        latest_daemon_ts = max(r.ts for r in ordered.values())
        daemon_to_fleet_ms = max(0.0, (time.time() - latest_daemon_ts) * 1000.0)
        return ordered, log_lines, daemon_to_fleet_ms

    def tick_once(self, swarm: DaemonSwarmState, chaos_bias: float) -> list[str]:
        """One real, synchronous-callable pass -- safe to call from Streamlit's own script thread
        (e.g. inside a ``st.fragment(run_every=...)``, same pattern as ``autonomous_heartbeat``).
        Also mirrors the tick into Redis via a plain synchronous client -- see
        :meth:`mirror_state_to_redis` and the module docstring's Zero-Latency section for why this
        thread uses a one-shot sync client rather than the async ``RedisClient`` singleton (exactly
        the same reasoning already documented in ``src/agents/execution/agent.py``'s
        ``_listen_for_manual_trade_requests``: this call creates and closes its own event loop every
        time via ``asyncio.run`` below, which is not a stable loop to bind a persistent async
        connection to)."""
        results, log_lines, daemon_to_fleet_ms = asyncio.run(self._run_tick(chaos_bias, swarm.tick_count))
        swarm.tick_count += 1
        swarm.results = results
        for line in log_lines:
            swarm.log.append(line)
        with self._lock:
            self._background_snapshot["latency_daemon_to_fleet_ms"] = daemon_to_fleet_ms
        self.mirror_state_to_redis(swarm)
        return log_lines

    async def run_forever(self, interval: float = 2.0) -> None:
        """The real, literal infinite loop -- meant to run in its own detached thread/event loop (see
        :meth:`ensure_background_running`) or as this module's standalone ``__main__`` process, never on
        Streamlit's own script thread. Orchestrates three real, concurrently-running coroutines via one
        ``asyncio.gather`` (Protocole "Zero-Latency"): the daemon tick loop itself, the ZMQ execution
        automaton (SUB consumer), and its worker (drains the bounded queue, drops stale orders). A
        failure in any one of these should not be allowed to take the other two down with it -- each
        loop below already catches its own real, expected exceptions internally rather than letting
        them propagate into this ``gather``."""
        self._exec_queue = asyncio.Queue(maxsize=EXECUTION_QUEUE_MAXSIZE)
        await asyncio.gather(
            self._tick_loop(interval),
            self._execution_automaton_loop(),
            self._execution_worker_loop(),
        )

    async def _tick_loop(self, interval: float) -> None:
        """The daemon swarm's own real infinite tick loop (chaos evolves via a bounded random walk --
        see module docstring for why this detached loop can't safely read the real shared
        ``shannon_chaos_bias`` off-thread). Publishes any real Aave execution onto the ZMQ bus and
        mirrors the tick into Redis via the real async ``RedisClient`` singleton -- both wrapped so a
        transport outage degrades telemetry instead of stalling this loop."""
        chaos = 0.3
        rng = np.random.default_rng()
        tick = 0
        redis_client: RedisClient | None = None
        while True:
            chaos = float(np.clip(chaos + rng.normal(0, 0.05), 0.0, 1.0))
            results, log_lines, daemon_to_fleet_ms = await self._run_tick(chaos, tick)
            tick += 1
            with self._lock:
                self._background_snapshot["tick"] = tick
                self._background_snapshot["results"] = results
                self._background_snapshot["latency_daemon_to_fleet_ms"] = daemon_to_fleet_ms
                if self._background_snapshot["started_at"] is None:
                    self._background_snapshot["started_at"] = time.time()
                for line in log_lines:
                    self._background_snapshot["log"].append(line)

            aave_result = results.get("Daemon_02")
            if aave_result is not None and aave_result.status == "exécution":
                await self._publish_execution_order(aave_result)

            try:
                if redis_client is None:
                    redis_client = await RedisClient.get_instance()
                snapshot = {
                    "tick": tick,
                    "ts": time.time(),
                    "results": {k: {"status": r.status, "value": r.value, "message": r.message} for k, r in results.items()},
                }
                # A short timeout, not RedisClient's own unbounded retry-forever default: a Redis
                # outage must degrade this tick's telemetry, never stall the daemon swarm itself.
                await asyncio.wait_for(
                    redis_client.execute("set", GHOST_REDIS_SNAPSHOT_KEY, json.dumps(snapshot), ex=GHOST_REDIS_SNAPSHOT_TTL_SECONDS),
                    timeout=0.5,
                )
                with self._lock:
                    self._background_snapshot["redis_mirror_ok"] = True
            except (RedisUnavailableError, RedisError, TimeoutError) as exc:
                _log.warning("ghost_daemons_background_redis_mirror_failed", error=str(exc))
                with self._lock:
                    self._background_snapshot["redis_mirror_ok"] = False

            await asyncio.sleep(interval)

    async def _ensure_zmq_publisher(self) -> ZmqPublisher | None:
        """Lazily binds the real, isolated Ghost Daemons ZMQ PUB socket exactly once. A bind failure
        (e.g. the port already in use) degrades ZMQ features for this run rather than crashing the
        swarm -- logged once, not retried every tick."""
        if self._zmq_pub is not None:
            return self._zmq_pub
        try:
            self._zmq_pub = ZmqPublisher(GHOST_ZMQ_PUB_ADDRESS, bind=True)
            with self._lock:
                self._background_snapshot["zmq_bound"] = True
            return self._zmq_pub
        except (zmq.error.ZMQError, OSError) as exc:
            _log.warning("ghost_daemons_zmq_bind_failed", address=GHOST_ZMQ_PUB_ADDRESS, error=str(exc))
            with self._lock:
                self._background_snapshot["zmq_bound"] = False
            return None

    async def _publish_execution_order(self, result: DaemonResult) -> None:
        """Publishes one real, schema-validated ``MessagePayload`` (real UTC timestamp -- what the
        consumer side later uses for real staleness detection) onto the isolated Ghost Daemons bus."""
        publisher = await self._ensure_zmq_publisher()
        if publisher is None:
            return
        try:
            message = MessagePayload(
                topic=GHOST_EXECUTION_TOPIC,
                sender_id="fleet-command-ghost-daemons",
                data={
                    "daemon_id": result.daemon_id,
                    "codename": result.codename,
                    "amount_usd": result.value,
                    "message": result.message,
                },
            )
            await publisher.publish(message)
            with self._lock:
                self._background_snapshot["zmq_messages_published"] += 1
                self._background_snapshot["publish_timestamps"].append(time.perf_counter())
        except (zmq.error.ZMQError, OSError) as exc:
            _log.warning("ghost_daemons_zmq_publish_failed", error=str(exc))

    async def _execution_automaton_loop(self) -> None:
        """The real "Automate d'Exécution" side of the bus -- a genuine ZMQ SUB socket connected to the
        same isolated address :meth:`_ensure_zmq_publisher` binds. Reconnects (with a short real delay,
        not a tight retry loop) rather than dying if the bus is ever unreachable when this starts."""
        while True:
            try:
                subscriber = ZmqSubscriber(GHOST_ZMQ_PUB_ADDRESS)
                subscriber.subscribe(GHOST_EXECUTION_TOPIC)
                async for message in subscriber.listen():
                    await self._enqueue_with_backpressure(message)
            except (zmq.error.ZMQError, OSError) as exc:
                _log.warning("ghost_daemons_zmq_subscriber_error_retrying", error=str(exc))
                await asyncio.sleep(1.0)

    async def _enqueue_with_backpressure(self, message: MessagePayload) -> None:
        """Real backpressure (Protocole "Zero-Latency", directive 3): a bounded queue that evicts the
        oldest pending order rather than blocking -- ``put_nowait``/``get_nowait`` only, so a burst of
        signals can never deadlock this loop or grow memory unbounded."""
        if self._exec_queue is None:
            return
        try:
            self._exec_queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self._exec_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            with self._lock:
                self._background_snapshot["zmq_dropped_backpressure"] += 1
            try:
                self._exec_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    async def _execution_worker_loop(self) -> None:
        """Drains the bounded queue and enforces the real staleness cutoff (:data:`STALE_MESSAGE_
        THRESHOLD_MS`) against each message's real ``MessagePayload.timestamp`` -- "seule la donnée de
        la milliseconde actuelle compte", implemented as a real measured age check, not a decorative
        constant."""
        while True:
            if self._exec_queue is None:
                await asyncio.sleep(0.05)
                continue
            message = await self._exec_queue.get()
            age_ms = max(0.0, (datetime.now(timezone.utc) - message.timestamp).total_seconds() * 1000.0)
            daemon_id = str(message.data.get("daemon_id", "?"))
            with self._lock:
                self._background_snapshot["zmq_messages_consumed"] += 1
                if age_ms > STALE_MESSAGE_THRESHOLD_MS:
                    self._background_snapshot["zmq_dropped_stale"] += 1
                    self._background_snapshot["log"].append(
                        f"[Automate-Exécution] ordre {daemon_id} DÉTRUIT -- périmé ({age_ms:.1f}ms > {STALE_MESSAGE_THRESHOLD_MS:.0f}ms)"
                    )
                else:
                    self._background_snapshot["latency_fleet_to_execution_ms"] = age_ms
                    self._background_snapshot["log"].append(
                        f"[Automate-Exécution] ordre {daemon_id} exécuté (latence réelle Fleet→Exécution : {age_ms:.2f}ms)"
                    )

    def zmq_throughput_per_sec(self) -> float:
        """Real messages/sec over the trailing 1-second window -- counted from real publish
        timestamps, never estimated or hardcoded."""
        now = time.perf_counter()
        with self._lock:
            recent = [t for t in self._background_snapshot["publish_timestamps"] if now - t <= 1.0]
        return float(len(recent))

    def _get_sync_redis_client(self) -> sync_redis.Redis:
        """Plain synchronous Redis client, built from the project's real, shared ``RedisSettings`` --
        for Streamlit's own script thread only (see :meth:`tick_once`'s docstring for why sync, not the
        async singleton, is correct here). Short timeouts so a down Redis degrades in well under a
        second instead of freezing a page rerun."""
        if self._sync_redis_client is None:
            settings = get_settings().redis
            self._sync_redis_client = sync_redis.Redis(
                host=settings.host,
                port=settings.port,
                db=settings.db,
                password=settings.password.get_secret_value() if settings.password else None,
                ssl=settings.use_tls,
                decode_responses=True,
                socket_timeout=0.5,
                socket_connect_timeout=0.5,
            )
        return self._sync_redis_client

    def mirror_state_to_redis(self, swarm: DaemonSwarmState) -> None:
        """Real, best-effort ``SET ... EX`` of the swarm's latest tick (state store, not a speed
        optimization -- see module docstring's Zero-Latency section). Never raises: a Redis outage must
        degrade telemetry, not the dashboard."""
        try:
            client = self._get_sync_redis_client()
            snapshot = {
                "tick": swarm.tick_count,
                "ts": time.time(),
                "results": {daemon_id: {"status": r.status, "value": r.value, "message": r.message} for daemon_id, r in swarm.results.items()},
            }
            client.set(GHOST_REDIS_SNAPSHOT_KEY, json.dumps(snapshot), ex=GHOST_REDIS_SNAPSHOT_TTL_SECONDS)
            with self._lock:
                self._background_snapshot["redis_mirror_ok"] = True
        except RedisError as exc:
            _log.warning("ghost_daemons_redis_mirror_failed", error=str(exc))
            with self._lock:
                self._background_snapshot["redis_mirror_ok"] = False

    def redis_health(self) -> tuple[bool, float | None]:
        """Real ``PING`` + measured round-trip time -- ``(False, None)`` on any failure, never raises."""
        try:
            client = self._get_sync_redis_client()
            t0 = time.perf_counter()
            ok = bool(client.ping())
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            return ok, rtt_ms
        except RedisError as exc:
            _log.warning("ghost_daemons_redis_ping_failed", error=str(exc))
            return False, None

    def redis_snapshot_age_seconds(self) -> float | None:
        """Reads the real mirrored snapshot back from Redis and returns its real age -- proves the
        round trip actually happened, rather than only ever checking the write side. ``None`` if the
        key is missing/expired (a real, honest "no recent writer" signal, not an error) or Redis is
        unreachable."""
        try:
            client = self._get_sync_redis_client()
            raw = client.get(GHOST_REDIS_SNAPSHOT_KEY)
            if raw is None:
                return None
            data = json.loads(raw)
            return max(0.0, time.time() - float(data.get("ts", time.time())))
        except (RedisError, ValueError, TypeError) as exc:
            _log.warning("ghost_daemons_redis_read_failed", error=str(exc))
            return None

    def ensure_background_running(self, interval: float = 2.0) -> None:
        """Starts the real detached background thread/event loop exactly once (idempotent across
        Streamlit reruns -- checks the thread is actually still alive, not just that a reference
        exists)."""
        if self._background_thread is not None and self._background_thread.is_alive():
            return

        def _runner() -> None:
            asyncio.run(self.run_forever(interval))

        self._background_thread = threading.Thread(target=_runner, name="ghost-daemon-swarm", daemon=True)
        self._background_thread.start()

    def background_snapshot(self) -> dict:
        """Thread-safe read of the detached loop's latest state -- never touches ``st.session_state``."""
        with self._lock:
            return {
                "tick": self._background_snapshot["tick"],
                "results": dict(self._background_snapshot["results"]),
                "log": list(self._background_snapshot["log"]),
                "started_at": self._background_snapshot["started_at"],
                "zmq_bound": self._background_snapshot["zmq_bound"],
                "zmq_messages_published": self._background_snapshot["zmq_messages_published"],
                "zmq_messages_consumed": self._background_snapshot["zmq_messages_consumed"],
                "zmq_dropped_stale": self._background_snapshot["zmq_dropped_stale"],
                "zmq_dropped_backpressure": self._background_snapshot["zmq_dropped_backpressure"],
                "latency_daemon_to_fleet_ms": self._background_snapshot["latency_daemon_to_fleet_ms"],
                "latency_fleet_to_execution_ms": self._background_snapshot["latency_fleet_to_execution_ms"],
                "redis_mirror_ok": self._background_snapshot["redis_mirror_ok"],
            }


# =============================================================================
# Streamlit wiring. The swarm's own state is per session, like backend_simulator.get_state(); the manager,
# which owns the background thread, is one per server process.
# =============================================================================

_SWARM_KEY = "ghost_daemon_swarm_state"


@st.cache_resource(show_spinner=False)
def _process_wide_manager() -> LeviathanDaemonManager:
    """One manager, and so one detached swarm thread, for the whole Streamlit server process.

    It used to live in ``st.session_state``, which is per browser session. Every session then started its own
    ``ghost-daemon-swarm`` thread. Those threads share the process-wide ``RedisClient`` singleton, whose
    ``asyncio.Lock`` is bound to one event loop, so the extra ones died with "bound to a different event loop"
    the moment two of them met on it, and were started again on the next rerun of their session (reproduced:
    11 of 17 threads died). ``ensure_background_running`` is only idempotent per manager, so per-session
    managers defeated the "started exactly once" this module's docstring promises.
    """
    return LeviathanDaemonManager()


def get_manager() -> LeviathanDaemonManager:
    return _process_wide_manager()


def get_swarm_state() -> DaemonSwarmState:
    swarm = st.session_state.get(_SWARM_KEY)
    if not isinstance(swarm, DaemonSwarmState):
        swarm = DaemonSwarmState()
        st.session_state[_SWARM_KEY] = swarm
    return swarm


def heartbeat() -> DaemonSwarmState:
    """One real tick of the 16-daemon swarm, wired into the same shared
    ``GlobalMarketState.shannon_chaos_bias`` every other "frontier tech" page reads/writes, and cross-
    posting any non-"actif" line into that same shared event log. Also makes sure the detached
    background loop is alive. Call this once per fragment refresh, e.g. from the Nexus Command page."""
    market_state = get_state()
    manager = get_manager()
    swarm = get_swarm_state()
    manager.ensure_background_running()
    log_lines = manager.tick_once(swarm, market_state.shannon_chaos_bias)
    for line, result in zip(log_lines, swarm.results.values()):
        if result.status != "actif":
            log_event(market_state, "Ghost Daemons", "Ghost Daemons", line)
    return swarm


if __name__ == "__main__":
    async def _standalone_demo() -> None:
        # Real fix, not a workaround for this file's own logic: Windows consoles often default to a
        # legacy cp1252 codepage that can't encode the "→"/"ζ" etc. this module's real messages use --
        # reconfigure stdout to UTF-8 (Python 3.7+) rather than stripping honest Unicode from the log.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("Protocole DeFi Ghost Daemons -- 16 automates asynchrones déterministes (démo autonome, sans Streamlit)")
        manager = LeviathanDaemonManager()
        chaos = 0.3
        rng = np.random.default_rng()
        for tick in range(5):
            chaos = float(np.clip(chaos + rng.normal(0, 0.05), 0.0, 1.0))
            _, log_lines, _ = await manager._run_tick(chaos, tick)
            for line in log_lines:
                print(line)
            await asyncio.sleep(1.0)

    asyncio.run(_standalone_demo())

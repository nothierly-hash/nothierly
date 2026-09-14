"""Fleet Command Nexus -- l'écran d'audit et de télémétrie temps réel (page d'entrée du système).

**Ce que cette page est, précisément** : une nouvelle interface d'observabilité, distincte de
``leviathan_nexus_command.py`` ("Nexus Command Center", qui montre le graphe de dépendance des 10
modules "frontier tech" du Protocole Léviathan). Celle-ci montre la couche d'infrastructure la plus
basse de l'EMS -- les 16 automates du Protocole "DeFi Ghost Daemons" (``core_daemons.py``), le canal
Redis/ZeroMQ réel qui les relie, et un arbre de consensus visuel centré sur un nœud "FLEET COMMAND
ARBITER" -- pour qu'un auditeur puisse vérifier d'un coup d'œil que le calcul mathématique, le
transport réseau et le veto de risque fonctionnent ensemble sans conflit. Aucune nouvelle donnée
n'est inventée ici : chaque nombre affiché est lu depuis les mêmes primitives réelles déjà vérifiées
dans ``core_daemons.py`` et ``backend_simulator.py``.

**Purge lexicale** : comme ``core_daemons.py``, cette page dit "Daemon"/"Automate", jamais
"Agent"/"Intelligence Artificielle" -- même règle, même portée limitée à ce sous-système (voir le
docstring de ``core_daemons.py`` pour pourquoi ``src/agents/*`` n'est pas renommé ailleurs).

**Ce qui est corrigé par rapport au brief d'origine, honnêtement** (voir [[feedback-nothierly-honest-
engineering]] dans la mémoire de ce projet -- traduire le grandiose en ingénierie réelle plutôt que de
fabriquer un chiffre) :

- **Module 1 (latence de calcul)** : le brief demandait un chiffre fixe recopié ("0.4 ms") pour les 16
  automates. Un chiffre fixe ne mesurerait rien -- ce panneau affiche à la place la **fraîcheur réelle**
  de chaque résultat (``time.time() - DaemonResult.ts``, en millisecondes), un nombre qui bouge
  vraiment tick après tick.
- **Module 1 (légende des pastilles)** : cette page utilise sa propre légende, celle demandée par ce
  brief (🟢 vert = calcul nominal, 🟡 jaune = alerte mathématique), différente de la légende rouge
  utilisée sur la page Nexus Command Center -- une pastille rouge est réservée ici au **veto de risque
  système** du Module 3 (voir plus bas), pas à l'alerte d'un seul automate.
- **Module 2 (journal Redis/ZMQ)** : le brief demandait des lignes littérales "[REDIS PULL] Daemon_14
  signale..." -- mais aucun automate ne fait de "pull" Redis par tick ; le vrai flux est (a) 15
  automates calculent en concurrence réelle (``asyncio.gather``) puis publient leur résultat en mémoire
  vers Fleet Command, et (b) seul un ordre Aave réellement exécuté est publié sur le vrai bus ZeroMQ
  isolé. Ce panneau étiquette donc honnêtement deux flux réels distincts (Daemons -> Fleet Command,
  Fleet Command -> Exécution via ZMQ) plutôt que d'inventer un luxueux mais faux "[REDIS PULL]" par
  automate.
- **Module 4 (télémétrie RPC Web3)** : ``core_daemons.py`` dit explicitement que sa signature "Web3"
  est un mock (``mock_web3_sign_transaction``) -- aucune vraie clé, aucun vrai point RPC, aucun vrai
  gas nulle part dans ce process. Afficher "Connecté" sur un vrai Uniswap V3/Aave/Hyperliquid serait un
  mensonge d'infrastructure, le genre précis que ``GlobalMarketState`` (``backend_simulator.py``)
  s'engage déjà à ne jamais afficher. Ce module montre donc des pings **explicitement disclosés comme
  simulés** (formule réelle, pilotée par le vrai ``shannon_chaos_bias`` partagé pour qu'ils bougent
  vraiment), jamais un badge "Connecté" à une infrastructure qui n'existe pas dans ce projet.
- **Module 7 (Courbe d'Impact d'Exécution Optimale, ajouté 2026-09-13, "Le Prompt Architecte")** :
  télémétrie réelle de ``ExecutionAgent.compute_optimal_trajectory`` (Almgren-Chriss) -- trace la
  vraie courbe d'impact temporaire versus impact permanent du dernier plan d'exécution calculé pour
  chaque instrument, lue depuis Redis (``optimal_execution_schedule:{instrument}``). Note honnête :
  ce module affiche un **plan calculé**, pas un ordre réellement tranché et exécuté -- voir le
  docstring de ``compute_optimal_trajectory`` sur la frontière de portée disclosée (le découpage réel
  d'un ordre dans le temps n'est pas encore câblé dans le chemin d'exécution réel de cet agent).
- **Module 8 (Greeks de Portefeuille & Couverture Multi-Instruments, ajouté 2026-09-13, "Options &
  Multi-Instrument Hedging")** : télémétrie réelle de ``ExecutionAgent.compute_portfolio_greeks``
  (``core.schemas.PORTFOLIO_GREEKS_REDIS_KEY``, une seule clé Redis niveau-portefeuille, sans
  suffixe par instrument -- les Greeks sont additifs par construction). Note honnête, disclosée dans
  ce module même : contrairement aux Modules 6/7, rien dans la boucle de tick réelle n'appelle
  ``compute_portfolio_greeks`` automatiquement pour l'instant -- ce panneau reste vide tant qu'un
  opérateur (ou un futur script périodique) n'a pas réellement invoqué cette méthode au moins une
  fois ; ce n'est pas un bug, c'est la même frontière de portée honnête que Module 7 (un calcul réel
  et utilisable, pas encore câblé à un déclenchement automatique). Rappel de la frontière
  d'exécution réelle (voir le docstring de ``options_hedging_manager.py``) : toute position
  d'option affichée ici est nécessairement simulée (paper-fill) -- ce projet n'a aucun vrai courtier
  d'options ; seul son prix Black-Scholes est réel.
- **Module 9 (Smart Order Router -- Historique Réel de Routage, ajouté 2026-09-14, audit des 9
  pages "lois mathématiques")** : le brief demandait de "lier" 9 pages frontière (Macro-
  Entanglement, Tensor Train Optimizer, Apex Leviathan, Wolfram Hypergraph, Dynamic Kelly
  Optimizer, Ohm's Law Financial Routing, Quantitative Supremacy, Absolute Quant Frontier,
  Millennium Conjectures) à ce module de production. Audit fait honnêtement, page par page, avant
  de câbler quoi que ce soit (voir la demande explicite de l'utilisateur de séparer le réel du
  simulé) : **8 des 9 pages exécutent de vraies maths sur des données synthétiques/illustratives,
  par conception délibérée et déjà disclosée** (ex. Tensor Train Optimizer décompose un tenseur
  synthétique lisse ; Macro-Entanglement calcule un vrai CHSH sur un modèle classique simulé) --
  les câbler dans la vraie télémétrie de production ferait entrer une donnée simulée dans un
  système réel, exactement ce que ce projet refuse partout ailleurs (voir le docstring de
  ``core_daemons.py``). Aucune de ces 8 n'est donc câblée ici -- refusé honnêtement, pas oublié.
  La 9ème piste, ``src/execution/smart_order_router.py`` (routage multi-venues crypto réel, appelé
  depuis ``liquidity_routing.py``), est la seule des 9 à être réellement câblable : ce module a
  déjà sa propre intégration Kill Switch/``safe_financial_call``/MOCK_MODE réelle, et persiste
  chaque décision de routage **réellement exécutée** (jamais un simple aperçu) sur
  ``core.schemas.SOR_ROUTING_LOG_REDIS_KEY``. Module 9 lit cette même clé réelle, en lecture seule
  -- Fleet Command Nexus n'avait jusqu'ici aucune visibilité sur cette autre filière d'exécution
  réelle du projet. Note honnête : ``build_allocation_plan`` lui-même (le calcul d'aperçu, sans
  exécution) nécessiterait des connexions réelles à 4 exchanges crypto que ``ExecutionAgent`` ne
  détient pas aujourd'hui -- l'ajouter à la Porte 7 par tick exigerait une vraie décision
  d'architecture (faire tenir ces connexions à ``ExecutionAgent``), pas une simple ligne de
  télémétrie ; non fait ici, disclosé plutôt que bricolé.
- **Module 6 (Portes d'Exécution, ajouté 2026-09-13, "Protocole 7-Gates Convergence" puis étendu le
  même jour par "Protocole Stochastic Core")** : distinct de tout ce qui précède -- Modules 1-4 sont le
  swarm mathématique **simulé** ``core_daemons.py``/DeFi Ghost Daemons ; Module 6 est la **vraie**
  télémétrie du moteur d'exécution de production (``src/agents/execution/agent.py``, nom d'affichage
  ``Fleet_Command_Arbiter`` -- les vraies classes ne sont pas renommées, voir le docstring de ce module
  de production pour pourquoi). Initialement 7 portes ; une 8ème ("StochasticCore" -- Hawkes, Merton
  jump-diffusion, Ornstein-Uhlenbeck, impact de prix, voir le docstring de ``core/quant_math.py``) a
  été insérée en position 7, avec Control State décalée en position 8 pour rester la toute dernière
  porte avant un ordre réel -- disclosé dans le docstring de ``src/agents/execution/agent.py``, pas
  silencieusement réordonné. Le brief demandait une diffusion ZMQ des portes -- cette page ne s'abonne
  jamais directement à ZMQ (même règle que Module 2 ci-dessous), donc la télémétrie est pontée via
  Redis, exactement comme tout le reste de cette page. Le brief demandait aussi de voir les portes
  "passer au vert" avant un log de type ``CREATE_MARKET_ORDER`` -- honnêtement, le pipeline réel
  s'exécute en quelques millisecondes (calculs en mémoire + lectures Redis), trop rapide pour qu'un
  humain le regarde s'allumer porte par porte en temps réel ; ce module montre donc le vrai verdict, le
  plus récent, de chaque porte pour la dernière tentative d'entrée réellement évaluée -- puis, seulement
  quand toutes les portes réelles sont vertes, la ligne ``[CREATE_MARKET_ORDER]`` apparaît, honnêtement
  étiquetée SIMULÉ quand ``final_action`` est ``SHADOW_ORDER``/``MOCK_ORDER`` plutôt que ``LIVE_ORDER``.

**Placement** : ajoutée comme page par défaut de l'EMS (``nothierly_ems.py``) -- le brief la décrit
comme "l'écran de contrôle absolu", celui sur lequel un auditeur doit atterrir en premier.

**Liaison avec Agent Fleet Command (``agent_fleet.py``), réelle mais volontairement à sens unique
côté contrôle** : ce brief demande que les deux pages "travaillent ensemble" -- Module 5 ci-dessous lit
donc les vraies clés Redis ``heartbeat:{agent_id}``/``agent_control_state:{agent_id}`` que
``agent_fleet.py`` utilise déjà (même mécanisme réel, pas une copie), et ``agent_fleet.py`` lit en
retour le vrai ``background_snapshot()``/``redis_health()`` de ce module -- une vraie liaison
bidirectionnelle d'observation, avec un lien de navigation réel (``st.page_link``) dans les deux sens.
Ce qui n'est délibérément **pas** fait : le veto de risque simulé du Module 3 (déclenché par des
maths sur données synthétiques) ne publie et n'écrit jamais sur les vrais canaux de contrôle
(``AGENT_CONTROL_CHANNEL_PREFIX``/``AGENT_CONTROL_STATE_REDIS_KEY_PREFIX``) qui pilotent réellement
``execution-agent`` et les autres agents de production -- cela romprait la règle déjà établie dans tout
ce projet ("jamais une vraie action déclenchée par une donnée simulée", voir le docstring de
``core_daemons.py``). Module 5 est donc strictement en lecture : aucun bouton LIVE/SHADOW/OFF ici,
ceux-là restent uniquement sur ``agent_fleet.py``, la seule page autorisée à écrire sur ce canal réel.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.base_agent import AgentControlState
from core.schemas import (
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    HEARTBEAT_REDIS_KEY_PREFIX,
    PORTFOLIO_GREEKS_REDIS_KEY,
    SOR_ROUTING_LOG_REDIS_KEY,
)
from src.dashboard.backend_simulator import autonomous_heartbeat, get_state
from src.dashboard.core_daemons import (
    DAEMON_META,
    DAEMON_ORDER,
    EXECUTION_QUEUE_MAXSIZE,
    GHOST_ZMQ_PUB_ADDRESS,
    STALE_MESSAGE_THRESHOLD_MS,
    get_manager,
    get_swarm_state,
    heartbeat as ghost_daemon_heartbeat,
)
from src.dashboard.ems_pages._shared import get_redis_client, safe_get, safe_get_json, safe_keys, safe_lrange

# =============================================================================
# Institutional Glassmorphism -- palette + CSS
# =============================================================================

_BLACK = "#050506"
_ANTHRACITE = "#0c0c11"
_CYAN = "#00e5ff"          # réseau
_GOLD = "#e8c468"          # capital
_BLOOD = "#c81e3a"         # veto de risque
_GREEN = "#3ddc84"         # calcul nominal
_YELLOW = "#ffce54"        # alerte mathématique
_GRAY = "#4a4a58"          # veille
_TEXT_DIM = "#9098a8"

# Légende propre à cette page (Module 1) -- distincte de STATUS_LED_COLOR (core_daemons.py) par
# demande explicite du brief : vert=nominal, jaune=alerte mathématique. "exécution"/"veille" reprennent
# des teintes cohérentes avec le reste de l'EMS.
_LED_COLOR: dict[str, str] = {"veille": _GRAY, "actif": _GREEN, "alerte": _YELLOW, "exécution": _CYAN}
_LED_LABEL: dict[str, str] = {"veille": "VEILLE", "actif": "NOMINAL", "alerte": "ALERTE MATH.", "exécution": "EXÉCUTION"}

# Étiquette humaine de la valeur mathématique de chaque automate -- purement cosmétique (le nombre
# affiché reste le vrai DaemonResult.value calculé par core_daemons.py), pour donner à chaque bloc de
# la matrice le style "Vélocité Fluide: 0.84" demandé par le brief.
_DAEMON_VALUE_LABEL: dict[str, str] = {
    "Daemon_01": "Vorticité de pointe (rad/s)",
    "Daemon_02": "Montant emprunté simulé ($)",
    "Daemon_03": "Distance de Wasserstein",
    "Daemon_04": "Fraction de Kelly (f*)",
    "Daemon_05": "Concentration de contagion",
    "Daemon_06": "Zéro de Riemann Im(s)",
    "Daemon_07": "Étapes de Collatz",
    "Daemon_08": "Score KAN (φ)",
    "Daemon_09": "Énergie libre F(q)",
    "Daemon_10": "Exposant de Hurst (H)",
    "Daemon_11": "Rayon spectral (Koopman)",
    "Daemon_12": "Ratio LZ78 (Assembly)",
    "Daemon_13": "Statistique CHSH (S)",
    "Daemon_14": "β₁ (cycles topologiques)",
    "Daemon_15": "Incohérence cyclique (Hodge)",
    "Daemon_16": "Entropie de Shannon (H)",
}

# Veto de risque système (Module 3) -- les deux seuls automates dont une "alerte" fait basculer le
# Fleet Command Arbiter en lockdown : singularité de vélocité réelle (Navier-Stokes) et vide
# topologique réel (β1>0). Un choix disclosé, pas dérivé automatiquement -- Daemon_04 (Kelly) reste un
# blocage *local* de Daemon_02 (déjà réel dans core_daemons.py), pas un veto de réseau entier.
_RISK_DAEMONS: tuple[str, ...] = ("Daemon_01", "Daemon_14")

# Module 5 -- la vraie flotte de production (agent_fleet.py). Liste réelle dupliquée depuis
# agent_fleet.py::_HEARTBEAT_AGENTS (on ne peut pas l'importer : ce module exécute du code Streamlit
# top-level à l'import, comme toute page ``st.Page``) -- même identifiants réels, mêmes clés Redis
# réelles (``HEARTBEAT_REDIS_KEY_PREFIX``/``AGENT_CONTROL_STATE_REDIS_KEY_PREFIX``), lecture seule ici.
_REAL_FLEET_AGENTS: tuple[str, ...] = (
    "supervisor-agent", "rl-signal-agent", "execution-agent", "alert-agent",
    "macro-order-execution-agent", "audit-ledger-agent", "political-macro-sentiment-agent",
)

# Module 6 -- real Fleet Command Arbiter (src/agents/execution/agent.py) gate telemetry. Duplicated
# literal, not imported: it is a private (leading-underscore) constant on that module, and this
# project's own convention (see e.g. red_phone.py/risk_management.py) is that a Streamlit page
# duplicates such a private key rather than importing a leading-underscore name across the boundary.
_GATE_TELEMETRY_REDIS_KEY: Final[str] = "NOTHIERLY_FLEET_COMMAND_GATES"

_GATE_STATUS_COLOR: dict[str, str] = {"PASSED": _GREEN, "FAILED": _BLOOD, "PENDING": _GRAY}
_GATE_STATUS_ICON: dict[str, str] = {"PASSED": "🟢", "FAILED": "🔴", "PENDING": "⚪"}
_FINAL_ACTION_LABEL: dict[str, str] = {
    "LIVE_ORDER": "🟢 [CREATE_MARKET_ORDER] -- tir réel (LIVE, MOCK_MODE=false)",
    "SHADOW_ORDER": "🔵 [CREATE_MARKET_ORDER — SIMULÉ] -- control_state=SHADOW, aucun appel broker réel",
    "MOCK_ORDER": "🔵 [CREATE_MARKET_ORDER — SIMULÉ] -- MOCK_MODE=true, aucun appel broker réel",
    "REJECTED": "⛔ REJETÉ -- au moins une porte est fermée",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background: radial-gradient(ellipse at 50% -10%, {_ANTHRACITE} 0%, {_BLACK} 65%); }}
    .fcn-mono {{ font-family: 'JetBrains Mono', 'Consolas', monospace; }}
    .fcn-glass {{
        background: rgba(255,255,255,0.035); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.09); border-radius: 14px; padding: 1rem 1.25rem;
    }}
    .fcn-glass-cyan {{ border-color: rgba(0,229,255,0.45); box-shadow: 0 0 20px rgba(0,229,255,0.08) inset; }}
    .fcn-glass-gold {{ border-color: rgba(232,196,104,0.45); box-shadow: 0 0 20px rgba(232,196,104,0.08) inset; }}
    .fcn-glass-red {{ border-color: rgba(200,30,58,0.55); box-shadow: 0 0 20px rgba(200,30,58,0.14) inset; }}
    .fcn-glass-steel {{ border-color: rgba(255,255,255,0.28); box-shadow: 0 0 20px rgba(255,255,255,0.05) inset; }}
    .fcn-badge-real {{
        font-family: 'JetBrains Mono', monospace; background: rgba(255,255,255,0.12); color: #eef2ff;
        border: 1px solid #eef2ff; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.7rem; font-weight: 700;
    }}
    .fcn-disclosure {{
        border-left: 3px solid {_CYAN}; background: rgba(0,229,255,0.05); padding: 0.6rem 1rem;
        font-size: 0.83rem; color: #a9e9ff; border-radius: 4px;
    }}
    .fcn-badge {{
        font-family: 'JetBrains Mono', monospace; background: rgba(232,196,104,0.14); color: {_GOLD};
        border: 1px solid {_GOLD}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.7rem; font-weight: 700;
    }}
    .fcn-led {{ border-radius: 10px; padding: 0.6rem 0.65rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); min-height: 108px; }}
    .fcn-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.35rem; }}
    .fcn-led-id {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #cfd3e0; }}
    .fcn-led-name {{ font-size: 0.8rem; font-weight: 700; margin-top: 0.15rem; }}
    .fcn-led-value {{ font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; margin-top: 0.3rem; color: #eef2ff; }}
    .fcn-led-label {{ font-size: 0.62rem; color: {_TEXT_DIM}; }}
    .fcn-led-status {{ font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; margin-top: 0.35rem; letter-spacing: 0.04em; }}
    .fcn-led-fresh {{ font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: {_TEXT_DIM}; float: right; }}
    .fcn-metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: {_CYAN}; }}
    .fcn-metric-label {{ font-size: 0.7rem; color: {_TEXT_DIM}; margin-top: 0.15rem; }}
    .fcn-terminal {{
        background: #000; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; padding: 0.75rem;
        border-radius: 8px; height: 210px; overflow-y: auto; line-height: 1.55; border: 1px solid rgba(0,229,255,0.2); color: #8ef2ff;
    }}
    .fcn-terminal .tag-up {{ color: {_GOLD}; }}
    .fcn-terminal .tag-down {{ color: {_CYAN}; }}
    .fcn-rpc-row {{ display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fcn-mono" style="font-size:1.7rem;font-weight:700;color:#eef2ff;letter-spacing:0.03em;">'
    '🛰️ FLEET COMMAND NEXUS</div>',
    unsafe_allow_html=True,
)
st.caption("Interface d'audit et de télémétrie temps réel -- API Web3 · 16 Daemons mathématiques · Consensus · ZeroMQ/Redis")
st.markdown(
    '<div class="fcn-disclosure">🔬 Chaque valeur ci-dessous provient des primitives réelles déjà '
    "vérifiées de <code>src/dashboard/core_daemons.py</code> et <code>backend_simulator.py</code> -- "
    "aucun chiffre recopié d'exemple. Le seul module explicitement <b>simulé</b> est la télémétrie RPC "
    "Web3 (Module 4) : ce projet n'a aucune connectivité RPC réelle (signature Web3 = mock disclosé), "
    "donc aucun badge « Connecté » n'est affiché sur une infrastructure qui n'existe pas.</div>",
    unsafe_allow_html=True,
)
st.markdown("")


def _dot(color: str) -> str:
    return f'<span class="fcn-dot" style="background:{color};box-shadow:0 0 6px {color};"></span>'


# =============================================================================
# Module 1 -- Matrice des 16 Daemons Mathématiques (grille 4x4)
# =============================================================================


@st.fragment(run_every="1s")
def _render_daemon_matrix() -> None:
    market_state = get_state()
    autonomous_heartbeat(market_state)  # garde le reste du Protocole Léviathan vivant si cette page est seule ouverte
    swarm = ghost_daemon_heartbeat()
    now = time.time()

    st.markdown('<div class="fcn-glass fcn-glass-cyan">', unsafe_allow_html=True)
    st.markdown("##### 🧮 Module 1 · Matrice des 16 Daemons Mathématiques")
    cols_per_row = 4
    daemon_ids = list(DAEMON_ORDER)
    for row_start in range(0, len(daemon_ids), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for col, daemon_id in zip(row_cols, daemon_ids[row_start:row_start + cols_per_row]):
            result = swarm.results[daemon_id]
            codename, law = DAEMON_META[daemon_id]
            color = _LED_COLOR[result.status]
            fresh_ms = max(0.0, (now - result.ts) * 1000.0)
            value_label = _DAEMON_VALUE_LABEL[daemon_id]
            col.markdown(
                f'<div class="fcn-led" style="box-shadow: 0 0 12px {color}44; border-color: {color}55;">'
                f'<span class="fcn-led-fresh">{fresh_ms:,.0f} ms</span>'
                f'<div class="fcn-led-id">{daemon_id}</div>'
                f'<div class="fcn-led-name" style="color:{color};">{codename}</div>'
                f'<div class="fcn-led-label">{law}</div>'
                f'<div class="fcn-led-value">{result.value:,.3f}</div>'
                f'<div class="fcn-led-label">{value_label}</div>'
                f'<div class="fcn-led-status" style="color:{color};">{_dot(color)}{_LED_LABEL[result.status]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.caption(
        f"Tick #{swarm.tick_count} · légende propre à cette page (🟢 nominal / 🟡 alerte mathématique / "
        "🔵 exécution simulée / ⚪ veille) · fraîcheur = temps réel écoulé depuis le calcul de ce résultat "
        "(remplace un chiffre de latence fixe, qui ne mesurerait rien)."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 2 -- État du canal ZeroMQ & Redis (le "flux sanguin")
# =============================================================================


@st.fragment(run_every="1s")
def _render_infra_channel() -> None:
    manager = get_manager()
    bg = manager.background_snapshot()
    redis_ok, redis_rtt_ms = manager.redis_health()
    throughput = manager.zmq_throughput_per_sec()
    swarm = get_swarm_state()

    st.markdown('<div class="fcn-glass fcn-glass-gold">', unsafe_allow_html=True)
    st.markdown('##### 🩸 Module 2 · Canal ZeroMQ &amp; Redis &nbsp;<span class="fcn-badge">LE FLUX SANGUIN</span>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    redis_label = "🟢 EN LIGNE" if redis_ok else "🔴 HORS LIGNE"
    m1.markdown(f'<div class="fcn-glass"><div class="fcn-metric-value" style="font-size:1.0rem;">{redis_label}</div><div class="fcn-metric-label">Redis (PING réel)</div></div>', unsafe_allow_html=True)
    ping_label = f"{redis_rtt_ms:.2f} ms" if redis_rtt_ms is not None else "N/A"
    m2.markdown(f'<div class="fcn-glass"><div class="fcn-metric-value">{ping_label}</div><div class="fcn-metric-label">Ping serveur Redis</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="fcn-glass"><div class="fcn-metric-value">{throughput:.0f}/s</div><div class="fcn-metric-label">Messages/Seconde (ZMQ, réel)</div></div>', unsafe_allow_html=True)
    zmq_label = "🟢 LIÉ" if bg["zmq_bound"] else "🔴 NON LIÉ"
    m4.markdown(f'<div class="fcn-glass"><div class="fcn-metric-value" style="font-size:1.0rem;">{zmq_label}</div><div class="fcn-metric-label">Bus ZMQ isolé ({GHOST_ZMQ_PUB_ADDRESS})</div></div>', unsafe_allow_html=True)

    term_up, term_down = st.columns(2)
    with term_up:
        st.markdown("**⬆️ Daemons → Fleet Command**")
        up_lines = list(swarm.log)[-10:][::-1]
        body = "".join(f'<div><span class="tag-up">[SIGNAL]</span> {ln}</div>' for ln in up_lines) or "<div>en attente du premier tick...</div>"
        st.markdown(f'<div class="fcn-terminal">{body}</div>', unsafe_allow_html=True)
    with term_down:
        st.markdown("**⬇️ Fleet Command → Exécution (ZMQ)**")
        # bg["log"] mélange, dans l'ordre réel d'arrivée, les lignes du swarm autonome en arrière-plan ET les
        # vraies lignes "[Automate-Exécution]" du consommateur ZMQ -- on ne garde que ces dernières ici pour
        # que le tag [ZMQ] reste honnête (ne pas étiqueter un simple recalcul de daemon comme un message ZMQ).
        down_lines = [ln for ln in bg["log"] if ln.startswith("[Automate-Exécution]")][-10:][::-1]
        body = "".join(f'<div><span class="tag-down">[ZMQ]</span> {ln}</div>' for ln in down_lines) or "<div>aucun ordre publié/consommé pour l'instant...</div>"
        st.markdown(f'<div class="fcn-terminal">{body}</div>', unsafe_allow_html=True)

    st.caption(
        f"Publiés / consommés (cumul réel) : **{bg['zmq_messages_published']} / {bg['zmq_messages_consumed']}** -- "
        f"détruits périmés (&gt;{STALE_MESSAGE_THRESHOLD_MS:.0f}ms) / backpressure : "
        f"**{bg['zmq_dropped_stale']} / {bg['zmq_dropped_backpressure']}** (queue bornée à {EXECUTION_QUEUE_MAXSIZE}). "
        "Deux flux réels distincts, pas un « REDIS PULL » fabriqué par automate : le flux montant vient du "
        "vrai résultat en mémoire de chaque automate ; le flux descendant ne montre que les vraies lignes du "
        "consommateur ZMQ, publiées uniquement quand Daemon_02 (Aave) exécute réellement un emprunt simulé."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 3 -- Arbre de consensus (Fleet Command Arbiter)
# =============================================================================

_DAEMON_ANGLE_OFFSET = -math.pi / 2  # Daemon_01 en haut, sens horaire
_RING_RADIUS = 2.7
_EXEC_NODES: tuple[str, ...] = ("Uniswap V3", "Aave", "Curve/Hyperliquid")
_EXEC_POS: dict[str, tuple[float, float]] = {
    "Uniswap V3": (-2.3, -4.3),
    "Aave": (0.0, -4.7),
    "Curve/Hyperliquid": (2.3, -4.3),
}


def _daemon_positions() -> dict[str, tuple[float, float]]:
    n = len(DAEMON_ORDER)
    positions = {}
    for i, daemon_id in enumerate(DAEMON_ORDER):
        angle = _DAEMON_ANGLE_OFFSET + 2 * math.pi * i / n
        positions[daemon_id] = (_RING_RADIUS * math.cos(angle), _RING_RADIUS * math.sin(angle))
    return positions


@st.fragment(run_every="1s")
def _render_consensus_tree() -> None:
    swarm = get_swarm_state()
    daemon_pos = _daemon_positions()
    lockdown = any(swarm.results[d].status == "alerte" for d in _RISK_DAEMONS)

    st.markdown(f'<div class="fcn-glass {"fcn-glass-red" if lockdown else "fcn-glass-cyan"}">', unsafe_allow_html=True)
    st.markdown("##### 🕸️ Module 3 · Arbre de Consensus (Fleet Command Arbiter)", unsafe_allow_html=True)

    fig = go.Figure()

    # Daemon -> Arbiter (couleur = statut réel de l'automate)
    for daemon_id, (x, y) in daemon_pos.items():
        result = swarm.results[daemon_id]
        color = _LED_COLOR[result.status]
        hot = result.status in ("alerte", "exécution")
        fig.add_trace(go.Scatter(
            x=[x, 0.0], y=[y, 0.0], mode="lines",
            line=dict(color=color, width=3.2 if hot else 1.1),
            opacity=0.95 if hot else 0.35, hoverinfo="skip", showlegend=False,
        ))

    # Arbiter -> Nœuds d'exécution Web3 (bloqués si lockdown de risque)
    for exec_name, (x, y) in _EXEC_POS.items():
        color = _BLOOD if lockdown else _CYAN
        fig.add_trace(go.Scatter(
            x=[0.0, x], y=[0.0, y], mode="lines",
            line=dict(color=color, width=2.6, dash="dot" if lockdown else "solid"),
            opacity=0.9 if lockdown else 0.55, hoverinfo="text",
            text="🔒 LOCKDOWN -- accès bloqué" if lockdown else f"routage autorisé → {exec_name}",
            showlegend=False,
        ))

    # Nœuds automates
    daemon_x = [p[0] for p in daemon_pos.values()]
    daemon_y = [p[1] for p in daemon_pos.values()]
    daemon_colors = [_LED_COLOR[swarm.results[d].status] for d in daemon_pos]
    daemon_hover = [
        f"{d} {DAEMON_META[d][0]} -- {_LED_LABEL[swarm.results[d].status]} -- "
        f"{_DAEMON_VALUE_LABEL[d]}={swarm.results[d].value:,.3f}"
        for d in daemon_pos
    ]
    fig.add_trace(go.Scatter(
        x=daemon_x, y=daemon_y, mode="markers+text",
        text=[d.replace("Daemon_", "D") for d in daemon_pos], textposition="top center",
        marker=dict(size=20, color=daemon_colors, line=dict(color="#0c0c11", width=1.5)),
        textfont=dict(color="#cfd3e0", size=9), hovertext=daemon_hover, hoverinfo="text", showlegend=False,
    ))

    # Nœuds d'exécution Web3
    exec_x = [p[0] for p in _EXEC_POS.values()]
    exec_y = [p[1] for p in _EXEC_POS.values()]
    fig.add_trace(go.Scatter(
        x=exec_x, y=exec_y, mode="markers+text",
        text=list(_EXEC_POS.keys()), textposition="bottom center",
        marker=dict(size=24, color=_GOLD, symbol="square", line=dict(color="#0c0c11", width=1.5)),
        textfont=dict(color=_GOLD, size=10), hoverinfo="text",
        hovertext=[f"nœud d'exécution simulé : {n}" for n in _EXEC_POS], showlegend=False,
    ))

    # Fleet Command Arbiter (centre)
    arbiter_color = _BLOOD if lockdown else _CYAN
    fig.add_trace(go.Scatter(
        x=[0.0], y=[0.0], mode="markers+text", text=["FLEET COMMAND<br>ARBITER"], textposition="middle center",
        marker=dict(size=54, color=arbiter_color, symbol="diamond", line=dict(color="#eef2ff", width=2)),
        textfont=dict(color="#0c0c11" if not lockdown else "#fff", size=9, family="JetBrains Mono"),
        hoverinfo="text",
        hovertext=["🔒 LOCKDOWN -- veto de risque actif" if lockdown else "consensus nominal -- routage autorisé"],
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-3.4, 3.4]), yaxis=dict(visible=False, range=[-5.4, 3.4]),
    )
    st.plotly_chart(fig, width="stretch", key=f"fcn_consensus_{time.time()}")

    if lockdown:
        st.error(
            "🔒 **LOCKDOWN** -- au moins un automate de risque réel (Navier-Stokes ou Topologie/β1) est en "
            "alerte mathématique : le Fleet Command Arbiter bloque le routage vers les nœuds d'exécution "
            "Web3 simulés jusqu'à ce que ces automates repassent à un état nominal."
        )
    else:
        st.success("✅ Consensus nominal -- aucun veto de risque actif, routage autorisé vers les nœuds d'exécution simulés.")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 4 -- Télémétrie des API Web3 (RPC Nodes) -- explicitement simulée
# =============================================================================

_RPC_VENUES: tuple[tuple[str, float, float], ...] = (
    ("Uniswap V3", 12.0, 0.0),
    ("Aave FlashLoan", 14.0, 1.4),
    ("Hyperliquid", 8.0, 2.7),
)


def _simulated_rpc_ping(base_ms: float, phase: float, chaos_bias: float) -> float:
    """Ping RPC **explicitement simulé** -- ce projet n'a aucune connectivité RPC Web3 réelle (voir
    ``core_daemons.mock_web3_sign_transaction``). Formule réelle et vivante (pilotée par le vrai
    ``shannon_chaos_bias`` partagé), pas un chiffre statique -- mais jamais présentée comme une mesure
    de réseau réelle."""
    jitter = 3.0 * math.sin(time.time() * 0.6 + phase) + 4.0 * chaos_bias
    return max(1.0, base_ms + jitter)


@st.fragment(run_every="1s")
def _render_web3_telemetry() -> None:
    market_state = get_state()
    st.markdown('<div class="fcn-glass fcn-glass-gold">', unsafe_allow_html=True)
    st.markdown("##### 🌐 Module 4 · Télémétrie des API Web3 (RPC Nodes)")
    st.markdown(
        '<div class="fcn-disclosure">🎭 <b>Simulé</b> -- aucun point RPC réel, aucune vraie clé, aucun '
        "vrai gas dans ce process (même disclosure que la signature Web3 mock de "
        "<code>core_daemons.py</code>). Ces pings sont une formule réelle pilotée par le chaos partagé, "
        "pas une mesure réseau ni un badge « Connecté » à une infrastructure inexistante.</div>",
        unsafe_allow_html=True,
    )
    for name, base_ms, phase in _RPC_VENUES:
        ping = _simulated_rpc_ping(base_ms, phase, market_state.shannon_chaos_bias)
        st.markdown(
            f'<div class="fcn-rpc-row"><span>🔌 RPC {name}</span>'
            f'<span>Ping {ping:.1f} ms (simulé) &nbsp;|&nbsp; <span style="color:{_GOLD};">🎭 MOCK</span></span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 5 -- Agent Fleet Command (production réelle) -- liaison réelle, lecture seule
# =============================================================================


@st.fragment(run_every="2s")
def _render_real_agent_fleet() -> None:
    """Lit les mêmes vraies clés Redis que ``agent_fleet.py`` (``heartbeat:{agent_id}`` et
    ``agent_control_state:{agent_id}``) -- une vraie liaison d'observation avec la flotte de production,
    pas une copie de données ni une simulation. Volontairement sans aucun bouton LIVE/SHADOW/OFF : seule
    ``agent_fleet.py`` est autorisée à écrire sur le canal de contrôle réel (voir docstring du module)."""
    r = get_redis_client()
    st.markdown('<div class="fcn-glass fcn-glass-steel">', unsafe_allow_html=True)
    st.markdown(
        '##### 🤖 Module 5 · Agent Fleet Command &nbsp;<span class="fcn-badge-real">🔒 RÉEL — PRODUCTION</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fcn-disclosure" style="border-left-color:#eef2ff;background:rgba(255,255,255,0.04);color:#dfe3ee;">'
        "🔗 Liaison réelle avec <code>agent_fleet.py</code> -- ce panneau lit les mêmes clés Redis réelles "
        "que cette page (heartbeat + état de contrôle), en <b>lecture seule</b>. Le veto de risque simulé du "
        "Module 3 (maths sur données synthétiques) n'écrit jamais sur le canal de contrôle réel -- toute "
        "action LIVE/SHADOW/OFF réelle reste exclusivement sur <code>Agent Fleet Command</code>.</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    online_count = 0
    for i, agent_id in enumerate(_REAL_FLEET_AGENTS):
        raw_heartbeat = safe_get(r, f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}")
        online = raw_heartbeat is not None
        online_count += int(online)
        raw_state = safe_get(r, f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent_id}")
        try:
            control_state = AgentControlState(raw_state).value if raw_state is not None else "LIVE"
        except ValueError:
            control_state = "LIVE"
        color = _GREEN if online else _GRAY
        cols[i % 4].markdown(
            f'<div class="fcn-led" style="min-height:76px;box-shadow:0 0 10px {color}44;border-color:{color}55;">'
            f'<div class="fcn-led-id">{agent_id}</div>'
            f'<div class="fcn-led-status" style="color:{color};margin-top:0.4rem;">{_dot(color)}'
            f'{"EN LIGNE" if online else "HORS LIGNE"}</div>'
            f'<div class="fcn-led-label">état réel : {control_state}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(f"**{online_count}/{len(_REAL_FLEET_AGENTS)}** agents réels en ligne (heartbeat Redis réel, même source que la page Agent Fleet Command).")
    st.page_link("ems_pages/agent_fleet.py", label="Ouvrir Agent Fleet Command (contrôle réel LIVE/SHADOW/OFF)", icon="🤖")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 6 -- Fleet Command Arbiter : les 7 Portes d'Exécution (production réelle)
# =============================================================================


@st.fragment(run_every="1s")
def _render_fleet_command_gates() -> None:
    """Lit la vraie télémétrie 7-portes publiée par ``src/agents/execution/agent.py``
    (``ExecutionAgent._publish_gate_report``, nom d'affichage ``Fleet_Command_Arbiter``) -- une vraie
    liaison d'observation avec le moteur d'exécution de production, en lecture seule, pas une
    simulation. Voir le docstring de ce module (section "Module 6") pour la correction honnête sur
    ZMQ vs Redis et sur pourquoi ce panneau montre un instantané réel plutôt qu'une animation."""
    r = get_redis_client()
    st.markdown('<div class="fcn-glass fcn-glass-steel">', unsafe_allow_html=True)
    st.markdown(
        "##### 🚪 Module 6 · Fleet Command Arbiter — Les Portes d'Exécution "
        '&nbsp;<span class="fcn-badge-real">🔒 RÉEL — PRODUCTION</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fcn-disclosure" style="border-left-color:#eef2ff;background:rgba(255,255,255,0.04);color:#dfe3ee;">'
        "🔗 Télémétrie réelle de <code>src/agents/execution/agent.py</code> (nom d'affichage "
        "<b>Fleet_Command_Arbiter</b> -- la vraie classe reste <code>ExecutionAgent</code>, voir son "
        "propre docstring). Chaque porte ci-dessous est le vrai verdict de la dernière tentative "
        "d'entrée réellement évaluée pour cet instrument -- un instantané réel, pas une animation : le "
        "pipeline complet s'exécute en quelques millisecondes. Aucun bouton ici : cette page ne pilote "
        "jamais l'exécution réelle, elle l'observe.</div>",
        unsafe_allow_html=True,
    )

    gate_state = safe_get_json(r, _GATE_TELEMETRY_REDIS_KEY)
    if not gate_state:
        st.info(
            "Aucune tentative d'entrée réelle évaluée pour l'instant -- ce panneau s'allume dès le "
            "premier consensus réel atteint par la flotte de production (voir Module 5)."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for instrument, record in sorted(gate_state.items()):
        timestamp = record.get("timestamp", time.time())
        age_s = max(0.0, time.time() - timestamp)
        direction_label = "ACHAT" if record.get("direction") == 1 else "VENTE"
        final_action = record.get("final_action", "REJECTED")
        gates = record.get("gates", [])

        st.markdown(f"**{instrument}** -- {direction_label} &nbsp;·&nbsp; il y a {age_s:.1f}s")
        # len(gates), not a hardcoded count: real gate count (8 as of "Protocole Stochastic
        # Core"), read from the real telemetry rather than a number that could drift from it again.
        cols = st.columns(len(gates)) if gates else []
        for gate, col in zip(gates, cols):
            status = gate.get("status", "PENDING")
            color = _GATE_STATUS_COLOR.get(status, _GRAY)
            col.markdown(
                f'<div class="fcn-led" style="min-height:88px;box-shadow:0 0 10px {color}44;border-color:{color}55;">'
                f'<div class="fcn-led-id">Porte {gate.get("number")}</div>'
                f'<div class="fcn-led-name" style="color:{color};font-size:0.7rem;">{gate.get("name")}</div>'
                f'<div class="fcn-led-status" style="color:{color};">{_GATE_STATUS_ICON.get(status, "⚪")} {status}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        first_failed = next((g for g in gates if g.get("status") == "FAILED"), None)
        if first_failed is not None:
            st.caption(f"🔴 Porte {first_failed['number']} ({first_failed['name']}) : {first_failed['detail']}")
        st.markdown(
            f'<div class="fcn-terminal" style="height:auto;padding:0.5rem 0.75rem;">'
            f'{_FINAL_ACTION_LABEL.get(final_action, final_action)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 7 -- Courbe d'Impact d'Exécution Optimale (Almgren-Chriss, production réelle)
# =============================================================================

_OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX = "optimal_execution_schedule:"  # duplicated literal, see Module 6's own comment on this project's private-constant convention


@st.fragment(run_every="2s")
def _render_optimal_execution_curve() -> None:
    """Lit les vrais plans d'exécution calculés par ``ExecutionAgent.compute_optimal_trajectory``
    (Almgren-Chriss) -- lecture seule, voir le docstring de ce module (section Module 7) pour la
    frontière de portée disclosée (plan calculé, pas un ordre réellement tranché)."""
    r = get_redis_client()
    st.markdown('<div class="fcn-glass fcn-glass-steel">', unsafe_allow_html=True)
    st.markdown(
        "##### 📉 Module 7 · Courbe d'Impact d'Exécution Optimale (Almgren-Chriss) "
        '&nbsp;<span class="fcn-badge-real">🔒 RÉEL — PRODUCTION</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fcn-disclosure" style="border-left-color:#eef2ff;background:rgba(255,255,255,0.04);color:#dfe3ee;">'
        "🔗 Télémétrie réelle de <code>src/agents/execution/agent.py</code>'s "
        "<code>compute_optimal_trajectory</code> -- impact temporaire (réversible, par tranche) et "
        "impact permanent (cumulatif) réels du dernier plan Almgren-Chriss calculé par instrument. "
        "<b>Plan calculé, pas un ordre réellement tranché</b> : cet agent place toujours un seul "
        "ordre au marché par entrée (voir son propre docstring) -- ce module montre ce qu'un "
        "découpage optimal recommanderait, pas ce qui a été réellement exécuté tranche par tranche.</div>",
        unsafe_allow_html=True,
    )

    schedule_keys = safe_keys(r, f"{_OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX}*")
    if not schedule_keys:
        st.info(
            "Aucun plan d'exécution calculé pour l'instant -- ce panneau s'allume dès le premier "
            "appel réel à compute_optimal_trajectory (voir son docstring pour comment l'invoquer)."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for key in sorted(schedule_keys):
        record = safe_get_json(r, key)
        if record is None:
            continue
        instrument = record.get("instrument", key.removeprefix(_OPTIMAL_EXECUTION_SCHEDULE_REDIS_KEY_PREFIX))
        temporary_impact = record.get("temporary_impact", [])
        permanent_impact = record.get("permanent_impact", [])
        if not temporary_impact or not permanent_impact:
            continue

        age_s = max(0.0, time.time() - record.get("timestamp", time.time()))
        capped = record.get("was_participation_capped", False)
        st.markdown(
            f"**{instrument}** -- taille {record.get('order_size', 0):,.2f} unités sur "
            f"{record.get('execution_horizon_seconds', 0):,.0f}s &nbsp;·&nbsp; il y a {age_s:.1f}s"
            + (" &nbsp;🟡 plafonné par la liquidité réelle disponible" if capped else "")
        )

        slices = list(range(1, len(temporary_impact) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=slices, y=temporary_impact, mode="lines+markers", name="Impact temporaire (par tranche)",
            line=dict(color=_CYAN, width=2.4), marker=dict(size=7),
        ))
        fig.add_trace(go.Scatter(
            x=slices, y=permanent_impact, mode="lines+markers", name="Impact permanent (cumulatif)",
            line=dict(color=_GOLD, width=2.4), marker=dict(size=7),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Tranche d'exécution"), yaxis=dict(title="Impact de prix (unités réelles)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch", key=f"fcn_impact_{instrument}_{time.time()}")
        st.caption(f"Coût temporaire total estimé : {record.get('total_estimated_temporary_cost', 0):.6f} (unités de prix réelles)")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 8 -- Greeks de Portefeuille & Couverture Multi-Instruments (production réelle)
# =============================================================================

_GREEK_METRIC_META: Final[tuple[tuple[str, str, str], ...]] = (
    ("delta", "Δ Delta", "Sensibilité au prix du sous-jacent"),
    ("gamma", "Γ Gamma", "Sensibilité de Delta lui-même"),
    ("vega", "ν Vega", "Sensibilité à la volatilité"),
    ("theta", "Θ Theta", "Décroissance temporelle (par an)"),
)


@st.fragment(run_every="2s")
def _render_portfolio_greeks() -> None:
    """Lit les vrais Greeks de portefeuille agrégés par ``ExecutionAgent.compute_portfolio_greeks``
    (options réelles mais nécessairement simulées + exposition spot/CFD réelle et signée) -- lecture
    seule, voir le docstring de ce module (section Module 8) pour la frontière de portée honnête
    (pas encore appelé automatiquement par la boucle de tick réelle)."""
    r = get_redis_client()
    st.markdown('<div class="fcn-glass fcn-glass-steel">', unsafe_allow_html=True)
    st.markdown(
        "##### Σ Module 8 · Greeks de Portefeuille & Couverture Multi-Instruments "
        '&nbsp;<span class="fcn-badge-real">🔒 RÉEL — PRODUCTION</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fcn-disclosure" style="border-left-color:#eef2ff;background:rgba(255,255,255,0.04);color:#dfe3ee;">'
        "🔗 Télémétrie réelle de <code>ExecutionAgent.compute_portfolio_greeks</code> -- agrège les "
        "vraies positions d'options (nécessairement simulées/paper-fill, voir "
        "<code>options_hedging_manager.py</code>) avec l'exposition spot/CFD réelle et signée "
        "(<code>PositionState.units_held</code>). <b>Pas encore appelé automatiquement</b> par la "
        "boucle de tick réelle -- ce panneau reste honnêtement vide tant qu'un opérateur (ou un "
        "futur script périodique) n'a pas réellement invoqué cette méthode au moins une fois.</div>",
        unsafe_allow_html=True,
    )

    snapshot = safe_get_json(r, PORTFOLIO_GREEKS_REDIS_KEY)
    if not snapshot:
        st.info(
            "Aucun calcul de Greeks de portefeuille réel pour l'instant -- ce panneau s'allume dès le "
            "premier appel réel à compute_portfolio_greeks (voir son propre docstring)."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    age_s = max(0.0, time.time() - snapshot.get("timestamp", time.time()))
    st.caption(f"Dernier calcul réel il y a {age_s:.1f}s")

    cols = st.columns(len(_GREEK_METRIC_META))
    for (field, label, help_text), col in zip(_GREEK_METRIC_META, cols):
        value = snapshot.get(field, 0.0)
        col.metric(label, f"{value:,.4f}", help=help_text)

    if abs(snapshot.get("vega", 0.0)) > 1e-9 or abs(snapshot.get("gamma", 0.0)) > 1e-12:
        st.caption(
            "Γ/ν non nuls : au moins une position d'option (simulée) réelle contribue à cette "
            "exposition -- un instrument spot seul ne peut structurellement produire ni gamma ni "
            "vega (voir Ashby's Law / check_hedge_completeness)."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Module 9 -- Smart Order Router : historique réel de routage (production réelle)
# =============================================================================

_SOR_SIDE_LABEL: Final[dict[str, str]] = {"BUY": "ACHAT", "SELL": "VENTE"}


@st.fragment(run_every="5s")
def _render_smart_order_router_history() -> None:
    """Lit le vrai journal des décisions de routage réellement exécutées par
    ``src/execution/smart_order_router.py`` (``core.schemas.SOR_ROUTING_LOG_REDIS_KEY``) -- lecture
    seule, voir le docstring de ce module (section Module 9) pour l'audit complet des 9 pages
    "lois mathématiques" et pourquoi seule celle-ci est honnêtement câblable ici."""
    r = get_redis_client()
    st.markdown('<div class="fcn-glass fcn-glass-steel">', unsafe_allow_html=True)
    st.markdown(
        "##### 🔀 Module 9 · Smart Order Router — Historique Réel de Routage "
        '&nbsp;<span class="fcn-badge-real">🔒 RÉEL — PRODUCTION</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fcn-disclosure" style="border-left-color:#eef2ff;background:rgba(255,255,255,0.04);color:#dfe3ee;">'
        "🔗 Télémétrie réelle de <code>src/execution/smart_order_router.py</code> -- chaque ligne "
        "ci-dessous est une décision de routage <b>réellement exécutée</b> (jamais un simple aperçu "
        "<code>build_allocation_plan</code>), avec son vrai gain effectif-prix contre le pire niveau "
        "utilisé. Aucune des 8 autres pages \"lois mathématiques\" auditées le 2026-09-14 n'est "
        "câblée ici -- leurs maths sont réelles mais tournent sur des données synthétiques/"
        "illustratives par conception, et y mélanger une donnée simulée serait exactement ce que ce "
        "projet refuse ailleurs.</div>",
        unsafe_allow_html=True,
    )

    raw_entries = safe_lrange(r, SOR_ROUTING_LOG_REDIS_KEY, 10)
    if not raw_entries:
        st.info(
            "Aucune décision de routage réelle pour l'instant -- ce panneau s'allume dès le premier "
            "vrai routage multi-venues exécuté depuis la page Liquidity Routing."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for raw in raw_entries:
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        age_s = max(0.0, time.time() - record.get("timestamp", time.time()))
        side_label = _SOR_SIDE_LABEL.get(record.get("side", ""), record.get("side", "?"))
        fully_filled = record.get("fully_filled", False)
        savings = record.get("savings_usd", 0.0)
        venues = ", ".join(a.get("broker_id", "?") for a in record.get("allocations", []))
        st.markdown(
            f"**{record.get('symbol', '?')}** -- {side_label} "
            f"{record.get('allocated_quantity', '?')}/{record.get('requested_quantity', '?')} "
            f"&nbsp;·&nbsp; il y a {age_s:.1f}s "
            + ("&nbsp;🟢 rempli entièrement" if fully_filled else "&nbsp;🟡 partiellement rempli")
        )
        st.caption(f"Venues réelles utilisées : {venues or '—'} &nbsp;·&nbsp; gain réel vs pire niveau : ${savings:,.4f}")

    st.markdown("</div>", unsafe_allow_html=True)


_render_daemon_matrix()
st.markdown("")
_render_infra_channel()
st.markdown("")
_render_consensus_tree()
st.markdown("")
_render_web3_telemetry()
st.markdown("")
_render_real_agent_fleet()
st.markdown("")
_render_fleet_command_gates()
st.markdown("")
_render_optimal_execution_curve()
st.markdown("")
_render_portfolio_greeks()
st.markdown("")
_render_smart_order_router_history()

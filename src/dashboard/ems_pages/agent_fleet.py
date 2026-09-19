"""Agent Fleet Command -- real liveness + real remote LIVE/SHADOW/OFF control, per micro-agent.

**Real data, both halves**: Tier 1 reads the same real
``heartbeat:{agent_id}`` Redis keys this page always has (see
:meth:`~core.base_agent.BaseAgent._write_heartbeat`), and now also
reads/writes each agent's real, persisted
:class:`~core.base_agent.AgentControlState` (``agent_control_state:
{agent_id}``) -- clicking a button here does not just change a
dashboard label: it persists the new state (so a restarted agent
resumes it) *and* ``PUBLISH``es it on that agent's real control
channel for instant effect on an already-running process, via
:meth:`~core.base_agent.BaseAgent._listen_for_control_commands`. See
that method's own docstring, and :class:`~src.agents.execution.agent.
ExecutionAgent`'s real, tested integration, for exactly what ``OFF``/
``SHADOW`` do for each real agent kind (an order-placing agent refuses/
simulates its entries; a signal-only agent never publishes onto the
real bus; the alert agent sends no SMS/trips no Kill Switch).

**Tier 2 -- "Shadow Fleet" R&D, honestly, not "verrouillé (locked)"
theater**: an earlier draft of this section proposed fabricated
progress metrics ("Ingestion: 4.2TB", "DAG Convergence: 84%",
"EPOCH 45/100") for the HMM regime detector, the SNN neuromorphic
engine, and the causal impact engine -- none of which run as live,
continuously-training services (confirmed by direct inspection: no
``BaseAgent`` subclass, no heartbeat, imported only by their own test
files). Inventing progress numbers for processes that are not actually
running would be exactly the fabricated-telemetry problem this
project's discipline exists to avoid. Instead, each Tier 2 card is
honestly labeled "recherche / sur demande" and carries a real "Lancer
maintenant" button that actually runs that module's real, tested
pipeline right now and displays its real result -- a genuine regime
label from a real Gaussian HMM fit on real BTCUSDT bars
(``research/sample_data/``), a real measured SNN-vs-LSTM latency
benchmark, or a real DoWhy causal-effect estimate on synthetic data
with known ground truth -- never a number that didn't just get
computed live.

**Liaison avec Fleet Command Nexus (2026-09-13)** : un petit panneau en haut de cette page lit le vrai
``background_snapshot()``/``redis_health()`` de ``core_daemons.LeviathanDaemonManager`` (Protocole
"DeFi Ghost Daemons", entièrement simulé -- voir son propre docstring) et pointe vers cette page via
``st.page_link``. Liaison réelle, bidirectionnelle en observation (Fleet Command Nexus lit en retour
les vraies clés heartbeat/état de contrôle de cette page), mais strictement à sens unique côté écriture :
seule cette page publie/écrit sur ``AGENT_CONTROL_CHANNEL_PREFIX``/``AGENT_CONTROL_STATE_REDIS_KEY_
PREFIX`` -- le veto de risque simulé de Fleet Command Nexus ne pilote jamais un agent réel.
"""

from __future__ import annotations

import json
import time
from typing import Literal

import streamlit as st

from core.base_agent import AgentControlState
from core.schemas import (
    AGENT_CONTROL_CHANNEL_PREFIX,
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    HEARTBEAT_REDIS_KEY_PREFIX,
    LLM_STATUS_REDIS_KEY,
)
from src.dashboard.core_daemons import get_manager as get_ghost_daemon_manager
from src.dashboard.ems_pages._shared import get_redis_client, safe_get
from src.dashboard.theme import status_badge

# A status this stale means political_agent.py has stopped scoring headlines altogether (crashed,
# or simply hasn't run a cycle yet) -- real signal, not a guessed threshold: _POLL_INTERVAL_SECONDS
# there is 15 minutes, so anything past ~2 cycles' worth is genuinely stale, not just "between polls".
_LLM_STATUS_STALE_AFTER_SECONDS = 30 * 60

_HEARTBEAT_AGENTS = (
    "supervisor-agent", "rl-signal-agent", "execution-agent", "alert-agent", "macro-order-execution-agent",
    "audit-ledger-agent",
    # Real gap closed here, found while wiring the "Statut de l'IA" indicator below: this agent runs
    # in its own container (docker-compose.yml's nothierly-political-agent), writes a real
    # heartbeat:{agent_id} key exactly like every agent below (see political_agent.py's own
    # _write_heartbeat) -- it was simply never added to this page's own list, so it never showed up
    # here at all despite being real and running, the same silently-invisible pattern this page's own
    # module docstring already disclosed for macro-agent/nlp-agent/archivist-agent (those three
    # instead never call _write_heartbeat in the first place -- a different, still-open gap).
    "political-macro-sentiment-agent",
)
# Real gap closed here, found auditing scripts/run_backend.py's own real 16-agent count against what
# this page actually showed: MacroAgent, NlpAgent, and ArchivistAgent are real, running,
# continuously-ticking agents (confirmed live: macro_snapshot_published, nlp_fetch_cycle_started,
# archivist_batch_written events fire on schedule) -- but unlike every other agent in this backend,
# none of their own run() overrides ever call BaseAgent._write_heartbeat(), so they were previously
# invisible here: not shown as "HORS LIGNE" (which would at least flag a real problem), just absent
# from the page entirely, indistinguishable from not existing. Real fix belongs in each agent's own
# run() loop (a genuine missing self._write_heartbeat() call, out of scope for a dashboard-only
# patch); until that lands, this section honestly labels them PENDING/"pas de heartbeat" rather than
# fabricating one or silently omitting them -- same "recherche / sur demande" honesty this page's own
# Tier 2 section already established, not a new convention.
_NO_HEARTBEAT_AGENTS = (
    ("market-data-agent", "2 instances réelles (BTC/USDT, ETH/USDT) -- boucle événementielle WebSocket, pas de tick loop, pas de heartbeat."),
    ("macro-agent", "Tourne réellement (poll_interval=3600s) -- publie macro_snapshot_published -- mais n'appelle jamais _write_heartbeat() (gap réel, pas encore corrigé)."),
    ("nlp-agent", "Tourne réellement (poll_interval=300s) -- publie nlp_fetch_cycle_started -- mais n'appelle jamais _write_heartbeat() (gap réel, pas encore corrigé)."),
    ("archivist-agent", "Tourne réellement (flush_interval=5s) -- publie archivist_batch_written -- mais n'appelle jamais _write_heartbeat() (gap réel, pas encore corrigé)."),
)

# market-data-agent is deliberately absent from the control panel below:
# it never calls a broker, sends an alert, or publishes a trading
# opinion -- there is no real "OFF"/"SHADOW" action for it to gate (see
# module docstring). Only agents with a real dangerous action to pause
# get a real control card.
#
# audit-ledger-agent is also absent, deliberately, for the opposite
# reason: it has a real heartbeat (shown below like every other Tier 1
# agent) but no real LIVE/SHADOW/OFF buttons, on purpose -- see
# src/execution/audit_ledger.py's own module docstring on why silencing
# a compliance audit trail is never a legitimate "pause" action for an
# operator to take from a dashboard.
_CONTROLLABLE_AGENTS = ("supervisor-agent", "rl-signal-agent", "execution-agent", "alert-agent", "macro-order-execution-agent")

st.title("🤖 Agent Fleet Command")
st.caption("Statut réel de chaque micro-agent, et contrôle réel LIVE / SHADOW / OFF -- voir core/base_agent.py.")

r = get_redis_client()

with st.container(border=True):
    st.caption("🔗 Liaison réelle avec Fleet Command Nexus (lecture seule -- voir docstring de ce module)")
    _ghost_manager = get_ghost_daemon_manager()
    _ghost_bg = _ghost_manager.background_snapshot()
    _ghost_redis_ok, _ghost_redis_rtt = _ghost_manager.redis_health()
    col_ghost_redis, col_ghost_zmq, col_ghost_tick = st.columns(3)
    col_ghost_redis.metric("Redis Ghost Daemons (PING réel)", f"{_ghost_redis_rtt:.2f} ms" if _ghost_redis_ok else "HORS LIGNE")
    col_ghost_zmq.metric("Bus ZMQ isolé (Protocole Ghost Daemons)", "LIÉ" if _ghost_bg["zmq_bound"] else "NON LIÉ")
    col_ghost_tick.metric("Tick swarm (16 daemons, simulé)", _ghost_bg["tick"])
    st.page_link("ems_pages/fleet_command_nexus.py", label="Ouvrir Fleet Command Nexus (audit complet, simulation mathématique)", icon="🛰️")

st.subheader("🟢 Tier 1 -- Flotte de production")

for agent_id in _HEARTBEAT_AGENTS:
    raw_heartbeat = safe_get(r, f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}")
    online = raw_heartbeat is not None
    age_seconds = time.time() - float(raw_heartbeat) if raw_heartbeat is not None else None

    with st.container(border=True):
        col_name, col_heartbeat, col_state = st.columns([2, 2, 2])
        col_name.markdown(f"**{agent_id}**")
        col_heartbeat.markdown(
            status_badge("SAFE" if online else "PANIC", f"{age_seconds:.1f}s" if age_seconds is not None else "HORS LIGNE"),
            unsafe_allow_html=True,
        )

        if agent_id not in _CONTROLLABLE_AGENTS:
            col_state.markdown(status_badge("SAFE", "TOUJOURS ACTIF"), unsafe_allow_html=True)
            if agent_id == "political-macro-sentiment-agent":
                st.caption("Pas de contrôle LIVE/SHADOW/OFF -- agent en lecture seule (RSS + LLM), jamais d'ordre.")
            else:
                st.caption("Pas de contrôle LIVE/SHADOW/OFF -- voir le docstring de src/execution/audit_ledger.py.")
            continue

        raw_state = safe_get(r, f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent_id}")
        try:
            current_state = AgentControlState(raw_state) if raw_state is not None else AgentControlState.LIVE
        except ValueError:
            current_state = AgentControlState.LIVE

        if current_state == AgentControlState.LIVE:
            state_badge_kind: Literal["SAFE", "PANIC", "PENDING"] = "SAFE"
        elif current_state == AgentControlState.SHADOW:
            state_badge_kind = "PENDING"
        else:
            state_badge_kind = "PANIC"
        col_state.markdown(status_badge(state_badge_kind, current_state.value), unsafe_allow_html=True)

        col_live, col_shadow, col_off = st.columns(3)
        for col, target_state, label in (
            (col_live, AgentControlState.LIVE, "⚡ LIVE"),
            (col_shadow, AgentControlState.SHADOW, "👻 SHADOW"),
            (col_off, AgentControlState.OFF, "🔴 OFF"),
        ):
            disabled = current_state == target_state
            if col.button(label, key=f"control_{agent_id}_{target_state.value}", width="stretch", disabled=disabled):
                r.set(f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent_id}", target_state.value)
                r.publish(f"{AGENT_CONTROL_CHANNEL_PREFIX}{agent_id}", target_state.value)
                st.toast(f"{agent_id} -- nouvel état réel : {target_state.value}.", icon="✅")
                st.rerun()

for agent_id, note in _NO_HEARTBEAT_AGENTS:
    with st.container(border=True):
        col_name, col_note = st.columns([2, 4])
        col_name.markdown(f"**{agent_id}**")
        col_note.markdown(status_badge("PENDING", "N/A"), unsafe_allow_html=True)
        st.caption(note)

online_count = sum(1 for agent_id in _HEARTBEAT_AGENTS if safe_get(r, f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}") is not None)
st.markdown(
    status_badge("SAFE" if online_count == len(_HEARTBEAT_AGENTS) else "PANIC", f"{online_count}/{len(_HEARTBEAT_AGENTS)} AGENTS EN LIGNE"),
    unsafe_allow_html=True,
)
st.caption(
    "\"HORS LIGNE\" reflète l'état réel de Redis (heartbeat manquant), pas une simulation -- dans un "
    "environnement de démonstration sans agent réel en cours d'exécution, tous les agents apparaissent "
    "HORS LIGNE, ce qui est honnête, pas un bug."
)

st.markdown("---")
st.subheader("🧠 Statut de l'IA -- Moat Propriétaire (Circuit Breaker)")
st.caption(
    "Reflète l'issue réelle du dernier appel LLM de political_agent.py, pas une simple présence de "
    "configuration -- un endpoint Vast.ai configuré mais injoignable bascule quand même en Secours. "
    "Voir political_agent.py::_assess_headline / scripts/deploy_custom_llm.py."
)

raw_llm_status = safe_get(r, LLM_STATUS_REDIS_KEY)
if raw_llm_status is None:
    st.markdown(status_badge("PENDING", "AUCUN APPEL LLM ENCORE ENREGISTRÉ"), unsafe_allow_html=True)
    st.caption("political_agent.py n'a pas encore terminé de cycle avec au moins un titre à scorer.")
else:
    mode: str | None
    status_age_seconds: float | None
    try:
        llm_status = json.loads(raw_llm_status)
        mode = llm_status["mode"]
        status_age_seconds = time.time() - float(llm_status["timestamp"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        mode, status_age_seconds = None, None

    if mode is None:
        st.markdown(status_badge("PANIC", "STATUT ILLISIBLE"), unsafe_allow_html=True)
        st.caption(f"Valeur brute inattendue dans Redis ('{LLM_STATUS_REDIS_KEY}') -- vérifier political_agent.py.")
    elif status_age_seconds is not None and status_age_seconds > _LLM_STATUS_STALE_AFTER_SECONDS:
        st.markdown(status_badge("PANIC", f"PÉRIMÉ ({status_age_seconds / 60:.0f} min)"), unsafe_allow_html=True)
        st.caption("Dernier statut trop ancien -- political_agent.py a probablement arrêté de scorer des titres.")
    elif mode == "proprietary":
        st.markdown(status_badge("SAFE", "🟢 PROPRIÉTAIRE (VAST.AI)"), unsafe_allow_html=True)
        st.caption(f"Dernier appel réel réussi il y a {status_age_seconds:.0f}s sur le modèle self-hosted.")
    elif mode == "fallback":
        st.markdown(status_badge("PENDING", "🟠 SECOURS (API PUBLIQUE)"), unsafe_allow_html=True)
        st.caption(
            f"Bascule active il y a {status_age_seconds:.0f}s -- le GPU Vast.ai est injoignable, en cours de "
            "chargement, ou non déployé. Le trading n'est pas affecté (voir Cognitive Guardrail)."
        )
    else:  # "unconfigured"
        st.markdown(status_badge("PANIC", "AUCUN PROVIDER LLM CONFIGURÉ"), unsafe_allow_html=True)
        st.caption("Ni le GPU propriétaire ni un provider public (GUARDIAN_API_KEY) ne sont configurés.")

st.markdown("---")
st.subheader("🔬 Tier 2 -- R&D & Shadow Fleet (moteurs de nouvelle génération)")
st.caption(
    "Ces moteurs ne tournent pas comme des services en direct (aucun heartbeat, aucun processus "
    "continu) -- ils n'ont donc pas l'autorisation de router du capital. Chaque bouton \"Lancer "
    "maintenant\" exécute réellement le pipeline correspondant et affiche son vrai résultat, à la "
    "demande -- jamais un chiffre de progression fabriqué."
)

col_hmm, col_snn, col_causal = st.columns(3)

with col_hmm:
    st.markdown("**Markov Regime Switcher**")
    st.caption("Meta-Controller (HMM, prix + flux d'ordres réel TFI) -- risk/regime_detector.py")
    st.markdown(status_badge("PENDING", "RECHERCHE / SUR DEMANDE"), unsafe_allow_html=True)
    if st.button("▶️ Lancer une détection réelle", key="run_hmm"):
        with st.spinner("Fit HMM réel sur les barres BTCUSDT réelles (prix + TFI)..."):
            from risk.regime_detector import (
                GaussianHMM,
                build_regime_features,
                label_regimes,
                load_real_bar_closes,
                load_real_order_flow_series,
            )

            # "Order-Flow-Fused Regime Detection" (2026-09-19): a 3rd real feature (rolling TFI,
            # bar-aligned with `closes` -- see load_real_order_flow_series's own docstring for the
            # alignment guarantee), so a microstructure-driven regime shift doesn't have to wait to
            # show up in price alone before this detector can react to it.
            closes = load_real_bar_closes("BTCUSDT", bar_interval="1m")
            order_flow = load_real_order_flow_series("BTCUSDT", bar_interval="1m")
            features = build_regime_features(closes, window=20, order_flow=order_flow)
            hmm = GaussianHMM(num_states=3, num_features=3, seed=0)
            hmm.fit(features, num_iterations=100)
            labels = label_regimes(hmm)
            posterior = hmm.predict_proba(features)
            current_regime = labels[int(posterior[-1].argmax())]
            st.session_state["hmm_last_result"] = {
                "regime": current_regime.value, "confidence": float(posterior[-1].max()), "n_bars": len(closes),
            }
    if "hmm_last_result" in st.session_state:
        result = st.session_state["hmm_last_result"]
        st.metric("Régime détecté (réel)", result["regime"])
        st.caption(f"Confiance : {result['confidence']:.1%} -- {result['n_bars']} barres réelles BTCUSDT (research/sample_data).")

with col_snn:
    st.markdown("**Neuromorphic Tick Processor**")
    st.caption("Spiking Neural Network (SNN) -- nothierly_genesis/spiking_tick_network.py")
    st.markdown(status_badge("PENDING", "RECHERCHE / SUR DEMANDE"), unsafe_allow_html=True)
    if st.button("▶️ Lancer un benchmark réel", key="run_snn"):
        with st.spinner("Benchmark SNN vs LSTM réel en cours..."):
            from nothierly_genesis.spiking_tick_network import (
                LSTMBaseline,
                SpikingTickNetwork,
                TickEvent,
                benchmark_inference_latency,
                encode_order_events_to_spikes,
            )

            events = [
                TickEvent(event_type="ADD", side="BID" if i % 2 == 0 else "ASK", price_offset_bps=float(i), quantity=1.0)
                for i in range(50)
            ]
            spikes = encode_order_events_to_spikes(events)
            snn_result = benchmark_inference_latency(
                SpikingTickNetwork(hidden_size=64), spikes, num_warmup=5, num_trials=20, model_name="SNN"
            )
            lstm_result = benchmark_inference_latency(
                LSTMBaseline(hidden_size=64), spikes, num_warmup=5, num_trials=20, model_name="LSTM"
            )
            st.session_state["snn_last_result"] = {
                "snn_us": snn_result.mean_latency_us, "lstm_us": lstm_result.mean_latency_us,
            }
    if "snn_last_result" in st.session_state:
        result = st.session_state["snn_last_result"]
        m1, m2 = st.columns(2)
        m1.metric("Latence SNN (réelle)", f"{result['snn_us']:.0f} µs")
        m2.metric("Latence LSTM (réelle)", f"{result['lstm_us']:.0f} µs")
        st.caption(f"SNN {result['snn_us'] / result['lstm_us']:.0f}x plus lent que LSTM sur ce CPU -- mesuré, pas asserté (voir docstring du module).")

with col_causal:
    st.markdown("**Causal Impact Oracle**")
    st.caption("Do-Calculus / DAG Bayésien -- nothierly_genesis/causal_impact_engine.py")
    st.markdown(status_badge("PENDING", "RECHERCHE / SUR DEMANDE"), unsafe_allow_html=True)
    if st.button("▶️ Lancer une estimation réelle", key="run_causal"):
        with st.spinner("Estimation causale DoWhy réelle en cours..."):
            from nothierly_genesis.causal_impact_engine import estimate_causal_effect, generate_synthetic_market_data

            data, true_effect = generate_synthetic_market_data(n_samples=5000, seed=0)
            result = estimate_causal_effect(data, action="do(Buy_100_BTC)", baseline_volume=0.0, run_refutation=False)
            st.session_state["causal_last_result"] = {
                "estimate": result.estimated_slippage_impact, "true_effect_per_unit": true_effect,
            }
    if "causal_last_result" in st.session_state:
        result = st.session_state["causal_last_result"]
        st.metric("Impact slippage estimé (réel)", f"{result['estimate']:.2f}")
        st.caption(
            f"Sur données synthétiques à vérité terrain connue (effet réel/unité = {result['true_effect_per_unit']:.3f}) -- "
            "voir docstring du module pour la validation complète."
        )

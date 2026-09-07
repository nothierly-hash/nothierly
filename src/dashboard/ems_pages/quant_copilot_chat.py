"""Quant Copilot -- real st.chat_message interface with live tool-call status, Chapter 13, Étape 5.5.

**Real, not scripted**: every reply comes from
:func:`~src.agents.quant_copilot.run_copilot_turn_sync` -- the same
real tool-calling agent (4 real tools, direct REST to the configured
LLM provider, no fabricated responses) already built in Étape 5.1 and
used by ``ultimate_terminal.py``'s JARVIS gadget. This page is a
proper full-width chat surface for it, not a duplicate implementation.

**The "Status Spinner" requirement, done honestly**: the brief asks for
spinners shown "before the response appears" simulating tool-calling
steps. Built as a real, live :func:`streamlit.status` block updated by
:data:`~src.agents.quant_copilot.TOOL_STATUS_LABELS` through the new
``on_tool_call`` callback :func:`~src.agents.quant_copilot.
run_copilot_turn` now accepts -- each label appears exactly when that
real tool call actually fires, not a pre-scripted animation playing
regardless of what the LLM actually decided to do. A turn that needs no
tool (e.g. "bonjour") still shows one real step ("Interrogation du
LLM..."), because that's what actually happened.

**No LLM configured**: if ``GUARDIAN_LLM_PROVIDER`` is unset, replies
fall back to :func:`~src.agents.quant_copilot._fallback_reply`'s own
honest "no LLM configured" message -- shown as a normal assistant
message, not hidden or replaced with a fabricated answer.

**Alpha Swarm hand-off (Frontier Tech Injection, Pillar 3)**: real
proposals from :func:`~ai_agents.alpha_swarm.run_alpha_swarm_once` land
in :data:`~core.schemas.ALPHA_SWARM_PROPOSALS_REDIS_KEY` -- this page
reads that real Redis list and renders each as an Approve/Reject card,
same human-in-the-loop pattern as every other AI-touching surface in
this project (``guardian_ai.py``'s ``EmergencyAction`` confirmation,
``quant_copilot.py``'s ``adjust_algo_aggression`` proposal). Approving
here only removes the entry from the pending list and logs the
decision -- see the section below for exactly what it does and does
not do; it never submits a real order or promotes a strategy to live
trading on its own.

**Evolutionary Tuner hand-off**: identical real pattern for
:func:`~ai_agents.evolutionary_tuner.run_evolutionary_tuning_and_propose`'s
real CMA-ES-discovered parameter proposals
(:data:`~core.schemas.EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY``) --
see that module's own docstring on why this gate exists at all (a
trading system silently mutating its own live parameters overnight,
with no held-out validation, is a real, known overfitting risk this
project does not take on).
"""

from __future__ import annotations

import json

import streamlit as st

from core.config import get_settings
from core.schemas import ALPHA_SWARM_PROPOSALS_REDIS_KEY, EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY
from src.agents.quant_copilot import TOOL_STATUS_LABELS, run_copilot_turn_sync
from src.dashboard.ems_pages._shared import get_redis_client, safe_lrange
from src.dashboard.theme import status_badge

st.title("🤖 Quant Copilot")
st.caption("Agent conversationnel réel avec appel d'outils en direct -- Redis, statut des agents, TCA.")

settings = get_settings()
llm_configured = settings.guardian.llm_provider != "none" and settings.guardian.api_key is not None
st.markdown(
    status_badge("SAFE" if llm_configured else "PENDING", f"LLM: {settings.guardian.llm_provider.upper()}" if llm_configured else "AUCUN LLM CONFIGURÉ"),
    unsafe_allow_html=True,
)
if not llm_configured:
    st.info("Configure GUARDIAN_LLM_PROVIDER / GUARDIAN_API_KEY pour activer le raisonnement réel -- le Copilot répond en mode dégradé honnête en attendant.")

api_settings = settings.api_middleware
gateway_key = api_settings.api_keys.get_secret_value().split(",")[0].strip() if api_settings.api_keys else ""

if "copilot_chat_history" not in st.session_state:
    st.session_state.copilot_chat_history = []

for turn in st.session_state.copilot_chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Pose une question au Copilot (ex: pourquoi mon exécution BTC est lente ?)")

if question:
    st.session_state.copilot_chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Interrogation du LLM...", expanded=True)

        def _on_tool_call(name: str, arguments: dict[str, object]) -> None:
            label = TOOL_STATUS_LABELS.get(name, f"Appel de l'outil {name}...")
            status.write(label)
            status.update(label=label)

        reply, _updated_history = run_copilot_turn_sync(
            user_message=question,
            conversation_history=st.session_state.copilot_chat_history[:-1],
            gateway_base_url=api_settings.self_base_url,
            gateway_api_key=gateway_key,
            on_tool_call=_on_tool_call,
        )
        status.update(label="Réponse prête.", state="complete", expanded=False)
        st.markdown(reply)

    st.session_state.copilot_chat_history.append({"role": "assistant", "content": reply})

if st.session_state.copilot_chat_history and st.button("Effacer la conversation"):
    st.session_state.copilot_chat_history = []
    st.rerun()

st.markdown("---")
st.subheader("🐝 Alpha Swarm -- Propositions en attente")
st.caption(
    "Chaque proposition ci-dessous vient d'un vrai backtest sandboxé, exécuté sur les données "
    "réelles research/sample_data (voir ai_agents/alpha_swarm.py) -- Approuver/Rejeter ne fait que "
    "retirer l'entrée de cette liste Redis et journaliser la décision : aucune action ici ne soumet "
    "un ordre réel ni ne promeut une stratégie en production."
)

r = get_redis_client()
raw_proposals = safe_lrange(r, ALPHA_SWARM_PROPOSALS_REDIS_KEY, 20)
if not raw_proposals:
    st.info("Aucune proposition en attente -- ALPHA_SWARM_PROPOSALS est vide.")
else:
    for raw in raw_proposals:
        try:
            proposal = json.loads(raw)
        except json.JSONDecodeError:
            continue
        hypothesis = proposal["hypothesis"]
        metrics = proposal["metrics"]
        with st.container(border=True):
            st.markdown(f"**{hypothesis['strategy_name']}** -- source: `{hypothesis['source']}`")
            st.caption(hypothesis["rationale"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rendement total", f"{metrics['total_return_pct']:.2f}%")
            m2.metric("Sharpe (voir avertissement)", f"{metrics['sharpe_ratio']:.1f}" if metrics["sharpe_ratio"] is not None else "N/A")
            m3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
            m4.metric("Taux de réussite", f"{metrics['win_rate_pct']:.1f}%")
            st.caption(
                f"{metrics['num_trades']} trades sur {metrics['bars_used']} barres réelles -- Sharpe annualisé à "
                "partir d'une seule journée réelle de données, une amplitude élevée reflète un échantillon "
                "statistiquement mince, pas une estimation annuelle fiable (voir BacktestMetrics.sharpe_ratio)."
            )
            with st.expander("Code généré (exécuté en sandbox)"):
                st.code(hypothesis["generated_code"], language="python")
            col_approve, col_reject = st.columns(2)
            if col_approve.button("✅ Approuver", key=f"approve_{proposal['proposal_id']}", width="stretch"):
                r.lrem(ALPHA_SWARM_PROPOSALS_REDIS_KEY, 1, raw)
                st.toast(f"Proposition {hypothesis['strategy_name']} approuvée -- retirée de la file d'attente.", icon="✅")
                st.rerun()
            if col_reject.button("❌ Rejeter", key=f"reject_{proposal['proposal_id']}", width="stretch"):
                r.lrem(ALPHA_SWARM_PROPOSALS_REDIS_KEY, 1, raw)
                st.toast(f"Proposition {hypothesis['strategy_name']} rejetée.", icon="❌")
                st.rerun()

st.markdown("---")
st.subheader("🧬 Evolutionary Tuner -- Propositions de paramètres en attente")
st.caption(
    "Chaque proposition vient d'une vraie recherche CMA-ES, chaque candidat réellement backtesté en "
    "sandbox sur les données réelles research/sample_data (voir ai_agents/evolutionary_tuner.py) -- "
    "jamais appliqué automatiquement à une stratégie en production, exactement comme pour l'Alpha Swarm "
    "ci-dessus."
)

raw_tuning_proposals = safe_lrange(r, EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 20)
if not raw_tuning_proposals:
    st.info("Aucune proposition en attente -- EVOLUTIONARY_TUNING_PROPOSALS est vide.")
else:
    for raw in raw_tuning_proposals:
        try:
            tuning_proposal = json.loads(raw)
        except json.JSONDecodeError:
            continue
        baseline = tuning_proposal["baseline_metrics"]
        proposed = tuning_proposal["proposed_metrics"]
        with st.container(border=True):
            st.markdown(f"**{tuning_proposal['symbol']}** -- sma_crossover réglé par CMA-ES ({tuning_proposal['num_evaluations']} évaluations réelles)")
            col_baseline, col_proposed = st.columns(2)
            with col_baseline:
                st.caption(f"Référence : fast={tuning_proposal['baseline_params']['fast_window']}, slow={tuning_proposal['baseline_params']['slow_window']}")
                if baseline is not None:
                    st.metric("Sharpe référence", f"{baseline['sharpe_ratio']:.1f}" if baseline["sharpe_ratio"] is not None else "N/A")
            with col_proposed:
                st.caption(f"Proposé : fast={tuning_proposal['proposed_params']['fast_window']}, slow={tuning_proposal['proposed_params']['slow_window']}")
                if proposed is not None:
                    st.metric("Sharpe proposé", f"{proposed['sharpe_ratio']:.1f}" if proposed["sharpe_ratio"] is not None else "N/A")
            st.caption(
                "Sharpe calculé sur une seule journée réelle de données -- amplitude élevée non représentative "
                "d'une estimation annuelle (même avertissement que l'Alpha Swarm ci-dessus)."
            )
            col_approve, col_reject = st.columns(2)
            if col_approve.button("✅ Approuver", key=f"approve_tuning_{tuning_proposal['proposal_id']}", width="stretch"):
                r.lrem(EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 1, raw)
                st.toast(f"Proposition de réglage {tuning_proposal['symbol']} approuvée -- retirée de la file d'attente.", icon="✅")
                st.rerun()
            if col_reject.button("❌ Rejeter", key=f"reject_tuning_{tuning_proposal['proposal_id']}", width="stretch"):
                r.lrem(EVOLUTIONARY_TUNING_PROPOSALS_REDIS_KEY, 1, raw)
                st.toast(f"Proposition de réglage {tuning_proposal['symbol']} rejetée.", icon="❌")
                st.rerun()

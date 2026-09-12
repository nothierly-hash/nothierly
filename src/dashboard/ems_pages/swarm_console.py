"""Swarm Console -- real, live status of the Scout / Analyst / Risk Gate reactive pipeline.

**Real data only**: every section below reads directly from the same real Redis keys
``scout_agent.py``/``analyst_agent.py``/``risk_gate_agent.py`` themselves write -- heartbeats, the
real anomaly/analysis hashes, and the real, bounded approved-opportunities list. No section here
simulates a cascade; an honestly empty page (no anomalies detected yet, no approvals yet) is a real
possible state, not a bug.

**Real, disclosed scope, same as ``p2p_premium_scanner.py``'s own "not an execution guarantee"
caveat**: an approved opportunity here is a real, actionable alert for a human operator --
this project holds no real broker/execution capability for P2P/fiat-rail arbitrage (see
``risk_gate_agent.py``'s own module docstring), so nothing on this page ever claims a trade was
placed.
"""

from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.schemas import (
    HEARTBEAT_REDIS_KEY_PREFIX,
    SWARM_ANALYSIS_REDIS_KEY_PREFIX,
    SWARM_ANOMALY_REDIS_KEY_PREFIX,
    SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY,
    SWARM_TDA_SAGA_REDIS_KEY,
)
from src.dashboard.ems_pages._shared import get_redis_client, safe_keys, safe_lrange
from src.dashboard.theme import status_badge

_SWARM_AGENT_IDS = ("swarm-scout", "swarm-analyst", "swarm-risk-gate")


def _decode(raw: dict[Any, Any]) -> dict[str, str]:
    """Same real, defensive decode as risk_gate_agent.py's own -- see that module's own comment
    on this project's pinned fakeredis not always honoring decode_responses for HGETALL."""
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else k): (v.decode("utf-8") if isinstance(v, bytes) else v)
        for k, v in raw.items()
    }


st.title("🐝 Swarm Console -- Scout / Analyst / Risk Gate")
st.caption(
    "Pipeline réactif réel : détection d'anomalie statistique (Scout) → analyse LLM réelle "
    "(Analyst, réutilise political_agent.py) → filtre de liquidité réel (Risk Gate) → alerte "
    "actionnable pour un opérateur humain. Aucune exécution automatique -- voir le docstring de "
    "risk_gate_agent.py."
)

refresh_col, _ = st.columns([1, 3])
with refresh_col:
    refresh_seconds = st.select_slider("Rafraîchissement auto", options=[3, 5, 10, 30], value=5, format_func=lambda s: f"{s}s")
st_autorefresh(interval=refresh_seconds * 1000, key="swarm_console_autorefresh")

r = get_redis_client()

st.markdown("---")
st.subheader("Statut des agents")
cols = st.columns(3)
for col, agent_id in zip(cols, _SWARM_AGENT_IDS):
    raw_heartbeat = r.get(f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}")
    online = raw_heartbeat is not None
    age_seconds = time.time() - float(raw_heartbeat) if raw_heartbeat is not None else None
    with col:
        st.markdown(f"**{agent_id}**")
        st.markdown(
            status_badge("SAFE" if online else "PANIC", f"{age_seconds:.1f}s" if age_seconds is not None else "HORS LIGNE"),
            unsafe_allow_html=True,
        )

st.markdown("---")
st.subheader("🔍 Anomalies détectées récemment (Scout)")
anomaly_keys = safe_keys(r, f"{SWARM_ANOMALY_REDIS_KEY_PREFIX}*")
if not anomaly_keys:
    st.info("Aucune anomalie réelle détectée pour l'instant -- honnêtement vide, pas une erreur.")
else:
    anomaly_rows = []
    for key in anomaly_keys:
        record = _decode(r.hgetall(key))
        if not record:
            continue
        anomaly_rows.append({
            "ID": record.get("anomaly_id", "?")[:8],
            "Devise": record.get("fiat", "?"),
            "Prime actuelle (%)": record.get("latest_premium_percentage", "?"),
            "Baseline (%)": record.get("baseline_mean_percentage", "?"),
            "Z-score": record.get("zscore", "?"),
            "Détecté à": time.strftime("%H:%M:%S", time.localtime(float(record["detected_at"]))) if record.get("detected_at") else "?",
        })
    anomaly_rows.sort(key=lambda row: row["Détecté à"], reverse=True)
    st.dataframe(anomaly_rows, width="stretch", hide_index=True)

st.markdown("---")
st.subheader("🧠 Analyses réelles (Analyst)")
analysis_keys = safe_keys(r, f"{SWARM_ANALYSIS_REDIS_KEY_PREFIX}*")
if not analysis_keys:
    st.info("Aucune analyse réelle pour l'instant.")
else:
    analysis_rows = []
    for key in analysis_keys:
        record = _decode(r.hgetall(key))
        if not record:
            continue
        analysis_rows.append({
            "ID": record.get("anomaly_id", "?")[:8],
            "Devise": record.get("fiat", "?"),
            "Statut": record.get("status", "?"),
            "Score de crise": record.get("crisis_score", "?"),
            "Titre réel": (record.get("headline") or "--")[:80],
        })
    st.dataframe(analysis_rows, width="stretch", hide_index=True)

st.markdown("---")
st.subheader("✅ Opportunités approuvées (Risk Gate) -- alertes réelles pour un opérateur")
approved_raw = safe_lrange(r, SWARM_APPROVED_OPPORTUNITIES_REDIS_KEY, 100)
if not approved_raw:
    st.info(
        "Aucune opportunité approuvée pour l'instant -- il faut une crise réellement confirmée par "
        "le LLM ET une liquidité P2P réelle suffisante. Honnêtement vide, pas un défaut."
    )
else:
    approved_entries = []
    for raw in approved_raw:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        approved_entries.append(entry)
    st.dataframe(
        [
            {
                "Devise": e.get("fiat", "?"),
                "Liquidité réelle ($)": e.get("liquidity_usd", "?"),
                "Score de crise": e.get("crisis_score", "?"),
                "Sentiment": e.get("sentiment", "?"),
                "Résumé réel": e.get("event_summary", "?"),
                "Approuvé à": time.strftime("%H:%M:%S", time.localtime(e["approved_at"])) if e.get("approved_at") else "?",
            }
            for e in approved_entries
        ],
        width="stretch", hide_index=True,
    )

st.markdown("---")
st.subheader("🌐 Propositions de cycle TDA (Genesis) -- pour revue humaine, aucune exécution automatique")
st.caption(
    "Traduction réelle du cycle détecté par nothierly_genesis/tda_engine.py + Bellman-Ford en proposition "
    "structurée (nothierly_genesis/saga_proposal_generator.py) -- le cycle sous-jacent reste celui du "
    "moteur de démonstration (non calibré sur données live), et la liquidité de goulot d'étranglement est "
    "réelle quand elle existe (scan P2P réel), honnêtement absente sinon."
)
saga_raw = safe_lrange(r, SWARM_TDA_SAGA_REDIS_KEY, 50)
if not saga_raw:
    st.info("Aucune proposition générée pour l'instant -- lancez `python -m nothierly_genesis.saga_proposal_generator`.")
else:
    saga_entries = []
    for raw in saga_raw:
        try:
            saga_entries.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    st.dataframe(
        [
            {
                "Cycle": " -> ".join(e.get("cycle", [])),
                "Profit net attendu (%)": e.get("expected_net_profit_pct", "?"),
                "Goulot d'étranglement ($)": e.get("bottleneck_liquidity_usd") if e.get("bottleneck_liquidity_usd") is not None else "Aucune donnée réelle",
                "Legs sans donnée réelle": ", ".join(e.get("legs_missing_real_liquidity_data", [])) or "—",
                "Statut": e.get("status", "?"),
                "Généré à": time.strftime("%H:%M:%S", time.localtime(e["generated_at"])) if e.get("generated_at") else "?",
            }
            for e in saga_entries
        ],
        width="stretch", hide_index=True,
    )

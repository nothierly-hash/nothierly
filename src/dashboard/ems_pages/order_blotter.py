"""Order Blotter & TCA -- real fills from ExecutionAgent, Chapter 13, Étape 5.5.

**Real, newly-persisted data**: every row is read from
:data:`~core.schemas.FILLS_BLOTTER_REDIS_KEY`, a Redis list
:meth:`~src.agents.execution.agent.ExecutionAgent._log_fill` now writes
on every real entry and exit (added alongside this page specifically so
it has real data, not a stub -- see that method's own docstring). Empty
until at least one real (or ``MOCK_MODE``-simulated) fill has happened
in this running session; that's the honest state, not backfilled. Two
more real writers share this same list: ``api_middleware/gateway.py``'s
``_write_shadow_fill`` (client SHADOW simulations) and
:class:`~src.agents.macro_order_execution.agent.MacroOrderExecutionAgent`
(real, client-attributed LIVE fills -- see that module's own docstring).

**Three real modes, not two**: ``Mode`` distinguishes ``MOCK`` (the
process-wide ``MOCK_MODE`` setting), ``SHADOW`` (this one agent was
individually paused into ``AgentControlState.SHADOW`` by an operator,
via ``src/dashboard/ems_pages/agent_fleet.py`` -- see that field's own
docstring on ``ExecutionAgent._log_fill`` for why it's a distinct,
separately-named concept from ``api_middleware/gateway.py``'s
unrelated ``ExecutionMode.SHADOW``), and ``RÉEL``.

**A real correction to the brief, stated upfront**: the brief asks for
an "Action" column of "Maker Skew / Taker Sweep" -- that vocabulary is
real, but only inside ``research/market_making_env.py``'s RL
market-making environment (Chapter 12 research, never wired into
production). :class:`~src.agents.execution.agent.ExecutionAgent` only
ever places simple market orders -- there is no maker/taker choice in
this production path to report on. This remains a real, permanent
architectural fact, not a gap this page's own metrics can close.

**"Toxic Orders Evaded", now a real counter, not a demo-only one**:
:class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent` now reuses ``execution.predator_honeypot``'s
real, already-tested ``AdverseFlowDefenseAgent``, fed real, live VPIN
(see that agent's own module docstring), to actually reduce or refuse a
real ``TOXICITY_AVOIDANCE`` order under detected toxic conditions --
:data:`~core.schemas.TOXIC_ORDERS_EVADED_REDIS_KEY` is incremented
every real time that happens. Real, but honestly scoped: only orders a
client explicitly tags ``execution_algo=TOXICITY_AVOIDANCE`` are ever
subject to this -- other algos never have a behavior their client
didn't ask for silently applied to them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from core.schemas import FILLS_BLOTTER_REDIS_KEY, TOXIC_ORDERS_EVADED_REDIS_KEY
from src.dashboard.ems_pages._shared import demo_mode_toggle, get_redis_client, not_yet_implemented, safe_get, safe_lrange

_ACTION_LABELS = {"ENTRY_LONG": "🟢 Entrée Long", "ENTRY_SHORT": "🔴 Entrée Short", "EXIT": "⚪ Sortie"}

st.title("📋 Order Blotter & TCA")
st.caption("Journal réel des fills exécutés par l'Agent d'Exécution -- pas de concept Maker/Taker en production (voir ci-dessous).")

r = get_redis_client()
raw_entries = safe_lrange(r, FILLS_BLOTTER_REDIS_KEY, 200)

rows: list[dict[str, object]] = []
latencies: list[float] = []
for raw in raw_entries:
    try:
        entry = json.loads(raw)
        latencies.append(float(entry["latency_ms"]))
        rows.append({
            "Horodatage": datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Agent ID": entry["agent_id"],
            "Instrument": entry["instrument"],
            "Action": _ACTION_LABELS.get(entry["action"], entry["action"]),
            "Prix": round(entry["price"], 6),
            "Unités": round(entry["units"], 6),
            "Latence (ms)": round(entry["latency_ms"], 2),
            "Raison": entry.get("reason") or "--",
            "Mode": "MOCK" if entry.get("mock_mode") else ("SHADOW" if entry.get("agent_control_shadow") else "RÉEL"),
        })
    except (json.JSONDecodeError, KeyError, TypeError):
        continue

demo_on = demo_mode_toggle(key="order_blotter_demo_mode")

raw_toxic_evaded = safe_get(r, TOXIC_ORDERS_EVADED_REDIS_KEY)
toxic_evaded_count = int(raw_toxic_evaded) if raw_toxic_evaded is not None else 0

if rows or not demo_on:
    st.subheader("Métriques agrégées réelles")
    if rows:
        total = len(rows)
        mock_count = sum(1 for row in rows if row["Mode"] == "MOCK")
        shadow_count = sum(1 for row in rows if row["Mode"] == "SHADOW")
        avg_latency = sum(latencies) / total
        entries_count = sum(1 for row in rows if row["Action"] != _ACTION_LABELS["EXIT"])
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Fills", total)
        col2.metric("Entrées / Sorties", f"{entries_count} / {total - entries_count}")
        col3.metric("Latence moyenne", f"{avg_latency:.1f} ms")
        col4.metric("Simulés (MOCK + SHADOW)", f"{mock_count + shadow_count} / {total}")
        col5.metric("Toxic Orders Evaded (réel)", toxic_evaded_count)
    else:
        st.info("Aucun fill réel enregistré pour l'instant dans cette session -- NOTHIERLY_FILLS_BLOTTER est vide.")
        st.metric("Toxic Orders Evaded (réel)", toxic_evaded_count)

    st.markdown("---")
    st.subheader("Journal réel des fills")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)
    else:
        st.caption("Aucune ligne à afficher -- active le Mode Démo ci-dessus pour visualiser la mise en page avec des données simulées.")

if demo_on:
    st.markdown("---")
    st.subheader("Aperçu Démo (données simulées, présentation B2B)")
    st.caption(
        "Simule le concept de recherche Maker Skew / Taker Sweep (research/market_making_env.py) -- "
        "jamais présent dans les données réelles ci-dessus, qui ne connaissent que ENTRY/EXIT."
    )
    rng = np.random.default_rng(42)
    n_demo = 40
    demo_actions = rng.choice(["Maker Skew (Bid)", "Maker Skew (Ask)", "Taker Sweep"], size=n_demo, p=[0.4, 0.4, 0.2])
    demo_agents = rng.choice(["rl-signal-agent", "supervisor-agent"], size=n_demo)
    demo_instruments = rng.choice(["BTC/USDT", "EUR_USD", "XAU_USD"], size=n_demo)
    base_prices = {"BTC/USDT": 65000.0, "EUR_USD": 1.085, "XAU_USD": 2380.0}
    demo_rows = []
    for i in range(n_demo):
        instrument = demo_instruments[i]
        price = base_prices[instrument] * (1 + rng.normal(0, 0.0006))
        slippage_saved = round(abs(rng.normal(1.8, 0.9)), 2) if demo_actions[i] != "Taker Sweep" else 0.0
        demo_rows.append({
            "Horodatage": (datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Agent ID": demo_agents[i],
            "Instrument": instrument,
            "Action (démo)": demo_actions[i],
            "Prix": round(price, 4),
            "Latence (ms)": round(abs(rng.normal(45, 20)), 1),
            "Slippage Sauvé ($, démo)": slippage_saved,
        })
    demo_df = pd.DataFrame(demo_rows)
    st.dataframe(demo_df, width="stretch", hide_index=True, height=380)

    maker_pct = (demo_df["Action (démo)"] != "Taker Sweep").mean() * 100
    toxic_evaded = int((demo_df["Action (démo)"] == "Maker Skew (Bid)").sum() * 0.3)
    col1, col2 = st.columns(2)
    col1.metric("Total Maker Fills (%, démo)", f"{maker_pct:.1f}%")
    col2.metric("Toxic Orders Evaded (Count, démo)", toxic_evaded)

st.markdown("---")
not_yet_implemented(
    "Maker Skew / Taker Sweep en production",
    reason="ExecutionAgent ne place que des ordres au marché (market orders) -- ce concept existe uniquement dans research/market_making_env.py (Chapitre 12, recherche RL hors-ligne). Fait architectural réel et permanent, pas un gap de cette page.",
)

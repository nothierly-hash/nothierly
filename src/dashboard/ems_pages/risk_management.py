"""Risk Management & Overrides -- the real risk-per-trade slider + global Kill Switch.

**Real, and reuses the existing Guardian-gated safety flow**: writes
the same real ``TRADING_RISK_PCT`` Redis key
:class:`~src.agents.execution.agent.ExecutionAgent` reads live before
every order (no restart required) -- exactly the mechanism
``src/dashboard/app.py``'s sidebar already implements. A risk increase
above :data:`_RISK_GATE_THRESHOLD_PCT` is gated through
:func:`~src.agents.guardian_ai.assess_risk_sync` (trigger
``"RISK_INCREASE"``) before being written, matching that established
safety precedent rather than a simpler, less-safe reimplementation.

**Taille de position maximale, now real**: ``TRADING_MAX_POSITION_UNITS``
was declared here, unused, before this -- :meth:`~src.agents.execution.
agent.ExecutionAgent.calculate_position_size` now reads it live (same
"dashboard writes, agent reads, no restart" pattern) and caps the
risk-derived size, when a real positive value is configured. A single
global cap in instrument units, not per-instrument -- a real, disclosed
scope limit (BTC-sized vs. forex-lot-sized instruments share one number)
kept simple rather than adding per-instrument config this project has
no other precedent for.

**Not implemented here**: a "Hurdle Rate" control (the brief's own
term) has no real backing anywhere in this codebase --
:class:`~src.agents.execution.agent.ExecutionAgent`'s entry rule is a
binary two-agent consensus (see that module's own docstring), not an
expected-value estimate a hurdle rate could gate against. Building one
for real would mean inventing a new expected-return model this project
doesn't have anywhere, not wiring an existing one -- shown as an honest
gap, not a fabricated threshold compared against nothing real.
"""

from __future__ import annotations

import streamlit as st

from core.security import STATUS_PANIC, STATUS_SAFE, SYSTEM_STATUS_KEY, reset_kill_switch, trip_kill_switch
from src.agents.execution.agent import TRADING_MAX_POSITION_UNITS_REDIS_KEY
from src.agents.guardian_ai import assess_risk_sync
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_get
from src.dashboard.theme import status_badge

_RISK_PCT_REDIS_KEY = "TRADING_RISK_PCT"
_DEFAULT_RISK_PCT = 1.0
_RISK_GATE_THRESHOLD_PCT = 2.0

st.title("⚙️ Risk Management & Overrides")
st.caption("Limites de risque en direct -- lues par l'Agent d'Exécution avant chaque ordre, sans redémarrage.")

r = get_redis_client()

system_status = safe_get(r, SYSTEM_STATUS_KEY)
is_panic = system_status == STATUS_PANIC
st.markdown(f"**SYSTEM_STATUS:** {status_badge('PANIC' if is_panic else 'SAFE')}", unsafe_allow_html=True)

col_kill, col_reset = st.columns(2)
with col_kill:
    if st.button("🚨 KILL SWITCH -- ARRÊT GLOBAL", type="primary", disabled=is_panic):
        import asyncio
        asyncio.run(trip_kill_switch("Déclenché manuellement depuis Risk Management & Overrides."))
        st.rerun()
with col_reset:
    if st.button("Réinitialiser (SAFE)", disabled=not is_panic):
        import asyncio
        asyncio.run(reset_kill_switch())
        st.rerun()

st.markdown("---")
st.subheader("Risque par trade (Money Management)")

raw_risk_pct = safe_get(r, _RISK_PCT_REDIS_KEY)
try:
    current_risk_pct = float(raw_risk_pct) * 100.0 if raw_risk_pct else _DEFAULT_RISK_PCT
except ValueError:
    current_risk_pct = _DEFAULT_RISK_PCT

new_risk_pct = st.slider(
    "Risque par trade (% du capital)", min_value=0.1, max_value=5.0, value=current_risk_pct, step=0.1,
    help="Fraction du capital risquée sur un seul trade -- l'Agent d'Exécution lit cette valeur en direct.",
)

if new_risk_pct != current_risk_pct:
    if new_risk_pct > _RISK_GATE_THRESHOLD_PCT:
        pending = st.session_state.get("risk_pct_pending")
        if pending is None or pending.get("value") != new_risk_pct:
            alert = assess_risk_sync(
                trigger="RISK_INCREASE",
                summary=f"Risque par trade porté à {new_risk_pct:.1f}% (seuil : {_RISK_GATE_THRESHOLD_PCT:.0f}%)",
                context="Ajustement demandé depuis la page Risk Management & Overrides du Master Control Terminal.",
            )
            st.session_state["risk_pct_pending"] = {"value": new_risk_pct, "verdict": alert.verdict}
            st.rerun()
    else:
        r.set(_RISK_PCT_REDIS_KEY, str(round(new_risk_pct / 100.0, 4)))
        st.session_state["risk_pct_pending"] = None
        st.toast(f"Risque par trade mis à jour : {new_risk_pct:.1f}%", icon="✅")

pending_risk = st.session_state.get("risk_pct_pending")
if pending_risk is not None:
    st.error(f"⚠️ **Alerte CRO Nothierly**\n\n{pending_risk['verdict']}")
    if st.button("Je comprends les risques, appliquer quand même"):
        r.set(_RISK_PCT_REDIS_KEY, str(round(pending_risk["value"] / 100.0, 4)))
        st.session_state["risk_pct_pending"] = None
        st.rerun()

st.markdown("---")
st.subheader("Taille de position maximale (plafond réel, indépendant)")
st.caption(
    "Réel : ExecutionAgent.calculate_position_size lit ce plafond en direct et l'applique après le "
    "dimensionnement par risque ci-dessus -- 0 (ou vide) signifie non plafonné, le comportement d'origine."
)

raw_max_units = safe_get(r, TRADING_MAX_POSITION_UNITS_REDIS_KEY)
try:
    current_max_units = float(raw_max_units) if raw_max_units else 0.0
except ValueError:
    current_max_units = 0.0

new_max_units = st.number_input(
    "Plafond de taille de position (unités d'instrument, 0 = illimité)",
    min_value=0.0, value=current_max_units, step=0.01, format="%.4f",
)
if new_max_units != current_max_units:
    if new_max_units > 0:
        r.set(TRADING_MAX_POSITION_UNITS_REDIS_KEY, str(new_max_units))
        st.toast(f"Plafond de position mis à jour : {new_max_units:g} unités.", icon="✅")
    else:
        r.delete(TRADING_MAX_POSITION_UNITS_REDIS_KEY)
        st.toast("Plafond de position retiré -- dimensionnement non plafonné.", icon="✅")

st.markdown("---")
not_yet_implemented(
    "Hurdle Rate configurable",
    reason=(
        "L'entrée d'ExecutionAgent est un consensus binaire à deux agents (voir le docstring du module), "
        "pas une estimation de valeur attendue -- il n'existe aucun chiffre réel à comparer à un hurdle rate "
        "aujourd'hui. Construire ça pour de vrai nécessiterait un nouveau modèle de rendement attendu, "
        "pas un simple branchement."
    ),
)

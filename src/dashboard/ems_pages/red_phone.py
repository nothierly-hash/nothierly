"""SRE & Red Phone -- real Twilio config, live-editable triggers, recipient CRUD, and a real Kill Switch.

**Real data**: reads :class:`~core.config.TwilioSettings` (never
displays the auth token/SID values -- only whether they're set) and
:data:`~src.agents.alert_agent._ALERT_HISTORY_REDIS_KEY`
(``RED_PHONE_ALERT_LOG``), the real alert history
:meth:`~src.agents.alert_agent.AlertAgent._log_alert_history` writes.

**Now live-editable, not just read-only** (Chapter 13, Étape 5.5): the
OBI catastrophic threshold is backed by
:data:`~src.agents.alert_agent.OBI_THRESHOLD_OVERRIDE_REDIS_KEY`, read
fresh by :meth:`~src.agents.alert_agent.AlertAgent.
_get_dynamic_obi_threshold` on every tick -- the same "dashboard
writes, agent reads live, no restart" pattern
``TRADING_RISK_PCT``/Risk Management already established. Extra SMS
recipients are a real Redis SET
(:data:`~src.agents.alert_agent.RECIPIENTS_REDIS_KEY`) actually read by
:meth:`~src.agents.alert_agent.AlertAgent._get_alert_recipients` before
every real send -- adding a number here means it actually receives the
next real alert. Infrastructure/debounce thresholds stay read-only
(shown below): no live-override mechanism exists for those yet, stated
honestly rather than half-wired.

**Max Drawdown slider, now real**: the 24h drawdown trigger in
:class:`~src.agents.execution.agent.ExecutionAgent`
(``_check_drawdown_trigger``) used to be a compiled-in 5% constant with
no live override -- it now reads
:data:`~src.agents.execution.agent.DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY`
fresh on every real balance observation, the same "dashboard writes,
agent reads live, no restart" pattern the OBI slider above and
``TRADING_RISK_PCT`` (Risk Management page) already establish. The
re-arm hysteresis threshold scales proportionally with whatever value
is set here (see that module's own docstring on why it is not an
independently fixed second slider).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.config import get_settings
from core.security import STATUS_PANIC, SYSTEM_STATUS_KEY, reset_kill_switch, trip_kill_switch
from src.agents.alert_agent import (
    DEBOUNCE_TTL_SECONDS,
    INFRA_MISSING_HEARTBEAT_STRIKES,
    OBI_CATASTROPHIC_THRESHOLD,
    OBI_THRESHOLD_OVERRIDE_REDIS_KEY,
    RECIPIENTS_REDIS_KEY,
    TICK_INTERVAL_SECONDS,
)
from src.agents.execution.agent import DRAWDOWN_ALERT_THRESHOLD_PCT, DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY
from src.dashboard.ems_pages._shared import get_redis_client, safe_get, safe_lrange
from src.dashboard.theme import status_badge

_ALERT_HISTORY_REDIS_KEY = "RED_PHONE_ALERT_LOG"

st.title("🚒 SRE & Red Phone")
st.caption("Configuration des triggers de Flash Crash, destinataires SMS, et arrêt d'urgence global.")

settings = get_settings()
twilio = settings.twilio
r = get_redis_client()

system_status = safe_get(r, SYSTEM_STATUS_KEY)
is_panic = system_status == STATUS_PANIC
st.markdown(f"**SYSTEM_STATUS:** {status_badge('PANIC' if is_panic else 'SAFE')}", unsafe_allow_html=True)
col_kill, col_reset = st.columns(2)
with col_kill:
    if st.button("🚨 TRIGGER GLOBAL KILL-SWITCH", type="primary", disabled=is_panic, width="stretch"):
        asyncio.run(trip_kill_switch("Déclenché manuellement depuis SRE & Red Phone."))
        st.rerun()
with col_reset:
    if st.button("✅ Réactiver (SAFE)", disabled=not is_panic, width="stretch"):
        asyncio.run(reset_kill_switch())
        st.rerun()
st.caption(
    "Appelle le vrai core.security.trip_kill_switch / reset_kill_switch (identiques aux boutons de la page "
    "Risk Management) -- le kill-switch bloque immédiatement tout nouvel ordre, système entier ; la "
    "réactivation est une action humaine explicite, jamais automatique (voir reset_kill_switch)."
)

st.markdown("---")
st.subheader("Configuration Twilio")
configured = all([twilio.account_sid, twilio.auth_token, twilio.phone_number, twilio.admin_phone_number])
st.markdown(status_badge("SAFE" if configured else "PENDING", "CONFIGURÉ" if configured else "NON CONFIGURÉ"), unsafe_allow_html=True)
col1, col2 = st.columns(2)
col1.metric("Numéro d'envoi (Twilio)", twilio.phone_number or "non configuré")
col2.metric("Numéro destinataire admin (.env)", twilio.admin_phone_number or "non configuré")
if not configured:
    st.info("Configure TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER / ADMIN_PHONE_NUMBER dans .env.")

st.markdown("---")
st.subheader("Destinataires SMS additionnels")
st.caption("Réel : chaque numéro ajouté ici reçoit effectivement la prochaine alerte SMS déclenchée, en plus du numéro admin ci-dessus.")

extra_recipients = sorted(str(n) for n in (r.smembers(RECIPIENTS_REDIS_KEY) or set()))
if extra_recipients:
    for number in extra_recipients:
        rec_col, del_col = st.columns([4, 1])
        rec_col.markdown(f"📱 `{number}`")
        if del_col.button("Retirer", key=f"remove_recipient_{number}"):
            r.srem(RECIPIENTS_REDIS_KEY, number)
            st.rerun()
else:
    st.caption("Aucun destinataire additionnel -- seul le numéro admin (.env) reçoit les alertes.")

with st.form("add_recipient_form", clear_on_submit=True):
    new_number = st.text_input("Ajouter un numéro (format E.164, ex: +15551234567)")
    if st.form_submit_button("Ajouter") and new_number.strip():
        candidate = new_number.strip()
        if candidate.startswith("+") and candidate[1:].isdigit() and len(candidate) >= 8:
            r.sadd(RECIPIENTS_REDIS_KEY, candidate)
            st.toast(f"Destinataire ajouté : {candidate}", icon="✅")
            st.rerun()
        else:
            st.error("Format invalide -- attendu : + suivi de l'indicatif et du numéro (E.164), ex: +15551234567.")

st.markdown("---")
st.subheader("Seuils de déclenchement")

st.markdown("**Microstructure (OBI catastrophique) -- éditable en direct**")
raw_obi_override = safe_get(r, OBI_THRESHOLD_OVERRIDE_REDIS_KEY)
try:
    current_obi_threshold = float(raw_obi_override) if raw_obi_override else OBI_CATASTROPHIC_THRESHOLD
except ValueError:
    current_obi_threshold = OBI_CATASTROPHIC_THRESHOLD

new_obi_threshold = st.slider(
    "Seuil de Flash Crash (|OBI| ≥ X)", min_value=0.5, max_value=0.99, value=current_obi_threshold, step=0.01,
    help="AlertAgent lit cette valeur en direct à chaque tick (~2s) -- aucun redémarrage requis.",
)
if new_obi_threshold != current_obi_threshold:
    r.set(OBI_THRESHOLD_OVERRIDE_REDIS_KEY, str(round(new_obi_threshold, 4)))
    st.toast(f"Seuil OBI mis à jour : {new_obi_threshold:.2f}", icon="✅")

st.markdown("**Max Drawdown (24h) -- éditable en direct**")
raw_drawdown_override = safe_get(r, DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY)
try:
    current_drawdown_threshold = float(raw_drawdown_override) if raw_drawdown_override else DRAWDOWN_ALERT_THRESHOLD_PCT
except ValueError:
    current_drawdown_threshold = DRAWDOWN_ALERT_THRESHOLD_PCT

new_drawdown_threshold = st.slider(
    "Seuil d'alerte Drawdown (perte de capital sur 24h glissantes)",
    min_value=0.01, max_value=0.5, value=current_drawdown_threshold, step=0.01, format="%.0f%%",
    help=(
        "ExecutionAgent lit cette valeur en direct avant chaque ordre -- aucun redémarrage requis. "
        "Le seuil de réarmement (hystérésis) suit proportionnellement à 60% de cette valeur."
    ),
)
if new_drawdown_threshold != current_drawdown_threshold:
    r.set(DRAWDOWN_THRESHOLD_OVERRIDE_REDIS_KEY, str(round(new_drawdown_threshold, 4)))
    st.toast(f"Seuil de drawdown mis à jour : {new_drawdown_threshold:.0%}", icon="✅")

st.caption("**Seuils suivants : réels, lecture seule** (pas encore de mécanisme de surcharge Redis pour ceux-ci).")
threshold_rows = [
    {"Danger": "Infrastructure (heartbeat manquant)", "Seuil": f"{INFRA_MISSING_HEARTBEAT_STRIKES} cycles consécutifs (~{INFRA_MISSING_HEARTBEAT_STRIKES * TICK_INTERVAL_SECONDS:.0f}s)"},
    {"Danger": "Anti-spam (debounce)", "Seuil": f"1 alerte / catégorie / {DEBOUNCE_TTL_SECONDS // 60} minutes"},
]
st.dataframe(pd.DataFrame(threshold_rows), width="stretch", hide_index=True)

st.markdown("---")
st.subheader("Historique réel des alertes")
entries = safe_lrange(r, _ALERT_HISTORY_REDIS_KEY, 20)
if not entries:
    st.info("Aucune alerte enregistrée -- RED_PHONE_ALERT_LOG est vide (aucun danger réel détecté récemment).")
else:
    parsed = []
    for raw in entries:
        try:
            entry = json.loads(raw)
            parsed.append({
                "Horodatage": datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Catégorie": entry["category"], "Détail": entry["detail"],
                "SMS envoyé": "✅" if entry.get("sms_sent") else "❌",
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    st.dataframe(pd.DataFrame(parsed), width="stretch", hide_index=True)

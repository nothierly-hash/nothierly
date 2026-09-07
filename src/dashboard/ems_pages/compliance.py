"""Compliance & Audit Logs -- real, aggregated view of every real decision log this system already keeps.

**Real data, honestly framed**: what's on *this* page are two Redis
lists aggregated chronologically: ``NOTHIERLY_GUARDIAN_ALERTS`` (every
real Guardian AI verdict -- risk increases, manual overrides,
drawdown, signal collisions) and ``RED_PHONE_ALERT_LOG`` (every real
circuit-breaker trip). Together they answer a real subset of "why did
the algo do that at 14:02" -- every *risk-gated* and *emergency*
decision. Neither list is actually immutable (a plain Redis
``LTRIM``-capped list, not an append-only/write-once store) -- stated
plainly, since "immutable" is a real compliance requirement, not a
cosmetic one.

**The routine order-by-order gap this docstring used to describe is
now closed, elsewhere**: :mod:`~src.execution.audit_ledger`'s real,
SHA-256 hash-chained ``order_audit_trail`` table now records every
real ``SUBMITTED``/``EXECUTED``/``REJECTED`` order event -- see the
"Order Audit Trail" page (:mod:`src.dashboard.ems_pages.audit_trail`)
and that module's own docstring for its real tamper-*evidence* (not
tamper-*prevention*) guarantee. This page's own two Redis lists remain
what they always were: real, but not immutable, and scoped to
risk/emergency decisions, not routine execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.dashboard.ems_pages._shared import get_redis_client, safe_lrange

_GUARDIAN_ALERTS_REDIS_KEY = "NOTHIERLY_GUARDIAN_ALERTS"
_RED_PHONE_ALERT_LOG_KEY = "RED_PHONE_ALERT_LOG"

st.title("📜 Compliance & Audit Logs")
st.caption(
    "Journal agrégé des décisions de risque réelles. Pour le registre d'ordres chaîné cryptographiquement, "
    "voir la page \"Order Audit Trail\"."
)

st.warning(
    "⚠️ Ces journaux sont des listes Redis plafonnées (LTRIM), pas un magasin immuable/append-only. "
    "Une conformité réglementaire stricte nécessiterait une persistance write-once (ex: TimescaleDB en mode append-only, ou un stockage objet WORM)."
)

r = get_redis_client()

rows: list[dict[str, str]] = []
for raw in safe_lrange(r, _GUARDIAN_ALERTS_REDIS_KEY, 30):
    try:
        entry = json.loads(raw)
        rows.append({
            "Horodatage": datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Source": "Guardian AI", "Type": entry["trigger"], "Détail": entry["summary"],
        })
    except (json.JSONDecodeError, KeyError, TypeError):
        continue

for raw in safe_lrange(r, _RED_PHONE_ALERT_LOG_KEY, 30):
    try:
        entry = json.loads(raw)
        rows.append({
            "Horodatage": datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Source": "Red Phone", "Type": entry["category"], "Détail": entry["detail"],
        })
    except (json.JSONDecodeError, KeyError, TypeError):
        continue

rows.sort(key=lambda row: row["Horodatage"], reverse=True)

if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
else:
    st.info("Aucune décision de risque enregistrée récemment (les deux journaux réels sont vides).")

st.markdown("---")
st.success(
    "✅ **Audit trail complet ordre-par-ordre, chaîné cryptographiquement** -- désormais réel, voir la page "
    "\"Order Audit Trail\" (SUBMITTED/EXECUTED/REJECTED, chaînage SHA-256, ancre Redis indépendante). "
    "Preuve d'altération réelle -- pas une immutabilité write-once absolue au sens strict (voir le docstring "
    "de src/execution/audit_ledger.py pour la nuance exacte)."
)

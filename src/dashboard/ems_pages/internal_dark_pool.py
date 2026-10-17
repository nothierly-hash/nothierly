"""Internal Dark Pool -- real, observation-only shadow-crossing matches.

**Real data, honestly scoped**: every row here comes from
:data:`~core.schemas.DARK_POOL_SHADOW_MATCHES_REDIS_KEY`, written by
:class:`~src.execution.shadow_crossing_engine.ShadowCrossingEngine` --
see that module's own docstring for the full, deliberately bounded
design: it observes real orders on :data:`~core.schemas.
CLIENT_ORDERS_TOPIC` and asks the real, already-tested
:class:`~core.crossing_network.CrossingNetwork` whether two of them
*could* have crossed internally, but **never intercepts, delays, or
reroutes a real order** -- every order still goes through the real
pipeline exactly as before. "Volume internalisé" and "Frais évités"
below are real, honest *shadow* figures (what could have been saved,
had this engine been authorized to actually internalize flow -- see
that module's docstring on why it is not), not a claim that any real
capital was ever crossed here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

import streamlit as st

from core.schemas import DARK_POOL_SHADOW_MATCHES_REDIS_KEY
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_lrange

st.title("🌑 Internal Dark Pool -- Shadow Crossing")
st.caption(
    "Observation réelle, non-exécutoire : voir src/execution/shadow_crossing_engine.py pour la portée "
    "exacte et pourquoi aucun ordre réel n'est jamais intercepté ou redirigé ici."
)

r = get_redis_client()
raw_entries = safe_lrange(r, DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 200)

matches = []
for raw in raw_entries:
    try:
        matches.append(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        continue

if not matches:
    not_yet_implemented(
        "Dark Pool -- correspondances réelles",
        reason=(
            "Aucune correspondance shadow réelle enregistrée pour l'instant -- ShadowCrossingEngine doit "
            "tourner en direct et observer au moins deux ordres réels opposés sur le même instrument "
            "(via CLIENT_ORDERS_TOPIC) pour peupler cette page avec des données réelles."
        ),
    )
else:
    total_internalized_volume = sum(Decimal(m["quantity"]) for m in matches)
    total_fees_saved = sum(float(m["fee_saved_usd"]) for m in matches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Volume internalisé (réel, shadow)", f"{total_internalized_volume:g}")
    col2.metric("Frais d'exchange évités (réel, shadow)", f"${total_fees_saved:,.2f}")
    col3.metric("Correspondances détectées", len(matches))
    st.caption(
        "Réel : chaque correspondance vient d'un vrai croisement au prix mid réel, calculé par "
        "core.crossing_network.CrossingNetwork -- jamais un ordre réellement exécuté ici (voir le "
        "docstring du module pour la distinction exacte)."
    )

    st.markdown("---")
    st.subheader("Live Matches -- correspondances réelles récentes")
    rows = []
    for match in matches[:100]:
        timestamp_str = datetime.fromtimestamp(float(match["timestamp"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        rows.append(
            {
                "Horodatage": timestamp_str,
                "Instrument": match["instrument"],
                "Quantité": match["quantity"],
                "Prix mid réel": round(float(match["price"]), 6),
                "Acheteur (client)": match["buy_client_id"],
                "Vendeur (client)": match["sell_client_id"],
                "Frais évités ($)": round(float(match["fee_saved_usd"]), 2),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True, height=420)

st.markdown("---")
not_yet_implemented(
    "Internalisation réelle (exécution effective)",
    reason=(
        "Croiser réellement le flux client nécessite une autorisation réglementaire réelle (Systematic "
        "Internaliser / MTF / OTF en UE, ATS aux US) que ce projet ne détient pas dans son modèle "
        "actuel -- voir le docstring de core/crossing_network.py. Cette page reste volontairement "
        "une observation, jamais une exécution."
    ),
)

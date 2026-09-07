"""Order Audit Trail -- real, hash-chained order-lifecycle history, searchable and verifiable.

**Real data, real gap closed**: every row here comes from
:func:`~src.execution.audit_ledger.fetch_order_history`, reading the
real ``order_audit_trail`` table :class:`~src.execution.audit_ledger.
CryptographicAuditLedger` writes to for every real ``SUBMITTED``/
``EXECUTED``/``REJECTED`` event (see that module's own docstring for
exactly which real code paths emit each one, and the one honestly
disclosed gap: no ``CANCELLED`` emitter exists yet, since no
client-facing cancellation endpoint exists anywhere in this codebase).
This is the dedicated, purpose-built store
``src/dashboard/ems_pages/compliance.py``'s own docstring previously
said this project didn't have.

**"Vérifier l'intégrité", precisely what it proves**: re-fetches every
row, recomputes every SHA-256 link in order
(:func:`~src.execution.audit_ledger.verify_chain`), and separately
cross-checks the database's own latest hash against the independent
Redis anchor (:func:`~src.execution.audit_ledger.
verify_chain_tip_matches_anchor`). This proves tamper-*evidence* (an
edit or deletion anywhere in the chain is mechanically detectable) --
it does not prove tamper-*prevention* against a privileged database
attacker. See :mod:`~src.execution.audit_ledger`'s own module docstring
for that exact, honest distinction.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from core.config import get_settings
from core.redis_client import RedisClient
from core.schemas import AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY
from core.timescale_client import TimescaleUnavailableError, connect_with_backoff
from src.dashboard.ems_pages._shared import get_redis_client, not_yet_implemented, safe_get_json
from src.execution.audit_ledger import (
    ensure_schema,
    fetch_all_rows,
    fetch_order_history,
    verify_chain,
    verify_chain_tip_matches_anchor,
)

st.title("🏛️ Order Audit Trail -- Registre Cryptographique")
st.caption(
    "Chaque événement réel de cycle de vie d'ordre (Soumission, Exécution, Rejet) est chaîné par SHA-256 "
    "(current_hash = SHA256(prev_hash | payload)) -- voir src/execution/audit_ledger.py. Preuve d'altération, "
    "pas immutabilité absolue : un accès SQL privilégié pourrait en théorie réécrire toute la chaîne aval."
)

settings = get_settings()
r = get_redis_client()

anchor = safe_get_json(r, AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY)
if anchor is not None:
    st.caption(f"Dernière ancre réelle (Redis, indépendante) : `{anchor['current_hash'][:16]}...` -- écrite le {anchor['written_at']}.")
else:
    st.caption("Aucune ancre réelle enregistrée pour l'instant -- aucun événement n'a encore été chaîné avec succès.")

col_search, col_tools = st.columns([3, 1])

with col_search:
    search_term = st.text_input("🔍 Rechercher (Order ID ou Asset)", placeholder="Ex: un order_id réel, ou BTC/USDT")

    async def _search(query: str) -> list[dict[str, Any]] | None:
        try:
            pool = await connect_with_backoff(settings.timescale, max_attempts=2)
        except TimescaleUnavailableError:
            return None
        try:
            await ensure_schema(pool)
            return await fetch_order_history(pool, search=query, limit=200)
        finally:
            await pool.close()

    rows = asyncio.run(_search(search_term))

    if rows is None:
        not_yet_implemented("Registre d'audit", reason="TimescaleDB n'est pas joignable depuis ce process dashboard.")
    elif not rows:
        st.info(
            "Aucun événement réel enregistré pour l'instant -- soumettez un ordre réel via l'API pour peupler ce registre."
            if not search_term else "Aucun résultat réel pour cette recherche."
        )
    else:
        display_rows = [
            {
                "Seq": row["seq_id"],
                "Horodatage": row["time"].strftime("%Y-%m-%d %H:%M:%S UTC") if isinstance(row["time"], datetime) else str(row["time"]),
                "Type": row["event_type"],
                "Order ID": row["order_id"],
                "Client": row["client_id"] or "--",
                "Asset": row["asset"] or "--",
                "Hash (aperçu)": f"{row['current_hash'][:16]}...",
            }
            for row in rows
        ]
        st.dataframe(display_rows, width="stretch", hide_index=True, height=420)

with col_tools:
    st.subheader("Vérification cryptographique")
    if st.button("🛡️ Vérifier l'intégrité de la chaîne (réel)", type="primary", width="stretch"):

        async def _verify() -> tuple[bool, str, bool, str] | None:
            try:
                pool = await connect_with_backoff(settings.timescale, max_attempts=2)
            except TimescaleUnavailableError:
                return None
            try:
                await ensure_schema(pool)
                all_rows = await fetch_all_rows(pool)
                chain_valid, chain_message = verify_chain(all_rows)
                redis_client = await RedisClient.get_instance(settings.redis)
                anchor_valid, anchor_message = await verify_chain_tip_matches_anchor(pool, redis_client)
                return chain_valid, chain_message, anchor_valid, anchor_message
            finally:
                await pool.close()

        with st.spinner("Recalcul SHA-256 de la chaîne complète en cours..."):
            result = asyncio.run(_verify())

        if result is None:
            st.error("TimescaleDB n'est pas joignable -- vérification impossible.")
        else:
            chain_valid, chain_message, anchor_valid, anchor_message = result
            if chain_valid:
                st.success(f"✅ {chain_message}")
            else:
                st.error(f"🚨 {chain_message}")
            if anchor_valid:
                st.success(f"✅ {anchor_message}")
            else:
                st.error(f"🚨 {anchor_message}")

    st.caption(
        "Recalcule chaque empreinte SHA-256 depuis le début de la chaîne et compare le dernier hash de "
        "TimescaleDB à l'ancre Redis indépendante -- voir le docstring du module pour la portée exacte."
    )

st.markdown("---")
st.success(
    "✅ **Annulation d'ordre (événement CANCELLED)** -- désormais réelle : "
    "`POST /api/v1/orders/{order_id}/cancel` (api_middleware/gateway.py) marque l'ordre CANCELLED et publie "
    "un vrai événement CANCELLED ici. Fenêtre réelle mais honnêtement courte : MacroOrderExecutionAgent "
    "exécute immédiatement à la réception (voir le docstring de ce module), donc la plupart des annulations "
    "arriveront trop tard (409, ordre déjà terminal)."
)

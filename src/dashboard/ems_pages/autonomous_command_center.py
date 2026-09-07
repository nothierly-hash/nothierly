"""Autonomous Command Center -- Master Toggle, live telemetry, and Emergency Liquidate for the Alpha Engine.

**Real data, real actions -- no fabricated telemetry** (this project's
own established discipline, see ``src/dashboard/ems_pages/
agent_fleet.py``'s module docstring on why an earlier draft's invented
Tier-2 progress numbers were rejected): every number on this page comes
from :data:`~core.schemas.ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX`, a
real JSON blob :class:`~src.agents.alpha_engine.agent.
AlphaMarketMakerAgent` publishes on every real tick -- if the engine is
not running, this page honestly shows "aucune télémétrie" instead of a
plausible-looking demo number.

**Master Toggle**: writes the real, existing
:class:`~core.base_agent.AgentControlState` (``LIVE``/``SHADOW``/``OFF``)
for this engine's ``agent_id`` -- the exact same mechanism
``agent_fleet.py`` already uses for every other agent, not a second,
parallel on/off primitive. See that page's own docstring, and
:class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`'s module
docstring point 3, for why ``NOTHIERLY_BOT_STATUS`` as literally briefed
is not what got built.

**Emergency Liquidate**: PUBLISHes to
:data:`~core.schemas.ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX`,
which the engine's own background listener
(``_listen_for_emergency_liquidate``) picks up and acts on within one
real tick. This page has no way to place an order itself -- it can only
signal a real, running engine process to do so, which requires that
process to actually be connected to Redis. A two-step confirm (click,
then confirm) guards this button: it fires a real market order against
real capital.
"""

from __future__ import annotations

import time
from typing import Literal

import streamlit as st

from core.base_agent import AgentControlState
from core.schemas import (
    AGENT_CONTROL_CHANNEL_PREFIX,
    AGENT_CONTROL_STATE_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX,
    ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX,
    HEARTBEAT_REDIS_KEY_PREFIX,
)
from src.dashboard.ems_pages._shared import get_redis_client, safe_get, safe_get_json
from src.dashboard.theme import status_badge

# Matches AlphaMarketMakerAgent's own f"alpha-engine-{symbol...}" derivation exactly (see that module) --
# duplicated here as a plain string rather than importing the agent class itself, which would pull
# ccxt.pro's full import graph into this Streamlit page for one string transform. If a second symbol's
# engine is ever run, add it to this tuple.
_ENGINES: tuple[tuple[str, str], ...] = (("BTC/USDT", "alpha-engine-btc-usdt"),)

st.title("🔴 Autonomous Command Center")
st.caption("Contrôle en temps réel du moteur de market making autonome -- capital réel, pas une démo.")

r = get_redis_client()

symbol, agent_id = st.selectbox(
    "Moteur", options=_ENGINES, format_func=lambda pair: f"{pair[0]} ({pair[1]})", index=0,
)

# ---- Liveness ----
raw_heartbeat = safe_get(r, f"{HEARTBEAT_REDIS_KEY_PREFIX}{agent_id}")
engine_online = raw_heartbeat is not None
heartbeat_age = time.time() - float(raw_heartbeat) if raw_heartbeat is not None else None

st.markdown(
    status_badge(
        "SAFE" if engine_online else "PANIC",
        f"EN LIGNE ({heartbeat_age:.1f}s)" if engine_online and heartbeat_age is not None else "HORS LIGNE",
    ),
    unsafe_allow_html=True,
)
if not engine_online:
    st.warning(
        "Aucun heartbeat réel détecté pour ce moteur -- soit il n'est pas démarré "
        "(`python scripts/run_alpha_engine.py`), soit Redis est injoignable. Les commandes ci-dessous "
        "restent fonctionnelles (elles écrivent un vrai état dans Redis), mais n'ont aucun effet tant "
        "qu'un processus réel n'écoute pas."
    )

st.markdown("---")

# ---- Master Toggle ----
st.subheader("⚡ Master Toggle")
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
st.markdown(
    f"**État réel actuel :** {status_badge(state_badge_kind, current_state.value)}", unsafe_allow_html=True,
)

col_on, col_off = st.columns(2)
with col_on:
    if st.button(
        "🟢 ACTIVER LE MOTEUR (LIVE)", type="primary", width="stretch",
        disabled=current_state == AgentControlState.LIVE, key=f"toggle_live_{agent_id}",
    ):
        r.set(f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent_id}", AgentControlState.LIVE.value)
        r.publish(f"{AGENT_CONTROL_CHANNEL_PREFIX}{agent_id}", AgentControlState.LIVE.value)
        st.toast("Moteur activé -- état réel LIVE.", icon="✅")
        st.rerun()
with col_off:
    if st.button(
        "⏸️ DÉSACTIVER LE MOTEUR (OFF)", width="stretch",
        disabled=current_state == AgentControlState.OFF, key=f"toggle_off_{agent_id}",
    ):
        r.set(f"{AGENT_CONTROL_STATE_REDIS_KEY_PREFIX}{agent_id}", AgentControlState.OFF.value)
        r.publish(f"{AGENT_CONTROL_CHANNEL_PREFIX}{agent_id}", AgentControlState.OFF.value)
        st.toast("Moteur désactivé -- état réel OFF. Cotations annulées au prochain cycle.", icon="🛑")
        st.rerun()

st.caption(
    "OFF arrête l'ouverture de nouvelles cotations et annule les cotations en cours -- ça ne ferme "
    "**pas** automatiquement les positions déjà ouvertes (voir Emergency Liquidate plus bas pour ça, "
    "explicitement). Choix délibéré : forcer une vente au marché uniquement parce que l'opérateur met "
    "en pause n'est pas un comportement souhaitable."
)

st.markdown("---")

# ---- Telemetry ----
st.subheader("📡 Télémétrie en direct")
telemetry = safe_get_json(r, f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}{agent_id}")

if telemetry is None:
    st.info(
        "**Aucune télémétrie réelle disponible** -- le moteur n'a publié aucune donnée récente "
        "(clé absente ou expirée). Rien n'est affiché ici tant que le processus réel ne tourne pas -- "
        "jamais une valeur simulée à sa place."
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    inventory = telemetry.get("inventory")
    base_currency = symbol.split("/")[0]
    col1.metric("Inventaire réel", f"{inventory} {base_currency}" if inventory is not None else "N/A")
    col2.metric("Plafond d'inventaire", f"{telemetry.get('max_inventory_units', 'N/A')} {base_currency}")
    pnl = telemetry.get("session_realized_pnl_usd")
    col3.metric("PnL réalisé (session)", f"${float(pnl):,.2f}" if pnl is not None else "N/A")
    col4.metric("Trades (24h)", telemetry.get("trade_count_24h", "N/A"))

    col5, col6, col7 = st.columns(3)
    daily_pnl = telemetry.get("daily_realized_pnl_usd")
    col5.metric("PnL réalisé (jour)", f"${float(daily_pnl):,.2f}" if daily_pnl is not None else "N/A")
    max_daily_loss = telemetry.get("max_daily_loss_usd")
    col6.metric("Plafond perte/jour", f"${float(max_daily_loss):,.2f}" if max_daily_loss is not None else "⚠️ NON CONFIGURÉ")
    max_single_order = telemetry.get("max_single_order_units")
    col7.metric("Plafond fat-finger", f"{max_single_order} {base_currency}" if max_single_order is not None else "N/A")
    if max_daily_loss is None:
        st.warning(
            "⚠️ **Aucune limite de perte quotidienne configurée** -- ce moteur ne s'arrêtera jamais "
            "automatiquement sur drawdown. Voir `--max-daily-loss-usd` dans `scripts/run_alpha_engine.py`."
        )

    status = telemetry.get("status", "?")
    if status in ("STALE_DATA", "HEDGING"):
        status_badge_kind: Literal["SAFE", "PANIC", "PENDING"] = "PANIC"
    elif status == "LIVE":
        status_badge_kind = "SAFE"
    else:
        status_badge_kind = "PENDING"
    st.markdown(f"**Statut du cycle :** {status_badge(status_badge_kind, status)}", unsafe_allow_html=True)

    age = telemetry.get("market_data_age_seconds")
    if age is not None:
        st.caption(f"Fraîcheur des données de marché : {age:.2f}s.")
    if telemetry.get("mock_mode"):
        st.warning(
            "🎭 **MOCK_MODE actif** -- ce moteur ne place aucun ordre réel "
            "(voir `core/config.py` -> `AppSettings.mock_mode`)."
        )

    updated_at = telemetry.get("updated_at")
    if updated_at is not None:
        st.caption(f"Dernière mise à jour : il y a {time.time() - updated_at:.1f}s.")

st.markdown("---")

# ---- Emergency Liquidate ----
st.subheader("🆘 Emergency Liquidate")
st.error(
    "Ferme immédiatement toute position réelle ouverte au prix du marché (ordre MARKET), annule "
    "toutes les cotations en cours, et force le moteur sur OFF."
)
with st.expander("⚠️ Limite connue -- à lire avant utilisation"):
    st.markdown(
        "Si le **Kill Switch global** (`SYSTEM_STATUS`) est déjà déclenché pour une autre raison, "
        "l'ordre de fermeture est lui-même refusé par ce même mécanisme -- exactement le même paradoxe "
        "que l'auto-unwind du Smart Order Router (voir `src/execution/smart_order_router.py`). Dans ce "
        "cas, le moteur envoie une alerte Telegram **INTERVENTION MANUELLE REQUISE** au lieu de "
        "prétendre que la position est fermée. Vérifiez la page SRE & Red Phone si ce bouton ne semble "
        "pas avoir d'effet."
    )

_confirm_key = f"confirm_emergency_liquidate_{agent_id}"
if _confirm_key not in st.session_state:
    st.session_state[_confirm_key] = False

if not st.session_state[_confirm_key]:
    if st.button("🆘 EMERGENCY LIQUIDATE", type="primary", width="stretch", key=f"liquidate_{agent_id}"):
        st.session_state[_confirm_key] = True
        st.rerun()
else:
    st.warning("Confirmez : ceci va réellement fermer la position au marché et couper le moteur.")
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("✅ CONFIRMER -- fermer maintenant", type="primary", width="stretch", key=f"confirm_{agent_id}"):
            r.publish(f"{ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX}{agent_id}", "LIQUIDATE")
            st.session_state[_confirm_key] = False
            st.toast("Commande Emergency Liquidate envoyée -- effective au prochain cycle réel du moteur.", icon="🆘")
            st.rerun()
    with col_cancel:
        if st.button("Annuler", width="stretch", key=f"cancel_{agent_id}"):
            st.session_state[_confirm_key] = False
            st.rerun()

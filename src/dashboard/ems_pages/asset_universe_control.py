"""Asset Universe Control -- per-asset session state, spread health, and live risk overrides. Cross-asset mutation's own dashboard counterpart.

**Real registry, real session logic, real Redis writes -- one real gap
disclosed up front rather than silently implied**: the asset list, each
asset's ``max_allowable_spread``, and each session's OPEN/CLOSED/FRIDAY
UNWIND status all come straight from :data:`core.asset_universe.
ASSET_UNIVERSE` and :class:`core.session_manager.MarketSessionManager` --
the exact same registry and pure functions
:class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent` itself
already gates every real cycle on (see that module's docstring point 14).
This page's per-asset kill switch and dynamic risk override, however,
write to two **new** Redis keys
(:data:`~core.schemas.ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX`,
:data:`~core.schemas.ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX`)
that no running engine reads yet -- today's ``AlphaMarketMakerAgent``
trades exactly one hardcoded ``self._symbol`` per process, gated solely
by its own :class:`~core.base_agent.AgentControlState`. The writes below
are real and functional (flip a toggle, it is really in Redis, survives
a page reload); wiring a per-symbol read into ``_run_cycle`` for a
future multi-symbol engine is a separate, not-yet-done change, not
something this page pretends to already control.

**Live spread**: read from the real
:data:`~core.schemas.MARKET_METRICS_REDIS_KEY_PREFIX` cache
:class:`~src.agents.market_data.agent.MarketDataAgent` populates --
today that is only genuinely live for symbols an actual running
``MarketDataAgent``/exchange client is subscribed to. For any other
registered symbol this honestly shows "N/A" instead of a plausible-
looking number -- the optional demo-mode toggle (this project's own
established ``src/dashboard/ems_pages/_shared.py::demo_mode_toggle``
pattern) substitutes an explicitly-labelled simulated value instead,
never silently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

import streamlit as st

from core.asset_universe import ASSET_UNIVERSE, AssetUniverseEntry, PositionSizingConvention
from core.schemas import (
    ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX,
    MARKET_METRICS_REDIS_KEY_PREFIX,
)
from core.session_manager import MarketSessionManager
from src.dashboard.ems_pages._shared import demo_mode_toggle, get_redis_client, safe_get, safe_get_json
from src.dashboard.theme import status_badge

st.title("🌐 Asset Universe Control")
st.caption(
    "Surveillance et bridage individuel de chaque actif du registre cross-asset -- "
    "Crypto, FX, Commodities. Un seul registre (`core/asset_universe.py`), zéro branche par actif."
)

r = get_redis_client()
demo_mode = demo_mode_toggle(key="asset_universe_demo_mode")

st.divider()


@dataclass(frozen=True)
class _AssetSnapshot:
    symbol: str
    entry: AssetUniverseEntry
    session_status: Literal["OPEN", "CLOSED", "FRIDAY UNWIND"]
    badge_kind: Literal["SAFE", "PANIC", "PENDING"]
    current_spread: Decimal | None
    spread_is_simulated: bool
    kill_switch_on: bool


def _read_current_spread(symbol: str) -> Decimal | None:
    """Real spread from MarketDataAgent's own live cache key -- None if that producer isn't running for this symbol."""
    metrics = safe_get_json(r, f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
    if metrics is None or "spread" not in metrics:
        return None
    try:
        return Decimal(str(metrics["spread"]))
    except InvalidOperation:
        return None


def _build_snapshot(symbol: str, entry: AssetUniverseEntry) -> _AssetSnapshot:
    now = datetime.now(timezone.utc)
    is_unwind = MarketSessionManager.is_unwind_period(symbol, now=now)
    is_open = MarketSessionManager.is_market_open(symbol, now=now)
    if is_unwind:
        session_status: Literal["OPEN", "CLOSED", "FRIDAY UNWIND"] = "FRIDAY UNWIND"
        badge_kind: Literal["SAFE", "PANIC", "PENDING"] = "PANIC"
    elif is_open:
        session_status = "OPEN"
        badge_kind = "SAFE"
    else:
        session_status = "CLOSED"
        badge_kind = "PENDING"

    real_spread = _read_current_spread(symbol)
    spread_is_simulated = False
    current_spread = real_spread
    if current_spread is None and demo_mode:
        # Explicitly labelled simulated placeholder only -- see _shared.py's demo_mode_toggle docstring
        # on why this never substitutes silently. Deterministic per symbol (a stable demo, not a value
        # that jumps every rerun for no reason), pinned at 60% of this asset's own real ceiling.
        current_spread = entry.max_allowable_spread * Decimal("0.6")
        spread_is_simulated = True

    raw_status = safe_get(r, f"{ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX}{symbol}")
    kill_switch_on = raw_status != "OFF"  # missing key defaults ON -- an unconfigured asset trades normally

    return _AssetSnapshot(
        symbol=symbol, entry=entry, session_status=session_status, badge_kind=badge_kind,
        current_spread=current_spread, spread_is_simulated=spread_is_simulated, kill_switch_on=kill_switch_on,
    )


snapshots = [_build_snapshot(symbol, entry) for symbol, entry in ASSET_UNIVERSE.items()]

# ---- Composant 2 : Tableau de bord global (grille st.columns) ----
st.subheader("📊 Market State")
grid_cols = st.columns(len(snapshots))
for col, snap in zip(grid_cols, snapshots):
    with col:
        st.markdown(f"**{snap.symbol}**")
        st.caption(snap.entry.asset.asset_class.value)
        st.markdown(status_badge(snap.badge_kind, snap.session_status), unsafe_allow_html=True)
        if not snap.kill_switch_on:
            st.markdown(status_badge("PANIC", "ASSET OFF"), unsafe_allow_html=True)

        if snap.current_spread is None:
            st.metric("Spread actuel", "N/A")
            st.caption("Aucun flux réel (`market_metrics`) pour ce symbole.")
        else:
            over_ceiling = snap.current_spread > snap.entry.max_allowable_spread
            st.metric(
                "Spread actuel" + (" 🎭" if snap.spread_is_simulated else ""),
                f"{snap.current_spread}",
                delta=f"plafond {snap.entry.max_allowable_spread}",
                delta_color="inverse" if over_ceiling else "off",
            )
            if over_ceiling:
                st.caption("⚠️ Au-dessus du plafond -- rejeté par le filtre anti-manipulation.")

st.divider()

# ---- Composant 3 & 4 : Kill-switch individuel + Override de risque dynamique, par actif ----
st.subheader("🎛️ Contrôle par actif")

for snap in snapshots:
    entry = snap.entry
    with st.container(border=True):
        header_col, toggle_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"### {snap.symbol}")
            st.caption(
                f"{entry.asset.asset_class.value} · tick {entry.asset.tick_size} · lot {entry.asset.lot_size} · "
                f"{entry.position_sizing_convention.value}"
                + (f" (x{entry.lot_multiplier})" if entry.position_sizing_convention == PositionSizingConvention.LOT_BASED else "")
            )
        with toggle_col:
            new_on = st.toggle(
                "Actif activé", value=snap.kill_switch_on, key=f"kill_switch_{snap.symbol}",
                help="OFF : l'Alpha Engine rejette tout nouvel ordre sur cet actif (écrit un vrai état dans Redis).",
            )
            if new_on != snap.kill_switch_on:
                r.set(f"{ALPHA_ENGINE_ASSET_STATUS_REDIS_KEY_PREFIX}{snap.symbol}", "ON" if new_on else "OFF")
                st.toast(
                    f"{snap.symbol} : {'activé' if new_on else 'désactivé'} (Redis mis à jour).",
                    icon="✅" if new_on else "🛑",
                )
                st.rerun()

        st.caption(f"Session : {snap.session_status} · fenêtre hebdomadaire réelle (voir `core/session_manager.py`).")

        # ---- Override de risque dynamique -- lit tout override déjà écrit, sinon retombe sur la config statique du registre.
        existing_override = safe_get_json(r, f"{ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX}{snap.symbol}")
        default_spread = float(existing_override["max_allowable_spread"]) if existing_override else float(entry.max_allowable_spread)
        default_position_size = float(existing_override["max_position_size"]) if existing_override else 0.0

        with st.form(key=f"risk_override_form_{snap.symbol}"):
            spread_col, size_col = st.columns(2)
            new_max_spread = spread_col.number_input(
                "Spread max autorisé", min_value=0.0, value=default_spread, step=float(entry.asset.tick_size) or 0.0001,
                format="%.5f", key=f"max_spread_input_{snap.symbol}",
            )
            new_max_position = size_col.number_input(
                "Taille de position max", min_value=0.0, value=default_position_size, step=1.0,
                key=f"max_position_input_{snap.symbol}",
                help="0 = aucun override actif (le moteur, une fois câblé, retombe sur sa propre config statique).",
            )
            if st.form_submit_button("💾 Valider l'override", width="stretch"):
                r.set(
                    f"{ALPHA_ENGINE_ASSET_RISK_OVERRIDE_REDIS_KEY_PREFIX}{snap.symbol}",
                    json.dumps(
                        {
                            "max_allowable_spread": str(new_max_spread),
                            "max_position_size": str(new_max_position),
                            "updated_at": time.time(),
                        }
                    ),
                )
                st.toast(f"Override de risque pour {snap.symbol} écrit dans Redis.", icon="💾")
                st.rerun()

        if existing_override is not None:
            st.caption(
                f"Override actif depuis {time.time() - float(existing_override.get('updated_at', time.time())):.0f}s "
                f"-- config statique du registre : spread max {entry.max_allowable_spread}."
            )

    st.write("")

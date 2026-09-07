"""Nothierly EMS -- Enterprise Multipage Architecture. Chapter 13, Étape 5.4-5.9.

The router for the full 14-page B2B Execution Management System, using
Streamlit's native ``st.navigation``/``st.Page`` (not a hand-rolled
``pages/`` folder or a session-state-driven if/elif switch): each
``st.Page`` is given a *file path*, not an imported function, which is
what actually gives "isolation parfaite des pages pour ne pas
surcharger la RAM" (the brief's own requirement) -- Streamlit only
loads and executes the selected page's module on each run, not all
fourteen every time. Every later "ENTERPRISE IAM" directive that asked
for new pages under a literal ``pages/0N_name.py`` path (Pre-Trade
Oracle, Cognitive Guardrail, Security & IAM) is placed under
``ems_pages/`` and added to the list below instead, for the same
reason -- this project has no ``pages/`` auto-discovery folder, by
design, since it would defeat the per-page isolation above.

Design system: :mod:`src.dashboard.theme`, the real, already-verified
"Nothierly" brand extracted from ``src/dashboard/app.py`` (see that
module's docstring), not a new visual identity -- this chapter's own
instruction was to style the new app after the existing one, not
invent a third look after ``ultimate_terminal.py``'s separate Obsidian
Glass theme.

**Real authentication, not none**: before Étape 5.9 this router called
no login gate at all -- only the retired single-page ``app.py`` ever
had one. :mod:`src.dashboard.auth_middleware` now gates every page
behind :class:`core.iam.IamStore`-backed accounts (real per-user
Argon2 password hashes, optional TOTP 2FA, session revocation, IP
whitelisting, and an idle screen lock) -- see that module's docstring
for the full design and its honest limits.

Run with::

    streamlit run src/dashboard/nothierly_ems.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from core.config import get_settings
from src.dashboard.auth_middleware import enforce_idle_lock, render_lock_overlay_if_needed, require_login, touch_activity
from src.dashboard.ems_pages._shared import get_iam_store
from src.dashboard.theme import render_brand_header, render_theme_css, render_theme_toggle

st.set_page_config(page_title="Nothierly EMS", page_icon="◆", layout="wide", initial_sidebar_state="expanded")

# Real login gate -- see src/dashboard/auth_middleware.py's module docstring for why this replaces the
# old app.py::check_password() and why it must run before st.navigation/st.Page ever load a page module.
_iam_store = get_iam_store()
_dashboard_settings = get_settings().dashboard
if _dashboard_settings.user is not None and _dashboard_settings.password is not None:
    _iam_store.ensure_bootstrap_admin(
        username=_dashboard_settings.user, password=_dashboard_settings.password.get_secret_value()
    )
require_login(_iam_store)
# Order matters: enforce_idle_lock must see last_activity_at as it was BEFORE this rerun -- calling
# touch_activity() first would stamp it to "now" and the staleness check could never fire (real bug,
# caught live: the lock never triggered because every rerun -- including the one being checked --
# reset its own baseline). touch_activity() only runs once this rerun is confirmed not locked, so a
# still-locked session's overlay renders (and st.stop()s) without ever counting itself as activity.
enforce_idle_lock(_iam_store)
render_lock_overlay_if_needed(_iam_store)
touch_activity()

_PAGES_DIR = Path(__file__).resolve().parent / "ems_pages"

with st.sidebar:
    render_brand_header(subtitle="MASTER CONTROL TERMINAL")
    st.markdown("---")
    mode = render_theme_toggle()
    st.markdown("---")

render_theme_css(mode)

# Grouped into 3 real sections (st.navigation's own Mapping[str, Sequence[Page]]
# form, not a hand-rolled grouping) -- real pages placed by what they actually
# do, not a literal re-reading of a brief that named pages ("SOR", "Dark Pool")
# this app didn't have yet when this grouping was introduced; each is added to
# its section as its own real page lands, not stubbed in ahead of time.
pages = {
    "📊 Execution & Routing": [
        st.Page(str(_PAGES_DIR / "command_center.py"), title="Command Center", icon="📊", default=True),
        st.Page(str(_PAGES_DIR / "liquidity_routing.py"), title="Liquidity Routing", icon="🔌"),
        st.Page(str(_PAGES_DIR / "smart_order_router_radar.py"), title="SOR Radar", icon="📡"),
        st.Page(str(_PAGES_DIR / "internal_dark_pool.py"), title="Internal Dark Pool", icon="🌑"),
        st.Page(str(_PAGES_DIR / "order_blotter.py"), title="Order Blotter & TCA", icon="📋"),
        st.Page(str(_PAGES_DIR / "digital_twin_backtest.py"), title="Digital Twin Backtesting", icon="🕰️"),
    ],
    "🧠 AI Fleet & Risk": [
        st.Page(str(_PAGES_DIR / "intelligence.py"), title="Intelligence & Brain", icon="🧠"),
        st.Page(str(_PAGES_DIR / "agent_fleet.py"), title="Multi-Agent Fleet", icon="🤖"),
        st.Page(str(_PAGES_DIR / "microstructure.py"), title="Microstructure Telemetry", icon="📈"),
        st.Page(str(_PAGES_DIR / "risk_management.py"), title="Risk Management", icon="⚙️"),
        st.Page(str(_PAGES_DIR / "quant_copilot_chat.py"), title="Quant Copilot", icon="🤖"),
        st.Page(str(_PAGES_DIR / "pre_trade_oracle.py"), title="Pre-Trade Oracle", icon="🔮"),
        st.Page(str(_PAGES_DIR / "cognitive_guardrail.py"), title="Cognitive Guardrail", icon="🧭"),
    ],
    "⚙️ Operations": [
        st.Page(str(_PAGES_DIR / "red_phone.py"), title="SRE & Red Phone", icon="🚒"),
        st.Page(str(_PAGES_DIR / "finops.py"), title="Cloud & FinOps", icon="☁️"),
        st.Page(str(_PAGES_DIR / "compliance.py"), title="Compliance & Audit", icon="📜"),
        st.Page(str(_PAGES_DIR / "audit_trail.py"), title="Order Audit Trail", icon="🏛️"),
        st.Page(str(_PAGES_DIR / "eod_reconciliation.py"), title="EOD Reconciliation", icon="📑"),
        st.Page(str(_PAGES_DIR / "security_iam.py"), title="Mon Compte & Sécurité", icon="🛡️"),
    ],
}

navigation = st.navigation(pages, expanded=True)
navigation.run()

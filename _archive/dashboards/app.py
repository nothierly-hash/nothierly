"""Nothierly Quantitative API: reads/writes the shared Redis state, nothing else.

Deliberately a separate, synchronous process from every ``asyncio``-based
agent in this system: Streamlit's execution model (top-to-bottom rerun on
every interaction) is incompatible with sharing an event loop, so this
dashboard never imports or calls into any agent directly. It only talks to
Redis -- the same state store every agent already reads/writes -- so it can
never slow down or block the trading loop no matter how it's used.

"Cyber-Institutional" visual theme (heavy CSS injection, glassmorphism,
custom SVG branding, a Bloomberg-terminal-style KPI bar and equity curve):
purely cosmetic on top of that same boundary. **Every number shown is
either real (read from Redis, computed from real position/balance data)
or an honest "réservé"/"N/A" placeholder -- never a fabricated demo
figure**, matching the same "cite your source, never invent a number"
discipline :mod:`src.agents.guardian_ai`'s chat persona is built around.
Concretely: AUM and Risk Exposure below are computed from real Redis
state; "Alpha Quotidien" has no real source yet (would need the Vector
Backtest Engine's benchmark-relative return calculation, not built as of
this chapter) and is shown as reserved rather than invented.

Run with::

    streamlit run src/dashboard/app.py

Reads connection settings from :class:`core.config.RedisSettings` (the same
``.env``/environment every agent uses), not a hardcoded URL -- so this
dashboard always points at the same Redis instance the agents do, in every
environment (dev/staging/prod), without a second place to configure it.

Gated by :func:`check_password` behind ``DASHBOARD_USER``/
``DASHBOARD_PASSWORD`` (see that function's docstring): nothing under
:func:`main_dashboard` -- no Redis read, no Kill Switch control -- runs
until that check passes.
"""

from __future__ import annotations

import html
import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# `streamlit run src/dashboard/app.py` adds this file's own directory
# (src/dashboard/) to sys.path, not the project root -- unlike `python -m`
# execution, which every other entrypoint in this project relies on. The
# `core`/`src` packages this dashboard imports live at the project root,
# so it must be added explicitly, before those imports, regardless of the
# working directory the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import redis
import streamlit as st

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.security import STATUS_PANIC, STATUS_SAFE, SYSTEM_STATUS_KEY
from src.agents.guardian_ai import EmergencyAction, GuardianAlert, assess_risk_sync, chat_with_nothierly_sync

st.set_page_config(page_title="Nothierly Quantitative API", page_icon="◆", layout="wide")

# ==============================================================================
# DESIGN SYSTEM -- "Cyber-Institutional"
# ==============================================================================
# Void black + glassmorphism + Inter/JetBrains Mono, injected once at module
# level so it covers both the pre-login form (check_password) and the main
# dashboard. Targets Streamlit's own data-testid hooks (the standard, if
# unofficial, way to theme a Streamlit app beyond config.toml's flat color
# set) rather than replacing any widget: every st.* call below is
# unchanged, only its rendered appearance is.
_MINT = "#00E676"
_CRIMSON = "#FF1744"
_INDIGO = "#2962FF"
_CYAN = "#00E5FF"

_NOTHIERLY_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --void-black: #030303;
    --panel-bg: rgba(15, 20, 25, 0.45);
    --panel-border: rgba(255, 255, 255, 0.06);
    --panel-border-hover: rgba(0, 229, 255, 0.4);
    --mint: {_MINT};
    --crimson: {_CRIMSON};
    --indigo: {_INDIGO};
    --cyan: {_CYAN};
}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
}}

.stApp {{
    background: radial-gradient(circle, #0a0f18 0%, #030303 100%);
}}

/* Hide Streamlit's own chrome -- default header/menu/footer. */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{background: transparent;}}
header[data-testid="stHeader"] [data-testid="stToolbar"] {{visibility: hidden;}}
/* stExpandSidebarButton (the only way to reopen a collapsed sidebar)
   lives inside stToolbar and was going invisible with it above --
   restored explicitly so a collapsed sidebar is never a dead end. */
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {{
    visibility: visible !important;
    color: #E8EAED !important;
}}

/* Glassmorphism cards -- metrics, expanders, forms, dataframes, chat bubbles,
   and any custom .nothierly-card wrapper below. */
div[data-testid="stMetric"], div[data-testid="stExpander"], div[data-testid="stForm"],
div[data-testid="stDataFrame"], div[data-testid="stChatMessage"], .nothierly-card {{
    background: var(--panel-bg) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--panel-border) !important;
    border-radius: 14px;
    transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
}}
div[data-testid="stMetric"]:hover, .nothierly-card:hover {{
    border-color: var(--panel-border-hover) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 229, 255, 0.08);
}}
div[data-testid="stMetric"] {{ padding: 1rem 1.1rem; }}
div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; }}
div[data-testid="stMetricLabel"] {{ font-family: 'Inter', sans-serif !important; opacity: 0.75; }}

section[data-testid="stSidebar"] {{
    background: rgba(8, 11, 16, 0.85);
    border-right: 1px solid var(--panel-border);
}}

/* Buttons -- flat, glowing cyber-buttons, never Streamlit's default grey. */
.stButton > button, .stFormSubmitButton > button {{
    background: rgba(41, 98, 255, 0.10);
    border: 1px solid rgba(0, 229, 255, 0.35);
    color: var(--cyan);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
    box-shadow: 0 0 12px rgba(0, 229, 255, 0.12);
    transition: all 0.2s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    box-shadow: 0 0 22px rgba(0, 229, 255, 0.4);
    border-color: var(--cyan);
    color: #ffffff;
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{
    background: rgba(255, 23, 68, 0.14) !important;
    border: 1px solid rgba(255, 23, 68, 0.55) !important;
    color: var(--crimson) !important;
    box-shadow: 0 0 12px rgba(255, 23, 68, 0.2);
}}
.stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{
    box-shadow: 0 0 24px rgba(255, 23, 68, 0.5);
    border-color: var(--crimson) !important;
    color: #ffffff !important;
}}

/* Brand header row (logo + wordmark + LIVE dot). */
.nothierly-brand {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.4rem;
}}
.nothierly-wordmark {{
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.08em;
    color: #E8EAED;
}}
.nothierly-live-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--mint);
    box-shadow: 0 0 8px var(--mint);
    display: inline-block;
    animation: nothierly-live-pulse 1.6s ease-in-out infinite;
}}
.nothierly-live-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: var(--mint);
}}
@keyframes nothierly-live-pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.25; }}
}}

.nothierly-panic-bar {{
    background: rgba(255, 23, 68, 0.15);
    border: 1px solid var(--crimson);
    color: #ffffff;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.7rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    animation: nothierly-pulse 1.4s ease-in-out infinite;
}}
@keyframes nothierly-pulse {{
    0% {{ box-shadow: 0 0 6px 2px rgba(255, 23, 68, 0.4); }}
    50% {{ box-shadow: 0 0 28px 12px rgba(255, 23, 68, 0.8); }}
    100% {{ box-shadow: 0 0 6px 2px rgba(255, 23, 68, 0.4); }}
}}

/* J.A.R.V.I.S. terminal -- hacker/quant console. */
.nothierly-terminal {{
    background: #000000;
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    height: 280px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.55;
}}
.nothierly-terminal div {{ white-space: pre; }}
.nothierly-terminal .tag-signal {{ color: var(--cyan); }}
.nothierly-terminal .tag-decision {{ color: var(--mint); }}
.nothierly-terminal .tag-veto {{ color: var(--crimson); }}
</style>
"""
st.markdown(_NOTHIERLY_CSS, unsafe_allow_html=True)


def _nothierly_logo_svg(size: int = 40) -> str:
    """Sleek, minimalist SVG mark: a metallic diamond monolith around a stylized indigo/cyan "N".

    Pure inline SVG (no external asset/icon font) -- geometric, not an
    emoji or literal text logo, per this chapter's explicit branding
    requirement.
    """
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="nothierlyMetal" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#F2F4F7"/>
                <stop offset="55%" stop-color="#9AA5B1"/>
                <stop offset="100%" stop-color="#4B5563"/>
            </linearGradient>
            <linearGradient id="nothierlyIndigo" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{_INDIGO}"/>
                <stop offset="100%" stop-color="{_CYAN}"/>
            </linearGradient>
        </defs>
        <path d="M21 2 L38 21 L21 40 L4 21 Z" fill="none" stroke="url(#nothierlyMetal)" stroke-width="1.3"/>
        <path d="M14 28 L14 13 L28 28 L28 13" fill="none" stroke="url(#nothierlyIndigo)"
              stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="14" cy="13" r="2.1" fill="{_CYAN}"/>
        <circle cx="28" cy="13" r="2.1" fill="{_CYAN}"/>
        <circle cx="14" cy="28" r="2.1" fill="{_INDIGO}"/>
        <circle cx="28" cy="28" r="2.1" fill="{_INDIGO}"/>
        <circle cx="21" cy="20.5" r="1.5" fill="#F2F4F7"/>
    </svg>
    """


def _render_brand_header(size: int = 40) -> None:
    """Logo + wordmark + pulsing LIVE indicator -- used in both the sidebar and the main header."""
    st.markdown(
        f'<div class="nothierly-brand">{_nothierly_logo_svg(size)}'
        f'<div><div class="nothierly-wordmark">NOTHIERLY QUANTITATIVE API</div>'
        f'<div><span class="nothierly-live-dot"></span> '
        f'<span class="nothierly-live-label">LIVE</span></div></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def _init_logging() -> None:
    """Configure structured logging exactly once per Streamlit server process.

    Streamlit reruns this whole script top-to-bottom on every interaction;
    without ``@st.cache_resource``, a naive top-level call would tear down
    and rebuild the background logging thread on every click.
    """
    configure_logging(log_level="INFO", log_dir=None, service_name="dashboard")


_init_logging()
_log = get_logger(__name__)


def check_password() -> bool:
    """Gate the rest of the dashboard behind a username/password form.

    Reads credentials via :func:`core.config.get_settings` (its
    ``dashboard.user``/``dashboard.password`` fields, sourced from the
    ``DASHBOARD_USER``/``DASHBOARD_PASSWORD`` environment variables --
    see :class:`~core.config.DashboardSettings`) on every attempt, not a
    Streamlit-level cache of our own -- a credential rotation via a
    process restart (which also clears ``get_settings``'s own
    ``lru_cache``) must take effect immediately. Deliberately goes
    through ``get_settings()`` rather than reading ``os.environ``
    directly: that is what actually loads a value that only lives in
    ``.env`` (pydantic-settings' own ``env_file`` support), whereas
    ``os.environ.get(...)`` only ever sees variables the launching shell
    itself exported. Either credential unset means every login attempt
    is refused, never silently accepted -- there is no "no password
    configured" bypass.

    Uses :func:`secrets.compare_digest` rather than ``==`` for both
    fields: a naive string comparison leaks how many leading characters
    matched through response-time differences (a timing side channel),
    which ``compare_digest`` is specifically designed to close.

    Returns:
        ``True`` once ``st.session_state["authenticated"]`` is set (either
        already, from a prior successful attempt this browser session, or
        just now). Never returns ``False`` -- a failed or not-yet-attempted
        login renders the form and calls ``st.stop()``, which halts the
        script for this rerun entirely, so no caller ever sees ``False``.
    """
    if st.session_state.get("authenticated", False):
        return True

    _render_brand_header(56)
    st.subheader("🔒 Authentification requise")

    with st.form("login_form"):
        username = st.text_input("Utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Déverrouiller")

    if submitted:
        dashboard_settings = get_settings().dashboard
        expected_user = dashboard_settings.user
        expected_password = (
            dashboard_settings.password.get_secret_value() if dashboard_settings.password is not None else None
        )
        valid = (
            expected_user is not None
            and expected_password is not None
            and secrets.compare_digest(username, expected_user)
            and secrets.compare_digest(password, expected_password)
        )
        if valid:
            st.session_state["authenticated"] = True
            _log.info("dashboard_login_succeeded", username=username)
            st.rerun()
        else:
            _log.warning("dashboard_login_failed", attempted_username=username)
            st.error("Accès Refusé : Tentative logguée.")

    st.stop()
    return False  # pragma: no cover -- unreachable, st.stop() halts the script above.


@st.cache_resource
def get_redis_client() -> redis.Redis[str]:
    """Build the dashboard's own synchronous Redis client, cached per server process.

    A dedicated *synchronous* client -- ``core.redis_client.RedisClient``
    (the shared client every ``asyncio`` agent uses) wraps
    ``redis.asyncio.Redis`` and requires a running event loop; Streamlit's
    script has none. Same connection parameters as that async client
    (:class:`~core.config.RedisSettings`), just the sync driver.
    """
    settings = get_settings().redis
    return redis.Redis(
        host=settings.host,
        port=settings.port,
        db=settings.db,
        password=settings.password.get_secret_value() if settings.password is not None else None,
        ssl=settings.use_tls,
        decode_responses=True,
    )


# Redis keys shared with src/agents/execution/agent.py -- see that module's
# matching constants (_RISK_PCT_REDIS_KEY, _MARKET_STATE_REDIS_KEY, etc.)
# for the producer side of each. No shared constants module between the two
# processes (same choice as "TRADE_EXECUTED" elsewhere in this project): a
# single producer here, a single consumer there, so if one changes, the
# other must change with it.
_RISK_PCT_REDIS_KEY = "TRADING_RISK_PCT"
_DEFAULT_RISK_PCT = 1.0  # %, si la clé n'existe pas encore
_MARKET_STATE_REDIS_KEY = "NOTHIERLY_MARKET_STATE"
_ACTIVE_POSITIONS_REDIS_KEY = "NOTHIERLY_ACTIVE_POSITIONS"
_DECISION_LOG_REDIS_KEY = "NOTHIERLY_DECISION_LOGS"
_RAW_SIGNAL_FEED_REDIS_KEY = "NOTHIERLY_RAW_SIGNAL_FEED"
_GUARDIAN_ALERTS_REDIS_KEY = "NOTHIERLY_GUARDIAN_ALERTS"
_GUARDIAN_ALERTS_MAX_ENTRIES = 30
_EQUITY_CURVE_REDIS_KEY = "NOTHIERLY_EQUITY_CURVE"
# Redis Pub/Sub channel (not a Redis key holding state) -- see
# src/agents/execution/agent.py's _listen_for_manual_trade_requests for
# the ExecutionAgent-side subscriber and why Pub/Sub, not ZMQ, carries
# this specific command.
_MANUAL_TRADE_CHANNEL = "NOTHIERLY_MANUAL_TRADE_CHANNEL"
# Guardian AI trigger 1's threshold: risk-per-trade values at or below
# this need no second opinion. Matches src/agents/execution/agent.py's
# own _MAX_RISK_PCT hard cap being well above this -- this is a lower,
# "ask before going further" line, not the agent's own safety ceiling.
_RISK_GATE_THRESHOLD_PCT = 2.0
_SENTIMENT_ICON = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}
# Fallback instrument list for the manual-override selector when
# NOTHIERLY_MARKET_STATE hasn't been populated yet (e.g. backend not
# running) -- matches scripts/run_backend.py's _ALL_INSTRUMENTS.
_DEFAULT_INSTRUMENTS = ("EUR_USD", "XAU_USD", "BTC/USDT", "ETH/USDT")
# Chat: how many prior turns to resend as conversation history on every
# call -- bounds both the LLM's context/cost and this process's memory,
# without needing real pruning logic for a dashboard nobody expects to
# hold a thousand-message conversation.
_CHAT_HISTORY_TURNS = 10


def _build_dashboard_context(market_state: dict[str, dict[str, Any]], active_positions: list[dict[str, Any]]) -> str:
    """Render current market/position state as plain text for the Guardian AI prompt.

    Dashboard-side counterpart to :meth:`~src.agents.execution.agent.
    ExecutionAgent._build_market_context` -- same purpose (feed "le
    contexte actuel du marché, les positions ouvertes" the CRO system
    prompt asks for), built from the already-parsed Redis JSON this
    process has on hand instead of live ``PositionState`` objects, since
    the two processes share no in-memory state to build it from once
    (see the module docstring on why this dashboard never imports agent
    code).
    """
    lines: list[str] = []
    if not active_positions:
        lines.append("Aucune position ouverte actuellement.")
    else:
        lines.append("Positions ouvertes :")
        for position in active_positions:
            lines.append(
                f"  - {position.get('instrument', '?')} : {position.get('status', '?')}, "
                f"entree {position.get('entry_price')}, prix actuel {position.get('current_price')}, "
                f"PnL {position.get('pnl')}$."
            )
    if market_state:
        lines.append("Derniers signaux connus :")
        for instrument, record in market_state.items():
            lines.append(
                f"  - {instrument} : XGBoost={record.get('xgboost_signal')}, RL={record.get('rl_signal')}, "
                f"sentiment={record.get('sentiment')}."
            )
    return "\n".join(lines)


def _build_chat_context(
    current_status: str, market_state: dict[str, dict[str, Any]], active_positions: list[dict[str, Any]]
) -> str:
    """Timestamped, source-labeled state text for the Nothierly Chat persona.

    The chat's own system prompt (``_CHAT_SYSTEM_PROMPT`` in
    :mod:`src.agents.guardian_ai`) instructs Nothierly to always cite
    which source a figure came from and when it was read -- this is what
    gives it something real to cite: every line is tagged with its
    source (``[REDIS_HOT_STATE]``) and this exact read's wall-clock time,
    rather than a bare, unattributed dump of numbers.
    """
    read_at = datetime.now().strftime("%H:%M:%S")
    return (
        f"[REDIS_HOT_STATE @ {read_at}] Kill Switch : {current_status}.\n"
        f"[REDIS_HOT_STATE @ {read_at}]\n{_build_dashboard_context(market_state, active_positions)}\n"
        "[TIMESCALEDB] Non interrogée pour ce message -- aucune requête historique déclenchée.\n"
        "[GPU_CLUSTER / VAST_AI] Non connecté -- le Vector Backtest Engine n'est pas encore "
        "implémenté dans ce système."
    )


def _push_guardian_alert(r: redis.Redis[str], alert: GuardianAlert) -> None:
    """Push one Guardian AI verdict onto ``NOTHIERLY_GUARDIAN_ALERTS`` -- the dashboard-side producer.

    Mirrors :meth:`~src.agents.execution.agent.ExecutionAgent.
    _trigger_guardian_alert`'s ``LPUSH``+``LTRIM`` exactly (same key,
    same cap), just via this process's own already-open synchronous
    ``redis`` client instead of ``ExecutionAgent``'s async one -- see the
    module docstring on why :mod:`src.agents.guardian_ai` itself never
    touches Redis directly.
    """
    r.lpush(_GUARDIAN_ALERTS_REDIS_KEY, alert.model_dump_json())
    r.ltrim(_GUARDIAN_ALERTS_REDIS_KEY, 0, _GUARDIAN_ALERTS_MAX_ENTRIES - 1)


def main_dashboard() -> None:
    """Render the full control room. Only reached once :func:`check_password` passes."""
    r = get_redis_client()

    # -- Read every piece of shared state up front, once, so every section
    # below never disagrees with another within the same rerun (and never
    # re-reads the same key twice).
    current_status = r.get(SYSTEM_STATUS_KEY) or STATUS_SAFE

    raw_market_state = r.get(_MARKET_STATE_REDIS_KEY)
    try:
        market_state: dict[str, dict[str, Any]] = json.loads(raw_market_state) if raw_market_state else {}
    except ValueError:
        market_state = {}
    market_records = list(market_state.values())

    raw_positions = r.get(_ACTIVE_POSITIONS_REDIS_KEY)
    try:
        active_positions: list[dict[str, Any]] = json.loads(raw_positions) if raw_positions else []
    except ValueError:
        active_positions = []

    # Equity curve: LPUSH'd newest-first -- reverse to chronological order
    # for plotting. Real samples only (see NOTHIERLY_EQUITY_CURVE's
    # producer-side docstring in execution/agent.py) -- an empty/short
    # list means genuinely too little trading history, not a bug to hide.
    raw_equity_points = r.lrange(_EQUITY_CURVE_REDIS_KEY, 0, -1)
    equity_points: list[dict[str, float]] = []
    for raw_point in reversed(raw_equity_points):
        try:
            equity_points.append(json.loads(raw_point))
        except ValueError:
            continue

    # AUM / Risk Exposure: real, computed figures -- see the module
    # docstring's "never fabricate a number" rule. None when genuinely
    # unavailable (no equity sample published yet), rendered as "N/A"
    # below rather than a guessed default.
    current_aum = equity_points[-1]["balance"] if equity_points else None
    gross_notional = sum(
        abs(float(p.get("units", 0)) * float(p.get("entry_price", 0)))
        for p in active_positions
        if isinstance(p.get("units"), (int, float)) and isinstance(p.get("entry_price"), (int, float))
    )
    risk_exposure_pct = (gross_notional / current_aum * 100.0) if current_aum else None

    # ==========================================
    # SIDEBAR : NOTHIERLY SETTINGS
    # ==========================================
    with st.sidebar:
        _render_brand_header(34)
        st.markdown("---")
        st.subheader("Money Management")
        raw_risk_pct = r.get(_RISK_PCT_REDIS_KEY)
        if raw_risk_pct is None:
            current_risk_pct_display = _DEFAULT_RISK_PCT
            r.set(_RISK_PCT_REDIS_KEY, str(_DEFAULT_RISK_PCT / 100.0))
        else:
            try:
                # Redis stores the fraction (e.g. 0.02); the slider shows
                # the percentage (e.g. 2.0) -- see src/agents/execution/
                # agent.py's _get_dynamic_risk_pct, which reads the
                # fraction back directly.
                current_risk_pct_display = float(raw_risk_pct) * 100.0
            except ValueError:
                current_risk_pct_display = _DEFAULT_RISK_PCT

        new_risk_pct_display = st.slider(
            "Risque par trade (% du capital)",
            min_value=0.1,
            max_value=5.0,
            value=current_risk_pct_display,
            step=0.1,
            help=(
                "Fraction du capital risquée sur un seul trade (Money "
                "Management à Risque Fractionnaire Fixe). L'agent "
                "d'exécution lit cette valeur en direct avant chaque "
                "ordre -- aucun redémarrage requis. Plafonné à 5% côté "
                "agent, quoi que cette valeur affiche."
            ),
        )
        if new_risk_pct_display != current_risk_pct_display:
            if new_risk_pct_display > _RISK_GATE_THRESHOLD_PCT:
                # Guardian AI trigger 1: gate, don't write yet. Only calls
                # assess_risk_sync once per distinct slider value (not on
                # every rerun the pending state itself causes) -- see the
                # section docstring in main_dashboard for the full flow.
                pending_risk = st.session_state.get("guardian_pending_risk")
                if pending_risk is None or pending_risk.get("value") != new_risk_pct_display:
                    context = (
                        f"L'utilisateur tente de régler le risque par trade à "
                        f"{new_risk_pct_display:.1f}% du capital (seuil recommandé : "
                        f"{_RISK_GATE_THRESHOLD_PCT:.0f}%).\n\n"
                        f"{_build_dashboard_context(market_state, active_positions)}"
                    )
                    guardian_alert = assess_risk_sync(
                        trigger="RISK_INCREASE",
                        summary=(
                            f"Risque par trade porté à {new_risk_pct_display:.1f}% "
                            f"(seuil : {_RISK_GATE_THRESHOLD_PCT:.0f}%)"
                        ),
                        context=context,
                    )
                    _push_guardian_alert(r, guardian_alert)
                    st.session_state["guardian_pending_risk"] = {
                        "value": new_risk_pct_display,
                        "verdict": guardian_alert.verdict,
                    }
                    st.rerun()
            else:
                r.set(_RISK_PCT_REDIS_KEY, str(round(new_risk_pct_display / 100.0, 4)))
                st.session_state["guardian_pending_risk"] = None
                _log.info(
                    "trading_risk_pct_updated_from_dashboard",
                    old_pct=current_risk_pct_display,
                    new_pct=new_risk_pct_display,
                )
                st.toast(f"Risque par trade mis à jour : {new_risk_pct_display:.1f}%", icon="✅")

        pending_risk = st.session_state.get("guardian_pending_risk")
        if pending_risk is not None:
            st.error(f"⚠️ **Alerte CRO Nothierly**\n\n{pending_risk['verdict']}")
            risk_confirm_col1, risk_confirm_col2 = st.columns(2)
            with risk_confirm_col1:
                if st.button("I understand the risks, execute anyway", key="confirm_risk"):
                    r.set(_RISK_PCT_REDIS_KEY, str(round(pending_risk["value"] / 100.0, 4)))
                    _log.warning(
                        "trading_risk_pct_updated_after_guardian_override",
                        new_pct=pending_risk["value"],
                    )
                    st.session_state["guardian_pending_risk"] = None
                    st.toast(f"Risque mis à jour après confirmation : {pending_risk['value']:.1f}%", icon="⚡")
                    st.rerun()
            with risk_confirm_col2:
                if st.button("Annuler", key="cancel_risk"):
                    st.session_state["guardian_pending_risk"] = None
                    st.rerun()

        st.markdown("---")
        st.subheader("🖥️ Cluster GPU Vast.ai")
        # NOTE: aucun agent n'alimente encore cette clé -- Vast.ai n'est
        # utilisé que par scripts/run_remote_training.py (entraînement
        # ponctuel hors-ligne), jamais par un agent live. Emplacement
        # réservé, même statut que le KPI "Compute Nodes" plus haut.
        vast_status = r.get("STATS_VAST_STATUS") or "NON CONNECTÉ"
        st.metric("Statut du cluster", vast_status)
        st.caption("Réservé -- aucun entraînement distant en cours (voir scripts/run_remote_training.py).")

    # ==========================================
    # BANDEAU PANIC (agressif, pulsant) -- toujours visible, hors onglets
    # ==========================================
    if current_status != STATUS_SAFE:
        st.markdown(
            '<div class="nothierly-panic-bar">🚨 KILL SWITCH ACTIF — TOUTES LES OPÉRATIONS '
            "FINANCIÈRES SONT BLOQUÉES 🚨</div>",
            unsafe_allow_html=True,
        )

    _render_brand_header(48)

    # ==========================================
    # TOP BAR : KPIs INSTITUTIONNELS -- toujours visibles, hors onglets
    # ==========================================
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("AUM (Capital Sous Gestion)", f"${current_aum:,.2f}" if current_aum is not None else "N/A")
        st.caption("Source : NOTHIERLY_EQUITY_CURVE (dernier point réel).")
    with kpi_cols[1]:
        st.metric("Alpha Quotidien", "Réservé")
        st.caption("Nécessite le Vector Backtest Engine (non implémenté).")
    with kpi_cols[2]:
        st.metric("Compute Nodes (GPU/CPU)", r.get("STATS_VAST_STATUS") or "NON CONNECTÉ")
        st.caption("Source : STATS_VAST_STATUS.")
    with kpi_cols[3]:
        st.metric(
            "Exposition Risque Globale", f"{risk_exposure_pct:.1f}%" if risk_exposure_pct is not None else "N/A"
        )
        st.caption("Notionnel brut des positions ÷ AUM.")

    st.markdown("---")

    overview_tab, chat_tab = st.tabs(["🌐 Vue d'Ensemble", "💬 Nothierly Chat"])

    with overview_tab:
        # ==========================================
        # BANDEAU D'ALERTE — NOTHIERLY GUARDIAN AI (CRO)
        # ==========================================
        # "NOTHIERLY_GUARDIAN_ALERTS" -- une liste Redis (LPUSH + LTRIM),
        # alimentée soit par ExecutionAgent (triggers 3/4 : drawdown, signaux
        # contradictoires), soit par ce process lui-même via _push_guardian_
        # alert (triggers 1/2 : risque, override manuel). Affichage seul ici
        # -- les triggers 1/2 ont en plus leur propre confirmation bloquante,
        # rendue au point où l'action dangereuse a été demandée, plus bas.
        raw_guardian_alerts = r.lrange(_GUARDIAN_ALERTS_REDIS_KEY, 0, 1)
        if raw_guardian_alerts:
            st.markdown("### ⚠️ Nothierly CRO — Alertes Récentes")
            for raw_alert in raw_guardian_alerts:
                try:
                    alert_data = json.loads(raw_alert)
                except ValueError:
                    continue
                st.error(
                    f"⚠️ **[{alert_data.get('trigger', '?')}]** {alert_data.get('summary', '')}\n\n"
                    f"{alert_data.get('verdict', '')}"
                )
            st.markdown("---")

        # ==========================================
        # MAIN ARENA : ÉQUITY CURVE  |  SIDE PANEL : ASYMMETRIC CONSENSUS
        # ==========================================
        arena_col, side_col = st.columns([2, 1])

        with arena_col:
            st.markdown("#### 📈 Équity Curve — Nothierly Live")
            if len(equity_points) < 2:
                st.info(
                    f"Historique insuffisant pour tracer une courbe ({len(equity_points)} point(s) "
                    "disponible(s)). Se remplit à chaque position fermée."
                )
            else:
                timestamps = [datetime.fromtimestamp(p["timestamp"]) for p in equity_points]
                balances = [p["balance"] for p in equity_points]
                starting_capital = balances[0]
                is_up = balances[-1] >= starting_capital
                curve_color = _MINT if is_up else _CRIMSON
                curve_fill = "rgba(0, 230, 118, 0.08)" if is_up else "rgba(255, 23, 68, 0.08)"

                equity_fig = go.Figure()
                equity_fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=balances,
                        mode="lines",
                        line={"color": curve_color, "width": 2.5},
                        fill="tozeroy",
                        fillcolor=curve_fill,
                        name="Nothierly (réel)",
                    )
                )
                equity_fig.add_hline(
                    y=starting_capital,
                    line={"color": "rgba(255,255,255,0.25)", "dash": "dash", "width": 1},
                    annotation_text="Capital Initial",
                    annotation_font={"color": "rgba(255,255,255,0.5)", "family": "JetBrains Mono"},
                )
                equity_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#c9d1d9", "family": "JetBrains Mono"},
                    height=380,
                    margin={"l": 10, "r": 10, "t": 10, "b": 10},
                    xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
                    yaxis={"gridcolor": "rgba(255,255,255,0.06)", "tickprefix": "$"},
                    showlegend=False,
                )
                st.plotly_chart(equity_fig, width="stretch", config={"displayModeBar": False})
                st.caption(
                    "Solde réel du compte MOCK_MODE au fil des clôtures de position -- aucune donnée "
                    "synthétique. Ligne pointillée = capital de départ (référence, pas un benchmark de marché)."
                )

        with side_col:
            st.markdown("#### 🧭 Asymmetric Consensus")
            xgb_values: list[float] = [
                float(rec["xgboost_signal"])
                for rec in market_records
                if isinstance(rec.get("xgboost_signal"), (int, float))
            ]
            rl_values: list[float] = [
                float(rec["rl_signal"]) for rec in market_records if isinstance(rec.get("rl_signal"), (int, float))
            ]
            xgb_balance = sum(xgb_values) / len(xgb_values) if xgb_values else 0.0
            rl_balance = sum(rl_values) / len(rl_values) if rl_values else 0.0
            tension = abs(xgb_balance - rl_balance)

            _gauge_steps = [
                {"range": [-1, -0.2], "color": "rgba(255, 23, 68, 0.18)"},
                {"range": [-0.2, 0.2], "color": "rgba(255,255,255,0.06)"},
                {"range": [0.2, 1], "color": "rgba(0, 230, 118, 0.18)"},
            ]
            gauge_fig = go.Figure()
            gauge_fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=xgb_balance,
                    title={"text": "XGBoost", "font": {"family": "JetBrains Mono", "size": 13}},
                    domain={"x": [0, 1], "y": [0.52, 1]},
                    number={"font": {"family": "JetBrains Mono", "color": _CYAN}},
                    gauge={
                        "axis": {"range": [-1, 1], "tickfont": {"size": 8}},
                        "bar": {"color": _CYAN},
                        "steps": _gauge_steps,
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                )
            )
            gauge_fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=rl_balance,
                    title={"text": "RL Policy", "font": {"family": "JetBrains Mono", "size": 13}},
                    domain={"x": [0, 1], "y": [0, 0.48]},
                    number={"font": {"family": "JetBrains Mono", "color": _INDIGO}},
                    gauge={
                        "axis": {"range": [-1, 1], "tickfont": {"size": 8}},
                        "bar": {"color": _INDIGO},
                        "steps": _gauge_steps,
                        "bgcolor": "rgba(0,0,0,0)",
                    },
                )
            )
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#c9d1d9", "family": "JetBrains Mono"},
                height=380,
                margin={"l": 20, "r": 20, "t": 10, "b": 10},
            )
            st.plotly_chart(gauge_fig, width="stretch", config={"displayModeBar": False})
            if tension > 1.0:
                st.error("🚨 VETO ACTIF — divergence maximale entre XGBoost et RL.")
            elif tension > 0.5:
                st.warning(f"⚠️ Tension élevée entre agents ({tension:.2f}).")
            else:
                st.success(f"✅ Agents alignés (tension {tension:.2f}).")

        st.markdown("---")

        # ==========================================
        # SÉCURITÉ SYSTÈME (KILL SWITCH)
        # ==========================================
        st.subheader("Sécurité Système")

        # SYSTEM_STATUS_KEY/STATUS_SAFE/STATUS_PANIC come from core.security --
        # the exact same constants every agent's requires_system_active check
        # reads (see core/security.py) -- so this dashboard can never drift onto
        # a different key name or value spelling than what actually gates a real
        # order/GPU-rental call.
        if current_status == STATUS_SAFE:
            st.success("✅ SYSTÈME OPÉRATIONNEL (SAFE) - Les agents sont autorisés à trader et louer des GPU.")
        else:
            st.error("🚨 SYSTÈME VERROUILLÉ (PANIC) - Toutes les opérations financières sont bloquées.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚨 ACTIVER LE KILL SWITCH", type="primary", width="stretch"):
                r.set(SYSTEM_STATUS_KEY, STATUS_PANIC)
                _log.critical("GLOBAL_KILL_SWITCH_TRIPPED", reason="manual activation from dashboard")
                st.rerun()

        with col2:
            if st.button("✅ RELANCER LE SYSTÈME", width="stretch"):
                r.set(SYSTEM_STATUS_KEY, STATUS_SAFE)
                _log.warning("GLOBAL_KILL_SWITCH_RESET", reason="manual reset from dashboard")
                st.rerun()

        st.markdown("---")

        # ==========================================
        # OVERRIDE MANUEL (Guardian AI trigger 2)
        # ==========================================
        st.subheader("🎯 Override Manuel (Danger)")
        st.caption(
            "Force une entrée en contournant le consensus XGBoost/RL. Passe toujours par "
            "les mêmes gardes-fous que toute autre entrée (Kill Switch, sizing, cap de "
            "positions) -- seul le consensus est contourné."
        )

        override_instrument_options = sorted(market_state.keys()) or list(_DEFAULT_INSTRUMENTS)
        override_col1, override_col2, override_col3 = st.columns([2, 2, 1])
        with override_col1:
            override_instrument = st.selectbox("Instrument", override_instrument_options, key="override_instrument")
        with override_col2:
            override_direction_label = st.radio(
                "Direction", ["Acheter (BUY)", "Vendre (SELL)"], horizontal=True, key="override_direction"
            )
        with override_col3:
            st.write("")
            st.write("")
            force_clicked = st.button("⚡ Forcer l'ordre", type="primary", key="force_order_button")

        override_direction = 1 if override_direction_label.startswith("Acheter") else -1

        if force_clicked:
            record = market_state.get(override_instrument, {})
            xgb_signal, rl_signal = record.get("xgboost_signal"), record.get("rl_signal")
            consensus_conflicts = (
                isinstance(xgb_signal, (int, float)) and int(xgb_signal) != override_direction
            ) or (isinstance(rl_signal, (int, float)) and int(rl_signal) != override_direction)

            if consensus_conflicts:
                direction_word = "ACHAT" if override_direction == 1 else "VENTE"
                override_summary = (
                    f"Forcer un ordre {direction_word} sur {override_instrument} malgré le consensus "
                    f"(XGBoost={xgb_signal}, RL={rl_signal})"
                )
                override_context = f"{override_summary}.\n\n{_build_dashboard_context(market_state, active_positions)}"
                guardian_alert = assess_risk_sync(
                    trigger="MANUAL_OVERRIDE", summary=override_summary, context=override_context
                )
                _push_guardian_alert(r, guardian_alert)
                st.session_state["guardian_pending_override"] = {
                    "instrument": override_instrument,
                    "direction": override_direction,
                    "verdict": guardian_alert.verdict,
                }
                st.rerun()
            else:
                r.publish(
                    _MANUAL_TRADE_CHANNEL,
                    json.dumps(
                        {"instrument": override_instrument, "direction": override_direction, "confirmed": True}
                    ),
                )
                st.toast(f"Ordre {override_instrument} envoyé (déjà aligné avec le consensus).", icon="✅")

        pending_override = st.session_state.get("guardian_pending_override")
        if pending_override is not None:
            st.error(f"⚠️ **Alerte CRO Nothierly**\n\n{pending_override['verdict']}")
            override_confirm_col1, override_confirm_col2 = st.columns(2)
            with override_confirm_col1:
                if st.button("I understand the risks, execute anyway", key="confirm_override"):
                    r.publish(
                        _MANUAL_TRADE_CHANNEL,
                        json.dumps(
                            {
                                "instrument": pending_override["instrument"],
                                "direction": pending_override["direction"],
                                "confirmed": True,
                            }
                        ),
                    )
                    st.session_state["guardian_pending_override"] = None
                    st.toast("Ordre forcé envoyé à l'agent d'exécution.", icon="⚡")
                    st.rerun()
            with override_confirm_col2:
                if st.button("Annuler", key="cancel_override"):
                    st.session_state["guardian_pending_override"] = None
                    st.rerun()

        st.markdown("---")

        # ==========================================
        # RADAR UNIVERSEL (multi-actifs)
        # ==========================================
        # "NOTHIERLY_MARKET_STATE" -- un seul objet JSON, indexé par instrument
        # -- {"EUR_USD": {...}, "BTC/USDT": {...}}. Chaque valeur est un
        # enregistrement PLAT (mêmes clés partout, `None` au lieu d'une clé
        # absente) : `list(market_state.values())` est donc déjà directement
        # compatible avec `pd.DataFrame(...)` sans aucune transformation --
        # exactement pour ça qu'ExecutionAgent structure ce JSON sous cette
        # forme plutôt qu'en objets imbriqués.
        st.subheader("🌐 Radar Universel")

        show_strong_only = st.toggle(
            "Afficher seulement les marchés à signal fort",
            value=False,
            help="Masque les instruments dont XGBoost et RL sont neutres ou en désaccord (⚪).",
        )

        visible_market_records = market_records
        if show_strong_only:
            visible_market_records = [rec for rec in market_records if rec.get("sentiment") != "NEUTRAL"]

        if not visible_market_records:
            st.info("Aucune donnée de marché reçue pour le moment (le backend doit être en cours d'exécution).")
        else:
            radar_cols = st.columns(min(len(visible_market_records), 4))
            for idx, rec in enumerate(visible_market_records):
                with radar_cols[idx % len(radar_cols)]:
                    icon = _SENTIMENT_ICON.get(str(rec.get("sentiment")), "⚪")
                    price = rec.get("last_price")
                    price_str = f"{price:.5f}" if isinstance(price, (int, float)) else "—"
                    st.metric(label=f"{icon} {rec.get('instrument', '?')}", value=price_str)
                    xgb_sig, rl_sig = rec.get("xgboost_signal"), rec.get("rl_signal")
                    st.caption(
                        f"XGBoost: {xgb_sig if xgb_sig is not None else '—'} · "
                        f"RL: {rl_sig if rl_sig is not None else '—'}"
                    )

        st.markdown("---")

        # ==========================================
        # POSITIONS ENGAGÉES
        # ==========================================
        st.subheader("⚔️ Positions Engagées")

        if not active_positions:
            st.info("Aucune position ouverte actuellement.")
        else:
            positions_df = pd.DataFrame(active_positions)

            def _color_pnl(value: object) -> str:
                if not isinstance(value, (int, float)):
                    return ""
                if value > 0:
                    return f"color: {_MINT}; font-weight: 600;"
                if value < 0:
                    return f"color: {_CRIMSON}; font-weight: 600;"
                return ""

            styled_positions = positions_df.style.map(_color_pnl, subset=["pnl", "pnl_pct"]).format(
                {
                    "entry_price": "{:.5f}",
                    "current_price": "{:.5f}",
                    "units": "{:.4f}",
                    "pnl": "{:+.2f} $",
                    "pnl_pct": "{:+.2%}",
                }
            )
            st.dataframe(styled_positions, width="stretch", hide_index=True)

        st.markdown("---")

        # ==========================================
        # STATISTIQUES & SURVEILLANCE
        # ==========================================
        st.subheader("Métriques en temps réel")

        # NOTE: aucun agent n'écrit encore ces clés (STATS_DAILY_PROFIT,
        # STATS_ACTIVE_AGENTS) -- ce sont des emplacements réservés. Valeurs
        # par défaut affichées tant qu'un agent ne les alimente pas.
        daily_profit = r.get("STATS_DAILY_PROFIT") or "+ $0.00"
        active_agents = r.get("STATS_ACTIVE_AGENTS") or "0"

        col_metrics1, col_metrics2 = st.columns(2)

        with col_metrics1:
            st.metric(label="Profit Journalier (Trading)", value=daily_profit, delta="À connecter")

        with col_metrics2:
            st.metric(label="Agents Actifs (ZeroMQ)", value=active_agents, delta="À surveiller")

        # ==========================================
        # LOGS EN DIRECT (Optionnel pour plus tard)
        # ==========================================
        with st.expander("Voir les derniers événements système"):
            st.info(
                "Ici, tu pourras brancher la lecture d'une liste Redis contenant tes derniers "
                "logs d'erreurs ou de transactions."
            )

        # ==========================================
        # NOTHIERLY AI BRAINWAVES (XAI)
        # ==========================================
        with st.expander("🧠 Nothierly AI Brainwaves — Journal explicatif des décisions"):
            st.caption(
                "Un rapport généré à chaque évaluation de consensus (entrée, veto, ou "
                "consensus ignoré car déjà en position). Langage templaté habillant le "
                "signal numérique brut de chaque agent -- pas une vraie explicabilité "
                "de modèle (ni XGBoost ni la politique RL n'exposent de chaîne de "
                "raisonnement réelle)."
            )
            # LPUSH insère en tête -- lrange(0, N) est donc déjà du plus récent au
            # plus ancien, sans tri supplémentaire nécessaire ici.
            raw_reports = r.lrange(_DECISION_LOG_REDIS_KEY, 0, 19)
            if not raw_reports:
                st.info("Aucune décision enregistrée pour le moment.")
            else:
                for raw_entry in raw_reports:
                    try:
                        entry = json.loads(raw_entry)
                    except ValueError:
                        continue
                    action = entry.get("action", "?")
                    icon = {"BUY": "🟢", "SELL": "🔴", "VETO": "🚫", "HOLD": "⚪"}.get(action, "⚪")
                    st.markdown(f"**{icon} {entry.get('instrument', '?')} — {action}**")
                    st.code(entry.get("report", ""), language=None)

        st.markdown("---")

        # ==========================================
        # J.A.R.V.I.S. TERMINAL — flux unifié (signaux bruts + décisions)
        # ==========================================
        # Fusionne deux sources 100% réelles -- NOTHIERLY_RAW_SIGNAL_FEED
        # (un TradingSignal individuel par entrée, avant tout filtrage de
        # fraîcheur/consensus) et NOTHIERLY_DECISION_LOGS (décisions de
        # niveau consensus) -- triées ensemble par horodatage, plutôt que
        # deux terminaux séparés. Aucune ligne n'est générée/inventée ici :
        # chaque ligne provient d'un événement que le backend a réellement
        # publié.
        st.subheader("📡 J.A.R.V.I.S. Terminal — Flux Système Unifié")

        # Each entry pairs a timestamp with a *trusted* HTML fragment: the
        # `<span class="tag-...">` markup is authored here, in code, but
        # every dynamic field interpolated into it (agent_id, instrument,
        # action) is escaped individually first via _esc -- never the
        # already-built fragment as a whole (escaping the whole string
        # and then un-escaping angle brackets back, as an earlier version
        # of this code did, defeats escaping entirely).
        def _esc(value: object) -> str:
            return html.escape(str(value), quote=False)

        terminal_events: list[tuple[float, str]] = []
        for raw_entry in r.lrange(_RAW_SIGNAL_FEED_REDIS_KEY, 0, 39):
            try:
                feed_entry = json.loads(raw_entry)
                ts = float(feed_entry["timestamp"])
                line = (
                    f'<span class="tag-signal">[SIGNAL]</span> {_esc(feed_entry["agent_id"])} · '
                    f'{_esc(feed_entry["instrument"])} · signal={int(feed_entry["signal"]):+d} '
                    f'conf={float(feed_entry["confidence"]):.2f}'
                )
                terminal_events.append((ts, line))
            except (ValueError, KeyError):
                continue
        for raw_entry in r.lrange(_DECISION_LOG_REDIS_KEY, 0, 19):
            try:
                entry = json.loads(raw_entry)
                ts = float(entry.get("timestamp", 0))
                action = str(entry.get("action", "?"))
                css_class = "tag-veto" if action == "VETO" else "tag-decision"
                tag = "VETO" if action == "VETO" else "DECISION"
                line = f'<span class="{css_class}">[{tag}]</span> {_esc(entry.get("instrument", "?"))} · {_esc(action)}'
                terminal_events.append((ts, line))
            except (ValueError, KeyError):
                continue

        if not terminal_events:
            st.info("Aucun événement reçu pour le moment.")
        else:
            terminal_events.sort(key=lambda pair: pair[0], reverse=True)
            terminal_lines = [
                f"<div>[{datetime.fromtimestamp(ts).strftime('%H:%M:%S.%f')[:-3]}] {line}</div>"
                for ts, line in terminal_events[:50]
            ]
            terminal_html = '<div class="nothierly-terminal">' + "".join(terminal_lines) + "</div>"
            st.markdown(terminal_html, unsafe_allow_html=True)

    with chat_tab:
        # ==========================================
        # NOTHIERLY CHAT — copilote conversationnel
        # ==========================================
        # Voir src.agents.guardian_ai's "Chat" section for the full
        # architecture. Summary: this tab NEVER decides on its own whether
        # to execute a dangerous command -- chat_with_nothierly_sync's
        # deterministic intent classifier does that, and even then only
        # ever returns a *pending* action requiring one more explicit
        # confirmation turn before this tab actually touches Redis/pubsub.
        st.caption(
            "Copilote institutionnel Nothierly. Commandes d'urgence supportées : \"ferme <instrument>\", "
            "\"active le kill switch\" -- chacune exige une confirmation explicite avant exécution."
        )

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        if "chat_pending_action" not in st.session_state:
            st.session_state["chat_pending_action"] = None

        if st.button("🗑️ Effacer la conversation", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.session_state["chat_pending_action"] = None
            st.rerun()

        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Parlez à Nothierly...")
        if user_input:
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})

            history = st.session_state["chat_messages"][:-1][-_CHAT_HISTORY_TURNS:]
            chat_context = _build_chat_context(current_status, market_state, active_positions)

            reply = chat_with_nothierly_sync(
                user_message=user_input,
                conversation_history=history,
                context=chat_context,
                pending_action=st.session_state["chat_pending_action"],
            )

            st.session_state["chat_messages"].append({"role": "assistant", "content": reply.text})
            st.session_state["chat_pending_action"] = reply.pending_action

            if reply.confirmed_action is not None:
                _execute_chat_confirmed_action(r, reply.confirmed_action)

            st.rerun()


def _execute_chat_confirmed_action(r: redis.Redis[str], action: EmergencyAction) -> None:
    """Carry out one chat-confirmed :class:`~src.agents.guardian_ai.EmergencyAction`.

    The only place in this file where a chat command actually touches
    Redis -- ``src.agents.guardian_ai`` deliberately never does (see that
    module's chat section docstring), so this dispatch has to live here,
    right next to every other Redis-writing action in this file.
    """
    if action.action_type == "KILL_SWITCH":
        r.set(SYSTEM_STATUS_KEY, STATUS_PANIC)
        _log.critical("GLOBAL_KILL_SWITCH_TRIPPED", reason="Nothierly Chat command")
    else:
        r.publish(
            _MANUAL_TRADE_CHANNEL,
            json.dumps({"instrument": action.instrument, "action": "close", "confirmed": True}),
        )
        _log.warning("nothierly_chat_close_position_command", instrument=action.instrument)


if check_password():
    main_dashboard()

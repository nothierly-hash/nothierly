"""Shared "Nothierly" design system -- Chapter 13, Étape 5.4.

Originally extracted from this project's first, single-page dashboard
(a "Cyber-Institutional" theme, built and visually verified earlier in
this chapter) rather than reinventing a new visual style for the
multipage EMS: this is the real, already-established brand -- void
black, mint/crimson/indigo/cyan accents, Inter + JetBrains Mono,
glassmorphism, the geometric diamond logo. That original single-page
dashboard was later retired in favor of the multipage EMS being the
project's one dashboard (see ``_archive/dashboards/`` for its
retained-but-inactive source) -- this module is what
``src/dashboard/nothierly_ems.py`` and its pages import, the sole
source of the design system now.

**Light mode is new**, not extracted from anywhere -- the original
dashboard only ever had the dark "void black" theme. Built to the
brief's own spec ("Blanc clinique, accents Gris/Bleu marin") while
keeping the same semantic accents (mint=safe/maker, crimson=danger/
panic, cyan/indigo=brand) recognizable across both modes, so a client
switching themes mid-session still reads the same status colors the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import streamlit as st

ThemeMode = Literal["dark", "light"]

MINT: Final[str] = "#00E676"
CRIMSON: Final[str] = "#FF1744"
INDIGO: Final[str] = "#2962FF"
CYAN: Final[str] = "#00E5FF"
NAVY: Final[str] = "#1E3A5F"  # light mode's "Bleu marin" accent, per the brief
STEEL_GRAY: Final[str] = "#5B6B7C"  # light mode's "Gris" accent


@dataclass(frozen=True)
class _Palette:
    background: str
    panel_bg: str
    panel_border: str
    panel_border_hover: str
    text_primary: str
    text_dim: str
    terminal_bg: str
    accent: str  # cyan (dark) / navy (light) -- the "brand" accent buttons/links use


_DARK = _Palette(
    background="radial-gradient(circle, #0a0f18 0%, #030303 100%)",
    panel_bg="rgba(15, 20, 25, 0.45)",
    panel_border="rgba(255, 255, 255, 0.06)",
    panel_border_hover="rgba(0, 229, 255, 0.4)",
    text_primary="#E8EAED",
    text_dim="rgba(232, 234, 237, 0.65)",
    terminal_bg="#000000",
    accent=CYAN,
)

_LIGHT = _Palette(
    background="radial-gradient(circle, #ffffff 0%, #eef1f5 100%)",
    panel_bg="rgba(255, 255, 255, 0.72)",
    panel_border="rgba(30, 58, 95, 0.10)",
    panel_border_hover="rgba(30, 58, 95, 0.45)",
    text_primary="#161A20",
    text_dim="rgba(22, 26, 32, 0.62)",
    terminal_bg="#0d1117",
    accent=NAVY,
)

_PALETTES: Final[dict[ThemeMode, _Palette]] = {"dark": _DARK, "light": _LIGHT}


def render_theme_css(mode: ThemeMode) -> None:
    """Inject the full Nothierly design system for ``mode``. Call once per page, near the top."""
    p = _PALETTES[mode]
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --panel-bg: {p.panel_bg};
        --panel-border: {p.panel_border};
        --panel-border-hover: {p.panel_border_hover};
        --mint: {MINT};
        --crimson: {CRIMSON};
        --indigo: {INDIGO};
        --accent: {p.accent};
        --text-primary: {p.text_primary};
        --text-dim: {p.text_dim};
    }}

    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; color: var(--text-primary); }}
    .stApp {{ background: {p.background}; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    header[data-testid="stHeader"] [data-testid="stToolbar"] {{ visibility: hidden; }}
    /* The rule above hides the whole toolbar (Deploy button, "..." menu)
       for a clean branded header -- but stExpandSidebarButton, the ONLY
       control that reopens a collapsed sidebar, lives inside that same
       toolbar container and was going invisible with it, leaving no way
       back in once the sidebar collapsed (narrow window, stray click).
       Restored explicitly, on top of the blanket hide above. */
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {{
        visibility: visible !important;
        color: var(--text-primary) !important;
    }}

    p, span, label, div[data-testid="stMarkdownContainer"] {{ color: var(--text-primary); }}

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
    }}
    div[data-testid="stMetric"] {{ padding: 1rem 1.1rem; }}
    div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; color: var(--text-primary) !important; }}
    div[data-testid="stMetricLabel"] {{ font-family: 'Inter', sans-serif !important; opacity: 0.75; }}

    section[data-testid="stSidebar"] {{
        background: {"rgba(8, 11, 16, 0.9)" if mode == "dark" else "rgba(255, 255, 255, 0.85)"};
        border-right: 1px solid var(--panel-border);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text-primary) !important; }}

    .stButton > button, .stFormSubmitButton > button {{
        background: rgba(41, 98, 255, 0.10);
        border: 1px solid var(--panel-border-hover);
        color: var(--accent);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.04em;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        box-shadow: 0 0 18px {"rgba(0, 229, 255, 0.4)" if mode == "dark" else "rgba(30, 58, 95, 0.25)"};
        border-color: var(--accent);
        transform: translateY(-1px);
    }}
    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background: rgba(255, 23, 68, 0.14) !important;
        border: 1px solid rgba(255, 23, 68, 0.55) !important;
        color: var(--crimson) !important;
    }}

    .nothierly-brand {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.4rem; }}
    .nothierly-wordmark {{ font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.08em; color: var(--text-primary); }}
    .nothierly-live-dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--mint); box-shadow: 0 0 8px var(--mint);
        display: inline-block; animation: nothierly-live-pulse 1.6s ease-in-out infinite;
    }}
    .nothierly-live-label {{ font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; color: var(--mint); }}
    @keyframes nothierly-live-pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.25; }} }}

    .nothierly-panic-bar {{
        background: rgba(255, 23, 68, 0.15); border: 1px solid var(--crimson); color: {("#ffffff" if mode == "dark" else "#3a0510")};
        text-align: center; font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.12em;
        padding: 0.7rem; border-radius: 10px; margin-bottom: 1.2rem; animation: nothierly-pulse 1.4s ease-in-out infinite;
    }}
    @keyframes nothierly-pulse {{
        0% {{ box-shadow: 0 0 6px 2px rgba(255, 23, 68, 0.4); }}
        50% {{ box-shadow: 0 0 28px 12px rgba(255, 23, 68, 0.8); }}
        100% {{ box-shadow: 0 0 6px 2px rgba(255, 23, 68, 0.4); }}
    }}

    .nothierly-terminal {{
        background: var(--terminal-bg, {p.terminal_bg}); border: 1px solid var(--panel-border-hover); border-radius: 10px;
        padding: 0.8rem 1rem; height: 280px; overflow-y: auto;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.55; color: #c9d1d9;
    }}
    .nothierly-terminal div {{ white-space: pre; }}
    .nothierly-terminal .tag-signal {{ color: {CYAN}; }}
    .nothierly-terminal .tag-decision {{ color: {MINT}; }}
    .nothierly-terminal .tag-veto {{ color: {CRIMSON}; }}

    .nothierly-status-badge {{
        display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        letter-spacing: 0.08em; padding: 3px 10px; border-radius: 100px; text-transform: uppercase;
    }}
    .nothierly-status-safe {{ background: rgba(0,230,118,0.12); color: var(--mint); border: 1px solid rgba(0,230,118,0.35); }}
    .nothierly-status-panic {{ background: rgba(255,23,68,0.14); color: var(--crimson); border: 1px solid rgba(255,23,68,0.4); }}
    .nothierly-status-pending {{ background: rgba(91,107,124,0.14); color: var(--text-dim); border: 1px solid var(--panel-border); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def nothierly_logo_svg(size: int = 40) -> str:
    """The real, existing Nothierly mark -- identical to app.py's own, not a new logo. Pure inline SVG."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 42 42" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="nothierlyMetal" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#F2F4F7"/>
                <stop offset="55%" stop-color="#9AA5B1"/>
                <stop offset="100%" stop-color="#4B5563"/>
            </linearGradient>
            <linearGradient id="nothierlyIndigo" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="{INDIGO}"/>
                <stop offset="100%" stop-color="{CYAN}"/>
            </linearGradient>
        </defs>
        <path d="M21 2 L38 21 L21 40 L4 21 Z" fill="none" stroke="url(#nothierlyMetal)" stroke-width="1.3"/>
        <path d="M14 28 L14 13 L28 28 L28 13" fill="none" stroke="url(#nothierlyIndigo)"
              stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="14" cy="13" r="2.1" fill="{CYAN}"/>
        <circle cx="28" cy="13" r="2.1" fill="{CYAN}"/>
        <circle cx="14" cy="28" r="2.1" fill="{INDIGO}"/>
        <circle cx="28" cy="28" r="2.1" fill="{INDIGO}"/>
        <circle cx="21" cy="20.5" r="1.5" fill="#F2F4F7"/>
    </svg>
    """


def render_brand_header(*, size: int = 34, subtitle: str = "EXECUTION MANAGEMENT SYSTEM") -> None:
    """Logo + wordmark + pulsing LIVE indicator -- the sidebar brand block for every EMS page."""
    st.markdown(
        f'<div class="nothierly-brand">{nothierly_logo_svg(size)}'
        f'<div><div class="nothierly-wordmark">NOTHIERLY</div>'
        f'<div style="font-size:0.62rem;letter-spacing:0.1em;color:var(--text-dim);">{subtitle}</div>'
        f'<div><span class="nothierly-live-dot"></span> '
        f'<span class="nothierly-live-label">LIVE</span></div></div></div>',
        unsafe_allow_html=True,
    )


def status_badge(status: Literal["SAFE", "PANIC", "PENDING"], label: str | None = None) -> str:
    """One real, styled status pill -- ``SYSTEM_STATUS``, agent liveness, order state, etc."""
    css_class = {"SAFE": "nothierly-status-safe", "PANIC": "nothierly-status-panic", "PENDING": "nothierly-status-pending"}[status]
    return f'<span class="nothierly-status-badge {css_class}">{label or status}</span>'


def _as_theme_mode(value: object) -> ThemeMode:
    """Narrow an untyped session-state/widget value to a real ThemeMode, defaulting to dark."""
    return "light" if value == "light" else "dark"


def get_theme_mode() -> ThemeMode:
    """The current theme, from session state -- defaults to dark (this brand's native mode)."""
    return _as_theme_mode(st.session_state.get("nothierly_theme_mode"))


def render_theme_toggle() -> ThemeMode:
    """Sidebar dark/light toggle. Call once per page, before :func:`render_theme_css`."""
    current = get_theme_mode()
    choice = st.sidebar.radio(
        "Thème", options=["dark", "light"], index=0 if current == "dark" else 1,
        format_func=lambda m: "🌑 Vantablack" if m == "dark" else "☀️ Blanc clinique",
        horizontal=True, key="nothierly_theme_mode",
    )
    return _as_theme_mode(choice)

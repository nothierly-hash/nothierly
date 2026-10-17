"""Institutional Due Diligence Dashboard -- honest version, for a real M&A audit audience.

**Why this page does not contain the brief's own literal numbers, stated plainly**: the brief asks
for "Sharpe Ratio : 4.2", "Win Rate : 98.4%", a specific wallet AML score, and a security guarantee
("les clés n'existent jamais en clair dans la RAM") to be shown to real Chief Risk Officers during
real M&A due diligence. None of the systems that would produce those numbers for real exist in
this codebase: there is no real 3-way MPC/KMS key-sharding architecture (``core/security_vault.py``
implements real Fernet-based single-vault encryption, not multi-party computation across AWS
KMS/Azure/on-prem), and no real Chainalysis/wallet-risk-scoring integration anywhere in this
project. A due-diligence audience's entire job is verifying exactly these claims -- presenting
fabricated performance/security/compliance numbers to them, even behind a small disclaimer, would
be a false statement to real auditors making a real financial decision, not a design choice. See
this session's own conversation for the fuller reasoning.

**What this page does instead**: real math applied to a disclosed, illustrative historical
reconstruction (real Sharpe/drawdown/win-rate formulas computed on a labeled simulated series, not
hardcoded "trust me" numbers), and honest, visibly-labeled conceptual diagrams for the security/AML
architecture this project does not yet have -- the same real/simulation disclosure discipline as
``singularity_engine.py`` (see that module's own docstring), reused here rather than reinvented.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DARK_BG = "#000000"
_PANEL_BG = "#0a0e0a"
_GREEN = "#00ff41"
_RED = "#ff2222"
_GRAY = "#8a9a8a"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {_DARK_BG}; font-family: 'Consolas', 'Courier New', monospace; }}
    .idd-panel {{ background: {_PANEL_BG}; border: 1px solid #1a3a1a; border-radius: 4px; padding: 1rem 1.2rem; }}
    .idd-disclosure {{
        border-left: 3px solid {_RED}; background: rgba(255,34,34,0.08);
        padding: 0.7rem 1rem; font-size: 0.85rem; color: #ffb3b3; border-radius: 4px;
    }}
    .idd-badge-real {{ background: rgba(0,255,65,0.15); color: {_GREEN}; border: 1px solid {_GREEN}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    .idd-badge-sim {{ background: rgba(255,34,34,0.12); color: #ff8080; border: 1px solid {_RED}; border-radius: 3px; padding: 0.05rem 0.5rem; font-size: 0.72rem; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏦 Institutional Risk & Compliance Terminal")
st.caption("Nothierly -- Due Diligence M&A")
st.markdown(
    '<div class="idd-disclosure">'
    "⚠️ <b>Note de transparence, destinée explicitement à un public d'audit</b> -- "
    "les chiffres de performance affichés ici sont calculés mathématiquement (Sharpe/drawdown réels) "
    "sur une reconstruction historique <b>illustrative</b>, pas sur un vrai résultat de trading en capital réel. "
    "Le coffre MPC et le bouclier AML sont des schémas d'architecture <b>conceptuels</b> -- "
    "<b>aucune intégration KMS/Azure multi-parties ni aucun scoring Chainalysis réel n'existe dans ce projet aujourd'hui</b>. "
    "Présenter ceci comme un résultat ou une certification déjà obtenue serait une fausse déclaration."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")


# =============================================================================
# Module 1 -- Historical Alpha Verification
# =============================================================================


def _simulate_usdc_depeg_reconstruction() -> pd.DataFrame:
    """A real, disclosed, stylized reconstruction of the real March 2023 USDC depeg's shape (spot
    fell to ~$0.88 around March 11 2023 on SVB exposure news, recovered by March 13) -- not a real
    tick-level replay (no real historical tick feed for this event exists in this project)."""
    n = 200
    t = np.linspace(0, 1, n)
    # Real bug this module's own first live run caught: `sin(...) ** 1.3` on a signed value
    # produced real NaNs the moment the sine went negative (a fractional power of a negative real
    # number is undefined) -- fixed by dropping the fractional exponent (a plain sine already
    # gives the right smooth depeg/recovery shape, no exponent needed).
    spot = np.where(t < 0.3, 1.0, np.where(t < 0.55, 1.0 - 0.12 * np.sin((t - 0.3) / 0.25 * np.pi / 2), 0.88 + 0.12 * (t - 0.55) / 0.45))
    rng = np.random.default_rng(42)
    spot = spot + rng.normal(0, 0.003, n)

    # **A second real bug this module's own first live run caught, fixed here**: an earlier
    # version summed only non-negative terms (`abs(...)` plus a positive floor) into the PnL via
    # `cumsum` -- mathematically guaranteed to never lose money, giving a real Sharpe/drawdown/
    # win-rate calculation a rigged, degenerate 100% win rate / 0% drawdown / Sharpe > 14 every
    # single run. A "real" statistic computed on data engineered to always win is more misleading
    # than an openly-asserted fake number, not less. Fixed with real, signed per-period returns,
    # calibrated (verified live, not guessed) so noise is large enough relative to the signal to
    # produce genuine losing periods -- a real ~7% max drawdown and ~70% win rate, not a fantasy
    # zero-drawdown curve.
    opportunity_signal = np.abs(spot - 1.0) * 0.12
    execution_noise = rng.normal(0.0, 0.006, n - 1)
    period_returns = opportunity_signal[1:] + execution_noise
    pnl = 100_000 * np.concatenate([[1.0], np.cumprod(1 + period_returns)])
    return pd.DataFrame({"t": t, "spot_usdc": spot, "pnl_nothierly": pnl})


@st.fragment(run_every="5s")
def _render_backtester_module() -> None:
    st.markdown(
        '### 📈 Historical Alpha Verification (Tick-Level) &nbsp; <span class="idd-badge-sim">SIMULATION</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Reconstruction illustrative de la forme réelle du décrochage USDC de mars 2023 (SVB) -- "
        "pas un rejeu de données tick réelles (aucun flux historique tick-level de cet événement "
        "n'est intégré à ce projet). Les métriques ci-dessous sont de vrais calculs (Sharpe, "
        "drawdown, taux de réussite) appliqués à cette série simulée, pas des chiffres affirmés."
    )

    df = _simulate_usdc_depeg_reconstruction()
    returns = np.diff(df["pnl_nothierly"]) / df["pnl_nothierly"].to_numpy()[:-1]
    # Real, disclosed methodology limit: no standard annualization factor (sqrt(252) for daily
    # returns, etc.) applies to this illustrative event window -- multiplying by sqrt(len(returns))
    # instead (a real bug an earlier version of this page had) treats the whole ~200-tick sample as
    # if it were 200 independent years of returns, inflating the ratio to an absurd, non-credible
    # value. Reported here as the real, raw (non-annualized) mean/std ratio instead.
    raw_sharpe_like_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    running_max = np.maximum.accumulate(df["pnl_nothierly"])
    drawdown: np.ndarray = (df["pnl_nothierly"] - running_max) / running_max
    max_drawdown = float(-drawdown.min() * 100)
    win_rate = float(np.mean(returns > 0) * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Ratio rendement/risque réel (non annualisé)", f"{raw_sharpe_like_ratio:.2f}")
    c2.metric("Max Drawdown réel (sur simulation)", f"{max_drawdown:.2f}%")
    c3.metric("Win Rate réel (sur simulation)", f"{win_rate:.1f}%")
    st.caption("Aucune convention d'annualisation standard ne s'applique à cette fenêtre illustrative -- ratio brut affiché, pas un Sharpe annualisé.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["spot_usdc"], name="USDC Spot (reconstruction)", line=dict(color=_RED)))
    fig.add_trace(go.Scatter(x=df["t"], y=df["pnl_nothierly"] / df["pnl_nothierly"].iloc[0], name="PnL Nothierly (illustratif, normalisé)", line=dict(color=_GREEN), yaxis="y2"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=_PANEL_BG, plot_bgcolor=_PANEL_BG, height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="USDC ($)"), yaxis2=dict(title="PnL (normalisé)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch", key=f"backtest_{time.time()}")


# =============================================================================
# Module 2 -- MPC Vault (conceptual architecture)
# =============================================================================


@st.fragment(run_every="3s")
def _render_mpc_vault_module() -> None:
    st.markdown(
        '### 🔐 Zero-Knowledge Security & MPC Architecture &nbsp; <span class="idd-badge-sim">SIMULATION / VISION R&D</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Schéma d'architecture conceptuel -- ce projet chiffre réellement ses secrets via "
        "`core/security_vault.py` (Fernet, coffre unique), **pas** un partage de clé multi-parties "
        "(MPC) réel entre AWS KMS, Azure Key Vault et un serveur on-premise. Aucune de ces trois "
        "intégrations n'existe dans ce projet aujourd'hui."
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, sub in [
        (c1, "🔑 Clé API (factice)", "Jamais stockée en clair -- concept"),
        (c2, "🇺🇸 Fragment 1", "AWS KMS (concept)"),
        (c3, "🇪🇺 Fragment 2", "Azure Key Vault (concept)"),
        (c4, "🏠 Fragment 3", "On-Premise Nothierly (concept)"),
    ]:
        col.markdown(f'<div class="idd-panel" style="text-align:center;"><b>{label}</b><br><span style="color:{_GRAY};font-size:0.8rem;">{sub}</span></div>', unsafe_allow_html=True)

    blink = int(time.time()) % 2 == 0
    color = _GREEN if blink else "#004411"
    st.markdown(
        f'<div style="margin-top:0.8rem;padding:0.6rem;border:1px solid {color};border-radius:4px;color:{color};text-align:center;font-weight:700;">'
        "CONCEPT -- No Single Point of Failure (vision architecturale, non implémentée)"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Module 3 -- RegTech / AML Shield (conceptual mockup)
# =============================================================================

_AML_SCRIPT = [
    (99, "Clean", True), (95, "Clean", True), (87, "Clean", True),
    (12, "High Risk (exposition simulée à un mixer)", False), (91, "Clean", True), (78, "Clean", True),
]


@st.fragment(run_every="2s")
def _render_aml_module() -> None:
    st.markdown(
        '### 🛡️ Real-Time AML Shield &nbsp; <span class="idd-badge-sim">SIMULATION / VISION R&D</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Mockup conceptuel -- **aucune intégration Chainalysis ou scoring de wallet réel n'existe "
        "dans ce projet aujourd'hui**. Les scores et statuts ci-dessous sont des exemples fixes "
        "pour illustrer le flux d'écran visé, jamais un vrai blocage de fonds."
    )

    col_table, col_gauge = st.columns([2, 1])
    with col_table:
        rows = []
        for score, label, ok in _AML_SCRIPT:
            status = "🟢 Approuvé (exemple)" if ok else "🔴 Bloqué (exemple)"
            rows.append({"Wallet Score (exemple)": score, "Évaluation (exemple)": label, "Statut": status})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=8,
            title={"text": "Latence de scan (illustrative, ms)"},
            gauge={"axis": {"range": [0, 50]}, "bar": {"color": _GREEN},
                   "steps": [{"range": [0, 15], "color": "#0a2a0a"}, {"range": [15, 50], "color": "#2a0a0a"}]},
        ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=_PANEL_BG, height=220, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", key=f"aml_gauge_{time.time()}")


_render_backtester_module()
_render_mpc_vault_module()
_render_aml_module()

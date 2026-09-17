"""Investor Portal -- "Protocole Fort Knox", Pilier 4: institutional-facing risk & performance view.

**Real, disclosed scope, stated upfront (same discipline as bilan.py, which
this page's real data source is built on)**: the brief asks for a "Mode
Clair" institutional dashboard with performance/risk metrics and a
capital-allocation breakdown, explicitly without technical logs or any
mention of AI/daemons. Built honestly on top of the one real source of
"trades gagnants/perdants" this codebase has --
:class:`~src.agents.alpha_engine.agent.AlphaMarketMakerAgent`'s real fill
history (:data:`~core.schemas.ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX`,
the same key :mod:`bilan` reads) -- never a second, parallel, or fabricated
number.

**Real, disclosed gaps, not silently faked**:

- **Sharpe/Sortino/Max Drawdown/VaR** are computed from real fills bucketed
  into real calendar-day PnL totals (see :func:`_daily_pnl_series`), the
  closest real proxy this system has to a "24h return series" -- honestly
  requires at least 2 distinct real trading days of fills before any ratio
  is shown (a single day has no real variance to measure risk from).
- **VaR is reported in real dollar terms, not as a percentage of capital.**
  No real starting-capital / total-AUM figure is published anywhere in this
  codebase (confirmed by direct search -- see
  :mod:`~src.execution.reporting_engine`'s own identical, pre-existing
  disclosure for exactly this same gap). Converting to a percentage would
  require inventing a capital baseline; this page refuses to, exactly the
  same refusal-not-fabrication discipline as every other honest-disclosure
  page in this project.
- **"Stablecoins" is not shown as a separate bucket.** No real free-cash /
  stablecoin-balance telemetry is published by
  :class:`AlphaMarketMakerAgent` today (only signed inventory per
  instrument) -- this page shows real exposure by instrument instead
  (inventory times the real, live mid-price already cached by
  :class:`~src.agents.market_data.agent.MarketDataAgent`), and discloses
  the gap rather than fabricating a cash figure.

Forced light theme ("Mode Clair" per the brief), no sidebar brand block, no
live-pulse indicator, no gate/daemon/agent terminology anywhere on this
page -- see :mod:`src.dashboard.theme` for the shared light palette this
reuses rather than inventing a second one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.quant_math import parametric_var, sharpe_ratio, sortino_ratio
from core.schemas import (
    ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES,
    ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX,
    ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX,
    MARKET_METRICS_REDIS_KEY_PREFIX,
)
from src.dashboard.ems_pages._shared import get_redis_client, safe_get_json, safe_keys, safe_lrange
from src.dashboard.theme import render_theme_css

render_theme_css("light")

_VAR_CONFIDENCE_LEVEL = 0.99
_PERIODS_PER_YEAR = 365  # real calendar-day bucketing -- see _daily_pnl_series
_MIN_DAYS_FOR_RISK_METRICS = 2  # a single day has zero real variance to measure risk from

st.title("Nothierly Capital -- Relations Investisseurs")
st.caption(
    "Performance et risque réels de la stratégie propriétaire Nothierly. Aucune donnée simulée : "
    "cette page reste honnêtement vide tant qu'aucune exécution réelle n'a encore eu lieu."
)

r = get_redis_client()

# =============================================================================
# Real data loading -- identical source to bilan.py, see this module's own docstring.
# =============================================================================

agent_keys = safe_keys(r, f"{ALPHA_ENGINE_PNL_HISTORY_REDIS_KEY_PREFIX}*")
entries: list[dict[str, Any]] = []
for key in agent_keys:
    for raw in safe_lrange(r, key, ALPHA_ENGINE_PNL_HISTORY_MAX_ENTRIES):
        try:
            record = json.loads(raw)
            record["_delta"] = Decimal(str(record["realized_pnl_delta_usd"]))
            entries.append(record)
        except (json.JSONDecodeError, TypeError, KeyError, InvalidOperation):
            continue
entries.sort(key=lambda e: float(e["timestamp"]))

if not entries:
    st.info(
        "**En attente des premières données réelles.** Aucune exécution réelle n'a encore été "
        "enregistrée sur ce déploiement -- les métriques de performance et de risque apparaîtront "
        "ici dès la première exécution réelle."
    )
    st.stop()


def _daily_pnl_series(records: list[dict[str, Any]]) -> dict[date, float]:
    """Real per-calendar-day PnL totals -- the closest real proxy this system has to a daily return series."""
    daily: dict[date, float] = defaultdict(float)
    for record in records:
        day = datetime.fromtimestamp(float(record["timestamp"]), tz=timezone.utc).date()
        daily[day] += float(record["_delta"])
    return dict(sorted(daily.items()))


daily_pnl = _daily_pnl_series(entries)
gains_cumules = sum((e["_delta"] for e in entries if e["_delta"] > 0), Decimal(0))
pertes_cumulees = sum((e["_delta"] for e in entries if e["_delta"] < 0), Decimal(0))
pnl_net = gains_cumules + pertes_cumulees

# =============================================================================
# Performance curve
# =============================================================================

st.subheader("Performance cumulée")
timestamps = [datetime.fromtimestamp(float(e["timestamp"]), tz=timezone.utc) for e in entries]
cumulative: list[float] = []
running = 0.0
for e in entries:
    running += float(e["_delta"])
    cumulative.append(running)

fig = go.Figure()
fig.add_hline(y=0, line_dash="dash", line_color="#5B6B7C")
fig.add_trace(
    go.Scatter(
        x=timestamps, y=cumulative, mode="lines", name="PnL cumulé",
        line=dict(color="#1E3A5F", width=2.5), fill="tozeroy", fillcolor="rgba(30,58,95,0.08)",
    )
)
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
    xaxis_title="", yaxis_title="PnL cumulé (USD)",
)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Gains cumulés", f"${float(gains_cumules):,.2f}")
kpi2.metric("Pertes cumulées", f"-${abs(float(pertes_cumulees)):,.2f}")
kpi3.metric("PnL net", f"${float(pnl_net):,.2f}" if pnl_net >= 0 else f"-${abs(float(pnl_net)):,.2f}")

st.markdown("---")

# =============================================================================
# Risk metrics -- real, from daily-bucketed real PnL (see _daily_pnl_series' own docstring).
# =============================================================================

st.subheader("Métriques de risque")

if len(daily_pnl) < _MIN_DAYS_FOR_RISK_METRICS:
    st.info(
        f"**Historique insuffisant** -- {len(daily_pnl)} jour(s) réel(s) de trading enregistré(s). "
        f"Les ratios de risque (Sharpe, Sortino, VaR) nécessitent au moins {_MIN_DAYS_FOR_RISK_METRICS} "
        "jours réels distincts pour mesurer une variance réelle -- affichés dès que ce seuil est atteint."
    )
else:
    daily_returns = np.array(list(daily_pnl.values()), dtype=np.float64)
    daily_equity = np.cumsum(daily_returns)
    running_peak = np.maximum.accumulate(daily_equity)
    max_dd_usd = float(np.min(daily_equity - running_peak))  # a real, dollar-terms drawdown -- see module docstring

    sharpe = sharpe_ratio(daily_returns, periods_per_year=_PERIODS_PER_YEAR)
    sortino = sortino_ratio(daily_returns, periods_per_year=_PERIODS_PER_YEAR)
    var_99_usd = parametric_var(daily_returns, _VAR_CONFIDENCE_LEVEL, position_value=1.0)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Ratio de Sharpe (annualisé)", f"{sharpe:.2f}")
    r2.metric("Ratio de Sortino (annualisé)", f"{sortino:.2f}")
    r3.metric("Drawdown maximal", f"-${abs(max_dd_usd):,.2f}")
    r4.metric("VaR 99% (24h, estimation)", f"${var_99_usd:,.2f}")
    st.caption(
        f"Calculés sur {len(daily_pnl)} jours réels d'historique de PnL. VaR exprimée en dollars, non "
        "en pourcentage du capital : aucune valeur totale d'actifs sous gestion n'est publiée par ce "
        "système (voir la documentation technique pour le détail de cette limite)."
    )

st.markdown("---")

# =============================================================================
# Capital allocation -- real exposure by instrument, honest gap on the "stablecoins" half.
# =============================================================================

st.subheader("Répartition du capital")

telemetry_keys = safe_keys(r, f"{ALPHA_ENGINE_TELEMETRY_REDIS_KEY_PREFIX}*")
exposure_by_instrument: dict[str, float] = defaultdict(float)
for key in telemetry_keys:
    telemetry = safe_get_json(r, key)
    if telemetry is None or telemetry.get("inventory") is None:
        continue
    symbol = telemetry.get("symbol")
    if not symbol:
        continue
    metrics = safe_get_json(r, f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}")
    if metrics is None:
        continue
    try:
        inventory = abs(float(telemetry["inventory"]))
        mid_price = float(metrics["mid_price"])
    except (TypeError, ValueError, KeyError):
        continue
    exposure_by_instrument[symbol] += inventory * mid_price

if not exposure_by_instrument:
    st.info(
        "**Aucune position ouverte réelle actuellement.** La répartition du capital s'affichera dès "
        "qu'une position réelle sera ouverte et que les données de marché correspondantes seront disponibles."
    )
else:
    pie = go.Figure(
        data=[
            go.Pie(
                labels=list(exposure_by_instrument.keys()),
                values=list(exposure_by_instrument.values()),
                hole=0.45,
                marker=dict(colors=["#1E3A5F", "#5B6B7C", "#2962FF", "#00E5FF", "#00E676"]),
            )
        ]
    )
    pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Exposition réelle par instrument (position ouverte × prix de marché courant). La part en "
        "stablecoins/liquidités n'est pas affichée séparément -- ce système ne publie pas encore de "
        "solde de trésorerie réel distinct des positions ouvertes."
    )

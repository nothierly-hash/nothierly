"""Cognitive Guardrail -- real UI for risk.behavioral_profiler.BehavioralProfiler. Chapter 13, Étape 5.9.

**Real, load-bearing correction to the brief, stated upfront**: the
brief asks for "l'historique des alertes de Revenge Trading
interceptées". There is no real interception anywhere in this codebase
-- no gate blocks or delays an order based on a Tilt Score (see
``api_middleware/gateway.py``: nothing in the order-submission path
calls this module). What this page shows instead is real: a **live**
Tilt Score computed, on every page load, against each client's actual
recorded fill history from ``NOTHIERLY_FILLS_BLOTTER`` (the same real
Redis list ``order_blotter.py`` reads) -- not a persisted "N alerts
intercepted" counter that would misrepresent something as blocked when
nothing ever was.

**Real, per-detector data honesty**: :mod:`risk.behavioral_profiler`'s
own docstring is explicit that ``realized_pnl_usd``/``stop_loss_price``
have no real producer in this codebase (Execution-as-a-Service has no
client P&L visibility, and no order model here has a stop-loss field).
``detect_revenge_trading``/``detect_stop_loss_creep`` therefore
genuinely return ``insufficient_data=True`` against real data below --
shown as such, not hidden. ``detect_overtrading`` is the one detector
with real supporting data (real fill timestamps), and is the only one
that can genuinely fire from real client activity today.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from risk.behavioral_profiler import ActionType, BehavioralProfiler, ClientAction, TiltReport
from src.dashboard.ems_pages._shared import demo_mode_toggle, get_redis_client, not_yet_implemented, safe_lrange

st.title("🧭 Cognitive Guardrail")
st.caption("Tilt Score réel par client, calculé en direct sur l'historique de fills réel -- pas un journal d'alertes interceptées (voir ci-dessous).")

_FILLS_BLOTTER_REDIS_KEY = "NOTHIERLY_FILLS_BLOTTER"
_TILT_COLOR = {"NO_ACTION": "🟢", "WARN_ADVISOR": "🟡", "LOCK_ACCOUNT": "🔴"}


def _load_real_actions_by_client() -> dict[str, list[ClientAction]]:
    r = get_redis_client()
    raw_entries = safe_lrange(r, _FILLS_BLOTTER_REDIS_KEY, 500)
    by_client: dict[str, list[ClientAction]] = {}
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            client_id = entry.get("client_id")
            if not client_id:
                continue
            action = ClientAction(
                client_id=client_id,
                timestamp=datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc),
                action_type=ActionType.ORDER_FILLED,
                instrument=entry["instrument"],
                quantity=abs(float(entry["units"])),
                side="buy" if entry["action"] == "ENTRY_LONG" else "sell",
            )
        except (KeyError, ValueError, TypeError):
            continue
        by_client.setdefault(client_id, []).append(action)
    return by_client


def _generate_report(client_id: str, actions: list[ClientAction]) -> TiltReport:
    profiler = BehavioralProfiler(client_id, max_history=1000)
    for action in sorted(actions, key=lambda a: a.timestamp):
        profiler.record_action(action)
    return profiler.generate_report()


def _demo_actions(client_id: str) -> list[ClientAction]:
    """A synthetic martingale pattern -- for illustration only, real P&L data doesn't exist in this system (see module docstring)."""
    from datetime import timedelta

    t0 = datetime.now(timezone.utc)
    return [
        ClientAction(client_id=client_id, timestamp=t0, action_type=ActionType.ORDER_FILLED, instrument="BTC/USDT", quantity=1.0, realized_pnl_usd=-800.0),
        ClientAction(client_id=client_id, timestamp=t0 + timedelta(seconds=20), action_type=ActionType.ORDER_FILLED, instrument="BTC/USDT", quantity=3.0, realized_pnl_usd=-1500.0),
        ClientAction(client_id=client_id, timestamp=t0 + timedelta(seconds=40), action_type=ActionType.ORDER_FILLED, instrument="BTC/USDT", quantity=8.0),
        ClientAction(client_id=client_id, timestamp=t0 - timedelta(minutes=1), action_type=ActionType.STOP_LOSS_MOVED, instrument="BTC/USDT", quantity=1.0, stop_loss_price=64000),
        ClientAction(client_id=client_id, timestamp=t0 - timedelta(seconds=30), action_type=ActionType.STOP_LOSS_MOVED, instrument="BTC/USDT", quantity=1.0, stop_loss_price=63500),
        ClientAction(client_id=client_id, timestamp=t0 - timedelta(seconds=5), action_type=ActionType.STOP_LOSS_MOVED, instrument="BTC/USDT", quantity=1.0, stop_loss_price=63000),
    ]


demo_on = demo_mode_toggle(key="guardrail_demo", label="🎭 Mode Démo (schéma de revenge trading simulé)")

by_client = _load_real_actions_by_client()
if demo_on:
    by_client["DEMO-client"] = _demo_actions("DEMO-client")

reports: dict[str, TiltReport] = {cid: _generate_report(cid, actions) for cid, actions in by_client.items()}

if not reports:
    not_yet_implemented(
        "Tilt Score par client",
        reason="Aucun fill avec client_id réel dans NOTHIERLY_FILLS_BLOTTER pour l'instant -- passez une commande via le Command Center "
        "(mode SHADOW ou réel) pour peupler des données réelles, ou activez le Mode Démo ci-dessus.",
    )
    st.stop()

st.subheader("Vue d'ensemble")
summary_rows = [
    {
        "Client": cid, "Tilt Score": round(r.tilt_score, 3),
        "Action recommandée": f"{_TILT_COLOR.get(r.recommended_action, '⚪')} {r.recommended_action}",
        "Détections actives": sum(1 for d in r.detections if d.detected),
        "Actions réelles enregistrées": len(by_client.get(cid, [])),
    }
    for cid, r in sorted(reports.items(), key=lambda kv: kv[1].tilt_score, reverse=True)
]
st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

st.subheader("Détail par client")
for cid, report in sorted(reports.items(), key=lambda kv: kv[1].tilt_score, reverse=True):
    with st.expander(f"{_TILT_COLOR.get(report.recommended_action, '⚪')} {cid} -- Tilt Score {report.tilt_score:.3f} -- {report.recommended_action}"):
        for detection in report.detections:
            if detection.insufficient_data:
                st.caption(f"**{detection.bias_name}** -- données réelles insuffisantes pour ce client (voir docstring du module).")
                continue
            icon = "🔴" if detection.detected else "🟢"
            st.markdown(f"{icon} **{detection.bias_name}** -- confiance {detection.confidence:.2f}")
            for ev in detection.evidence:
                st.caption(f"↳ {ev}")

st.caption(
    "Aucune commande n'a jamais été réellement bloquée par ce Tilt Score -- ce module calcule un signal, "
    "il n'existe aujourd'hui aucun point d'intégration dans api_middleware/gateway.py qui agirait dessus."
)

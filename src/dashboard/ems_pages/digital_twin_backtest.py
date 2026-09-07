"""Digital Twin Backtesting -- upload a real trade history, replay it against real historical L2 order-book data.

**Real engine reused, not rebuilt**: this page is a thin UI layer over
:func:`~simulation.digital_twin.run_digital_twin_backtest` (Chapter 15,
Pillar 3, "The Causal Digital Twin"). See that module's own docstring for
exactly what's real (TWAP slicing via ``src.execution.macro_strategist.
slice_twap``, book-walking via ``pricing.pre_trade_oracle._walk_book``, a
real :class:`~src.agents.rl_signal.policy_model.RLPolicyModel` signal
computed at each historical moment) versus deliberately, honestly mocked
(:class:`~simulation.digital_twin.MockGuardianAgent` -- a real risk rule,
not the production LLM-based Guardian, for cost/speed reasons documented
there).

**Honest coverage, stated upfront, not buried in a report footer**: this
project has real historical L2 order-book + trade data (Tardis format)
for exactly one real window --
:data:`~simulation.digital_twin.COVERAGE_START_UTC` to
:data:`~simulation.digital_twin.COVERAGE_END_UTC`, covering only
:data:`~simulation.digital_twin.SAMPLE_BOOK_FILES`. A real prospect's
uploaded CSV will very likely fall outside this window in whole or in
part -- those trades come back with ``replay_available=False`` and are
excluded from every aggregate figure below, never silently padded with an
invented number.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from simulation.digital_twin import (
    COVERAGE_END_UTC,
    COVERAGE_START_UTC,
    DigitalTwinError,
    SAMPLE_BOOK_FILES,
    build_sample_client_trades_csv,
    run_digital_twin_backtest,
)

st.title("🕰️ Digital Twin Backtesting")
st.caption(
    "Rejoue un historique réel de trades client contre les vraies données L2 historiques (Tardis) -- "
    "compare le coût naïf réellement rapporté au coût simulé d'une exécution Nothierly AI (TWAP + carnet réel)."
)

st.warning(
    f"⚠️ **Couverture réelle limitée** -- ce backtest ne peut rejouer que des trades sur "
    f"{', '.join(sorted(SAMPLE_BOOK_FILES))}, entre {COVERAGE_START_UTC.strftime('%Y-%m-%d %H:%M:%S UTC')} "
    f"et {COVERAGE_END_UTC.strftime('%Y-%m-%d %H:%M:%S UTC')}. Tout trade hors de cette fenêtre est honnêtement "
    "marqué non-rejouable et exclu des totaux ci-dessous -- jamais remplacé par un chiffre inventé."
)

col_upload, col_sample = st.columns([2, 1])

with col_sample:
    st.caption(
        "Pas de CSV sous la main ? Télécharge un exemple réel -- vrais trades historiques de la fenêtre de "
        "couverture ci-dessus, pas des données fabriquées -- pour tester l'outil immédiatement."
    )
    sample_symbol = st.selectbox("Symbole de l'exemple", sorted(SAMPLE_BOOK_FILES), key="digital_twin_sample_symbol")
    try:
        sample_csv = build_sample_client_trades_csv(sample_symbol, n_trades=8)
    except DigitalTwinError as exc:
        st.error(str(exc))
    else:
        st.download_button(
            "📥 Télécharger un exemple réel (CSV)",
            data=sample_csv,
            file_name=f"sample_client_trades_{sample_symbol.lower()}.csv",
            mime="text/csv",
            width="stretch",
        )

with col_upload:
    uploaded = st.file_uploader(
        "📤 Historique de trades client (CSV -- colonnes requises : timestamp, side, price, size ou quantity ; symbol optionnel)",
        type=["csv"],
    )

col_params1, col_params2 = st.columns(2)
execution_window_minutes = col_params1.number_input(
    "Fenêtre d'exécution simulée (minutes)", min_value=1, max_value=60, value=1, step=1,
    help="Durée que met le TWAP simulé de Nothierly AI pour exécuter chaque trade rejoué.",
)
num_slices = col_params2.number_input(
    "Nombre de slices TWAP", min_value=1, max_value=20, value=5, step=1,
    help="Nombre de sous-ordres dans lesquels chaque trade rejoué est découpé.",
)

if uploaded is None:
    st.info("Upload un CSV réel de trades client (ou télécharge l'exemple réel ci-dessus) pour lancer un backtest.")
elif st.button("▶️ Lancer le backtest réel", type="primary"):
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        with st.spinner("Rejeu réel en cours contre les données historiques L2..."):
            report = run_digital_twin_backtest(
                tmp_path, execution_window_minutes=int(execution_window_minutes), num_slices=int(num_slices),
            )
    except DigitalTwinError as exc:
        st.error(f"🚨 {exc}")
    else:
        st.success(f"✅ {report.coverage_note}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trades totaux", report.trades_total)
        col2.metric("Trades rejoués (réel)", report.trades_replayed)
        col3.metric("Coût naïf total (réel)", f"${float(report.total_naive_cost_usd):,.2f}")
        col4.metric("Slippage économisé (réel)", f"${float(report.total_slippage_saved_usd):,.2f}")

        st.markdown("---")
        st.subheader("Détail par trade (réel)")
        display_rows = [
            {
                "Horodatage": row.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Symbole": row.symbol,
                "Côté": row.side.value,
                "Quantité": float(row.quantity),
                "Rejouable": "✅" if row.replay_available else "❌",
                "Prix Naïf": float(row.naive_price),
                "Prix Simulé IA": float(row.ai_simulated_price) if row.ai_simulated_price is not None else None,
                "Slippage Économisé ($)": float(row.slippage_saved_usd) if row.slippage_saved_usd is not None else None,
                "Signal RL": row.rl_signal,
                "Confiance RL": round(row.rl_confidence, 3) if row.rl_confidence is not None else None,
                "Verdict Guardian": row.guardian_verdict.reason if row.guardian_verdict is not None else "--",
                "Raison (si non-rejoué)": row.unavailable_reason or "--",
            }
            for row in report.rows
        ]
        st.dataframe(display_rows, width="stretch", hide_index=True, height=420)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

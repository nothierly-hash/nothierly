"""Intelligence & Brain -- data-science evidence, real research results plus a real, on-demand live analysis.

**Real, but research-only, historical evidence**: this project's
already-produced model-analysis artifacts (Chapter 12's research
chapter, ``research/``, ``scripts/audit_matrix.py``) are real results,
frozen at the time they were run -- not a live, continuously-updating
feed.

**SHAP / RL reward curves / the 3-run ablation study, now real and
on-demand below**: this page used to say none of these ran anywhere but
an offline script. :func:`~scripts.audit_matrix.build_real_audit_dataset`,
:func:`~scripts.audit_matrix.audit_xgboost_shap`,
:func:`~scripts.audit_matrix.audit_rl_convergence`, and
:func:`~scripts.audit_matrix.audit_ablation_study` are reused here,
unchanged, from one real button -- a genuine XGBoost + PPO training run
against real FRED (EUR/USD, 10Y Treasury) data, not a re-plot of stale
numbers or a second, parallel implementation.

**"Continu", corrected**: the brief's own "preuve mathématique continue
de non-surapprentissage" framing assumed an always-running verification
process. None exists (and building one -- a scheduled retrain-and-
compare job with real alerting on regression -- is real, separate,
larger scope than this page). What *is* real and shipped here: the same
3-run ablation study ``scripts/audit_matrix.py`` already computes
offline, run live, on demand, from this button -- honestly labeled
on-demand, not continuous.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from scripts.audit_matrix import (
    audit_ablation_study,
    audit_rl_convergence,
    audit_xgboost_shap,
    build_real_audit_dataset,
)
from scripts.train_model import save_model, train_xgboost_classifier
from scripts.train_ppo import export_model
from src.agents.supervisor.ml_engine import SupervisorMLModel

st.title("🧠 Intelligence & Brain")
st.caption("Preuves de convergence des modèles -- feature importance, courbes de reward, détection de surapprentissage.")

st.subheader("Résultats réels déjà produits (Chapitre 12, recherche hors-ligne)")
st.markdown(
    """
- **Checkpoint XGBoost (Étape 3)** : 58.8% de précision directionnelle réelle sur données Tardis.dev réelles -- `research/checkpoint_xgboost.py`.
- **Behavioral Cloning (Étape 3.6)** : 92.3% de précision de correspondance Phase A -- `research/imitation_learning.py`.
- **4 itérations RL de market making (Étape 4.0-4.3)** : toxic-fill rate, Maker/Taker split, tous mesurés réellement -- `research/train_market_maker.py`.
    """
)
st.caption("Ces résultats sont réels et documentés, mais figés au moment de leur exécution -- pas un flux en direct.")

st.markdown("---")
st.subheader("🔬 Analyse ML réelle, à la demande")
st.caption(
    "Un vrai entraînement (XGBoost + PPO) sur des données réelles FRED (EUR/USD, 10Y Treasury), en direct -- "
    "réutilise scripts/audit_matrix.py sans le modifier. Réel mais **à la demande, pas continu** (voir le "
    "docstring de cette page)."
)

total_timesteps = st.number_input(
    "Pas d'entraînement RL réels (PPO)", min_value=2048, max_value=200_000, value=8192, step=2048,
    help="Plus élevé = courbe RL plus mature mais plus lent. 40 000 est la référence hors-ligne complète de scripts/audit_matrix.py.",
)

if st.button("▶️ Lancer l'analyse ML réelle complète (peut prendre ~1 minute)", type="primary"):
    with st.spinner("Récupération des données réelles FRED + entraînement réel (XGBoost + PPO) en cours..."):
        features, prices, aligned_frame = build_real_audit_dataset()
        labels = aligned_frame["label"].to_numpy(dtype=np.int64)

        xgb_raw_model, scaler, xgb_metrics = train_xgboost_classifier(features, labels)
        shap_result = audit_xgboost_shap(xgb_raw_model, features, SupervisorMLModel.FEATURE_NAMES)
        rl_result, rl_model = audit_rl_convergence(features, prices, total_timesteps=int(total_timesteps))

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            xgb_model_path, xgb_scaler_path = save_model(xgb_raw_model, scaler, models_dir=tmp_path)
            rl_model_path = export_model(rl_model, models_dir=tmp_path)
            xgb_wrapped = SupervisorMLModel(model_path=str(xgb_model_path), scaler_path=str(xgb_scaler_path))
            ablation_result = audit_ablation_study(xgb_wrapped, str(rl_model_path), features, prices, aligned_frame)

    st.session_state["intelligence_ml_analysis"] = {
        "xgb_metrics": xgb_metrics, "shap": shap_result, "rl": rl_result, "ablation": ablation_result,
    }
    st.toast("Analyse ML réelle terminée.", icon="✅")

analysis: dict[str, Any] | None = st.session_state.get("intelligence_ml_analysis")
if analysis is not None:
    st.markdown("#### Feature Importance SHAP (réel)")
    shap_rows = [
        {"Feature": name, "Importance SHAP (%)": pct}
        for name, pct in analysis["shap"]["feature_importance_pct"].items()
    ]
    shap_rows.sort(key=lambda row: row["Importance SHAP (%)"], reverse=True)
    st.dataframe(shap_rows, width="stretch", hide_index=True)
    st.caption(
        f"Précision holdout réelle : {analysis['xgb_metrics']['accuracy']:.1%} · "
        f"ROC-AUC réel : {analysis['xgb_metrics']['roc_auc']:.3f} · {analysis['shap']['detail']}"
    )

    st.markdown("#### Courbe de Reward RL (réelle)")
    col_reward, col_entropy = st.columns(2)
    col_reward.line_chart(pd.DataFrame({"Reward moyen (réel)": analysis["rl"]["reward_curve"]}), height=220)
    col_entropy.line_chart(pd.DataFrame({"Entropie (réelle)": analysis["rl"]["entropy_curve"]}), height=220)
    st.caption(f"{analysis['rl']['num_rollouts']} rollouts réels · {analysis['rl']['detail']}")

    st.markdown("#### Étude d'ablation réelle (3 runs) -- preuve de non-surapprentissage, à la demande")
    ablation_rows = [{"Run": name, **metrics} for name, metrics in analysis["ablation"]["runs"].items()]
    st.dataframe(ablation_rows, width="stretch", hide_index=True)
    for finding in analysis["ablation"]["findings"]:
        st.info(finding)

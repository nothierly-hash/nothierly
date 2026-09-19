"""Real, executable tests for risk/meta_labeling.py.

See that module's own docstring for the real gap this closes
(ExecutionAgent already computes a consensus confidence and throws it
away after one log line) and the real, load-bearing correctness
property these tests check first: an untrained skeleton model's
uninformative 0.5 output must map to an exact 1.0 size multiplier, so
wiring this in changes nothing about today's real sizing until a real
model is actually trained and loaded.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from risk.meta_labeling import MetaLabelModel, size_multiplier_from_confidence

# =============================================================================
# size_multiplier_from_confidence
# =============================================================================


def test_neutral_confidence_maps_to_exactly_one_multiplier() -> None:
    """The real correctness property this module exists to guarantee: no real trained model yet
    (skeleton always predicts 0.5) must mean zero change to today's real Gate 3 sizing."""
    assert size_multiplier_from_confidence(0.5) == pytest.approx(1.0)


def test_high_confidence_scales_above_one_capped_at_max_multiplier() -> None:
    assert size_multiplier_from_confidence(1.0, max_multiplier=1.5) == pytest.approx(1.5)
    assert size_multiplier_from_confidence(0.6, max_multiplier=10.0) == pytest.approx(0.6 / 0.5)


def test_confidence_below_veto_threshold_is_a_real_hard_veto() -> None:
    assert size_multiplier_from_confidence(0.1, veto_below=0.35) == 0.0
    assert size_multiplier_from_confidence(0.34, veto_below=0.35) == 0.0


def test_confidence_at_or_above_veto_threshold_is_not_vetoed() -> None:
    result = size_multiplier_from_confidence(0.35, veto_below=0.35)
    assert result > 0.0
    assert result == pytest.approx(0.35 / 0.5)


def test_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        size_multiplier_from_confidence(-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        size_multiplier_from_confidence(1.1)


def test_rejects_veto_below_at_or_above_neutral_confidence() -> None:
    """A veto threshold >= 0.5 would veto the untrained skeleton's own default output -- must be refused."""
    with pytest.raises(ValueError, match="veto_below must be"):
        size_multiplier_from_confidence(0.5, veto_below=0.5)
    with pytest.raises(ValueError, match="veto_below must be"):
        size_multiplier_from_confidence(0.5, veto_below=-0.1)


def test_rejects_max_multiplier_below_one() -> None:
    with pytest.raises(ValueError, match="max_multiplier must be"):
        size_multiplier_from_confidence(0.5, max_multiplier=0.9)


# =============================================================================
# MetaLabelModel -- skeleton mode
# =============================================================================


def test_skeleton_model_always_predicts_neutral_confidence() -> None:
    model = MetaLabelModel()
    features_a = MetaLabelModel.build_feature_vector(
        min_confidence=0.99, confidence_spread=0.0, spread=0.5, recent_conflict_count=0,
    )
    features_b = MetaLabelModel.build_feature_vector(
        min_confidence=0.1, confidence_spread=0.8, spread=5.0, recent_conflict_count=10,
    )
    assert model.predict_confidence(features_a) == pytest.approx(0.5)
    assert model.predict_confidence(features_b) == pytest.approx(0.5)


def test_build_feature_vector_preserves_field_order() -> None:
    features = MetaLabelModel.build_feature_vector(
        min_confidence=0.7, confidence_spread=0.2, spread=1.5, recent_conflict_count=3,
    )
    assert features.tolist() == [0.7, 0.2, 1.5, 3.0]
    assert MetaLabelModel.FEATURE_NAMES == ("min_confidence", "confidence_spread", "spread", "recent_conflict_count")


def test_predict_confidence_rejects_wrong_shape() -> None:
    model = MetaLabelModel()
    with pytest.raises(ValueError, match="Expected feature vector of shape"):
        model.predict_confidence(np.array([0.5, 0.5]))


# =============================================================================
# MetaLabelModel -- real fitted artifact round-trip
# =============================================================================


def test_real_fitted_model_round_trips_through_save_and_load(tmp_path: Path) -> None:
    """A genuinely fitted model (min_confidence alone perfectly separates the two classes) must
    still discriminate after a real save_model/joblib round-trip -- not just in-process."""
    rng = np.random.default_rng(0)
    n = 200
    min_confidence = rng.uniform(0.0, 1.0, n)
    other_features = rng.normal(size=(n, 3))
    features = np.column_stack([min_confidence, other_features])
    labels = (min_confidence > 0.5).astype(int)

    scaler = StandardScaler().fit(features)
    real_model = xgb.XGBClassifier(n_estimators=20, max_depth=3, objective="binary:logistic")
    real_model.fit(scaler.transform(features), labels)

    model_path = tmp_path / "meta_label_model.json"
    scaler_path = tmp_path / "meta_label_scaler.joblib"
    # Same real xgboost 2.1.x compatibility shim as scripts/train_model.py's own save_model():
    # XGBClassifier.save_model() asserts _estimator_type, which recent scikit-learn no longer
    # supplies as a class attribute on ClassifierMixin.
    if not hasattr(real_model, "_estimator_type"):
        real_model._estimator_type = "classifier"
    real_model.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)

    loaded = MetaLabelModel(model_path=model_path, scaler_path=scaler_path)
    high = MetaLabelModel.build_feature_vector(
        min_confidence=0.95, confidence_spread=0.0, spread=0.0, recent_conflict_count=0,
    )
    low = MetaLabelModel.build_feature_vector(
        min_confidence=0.05, confidence_spread=0.0, spread=0.0, recent_conflict_count=0,
    )
    assert loaded.predict_confidence(high) > 0.5
    assert loaded.predict_confidence(low) < 0.5

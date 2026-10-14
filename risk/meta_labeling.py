"""Meta-Labeling: a real secondary confidence layer over an already-reached consensus, never a primary signal.

**The real, confirmed gap this closes**: :class:`~src.agents.execution.
agent.ExecutionAgent` already has a real signal-ensemble mechanism --
entry requires **both** :class:`~src.agents.supervisor.agent.
XgboostSignalAgent` and :class:`~src.agents.rl_signal.agent.RlSignalAgent`
to agree, fresh, non-neutral, before consensus is even reached (see that
module's own docstring). This is real, live, tested ensemble logic, not
absent -- an earlier pass's due-diligence audit calling it "no ensemble"
was imprecise and is corrected here, not repeated. **What actually is
missing, verified directly in the code, not assumed**: :meth:`~src.
agents.execution.agent.ExecutionAgent._act_on_consensus` already computes
``confidence = min(signal.confidence for signal in signals.values())``
and threads it all the way to :meth:`~src.agents.execution.agent.
ExecutionAgent._place_order` -- where it is used for exactly one thing,
a log line (``confidence=round(confidence, 4)``), and nowhere else.
:meth:`~src.agents.execution.agent.ExecutionAgent.calculate_position_size`
(Gate 3) is pure fixed-fractional sizing -- account balance and
stop-loss distance only, no confidence term at all. Two consensus events
with confidences 0.99 and 0.51 are sized identically today. This module
is the real secondary model -- López de Prado's "meta-labeling" (Advances
in Financial Machine Learning, 2018, ch. 3): a model that does not decide
*direction* (the two primary signal agents already do that, via
consensus), only *how much to trust this particular consensus event*,
turning an unused ``[0, 1]`` confidence number into a real position-size
multiplier (including a real veto at the low end) instead of a discarded
log field.

**Real, disclosed feature scope**: :func:`MetaLabelModel.build_feature_vector`
uses exactly what is already tracked, unconditionally, at the point
:class:`~src.agents.execution.agent.ExecutionAgent`'s Gate 3 runs --
the two agreeing agents' own confidences (min and spread between them),
the instrument's latest real spread (``self._last_spreads``), and the
real recent consensus-conflict count (``self._conflict_history``, which
already existed for Trigger 4's own "signal flip-flop" alert and was
otherwise unused past that). VPIN and Gate 7's Hawkes intensity are
**not** included here, disclosed rather than silently assumed available:
both are computed later in the same gate pipeline (Gate 7,
``StochasticCore``) than Gate 3 runs, and pulling them forward would
either duplicate that computation or reorder the gate pipeline -- a
real, separate, larger change, not made unasked as part of this one.

**Update (2026-10-14)**: ``scripts/train_metalabel.py`` now exists (decision log + real round trips from the workers' fills, chronological
split, strict acceptance gate) and :meth:`MetaLabelModel.load_or_skeleton` loads its artifact, falling back to the neutral skeleton
when it is absent or unusable. It has never produced an artifact on real data: the history below is still the true state of the data.

**Real, disclosed training status, following :class:`~src.agents.
supervisor.ml_engine.SupervisorMLModel`'s own precedent exactly**: this
project has no real, sufficiently-long live trade-outcome history to fit
a real meta-model on yet (the same honest gap that project's own
Chaos Shadow Run / quant due-diligence passes already found: simulated
signals, no live audited P&L). :class:`MetaLabelModel` therefore ships
the same two-mode design as :class:`~src.agents.supervisor.ml_engine.
SupervisorMLModel`: load a real fitted XGBoost + scaler once real
training data and a real training pipeline exist, or fall back to a
structurally valid but untrained skeleton that always predicts exactly
``0.5`` (maximum uncertainty). **A real, load-bearing correctness
property, not an incidental default**: :func:`size_multiplier_from_confidence`
maps confidence ``0.5`` to a multiplier of exactly ``1.0`` -- wiring this
module in with no real trained artifact yet must leave today's real
sizing behavior completely unchanged, never silently alter it with an
untrained model's arbitrary output.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from core.logger import get_logger

_log = get_logger(__name__)

# See module docstring's "load-bearing correctness property": the skeleton model's own uninformative
# output (0.5) must round-trip through this function to exactly 1.0 (no change to Gate 3's sizing
# before this module existed). veto_below is a real, disclosed starting default, not backtested.
_DEFAULT_VETO_BELOW: Final[float] = 0.35
_DEFAULT_MAX_MULTIPLIER: Final[float] = 1.5
_NEUTRAL_CONFIDENCE: Final[float] = 0.5

# Feature set "v2", produced by ``scripts/train_metalabel.py`` and recorded in the artifact's sidecar. Order is the
# model's input order. ``imbalance_aligned`` is the signed order-book imbalance multiplied by the trade direction
# (+1 buy / -1 sell): positive when the book's pressure agrees with the trade. ``volatility`` is the per-tick
# standard deviation of recent log mid-price returns.
FEATURES_V2: Final[tuple[str, ...]] = (
    "ppo_confidence",
    "xgb_confidence",
    "confidence_gap",
    "spread",
    "volatility",
    "imbalance_aligned",
)
METADATA_SUFFIX: Final[str] = ".meta.json"
_DEFAULT_PAYOFF_RATIO: Final[float] = 1.0


def metadata_path_for(model_path: str | Path) -> Path:
    """``models/metalabel_weights.json`` -> ``models/metalabel_weights.meta.json`` (feature names, payoff ratio, metrics)."""
    return Path(model_path).with_suffix(METADATA_SUFFIX)


class MetaLabelModel:
    """XGBoost-based secondary confidence model over an already-reached consensus. See module docstring.

    Attributes:
        FEATURE_NAMES: The feature vector's fields, in the fixed order
            :meth:`build_feature_vector` produces and the model expects.
    """

    FEATURE_NAMES: Final[tuple[str, ...]] = (
        "min_confidence",
        "confidence_spread",
        "spread",
        "recent_conflict_count",
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        *,
        feature_names: tuple[str, ...] | None = None,
        payoff_ratio: float = _DEFAULT_PAYOFF_RATIO,
    ) -> None:
        """Load a pre-trained model+scaler, or build an untrained inference-ready skeleton.

        Same real two-mode contract as :class:`~src.agents.supervisor.
        ml_engine.SupervisorMLModel.__init__` -- see that class's own
        docstring for why (a training pipeline for this real model does
        not exist yet, disclosed here rather than fabricated).

        Args:
            model_path: Path to a model file previously saved via
                ``xgboost.XGBClassifier.save_model``. ``None`` (the
                default) builds a skeleton instead.
            scaler_path: Path to a fitted
                :class:`~sklearn.preprocessing.StandardScaler` saved via
                ``joblib.dump``. Only meaningful alongside
                ``model_path``.
        """
        self.feature_names: tuple[str, ...] = tuple(feature_names) if feature_names is not None else self.FEATURE_NAMES
        self.payoff_ratio = payoff_ratio
        self._is_fitted: bool
        if model_path is not None:
            self._model = xgb.XGBClassifier()
            if not hasattr(self._model, "_estimator_type"):
                self._model._estimator_type = "classifier"
            self._model.load_model(str(model_path))
            expected = len(self.feature_names)
            if self._model.get_booster().num_features() != expected:
                raise ValueError(
                    f"The model was fitted on {self._model.get_booster().num_features()} features, "
                    f"but its feature set {self.feature_names} has {expected}."
                )
            self._is_fitted = True
            _log.info("meta_label_model_loaded", model_path=str(model_path))
        else:
            self._model = self._build_skeleton_model()
            self._is_fitted = False
            _log.warning(
                "meta_label_model_using_untrained_skeleton",
                reason="no model_path provided; no real trade-outcome training pipeline exists yet",
            )

        self._scaler: Any
        if scaler_path is not None:
            self._scaler = joblib.load(scaler_path)
            _log.info("meta_label_scaler_loaded", scaler_path=str(scaler_path))
        elif feature_names is not None and model_path is not None:
            self._scaler = None  # trees are invariant to monotone scaling: a v2 artifact ships without one
        else:
            if model_path is not None:
                _log.warning(
                    "meta_label_model_missing_paired_scaler",
                    model_path=str(model_path),
                    detail="using a skeleton (synthetic-data-fitted) scaler alongside a real model",
                )
            self._scaler = self._build_skeleton_scaler()

    def _build_skeleton_model(self) -> xgb.XGBClassifier:
        """Same real "fit on 2 opposite-label all-zero rows" honest placeholder as ``SupervisorMLModel``'s own."""
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, objective="binary:logistic")
        dummy_features = np.zeros((2, len(self.feature_names)), dtype=np.float64)
        dummy_labels = np.array([0, 1])
        model.fit(dummy_features, dummy_labels)
        return model

    def _build_skeleton_scaler(self) -> StandardScaler:
        scaler = StandardScaler()
        dummy_features = np.zeros((2, len(self.feature_names)), dtype=np.float64)
        scaler.fit(dummy_features)
        return scaler

    @staticmethod
    def build_feature_vector(
        *, min_confidence: float, confidence_spread: float, spread: float, recent_conflict_count: int,
    ) -> npt.NDArray[np.float64]:
        """Real feature vector, in :attr:`FEATURE_NAMES` order -- see module docstring for what each one is and why.

        Args:
            min_confidence: The lower of the two agreeing agents' real
                published confidences -- the same value
                :meth:`~src.agents.execution.agent.
                ExecutionAgent._act_on_consensus` already computes.
            confidence_spread: ``max - min`` of the two agents' real
                confidences -- how much they agree beyond just direction
                (two agents at 0.51/0.99 both "agree", but very
                differently).
            spread: The instrument's latest real bid-ask spread
                (``ExecutionAgent._last_spreads``).
            recent_conflict_count: How many real "conflict" consensus
                outcomes this instrument has had recently
                (``ExecutionAgent._conflict_history``) -- an unstable
                instrument flip-flopping between agreement and conflict
                is real, relevant context a static confidence number
                alone does not capture.
        """
        return np.array([min_confidence, confidence_spread, spread, float(recent_conflict_count)], dtype=np.float64)

    def predict_confidence(self, features: npt.NDArray[np.float64]) -> float:
        """Real secondary confidence in ``[0, 1]`` that this consensus event is worth trusting fully.

        Returns exactly ``0.5`` (maximum uncertainty) if this model is
        an untrained skeleton -- see module docstring and
        :meth:`~src.agents.supervisor.ml_engine.SupervisorMLModel.
        predict_probability`'s identical honest-default framing.

        Raises:
            ValueError: If ``features`` doesn't have the expected shape.
        """
        expected_shape = (len(self.feature_names),)
        if features.shape != expected_shape:
            raise ValueError(f"Expected feature vector of shape {expected_shape}, got {features.shape}.")

        row = features.reshape(1, -1)
        scaled = row if self._scaler is None else self._scaler.transform(row)
        probabilities = self._model.predict_proba(scaled)[0]
        return float(probabilities[1])

    @property
    def is_fitted(self) -> bool:
        """``False`` for the untrained skeleton, which always answers exactly 0.5."""
        return self._is_fitted

    def vectorize(self, values: Mapping[str, float]) -> npt.NDArray[np.float64] | None:
        """This model's input vector from named values, in its own feature order; ``None`` if any is missing or not finite.

        The caller decides what ``None`` means (the execution agent falls back to the neutral 0.5): a missing
        input is never replaced by a made-up number here.
        """
        try:
            row = [float(values[name]) for name in self.feature_names]
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in row):
            return None
        return np.array(row, dtype=np.float64)

    @classmethod
    def from_artifact(cls, model_path: str | Path, scaler_path: str | Path | None = None) -> MetaLabelModel:
        """Strict load of a trained artifact and its ``.meta.json`` sidecar. Raises on any problem.

        With a sidecar, the feature set and payoff ratio come from it. Without one the artifact is read as the legacy
        v1 layout (4 features, scaler expected), exactly as before.
        """
        path = Path(model_path)
        sidecar = metadata_path_for(path)
        if not sidecar.is_file():
            return cls(path, scaler_path)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        names = tuple(str(n) for n in metadata["feature_names"])
        payoff = float(metadata.get("payoff_ratio", _DEFAULT_PAYOFF_RATIO))
        if not names or payoff <= 0 or not math.isfinite(payoff):
            raise ValueError("invalid metadata: empty feature_names or non-positive payoff_ratio")
        return cls(path, scaler_path, feature_names=names, payoff_ratio=payoff)

    @classmethod
    def load_or_skeleton(cls, model_path: str | Path | None, scaler_path: str | Path | None = None) -> MetaLabelModel:
        """Never raises: a missing, unreadable or inconsistent artifact gives the untrained skeleton (always 0.5).

        The skeleton maps to a multiplier of exactly 1.0, so a deployment without a trained model, or with a broken
        file, sizes exactly as before. A configured-but-unusable file is logged as an ERROR, not hidden.
        """
        if model_path is None:
            return cls()
        path = Path(model_path)
        if not path.is_file():
            _log.error("meta_label_artifact_missing", model_path=str(path), fallback="untrained skeleton (0.5, multiplier 1.0)")
            return cls()
        try:
            return cls.from_artifact(path, scaler_path)
        except Exception as exc:  # noqa: BLE001 -- a broken artifact must never stop the execution agent from starting
            _log.error("meta_label_artifact_unusable", model_path=str(path), error=type(exc).__name__, fallback="untrained skeleton (0.5, multiplier 1.0)")
            return cls()


def size_multiplier_from_confidence(
    confidence: float,
    *,
    veto_below: float = _DEFAULT_VETO_BELOW,
    max_multiplier: float = _DEFAULT_MAX_MULTIPLIER,
) -> float:
    r"""Real position-size multiplier from a real meta-label confidence. See module docstring's correctness property.

    ``0.0`` (a real, hard veto -- Gate 3 fails the entry entirely) below
    ``veto_below``; otherwise ``confidence / 0.5``, capped at
    ``max_multiplier``. This makes :data:`_NEUTRAL_CONFIDENCE` (``0.5``,
    the skeleton model's own uninformative output) map to exactly
    ``1.0`` -- unchanged sizing -- by construction, not by a special
    case: :math:`0.5 / 0.5 = 1.0` falls directly out of the same formula
    every other confidence value uses.

    Args:
        confidence: Real meta-label confidence in ``[0, 1]``.
        veto_below: Confidence strictly below this returns ``0.0``. Must
            be in ``[0, 0.5)`` -- a veto threshold at or above the
            neutral midpoint would veto the untrained skeleton's own
            default output, breaking the "no real trained model yet ->
            no behavior change" property above.
        max_multiplier: Hard cap on the returned multiplier. Must be
            ``>= 1.0``.

    Returns:
        The real size multiplier, ``0.0`` or in ``[veto_below / 0.5,
        max_multiplier]``.

    Raises:
        ValueError: If ``confidence`` is outside ``[0, 1]``,
            ``veto_below`` is outside ``[0, 0.5)``, or ``max_multiplier``
            is below ``1.0``.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if not 0.0 <= veto_below < _NEUTRAL_CONFIDENCE:
        raise ValueError(f"veto_below must be in [0, {_NEUTRAL_CONFIDENCE}).")
    if max_multiplier < 1.0:
        raise ValueError("max_multiplier must be >= 1.0.")

    if confidence < veto_below:
        return 0.0
    return min(confidence / _NEUTRAL_CONFIDENCE, max_multiplier)


def kelly_size_multiplier(
    probability: float,
    *,
    payoff_ratio: float = _DEFAULT_PAYOFF_RATIO,
    kelly_fraction: float = 0.5,
    veto_below: float = _DEFAULT_VETO_BELOW,
    max_multiplier: float = _DEFAULT_MAX_MULTIPLIER,
) -> float:
    r"""Position-size multiplier from a success probability, as a fractional-Kelly slope around the neutral point.

    For a bet that wins with probability :math:`p` and pays :math:`b` times what it risks, the Kelly fraction is
    :math:`f^* = p - (1-p)/b`. The engine's own sizing (fixed-fractional risk, Gate 3) already fixes a base risk amount
    whose Kelly-optimality nobody has measured, so ``f*`` is **not** used as an absolute allocation: at the neutral
    :math:`p = 0.5` and :math:`b = 1` it is exactly zero, which would veto the untrained skeleton and break the
    "no trained model yet -> sizing unchanged" property this module guarantees. What is used is the change in ``f*``
    relative to that neutral point, scaled by a safety fraction:

    .. math:: m = 1 + \lambda\,\bigl(f^*(p) - f^*(0.5)\bigr) = 1 + \lambda\,(1 + 1/b)\,(p - 0.5)

    - :math:`p = 0.5` gives exactly ``1.0`` (neutral, by construction); below it the size shrinks, above it grows;
    - the result is capped at ``max_multiplier`` and floored at ``0``; ``p`` strictly below ``veto_below`` is a hard
      veto (``0.0``), same as :func:`size_multiplier_from_confidence`;
    - ``kelly_fraction`` (default one half, "half Kelly") scales the slope; ``payoff_ratio`` is the average win over the
      average loss measured at training time (``1.0`` when unknown).

    The defaults are a disclosed starting point, not backtested values.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    if payoff_ratio <= 0.0 or not math.isfinite(payoff_ratio):
        raise ValueError("payoff_ratio must be a positive finite number.")
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must be in (0, 1].")
    if not 0.0 <= veto_below < _NEUTRAL_CONFIDENCE:
        raise ValueError(f"veto_below must be in [0, {_NEUTRAL_CONFIDENCE}).")
    if max_multiplier < 1.0:
        raise ValueError("max_multiplier must be >= 1.0.")
    if probability < veto_below:
        return 0.0
    slope = kelly_fraction * (1.0 + 1.0 / payoff_ratio)
    return max(0.0, min(1.0 + slope * (probability - _NEUTRAL_CONFIDENCE), max_multiplier))

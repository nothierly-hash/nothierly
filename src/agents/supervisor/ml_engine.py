"""ML decision model: fuses market, macro, and NLP signals into a probability.

:class:`SupervisorMLModel` combines the latest reading from each of the
three collection agents -- :class:`~core.schemas.MarketMetrics` (Agent
2), :class:`~core.schemas.MacroSnapshot` (Agent 3), and
:class:`~core.schemas.SentimentSignal` (Agent 4) -- into one fixed-size
feature vector, and runs it through an XGBoost classifier to produce a
probability of an upward price move.

:mod:`scripts.train_model` (chapter 6) is the training pipeline that
produces real artifacts: a fitted XGBoost model
(``xgboost.XGBClassifier.save_model``) and its paired fitted
:class:`~sklearn.preprocessing.StandardScaler` (``joblib.dump``).
:class:`SupervisorMLModel` supports two modes: load both real artifacts
from disk once they exist, or fall back to a structurally valid but
untrained "skeleton" model+scaler -- callable via the exact same
:meth:`~SupervisorMLModel.predict_probability` API, but always returning
~0.5 (maximum uncertainty) since it encodes no real trained knowledge.
This keeps the inference *interface* stable for whatever calls it today,
independent of when real training data and a real model become
available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from core.logger import get_logger
from core.schemas import MacroSnapshot, MarketMetrics, SentimentPolarity, SentimentSignal

_log = get_logger(__name__)


class SupervisorMLModel:
    """XGBoost-based fusion model producing an up/down move probability.

    Attributes:
        FEATURE_NAMES: The feature vector's fields, in the fixed order
            :meth:`build_feature_vector` produces and the model expects.
    """

    FEATURE_NAMES: Final[tuple[str, ...]] = (
        "spread",
        "order_book_imbalance",
        "macro_us_10y_yield",
        "macro_btc_exchange_flow",
        "sentiment_score_signed",
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
    ) -> None:
        """Load a pre-trained model+scaler, or build an untrained inference-ready skeleton.

        Args:
            model_path: Path to a model file previously saved via
                ``xgboost.XGBClassifier.save_model`` (see
                :mod:`scripts.train_model`). If ``None`` (the default), a
                skeleton model is built instead -- see the module
                docstring.
            scaler_path: Path to a fitted
                :class:`~sklearn.preprocessing.StandardScaler`
                previously saved via ``joblib.dump``. Only meaningful
                alongside ``model_path``; if ``model_path`` is given but
                ``scaler_path`` is not, a warning is logged and a
                skeleton (synthetic-data-fitted) scaler is used instead
                -- inference will run, but feature scaling won't reflect
                the real model's training distribution, which will
                degrade prediction quality.
        """
        self._is_fitted: bool
        if model_path is not None:
            self._model = xgb.XGBClassifier()
            # Same compatibility shim as scripts/train_model.py's
            # save_model(): XGBClassifier.load_model() (xgboost 2.1.x)
            # also asserts _estimator_type, which recent scikit-learn no
            # longer supplies as a class attribute on ClassifierMixin.
            # Must be set before load_model() is called, not after --
            # load_model() reads it during its own version-compatibility
            # check.
            if not hasattr(self._model, "_estimator_type"):
                self._model._estimator_type = "classifier"
            self._model.load_model(str(model_path))
            self._is_fitted = True
            _log.info("supervisor_ml_model_loaded", model_path=str(model_path))
        else:
            self._model = self._build_skeleton_model()
            self._is_fitted = False
            _log.warning(
                "supervisor_ml_model_using_untrained_skeleton",
                reason="no model_path provided; no training pipeline exists yet",
            )

        if scaler_path is not None:
            self._scaler = joblib.load(scaler_path)
            _log.info("supervisor_ml_scaler_loaded", scaler_path=str(scaler_path))
        else:
            if model_path is not None:
                _log.warning(
                    "supervisor_ml_model_missing_paired_scaler",
                    model_path=str(model_path),
                    detail="using a skeleton (synthetic-data-fitted) scaler alongside a real model",
                )
            self._scaler = self._build_skeleton_scaler()

    def _build_skeleton_model(self) -> xgb.XGBClassifier:
        """Build an XGBoost classifier structurally ready for inference.

        XGBoost requires at least one ``.fit()`` call before
        ``predict_proba`` is callable. Fitting on two rows of all-zero
        features with opposite labels gives the trees nothing real to
        split on, so the resulting model outputs ~0.5 for effectively any
        input -- an honest "no trained knowledge yet" placeholder, rather
        than a model that looks trained but is silently meaningless.

        Returns:
            A fitted-but-uninformative :class:`xgboost.XGBClassifier`.
        """
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, objective="binary:logistic")
        dummy_features = np.zeros((2, len(self.FEATURE_NAMES)), dtype=np.float64)
        dummy_labels = np.array([0, 1])
        model.fit(dummy_features, dummy_labels)
        return model

    def _build_skeleton_scaler(self) -> StandardScaler:
        """Build a :class:`StandardScaler` fitted on the same synthetic data.

        See the "Note" in :meth:`__init__` for why this is used
        regardless of whether a real model was loaded.
        """
        scaler = StandardScaler()
        dummy_features = np.zeros((2, len(self.FEATURE_NAMES)), dtype=np.float64)
        scaler.fit(dummy_features)
        return scaler

    @staticmethod
    def build_feature_vector(
        market_metrics: MarketMetrics,
        macro_snapshots: Mapping[str, MacroSnapshot],
        sentiment_signals: Sequence[SentimentSignal],
    ) -> npt.NDArray[np.float64]:
        """Combine the three agents' latest signals into one feature vector.

        A ``@staticmethod`` -- deliberately: it touches no instance state
        (unlike :meth:`predict_probability`, which needs the loaded
        model/scaler), which is exactly what lets
        :class:`~src.agents.rl_signal.agent.RlSignalAgent` call it
        without constructing (and paying the load cost of) a full
        :class:`SupervisorMLModel` just to build the same 5-feature
        vector both signal agents must agree on.

        Args:
            market_metrics: Latest order book metrics for the traded
                symbol, from :class:`~src.agents.market_data.agent.
                MarketDataAgent`.
            macro_snapshots: Latest :class:`~core.schemas.MacroSnapshot`
                per indicator name, from :class:`~src.agents.macro.agent.
                MacroAgent` (e.g. ``{"US_10Y_YIELD": ..., ...}``). An
                indicator missing from this mapping contributes ``0.0``
                (neutral) rather than raising, since any single macro
                provider being temporarily unavailable shouldn't prevent
                a decision.
            sentiment_signals: Recent :class:`~core.schemas.
                SentimentSignal` readings from
                :class:`~src.agents.nlp.agent.NlpAgent`, aggregated into
                one signed score via :meth:`_aggregate_sentiment`.

        Returns:
            A 1D array of shape ``(len(FEATURE_NAMES),)``, in the order
            :attr:`FEATURE_NAMES` documents.
        """
        us_10y_yield = macro_snapshots.get("US_10Y_YIELD")
        btc_exchange_flow = macro_snapshots.get("BTC_EXCHANGE_FLOW")

        return np.array(
            [
                market_metrics.spread,
                market_metrics.order_book_imbalance,
                us_10y_yield.value if us_10y_yield is not None else 0.0,
                btc_exchange_flow.value if btc_exchange_flow is not None else 0.0,
                SupervisorMLModel._aggregate_sentiment(sentiment_signals),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _aggregate_sentiment(signals: Sequence[SentimentSignal]) -> float:
        """Average a batch of sentiment signals into one signed score.

        ``POSITIVE`` contributes ``+score``, ``NEGATIVE`` contributes
        ``-score``, ``NEUTRAL`` contributes ``0``.

        Args:
            signals: Recent sentiment readings. May be empty.

        Returns:
            The mean signed score in ``[-1, 1]``, or ``0.0`` (neutral)
            if ``signals`` is empty.
        """
        if not signals:
            return 0.0

        signed_scores = [
            signal.score
            if signal.polarity == SentimentPolarity.POSITIVE
            else -signal.score
            if signal.polarity == SentimentPolarity.NEGATIVE
            else 0.0
            for signal in signals
        ]
        return float(np.mean(signed_scores))

    def predict_probability(self, features: npt.NDArray[np.float64]) -> float:
        """Predict the probability of an upward price move.

        Args:
            features: A feature vector from :meth:`build_feature_vector`,
                of shape ``(len(FEATURE_NAMES),)``.

        Returns:
            Probability in ``[0, 1]`` that the next price move is
            upward. If this model is an untrained skeleton (see
            :meth:`__init__`), this is always ~0.5 regardless of input,
            honestly reflecting "no real trained knowledge yet" rather
            than a confident-looking but meaningless number.

        Raises:
            ValueError: If ``features`` doesn't have the expected shape.
        """
        expected_shape = (len(self.FEATURE_NAMES),)
        if features.shape != expected_shape:
            raise ValueError(
                f"Expected feature vector of shape {expected_shape}, got {features.shape}."
            )

        scaled = self._scaler.transform(features.reshape(1, -1))
        probabilities = self._model.predict_proba(scaled)[0]
        return float(probabilities[1])

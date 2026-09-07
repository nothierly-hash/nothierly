"""Inference wrapper around the PPO :class:`~src.rl.networks.ActorCritic` policy.

Mirrors :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel`'s two
modes and its untrained-skeleton fallback: load a real
``torch.jit.script``-exported policy (see :func:`scripts.train_ppo.
export_model`) if one exists, or fall back to a freshly-initialized,
untrained :class:`~src.rl.networks.ActorCritic` -- callable via the exact
same :meth:`RLPolicyModel.predict_action` API either way, so this
project's own no-training-pipeline-run-yet state (true as of this
writing: no PPO policy has actually been exported) is an honest "no
opinion" rather than a crash or a confident-looking but meaningless
prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import torch

from core.logger import get_logger
from src.rl.networks import ActorCritic
from src.rl.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL

_log = get_logger(__name__)

# Must match SupervisorMLModel.FEATURE_NAMES's count (5) plus the
# trading-env observation's trailing "current position" element -- see
# src/rl/trading_env.py's MarketEnv docstring for the exact layout this
# policy was (or will be) trained against.
_OBSERVATION_DIM: Final[int] = 6
_ACTION_DIM: Final[int] = 3

# MarketEnv's action space (0=HOLD/1=BUY/2=SELL, long-only/flat -- SELL
# means "exit to flat", not "go short") mapped onto
# core.schemas.TradingSignal's signed vocabulary (0=neutral/1=buy/-1=sell).
_ACTION_TO_SIGNAL: Final[dict[int, Literal[-1, 0, 1]]] = {ACTION_HOLD: 0, ACTION_BUY: 1, ACTION_SELL: -1}


class RLPolicyModel:
    """Loads a trained (or untrained-skeleton) PPO policy and predicts one action.

    Attributes:
        is_fitted: Whether a real, trained policy was loaded (``True``)
            or this is the untrained skeleton (``False``).
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Load a ``torch.jit.script``-exported policy, or build an untrained skeleton.

        Args:
            model_path: Path to a policy previously saved via
                :func:`scripts.train_ppo.export_model`. If ``None`` (the
                default), an untrained skeleton is built instead -- see
                the module docstring.
        """
        if model_path is not None:
            # torch.jit.load is untyped in torch's stubs.
            self._model: torch.nn.Module = torch.jit.load(str(model_path))  # type: ignore[no-untyped-call]
            self.is_fitted = True
            _log.info("rl_policy_model_loaded", model_path=str(model_path))
        else:
            self._model = ActorCritic(observation_dim=_OBSERVATION_DIM, action_dim=_ACTION_DIM)
            self.is_fitted = False
            _log.warning(
                "rl_policy_model_using_untrained_skeleton",
                reason="no model_path provided; scripts.train_ppo has not been run yet",
            )
        self._model.eval()

    def predict_action(
        self, features: npt.NDArray[np.float64], *, current_position: float = 0.0
    ) -> tuple[Literal[-1, 0, 1], float]:
        """Predict a signed trading signal + confidence from this cycle's features.

        Args:
            features: Shape ``(5,)``, in
                :attr:`~src.agents.supervisor.ml_engine.
                SupervisorMLModel.FEATURE_NAMES` order (this policy
                shares that exact feature set, plus position below).
            current_position: ``0.0`` (flat) or ``1.0`` (long), the same
                convention :class:`~src.rl.trading_env.MarketEnv` was
                trained against. Live signal generation has no
                execution-tracked position of its own to report here
                (see the module docstring in
                :mod:`src.agents.rl_signal.agent`) -- defaults to
                ``0.0`` (evaluate as if flat), a known, documented
                simplification.

        Returns:
            ``(signal, confidence)``: ``signal`` is ``1`` (buy), ``-1``
            (sell), or ``0`` (neutral), and ``confidence`` is the
            policy's own probability for that action, in ``[0, 1]``.

        Raises:
            ValueError: If ``features`` doesn't have shape ``(5,)``.
        """
        if features.shape != (_OBSERVATION_DIM - 1,):
            raise ValueError(
                f"Expected feature vector of shape ({_OBSERVATION_DIM - 1},), got {features.shape}."
            )

        observation = np.concatenate([features, [current_position]]).astype(np.float32)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probabilities, _state_value = self._model(observation_tensor)

        probabilities = action_probabilities.squeeze(0)
        best_action = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[best_action].item())
        return _ACTION_TO_SIGNAL[best_action], confidence

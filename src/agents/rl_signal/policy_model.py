"""Inference wrapper around the PPO :class:`~src.rl.networks.ActorCritic` policy.

Mirrors :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel`'s two
modes and its untrained-skeleton fallback: load a real
``torch.jit.script``-exported policy (see :func:`scripts.train_ppo.
export_model`) if one exists, or fall back to a freshly-initialized,
untrained :class:`~src.rl.networks.ActorCritic` -- callable via the exact
same :meth:`RLPolicyModel.predict_action` API either way, so a missing
policy file is an honest "no opinion" rather than a crash or a
confident-looking but meaningless prediction.

**Real update (2026-09-13)**: :mod:`scripts.train_ppo` has in fact been
run -- 4 real ``torch.jit.script``-exported checkpoints exist under
``models/`` (trained 2026-09-02). ``scripts/run_backend.py`` now passes
the latest one to every real :class:`~src.agents.rl_signal.agent.
RlSignalAgent` it constructs (see that script's own
``_resolve_rl_policy_model_path``), closing a real, found gap: the
training pipeline had produced real artifacts that were sitting unused
on disk while production silently kept running the untrained skeleton.
No live-trading accuracy/edge claim is made for this checkpoint --
confirmed only that it loads correctly against the current
:class:`~src.rl.networks.ActorCritic` architecture and produces a valid,
deterministic action-probability distribution; a real edge claim would
need actual backtesting this project has not run against this specific
checkpoint (contrast :mod:`research.evaluate_apprentice`, which does run
that real evaluation, but against a separate, research-only policy).
Any caller (e.g. a test) that constructs this class directly with
``model_path=None`` still gets the honest untrained skeleton, unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import torch

from core.logger import get_logger
from src.rl.networks import ActorCritic, masked_probabilities
from src.rl.risk_shaping import compute_action_mask
from src.rl.trading_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, N_VPIN_FEATURES, OBSERVATION_DIM, OBSERVATION_DIM_WITH_VPIN

_log = get_logger(__name__)

# Must match SupervisorMLModel.FEATURE_NAMES's count (5) plus the
# trading-env observation's trailing "current position" element -- see
# src/rl/trading_env.py's MarketEnv docstring for the exact layout this
# policy was (or will be) trained against.
_OBSERVATION_DIM: Final[int] = OBSERVATION_DIM
_ACTION_DIM: Final[int] = 3
DEFAULT_VPIN_WINDOW_S: Final[float] = 60.0


class VpinUnavailableError(RuntimeError):
    """La politique a été entraînée avec le VPIN (observation de 9 valeurs) mais aucun VPIN complet (niveau, vitesse, accélération) n'est disponible : aucune opinion honnête n'est possible."""

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

    def __init__(self, model_path: str | Path | None = None, *, kill_switch_vpin: float | None = None) -> None:
        """Load a ``torch.jit.script``-exported policy, or build an untrained skeleton.

        Args:
            model_path: Path to a policy previously saved via
                :func:`scripts.train_ppo.export_model`. If ``None`` (the
                default), an untrained skeleton is built instead -- see
                the module docstring.
            kill_switch_vpin: Deterministic safety enclosure, applied to ANY policy (trained or not, with or without the VPIN in its
                observation): above this VPIN, ``BUY`` is masked and a held position can only exit. Defaults to the value stored in the
                policy's ``.meta.json`` (the one it was trained under); ``None`` = no mask.

        Raises:
            ValueError: If the policy's metadata declares an observation size this wrapper cannot feed (neither 6 nor 9).
        """
        meta = self._read_metadata(model_path)
        self._observation_dim = int(meta.get("observation_dim", _OBSERVATION_DIM))
        if self._observation_dim not in (OBSERVATION_DIM, OBSERVATION_DIM_WITH_VPIN):
            raise ValueError(f"observation_dim {self._observation_dim} non pris en charge (attendu {OBSERVATION_DIM} ou {OBSERVATION_DIM_WITH_VPIN})")
        self.uses_vpin = self._observation_dim == OBSERVATION_DIM_WITH_VPIN
        self.vpin_window_s = float(meta.get("vpin_window_s", DEFAULT_VPIN_WINDOW_S))
        meta_kill_switch = meta.get("kill_switch_vpin")
        self.kill_switch_vpin: float | None = kill_switch_vpin if kill_switch_vpin is not None else (float(meta_kill_switch) if meta_kill_switch is not None else None)
        if model_path is not None:
            # torch.jit.load is untyped in torch's stubs.
            self._model: torch.nn.Module = torch.jit.load(str(model_path))  # type: ignore[no-untyped-call]
            self.is_fitted = True
            _log.info("rl_policy_model_loaded", model_path=str(model_path))
        else:
            self._model = ActorCritic(observation_dim=self._observation_dim, action_dim=_ACTION_DIM)
            self.is_fitted = False
            _log.warning(
                "rl_policy_model_using_untrained_skeleton",
                reason="no model_path provided; scripts.train_ppo has not been run yet",
            )
        self._model.eval()

    @staticmethod
    def _read_metadata(model_path: str | Path | None) -> dict[str, Any]:
        """``<politique>.meta.json`` (écrit par ``scripts/train_ppo.py``), ou ``{}`` : une politique sans fichier de métadonnées est une politique d'avant (observation de 6 valeurs)."""
        if model_path is None:
            return {}
        sidecar = Path(model_path).with_suffix(".meta.json")
        if not sidecar.is_file():
            return {}
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{sidecar} doit contenir un objet JSON")
        return loaded

    def predict_action(
        self,
        features: npt.NDArray[np.float64],
        *,
        current_position: float = 0.0,
        vpin_features: npt.NDArray[np.float64] | None = None,
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
            vpin_features: ``(3,)`` : VPIN, sa vitesse et son accélération à cet instant (voir :func:`src.rl.risk_shaping.vpin_derivatives`).
                Obligatoire pour une politique entraînée avec le VPIN ; pour les autres, sert seulement au masque du coupe-circuit.

        Returns:
            ``(signal, confidence)``: ``signal`` is ``1`` (buy), ``-1``
            (sell), or ``0`` (neutral), and ``confidence`` is the
            policy's own probability for that action, in ``[0, 1]``.

        Raises:
            ValueError: If ``features`` doesn't have shape ``(5,)`` or ``vpin_features`` doesn't have shape ``(3,)``.
            VpinUnavailableError: If the policy needs the VPIN and ``vpin_features`` is missing or not finite.
        """
        if features.shape != (_OBSERVATION_DIM - 1,):
            raise ValueError(
                f"Expected feature vector of shape ({_OBSERVATION_DIM - 1},), got {features.shape}."
            )
        if vpin_features is not None and vpin_features.shape != (N_VPIN_FEATURES,):
            raise ValueError(f"Expected vpin_features of shape ({N_VPIN_FEATURES},), got {vpin_features.shape}.")
        vpin_known = vpin_features is not None and bool(np.isfinite(vpin_features).all())
        if self.uses_vpin and not vpin_known:
            raise VpinUnavailableError("politique entraînée avec le VPIN : niveau, vitesse et accélération sont requis")

        parts: list[Any] = [features, [current_position]]
        if self.uses_vpin and vpin_features is not None:
            parts.append(vpin_features)
        observation = np.concatenate(parts).astype(np.float32)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_probabilities, _state_value = self._model(observation_tensor)

        # Masque déterministe appliqué APRÈS l'inférence : le réseau sort des probabilités (Softmax), pas des logits ; on met à zéro les actions interdites et on renormalise.
        vpin_level = float(vpin_features[0]) if vpin_known and vpin_features is not None else None
        mask = compute_action_mask(current_position, vpin_level, self.kill_switch_vpin)
        probabilities = masked_probabilities(action_probabilities, torch.as_tensor(mask).unsqueeze(0)).squeeze(0)
        best_action = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[best_action].item())
        return _ACTION_TO_SIGNAL[best_action], confidence

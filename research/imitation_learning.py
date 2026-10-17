"""Chapter 12, Étape 3.6: The Apprentice -- Behavioral Cloning + PPO fine-tuning.

research/ only -- never imported by, and never imports, any live agent or
production module. Diagnosis this responds to (Étape 3.5, real, measured
results): pure trial-and-error PPO reached only 15.1% win-rate-net-of-fees
after 150k steps, while the already-validated XGBoost model reaches 58.8%
directional accuracy (research/checkpoint_xgboost.py) via ordinary
supervised learning on the same real data. Rather than making PPO
rediscover from scratch what XGBoost already knows, this script:

**Phase A -- Behavioral Cloning**: XGBoost's own predictions become
"expert" HOLD/BUY/SELL labels (thresholded the same way
:class:`~src.agents.supervisor.agent.XgboostSignalAgent` converts a
probability into a signal in production -- see
:data:`_DECISION_THRESHOLD`). The PPO policy network
(``stable_baselines3``'s ``model.policy``, the exact object Phase B will
later fine-tune) is trained with a plain supervised cross-entropy loss
against those labels -- computed as ``-policy.evaluate_actions(obs,
expert_action).log_prob.mean()``, mathematically identical to
cross-entropy for a categorical policy, using SB3's own public
``evaluate_actions`` API rather than reaching into private internals.
Observations use a fixed ``current_position=0.0`` (flat) throughout:
Phase A is deliberately scoped to teach only "recognize XGBoost's
entry/exit call from market features", not position management -- that
is explicitly Phase B's job (see module docstring below), matching what
XGBoost itself was never trained to do either.

**Phase B -- PPO fine-tuning**: the now-BC-pretrained policy continues
training via ordinary PPO (`model.learn()`) inside
:class:`~research.rl_sandbox_env.RlSandboxEnv` (Étape 3.5's hurdle-rate /
OBI-gated passivity / holding-time reward shaping, unchanged), at a much
lower learning rate and entropy coefficient than Étape 3.5 used, so
fine-tuning refines around the cloned behavior rather than overwriting
it. This is the phase meant to learn what XGBoost's single-shot
probability threshold cannot express at all: filtering an entry whose
margin cannot plausibly clear the round-trip fee, and holding a winning
position longer than one tick.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.checkpoint_xgboost import BOOK_PATH, build_feature_matrix
from research.rl_sandbox_env import ACTION_BUY, ACTION_HOLD, ACTION_SELL, RlSandboxEnv
from research.train_rl_sandbox import TB_LOG_DIR, build_sandbox_dataset

# Matches XgboostSignalAgent's own production decision_threshold default
# -- P(up) must clear 0.5 +/- this before the "expert" calls a direction
# instead of HOLD, exactly mirroring how the live agent would decide.
_DECISION_THRESHOLD = 0.1
_BC_MATCH_ACCURACY_TARGET = 0.95
_BC_EPOCHS = 30
_BC_BATCH_SIZE = 256
_BC_LEARNING_RATE = 1e-3

_FINE_TUNE_LEARNING_RATE = 1e-5
_FINE_TUNE_ENT_COEF = 0.001
_FINE_TUNE_TOTAL_TIMESTEPS = 100_000


def train_xgboost_expert() -> tuple[xgb.XGBClassifier, StandardScaler, np.ndarray, np.ndarray]:
    """Retrain XGBoost fresh on the same real Tardis data (research/checkpoint_xgboost.py's own pipeline).

    Retrained here rather than loading a specific saved timestamped
    model file, so this script stays reproducible and self-contained --
    not dependent on whichever model file happened to be saved last.

    Returns the **raw** fitted classifier and its paired scaler, not
    :class:`~src.agents.supervisor.ml_engine.SupervisorMLModel`: that
    wrapper hard-validates its input against the *production* 5-feature
    shape (``spread``/``order_book_imbalance``/``macro_us_10y_yield``/
    ``macro_btc_exchange_flow``/``sentiment_score_signed``) -- a
    completely different feature space from this research model's real
    6-column microstructure set (``obi_depth1``/``obi_depth5``/
    ``obi_depth10``/``spread``/``micro_price_momentum_5``/
    ``micro_price_momentum_20``, see :mod:`research.checkpoint_xgboost`).
    Wrapping it in ``SupervisorMLModel`` raises ``ValueError`` on shape
    mismatch immediately -- caught by this module's own smoke test, not
    a hypothetical.

    Returns:
        ``(raw_model, scaler, X, y)`` -- the fitted classifier, its
        fitted feature scaler, and the full real feature matrix / label
        vector it was built from.
    """
    import polars as pl
    from scripts.train_model import save_model, train_xgboost_classifier

    book = pl.read_csv(BOOK_PATH)
    featured = build_feature_matrix(book)
    feature_columns = [
        "obi_depth1", "obi_depth5", "obi_depth10", "spread",
        "micro_price_momentum_5", "micro_price_momentum_20",
    ]
    clean = featured.drop_nulls(subset=[*feature_columns, "label"])
    X = clean.select(feature_columns).to_numpy()
    y = clean["label"].to_numpy().astype(np.int64)

    raw_model, scaler, metrics = train_xgboost_classifier(X, y)
    print(f"Expert XGBoost retrained: accuracy={metrics['accuracy']:.4f}, ROC AUC={metrics['roc_auc']:.4f}")
    save_model(raw_model, scaler)  # persisted for reference; this run uses raw_model/scaler directly
    return raw_model, scaler, X, y


def expert_actions_from_xgboost(model: xgb.XGBClassifier, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Map each row's XGBoost P(up) to an expert HOLD/BUY/SELL action, same threshold rule as production."""
    probabilities_up = model.predict_proba(scaler.transform(X))[:, 1]
    actions = np.full(len(X), ACTION_HOLD, dtype=np.int64)
    actions[probabilities_up > 0.5 + _DECISION_THRESHOLD] = ACTION_BUY
    actions[probabilities_up < 0.5 - _DECISION_THRESHOLD] = ACTION_SELL
    return actions


def behavioral_cloning(
    policy: ActorCriticPolicy,
    observations: np.ndarray,
    expert_actions: np.ndarray,
    *,
    epochs: int = _BC_EPOCHS,
    batch_size: int = _BC_BATCH_SIZE,
    learning_rate: float = _BC_LEARNING_RATE,
) -> float:
    """Supervised cross-entropy training of an SB3 policy network against expert actions.

    Args:
        policy: ``model.policy`` from an already-constructed SB3
            ``PPO`` model (so its architecture matches
            :class:`~research.rl_sandbox_env.RlSandboxEnv`'s observation/
            action spaces exactly) -- mutated in place.
        observations: Shape ``(N, obs_dim)`` -- real market features with
            ``current_position`` fixed at ``0.0`` (see module docstring).
        expert_actions: Shape ``(N,)``, ``int64`` -- XGBoost-derived
            HOLD/BUY/SELL labels, same order as ``observations``.
        epochs: Full passes over the training split.
        batch_size: Minibatch size.
        learning_rate: A separate, dedicated optimizer's learning rate
            for this supervised phase -- deliberately not
            ``policy.optimizer`` (PPO's own, configured for Phase B's
            much smaller fine-tuning learning rate), so Phase A's
            momentum state doesn't leak into Phase B.

    Returns:
        Final held-out match-accuracy (fraction of validation rows where
        ``argmax(policy(obs)) == expert_action``).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        observations, expert_actions, test_size=0.2, random_state=42, stratify=expert_actions
    )
    obs_train = torch.as_tensor(X_train, dtype=torch.float32)
    actions_train = torch.as_tensor(y_train, dtype=torch.int64)
    obs_val = torch.as_tensor(X_val, dtype=torch.float32)
    actions_val = torch.as_tensor(y_val, dtype=torch.int64)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    n = len(obs_train)

    for epoch in range(epochs):
        permutation = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            batch_obs = obs_train[idx]
            batch_actions = actions_train[idx]

            # -log_prob(expert_action) under the current policy is
            # exactly cross-entropy for a categorical distribution --
            # SB3's own public evaluate_actions() API, not a private
            # internal.
            _values, log_prob, _entropy = policy.evaluate_actions(batch_obs, batch_actions)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            # Tensor.backward() is untyped in torch's stubs (same known
            # gap scripts/train_ppo.py's ppo_update() already works
            # around identically).
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)

        with torch.no_grad():
            val_actions, _values, _log_prob = policy(obs_val, deterministic=True)
            val_accuracy = float((val_actions == actions_val).float().mean().item())

        print(f"  BC epoch {epoch + 1}/{epochs}: train_loss={total_loss / n:.4f}  val_match_accuracy={val_accuracy:.1%}")

    return val_accuracy


def main() -> None:
    print("=" * 70)
    print("PHASE A: BEHAVIORAL CLONING")
    print("=" * 70)

    print("\nRetraining the XGBoost expert on real data...")
    xgb_model, scaler, X, _y = train_xgboost_expert()

    print("\nGenerating expert HOLD/BUY/SELL actions from XGBoost predictions...")
    expert_actions = expert_actions_from_xgboost(xgb_model, scaler, X)
    hold_n, buy_n, sell_n = (int((expert_actions == a).sum()) for a in (ACTION_HOLD, ACTION_BUY, ACTION_SELL))
    print(f"Expert action distribution: HOLD={hold_n} BUY={buy_n} SELL={sell_n}")

    # Observations with current_position fixed at 0.0 (flat) -- see
    # module docstring on why Phase A is deliberately scoped this way.
    flat_position_column = np.zeros((len(X), 1), dtype=np.float64)
    observations = np.concatenate([X, flat_position_column], axis=1).astype(np.float32)

    print(f"\nBuilding the sandbox environment (for a matching policy architecture)...")
    features, prices, obi_values = build_sandbox_dataset(BOOK_PATH)
    env: Monitor[np.ndarray, int] = Monitor(RlSandboxEnv(features, prices, obi_values, episode_length=2000))
    model = PPO(
        "MlpPolicy", env, learning_rate=_FINE_TUNE_LEARNING_RATE, ent_coef=_FINE_TUNE_ENT_COEF,
        n_steps=2048, batch_size=64, verbose=0, tensorboard_log=str(TB_LOG_DIR),
    )

    print(f"\nCloning XGBoost's behavior into the PPO policy network ({_BC_EPOCHS} epochs)...")
    match_accuracy = behavioral_cloning(model.policy, observations, expert_actions)

    print(f"\n{'='*70}")
    if match_accuracy >= _BC_MATCH_ACCURACY_TARGET:
        print(f"PHASE A TARGET MET: {match_accuracy:.1%} >= {_BC_MATCH_ACCURACY_TARGET:.0%}")
    else:
        print(f"PHASE A TARGET NOT MET: {match_accuracy:.1%} < {_BC_MATCH_ACCURACY_TARGET:.0%}")
        print("Proceeding to Phase B regardless (per the brief), but starting from an imperfect clone.")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PHASE B: PPO FINE-TUNING (low LR, low entropy, on top of the clone)")
    print("=" * 70)
    print(f"learning_rate={_FINE_TUNE_LEARNING_RATE}  ent_coef={_FINE_TUNE_ENT_COEF}  timesteps={_FINE_TUNE_TOTAL_TIMESTEPS}")

    model.learn(total_timesteps=_FINE_TUNE_TOTAL_TIMESTEPS, progress_bar=False)

    model.save(str(Path(__file__).resolve().parent / "apprentice_policy.zip"))
    print("\nSaved fine-tuned policy: research/apprentice_policy.zip")


if __name__ == "__main__":
    main()

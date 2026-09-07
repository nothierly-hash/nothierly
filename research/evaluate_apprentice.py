"""Evaluate the saved Phase B (Étape 3.6) apprentice policy: real win-rate-net-of-fees,
trade count, and PnL, run deterministically (no further training) over
real episodes -- the metrics research/imitation_learning.py's own
training run never captured (a gap found and fixed here, not part of
that script's original scope).
"""

from __future__ import annotations

import sys
from pathlib import Path

from typing import Any

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.rl_sandbox_env import RlSandboxEnv
from research.train_rl_sandbox import BOOK_PATH, build_sandbox_dataset

N_EVAL_EPISODES = 75  # same count as Étape 3.5's evaluation, for a direct comparison


def main() -> None:
    print("Loading real order book and the saved apprentice policy...")
    features, prices, obi_values = build_sandbox_dataset(BOOK_PATH)
    model = PPO.load(str(Path(__file__).resolve().parent / "apprentice_policy.zip"))

    env = RlSandboxEnv(features, prices, obi_values, episode_length=2000)

    episode_pnls: list[float] = []
    episode_trade_counts: list[int] = []
    episode_win_rates: list[float] = []  # None (no closed trade) excluded entirely, not conflated with a genuine 0.0

    for episode in range(N_EVAL_EPISODES):
        obs, _info = env.reset(seed=episode)
        done = False
        info: dict[str, Any] = {}
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        episode_pnls.append(float(info["episode_pnl"]))
        episode_trade_counts.append(int(info["trade_count"]))
        win_rate = info.get("win_rate_net_of_fees")
        # None (no closed trade this episode) is genuinely excluded here
        # -- NOT coerced to 0.0, which would be indistinguishable from a
        # real "traded and lost every time" 0.0 win rate. An earlier
        # version of this script filtered on `w > 0`, which wrongly
        # excluded real 0.0 win-rate episodes right alongside the
        # no-trade ones, making the reported rate look far better than
        # what was actually happening -- caught by manually inspecting
        # 10 individual episodes and finding `win_rate_net_of_fees=0.0`
        # (not None) on nearly all of them.
        if win_rate is not None:
            episode_win_rates.append(float(win_rate))

    print(f"\n{'='*70}")
    print(f"APPRENTICE POLICY EVALUATION ({N_EVAL_EPISODES} real episodes, deterministic)")
    print(f"{'='*70}")
    print(f"Mean trades/episode: {np.mean(episode_trade_counts):.1f}")
    print(f"Mean PnL/episode: {np.mean(episode_pnls):+.6f}")
    if episode_win_rates:
        print(f"Mean win-rate-net-of-fees (episodes with >=1 closed trade): {np.mean(episode_win_rates):.1%}")
        print(f"Episodes with >=1 closed trade: {len(episode_win_rates)}/{N_EVAL_EPISODES}")
        print(f"Of those, episodes with a >0% win rate: {sum(1 for w in episode_win_rates if w > 0)}/{len(episode_win_rates)}")
    else:
        print("No closed trades across any evaluation episode.")

    print(f"\nÉtape 3.5 baseline (pure RL, no imitation learning): trades/ep=54.2, PnL/ep=-0.021301, win_rate=15.1%")


if __name__ == "__main__":
    main()

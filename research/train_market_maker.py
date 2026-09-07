"""Chapter 12, Étape 4.0: train PPO in the market-making (quote-skewing) environment.

research/ only -- never imported by, and never imports, any live agent
or production module. Built on the same real BTC/ETH data and real
13-feature set already validated in checkpoint_xgboost_orthogonal.py
(Étape 3.8), plus the real fill-probability/toxic-fill calibration from
maker_execution_sim.py's queue simulation (Étape 3.9). See
market_making_env.py's module docstring for the environment itself.

Tests directly whether the agent learns the lesson the brief asked for:
does it learn to prefer NEUTRAL_MAKE during calm regimes and pull back
(HOLD) or skew defensively ahead of a real, feature-detected violent
move, rather than sitting exposed on both sides indiscriminately --
i.e., does *selective* quoting reduce the real, measured adverse-
selection cost found in Étape 3.9?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.alpha_features import (
    attach_cross_momentum,
    attach_trade_flow_features,
    attach_vpin_feature,
    compute_micro_price,
    compute_obi,
    compute_rolling_volatility,
    compute_vpin,
    load_tardis_book_snapshot,
    load_tardis_trades,
    rolling_zscore,
)
from research.market_making_env import N_ACTIONS, MarketMakingEnv

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
BTC_BOOK_PATH = SAMPLE_DIR / "btcusdt_book_snapshot_2020-01-01.csv"
BTC_TRADES_PATH = SAMPLE_DIR / "btcusdt_trades_2020-01-01.csv"
ETH_BOOK_PATH = SAMPLE_DIR / "ethusdt_book_snapshot_2020-01-01.csv"
TB_LOG_DIR = Path(__file__).resolve().parent / "tb_logs"
TOTAL_TIMESTEPS = 150_000  # matches Étape 3's own training scale, for a comparable result

FEATURE_COLUMNS = [
    "obi_depth1", "obi_depth5", "obi_depth10", "spread",
    "micro_price_momentum_5", "micro_price_momentum_20",
    "tfi_1s_zscore", "tfi_5s_zscore", "tfi_30s_zscore", "vpin_zscore",
    "cross_momentum_eth_1s_zscore", "cross_momentum_eth_5s_zscore", "cross_momentum_eth_30s_zscore",
]
_RAW_ZSCORE_COLUMNS = [
    "tfi_1s", "tfi_5s", "tfi_30s", "vpin",
    "cross_momentum_eth_1s", "cross_momentum_eth_5s", "cross_momentum_eth_30s",
    "obi_depth1", "volatility",
]


def build_market_making_dataset(
    book_path: Path, trades_path: Path, eth_book_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Real BTC/ETH data -> (features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore)."""
    book = load_tardis_book_snapshot(book_path)
    trades = load_tardis_trades(trades_path)
    eth_book = load_tardis_book_snapshot(eth_book_path)

    book = compute_obi(book, depth=1).rename({"obi": "obi_depth1"})
    book = compute_obi(book, depth=5).rename({"obi": "obi_depth5"})
    book = compute_obi(book, depth=10).rename({"obi": "obi_depth10"})
    book = compute_micro_price(book)
    book = compute_rolling_volatility(book)
    book = book.with_columns(
        (pl.col("asks[0].price") - pl.col("bids[0].price")).alias("spread"),
        (pl.col("micro_price") - pl.col("micro_price").shift(5)).alias("micro_price_momentum_5"),
        (pl.col("micro_price") - pl.col("micro_price").shift(20)).alias("micro_price_momentum_20"),
    )
    book = attach_trade_flow_features(book, trades)
    vpin_result = compute_vpin(trades)
    book = attach_vpin_feature(book, vpin_result)
    book = attach_cross_momentum(book, eth_book, asset_b_label="eth")

    for raw_column in _RAW_ZSCORE_COLUMNS:
        book = rolling_zscore(book, raw_column)

    required = [*FEATURE_COLUMNS, "obi_depth1_zscore", "vpin_zscore", "volatility_zscore", "tfi_1s_zscore", "micro_price", "spread"]
    clean = book.drop_nulls(subset=required)

    features = clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
    mid_prices = clean["micro_price"].to_numpy().astype(np.float64)
    spreads = clean["spread"].to_numpy().astype(np.float64)
    obi_zscore = clean["obi_depth1_zscore"].to_numpy().astype(np.float64)
    vpin_zscore = clean["vpin_zscore"].to_numpy().astype(np.float64)
    volatility_zscore = clean["volatility_zscore"].to_numpy().astype(np.float64)
    # Étape 4.3: real Trade Flow Imbalance (executed aggressive volume,
    # 1s window -- the most immediate real flow signal), now the
    # primary fill-probability driver -- see market_making_env.py.
    tfi_zscore = clean["tfi_1s_zscore"].to_numpy().astype(np.float64)
    return features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore


class MarketMakingCallback(BaseCallback):
    """Tracks per-episode fill/toxic-fill counts, inventory duration, and reward; streams to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_fill_counts: list[int] = []
        self.episode_toxic_fill_counts: list[int] = []
        self.episode_inventory_penalties: list[float] = []
        self.episode_mean_inventory_durations: list[float] = []
        self.episode_maker_exits: list[int] = []
        self.episode_taker_exits: list[int] = []
        self.episode_mean_hold_duration_maker_exit: list[float] = []
        self.episode_action_counts: list[dict[int, int]] = []
        self._current_episode_actions: dict[int, int] = {a: 0 for a in range(N_ACTIONS)}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        if actions is not None:
            for action in np.asarray(actions).flatten():
                self._current_episode_actions[int(action)] = self._current_episode_actions.get(int(action), 0) + 1

        for info in infos:
            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_rewards.append(float(episode_info.get("r", 0.0)))
                self.episode_fill_counts.append(int(info.get("fill_count", 0)))
                self.episode_toxic_fill_counts.append(int(info.get("toxic_fill_count", 0)))
                self.episode_inventory_penalties.append(float(info.get("episode_inventory_penalty", 0.0)))
                self.episode_maker_exits.append(int(info.get("maker_exit_count", 0)))
                self.episode_taker_exits.append(int(info.get("taker_exit_count", 0)))
                mean_duration = info.get("mean_inventory_duration")
                if mean_duration is not None:
                    self.episode_mean_inventory_durations.append(float(mean_duration))
                mean_maker_exit_duration = info.get("mean_hold_duration_maker_exit")
                if mean_maker_exit_duration is not None:
                    self.episode_mean_hold_duration_maker_exit.append(float(mean_maker_exit_duration))
                self.episode_action_counts.append(dict(self._current_episode_actions))
                self._current_episode_actions = {a: 0 for a in range(N_ACTIONS)}

                toxic_fills = info.get("toxic_fill_count", 0)
                fills = info.get("fill_count", 0)
                if fills > 0:
                    self.logger.record("rollout/toxic_fill_rate", toxic_fills / fills)
                if mean_duration is not None:
                    self.logger.record("rollout/Mean_Inventory_Duration", mean_duration)
                if mean_maker_exit_duration is not None:
                    self.logger.record("rollout/Mean_Hold_Duration_Maker_Exit", mean_maker_exit_duration)
                maker_exits = info.get("maker_exit_count", 0)
                taker_exits = info.get("taker_exit_count", 0)
                total_exits = maker_exits + taker_exits
                if total_exits > 0:
                    self.logger.record("rollout/Maker_Exit_Fraction", maker_exits / total_exits)
                    self.logger.record("rollout/Taker_Exit_Fraction", taker_exits / total_exits)

        return True


def compute_random_policy_baseline(
    features: np.ndarray, mid_prices: np.ndarray, spreads: np.ndarray,
    obi_zscore: np.ndarray, vpin_zscore: np.ndarray, volatility_zscore: np.ndarray, tfi_zscore: np.ndarray,
    *, n_episodes: int = 30,
) -> float:
    """Real toxic-fill rate under a uniform-random policy -- the "no selectivity" comparison point."""
    env = MarketMakingEnv(features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore, episode_length=2000)
    rng = np.random.default_rng(0)
    total_fills = 0
    total_toxic = 0
    for episode in range(n_episodes):
        env.reset(seed=episode)
        done = False
        while not done:
            action = int(rng.integers(0, N_ACTIONS))
            _obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        total_fills += info["fill_count"]
        total_toxic += info["toxic_fill_count"]
    return total_toxic / total_fills if total_fills else float("nan")


def main() -> None:
    for path in (BTC_BOOK_PATH, BTC_TRADES_PATH, ETH_BOOK_PATH):
        if not path.exists():
            print(f"Real sample data not found at {path}.", file=sys.stderr)
            sys.exit(1)

    print("Building real market-making dataset (BTC + ETH, 13 features + fill-model inputs)...")
    features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore = build_market_making_dataset(
        BTC_BOOK_PATH, BTC_TRADES_PATH, ETH_BOOK_PATH
    )
    print(f"Real dataset: {len(features)} timesteps, {features.shape[1]} features")

    print("\nComputing the real uniform-random-policy toxic-fill-rate baseline (30 episodes)...")
    random_toxic_rate = compute_random_policy_baseline(features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore)
    print(f"Random-policy toxic-fill rate: {random_toxic_rate:.2%}")

    env: Monitor[np.ndarray, int] = Monitor(
        MarketMakingEnv(features, mid_prices, spreads, obi_zscore, vpin_zscore, volatility_zscore, tfi_zscore, episode_length=2000)
    )
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.02, verbose=1,
        tensorboard_log=str(TB_LOG_DIR),
    )
    callback = MarketMakingCallback()

    print(f"\nTraining PPO for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model.save(str(Path(__file__).resolve().parent / "market_maker_policy.zip"))
    print("\nSaved: research/market_maker_policy.zip")

    print(f"\nCompleted episodes: {len(callback.episode_rewards)}")
    if callback.episode_rewards:
        print(f"Mean reward/episode: {np.mean(callback.episode_rewards):+.6f}")
        print(f"Mean fills/episode: {np.mean(callback.episode_fill_counts):.1f}")
        total_fills = sum(callback.episode_fill_counts)
        total_toxic = sum(callback.episode_toxic_fill_counts)
        toxic_rate = total_toxic / total_fills if total_fills else float("nan")
        print(f"Overall toxic-fill rate (learned policy): {toxic_rate:.2%} ({total_toxic}/{total_fills} fills)")
        print(f"Mean inventory penalty/episode: {np.mean(callback.episode_inventory_penalties):.6f}")
        if callback.episode_mean_inventory_durations:
            print(f"Mean_Inventory_Duration (ticks, averaged across episodes with a completed unwind): {np.mean(callback.episode_mean_inventory_durations):.1f}")
        else:
            print("Mean_Inventory_Duration: no completed unwinds recorded.")

        total_maker_exits = sum(callback.episode_maker_exits)
        total_taker_exits = sum(callback.episode_taker_exits)
        total_exits = total_maker_exits + total_taker_exits
        if callback.episode_mean_hold_duration_maker_exit:
            mean_hold_maker = np.mean(callback.episode_mean_hold_duration_maker_exit)
            print(
                f"Mean_Hold_Duration_Maker_Exit: {mean_hold_maker:.1f} ticks "
                f"({'inside' if mean_hold_maker <= 10 else 'OUTSIDE'} the 10-tick grace period)"
            )
        else:
            print("Mean_Hold_Duration_Maker_Exit: no maker exits recorded.")

        if total_exits > 0:
            print(
                f"Maker_Exits vs Taker_Exits: {total_maker_exits} maker ({total_maker_exits/total_exits:.1%}) "
                f"vs {total_taker_exits} taker ({total_taker_exits/total_exits:.1%})"
            )
        else:
            print("No completed exits recorded.")

        # Action mix, first 10% of episodes (early/random policy) vs last 10% (learned policy).
        n_eps = len(callback.episode_action_counts)
        early = callback.episode_action_counts[: max(n_eps // 10, 1)]
        late = callback.episode_action_counts[-max(n_eps // 10, 1) :]
        action_names = ["HOLD", "SKEW_BULLISH", "SKEW_BEARISH", "NEUTRAL_MAKE", "TAKER_SWEEP"]
        print("\nAction mix, early vs. late training (fraction of steps):")
        for a, name in enumerate(action_names):
            early_frac = sum(ep.get(a, 0) for ep in early) / max(sum(sum(ep.values()) for ep in early), 1)
            late_frac = sum(ep.get(a, 0) for ep in late) / max(sum(sum(ep.values()) for ep in late), 1)
            print(f"  {name}: {early_frac:.1%} -> {late_frac:.1%}")

        early_exits_maker = sum(callback.episode_maker_exits[: max(n_eps // 10, 1)])
        early_exits_taker = sum(callback.episode_taker_exits[: max(n_eps // 10, 1)])
        late_exits_maker = sum(callback.episode_maker_exits[-max(n_eps // 10, 1) :])
        late_exits_taker = sum(callback.episode_taker_exits[-max(n_eps // 10, 1) :])
        early_total = early_exits_maker + early_exits_taker
        late_total = late_exits_maker + late_exits_taker
        print("\nMaker-exit preference, early vs. late training:")
        print(f"  early: {early_exits_maker/early_total:.1%} maker" if early_total else "  early: no exits")
        print(f"  late:  {late_exits_maker/late_total:.1%} maker" if late_total else "  late: no exits")

        print(f"\nRandom-policy toxic-fill-rate baseline (no selectivity): {random_toxic_rate:.2%}")
        print(f"Learned policy toxic-fill rate: {toxic_rate:.2%}")
        if toxic_rate < random_toxic_rate:
            print("The learned policy reduced the toxic-fill rate relative to random/indiscriminate quoting.")
        else:
            print("The learned policy did NOT reduce the toxic-fill rate relative to random/indiscriminate quoting.")


if __name__ == "__main__":
    main()

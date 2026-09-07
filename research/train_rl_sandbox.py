"""Chapter 12, Étape 3 + 3.5: train PPO (Stable-Baselines3) in the RL sandbox.

research/ only -- never imported by, and never imports, any live agent
or production module. Trains on the same real Tardis order book data and
real OBI/micro-price features already validated in
research/checkpoint_xgboost.py (58.8% directional accuracy, crossing the
55% checkpoint -- see that script).

Hard callback, as specified: if the agent has taken zero real trades
(BUY or SELL, not HOLD) after 50,000 steps, training stops with an
AssertionError -- a loud, honest failure instead of silently completing
and looking finished while having learned nothing.

Logs ``WinRate_Net_Of_Fees`` (fraction of closed trades whose return,
net of the full round-trip 2x taker fee, was positive) to TensorBoard
every step it changes, as specified -- run ``tensorboard --logdir
research/tb_logs`` to view it, or see the local PNG this script also
saves (no TensorBoard viewer needed to see the same numbers).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.alpha_features import compute_micro_price, compute_obi, load_tardis_book_snapshot
from research.rl_sandbox_env import ACTION_HOLD, RlSandboxEnv

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
BOOK_PATH = SAMPLE_DIR / "btcusdt_book_snapshot_2020-01-01.csv"
TB_LOG_DIR = Path(__file__).resolve().parent / "tb_logs"
NO_TRADE_ASSERTION_STEPS = 50_000
TOTAL_TIMESTEPS = 150_000
FEATURE_COLUMNS = ["obi_depth1", "obi_depth5", "obi_depth10", "spread", "micro_price_momentum_5", "micro_price_momentum_20"]


def build_sandbox_dataset(book_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real order book -> (features, prices, obi_values) for RlSandboxEnv.

    ``obi_values`` is depth-1 OBI specifically -- the same column already
    included in ``features``, exposed separately since
    :class:`~research.rl_sandbox_env.RlSandboxEnv` needs it as its own
    argument to gate the Règle 2 passivity penalty (see that class's
    docstring for why it isn't just read out of the feature matrix by
    column index instead).
    """
    book = load_tardis_book_snapshot(book_path)
    book = compute_obi(book, depth=1).rename({"obi": "obi_depth1"})
    book = compute_obi(book, depth=5).rename({"obi": "obi_depth5"})
    book = compute_obi(book, depth=10).rename({"obi": "obi_depth10"})
    book = compute_micro_price(book)
    book = book.with_columns(
        (pl.col("asks[0].price") - pl.col("bids[0].price")).alias("spread"),
        (pl.col("micro_price") - pl.col("micro_price").shift(5)).alias("micro_price_momentum_5"),
        (pl.col("micro_price") - pl.col("micro_price").shift(20)).alias("micro_price_momentum_20"),
    )
    clean = book.drop_nulls(subset=FEATURE_COLUMNS)
    features = clean.select(FEATURE_COLUMNS).to_numpy().astype(np.float64)
    prices = clean["micro_price"].to_numpy().astype(np.float64)
    obi_values = clean["obi_depth1"].to_numpy().astype(np.float64)
    return features, prices, obi_values


class TradeAssertionCallback(BaseCallback):
    """Fails loudly (AssertionError) if the agent hasn't traded by a fixed step count.

    Also records, at every real episode end (via SB3's Monitor wrapper,
    which injects an ``"episode"`` key into ``info`` on episode
    completion), that episode's PnL, trade count, and win-rate-net-of-
    fees for :func:`plot_training_curves` -- and streams the latter to
    TensorBoard every step it's available, per the brief's specific
    logging request.
    """

    def __init__(self, no_trade_assertion_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._no_trade_assertion_steps = no_trade_assertion_steps
        self._any_trade_seen = False
        self.episode_pnls: list[float] = []
        self.episode_trade_counts: list[int] = []
        self.episode_win_rates: list[float] = []

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        if actions is not None:
            for action in np.asarray(actions).flatten():
                if int(action) != ACTION_HOLD:
                    self._any_trade_seen = True

        for info in infos:
            win_rate = info.get("win_rate_net_of_fees")
            if win_rate is not None:
                self.logger.record("rollout/win_rate_net_of_fees", win_rate)

            episode_info = info.get("episode")
            if episode_info is not None:
                self.episode_pnls.append(float(info.get("episode_pnl", 0.0)))
                self.episode_trade_counts.append(int(info.get("trade_count", 0)))
                self.episode_win_rates.append(float(win_rate) if win_rate is not None else 0.0)

        if self.num_timesteps >= self._no_trade_assertion_steps:
            assert self._any_trade_seen, (
                f"BRAINDEAD POLICY DETECTED: {self.num_timesteps} steps elapsed with zero BUY/SELL actions "
                "taken (only HOLD). This callback exists specifically so that failure is loud, not silent."
            )

        return True


def plot_training_curves(callback: TradeAssertionCallback, output_path: Path) -> None:
    """Save a real PnL/trade-count/win-rate-per-episode figure -- not a placeholder."""
    import matplotlib

    matplotlib.use("Agg")  # headless -- this script never runs with a display attached
    import matplotlib.pyplot as plt

    if not callback.episode_pnls:
        print("No completed episodes to plot (training ended before any episode finished).")
        return

    fig, (ax_pnl, ax_trades, ax_winrate) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax_pnl.plot(callback.episode_pnls, color="#2962FF")
    ax_pnl.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax_pnl.set_ylabel("Episode PnL (net log-return)")
    ax_pnl.set_title("Real PnL per episode (Sandbox PPO, real Tardis data, Étape 3.5 reward shaping)")

    ax_trades.plot(callback.episode_trade_counts, color="#00E676")
    ax_trades.set_ylabel("Trades per episode")
    ax_trades.set_title("Real trade count per episode")

    ax_winrate.plot(callback.episode_win_rates, color="#FF6D00")
    ax_winrate.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax_winrate.set_ylabel("Win rate net of fees")
    ax_winrate.set_xlabel("Episode")
    ax_winrate.set_title("WinRate_Net_Of_Fees per episode")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print(f"Saved: {output_path}")


def main() -> None:
    if not BOOK_PATH.exists():
        print(f"Real sample data not found at {BOOK_PATH}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading real order book: {BOOK_PATH.name}")
    features, prices, obi_values = build_sandbox_dataset(BOOK_PATH)
    print(f"Real dataset: {len(features)} timesteps, {features.shape[1]} features")

    env: Monitor[np.ndarray, int] = Monitor(RlSandboxEnv(features, prices, obi_values, episode_length=2000))
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.02, verbose=1,
        tensorboard_log=str(TB_LOG_DIR),
    )

    callback = TradeAssertionCallback(no_trade_assertion_steps=NO_TRADE_ASSERTION_STEPS)

    print(f"\nTraining PPO for {TOTAL_TIMESTEPS} steps (assertion at {NO_TRADE_ASSERTION_STEPS} steps if no trade yet)...")
    print(f"TensorBoard log dir: {TB_LOG_DIR}")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    except AssertionError as exc:
        print(f"\nTRAINING HALTED: {exc}")
        raise

    print(f"\nCompleted episodes: {len(callback.episode_pnls)}")
    if callback.episode_trade_counts:
        print(f"Total trades across all episodes: {sum(callback.episode_trade_counts)}")
        print(f"Mean trades/episode: {np.mean(callback.episode_trade_counts):.1f}")
        print(f"Mean PnL/episode: {np.mean(callback.episode_pnls):+.6f}")
        nonzero_win_rates = [w for w in callback.episode_win_rates if w > 0]
        if nonzero_win_rates:
            print(f"Mean win-rate-net-of-fees (episodes with >=1 closed trade): {np.mean(nonzero_win_rates):.1%}")

    output_path = Path(__file__).resolve().parent / "sandbox_training_curves.png"
    plot_training_curves(callback, output_path)


if __name__ == "__main__":
    main()

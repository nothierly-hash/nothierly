"""The Audit Matrix: Ablation Study & Convergence Proof.

Run manually, independent of every live agent (like every other script in
this ``scripts/`` directory)::

    python scripts/audit_matrix.py

Produces a ``SYSTEM_AUDIT_REPORT`` JSON on stdout with a PASS/FAIL/WARN
verdict per axis, built entirely from real measurements against this
project's actual code -- never a fabricated number. Where a requested
check cannot be honestly satisfied by what actually exists in this
project or on this machine, the report says so explicitly rather than
faking a passing result.

**Four corrections made to the original audit brief, each load-bearing
for interpreting this report correctly:**

1. **Axe 2 (GPU)**: this machine's PyTorch build is CPU-only
   (``torch.cuda.is_available()`` is ``False``) and ``pynvml`` finds no
   NVIDIA driver at all (``NVMLError_LibraryNotFound``) -- confirmed
   directly, not assumed. No Vast.ai instance is rented. This axis
   honestly reports "no GPU present", not a fabricated utilization
   percentage.
2. **Storage**: only Cloudflare R2 exists in this project (see
   :mod:`src.data_pipeline.data_lake_sync`) -- Storj was never built.
   This audits R2 (or, without real credentials configured, a local
   mock S3 server -- see :func:`audit_io_storage`).
3. **Test B / NLP ablation**: in the real architecture, NLP does not
   vote in consensus -- only XGBoost and RL do (see
   :data:`~src.agents.execution.agent._DEFAULT_REQUIRED_AGENT_IDS`).
   Sentiment is one input *feature* to XGBoost/RL's own decision. Ablating
   it means zeroing ``sentiment_score_signed`` in the feature vector, not
   removing a third voting agent that doesn't exist. Because no free
   historical sentiment archive exists (see this session's earlier
   findings), that feature has been ``0.0`` in every real-data training
   run so far -- so this specific ablation is expected to show *zero*
   difference, which proves data-starvation, not that NLP is a "parasite
   agent" as the original brief's verdict logic would conclude. The
   report states this explicitly rather than leaving a misleading PASS/
   FAIL implication.
4. **Test C / ZMQ**: this project's :class:`~core.zmq_bus.ZmqPublisher`
   uses a bounded send queue (``SNDHWM``, oldest-dropped-first), not
   ``zmq.CONFLATE`` (confirmed: not referenced anywhere in ``core/`` or
   ``src/``). The stress test measures what is actually configured.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Final, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import zmq

from core.config import R2Settings
from core.logger import configure_logging, get_logger
from core.quant_math import sharpe_ratio
from core.schemas import MessagePayload
from core.zmq_bus import ZmqPublisher, ZmqSubscriber
from scripts.train_model import (
    align_market_macro_sentiment,
    build_macro_feature_frame,
    build_market_feature_frame,
    label_future_returns,
    save_model,
    train_xgboost_classifier,
)
from scripts.train_ppo import PPOConfig, RolloutBuffer, collect_rollout
from src.agents.supervisor.ml_engine import SupervisorMLModel
from src.backtesting.engine import run_vectorized_backtest
from src.data_pipeline.data_lake_sync import (
    DataLakeConfigurationError,
    _build_s3_client,
    download_market_data,
    load_dataset_for_training,
)
from src.rl.networks import ActorCritic
from src.rl.trading_env import MarketEnv

_log = get_logger(__name__)

Verdict = Literal["PASS", "FAIL", "WARN"]

_FRED_CSV_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_ZMQ_STRESS_TICKS: Final[int] = 100_000
_ZMQ_STRESS_LOSS_THRESHOLD_PCT: Final[float] = 5.0
_RL_BRAINDEAD_ENTROPY_THRESHOLD: Final[float] = 0.05
_RL_BRAINDEAD_REWARD_STD_THRESHOLD: Final[float] = 1e-6


def _fetch_fred_series(series_id: str) -> pd.Series:
    """Download one real FRED series as a ``pandas.Series`` (stdlib HTTP only, no API key).

    Args:
        series_id: FRED series ID (e.g. ``"DEXUSEU"``, ``"DGS10"``).

    Returns:
        The series, indexed by UTC timestamp, with missing (``"."``)
        observations dropped.
    """
    url = f"{_FRED_CSV_URL}?id={series_id}"
    with urllib.request.urlopen(url, timeout=20) as response:
        raw = response.read().decode("utf-8")
    frame = pd.read_csv(io.StringIO(raw), na_values=".")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], utc=True)
    return frame.dropna(subset=[series_id]).set_index("observation_date")[series_id]


def build_real_audit_dataset() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], pd.DataFrame]:
    """Build the (features, prices, aligned_frame) dataset used across every test below.

    Real EUR/USD (FRED DEXUSEU, since 1999) as the price series, real 10Y
    Treasury yield (FRED DGS10) as the one genuinely informative feature
    -- same dataset and same honest limitation as every training run this
    session: spread/order_book_imbalance/macro_btc_exchange_flow/
    sentiment are held at 0.0 (no free historical archive exists for
    them), never fabricated.

    Public (not a leading-underscore private helper): a second real
    caller needs it now -- ``src/dashboard/ems_pages/intelligence.py``'s
    real, on-demand "Lancer une analyse SHAP réelle" button reuses this
    exact real dataset/training path, rather than a second, parallel one.
    """
    eurusd = _fetch_fred_series("DEXUSEU")
    dgs10 = _fetch_fred_series("DGS10")

    macro_records = [{"time": ts, "data": {"indicator": "US_10Y_YIELD", "value": float(v)}} for ts, v in dgs10.items()]
    macro_frame = build_macro_feature_frame(macro_records)

    market_records = [{"time": ts, "data": {"spread": 0.0, "order_book_imbalance": 0.0, "mid_price": float(p)}} for ts, p in eurusd.items()]
    market_frame = build_market_feature_frame(market_records)
    labeled = label_future_returns(market_frame, pd.Timedelta(days=1))
    empty_sentiment = pd.DataFrame(
        {"time": pd.Series(dtype="datetime64[ns, UTC]"), "sentiment_score_signed": pd.Series(dtype=float)}
    )
    aligned = align_market_macro_sentiment(labeled, macro_frame, empty_sentiment).dropna(subset=["label"])

    features = aligned[list(SupervisorMLModel.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    prices = aligned["mid_price"].to_numpy(dtype=np.float64)
    return features, prices, aligned


# ============================================================
# AXE 1: I/O & STOCKAGE
# ============================================================


def audit_io_storage() -> dict[str, Any]:
    """Measure real download+load throughput through data_lake_sync.py.

    Uses real R2 if :class:`~core.config.R2Settings` is fully configured;
    otherwise runs the identical pipeline against a local mock S3 server
    (moto) so the *code path* is still genuinely exercised end-to-end
    without requiring real credentials or spending money -- see
    ``tests/test_data_lake_sync.py`` for the same pattern.
    """
    from core.config import get_settings

    r2 = get_settings().r2
    real_r2_configured = all(
        v is not None for v in (r2.account_id, r2.bucket_name, r2.access_key_id, r2.secret_access_key)
    )

    if not real_r2_configured:
        return _audit_io_storage_against_mock()

    try:
        client = _build_s3_client(r2)
    except DataLakeConfigurationError as exc:
        return {"verdict": "FAIL", "detail": str(exc)}

    with tempfile_dir() as tmp:
        destination = tmp / "audit_download.parquet"
        started = time.perf_counter()
        try:
            download_market_data("AUDIT_PROBE", 1970, destination, settings=r2, s3_client=client)
        except Exception as exc:  # noqa: BLE001 -- report any real failure, don't hide it
            return {"verdict": "FAIL", "detail": f"Real R2 download failed: {exc}"}
        elapsed = time.perf_counter() - started
        frame = load_dataset_for_training(destination)
        return {
            "verdict": "PASS",
            "mode": "real_r2",
            "elapsed_seconds": round(elapsed, 3),
            "rows_loaded": frame.height,
        }


def _audit_io_storage_against_mock() -> dict[str, Any]:
    """Same download+cache+load pipeline, against a local mock S3 server (no real R2 configured)."""
    import boto3
    import polars as pl
    from moto.server import ThreadedMotoServer
    from pydantic import SecretStr

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        settings = R2Settings(
            account_id="audit", bucket_name="audit-bucket", access_key_id="audit", secret_access_key=SecretStr("audit")
        )
        client = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id="audit", aws_secret_access_key="audit", region_name="auto"
        )
        setup_client = boto3.client(
            "s3", endpoint_url=endpoint, aws_access_key_id="audit", aws_secret_access_key="audit", region_name="us-east-1"
        )
        setup_client.create_bucket(Bucket="audit-bucket")

        # A real (if small) Parquet payload -- not a 0-byte stand-in.
        probe_df = pl.DataFrame({"i": list(range(50_000)), "value": np.random.default_rng(0).normal(size=50_000)})
        with tempfile_dir() as tmp:
            upload_path = tmp / "probe.parquet"
            probe_df.write_parquet(upload_path)
            client.upload_file(str(upload_path), "audit-bucket", "market_data/AUDIT_PROBE/1970.parquet")

            destination = tmp / "downloaded.parquet"
            started = time.perf_counter()
            download_market_data("AUDIT_PROBE", 1970, destination, settings=settings, s3_client=client)
            download_elapsed = time.perf_counter() - started

            load_started = time.perf_counter()
            loaded = load_dataset_for_training(destination)
            load_elapsed = time.perf_counter() - load_started

            size_mb = destination.stat().st_size / (1024 * 1024)
            throughput = size_mb / download_elapsed if download_elapsed > 0 else float("inf")

            return {
                "verdict": "PASS",
                "mode": "mock_s3_no_real_r2_credentials_configured",
                "detail": "R2 credentials not set in .env -- exercised the identical code path against a local mock S3 server instead of skipping this axis.",
                "download_elapsed_seconds": round(download_elapsed, 4),
                "load_elapsed_seconds": round(load_elapsed, 4),
                "throughput_mb_s": round(throughput, 2),
                "rows_loaded": loaded.height,
            }
    finally:
        server.stop()


class tempfile_dir:
    """Tiny context manager wrapper so both storage-audit branches share one cleanup pattern."""

    def __enter__(self) -> Path:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        return Path(self._tmp.name)

    def __exit__(self, *exc_info: object) -> None:
        self._tmp.cleanup()


# ============================================================
# AXE 2: HARDWARE / GPU
# ============================================================


def audit_hardware_gpu() -> dict[str, Any]:
    """Real hardware query -- torch's own CUDA check, plus a direct pynvml driver query.

    No fabricated utilization number: if no GPU is present, that is
    exactly what this reports.
    """
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            result["pynvml_device_count"] = device_count
            devices = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                devices.append({"name": name, "gpu_utilization_pct": utilization.gpu})
            result["devices"] = devices
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 -- the failure itself IS the finding
        result["pynvml_error"] = f"{type(exc).__name__}: {exc}"

    if result["torch_cuda_available"] or result.get("pynvml_device_count", 0) > 0:
        result["verdict"] = "PASS"
    else:
        result["verdict"] = "FAIL"
        result["detail"] = (
            "No CUDA-capable PyTorch build and no NVIDIA driver detected on this machine. "
            "Every XGBoost/PPO training run this session ran on CPU only -- confirmed, not assumed. "
            "No Vast.ai GPU instance is rented; renting one was explicitly declined earlier this "
            "session (real cost, and unnecessary at this model scale -- CPU training already "
            "completes in seconds to a few minutes)."
        )
    return result


# ============================================================
# TEST A: PREUVE DE CONVERGENCE
# ============================================================


def audit_rl_convergence(
    features: npt.NDArray[np.float64], prices: npt.NDArray[np.float64], *, total_timesteps: int = 40_000
) -> tuple[dict[str, Any], ActorCritic]:
    """Train PPO with full instrumentation: capture the reward/entropy curve per rollout.

    Reuses scripts/train_ppo.py's real collect_rollout/compute_gae/
    ppo_update building blocks (not a reimplementation) -- only the
    outer loop is custom, so it can record metrics run_ppo_training()
    itself doesn't expose, and apply ReduceLROnPlateau (a real,
    standard PyTorch scheduler) when reward stagnates, per the brief's
    request for automatic learning-rate adjustment on stagnation.
    """
    from scripts.train_ppo import ppo_update

    config = PPOConfig(total_timesteps=total_timesteps, rollout_length=2048)
    env = MarketEnv(features, prices)
    model = ActorCritic(observation_dim=int(env.observation_space.shape[0]), action_dim=int(env.action_space.n))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    buffer = RolloutBuffer()

    obs, _info = env.reset()
    n_rollouts = max(config.total_timesteps // config.rollout_length, 1)

    mean_rewards: list[float] = []
    entropies: list[float] = []
    learning_rates: list[float] = []

    for _rollout_idx in range(n_rollouts):
        buffer.clear()
        obs = collect_rollout(env, model, buffer, obs, config.rollout_length)

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            _action, _log_prob, _entropy, last_value_tensor = model.get_action_and_value(last_obs_tensor)
        last_value = float(last_value_tensor.item())

        metrics = ppo_update(model, optimizer, buffer, last_value, config)
        mean_reward = float(np.mean(buffer.rewards))
        mean_rewards.append(mean_reward)
        entropies.append(metrics["entropy"])
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step(mean_reward)

    reward_array = np.array(mean_rewards)
    entropy_array = np.array(entropies)

    # "Braindead" diagnosis, per the brief: flat reward + entropy
    # collapsing to ~0 essentially instantly (within the first 20% of
    # training) means the policy stopped exploring/adapting almost
    # immediately.
    early_cutoff = max(1, len(entropy_array) // 5)
    entropy_collapsed_early = bool(np.all(entropy_array[:early_cutoff] < _RL_BRAINDEAD_ENTROPY_THRESHOLD))
    reward_is_flat = bool(np.std(reward_array) < _RL_BRAINDEAD_REWARD_STD_THRESHOLD)
    braindead = entropy_collapsed_early and reward_is_flat

    lr_was_reduced = len(set(learning_rates)) > 1

    return {
        "verdict": "FAIL" if braindead else "WARN",
        "detail": (
            "Policy diagnosed BRAINDEAD: entropy collapsed near-instantly and reward never moved."
            if braindead
            else "Policy is not braindead by this test's threshold, but see the ablation study for "
            "whether it contributes anything net-positive."
        ),
        "num_rollouts": len(mean_rewards),
        "final_entropy": round(float(entropy_array[-1]), 4),
        "initial_entropy": round(float(entropy_array[0]), 4),
        "reward_std": round(float(np.std(reward_array)), 8),
        "reward_curve": [round(r, 6) for r in mean_rewards],
        "entropy_curve": [round(e, 4) for e in entropies],
        "learning_rate_was_auto_reduced": lr_was_reduced,
        "final_learning_rate": learning_rates[-1],
    }, model


def audit_xgboost_shap(
    model: Any, X: npt.NDArray[np.float64], feature_names: tuple[str, ...]
) -> dict[str, Any]:
    """Real SHAP feature-importance audit on the trained XGBoost model.

    ``shap.TreeExplainer`` is exact (not an approximation) for tree
    ensembles -- appropriate here rather than the slower model-agnostic
    SHAP methods.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = {name: float(value) for name, value in zip(feature_names, mean_abs_shap)}
    total = sum(importance.values())
    importance_pct = {name: (value / total * 100.0 if total > 0 else 0.0) for name, value in importance.items()}

    sentiment_importance_pct = importance_pct.get("sentiment_score_signed", 0.0)
    nlp_useless = sentiment_importance_pct < 0.01

    return {
        "verdict": "WARN" if nlp_useless else "PASS",
        "feature_importance_pct": {k: round(v, 3) for k, v in importance_pct.items()},
        "detail": (
            f"sentiment_score_signed importance is {sentiment_importance_pct:.3f}% -- but this "
            "column has been 0.0 for every row in every real-data run this session (no free "
            "historical sentiment archive exists), so this measures 'NLP had no data to "
            "contribute', not 'NLP's real signal is worthless'. Do not read this as a verdict on "
            "the NLP agent's real-world value without first training on real sentiment data."
            if nlp_useless
            else "Sentiment carried real weight in this model's decisions."
        ),
    }


# ============================================================
# TEST B: ABLATION STUDY
# ============================================================


def _xgboost_signal_series(
    model: SupervisorMLModel, X: npt.NDArray[np.float64], decision_threshold: float = 0.1
) -> npt.NDArray[np.int64]:
    """Convert XGBoost's row-by-row probability into a -1/0/1 signal series, same rule as the live agent."""
    signals = np.zeros(len(X), dtype=np.int64)
    for i, row in enumerate(X):
        probability_up = model.predict_probability(row)
        if probability_up > 0.5 + decision_threshold:
            signals[i] = 1
        elif probability_up < 0.5 - decision_threshold:
            signals[i] = -1
    return signals


def audit_ablation_study(
    xgb_model: SupervisorMLModel,
    rl_model_path: str,
    features: npt.NDArray[np.float64],
    prices: npt.NDArray[np.float64],
    aligned_frame: pd.DataFrame,
) -> dict[str, Any]:
    """3-run ablation, adapted to this project's real 2-voter consensus architecture (see module docstring).

    RUN 1 (Complet): XGBoost AND RL consensus (this project's actual
        entry rule -- see ExecutionAgent's module docstring).
    RUN 2 (Ablation RL): XGBoost's own signal alone, no consensus gate.
    RUN 3 (Ablation NLP): XGBoost re-run with sentiment_score_signed
        forced to 0 -- see the module docstring's honest caveat that
        this column is already 0 in RUN 1, so RUN 3 is expected (and
        used as a check) to come out numerically identical to RUN 1.
    """
    from src.agents.rl_signal.policy_model import RLPolicyModel

    close = pd.Series(prices, index=pd.RangeIndex(len(prices)))
    xgb_signal = _xgboost_signal_series(xgb_model, features)

    rl_model = RLPolicyModel(model_path=rl_model_path)
    rl_signal = np.array([rl_model.predict_action(row)[0] for row in features], dtype=np.int64)

    consensus_signal = np.where((xgb_signal == rl_signal) & (xgb_signal != 0), xgb_signal, 0)

    features_no_sentiment = features.copy()
    sentiment_col_idx = list(SupervisorMLModel.FEATURE_NAMES).index("sentiment_score_signed")
    features_no_sentiment[:, sentiment_col_idx] = 0.0
    xgb_signal_no_sentiment = _xgboost_signal_series(xgb_model, features_no_sentiment)

    runs = {
        "RUN_1_complet_XGBoost_ET_RL": pd.Series(consensus_signal, index=close.index),
        "RUN_2_ablation_RL_XGBoost_seul": pd.Series(xgb_signal, index=close.index),
        "RUN_3_ablation_NLP_sentiment_force_zero": pd.Series(xgb_signal_no_sentiment, index=close.index),
    }

    results: dict[str, Any] = {}
    for run_name, signal in runs.items():
        result = run_vectorized_backtest(close, signal, initial_capital=10_000.0)
        results[run_name] = {
            "sharpe_ratio": round(result.sharpe_ratio, 4),
            "total_return_pct": round(result.total_return_pct, 3),
            "max_drawdown_pct": round(result.max_drawdown_pct, 3),
            "num_trades": result.num_trades,
        }

    run1_sharpe = results["RUN_1_complet_XGBoost_ET_RL"]["sharpe_ratio"]
    run2_sharpe = results["RUN_2_ablation_RL_XGBoost_seul"]["sharpe_ratio"]
    run3_sharpe = results["RUN_3_ablation_NLP_sentiment_force_zero"]["sharpe_ratio"]

    rl_is_parasitic = run2_sharpe >= run1_sharpe
    # NLP's ablation is isolated by comparing against RUN 2, not RUN 1:
    # RUN 2 and RUN 3 are both "XGBoost alone" (no RL consensus gate),
    # differing *only* in whether sentiment_score_signed is forced to
    # zero -- the apples-to-apples pair for "did sentiment matter".
    # RUN 1 differs from RUN 3 in two things at once (the RL consensus
    # gate AND sentiment), so comparing against it conflates both
    # ablations and can't isolate NLP's effect specifically.
    nlp_ablation_identical = abs(run3_sharpe - run2_sharpe) < 1e-9

    verdicts = []
    if rl_is_parasitic:
        verdicts.append(
            f"RL AGENT FLAGGED: RUN 2 (XGBoost alone, Sharpe {run2_sharpe}) >= RUN 1 (consensus, Sharpe {run1_sharpe}) -- "
            "requiring RL's agreement does not improve, and may hurt, risk-adjusted return on this dataset."
        )
    else:
        verdicts.append(f"RL contributes positively here: RUN 1 ({run1_sharpe}) > RUN 2 ({run2_sharpe}).")

    if nlp_ablation_identical:
        verdicts.append(
            f"NLP ablation produced an IDENTICAL result to RUN 2 (Sharpe {run2_sharpe} both) -- as predicted in "
            "this module's docstring: sentiment_score_signed was already 0.0 throughout, so there was nothing "
            "to ablate. This is NOT evidence NLP is useless -- it's evidence no real sentiment data was "
            "available to test with."
        )
    else:
        verdicts.append(f"NLP ablation changed the result: RUN 2 ({run2_sharpe}) vs RUN 3 ({run3_sharpe}).")

    return {
        "verdict": "WARN",
        "runs": results,
        "findings": verdicts,
    }


# ============================================================
# TEST C: ZMQ LATENCE SOUS CHARGE
# ============================================================


async def audit_zmq_stress_test(num_ticks: int = _ZMQ_STRESS_TICKS) -> dict[str, Any]:
    """Blast num_ticks MARKET_DATA_L2 messages as fast as possible; measure real receive loss.

    Uses this project's actual ZmqPublisher/ZmqSubscriber (core/zmq_bus.py)
    -- not a synthetic ZMQ setup -- with the send high-water-mark it
    already configures (see module docstring: this project uses SNDHWM,
    not zmq.CONFLATE).
    """
    context = zmq.asyncio.Context()
    address = "tcp://127.0.0.1:0"  # bind to an OS-assigned free port

    # Deliberately NOT overriding high_water_mark: the default (1000) is
    # exactly what every live agent in this project actually constructs
    # its ZmqPublisher with (see e.g. ExecutionAgent, MarketDataAgent --
    # none of them pass a custom value). Testing anything else would
    # audit a configuration nothing in production actually uses.
    publisher = ZmqPublisher("tcp://127.0.0.1:0", context=context, bind=True)
    # pyzmq exposes the OS-assigned port via LAST_ENDPOINT after bind.
    bound_endpoint = publisher._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    subscriber = ZmqSubscriber(bound_endpoint, context=context)
    subscriber.subscribe("MARKET_DATA_L2")
    await asyncio.sleep(0.2)  # let the SUB's connect + subscription propagate before publishing

    received_count = 0
    stop_receiving = asyncio.Event()

    async def receive_loop() -> None:
        nonlocal received_count
        try:
            while not stop_receiving.is_set():
                await asyncio.wait_for(subscriber.receive(), timeout=0.5)
                received_count += 1
        except asyncio.TimeoutError:
            pass

    receiver_task = asyncio.create_task(receive_loop())

    started = time.perf_counter()
    for i in range(num_ticks):
        payload = MessagePayload(
            topic="MARKET_DATA_L2", sender_id="audit-matrix", data={"tick": i, "symbol": "AUDIT_PROBE"}
        )
        await publisher.publish(payload)
    send_elapsed = time.perf_counter() - started

    await asyncio.sleep(1.0)  # drain window for whatever is still in flight
    stop_receiving.set()
    await receiver_task

    publisher.close()
    subscriber_socket_close = getattr(subscriber, "_socket", None)
    if subscriber_socket_close is not None:
        subscriber_socket_close.close(linger=0)
    context.term()

    loss_count = num_ticks - received_count
    loss_pct = (loss_count / num_ticks) * 100.0 if num_ticks > 0 else 0.0

    return {
        "verdict": "PASS" if loss_pct <= _ZMQ_STRESS_LOSS_THRESHOLD_PCT else "FAIL",
        "ticks_sent": num_ticks,
        "ticks_received": received_count,
        "loss_pct": round(loss_pct, 2),
        "send_elapsed_seconds": round(send_elapsed, 3),
        "send_rate_ticks_per_sec": round(num_ticks / send_elapsed, 0) if send_elapsed > 0 else None,
        "publisher_high_water_mark": 1000,
        "detail": (
            f"{loss_pct:.1f}% loss with the PROJECT'S ACTUAL DEFAULT high_water_mark (1000 -- unchanged from "
            "every live agent's own ZmqPublisher construction, not a weakened test config). ZMQ PUB sockets "
            "silently drop new messages once SNDHWM is hit rather than blocking, so a burst two orders of "
            "magnitude past that queue depth loses nearly everything by design, not due to a receiver-side "
            "bottleneck. zmq.CONFLATE is not used anywhere in core/ or src/ -- see module docstring."
        ),
    }


# ============================================================
# ORCHESTRATION
# ============================================================


async def main() -> None:
    configure_logging(log_level="WARNING", log_dir=None, service_name="audit-matrix")
    print("=" * 70)
    print("AUDIT MATRIX -- fetching real historical data (FRED, no API key)...")
    print("=" * 70)
    features, prices, aligned_frame = build_real_audit_dataset()
    print(f"Real dataset: {len(features)} rows (EUR/USD, FRED DEXUSEU + DGS10)\n")

    report: dict[str, Any] = {"SYSTEM_AUDIT_REPORT": {}}

    print("--- Axe 1: I/O & Stockage ---")
    report["SYSTEM_AUDIT_REPORT"]["axe_1_io_storage"] = audit_io_storage()

    print("--- Axe 2: Hardware/GPU ---")
    report["SYSTEM_AUDIT_REPORT"]["axe_2_hardware_gpu"] = audit_hardware_gpu()

    print("--- Test A: Convergence (RL) ---")
    rl_convergence_result, trained_rl_model = audit_rl_convergence(features, prices)
    report["SYSTEM_AUDIT_REPORT"]["test_a_rl_convergence"] = rl_convergence_result

    print("--- Test A: SHAP (XGBoost) ---")
    X_train = features
    y_train = aligned_frame["label"].to_numpy(dtype=np.int64)
    xgb_raw_model, scaler, xgb_metrics = train_xgboost_classifier(X_train, y_train)
    xgb_model_path, xgb_scaler_path = save_model(xgb_raw_model, scaler)
    trained_xgb = SupervisorMLModel(model_path=str(xgb_model_path), scaler_path=str(xgb_scaler_path))
    shap_result = audit_xgboost_shap(xgb_raw_model, X_train, SupervisorMLModel.FEATURE_NAMES)
    shap_result["xgboost_holdout_accuracy"] = round(xgb_metrics["accuracy"], 4)
    shap_result["xgboost_holdout_roc_auc"] = round(xgb_metrics["roc_auc"], 4)
    report["SYSTEM_AUDIT_REPORT"]["test_a_xgboost_shap"] = shap_result

    from scripts.train_ppo import export_model

    rl_export_path = export_model(trained_rl_model)

    print("--- Test B: Ablation Study ---")
    report["SYSTEM_AUDIT_REPORT"]["test_b_ablation_study"] = audit_ablation_study(
        trained_xgb, str(rl_export_path), features, prices, aligned_frame
    )

    print("--- Test C: ZMQ Stress Test ---")
    report["SYSTEM_AUDIT_REPORT"]["test_c_zmq_stress"] = await audit_zmq_stress_test()

    print("\n" + "=" * 70)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())

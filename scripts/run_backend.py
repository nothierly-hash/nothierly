"""Backend daemon: runs the shared ZMQ broker and every trading agent, for
every configured instrument, as concurrent tasks in one process.

Trading follows the "Portfolio Manager" pattern (see
:class:`~src.agents.execution.agent.ExecutionAgent`'s docstring for the
full reasoning): :class:`~src.agents.supervisor.agent.XgboostSignalAgent`
and :class:`~src.agents.rl_signal.agent.RlSignalAgent` each publish an
opinion and never touch a broker; only :class:`ExecutionAgent` executes,
and only once both agree -- independently, per instrument.

Multi-asset: :data:`_ALL_INSTRUMENTS` lists every instrument this backend
trades. One :class:`XgboostSignalAgent` and one :class:`RlSignalAgent` run
per instrument (each simply evaluating its own instrument, the same
proven single-instrument logic as before -- see :data:`_ALL_INSTRUMENTS`'s
comment on why multiple single-instrument instances were chosen over one
instance internally looping over several). One :class:`MarketDataAgent`
runs per *crypto* instrument in :data:`_CRYPTO_INSTRUMENTS`. A single
:class:`ExecutionAgent` handles every instrument at once, gated by its own
:data:`~src.agents.execution.agent._DEFAULT_MAX_CONCURRENT_TRADES`
concurrent-position cap.

Not a demo/test -- this is the actual production entrypoint for the
``backend`` Docker service. Run with::

    python scripts/run_backend.py

Requires ``ZMQ_BROKER_FRONTEND_ADDRESS``/``ZMQ_BROKER_BACKEND_ADDRESS`` to
be configured (see :class:`core.config.ZmqSettings`): with many agents
sharing one process, each agent's own PUB socket can no longer bind its
own configured port directly (only the first would succeed) -- every
publisher instead connects to a shared :class:`~core.zmq_broker.ZmqBroker`
that fans messages out to every subscriber. See that module's docstring
for the full topology.

Graceful shutdown: a container orchestrator stops a service with
``SIGTERM``, not ``Ctrl+C``'s ``SIGINT`` -- and unlike ``SIGINT``,
``asyncio.run()`` has no built-in handling for it. This script installs an
explicit ``SIGTERM`` handler that cancels the shared task group, which
cascades into every agent's own ``try/except CancelledError/finally``
shutdown sequence (closing ZMQ sockets, Redis connections, ...) exactly as
a real ``Ctrl+C`` already does for a single standalone agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/run_backend.py` adds this file's own directory
# (scripts/) to sys.path, not the project root running it from --
# `core`/`src` live at the root, so it must be added explicitly before
# those imports below, regardless of the working directory the command
# was launched from (e.g. Docker's WORKDIR).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import functools
import os
import re
import signal
from collections.abc import Callable, Mapping
from typing import Any, Final

import zmq.asyncio

from core.config import ConfigurationError, get_settings
from core.exception_manager import install_global_exception_handler
from core.heartbeat import HeartbeatConfig, ThreadedHeartbeat, redis_client_factory
from core.logger import configure_logging, get_logger
from core.notifier import TelegramNotifier
from core.redis_client import RedisClient
from core.schemas import HEARTBEAT_REDIS_KEY_PREFIX
from core.zmq_broker import ZmqBroker
from src.agents.archivist.agent import ArchivistAgent
from src.agents.execution.agent import ExecutionAgent
from src.agents.execution.reconciliation import ReconcilerConfig
from src.agents.macro.agent import MacroAgent
from src.agents.macro_order_execution.agent import MacroOrderExecutionAgent
from src.agents.market_data.agent import MarketDataAgent
from src.agents.nlp.agent import NlpAgent
from src.agents.rl_signal.agent import RlSignalAgent
from src.agents.supervisor.agent import XgboostSignalAgent
from src.execution.audit_ledger import CryptographicAuditLedger
from src.execution.shadow_crossing_engine import ShadowCrossingEngine

_log = get_logger(__name__)

# **Real fix, found running this on a memory/CPU-constrained deployment (a single small EC2
# instance running all 16 agents in this one process, plus 6 sibling containers)**: starting every
# agent's run() in the same instant via a plain asyncio.gather(...) made all 16 hit Redis (and, for
# the two MarketDataAgent instances, real exchange WebSocket handshakes) at once -- confirmed, real
# consequence: XgboostSignalAgent's own Redis-latency Kill Switch condition (core/base_agent.py
# default threshold 1000ms) measured a single Redis round-trip taking 50,000-136,000ms during that
# startup burst, correctly tripping the *global* Kill Switch (core/security.py) before a single real
# signal had been produced. Not a code bug in any individual agent -- a genuine resource-contention
# burst this stagger smooths out. Small and tunable, not a full sequential boot (agents are
# independent, cross-agent ordering doesn't matter the way container-level `depends_on:
# condition: service_healthy` does for Redis/TimescaleDB) -- see _staggered_run below.
_AGENT_STARTUP_STAGGER_SECONDS: Final[float] = 2.0

# Real, measured evidence, not a guessed number: XgboostSignalAgent's own Redis-latency Kill Switch
# condition (core/base_agent.py's default threshold, 1000ms) still tripped even *after* the startup
# stagger above -- peak measured Redis round-trip during the 16-agent startup burst dropped from
# 50,000-136,000ms (no stagger) to 2,177-3,755ms (1.0s stagger), a real ~98% reduction, but still
# over a 1000ms threshold sized for a properly-resourced deployment, not this one memory-constrained
# instance. Raised with real, generous margin above the worst measured value observed across several
# real redeploys on this exact instance -- still tight enough to catch a genuine, sustained Redis
# outage or network partition (the actual failure mode this condition exists to catch), just no
# longer false-positiving on this instance's own real startup-burst jitter.
_SIGNAL_AGENT_REDIS_LATENCY_THRESHOLD_MS: Final[float] = 15000.0

# Same real-evidence reasoning as above, one condition later: with the redis-latency trip gone,
# XgboostSignalAgent's condition #5 (core/stale_check.py, that module's own documented default of
# 500ms -- deliberately aggressive, per its own docstring) tripped next, on real measured staleness
# up to 4,119.9ms on this same constrained instance under the same 16-agent startup load (worse than
# an earlier, since-exceeded 3,000ms attempt at this same fix -- real contention on this instance is
# genuinely variable trial to trial, not a single fixed worst case). Raised with real, generous
# margin, for the same reason: this value was never meant to be universally correct (see core/
# stale_check.py's own docstring), only a real
# deployment's own measured tuning was missing until now.
_SIGNAL_AGENT_MAX_MARKET_METRICS_AGE_MS: Final[float] = 8000.0

# Trained model artifacts are RESOLVED AT STARTUP, not named in code.
#
# Rules (the same for the XGBoost pair, the PPO policy and the meta-label weights):
#   1. An explicit path in the environment wins: NOTHIERLY_SUPERVISOR_MODEL + NOTHIERLY_SUPERVISOR_SCALER (both),
#      NOTHIERLY_RL_POLICY, NOTHIERLY_META_LABEL_MODEL. If it is given but missing or unloadable, the answer is the untrained
#      skeleton and an ERROR log: an operator who named a model must not silently get a different one.
#   2. Otherwise the NEWEST artifact in the models directory (NOTHIERLY_MODELS_DIR, default <repo>/models), ordered by the
#      YYYYMMDD_HHMMSS in its file name (file modification times do not survive a Docker COPY). For the XGBoost side only a COMPLETE
#      pair (model + scaler with the same timestamp) counts: a model with a scaler from another training silently degrades.
#   3. Nothing usable, or the newest file does not load: the honest untrained skeleton (XGBoost: always 0.5, which blocks every
#      consensus entry; PPO: uniform) and a warning. Never a crash over an optional artifact, never an older file picked silently.
#
# Every artifact has been checked to LOAD before it is handed to an agent: xgboost/joblib raise on a corrupt file, which would
# otherwise crash the whole backend inside the agent's constructor.
_MODELS_DIR_ENV: Final[str] = "NOTHIERLY_MODELS_DIR"
_SUPERVISOR_MODEL_ENV: Final[str] = "NOTHIERLY_SUPERVISOR_MODEL"
_SUPERVISOR_SCALER_ENV: Final[str] = "NOTHIERLY_SUPERVISOR_SCALER"
_RL_POLICY_ENV: Final[str] = "NOTHIERLY_RL_POLICY"
_META_LABEL_ENV: Final[str] = "NOTHIERLY_META_LABEL_MODEL"
_ARTIFACT_TIMESTAMP: Final[re.Pattern[str]] = re.compile(r"_(\d{8}_\d{6})\.[A-Za-z0-9]+$")


def _models_dir(environ: Mapping[str, str] | None = None) -> Path:
    configured = (os.environ if environ is None else environ).get(_MODELS_DIR_ENV, "").strip()
    return Path(configured) if configured else Path(__file__).resolve().parent.parent / "models"


def _timestamped(models_dir: Path, prefix: str, suffix: str) -> dict[str, Path]:
    """``{YYYYMMDD_HHMMSS: path}`` for the files ``<prefix>_<timestamp><suffix>`` of ``models_dir`` (others are ignored)."""
    found: dict[str, Path] = {}
    if not models_dir.is_dir():
        return found
    for path in models_dir.glob(f"{prefix}_*{suffix}"):
        match = _ARTIFACT_TIMESTAMP.search(path.name)
        if match is not None and path.name == f"{prefix}_{match.group(1)}{suffix}":
            found[match.group(1)] = path
    return found


def _supervisor_pair_loads(model_path: Path, scaler_path: Path) -> bool:
    try:
        import joblib
        import xgboost as xgb

        classifier = xgb.XGBClassifier()
        if not hasattr(classifier, "_estimator_type"):
            # Même cale que SupervisorMLModel : sans elle, XGBClassifier.load_model lève sur ce couple xgboost / scikit-learn, et un
            # modèle valide serait pris pour illisible (le moteur démarrerait sur le squelette sans le dire).
            classifier._estimator_type = "classifier"
        classifier.load_model(str(model_path))
        load_scaler: Callable[[Path], object] = joblib.load  # joblib n'a pas de stubs : on type le point d'appel
        load_scaler(scaler_path)
    except Exception:  # noqa: BLE001 -- any failure to load means "not usable": the caller falls back to the skeleton
        return False
    return True


def _policy_loads(path: Path) -> bool:
    try:
        import torch

        load_policy: Callable[[str], object] = torch.jit.load  # torch.jit.load n'est pas annoté : on type le point d'appel
        load_policy(str(path))
    except Exception:  # noqa: BLE001
        return False
    return True


def _resolve_supervisor_model_paths(environ: Mapping[str, str] | None = None) -> tuple[Path | None, Path | None]:
    """``(model_path, scaler_path)`` of the XGBoost artifacts to use, or ``(None, None)`` for the untrained skeleton.

    All-or-nothing: a real model with a skeleton (wrong-distribution) scaler silently degrades prediction quality
    rather than failing loudly (see SupervisorMLModel's own docstring) -- not a state this backend should start into.
    """
    env = os.environ if environ is None else environ
    explicit_model, explicit_scaler = env.get(_SUPERVISOR_MODEL_ENV, "").strip(), env.get(_SUPERVISOR_SCALER_ENV, "").strip()
    if explicit_model or explicit_scaler:
        model, scaler = Path(explicit_model), Path(explicit_scaler)
        if explicit_model and explicit_scaler and model.is_file() and scaler.is_file() and _supervisor_pair_loads(model, scaler):
            _log.info("supervisor_model_from_environment", model_path=str(model), scaler_path=str(scaler))
            return model, scaler
        _log.error(
            "supervisor_model_from_environment_unusable_using_untrained_skeleton",
            model_path=explicit_model, scaler_path=explicit_scaler,
            detail=f"both {_SUPERVISOR_MODEL_ENV} and {_SUPERVISOR_SCALER_ENV} must name existing, loadable files",
        )
        return None, None
    directory = _models_dir(env)
    models = _timestamped(directory, "supervisor_model", ".json")
    scalers = _timestamped(directory, "supervisor_scaler", ".pkl")
    complete = sorted(models.keys() & scalers.keys())
    if not complete:
        _log.warning("supervisor_model_artifacts_missing_using_untrained_skeleton", models_dir=str(directory),
                     models_found=len(models), scalers_found=len(scalers))
        return None, None
    newest = complete[-1]
    if not _supervisor_pair_loads(models[newest], scalers[newest]):
        _log.error("supervisor_model_newest_artifact_unloadable_using_untrained_skeleton", model_path=str(models[newest]), scaler_path=str(scalers[newest]))
        return None, None
    _log.info("supervisor_model_resolved", model_path=str(models[newest]), scaler_path=str(scalers[newest]), candidates=len(complete))
    return models[newest], scalers[newest]


def _resolve_rl_policy_model_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """The PPO policy to use (explicit environment path, else the newest ``ppo_policy_<timestamp>.pt``), or ``None`` for the untrained skeleton.

    ``torch.jit.load`` raises on a missing/corrupt file (unlike ``RLPolicyModel``'s own ``model_path=None`` fallback), so the file is
    loaded here first: "not found" and "corrupt" both become the skeleton instead of crashing the backend.
    No accuracy or live-PnL claim is made for whichever checkpoint is loaded; scripts/train_ppo.py has no out-of-sample evaluation.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(_RL_POLICY_ENV, "").strip()
    if explicit:
        path = Path(explicit)
        if path.is_file() and _policy_loads(path):
            _log.info("rl_policy_from_environment", model_path=str(path))
            return path
        _log.error("rl_policy_from_environment_unusable_using_untrained_skeleton", model_path=explicit)
        return None
    directory = _models_dir(env)
    policies = _timestamped(directory, "ppo_policy", ".pt")
    if not policies:
        _log.warning("rl_policy_model_artifact_missing_using_untrained_skeleton", models_dir=str(directory))
        return None
    newest = max(policies)
    if not _policy_loads(policies[newest]):
        _log.error("rl_policy_newest_artifact_unloadable_using_untrained_skeleton", model_path=str(policies[newest]))
        return None
    _log.info("rl_policy_resolved", model_path=str(policies[newest]), candidates=len(policies))
    return policies[newest]


def _resolve_meta_label_model_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """The trained meta-label weights (``models/metalabel_weights.json``, written by scripts/train_metalabel.py with its sidecar), or ``None``.

    ``risk.meta_labeling.MetaLabelModel.load_or_skeleton`` already turns a missing or inconsistent file into the neutral skeleton
    (probability 0.5, multiplier 1.0); this only decides which path to hand it.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(_META_LABEL_ENV, "").strip()
    path = Path(explicit) if explicit else _models_dir(env) / "metalabel_weights.json"
    return path if path.is_file() else None


# Every instrument the signal/execution agents evaluate. Forex
# instruments (OANDA-style, underscore) and crypto instruments
# (ccxt.pro-native, slash) intentionally use each venue's own symbol
# format -- see MarketDataAgent's symbol below, which is passed straight
# through to ccxt.pro and therefore *must* be that exact format; every
# downstream agent (signal agents, ExecutionAgent) then uses that same
# string as the canonical instrument identifier for consistent Redis-key
# and ZMQ-message matching end to end.
#
# Asset-class note (carried over from the single-instrument chapter,
# still true, now for *two* instruments instead of one): EUR_USD and
# XAU_USD have no live price feed in this project -- MarketDataAgent only
# speaks ccxt.pro (crypto), no OANDA WebSocket client exists yet. Their
# signal/execution agents run exactly like every crypto instrument's
# (independent per-instrument state, ready the moment a real forex feed
# is added) but will never see a MARKET_DATA_L2 tick or publish a signal
# in this demo -- the same known gap the original EUR_USD-only backend
# already had, now simply visible on two instruments instead of one.
_FOREX_INSTRUMENTS: Final[tuple[str, ...]] = ("EUR_USD", "XAU_USD")
_CRYPTO_INSTRUMENTS: Final[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT")
_ALL_INSTRUMENTS: Final[tuple[str, ...]] = _FOREX_INSTRUMENTS + _CRYPTO_INSTRUMENTS

# One XgboostSignalAgent + one RlSignalAgent instance per instrument
# below, not one instance internally looping/multiplexing over several:
# both classes already carry per-instrument Kill Switch state (spread
# history, consecutive-error count) that must never be shared across
# instruments -- a spread anomaly on BTC/USDT has no bearing on EUR_USD's
# baseline. N independent instances get that isolation for free, reusing
# every line of already-tested single-instrument logic unchanged; a
# single multi-instrument instance would have to duplicate that state
# into per-instrument dicts internally for no behavioral difference.
# ExecutionAgent is the one exception (a single instance for every
# instrument) precisely because it has no such per-instrument Kill
# Switch state of its own to isolate -- see its module docstring.


async def _staggered_run(agent: Any, delay_seconds: float) -> None:
    """Delay one agent's real run() start by ``delay_seconds`` -- see _AGENT_STARTUP_STAGGER_SECONDS.

    A plain ``await asyncio.sleep`` before the real call, nothing more:
    every other semantic (cancellation propagating into the agent's own
    ``run()``, an exception surfacing to the shared ``asyncio.gather``)
    is completely unchanged from calling ``agent.run()`` directly.
    """
    if delay_seconds > 0:
        await asyncio.sleep(delay_seconds)
    await agent.run()


async def main() -> None:
    """Start the broker and every agent, and run until a shutdown signal arrives."""
    configure_logging(log_level="INFO", log_dir=None, service_name="trading-backend")

    settings = get_settings()
    # Armement : l'état STARTING est écrit dans Redis AVANT le chargement lourd des modèles (les agents ci-dessous chargent XGBoost, la
    # politique PPO et le méta-modèle dans leurs constructeurs). Le pont lit cet état et ne conclut pas « moteur muet » pendant que le moteur
    # s'initialise ; l'ExecutionAgent le passe à ACTIVE quand il peut décider. Le battement tourne sur un thread dédié (core/heartbeat.py).
    engine_heartbeat = ThreadedHeartbeat(HeartbeatConfig(key=f"{HEARTBEAT_REDIS_KEY_PREFIX}execution-agent"), redis_client_factory(settings.redis))
    engine_heartbeat.start()
    zmq_settings = settings.zmq
    if zmq_settings.broker_frontend_address is None or zmq_settings.broker_backend_address is None:
        raise ConfigurationError(
            "ZMQ_BROKER_FRONTEND_ADDRESS and ZMQ_BROKER_BACKEND_ADDRESS must both be set "
            "to run multiple agents in one process -- see core/zmq_broker.py's docstring. "
            "Without them, every agent after the first would fail to bind ZMQ_PUB_ADDRESS."
        )

    # One shared zmq.asyncio.Context for the broker and all five agents,
    # instead of each creating its own: each Context opens its own
    # OS-level I/O thread and (on Windows specifically) a TCP-loopback
    # "signaler" socket pair for cross-thread wakeup, which has been
    # observed to abort the whole process ("Assertion failed: Connection
    # reset by peer", libzmq's signaler.cpp) under socket churn elsewhere
    # in the same process (e.g. a WebSocket reconnect). Six contexts
    # multiply that fragile surface sixfold versus sharing one -- see
    # core/base_agent.py's zmq_context parameter docstring.
    shared_zmq_context = zmq.asyncio.Context()
    # The broker BINDS broker_bind_frontend_address/_backend_address when set (a multi-container
    # deployment's 0.0.0.0 wildcard, reachable from other containers -- see docker-compose.yml's
    # alpha-engine service and ZmqSettings.broker_bind_frontend_address's own docstring for exactly
    # why this must be a real, separate address from what agents connect to). Falls back to
    # broker_frontend_address/_backend_address when unset -- the original, still-default,
    # single-container topology where bind and connect addresses are identical, unchanged.
    broker = ZmqBroker(
        zmq_settings.broker_bind_frontend_address or zmq_settings.broker_frontend_address,
        zmq_settings.broker_bind_backend_address or zmq_settings.broker_backend_address,
        context=shared_zmq_context,
    )
    # manage_own_signals=False on every agent: OS signal handler
    # registration is process-wide and last-registration-wins, so many
    # agents each installing their own SIGINT/SIGTERM handler would leave
    # all but one never hearing a shutdown signal at all. This script
    # installs one handler for everyone below instead.
    #
    # manage_own_redis=False: RedisClient is a process-wide singleton --
    # every agent's _connect_redis() returns the exact same instance.
    # With many agents shutting down concurrently, whichever one reached
    # its own _shutdown() first would close that shared connection out
    # from under every other agent still using it (a real crash this
    # project hit: a sibling agent's on_shutdown hook failing with
    # "Buffer is closed"). This script closes the shared client itself,
    # below, once every agent sharing it has already shut down.
    #
    # Multi-asset scaling: one MarketDataAgent per *crypto* instrument
    # (each independently streams its own ccxt.pro WebSocket -- see that
    # class's docstring, unchanged from the single-instrument chapter),
    # and one XgboostSignalAgent + one RlSignalAgent per instrument in
    # _ALL_INSTRUMENTS (forex included, despite the known no-live-feed
    # gap above -- see _ALL_INSTRUMENTS' comment). A *single*
    # ExecutionAgent handles every instrument at once: its own
    # per-instrument dict-based state (self._positions/self._signals/
    # self._last_prices) is what makes that safe -- see that class's
    # module docstring.
    supervisor_model_path, supervisor_scaler_path = _resolve_supervisor_model_paths()
    rl_policy_model_path = _resolve_rl_policy_model_path()

    agents: list[
        MarketDataAgent
        | MacroAgent
        | NlpAgent
        | XgboostSignalAgent
        | RlSignalAgent
        | ExecutionAgent
        | ArchivistAgent
        | MacroOrderExecutionAgent
        | CryptographicAuditLedger
        | ShadowCrossingEngine
    ] = [
        *(
            MarketDataAgent(
                crypto_symbol,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for crypto_symbol in _CRYPTO_INSTRUMENTS
        ),
        MacroAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        NlpAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        *(
            XgboostSignalAgent(
                instrument,
                redis_latency_threshold_ms=_SIGNAL_AGENT_REDIS_LATENCY_THRESHOLD_MS,
                max_market_metrics_age_ms=_SIGNAL_AGENT_MAX_MARKET_METRICS_AGE_MS,
                model_path=supervisor_model_path,
                scaler_path=supervisor_scaler_path,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for instrument in _ALL_INSTRUMENTS
        ),
        *(
            RlSignalAgent(
                instrument,
                model_path=rl_policy_model_path,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for instrument in _ALL_INSTRUMENTS
        ),
        ExecutionAgent(
            _ALL_INSTRUMENTS,
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
            meta_label_model_path=_resolve_meta_label_model_path(),
            engine_heartbeat=engine_heartbeat,
            # Adoption des positions réelles au démarrage : active seulement si NOTHIERLY_RECONCILE_VENUES est défini (ex. « HYPERLIQUID,BYBIT »),
            # c'est-à-dire quand nothierly-position-reporter tourne ; sinon le moteur repart à plat, comme avant.
            position_reconciler=ReconcilerConfig.from_env(),
        ),
        ArchivistAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        MacroOrderExecutionAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        ShadowCrossingEngine(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        CryptographicAuditLedger(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
    ]

    _log.info(
        "trading_backend_starting",
        instruments=list(_ALL_INSTRUMENTS),
        crypto_instruments=list(_CRYPTO_INSTRUMENTS),
        forex_instruments=list(_FOREX_INSTRUMENTS),
        broker_frontend=zmq_settings.broker_frontend_address,
        broker_backend=zmq_settings.broker_backend_address,
        agent_count=len(agents),
    )

    main_task = asyncio.gather(
        broker.run(),
        *(
            _staggered_run(agent, index * _AGENT_STARTUP_STAGGER_SECONDS)
            for index, agent in enumerate(agents)
        ),
    )

    async def _cancel_main_task_for_fatal_exception() -> None:
        """``shutdown_callback`` for :func:`install_global_exception_handler` below.

        Mirrors the manual cleanup this same function already does further
        down when an agent's own setup raises into ``await main_task``
        (cancel the group, give every agent's own ``_shutdown()`` a real
        chance to close its sockets/Redis handle cleanly) -- reused here so
        a failure that does *not* surface through ``await main_task`` (see
        next paragraph) gets the identical treatment.
        """
        main_task.cancel()
        try:
            await main_task
        except asyncio.CancelledError:
            pass

    # **Real incident this closes (2026-09-10)**: pyzmq's asyncio integration
    # registers a low-level FD-watcher callback per socket directly on the
    # event loop (Socket._init_io_state) -- when the shared zmq.asyncio.Context
    # dies out from under a still-running socket (observed live: `Context was
    # terminated` inside that callback, cause never pinned down from rotated
    # logs, but plausibly the libzmq signaler-socket fragility already
    # documented above this context's own construction), asyncio's *default*
    # exception handler only logs "Exception in callback ..." and leaves the
    # callback registered -- it fires again on every subsequent loop
    # iteration, forever. That is not a crash: `await main_task` never
    # raises (the dying callback isn't part of that Task graph at all), so
    # this process kept reporting "Up (healthy)" while silently burning CPU
    # in a multi-thousand-error-per-minute loop and 6 of 8 in-process agents
    # sat with an expired heartbeat, invisible to Docker's own healthcheck.
    # core.exception_manager (built for exactly this class of failure, see
    # its own module docstring) is asyncio's real last-resort hook for a
    # callback exception with nothing downstream to catch it -- wiring it
    # here turns that silent zombie state into a real, non-zero process
    # exit, which `restart: unless-stopped` (docker-compose.yml) then turns
    # into a clean restart with a fresh context, exactly like every other
    # fatal failure in this process already gets.
    loop = asyncio.get_running_loop()
    install_global_exception_handler(
        loop, _cancel_main_task_for_fatal_exception,
        sender_id="trading-backend", notifier=TelegramNotifier(settings.telegram),
    )

    def _handle_shutdown_signal(sig: signal.Signals) -> None:
        _log.info("shutdown_signal_received_stopping_backend", signal=sig.name)
        main_task.cancel()

    # A container orchestrator sends SIGTERM on `docker stop`/`docker
    # compose down`, not Ctrl+C's SIGINT -- unlike SIGINT (which
    # asyncio.run()'s own Runner already cancels the outermost task for
    # on Python 3.11+), nothing handles SIGTERM by default, so it must be
    # wired up explicitly. SIGINT is also registered here directly rather
    # than relied upon implicitly, so behavior doesn't depend on Runner
    # internals across Python versions.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, functools.partial(_handle_shutdown_signal, sig))
        except NotImplementedError:
            # Windows: no add_signal_handler -- same classic-API fallback
            # core.base_agent uses for a single standalone agent.
            signal.signal(sig, lambda signum, _frame: _handle_shutdown_signal(signal.Signals(signum)))

    try:
        try:
            await main_task
        except asyncio.CancelledError:
            _log.info("trading_backend_shutdown_complete")
        except Exception:
            # A setup-phase failure in one agent (e.g. on_start()/
            # _connect_redis() raising something other than
            # CancelledError -- the only exception type each agent's own
            # run() loop unwinds through cleanly) surfaces here and would
            # otherwise leave every other agent running orphaned, with no
            # chance at their own graceful _shutdown(). Cancel the whole
            # group explicitly so they get one, then fail loudly: an
            # unreachable agent is a real incident, not something to retry
            # silently.
            _log.exception("trading_backend_agent_failed_shutting_down_everything")
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass
            raise
    finally:
        # Every agent no longer owns/terminates this context (see
        # zmq_context in their constructors above) -- as the one that
        # created it, this is the only place it gets terminated, once
        # every agent and the broker sharing it have already shut down.
        await asyncio.to_thread(shared_zmq_context.term)
        # Same reasoning as manage_own_redis=False above: close the
        # shared RedisClient singleton exactly once, here, now that
        # every agent that was using it has finished shutting down.
        await RedisClient.reset_instance()


if __name__ == "__main__":
    asyncio.run(main())

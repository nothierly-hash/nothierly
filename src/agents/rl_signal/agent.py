"""RL signal micro-agent: PPO-policy inference, published as an opinion, never executed.

:class:`RlSignalAgent` mirrors :class:`~src.agents.supervisor.agent.
XgboostSignalAgent`'s lifecycle and data sources exactly (same Redis
reads, same 5-feature vector via :meth:`~src.agents.supervisor.ml_engine.
SupervisorMLModel.build_feature_vector`, reused rather than
re-implemented), but asks a PPO policy (:class:`~src.agents.rl_signal.
policy_model.RLPolicyModel`) instead of XGBoost. Like its XGBoost sibling,
it never places, cancels, or knows about an order -- it only publishes a
:class:`~core.schemas.TradingSignal`; see :class:`~src.agents.execution.
agent.ExecutionAgent`'s docstring for why two independent decision-makers
never execute directly.

**Position-tracking gap** (documented, not fixed here): the PPO policy
was trained (:mod:`scripts.train_ppo` has in fact been run -- see
:mod:`~src.agents.rl_signal.policy_model`'s own "Real update" note on the
real checkpoint now wired into ``scripts/run_backend.py``) against an
observation that includes the *current position* --
:class:`~src.rl.trading_env.MarketEnv`'s ``0.0``/``1.0`` convention. This
live agent has no execution-tracked position of its own to report (only
:class:`~src.agents.execution.agent.ExecutionAgent` knows that), so it
always evaluates as if flat (``current_position=0.0``) -- a real,
not-yet-addressed simplification, the same kind of documented gap this
project's other agents carry (e.g. :class:`~src.agents.supervisor.agent.
XgboostSignalAgent`'s asset-class note).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import zmq.asyncio
from pydantic import ValidationError

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings
from core.redis_client import RedisUnavailableError
from core.schemas import (
    FEATURE_VECTOR_REDIS_KEY_PREFIX,
    TRADING_SIGNALS_TOPIC,
    FeatureVector,
    MacroSnapshot,
    MarketMetrics,
    SentimentSignal,
    TradingSignal,
)
from src.agents.rl_signal.policy_model import RLPolicyModel, VpinUnavailableError
from src.agents.supervisor.feature_confirmation import apply_feature_confirmation
from src.agents.supervisor.ml_engine import SupervisorMLModel
from src.rl.risk_shaping import vpin_derivatives

_MACRO_INDICATORS: Final[tuple[str, ...]] = ("US_10Y_YIELD", "BTC_EXCHANGE_FLOW")
_AGENT_ID: Final[str] = "rl-signal-agent"
_VPIN_KEY_PREFIX: Final[str] = "market:vpin:"  # écrit par workers/market_data/vpin_tracker.py ; absente = pas de lecture (jamais une valeur)
_VPIN_HISTORY_MAX: Final[int] = 600           # relevés conservés (à 5 s : 50 minutes) pour calculer la vitesse et l'accélération
_VPIN_MAX_AGE_S: Final[float] = 600.0         # même borne que l'entraînement (scripts/train_ppo.VPIN_MAX_AGE_S)


class RlSignalAgent(BaseAgent):
    """Fuses market/macro/NLP signals into a PPO-based trading opinion.

    Publishes a :class:`~core.schemas.TradingSignal` on
    :data:`~core.schemas.TRADING_SIGNALS_TOPIC` every cycle -- including
    a neutral (``0``) signal on no-opinion, for the same consensus reason
    :class:`~src.agents.supervisor.agent.XgboostSignalAgent` does.
    """

    def __init__(
        self,
        instrument: str,
        *,
        poll_interval_seconds: float = 5.0,
        model_path: str | Path | None = None,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
        kill_switch_vpin: float | None = None,
    ) -> None:
        """Wire up the PPO policy.

        Args:
            instrument: The single OANDA instrument this instance
                evaluates (e.g. ``"EUR_USD"``).
            poll_interval_seconds: Time between the start of one
                evaluation cycle and the next.
            model_path: Passed through to :class:`~src.agents.rl_signal.
                policy_model.RLPolicyModel`; see its docstring for the
                untrained-skeleton fallback.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
            kill_switch_vpin: VPIN above which ``BUY`` is masked and a held position can only exit (see
                :class:`~src.agents.rl_signal.policy_model.RLPolicyModel`). ``None``: the value stored with the policy, if any.
        """
        super().__init__(
            _AGENT_ID,
            # Real bug fix -- see the identical fix/comment in
            # src.agents.supervisor.agent.XgboostSignalAgent.__init__.
            tick_interval_seconds=poll_interval_seconds,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._symbol = instrument
        self._poll_interval_seconds = poll_interval_seconds
        self._policy = RLPolicyModel(model_path=model_path, kill_switch_vpin=kill_switch_vpin)
        self._vpin_history: deque[tuple[float, float]] = deque(maxlen=_VPIN_HISTORY_MAX)  # (instant, VPIN) lus dans Redis à chaque cycle

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent polls on its own schedule inside :meth:`run`.

        Implemented (rather than omitted) only to satisfy the abstract
        base class, and raises loudly if it is ever invoked by mistake.
        """
        raise NotImplementedError("RlSignalAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Log readiness. No trading client to connect -- this agent never executes."""
        self._log.info("rl_signal_agent_ready", agent_id=self.agent_id, symbol=self._symbol)

    async def run(self) -> None:
        """Connect dependencies, then run the fetch-infer-publish loop.

        Raises:
            RuntimeError: If ``run`` is called while already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self.agent_id!r} is already running.")

        self._main_task = asyncio.current_task()
        self._install_signal_handlers()

        try:
            await self._connect_redis()
            await self.on_start()

            self._running = True
            self._log.info(
                "agent_started",
                agent_id=self.agent_id,
                symbol=self._symbol,
                poll_interval_seconds=self._poll_interval_seconds,
            )

            await asyncio.gather(self._run_polling_loop(), self._listen_for_control_commands())
        except asyncio.CancelledError:
            self._log.info("agent_run_loop_cancelled_for_shutdown", agent_id=self.agent_id)
        finally:
            self._running = False
            await self._shutdown()

    async def _run_polling_loop(self) -> None:
        """The real fetch-infer-publish loop -- a separate coroutine (not inlined in :meth:`run`) so it can run alongside :meth:`~core.base_agent.BaseAgent._listen_for_control_commands`."""
        while self._running:
            cycle_started_at = time.monotonic()
            try:
                await self._run_signal_cycle()
            except Exception:
                self._log.exception("rl_signal_cycle_failed", agent_id=self.agent_id)

            await self._write_heartbeat()

            elapsed = time.monotonic() - cycle_started_at
            remaining = max(0.0, self._poll_interval_seconds - elapsed)
            await asyncio.sleep(remaining)

    # -- Signal cycle -------------------------------------------------

    async def _run_signal_cycle(self) -> None:
        """Fetch state, run PPO inference, and publish the resulting signal -- unless ``control_state`` says not to.

        ``AgentControlState.OFF``/``SHADOW`` both suppress the real
        publish onto :data:`~core.schemas.TRADING_SIGNALS_TOPIC` (so
        :class:`~src.agents.execution.agent.ExecutionAgent`'s two-agent
        consensus can never be satisfied by an opinion this agent was
        told not to really offer) -- ``SHADOW`` additionally logs what
        would have been published, for observability; ``OFF`` does not.
        """
        try:
            market_metrics, macro_snapshots, sentiment_signals, feature_vector = (
                await self._fetch_latest_state()
            )
        except (RedisUnavailableError, ValidationError) as exc:
            self._log.warning(
                "rl_signal_state_fetch_failed", agent_id=self.agent_id, error_type=type(exc).__name__
            )
            return

        if market_metrics is None:
            self._log.debug("rl_signal_no_market_data_yet", agent_id=self.agent_id)
            return

        # Reused, not re-implemented: the exact same 5-feature vector
        # SupervisorMLModel.build_feature_vector builds for the XGBoost
        # signal agent -- both signal agents must evaluate the same
        # inputs for ExecutionAgent's consensus check to mean anything.
        features = SupervisorMLModel.build_feature_vector(
            market_metrics, macro_snapshots, sentiment_signals
        )
        vpin_features = await self._read_vpin_features()
        try:
            raw_signal_value, raw_confidence = self._policy.predict_action(features, vpin_features=vpin_features)
        except VpinUnavailableError:
            # Politique entraînée avec le VPIN mais VPIN absent, périmé ou en phase de chauffe (les deux fenêtres de la vitesse et de l'accélération ne sont pas encore remplies) :
            # aucune opinion plutôt qu'une opinion fabriquée. Le signal neutre n'ouvre rien dans le consensus.
            self._log.warning("rl_signal_vpin_unavailable_neutral", agent_id=self.agent_id, instrument=self._symbol)
            raw_signal_value, raw_confidence = 0, 0.0
        # See src.agents.supervisor.feature_confirmation: adjusts
        # confidence only (never direction) from the Feature Store's RSI
        # read, when one exists for this instrument.
        signal_value, confidence = apply_feature_confirmation(
            raw_signal_value, raw_confidence, feature_vector
        )

        signal = TradingSignal(
            timestamp=time.time(),
            agent_id=self.agent_id,
            instrument=self._symbol,
            signal=signal_value,
            confidence=confidence,
        )
        if self.control_state == AgentControlState.OFF:
            self._log.warning(
                "rl_signal_suppressed_agent_off", agent_id=self.agent_id, instrument=self._symbol,
                detail="control_state=OFF: not published to the real bus.",
            )
            return
        if self.control_state == AgentControlState.SHADOW:
            self._log.warning(
                "rl_signal_suppressed_agent_shadow",
                agent_id=self.agent_id, instrument=self._symbol, signal=signal_value, confidence=confidence,
                detail="control_state=SHADOW: this is what would have been published -- not sent to the real bus.",
            )
            return

        self._log.info(
            "rl_signal_published",
            agent_id=self.agent_id,
            instrument=self._symbol,
            signal=signal_value,
            confidence=confidence,
        )
        await self.publish(TRADING_SIGNALS_TOPIC, signal.model_dump(mode="json"))

    async def _fetch_latest_state(
        self,
    ) -> tuple[
        MarketMetrics | None, dict[str, MacroSnapshot], list[SentimentSignal], FeatureVector | None
    ]:
        """Read every sub-agent's latest published state from Redis.

        Returns:
            ``(market_metrics, macro_snapshots, sentiment_signals,
            feature_vector)``, an all-``None``/empty tuple if no Redis
            connection exists.

        Raises:
            RedisUnavailableError: If a read fails after Redis's own
                bounded internal reconnect attempts are exhausted.
            pydantic.ValidationError: If a value in Redis doesn't
                deserialize into its expected schema.
        """
        if self._redis_client is None:
            return None, {}, [], None

        raw_metrics = await self._redis_client.execute("get", f"market_metrics:{self._symbol}")
        market_metrics = MarketMetrics.model_validate_json(raw_metrics) if raw_metrics is not None else None

        macro_snapshots: dict[str, MacroSnapshot] = {}
        for indicator in _MACRO_INDICATORS:
            raw = await self._redis_client.execute("get", f"macro_snapshot:{indicator}")
            if raw is not None:
                macro_snapshots[indicator] = MacroSnapshot.model_validate_json(raw)

        sentiment_signals: list[SentimentSignal] = []
        async for key in self._redis_client.client.scan_iter(match="nlp_signal:*"):
            raw = await self._redis_client.execute("get", key)
            if raw is not None:
                sentiment_signals.append(SentimentSignal.model_validate_json(raw))

        feature_vector = await self._read_feature_vector()

        return market_metrics, macro_snapshots, sentiment_signals, feature_vector

    async def _read_vpin_features(self) -> npt.NDArray[np.float64] | None:
        """``[VPIN, vitesse, accélération]`` à cet instant, ou ``None`` (clé absente ou expirée, valeur illisible ou hors [0, 1], historique encore trop court).

        L'historique est tenu en mémoire d'un cycle à l'autre : il faut deux fenêtres (2 x ``vpin_window_s``) de relevés avant que la vitesse et
        l'accélération existent ; d'ici là la réponse est ``None``, jamais une dérivée inventée.
        """
        if self._redis_client is None:
            return None
        key = f"{_VPIN_KEY_PREFIX}{''.join(ch for ch in self._symbol.upper() if ch.isalnum())}"
        try:
            raw = await self._redis_client.execute("get", key)
        except RedisUnavailableError:
            return None
        now = time.time()
        if raw is None:
            return None  # clé absente ou expirée (le tracker est arrêté ou coupé) : le niveau de CE cycle est inconnu, on ne ressert pas le dernier relevé comme s'il était frais
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = math.nan
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            return None  # valeur illisible ou hors de [0, 1] : ignorée, jamais ajoutée à l'historique
        self._vpin_history.append((now, value))
        times = np.array([t for t, _ in self._vpin_history], dtype=np.float64)
        values = np.array([v for _, v in self._vpin_history], dtype=np.float64)
        derived = vpin_derivatives(np.array([now]), times, values, window_s=self._policy.vpin_window_s, max_age_s=_VPIN_MAX_AGE_S)[0]
        return derived if bool(np.isfinite(derived).all()) else None

    async def _read_feature_vector(self) -> FeatureVector | None:
        """Read the traded symbol's latest :class:`~core.schemas.FeatureVector`, if any.

        ``None`` (key absent, expired, or malformed) is a normal outcome
        -- see :mod:`~src.agents.supervisor.feature_confirmation`.
        """
        assert self._redis_client is not None
        raw = await self._redis_client.execute("get", f"{FEATURE_VECTOR_REDIS_KEY_PREFIX}{self._symbol}")
        if raw is None:
            return None
        try:
            return FeatureVector.model_validate_json(raw)
        except ValidationError:
            self._log.warning("rl_signal_feature_vector_malformed", agent_id=self.agent_id)
            return None

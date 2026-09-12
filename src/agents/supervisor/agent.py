"""XGBoost signal micro-agent: signal fusion, ML inference, and a local Kill Switch.

:class:`XgboostSignalAgent` (module kept at its original ``supervisor``
path for historical continuity, even though it no longer supervises
execution -- see below) fuses the other agents' latest published state
into a directional opinion on one instrument, and publishes it as a
:class:`~core.schemas.TradingSignal`. It never places, cancels, or even
knows about an order: **execution now lives entirely in
:class:`~src.agents.execution.agent.ExecutionAgent`**, the only thing in
this system authorized to touch a broker.

This split (the "Portfolio Manager" pattern) exists to prevent exactly
the failure mode a single execution-capable agent per signal source would
invite: two independent decision-makers (this one, XGBoost-based, and
:class:`~src.agents.rl_signal.agent.RlSignalAgent`, RL-based) each able to
place real orders on the same OANDA account would risk contradicting and
fighting each other. Neither can place an order at all now; only
:class:`ExecutionAgent` can, and only once it has independently
arbitrated consensus between both agents' signals.

Safety-by-default carries over even though execution moved out: this
agent's local Kill Switch (redis-latency / spread-anomaly /
consecutive-error conditions) still exists, but since it no longer holds
a trading client of its own to cancel orders on, tripping it now
escalates to the *global* Kill Switch (:func:`core.security.
trip_kill_switch`) instead -- stopping :class:`ExecutionAgent` from
acting on *anyone's* signal, not just this agent's.

**Asset-class note** (carried over, still true): this agent evaluates
OANDA forex instruments (e.g. ``"EUR_USD"``), but the market data it
reads from Redis (:class:`~core.schemas.MarketMetrics`) is still
published by :class:`~src.agents.market_data.agent.MarketDataAgent`,
which watches a *crypto* order book via ``ccxt.pro``. Feeding a crypto
order book's spread/imbalance into a forex signal has no economic basis
-- a real, not-yet-addressed prerequisite for this agent to produce
economically meaningful signals in practice.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Final, Literal

import zmq.asyncio
from pydantic import ValidationError

from core.base_agent import AgentControlState, BaseAgent
from core.circuit_breaker import MarketAnomalyDetector
from core.config import AppSettings
from core.message_integrity import SignatureVerificationError, read_verified_market_metrics
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
from core.security import trip_kill_switch
from core.stale_check import staleness_reason
from src.agents.supervisor.feature_confirmation import apply_feature_confirmation
from src.agents.supervisor.ml_engine import SupervisorMLModel

# The two macro indicators SupervisorMLModel.build_feature_vector looks
# for by name (see that module). Kept here, not imported, since this is
# this agent's own Redis-key-naming concern, not the model's.
_MACRO_INDICATORS: Final[tuple[str, ...]] = ("US_10Y_YIELD", "BTC_EXCHANGE_FLOW")

_AGENT_ID: Final[str] = "supervisor-agent"


class EmergencyStopError(Exception):
    """Raised internally to unwind the current cycle once the local Kill Switch trips.

    Never escapes :meth:`XgboostSignalAgent.run`: it is caught there to
    break the main loop and fall through to the normal graceful-shutdown
    path, which the Kill Switch has already triggered via
    :meth:`~core.base_agent.BaseAgent.request_shutdown`.
    """


class SignalState:
    """One cycle's snapshot of every sub-agent's latest published state.

    Attributes:
        market_metrics: Latest order book metrics for the traded symbol,
            or ``None`` if :class:`~src.agents.market_data.agent.
            MarketDataAgent` hasn't published anything yet.
        macro_snapshots: Latest :class:`~core.schemas.MacroSnapshot` per
            indicator name currently in Redis.
        sentiment_signals: Every currently-live (TTL not yet expired)
            :class:`~core.schemas.SentimentSignal`.
        feature_vector: The traded symbol's latest :class:`~core.schemas.
            FeatureVector`, or ``None`` if :class:`~src.data_pipeline.
            feature_store.FeatureStore` has never published one for it
            (e.g. a forex instrument -- see that class's crypto-only
            default) or the key has expired. See :mod:`~src.agents.
            supervisor.feature_confirmation` for how this is used.
        redis_latency_ms: Wall-clock time this whole read took. Doubles
            as the Kill Switch's Redis-latency health check.
    """

    __slots__ = (
        "market_metrics",
        "macro_snapshots",
        "sentiment_signals",
        "feature_vector",
        "redis_latency_ms",
    )

    def __init__(
        self,
        market_metrics: MarketMetrics | None,
        macro_snapshots: dict[str, MacroSnapshot],
        sentiment_signals: list[SentimentSignal],
        feature_vector: FeatureVector | None = None,
        redis_latency_ms: float = 0.0,
    ) -> None:
        self.market_metrics = market_metrics
        self.macro_snapshots = macro_snapshots
        self.sentiment_signals = sentiment_signals
        self.feature_vector = feature_vector
        self.redis_latency_ms = redis_latency_ms


class XgboostSignalAgent(BaseAgent):
    """Fuses market/macro/NLP signals into an XGBoost-based trading opinion.

    Publishes a :class:`~core.schemas.TradingSignal` on
    :data:`~core.schemas.TRADING_SIGNALS_TOPIC` every cycle -- including a
    neutral (``0``) signal on ``HOLD``, since :class:`~src.agents.
    execution.agent.ExecutionAgent`'s consensus arbitration needs to see
    an explicit veto, not silence, to tell "no opinion yet" apart from
    "opinion not published this cycle".

    See the module docstring for the local Kill Switch's new
    global-escalation behavior.
    """

    def __init__(
        self,
        instrument: str,
        *,
        poll_interval_seconds: float = 5.0,
        decision_threshold: float = 0.1,
        redis_latency_threshold_ms: float = 1000.0,
        spread_anomaly_multiplier: float = 3.0,
        spread_history_window: int = 20,
        max_consecutive_errors: int = 3,
        flash_crash_window_seconds: float = 60.0,
        flash_crash_max_move_fraction: float = 0.05,
        max_market_metrics_age_ms: float = 500.0,
        model_path: str | Path | None = None,
        scaler_path: str | Path | None = None,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Wire up the ML model and the local Kill Switch.

        Args:
            instrument: The single OANDA instrument this instance
                evaluates (e.g. ``"EUR_USD"``). See the module
                docstring's asset-class note: the market data this agent
                reads is still crypto-flavored, a real gap this does not
                close.
            poll_interval_seconds: Time between the start of one
                evaluation cycle and the next.
            decision_threshold: How far the ML model's up-probability
                must be from ``0.5`` before this agent signals a
                direction instead of neutral (e.g. ``0.1`` means signal
                only outside ``[0.4, 0.6]``).
            redis_latency_threshold_ms: Kill Switch condition #1: trip if
                a full state read from Redis takes longer than this.
            spread_anomaly_multiplier: Kill Switch condition #2: trip if
                the current spread exceeds this many times the rolling
                average of the last ``spread_history_window`` readings.
            spread_history_window: Number of past spread readings kept
                for the anomaly baseline. The check only activates once
                this many *prior* readings exist.
            max_consecutive_errors: Kill Switch condition #3: trip once
                consecutive read/inference errors exceed this count.
            flash_crash_window_seconds: Kill Switch condition #4: lookback
                window :class:`~core.circuit_breaker.MarketAnomalyDetector`
                measures a price move over -- independent of spread (see
                that class's own docstring for why this exists alongside
                condition #2, not instead of it).
            flash_crash_max_move_fraction: Kill Switch condition #4:
                absolute fractional price move within
                ``flash_crash_window_seconds`` that trips it.
            max_market_metrics_age_ms: Kill Switch condition #5: trip if
                the latest :class:`~core.schemas.MarketMetrics` reading is
                older than this many milliseconds (see
                :mod:`core.stale_check`'s own docstring on why 500ms is
                deliberately aggressive).
            model_path: Passed through to :class:`SupervisorMLModel`; see
                its docstring for the untrained-skeleton fallback.
            scaler_path: Passed through to :class:`SupervisorMLModel`
                alongside ``model_path`` -- without this, a real
                ``model_path`` still loads and runs, but against a
                skeleton (synthetic-data-fitted) scaler, silently
                degrading prediction quality (see that class's own
                docstring). Real bug fixed here: this parameter never
                existed on this constructor at all, so no caller could
                ever supply a real fitted scaler alongside a real model.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`).
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.
        """
        super().__init__(
            _AGENT_ID,
            # Real bug, found and fixed alongside Chapter 13's "Red
            # Phone" heartbeat-staleness alert: without this,
            # tick_interval_seconds silently stayed at BaseAgent's 1.0s
            # default even though this agent's real cadence is
            # poll_interval_seconds (5.0s default) -- since run() is
            # fully overridden below and never consults
            # tick_interval_seconds for its own timing, this had no
            # visible effect until _write_heartbeat started reading it
            # for its TTL calculation, which would then have raced (a
            # 5.0s-TTL heartbeat written only every real 5.0s).
            tick_interval_seconds=poll_interval_seconds,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._symbol = instrument
        self._poll_interval_seconds = poll_interval_seconds
        self._decision_threshold = decision_threshold
        self._redis_latency_threshold_ms = redis_latency_threshold_ms
        self._spread_anomaly_multiplier = spread_anomaly_multiplier
        self._max_consecutive_errors = max_consecutive_errors

        self._ml_model = SupervisorMLModel(model_path=model_path, scaler_path=scaler_path)

        self._consecutive_error_count = 0
        self._spread_history_window = spread_history_window
        self._spread_history: deque[float] = deque(maxlen=spread_history_window)
        self._emergency_mode = False

        self._anomaly_detector = MarketAnomalyDetector(
            window_seconds=flash_crash_window_seconds, max_move_fraction=flash_crash_max_move_fraction
        )
        self._max_market_metrics_age_ms = max_market_metrics_age_ms

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent polls on its own schedule inside :meth:`run`.

        :meth:`run` is fully overridden with a fetch-decide-publish loop
        instead of :class:`~core.base_agent.BaseAgent`'s tick loop, so
        this is never called. Implemented (rather than omitted) only to
        satisfy the abstract base class, and raises loudly if it is ever
        invoked by mistake.
        """
        raise NotImplementedError("XgboostSignalAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Log readiness. No trading client to connect -- this agent never executes."""
        self._log.info(
            "xgboost_signal_agent_ready",
            agent_id=self.agent_id,
            symbol=self._symbol,
        )

    async def run(self) -> None:
        """Connect dependencies, then run the fetch-decide-publish loop.

        Mirrors :meth:`BaseAgent.run`'s lifecycle (signal handling, Redis
        connection, cancellation-safe shutdown). Every iteration first
        checks the local Kill Switch (see :meth:`_check_kill_switch`);
        tripping it ends the loop and falls through to the normal
        shutdown sequence, which the Kill Switch has already put in
        motion via :meth:`~core.base_agent.BaseAgent.request_shutdown`.

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
        """The real fetch-decide-publish loop -- a separate coroutine (not inlined in :meth:`run`) so it can run alongside :meth:`~core.base_agent.BaseAgent._listen_for_control_commands`."""
        while self._running:
            cycle_started_at = time.monotonic()
            try:
                await self._run_signal_cycle()
            except EmergencyStopError:
                break
            except Exception:
                self._consecutive_error_count += 1
                self._log.exception(
                    "xgboost_signal_cycle_failed",
                    agent_id=self.agent_id,
                    consecutive_error_count=self._consecutive_error_count,
                )

            await self._write_heartbeat()

            elapsed = time.monotonic() - cycle_started_at
            remaining = max(0.0, self._poll_interval_seconds - elapsed)
            await asyncio.sleep(remaining)

    # -- Signal cycle -------------------------------------------------

    async def _run_signal_cycle(self) -> None:
        """Fetch state, enforce the Kill Switch, then decide and publish -- unless ``control_state`` says not to.

        ``AgentControlState.OFF``/``SHADOW`` both suppress the real
        publish onto :data:`~core.schemas.TRADING_SIGNALS_TOPIC` -- see
        the identical real reasoning in :meth:`~src.agents.rl_signal.
        agent.RlSignalAgent._run_signal_cycle`'s own docstring.

        Raises:
            EmergencyStopError: If the Kill Switch trips (the global
                Kill Switch has already been tripped -- see
                :meth:`_trigger_kill_switch` -- by the time this raises;
                it exists purely to unwind this agent's own cycle).
        """
        try:
            state = await self._fetch_latest_state()
        except (RedisUnavailableError, ValidationError, SignatureVerificationError) as exc:
            self._consecutive_error_count += 1
            self._log.warning(
                "xgboost_signal_state_fetch_failed",
                agent_id=self.agent_id,
                error_type=type(exc).__name__,
                consecutive_error_count=self._consecutive_error_count,
            )
            await self._check_kill_switch(SignalState(None, {}, [], redis_latency_ms=float("inf")))
            return

        await self._check_kill_switch(state)

        # Reaching here means every Kill Switch condition passed.
        self._consecutive_error_count = 0

        if state.market_metrics is None:
            self._log.debug("xgboost_signal_no_market_data_yet", agent_id=self.agent_id)
            return

        signal_value, confidence = self._make_decision(state)
        signal = TradingSignal(
            timestamp=time.time(),
            agent_id=self.agent_id,
            instrument=self._symbol,
            signal=signal_value,
            confidence=confidence,
        )
        if self.control_state == AgentControlState.OFF:
            self._log.warning(
                "xgboost_signal_suppressed_agent_off", agent_id=self.agent_id, instrument=self._symbol,
                detail="control_state=OFF: not published to the real bus.",
            )
            return
        if self.control_state == AgentControlState.SHADOW:
            self._log.warning(
                "xgboost_signal_suppressed_agent_shadow",
                agent_id=self.agent_id, instrument=self._symbol, signal=signal_value, confidence=confidence,
                detail="control_state=SHADOW: this is what would have been published -- not sent to the real bus.",
            )
            return

        self._log.info(
            "xgboost_signal_published",
            agent_id=self.agent_id,
            instrument=self._symbol,
            signal=signal_value,
            confidence=confidence,
        )
        await self.publish(TRADING_SIGNALS_TOPIC, signal.model_dump(mode="json"))

    # -- State fetching / Redis latency measurement ----------------------

    async def _fetch_latest_state(self) -> SignalState:
        """Read every sub-agent's latest published state from Redis.

        The wall-clock time of this whole read **is** the Kill Switch's
        Redis-latency health check (see :meth:`_check_kill_switch`): a
        slow round trip here means Redis, the network to it, or the
        agents feeding it are struggling, regardless of the specific
        cause.

        Returns:
            The fetched state, with ``redis_latency_ms`` set to
            ``float("inf")`` if no Redis connection exists at all
            (treated as maximally unsafe, not a separate error path).

        Raises:
            RedisUnavailableError: If a read fails after Redis's own
                bounded internal reconnect attempts are exhausted.
            pydantic.ValidationError: If a value in Redis doesn't
                deserialize into its expected schema.
        """
        if self._redis_client is None:
            return SignalState(None, {}, [], redis_latency_ms=float("inf"))

        start = time.monotonic()
        market_metrics = await self._read_market_metrics()
        macro_snapshots = await self._read_macro_snapshots()
        sentiment_signals = await self._read_sentiment_signals()
        feature_vector = await self._read_feature_vector()
        elapsed_ms = (time.monotonic() - start) * 1000.0

        return SignalState(
            market_metrics, macro_snapshots, sentiment_signals, feature_vector, redis_latency_ms=elapsed_ms
        )

    async def _read_market_metrics(self) -> MarketMetrics | None:
        """Read + signature-verify the traded symbol's latest :class:`~core.schemas.MarketMetrics`.

        Real fix for ``tests/red_team_hacker.py``'s Attack 1 (Redis
        MarketMetrics cache poisoning): a plain ``GET`` accepted anything
        shaped like a valid reading, including a fabricated one written
        directly to this key by anyone with Redis access. See
        :func:`~core.message_integrity.read_verified_market_metrics`'s own
        docstring for the real, disclosed scope of this fix.
        """
        assert self._redis_client is not None
        return await read_verified_market_metrics(self._redis_client, self._symbol)

    async def _read_feature_vector(self) -> FeatureVector | None:
        """Read the traded symbol's latest :class:`~core.schemas.FeatureVector`, if any.

        ``None`` (key absent, expired, or malformed) is a normal outcome,
        not an error -- see :mod:`~src.agents.supervisor.
        feature_confirmation`'s docstring on why a missing vector simply
        means "behave as before this integration existed", not a fetch
        failure worth counting toward the Kill Switch's consecutive-error
        condition.
        """
        assert self._redis_client is not None
        raw = await self._redis_client.execute("get", f"{FEATURE_VECTOR_REDIS_KEY_PREFIX}{self._symbol}")
        if raw is None:
            return None
        try:
            return FeatureVector.model_validate_json(raw)
        except ValidationError:
            self._log.warning("xgboost_signal_feature_vector_malformed", agent_id=self.agent_id)
            return None

    async def _read_macro_snapshots(self) -> dict[str, MacroSnapshot]:
        """Read the latest :class:`~core.schemas.MacroSnapshot` per known indicator."""
        assert self._redis_client is not None
        snapshots: dict[str, MacroSnapshot] = {}
        for indicator in _MACRO_INDICATORS:
            raw = await self._redis_client.execute("get", f"macro_snapshot:{indicator}")
            if raw is not None:
                snapshots[indicator] = MacroSnapshot.model_validate_json(raw)
        return snapshots

    async def _read_sentiment_signals(self) -> list[SentimentSignal]:
        """Read every currently-live :class:`~core.schemas.SentimentSignal`.

        Uses ``SCAN`` (via ``scan_iter``), not ``KEYS``: ``KEYS`` walks
        the entire keyspace in one blocking call, which is fine on a toy
        dataset but is a well-known way to stall a shared Redis instance
        under real load. ``SCAN`` iterates in small non-blocking steps.
        """
        assert self._redis_client is not None
        signals: list[SentimentSignal] = []
        async for key in self._redis_client.client.scan_iter(match="nlp_signal:*"):
            raw = await self._redis_client.execute("get", key)
            if raw is not None:
                signals.append(SentimentSignal.model_validate_json(raw))
        return signals

    # -- Local Kill Switch -----------------------------------------------

    async def _check_kill_switch(self, state: SignalState) -> None:
        """Check the three local safety conditions; trip if any is met.

        Args:
            state: This cycle's fetched state.

        Raises:
            EmergencyStopError: If any condition is met. By the time
                this raises, :meth:`_trigger_kill_switch` has already
                tripped the *global* Kill Switch and requested shutdown.
        """
        reasons: list[str] = []

        if state.redis_latency_ms > self._redis_latency_threshold_ms:
            reasons.append(
                f"redis_latency_ms={state.redis_latency_ms:.1f} exceeds threshold "
                f"{self._redis_latency_threshold_ms:.1f} (possible system freeze or network outage)"
            )

        if state.market_metrics is not None:
            spread = state.market_metrics.spread
            if len(self._spread_history) >= self._spread_history_window:
                baseline = statistics.fmean(self._spread_history)
                if baseline > 0 and spread > self._spread_anomaly_multiplier * baseline:
                    reasons.append(
                        f"spread={spread:.6f} exceeds {self._spread_anomaly_multiplier}x "
                        f"the recent average {baseline:.6f} (destructive slippage risk)"
                    )
            # Recorded *after* the check so the current reading never
            # contaminates its own anomaly baseline.
            self._spread_history.append(spread)

            # Condition #4 -- flash crash, independent of spread. Real gap closed: see
            # core/circuit_breaker.py's module docstring (tests/chaos_monkey.py Chaos 1a).
            price_move_reason = self._anomaly_detector.check(state.market_metrics.mid_price)
            if price_move_reason is not None:
                reasons.append(price_move_reason)

            # Condition #5 -- stale-but-present reading. Real gap closed: see
            # core/stale_check.py's module docstring (tests/chaos_monkey.py Chaos 3).
            stale_reason = staleness_reason(state.market_metrics.timestamp, max_age_ms=self._max_market_metrics_age_ms)
            if stale_reason is not None:
                reasons.append(stale_reason)

        if self._consecutive_error_count > self._max_consecutive_errors:
            reasons.append(
                f"consecutive_error_count={self._consecutive_error_count} exceeds "
                f"{self._max_consecutive_errors}"
            )

        if reasons:
            await self._trigger_kill_switch(reasons)
            raise EmergencyStopError("; ".join(reasons))

    async def _trigger_kill_switch(self, reasons: list[str]) -> None:
        """Enter emergency mode: log critical, trip the global Kill Switch, stop the loop.

        Unlike this agent's previous (chapter 8) design, this no longer
        cancels orders directly -- it has no trading client to cancel
        them on. Instead it trips the *global* Kill Switch (see
        :mod:`core.security`), which blocks :class:`~src.agents.
        execution.agent.ExecutionAgent` from placing any new order on
        *any* agent's signal, a stronger and more appropriate reaction
        now that execution is centralized there.

        Args:
            reasons: Human-readable description of every Kill Switch
                condition that was met.
        """
        self._emergency_mode = True
        self._running = False
        self._log.critical(
            "LOCAL_KILL_SWITCH_ACTIVATED",
            agent_id=self.agent_id,
            symbol=self._symbol,
            reasons=reasons,
        )
        await trip_kill_switch(
            f"{self.agent_id} local Kill Switch tripped for {self._symbol}: {'; '.join(reasons)}"
        )

    # -- Decision ---------------------------------------------------------

    def _make_decision(self, state: SignalState) -> tuple[Literal[-1, 0, 1], float]:
        """Fuse this cycle's state into a signed signal + confidence via the ML model.

        The ML model's own call always determines direction. When this
        cycle also has a fresh :class:`~core.schemas.FeatureVector` for
        this instrument, :func:`~src.agents.supervisor.
        feature_confirmation.apply_feature_confirmation` additionally
        adjusts confidence (never direction) from its RSI momentum read
        -- see that function's docstring.

        Args:
            state: This cycle's fetched state. ``state.market_metrics``
                must not be ``None`` (checked by the caller).

        Returns:
            ``(signal, confidence)``: ``signal`` is ``1`` (buy), ``-1``
            (sell), or ``0`` (neutral); ``confidence`` is in ``[0, 1]``.
        """
        assert state.market_metrics is not None
        features = self._ml_model.build_feature_vector(
            state.market_metrics, state.macro_snapshots, state.sentiment_signals
        )
        probability_up = self._ml_model.predict_probability(features)

        if probability_up > 0.5 + self._decision_threshold:
            signal_value: Literal[-1, 0, 1] = 1
            confidence = probability_up
        elif probability_up < 0.5 - self._decision_threshold:
            signal_value = -1
            confidence = 1.0 - probability_up
        else:
            signal_value = 0
            confidence = 1.0 - 2.0 * abs(probability_up - 0.5)

        return apply_feature_confirmation(signal_value, confidence, state.feature_vector)

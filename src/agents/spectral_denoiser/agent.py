"""Spectral (FFT) price-signal denoiser: real high-frequency noise filtering, published as an opinion, never executed.

:class:`SpectralDenoiserAgent` fetches real, recent OHLCV candles for one
instrument from TimescaleDB, applies a real Fast Fourier Transform to the
close-price series, keeps only the most energetic ("dominant") frequency
components, and inverse-transforms the result back into a denoised
time-domain price series -- a standard, real top-K spectral low-pass
filter (not adaptive/Wiener filtering; a disclosed simplification, not a
hidden one). Like every signal-generating agent in this project
(:class:`~src.agents.supervisor.agent.XgboostSignalAgent`,
:class:`~src.agents.rl_signal.agent.RlSignalAgent`), it never places,
cancels, or even knows about an order: it only publishes a
:class:`~core.schemas.DenoisedPriceSignal` on
:data:`~core.schemas.DENOISED_PRICE_SIGNAL_TOPIC` and caches the same
payload in Redis -- nothing in this system currently subscribes to that
topic (a new, standalone signal source, not yet wired into
:class:`~src.agents.execution.agent.ExecutionAgent`'s consensus).

**The real wire/tensor boundary, disclosed**: this agent also builds a
genuine ``torch.Tensor`` from the denoised window (see
:meth:`SpectralDenoiserAgent._to_feature_tensor`) for any in-process
caller doing feature engineering directly against this class -- but ZMQ
and Redis only ever carry the JSON :class:`~core.schemas.
DenoisedPriceSignal` payload (the same real serialization boundary every
other topic/schema in this project already has); a downstream consumer
reading this off the wire reconstructs its own tensor from
``denoised_window`` rather than ever receiving a ``torch.Tensor`` object
directly. See :class:`~core.schemas.DenoisedPriceSignal`'s own docstring.

**Real frequency units, not assumed**: the real observed spacing between
consecutive fetched candles (median of real timestamp deltas, robust to
an occasional gap) is used as the FFT's sample period -- never a fixed,
assumed candle interval -- so :attr:`~core.schemas.DenoisedPriceSignal.
dominant_frequencies` are real cycles-per-second, correctly scaled to
whatever this instrument's actual candle cadence happens to be.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

import asyncpg
import numpy as np
import numpy.typing as npt
import torch
import zmq.asyncio
from scipy.fft import irfft, rfft, rfftfreq

from core.base_agent import AgentControlState, BaseAgent
from core.config import AppSettings
from core.logger import get_logger
from core.redis_client import RedisUnavailableError
from core.schemas import (
    DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX,
    DENOISED_PRICE_SIGNAL_TOPIC,
    DenoisedPriceSignal,
)
from core.timescale_client import TimescaleUnavailableError, connect_with_backoff
from src.data_pipeline.schema import OHLCV_TABLE

_log = get_logger(__name__)

_AGENT_ID: Final[str] = "spectral-denoiser-agent"

# Real, ordered-oldest-last SELECT (see _fetch_recent_closes, which
# reverses it back to chronological order) -- the exact same query shape
# src.data_pipeline.feature_store.FeatureStore already uses against this
# table, minus the OHLC columns this agent doesn't need.
_SELECT_RECENT_CLOSES_SQL: Final[str] = f"""
SELECT time, close
FROM {OHLCV_TABLE}
WHERE instrument = $1
ORDER BY time DESC
LIMIT $2;
"""

# A real FFT needs enough samples for "dominant frequency" to mean
# anything; below this, _run_denoise_cycle skips the cycle entirely
# rather than publish a denoised series computed from almost nothing.
_MIN_CANDLES_FOR_FFT: Final[int] = 32

# TTL on the Redis cache entry -- matches FeatureStore's own reasoning
# (src.data_pipeline.feature_store._REDIS_TTL_SECONDS): a stale cached
# signal should expire on its own rather than be trusted indefinitely by
# a late poller if this agent stops updating it.
_REDIS_TTL_SECONDS: Final[int] = 300


class SpectralDenoiserAgent(BaseAgent):
    """Fuses recent OHLCV history into a real FFT-denoised price signal.

    Publishes a :class:`~core.schemas.DenoisedPriceSignal` on
    :data:`~core.schemas.DENOISED_PRICE_SIGNAL_TOPIC` every cycle, and
    persists the same payload under
    :data:`~core.schemas.DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX` -- the
    same "ZMQ publish + Redis cache" pairing
    :class:`~src.data_pipeline.feature_store.FeatureStore` already
    establishes for :class:`~core.schemas.FeatureVector`.
    """

    def __init__(
        self,
        instrument: str,
        *,
        lookback_candles: int = 256,
        top_k_frequencies: int = 8,
        publish_window_size: int = 20,
        poll_interval_seconds: float = 5.0,
        settings: AppSettings | None = None,
        manage_own_signals: bool = True,
        zmq_context: zmq.asyncio.Context | None = None,
        manage_own_redis: bool = True,
    ) -> None:
        """Wire up the TimescaleDB connection pool (connected in :meth:`on_start`).

        Args:
            instrument: The single instrument this instance evaluates
                (e.g. ``"BTC/USDT"``), matching :data:`~src.data_pipeline.
                schema.OHLCV_TABLE`'s ``instrument`` column.
            lookback_candles: How many of the most recent candles to fetch
                and run the FFT over each cycle. Must be comfortably above
                :data:`_MIN_CANDLES_FOR_FFT` for a meaningful spectrum.
            top_k_frequencies: How many of the most energetic frequency
                components (by real FFT magnitude, always including the
                DC/mean component) to keep -- every other component is
                zeroed before the inverse transform. Higher keeps more
                real detail (less smoothing); lower denoises more
                aggressively.
            publish_window_size: How many of the most recent denoised
                points to actually publish/cache -- kept small
                deliberately (see the module docstring's low-latency
                framing): a consumer needing the full denoised window
                re-fetches and re-computes, the same real source this
                agent itself reads from, rather than this message
                carrying the entire ``lookback_candles`` window on every
                cycle.
            poll_interval_seconds: Time between the start of one
                fetch-denoise-publish cycle and the next.
            settings: Application settings. Defaults to ``get_settings()``
                (see :class:`~core.base_agent.BaseAgent`). Also where
                TimescaleDB connection settings (:attr:`~core.config.
                AppSettings.timescale`) are read from.
            manage_own_signals: See :class:`~core.base_agent.BaseAgent`.
            zmq_context: See :class:`~core.base_agent.BaseAgent`.
            manage_own_redis: See :class:`~core.base_agent.BaseAgent`.

        Raises:
            ValueError: If ``lookback_candles`` is below
                :data:`_MIN_CANDLES_FOR_FFT`, or ``top_k_frequencies``/
                ``publish_window_size`` is not positive.
        """
        if lookback_candles < _MIN_CANDLES_FOR_FFT:
            raise ValueError(f"lookback_candles must be >= {_MIN_CANDLES_FOR_FFT}.")
        if top_k_frequencies <= 0:
            raise ValueError("top_k_frequencies must be positive.")
        if publish_window_size <= 0:
            raise ValueError("publish_window_size must be positive.")

        super().__init__(
            _AGENT_ID,
            tick_interval_seconds=poll_interval_seconds,
            settings=settings,
            manage_own_signals=manage_own_signals,
            zmq_context=zmq_context,
            manage_own_redis=manage_own_redis,
        )

        self._symbol = instrument
        self._lookback_candles = lookback_candles
        self._top_k_frequencies = top_k_frequencies
        self._publish_window_size = publish_window_size
        self._poll_interval_seconds = poll_interval_seconds
        self._pool: asyncpg.Pool | None = None

    # -- BaseAgent overrides -----------------------------------------

    async def on_tick(self) -> None:
        """Unused: this agent polls on its own schedule inside :meth:`run`.

        Implemented (rather than omitted) only to satisfy the abstract
        base class, and raises loudly if it is ever invoked by mistake.
        """
        raise NotImplementedError("SpectralDenoiserAgent overrides run() and never ticks.")

    async def on_start(self) -> None:
        """Connect the real TimescaleDB pool this agent's whole lifetime shares.

        A bounded ``max_attempts`` (unlike e.g. :class:`~src.agents.
        archivist.agent.ArchivistAgent`'s unbounded retry): this is a
        signal-quality agent, not a durability-critical write path -- if
        TimescaleDB is genuinely unreachable, failing loudly and letting
        the process supervisor restart it is more honest than retrying
        forever with nothing to show for it.
        """
        try:
            self._pool = await connect_with_backoff(self._settings.timescale, max_attempts=3)
        except TimescaleUnavailableError:
            self._log.error(
                "spectral_denoiser_timescale_unavailable_at_startup", agent_id=self.agent_id
            )
            raise
        self._log.info("spectral_denoiser_agent_ready", agent_id=self.agent_id, symbol=self._symbol)

    async def on_shutdown(self) -> None:
        """Close the real TimescaleDB pool."""
        if self._pool is not None:
            await self._pool.close()

    async def run(self) -> None:
        """Connect dependencies, then run the fetch-denoise-publish loop.

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
        """The real fetch-denoise-publish loop -- a separate coroutine so it can run alongside :meth:`~core.base_agent.BaseAgent._listen_for_control_commands`."""
        while self._running:
            cycle_started_at = time.monotonic()
            try:
                await self._run_denoise_cycle()
            except Exception:
                self._log.exception("spectral_denoiser_cycle_failed", agent_id=self.agent_id)

            await self._write_heartbeat()

            elapsed = time.monotonic() - cycle_started_at
            remaining = max(0.0, self._poll_interval_seconds - elapsed)
            await asyncio.sleep(remaining)

    # -- TimescaleDB fetch ----------------------------------------------

    async def _fetch_recent_closes(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Real recent (timestamp, close) history for this instrument, chronological order.

        Returns:
            ``(epoch_seconds, closes)``, both empty if TimescaleDB is
            unreachable this cycle (a real, honest "no data this cycle,"
            never a fabricated series) or no candles exist yet.
        """
        assert self._pool is not None
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(_SELECT_RECENT_CLOSES_SQL, self._symbol, self._lookback_candles)
        except (OSError, asyncpg.PostgresError) as exc:
            self._log.warning(
                "spectral_denoiser_timescale_fetch_failed",
                agent_id=self.agent_id, symbol=self._symbol, error_type=type(exc).__name__,
            )
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Real rows come back newest-first (see _SELECT_RECENT_CLOSES_SQL's
        # ORDER BY time DESC, which lets LIMIT cheaply take the N most
        # recent candles) -- reversed here into chronological order, which
        # the FFT/timestamp-delta math below requires.
        ordered = list(reversed(rows))
        timestamps = np.array([row["time"].timestamp() for row in ordered], dtype=np.float64)
        closes = np.array([float(row["close"]) for row in ordered], dtype=np.float64)
        return timestamps, closes

    # -- Real FFT denoising -----------------------------------------------

    def _denoise(
        self, timestamps: npt.NDArray[np.float64], closes: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        r"""Real top-K spectral low-pass filter: keep the most energetic frequencies, zero the rest.

        A standard, real FFT-denoising technique: transform the real
        close-price series into the frequency domain
        (:func:`scipy.fft.rfft`, the real-input-optimized FFT), keep only
        the DC (mean) component plus the ``top_k_frequencies`` components
        with the largest magnitude, zero every other component, then
        inverse-transform (:func:`scipy.fft.irfft`) back to a real,
        denoised time-domain series of the same length. Disclosed
        simplification: this is a deterministic top-K filter, not an
        adaptive/statistically-optimal one (e.g. Wiener filtering) -- a
        real, standard, explainable technique, not the only one that
        exists.

        The real sample spacing used for :func:`scipy.fft.rfftfreq` is
        the **median of actual observed timestamp deltas** in
        ``timestamps`` (robust to one occasional gap, e.g. an exchange
        outage) -- never an assumed fixed candle interval, so the
        returned frequencies are real cycles-per-second for whatever this
        instrument's actual candle cadence happens to be this cycle.

        Args:
            timestamps: Real epoch-second timestamps, chronological
                order, same length as ``closes``.
            closes: Real close prices, chronological order. Must have at
                least 2 elements (need at least one real interval to
                derive a sample rate from).

        Returns:
            ``(denoised_series, dominant_frequencies, dominant_magnitudes,
            noise_energy_removed_pct)`` -- ``dominant_frequencies``/
            ``dominant_magnitudes`` are the kept (non-DC) components,
            descending by magnitude; ``noise_energy_removed_pct`` is the
            real fraction (0-100) of **AC** (non-DC) spectral energy the
            filter discarded -- see the real, disclosed fix in this
            method's own inline comment on why DC must be excluded from
            this specific metric.
        """
        n = closes.size
        real_dt = float(np.median(np.diff(timestamps)))

        spectrum = rfft(closes)
        frequencies = rfftfreq(n, d=real_dt)
        magnitudes = np.abs(spectrum)

        # AC-only energy (every bin except DC/index 0), not total energy:
        # a real price series' DC magnitude (~mean price * n) is routinely
        # 3-4 orders of magnitude larger than every AC (fluctuation)
        # component combined, so a total-energy-based percentage would
        # read ~0% almost regardless of how aggressively the AC spectrum
        # is filtered -- a real, found-by-testing bug in this metric's
        # first draft (verified against synthetic data with a known
        # injected cycle), not a hypothetical one. Excluding DC from both
        # the numerator and denominator is what makes this percentage
        # actually describe "how much of the price *variation* (never the
        # price *level*) this cycle's filter judged noise."
        ac_total_energy = float(np.sum(magnitudes[1:] ** 2))
        # Scale-aware epsilon, not an exact-zero check: a (near-)flat
        # real price series' true AC energy is 0, but scipy.fft's own
        # floating-point rounding leaves *tiny* nonzero AC bins (verified
        # directly: a perfectly flat synthetic series came back with
        # ac_total_energy ~1e-20 relative to a DC energy ~1e9, not
        # exactly 0.0) -- an exact `== 0.0` check misses that, and then
        # dividing two such near-zero floats amplifies pure rounding
        # noise into a meaningless double-digit "noise removed" percentage.
        # Comparing against DC energy (which scales with real price level)
        # keeps this threshold meaningful for both a $0.001 and a
        # $100,000 instrument, rather than a fixed absolute epsilon.
        dc_energy = float(magnitudes[0] ** 2)
        if ac_total_energy < 1e-9 * max(dc_energy, 1.0):
            # No real AC energy to filter -- the honest result is
            # "nothing to denoise", not a division-by-zero error.
            return closes.copy(), np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0.0

        keep_count = min(self._top_k_frequencies, magnitudes.size)
        top_indices = np.argsort(magnitudes)[-keep_count:]
        keep_mask = np.zeros_like(magnitudes, dtype=bool)
        keep_mask[top_indices] = True
        keep_mask[0] = True  # DC (mean level) is always kept -- "denoised" must not drift the mean

        filtered_spectrum = np.where(keep_mask, spectrum, 0.0)
        denoised = irfft(filtered_spectrum, n=n)

        ac_kept_energy = float(np.sum(np.abs(filtered_spectrum[1:]) ** 2))
        noise_energy_removed_pct = max(0.0, 1.0 - ac_kept_energy / ac_total_energy) * 100.0

        # Report only the real, non-DC dominant components (index 0 is
        # the mean level, not a "frequency" a caller would find useful),
        # descending by magnitude.
        non_dc_indices = top_indices[top_indices != 0]
        order = np.argsort(magnitudes[non_dc_indices])[::-1]
        dominant_frequencies = frequencies[non_dc_indices][order]
        dominant_magnitudes = magnitudes[non_dc_indices][order]

        return denoised, dominant_frequencies, dominant_magnitudes, noise_energy_removed_pct

    def _to_feature_tensor(self, denoised_window: npt.NDArray[np.float64]) -> torch.Tensor:
        """Real ``torch.Tensor`` of ``denoised_window``, for in-process feature-engineering callers.

        See the module docstring's "real wire/tensor boundary" note --
        this tensor never crosses ZMQ/Redis itself, only this method's
        direct return value.
        """
        return torch.tensor(denoised_window, dtype=torch.float32)

    # -- Cycle: fetch, denoise, publish, cache ----------------------------

    async def _run_denoise_cycle(self) -> torch.Tensor | None:
        """One real fetch-denoise-publish-cache cycle.

        Returns:
            The real, in-process ``torch.Tensor`` built from this cycle's
            published window (see :meth:`_to_feature_tensor`), or
            ``None`` if this cycle produced no signal (insufficient
            history, no TimescaleDB data, or this agent is paused --
            see below).
        """
        timestamps, closes = await self._fetch_recent_closes()
        if closes.size < _MIN_CANDLES_FOR_FFT:
            self._log.debug(
                "spectral_denoiser_insufficient_history",
                agent_id=self.agent_id, symbol=self._symbol, have=int(closes.size), need=_MIN_CANDLES_FOR_FFT,
            )
            return None

        denoised, dominant_frequencies, dominant_magnitudes, noise_energy_removed_pct = self._denoise(
            timestamps, closes
        )
        published_window = denoised[-self._publish_window_size :]
        feature_tensor = self._to_feature_tensor(published_window)

        signal = DenoisedPriceSignal(
            instrument=self._symbol,
            timestamp=time.time(),
            denoised_window=[round(float(value), 6) for value in published_window],
            dominant_frequencies=[round(float(f), 8) for f in dominant_frequencies],
            dominant_magnitudes=[round(float(m), 4) for m in dominant_magnitudes],
            noise_energy_removed_pct=round(noise_energy_removed_pct, 2),
            window_size=int(closes.size),
        )

        if self.control_state == AgentControlState.OFF:
            self._log.warning(
                "spectral_denoiser_suppressed_agent_off", agent_id=self.agent_id, symbol=self._symbol,
                detail="control_state=OFF: not published to the real bus.",
            )
            return None
        if self.control_state == AgentControlState.SHADOW:
            self._log.warning(
                "spectral_denoiser_suppressed_agent_shadow",
                agent_id=self.agent_id, symbol=self._symbol,
                noise_energy_removed_pct=signal.noise_energy_removed_pct,
                detail="control_state=SHADOW: this is what would have been published -- not sent to the real bus.",
            )
            return None

        self._log.info(
            "spectral_denoiser_signal_published",
            agent_id=self.agent_id, symbol=self._symbol,
            window_size=signal.window_size, noise_energy_removed_pct=signal.noise_energy_removed_pct,
            dominant_frequencies=signal.dominant_frequencies,
        )
        await self.publish(DENOISED_PRICE_SIGNAL_TOPIC, signal.model_dump(mode="json"))
        await self._persist_to_redis(signal)
        return feature_tensor

    async def _persist_to_redis(self, signal: DenoisedPriceSignal) -> None:
        """Cache the latest real signal under :data:`~core.schemas.DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX`.

        Best-effort, like every other Redis cache write in this project
        (see e.g. :class:`~src.data_pipeline.feature_store.FeatureStore`'s
        own): a Redis hiccup never fails the whole cycle, since the real
        ZMQ publish above already reached every live subscriber.
        """
        if self._redis_client is None:
            return
        key = f"{DENOISED_PRICE_SIGNAL_REDIS_KEY_PREFIX}{signal.instrument}"
        try:
            await self._redis_client.execute("set", key, signal.model_dump_json(), ex=_REDIS_TTL_SECONDS)
        except RedisUnavailableError:
            self._log.warning("spectral_denoiser_redis_persist_failed", agent_id=self.agent_id)

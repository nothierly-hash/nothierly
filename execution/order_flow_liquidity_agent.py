"""Real, legitimate order-flow-based liquidity-provision signal.

**A necessary correction to the brief, made explicitly rather than
silently complied with -- same discipline as this module's sibling,
:mod:`execution.predator_honeypot` (read that module's own docstring
first if you haven't).** The brief that motivated this file describes
detecting a large market participant "absorbing all the liquidity on
one side of the book" and then having this system's own agent "place
itself intelligently in front of them" to resell moments later --
explicitly labelled, in the same breath, "legal algorithmic front-
running."

That label needs to be pinned down precisely, because "front-running"
already has one narrow, specific, real illegal meaning, and a much
broader, legal one this module actually implements:

- **Illegal front-running** (FINRA Rule 5270, IIROC UMIR 6.3, and the
  fiduciary-breach line of cases under SEC Rule 10b-5) is a
  broker-dealer or fiduciary trading ahead of a **specific,
  identifiable, non-public client order** it has a duty to handle,
  using knowledge of *that order* to profit before executing it. This
  system has no brokerage relationship with any third party and never
  receives, stores, or infers anything about any specific counterparty's
  individual pending order -- there is no non-public order here to
  front-run, categorically, not just as a matter of self-restraint.
- **What this module actually does** -- infer a real, aggregate,
  directional pressure from the exact same real, already-public
  ``order_book_imbalance`` signal Gate 7 (:mod:`src.agents.execution.
  agent`) already reads off every live tick, and recommend tilting this
  system's *own* discretionary sizing toward the side real flow is
  pushing -- is the same "order-flow-based" or "flow-following"
  category of strategy long documented in the market-microstructure
  literature (order-flow imbalance as a short-horizon price-pressure
  predictor is a standard, published result, not a novel loophole) and
  practiced openly by market makers and CTAs. It reads only the public
  book; it never places a deceptive order of its own (no layering, no
  spoofing, no wash trades) to manufacture that pressure -- the same
  "detection only, no bait" line :mod:`execution.predator_honeypot`
  already draws for the defensive side of this same signal.

**Relationship to :mod:`execution.predator_honeypot`**: that module
answers "toxic flow was detected -- how do we protect our own resting
liquidity from it?" (purely defensive, real; only ever shrinks size or
switches to Post-Only). This module answers a different, real
question: "sustained, real, one-sided public flow was detected -- does
our own discretionary order deserve a size tilt in that direction?"
(opportunistic; only ever recommends, never places an order itself,
same "decision logic here, execution wiring is the caller's job"
pattern every other real signal in this codebase follows).

**Real math, reusing already-verified primitives, not new ones**: the
per-side "buy pressure" and "sell pressure" intensities are each a real
:func:`core.quant_math.hawkes_intensity` evaluation (the exact same,
already-verified self-exciting process Gate 7's own Hawkes cascade
uses) fed only the ticks whose real, signed ``order_book_imbalance``
favored that side, weighted by the real magnitude of that imbalance. A
sustained run of same-direction ticks pushes that side's intensity
sharply above the other's, exactly the "cascade" a burst of same-
direction order arrivals really is; a single stale/outlying tick decays
back out within seconds, same as Gate 7's own cascade detection. No new
quant primitive was needed -- this is one more legitimate application
of a formula this codebase had already built and verified.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Final, Literal

import numpy as np
import numpy.typing as npt

from core.logger import get_logger
from core.quant_math import hawkes_intensity

_log = get_logger(__name__)

_DEFAULT_BASELINE_INTENSITY: Final[float] = 0.1
_DEFAULT_DECAY_RATE: Final[float] = 0.5
_DEFAULT_EXCITATION_STRENGTH: Final[float] = 2.0
_DEFAULT_HISTORY_MAXLEN: Final[int] = 200
_DEFAULT_MIN_EVENTS_PER_SIDE: Final[int] = 3
_DEFAULT_DOMINANCE_THRESHOLD: Final[float] = 0.3
_DEFAULT_MAX_SIZE_MULTIPLIER: Final[float] = 2.0

FlowDirection = Literal["BUY", "SELL", "NEUTRAL"]


@dataclass(frozen=True)
class LiquidityProvisionRecommendation:
    """One real, current opportunistic liquidity-provision recommendation for one instrument.

    Never a claim about what any other participant will do -- only a
    real, disclosed function of this instrument's own recent, public
    order-book-imbalance history.

    Attributes:
        instrument: Which instrument this recommendation is for.
        direction: ``"BUY"``/``"SELL"`` when real flow dominance clears
            :attr:`~OrderFlowLiquidityAgent`'s configured threshold,
            ``"NEUTRAL"`` otherwise (including "not enough real events
            yet").
        dominance: Real, signed ``(buy_intensity - sell_intensity) /
            (buy_intensity + sell_intensity)``, in ``[-1, 1]``. Positive
            means real buy-side pressure dominates.
        confidence: ``abs(dominance)`` when ``direction != "NEUTRAL"``,
            else ``0.0``.
        size_multiplier: Real, ``>= 1.0`` multiplier a caller *may*
            choose to apply to its own normal order size when trading
            *with* ``direction`` -- ``1.0`` (no tilt) at
            ``confidence == 0``, rising toward the configured cap as
            confidence rises. This class never applies it itself.
        buy_intensity: Real, current Hawkes intensity of same-direction
            (buy-favoring) ticks.
        sell_intensity: Real, current Hawkes intensity of
            sell-favoring ticks.
        reason: Real, human-readable justification citing the actual
            numbers behind this recommendation -- never a canned string.
    """

    instrument: str
    direction: FlowDirection
    dominance: float
    confidence: float
    size_multiplier: float
    buy_intensity: float
    sell_intensity: float
    reason: str


class OrderFlowLiquidityAgent:
    """Real, per-instrument, opportunistic order-flow liquidity-provision signal. See module docstring.

    Args:
        baseline_intensity: Shared :math:`\\mu` for both sides' Hawkes
            evaluation -- see :func:`core.quant_math.hawkes_intensity`.
            Matches Gate 7's own tuning by deliberate default choice
            (an already-tested magnitude for this exact tick-arrival
            process), not re-derived independently.
        decay_rate: Shared :math:`\\beta`. Same default-matching
            reasoning as ``baseline_intensity``.
        excitation_strength: Shared :math:`\\alpha`. Same reasoning.
        history_maxlen: Real, bounded per-side, per-instrument event
            history length -- prevents unbounded memory growth over a
            long-running process, same convention as
            :class:`~src.agents.execution.agent.ExecutionAgent`'s own
            ``_tick_events``.
        min_events_per_side: A real recommendation is only ever
            non-neutral once at least one side has observed this many
            real ticks -- guards against a confident-looking dominance
            ratio computed from one or two noisy early observations.
        dominance_threshold: Minimum ``abs(dominance)`` required before
            recommending a real, non-neutral direction. Must be in
            ``(0, 1)``.
        max_size_multiplier: The real cap :attr:`LiquidityProvisionRecommendation.size_multiplier`
            approaches as confidence approaches ``1.0``. Must be
            ``>= 1.0``.

    Raises:
        ValueError: If ``dominance_threshold`` is not in ``(0, 1)`` or
            ``max_size_multiplier < 1.0``.
    """

    def __init__(
        self,
        *,
        baseline_intensity: float = _DEFAULT_BASELINE_INTENSITY,
        decay_rate: float = _DEFAULT_DECAY_RATE,
        excitation_strength: float = _DEFAULT_EXCITATION_STRENGTH,
        history_maxlen: int = _DEFAULT_HISTORY_MAXLEN,
        min_events_per_side: int = _DEFAULT_MIN_EVENTS_PER_SIDE,
        dominance_threshold: float = _DEFAULT_DOMINANCE_THRESHOLD,
        max_size_multiplier: float = _DEFAULT_MAX_SIZE_MULTIPLIER,
    ) -> None:
        if not 0.0 < dominance_threshold < 1.0:
            raise ValueError("dominance_threshold must be strictly between 0 and 1.")
        if max_size_multiplier < 1.0:
            raise ValueError("max_size_multiplier must be >= 1.0.")

        self._baseline_intensity = baseline_intensity
        self._decay_rate = decay_rate
        self._excitation_strength = excitation_strength
        self._history_maxlen = history_maxlen
        self._min_events_per_side = min_events_per_side
        self._dominance_threshold = dominance_threshold
        self._max_size_multiplier = max_size_multiplier
        self._buy_events: dict[str, Deque[tuple[float, float]]] = {}
        self._sell_events: dict[str, Deque[tuple[float, float]]] = {}

    def on_tick(self, instrument: str, order_book_imbalance: float, *, timestamp: float | None = None) -> None:
        """Feed one real, public tick's signed order-book imbalance.

        Args:
            instrument: Which instrument this reading is for.
            order_book_imbalance: Real, signed value in ``[-1, 1]`` --
                the exact same field as
                :attr:`core.schemas.MarketMetrics.order_book_imbalance`
                (positive = more real bid-side volume, negative = more
                real ask-side volume). A reading of exactly ``0.0``
                (perfectly balanced) contributes to neither side --
                genuinely no directional pressure to record, not an
                omission.
            timestamp: Real event time (Unix epoch seconds). Defaults
                to ``time.time()`` -- real wall-clock receipt time,
                matching :class:`~src.agents.execution.agent.
                ExecutionAgent`'s own ``_record_tick_event`` convention.

        Raises:
            ValueError: If ``order_book_imbalance`` is outside ``[-1, 1]``.
        """
        if not -1.0 <= order_book_imbalance <= 1.0:
            raise ValueError("order_book_imbalance must be within [-1, 1].")

        ts = time.time() if timestamp is None else timestamp
        if order_book_imbalance > 0:
            history = self._buy_events.setdefault(instrument, deque(maxlen=self._history_maxlen))
            history.append((ts, order_book_imbalance))
        elif order_book_imbalance < 0:
            history = self._sell_events.setdefault(instrument, deque(maxlen=self._history_maxlen))
            history.append((ts, -order_book_imbalance))

    def recommendation_for(self, instrument: str, *, at_time: float | None = None) -> LiquidityProvisionRecommendation:
        """The real, current recommendation for ``instrument``, from every tick fed so far via :meth:`on_tick`.

        Args:
            instrument: Which instrument to evaluate.
            at_time: Real evaluation time. Defaults to ``time.time()``.
                Both sides are always evaluated at this **same** instant
                (unlike calling :func:`~core.quant_math.hawkes_intensity`
                independently per side with its own default, which would
                evaluate each side at its own latest event time and make
                the two intensities not actually comparable).

        Returns:
            The real :class:`LiquidityProvisionRecommendation`.
        """
        evaluation_time = time.time() if at_time is None else at_time
        buy_history = self._buy_events.get(instrument, ())
        sell_history = self._sell_events.get(instrument, ())

        buy_intensity = self._intensity(buy_history, evaluation_time)
        sell_intensity = self._intensity(sell_history, evaluation_time)
        total = buy_intensity + sell_intensity
        dominance = (buy_intensity - sell_intensity) / total if total > 0 else 0.0

        enough_events = (
            len(buy_history) >= self._min_events_per_side or len(sell_history) >= self._min_events_per_side
        )
        if not enough_events or abs(dominance) < self._dominance_threshold:
            return LiquidityProvisionRecommendation(
                instrument=instrument, direction="NEUTRAL", dominance=round(dominance, 6), confidence=0.0,
                size_multiplier=1.0, buy_intensity=round(buy_intensity, 6), sell_intensity=round(sell_intensity, 6),
                reason=(
                    f"Pas de dominance de flux publique suffisante sur {instrument} "
                    f"(dominance={dominance:.3f}, seuil={self._dominance_threshold:.2f}) -- posture neutre."
                ),
            )

        direction: FlowDirection = "BUY" if dominance > 0 else "SELL"
        confidence = abs(dominance)
        size_multiplier = 1.0 + confidence * (self._max_size_multiplier - 1.0)
        return LiquidityProvisionRecommendation(
            instrument=instrument, direction=direction, dominance=round(dominance, 6), confidence=round(confidence, 6),
            size_multiplier=round(size_multiplier, 4), buy_intensity=round(buy_intensity, 6),
            sell_intensity=round(sell_intensity, 6),
            reason=(
                f"Pression de flux publique {direction} detectee sur {instrument} "
                f"(dominance={dominance:.3f}, buy_intensity={buy_intensity:.4f}, sell_intensity={sell_intensity:.4f}) "
                "-- signal derive uniquement de order_book_imbalance public ; aucune information non-publique, "
                "aucun ordre specifique d'un tiers connu (voir le docstring du module)."
            ),
        )

    def _intensity(self, history: Deque[tuple[float, float]] | tuple[()], at_time: float) -> float:
        if not history:
            return self._baseline_intensity
        timestamps: npt.NDArray[np.float64] = np.array([ts for ts, _ in history], dtype=np.float64)
        weights: npt.NDArray[np.float64] = np.array([w for _, w in history], dtype=np.float64)
        return hawkes_intensity(
            timestamps, baseline_intensity=self._baseline_intensity, decay_rate=self._decay_rate,
            excitation_strength=self._excitation_strength, event_weights=weights, at_time=at_time,
        )

"""Chapter 13, Étape 5.9, Pilier 1: stateful behavioral-bias detection for B2B clients.

**A real, load-bearing gap in the input data, stated upfront**: the
brief's three detectors need three real signals this project's data
model does not produce today, and this module does not pretend
otherwise:

1. ``detect_revenge_trading`` needs a real **realized loss event** per
   client action. This project's only durable, client-indexed
   record -- ``client_fills`` (:mod:`src.execution.billing_schema`,
   Chapter 13 Étape 5.7) -- logs *execution slices of one macro order*
   (Shadow Mode TWAP/Iceberg children), never a directional round-trip
   trade with a realized P&L: Nothierly's Execution-as-a-Service model
   never takes a client-facing position at all (see
   ``src/execution/reporting_engine.py``'s own module docstring on the
   same point, one chapter earlier). A client's real trade-level P&L
   is not observable from anywhere in this codebase.
2. ``detect_stop_loss_creep`` needs real stop-loss *order* tracking.
   ``api_middleware/gateway.py``'s ``MacroOrderRequest`` has no
   stop-loss field at all today -- only ``execution_algo``.
3. ``detect_overtrading`` is the one detector this project's real data
   genuinely supports already: order *timestamps* per client are real
   and durable (``client_fills.time``, or the gateway's own
   ``order_status:{id}`` records).

Rather than silently fabricate loss/stop-loss data to make all three
"work", this module defines the real, honest input contract each
detector actually needs (:class:`ClientAction`), implements the
detection algorithms correctly and rigorously against that contract,
and is explicit that wiring a real producer for ``realized_pnl_usd``/
``stop_loss_price`` (extending ``MacroOrderRequest`` with an optional
stop-loss field, and giving Nothierly real visibility into a client's
own P&L, which the current architecture deliberately does not have)
is real, separate future work -- the same class of gap this project has
flagged, not silently papered over, at every prior chapter.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Deque, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger

_log = get_logger(__name__)


class ActionType(str, Enum):
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    STOP_LOSS_MOVED = "STOP_LOSS_MOVED"


class ClientAction(BaseModel):
    """One real, timestamped client action -- the honest input contract every detector below is defined against.

    Not every field is populated by any real producer in this codebase
    today -- see module docstring. A caller integrating this for real
    against ``client_fills`` can populate ``timestamp``/``instrument``/
    ``quantity``/``action_type`` genuinely; ``realized_pnl_usd`` and
    ``stop_loss_price`` require real upstream work this module does not
    claim to have done.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str = Field(min_length=1)
    timestamp: datetime
    action_type: ActionType
    instrument: str
    quantity: float = Field(gt=0)
    realized_pnl_usd: float | None = Field(default=None, description="None unless a real per-trade P&L source exists -- see module docstring.")
    stop_loss_price: float | None = Field(default=None, description="Only present for STOP_LOSS_MOVED actions, and only if the caller's order model actually has a stop-loss concept -- see module docstring.")
    side: Literal["buy", "sell"] = "buy"


class BiasDetectionResult(BaseModel):
    """One detector's real, structured verdict -- always returned, even when the verdict is "not enough data"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bias_name: str
    detected: bool
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)
    insufficient_data: bool = Field(default=False, description="True when the real input this detector needs (see module docstring) wasn't present in the action history at all -- distinct from 'checked and found nothing'.")


class TiltReport(BaseModel):
    """The engine's real, structured output -- one per profiling run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str
    generated_at: datetime
    tilt_score: float = Field(ge=0, le=1, description="Weighted real combination of every detector's confidence -- see _compute_tilt_score.")
    detections: list[BiasDetectionResult]
    recommended_action: str


# -- Real, tunable thresholds -- kept as module constants, not magic literals -----
_REVENGE_TRADE_WINDOW_SECONDS: Final[float] = 60.0
_REVENGE_TRADE_SIZE_INCREASE_THRESHOLD: Final[float] = 1.5  # Martingale-style: next order >= 1.5x the prior size
_REVENGE_TRADE_LOSS_THRESHOLD_USD: Final[float] = 0.0  # any real realized loss counts -- see detect_revenge_trading
_STOP_LOSS_CREEP_MIN_MOVES: Final[int] = 2  # "more than 2 times", per the brief -- >2 means >= 3
_OVERTRADING_LOOKBACK_DAYS: Final[int] = 30
_OVERTRADING_ZSCORE_THRESHOLD: Final[float] = 3.0  # "statistically anormal" -- 3 sigma, the real standard threshold for a rare-event flag
_LOCK_ACCOUNT_TILT_THRESHOLD: Final[float] = 0.7
_WARN_TILT_THRESHOLD: Final[float] = 0.4
_LOCK_DURATION_MINUTES: Final[int] = 15


class BehavioralProfiler:
    """Stateful, per-client rolling action history + the three real bias detectors.

    "Stateful" in the real sense the brief asks for: one instance holds
    one client's last ``max_history`` real actions in memory
    (:class:`collections.deque`, O(1) append/evict) -- a caller feeding
    it real ``client_fills``/order-status events keeps it current; this
    class never queries a database itself (kept dependency-free and
    directly testable, matching this project's established "the
    detector is pure, the wiring is the caller's job" pattern).
    """

    def __init__(self, client_id: str, *, max_history: int = 500) -> None:
        self.client_id = client_id
        self._history: Deque[ClientAction] = deque(maxlen=max_history)

    def record_action(self, action: ClientAction) -> None:
        """Append one real action -- must belong to this profiler's own client_id."""
        if action.client_id != self.client_id:
            raise ValueError(f"Action for client_id {action.client_id!r} does not belong to this profiler ({self.client_id!r}).")
        self._history.append(action)

    @property
    def history(self) -> list[ClientAction]:
        return list(self._history)

    # -- Detector 1: revenge trading ------------------------------------

    def detect_revenge_trading(self) -> BiasDetectionResult:
        """A real order within :data:`_REVENGE_TRADE_WINDOW_SECONDS` of a real realized loss, sized up (Martingale/tilt).

        Requires real ``realized_pnl_usd`` on at least one
        ``ORDER_FILLED`` action -- see module docstring on why that
        field has no real producer in this codebase today.
        """
        fills = [a for a in self._history if a.action_type == ActionType.ORDER_FILLED]
        if not any(a.realized_pnl_usd is not None for a in fills):
            return BiasDetectionResult(bias_name="revenge_trading", detected=False, confidence=0.0, insufficient_data=True)

        evidence: list[str] = []
        for i in range(len(fills) - 1):
            loss_action = fills[i]
            next_action = fills[i + 1]
            if loss_action.realized_pnl_usd is None or loss_action.realized_pnl_usd >= _REVENGE_TRADE_LOSS_THRESHOLD_USD:
                continue
            gap_seconds = (next_action.timestamp - loss_action.timestamp).total_seconds()
            if gap_seconds < 0 or gap_seconds > _REVENGE_TRADE_WINDOW_SECONDS:
                continue
            if next_action.quantity < loss_action.quantity * _REVENGE_TRADE_SIZE_INCREASE_THRESHOLD:
                continue
            evidence.append(
                f"Perte réelle de ${abs(loss_action.realized_pnl_usd):.2f} sur {loss_action.instrument} à "
                f"{loss_action.timestamp.isoformat()}, suivie {gap_seconds:.0f}s plus tard d'un ordre "
                f"{next_action.quantity / loss_action.quantity:.1f}x plus gros sur {next_action.instrument}."
            )

        detected = len(evidence) > 0
        confidence = min(1.0, 0.4 + 0.2 * len(evidence)) if detected else 0.0
        return BiasDetectionResult(bias_name="revenge_trading", detected=detected, confidence=confidence, evidence=evidence)

    # -- Detector 2: stop-loss creep ------------------------------------

    def detect_stop_loss_creep(self) -> BiasDetectionResult:
        """Repeated (> :data:`_STOP_LOSS_CREEP_MIN_MOVES`) downward stop-loss moves on one long position -- refusing to accept a loss.

        Requires real ``STOP_LOSS_MOVED`` actions -- see module
        docstring on why no real order model in this codebase has a
        stop-loss concept to produce these from today.
        """
        moves = [a for a in self._history if a.action_type == ActionType.STOP_LOSS_MOVED and a.stop_loss_price is not None]
        if not moves:
            return BiasDetectionResult(bias_name="stop_loss_creep", detected=False, confidence=0.0, insufficient_data=True)

        evidence: list[str] = []
        by_instrument: dict[str, list[ClientAction]] = {}
        for move in moves:
            by_instrument.setdefault(move.instrument, []).append(move)

        max_downward_streak = 0
        for instrument, instrument_moves in by_instrument.items():
            instrument_moves.sort(key=lambda a: a.timestamp)
            downward_streak = 0
            for prev, curr in zip(instrument_moves, instrument_moves[1:]):
                assert prev.stop_loss_price is not None and curr.stop_loss_price is not None
                if curr.stop_loss_price < prev.stop_loss_price:  # long position: moving the stop DOWN = more room to lose
                    downward_streak += 1
                    evidence.append(
                        f"{instrument}: stop-loss déplacé de {prev.stop_loss_price} à {curr.stop_loss_price} "
                        f"(vers le bas, position longue) à {curr.timestamp.isoformat()}."
                    )
                else:
                    downward_streak = 0
            max_downward_streak = max(max_downward_streak, downward_streak)

        detected = max_downward_streak > _STOP_LOSS_CREEP_MIN_MOVES
        confidence = min(1.0, max_downward_streak / (_STOP_LOSS_CREEP_MIN_MOVES + 2)) if detected else 0.0
        return BiasDetectionResult(bias_name="stop_loss_creep", detected=detected, confidence=confidence, evidence=evidence if detected else [])

    # -- Detector 3: overtrading ------------------------------------

    def detect_overtrading(self, *, now: datetime | None = None) -> BiasDetectionResult:
        """Real z-score of today's order count vs. this client's own trailing 30-day daily average.

        The one detector whose real input (order timestamps) this
        project's data model genuinely already supports -- see module
        docstring.
        """
        now = now or datetime.now(timezone.utc)
        submissions = [a for a in self._history if a.action_type == ActionType.ORDER_SUBMITTED]
        lookback_start = now - timedelta(days=_OVERTRADING_LOOKBACK_DAYS)
        window = [a for a in submissions if lookback_start <= a.timestamp <= now]

        if len(window) < 2:
            return BiasDetectionResult(bias_name="overtrading", detected=False, confidence=0.0, insufficient_data=True)

        daily_counts: dict[str, int] = {}
        for action in window:
            day_key = action.timestamp.date().isoformat()
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        counts = list(daily_counts.values())
        today_key = now.date().isoformat()
        today_count = daily_counts.get(today_key, 0)

        # z-score against every OTHER day in the window -- today's own count must
        # not inflate the baseline it's being compared against.
        other_counts = [c for day, c in daily_counts.items() if day != today_key]
        if len(other_counts) < 2:
            return BiasDetectionResult(bias_name="overtrading", detected=False, confidence=0.0, insufficient_data=True)

        mean = sum(other_counts) / len(other_counts)
        variance = sum((c - mean) ** 2 for c in other_counts) / len(other_counts)
        std_dev = variance**0.5

        if std_dev == 0:
            # A perfectly-consistent baseline broken by *more* orders than usual is a
            # real anomaly; broken by *fewer* (even zero) is not "overtrading" -- only
            # the upward direction should ever read as an infinite z-score here.
            z_score = float("inf") if today_count > mean else 0.0
        else:
            z_score = (today_count - mean) / std_dev

        detected = z_score >= _OVERTRADING_ZSCORE_THRESHOLD
        confidence = min(1.0, z_score / (_OVERTRADING_ZSCORE_THRESHOLD * 2)) if detected else 0.0
        evidence = (
            [f"{today_count} ordres aujourd'hui vs. moyenne de {mean:.1f}/jour sur {len(other_counts)} jours (écart-type {std_dev:.2f}, z-score {z_score:.2f})"]
            if detected
            else []
        )
        return BiasDetectionResult(bias_name="overtrading", detected=detected, confidence=confidence, evidence=evidence)

    # -- Aggregate report ------------------------------------

    def generate_report(self, *, now: datetime | None = None) -> TiltReport:
        """Run all three real detectors, combine into one real Tilt_Score, and recommend a real action.

        Never raises on a single detector's failure -- one bad
        detection must not prevent the report from being generated
        (the same "one bad check must not crash the whole run"
        discipline :class:`~src.agents.alert_agent.AlertAgent.on_tick`
        already established).
        """
        detections: list[BiasDetectionResult] = []
        for detector_name, run_detector in (
            ("revenge_trading", self.detect_revenge_trading),
            ("stop_loss_creep", self.detect_stop_loss_creep),
            ("overtrading", lambda: self.detect_overtrading(now=now)),
        ):
            try:
                detections.append(run_detector())
            except Exception:
                _log.exception("behavioral_detector_failed", client_id=self.client_id, detector=detector_name)

        tilt_score = self._compute_tilt_score(detections)
        recommended_action = self._recommend_action(tilt_score)

        _log.info(
            "behavioral_report_generated", client_id=self.client_id, tilt_score=tilt_score,
            recommended_action=recommended_action, detected_biases=[d.bias_name for d in detections if d.detected],
        )
        return TiltReport(
            client_id=self.client_id, generated_at=now or datetime.now(timezone.utc), tilt_score=tilt_score,
            detections=detections, recommended_action=recommended_action,
        )

    @staticmethod
    def _compute_tilt_score(detections: list[BiasDetectionResult]) -> float:
        """Real, simple weighted average of every detector that actually had data -- never inflated by a detector that abstained."""
        scored = [d.confidence for d in detections if not d.insufficient_data]
        if not scored:
            return 0.0
        return round(sum(scored) / len(scored), 4)

    @staticmethod
    def _recommend_action(tilt_score: float) -> str:
        if tilt_score >= _LOCK_ACCOUNT_TILT_THRESHOLD:
            return f"LOCK_ACCOUNT_{_LOCK_DURATION_MINUTES}_MIN"
        if tilt_score >= _WARN_TILT_THRESHOLD:
            return "WARN_CLIENT"
        return "NO_ACTION"

"""Blue Ocean Singularity, Pillar 3: real spoofing detection + real, legitimate defensive execution.

**A necessary correction to the brief, made explicitly rather than
silently complied with -- read this before reading anything else in
this file.** The brief asks for an agent that, on detecting apparent
spoofing, deploys "ultra-aggressive Post-Only and Iceberg orders
placed just behind the spread to FORCE the adversarial HFT algo to
execute against our hidden liquidity when it tries to manipulate the
price". That is not a request for surveillance or risk management -- it
is a request for a system whose *explicit design purpose* is to lure
another market participant into an unfavorable trade using
deliberately positioned, bait-like hidden liquidity.

This is not a terminology quibble. Real market-manipulation law in
every major jurisdiction this project's own docstrings already
reference (see ``core.crossing_network``'s MiFID II/Reg ATS section)
turns on the *intent* behind order placement, not just its mechanics:

- **US**: the Commodity Exchange Act §4c(a)(5)(C) (added by Dodd-Frank
  §747) defines spoofing as bidding/offering "with the intent to
  cancel the bid or offer before execution"; SEC Rule 10b-5 and CFTC
  Reg 180.1 independently reach any "manipulative or deceptive device"
  more broadly.
- **EU**: Market Abuse Regulation (596/2014), Article 12, defines
  market manipulation to include placing orders "likely to give false
  or misleading signals" about supply, demand, or price.

A system deliberately engineered to place hidden liquidity *for the
purpose of* luring a counterparty's detected (mis)behavior into an
unfavorable trade sits close enough to "orders placed to influence
another participant's trading decision through a misleading signal"
that a real compliance/legal review would need to examine it before
any deployment -- which this deliverable is not, and cannot substitute
for. Building it silently, under a name like "predator_honeypot", and
presenting it as a working execution strategy would be irresponsible
regardless of how technically interesting the mechanics are. The
filename is kept as the brief's own working title (nothing here tries
to quietly relabel or obscure what was asked for) -- what changes is
the actual content of the file.

**A second, smaller correction, partially updated since first written**:
the brief names a ``XGBoost_Toxicity_Score`` feed to listen to -- no
such signal ever existed, or exists today. At the time this module was
first written, the closest real signal was order book imbalance (OBI);
VPIN was explicitly documented as not wired into production. That has
changed: :mod:`~src.agents.market_data.microstructure_features` now
computes real, live VPIN from a real trade tape, and
:class:`~src.agents.macro_order_execution.agent.
MacroOrderExecutionAgent` feeds it into this module's own
``external_toxicity_score`` extension point, unchanged, for real
``TOXICITY_AVOIDANCE``-tagged client orders -- see that agent's own
module docstring for the full wiring. This module still takes a real,
injected callable rather than importing a network/Redis client itself
(this project's established "pure decision logic, network/Redis wiring
is the caller's job" pattern -- see ``core.crossing_network``,
``risk.behavioral_profiler``); what changed is that a real caller now
exists.

**What this module builds instead -- the real, legitimate, and
considerably more broadly useful half of the brief**:

1. :class:`SpoofingPatternDetector` -- a real, order-lifetime-based
   toxic-flow detector. The exact heuristic the brief describes
   ("large volume added then cancelled within <10ms, repeatedly") is a
   real, legitimate, widely-used market-surveillance technique: real
   exchanges and regulators (FINRA among them) use fleeting-order-ratio
   and order-lifetime-distribution analysis for exactly this. Detecting
   manipulative-looking flow is unambiguously legitimate -- it is
   closer to what exchange surveillance systems and regulators
   themselves run than to anything adversarial.
2. :class:`AdverseFlowDefenseAgent` -- a real, standard *defensive*
   response. When sustained toxic flow is detected for an instrument,
   real market makers protect themselves with real, universally-
   accepted risk controls: reduce or pull resting size, widen the
   effective tolerance required before taking liquidity, and prefer
   Post-Only order placement (an order type that, by construction, can
   never take liquidity -- choosing it during a detected toxic period
   is declining to participate as an aggressor, not manipulation). This
   protects *Nothierly's own execution* from a detected hazard; it
   never engineers bait for anyone else's.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Callable, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.nothierly_ontology import OrderSide

_log = get_logger(__name__)

# =============================================================================
# Real order-book event model this detector consumes
# =============================================================================


class OrderEvent(BaseModel):
    """One real order-book event -- an ADD, a CANCEL, or a TRADE, for one order this detector tracks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: Literal["ADD", "CANCEL", "TRADE"]
    order_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    side: OrderSide
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    timestamp: datetime


class ToxicityLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    TOXIC = "TOXIC"


# =============================================================================
# Real detector: fleeting-large-order pattern, per instrument
# =============================================================================


@dataclass
class _TrackedOrder:
    quantity: Decimal
    side: OrderSide
    added_at: datetime


class SpoofingPatternDetector:
    """Real, order-lifetime-based toxic/spoofing-like flow detector -- see module docstring on its real legitimacy.

    Tracks every ``ADD`` until its matching ``CANCEL`` (or ``TRADE``,
    which clears tracking without counting as a fleeting cancel -- an
    order that actually executes was never "spoofed", by definition).
    A cancel counts as a real "fleeting large cancel" only if *both*
    real conditions hold: the order lived for less than
    ``fleeting_threshold_ms`` (the brief's own "<10ms" example) *and*
    its quantity met ``large_size_threshold`` -- a large order that
    rests for a full second, or a tiny one that blinks in and out, are
    each individually unremarkable; the real spoofing signature this
    detector targets is the *combination*.
    """

    def __init__(
        self,
        *,
        fleeting_threshold_ms: float = 10.0,
        large_size_threshold: Decimal = Decimal("5.0"),
        pattern_window_seconds: float = 5.0,
        pattern_count_threshold: int = 3,
    ) -> None:
        self._fleeting_threshold = timedelta(milliseconds=fleeting_threshold_ms)
        self._large_size_threshold = large_size_threshold
        self._pattern_window = timedelta(seconds=pattern_window_seconds)
        self._pattern_count_threshold = pattern_count_threshold
        self._pending: dict[str, _TrackedOrder] = {}
        self._fleeting_cancel_events: dict[str, list[datetime]] = defaultdict(list)

    def on_order_added(self, event: OrderEvent) -> None:
        if event.event_type != "ADD":
            raise ValueError(f"on_order_added called with event_type={event.event_type!r}, expected 'ADD'")
        self._pending[event.order_id] = _TrackedOrder(quantity=event.quantity, side=event.side, added_at=event.timestamp)

    def on_order_traded(self, event: OrderEvent) -> None:
        """A real fill clears tracking without counting as fleeting -- an executed order was never spoofed."""
        if event.event_type != "TRADE":
            raise ValueError(f"on_order_traded called with event_type={event.event_type!r}, expected 'TRADE'")
        self._pending.pop(event.order_id, None)

    def on_order_cancelled(self, event: OrderEvent) -> ToxicityLevel:
        """Real detection step -- returns the real, current toxicity level for this instrument after this cancel."""
        if event.event_type != "CANCEL":
            raise ValueError(f"on_order_cancelled called with event_type={event.event_type!r}, expected 'CANCEL'")

        tracked = self._pending.pop(event.order_id, None)
        if tracked is None:
            return self.current_level(event.instrument)  # a cancel for an order this detector never saw ADD for -- nothing to evaluate

        lifetime = event.timestamp - tracked.added_at
        if lifetime <= self._fleeting_threshold and tracked.quantity >= self._large_size_threshold:
            events = self._fleeting_cancel_events[event.instrument]
            events.append(event.timestamp)
            cutoff = event.timestamp - self._pattern_window
            self._fleeting_cancel_events[event.instrument] = [t for t in events if t >= cutoff]
            _log.debug(
                "spoofing_pattern_fleeting_cancel_observed", instrument=event.instrument,
                lifetime_ms=lifetime.total_seconds() * 1000.0, quantity=str(tracked.quantity),
            )

        return self.current_level(event.instrument)

    def current_level(self, instrument: str) -> ToxicityLevel:
        """Real, current toxicity level -- purely a function of how many fleeting large cancels are still within the real time window."""
        count = len(self._fleeting_cancel_events.get(instrument, []))
        if count >= self._pattern_count_threshold:
            return ToxicityLevel.TOXIC
        if count > 0:
            return ToxicityLevel.ELEVATED
        return ToxicityLevel.NORMAL

    def fleeting_cancel_count(self, instrument: str) -> int:
        """Real, current count within the pattern window -- exposed for observability/dashboards, not used by the detector itself."""
        return len(self._fleeting_cancel_events.get(instrument, []))


# =============================================================================
# Real, legitimate defensive response -- protects Nothierly, never baits anyone
# =============================================================================


class DefensiveExecutionMode(str, Enum):
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    POST_ONLY_ONLY = "POST_ONLY_ONLY"


@dataclass(frozen=True)
class DefensivePosture:
    """One real, current defensive stance for one instrument -- what :class:`AdverseFlowDefenseAgent` actually recommends.

    Attributes:
        mode: The real posture level.
        max_order_size_multiplier: Real real-valued scale (``1.0`` =
            no reduction, ``0.0`` = do not add new resting size at
            all) the caller should apply to its normal order-sizing
            logic -- this class never places an order itself, matching
            this project's established "decision logic here, execution
            wiring is the caller's job" pattern.
        require_post_only: When ``True``, the caller must not submit
            any liquidity-taking (market/marketable-limit) order for
            this instrument until the posture improves -- the real,
            legitimate mechanism this module uses instead of the
            brief's "force the adversary to execute against us": simply
            declining to take liquidity during a detected toxic window.
        reason: Real, human-readable justification -- always cites the
            real detector state behind the posture, never a canned
            string independent of what was actually observed.
    """

    mode: DefensiveExecutionMode
    max_order_size_multiplier: float
    require_post_only: bool
    reason: str


ToxicityScoreProvider = Callable[[str], float | None]


class AdverseFlowDefenseAgent:
    """Real, legitimate posture controller -- turns :class:`SpoofingPatternDetector`'s real signal into a real, defensive execution stance.

    Args:
        detector: The real, primary signal source (see module
            docstring -- order-lifetime pattern detection).
        external_toxicity_score: Optional, real secondary signal --
            e.g. a caller-supplied OBI-based score read from this
            project's real ``market_metrics:{symbol}`` Redis key (see
            module docstring's correction on why no real
            "XGBoost_Toxicity_Score" exists to wire in directly).
            Expected range ``[0.0, 1.0]``; ``None`` (the default) means
            no secondary signal is available, and posture is decided
            from :attr:`detector` alone.
        external_toxicity_threshold: A real external score at or above
            this value independently escalates posture to at least
            :attr:`DefensiveExecutionMode.CAUTIOUS`, even if the
            detector itself currently reads NORMAL -- real corroborating
            evidence from a second, independent signal is treated as
            real evidence, not discarded just because it didn't come
            from the primary detector.
    """

    def __init__(
        self,
        detector: SpoofingPatternDetector,
        *,
        external_toxicity_score: ToxicityScoreProvider | None = None,
        external_toxicity_threshold: float = 0.7,
    ) -> None:
        self._detector = detector
        self._external_toxicity_score = external_toxicity_score
        self._external_toxicity_threshold = external_toxicity_threshold

    def posture_for(self, instrument: str) -> DefensivePosture:
        """The real, current defensive posture for ``instrument`` -- never places or modifies any real order itself."""
        level = self._detector.current_level(instrument)
        external_score = self._external_toxicity_score(instrument) if self._external_toxicity_score is not None else None
        external_flag = external_score is not None and external_score >= self._external_toxicity_threshold

        if level == ToxicityLevel.TOXIC:
            count = self._detector.fleeting_cancel_count(instrument)
            return DefensivePosture(
                mode=DefensiveExecutionMode.POST_ONLY_ONLY, max_order_size_multiplier=0.0, require_post_only=True,
                reason=(
                    f"Flux toxique détecté sur {instrument} : {count} annulations rapides de gros volume "
                    "dans la fenêtre récente -- passage en Post-Only exclusif, aucune nouvelle prise de "
                    "liquidité tant que le motif persiste (défense réelle, pas un piège -- voir le docstring "
                    "du module)."
                ),
            )
        if level == ToxicityLevel.ELEVATED or external_flag:
            reason_parts = []
            if level == ToxicityLevel.ELEVATED:
                reason_parts.append(f"activité d'annulation rapide élevée sur {instrument}")
            if external_flag:
                reason_parts.append(f"score de toxicité externe = {external_score:.2f} (seuil {self._external_toxicity_threshold:.2f})")
            return DefensivePosture(
                mode=DefensiveExecutionMode.CAUTIOUS, max_order_size_multiplier=0.3, require_post_only=True,
                reason=f"Posture prudente sur {instrument} : {', '.join(reason_parts)}.",
            )

        return DefensivePosture(
            mode=DefensiveExecutionMode.NORMAL, max_order_size_multiplier=1.0, require_post_only=False,
            reason="Aucune activité suspecte détectée -- posture normale.",
        )

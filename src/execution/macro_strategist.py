"""Chapter 13, Étape 5.6, Pilier 2: synthetic algorithmic orders -- TWAP and Iceberg slicing.

Splits one client "parent" macro order (the shape
``api_middleware/gateway.py``'s ``MacroOrderRequest`` already defines)
into many "child" orders, sized and timed by a real, standard
execution-algo formula -- not a random/uniform split.

**Real scope, stated upfront**: this module implements the *slicing
logic* -- deciding how big each child order is and when it should fire
-- as a pure, directly-testable component. It does not itself submit
orders to a broker or run as a live scheduled process: wiring child
orders into a real, continuously-running executor that calls
:class:`~src.execution.broker_gateway.BaseBroker` on each child's
schedule is the same kind of real, separate future work
``api_middleware/gateway.py``'s own docstring already flags for
``CLIENT_ORDERS_TOPIC`` (no "Ingestor Agent" consumes it today) --
:class:`MacroStrategist` is what such a consumer would call once
built, not a claim that it already runs live.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

OrderSide = Literal["buy", "sell"]

_DEFAULT_ICEBERG_DISPLAY_PCT: Final[float] = 0.01  # brief's own "1% of the real order" spec
_DEFAULT_TWAP_JITTER_PCT: Final[float] = 0.3  # +-30% randomization on each slice's delay


class ChildOrder(BaseModel):
    """One slice of a parent macro order -- what a real executor would eventually submit to a :class:`~src.execution.broker_gateway.BaseBroker`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    child_order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_order_id: str
    sequence: int = Field(ge=0, description="0-indexed position of this child within its parent's slicing plan.")
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    scheduled_at: datetime = Field(description="When this child should fire (TWAP) or become eligible (Iceberg, informational only).")


def slice_twap(
    *,
    parent_order_id: str,
    symbol: str,
    side: OrderSide,
    total_quantity: float,
    time_window_minutes: int,
    num_slices: int,
    jitter_pct: float = _DEFAULT_TWAP_JITTER_PCT,
    rng: random.Random | None = None,
) -> list[ChildOrder]:
    """Split ``total_quantity`` into ``num_slices`` equal-sized children, spread over ``time_window_minutes``.

    Time-Weighted Average Price: children are scheduled at roughly
    regular intervals across the window, each interval's delay
    randomized by up to ``jitter_pct`` so the resulting submission
    pattern doesn't look like a fixed-interval bot to other market
    participants (the brief's own stated goal) -- a perfectly regular
    cadence is itself a detectable signature.

    Args:
        parent_order_id: The macro order this slicing plan belongs to.
        symbol: Trading pair/instrument.
        side: ``"buy"`` or ``"sell"`` -- applied identically to every child.
        total_quantity: Total size to execute, in base-asset units.
        time_window_minutes: Total time budget to spread execution over.
        num_slices: How many child orders to split into. Must be >= 1.
        jitter_pct: Fraction of each slice's nominal interval that its
            actual delay may randomly vary by (e.g. ``0.3`` = +-30%).
        rng: Injected for deterministic tests; defaults to a fresh
            :class:`random.Random` (module-global ``random`` state is
            never touched, so this is safe to call concurrently).

    Returns:
        ``num_slices`` child orders, in scheduled order, each sized
        ``total_quantity / num_slices`` (the last slice absorbs any
        floating-point remainder so the sum exactly equals
        ``total_quantity``).

    Raises:
        ValueError: If ``num_slices < 1``, ``total_quantity <= 0``, or
            ``time_window_minutes <= 0``.
    """
    if num_slices < 1:
        raise ValueError("num_slices must be >= 1.")
    if total_quantity <= 0:
        raise ValueError("total_quantity must be > 0.")
    if time_window_minutes <= 0:
        raise ValueError("time_window_minutes must be > 0.")

    rng = rng or random.Random()
    base_slice_qty = total_quantity / num_slices
    nominal_interval = timedelta(minutes=time_window_minutes / num_slices)
    now = datetime.now(timezone.utc)

    children: list[ChildOrder] = []
    elapsed = timedelta(0)
    for i in range(num_slices):
        jitter_factor = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
        delay = nominal_interval * jitter_factor if i > 0 else timedelta(0)
        elapsed += delay
        # Last slice absorbs the float remainder so sum(quantities) == total_quantity exactly.
        quantity = total_quantity - base_slice_qty * (num_slices - 1) if i == num_slices - 1 else base_slice_qty
        children.append(
            ChildOrder(
                parent_order_id=parent_order_id,
                sequence=i,
                symbol=symbol,
                side=side,
                quantity=quantity,
                scheduled_at=now + elapsed,
            )
        )
    return children


class IcebergSlicer:
    """Stateful, fill-reactive slicer: only ever exposes ``display_pct`` of the *remaining* size at once.

    Unlike :func:`slice_twap` (a fixed, time-scheduled plan computed
    once), an iceberg order is inherently reactive: the next slice is
    only revealed once the previous one is actually filled, so it must
    be driven by a real executor calling :meth:`record_fill` as fills
    happen, not read as a flat pre-computed list.
    """

    def __init__(
        self,
        *,
        parent_order_id: str,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        display_pct: float = _DEFAULT_ICEBERG_DISPLAY_PCT,
    ) -> None:
        """Configure one iceberg order's slicing state.

        Args:
            parent_order_id: The macro order this slicer belongs to.
            symbol: Trading pair/instrument.
            side: ``"buy"`` or ``"sell"``.
            total_quantity: Total real size to execute -- never shown
                on the book all at once.
            display_pct: Fraction of ``total_quantity`` revealed per
                slice (the brief's own "1%" spec, as a tunable default).

        Raises:
            ValueError: If ``total_quantity <= 0`` or ``display_pct`` is
                not in ``(0, 1]``.
        """
        if total_quantity <= 0:
            raise ValueError("total_quantity must be > 0.")
        if not (0 < display_pct <= 1):
            raise ValueError("display_pct must be in (0, 1].")

        self._parent_order_id = parent_order_id
        self._symbol = symbol
        self._side: OrderSide = side
        self._display_quantity = total_quantity * display_pct
        self.remaining_quantity = total_quantity
        self._next_sequence = 0

    def next_slice(self) -> ChildOrder | None:
        """The next child order to show on the book, or ``None`` if the parent is fully executed.

        Does **not** reduce :attr:`remaining_quantity` -- that only
        happens once the slice actually fills (see :meth:`record_fill`),
        so calling this repeatedly without a matching fill returns the
        same slice again rather than silently consuming size that was
        never actually executed.
        """
        if self.remaining_quantity <= 0:
            return None
        quantity = min(self._display_quantity, self.remaining_quantity)
        child = ChildOrder(
            parent_order_id=self._parent_order_id,
            sequence=self._next_sequence,
            symbol=self._symbol,
            side=self._side,
            quantity=quantity,
            scheduled_at=datetime.now(timezone.utc),
        )
        self._next_sequence += 1
        return child

    def record_fill(self, filled_quantity: float) -> None:
        """Reduce the real remaining size by a real fill, clamped at zero.

        Args:
            filled_quantity: How much of the most recent
                :meth:`next_slice` actually filled.

        Raises:
            ValueError: If ``filled_quantity`` is negative.
        """
        if filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative.")
        self.remaining_quantity = max(0.0, self.remaining_quantity - filled_quantity)

    @property
    def is_exhausted(self) -> bool:
        """``True`` once every unit of the parent order has been filled."""
        return self.remaining_quantity <= 0

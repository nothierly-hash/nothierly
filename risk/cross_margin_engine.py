"""Chapter 13, Étape 5.8, Pilier 2: cross-venue margin aggregation and real risk reduction.

**Real, load-bearing corrections to the brief, stated upfront:**

1. **No "Broker MT5" integration exists in this project**, and this
   module does not invent one. This project's real, working brokers
   are exactly the three :class:`~src.execution.broker_gateway.
   BaseBroker` adapters built this chapter: crypto (any ``ccxt``
   exchange), OANDA (forex), and Interactive Brokers. This engine works
   against that real ``BaseBroker`` abstraction -- it would extend to a
   real MT5 adapter the moment one exists, without changing a line here,
   which is the entire point of coding against the ABC rather than a
   named broker.

2. **There is no "RL Agent" this engine can order to close a position.**
   Verified before writing this (and already documented in
   ``src/agents/alert_agent.py``'s own module docstring): the RL signal
   agent only ever *publishes an opinion*
   (:class:`~core.schemas.TradingSignal`); it never places or holds
   orders. The real mechanism for closing a position is calling
   :meth:`~src.execution.broker_gateway.BaseBroker.place_market_order`
   directly on the broker that actually holds it -- exactly what
   :meth:`CrossMarginEngine.enforce_margin_safety` does.

3. **The single most important correction**: the brief's own scenario
   -- "the client is losing money on Forex to the point of a global
   liquidation risk, so close some Crypto positions to free margin" --
   describes something that **cannot work as literally stated** in this
   project's real architecture, and this module does not pretend
   otherwise. OANDA and a crypto exchange are two **separate brokerage
   accounts** with no fund-transfer rail between them anywhere in this
   codebase (or, generally, in real brokerage relationships without a
   real bank wire initiated by a human). Selling BTC on Binance does
   not put one more dollar of margin into an OANDA account. What this
   engine does instead, honestly:

   - Detects real danger **per margined venue** (OANDA, IBKR -- crypto
     spot has no margin concept at all, see
     :class:`~core.nothierly_ontology.MarginSnapshot`'s own docstring)
     and reduces risk **on that same venue** by closing its own
     largest real open position -- the one thing that is actually
     possible without a fund transfer.
   - Separately reports how much real value sits in unmargined
     (spot) venues that *could* be manually liquidated and wired over
     by a human treasury operation --
     :meth:`CrossMarginEngine.compute_liquidatable_spot_value_usd` is
     informational only; this module never initiates a real bank
     transfer, and would not fabricate one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from core.nothierly_ontology import MarginSnapshot, UniversalOrder, order_from_ccxt, order_from_oanda
from src.execution.broker_gateway import BaseBroker, OrderSide

_log = get_logger(__name__)

_DEFAULT_WARNING_THRESHOLD_PCT: Final[Decimal] = Decimal(70)
_DEFAULT_CRITICAL_THRESHOLD_PCT: Final[Decimal] = Decimal(90)


class GlobalMarginReport(BaseModel):
    """One real, point-in-time snapshot of margin utilization across every connected broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    total_balance_usd: Decimal = Field(ge=0, description="Sum across every connected broker, margined or not.")
    total_margin_used_usd: Decimal = Field(ge=0, description="Sum across margined brokers only.")
    total_margin_available_usd: Decimal = Field(ge=0, description="Sum across margined brokers only.")
    global_utilization_pct: Decimal = Field(
        ge=0, description="0 if no margined broker is connected -- see per_margined_broker for the real per-venue detail."
    )
    per_broker: dict[str, MarginSnapshot]
    unmargined_broker_ids: list[str] = Field(
        default_factory=list, description="Brokers excluded from the utilization ratio -- spot venues with no real margin concept."
    )


class CrossMarginEngine:
    """Aggregates real margin state across every connected :class:`~src.execution.broker_gateway.BaseBroker`, and reduces real risk when needed.

    See module docstring for the real, honest scope of
    :meth:`enforce_margin_safety` -- it reduces risk *per venue*, it
    does not (and structurally cannot, without a real bank transfer
    this project doesn't implement) move capital between brokers.
    """

    def __init__(
        self,
        brokers: dict[str, BaseBroker],
        *,
        warning_threshold_pct: Decimal = _DEFAULT_WARNING_THRESHOLD_PCT,
        critical_threshold_pct: Decimal = _DEFAULT_CRITICAL_THRESHOLD_PCT,
    ) -> None:
        """Configure the engine over a real set of already-connected brokers.

        Args:
            brokers: ``{broker_id: connected BaseBroker}`` -- callers
                own connecting/disconnecting these (this engine never
                calls :meth:`~src.execution.broker_gateway.BaseBroker.connect`
                itself); typically built via
                :class:`~src.execution.broker_gateway.BrokerFactory`.
            warning_threshold_pct: Utilization level (0-100) at which
                :meth:`compute_global_margin_report` is worth alerting
                on, even though :meth:`enforce_margin_safety` doesn't
                act until ``critical_threshold_pct``.
            critical_threshold_pct: Utilization level at which
                :meth:`enforce_margin_safety` actually closes a real
                position.
        """
        self._brokers = brokers
        self._warning_threshold_pct = warning_threshold_pct
        self._critical_threshold_pct = critical_threshold_pct

    async def compute_global_margin_report(self) -> GlobalMarginReport:
        """Real, current margin state across every connected broker -- one real API call per broker, no caching."""
        snapshots: dict[str, MarginSnapshot] = {}
        for broker_id, broker in self._brokers.items():
            try:
                snapshots[broker_id] = await broker.get_margin_snapshot()
            except Exception:
                _log.exception("cross_margin_snapshot_failed", broker_id=broker_id)

        margined = {k: v for k, v in snapshots.items() if v.margin_used_usd is not None and v.margin_available_usd is not None}
        unmargined_ids = [k for k, v in snapshots.items() if v.margin_used_usd is None]

        total_balance = sum((v.account_balance_usd for v in snapshots.values()), Decimal(0))
        total_used = sum((v.margin_used_usd for v in margined.values() if v.margin_used_usd is not None), Decimal(0))
        total_available = sum((v.margin_available_usd for v in margined.values() if v.margin_available_usd is not None), Decimal(0))
        total_capacity = total_used + total_available
        global_pct = (total_used / total_capacity * Decimal(100)) if total_capacity > 0 else Decimal(0)

        return GlobalMarginReport(
            generated_at=datetime.now(timezone.utc),
            total_balance_usd=total_balance,
            total_margin_used_usd=total_used,
            total_margin_available_usd=total_available,
            global_utilization_pct=global_pct,
            per_broker=snapshots,
            unmargined_broker_ids=unmargined_ids,
        )

    async def compute_liquidatable_spot_value_usd(self) -> Decimal:
        """Real, informational total: unmargined (spot) holdings' current notional value across every connected broker.

        **Never automatically transferred anywhere** -- see module
        docstring. A real operator deciding to top up a margined
        account manually (sell spot, wire the proceeds) would use this
        number as a starting point, not this engine acting on it.
        """
        total = Decimal(0)
        for broker_id, broker in self._brokers.items():
            try:
                snapshot = await broker.get_margin_snapshot()
                if snapshot.margin_used_usd is not None:
                    continue  # margined venue, not spot -- excluded
                positions = await broker.get_open_positions()
            except Exception:
                _log.exception("cross_margin_liquidatable_value_check_failed", broker_id=broker_id)
                continue
            total += sum((p.notional_usd for p in positions), Decimal(0))
        return total

    async def enforce_margin_safety(self) -> list[UniversalOrder]:
        """Close each critically-margined broker's own largest position -- real orders, on the same venue only.

        Returns:
            One :class:`~core.nothierly_ontology.UniversalOrder` per
            real position actually closed. Empty if every margined
            broker is under :attr:`_critical_threshold_pct`, or if no
            margined broker is connected at all (an all-spot portfolio
            has nothing this method can act on -- see module docstring).
        """
        report = await self.compute_global_margin_report()
        closed_orders: list[UniversalOrder] = []

        for broker_id, snapshot in report.per_broker.items():
            utilization = snapshot.margin_utilization_pct
            if utilization is None or utilization < self._critical_threshold_pct:
                continue

            broker = self._brokers[broker_id]
            _log.critical(
                "cross_margin_critical_utilization_detected",
                broker_id=broker_id, utilization_pct=str(utilization), threshold_pct=str(self._critical_threshold_pct),
            )
            try:
                positions = await broker.get_open_positions()
            except Exception:
                _log.exception("cross_margin_position_lookup_failed", broker_id=broker_id)
                continue
            if not positions:
                _log.warning("cross_margin_critical_but_no_open_positions_to_close", broker_id=broker_id)
                continue

            largest = max(positions, key=lambda p: p.notional_usd)
            side: OrderSide = "sell" if largest.quantity > 0 else "buy"
            decision_timestamp = datetime.now(timezone.utc)
            try:
                raw_result = await broker.place_market_order(largest.asset.symbol, side, float(abs(largest.quantity)))
            except Exception:
                _log.exception("cross_margin_emergency_close_failed", broker_id=broker_id, symbol=largest.asset.symbol)
                continue

            # Real, honest attribution: this order is Nothierly's own risk system acting
            # autonomously on margin safety, not a specific client's investment decision --
            # see UniversalOrder's docstring on what these fields represent.
            order = (
                order_from_oanda(
                    raw_result, asset=largest.asset, algo_id="CROSS_MARGIN_EMERGENCY_CLOSE",
                    decision_timestamp_utc=decision_timestamp, regulatory_client_shortcode="NOTHIERLY_RISK_SYSTEM",
                )
                if broker_id == "oanda"
                else order_from_ccxt(
                    raw_result, broker_id=broker_id, asset=largest.asset, algo_id="CROSS_MARGIN_EMERGENCY_CLOSE",
                    decision_timestamp_utc=decision_timestamp, regulatory_client_shortcode="NOTHIERLY_RISK_SYSTEM",
                )
            )
            closed_orders.append(order)
            _log.critical(
                "cross_margin_emergency_close_executed",
                broker_id=broker_id, symbol=largest.asset.symbol, quantity=str(largest.quantity), order_id=order.order_id,
            )

        return closed_orders

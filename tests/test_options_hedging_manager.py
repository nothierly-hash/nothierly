"""Real, executable tests for OptionsHedgingManager.estimate_discrete_hedge_replication_error.

See src/agents/supervisor/options_hedging_manager.py's own "Discrete
Hedge Replication Error" docstring addendum for the real gap this
closes: core.quant_math.simulate_discrete_delta_hedge existed with zero
real callers anywhere in this project before this wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.quant_math import simulate_discrete_delta_hedge
from core.schemas import OptionContract, OptionPosition
from src.agents.supervisor.options_hedging_manager import OptionsHedgingManager


def _contract(*, option_type: str = "call", strike: float = 100.0, contract_size: float = 1.0) -> OptionContract:
    return OptionContract(
        underlying="BTC/USDT", strike=strike, expiry=datetime.now(timezone.utc) + timedelta(days=30),
        option_type=option_type, contract_size=contract_size,
    )


def _position(*, quantity: float = 1.0, **contract_kwargs: object) -> OptionPosition:
    return OptionPosition(
        contract=_contract(**contract_kwargs), quantity=quantity, entry_premium=5.0, entry_timestamp=1.0,
    )


def test_matches_the_real_quant_math_function_scaled_by_position_size() -> None:
    manager = OptionsHedgingManager(risk_free_rate=0.01)
    price_path = np.array([100.0, 101.0, 99.0, 102.0, 100.0])
    position = _position(quantity=3.0, contract_size=2.0)

    result = manager.estimate_discrete_hedge_replication_error(
        position, price_path, time_step=1.0 / 252.0, volatility=0.5,
    )

    _replicating_path, per_unit_error = simulate_discrete_delta_hedge(
        price_path, 1.0 / 252.0, 100.0, 0.01, 0.5, option_type="call",
    )
    assert result == pytest.approx(per_unit_error * 3.0 * 2.0)


def test_short_position_flips_the_sign_of_the_scaled_error() -> None:
    manager = OptionsHedgingManager(risk_free_rate=0.0)
    price_path = np.array([100.0, 105.0, 95.0, 110.0])
    long_position = _position(quantity=1.0)
    short_position = _position(quantity=-1.0)

    long_error = manager.estimate_discrete_hedge_replication_error(
        long_position, price_path, time_step=1.0 / 252.0, volatility=0.4,
    )
    short_error = manager.estimate_discrete_hedge_replication_error(
        short_position, price_path, time_step=1.0 / 252.0, volatility=0.4,
    )
    assert short_error == pytest.approx(-long_error)


def test_different_strikes_produce_different_real_replication_errors() -> None:
    """Put-call parity makes the same-strike call/put replication error coincide almost exactly
    (a real, expected property of discrete gamma-driven hedging error -- not something this test
    should assume away) -- a different strike is a genuine, independent free parameter instead."""
    manager = OptionsHedgingManager(risk_free_rate=0.03)
    price_path = np.array([100.0, 90.0, 80.0, 70.0, 90.0, 110.0])
    near_the_money = manager.estimate_discrete_hedge_replication_error(
        _position(strike=100.0), price_path, time_step=30.0 / 252.0, volatility=0.6,
    )
    deep_out_of_the_money = manager.estimate_discrete_hedge_replication_error(
        _position(strike=500.0), price_path, time_step=30.0 / 252.0, volatility=0.6,
    )
    assert near_the_money != pytest.approx(deep_out_of_the_money)


def test_rejects_too_short_a_price_path() -> None:
    manager = OptionsHedgingManager()
    with pytest.raises(ValueError, match="at least 2 elements"):
        manager.estimate_discrete_hedge_replication_error(
            _position(), np.array([100.0]), time_step=1.0 / 252.0, volatility=0.3,
        )

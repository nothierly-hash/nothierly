"""Real, executable test for SpectralDenoiserAgent._compute_cwt_dominant_period.

Ground truth: a synthetic close-price series built from a real, known
sinusoidal cycle length -- the CWT-based reading must recover a real
period close to the true one, not merely return *some* number.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
import pytest

from src.agents.spectral_denoiser.agent import SpectralDenoiserAgent


@pytest.fixture
def agent() -> Generator[SpectralDenoiserAgent, None, None]:
    instance = SpectralDenoiserAgent("BTC/USDT", manage_own_signals=False, manage_own_redis=False)
    yield instance
    # Real cleanup, not optional: BaseAgent.__init__ opens real ZMQ PUB/SUB sockets on construction
    # (see its own docstring -- "ZeroMQ socket creation/bind/connect ... happens here"). Leaving them
    # open let the test process hang indefinitely at interpreter/pytest-session teardown trying to
    # clean up an un-terminated zmq.asyncio.Context -- a real bug caught by testing this new agent
    # fixture, not present in this project's other agent tests only because they immediately replace
    # `_publisher` with a fake (see e.g. test_execution_agent_control_state.py's own fixture).
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def test_cwt_dominant_period_recovers_a_known_real_cycle_length(agent: SpectralDenoiserAgent) -> None:
    n = 500
    candle_interval_seconds = 60.0
    timestamps = np.arange(n, dtype=np.float64) * candle_interval_seconds
    true_period_candles = 20.0
    closes = 100.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / true_period_candles)

    period_seconds, power = agent._compute_cwt_dominant_period(timestamps, closes)

    assert period_seconds is not None
    assert power is not None
    assert power > 0
    true_period_seconds = true_period_candles * candle_interval_seconds
    assert period_seconds == pytest.approx(true_period_seconds, rel=0.25)


def test_cwt_dominant_period_is_none_for_too_short_a_series(agent: SpectralDenoiserAgent) -> None:
    n = 10  # quarter-window (2.5 samples) is below _CWT_MIN_PERIOD_SAMPLES (4.0)
    timestamps = np.arange(n, dtype=np.float64) * 60.0
    closes = 100.0 + np.sin(np.arange(n, dtype=np.float64))

    period_seconds, power = agent._compute_cwt_dominant_period(timestamps, closes)

    assert period_seconds is None
    assert power is None

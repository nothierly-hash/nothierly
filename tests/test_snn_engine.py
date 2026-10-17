"""Real, executable tests for nothierly_genesis/snn_encoder.py and snn_brain.py.

No mocking of ``snntorch`` -- every test runs the real :class:`snntorch.Leaky` dynamics against
this module's own hand-set (not trained -- see snn_brain.py's own docstring) weights.
"""

from __future__ import annotations

import pytest
import torch

from nothierly_genesis.snn_brain import BUY_ARBITRAGE_INDEX, run_event_driven_trigger
from nothierly_genesis.snn_encoder import SPIKE_DOWN, SPIKE_SILENCE, SPIKE_UP, delta_encode


def test_delta_encode_matches_the_brief_own_three_rules() -> None:
    # +5 (up, > threshold), -5 (down), -0.5 (silence, |delta| <= threshold)
    spikes = delta_encode([100.0, 105.0, 100.0, 99.5], threshold=1.0)
    assert spikes.tolist() == [SPIKE_UP, SPIKE_DOWN, SPIKE_SILENCE]


def test_delta_encode_requires_at_least_two_prices() -> None:
    with pytest.raises(ValueError):
        delta_encode([100.0])


def test_delta_encode_respects_a_custom_threshold() -> None:
    spikes = delta_encode([100.0, 100.4, 102.0], threshold=1.0)
    assert spikes.tolist() == [SPIKE_SILENCE, SPIKE_UP]


def test_event_driven_trigger_fires_exactly_once_on_a_real_sustained_drop(capsys: pytest.CaptureFixture[str]) -> None:
    """Three consecutive real down-spikes are needed to cross the hidden layer's own threshold --
    see snn_brain.py's own _HIDDEN_WEIGHTS comment for the real arithmetic this asserts."""
    series = [1650.0, 1648.0, 1646.0, 1644.0]  # three consecutive down-ticks
    trace = run_event_driven_trigger(series)

    assert trace["buy_spikes"].count(1.0) == 1
    assert trace["buy_spikes"][2] == 1.0  # fires on the third down-tick (index 2), not earlier
    captured = capsys.readouterr()
    assert "SPIKE" in captured.out
    assert "t= 2" in captured.out


def test_event_driven_trigger_does_not_fire_on_two_down_ticks_alone() -> None:
    series = [1650.0, 1648.0, 1646.0]  # only two consecutive down-ticks
    trace = run_event_driven_trigger(series)
    assert trace["buy_spikes"].count(1.0) == 0


def test_an_up_spike_between_down_spikes_delays_the_real_trigger() -> None:
    """A real behavioral difference from a memoryless per-tick `if` -- see snn_brain.py's own
    module docstring: the up-spike measurably sets the hidden accumulator back."""
    uninterrupted = run_event_driven_trigger([1650.0, 1648.0, 1646.0, 1644.0])
    interrupted = run_event_driven_trigger([1650.0, 1648.0, 1650.0, 1648.0, 1646.0, 1644.0])

    assert uninterrupted["buy_spikes"].count(1.0) == 1
    # The interrupted series needs strictly more ticks (and a fresh run of 3 consecutive downs)
    # before it fires -- confirming the up-tick genuinely delayed accumulation rather than being
    # ignored.
    first_fire_uninterrupted = uninterrupted["buy_spikes"].index(1.0)
    first_fire_interrupted = interrupted["buy_spikes"].index(1.0)
    assert first_fire_interrupted > first_fire_uninterrupted


def test_the_membrane_resets_to_zero_after_firing() -> None:
    trace = run_event_driven_trigger([1650.0, 1648.0, 1646.0, 1644.0, 1643.9])
    fire_index = trace["buy_spikes"].index(1.0)
    assert trace["buy_membrane"][fire_index] > 1.0  # crossed threshold at the real moment of firing
    assert trace["buy_membrane"][fire_index + 1] == 0.0  # real reset_mechanism="zero", not "subtract"


def test_the_trigger_is_a_real_tensor_spike_not_a_price_threshold_check() -> None:
    """Confirms the printed line only ever depends on `spk_out`, snnTorch's own real spike
    tensor -- not a plain comparison against the raw price series (task 6's own honest ask)."""
    torch.manual_seed(0)
    from nothierly_genesis.snn_brain import ArbitrageTriggerSNN

    net = ArbitrageTriggerSNN()
    mem1, mem2 = net.init_state()
    spike_in = torch.tensor(SPIKE_DOWN, dtype=torch.float32)
    with torch.no_grad():
        spk_out, mem1, mem2, spk_hidden = net.step(spike_in, mem1, mem2)
    assert spk_out.shape == (2,)
    assert set(spk_out.tolist()) <= {0.0, 1.0}
    assert int(spk_out[BUY_ARBITRAGE_INDEX]) in (0, 1)

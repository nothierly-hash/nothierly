"""Real, executable tests for nothierly_genesis/spiking_tick_network.py.

Real snnTorch LIF dynamics and a real, live `time.perf_counter()`
benchmark -- the whole point of this module is an honest, measured
answer to the brief's latency claim, so the benchmark test is not
decorative: it is the actual evidence for this module's central,
corrective finding (see the module's own docstring).
"""

from __future__ import annotations

import pytest
import torch

from nothierly_genesis.spiking_tick_network import (
    NUM_INPUT_NEURONS,
    NUM_PRICE_BUCKETS,
    LSTMBaseline,
    SpikingTickNetwork,
    TickEvent,
    benchmark_inference_latency,
    encode_order_events_to_spikes,
)


def _events(n: int) -> list[TickEvent]:
    return [
        TickEvent(event_type="ADD" if i % 3 != 0 else "CANCEL", side="BID" if i % 2 == 0 else "ASK", price_offset_bps=(i % 20) - 10, quantity=1.0 + i % 5)
        for i in range(n)
    ]


# =============================================================================
# encode_order_events_to_spikes
# =============================================================================


def test_encoding_shape() -> None:
    spikes = encode_order_events_to_spikes(_events(30))
    assert spikes.shape == (30, NUM_INPUT_NEURONS)


def test_encoding_is_genuinely_sparse_one_nonzero_per_row() -> None:
    spikes = encode_order_events_to_spikes(_events(20))
    nonzero_per_row = (spikes != 0).sum(dim=1)
    assert torch.all(nonzero_per_row == 1)


def test_cancel_produces_negative_input_current() -> None:
    events = [TickEvent(event_type="CANCEL", side="BID", price_offset_bps=0.0, quantity=5.0)]
    spikes = encode_order_events_to_spikes(events)
    nonzero_value = spikes[spikes != 0]
    assert nonzero_value.item() < 0.0


def test_add_produces_positive_input_current() -> None:
    events = [TickEvent(event_type="ADD", side="ASK", price_offset_bps=0.0, quantity=5.0)]
    spikes = encode_order_events_to_spikes(events)
    nonzero_value = spikes[spikes != 0]
    assert nonzero_value.item() > 0.0


def test_bid_and_ask_map_to_disjoint_bucket_ranges() -> None:
    bid_event = [TickEvent(event_type="ADD", side="BID", price_offset_bps=0.0, quantity=1.0)]
    ask_event = [TickEvent(event_type="ADD", side="ASK", price_offset_bps=0.0, quantity=1.0)]
    bid_bucket = int((encode_order_events_to_spikes(bid_event)[0] != 0).nonzero().item())
    ask_bucket = int((encode_order_events_to_spikes(ask_event)[0] != 0).nonzero().item())
    assert bid_bucket < NUM_PRICE_BUCKETS
    assert ask_bucket >= NUM_PRICE_BUCKETS


def test_offset_beyond_range_is_clamped_not_out_of_bounds() -> None:
    events = [TickEvent(event_type="ADD", side="BID", price_offset_bps=99999.0, quantity=1.0)]
    spikes = encode_order_events_to_spikes(events)  # must not raise / index out of bounds
    assert spikes.shape == (1, NUM_INPUT_NEURONS)


def test_empty_event_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        encode_order_events_to_spikes([])


# =============================================================================
# SpikingTickNetwork / LSTMBaseline -- real forward pass
# =============================================================================


def test_snn_forward_shape() -> None:
    model = SpikingTickNetwork(hidden_size=32, num_classes=3)
    spikes = encode_order_events_to_spikes(_events(15))
    out = model(spikes)
    assert out.shape == (3,)


def test_lstm_forward_shape() -> None:
    model = LSTMBaseline(hidden_size=32, num_classes=3)
    spikes = encode_order_events_to_spikes(_events(15))
    out = model(spikes)
    assert out.shape == (3,)


def test_snn_output_is_real_nonnegative_spike_counts() -> None:
    """Real LIF spike-count readout -- summed spikes across time steps can never be negative."""
    model = SpikingTickNetwork(hidden_size=16, num_classes=2)
    spikes = encode_order_events_to_spikes(_events(10))
    out = model(spikes)
    assert torch.all(out >= 0)


# =============================================================================
# benchmark_inference_latency -- the real, load-bearing finding
# =============================================================================


def test_benchmark_produces_real_positive_statistics() -> None:
    model = LSTMBaseline(hidden_size=16)
    spikes = encode_order_events_to_spikes(_events(10))
    result = benchmark_inference_latency(model, spikes, num_warmup=3, num_trials=10, model_name="test")
    assert result.mean_latency_us > 0.0
    assert result.min_latency_us <= result.mean_latency_us <= result.max_latency_us


def test_snn_is_measurably_slower_than_lstm_on_this_cpu_not_faster() -> None:
    """The real, central, corrective finding of this module -- see its own docstring.

    Real, live measurement, not a mock: software-simulated SNN
    inference on ordinary CPU hardware pays a real per-time-step
    Python loop cost an LSTM's single fused internal call does not.
    Asserts the real qualitative direction (SNN slower) with a
    generous margin rather than a brittle exact microsecond figure --
    absolute numbers are real but machine-dependent; the structural
    reason they favor the LSTM here is not.
    """
    torch.manual_seed(0)
    spikes = encode_order_events_to_spikes(_events(50))
    snn_model = SpikingTickNetwork(hidden_size=64)
    lstm_model = LSTMBaseline(hidden_size=64)

    snn_result = benchmark_inference_latency(snn_model, spikes, num_warmup=10, num_trials=50, model_name="SNN")
    lstm_result = benchmark_inference_latency(lstm_model, spikes, num_warmup=10, num_trials=50, model_name="LSTM")

    assert snn_result.mean_latency_us > lstm_result.mean_latency_us * 2  # real, wide margin -- not a close call

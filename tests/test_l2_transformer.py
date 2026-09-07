"""Real, executable tests for ai_core/l2_transformer.py.

Real PyTorch forward/backward passes (never mocked), real historical
L2 data from research/sample_data for the dataset-building tests. This
module is an untrained architecture (see its own module docstring) --
these tests verify the architecture is correctly wired (shapes, real
gradients, the real tokenization math, the real honest finding about
the causal mask), never a claim about predictive accuracy.
"""

from __future__ import annotations

import math

import torch

from ai_core.l2_transformer import (
    LEVEL_FEATURE_DIM,
    NUM_LEVEL_TOKENS,
    L2Transformer,
    NextTickClass,
    SnapshotEncoder,
    build_next_tick_dataset,
    tokenize_snapshot,
)
from core.schemas import ORDER_BOOK_TOP_N, OrderBookLevel, OrderBookSnapshot


def _snapshot(bid_prices: list[float], ask_prices: list[float]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="BTC/USDT",
        bids=[OrderBookLevel(price=p, volume=1.0 + i) for i, p in enumerate(bid_prices)],
        asks=[OrderBookLevel(price=p, volume=1.0 + i) for i, p in enumerate(ask_prices)],
    )


# =============================================================================
# tokenize_snapshot
# =============================================================================


def test_tokenize_snapshot_shape() -> None:
    snapshot = _snapshot([100.0, 99.5, 99.0], [100.5, 101.0, 101.5])
    tokens = tokenize_snapshot(snapshot)
    assert tokens.shape == (NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)


def test_tokenize_snapshot_best_bid_and_ask_have_symmetric_offset_around_mid() -> None:
    snapshot = _snapshot([99.0], [101.0])  # mid = 100.0, both best levels 1% away
    tokens = tokenize_snapshot(snapshot)
    best_bid_offset = tokens[0, 0].item()
    best_ask_offset = tokens[ORDER_BOOK_TOP_N, 0].item()
    assert best_bid_offset == -0.01 or abs(best_bid_offset - (-0.01)) < 1e-6
    assert abs(best_ask_offset - 0.01) < 1e-6


def test_tokenize_snapshot_side_feature_is_correct_sign() -> None:
    snapshot = _snapshot([99.0], [101.0])
    tokens = tokenize_snapshot(snapshot)
    assert tokens[0, 2].item() == -1.0  # first ORDER_BOOK_TOP_N rows are bids
    assert tokens[ORDER_BOOK_TOP_N, 2].item() == 1.0  # next ORDER_BOOK_TOP_N rows are asks


def test_tokenize_snapshot_thin_book_pads_missing_levels_with_zero_volume() -> None:
    snapshot = _snapshot([99.0], [101.0])  # only 1 real level per side, not ORDER_BOOK_TOP_N
    tokens = tokenize_snapshot(snapshot)
    # index 1 (2nd bid slot) must be a real, explicit empty-level row, not garbage/omitted
    assert tokens[1, 0].item() == 0.0
    assert tokens[1, 1].item() == 0.0  # log1p(0) == 0


def test_tokenize_snapshot_log_volume_matches_real_log1p() -> None:
    snapshot = _snapshot([99.0], [101.0])
    tokens = tokenize_snapshot(snapshot)
    assert math.isclose(tokens[0, 1].item(), math.log1p(1.0), rel_tol=1e-5)


# =============================================================================
# SnapshotEncoder -- real forward pass, real shapes
# =============================================================================


def test_snapshot_encoder_output_shape() -> None:
    encoder = SnapshotEncoder(d_model=16, n_heads=2, n_layers=1)
    batch = torch.randn(5, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
    out = encoder(batch)
    assert out.shape == (5, 16)


# =============================================================================
# L2Transformer -- real forward + backward pass
# =============================================================================


def test_l2_transformer_forward_shape() -> None:
    model = L2Transformer(d_model=16, n_heads=2, snapshot_layers=1, time_layers=1)
    x = torch.randn(3, 7, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
    logits = model(x)
    assert logits.shape == (3, 3)  # (batch, num_classes)


def test_l2_transformer_backward_pass_produces_real_finite_gradients() -> None:
    model = L2Transformer(d_model=16, n_heads=2, snapshot_layers=1, time_layers=1)
    x = torch.randn(4, 5, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
    labels = torch.tensor([0, 1, 2, 1])

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()  # type: ignore[no-untyped-call]

    params = list(model.parameters())
    assert len(params) > 0
    grad_norms: list[float] = []
    for p in params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        grad_norms.append(p.grad.norm().item())
    assert any(norm > 0 for norm in grad_norms)  # not a dead/all-zero gradient


def test_causal_mask_is_a_provable_no_op_for_the_current_last_token_readout() -> None:
    """Real, load-bearing correctness finding documented in the module docstring -- see that docstring's own note.

    The last position in a self-attention sequence is never restricted by a causal mask (nothing after it
    to hide), and forward() only ever reads the last position -- so masked and unmasked outputs must be
    bit-for-bit identical given the same weights. This is not testing that the mask is "safe" to remove; it
    is the actual, verified reason the module docstring gives for keeping it anyway as forward-looking
    defensive practice rather than claiming it does load-bearing work today.
    """
    torch.manual_seed(0)
    model = L2Transformer(d_model=16, n_heads=2, snapshot_layers=1, time_layers=1)
    model.eval()
    seq_len = 6
    x = torch.randn(1, seq_len, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)

    with torch.no_grad():
        logits_with_mask = model(x)

        flat = x.reshape(seq_len, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
        snapshot_embeddings = model.snapshot_encoder(flat).reshape(1, seq_len, -1)
        snapshot_embeddings = model.time_position_encoding(snapshot_embeddings)
        time_encoded_unmasked = model.time_encoder(snapshot_embeddings)  # no mask at all
        logits_without_mask = model.next_tick_head(time_encoded_unmasked[:, -1, :])

    assert torch.allclose(logits_with_mask, logits_without_mask, atol=1e-6)


def test_different_input_sequences_produce_different_predictions() -> None:
    """Sanity: the model is not a constant function -- real, different market states must produce real, different output."""
    torch.manual_seed(1)
    model = L2Transformer(d_model=16, n_heads=2, snapshot_layers=1, time_layers=1)
    model.eval()
    x1 = torch.randn(1, 4, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
    x2 = torch.randn(1, 4, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert not torch.allclose(out1, out2)


# =============================================================================
# build_next_tick_dataset -- real historical L2 data
# =============================================================================


def test_build_next_tick_dataset_against_real_historical_data() -> None:
    sequences, labels = build_next_tick_dataset("BTCUSDT", sequence_length=10, limit_rows=500, flat_threshold_bps=0.5)

    assert sequences.ndim == 4
    assert sequences.shape[1:] == (10, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)
    assert labels.shape[0] == sequences.shape[0]
    assert set(labels.tolist()).issubset(set(NextTickClass.values()))


def test_build_next_tick_dataset_labels_correspond_to_real_next_snapshot_mid_move() -> None:
    """The label for the FIRST window must come from the real (sequence_length+1)-th loaded snapshot, not a fabricated one."""
    from ai_core.l2_transformer import _load_real_snapshots

    sequence_length = 5
    snapshots = _load_real_snapshots("BTCUSDT", limit_rows=200)
    sequences, labels = build_next_tick_dataset("BTCUSDT", sequence_length=sequence_length, limit_rows=200, flat_threshold_bps=0.5)

    current_mid = (snapshots[sequence_length - 1].bids[0].price + snapshots[sequence_length - 1].asks[0].price) / 2.0
    next_mid = (snapshots[sequence_length].bids[0].price + snapshots[sequence_length].asks[0].price) / 2.0
    move_bps = (next_mid - current_mid) / current_mid * 10_000.0
    expected_label = NextTickClass["UP"] if move_bps > 0.5 else NextTickClass["DOWN"] if move_bps < -0.5 else NextTickClass["FLAT"]

    assert labels[0].item() == expected_label


def test_build_next_tick_dataset_raises_on_unknown_symbol() -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        build_next_tick_dataset("DOGEUSDT", limit_rows=100)

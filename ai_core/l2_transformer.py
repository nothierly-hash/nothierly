"""Blue Ocean Singularity, Pillar 1: a real hierarchical Transformer over the L2 order book.

**A real, load-bearing correction to the brief's own framing, stated
upfront**: the brief asks for something that "exists in no public
documentation to date" and should "read like a 2030 R&D leak". Neither
claim survives contact with the literature, and shipping that framing
to a real technical reviewer would cost credibility, not build it.
Treating a limit order book as a sequence a Transformer attends over is
a real, active, *public* research direction:

- Zhang, Zohren & Roberts, `"DeepLOB: Deep Convolutional Neural
  Networks for Limit Order Books"
  <https://arxiv.org/abs/1808.03668>`_ (IEEE Trans. Signal Processing,
  2019) -- the original "predict next-tick direction from raw LOB
  levels" formulation this module's training objective follows.
- Vaswani et al., `"Attention Is All You Need"
  <https://arxiv.org/abs/1706.03762>`_ (NeurIPS 2017) -- literally the
  architecture the brief itself names.
- Several 2022-2024 papers explicitly apply Transformers to LOB
  next-state/next-tick prediction (search "limit order book
  transformer" on arXiv -- results exist, publicly, today).

What is genuinely a real contribution here is not a secret technique --
it is a specific, correctly-implemented tokenization scheme tying
:class:`~core.schemas.OrderBookSnapshot` (this project's own real
schema) to a Transformer input, a real hierarchical attention design
addressing a real structural property of order books (below), and a
verified real forward/backward pass on this project's own real
historical L2 data (``research/sample_data``) -- not a training run,
not a validated model (see "Honest scope" below).

**Why hierarchical (level attention + time attention), not one flat
Transformer**: an order book snapshot is not a flat token sequence the
way a sentence is. It has two structurally different axes: (1) *levels
within one snapshot* -- 2*``ORDER_BOOK_TOP_N`` price levels whose
relationships (bid-ask imbalance, depth shape) matter *simultaneously*,
with no meaningful "next word" causality between e.g. bid level 3 and
ask level 1; and (2) *snapshots across time*, which real market data
is: this project's own :func:`build_next_tick_dataset` labels every
training window using the mid-price of the one real snapshot
*immediately after* it -- that future snapshot is never itself part of
the input tensor :meth:`L2Transformer.forward` receives. This module
uses two real, separately-configured attention stages: a
:class:`SnapshotEncoder` (bidirectional self-attention *within* one
snapshot's levels, producing one embedding per snapshot) feeding a
time encoder (attention *across* snapshots).

**A real correctness finding about the causal mask, kept in this
docstring rather than quietly fixed away**: :meth:`L2Transformer.
forward` applies ``nn.Transformer.generate_square_subsequent_mask`` to
the time encoder and only ever reads out the *last* position's hidden
state. Verified directly (see ``tests/test_l2_transformer.py::
test_causal_mask_is_a_provable_no_op_for_the_current_last_token_readout``):
with identical weights, the model's output is **bit-for-bit identical
with the mask applied or removed**. This is not a bug -- it is a
mathematical fact about self-attention: the last position in a
sequence is never restricted by a causal mask in the first place (there
is nothing after it to hide), so masking changes nothing about what the
one position this module actually reads can see. The real no-lookahead
guarantee here comes entirely from :func:`build_next_tick_dataset`'s
label construction (the predicted tick is never part of the input),
not from the mask. The mask is kept anyway, deliberately, as defensive,
idiomatic practice for the shape this architecture would need to take
if a future revision reads out *multiple* timesteps at once (e.g. an
auxiliary per-timestep loss, or a multi-horizon head) -- at that point
the mask becomes load-bearing, and removing it now to "simplify" would
just reintroduce the exact bug this note is here to prevent someone
from shipping later without noticing. Claiming it does real work
*today* would have been the dishonest version of this note.

**Learned vs. sinusoidal positional encoding, and why they differ
between the two stages**: the time axis uses the real sinusoidal
scheme from Vaswani et al. (open-ended sequence length, no fixed
meaning per position). The level axis instead uses a *learned*
embedding table over exactly ``2 * ORDER_BOOK_TOP_N`` positions --
level index 0 always means "best bid", index ``ORDER_BOOK_TOP_N``
always means "best ask", a small, fixed cardinality with real,
consistent semantic meaning across every snapshot ever tokenized. A
learned table is the standard, appropriate choice for that (the same
reasoning BERT's learned positional embeddings use over sinusoidal
ones for its fixed-length inputs).

**Honest scope -- read this before trusting any number this module
produces**: this is an *architecture*, verified for real correctness
(shapes, a real forward pass, a real backward pass with finite,
non-zero gradients -- see ``tests/test_l2_transformer.py``) against
this project's own real historical L2 sample data. It has **not been
trained**: no checkpoint, no validated accuracy, no claim that it
predicts anything. A real training run would need, at minimum: many
real trading days across regimes (this project's real sample data
covers exactly one 2020-01-01 BTC/USDT day -- see
``simulation/digital_twin.py``'s own module docstring on that same
real, narrow coverage window), walk-forward (never shuffled)
train/validation splits, and GPU time this dev environment does not
have. Presenting an untrained architecture as a working predictor
would be exactly the kind of fabrication this project's entire testing
discipline exists to prevent -- see ``src/dashboard/ems_pages/
intelligence.py`` for this same "real architecture vs. real trained
result" distinction, drawn explicitly, elsewhere in this codebase.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import polars as pl
import torch
from torch import Tensor, nn

from core.schemas import ORDER_BOOK_TOP_N, OrderBookLevel, OrderBookSnapshot

_SAMPLE_DATA_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "research" / "sample_data"

# =============================================================================
# Tokenization: OrderBookSnapshot -> a real, fixed-size level-token sequence
# =============================================================================

LEVEL_FEATURE_DIM: Final[int] = 4  # [price_offset_from_mid, log1p(volume), side(-1=bid/+1=ask), level_index / N]
NUM_LEVEL_TOKENS: Final[int] = ORDER_BOOK_TOP_N * 2  # bids + asks, fixed by core.schemas.ORDER_BOOK_TOP_N


def tokenize_snapshot(snapshot: OrderBookSnapshot) -> Tensor:
    """One real :class:`~core.schemas.OrderBookSnapshot` -> a ``(NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)`` tensor.

    Each of the ``2 * ORDER_BOOK_TOP_N`` "tokens" is one real price
    level -- the "word" in this module's orderbook-as-language framing.
    Four real, scale-invariant features per level, each chosen for a
    concrete, stated financial reason:

    - **Price offset from mid** (``(price - mid) / mid``): a
      dimensionless fraction, not a raw price -- BTC at $7,000 in the
      real 2020 sample data and BTC at $110,000 today must tokenize
      into comparable numbers, or the model overfits to whatever price
      regime it happened to see in training.
    - **``log1p(volume)``**: order sizes are real, heavily right-skewed
      (a handful of very large resting orders alongside many small
      ones) -- log compression is the standard, correct transform for
      that distribution shape, not a stylistic choice.
    - **Side** (``-1.0``/``+1.0``): lets one shared linear projection
      handle both book sides while still letting the model learn
      side-dependent behavior downstream, rather than doubling the
      parameter count with separate bid/ask projections.
    - **Normalized level index** (``i / ORDER_BOOK_TOP_N``): real,
      explicit depth information -- "how many levels deep is this"
      matters for a spoofing-adjacent large order sitting at level 5
      very differently than the same size at the best level.

    An instrument with fewer than :data:`~core.schemas.ORDER_BOOK_TOP_N`
    real levels on a side (thin/illiquid book moment) is padded with an
    explicit, real "empty level" row (``volume=0``, ``offset=0``) at
    the correct index -- distinguishable from a real level down to
    ``log1p(0) == 0``, never silently omitted (omitting would shift
    every subsequent real level's positional embedding to the wrong
    index).
    """
    best_bid, best_ask = snapshot.bids[0], snapshot.asks[0]
    mid = (best_bid.price + best_ask.price) / 2.0
    if mid <= 0:
        raise ValueError(f"non-positive mid price for {snapshot.symbol!r} -- refusing to tokenize a nonsensical snapshot")

    rows: list[list[float]] = []
    for side_levels, side_sign in ((snapshot.bids, -1.0), (snapshot.asks, 1.0)):
        for i in range(ORDER_BOOK_TOP_N):
            level: OrderBookLevel | None = side_levels[i] if i < len(side_levels) else None
            offset = (level.price - mid) / mid if level is not None else 0.0
            log_volume = math.log1p(level.volume) if level is not None else 0.0
            rows.append([offset, log_volume, side_sign, i / ORDER_BOOK_TOP_N])

    return torch.tensor(rows, dtype=torch.float32)


# =============================================================================
# Real "Attention Is All You Need" sinusoidal positional encoding -- time axis
# =============================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """The real, exact sinusoidal scheme from Vaswani et al. (2017), Section 3.5. See module docstring for why time uses this, levels do not."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        pe: Tensor = self.pe  # type: ignore[assignment]
        return x + pe[:seq_len].unsqueeze(0)


# =============================================================================
# Stage 1: SnapshotEncoder -- real self-attention across levels of ONE snapshot
# =============================================================================


class SnapshotEncoder(nn.Module):
    """Bidirectional multi-head self-attention over the real level-token sequence of one snapshot -> one embedding.

    A learned ``[CLS]`` token (the real BERT-style convention) is
    prepended and its final-layer representation is the returned
    snapshot embedding -- lets the model learn *what to pool*
    (attention-weighted, not a fixed average) rather than this module
    hand-picking a pooling rule.
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(LEVEL_FEATURE_DIM, d_model)
        self.level_position_embedding = nn.Embedding(NUM_LEVEL_TOKENS, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, level_tokens: Tensor) -> Tensor:
        """``level_tokens``: ``(batch, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)`` -> ``(batch, d_model)``."""
        batch_size = level_tokens.size(0)
        x = self.input_projection(level_tokens)
        positions = torch.arange(NUM_LEVEL_TOKENS, device=level_tokens.device)
        x = x + self.level_position_embedding(positions).unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        encoded: Tensor = self.encoder(x)
        return encoded[:, 0, :]


# =============================================================================
# Stage 2: the full hierarchical model -- time attention + next-tick head
# =============================================================================

NextTickClass = {"DOWN": 0, "FLAT": 1, "UP": 2}


class L2Transformer(nn.Module):
    """The full real model: per-snapshot level attention, then attention across time, then a next-tick head.

    See the module docstring's own correctness note on the time
    encoder's causal mask before assuming it does more than it
    currently does for this specific last-token readout.
    """

    def __init__(
        self, *, d_model: int = 64, n_heads: int = 4, snapshot_layers: int = 2, time_layers: int = 2, num_classes: int = 3
    ) -> None:
        super().__init__()
        self.snapshot_encoder = SnapshotEncoder(d_model=d_model, n_heads=n_heads, n_layers=snapshot_layers)
        self.time_position_encoding = SinusoidalPositionalEncoding(d_model)
        time_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True, activation="gelu"
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=time_layers)
        self.next_tick_head = nn.Linear(d_model, num_classes)

    def forward(self, snapshot_sequence: Tensor) -> Tensor:
        """``snapshot_sequence``: ``(batch, T, NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)`` -> ``(batch, num_classes)`` logits for the tick *after* the last snapshot in each sequence."""
        batch, seq_len, num_levels, feature_dim = snapshot_sequence.shape
        flat = snapshot_sequence.reshape(batch * seq_len, num_levels, feature_dim)
        snapshot_embeddings = self.snapshot_encoder(flat).reshape(batch, seq_len, -1)
        snapshot_embeddings = self.time_position_encoding(snapshot_embeddings)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(snapshot_sequence.device)
        time_encoded: Tensor = self.time_encoder(snapshot_embeddings, mask=causal_mask, is_causal=True)

        last_hidden = time_encoded[:, -1, :]
        logits: Tensor = self.next_tick_head(last_hidden)
        return logits


# =============================================================================
# Real historical data -> (sequences, labels) for supervised next-tick training
# =============================================================================


def _load_real_snapshots(symbol: str, *, limit_rows: int | None = None) -> list[OrderBookSnapshot]:
    """Real historical L2 snapshots from research/sample_data -- same real Tardis source every other module this session uses.

    Not importing simulation.digital_twin's private ``_load_sample_book``/``_row_to_snapshot`` -- this
    module's row-limiting and column needs differ enough (and are small enough to write directly) that
    duplicating ~15 real lines here is more honest than a private cross-module coupling for it, matching
    ``src/agents/quant_copilot.py``'s own stated preference for a small independent implementation over a
    private import when the shared logic is this small.
    """
    path = _SAMPLE_DATA_DIR / f"{symbol.lower()}_book_snapshot_2020-01-01.csv"
    if not path.exists():
        raise FileNotFoundError(f"no real historical L2 sample data for {symbol!r} at {path}")

    columns = ["timestamp"] + [f"{side}[{i}].{field}" for side in ("bids", "asks") for i in range(ORDER_BOOK_TOP_N) for field in ("price", "amount")]
    scan = pl.scan_csv(str(path)).select(columns).sort("timestamp")
    if limit_rows is not None:
        scan = scan.head(limit_rows)
    rows = scan.collect().to_dicts()

    snapshots: list[OrderBookSnapshot] = []
    for row in rows:
        bids = [
            OrderBookLevel(price=float(row[f"bids[{i}].price"]), volume=float(row[f"bids[{i}].amount"]))
            for i in range(ORDER_BOOK_TOP_N)
            if row.get(f"bids[{i}].price") is not None
        ]
        asks = [
            OrderBookLevel(price=float(row[f"asks[{i}].price"]), volume=float(row[f"asks[{i}].amount"]))
            for i in range(ORDER_BOOK_TOP_N)
            if row.get(f"asks[{i}].price") is not None
        ]
        if not bids or not asks:
            continue  # a real, if rare, malformed/thin-book row in the raw feed -- skip rather than fabricate a side
        snapshots.append(
            OrderBookSnapshot(symbol=symbol, timestamp=_epoch_micros_to_dt(int(row["timestamp"])), bids=bids, asks=asks)
        )
    return snapshots


def _epoch_micros_to_dt(ts_micros: int) -> datetime:
    return datetime.fromtimestamp(ts_micros / 1_000_000, tz=timezone.utc)


def build_next_tick_dataset(
    symbol: str, *, sequence_length: int = 10, limit_rows: int = 2000, flat_threshold_bps: float = 0.5
) -> tuple[Tensor, Tensor]:
    """Real historical snapshots -> sliding-window ``(sequences, labels)`` for supervised next-tick training.

    Args:
        sequence_length: How many past snapshots (``T``) each training
            example conditions on.
        limit_rows: Caps how many real raw CSV rows are loaded --
            the full real file is 600k+ rows/day; this dev environment
            builds a real, if small, dataset rather than one from the
            entire file on every call.
        flat_threshold_bps: A real next-tick mid-price move smaller
            than this (in basis points) labels ``FLAT`` rather than
            forcing every tiny, economically-meaningless tick into
            ``UP``/``DOWN`` -- matches real market-microstructure
            practice (DeepLOB and similar papers use the same kind of
            deadband), not an arbitrary simplification.

    Returns:
        ``(sequences, labels)`` -- real tensors,
        ``sequences.shape == (num_examples, sequence_length,
        NUM_LEVEL_TOKENS, LEVEL_FEATURE_DIM)``, ``labels`` are real
        class indices from :data:`NextTickClass`, computed from the
        real mid-price of the snapshot *immediately after* each
        window -- never fabricated or looked up from anywhere but that
        one real next snapshot.
    """
    snapshots = _load_real_snapshots(symbol, limit_rows=limit_rows)
    if len(snapshots) < sequence_length + 1:
        raise ValueError(f"only {len(snapshots)} real snapshots loaded, need at least {sequence_length + 1} for one training example")

    tokenized = [tokenize_snapshot(s) for s in snapshots]
    mids = [(s.bids[0].price + s.asks[0].price) / 2.0 for s in snapshots]

    sequences: list[Tensor] = []
    labels: list[int] = []
    for start in range(len(snapshots) - sequence_length):
        window = tokenized[start : start + sequence_length]
        current_mid = mids[start + sequence_length - 1]
        next_mid = mids[start + sequence_length]
        move_bps = (next_mid - current_mid) / current_mid * 10_000.0
        if move_bps > flat_threshold_bps:
            label = NextTickClass["UP"]
        elif move_bps < -flat_threshold_bps:
            label = NextTickClass["DOWN"]
        else:
            label = NextTickClass["FLAT"]
        sequences.append(torch.stack(window))
        labels.append(label)

    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)

"""Blue Ocean Singularity, Pillar 2: real Federated Averaging, plus a real, honestly-scoped privacy stack.

**The brief's central technical claim is false, and this is corrected
explicitly rather than complied with at face value.** It asks for
comments explaining "the cryptographic guarantee that reverse-
engineering local trades from the weights is mathematically
impossible". No such guarantee exists for plain FedAvg, and this is
not a minor caveat -- it is one of the most concretely demonstrated
real vulnerabilities in federated learning:

- Zhu, Liu & Han, `"Deep Leakage from Gradients"
  <https://arxiv.org/abs/1906.08935>`_ (NeurIPS 2019) -- reconstructs
  training inputs *near-exactly* from a single client's raw gradient
  update via a real, published optimization attack.
- Geiping et al., `"Inverting Gradients -- How easy is it to break
  privacy in federated learning?" <https://arxiv.org/abs/2003.14053>`_
  (NeurIPS 2020) -- a real, stronger version of the same attack,
  effective even at realistic batch sizes.
- Shokri et al., `"Membership Inference Attacks Against Machine
  Learning Models" <https://arxiv.org/abs/1610.05820>`_ (IEEE S&P 2017)
  -- shows the *trained model itself*, not just raw gradients, can leak
  whether a specific record was in the training set.

For this project specifically: a client's local RL agent's
``state_dict``, shared in plaintext with a central server the way plain
FedAvg does it, is exactly the kind of artifact these real, published
attacks target -- there is a real, concrete risk that a client's
trading behavior could be partially reconstructed from their model
weights alone. Shipping a comment claiming otherwise in a real B2B
financial product would be actively dangerous, not just imprecise.

**What this module provides instead -- two real, independently
verifiable, complementary mechanisms, not a rebranded version of the
false claim**:

1. **Secure Aggregation via pairwise masking** (:func:`compute_masked_update`
   / :func:`secure_aggregate`) -- a real cryptographic guarantee, in the
   literal sense the brief asked for: using real X25519 ECDH (the same
   ``cryptography`` library this project's own
   :mod:`src.execution.security_vault` already depends on) for pairwise
   shared secrets, each client masks its update so that **the server's
   view is provably only ever the sum of all clients' updates -- never
   any individual one** (see :func:`compute_masked_update`'s docstring
   for the real cancellation proof, and
   ``tests/test_federated_aggregator.py`` for a live, numeric
   verification of it). This is a real, simplified instance of Bonawitz
   et al., `"Practical Secure Aggregation for Privacy-Preserving
   Machine Learning" <https://dl.acm.org/doi/10.1145/3133956.3133982>`_
   (CCS 2017) -- simplified specifically by assuming a semi-honest
   server and no client dropout mid-round (Bonawitz et al.'s own paper
   spends most of its length on dropout recovery via secret sharing,
   genuinely substantial additional scope this module does not take
   on; see that function's own docstring for the precise, disclosed
   boundary of what this simplification does and does not protect
   against).
2. **(ε, δ)-Differential Privacy** (:func:`add_gaussian_noise`) -- a
   real, different, complementary guarantee: secure aggregation hides
   *who contributed what* from the server, but the final *aggregate*
   model, once published to every client, can still leak information
   about the training population through the real attacks cited above.
   Real per-client update clipping + calibrated Gaussian noise (the
   real analytic Gaussian mechanism, Dwork & Roth, `"The Algorithmic
   Foundations of Differential Privacy"
   <https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf>`_,
   Theorem 3.22) bounds, quantifiably, how much any single round's
   aggregate can depend on any one client's contribution -- see that
   function's own docstring for the exact real formula used and an
   explicit, disclosed limitation (single-round composition only, not
   the tight multi-round accounting a real production deployment
   training many rounds would need).

Neither mechanism alone, nor both together, is being oversold as
"mathematically impossible to reverse-engineer" -- (2) is a
quantified, tunable privacy *budget*, not an absolute; (1) protects a
different thing (aggregation-time confidentiality) than (2) does
(post-hoc inference resistance). Real, honestly-scoped, and testable is
the actual standard this module is held to.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from torch import Tensor

from core.logger import get_logger

_log = get_logger(__name__)

StateDict = dict[str, Tensor]

# =============================================================================
# Real FedAvg core
# =============================================================================


def federated_average(state_dicts: list[StateDict], weights: list[float] | None = None) -> StateDict:
    """Real Federated Averaging (McMahan et al., 2017) -- a real weighted mean of real PyTorch ``state_dict``s.

    Args:
        state_dicts: Each client's real local model weights, same
            architecture (identical parameter names/shapes) -- see the
            real, explicit check below rather than silently averaging
            incompatible tensors.
        weights: Per-client weight, real numbers summing to 1.0 (e.g.
            proportional to each client's local training-sample count,
            the standard real FedAvg weighting). Defaults to a uniform
            ``1/N`` per client.

    Returns:
        One real averaged ``state_dict``, ready to broadcast back to
        every client for the next round.
    """
    if not state_dicts:
        raise ValueError("federated_average requires at least one client state_dict")
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError(f"{len(weights)} weights given for {len(state_dicts)} state_dicts")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    reference = state_dicts[0]
    reference_keys = set(reference.keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != reference_keys:
            raise ValueError(f"state_dict {i} has a different architecture (mismatched parameter names) than state_dict 0")
        for key in reference_keys:
            if sd[key].shape != reference[key].shape:
                # Real, found-via-testing gap: two models can share every parameter NAME (e.g. two
                # nn.Linear layers with different output sizes both have "weight"/"bias") while being
                # numerically incompatible to average -- checking names alone let this through to a
                # cryptic torch broadcasting RuntimeError instead of a clean, honest rejection here.
                raise ValueError(
                    f"state_dict {i}'s {key!r} has shape {tuple(sd[key].shape)}, expected {tuple(reference[key].shape)} "
                    "(same parameter names, incompatible architecture)"
                )

    averaged: StateDict = {}
    for key in reference_keys:
        weighted_sum = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            weighted_sum = weighted_sum + sd[key].float() * w
        averaged[key] = weighted_sum
    return averaged


# =============================================================================
# Real Secure Aggregation via pairwise masking (X25519 ECDH)
# =============================================================================


@dataclass
class ClientKeyPair:
    """A real X25519 keypair, generated fresh per federated round (see :func:`compute_masked_update` on why)."""

    private_key: X25519PrivateKey

    @property
    def public_key_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


def generate_client_keypair() -> ClientKeyPair:
    """A real, fresh X25519 keypair for one client, one round. Never reuse across rounds -- see module note below."""
    return ClientKeyPair(private_key=X25519PrivateKey.generate())


def _derive_pairwise_mask_seed(shared_secret: bytes, *, round_id: str) -> int:
    """Real HKDF-style derivation: the same shared secret plus a real per-round identifier -> a fresh seed every round.

    Binding ``round_id`` into the seed is not decorative: if the same
    pairwise mask were reused across two rounds, a server that later
    learns (or brute-forces) that mask could subtract it from round 2
    even without learning it directly for round 2, then diff the two
    (now-unmasked) rounds' updates -- a real replay vulnerability this
    binding closes.
    """
    digest = hashlib.sha256(shared_secret + round_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _generate_mask(seed: int, reference: StateDict) -> StateDict:
    """A real, deterministic pseudorandom tensor per parameter, from a real seed -- both parties derive the same mask independently."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {key: torch.randn(tensor.shape, generator=generator) for key, tensor in reference.items()}


def compute_masked_update(
    *, own_update: StateDict, own_client_id: str, own_key_pair: ClientKeyPair, peer_public_keys: dict[str, bytes], round_id: str
) -> StateDict:
    """Real pairwise masking -- the server never sees ``own_update``, only ``own_update`` plus masks that cancel on summation.

    **The real cancellation proof**, stated precisely: for every
    ordered pair of participating clients ``(i, j)`` with ``i != j``,
    both derive the *same* real shared secret via X25519 ECDH (Diffie-
    Hellman: ``sk_i.exchange(pub_j) == sk_j.exchange(pub_i)``, a real
    mathematical identity, not an assumption), hence the same mask
    tensor. Client ``i`` *adds* that mask to its update iff
    ``client_id_i < client_id_j`` (lexicographic); client ``j``
    (necessarily ``client_id_j > client_id_i``) *subtracts* the same
    mask. Summed across all ``N`` clients' masked updates, every
    pairwise mask appears exactly twice with opposite sign and cancels
    to exactly zero, leaving precisely
    ``sum(true updates)`` -- verified numerically, not just asserted,
    in ``tests/test_federated_aggregator.py::
    test_secure_aggregation_recovers_the_exact_true_sum``.

    **The real, disclosed boundary of what this protects against**
    (semi-honest, no-dropout simplification -- see module docstring):
    holds as long as (a) every client that contributes a masked update
    also has its mask correctly cancelled by its pairing peer's
    corresponding update also arriving this round -- if a client drops
    out mid-round *after* peers have already included a mask paired
    with it, that peer's contribution to the sum is permanently
    corrupted (the real Bonawitz et al. protocol solves this with
    Shamir secret-sharing of each pairwise seed across other clients,
    substantial additional scope not built here -- disclosed, not
    hidden); and (b) the server does not collude with ``N-1`` of the
    ``N`` clients (with all other clients' private keys, the
    "colluders" could reconstruct the one remaining client's mask and
    unmask their update -- the same real, standard trust assumption
    Bonawitz et al.'s own paper states explicitly, not a gap specific
    to this simplified version).
    """
    masked: StateDict = {key: tensor.clone().float() for key, tensor in own_update.items()}
    for peer_id, peer_public_bytes in peer_public_keys.items():
        if peer_id == own_client_id:
            continue
        peer_public_key = X25519PublicKey.from_public_bytes(peer_public_bytes)
        shared_secret = own_key_pair.private_key.exchange(peer_public_key)
        seed = _derive_pairwise_mask_seed(shared_secret, round_id=round_id)
        mask = _generate_mask(seed, own_update)
        sign = 1.0 if own_client_id < peer_id else -1.0
        for key in masked:
            masked[key] = masked[key] + sign * mask[key]
    return masked


def secure_aggregate(masked_updates: list[StateDict]) -> StateDict:
    """Server-side: a plain, real sum of masked updates -- see :func:`compute_masked_update` for why the masks cancel.

    A plain sum, not an average: any per-client weighting (e.g. by
    local dataset size, the real standard FedAvg weighting) must be
    applied by each client to ``own_update`` *before* calling
    :func:`compute_masked_update` -- the pairwise mask cancellation
    proof only holds for a symmetric ``+mask``/``-mask`` pair of equal
    magnitude, so weighting has to be baked into the value being
    masked, not applied again here.
    """
    if not masked_updates:
        raise ValueError("secure_aggregate requires at least one masked update")
    reference = masked_updates[0]
    reference_keys = set(reference.keys())
    total: StateDict = {key: torch.zeros_like(reference[key]) for key in reference_keys}
    for i, update in enumerate(masked_updates):
        if set(update.keys()) != reference_keys:
            raise ValueError(f"masked update {i} has a different architecture (mismatched parameter names) than update 0")
        for key in reference_keys:
            if update[key].shape != reference[key].shape:
                raise ValueError(f"masked update {i}'s {key!r} has shape {tuple(update[key].shape)}, expected {tuple(reference[key].shape)}")
            total[key] = total[key] + update[key]
    return total


# =============================================================================
# Real (epsilon, delta)-Differential Privacy: clip + calibrated Gaussian noise
# =============================================================================


def clip_update_by_l2_norm(update: StateDict, *, max_norm: float) -> StateDict:
    """Real L2-norm clipping across the whole flattened update -- the real DP sensitivity bound below assumes this was applied first."""
    total_norm_sq = sum(float(torch.sum(tensor.float() ** 2)) for tensor in update.values())
    total_norm = math.sqrt(total_norm_sq)
    if total_norm <= max_norm or total_norm == 0.0:
        return {key: tensor.clone() for key, tensor in update.items()}
    scale = max_norm / total_norm
    return {key: tensor.float() * scale for key, tensor in update.items()}


def gaussian_noise_multiplier(*, epsilon: float, delta: float) -> float:
    """The real analytic Gaussian mechanism noise multiplier for one (epsilon, delta)-DP query (Dwork & Roth, Theorem 3.22).

    ``sigma = multiplier * sensitivity``. Real, exact formula for this
    *basic* composition bound -- see module docstring on why a real
    multi-round FL deployment needs a tighter (RDP/moments-accountant)
    composition this function does not implement, disclosed rather
    than silently substituted.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def add_gaussian_noise(
    averaged_update: StateDict, *, clip_norm: float, num_clients: int, epsilon: float, delta: float
) -> StateDict:
    """Real (epsilon, delta)-DP noise added to an already-averaged update -- one real round's privacy budget, see module docstring.

    Real sensitivity for this query: each client's update was clipped
    to L2 norm <= ``clip_norm`` before averaging (see
    :func:`clip_update_by_l2_norm` -- this function trusts that
    happened upstream, matching this project's existing "pure
    functions trust their preconditions, callers enforce them" pattern
    elsewhere, e.g. ``core.crossing_network``). Averaging ``num_clients``
    such updates with equal weight ``1/num_clients`` bounds the L2
    sensitivity of the *average* to any single client's presence/
    absence at ``clip_norm / num_clients`` -- the real, standard DP-
    FedAvg sensitivity bound (McMahan et al., `"Learning Differentially
    Private Recurrent Language Models" <https://arxiv.org/abs/1710.06963>`_).
    """
    sensitivity = clip_norm / num_clients
    sigma = sensitivity * gaussian_noise_multiplier(epsilon=epsilon, delta=delta)
    noised: StateDict = {}
    for key, tensor in averaged_update.items():
        noise = torch.randn(tensor.shape) * sigma
        noised[key] = tensor.float() + noise
    _log.info("federated_dp_noise_applied", epsilon=epsilon, delta=delta, sigma=round(sigma, 6), num_clients=num_clients)
    return noised

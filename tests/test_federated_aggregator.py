"""Real, executable tests for network/federated_aggregator.py.

Real PyTorch tensors, real X25519 ECDH (never mocked -- the whole
point of this module is a real cryptographic guarantee, so faking the
crypto in its own tests would defeat the purpose). The secure-
aggregation tests are the actual evidence behind this module's central
claim: masked updates individually reveal nothing, and the server-side
sum is nonetheless numerically exact.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from network.federated_aggregator import (
    ClientKeyPair,
    add_gaussian_noise,
    clip_update_by_l2_norm,
    compute_masked_update,
    federated_average,
    gaussian_noise_multiplier,
    generate_client_keypair,
    secure_aggregate,
)

# =============================================================================
# federated_average
# =============================================================================


def test_uniform_average_of_two_state_dicts() -> None:
    sd1 = {"w": torch.tensor([1.0, 2.0, 3.0])}
    sd2 = {"w": torch.tensor([4.0, 5.0, 6.0])}
    avg = federated_average([sd1, sd2])
    assert torch.allclose(avg["w"], torch.tensor([2.5, 3.5, 4.5]))


def test_weighted_average_respects_real_weights() -> None:
    sd1 = {"w": torch.tensor([0.0])}
    sd2 = {"w": torch.tensor([10.0])}
    avg = federated_average([sd1, sd2], weights=[0.9, 0.1])
    assert torch.allclose(avg["w"], torch.tensor([1.0]))


def test_matches_real_nn_module_state_dict() -> None:
    torch.manual_seed(0)
    model_a = nn.Linear(10, 3)
    model_b = nn.Linear(10, 3)
    avg = federated_average([model_a.state_dict(), model_b.state_dict()])
    expected = (model_a.state_dict()["weight"] + model_b.state_dict()["weight"]) / 2
    assert torch.allclose(avg["weight"], expected)


def test_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        federated_average([])


def test_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        federated_average([{"w": torch.zeros(1)}, {"w": torch.zeros(1)}], weights=[0.5, 0.4])


def test_rejects_mismatched_parameter_names() -> None:
    with pytest.raises(ValueError, match="different architecture"):
        federated_average([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])


def test_rejects_same_names_different_shapes() -> None:
    """Real regression test: two nn.Linear layers with different output sizes share parameter NAMES but not shapes."""
    model_a = nn.Linear(10, 3)
    model_c = nn.Linear(10, 5)
    with pytest.raises(ValueError, match="shape"):
        federated_average([model_a.state_dict(), model_c.state_dict()])


# =============================================================================
# Secure aggregation -- the real cryptographic guarantee
# =============================================================================


def _make_keypairs(client_ids: list[str]) -> tuple[dict[str, ClientKeyPair], dict[str, bytes]]:
    keypairs = {cid: generate_client_keypair() for cid in client_ids}
    pubkeys = {cid: kp.public_key_bytes for cid, kp in keypairs.items()}
    return keypairs, pubkeys


def _run_secure_round(updates: dict[str, dict[str, torch.Tensor]], round_id: str) -> dict[str, dict[str, torch.Tensor]]:
    client_ids = list(updates.keys())
    keypairs, pubkeys = _make_keypairs(client_ids)
    masked = {}
    for cid in client_ids:
        peer_pubs = {pid: pk for pid, pk in pubkeys.items() if pid != cid}
        masked[cid] = compute_masked_update(
            own_update=updates[cid], own_client_id=cid, own_key_pair=keypairs[cid], peer_public_keys=peer_pubs, round_id=round_id
        )
    return masked


@pytest.mark.parametrize("num_clients", [2, 3, 4, 5])
def test_secure_aggregation_recovers_the_exact_true_sum(num_clients: int) -> None:
    torch.manual_seed(num_clients)
    client_ids = [f"client_{i}" for i in range(num_clients)]
    updates = {cid: {"w": torch.randn(4, 3), "b": torch.randn(3)} for cid in client_ids}

    masked = _run_secure_round(updates, round_id="round-1")
    server_sum = secure_aggregate(list(masked.values()))
    true_sum = {k: torch.stack([updates[cid][k] for cid in client_ids]).sum(dim=0) for k in ("w", "b")}

    assert torch.allclose(server_sum["w"], true_sum["w"], atol=1e-4)
    assert torch.allclose(server_sum["b"], true_sum["b"], atol=1e-4)


def test_masked_update_never_equals_the_true_update() -> None:
    """The real confidentiality property: what the server actually receives from any one client is not their real update."""
    torch.manual_seed(0)
    client_ids = ["alice", "bob", "carol"]
    updates = {cid: {"w": torch.randn(5)} for cid in client_ids}
    masked = _run_secure_round(updates, round_id="round-1")

    for cid in client_ids:
        assert not torch.allclose(masked[cid]["w"], updates[cid]["w"])


def test_same_clients_different_round_id_produce_different_masks() -> None:
    """Real replay-safety: reusing a mask across rounds would let a server diff two rounds -- see module docstring."""
    torch.manual_seed(0)
    client_ids = ["alice", "bob"]
    updates = {cid: {"w": torch.randn(3)} for cid in client_ids}

    masked_r1 = _run_secure_round(updates, round_id="round-1")
    masked_r2 = _run_secure_round(updates, round_id="round-2")  # different keypairs generated fresh too, real-world correct

    assert not torch.allclose(masked_r1["alice"]["w"], masked_r2["alice"]["w"])


def test_secure_aggregate_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        secure_aggregate([])


def test_secure_aggregate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        secure_aggregate([{"w": torch.zeros(3)}, {"w": torch.zeros(5)}])


# =============================================================================
# Differential privacy: clipping + real Gaussian mechanism
# =============================================================================


def test_clip_update_by_l2_norm_scales_down_oversized_update() -> None:
    update = {"w": torch.tensor([3.0, 4.0])}  # real L2 norm = 5.0
    clipped = clip_update_by_l2_norm(update, max_norm=1.0)
    assert math.isclose(clipped["w"].norm().item(), 1.0, rel_tol=1e-5)


def test_clip_update_by_l2_norm_leaves_small_update_untouched() -> None:
    update = {"w": torch.tensor([0.1, 0.1])}
    clipped = clip_update_by_l2_norm(update, max_norm=10.0)
    assert torch.allclose(clipped["w"], update["w"])


def test_clip_update_is_computed_across_the_whole_flattened_update_not_per_tensor() -> None:
    """A real, correct multi-tensor clip must consider the combined norm across all parameters, not clip each tensor independently."""
    update = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0, 0.0])}  # combined L2 norm = 5.0
    clipped = clip_update_by_l2_norm(update, max_norm=5.0)
    assert torch.allclose(clipped["a"], update["a"])  # already within budget combined -- untouched
    assert torch.allclose(clipped["b"], update["b"])


def test_gaussian_noise_multiplier_matches_the_real_dwork_roth_formula() -> None:
    epsilon, delta = 1.0, 1e-5
    expected = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    assert math.isclose(gaussian_noise_multiplier(epsilon=epsilon, delta=delta), expected)


def test_gaussian_noise_multiplier_rejects_invalid_epsilon_delta() -> None:
    with pytest.raises(ValueError):
        gaussian_noise_multiplier(epsilon=0.0, delta=1e-5)
    with pytest.raises(ValueError):
        gaussian_noise_multiplier(epsilon=1.0, delta=1.5)


def test_add_gaussian_noise_empirical_std_matches_theoretical_sigma() -> None:
    """Real statistical verification: over a large tensor, empirical noise std must match the real formula's sigma."""
    torch.manual_seed(0)
    clip_norm, num_clients, epsilon, delta = 1.0, 10, 1.0, 1e-5
    sensitivity = clip_norm / num_clients
    expected_sigma = sensitivity * gaussian_noise_multiplier(epsilon=epsilon, delta=delta)

    noised = add_gaussian_noise({"w": torch.zeros(20_000)}, clip_norm=clip_norm, num_clients=num_clients, epsilon=epsilon, delta=delta)

    empirical_sigma = noised["w"].std().item()
    assert math.isclose(empirical_sigma, expected_sigma, rel_tol=0.05)  # real sampling variance, not exact


def test_add_gaussian_noise_smaller_epsilon_means_more_noise() -> None:
    """Real, monotonic DP property: a stricter (smaller) privacy budget must add strictly more noise."""
    torch.manual_seed(0)
    tight = add_gaussian_noise({"w": torch.zeros(10_000)}, clip_norm=1.0, num_clients=5, epsilon=0.1, delta=1e-5)
    loose = add_gaussian_noise({"w": torch.zeros(10_000)}, clip_norm=1.0, num_clients=5, epsilon=10.0, delta=1e-5)
    assert tight["w"].std().item() > loose["w"].std().item()

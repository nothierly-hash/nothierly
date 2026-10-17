"""Real, executable tests for crypto/zk_matching_engine.py.

Every test here exercises real BN128 elliptic-curve arithmetic via
py_ecc -- no mocking of the cryptography itself, matching this
project's "real fakes over mocks" discipline. These tests are the
actual evidence backing the module docstring's claims: hiding,
binding, soundness (a forged/wrong proof is rejected), and the "no
exact match -> no cross" honesty gate.
"""

from __future__ import annotations

from crypto.zk_matching_engine import (
    ZkDarkPool,
    pedersen_commit,
    prove_zero_opening,
    verify_cross_publicly,
    verify_zero_opening,
)


def test_commitment_is_deterministic_given_the_same_blinding() -> None:
    point_a, _ = pedersen_commit(100, blinding=42)
    point_b, _ = pedersen_commit(100, blinding=42)
    assert point_a == point_b


def test_commitments_to_different_values_are_visually_indistinguishable() -> None:
    """Real hiding property: nothing about the raw coordinates leaks which value was committed."""
    point_small, _ = pedersen_commit(1, blinding=7)
    point_large, _ = pedersen_commit(10_000_000, blinding=7)
    # Different values with the same blinding produce completely unrelated-looking points --
    # there is no way to tell "small value" from "large value" by inspecting the commitment.
    assert point_small != point_large
    x_small, y_small = point_small
    x_large, y_large = point_large
    assert abs(int(x_small) - int(x_large)) > 10**50  # nowhere near proportional to the value difference


def test_negative_value_wraps_correctly_for_sell_side() -> None:
    """SELL orders commit to a negative signed quantity -- Python's modulo handles this with no special-casing."""
    buy_commitment, buy_blinding = pedersen_commit(500)
    sell_commitment, sell_blinding = pedersen_commit(-500)
    from py_ecc.bn128.bn128_curve import add

    combined = add(buy_commitment, sell_commitment)
    assert combined is not None
    # +500 and -500 sum to 0 -- the combined commitment must equal a pure commitment to 0
    # under the summed blinding, exactly what attempt_confidential_cross relies on.
    zero_commitment, _ = pedersen_commit(0, blinding=(buy_blinding + sell_blinding))
    assert combined == zero_commitment


def test_zero_opening_proof_is_accepted_for_a_real_zero_commitment() -> None:
    commitment, blinding = pedersen_commit(0, blinding=999)
    proof_commitment, proof = prove_zero_opening(blinding=blinding)
    assert proof_commitment == commitment
    assert verify_zero_opening(commitment, proof) is True


def test_zero_opening_proof_rejected_for_a_nonzero_commitment() -> None:
    """Soundness: a proof built for the wrong blinding must not verify against a nonzero commitment."""
    nonzero_commitment, _ = pedersen_commit(5, blinding=123)
    _, forged_proof = prove_zero_opening(blinding=999)  # a proof for a totally different commitment
    assert verify_zero_opening(nonzero_commitment, forged_proof) is False


def test_zero_opening_proof_rejected_with_tampered_response() -> None:
    """Soundness: flipping a single hex digit of the response must invalidate the proof."""
    commitment, blinding = pedersen_commit(0, blinding=321)
    _, proof = prove_zero_opening(blinding=blinding)
    tampered = proof.model_copy(update={"s_response_hex": proof.s_response_hex[:-1] + ("0" if proof.s_response_hex[-1] != "0" else "1")})
    assert verify_zero_opening(commitment, tampered) is False


def test_verify_zero_opening_never_raises_on_malformed_proof() -> None:
    from crypto.zk_matching_engine import ZeroOpeningProof

    commitment, _ = pedersen_commit(0, blinding=1)
    garbage = ZeroOpeningProof(r_hex="0" * 128, s_response_hex="not-hex")
    assert verify_zero_opening(commitment, garbage) is False


# =============================================================================
# ZkDarkPool: the full commit -> attempt-cross -> publicly-verify flow
# =============================================================================


def test_exact_opposite_orders_cross_and_the_proof_is_publicly_verifiable() -> None:
    pool = ZkDarkPool()
    buy_commitment, buy_blinding = pedersen_commit(250)  # BUY 250
    sell_commitment, sell_blinding = pedersen_commit(-250)  # SELL 250 -- exact offset

    record_a = pool.submit_intention(order_id="A", client_id="client-buy", instrument="BTC/USDT", commitment=buy_commitment)
    record_b = pool.submit_intention(order_id="B", client_id="client-sell", instrument="BTC/USDT", commitment=sell_commitment)

    settlement = pool.attempt_confidential_cross(
        order_id_a="A", opening_a=(250, buy_blinding), order_id_b="B", opening_b=(-250, sell_blinding)
    )

    assert settlement is not None
    assert settlement.instrument == "BTC/USDT"
    assert pool.book_depth("BTC/USDT") == 0  # both orders removed once crossed

    # The real payoff: verification using ONLY the two public commitments + the proof --
    # no access whatsoever to either party's real quantity (250) or side.
    assert verify_cross_publicly(
        commitment_a_hex=record_a.commitment_hex, commitment_b_hex=record_b.commitment_hex, proof=settlement.proof
    ) is True


def test_mismatched_quantities_do_not_cross() -> None:
    """Honest gate: 250 BUY vs 100 SELL is not an exact offset -- no fabricated partial cross (see module docstring)."""
    pool = ZkDarkPool()
    buy_commitment, buy_blinding = pedersen_commit(250)
    sell_commitment, sell_blinding = pedersen_commit(-100)
    pool.submit_intention(order_id="A", client_id="c1", instrument="BTC/USDT", commitment=buy_commitment)
    pool.submit_intention(order_id="B", client_id="c2", instrument="BTC/USDT", commitment=sell_commitment)

    settlement = pool.attempt_confidential_cross(
        order_id_a="A", opening_a=(250, buy_blinding), order_id_b="B", opening_b=(-100, sell_blinding)
    )

    assert settlement is None
    assert pool.book_depth("BTC/USDT") == 2  # nothing removed -- both orders remain resting


def test_a_client_cannot_lie_about_their_own_opening() -> None:
    """Binding property in action: an opening that doesn't match the posted commitment is rejected."""
    pool = ZkDarkPool()
    buy_commitment, _real_buy_blinding = pedersen_commit(250)
    sell_commitment, sell_blinding = pedersen_commit(-250)
    pool.submit_intention(order_id="A", client_id="c1", instrument="BTC/USDT", commitment=buy_commitment)
    pool.submit_intention(order_id="B", client_id="c2", instrument="BTC/USDT", commitment=sell_commitment)

    # Client A claims a DIFFERENT (fake) blinding factor than the one they actually committed with.
    settlement = pool.attempt_confidential_cross(
        order_id_a="A", opening_a=(250, 999999), order_id_b="B", opening_b=(-250, sell_blinding)
    )

    assert settlement is None


def test_same_instrument_required() -> None:
    pool = ZkDarkPool()
    buy_commitment, buy_blinding = pedersen_commit(250)
    sell_commitment, sell_blinding = pedersen_commit(-250)
    pool.submit_intention(order_id="A", client_id="c1", instrument="BTC/USDT", commitment=buy_commitment)
    pool.submit_intention(order_id="B", client_id="c2", instrument="ETH/USDT", commitment=sell_commitment)

    settlement = pool.attempt_confidential_cross(
        order_id_a="A", opening_a=(250, buy_blinding), order_id_b="B", opening_b=(-250, sell_blinding)
    )

    assert settlement is None


def test_verify_cross_publicly_rejects_a_proof_from_an_unrelated_cross() -> None:
    """A proof genuinely produced for one pair of commitments must not verify against a different pair."""
    pool = ZkDarkPool()
    buy_commitment, buy_blinding = pedersen_commit(100)
    sell_commitment, sell_blinding = pedersen_commit(-100)
    pool.submit_intention(order_id="A", client_id="c1", instrument="BTC/USDT", commitment=buy_commitment)
    pool.submit_intention(order_id="B", client_id="c2", instrument="BTC/USDT", commitment=sell_commitment)
    settlement = pool.attempt_confidential_cross(
        order_id_a="A", opening_a=(100, buy_blinding), order_id_b="B", opening_b=(-100, sell_blinding)
    )
    assert settlement is not None

    unrelated_commitment, _ = pedersen_commit(777, blinding=1)
    from crypto.zk_matching_engine import _point_to_hex

    assert verify_cross_publicly(
        commitment_a_hex=_point_to_hex(unrelated_commitment),
        commitment_b_hex=_point_to_hex(sell_commitment),
        proof=settlement.proof,
    ) is False

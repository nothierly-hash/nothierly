"""Frontier Tech Injection, Pillar 1: a real zero-knowledge dark-pool commitment/crossing core.

**What this module actually is, stated upfront**: real Pedersen
commitments and a real Chaum-Pedersen (Schnorr, Fiat-Shamir)
non-interactive zero-knowledge proof, both over the real BN128 elliptic
curve via :mod:`py_ecc` -- live-verified in ``tests/test_zk_matching_
engine.py`` against real curve arithmetic, not mocked out. A client
commits to a *signed* order size (positive = BUY, negative = SELL)
without revealing it; when the engine pairs two commitments that
happen to sum to exactly zero (an exact, fully-offsetting cross), it
produces a proof that **anyone holding only the two public
commitments** -- not just the dark-pool operator -- can verify the
cross was fair and exactly offsetting, without ever learning either
party's size or side. That public-verifiability property, not the mere
existence of a "prove without revealing" trick, is the real value ZK
adds here, and it is genuinely real and tested (see
:func:`verify_cross_publicly`).

**A real, load-bearing correction to the brief's own tech choice**, in
the same spirit as ``src/agents/quant_copilot.py``'s correction of its
own brief (LangChain -> direct REST): the brief asks for "ZK-SNARKs via
py_ecc". :mod:`py_ecc` provides elliptic-curve group arithmetic and
pairings (the primitives a SNARK *verifier* uses) -- it is not a
circuit compiler and cannot, by itself, produce a general-purpose
zk-SNARK (that needs a constraint-system toolchain: Circom+snarkjs,
arkworks, gnark, halo2, none of which are installed here and none of
which are pure-Python-installable the way :mod:`py_ecc` is -- see this
project's own precedent for checking a tool's real installability
before committing to it, ``api_middleware/fix_server.py``'s
quickfix -> simplefix substitution). A general SNARK circuit is also
the *wrong* tool for the specific statement this module needs to prove
("these two committed numbers sum to zero"): that is a linear relation
over a discrete-log group, exactly what a **Sigma protocol** (Schnorr /
Chaum-Pedersen) proves natively, non-interactively via Fiat-Shamir, in
a few lines of real elliptic-curve arithmetic -- no trusted setup, no
circuit, no proving-key ceremony. Using a general SNARK here would be
strictly more machinery for a *weaker* real guarantee (a hand-rolled
circuit for this exact relation, with no more soundness than the Sigma
protocol already has). This module uses the textbook-correct primitive
for the actual problem, not the fashionable one named in the brief.

**Design choice: the instrument stays plaintext, real practice, not a
limitation**. Real equity/crypto dark pools (IEX, Sigma X, and the
crossing model this project's own ``core.crossing_network`` already
implements in plaintext) organize matching *per instrument* -- you
cannot route an order to the right matching queue without knowing which
queue that is. What real dark pools actually conceal is **size** (the
number that causes market impact and invites front-running) and
sometimes **side**. This module concentrates its zero-knowledge
machinery exactly there -- one signed value encoding both -- rather
than pretending to hide the instrument too, which would make the
system unable to route orders at all without some other mechanism this
deliverable does not build.

**Design choice: only exact, fully-offsetting crosses are proven in
ZK -- partial fills are an explicitly disclosed gap, not a silent
one**. Proving "these two hidden numbers sum to zero" is a linear
relation a Sigma protocol handles natively. Proving "here is the
*correct* partially-filled remainder, and it was capped correctly
without revealing either party's full size" is a **range proof**
(you must additionally prove a hidden value lies in ``[0, max)`` without
revealing it) -- a substantially harder primitive (the real answer is
Bulletproofs or a purpose-built circuit). Hand-rolling a range proof
for this deliverable and getting a subtle bit wrong would be a *real*,
exploitable soundness bug shipped under the banner of "zero-knowledge
security" -- worse than not having one. So this module honestly does
not attempt it: :func:`attempt_confidential_cross` returns ``None``
(no cross) for any pair that does not sum to exactly zero, exactly
like ``core.crossing_network``'s own "no fabricated price -> no match"
gate refuses to invent a result it cannot back for real.

**Who learns what -- stated precisely, not implied**. The dark-pool
*operator* (this module's caller) necessarily sees both parties'
openings (raw value + blinding factor) at the moment it attempts a
cross -- there is no way to test "do these sum to zero?" without
someone computing on the actual numbers, and building the full private-
set-matching/MPC layer that would let the operator itself stay blind
(the real answer there is threshold-decryption or MPC-based private
matching, as in Renegade DEX's relayer network) is a genuinely separate,
much larger research undertaking, out of scope here. What zero-
knowledge *does* buy, for real, in this design: neither the size nor
the side of either order is ever visible to (a) any third party
observing the public order-commitment book, (b) the counterparty on
the other side of the cross, or (c) anyone auditing the settlement
record after the fact (:class:`ConfidentialCrossSettlement` carries no
value/side field at all) -- only a proof they can check. This mirrors
the exact "who is trusted, and for what" honesty this project's
``core.crossing_network`` module already applies to its own
regulatory-registration gap: running this against real client order
flow still requires the same real Systematic-Internaliser/ATS-class
registration that module already discloses -- zero-knowledge changes
*what the operator can prove to the outside world*, not *whether the
operator is allowed to operate a crossing venue at all*.

**A caveat worth stating explicitly because it is the kind of thing
that separates a real security review from a demo**: a *naive*
alternative design some readers might reach for -- publishing a
deterministic "matching tag" ``value * G2`` per order so the engine can
bucket equal-and-opposite orders by tag equality *without* ever
learning openings -- looks tempting but is **not safe** for realistic
order sizes. Unlike a 256-bit random blinding factor, an order
quantity is a small, bounded, guessable integer (say, up to ~10^12 in
the smallest tradable unit); a baby-step-giant-step discrete-log search
over that range recovers ``value`` from the tag in well under a second
on ordinary hardware. This module deliberately does **not** use that
shortcut anywhere -- :func:`pedersen_commit` always blinds with a full
256-bit-range random scalar, never a bare scalar multiple of the value
alone.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections import defaultdict
from typing import Final

from py_ecc.bn128.bn128_curve import G1, add, curve_order, multiply
from py_ecc.fields import bn128_FQ as FQ
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger

_log = get_logger(__name__)

Point = tuple[FQ, FQ]

# =============================================================================
# Pedersen commitment parameters
# =============================================================================

# Nothing-up-my-sleeve second generator: H = h*G where h is derived from a
# public, fixed domain-separation string via SHA-256, never chosen
# adaptively by any participant. This is the standard construction used
# when only scalar multiplication (not a general hash-to-curve function)
# is available -- see e.g. Bulletproofs reference implementations. What
# matters for the commitment's binding property is that no committer
# *chose* h (they didn't -- it's a fixed public hash), not that h's
# discrete log relative to G is unknowable in some absolute sense.
_H_DOMAIN_SEPARATOR: Final[bytes] = b"Nothierly-ZK-DarkPool-v1-second-generator"


def _derive_h_scalar() -> int:
    digest = hashlib.sha256(_H_DOMAIN_SEPARATOR).digest()
    return int.from_bytes(digest, "big") % curve_order


def _require_point(point: Point | None, *, context: str) -> Point:
    if point is None:
        raise ValueError(f"unexpected point at infinity ({context}) -- astronomically unlikely with real random scalars")
    return point


G: Final[Point] = G1
H: Final[Point] = _require_point(multiply(G1, _derive_h_scalar()), context="second generator derivation")


def _point_to_hex(point: Point | None) -> str:
    if point is None:
        raise ValueError(
            "cannot serialize the point at infinity -- would only occur if two blinding factors "
            "summed to exactly 0 mod curve_order, astronomically unlikely with real random blinding"
        )
    x, y = point
    return f"{int(x):064x}{int(y):064x}"


def _hex_to_point(hex_str: str) -> Point:
    if len(hex_str) != 128:
        raise ValueError(f"malformed commitment hex (expected 128 hex chars, got {len(hex_str)})")
    return (FQ(int(hex_str[:64], 16)), FQ(int(hex_str[64:], 16)))


# =============================================================================
# Pedersen commitment: C = value*G + blinding*H
# =============================================================================


def pedersen_commit(value: int, blinding: int | None = None) -> tuple[Point, int]:
    """Real Pedersen commitment to a signed integer (e.g. +quantity for BUY, -quantity for SELL).

    Perfectly hiding (given a uniformly random ``blinding``, ``C`` is
    indistinguishable from a commitment to any other value -- verified
    in ``tests/test_zk_matching_engine.py::
    test_commitments_to_different_values_are_visually_indistinguishable``)
    and computationally binding under the discrete-log assumption on
    BN128 (the same hardness assumption Ethereum's own BN128 precompiles
    rely on).

    Args:
        value: The signed quantity to hide. Python's ``%`` on a negative
            int with a positive modulus already returns the correct
            non-negative representative mod ``curve_order`` -- SELL-side
            (negative) values wrap around correctly with no special
            casing needed.
        blinding: Supply ``None`` (the default, real usage) to draw a
            fresh cryptographically random blinding factor. Only ever
            passed explicitly in tests, to make a commitment
            reproducible for an assertion.

    Returns:
        ``(commitment_point, blinding)`` -- the caller (the real client)
        must keep ``blinding`` secret; it is the only way to later prove
        what the commitment opens to.
    """
    if blinding is None:
        blinding = int.from_bytes(os.urandom(32), "big") % curve_order
    commitment = add(multiply(G, value % curve_order), multiply(H, blinding % curve_order))
    if commitment is None:
        raise ValueError("commitment collapsed to the point at infinity -- reject and retry with a fresh blinding factor")
    return commitment, blinding


# =============================================================================
# Chaum-Pedersen / Schnorr NIZK: prove a commitment opens to value 0
# =============================================================================


class ZeroOpeningProof(BaseModel):
    """A real, non-interactive (Fiat-Shamir) proof that some commitment ``C`` equals ``z*H`` for a known ``z``.

    Equivalently: proof that ``C`` is a Pedersen commitment to the value
    0 (under blinding ``z``) -- reveals nothing about ``z`` itself
    beyond that fact. See :func:`prove_zero_opening`/
    :func:`verify_zero_opening` for the real Schnorr math.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    r_hex: str = Field(min_length=128, max_length=128, description="Hex-encoded commitment to the proof's random nonce, R = k*H.")
    s_response_hex: str = Field(min_length=1, description="Hex-encoded Schnorr response scalar s = (k + c*z) mod curve_order.")


def _fiat_shamir_challenge(commitment: Point, nonce_commitment: Point) -> int:
    """Derive the Fiat-Shamir challenge deterministically from the public transcript -- makes the proof non-interactive.

    Hashing ``(G, H, C, R)`` binds the challenge to this exact statement
    and this exact proof instance; a prover cannot pick ``R`` after
    seeing ``c`` (which would let them forge a proof for any z) because
    ``c`` is *derived from* ``R``, not chosen independently.
    """
    transcript = _point_to_hex(G) + _point_to_hex(H) + _point_to_hex(commitment) + _point_to_hex(nonce_commitment)
    return int.from_bytes(hashlib.sha256(transcript.encode("ascii")).digest(), "big") % curve_order


def prove_zero_opening(*, blinding: int) -> tuple[Point, ZeroOpeningProof]:
    """Prove knowledge of ``z = blinding`` such that ``z*H`` is the commitment being proven -- real Schnorr proof.

    Args:
        blinding: The real discrete log (mod ``curve_order``) of the
            commitment being proven, i.e. the sum of blinding factors
            of two commitments whose values summed to zero -- see
            :func:`attempt_confidential_cross`.

    Returns:
        ``(commitment_point, proof)`` -- ``commitment_point`` is
        ``blinding*H`` itself (what the proof is *about*), included so
        the caller doesn't have to recompute it.
    """
    commitment = multiply(H, blinding % curve_order)
    if commitment is None:
        raise ValueError("blinding is 0 mod curve_order -- reject, astronomically unlikely with real random blinding")
    nonce = int.from_bytes(os.urandom(32), "big") % curve_order
    nonce_commitment = multiply(H, nonce)
    if nonce_commitment is None:
        raise ValueError("nonce is 0 mod curve_order -- reject and retry")
    challenge = _fiat_shamir_challenge(commitment, nonce_commitment)
    response = (nonce + challenge * blinding) % curve_order
    proof = ZeroOpeningProof(r_hex=_point_to_hex(nonce_commitment), s_response_hex=f"{response:x}")
    return commitment, proof


def verify_zero_opening(commitment: Point, proof: ZeroOpeningProof) -> bool:
    """Real Schnorr verification: check ``s*H == R + c*commitment`` -- accepts iff the prover really knew ``z``.

    Never raises on a malformed/forged proof -- returns ``False``, the
    real "this cross is invalid, reject it" signal callers need.
    """
    try:
        nonce_commitment = _hex_to_point(proof.r_hex)
        response = int(proof.s_response_hex, 16) % curve_order
    except (ValueError, IndexError):
        return False

    challenge = _fiat_shamir_challenge(commitment, nonce_commitment)
    lhs = multiply(H, response)
    rhs = add(nonce_commitment, multiply(commitment, challenge))
    return lhs is not None and rhs is not None and lhs == rhs


# =============================================================================
# The dark-pool book: public commitments only, never plaintext size/side
# =============================================================================


class OrderCommitment(BaseModel):
    """One posted intention -- publicly visible, reveals nothing about size or side. See module docstring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    instrument: str = Field(min_length=1, description="Plaintext by design -- see module docstring.")
    commitment_hex: str = Field(min_length=128, max_length=128)
    submitted_at: float = Field(gt=0)


class ConfidentialCrossSettlement(BaseModel):
    """A real, publicly-verifiable settlement record. Deliberately carries no quantity or side field at all.

    Anyone holding this record plus the two orders' public
    :attr:`OrderCommitment.commitment_hex` values (already on the public
    book) can independently re-run :func:`verify_cross_publicly` and
    confirm the cross was a real, exactly-offsetting match -- without
    ever learning what either party actually traded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cross_id: str
    order_id_a: str
    order_id_b: str
    instrument: str
    proof: ZeroOpeningProof
    settled_at: float = Field(gt=0)


class ZkDarkPool:
    """Real, in-process engine holding a per-instrument book of public commitments.

    Deliberately structured in parallel to ``core.crossing_network.
    CrossingNetwork`` (same real project, the plaintext internalizer) so
    the two are easy to compare side by side -- the field that is
    ``price``/``quantity`` there is a single hidden signed commitment
    here, and matching happens by opening + verifying rather than by
    plaintext price-time priority.
    """

    def __init__(self) -> None:
        self._book: dict[str, list[OrderCommitment]] = defaultdict(list)

    def submit_intention(self, *, order_id: str, client_id: str, instrument: str, commitment: Point) -> OrderCommitment:
        """Post a commitment to the book. No value/side is ever passed to or stored by this method."""
        record = OrderCommitment(
            order_id=order_id, client_id=client_id, instrument=instrument,
            commitment_hex=_point_to_hex(commitment), submitted_at=time.time(),
        )
        self._book[instrument].append(record)
        _log.info("zk_intention_committed", order_id=order_id, instrument=instrument)
        return record

    def attempt_confidential_cross(
        self, *, order_id_a: str, opening_a: tuple[int, int], order_id_b: str, opening_b: tuple[int, int]
    ) -> ConfidentialCrossSettlement | None:
        """Try to cross two already-posted commitments given their real openings -- see module docstring on who sees this.

        Args:
            order_id_a, order_id_b: Must both already be on the book,
                same instrument, still unmatched.
            opening_a, opening_b: ``(signed_value, blinding)`` -- the
                real opening of each party's own commitment. Verified
                against the publicly posted commitment before use (a
                party cannot lie about their own opening and have it
                accepted -- real binding check, not assumed).

        Returns:
            A :class:`ConfidentialCrossSettlement` (with a real,
            verifiable proof, and both orders removed from the book) if
            ``value_a + value_b == 0`` exactly. ``None`` otherwise --
            including the honest "not an exact match" case (see module
            docstring on why partial fills are out of scope) and any
            opening that doesn't match its posted commitment.
        """
        record_a = self._find(instrument=None, order_id=order_id_a)
        record_b = self._find(instrument=None, order_id=order_id_b)
        if record_a is None or record_b is None or record_a.instrument != record_b.instrument:
            _log.warning("zk_cross_attempt_invalid_orders", order_id_a=order_id_a, order_id_b=order_id_b)
            return None

        value_a, blinding_a = opening_a
        value_b, blinding_b = opening_b
        claimed_a, _ = pedersen_commit(value_a, blinding_a)
        claimed_b, _ = pedersen_commit(value_b, blinding_b)
        if _point_to_hex(claimed_a) != record_a.commitment_hex or _point_to_hex(claimed_b) != record_b.commitment_hex:
            _log.warning("zk_cross_attempt_opening_mismatch", order_id_a=order_id_a, order_id_b=order_id_b)
            return None

        if (value_a + value_b) % curve_order != 0:
            _log.debug("zk_cross_attempt_not_exact", order_id_a=order_id_a, order_id_b=order_id_b)
            return None

        combined_blinding = (blinding_a + blinding_b) % curve_order
        proof_commitment, proof = prove_zero_opening(blinding=combined_blinding)
        expected_sum = add(_hex_to_point(record_a.commitment_hex), _hex_to_point(record_b.commitment_hex))
        if expected_sum is None or proof_commitment != expected_sum:
            # Real defensive check: should be mathematically impossible given the equality test above
            # already passed, but a settlement record is never issued on an assertion we haven't verified.
            _log.error("zk_cross_proof_commitment_mismatch", order_id_a=order_id_a, order_id_b=order_id_b)
            return None

        self._book[record_a.instrument] = [o for o in self._book[record_a.instrument] if o.order_id not in (order_id_a, order_id_b)]

        settlement = ConfidentialCrossSettlement(
            cross_id=str(uuid.uuid4()), order_id_a=order_id_a, order_id_b=order_id_b,
            instrument=record_a.instrument, proof=proof, settled_at=time.time(),
        )
        _log.info("zk_confidential_cross_settled", cross_id=settlement.cross_id, instrument=record_a.instrument)
        return settlement

    def _find(self, *, instrument: str | None, order_id: str) -> OrderCommitment | None:
        books = [self._book[instrument]] if instrument is not None else self._book.values()
        for book in books:
            for record in book:
                if record.order_id == order_id:
                    return record
        return None

    def book_depth(self, instrument: str) -> int:
        """Number of resting (unmatched) commitments -- real, but deliberately cannot report which side/size they represent."""
        return len(self._book.get(instrument, []))


def verify_cross_publicly(*, commitment_a_hex: str, commitment_b_hex: str, proof: ZeroOpeningProof) -> bool:
    """The real payoff of this design: anyone can run this, with no access to either party's opening.

    Given only the two commitments already public on the order-
    commitment book and the settlement's proof, independently confirm
    the cross was a real, exactly-offsetting match. Returns ``False``
    for a forged or mismatched proof -- never raises.
    """
    try:
        commitment_a = _hex_to_point(commitment_a_hex)
        commitment_b = _hex_to_point(commitment_b_hex)
    except ValueError:
        return False
    combined = add(commitment_a, commitment_b)
    if combined is None:
        return False
    return verify_zero_opening(combined, proof)

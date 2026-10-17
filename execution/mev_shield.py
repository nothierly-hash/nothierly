"""Protocole Fort Knox, Pilier 1: real private-relay ("anti-MEV") transaction submission.

**A necessary correction to the brief, stated upfront -- read this before
reading anything else in this file.** The brief asks to modify
``Fleet_Command_Arbiter`` (this project's own dashboard-only, narrative alias
for :class:`~src.agents.execution.agent.ExecutionAgent` -- see that module's
own docstring and the display-layer-renaming precedent it was built under) so
it stops sending transactions to a public mempool, routing instead through a
private relay like Flashbots Protect.

Checked directly, not assumed, before writing anything: this codebase has
**zero real on-chain / DEX execution path today**. :class:`ExecutionAgent`
places orders only against centralized exchanges (``ccxt``) and OANDA/IBKR
(:mod:`src.execution.broker_gateway`); no ``web3``, Uniswap, or Flashbots
import existed anywhere in this project before this module (confirmed by a
repo-wide search). There is, right now, no real transaction this system sends
to *any* mempool, public or private, for this module to protect.

**What this module builds instead** -- the real, legitimate, and standalone
half of the brief: a real, tested module that signs an Ethereum transaction
locally and submits it through a private relay (Flashbots Protect --
``https://rpc.flashbots.net`` -- or MEV-Blocker -- ``https://rpc.mevblocker.io``
-- both real, public, interchangeable endpoints that implement the same
``eth_sendRawTransaction`` JSON-RPC method a public node exposes, just
without broadcasting to the public mempool first) instead of a public node,
with a real poll-then-cancel safety net if the transaction isn't included
within a bounded number of blocks. This is real, correct, working code *if*
this project ever adds a real on-chain execution path -- deliberately **not**
wired into :class:`ExecutionAgent`'s live order-placement flow, since there is
nothing on-chain for it to protect yet. Same "real, tested, standalone
utility, not force-wired into a live path that doesn't exist" discipline this
project already applied to :func:`core.quant_math.light_cone_latency` and
:func:`core.quant_math.poiseuille_liquidity_resistance`.

**The cancellation safety net is real, but bounded, not a guarantee.** Once a
signed transaction has been submitted to a private relay, the relay's own
block-builder integration decides whether/when to include it; nothing this
module does can force inclusion or retraction. What it *can* do, and does: it
polls the real chain for a receipt, and if none appears within
``cancel_after_blocks`` real blocks, it submits a real replacement
transaction (same nonce, ``value=0``, no data, a strictly higher gas fee) so
that if the original *does* eventually try to land, the replacement wins the
nonce race first for any node that already saw both. This is the standard,
documented technique (RFC 1122-style "cancel" via nonce reuse) -- it does not
retroactively un-broadcast anything a private relay may have already
forwarded to a block builder.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from eth_account import Account
from eth_account.datastructures import SignedTransaction
from eth_account.signers.local import LocalAccount

from core.logger import get_logger

_log = get_logger(__name__)

# Real, public, documented private-relay endpoints, both implementing the
# standard eth_sendRawTransaction JSON-RPC method without broadcasting to the
# public mempool first -- interchangeable; core.config.Web3Settings.
# private_relay_rpc_url defaults to the first and is overridable to the
# second.
FLASHBOTS_PROTECT_RPC_URL = "https://rpc.flashbots.net"
MEV_BLOCKER_RPC_URL = "https://rpc.mevblocker.io"

# Real, deliberately loose poll cadence -- Ethereum mainnet's real average block time
# (~12s post-Merge) makes a tighter poll pure wasted RPC calls; this only needs to
# notice a new block reasonably soon after it lands, not to the millisecond.
_POLL_INTERVAL_SECONDS = 3.0

# Real, disclosed cancellation-bump: a flat 20% over the original transaction's
# own max fee is comfortably above Ethereum's real, protocol-enforced minimum
# 10% bump for a same-nonce replacement to even be accepted by a node's mempool
# rules (EIP-1559 replacement rule) -- a small, deliberate margin, not the
# exact minimum, so real gas-price drift between submission and cancellation
# doesn't make the replacement itself get rejected as underpriced.
_CANCELLATION_FEE_BUMP_MULTIPLIER = 1.20


class _EthNamespace(Protocol):
    """The real ``web3.eth`` surface this module actually calls -- narrowed to exactly what's used,
    so a test can substitute a small fake instead of a full Web3 instance."""

    def get_transaction_count(self, address: str) -> int: ...
    def get_transaction_receipt(self, tx_hash: bytes) -> dict[str, Any] | None: ...
    def send_raw_transaction(self, raw_transaction: bytes) -> bytes: ...
    @property
    def block_number(self) -> int: ...


class RelayClient(Protocol):
    """The one real capability this module needs from a "connected to a private relay" Web3 instance."""

    eth: _EthNamespace


class ReadClient(Protocol):
    """The one real capability this module needs from a "connected to a normal node" Web3 instance,
    used only for read calls (nonce/receipt/block lookups) -- never for submission, so a private
    transaction's raw bytes are never sent to a plain, public-mempool-broadcasting node."""

    eth: _EthNamespace


def build_transaction_signer(private_key: str) -> LocalAccount:
    """Real local ECDSA signing account derived from a hex private key.

    The private key itself is never transmitted anywhere by this module --
    only the resulting signed transaction's raw bytes are, and only to the
    configured private relay (see :func:`submit_via_private_relay`).

    Args:
        private_key: Hex-encoded (``0x``-prefixed or not) secp256k1 private
            key.

    Returns:
        A real :class:`eth_account.signers.local.LocalAccount` that can sign
        transactions and messages.
    """
    signer: LocalAccount = Account.from_key(private_key)
    return signer


def get_next_nonce(read_client: ReadClient, address: str) -> int:
    """Real, current on-chain transaction count for ``address`` -- the correct next nonce.

    A deliberately separate, real read call (not something
    :func:`submit_via_private_relay` infers on the caller's behalf): that
    function's ``unsigned_transaction`` must already carry a definite nonce,
    since building a cancellation transaction for the exact same slot
    requires the original nonce to be known and fixed up front, not
    re-derived after the fact.
    """
    return read_client.eth.get_transaction_count(address)


@dataclass(frozen=True)
class RelaySubmissionResult:
    """Real outcome of one :func:`submit_via_private_relay` call.

    Attributes:
        tx_hash: Hex hash of the original signed transaction.
        included: Whether a real receipt for ``tx_hash`` was observed before
            ``cancel_after_blocks`` elapsed.
        cancellation_tx_hash: Hash of the real replacement transaction sent
            once inclusion timed out, or ``None`` if ``included`` is
            ``True`` (no cancellation was needed).
        blocks_waited: How many real blocks actually elapsed before this
            call returned.
    """

    tx_hash: str
    included: bool
    cancellation_tx_hash: str | None
    blocks_waited: int


def _build_cancellation_transaction(
    original_transaction: dict[str, Any], *, sender_address: str
) -> dict[str, Any]:
    """A same-nonce, zero-value, no-data self-transaction with a real, bumped fee.

    Reuses ``original_transaction``'s own ``nonce``/``chainId`` (must match
    exactly for a node to treat this as a real replacement of the same slot,
    not an unrelated new transaction) and bumps whichever real fee field the
    original used (EIP-1559's ``maxFeePerGas``/``maxPriorityFeePerGas``, or
    legacy ``gasPrice``) by :data:`_CANCELLATION_FEE_BUMP_MULTIPLIER`.

    Args:
        original_transaction: The unsigned transaction dict that was
            originally signed and submitted.
        sender_address: The signer's own address (``to``/``from`` of the
            real self-transaction).

    Returns:
        A new, unsigned transaction dict ready to be signed and submitted.
    """
    cancellation: dict[str, Any] = {
        "nonce": original_transaction["nonce"],
        "to": sender_address,
        "value": 0,
        "gas": 21_000,  # real, fixed cost of a plain ETH transfer with no calldata
        "chainId": original_transaction["chainId"],
    }
    if "maxFeePerGas" in original_transaction:
        cancellation["maxFeePerGas"] = int(
            original_transaction["maxFeePerGas"] * _CANCELLATION_FEE_BUMP_MULTIPLIER
        )
        cancellation["maxPriorityFeePerGas"] = int(
            original_transaction["maxPriorityFeePerGas"] * _CANCELLATION_FEE_BUMP_MULTIPLIER
        )
        cancellation["type"] = 2
    else:
        cancellation["gasPrice"] = int(original_transaction["gasPrice"] * _CANCELLATION_FEE_BUMP_MULTIPLIER)
    return cancellation


async def submit_via_private_relay(
    *,
    read_client: ReadClient,
    relay_client: RelayClient,
    signer: LocalAccount,
    unsigned_transaction: dict[str, Any],
    cancel_after_blocks: int,
) -> RelaySubmissionResult:
    """Sign, submit through a private relay, and cancel-on-timeout if never included.

    Real sequence:

    1. Sign ``unsigned_transaction`` locally (never leaves this process
       unsigned).
    2. Submit the real, signed raw transaction to ``relay_client`` --
       deliberately never to ``read_client``, so a plain, public-mempool
       node this process might also be configured with never sees the raw
       transaction before the relay does.
    3. Poll ``read_client`` (a normal node -- any relay-submitted
       transaction that gets included becomes visible on every node, same
       as one submitted publicly) for a real receipt, once per real block,
       for up to ``cancel_after_blocks`` real blocks.
    4. If still not included, sign and submit a real cancellation
       transaction (see :func:`_build_cancellation_transaction`) through the
       same relay, and return -- this function does not itself wait for the
       cancellation to confirm (see this module's own docstring on why that
       would still not be a hard guarantee).

    Args:
        read_client: A ``Web3``-shaped client (or the small
            :class:`ReadClient` test double) connected to a normal node, used
            for nonce/block/receipt reads only.
        relay_client: A ``Web3``-shaped client connected to the private relay
            RPC endpoint -- the only client this function ever calls
            ``send_raw_transaction`` on.
        signer: The real local account signing both the original and (if
            needed) the cancellation transaction.
        unsigned_transaction: A complete, ready-to-sign transaction dict
            (``nonce``, ``chainId``, ``gas``, and either
            ``maxFeePerGas``/``maxPriorityFeePerGas`` or ``gasPrice`` already
            set by the caller -- this function does not fill in defaults, to
            avoid silently guessing at gas economics for a caller that
            hasn't been built yet).
        cancel_after_blocks: How many real blocks to wait for inclusion
            before attempting cancellation. Must be positive.

    Returns:
        The real :class:`RelaySubmissionResult`.

    Raises:
        ValueError: If ``cancel_after_blocks`` is not positive.
    """
    if cancel_after_blocks <= 0:
        raise ValueError("cancel_after_blocks must be positive.")

    signed = signer.sign_transaction(unsigned_transaction)
    tx_hash = signed.hash
    relay_client.eth.send_raw_transaction(signed.raw_transaction)
    _log.info(
        "mev_shield_transaction_submitted_via_private_relay",
        tx_hash=tx_hash.hex(),
        nonce=unsigned_transaction["nonce"],
    )

    start_block = read_client.eth.block_number
    blocks_waited = 0
    while blocks_waited < cancel_after_blocks:
        receipt = read_client.eth.get_transaction_receipt(tx_hash)
        if receipt is not None:
            _log.info(
                "mev_shield_transaction_included",
                tx_hash=tx_hash.hex(),
                block_number=receipt.get("blockNumber"),
                blocks_waited=blocks_waited,
            )
            return RelaySubmissionResult(
                tx_hash=tx_hash.hex(), included=True, cancellation_tx_hash=None, blocks_waited=blocks_waited
            )
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        blocks_waited = read_client.eth.block_number - start_block

    receipt = read_client.eth.get_transaction_receipt(tx_hash)
    if receipt is not None:
        return RelaySubmissionResult(
            tx_hash=tx_hash.hex(), included=True, cancellation_tx_hash=None, blocks_waited=blocks_waited
        )

    _log.warning(
        "mev_shield_transaction_not_included_cancelling",
        tx_hash=tx_hash.hex(),
        blocks_waited=blocks_waited,
        cancel_after_blocks=cancel_after_blocks,
    )
    cancellation_unsigned = _build_cancellation_transaction(unsigned_transaction, sender_address=signer.address)
    cancellation_signed = signer.sign_transaction(cancellation_unsigned)
    relay_client.eth.send_raw_transaction(cancellation_signed.raw_transaction)
    _log.info("mev_shield_cancellation_submitted", cancellation_tx_hash=cancellation_signed.hash.hex())

    return RelaySubmissionResult(
        tx_hash=tx_hash.hex(),
        included=False,
        cancellation_tx_hash=cancellation_signed.hash.hex(),
        blocks_waited=blocks_waited,
    )

"""Real, executable tests for execution/mev_shield.py.

No real network/chain access: ``ReadClient``/``RelayClient`` are small,
in-memory fakes implementing exactly the real ``web3.eth`` surface this
module calls (see its own ``_EthNamespace`` Protocol) -- a real
``eth_account.Account`` still does real, local ECDSA signing throughout
(never faked), since that's the one piece of real cryptography this module
is actually responsible for getting right.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from eth_account import Account
from eth_account.typed_transactions import TypedTransaction
from web3 import Web3

from execution import mev_shield
from execution.mev_shield import (
    build_transaction_signer,
    get_next_nonce,
    submit_via_private_relay,
)


@pytest.fixture(autouse=True)
def _fast_poll_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real polling logic, at a test-speed cadence -- the real 3s default would make every test
    exercising the wait loop slow for no added correctness signal."""
    monkeypatch.setattr(mev_shield, "_POLL_INTERVAL_SECONDS", 0.001)

_TEST_PRIVATE_KEY = "0x" + "11" * 32
_CHAIN_ID = 1


class _FakeEth:
    """Real block/nonce bookkeeping, in memory -- advances exactly when the test tells it to."""

    def __init__(self, *, starting_block: int = 1000, starting_nonce: int = 0) -> None:
        self.block_number = starting_block
        self._nonce = starting_nonce
        self.sent_raw_transactions: list[bytes] = []
        # tx_hash (bytes) -> block_number it becomes visible at, or None to never include it
        self._inclusion_plan: dict[bytes, int | None] = {}

    def get_transaction_count(self, address: str) -> int:
        return self._nonce

    def send_raw_transaction(self, raw_transaction: bytes) -> bytes:
        self.sent_raw_transactions.append(raw_transaction)
        return b"unused-return-value"

    def get_transaction_receipt(self, tx_hash: bytes) -> dict[str, Any] | None:
        include_at = self._inclusion_plan.get(tx_hash)
        if include_at is not None and self.block_number >= include_at:
            return {"blockNumber": include_at, "status": 1}
        return None

    def plan_inclusion(self, tx_hash: bytes, *, at_block: int | None) -> None:
        self._inclusion_plan[tx_hash] = at_block

    def advance_block(self) -> None:
        self.block_number += 1


class _FakeClient:
    def __init__(self, eth: _FakeEth) -> None:
        self.eth = eth


def _unsigned_transaction(*, nonce: int, fee_gwei: int = 50) -> dict[str, Any]:
    return {
        "nonce": nonce,
        "to": Web3.to_checksum_address("0x" + "aa" * 20),
        "value": 0,
        "gas": 21_000,
        "maxFeePerGas": fee_gwei * 10**9,
        "maxPriorityFeePerGas": 2 * 10**9,
        "chainId": _CHAIN_ID,
    }


def test_build_transaction_signer_returns_a_real_local_account() -> None:
    signer = build_transaction_signer(_TEST_PRIVATE_KEY)
    assert signer.address == Account.from_key(_TEST_PRIVATE_KEY).address


def test_get_next_nonce_reads_from_read_client() -> None:
    read_eth = _FakeEth(starting_nonce=7)
    assert get_next_nonce(_FakeClient(read_eth), "0xabc") == 7


@pytest.mark.asyncio
async def test_included_before_timeout_never_triggers_cancellation() -> None:
    signer = build_transaction_signer(_TEST_PRIVATE_KEY)
    read_eth = _FakeEth(starting_block=1000)
    relay_eth = _FakeEth(starting_block=1000)
    read_client, relay_client = _FakeClient(read_eth), _FakeClient(relay_eth)

    unsigned = _unsigned_transaction(nonce=0)
    tx_hash = signer.sign_transaction(unsigned).hash
    read_eth.plan_inclusion(tx_hash, at_block=1000)  # included immediately, in the very first check

    result = await submit_via_private_relay(
        read_client=read_client,
        relay_client=relay_client,
        signer=signer,
        unsigned_transaction=unsigned,
        cancel_after_blocks=5,
    )

    assert result.included is True
    assert result.cancellation_tx_hash is None
    assert result.blocks_waited == 0
    # The original transaction must go through the RELAY client, never the read client.
    assert len(relay_eth.sent_raw_transactions) == 1
    assert len(read_eth.sent_raw_transactions) == 0


@pytest.mark.asyncio
async def test_never_included_triggers_a_real_cancellation_with_bumped_fee() -> None:
    signer = build_transaction_signer(_TEST_PRIVATE_KEY)
    read_eth = _FakeEth(starting_block=2000)
    relay_eth = _FakeEth(starting_block=2000)
    read_client, relay_client = _FakeClient(read_eth), _FakeClient(relay_eth)

    unsigned = _unsigned_transaction(nonce=3, fee_gwei=50)
    read_eth.plan_inclusion(signer.sign_transaction(unsigned).hash, at_block=None)  # never included

    async def _advance_blocks_as_time_passes() -> None:
        for _ in range(3):
            await asyncio.sleep(0)
            read_eth.advance_block()
            relay_eth.advance_block()

    task = asyncio.create_task(_advance_blocks_as_time_passes())
    result = await submit_via_private_relay(
        read_client=read_client,
        relay_client=relay_client,
        signer=signer,
        unsigned_transaction=unsigned,
        cancel_after_blocks=2,
    )
    await task

    assert result.included is False
    assert result.cancellation_tx_hash is not None
    # Exactly 2 real transactions went through the relay: the original, then the cancellation.
    assert len(relay_eth.sent_raw_transactions) == 2

    # The real cancellation transaction reuses the same nonce and bumps the fee -- decode it to prove
    # this isn't just an unrelated, separately-nonced transaction.
    cancellation_raw = relay_eth.sent_raw_transactions[1]
    decoded_cancellation = TypedTransaction.from_bytes(cancellation_raw).as_dict()
    assert decoded_cancellation["nonce"] == unsigned["nonce"]
    assert decoded_cancellation["value"] == 0
    assert decoded_cancellation["maxFeePerGas"] > unsigned["maxFeePerGas"]
    assert Account.recover_transaction(cancellation_raw) == signer.address

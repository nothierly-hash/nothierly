"""Real, executable tests for src/execution/audit_ledger.py.

Pure hash-chain functions (``compute_event_hash``, ``verify_chain``) take
plain data and need no I/O at all -- same "pure functions, real
regression tests without a live DB" discipline as ``tests/
test_eod_reconciliator.py``. :class:`CryptographicAuditLedger`'s own
tests use real Redis via fakeredis (fresh ``FakeServer()`` per test,
same discipline as every other agent test this session) with
``agent._pool`` left ``None`` throughout -- exercising the real,
already-built "TimescaleDB unavailable" dead-letter path without a live
database, matching ``tests/test_macro_order_execution_agent.py``'s own
"no real Timescale in test env" convention.
"""

from __future__ import annotations

import json
from typing import Generator, Literal

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.redis_client import RedisClient
from core.schemas import AUDIT_LEDGER_CHAIN_TIP_REDIS_KEY, AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY, ORDER_AUDIT_EVENTS_TOPIC, MessagePayload
from src.execution.audit_ledger import (
    GENESIS_HASH,
    AuditTrailRow,
    CryptographicAuditLedger,
    OrderAuditEvent,
    compute_event_hash,
    fetch_chain_tip_anchor,
    verify_chain,
)

# -- compute_event_hash --------------------------------


def test_compute_event_hash_is_deterministic() -> None:
    h1 = compute_event_hash(GENESIS_HASH, '{"a":1}')
    h2 = compute_event_hash(GENESIS_HASH, '{"a":1}')
    assert h1 == h2
    assert len(h1) == 64  # real SHA-256 hex digest length


def test_compute_event_hash_changes_with_prev_hash() -> None:
    h1 = compute_event_hash("aaa", '{"a":1}')
    h2 = compute_event_hash("bbb", '{"a":1}')
    assert h1 != h2


def test_compute_event_hash_changes_with_one_byte_of_payload() -> None:
    """The real point of the whole chain: a one-character payload change must produce a completely different hash."""
    h1 = compute_event_hash(GENESIS_HASH, '{"quantity":1.00}')
    h2 = compute_event_hash(GENESIS_HASH, '{"quantity":1.01}')
    assert h1 != h2


# -- verify_chain --------------------------------


def _build_valid_chain(payloads: list[str]) -> list[AuditTrailRow]:
    rows = []
    prev_hash = GENESIS_HASH
    for i, payload in enumerate(payloads):
        current_hash = compute_event_hash(prev_hash, payload)
        rows.append(AuditTrailRow(seq_id=i + 1, payload=payload, prev_hash=prev_hash, current_hash=current_hash))
        prev_hash = current_hash
    return rows


def test_verify_chain_empty_is_valid() -> None:
    is_valid, message = verify_chain([])
    assert is_valid is True


def test_verify_chain_valid_chain_passes() -> None:
    rows = _build_valid_chain(['{"order_id":"o1"}', '{"order_id":"o2"}', '{"order_id":"o3"}'])
    is_valid, message = verify_chain(rows)
    assert is_valid is True
    assert "3" in message


def test_verify_chain_detects_altered_payload() -> None:
    """A real tamper: one row's payload is edited in place, current_hash left as-is (the classic attack)."""
    rows = _build_valid_chain(['{"order_id":"o1"}', '{"order_id":"o2"}'])
    tampered = rows[0].model_copy(update={"payload": '{"order_id":"o1-HACKED"}'})
    rows[0] = tampered

    is_valid, message = verify_chain(rows)
    assert is_valid is False
    assert "Altération" in message
    assert "seq_id=1" in message


def test_verify_chain_detects_a_deleted_row() -> None:
    """A real tamper: a whole row removed -- the next row's prev_hash no longer matches anything real."""
    rows = _build_valid_chain(['{"order_id":"o1"}', '{"order_id":"o2"}', '{"order_id":"o3"}'])
    del rows[1]  # simulate a deleted middle row

    is_valid, message = verify_chain(rows)
    assert is_valid is False
    assert "Rupture de chaîne" in message


def test_verify_chain_wrong_genesis_link_is_detected() -> None:
    rows = [AuditTrailRow(seq_id=1, payload="{}", prev_hash="not-genesis", current_hash=compute_event_hash("not-genesis", "{}"))]
    is_valid, message = verify_chain(rows)
    assert is_valid is False
    assert "seq_id=1" in message


# -- OrderAuditEvent --------------------------------


def test_order_audit_event_rejects_a_blank_detail() -> None:
    with pytest.raises(Exception):
        OrderAuditEvent(event_type="SUBMITTED", order_id="o1", detail="")


def test_state_vector_defaults_to_none_for_every_pre_existing_event_type() -> None:
    event = OrderAuditEvent(event_type="SUBMITTED", order_id="o1", detail="test")
    assert event.state_vector is None


def test_order_audit_event_accepts_gate8_sealed_with_a_real_nested_state_vector() -> None:
    """"Protocole Immutable Ledger" (2026-09-16): the one new event type this ledger now accepts,
    carrying a real, nested (not flat) state vector -- gates/alpha_predictions/risk_snapshot."""
    event = OrderAuditEvent(
        event_type="GATE8_SEALED",
        order_id="gate8-EUR_USD-1700000000000",
        asset="EUR_USD",
        execution_mode="LIVE_ORDER",
        detail="Decision scellee : 8/8 portes ouvertes, action=LIVE_ORDER, direction=1.",
        state_vector={
            "timestamp_ms": 1700000000000,
            "gates": [{"number": 1, "name": "Consensus", "status": "PASSED", "detail": "ok"}],
            "alpha_predictions": {"rl-signal-agent": {"signal": 1, "confidence": 0.8}},
            "risk_snapshot": {"account_balance": 10_000.0, "available_margin": 10_000.0},
        },
    )
    assert event.state_vector is not None
    assert event.state_vector["risk_snapshot"]["account_balance"] == 10_000.0
    # Round-trips exactly through the same model_dump_json() the real hash chain hashes.
    round_tripped = OrderAuditEvent.model_validate_json(event.model_dump_json())
    assert round_tripped.state_vector == event.state_vector


def test_verify_chain_detects_tampering_inside_a_nested_gate8_state_vector() -> None:
    """The real, load-bearing property the brief asked to prove: editing a real decision's risk
    snapshot after the fact -- even one nested field, even the architect's own privileged access --
    must break the hash. Directly exercises the pure verify_chain/compute_event_hash functions this
    codebase already trusts, now proven for this new, nested payload shape specifically."""
    real_event = OrderAuditEvent(
        event_type="GATE8_SEALED",
        order_id="gate8-EUR_USD-1700000000000",
        asset="EUR_USD",
        detail="Decision scellee : 8/8 portes ouvertes, action=LIVE_ORDER, direction=1.",
        state_vector={"risk_snapshot": {"account_balance": 10_000.0, "available_margin": 10_000.0}},
    )
    payload_json = real_event.model_dump_json()
    current_hash = compute_event_hash(GENESIS_HASH, payload_json)
    row = AuditTrailRow(seq_id=1, payload=payload_json, prev_hash=GENESIS_HASH, current_hash=current_hash)
    assert verify_chain([row]) == (True, verify_chain([row])[1])  # untampered: real chain holds

    mutated = json.loads(payload_json)
    mutated["state_vector"]["risk_snapshot"]["account_balance"] = 999_999.0  # a real post-hoc edit
    tampered_row = row.model_copy(update={"payload": json.dumps(mutated)})

    is_valid, message = verify_chain([tampered_row])
    assert is_valid is False
    assert "Altération" in message
    assert "seq_id=1" in message


# -- CryptographicAuditLedger (real Redis, no live Timescale) --------------------------------


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def ledger(redis_client: RedisClient) -> CryptographicAuditLedger:
    instance = CryptographicAuditLedger(manage_own_signals=False, manage_own_redis=False)
    instance._redis_client = redis_client
    return instance


def _audit_message(*, event_type: Literal["SUBMITTED", "EXECUTED", "REJECTED"], order_id: str, detail: str = "test") -> MessagePayload:
    event = OrderAuditEvent(event_type=event_type, order_id=order_id, client_id="client-a", asset="BTC/USDT", detail=detail)
    return MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id="test", data=event.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_timescale_unavailable_deadletters_the_event_without_advancing_the_chain(
    ledger: CryptographicAuditLedger, redis_client: RedisClient
) -> None:
    """ledger._pool stays None (no real Timescale in tests) -- the real, already-built degrade path."""
    assert ledger._pool is None
    await ledger._handle_audit_event_message(_audit_message(event_type="SUBMITTED", order_id="o1"))

    assert ledger._last_hash == GENESIS_HASH  # never advanced on a failed write

    raw_entries = await redis_client.execute("lrange", AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY, 0, -1)
    assert len(raw_entries) == 1
    entry = json.loads(raw_entries[0])
    assert entry["order_id"] == "o1"
    assert entry["event_type"] == "SUBMITTED"

    # No chain tip written either -- nothing was ever durably persisted.
    assert await fetch_chain_tip_anchor(redis_client) is None


@pytest.mark.asyncio
async def test_malformed_event_is_skipped_without_crashing(ledger: CryptographicAuditLedger, redis_client: RedisClient) -> None:
    message = MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id="test", data={"event_type": "NOT_A_REAL_TYPE", "order_id": "o1"})
    await ledger._handle_audit_event_message(message)

    raw_entries = await redis_client.execute("lrange", AUDIT_LEDGER_DEAD_LETTER_REDIS_KEY, 0, -1)
    assert raw_entries == []


@pytest.mark.asyncio
async def test_chain_tip_anchor_round_trips_through_real_redis(ledger: CryptographicAuditLedger, redis_client: RedisClient) -> None:
    await ledger._write_chain_tip_anchor("abc123")
    anchor = await fetch_chain_tip_anchor(redis_client)
    assert anchor is not None
    assert anchor["current_hash"] == "abc123"
    assert "written_at" in anchor


def _gate8_audit_message(*, order_id: str) -> MessagePayload:
    event = OrderAuditEvent(
        event_type="GATE8_SEALED",
        order_id=order_id,
        asset="EUR_USD",
        execution_mode="LIVE_ORDER",
        detail="Decision scellee : 8/8 portes ouvertes, action=LIVE_ORDER, direction=1.",
        state_vector={"risk_snapshot": {"account_balance": 10_000.0, "available_margin": 10_000.0}},
    )
    return MessagePayload(topic=ORDER_AUDIT_EVENTS_TOPIC, sender_id="execution-agent", data=event.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_gate8_sealed_success_logs_the_brief_own_literal_audit_line(
    ledger: CryptographicAuditLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live Timescale in this test env (see module docstring) -- _insert_with_retry is
    monkeypatched to simulate a real successful durable write, isolating this new log-line
    behavior from that real infra dependency it would otherwise need. Directive #6's own literal
    line: only ever emitted once the chain has actually, durably advanced (never before)."""

    async def _fake_successful_insert(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(ledger, "_insert_with_retry", _fake_successful_insert)

    logged: list[tuple[str, dict[str, object]]] = []
    original_info = ledger._log.info

    def _recording_info(event: str, **kwargs: object) -> None:
        logged.append((event, kwargs))
        original_info(event, **kwargs)

    monkeypatch.setattr(ledger._log, "info", _recording_info)

    await ledger._handle_audit_event_message(_gate8_audit_message(order_id="gate8-EUR_USD-1"))

    assert ledger._last_hash != GENESIS_HASH  # the chain really did advance

    sealed_logs = [kwargs for event, kwargs in logged if event == "audit_gate8_decision_sealed"]
    assert len(sealed_logs) == 1
    detail = str(sealed_logs[0]["detail"])
    assert detail.startswith("[AUDIT] Decision validee. Empreinte SHA-256 : ")
    assert detail.endswith("Enregistre dans le Ledger.")
    assert sealed_logs[0]["current_hash"] == ledger._last_hash  # the real hash actually chained


@pytest.mark.asyncio
async def test_gate8_sealed_never_logged_when_the_write_was_deadlettered(
    ledger: CryptographicAuditLedger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converse of the test above: a real, honest guarantee -- never claim '[AUDIT] ... Enregistre
    dans le Ledger.' for a write that was not, in fact, durably recorded (ledger._pool stays None here,
    the real dead-letter path)."""
    logged: list[str] = []
    original_info = ledger._log.info

    def _recording_info(event: str, **kwargs: object) -> None:
        logged.append(event)
        original_info(event, **kwargs)

    monkeypatch.setattr(ledger._log, "info", _recording_info)

    await ledger._handle_audit_event_message(_gate8_audit_message(order_id="gate8-EUR_USD-2"))

    assert ledger._last_hash == GENESIS_HASH  # never advanced -- deadlettered
    assert "audit_gate8_decision_sealed" not in logged

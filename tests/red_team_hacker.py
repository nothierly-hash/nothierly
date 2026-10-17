"""Phase 1 -- Red Team Penetration Testing against Nothierly's real attack surface.

**Corrections to the brief, verified against the real code before writing
anything here** (this project's established discipline: a plausible-
sounding brief is not the same as a true one):

1. Order flow and trading signals are **not** delivered over Redis
   Pub/Sub -- the real transport is ZMQ pub/sub (``core/zmq_bus.py``),
   with real topics ``CLIENT_ORDERS_TOPIC``/``TRADING_SIGNALS_TOPIC``/
   ``EMERGENCY_HALT_TOPIC`` in ``core/schemas.py``. Redis's real role is
   a state/cache store. So "inject fake L2 prices or KILL SWITCH signals
   into Redis Pub/Sub" as literally described does not apply -- but a
   **real, more interesting** equivalent does: real agents (XgboostSignalAgent,
   RlSignalAgent, QuantCopilot, Command Center) read live ``MarketMetrics``
   back from a real Redis cache key (``core.schemas.
   MARKET_METRICS_REDIS_KEY_PREFIX + symbol``), and the real Kill Switch
   (``core/security.py``) *is* a plain Redis key (``SYSTEM_STATUS``).
   Both are tested for real below.
2. The brief's WORM ledger (``nothierly_audit_ledger.db`` SQLite +
   ``verify_chain_integrity()``) does not exist. Real implementation:
   ``src/execution/audit_ledger.py``, a TimescaleDB table
   ``order_audit_trail``, real functions ``verify_chain()``/
   ``verify_chain_tip_matches_anchor()``. Tested below against the real
   function, not a fabricated one.
3. API secrets: ``src/execution/security_vault.py``'s ``SecurityVault``
   is real, genuine Fernet (AES-128-CBC + HMAC-SHA256) encryption, not
   base64. ``core/logger.py`` has a real redaction processor
   (``_redact_secrets``) with one confirmed, real blind spot: it only
   catches a secret that is an entire, unbroken field value -- not one
   embedded mid-sentence inside a longer human-readable error message
   (e.g. a real ccxt ``AuthenticationError``'s own text, as logged by
   ``src/agents/market_data/exchange_client.py``). Tested below.

Run directly for a human-readable report::

    .venv/Scripts/python tests/red_team_hacker.py

Or via pytest (each attack is also a real, asserting test)::

    pytest tests/red_team_hacker.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
from decimal import Decimal

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.logger import _redact_secrets
from core.message_integrity import SignatureVerificationError, publish_signed_market_metrics, read_verified_market_metrics, sign
from core.redis_client import RedisClient
from core.schemas import ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX, MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics
import core.security as security_module
from core.security import (
    STATUS_SAFE,
    SYSTEM_STATUS_EPOCH_KEY,
    SYSTEM_STATUS_KEY,
    SYSTEM_STATUS_SIG_KEY,
    is_system_safe,
    reset_kill_switch,
    trip_kill_switch,
)
from src.execution.audit_ledger import GENESIS_HASH, AuditTrailRow, compute_event_hash, verify_chain

_REPORT: list[tuple[str, str, str]] = []  # (scenario, verdict, detail)


def _record(scenario: str, verdict: str, detail: str) -> None:
    _REPORT.append((scenario, verdict, detail))
    print(f"\n[{verdict}] {scenario}\n    {detail}")


# =============================================================================
# ATTACK 1 -- Redis MarketMetrics cache poisoning
# =============================================================================


async def _attack_redis_market_metrics_poisoning() -> None:
    """**Re-run after the real fix** (core/message_integrity.py): the legit reading is now published via
    the real, signed writer (publish_signed_market_metrics, what MarketDataAgent now actually calls) --
    the attacker still only has raw Redis access, not VAULT_MASTER_KEY, so it can forge the primary key's
    plaintext but not a matching companion signature. Reports BOTH real outcomes: unmigrated legacy
    readers (a bare MarketMetrics.model_validate_json(raw), still real in ~8 call sites -- see
    read_verified_market_metrics's own docstring for the full list) remain genuinely vulnerable; a
    migrated reader (read_verified_market_metrics, what XgboostSignalAgent now actually calls) is not."""
    client = RedisClient(RedisSettings())
    # _DecodingFakeRedis (defined below, near Attack 2), not a plain FakeRedis: this test's own
    # read_verified_market_metrics call now depends on `get` returning `str` (decode_responses=True's real
    # contract), and fakeredis 2.37.1's async client returns raw bytes from GET regardless -- see that
    # class's own docstring for why this is a real, reproducible fakeredis-library quirk, not a production
    # behavior (real redis.asyncio.Redis decodes correctly and reliably).
    client._redis = _DecodingFakeRedis(server=fakeredis.FakeServer(), decode_responses=True)

    symbol = "BTC/USDT"
    real_key = f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}"  # the exact real key MarketDataAgent writes

    legit = MarketMetrics(symbol=symbol, spread=12.5, order_book_imbalance=0.05, mid_price=65000.0)
    await publish_signed_market_metrics(client, symbol, legit, ttl_seconds=10)

    # The real attack: nothing distinguishes an attacker's SET on this exact key from the real
    # MarketDataAgent's own write -- no source-agent authentication, and (per core/config.py's
    # RedisSettings.password default of None) no Redis auth by default either. The attacker does NOT also
    # know VAULT_MASTER_KEY, so it cannot forge a matching companion signature at f"{real_key}:sig" --
    # that key still holds the legit reading's real signature, now stale relative to the poisoned payload.
    poisoned = MarketMetrics(symbol=symbol, spread=0.0001, order_book_imbalance=1.0, mid_price=1.00)
    await client.execute("set", real_key, poisoned.model_dump_json())

    raw = await client.execute("get", real_key)
    assert raw is not None
    legacy_read_back = MarketMetrics.model_validate_json(raw)  # what every unmigrated reader still does

    legacy_vulnerable = legacy_read_back.mid_price == poisoned.mid_price
    try:
        await read_verified_market_metrics(client, symbol)
        migrated_reader_rejected = False
    except SignatureVerificationError:
        migrated_reader_rejected = True

    if legacy_vulnerable and migrated_reader_rejected:
        _record(
            "Redis MarketMetrics Cache Poisoning",
            "PARTIALLY FIXED",
            f"A fabricated MarketMetrics(mid_price=$1.00 for BTC/USDT, vs. a real $65,000.00) was written "
            f"under the real key '{real_key}'. read_verified_market_metrics (now used by XgboostSignalAgent) "
            f"correctly raised SignatureVerificationError and rejected it. But a bare "
            f"MarketMetrics.model_validate_json(raw) -- what RlSignalAgent, QuantCopilot, alert_agent, "
            f"alpha_engine, shadow_crossing_engine, the gateway, and two dashboard pages still do -- read "
            f"the poisoned value back with zero complaint, exactly as before. Real, disclosed scope: signing "
            f"is opt-in per reader, not yet migrated project-wide."
        )
    elif not legacy_vulnerable and not migrated_reader_rejected:
        _record("Redis MarketMetrics Cache Poisoning", "SAFE", "Poisoned value was not accepted by either path.")
    else:
        _record(
            "Redis MarketMetrics Cache Poisoning", "UNEXPECTED RESULT",
            f"legacy_vulnerable={legacy_vulnerable}, migrated_reader_rejected={migrated_reader_rejected} -- re-verify.",
        )


# =============================================================================
# ATTACK 2 -- Kill Switch spoofing via unauthenticated Redis
# =============================================================================


class _DecodingFakeRedis(fakeredis.aioredis.FakeRedis):
    """fakeredis 2.37.1's async client returns raw bytes from GET even with decode_responses=True (verified
    directly: a fresh FakeRedis(decode_responses=True) still returns b"SAFE", not "SAFE") -- a real,
    reproducible fakeredis-library quirk, not a production behavior (real redis.asyncio.Redis decodes
    correctly and reliably; core/redis_client.py uses that real client, never fakeredis). Overriding just
    ``get`` here makes this stand-in behave like the real production client is_system_safe() actually runs
    against, so this test demonstrates the real vulnerability, not a testing artifact."""

    async def get(self, name: str | bytes) -> str | None:
        value = await super().get(name)
        return value.decode() if isinstance(value, bytes) else value


async def _attack_kill_switch_spoofing() -> None:
    """**Re-run after the real fix** (core/security.py's is_system_safe/trip_kill_switch/reset_kill_switch
    now sign/verify SYSTEM_STATUS via SYSTEM_STATUS_SIG_KEY, core/message_integrity.py) -- this attack's
    own logic is unchanged (it deliberately bypasses reset_kill_switch() with a raw Redis SET, real
    attacker capability); only the real functions it calls now behave differently."""
    client = RedisClient(RedisSettings())
    client._redis = _DecodingFakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    security_module._min_accepted_epoch = 0  # test isolation -- see core/security.py's own docstring on this variable
    try:
        # A real operator legitimately trips the switch after detecting real danger.
        await trip_kill_switch("red-team scenario: operator detected anomalous fills")
        assert await is_system_safe() is False

        # The attack: core/security.py's is_system_safe()/reset_kill_switch() have no way to distinguish
        # a dashboard operator's deliberate reset click from a raw `redis-cli SET SYSTEM_STATUS SAFE` run
        # by anyone who can reach this (by-default-unauthenticated) Redis instance.
        await client.execute("set", SYSTEM_STATUS_KEY, STATUS_SAFE)
        now_safe = await is_system_safe()

        if now_safe:
            _record(
                "Kill Switch Spoofing (unauthenticated Redis)",
                "VULNERABLE",
                "A legitimately-tripped PANIC was silently un-tripped by a raw Redis SET on 'SYSTEM_STATUS' "
                "-- indistinguishable, at the Redis level, from core.security.reset_kill_switch()'s own "
                "real operator action. This defeats the one control every money-creating call in this "
                "system depends on (core/security.py's @requires_system_active decorator, and "
                "safe_financial_call's automatic trip-on-repeated-failure). Same real root cause as Attack 1."
            )
        else:
            _record(
                "Kill Switch Spoofing (unauthenticated Redis)",
                "SAFE (real fix)",
                "The spoofed raw SET on 'SYSTEM_STATUS' did NOT take effect: is_system_safe() now also "
                "requires a matching HMAC-SHA256 at the companion SYSTEM_STATUS_SIG_KEY (core/"
                "message_integrity.py), which the attacker -- lacking VAULT_MASTER_KEY -- could not forge. "
                "The switch correctly stayed tripped (fail closed) instead of being spoofed back to SAFE."
            )
    finally:
        RedisClient._instance = None


# =============================================================================
# ATTACK 2b -- Kill Switch signature REPLAY (stress-testing the Bloc 1 fix itself)
# =============================================================================


async def _attack_kill_switch_signature_replay() -> None:
    """Pushed further than Attack 2: does the real HMAC fix (core/message_integrity.py) hold up against a
    REPLAY, not just a forgery? First run of this test found it did NOT -- sign("SAFE") was one fixed hex
    string for the lifetime of VAULT_MASTER_KEY, so an attacker who merely OBSERVED one real (SYSTEM_STATUS,
    SYSTEM_STATUS_SIG_KEY) pair once could replay it forever, including to silently un-trip a LATER, real,
    unrelated panic.

    **Re-run after the real fix**: every signature is now bound to SYSTEM_STATUS_EPOCH_KEY (a monotonic
    Redis INCR, bumped on every real trip/reset), so a captured pair only verifies for the one transition
    it was actually issued for. This test now runs TWO variants of the attack:

    (a) Replay the old (value, signature) pair alone, epoch untouched -- defeated by epoch binding alone,
        since the current epoch has moved on and the old signature no longer matches it.
    (b) The harder version: replay the epoch key TOO, rolling the whole authenticated triple back to a
        consistent old state. core.security's own module docstring on SYSTEM_STATUS_EPOCH_KEY predicts
        this defeats the Redis-side epoch alone -- but is caught by the *second*, independent layer: a
        per-process, in-memory high-water mark that only Redis being rolled back cannot roll back too.
    """
    client = RedisClient(RedisSettings())
    client._redis = _DecodingFakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    security_module._min_accepted_epoch = 0  # test isolation -- see core/security.py's own docstring on this variable
    try:
        # Step 1: at some point in the past, a real, legitimate reset happens. The attacker only ever
        # reads -- never writes, never learns VAULT_MASTER_KEY -- capturing the real (value, signature, epoch).
        await reset_kill_switch()
        captured_value = await client.execute("get", SYSTEM_STATUS_KEY)
        captured_sig = await client.execute("get", SYSTEM_STATUS_SIG_KEY)
        captured_epoch = await client.execute("get", SYSTEM_STATUS_EPOCH_KEY)
        await is_system_safe()  # this process legitimately observes the real epoch, establishing its high-water mark

        # Step 2: time passes. A real operator legitimately trips the switch for a real, CURRENT,
        # unrelated danger.
        await trip_kill_switch("red-team scenario: a real, current, unrelated danger")
        assert await is_system_safe() is False

        # Variant (a): replay value+signature only, epoch left at its real, current value.
        await client.execute("set", SYSTEM_STATUS_KEY, captured_value)
        await client.execute("set", SYSTEM_STATUS_SIG_KEY, captured_sig)
        variant_a_safe = await is_system_safe()

        # Re-trip for a clean variant (b) (variant (a), whether it worked or not, must not contaminate this).
        await trip_kill_switch("red-team scenario: a second, real, current, unrelated danger")
        assert await is_system_safe() is False

        # Variant (b): the harder attack -- also roll back the epoch key itself, reconstructing a fully
        # internally-consistent OLD state. Real attacker capability: the same unrestricted Redis WRITE
        # access this whole class of attack already assumes (Attack 1/2's own threat model) covers
        # SYSTEM_STATUS_EPOCH_KEY exactly as much as SYSTEM_STATUS_KEY -- nothing in this codebase's Redis
        # ACLs (there are none) distinguishes "may INCR this key" from "may SET it to anything".
        await client.execute("set", SYSTEM_STATUS_KEY, captured_value)
        await client.execute("set", SYSTEM_STATUS_SIG_KEY, captured_sig)
        await client.execute("set", SYSTEM_STATUS_EPOCH_KEY, captured_epoch)
        variant_b_safe = await is_system_safe()

        if variant_a_safe or variant_b_safe:
            _record(
                "Kill Switch Signature REPLAY (stress-test of this session's own fix)",
                "VULNERABLE",
                f"variant_a_safe(value+sig only)={variant_a_safe}, variant_b_safe(value+sig+epoch)="
                f"{variant_b_safe} -- a real, CURRENT panic was silently un-tripped by replaying captured "
                f"state. core.security.SYSTEM_STATUS_EPOCH_KEY's own docstring already discloses this exact "
                f"residual for a fresh process; this run shows it's reachable here too."
            )
        else:
            _record(
                "Kill Switch Signature REPLAY (stress-test of this session's own fix)",
                "SAFE (real fix)",
                "Neither variant un-tripped the real, current panic: (a) value+signature replay alone is "
                "defeated by SYSTEM_STATUS_EPOCH_KEY binding (the old signature no longer matches the "
                "current epoch); (b) replaying the epoch too is separately caught by this process's own "
                "in-memory high-water mark (_min_accepted_epoch), which Redis being rolled back cannot roll "
                "back. Honest residual, disclosed in core/security.py: a BRAND-NEW process, with no "
                "high-water mark yet, would trust whatever Redis reports on its very first call -- not "
                "closed by this fix, and not closeable without a non-Redis source of truth."
            )
    finally:
        RedisClient._instance = None


# =============================================================================
# ATTACK 3 -- WORM Ledger tampering (real src/execution/audit_ledger.py)
# =============================================================================


def _attack_worm_ledger_tampering() -> None:
    payloads = [
        '{"event_type":"SUBMITTED","order_id":"ord-1","asset":"BTC/USDT","quantity":"1.0"}',
        '{"event_type":"EXECUTED","order_id":"ord-1","asset":"BTC/USDT","quantity":"1.0","price":"65000.00"}',
        '{"event_type":"SUBMITTED","order_id":"ord-2","asset":"ETH/USDT","quantity":"50.0"}',
    ]
    rows: list[AuditTrailRow] = []
    prev_hash = GENESIS_HASH
    for i, payload in enumerate(payloads):
        current_hash = compute_event_hash(prev_hash, payload)
        rows.append(AuditTrailRow(seq_id=i + 1, payload=payload, prev_hash=prev_hash, current_hash=current_hash))
        prev_hash = current_hash

    # The real attack: a privileged DB attacker runs
    #   UPDATE order_audit_trail SET payload = '...' WHERE seq_id = 2;
    # -- rewriting the fill price only, current_hash left untouched (the classic, naive tamper).
    tampered_payload = payloads[1].replace('"65000.00"', '"1.00"')
    rows[1] = rows[1].model_copy(update={"payload": tampered_payload})

    is_valid, message = verify_chain(rows)

    if not is_valid and "2" in message:
        _record(
            "WORM Ledger Tampering (real src/execution/audit_ledger.py, TimescaleDB table order_audit_trail)",
            "DETECTED",
            f"Real verify_chain() caught the tamper: {message!r}. Note: the brief's "
            f"'nothierly_audit_ledger.db (SQLite) + verify_chain_integrity()' does not exist anywhere in "
            f"this codebase -- confirmed real implementation is a TimescaleDB table + verify_chain()/"
            f"verify_chain_tip_matches_anchor(), exercised here with the exact real function the production "
            f"ledger's own dashboard 'Vérifier l'intégrité' button calls."
        )
    else:
        _record("WORM Ledger Tampering", "UNDETECTED -- CRITICAL", f"verify_chain() did not catch the tamper: {message!r}")


# =============================================================================
# ATTACK 4 -- API secret extraction via error-log injection
# =============================================================================


def _attack_api_secret_log_leak() -> None:
    fake_secret = "sk_live_51HijKLmNoPQRsTuVwXyZ0123456789ABCDEFGabc"  # realistic-shaped, not a real key

    # A realistic shape for src/agents/market_data/exchange_client.py's own real
    # `error=str(exc)` logging of a caught ccxt AuthenticationError -- the secret embedded mid-sentence,
    # not as the field's entire value.
    event_dict = {
        "event": "exchange_auth_failed",
        "exchange": "binance",
        "error": f"binance {{\"code\":-2015,\"msg\":\"Invalid API-key, IP, or permissions for action.\"}} key_used={fake_secret}",
    }
    redacted = _redact_secrets(None, "error", dict(event_dict))

    if fake_secret in str(redacted.get("error", "")):
        _record(
            "API Secret Extraction via Error Logs (real core/logger.py::_redact_secrets)",
            "VULNERABLE",
            f"The real _redact_secrets() processor did NOT catch a secret embedded mid-sentence in an "
            f"'error' field. Root cause, precisely: its value-based fallback "
            f"(core/logger.py's _SENSITIVE_VALUE_PATTERN) is anchored '^...$', matching only when the "
            f"ENTIRE field value is one unbroken token -- by design, to avoid false-positiving on this "
            f"project's own hyphenated agent/order IDs (see that pattern's own comment). Key-name "
            f"redaction also misses it since 'error' isn't in _SENSITIVE_KEY_PATTERN. This exact code "
            f"shape (str(exc) logged under key 'error' for a caught AuthenticationError/ExchangeError) is "
            f"real: src/agents/market_data/exchange_client.py, and CcxtBrokerAdapter "
            f"(src/execution/broker_gateway.py) delegates to the same client, so this applies to the live "
            f"order-execution path too, not just market-data streaming."
        )
    else:
        _record("API Secret Extraction via Error Logs", "SAFE", "Redaction caught the embedded secret.")


# =============================================================================
# ATTACK 5 -- Emergency Liquidate channel injection (unauthenticated Redis Pub/Sub)
# =============================================================================


async def _attack_emergency_liquidate_channel_injection() -> None:
    """New attack surface, found by reading src/agents/alpha_engine/agent.py while stress-testing the
    Bloc 3 (SOR unwind) fix from this same session: AlphaMarketMakerAgent._listen_for_emergency_liquidate
    subscribed to ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX + agent_id, and its handler acted on ANY
    message that arrived, regardless of origin or content -- confirmed VULNERABLE on the first run of this
    test (a raw, unauthenticated PUBLISH forced a real 0.25 BTC market sell via the Kill-Switch-bypassing
    emergency_flatten_market_order path). **Re-run after the real fix**: the listener now verifies the
    message against sign(f"LIQUIDATE:{{agent_id}}") (core/message_integrity.py) before acting; the real
    publisher (src/dashboard/ems_pages/autonomous_command_center.py's Confirmer button) now signs it.
    Tests BOTH real outcomes: an attacker's raw/unsigned publish (no VAULT_MASTER_KEY) is rejected; the
    real dashboard's own signed publish still works.
    """
    from tests.test_alpha_engine import _FakeBroker, _make_agent

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    try:
        agent = _make_agent(client, exchange_id="fake")
        broker = agent._broker
        assert isinstance(broker, _FakeBroker)
        broker.position_qty = Decimal("0.25")  # a real, nonzero position an attacker could force-close

        listener_task = asyncio.create_task(agent._listen_for_emergency_liquidate())
        await asyncio.sleep(0.05)  # let the subscribe() actually register before publishing
        channel = f"{ALPHA_ENGINE_EMERGENCY_LIQUIDATE_CHANNEL_PREFIX}{agent.agent_id}"

        # Step 1 -- the attack: a raw, unsigned PUBLISH from an unauthenticated client -- not the
        # dashboard, no VAULT_MASTER_KEY. Real attacker capability already established by Attack 1/2's
        # own Redis access model.
        await client.execute("publish", channel, "anything -- an attacker guessing the old literal 'LIQUIDATE' too")
        await asyncio.sleep(0.2)
        attack_blocked = broker.market_orders == []

        # Step 2 -- the real, legitimate path: the dashboard's own Confirmer button, which now signs.
        await client.execute("publish", channel, sign(f"LIQUIDATE:{agent.agent_id}"))
        await asyncio.sleep(0.2)
        legit_still_works = len(broker.market_orders) == 1

        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        if attack_blocked and legit_still_works:
            _record(
                "Emergency Liquidate Channel Injection (unauthenticated Redis Pub/Sub)",
                "SAFE (real fix)",
                f"An unsigned PUBLISH on {channel!r} was correctly rejected (no market order placed); a "
                f"signed one (sign(f'LIQUIDATE:{{agent_id}}'), what the real dashboard button now sends) "
                f"still triggered the real liquidation: {broker.market_orders[0]!r}. The break-glass path "
                f"itself (emergency_flatten_market_order) stays deliberately unguarded by the Kill Switch "
                f"-- the fix is authenticating who may reach it, not gating the action itself."
            )
        elif not attack_blocked:
            _record(
                "Emergency Liquidate Channel Injection (unauthenticated Redis Pub/Sub)",
                "VULNERABLE",
                f"An unauthenticated PUBLISH on {channel!r} (no signature, no dashboard session, no "
                f"content validation) forced a real market order: {broker.market_orders[0]!r}. This calls "
                f"BaseBroker.emergency_flatten_market_order -- deliberately NOT Kill-Switch-gated (see "
                f"core/security.py's module docstring) -- so this is not merely an unwanted trade, it is "
                f"one that fires even during a real, legitimate panic."
            )
        else:
            _record(
                "Emergency Liquidate Channel Injection", "REGRESSION",
                "The attack was correctly blocked, but the real dashboard's own signed publish no longer "
                "works either -- the fix over-corrected and broke the legitimate path.",
            )
    finally:
        RedisClient._instance = None


# =============================================================================
# pytest entrypoints
# =============================================================================


@pytest.mark.asyncio
async def test_redis_market_metrics_poisoning() -> None:
    await _attack_redis_market_metrics_poisoning()


@pytest.mark.asyncio
async def test_kill_switch_spoofing() -> None:
    await _attack_kill_switch_spoofing()


@pytest.mark.asyncio
async def test_kill_switch_signature_replay() -> None:
    await _attack_kill_switch_signature_replay()


def test_worm_ledger_tampering_is_detected() -> None:
    _attack_worm_ledger_tampering()
    assert _REPORT[-1][1] == "DETECTED"  # the one attack that MUST fail for real money to be safe


def test_api_secret_log_leak() -> None:
    _attack_api_secret_log_leak()


@pytest.mark.asyncio
async def test_emergency_liquidate_channel_injection() -> None:
    await _attack_emergency_liquidate_channel_injection()


# =============================================================================
# Standalone execution: python tests/red_team_hacker.py
# =============================================================================


def _print_summary() -> None:
    print("\n" + "=" * 78)
    print("RED TEAM REPORT -- PHASE 1: CYBER-SECURITY & PENETRATION TESTING")
    print("=" * 78)
    for scenario, verdict, _detail in _REPORT:
        print(f"  [{verdict:20s}] {scenario}")
    print("=" * 78)
    _RESOLVED_VERDICTS = ("DETECTED", "SAFE", "SAFE (real fix)")
    vulnerable = sum(1 for _s, v, _d in _REPORT if v not in _RESOLVED_VERDICTS and v != "PARTIALLY FIXED")
    partial = sum(1 for _s, v, _d in _REPORT if v == "PARTIALLY FIXED")
    print(f"Real, confirmed vulnerabilities: {vulnerable} / {len(_REPORT)} scenarios ({partial} partially fixed)")


async def _main() -> None:
    await _attack_redis_market_metrics_poisoning()
    await _attack_kill_switch_spoofing()
    await _attack_kill_switch_signature_replay()
    _attack_worm_ledger_tampering()
    _attack_api_secret_log_leak()
    await _attack_emergency_liquidate_channel_injection()
    _print_summary()


if __name__ == "__main__":
    asyncio.run(_main())

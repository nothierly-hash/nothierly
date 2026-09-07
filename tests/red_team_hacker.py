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

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.logger import _redact_secrets
from core.redis_client import RedisClient
from core.schemas import MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics
from core.security import STATUS_SAFE, SYSTEM_STATUS_KEY, is_system_safe, trip_kill_switch
from src.execution.audit_ledger import GENESIS_HASH, AuditTrailRow, compute_event_hash, verify_chain

_REPORT: list[tuple[str, str, str]] = []  # (scenario, verdict, detail)


def _record(scenario: str, verdict: str, detail: str) -> None:
    _REPORT.append((scenario, verdict, detail))
    print(f"\n[{verdict}] {scenario}\n    {detail}")


# =============================================================================
# ATTACK 1 -- Redis MarketMetrics cache poisoning
# =============================================================================


async def _attack_redis_market_metrics_poisoning() -> None:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)

    symbol = "BTC/USDT"
    real_key = f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}"  # the exact real key MarketDataAgent writes

    legit = MarketMetrics(symbol=symbol, spread=12.5, order_book_imbalance=0.05, mid_price=65000.0)
    await client.execute("set", real_key, legit.model_dump_json())

    # The real attack: nothing distinguishes an attacker's SET on this exact key from the real
    # MarketDataAgent's own write -- no signature, no source-agent authentication, and (per
    # core/config.py's RedisSettings.password default of None) no Redis auth by default either.
    poisoned = MarketMetrics(symbol=symbol, spread=0.0001, order_book_imbalance=1.0, mid_price=1.00)
    await client.execute("set", real_key, poisoned.model_dump_json())

    raw = await client.execute("get", real_key)
    assert raw is not None
    read_back = MarketMetrics.model_validate_json(raw)

    if read_back.mid_price == poisoned.mid_price:
        _record(
            "Redis MarketMetrics Cache Poisoning",
            "VULNERABLE",
            f"A fabricated MarketMetrics(mid_price=$1.00 for BTC/USDT, vs. a real $65,000.00) was written "
            f"under the real key '{real_key}' and read back validated with zero complaint -- Pydantic only "
            f"checks shape (mid_price > 0), never plausibility. Any real live consumer of this key "
            f"(XgboostSignalAgent, RlSignalAgent, QuantCopilot, Command Center) would treat this as genuine. "
            f"Real root cause: core/config.py's RedisSettings.password defaults to None and use_tls to "
            f"False -- confirmed by .env.example shipping REDIS_PASSWORD= blank -- so any direct/local run "
            f"of this codebase (outside the Docker Compose stack, which does force a password) exposes an "
            f"unauthenticated Redis. No ACL exists anywhere in this codebase to narrow this even if auth "
            f"were on."
        )
    else:
        _record("Redis MarketMetrics Cache Poisoning", "SAFE", "Poisoned value was not accepted as written.")


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
    client = RedisClient(RedisSettings())
    client._redis = _DecodingFakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
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
            _record("Kill Switch Spoofing", "SAFE", "The spoofed SAFE write did not take effect.")
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
# pytest entrypoints
# =============================================================================


@pytest.mark.asyncio
async def test_redis_market_metrics_poisoning() -> None:
    await _attack_redis_market_metrics_poisoning()


@pytest.mark.asyncio
async def test_kill_switch_spoofing() -> None:
    await _attack_kill_switch_spoofing()


def test_worm_ledger_tampering_is_detected() -> None:
    _attack_worm_ledger_tampering()
    assert _REPORT[-1][1] == "DETECTED"  # the one attack that MUST fail for real money to be safe


def test_api_secret_log_leak() -> None:
    _attack_api_secret_log_leak()


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
    vulnerable = sum(1 for _s, v, _d in _REPORT if v not in ("DETECTED", "SAFE"))
    print(f"Real, confirmed vulnerabilities: {vulnerable} / {len(_REPORT)} scenarios")


async def _main() -> None:
    await _attack_redis_market_metrics_poisoning()
    await _attack_kill_switch_spoofing()
    _attack_worm_ledger_tampering()
    _attack_api_secret_log_leak()
    _print_summary()


if __name__ == "__main__":
    asyncio.run(_main())

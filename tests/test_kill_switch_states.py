"""Real, executable tests for core/security.py's "Resilient Kill Switch" addition:
STATUS_LATCHED_DATA, latch_data_loss, auto_recover_from_data_loss, manual_authenticated_reset.

See core/security.py's own module-level comment on STATUS_LATCHED_DATA for the real gap this
closes: a 24h Chaos Shadow Run found a single 30s WebSocket blip latching the entire system on
PANIC, indistinguishable from a flash crash. These tests are the first in this codebase to
exercise the real signing/verification round trip end to end against real fakeredis (the existing
kill-switch tests in tests/test_supervisor_agent_kill_switch_recovery.py monkeypatch trip_kill_switch
itself) -- which is how the bytes-vs-str fakeredis decoding bug this file also covers was found.
"""

from __future__ import annotations

import time
from typing import Generator

import fakeredis.aioredis
import pytest

from core.config import RedisSettings
from core.message_integrity import sign
from core.redis_client import RedisClient
from core.security import (
    STATUS_LATCHED_DATA,
    STATUS_PANIC,
    STATUS_SAFE,
    auto_recover_from_data_loss,
    is_system_safe,
    latch_data_loss,
    manual_authenticated_reset,
    reset_kill_switch,
    trip_kill_switch,
)


@pytest.fixture
def redis_client(monkeypatch: pytest.MonkeyPatch) -> Generator[RedisClient, None, None]:
    """Real fakeredis-backed RedisClient, installed as the process singleton -- core.security's
    functions all resolve RedisClient.get_instance() independently, same discipline as
    tests/test_supervisor_agent_kill_switch_recovery.py's identical fixture.

    Also resets core.security's own real anti-replay high-water mark
    (_min_accepted_epoch) to 0: that module-level global deliberately
    never resets in production ("resets only on process restart, never
    manually" -- see SYSTEM_STATUS_EPOCH_KEY's own docstring), but each
    test here gets a brand-new fakeredis instance whose own epoch
    counter restarts at 1 -- without this, a later test's real epoch=1
    would be rejected as a false "rollback" against a high-water mark a
    completely different (fake) backing store left behind.
    """
    monkeypatch.setattr("core.security._min_accepted_epoch", 0)
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


def _decode(value: str | bytes | None) -> str | None:
    """Real, disclosed fakeredis quirk (see core/security.py's own _decode_redis_str docstring) --
    this test reads SYSTEM_STATUS directly, bypassing core.security's own normalization."""
    return value.decode("utf-8") if isinstance(value, bytes) else value


# =============================================================================
# latch_data_loss / STATUS_LATCHED_DATA
# =============================================================================


@pytest.mark.asyncio
async def test_latch_data_loss_blocks_trading_same_as_panic(redis_client: RedisClient) -> None:
    await latch_data_loss("real test: simulated total feed loss")
    assert await is_system_safe() is False

    raw_status = await redis_client.execute("get", "SYSTEM_STATUS")
    assert _decode(raw_status) == STATUS_LATCHED_DATA


# =============================================================================
# auto_recover_from_data_loss
# =============================================================================


@pytest.mark.asyncio
async def test_auto_recover_clears_a_real_latched_data_state(redis_client: RedisClient) -> None:
    await latch_data_loss("real test: simulated total feed loss")
    recovered = await auto_recover_from_data_loss()
    assert recovered is True
    assert await is_system_safe() is True


@pytest.mark.asyncio
async def test_auto_recover_never_clears_a_real_panic(redis_client: RedisClient) -> None:
    await trip_kill_switch("real test: simulated flash crash")
    recovered = await auto_recover_from_data_loss()
    assert recovered is False
    assert await is_system_safe() is False
    raw_status = await redis_client.execute("get", "SYSTEM_STATUS")
    assert _decode(raw_status) == STATUS_PANIC


@pytest.mark.asyncio
async def test_auto_recover_is_a_real_no_op_when_already_safe(redis_client: RedisClient) -> None:
    recovered = await auto_recover_from_data_loss()
    assert recovered is False
    assert await is_system_safe() is True


# =============================================================================
# manual_authenticated_reset
# =============================================================================


@pytest.mark.asyncio
async def test_manual_authenticated_reset_clears_panic_with_a_real_valid_signature(
    redis_client: RedisClient,
) -> None:
    await trip_kill_switch("real test: simulated flash crash")
    assert await is_system_safe() is False

    timestamp = time.time()
    signature = sign(f"RESET:{timestamp}")
    accepted = await manual_authenticated_reset(timestamp, signature)

    assert accepted is True
    assert await is_system_safe() is True


@pytest.mark.asyncio
async def test_manual_authenticated_reset_rejects_a_stale_timestamp(redis_client: RedisClient) -> None:
    await trip_kill_switch("real test: simulated flash crash")

    stale_timestamp = time.time() - 11.0  # past the real 10s anti-replay tolerance
    signature = sign(f"RESET:{stale_timestamp}")
    accepted = await manual_authenticated_reset(stale_timestamp, signature)

    assert accepted is False
    assert await is_system_safe() is False


@pytest.mark.asyncio
async def test_manual_authenticated_reset_accepts_the_exact_tolerance_boundary(
    redis_client: RedisClient,
) -> None:
    await trip_kill_switch("real test: simulated flash crash")

    boundary_timestamp = time.time() - 9.9  # inside the real 10s tolerance
    signature = sign(f"RESET:{boundary_timestamp}")
    accepted = await manual_authenticated_reset(boundary_timestamp, signature)

    assert accepted is True


@pytest.mark.asyncio
async def test_manual_authenticated_reset_rejects_an_invalid_signature(redis_client: RedisClient) -> None:
    await trip_kill_switch("real test: simulated flash crash")

    timestamp = time.time()
    accepted = await manual_authenticated_reset(timestamp, "0" * 64)

    assert accepted is False
    assert await is_system_safe() is False


@pytest.mark.asyncio
async def test_manual_authenticated_reset_rejects_a_signature_for_a_different_timestamp(
    redis_client: RedisClient,
) -> None:
    """A real, captured (timestamp, signature) pair must not verify against a different timestamp --
    the core anti-replay property, not just a stale-clock check."""
    await trip_kill_switch("real test: simulated flash crash")

    real_timestamp = time.time()
    real_signature = sign(f"RESET:{real_timestamp}")
    forged_timestamp = real_timestamp + 1.0  # still within tolerance of "now", but not what was signed

    accepted = await manual_authenticated_reset(forged_timestamp, real_signature)

    assert accepted is False
    assert await is_system_safe() is False


@pytest.mark.asyncio
async def test_manual_authenticated_reset_also_clears_latched_data(redis_client: RedisClient) -> None:
    await latch_data_loss("real test: simulated total feed loss")

    timestamp = time.time()
    signature = sign(f"RESET:{timestamp}")
    accepted = await manual_authenticated_reset(timestamp, signature)

    assert accepted is True
    assert await is_system_safe() is True


@pytest.mark.asyncio
async def test_manual_authenticated_reset_is_a_real_no_op_effect_when_already_safe(
    redis_client: RedisClient,
) -> None:
    timestamp = time.time()
    signature = sign(f"RESET:{timestamp}")
    accepted = await manual_authenticated_reset(timestamp, signature)

    assert accepted is True
    assert await is_system_safe() is True


@pytest.mark.asyncio
async def test_reset_kill_switch_still_works_unchanged(redis_client: RedisClient) -> None:
    """The existing, dashboard-session-gated reset path must be completely untouched by this addition."""
    await trip_kill_switch("real test: simulated flash crash")
    assert await is_system_safe() is False
    await reset_kill_switch()
    assert await is_system_safe() is True

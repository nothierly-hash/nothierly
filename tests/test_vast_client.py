"""Real, executable tests for src/mlops/vast_client.py.

Two disciplines combined:

1. :func:`summarize_instance_costs` is a pure function over already-
   fetched instance dicts (the same "pure functions, real regression
   tests without live I/O" discipline as ``tests/test_eod_reconciliator.py``).
   Real Vast.ai field names used here (``id``, ``dph_total``,
   ``actual_status``, ``start_date``, ``gpu_name``) match the ones
   :meth:`~src.mlops.vast_client.VastAIClient.search_gpu` already reads
   from the real API elsewhere in this module.
2. The master switch (:func:`~src.mlops.vast_client.is_vast_ai_enabled`,
   :func:`~src.mlops.vast_client.requires_vast_ai_enabled`) and the real
   price-ceiling circuit breaker in :meth:`~src.mlops.vast_client.
   VastAIClient.rent_instance` -- Redis faked with ``fakeredis``, same
   ``redis_client`` fixture convention as ``tests/test_alert_agent.py``
   (installed as the real process singleton, since
   ``is_vast_ai_enabled`` deliberately goes through
   ``RedisClient.get_instance()``, never a passed-in client, for the
   same "one true process-wide flag" reasoning as the global Kill
   Switch). No real network call anywhere here: ``VastAIClient._request``
   is monkeypatched at the boundary, the same "fake I/O only at the
   edge" discipline as every other test in this suite.
"""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from core.config import get_settings
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import VAST_AI_ENABLED_REDIS_KEY
from src.mlops.vast_client import (
    VastAIClient,
    VastAIDisabledError,
    VastAIPriceLimitExceededError,
    is_vast_ai_enabled,
    summarize_instance_costs,
)

_NOW = 1_000_000.0  # an arbitrary, fixed "now" for deterministic accrual math


def test_empty_instance_list_returns_a_real_zeroed_summary() -> None:
    summary = summarize_instance_costs([], now=_NOW)
    assert summary.running_instance_count == 0
    assert summary.total_burn_rate_usd_per_hour == 0.0
    assert summary.total_accrued_cost_usd == 0.0
    assert summary.per_instance == []


def test_burn_rate_only_sums_running_instances() -> None:
    instances = [
        {"id": 1, "dph_total": 0.50, "actual_status": "running", "start_date": _NOW - 3600.0},
        {"id": 2, "dph_total": 0.30, "actual_status": "exited", "start_date": _NOW - 3600.0},
        {"id": 3, "dph_total": 0.20, "actual_status": "running", "start_date": _NOW - 3600.0},
    ]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.running_instance_count == 2
    assert summary.total_burn_rate_usd_per_hour == pytest.approx(0.70)


def test_accrued_cost_is_real_dph_times_real_elapsed_hours() -> None:
    """2 real hours elapsed at $0.40/hr on one instance -> $0.80 real accrued cost."""
    instances = [{"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": _NOW - 2 * 3600.0}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.total_accrued_cost_usd == pytest.approx(0.80)
    assert summary.per_instance[0].accrued_cost_usd == pytest.approx(0.80)


def test_accrued_cost_sums_across_every_instance_regardless_of_status() -> None:
    """Accrued (unlike burn rate) is real spend-to-date -- an exited instance still cost real money while it ran."""
    instances = [
        {"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": _NOW - 3600.0},
        {"id": 2, "dph_total": 0.20, "actual_status": "exited", "start_date": _NOW - 3600.0},
    ]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.total_accrued_cost_usd == pytest.approx(0.60)


def test_missing_start_date_yields_zero_accrued_not_a_guess() -> None:
    """No real start_date means nothing real to compute accrual from -- 0.0, never a fabricated estimate."""
    instances = [{"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": None}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.per_instance[0].accrued_cost_usd == 0.0
    assert summary.total_accrued_cost_usd == 0.0


def test_missing_dph_total_defaults_to_zero_not_an_error() -> None:
    instances = [{"id": 1, "actual_status": "running", "start_date": _NOW - 3600.0}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.per_instance[0].dph_total == 0.0


def test_per_instance_rows_carry_real_gpu_name_and_status() -> None:
    instances = [{"id": 42, "dph_total": 0.55, "actual_status": "running", "start_date": _NOW, "gpu_name": "RTX_4090"}]
    summary = summarize_instance_costs(instances, now=_NOW)
    row = summary.per_instance[0]
    assert row.instance_id == 42
    assert row.gpu_name == "RTX_4090"
    assert row.actual_status == "running"


# =============================================================================
# Master switch + price-ceiling circuit breaker -- see this file's own module
# docstring, point 2.
# =============================================================================


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    """A real RedisClient over a real in-memory fakeredis server, installed as the process singleton.

    Same rationale/pattern as ``tests/test_alert_agent.py``'s identically-
    named fixture: ``is_vast_ai_enabled`` deliberately always goes
    through ``RedisClient.get_instance()``, so this is what makes its
    assertions actually observe what each test sets up.
    """
    from core.config import RedisSettings

    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    RedisClient._instance = client
    yield client
    RedisClient._instance = None


@pytest.fixture(autouse=True)
def _reset_vastai_settings() -> Generator[None, None, None]:
    """Every test in this section gets a clean, default VastAiSettings -- never leaks into another test."""
    settings = get_settings()
    original_enabled = settings.vastai.enabled
    original_limit = settings.vastai.max_hourly_price_limit_usd
    yield
    settings.vastai.enabled = original_enabled
    settings.vastai.max_hourly_price_limit_usd = original_limit


def _client() -> VastAIClient:
    """A VastAIClient with a fake, non-``None`` session -- ``_request`` itself is monkeypatched per-test, so no real connect() is needed."""
    client = VastAIClient(api_key="fake-key-for-tests")
    client._session = AsyncMock()  # only needs to be non-None; _request is replaced directly below
    return client


@pytest.mark.asyncio
async def test_master_switch_defaults_to_disabled_with_no_redis_key_and_default_settings(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = False
    assert await is_vast_ai_enabled() is False


@pytest.mark.asyncio
async def test_master_switch_falls_back_to_env_settings_when_redis_key_never_written(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = True
    assert await is_vast_ai_enabled() is True


@pytest.mark.asyncio
async def test_master_switch_redis_key_overrides_env_settings_live(redis_client: RedisClient) -> None:
    """The whole point of the Redis key: the dashboard toggle wins over the .env cold-start default, live."""
    get_settings().vastai.enabled = False
    await redis_client.execute("set", VAST_AI_ENABLED_REDIS_KEY, "true")
    assert await is_vast_ai_enabled() is True

    await redis_client.execute("set", VAST_AI_ENABLED_REDIS_KEY, "false")
    assert await is_vast_ai_enabled() is False


@pytest.mark.asyncio
async def test_master_switch_fails_closed_when_redis_is_unreachable() -> None:
    """A fake, already-installed singleton whose execute() raises -- never a real network attempt.

    Deliberately does NOT set ``RedisClient._instance = None`` and let
    ``get_instance()`` construct a real client: on this dev machine that
    would retry a real TCP connect with real (slow, unbounded)
    exponential backoff -- the exact pytest-hang-at-a-blocking-I/O
    pattern this project's own pyproject.toml already documents fighting
    elsewhere. Installing a fake whose ``execute`` raises
    ``RedisUnavailableError`` directly exercises the same fail-closed
    code path in milliseconds.
    """
    from core.config import RedisSettings

    class _AlwaysUnavailable(RedisClient):
        async def execute(self, command: str, *args: object, **kwargs: object) -> object:
            raise RedisUnavailableError("simulated: Redis unreachable")

    fake = _AlwaysUnavailable(RedisSettings())
    RedisClient._instance = fake
    try:
        get_settings().vastai.enabled = True  # even "enabled" by config must not matter when Redis can't be reached
        assert await is_vast_ai_enabled() is False
    finally:
        RedisClient._instance = None


@pytest.mark.asyncio
async def test_search_gpu_blocked_when_master_switch_disabled(redis_client: RedisClient) -> None:
    get_settings().vastai.enabled = False
    client = _client()
    with pytest.raises(VastAIDisabledError, match="Vast.ai Access is disabled"):
        await client.search_gpu()


@pytest.mark.asyncio
async def test_rent_instance_blocked_when_master_switch_disabled(redis_client: RedisClient) -> None:
    get_settings().vastai.enabled = False
    client = _client()
    with pytest.raises(VastAIDisabledError):
        await client.rent_instance(123, ssh_public_key="ssh-ed25519 AAAA...")


@pytest.mark.asyncio
async def test_start_instance_blocked_when_master_switch_disabled(redis_client: RedisClient) -> None:
    get_settings().vastai.enabled = False
    client = _client()
    with pytest.raises(VastAIDisabledError):
        await client.start_instance(123)


@pytest.mark.asyncio
async def test_stop_instance_is_never_blocked_by_the_master_switch(redis_client: RedisClient) -> None:
    """Cost-reducing calls must keep working even with the module 'disabled' -- see requires_vast_ai_enabled's own docstring."""
    get_settings().vastai.enabled = False
    client = _client()
    client._request = AsyncMock(return_value={"success": True})  # type: ignore[method-assign]
    await client.stop_instance(123)
    client._request.assert_awaited_once_with("PUT", "/instances/123/", json_body={"state": "stopped"})


@pytest.mark.asyncio
async def test_destroy_instance_is_never_blocked_by_the_master_switch(redis_client: RedisClient) -> None:
    get_settings().vastai.enabled = False
    client = _client()
    client._request = AsyncMock(return_value={"success": True})  # type: ignore[method-assign]
    await client.destroy_instance(123)
    client._request.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_instances_is_never_blocked_by_the_master_switch(redis_client: RedisClient) -> None:
    get_settings().vastai.enabled = False
    client = _client()
    client._request = AsyncMock(return_value={"instances": []})  # type: ignore[method-assign]
    result = await client.list_instances()
    assert result == []


@pytest.mark.asyncio
async def test_rent_instance_refuses_when_current_price_exceeds_configured_ceiling(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = True
    get_settings().vastai.max_hourly_price_limit_usd = 0.50
    client = _client()
    client._request = AsyncMock(return_value={"offers": [{"id": 999, "dph_total": 1.25}]})  # type: ignore[method-assign]

    with pytest.raises(VastAIPriceLimitExceededError, match=r"\$1\.2500/hr exceeds"):
        await client.rent_instance(999, ssh_public_key="ssh-ed25519 AAAA...")

    # Only the price-verification GET happened -- the real rent PUT must never fire once refused.
    client._request.assert_awaited_once()
    assert client._request.await_args is not None
    assert client._request.await_args.args[0] == "GET"


@pytest.mark.asyncio
async def test_rent_instance_proceeds_when_current_price_is_within_the_configured_ceiling(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = True
    get_settings().vastai.max_hourly_price_limit_usd = 2.00

    async def _fake_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "GET":
            return {"offers": [{"id": 999, "dph_total": 0.30}]}
        return {"new_contract": 555}

    client = _client()
    client._request = AsyncMock(side_effect=_fake_request)  # type: ignore[method-assign]

    instance_id = await client.rent_instance(999, ssh_public_key="ssh-ed25519 AAAA...")
    assert instance_id == 555
    assert client._request.await_count == 2  # the price-verification GET, then the real rent PUT


@pytest.mark.asyncio
async def test_rent_instance_refuses_when_ask_no_longer_exists_for_price_verification(
    redis_client: RedisClient,
) -> None:
    """A ceiling is configured but the offer vanished -- 'cannot verify' must refuse, never 'no limit'."""
    get_settings().vastai.enabled = True
    get_settings().vastai.max_hourly_price_limit_usd = 1.00
    client = _client()
    client._request = AsyncMock(return_value={"offers": []})  # type: ignore[method-assign]

    with pytest.raises(VastAIPriceLimitExceededError, match="could not verify"):
        await client.rent_instance(999, ssh_public_key="ssh-ed25519 AAAA...")


@pytest.mark.asyncio
async def test_rent_instance_skips_price_verification_when_no_ceiling_configured(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = True
    get_settings().vastai.max_hourly_price_limit_usd = None
    client = _client()
    client._request = AsyncMock(return_value={"new_contract": 777})  # type: ignore[method-assign]

    instance_id = await client.rent_instance(999, ssh_public_key="ssh-ed25519 AAAA...")
    assert instance_id == 777
    client._request.assert_awaited_once()  # only the real rent PUT -- no price-verification GET at all

    request_call = client._request.await_args
    assert request_call is not None
    assert request_call.args[0] == "PUT"


@pytest.mark.asyncio
async def test_rent_instance_threads_env_vars_and_onstart_script_into_the_real_payload(
    redis_client: RedisClient,
) -> None:
    get_settings().vastai.enabled = True
    get_settings().vastai.max_hourly_price_limit_usd = None
    client = _client()
    client._request = AsyncMock(return_value={"new_contract": 1})  # type: ignore[method-assign]

    await client.rent_instance(
        999, ssh_public_key="ssh-ed25519 AAAA...",
        env_vars={"HF_TOKEN": "secret123", "WANDB_API_KEY": "abc"},
        onstart_script="pip install -r requirements.txt && python train.py",
    )

    assert client._request.await_args is not None
    payload = client._request.await_args.kwargs["json_body"]
    assert payload["env"] == "-e HF_TOKEN=secret123 -e WANDB_API_KEY=abc"
    assert payload["onstart"] == "pip install -r requirements.txt && python train.py"

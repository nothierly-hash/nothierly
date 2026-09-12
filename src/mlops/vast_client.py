"""Async client for the Vast.ai GPU rental REST API.

:class:`VastAIClient` owns its own ``aiohttp.ClientSession`` directly,
rather than going through :class:`~core.http_client.ResilientHttpClient`
(built for read-heavy, idempotent GET-based data feeds elsewhere in this
system). That's deliberate: renting and destroying a GPU instance are
each one-shot, side-effecting actions on real infrastructure -- a
generically retried "rent" on a transient 5xx could rent *two* instances
of the same GPU, silently doubling the bill. Every method here fails
loudly and immediately instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, Final, ParamSpec, TypeVar

import aiohttp

from core.config import get_settings
from core.logger import get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import VAST_AI_ENABLED_REDIS_KEY
from core.security import requires_system_active

_log = get_logger(__name__)

_API_BASE_URL: Final[str] = "https://console.vast.ai/api/v0"

_P = ParamSpec("_P")
_R = TypeVar("_R")


class VastAIError(Exception):
    """Base class for every error this module raises."""


class NoMatchingGpuError(VastAIError):
    """Raised when no available, rentable offer matches the search criteria."""


class VastAIDisabledError(VastAIError):
    """Raised when the master switch (VASTAI_ENABLED / VAST_AI_ENABLED_REDIS_KEY) is off. See :func:`is_vast_ai_enabled`."""


class VastAIAuthenticationError(VastAIError):
    """Raised on a real HTTP 401 -- the configured API key is missing/invalid. Never retried."""


class VastAIRateLimitError(VastAIError):
    """Raised on a real HTTP 429 -- Vast.ai is rate-limiting this key. See the class docstring on why this module does not auto-retry."""


class VastAIServerError(VastAIError):
    """Raised on a real HTTP 5xx -- an upstream Vast.ai failure, plausibly transient but not auto-retried here either (see class docstring)."""


class VastAIPriceLimitExceededError(VastAIError):
    """Raised by :meth:`VastAIClient.rent_instance` when the offer's real, freshly re-queried price exceeds ``VastAiSettings.max_hourly_price_limit_usd``."""


async def is_vast_ai_enabled() -> bool:
    """The real, live master switch -- see :data:`~core.schemas.VAST_AI_ENABLED_REDIS_KEY`'s own docstring for the full precedence rule.

    Prefers the Redis key (set by the dashboard's toggle, live, no
    restart needed) when one has ever been written; falls back to
    :attr:`~core.config.VastAiSettings.enabled` (the ``.env``-configured
    cold-start default, itself ``False`` by default) only when the key
    has never been set. Fails **closed** (disabled) on any Redis error --
    this is a cost-control switch, so doubt must block, the opposite of
    :func:`~core.security.is_system_safe`'s "absent key means safe"
    convention, which exists to protect a system that is normally on.
    """
    try:
        client = await RedisClient.get_instance()
        raw = await client.execute("get", VAST_AI_ENABLED_REDIS_KEY)
    except RedisUnavailableError:
        _log.warning("vast_ai_enabled_check_failed_failing_closed")
        return False
    if raw is None:
        return get_settings().vastai.enabled
    # Real, verified-not-assumed quirk: the fakeredis version pinned for this project's tests
    # returns raw bytes for some commands despite decode_responses=True on the client (see
    # tests/test_alpha_engine.py's own comment on the identical quirk) -- decoded defensively here
    # so this check is correct against either, not just whichever this dev environment's real Redis
    # happens to return.
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw).strip().lower() == "true"


def requires_vast_ai_enabled(func: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """Decorator: block a call while the Vast.ai master switch is off. See :func:`is_vast_ai_enabled`.

    Applied only to calls that create or resume real GPU billing
    (:meth:`~VastAIClient.search_offers`/``search_gpu``,
    :meth:`~VastAIClient.rent_instance`, :meth:`~VastAIClient.
    start_instance`) -- deliberately **not** applied to
    :meth:`~VastAIClient.list_instances`/:meth:`~VastAIClient.
    get_instance_details` (monitoring must keep working even with the
    module "disabled", or an operator who just flipped it off can no
    longer see what is still running) or :meth:`~VastAIClient.
    stop_instance`/:meth:`~VastAIClient.destroy_instance` (blocking a
    cost-*reducing* call while "disabled" would strand a still-billing
    instance -- the exact same reasoning :func:`~core.security.
    requires_system_active`'s own docstring already applies to the
    global Kill Switch).
    """

    @wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not await is_vast_ai_enabled():
            _log.warning("vast_ai_call_blocked_disabled", function=func.__qualname__)
            raise VastAIDisabledError("Vast.ai Access is disabled")
        return await func(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class GpuOffer:
    """One real, rentable Vast.ai marketplace offer -- see :meth:`VastAIClient.search_offers`."""

    ask_id: int
    gpu_name: str
    dph_total: float
    reliability: float
    cpu_ram_gb: float | None


class VastAIClient:
    """Thin async wrapper around Vast.ai's REST API.

    Must be used as an async context manager (or have :meth:`connect`
    awaited) before making requests, matching this codebase's convention
    for every other async-resource-owning client.
    """

    def __init__(self, *, api_key: str, timeout_seconds: float = 30.0) -> None:
        """Configure the client. Does not open any connection yet.

        Args:
            api_key: Vast.ai API key.
            timeout_seconds: Total per-request timeout.
        """
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        """Open the underlying ``aiohttp.ClientSession``."""
        self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def close(self) -> None:
        """Close the underlying session. Safe to call multiple times."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> VastAIClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- Public API --------------------------------------------------------

    @requires_vast_ai_enabled
    async def search_gpu(
        self, *, gpu_name: str = "RTX_4090", max_price_per_hour: float = 0.40
    ) -> int:
        """Search Vast.ai's marketplace for the cheapest matching rentable offer.

        Args:
            gpu_name: GPU model to search for (e.g. ``"RTX_4090"``).
            max_price_per_hour: Maximum acceptable on-demand price, in
                USD/hour.

        Returns:
            The cheapest matching offer's ask ID (pass to
            :meth:`rent_instance`).

        Raises:
            NoMatchingGpuError: If no rentable offer matches both criteria.
            VastAIError: On any request failure.
        """
        query = {
            "gpu_name": {"eq": gpu_name},
            "dph_total": {"lte": max_price_per_hour},
            "rentable": {"eq": True},
            "order": [["dph_total", "asc"]],
        }
        data = await self._request("GET", "/bundles/", params={"q": json.dumps(query)})
        offers = data.get("offers", [])
        if not offers:
            raise NoMatchingGpuError(
                f"No rentable Vast.ai offer for gpu_name={gpu_name!r} at "
                f"<= ${max_price_per_hour:.2f}/hr."
            )

        cheapest = min(offers, key=lambda offer: offer["dph_total"])
        ask_id = int(cheapest["id"])
        _log.info(
            "vast_gpu_found", ask_id=ask_id, gpu_name=gpu_name, price_per_hour=cheapest["dph_total"]
        )
        return ask_id

    @requires_vast_ai_enabled
    async def search_offers(
        self,
        *,
        gpu_name: str | None = None,
        max_dph: float | None = None,
        min_ram_gb: float | None = None,
        min_reliability: float = 0.95,
        limit: int = 20,
    ) -> list[GpuOffer]:
        """Search Vast.ai's marketplace, returning every matching rentable offer, cheapest first.

        Broader sibling of :meth:`search_gpu` (which returns only the
        single cheapest ask ID): this returns the real, structured top
        ``limit`` offers so a caller (e.g. ``tests/test_vastai.py``, or a
        future "pick one" UI) can compare more than one. Real, verified
        raw ``/bundles/`` query field names (not the CLI's own friendly
        aliases, which differ): ``reliability`` and ``cpu_ram`` (in MB,
        not GB -- ``min_ram_gb`` is converted here so this method's own
        signature stays in the unit an operator actually thinks in).

        Args:
            gpu_name: GPU model filter (e.g. ``"RTX_3090"``, ``"RTX_4090"``,
                ``"A100"``) -- unfiltered by model when ``None``.
            max_dph: Maximum acceptable price, USD/hour -- unfiltered when ``None``.
            min_ram_gb: Minimum system RAM, in GB -- unfiltered when ``None``.
            min_reliability: Minimum machine reliability score (0-1).
                Defaults to ``0.95``, matching this project's other
                financial-action defaults (safe/strict unless told
                otherwise).
            limit: Maximum number of offers to return.

        Returns:
            Real matching offers, cheapest first. Empty list (never an
            exception) when nothing matches -- an empty market is a real,
            valid outcome for a search, unlike :meth:`search_gpu`'s
            stricter "I need exactly one" contract.

        Raises:
            VastAIError: On any request failure.
        """
        query: dict[str, Any] = {
            "rentable": {"eq": True},
            "reliability": {"gte": min_reliability},
            "order": [["dph_total", "asc"]],
        }
        if gpu_name is not None:
            query["gpu_name"] = {"eq": gpu_name}
        if max_dph is not None:
            query["dph_total"] = {"lte": max_dph}
        if min_ram_gb is not None:
            query["cpu_ram"] = {"gte": min_ram_gb * 1024.0}

        data = await self._request("GET", "/bundles/", params={"q": json.dumps(query)})
        offers = [
            GpuOffer(
                ask_id=int(offer["id"]),
                gpu_name=str(offer.get("gpu_name", "?")),
                dph_total=float(offer["dph_total"]),
                reliability=float(offer.get("reliability2", offer.get("reliability", 0.0))),
                cpu_ram_gb=float(offer["cpu_ram"]) / 1024.0 if offer.get("cpu_ram") is not None else None,
            )
            for offer in data.get("offers", [])
        ]
        return offers[:limit]

    @requires_system_active
    @requires_vast_ai_enabled
    async def rent_instance(
        self,
        ask_id: int,
        *,
        ssh_public_key: str,
        image: str = "pytorch/pytorch:latest",
        disk_gb: int = 20,
        env_vars: dict[str, str] | None = None,
        onstart_script: str | None = None,
    ) -> int:
        """Rent ``ask_id``, launching ``image`` with ``ssh_public_key`` authorized.

        Guarded by the global Kill Switch (see :mod:`core.security`) and
        the Vast.ai master switch (see :func:`is_vast_ai_enabled`):
        refuses to rent a new instance while either is off.
        :meth:`destroy_instance`/:meth:`stop_instance` are deliberately
        *not* guarded the same way by either -- an emergency (or a
        disabled module) must still be able to stop billing on an
        instance already rented.

        **Real price ceiling, enforced here, not just at search time**:
        when :attr:`~core.config.VastAiSettings.max_hourly_price_limit_usd`
        is configured, this re-queries ``ask_id``'s own *current* real
        ``dph_total`` (prices can drift between a search and this call)
        and refuses to rent if it exceeds the ceiling -- independent of
        whatever a caller's own search already filtered for, so a bug or
        a caller that skips :meth:`search_offers`/``search_gpu``
        entirely cannot silently bypass it.

        Args:
            ask_id: From :meth:`search_gpu`/:meth:`search_offers`.
            ssh_public_key: The public key's file *contents* (not a
                path) to authorize for SSH access on the instance.
            image: Docker image to launch.
            disk_gb: Disk space to allocate, in GB.
            env_vars: Environment variables to inject into the instance,
                real Vast.ai ``env`` field (Docker ``-e KEY=value`` flag
                syntax, verified against Vast.ai's own API docs) -- unset
                when ``None``.
            onstart_script: Shell commands to run once the instance
                starts, real Vast.ai ``onstart`` field (4,048-character
                limit -- Vast.ai's own, not enforced client-side here) --
                unset when ``None``.

        Returns:
            The new instance's contract ID (pass to
            :meth:`get_instance_details` / :meth:`destroy_instance`).

        Raises:
            ~core.security.KillSwitchActiveError: If the global Kill
                Switch is active.
            VastAIDisabledError: If the master switch is off.
            VastAIPriceLimitExceededError: If a ceiling is configured and
                the offer's real current price exceeds it.
            VastAIError: If the request fails, or Vast.ai's response
                doesn't include a new contract ID.
        """
        price_limit = get_settings().vastai.max_hourly_price_limit_usd
        if price_limit is not None:
            current_price = await self._current_price(ask_id)
            if current_price is None:
                raise VastAIPriceLimitExceededError(
                    f"Refusing to rent ask {ask_id}: could not verify its current price against "
                    f"the configured ${price_limit:.2f}/hr ceiling (offer may no longer exist)."
                )
            if current_price > price_limit:
                raise VastAIPriceLimitExceededError(
                    f"Refusing to rent ask {ask_id}: current price ${current_price:.4f}/hr exceeds "
                    f"the configured MAX_HOURLY_PRICE_LIMIT of ${price_limit:.2f}/hr."
                )

        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "runtype": "ssh",
            "ssh_key": ssh_public_key,
        }
        if env_vars is not None:
            payload["env"] = " ".join(f"-e {key}={value}" for key, value in env_vars.items())
        if onstart_script is not None:
            payload["onstart"] = onstart_script
        data = await self._request("PUT", f"/asks/{ask_id}/", json_body=payload)
        instance_id = data.get("new_contract")
        if instance_id is None:
            raise VastAIError(f"Vast.ai did not return a new contract ID for ask {ask_id}: {data}")

        instance_id = int(instance_id)
        _log.info("vast_instance_rented", instance_id=instance_id, ask_id=ask_id, image=image)
        return instance_id

    async def get_instance_details(self, instance_id: int) -> dict[str, Any]:
        """Fetch the instance's current status, including its dynamic ``ssh_host``/``ssh_port``.

        Args:
            instance_id: From :meth:`rent_instance`.

        Returns:
            The instance's status dict (raw Vast.ai fields), including at
            least ``actual_status``, ``ssh_host``, ``ssh_port`` once
            Vast.ai has assigned them (may be ``None`` while the instance
            is still starting up).

        Raises:
            VastAIError: On any request failure.
        """
        data = await self._request("GET", f"/instances/{instance_id}/")
        instance = data.get("instances", data)
        return dict(instance)

    async def list_instances(self) -> list[dict[str, Any]]:
        """Fetch every instance currently rented under this API key's account.

        Real Vast.ai endpoint: ``GET /instances/`` with no instance ID
        path segment returns every instance on the account (the same
        real per-instance fields :meth:`get_instance_details` returns
        for one -- ``dph_total``, ``actual_status``, ``start_date``,
        ...), not just one. This is the real data
        :func:`summarize_instance_costs` needed to close this project's
        previously-disclosed gap: "no code sums real spend across
        instances" (see ``src/dashboard/ems_pages/finops.py``'s own
        module docstring).

        Returns:
            Every real instance dict on the account, empty list if none
            are currently rented.

        Raises:
            VastAIError: On any request failure.
        """
        data = await self._request("GET", "/instances/")
        instances = data.get("instances", [])
        return [dict(instance) for instance in instances]

    async def destroy_instance(self, instance_id: int) -> None:
        """Destroy the instance, stopping further billing.

        This is the pipeline's critical safety action (see
        :mod:`scripts.run_remote_training`'s module docstring) -- it
        deliberately does not swallow failures: callers must treat any
        exception from this method as "the instance may still be
        running", never as "the instance is definitely gone".

        Args:
            instance_id: The instance to destroy.

        Raises:
            VastAIError: On any request failure.
        """
        await self._request("DELETE", f"/instances/{instance_id}/")
        _log.info("vast_instance_destroyed", instance_id=instance_id)

    async def stop_instance(self, instance_id: int) -> None:
        """Pause the instance: stops GPU billing, keeps disk billing and the contract intact.

        Real Vast.ai endpoint (verified against Vast.ai's own API docs):
        ``PUT /instances/{id}/`` with ``{"state": "stopped"}``. Not
        guarded by :func:`is_vast_ai_enabled` -- same reasoning as
        :meth:`destroy_instance`: a cost-*reducing* call must keep
        working even with the module "disabled" (see
        :func:`requires_vast_ai_enabled`'s own docstring). Not guarded by
        the global Kill Switch either, for the identical reason
        :meth:`destroy_instance` isn't.

        Args:
            instance_id: The instance to pause.

        Raises:
            VastAIError: On any request failure.
        """
        await self._request("PUT", f"/instances/{instance_id}/", json_body={"state": "stopped"})
        _log.info("vast_instance_stopped", instance_id=instance_id)

    @requires_vast_ai_enabled
    async def start_instance(self, instance_id: int) -> None:
        """Resume a previously-stopped instance -- resumes real GPU billing.

        Real Vast.ai endpoint: ``PUT /instances/{id}/`` with
        ``{"state": "running"}``. Guarded by :func:`is_vast_ai_enabled`
        (unlike :meth:`stop_instance`): this resumes billing, the same
        category of action as :meth:`rent_instance`, not a cleanup call.

        Args:
            instance_id: The instance to resume.

        Raises:
            VastAIDisabledError: If the master switch is off.
            VastAIError: On any request failure.
        """
        await self._request("PUT", f"/instances/{instance_id}/", json_body={"state": "running"})
        _log.info("vast_instance_started", instance_id=instance_id)

    # -- Internal ---------------------------------------------------------

    async def _current_price(self, ask_id: int) -> float | None:
        """Real, freshly-queried current ``dph_total`` for one specific ask ID -- see :meth:`rent_instance`'s price-ceiling check.

        Returns:
            The real current price, or ``None`` if this ask no longer
            appears in a fresh search (already rented by someone else,
            expired, etc.) -- treated as "cannot verify, refuse" by the
            caller, never as "no limit".
        """
        data = await self._request("GET", "/bundles/", params={"q": json.dumps({"id": {"eq": ask_id}})})
        offers = data.get("offers", [])
        if not offers:
            return None
        return float(offers[0]["dph_total"])

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        """Issue one request against the Vast.ai API and return its parsed JSON body.

        Deliberately does *not* retry: see the class docstring on why a
        generic retry policy is the wrong default for these
        infrastructure-mutating calls.

        Args:
            method: HTTP method (``"GET"``, ``"PUT"``, ``"DELETE"``).
            path: API path, e.g. ``"/bundles/"``.
            params: Optional query parameters.
            json_body: Optional JSON request body.

        Returns:
            The parsed JSON response body.

        Raises:
            RuntimeError: If called before :meth:`connect`.
            VastAIAuthenticationError: On a real HTTP 401 (invalid/missing key).
            VastAIRateLimitError: On a real HTTP 429.
            VastAIServerError: On a real HTTP 5xx.
            VastAIError: On any other non-2xx response or network failure.
        """
        if self._session is None:
            raise RuntimeError(
                "VastAIClient must be connected (use 'async with' or await .connect()) "
                "before making requests."
            )

        url = f"{_API_BASE_URL}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with self._session.request(
                method, url, params=params, json=json_body, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as exc:
            _log.error("vast_api_request_failed", method=method, path=path, status=exc.status)
            message = f"Vast.ai API request failed: {method} {path} -> HTTP {exc.status}."
            if exc.status == 401:
                raise VastAIAuthenticationError(f"{message} Invalid or missing VASTAI_API_KEY.") from exc
            if exc.status == 429:
                raise VastAIRateLimitError(f"{message} Rate-limited by Vast.ai.") from exc
            if exc.status >= 500:
                raise VastAIServerError(message) from exc
            raise VastAIError(message) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            _log.error(
                "vast_api_request_network_error",
                method=method,
                path=path,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise VastAIError(f"Vast.ai API network error: {method} {path}: {exc}") from exc


# =============================================================================
# Pure aggregation -- no I/O, real regression tests without a live Vast.ai
# account. Closes the gap src/dashboard/ems_pages/finops.py's own module
# docstring names: "no code sums real spend across instances or projects it
# forward."
# =============================================================================


@dataclass(frozen=True)
class InstanceCostRow:
    """One real instance's cost figures, derived from its own Vast.ai fields -- never a fabricated estimate."""

    instance_id: int
    gpu_name: str | None
    actual_status: str
    dph_total: float
    accrued_cost_usd: float


@dataclass(frozen=True)
class FinOpsCostSummary:
    """Real, aggregated cost figures across every instance on the account."""

    running_instance_count: int
    total_burn_rate_usd_per_hour: float
    total_accrued_cost_usd: float
    per_instance: list[InstanceCostRow] = field(default_factory=list)


def summarize_instance_costs(instances: list[dict[str, Any]], *, now: float | None = None) -> FinOpsCostSummary:
    """Real burn-rate (running instances only) + real accrued-cost (every instance) aggregation.

    Args:
        instances: Raw Vast.ai instance dicts, e.g. from
            :meth:`VastAIClient.list_instances`.
        now: Unix timestamp to compute elapsed accrual against.
            Defaults to ``time.time()`` -- overridable for deterministic
            tests.

    Returns:
        A :class:`FinOpsCostSummary`. Accrued cost for an instance
        missing a real ``start_date`` is ``0.0`` (nothing real to
        compute from) rather than a guess -- consistently with this
        project's "never fabricate a number in place of a real gap"
        discipline.
    """
    now = now if now is not None else time.time()
    rows: list[InstanceCostRow] = []
    for instance in instances:
        dph_total = float(instance.get("dph_total") or 0.0)
        start_date = instance.get("start_date")
        accrued = ((now - float(start_date)) / 3600.0) * dph_total if start_date is not None else 0.0
        rows.append(
            InstanceCostRow(
                instance_id=int(instance["id"]),
                gpu_name=instance.get("gpu_name"),
                actual_status=str(instance.get("actual_status") or "unknown"),
                dph_total=dph_total,
                accrued_cost_usd=round(max(0.0, accrued), 4),
            )
        )

    running_rows = [row for row in rows if row.actual_status == "running"]
    return FinOpsCostSummary(
        running_instance_count=len(running_rows),
        total_burn_rate_usd_per_hour=round(sum(row.dph_total for row in running_rows), 4),
        total_accrued_cost_usd=round(sum(row.accrued_cost_usd for row in rows), 4),
        per_instance=rows,
    )

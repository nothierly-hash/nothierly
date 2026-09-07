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
from typing import Any, Final

import aiohttp

from core.logger import get_logger
from core.security import requires_system_active

_log = get_logger(__name__)

_API_BASE_URL: Final[str] = "https://console.vast.ai/api/v0"


class VastAIError(Exception):
    """Base class for every error this module raises."""


class NoMatchingGpuError(VastAIError):
    """Raised when no available, rentable offer matches the search criteria."""


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

    @requires_system_active
    async def rent_instance(
        self,
        ask_id: int,
        *,
        ssh_public_key: str,
        image: str = "pytorch/pytorch:latest",
        disk_gb: int = 20,
    ) -> int:
        """Rent ``ask_id``, launching ``image`` with ``ssh_public_key`` authorized.

        Guarded by the global Kill Switch (see :mod:`core.security`):
        refuses to rent a new instance while ``SYSTEM_STATUS`` is
        ``PANIC``. :meth:`destroy_instance` is deliberately *not*
        guarded the same way -- an emergency must still be able to stop
        billing on an instance already rented.

        Args:
            ask_id: From :meth:`search_gpu`.
            ssh_public_key: The public key's file *contents* (not a
                path) to authorize for SSH access on the instance.
            image: Docker image to launch.
            disk_gb: Disk space to allocate, in GB.

        Returns:
            The new instance's contract ID (pass to
            :meth:`get_instance_details` / :meth:`destroy_instance`).

        Raises:
            ~core.security.KillSwitchActiveError: If the global Kill
                Switch is active.
            VastAIError: If the request fails, or Vast.ai's response
                doesn't include a new contract ID.
        """
        payload = {
            "client_id": "me",
            "image": image,
            "disk": disk_gb,
            "runtype": "ssh",
            "ssh_key": ssh_public_key,
        }
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

    # -- Internal ---------------------------------------------------------

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
            VastAIError: On any non-2xx response or network failure.
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
            raise VastAIError(
                f"Vast.ai API request failed: {method} {path} -> HTTP {exc.status}."
            ) from exc
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

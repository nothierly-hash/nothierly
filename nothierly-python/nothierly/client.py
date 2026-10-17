"""NothierlyClient -- the official synchronous Python SDK for Nothierly's institutional market-data API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import pydantic
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .exceptions import (
    APITimeoutError,
    AuthenticationError,
    NothierlyError,
    RateLimitError,
)
from .models import (
    MacroEvent,
    MacroSentimentResponse,
    P2PCandidate,
    P2PPremiumsResponse,
)

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_BASE_URL = "http://localhost:8000"

# Retenter uniquement sur les erreurs réseau/timeout et ces codes HTTP -- jamais sur un 401 (clé
# invalide, retenter ne changera rien) ni un 404/422 (erreur de requête, pas une panne transitoire).
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503})


def _is_retryable(exc: BaseException) -> bool:
    """True for network errors, timeouts, and the configured retryable HTTP status codes.

    Deliberately inspects only *raw* ``httpx`` exception types (never this
    SDK's own :class:`~nothierly.exceptions.NothierlyError` subclasses):
    see :meth:`NothierlyClient._get`'s own comment for why the
    custom-exception translation happens strictly outside the retried
    call, so this predicate always sees the real failure tenacity needs to
    reason about.
    """
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return False


class NothierlyClient:
    """Synchronous client for the Nothierly institutional market-data API.

    Args:
        api_key: Institutional API key, sent as the ``X-API-Key`` header.
            Falls back to the ``NOTHIERLY_API_KEY`` environment variable
            when omitted -- never hardcode a real key in source control.
        base_url: API origin. Defaults to ``http://localhost:8000`` for
            local development against this repo's own ``p2p_api.py``.
        timeout_seconds: Hard per-request timeout, in seconds, applied to
            every phase (connect/read/write) -- a client thread must never
            block indefinitely on a stalled backend.

    Usage::

        with NothierlyClient(api_key="...") as client:
            df = client.get_p2p_premium("ARS")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved_key = api_key or os.environ.get("NOTHIERLY_API_KEY")
        self._api_key = resolved_key
        headers = {"X-API-Key": resolved_key} if resolved_key else {}
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers=headers,
        )

    def close(self) -> None:
        """Release the underlying connection pool. Safe to call more than once."""
        self._client.close()

    def __enter__(self) -> NothierlyClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _request_raw(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """The one real, retried HTTP call.

        Raises only raw ``httpx`` exceptions -- never one of this SDK's
        own :class:`~nothierly.exceptions.NothierlyError` subclasses. That
        translation happens one level up, in :meth:`_get`, strictly
        *after* tenacity is done retrying: if it happened here instead,
        every attempt after the first would be retried against an
        already-translated exception type that ``_is_retryable`` does not
        recognize, silently disabling retries on exactly the failures
        this decorator exists to retry (timeouts, 429/500/502/503).
        """
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Real HTTP GET with retries already applied, translated into this SDK's own exceptions."""
        try:
            response = self._request_raw(path, params=params)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Request to {path} timed out after retries") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise AuthenticationError("Invalid or missing API key", status_code=401) from exc
            if status == 429:
                raise RateLimitError("Rate limit exceeded (retries exhausted)", status_code=429) from exc
            raise NothierlyError(
                f"{path} returned HTTP {status}: {exc.response.text[:200]}", status_code=status,
            ) from exc
        except httpx.TransportError as exc:
            raise NothierlyError(f"Network error calling {path} (retries exhausted): {exc}") from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            # A 2xx response with a non-JSON body (an upstream proxy's HTML error page, a truncated
            # response, ...) -- still a real API failure, and still owed a NothierlyError, not a raw
            # json.JSONDecodeError a caller catching only NothierlyError would never see coming.
            raise NothierlyError(f"{path} returned a non-JSON response body") from exc

    @staticmethod
    def _validate(model: type[pydantic.BaseModel], raw: Any, *, path: str) -> Any:
        """Validate a JSON payload against a response model, translating a shape mismatch into `NothierlyError`.

        Every model here already uses ``extra="ignore"`` (see
        ``models.py``'s own docstring) so a genuinely *new* field never
        reaches this -- this only fires when a field this SDK actually
        depends on is missing or the wrong type, a real, actionable API
        contract break a caller catching only `NothierlyError` must still
        be able to see.
        """
        try:
            return model.model_validate(raw)
        except pydantic.ValidationError as exc:
            raise NothierlyError(f"{path} returned a response that does not match the expected shape: {exc}") from exc

    # -- Methodes d'acces (Quant-Friendly) ----------------------------------

    def get_p2p_premium(
        self, fiat: str, *, as_dataframe: bool = True
    ) -> pd.DataFrame | list[P2PCandidate]:
        """Fetch the current real P2P/forex arbitrage premium for ``fiat``.

        Args:
            fiat: A fiat currency code from the scan (e.g. ``"NGN"``,
                ``"ARS"``), case-insensitive.
            as_dataframe: When True (default), returns a `pandas.DataFrame`
                with one row per matching candidate and the real candidate
                columns always present -- an **empty** DataFrame (same
                columns, zero rows) when ``fiat`` has no current real P2P
                liquidity, never a `KeyError`. When False, returns the raw
                validated :class:`~nothierly.models.P2PCandidate` list.
        """
        raw = self._get("/api/v1/p2p-premiums")
        parsed = self._validate(P2PPremiumsResponse, raw, path="/api/v1/p2p-premiums")
        matching = [c for c in parsed.top_candidates if c.country_fiat.upper() == fiat.upper()]

        if not as_dataframe:
            return matching

        columns = list(P2PCandidate.model_fields.keys())
        if not matching:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([c.model_dump() for c in matching], columns=columns)

    def get_macro_sentiment(
        self, fiat: str, *, as_dataframe: bool = True
    ) -> pd.DataFrame | MacroSentimentResponse:
        """Fetch the current risk score + recent crisis-event history for ``fiat``.

        **Disclosure**: as of this SDK version, no Nothierly API endpoint
        serves this yet (targets ``GET /api/v1/macro-sentiment/{fiat}``,
        which does not exist on the backend at the time this SDK ships) --
        ``political_agent.py`` already computes and stores this exact data
        in Redis, but nothing exposes it over REST yet. Calling this
        method against today's backend raises :class:`~nothierly.exceptions.NothierlyError`
        (HTTP 404); the method itself, its validation, and its retry
        behavior are real and ready for when that endpoint ships.

        Args:
            fiat: A fiat currency code, case-insensitive.
            as_dataframe: When True (default), returns a `pandas.DataFrame`
                of ``recent_events`` (one row per historical crisis event,
                empty with the real event columns if there is no history)
                instead of the raw model.
        """
        path = f"/api/v1/macro-sentiment/{fiat.upper()}"
        raw = self._get(path)
        parsed = self._validate(MacroSentimentResponse, raw, path=path)

        if not as_dataframe:
            return parsed

        columns = list(MacroEvent.model_fields.keys())
        if not parsed.recent_events:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([e.model_dump() for e in parsed.recent_events], columns=columns)

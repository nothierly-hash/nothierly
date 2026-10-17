"""Resilient, concurrency-limited async HTTP client for external data feeds.

:class:`ResilientHttpClient` wraps a single ``aiohttp.ClientSession`` and
adds two things bare ``aiohttp`` doesn't give you: a hard cap on
in-flight requests (so a burst of calls from, say, a macro-data agent
polling many indicators can never overwhelm a rate-limited upstream API
or this process' own connection pool), and automatic retry-with-backoff
for exactly the class of failures worth retrying -- network/timeout
errors and ``429``/``5xx`` responses -- via ``tenacity``.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Final, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from core.logger import get_logger

_log = get_logger(__name__)

_MAX_ATTEMPTS: Final[int] = 5
_BASE_DELAY_SECONDS: Final[float] = 1.0
_MAX_DELAY_SECONDS: Final[float] = 30.0

# HTTP statuses worth retrying: 429 (rate-limited -- the upstream is
# explicitly asking us to back off) and 5xx (upstream-side failure,
# plausibly transient). Everything else (4xx client errors like 400/401/
# 404) reflects a request that will never succeed by retrying it, so it
# is surfaced to the caller immediately instead.
_RETRYABLE_HTTP_STATUSES: Final[frozenset[int]] = frozenset({429})


class RetryableHttpError(Exception):
    """Raised for a response status that should trigger a retry (429/5xx).

    Never raised for other non-2xx statuses -- those surface as
    ``aiohttp.ClientResponseError`` instead, immediately, uncaught by the
    retry policy.
    """

    def __init__(self, status: int, sanitized_url: str) -> None:
        self.status = status
        super().__init__(f"Retryable HTTP {status} from {sanitized_url}")


def _is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE_HTTP_STATUSES or status >= 500


def _sanitize_url(url: str) -> str:
    """Strip the query string and fragment from a URL before logging it.

    API keys/tokens are very commonly passed as query parameters (e.g.
    ``?apikey=...``); headers (which may carry ``Authorization``) are
    never logged by this module at all. Scheme/host/path are kept since
    they're useful for diagnosing which endpoint is failing.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _is_retryable_error(exc: BaseException) -> bool:
    """Retry predicate: network/timeout errors and :class:`RetryableHttpError` only."""
    return isinstance(
        exc,
        (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientConnectionError, RetryableHttpError),
    )


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """``tenacity`` ``before_sleep`` hook: log the retry decision.

    Deliberately logs only the attempt number, computed wait time, and
    exception type/message -- never the request's headers or URL query
    string, which is the caller's responsibility to keep secret-free (see
    :func:`_sanitize_url`) and is logged, sanitized, at the call site
    instead of here.
    """
    outcome = retry_state.outcome
    exc = outcome.exception() if outcome is not None else None
    wait_seconds = retry_state.next_action.sleep if retry_state.next_action is not None else None
    _log.warning(
        "http_request_retrying",
        attempt=retry_state.attempt_number,
        max_attempts=_MAX_ATTEMPTS,
        wait_seconds=round(wait_seconds, 2) if wait_seconds is not None else None,
        error_type=type(exc).__name__ if exc is not None else None,
        error=str(exc) if exc is not None else None,
    )


class ResilientHttpClient:
    """Async HTTP client with bounded concurrency and automatic retry.

    Must be used as an async context manager (or have :meth:`connect`
    awaited) before making requests, matching this codebase's convention
    for every other async-resource-owning class (see
    :class:`~core.zmq_bus.ZmqPublisher`, :class:`~src.agents.market_data.
    exchange_client.AsyncExchangeClient`).
    """

    def __init__(
        self,
        *,
        max_concurrent_requests: int = 10,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Configure the client. Does not open any connection yet.

        Args:
            max_concurrent_requests: Hard cap on in-flight requests this
                client will issue at once; further calls to
                :meth:`get_json` block until a slot frees up.
            timeout_seconds: Total per-request timeout (connect + read),
                passed to ``aiohttp.ClientTimeout``.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
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

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """``GET`` ``url`` and return its parsed JSON body. See :meth:`_request_json`."""
        return await self._request_json("GET", url, params=params, headers=headers)

    async def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """``GET`` ``url`` and return its raw response body as text -- same retry/concurrency policy as :meth:`get_json`.

        Added for :mod:`political_agent` (RSS/Atom feeds are XML, not
        JSON -- ``response.json()`` would raise on a perfectly valid
        feed). Every other real caller in this project so far only
        needed JSON.
        """
        return cast(str, await self._request_json("GET", url, params=params, headers=headers, response_format="text"))

    async def post_json(
        self,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """``POST`` ``url`` with a JSON body and return its parsed JSON response.

        Added for :mod:`src.agents.guardian_ai` (LLM chat-completion APIs
        are POST-only) -- every other caller in this project so far only
        needed GET/PUT/DELETE.
        """
        return await self._request_json("POST", url, json_body=json_body, headers=headers)

    async def put_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """``PUT`` ``url`` and return its parsed JSON body. See :meth:`_request_json`."""
        return await self._request_json("PUT", url, params=params, headers=headers)

    async def delete_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """``DELETE`` ``url`` and return its parsed JSON body. See :meth:`_request_json`."""
        return await self._request_json("DELETE", url, params=params, headers=headers)

    @retry(
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_random_exponential(multiplier=_BASE_DELAY_SECONDS, max=_MAX_DELAY_SECONDS),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=_log_before_sleep,
        reraise=True,
    )
    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_format: Literal["json", "text"] = "json",
    ) -> Any:
        """Issue one HTTP request and return its parsed JSON body (or raw text -- see :meth:`get_text`).

        Shared implementation behind :meth:`get_json`/:meth:`get_text`/
        :meth:`put_json`/:meth:`delete_json`: same concurrency limiting,
        retry policy, and error handling regardless of verb.

        Retries automatically, with capped exponential backoff + jitter,
        on network/timeout errors and ``429``/``5xx`` responses. Any
        other non-2xx status (e.g. ``400``, ``401``, ``404``) raises
        ``aiohttp.ClientResponseError`` immediately, without retrying --
        retrying a request that will never succeed just wastes time and
        the upstream's rate-limit budget.

        Args:
            method: HTTP method (``"GET"``, ``"PUT"``, ``"DELETE"``, ...).
            url: Full request URL.
            params: Optional query parameters.
            json_body: Optional JSON request body (``POST``/``PUT`` only).
                Never logged, by this method or by the retry logger.
            headers: Optional request headers (e.g. an API key). Never
                logged, by this method or by the retry logger.

        Returns:
            The parsed JSON response body.

        Raises:
            RuntimeError: If called before :meth:`connect` (or entering
                the client as an async context manager).
            aiohttp.ClientResponseError: For a non-retryable non-2xx
                status.
            aiohttp.ClientConnectionError: For a network failure that
                persisted through every retry attempt.
            asyncio.TimeoutError: For a timeout that persisted through
                every retry attempt.
        """
        if self._session is None:
            raise RuntimeError(
                "ResilientHttpClient must be connected (use 'async with' or "
                "await .connect()) before making requests."
            )

        sanitized_url = _sanitize_url(url)

        async with self._semaphore:
            try:
                async with self._session.request(
                    method, url, params=params, json=json_body, headers=headers
                ) as response:
                    if _is_retryable_status(response.status):
                        _log.warning(
                            "http_request_failed_retryable",
                            method=method,
                            url=sanitized_url,
                            status=response.status,
                        )
                        raise RetryableHttpError(response.status, sanitized_url)

                    response.raise_for_status()
                    if response_format == "text":
                        return await response.text()
                    return await response.json()
            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientConnectionError) as exc:
                _log.warning(
                    "http_request_network_error",
                    method=method,
                    url=sanitized_url,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            except aiohttp.ClientResponseError as exc:
                _log.error(
                    "http_request_failed_non_retryable",
                    method=method,
                    url=sanitized_url,
                    status=exc.status,
                )
                raise

    async def __aenter__(self) -> ResilientHttpClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

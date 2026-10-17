"""Exception hierarchy raised by the Nothierly SDK.

Every failure the client surfaces -- a bad API key, a rate limit, a
timeout, a 5xx after retries are exhausted -- is a :class:`NothierlyError`
subclass, never a bare ``httpx``/``pydantic`` exception. A caller can
catch ``NothierlyError`` alone and handle every real failure mode without
knowing this SDK's internal HTTP library.
"""

from __future__ import annotations


class NothierlyError(Exception):
    """Base class for every error raised by the Nothierly SDK.

    Args:
        message: Human-readable description of what went wrong.
        status_code: The API's HTTP status code, if the failure came from
            a real response (``None`` for a network-level failure that
            never reached the server, e.g. a DNS error or connection
            refused).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(NothierlyError):
    """Raised on HTTP 401 -- the API key is missing, invalid, or revoked."""


class RateLimitError(NothierlyError):
    """Raised on HTTP 429 -- the caller exceeded its request quota (after retries were exhausted)."""


class APITimeoutError(NothierlyError):
    """Raised when a request does not complete within the client's configured timeout."""

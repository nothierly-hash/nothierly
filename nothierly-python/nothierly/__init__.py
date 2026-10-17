"""Official Python SDK for the Nothierly institutional market-data API."""

from __future__ import annotations

from .client import NothierlyClient
from .exceptions import (
    APITimeoutError,
    AuthenticationError,
    NothierlyError,
    RateLimitError,
)
from .models import (
    CrossCountryOpportunity,
    LiquidityDepthLevel,
    LiquidityDepthResponse,
    MacroEvent,
    MacroSentimentResponse,
    P2PCandidate,
    P2PPremiumsResponse,
)

__version__ = "0.1.0"

__all__ = [
    "NothierlyClient",
    "NothierlyError",
    "AuthenticationError",
    "RateLimitError",
    "APITimeoutError",
    "P2PCandidate",
    "CrossCountryOpportunity",
    "P2PPremiumsResponse",
    "MacroEvent",
    "MacroSentimentResponse",
    "LiquidityDepthLevel",
    "LiquidityDepthResponse",
]

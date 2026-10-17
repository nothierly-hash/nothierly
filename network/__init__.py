"""Cross-broker network resilience: predictive rate-limiting and stream-gap state recovery.

Distinct from :mod:`core` (generic, trading-domain-free infra like
Redis/ZMQ connection retry) and :mod:`risk` (position/margin-level risk):
this package is about the network layer specifically -- staying under a
broker's rate limit before it bans the connection, and correctly
recovering local state after a WebSocket gap, rather than resuming
blindly on stale data.
"""

from __future__ import annotations

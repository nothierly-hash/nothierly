"""Shared pytest configuration for ``tests/``.

Every socket created during the session gets ``setsockopt(zmq.LINGER, 0)`` immediately, so a context
that is torn down (by a fixture, by GC, or at exit) discards pending messages instead of waiting on
them. Test-only; no production behavior changes.

This is defense in depth, not the fix for the pytest exit hang. That hang's root cause was fixtures
that built a ``BaseAgent`` (which opens its own PUB/SUB sockets and ``zmq.asyncio.Context``) and never
closed it: ``tests/test_audit_ledger.py::ledger`` and ``tests/test_shadow_crossing_engine.py::engine``.
Any new agent fixture must yield and then close ``_publisher``, ``_subscriber`` and ``_zmq_context``;
see TESTING.md.

``fake_clock`` pins the clock the p2p_api rate limiter buckets requests by (its ``get_clock`` dependency), so
a rate-limit test never depends on where in the real second it happens to run.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import zmq

# core.config refuses to load without these ("required, no default"), and several test modules call get_settings() while being collected. A
# machine with a .env supplies them; a clean checkout (CI image) has none, so give them test-only placeholders there. Never overrides a real value.
if not Path(".env").exists():
    for _required in ("BROKER_API_KEY", "BROKER_API_SECRET", "OANDA_API_TOKEN", "OANDA_ACCOUNT_ID", "VAULT_MASTER_KEY"):
        os.environ.setdefault(_required, "test-placeholder")
    # With a broker configured, publishers CONNECT instead of binding ZMQ_PUB_ADDRESS (tcp://127.0.0.1:5555): several tests build two agents in one
    # process, and without this the second one dies with "Address in use". A developer's .env sets these; a clean checkout has none. Nothing listens
    # there -- connect() to a dead port is fine for ZMQ, the tests only need the sockets to exist.
    os.environ.setdefault("ZMQ_BROKER_FRONTEND_ADDRESS", "tcp://127.0.0.1:55559")
    os.environ.setdefault("ZMQ_BROKER_BACKEND_ADDRESS", "tcp://127.0.0.1:55560")


@pytest.fixture(autouse=True, scope="session")
def _zmq_sockets_linger_zero() -> Iterator[None]:
    original_socket = zmq.Context.socket

    def socket_with_linger_zero(self: zmq.Context, socket_type: int, *args: Any, **kwargs: Any) -> Any:
        sock = original_socket(self, socket_type, *args, **kwargs)
        sock.setsockopt(zmq.LINGER, 0)
        return sock

    zmq.Context.socket = socket_with_linger_zero  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        zmq.Context.socket = original_socket  # type: ignore[method-assign]


class FakeClock:
    """A settable clock in epoch seconds. The starting value is a whole second, so ``advance(0.5)`` stays in
    the same 1-second rate-limit window and ``advance(1.0)`` moves to the next one, whatever the real time is."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock() -> Iterator[FakeClock]:
    import p2p_api  # lazy: only the tests that ask for this fixture pay for importing the app

    clock = FakeClock()
    p2p_api.app.dependency_overrides[p2p_api.get_clock] = lambda: clock
    try:
        yield clock
    finally:
        p2p_api.app.dependency_overrides.pop(p2p_api.get_clock, None)

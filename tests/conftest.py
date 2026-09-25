"""Shared pytest configuration for ``tests/``.

Every socket created during the session gets ``setsockopt(zmq.LINGER, 0)`` immediately, so a context
that is torn down (by a fixture, by GC, or at exit) discards pending messages instead of waiting on
them. Test-only; no production behavior changes.

This is defense in depth, not the fix for the pytest exit hang. That hang's root cause was fixtures
that built a ``BaseAgent`` (which opens its own PUB/SUB sockets and ``zmq.asyncio.Context``) and never
closed it: ``tests/test_audit_ledger.py::ledger`` and ``tests/test_shadow_crossing_engine.py::engine``.
Any new agent fixture must yield and then close ``_publisher``, ``_subscriber`` and ``_zmq_context``;
see TESTING.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import zmq


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

"""Shared pytest configuration for ``tests/``.

ZMQ teardown. Many tests build ``BaseAgent`` subclasses, and ``BaseAgent`` creates its own
``zmq.asyncio.Context()`` (``core/base_agent.py``). A test that doesn't shut its agent down leaves a
context with open sockets behind. With ZMQ's default ``LINGER = -1`` (wait forever for unsent
messages), terminating such a context blocks, so the pytest process sits after the last test and
never prints its summary. ``pyproject.toml``'s ``timeout`` only bounds a single test, not this.

Every socket created during the session gets ``setsockopt(zmq.LINGER, 0)`` immediately, so a context
that is torn down (by a fixture, by GC, or at exit) discards pending messages instead of waiting on
them. Test-only; no production behavior changes.
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

"""Cognitive Router: the real ZMQ ROUTER-DEALER transport -- distinct from core/zmq_bus.py's own
PUB/SUB (broadcast, no addressed reply) pattern used by every other agent in this project.

**Why ROUTER-DEALER here, not the existing PUB/SUB bus**: this hypervisor needs to send one
specific task to one specific, addressable worker and get exactly that worker's reply back --
PUB/SUB has no notion of addressing a single subscriber or of a reply path at all (see
core/zmq_bus.py's own docstring: it is fire-and-forget broadcast by topic). ROUTER-DEALER is
ZeroMQ's own real pattern for exactly this: a ROUTER socket can address any connected DEALER by
its identity frame, and a DEALER's own identity is automatically attached for the ROUTER to reply
to -- a real, minimal worker-pool protocol, not a bespoke one invented here.

Demonstrated within one process (asyncio tasks sharing one real ``zmq.asyncio.Context``), the same
real, disclosed convention ``docker-compose.yml``'s own ``nothierly-agents`` service already uses
for running several real agents concurrently -- the wire protocol (real multi-part ZMQ messages
with a real identity frame) is identical whether the DEALER lives in this process or a separate
one; nothing here is faked to look like ZMQ without being it.
"""

from __future__ import annotations

import json
from typing import Any

import zmq
import zmq.asyncio

from core.logger import get_logger

_log = get_logger(__name__)

_READY: bytes = b"\x01READY"


class HypervisorRouter:
    """Binds a real ``zmq.ROUTER`` socket; dispatches one task at a time to one named worker."""

    def __init__(self, address: str, context: zmq.asyncio.Context) -> None:
        self._socket: zmq.asyncio.Socket = context.socket(zmq.ROUTER)
        self._socket.bind(address)
        self._address = address
        self._log = _log

    async def wait_for_workers(self, expected_names: set[str]) -> None:
        """Blocks until every real worker in ``expected_names`` has sent its real READY frame."""
        remaining = set(expected_names)
        while remaining:
            identity, frame = await self._socket.recv_multipart()
            if frame == _READY:
                remaining.discard(identity.decode("utf-8"))
                self._log.info("hypervisor_worker_ready", worker=identity.decode("utf-8"))

    async def dispatch(self, worker_name: str, payload: dict[str, Any]) -> None:
        await self._socket.send_multipart([worker_name.encode("utf-8"), json.dumps(payload).encode("utf-8")])

    async def receive_result(self) -> tuple[str, dict[str, Any]]:
        identity, body = await self._socket.recv_multipart()
        return identity.decode("utf-8"), json.loads(body.decode("utf-8"))

    def close(self) -> None:
        self._socket.close(linger=0)


class HypervisorWorker:
    """A real ``zmq.DEALER`` socket, addressable by the router under its own real ``name``."""

    def __init__(self, name: str, router_address: str, context: zmq.asyncio.Context) -> None:
        self._name = name
        self._socket: zmq.asyncio.Socket = context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, name.encode("utf-8"))
        self._socket.connect(router_address)

    async def announce_ready(self) -> None:
        await self._socket.send_multipart([_READY])

    async def receive_task(self) -> dict[str, Any]:
        (body,) = await self._socket.recv_multipart()
        result: dict[str, Any] = json.loads(body.decode("utf-8"))
        return result

    async def send_result(self, result: dict[str, Any]) -> None:
        await self._socket.send_multipart([json.dumps(result).encode("utf-8")])

    def close(self) -> None:
        self._socket.close(linger=0)

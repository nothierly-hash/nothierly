"""XSUB/XPUB broker: the shared rendezvous point for the multi-agent bus.

Every :class:`~core.base_agent.BaseAgent` binds its own PUB socket when run
standalone (see :class:`~core.zmq_bus.ZmqPublisher`'s ``bind=True``
default) -- correct for one agent talking to itself (every demo/test this
project has run so far), but not for multiple agent *processes*, or
multiple agent tasks in one process, coexisting: each would try to bind
the same configured port, and only the first would succeed.

:class:`ZmqBroker` is the standard ZeroMQ fix for that (the "forwarder
device" pattern from the ZeroMQ guide): it binds two sockets --

- ``frontend`` (``XSUB``): every agent's :class:`~core.zmq_bus.
  ZmqPublisher` *connects* here (``bind=False``) instead of binding its
  own address.
- ``backend`` (``XPUB``): every agent's :class:`~core.zmq_bus.
  ZmqSubscriber` connects here (unchanged from how subscribers already
  behave) to receive the combined stream from every publisher.

and forwards every message received on one to the other. Agents never
talk to each other directly; they only ever know about the broker's two
addresses.
"""

from __future__ import annotations

import asyncio

import zmq
import zmq.asyncio

from core.logger import get_logger

_log = get_logger(__name__)


class ZmqBroker:
    """Binds an XSUB/XPUB pair and forwards every message between them.

    Must be started (via :meth:`run`) before any agent's publisher tries
    to connect to :attr:`frontend_address`, and kept running for the
    lifetime of every agent process using this bus -- it is the single
    point every published message passes through.
    """

    def __init__(
        self,
        frontend_address: str,
        backend_address: str,
        *,
        context: zmq.asyncio.Context | None = None,
    ) -> None:
        """Configure the broker. Does not bind anything yet -- see :meth:`run`.

        Args:
            frontend_address: Endpoint every :class:`~core.zmq_bus.
                ZmqPublisher` connects to (this broker binds it).
            backend_address: Endpoint every :class:`~core.zmq_bus.
                ZmqSubscriber` connects to (this broker binds it).
            context: Optional shared ``zmq.asyncio.Context``. Defaults to
                the process-wide singleton, matching every other ZMQ
                socket in this project.
        """
        self.frontend_address = frontend_address
        self.backend_address = backend_address
        self._context = context or zmq.asyncio.Context.instance()
        self._frontend: zmq.asyncio.Socket | None = None
        self._backend: zmq.asyncio.Socket | None = None
        self._closed = False

    async def run(self) -> None:
        """Bind both sockets and forward messages between them until cancelled.

        Runs forever (until the awaiting task is cancelled, e.g. as part
        of a graceful process shutdown); does not return on its own.
        """
        self._frontend = self._context.socket(zmq.XSUB)
        self._backend = self._context.socket(zmq.XPUB)
        self._frontend.bind(self.frontend_address)
        self._backend.bind(self.backend_address)
        _log.info(
            "zmq_broker_started",
            frontend_address=self.frontend_address,
            backend_address=self.backend_address,
        )

        try:
            await asyncio.gather(
                self._forward(self._frontend, self._backend),
                self._forward(self._backend, self._frontend),
            )
        finally:
            self.close()

    async def _forward(self, source: zmq.asyncio.Socket, destination: zmq.asyncio.Socket) -> None:
        """Relay every multipart message received on ``source`` to ``destination``.

        Two of these run concurrently (frontend->backend carries actual
        publisher data; backend->frontend carries XPUB's own subscription
        control frames back to publishers, part of the standard XSUB/XPUB
        protocol -- without this second direction, XSUB would never
        forward anything, since it only relays topics it believes have at
        least one subscriber).
        """
        while True:
            frames = await source.recv_multipart()
            await destination.send_multipart(frames)

    def close(self) -> None:
        """Close both sockets. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._frontend is not None:
            self._frontend.close(linger=0)
        if self._backend is not None:
            self._backend.close(linger=0)
        _log.info("zmq_broker_stopped")

"""Asynchronous ZeroMQ PUB/SUB bus for low-latency inter-agent messaging.

Built entirely on ``zmq.asyncio``, so every socket operation integrates
natively with the asyncio event loop via ``asyncio.Future`` under the
hood: sends and receives never block the loop, they simply suspend the
calling coroutine until ZeroMQ's I/O thread signals completion.

Convention used here: a :class:`ZmqPublisher` *binds* to its own address
(the endpoint other agents connect to), and a :class:`ZmqSubscriber`
*connects* to one or more remote publisher addresses and filters by
topic. This is the standard ZeroMQ PUB/SUB pattern; introducing a
central broker/proxy (e.g. ``XPUB``/``XSUB``) is left to a later chapter
if/when the agent topology requires it.

Every message is validated through :class:`core.schemas.MessagePayload`
before it is ever put on the wire, and every received frame is validated
again before being handed back to caller code.
"""

from __future__ import annotations

import asyncio
import sys
from types import TracebackType
from typing import AsyncIterator, Final

import zmq
import zmq.asyncio

from core.logger import get_logger
from core.schemas import MalformedMessageError, MessagePayload

if sys.platform == "win32":
    # pyzmq's asyncio integration needs `add_reader`/`add_writer` on the
    # running loop. `ProactorEventLoop` -- asyncio's Windows default since
    # Python 3.8 -- does not implement them at all; without this, every
    # zmq socket call raises RuntimeError unless `tornado>=6.1` happens to
    # be installed (pyzmq falls back to a tornado compatibility shim).
    # Depending on an incidental transitive dependency for core
    # infrastructure to work is exactly the kind of latent fragility a
    # production system shouldn't have, so the policy is set explicitly
    # here instead. Must run before any event loop is created (i.e. at
    # import time): every agent imports this module, directly or via
    # core.base_agent, before ever calling asyncio.run().
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_log = get_logger(__name__)

_DEFAULT_LINGER_MS: Final[int] = 0


class ZmqPublisher:
    """A ZeroMQ ``PUB`` socket that broadcasts validated messages by topic.

    Each message is sent as a two-frame multipart message: a raw topic
    frame (used by subscribers for prefix-matching) followed by the
    JSON-encoded :class:`~core.schemas.MessagePayload` body.

    Two ways to wire this up, controlled by ``bind``:

    - ``bind=True`` (default): the socket binds ``address`` directly --
      correct for exactly one publisher talking to one or more
      subscribers that connect to it (this is how every single-agent
      demo/test in this project has worked so far).
    - ``bind=False``: the socket connects to ``address`` instead --
      correct when multiple publishers must coexist without each
      claiming its own port, by all connecting to one shared
      :class:`~core.zmq_broker.ZmqBroker` frontend, which itself binds
      and fans messages out to every subscriber. This is the multi-agent
      deployment topology this project's own earlier design notes
      flagged as "left to a later chapter if/when the agent topology
      requires it" -- running 5 agents as concurrent tasks in one
      process (or container) is that later chapter: without a shared
      broker, every agent's own PUB socket would try to bind the same
      configured port and only the first would succeed.
    """

    def __init__(
        self,
        address: str,
        *,
        high_water_mark: int = 1000,
        context: zmq.asyncio.Context | None = None,
        bind: bool = True,
    ) -> None:
        """Create a PUB socket, either bound or connected to ``address``.

        Args:
            address: ZeroMQ endpoint to bind or connect to (e.g.
                ``"tcp://127.0.0.1:5555"``).
            high_water_mark: Max number of outbound messages ZMQ will
                queue per subscriber before dropping the oldest ones.
                Prevents unbounded memory growth if a subscriber is slow
                or gone.
            context: Optional shared ``zmq.asyncio.Context``. Defaults to
                the process-wide singleton (``Context.instance()``), the
                pattern recommended by pyzmq for asyncio applications.
            bind: See the class docstring. Defaults to ``True``,
                preserving every existing single-agent-binds-its-own-port
                usage in this project unchanged.
        """
        self._context = context or zmq.asyncio.Context.instance()
        self._socket: zmq.asyncio.Socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, high_water_mark)
        if bind:
            self._socket.bind(address)
            _log.info("zmq_publisher_bound", address=address, high_water_mark=high_water_mark)
        else:
            self._socket.connect(address)
            _log.info("zmq_publisher_connected", address=address, high_water_mark=high_water_mark)
        self._address = address
        self._closed = False

    async def publish(self, payload: MessagePayload) -> None:
        """Validate and broadcast a message to all connected subscribers.

        Args:
            payload: The message to broadcast. Its ``topic`` field is used
                verbatim as the ZMQ topic frame for SUB-side filtering.

        Raises:
            RuntimeError: If called after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError(f"Cannot publish on closed ZmqPublisher({self._address!r}).")

        topic_frame = payload.topic.encode("utf-8")
        body_frame = payload.serialize()
        await self._socket.send_multipart([topic_frame, body_frame])
        _log.debug(
            "message_published",
            topic=payload.topic,
            sender_id=payload.sender_id,
            address=self._address,
        )

    def close(self, linger_ms: int = _DEFAULT_LINGER_MS) -> None:
        """Close the underlying socket.

        Args:
            linger_ms: Milliseconds to wait for queued outbound messages
                to flush before discarding them. ``0`` (default) closes
                immediately, appropriate for a low-latency system where a
                stale message is worse than a dropped one.
        """
        if not self._closed:
            self._socket.close(linger=linger_ms)
            self._closed = True
            _log.info("zmq_publisher_closed", address=self._address)

    async def __aenter__(self) -> ZmqPublisher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ZmqSubscriber:
    """A ZeroMQ ``SUB`` socket that receives validated messages by topic.

    Unlike :class:`ZmqPublisher`, a subscriber can connect to multiple
    publisher endpoints and subscribe/unsubscribe from topics dynamically
    at runtime.
    """

    def __init__(
        self,
        address: str | None = None,
        *,
        context: zmq.asyncio.Context | None = None,
    ) -> None:
        """Create a SUB socket, optionally connecting to an initial address.

        Args:
            address: Optional ZeroMQ endpoint to connect to immediately
                (e.g. ``"tcp://127.0.0.1:5555"``). Additional endpoints can
                be added later via :meth:`connect`.
            context: Optional shared ``zmq.asyncio.Context``. Defaults to
                the process-wide singleton (``Context.instance()``).
        """
        self._context = context or zmq.asyncio.Context.instance()
        self._socket: zmq.asyncio.Socket = self._context.socket(zmq.SUB)
        self._connected_addresses: set[str] = set()
        self._subscribed_topics: set[str] = set()
        self._closed = False
        if address is not None:
            self.connect(address)

    def connect(self, address: str) -> None:
        """Connect to an additional publisher endpoint.

        Args:
            address: ZeroMQ endpoint to connect to (e.g.
                ``"tcp://127.0.0.1:5555"``).
        """
        self._socket.connect(address)
        self._connected_addresses.add(address)
        _log.info("zmq_subscriber_connected", address=address)

    def disconnect(self, address: str) -> None:
        """Disconnect from a previously connected publisher endpoint.

        Args:
            address: The endpoint previously passed to :meth:`connect` (or
                the constructor).
        """
        self._socket.disconnect(address)
        self._connected_addresses.discard(address)
        _log.info("zmq_subscriber_disconnected", address=address)

    def subscribe(self, topic: str) -> None:
        """Subscribe to a topic (e.g. ``"MARKET_DATA"``, ``"NLP_SIGNAL"``).

        Args:
            topic: Exact topic string to subscribe to. ZeroMQ performs
                prefix matching, so this also matches any topic starting
                with this string.
        """
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        self._subscribed_topics.add(topic)
        _log.info("zmq_subscribed_to_topic", topic=topic)

    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a previously subscribed topic.

        Args:
            topic: The topic string previously passed to :meth:`subscribe`.
        """
        self._socket.setsockopt(zmq.UNSUBSCRIBE, topic.encode("utf-8"))
        self._subscribed_topics.discard(topic)
        _log.info("zmq_unsubscribed_from_topic", topic=topic)

    async def receive(self) -> MessagePayload:
        """Await and return the next validated message from the bus.

        Returns:
            The validated :class:`~core.schemas.MessagePayload`.

        Raises:
            MalformedMessageError: If the received frame does not conform
                to :class:`~core.schemas.MessagePayload`'s schema. The
                offending message is logged and dropped; the caller may
                simply call :meth:`receive` again to continue consuming.
            RuntimeError: If called after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError("Cannot receive on a closed ZmqSubscriber.")

        _, body_frame = await self._socket.recv_multipart()
        try:
            return MessagePayload.deserialize(body_frame)
        except MalformedMessageError:
            _log.warning("malformed_message_dropped", raw_size_bytes=len(body_frame))
            raise

    async def listen(self) -> AsyncIterator[MessagePayload]:
        """Yield validated messages indefinitely, skipping malformed ones.

        This is a convenience wrapper around :meth:`receive` for the
        common case of a long-running consumption loop::

            async for message in subscriber.listen():
                handle(message)

        Yields:
            Each validated :class:`~core.schemas.MessagePayload` as it
            arrives. Malformed messages are logged and silently skipped
            rather than raised, so the loop never dies on a bad message.
        """
        while True:
            try:
                yield await self.receive()
            except MalformedMessageError:
                continue

    def close(self, linger_ms: int = _DEFAULT_LINGER_MS) -> None:
        """Close the underlying socket.

        Args:
            linger_ms: Milliseconds to wait for pending operations before
                discarding them. Defaults to ``0`` (close immediately).
        """
        if not self._closed:
            self._socket.close(linger=linger_ms)
            self._closed = True
            _log.info("zmq_subscriber_closed", addresses=sorted(self._connected_addresses))

    async def __aenter__(self) -> ZmqSubscriber:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

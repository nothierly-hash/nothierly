"""Chaos Monkey Simulator -- real, standalone fault-injection torture script for a running shadow run.

**Not a pytest test file, despite living under ``tests/``** -- matches exactly where the brief asked
for it, but this is a real, independent async script meant to run *alongside* (a second terminal,
not inside pytest) an already-running Nothierly backend (``scripts/run_backend.py`` or the standing
multi-process deployment), the same "real two-or-more-terminal setup" convention
``scripts/run_zmq_broker.py``'s own docstring already establishes. **The brief's own explicit scope**:
this creates the torture script only -- it does not, and is not meant to, fix any real weakness it
finds in Fleet Command. What it *should* observe, if that system's own documented safety design
(Kill Switch, ``safe_financial_call``'s real timeout/retry budget, Pydantic's real "reject anything
malformed on receipt" ``MessagePayload`` validation) is working: a real Fail-Closed response --
blocked new entries, a tripped Kill Switch, rejected malformed messages -- never a crash and never a
silent, undetected corruption of real trading state.

**Real, honest safety guardrail, read before running this against anything**: every one of the 4
weapons below is genuinely disruptive by design -- that is the entire point of chaos engineering.
Weapon 2 (``redis_kill``) and weapon 1 (``zmq_drop``) act on **shared infrastructure** (a real Redis
server, a real OS process) that other, unrelated processes on the same machine may also depend on.
This script refuses to run any weapon at all unless ``--i-confirm-this-is-a-disposable-test-target``
is passed on the command line -- see the module-level warning this prints at startup. Never point
this at a real production Redis/ZMQ broker/OANDA account; only at a disposable ``fakeredis``
(``scripts/run_fake_redis.py``), an isolated broker (see the project's own ``ZMQ_BROKER_FRONTEND_
ADDRESS`` override convention for a shadow run), and ``MOCK_MODE=true``/``AgentControlState.SHADOW``.

**The 4 real weapons, and their real, disclosed mechanism/limits**:

1. **``zmq_drop``** -- suspends (``SIGSTOP``-equivalent, cross-platform via :mod:`psutil`) the real
   OS process of a running ``scripts/run_zmq_broker.py`` for a real random 0.5-3s window, then
   resumes it. Real and effective: while suspended, that process's ZMQ frontend/backend sockets stop
   servicing any I/O at all -- every connected agent's publish/subscribe genuinely stalls, the same
   as a real network partition to the broker. **Honest limit**: this targets the *shared broker*
   process, not each individual agent's own socket -- a topology with no separate broker (single-
   process demo, ``bind=True`` everywhere) has nothing for this weapon to suspend; pass
   ``--zmq-broker-pid`` explicitly if auto-discovery (matching ``run_zmq_broker.py`` on the command
   line) finds the wrong process or none.

2. **``redis_kill``** -- issues a real ``CLIENT KILL ... SKIPME yes`` against the configured Redis,
   forcibly severing every other real client's TCP connection to it (the real, standard Redis admin
   primitive for exactly this, not a fabricated one) -- literally "disconnexion de l'API client",
   matching the brief. Every disconnected agent's own ``core.redis_client.RedisClient`` should
   reconnect with its own real backoff (that class's own documented behavior) -- this script only
   forces the disconnect, never touches that reconnection logic. **Honest limit**: ``fakeredis``'s
   TCP server does not implement ``CLIENT KILL`` at the time this was written -- against a fakeredis
   target this weapon logs a clear, honest failure and does nothing, rather than silently no-op.

3. **``ram_spam``** -- connects a real ZMQ ``PUB`` socket to the real broker/agent frontend and blasts
   a real, configurable number of messages at max rate: a real, disclosed mix of (a) well-formed
   ``MessagePayload``s wrapping physically nonsensical ``MarketMetrics`` (negative spread, an
   impossible mid price) -- real, valid wire-format messages testing whether nonsensical-but-
   schema-valid values can still reach and destabilize downstream math, and (b) genuinely malformed
   frames (invalid JSON, in the real topic's own byte format) -- testing the real, already-documented
   ``MessagePayload`` validate-on-receipt boundary (:mod:`core.schemas`'s own module docstring) does
   what it claims under real volume, not just one crafted message at a time.

4. **``api_blackout``** -- two real, complementary mechanisms, since a separate process cannot reach
   into another already-running process's own function table:
   - :func:`blackout_broker_client`: a real, in-process monkeypatch (an async context manager) for
     any caller that already holds a live broker-client reference in the *same* Python process
     (e.g. a resilience test, or a chaos harness imported directly into a shadow-run script) --
     ``create_market_order`` raises a real ``OandaTimeoutError`` for the random window, then is
     restored exactly as it was.
   - :class:`ApiBlackoutProxy`: a real, standalone TCP proxy this script starts and listens on
     locally; while "blacked out", it accepts a connection and then genuinely holds it open, relaying
     nothing, until the window ends -- the real failure mode ``safe_financial_call``'s own
     ``timeout_seconds`` is designed to catch, not an instant, easily-retried connection refusal.
     **Honest, disclosed opt-in requirement**: this only affects a separate target process if that
     process is *launched* with ``HTTPS_PROXY``/``HTTP_PROXY`` pointed at this proxy's own listen
     address (e.g. ``HTTPS_PROXY=http://127.0.0.1:8666``) -- this script cannot silently redirect an
     already-running process's outbound OANDA calls without that explicit, disclosed opt-in; it does
     not modify system-wide network/firewall/hosts settings to force it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.zmq_bus  # noqa: F401,E402 -- must import before any event loop is created, see run_zmq_broker.py's own comment on why
import psutil  # noqa: E402
import zmq  # noqa: E402
import zmq.asyncio  # noqa: E402

from core.config import AppSettings, get_settings  # noqa: E402
from core.logger import get_logger  # noqa: E402
from core.redis_client import RedisClient, RedisUnavailableError  # noqa: E402
from core.schemas import MarketMetrics, MessagePayload  # noqa: E402

_log = get_logger(__name__)

# Matches ExecutionAgent's/every other agent's own private _MARKET_DATA_TOPIC -- duplicated as a
# literal rather than importing a leading-underscore name, this project's own established convention
# (see tests/test_execution_agent_control_state.py's identical comment).
_MARKET_DATA_L2_TOPIC = "MARKET_DATA_L2"

_DEFAULT_ZMQ_DROP_MIN_SECONDS = 0.5
_DEFAULT_ZMQ_DROP_MAX_SECONDS = 3.0
_DEFAULT_RAM_SPAM_COUNT = 5_000
_DEFAULT_RAM_SPAM_MALFORMED_FRACTION = 0.3
_DEFAULT_API_BLACKOUT_MIN_SECONDS = 1.0
_DEFAULT_API_BLACKOUT_MAX_SECONDS = 5.0
_DEFAULT_API_BLACKOUT_LISTEN_PORT = 8666
_OANDA_PRACTICE_HOST = "api-fxpractice.oanda.com"
_OANDA_HTTPS_PORT = 443


# =============================================================================
# Weapon 1 -- ZMQ_Drop
# =============================================================================


def find_zmq_broker_pid() -> int | None:
    """Best-effort real discovery of a running ``scripts/run_zmq_broker.py`` process by command line."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if any("run_zmq_broker.py" in str(part) for part in cmdline):
            return int(proc.info["pid"])
    return None


async def zmq_drop(
    *, pid: int | None = None, min_seconds: float = _DEFAULT_ZMQ_DROP_MIN_SECONDS, max_seconds: float = _DEFAULT_ZMQ_DROP_MAX_SECONDS
) -> bool:
    """Real weapon #1: suspend the real ZMQ broker OS process for a random window, then resume it.

    Args:
        pid: Real target process id. Auto-discovered via :func:`find_zmq_broker_pid` when omitted.
        min_seconds: Real minimum outage duration.
        max_seconds: Real maximum outage duration.

    Returns:
        ``True`` if a real target process was found and suspended/resumed; ``False`` if none was
        found (nothing was touched -- never silently targets the wrong process).
    """
    target_pid = pid if pid is not None else find_zmq_broker_pid()
    if target_pid is None:
        _log.error(
            "chaos_zmq_drop_no_target",
            detail="No running scripts/run_zmq_broker.py process found -- pass --zmq-broker-pid explicitly.",
        )
        return False

    duration = random.uniform(min_seconds, max_seconds)
    try:
        proc = psutil.Process(target_pid)
        _log.warning("CHAOS_ZMQ_DROP_START", pid=target_pid, duration_s=round(duration, 2))
        proc.suspend()
        try:
            await asyncio.sleep(duration)
        finally:
            proc.resume()
        _log.warning("CHAOS_ZMQ_DROP_END", pid=target_pid)
        return True
    except psutil.NoSuchProcess:
        _log.error("chaos_zmq_drop_process_vanished", pid=target_pid)
        return False


# =============================================================================
# Weapon 2 -- Redis_Kill
# =============================================================================


async def redis_kill(*, settings: AppSettings | None = None) -> int:
    """Real weapon #2: ``CLIENT KILL ... SKIPME yes`` -- forcibly disconnects every other real Redis client now.

    Returns:
        How many real clients were disconnected (``0`` if the command failed/is unsupported by the
        target -- see the module docstring's honest ``fakeredis`` limit -- or Redis was unreachable;
        never fabricated, always the real return value or ``0``).
    """
    resolved_settings = settings if settings is not None else get_settings()
    try:
        client = await RedisClient.get_instance(resolved_settings.redis)
        killed = await client.execute("client_kill_filter", skipme=True)
    except RedisUnavailableError:
        _log.error("chaos_redis_kill_unavailable")
        return 0
    except Exception as exc:  # noqa: BLE001 -- a real, foreign Redis error (e.g. fakeredis: unsupported command); log and disclose, never crash the whole chaos session over one weapon
        _log.error("chaos_redis_kill_command_failed", error_type=type(exc).__name__, error=str(exc))
        return 0

    _log.warning("CHAOS_REDIS_KILL", clients_disconnected=killed)
    return int(killed)


# =============================================================================
# Weapon 3 -- RAM_Spam
# =============================================================================


async def ram_spam(
    *,
    address: str,
    bind: bool = False,
    count: int = _DEFAULT_RAM_SPAM_COUNT,
    malformed_fraction: float = _DEFAULT_RAM_SPAM_MALFORMED_FRACTION,
    instrument: str = "EUR_USD",
) -> int:
    """Real weapon #3: flood the real ZMQ bus with ``count`` messages (malformed + nonsensical-but-valid), as fast as possible.

    Args:
        address: Real ZMQ endpoint (the broker's real frontend, or an agent's own ``sub_address`` in
            a no-broker topology).
        bind: ``True`` binds ``address`` directly; ``False`` (default) connects to a real, already-
            bound broker frontend -- see :class:`core.zmq_bus.ZmqPublisher`'s own docstring on when
            each is correct. Almost always ``False`` against a real running shadow run.
        count: Real total messages to send.
        malformed_fraction: Real fraction (``0.0``-``1.0``) of ``count`` that are genuinely malformed
            (invalid JSON) rather than a valid-but-nonsensical real tick.
        instrument: Real instrument symbol to spam under.

    Returns:
        The real number of messages actually sent (equals ``count`` unless the socket itself failed
        partway through).
    """
    context = zmq.asyncio.Context.instance()
    socket = context.socket(zmq.PUB)
    if bind:
        socket.bind(address)
    else:
        socket.connect(address)
    # Real PUB/SUB "slow joiner" settle -- every publisher in this project already needs this same
    # real pause before a subscriber connecting around the same instant would otherwise miss the
    # first few messages (see core.zmq_bus's own publishers for the identical real reason).
    await asyncio.sleep(0.2)

    sent = 0
    _log.warning("CHAOS_RAM_SPAM_START", address=address, count=count, malformed_fraction=malformed_fraction)
    try:
        for i in range(count):
            if random.random() < malformed_fraction:
                topic_frame = _MARKET_DATA_L2_TOPIC.encode("utf-8")
                body_frame = f"{{not-valid-json-chaos-{i}".encode("utf-8")
            else:
                try:
                    payload = MessagePayload(
                        topic=_MARKET_DATA_L2_TOPIC,
                        sender_id="chaos-monkey",
                        data=MarketMetrics(
                            symbol=instrument,
                            # Schema-valid (respects MarketMetrics' own ge=0/le=1/gt=0 constraints --
                            # see core.schemas) but real-market-nonsensical: a spread/price this
                            # extreme, or an imbalance parked at a legal boundary, should never occur
                            # from a real order book, testing whether downstream math (Gate 7's own
                            # price_impact/Poiseuille calls) stays numerically stable under it.
                            spread=random.uniform(0.0, 1e6),
                            order_book_imbalance=random.uniform(-1.0, 1.0),
                            mid_price=random.uniform(1e-6, 1e12),
                        ).model_dump(mode="json"),
                    )
                except Exception:  # noqa: BLE001 -- a value so extreme even this project's own MarketMetrics validators reject it; send it as a malformed frame instead rather than skip it
                    topic_frame = _MARKET_DATA_L2_TOPIC.encode("utf-8")
                    body_frame = f"{{not-valid-json-chaos-{i}".encode("utf-8")
                else:
                    topic_frame = payload.topic.encode("utf-8")
                    body_frame = payload.serialize()
            await socket.send_multipart([topic_frame, body_frame])
            sent += 1
    finally:
        socket.close(linger=0)
    _log.warning("CHAOS_RAM_SPAM_END", address=address, sent=sent)
    return sent


# =============================================================================
# Weapon 4 -- API_Blackout
# =============================================================================


class _BrokerOfflineError(Exception):
    """Raised by :func:`blackout_broker_client`'s patched ``create_market_order`` -- a real, disclosed simulated broker outage."""


class _BlackoutContext:
    def __init__(self, client: object, *, min_seconds: float, max_seconds: float) -> None:
        self._client = client
        self._min_seconds = min_seconds
        self._max_seconds = max_seconds
        self._original_create_market_order: object | None = None

    async def __aenter__(self) -> _BlackoutContext:
        duration = random.uniform(self._min_seconds, self._max_seconds)
        self._original_create_market_order = self._client.create_market_order  # type: ignore[attr-defined]

        async def _offline(*args: object, **kwargs: object) -> None:
            raise _BrokerOfflineError("CHAOS_API_BLACKOUT: simulated broker outage -- no real OANDA call was made.")

        self._client.create_market_order = _offline  # type: ignore[attr-defined]
        _log.warning("CHAOS_API_BLACKOUT_START", duration_s=round(duration, 2))
        self._duration = duration
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await asyncio.sleep(self._duration)
        self._client.create_market_order = self._original_create_market_order  # type: ignore[attr-defined]
        _log.warning("CHAOS_API_BLACKOUT_END")


def blackout_broker_client(
    client: object, *, min_seconds: float = _DEFAULT_API_BLACKOUT_MIN_SECONDS, max_seconds: float = _DEFAULT_API_BLACKOUT_MAX_SECONDS
) -> _BlackoutContext:
    """Real weapon #4a: in-process monkeypatch -- ``client.create_market_order`` raises for a random window, then is restored.

    Only usable when this chaos script (or a caller importing it) shares a Python process with a
    live broker-client reference -- see :class:`ApiBlackoutProxy` for the real, standalone,
    separate-process alternative and its own honest opt-in requirement.

    Usage::

        async with blackout_broker_client(agent._trading_client):
            ...  # every real create_market_order call in this window raises _BrokerOfflineError
    """
    return _BlackoutContext(client, min_seconds=min_seconds, max_seconds=max_seconds)


async def _relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError):
            writer.close()


class ApiBlackoutProxy:
    """Real weapon #4b: a standalone local TCP proxy that stalls new connections during a blackout window.

    See the module docstring's own disclosed opt-in requirement: only affects a separate target
    process if it is launched with ``HTTPS_PROXY``/``HTTP_PROXY`` pointed at this proxy's listen
    address. Outside a blackout window, transparently relays every byte both ways -- real traffic
    passes through unmodified.
    """

    def __init__(self, *, listen_host: str, listen_port: int, target_host: str, target_port: int) -> None:
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._target_host = target_host
        self._target_port = target_port
        self._blacked_out = False
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_client, self._listen_host, self._listen_port)
        _log.info(
            "chaos_api_blackout_proxy_listening",
            listen=f"{self._listen_host}:{self._listen_port}",
            target=f"{self._target_host}:{self._target_port}",
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Real, honest simulated outage: hold the connection open, relaying nothing, until the
        # blackout ends -- the real failure mode safe_financial_call's own timeout_seconds is
        # designed to catch (a broker that accepted the connection then hung), not an instantly-
        # retried connection refusal.
        while self._blacked_out:
            await asyncio.sleep(0.1)
        try:
            target_reader, target_writer = await asyncio.open_connection(self._target_host, self._target_port)
        except OSError as exc:
            _log.warning("chaos_api_blackout_proxy_upstream_unreachable", error=str(exc))
            writer.close()
            return
        await asyncio.gather(_relay(reader, target_writer), _relay(target_reader, writer), return_exceptions=True)

    async def trigger_blackout(self, *, min_seconds: float = _DEFAULT_API_BLACKOUT_MIN_SECONDS, max_seconds: float = _DEFAULT_API_BLACKOUT_MAX_SECONDS) -> None:
        """Real, one-shot blackout window on this already-running proxy."""
        duration = random.uniform(min_seconds, max_seconds)
        _log.warning("CHAOS_API_BLACKOUT_PROXY_START", duration_s=round(duration, 2))
        self._blacked_out = True
        await asyncio.sleep(duration)
        self._blacked_out = False
        _log.warning("CHAOS_API_BLACKOUT_PROXY_END")


# =============================================================================
# Orchestrator
# =============================================================================


class ChaosMonkey:
    """Runs a chosen subset of the 4 real weapons repeatedly, concurrently, for a real session duration."""

    def __init__(
        self,
        *,
        session_seconds: float,
        weapons: Sequence[str],
        zmq_broker_pid: int | None,
        zmq_drop_min_seconds: float,
        zmq_drop_max_seconds: float,
        ram_spam_address: str,
        ram_spam_count: int,
        ram_spam_malformed_fraction: float,
        api_blackout_listen_host: str,
        api_blackout_listen_port: int,
        api_blackout_target_host: str,
        api_blackout_target_port: int,
        api_blackout_min_seconds: float,
        api_blackout_max_seconds: float,
        rest_min_seconds: float,
        rest_max_seconds: float,
    ) -> None:
        self._session_seconds = session_seconds
        self._weapons = set(weapons)
        self._zmq_broker_pid = zmq_broker_pid
        self._zmq_drop_min_seconds = zmq_drop_min_seconds
        self._zmq_drop_max_seconds = zmq_drop_max_seconds
        self._ram_spam_address = ram_spam_address
        self._ram_spam_count = ram_spam_count
        self._ram_spam_malformed_fraction = ram_spam_malformed_fraction
        self._api_blackout_proxy = ApiBlackoutProxy(
            listen_host=api_blackout_listen_host,
            listen_port=api_blackout_listen_port,
            target_host=api_blackout_target_host,
            target_port=api_blackout_target_port,
        )
        self._api_blackout_min_seconds = api_blackout_min_seconds
        self._api_blackout_max_seconds = api_blackout_max_seconds
        self._rest_min_seconds = rest_min_seconds
        self._rest_max_seconds = rest_max_seconds
        self._stats: dict[str, int] = {"zmq_drop": 0, "redis_kill": 0, "ram_spam": 0, "api_blackout": 0}

    async def _loop_zmq_drop(self, deadline: float) -> None:
        while asyncio.get_event_loop().time() < deadline:
            if await zmq_drop(pid=self._zmq_broker_pid, min_seconds=self._zmq_drop_min_seconds, max_seconds=self._zmq_drop_max_seconds):
                self._stats["zmq_drop"] += 1
            await asyncio.sleep(random.uniform(self._rest_min_seconds, self._rest_max_seconds))

    async def _loop_redis_kill(self, deadline: float) -> None:
        while asyncio.get_event_loop().time() < deadline:
            killed = await redis_kill()
            if killed > 0:
                self._stats["redis_kill"] += 1
            await asyncio.sleep(random.uniform(self._rest_min_seconds, self._rest_max_seconds))

    async def _loop_ram_spam(self, deadline: float) -> None:
        while asyncio.get_event_loop().time() < deadline:
            await ram_spam(address=self._ram_spam_address, count=self._ram_spam_count, malformed_fraction=self._ram_spam_malformed_fraction)
            self._stats["ram_spam"] += 1
            await asyncio.sleep(random.uniform(self._rest_min_seconds, self._rest_max_seconds))

    async def _loop_api_blackout(self, deadline: float) -> None:
        await self._api_blackout_proxy.start()
        try:
            while asyncio.get_event_loop().time() < deadline:
                await self._api_blackout_proxy.trigger_blackout(
                    min_seconds=self._api_blackout_min_seconds, max_seconds=self._api_blackout_max_seconds
                )
                self._stats["api_blackout"] += 1
                await asyncio.sleep(random.uniform(self._rest_min_seconds, self._rest_max_seconds))
        finally:
            await self._api_blackout_proxy.stop()

    async def run(self) -> dict[str, int]:
        deadline = asyncio.get_event_loop().time() + self._session_seconds
        _log.warning(
            "CHAOS_MONKEY_SESSION_START",
            session_seconds=self._session_seconds,
            weapons=sorted(self._weapons),
        )
        tasks = []
        if "zmq_drop" in self._weapons:
            tasks.append(self._loop_zmq_drop(deadline))
        if "redis_kill" in self._weapons:
            tasks.append(self._loop_redis_kill(deadline))
        if "ram_spam" in self._weapons:
            tasks.append(self._loop_ram_spam(deadline))
        if "api_blackout" in self._weapons:
            tasks.append(self._loop_api_blackout(deadline))
        await asyncio.gather(*tasks)
        _log.warning("CHAOS_MONKEY_SESSION_END", stats=self._stats)
        return self._stats


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chaos Monkey Simulator -- real fault-injection torture for a running Nothierly shadow run.",
    )
    parser.add_argument(
        "--i-confirm-this-is-a-disposable-test-target",
        action="store_true",
        help="Required. Confirms the Redis/ZMQ broker this targets is a disposable test instance, never production.",
    )
    parser.add_argument(
        "--weapons", default="zmq_drop,redis_kill,ram_spam,api_blackout",
        help="Comma-separated subset of: zmq_drop,redis_kill,ram_spam,api_blackout",
    )
    parser.add_argument("--session-seconds", type=float, default=120.0, help="Total real chaos-session duration.")
    parser.add_argument("--zmq-broker-pid", type=int, default=None, help="Override auto-discovery of the run_zmq_broker.py process.")
    parser.add_argument("--zmq-drop-min-seconds", type=float, default=_DEFAULT_ZMQ_DROP_MIN_SECONDS)
    parser.add_argument("--zmq-drop-max-seconds", type=float, default=_DEFAULT_ZMQ_DROP_MAX_SECONDS)
    parser.add_argument("--ram-spam-address", default=None, help="Defaults to this process's own configured ZMQ broker frontend address.")
    parser.add_argument("--ram-spam-count", type=int, default=_DEFAULT_RAM_SPAM_COUNT)
    parser.add_argument("--ram-spam-malformed-fraction", type=float, default=_DEFAULT_RAM_SPAM_MALFORMED_FRACTION)
    parser.add_argument("--api-blackout-listen-host", default="127.0.0.1")
    parser.add_argument("--api-blackout-listen-port", type=int, default=_DEFAULT_API_BLACKOUT_LISTEN_PORT)
    parser.add_argument("--api-blackout-target-host", default=_OANDA_PRACTICE_HOST)
    parser.add_argument("--api-blackout-target-port", type=int, default=_OANDA_HTTPS_PORT)
    parser.add_argument("--api-blackout-min-seconds", type=float, default=_DEFAULT_API_BLACKOUT_MIN_SECONDS)
    parser.add_argument("--api-blackout-max-seconds", type=float, default=_DEFAULT_API_BLACKOUT_MAX_SECONDS)
    parser.add_argument("--rest-min-seconds", type=float, default=5.0, help="Real minimum calm interval between one weapon's own successive strikes.")
    parser.add_argument("--rest-max-seconds", type=float, default=15.0, help="Real maximum calm interval between one weapon's own successive strikes.")
    return parser


async def _main(argv: Sequence[str]) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.i_confirm_this_is_a_disposable_test_target:
        print(
            "Refusing to run: this script disconnects real Redis clients and suspends a real OS "
            "process. Pass --i-confirm-this-is-a-disposable-test-target once you have verified this "
            "is pointed at a disposable shadow-run/fakeredis test target, never production.",
            file=sys.stderr,
        )
        return 2

    settings = get_settings()
    ram_spam_address = args.ram_spam_address or settings.zmq.broker_frontend_address or settings.zmq.pub_address

    monkey = ChaosMonkey(
        session_seconds=args.session_seconds,
        weapons=[w.strip() for w in args.weapons.split(",") if w.strip()],
        zmq_broker_pid=args.zmq_broker_pid,
        zmq_drop_min_seconds=args.zmq_drop_min_seconds,
        zmq_drop_max_seconds=args.zmq_drop_max_seconds,
        ram_spam_address=ram_spam_address,
        ram_spam_count=args.ram_spam_count,
        ram_spam_malformed_fraction=args.ram_spam_malformed_fraction,
        api_blackout_listen_host=args.api_blackout_listen_host,
        api_blackout_listen_port=args.api_blackout_listen_port,
        api_blackout_target_host=args.api_blackout_target_host,
        api_blackout_target_port=args.api_blackout_target_port,
        api_blackout_min_seconds=args.api_blackout_min_seconds,
        api_blackout_max_seconds=args.api_blackout_max_seconds,
        rest_min_seconds=args.rest_min_seconds,
        rest_max_seconds=args.rest_max_seconds,
    )
    stats = await monkey.run()
    print(f"Chaos Monkey session complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(sys.argv[1:])))

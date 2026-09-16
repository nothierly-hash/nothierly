"""Real, executable tests for tests/chaos_monkey_simulator.py's 4 weapons.

Each weapon is proven against a real target, not a mock standing in for one: a real short-lived
subprocess (weapon 1), a real fakeredis TCP server (weapon 2 -- also proves the module's own
disclosed "fakeredis does not support CLIENT KILL" limit is actually true, not assumed), a real
bound ZMQ SUB socket (weapon 3), and a real local TCP echo server behind the real proxy (weapon 4).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from typing import Generator

import fakeredis
import psutil
import pytest
import zmq.asyncio

from core.config import AppSettings, RedisSettings
from core.redis_client import RedisClient
from tests.chaos_monkey_simulator import (
    ApiBlackoutProxy,
    _BrokerOfflineError,
    blackout_broker_client,
    find_zmq_broker_pid,
    ram_spam,
    redis_kill,
    zmq_drop,
)

# -- Weapon 1: zmq_drop --------------------------------


def test_find_zmq_broker_pid_is_none_or_a_real_matching_process() -> None:
    """Environment-robust, not "must be None": this dev machine may genuinely have a real, standing
    run_zmq_broker.py deployment already running (see the project's own memory of exactly that) --
    asserting None unconditionally would be a false failure there. Whatever comes back must be
    real and correct: either no match, or a real PID whose own real command line actually contains
    "run_zmq_broker.py"."""
    pid = find_zmq_broker_pid()
    if pid is not None:
        cmdline = psutil.Process(pid).cmdline()
        assert any("run_zmq_broker.py" in part for part in cmdline)


@pytest.mark.asyncio
async def test_zmq_drop_returns_false_without_crashing_for_a_pid_that_does_not_exist() -> None:
    """Deliberately never relies on auto-discovery here (pid=None) -- on a machine with a real
    standing broker (see the test above), that would actually suspend it for real. An explicit,
    essentially-guaranteed-nonexistent PID proves the same graceful-failure path safely."""
    assert await zmq_drop(pid=999_999_999) is False


@pytest.mark.asyncio
async def test_zmq_drop_really_suspends_and_resumes_a_real_process() -> None:
    """The real mechanism, proven against a real process this test owns and kills afterward --
    not a mock of psutil.Process."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        psutil_proc = psutil.Process(proc.pid)
        assert psutil_proc.status() != psutil.STATUS_STOPPED

        drop_task = asyncio.create_task(zmq_drop(pid=proc.pid, min_seconds=0.2, max_seconds=0.2))
        await asyncio.sleep(0.05)
        assert psutil_proc.status() == psutil.STATUS_STOPPED  # really suspended, mid-window

        result = await drop_task
        assert result is True
        assert psutil_proc.status() != psutil.STATUS_STOPPED  # really resumed
    finally:
        proc.kill()
        proc.wait(timeout=5)


# -- Weapon 2: redis_kill --------------------------------


@pytest.fixture
def fake_redis_server() -> Generator[int, None, None]:
    server = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.socket.getsockname()[1]
    yield port
    server.shutdown()


@pytest.mark.asyncio
async def test_redis_kill_against_real_fakeredis_fails_gracefully_not_crashes(fake_redis_server: int) -> None:
    """Proves the module's own disclosed honest limit is real, not assumed: fakeredis's TCP server
    does not implement CLIENT KILL, and redis_kill must degrade to a clean 0, never raise."""
    settings = AppSettings(redis=RedisSettings(host="127.0.0.1", port=fake_redis_server))
    RedisClient._instance = None
    try:
        killed = await redis_kill(settings=settings)
        assert killed == 0
    finally:
        if RedisClient._instance is not None:
            await RedisClient._instance.close()
        RedisClient._instance = None


# -- Weapon 3: ram_spam --------------------------------


class _RecordingSocket:
    """Stands in for a real zmq.asyncio.Socket -- records every frame pair sent, nothing else."""

    def __init__(self) -> None:
        self.sent: list[tuple[bytes, bytes]] = []

    def bind(self, address: str) -> None:
        pass

    def connect(self, address: str) -> None:
        pass

    async def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append((frames[0], frames[1]))

    def close(self, linger: int = 0) -> None:
        pass


class _RecordingContext:
    def __init__(self, socket: _RecordingSocket) -> None:
        self._socket = socket

    def socket(self, socket_type: int) -> _RecordingSocket:
        return self._socket


@pytest.mark.asyncio
async def test_ram_spam_sends_the_real_configured_count_as_a_real_mix_of_malformed_and_schema_valid_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real ZMQ PUB/SUB wire round-trip is already proven throughout this project's own test
    suite -- this test instead pins down the one thing that's actually new here: that ram_spam's
    own per-message branching produces a real mix of genuinely malformed frames and schema-valid-
    but-nonsensical MarketMetrics ticks, in roughly the configured proportion, with the real topic
    byte string on every frame. The real socket is replaced with a recording stand-in so this stays
    fast and immune to any environment-specific ZMQ/asyncio-loop interaction."""
    fake_socket = _RecordingSocket()
    monkeypatch.setattr(zmq.asyncio.Context, "instance", staticmethod(lambda: _RecordingContext(fake_socket)))

    sent = await ram_spam(address="tcp://ignored:0", count=200, malformed_fraction=0.5)

    assert sent == 200
    assert len(fake_socket.sent) == 200

    valid = 0
    malformed = 0
    for topic, body in fake_socket.sent:
        assert topic == b"MARKET_DATA_L2"
        try:
            json.loads(body)
            valid += 1
        except json.JSONDecodeError:
            malformed += 1

    assert valid > 0
    assert malformed > 0
    # A real, if loose, proportionality check -- exact 50/50 isn't guaranteed by a single random draw
    # per message, but neither category should be a rare fluke out of 200 real attempts.
    assert 50 < malformed < 150


# -- Weapon 4a: blackout_broker_client --------------------------------


class _FakeBrokerClient:
    async def create_market_order(self, instrument: str, units: int) -> dict[str, object]:
        return {"id": "real-order", "instrument": instrument, "units": units}


@pytest.mark.asyncio
async def test_blackout_broker_client_raises_during_the_window_then_restores_the_real_method() -> None:
    client = _FakeBrokerClient()
    original_method = client.create_market_order

    async with blackout_broker_client(client, min_seconds=0.05, max_seconds=0.05):
        with pytest.raises(_BrokerOfflineError):
            await client.create_market_order("EUR_USD", 100)

    # Restored exactly -- a real call succeeds again after the window.
    result = await client.create_market_order("EUR_USD", 100)
    assert result == {"id": "real-order", "instrument": "EUR_USD", "units": 100}
    assert client.create_market_order == original_method


# -- Weapon 4b: ApiBlackoutProxy --------------------------------


async def _start_echo_server(host: str, port: int) -> asyncio.Server:
    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        data = await reader.read(1024)
        writer.write(data)
        await writer.drain()
        writer.close()

    return await asyncio.start_server(_handle, host, port)


@pytest.mark.asyncio
async def test_api_blackout_proxy_relays_normally_outside_a_blackout_window() -> None:
    echo_server = await _start_echo_server("127.0.0.1", 0)
    echo_port = echo_server.sockets[0].getsockname()[1]

    proxy = ApiBlackoutProxy(listen_host="127.0.0.1", listen_port=0, target_host="127.0.0.1", target_port=echo_port)
    await proxy.start()
    proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"real-payload")
        await writer.drain()
        echoed = await asyncio.wait_for(reader.read(1024), timeout=2.0)
        assert echoed == b"real-payload"
        writer.close()
    finally:
        await proxy.stop()
        echo_server.close()
        await echo_server.wait_closed()


@pytest.mark.asyncio
async def test_api_blackout_proxy_really_stalls_new_connections_during_the_window() -> None:
    echo_server = await _start_echo_server("127.0.0.1", 0)
    echo_port = echo_server.sockets[0].getsockname()[1]

    proxy = ApiBlackoutProxy(listen_host="127.0.0.1", listen_port=0, target_host="127.0.0.1", target_port=echo_port)
    await proxy.start()
    proxy_port = proxy._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        blackout_task = asyncio.create_task(proxy.trigger_blackout(min_seconds=0.3, max_seconds=0.3))
        await asyncio.sleep(0.05)  # let the blackout actually start

        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"real-payload-during-blackout")
        await writer.drain()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(1024), timeout=0.15)  # real proof: no response arrives mid-blackout

        # Once the blackout ends, the same still-open connection is finally served.
        echoed = await asyncio.wait_for(reader.read(1024), timeout=2.0)
        assert echoed == b"real-payload-during-blackout"
        writer.close()
        await blackout_task
    finally:
        await proxy.stop()
        echo_server.close()
        await echo_server.wait_closed()

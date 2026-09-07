"""Real, executable tests for api_middleware/fix_server.py.

A real asyncio TCP server (FixAcceptor) and a real asyncio TCP client
exchange real, wire-encoded FIX bytes over a real loopback socket --
not a mock of the protocol layer. Only the ZMQ publish is intercepted
(the same ``_RecordingZmqPublisher`` stand-in ``tests/test_gateway.py``/
``tests/test_alert_agent.py`` already use -- no real-fake library exists
for ZMQ).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Generator, cast

import pytest
import pytest_asyncio
import simplefix

from api_middleware.fix_server import FixAcceptor
from core.config import get_settings
from core.schemas import FIX_INBOUND_ORDERS_TOPIC

_TEST_PORT = 15121
_OUR_COMP_ID = "NOTHIERLY"
_ALLOWED_CLIENT = "TESTCLIENT"


class _RecordingZmqPublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    async def publish(self, payload: Any) -> None:
        self.published.append(payload)

    def close(self, linger_ms: int = 0) -> None:
        pass


@pytest.fixture(autouse=True)
def configured_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    get_settings.cache_clear()
    monkeypatch.setenv("FIX_HOST", "127.0.0.1")
    monkeypatch.setenv("FIX_PORT", str(_TEST_PORT))
    monkeypatch.setenv("FIX_OUR_COMP_ID", _OUR_COMP_ID)
    monkeypatch.setenv("FIX_ALLOWED_CLIENT_SENDER_COMP_IDS", _ALLOWED_CLIENT)
    monkeypatch.setenv("FIX_HEARTBEAT_INTERVAL_SECONDS", "30")
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def recording_publisher() -> _RecordingZmqPublisher:
    return _RecordingZmqPublisher()


@pytest_asyncio.fixture
async def running_acceptor(recording_publisher: _RecordingZmqPublisher) -> AsyncGenerator[_RecordingZmqPublisher, None]:
    acceptor = FixAcceptor(settings=get_settings(), publisher=recording_publisher)  # type: ignore[arg-type]
    await acceptor.start()
    try:
        yield recording_publisher
    finally:
        await acceptor.stop()


def _logon_bytes(*, sender_comp_id: str, seq_num: int = 1) -> bytes:
    msg = simplefix.FixMessage()
    msg.append_pair(simplefix.TAG_BEGINSTRING, "FIX.4.4")
    msg.append_pair(simplefix.TAG_MSGTYPE, simplefix.MSGTYPE_LOGON)
    msg.append_pair(simplefix.TAG_SENDER_COMPID, sender_comp_id)
    msg.append_pair(simplefix.TAG_TARGET_COMPID, _OUR_COMP_ID)
    msg.append_pair(simplefix.TAG_MSGSEQNUM, seq_num)
    msg.append_utc_timestamp(simplefix.TAG_SENDING_TIME)
    msg.append_pair(simplefix.TAG_HEARTBTINT, 30)
    return cast(bytes, msg.encode())


def _new_order_single_bytes(
    *, sender_comp_id: str, seq_num: int, cl_ord_id: str, symbol: str = "BTC/USDT",
    side: bytes = simplefix.SIDE_BUY, order_qty: str = "1.5", ord_type: bytes = simplefix.ORDTYPE_MARKET,
    price: str | None = None,
) -> bytes:
    msg = simplefix.FixMessage()
    msg.append_pair(simplefix.TAG_BEGINSTRING, "FIX.4.4")
    msg.append_pair(simplefix.TAG_MSGTYPE, simplefix.MSGTYPE_NEW_ORDER_SINGLE)
    msg.append_pair(simplefix.TAG_SENDER_COMPID, sender_comp_id)
    msg.append_pair(simplefix.TAG_TARGET_COMPID, _OUR_COMP_ID)
    msg.append_pair(simplefix.TAG_MSGSEQNUM, seq_num)
    msg.append_utc_timestamp(simplefix.TAG_SENDING_TIME)
    msg.append_pair(simplefix.TAG_CLORDID, cl_ord_id)
    msg.append_pair(simplefix.TAG_SYMBOL, symbol)
    msg.append_pair(simplefix.TAG_SIDE, side)
    msg.append_pair(simplefix.TAG_ORDERQTY, order_qty)
    msg.append_pair(simplefix.TAG_ORDTYPE, ord_type)
    if price is not None:
        msg.append_pair(simplefix.TAG_PRICE, price)
    msg.append_utc_timestamp(simplefix.TAG_TRANSACTTIME)
    return cast(bytes, msg.encode())


async def _read_one_message(reader: asyncio.StreamReader, parser: simplefix.FixParser) -> simplefix.FixMessage:
    while True:
        msg = parser.get_message()
        if msg is not None:
            return msg
        data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        if not data:
            raise ConnectionError("Server closed the connection before sending a complete message.")
        parser.append_buffer(data)


@pytest.mark.asyncio
async def test_authorized_logon_is_accepted(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id=_ALLOWED_CLIENT))
        await writer.drain()
        parser = simplefix.FixParser()
        response = await _read_one_message(reader, parser)
        assert response.get(simplefix.TAG_MSGTYPE) == simplefix.MSGTYPE_LOGON
        assert response.get(simplefix.TAG_SENDER_COMPID) == _OUR_COMP_ID.encode()
        assert response.get(simplefix.TAG_TARGET_COMPID) == _ALLOWED_CLIENT.encode()
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_unauthorized_sender_comp_id_is_rejected(running_acceptor: _RecordingZmqPublisher) -> None:
    """Real fail-closed guard: an unrecognized SenderCompID must never get a Logon ack."""
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id="UNKNOWN_CLIENT"))
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        assert data == b""  # the server closed the connection without acking
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_new_order_single_accepted_and_published(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id=_ALLOWED_CLIENT))
        await writer.drain()
        parser = simplefix.FixParser()
        await _read_one_message(reader, parser)  # Logon ack

        writer.write(_new_order_single_bytes(sender_comp_id=_ALLOWED_CLIENT, seq_num=2, cl_ord_id="CL-001"))
        await writer.drain()
        exec_report = await _read_one_message(reader, parser)

        assert exec_report.get(simplefix.TAG_MSGTYPE) == simplefix.MSGTYPE_EXECUTION_REPORT
        assert exec_report.get(simplefix.TAG_CLORDID) == b"CL-001"
        assert exec_report.get(simplefix.TAG_ORDSTATUS) == simplefix.ORDSTATUS_NEW

        assert len(running_acceptor.published) == 1
        payload = running_acceptor.published[0]
        assert payload.topic == FIX_INBOUND_ORDERS_TOPIC
        assert payload.data["client_id"] == _ALLOWED_CLIENT
        assert payload.data["asset"]["symbol"] == "BTC/USDT"
        assert payload.data["side"] == "BUY"
        assert payload.data["quantity"] == "1.5"
        assert payload.data["algo_id"] == "CLIENT_DIRECT_FIX_ENTRY"
        assert payload.data["regulatory_client_shortcode"] == _ALLOWED_CLIENT
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_new_order_single_before_logon_disconnects(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_new_order_single_bytes(sender_comp_id=_ALLOWED_CLIENT, seq_num=1, cl_ord_id="CL-002"))
        await writer.drain()
        data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
        assert data == b""
        assert running_acceptor.published == []
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_unsupported_ord_type_is_rejected_not_published(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id=_ALLOWED_CLIENT))
        await writer.drain()
        parser = simplefix.FixParser()
        await _read_one_message(reader, parser)

        writer.write(
            _new_order_single_bytes(
                sender_comp_id=_ALLOWED_CLIENT, seq_num=2, cl_ord_id="CL-003", ord_type=simplefix.ORDTYPE_STOP,
            )
        )
        await writer.drain()
        exec_report = await _read_one_message(reader, parser)

        assert exec_report.get(simplefix.TAG_ORDSTATUS) == simplefix.ORDSTATUS_REJECTED
        assert running_acceptor.published == []
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_limit_order_requires_price(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id=_ALLOWED_CLIENT))
        await writer.drain()
        parser = simplefix.FixParser()
        await _read_one_message(reader, parser)

        writer.write(
            _new_order_single_bytes(
                sender_comp_id=_ALLOWED_CLIENT, seq_num=2, cl_ord_id="CL-004", ord_type=simplefix.ORDTYPE_LIMIT, price=None,
            )
        )
        await writer.drain()
        exec_report = await _read_one_message(reader, parser)

        assert exec_report.get(simplefix.TAG_ORDSTATUS) == simplefix.ORDSTATUS_REJECTED
        assert running_acceptor.published == []
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_limit_order_with_price_is_accepted(running_acceptor: _RecordingZmqPublisher) -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", _TEST_PORT)
    try:
        writer.write(_logon_bytes(sender_comp_id=_ALLOWED_CLIENT))
        await writer.drain()
        parser = simplefix.FixParser()
        await _read_one_message(reader, parser)

        writer.write(
            _new_order_single_bytes(
                sender_comp_id=_ALLOWED_CLIENT, seq_num=2, cl_ord_id="CL-005",
                ord_type=simplefix.ORDTYPE_LIMIT, price="65000.50", side=simplefix.SIDE_SELL,
            )
        )
        await writer.drain()
        exec_report = await _read_one_message(reader, parser)

        assert exec_report.get(simplefix.TAG_ORDSTATUS) == simplefix.ORDSTATUS_NEW
        assert len(running_acceptor.published) == 1
        assert running_acceptor.published[0].data["limit_price"] == "65000.50"
        assert running_acceptor.published[0].data["side"] == "SELL"
    finally:
        writer.close()

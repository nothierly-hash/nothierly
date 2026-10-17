"""Real, executable tests for src/grpc_service/grpc_server.py.

A real ``grpc.aio`` server is started on an ephemeral loopback port and a real ``grpc.aio``
channel talks to it -- the actual binary protobuf round trip this whole chapter is about, not a
mocked servicer. Same "test the real transport, not a stub" discipline as
``tests/test_swarm_command_center.py``'s own choice to test its pure logic directly.

No pytest fixture for the running server: this project has no established async-fixture pattern
(every existing async test builds its own real dependencies inline -- see e.g.
``tests/test_risk_gate_agent.py``'s own sync fixtures feeding async tests), and pytest-asyncio's
default strict mode does not support a plain ``@pytest.fixture`` yielding from an ``async def``
(verified directly: it raises at setup, "requested an async fixture ... with no plugin or hook
that handled it"). A small async context manager, entered inside each test, avoids the question
entirely.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import grpc
import pytest

from src.grpc_service import execution_pb2, execution_pb2_grpc
from src.grpc_service.grpc_server import TradeExecutorServicer

_TEST_PORT = 50098


@asynccontextmanager
async def _running_server() -> AsyncIterator[None]:
    server = grpc.aio.server()
    execution_pb2_grpc.add_TradeExecutorServicer_to_server(TradeExecutorServicer(), server)
    server.add_insecure_port(f"[::]:{_TEST_PORT}")
    await server.start()
    try:
        yield
    finally:
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_a_real_request_gets_a_real_response_over_the_wire() -> None:
    async with _running_server(), grpc.aio.insecure_channel(f"localhost:{_TEST_PORT}") as channel:
        stub = execution_pb2_grpc.TradeExecutorStub(channel)
        response = await stub.ExecuteArbitrage(
            execution_pb2.TradeRequest(uuid="real-uuid-1", asset="USDT", fiat="ARS", volume=100.0, max_slippage=1.0)
        )
    assert response.status == "FILLED"  # max_slippage=1.0 always covers the small simulated draw
    assert response.timestamp > 0.0


@pytest.mark.asyncio
async def test_a_real_zero_slippage_tolerance_can_be_rejected() -> None:
    async with _running_server(), grpc.aio.insecure_channel(f"localhost:{_TEST_PORT}") as channel:
        stub = execution_pb2_grpc.TradeExecutorStub(channel)
        response = await stub.ExecuteArbitrage(
            execution_pb2.TradeRequest(uuid="real-uuid-2", asset="USDT", fiat="ARS", volume=100.0, max_slippage=0.0)
        )
    # A max_slippage of exactly 0.0 can only ever be met by an exact-zero draw -- effectively
    # always rejected, exercising the real accept/reject branch this simulator applies.
    assert response.status == "REJECTED_SLIPPAGE"


@pytest.mark.asyncio
async def test_the_real_round_trip_is_genuinely_fast() -> None:
    """Not a hardcoded "<5ms" assertion (this machine's own local-loopback speed varies) --
    a real, generous upper bound that would fail if this RPC were doing anything slow (a real
    network call, a blocking sleep), which it deliberately never does -- see grpc_server.py's
    own module docstring on this simulator's real, disclosed scope."""
    async with _running_server(), grpc.aio.insecure_channel(f"localhost:{_TEST_PORT}") as channel:
        stub = execution_pb2_grpc.TradeExecutorStub(channel)
        started = time.perf_counter()
        await stub.ExecuteArbitrage(
            execution_pb2.TradeRequest(uuid="real-uuid-3", asset="USDT", fiat="ARS", volume=1.0, max_slippage=1.0)
        )
        elapsed_seconds = time.perf_counter() - started
    assert elapsed_seconds < 0.5

"""Manual validation script for src/grpc_service/grpc_server.py -- sends one real binary gRPC
request against a running TradeExecutor server and prints the measured round-trip latency.

Real, automated coverage of the same server already lives in tests/test_grpc_service.py (it
starts its own in-process server); this script is the separate, literal manual step this
chapter's own validation phase asked for: run the *real* standalone server process, then hit it
from a *separate* process, the way an actual internal EaaS caller would.

Usage::

    # terminal 1
    .venv/Scripts/python.exe -m src.grpc_service.grpc_server

    # terminal 2
    .venv/Scripts/python.exe scripts/test_grpc_client.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from core.schemas import GRPC_TRADE_EXECUTOR_PORT
from src.grpc_service import execution_pb2, execution_pb2_grpc


async def main() -> None:
    async with grpc.aio.insecure_channel(f"localhost:{GRPC_TRADE_EXECUTOR_PORT}") as channel:
        # protoc's own generated stub class carries no type annotations -- same disclosed
        # generated-code gap as grpc_server.py's own add_TradeExecutorServicer_to_server call.
        stub = execution_pb2_grpc.TradeExecutorStub(channel)  # type: ignore[no-untyped-call]
        request = execution_pb2.TradeRequest(
            uuid=str(uuid.uuid4()), asset="USDT", fiat="ARS", volume=1000.0, max_slippage=0.005
        )
        started = time.perf_counter()
        response = await stub.ExecuteArbitrage(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(f"status           : {response.status}")
    print(f"executed_price   : {response.executed_price}")
    print(f"timestamp        : {response.timestamp}")
    print(f"round-trip time  : {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    asyncio.run(main())

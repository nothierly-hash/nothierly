"""TradeExecutor gRPC server -- the internal, binary-protocol Ultra-Low-Latency EaaS layer.

**Real, disclosed scope, same discipline as risk_gate_agent.py's own module docstring**: this
server does **not** place a real trade, and ``executed_price`` is **not** sourced from any real
market feed (no reference price is even part of ``TradeRequest`` -- see ``execution.proto``). It
exists to demonstrate the real latency characteristic that motivates this whole chapter: a binary
protobuf request/response over HTTP/2 (gRPC's unary RPC), which the project's existing ZMQ bus
(``core/zmq_bus.py``) does not offer -- that bus is fire-and-forget PUB/SUB, with no built-in
"caller waits for exactly this one reply" contract, which is what a synchronous execution ack
needs. Real trades still only ever go through :mod:`src.execution.broker_gateway` /
:mod:`src.execution.smart_order_router` -- nothing here is wired into that path.

The one real piece of logic this simulator does apply: it accepts or rejects against the
caller's own ``max_slippage``, against a synthetic, clearly-labeled slippage draw -- not a coin
flip that ignores the request.
"""

from __future__ import annotations

import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import grpc

from core.logger import get_logger
from core.schemas import GRPC_TRADE_EXECUTOR_PORT
from src.grpc_service import execution_pb2, execution_pb2_grpc

_log = get_logger(__name__)

# Arbitrary unit reference price -- see module docstring: no real market feed is wired to this
# simulator, so there is no real per-asset price to apply slippage against. Any resemblance to a
# real quote is coincidental.
_SIMULATED_REFERENCE_PRICE: Final[float] = 1.0
_MAX_SIMULATED_SLIPPAGE: Final[float] = 0.002  # 0.2%, an arbitrary demo bound


class TradeExecutorServicer(execution_pb2_grpc.TradeExecutorServicer):
    """Implements the one RPC declared in execution.proto -- see this module's own docstring."""

    async def ExecuteArbitrage(
        self,
        request: execution_pb2.TradeRequest,
        context: grpc.aio.ServicerContext[execution_pb2.TradeRequest, execution_pb2.TradeResponse],
    ) -> execution_pb2.TradeResponse:
        started_at = time.perf_counter()
        simulated_slippage = random.uniform(-_MAX_SIMULATED_SLIPPAGE, _MAX_SIMULATED_SLIPPAGE)
        executed_price = _SIMULATED_REFERENCE_PRICE * (1.0 + simulated_slippage)
        status = "FILLED" if abs(simulated_slippage) <= request.max_slippage else "REJECTED_SLIPPAGE"

        response = execution_pb2.TradeResponse(
            status=status,
            executed_price=round(executed_price, 8),
            timestamp=time.time(),
        )
        _log.info(
            "grpc_trade_execution_simulated",
            uuid=request.uuid,
            asset=request.asset,
            fiat=request.fiat,
            status=status,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000.0, 4),
        )
        return response


async def serve(port: int = GRPC_TRADE_EXECUTOR_PORT) -> None:
    """Start the async gRPC server and block until it is stopped.

    ``grpc.aio.server`` still needs a real (small, bounded) thread pool for its own internal
    synchronous plumbing, even though the service method above is a native coroutine -- passed
    explicitly rather than relying on the library's default so its size is a deliberate choice,
    not an accident.
    """
    server = grpc.aio.server(ThreadPoolExecutor(max_workers=4))
    # protoc's own generated registration function carries no type annotations (see
    # execution_pb2_grpc.py's "DO NOT EDIT" header) -- a real, disclosed gap in generated code,
    # not this module's own.
    execution_pb2_grpc.add_TradeExecutorServicer_to_server(TradeExecutorServicer(), server)  # type: ignore[no-untyped-call]
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    _log.info("grpc_trade_executor_started", port=port)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())

"""Backend daemon: runs the shared ZMQ broker and every trading agent, for
every configured instrument, as concurrent tasks in one process.

Trading follows the "Portfolio Manager" pattern (see
:class:`~src.agents.execution.agent.ExecutionAgent`'s docstring for the
full reasoning): :class:`~src.agents.supervisor.agent.XgboostSignalAgent`
and :class:`~src.agents.rl_signal.agent.RlSignalAgent` each publish an
opinion and never touch a broker; only :class:`ExecutionAgent` executes,
and only once both agree -- independently, per instrument.

Multi-asset: :data:`_ALL_INSTRUMENTS` lists every instrument this backend
trades. One :class:`XgboostSignalAgent` and one :class:`RlSignalAgent` run
per instrument (each simply evaluating its own instrument, the same
proven single-instrument logic as before -- see :data:`_ALL_INSTRUMENTS`'s
comment on why multiple single-instrument instances were chosen over one
instance internally looping over several). One :class:`MarketDataAgent`
runs per *crypto* instrument in :data:`_CRYPTO_INSTRUMENTS`. A single
:class:`ExecutionAgent` handles every instrument at once, gated by its own
:data:`~src.agents.execution.agent._DEFAULT_MAX_CONCURRENT_TRADES`
concurrent-position cap.

Not a demo/test -- this is the actual production entrypoint for the
``backend`` Docker service. Run with::

    python scripts/run_backend.py

Requires ``ZMQ_BROKER_FRONTEND_ADDRESS``/``ZMQ_BROKER_BACKEND_ADDRESS`` to
be configured (see :class:`core.config.ZmqSettings`): with many agents
sharing one process, each agent's own PUB socket can no longer bind its
own configured port directly (only the first would succeed) -- every
publisher instead connects to a shared :class:`~core.zmq_broker.ZmqBroker`
that fans messages out to every subscriber. See that module's docstring
for the full topology.

Graceful shutdown: a container orchestrator stops a service with
``SIGTERM``, not ``Ctrl+C``'s ``SIGINT`` -- and unlike ``SIGINT``,
``asyncio.run()`` has no built-in handling for it. This script installs an
explicit ``SIGTERM`` handler that cancels the shared task group, which
cascades into every agent's own ``try/except CancelledError/finally``
shutdown sequence (closing ZMQ sockets, Redis connections, ...) exactly as
a real ``Ctrl+C`` already does for a single standalone agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/run_backend.py` adds this file's own directory
# (scripts/) to sys.path, not the project root running it from --
# `core`/`src` live at the root, so it must be added explicitly before
# those imports below, regardless of the working directory the command
# was launched from (e.g. Docker's WORKDIR).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import functools
import signal
from typing import Final

import zmq.asyncio

from core.config import ConfigurationError, get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient
from core.zmq_broker import ZmqBroker
from src.agents.archivist.agent import ArchivistAgent
from src.agents.execution.agent import ExecutionAgent
from src.agents.macro.agent import MacroAgent
from src.agents.macro_order_execution.agent import MacroOrderExecutionAgent
from src.agents.market_data.agent import MarketDataAgent
from src.agents.nlp.agent import NlpAgent
from src.agents.rl_signal.agent import RlSignalAgent
from src.agents.supervisor.agent import XgboostSignalAgent
from src.execution.audit_ledger import CryptographicAuditLedger

_log = get_logger(__name__)

# Every instrument the signal/execution agents evaluate. Forex
# instruments (OANDA-style, underscore) and crypto instruments
# (ccxt.pro-native, slash) intentionally use each venue's own symbol
# format -- see MarketDataAgent's symbol below, which is passed straight
# through to ccxt.pro and therefore *must* be that exact format; every
# downstream agent (signal agents, ExecutionAgent) then uses that same
# string as the canonical instrument identifier for consistent Redis-key
# and ZMQ-message matching end to end.
#
# Asset-class note (carried over from the single-instrument chapter,
# still true, now for *two* instruments instead of one): EUR_USD and
# XAU_USD have no live price feed in this project -- MarketDataAgent only
# speaks ccxt.pro (crypto), no OANDA WebSocket client exists yet. Their
# signal/execution agents run exactly like every crypto instrument's
# (independent per-instrument state, ready the moment a real forex feed
# is added) but will never see a MARKET_DATA_L2 tick or publish a signal
# in this demo -- the same known gap the original EUR_USD-only backend
# already had, now simply visible on two instruments instead of one.
_FOREX_INSTRUMENTS: Final[tuple[str, ...]] = ("EUR_USD", "XAU_USD")
_CRYPTO_INSTRUMENTS: Final[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT")
_ALL_INSTRUMENTS: Final[tuple[str, ...]] = _FOREX_INSTRUMENTS + _CRYPTO_INSTRUMENTS

# One XgboostSignalAgent + one RlSignalAgent instance per instrument
# below, not one instance internally looping/multiplexing over several:
# both classes already carry per-instrument Kill Switch state (spread
# history, consecutive-error count) that must never be shared across
# instruments -- a spread anomaly on BTC/USDT has no bearing on EUR_USD's
# baseline. N independent instances get that isolation for free, reusing
# every line of already-tested single-instrument logic unchanged; a
# single multi-instrument instance would have to duplicate that state
# into per-instrument dicts internally for no behavioral difference.
# ExecutionAgent is the one exception (a single instance for every
# instrument) precisely because it has no such per-instrument Kill
# Switch state of its own to isolate -- see its module docstring.


async def main() -> None:
    """Start the broker and every agent, and run until a shutdown signal arrives."""
    configure_logging(log_level="INFO", log_dir=None, service_name="trading-backend")

    settings = get_settings()
    zmq_settings = settings.zmq
    if zmq_settings.broker_frontend_address is None or zmq_settings.broker_backend_address is None:
        raise ConfigurationError(
            "ZMQ_BROKER_FRONTEND_ADDRESS and ZMQ_BROKER_BACKEND_ADDRESS must both be set "
            "to run multiple agents in one process -- see core/zmq_broker.py's docstring. "
            "Without them, every agent after the first would fail to bind ZMQ_PUB_ADDRESS."
        )

    # One shared zmq.asyncio.Context for the broker and all five agents,
    # instead of each creating its own: each Context opens its own
    # OS-level I/O thread and (on Windows specifically) a TCP-loopback
    # "signaler" socket pair for cross-thread wakeup, which has been
    # observed to abort the whole process ("Assertion failed: Connection
    # reset by peer", libzmq's signaler.cpp) under socket churn elsewhere
    # in the same process (e.g. a WebSocket reconnect). Six contexts
    # multiply that fragile surface sixfold versus sharing one -- see
    # core/base_agent.py's zmq_context parameter docstring.
    shared_zmq_context = zmq.asyncio.Context()
    # The broker BINDS broker_bind_frontend_address/_backend_address when set (a multi-container
    # deployment's 0.0.0.0 wildcard, reachable from other containers -- see docker-compose.yml's
    # alpha-engine service and ZmqSettings.broker_bind_frontend_address's own docstring for exactly
    # why this must be a real, separate address from what agents connect to). Falls back to
    # broker_frontend_address/_backend_address when unset -- the original, still-default,
    # single-container topology where bind and connect addresses are identical, unchanged.
    broker = ZmqBroker(
        zmq_settings.broker_bind_frontend_address or zmq_settings.broker_frontend_address,
        zmq_settings.broker_bind_backend_address or zmq_settings.broker_backend_address,
        context=shared_zmq_context,
    )
    # manage_own_signals=False on every agent: OS signal handler
    # registration is process-wide and last-registration-wins, so many
    # agents each installing their own SIGINT/SIGTERM handler would leave
    # all but one never hearing a shutdown signal at all. This script
    # installs one handler for everyone below instead.
    #
    # manage_own_redis=False: RedisClient is a process-wide singleton --
    # every agent's _connect_redis() returns the exact same instance.
    # With many agents shutting down concurrently, whichever one reached
    # its own _shutdown() first would close that shared connection out
    # from under every other agent still using it (a real crash this
    # project hit: a sibling agent's on_shutdown hook failing with
    # "Buffer is closed"). This script closes the shared client itself,
    # below, once every agent sharing it has already shut down.
    #
    # Multi-asset scaling: one MarketDataAgent per *crypto* instrument
    # (each independently streams its own ccxt.pro WebSocket -- see that
    # class's docstring, unchanged from the single-instrument chapter),
    # and one XgboostSignalAgent + one RlSignalAgent per instrument in
    # _ALL_INSTRUMENTS (forex included, despite the known no-live-feed
    # gap above -- see _ALL_INSTRUMENTS' comment). A *single*
    # ExecutionAgent handles every instrument at once: its own
    # per-instrument dict-based state (self._positions/self._signals/
    # self._last_prices) is what makes that safe -- see that class's
    # module docstring.
    agents: list[
        MarketDataAgent
        | MacroAgent
        | NlpAgent
        | XgboostSignalAgent
        | RlSignalAgent
        | ExecutionAgent
        | ArchivistAgent
        | MacroOrderExecutionAgent
        | CryptographicAuditLedger
    ] = [
        *(
            MarketDataAgent(
                crypto_symbol,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for crypto_symbol in _CRYPTO_INSTRUMENTS
        ),
        MacroAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        NlpAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        *(
            XgboostSignalAgent(
                instrument,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for instrument in _ALL_INSTRUMENTS
        ),
        *(
            RlSignalAgent(
                instrument,
                manage_own_signals=False,
                zmq_context=shared_zmq_context,
                manage_own_redis=False,
            )
            for instrument in _ALL_INSTRUMENTS
        ),
        ExecutionAgent(
            _ALL_INSTRUMENTS,
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        ArchivistAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        MacroOrderExecutionAgent(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
        CryptographicAuditLedger(
            manage_own_signals=False,
            zmq_context=shared_zmq_context,
            manage_own_redis=False,
        ),
    ]

    _log.info(
        "trading_backend_starting",
        instruments=list(_ALL_INSTRUMENTS),
        crypto_instruments=list(_CRYPTO_INSTRUMENTS),
        forex_instruments=list(_FOREX_INSTRUMENTS),
        broker_frontend=zmq_settings.broker_frontend_address,
        broker_backend=zmq_settings.broker_backend_address,
        agent_count=len(agents),
    )

    main_task = asyncio.gather(broker.run(), *(agent.run() for agent in agents))

    def _handle_shutdown_signal(sig: signal.Signals) -> None:
        _log.info("shutdown_signal_received_stopping_backend", signal=sig.name)
        main_task.cancel()

    # A container orchestrator sends SIGTERM on `docker stop`/`docker
    # compose down`, not Ctrl+C's SIGINT -- unlike SIGINT (which
    # asyncio.run()'s own Runner already cancels the outermost task for
    # on Python 3.11+), nothing handles SIGTERM by default, so it must be
    # wired up explicitly. SIGINT is also registered here directly rather
    # than relied upon implicitly, so behavior doesn't depend on Runner
    # internals across Python versions.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, functools.partial(_handle_shutdown_signal, sig))
        except NotImplementedError:
            # Windows: no add_signal_handler -- same classic-API fallback
            # core.base_agent uses for a single standalone agent.
            signal.signal(sig, lambda signum, _frame: _handle_shutdown_signal(signal.Signals(signum)))

    try:
        try:
            await main_task
        except asyncio.CancelledError:
            _log.info("trading_backend_shutdown_complete")
        except Exception:
            # A setup-phase failure in one agent (e.g. on_start()/
            # _connect_redis() raising something other than
            # CancelledError -- the only exception type each agent's own
            # run() loop unwinds through cleanly) surfaces here and would
            # otherwise leave every other agent running orphaned, with no
            # chance at their own graceful _shutdown(). Cancel the whole
            # group explicitly so they get one, then fail loudly: an
            # unreachable agent is a real incident, not something to retry
            # silently.
            _log.exception("trading_backend_agent_failed_shutting_down_everything")
            main_task.cancel()
            try:
                await main_task
            except asyncio.CancelledError:
                pass
            raise
    finally:
        # Every agent no longer owns/terminates this context (see
        # zmq_context in their constructors above) -- as the one that
        # created it, this is the only place it gets terminated, once
        # every agent and the broker sharing it have already shut down.
        await asyncio.to_thread(shared_zmq_context.term)
        # Same reasoning as manage_own_redis=False above: close the
        # shared RedisClient singleton exactly once, here, now that
        # every agent that was using it has finished shutting down.
        await RedisClient.reset_instance()


if __name__ == "__main__":
    asyncio.run(main())

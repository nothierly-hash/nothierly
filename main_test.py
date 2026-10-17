"""Manual/interactive proof that BaseAgent's lifecycle works end-to-end.

Runs a single toy agent that, every second, publishes a Pydantic-validated
MARKET_DATA message on its own ZMQ PUB socket and immediately receives it
back on its own SUB socket (both are pointed at the same demo endpoint via
the env vars below, so this proves a genuine PUB->SUB round trip, not just
that ``publish()`` didn't raise).

Usage::

    python main_test.py

Then press Ctrl+C: BaseAgent should log the shutdown signal, tear down its
ZMQ sockets/context and Redis connection, and exit cleanly with no
event-loop warnings or tracebacks. No real Redis server is required — if
one isn't running locally, the agent logs that it's continuing without it
after a few quick bounded retries, exactly as designed.
"""

from __future__ import annotations

import asyncio
import os

# Demo-only configuration, set before anything reads settings from the
# environment. BROKER_* has no default (see core/config.py) so a value is
# required even though this demo never calls a real broker API. Both ZMQ
# addresses point at the same endpoint so the demo agent can publish to,
# and subscribe from, itself.
os.environ.setdefault("BROKER_API_KEY", "demo_key_for_local_testing_only")
os.environ.setdefault("BROKER_API_SECRET", "demo_secret_for_local_testing_only")
os.environ.setdefault("ZMQ_PUB_ADDRESS", "tcp://127.0.0.1:15570")
os.environ.setdefault("ZMQ_SUB_ADDRESS", "tcp://127.0.0.1:15570")

from core.base_agent import BaseAgent  # noqa: E402  (must follow env setup above)
from core.logger import configure_logging  # noqa: E402


class FakeMarketDataAgent(BaseAgent):
    """Toy agent used only to exercise BaseAgent's lifecycle."""

    def __init__(self) -> None:
        super().__init__(
            "demo-agent-1",
            tick_interval_seconds=1.0,
            subscribed_topics=("MARKET_DATA",),
        )
        self._tick_count = 0

    async def on_start(self) -> None:
        self._log.info("fake_agent_ready")

    async def on_tick(self) -> None:
        self._tick_count += 1
        price = round(1.0500 + self._tick_count * 0.0001, 5)
        await self.publish("MARKET_DATA", {"tick": self._tick_count, "price": price})

        try:
            received = await asyncio.wait_for(self._subscriber.receive(), timeout=1.0)
        except asyncio.TimeoutError:
            self._log.warning("fake_agent_did_not_receive_own_message_in_time")
            return

        assert received.data["tick"] == self._tick_count, "pub/sub round-trip mismatch"
        self._log.info(
            "fake_agent_round_trip_ok",
            tick=received.data["tick"],
            price=received.data["price"],
        )

    async def on_shutdown(self) -> None:
        self._log.info("fake_agent_custom_cleanup_done", total_ticks=self._tick_count)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None)
    agent = FakeMarketDataAgent()
    print("Agent démarré. Appuyez sur Ctrl+C pour déclencher l'arrêt propre (graceful shutdown)...")
    await agent.run()
    print("Agent arrêté proprement, aucune erreur de l'event loop.")


if __name__ == "__main__":
    asyncio.run(main())

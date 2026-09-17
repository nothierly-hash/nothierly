"""Standalone launcher for a single, real :class:`~src.agents.whale_forensics.agent.WhaleForensicsDaemon`.

Ticks as a real, logged no-op unless ``WEB3_HTTP_PROVIDER_URL`` and
``WEB3_WATCHED_TOKEN_CONTRACTS`` are both configured in ``.env`` -- see that
class's own module docstring for the full, disclosed scope (polling, not a
persistent subscription; no hardcoded institutional address list).

Usage::

    python scripts/run_whale_forensics_daemon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as scripts/run_market_data_agent.py: `core`/`src` live at the
# project root, which must be on sys.path before those imports below,
# regardless of the working directory this was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from core.logger import configure_logging, get_logger
from src.agents.whale_forensics.agent import WhaleForensicsDaemon

_log = get_logger(__name__)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="whale-forensics-daemon")
    agent = WhaleForensicsDaemon()
    _log.info("whale_forensics_daemon_launcher_starting")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())

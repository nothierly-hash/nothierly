"""Explicit, one-time, human-run step: set ``SYSTEM_STATUS=SAFE`` before a Chaos Shadow Run.

**Deliberately not called by ``scripts/run_chaos_shadow_run.py`` or any
other automated script here.** :func:`core.security.reset_kill_switch`'s
own docstring is unambiguous: "No code path in this system calls this
automatically -- recovering from a global panic is a decision for a
human to make on purpose, never something an agent decides for itself."
This file exists so that decision is still a single, explicit, disclosed
command -- not so a chaos launcher can quietly make it for you.

**Why you need this at all**: :func:`core.security.is_system_safe`
"fails closed on any doubt" -- a brand-new Redis (e.g. this project's own
``scripts/run_fake_redis.py``, fresh on every Chaos Shadow Run) has no
``SYSTEM_STATUS`` key at all yet, which reads as unsafe by design.
Without running this once, first, every real consensus-cleared entry
:class:`~src.agents.execution.agent.ExecutionAgent` evaluates will be
blocked by the *global* Kill Switch before ever reaching Gates 4-7
(Margin/RiskManager/PortfolioRiskParityManager/StochasticCore) -- a real,
correct, but different rejection path than the one Directive 3 is meant
to stress. Run this after ``scripts/run_fake_redis.py`` is up and before
``scripts/run_execution_agent.py``/``scripts/chaos_injector.py`` start.

Usage::

    python scripts/run_fake_redis.py --port 16399   # terminal 1
    python scripts/chaos_reset_kill_switch.py        # terminal 2, once
    python scripts/run_chaos_shadow_run.py ...        # then the real chaos fleet
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient
from core.security import reset_kill_switch

_log = get_logger(__name__)


async def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="chaos-reset-kill-switch")
    settings = get_settings()
    await RedisClient.get_instance(settings.redis)

    print("=" * 70)
    print("This sets SYSTEM_STATUS=SAFE on the Redis at "
          f"{settings.redis.host}:{settings.redis.port} -- a real, global,\n"
          "signed flag every agent's Kill Switch check reads. Only run this against\n"
          "throwaway/test infrastructure (fake-redis, a local dev instance), never a\n"
          "real production Redis, and only because you deliberately intend a chaos/test\n"
          "run to reach real order-placement gates instead of being blocked upstream.")
    print("=" * 70)

    await reset_kill_switch()
    _log.warning("chaos_kill_switch_reset_confirmed", host=settings.redis.host, port=settings.redis.port)
    print("Done. SYSTEM_STATUS=SAFE.")

    await RedisClient.reset_instance()


if __name__ == "__main__":
    asyncio.run(main())

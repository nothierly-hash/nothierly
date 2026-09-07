"""Functional test: ExecutionAgent's multi-asset scaling (4 concurrent instruments).

Verifies the per-instrument state-dict design (``_signals``/``_positions``/
``_last_prices`` all keyed by instrument -- see the module docstring of
``src/agents/execution/agent.py``) actually behaves independently per
instrument, and that ``MAX_CONCURRENT_TRADES`` is enforced across them:

1. Consensus reached on one instrument opens only that instrument; every
   other instrument's state is untouched.
2. A second instrument opens normally (2 of a cap of 2); a third is refused
   once the cap is reached, without disturbing the first two.
3. Asymmetric exit: a single dissenting signal closes an open position
   immediately, with no effect on any other open instrument.
4. An instrument outside this agent's configured set is never tracked.

Talks to real code paths only -- no mocked ``ExecutionAgent`` internals.
The trading client is never actually called (``MOCK_MODE=true``), and
signals/price ticks are delivered by calling the agent's own message
handlers directly, exactly as the real ZMQ bus would dispatch them, without
needing a running ZMQ broker (this agent's ZMQ subscription only starts
inside ``run()``, which this test never calls).

Redis is optional but recommended, matching this project's real
infrastructure rather than an in-process fake -- start one first:

    python scripts/run_fake_redis.py

Then, in a second terminal (PowerShell shown; adjust for bash):

    $env:REDIS_HOST = "127.0.0.1"
    $env:REDIS_PORT = "6379"
    python tests/test_execution_agent_multi_asset.py

Without a reachable Redis, the same checks still run (ExecutionAgent
degrades to in-memory-only per its established fail-open contract) --
only the decision-log/equity-curve/risk-pct persistence is skipped.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Dummy credentials so AppSettings/OandaClient can construct -- never used:
# MOCK_MODE=true means _place_order/_close_position never call OANDA, and
# on_start()'s self._trading_client.connect() is documented as a no-op even
# under MOCK_MODE. setdefault() so a real .env value (if any) still wins.
os.environ.setdefault("BROKER_API_KEY", "demo_key_for_local_testing_only")
os.environ.setdefault("BROKER_API_SECRET", "demo_secret_for_local_testing_only")
os.environ.setdefault("OANDA_API_TOKEN", "demo_token_for_local_testing_only")
os.environ.setdefault("OANDA_ACCOUNT_ID", "demo-account-for-local-testing-only")
# Forced, not setdefault(): this test must never place a real order even if
# the user's own .env has MOCK_MODE=false set for an unrelated live session.
os.environ["MOCK_MODE"] = "true"

from core.schemas import MarketMetrics, MessagePayload, TradingSignal, TRADING_SIGNALS_TOPIC  # noqa: E402
from src.agents.execution.agent import ExecutionAgent  # noqa: E402

# Matches ExecutionAgent's own private _MARKET_DATA_TOPIC -- duplicated as a
# literal rather than importing a leading-underscore name from that module.
_MARKET_DATA_TOPIC = "MARKET_DATA_L2"

INSTRUMENTS = frozenset({"EUR_USD", "XAU_USD", "BTC/USDT", "ETH/USDT"})
_PRICES = {"EUR_USD": 1.10000, "XAU_USD": 2650.30000, "BTC/USDT": 78120.50000, "ETH/USDT": 2460.80000}

_checks: list[tuple[str, bool]] = []


def check(label: str, condition: bool) -> None:
    _checks.append((label, condition))
    print(f"{'PASS' if condition else 'FAIL'} - {label}")


async def send_price(agent: ExecutionAgent, instrument: str, price: float) -> None:
    await agent._handle_price_message(
        MessagePayload(
            topic=_MARKET_DATA_TOPIC,
            sender_id="test-harness",
            data=MarketMetrics(
                symbol=instrument, spread=0.0001, order_book_imbalance=0.0, mid_price=price
            ).model_dump(mode="json"),
        )
    )


async def send_signal(
    agent: ExecutionAgent, agent_id: str, instrument: str, signal: int, confidence: float = 0.8
) -> None:
    await agent._handle_signal_message(
        MessagePayload(
            topic=TRADING_SIGNALS_TOPIC,
            sender_id=agent_id,
            data=TradingSignal(
                timestamp=time.time(),
                agent_id=agent_id,
                instrument=instrument,
                signal=signal,  # type: ignore[arg-type]
                confidence=confidence,
            ).model_dump(mode="json"),
        )
    )


async def open_via_consensus(agent: ExecutionAgent, instrument: str) -> None:
    """Feed a price tick, then both required agents agreeing BUY -- should reach consensus."""
    await send_price(agent, instrument, _PRICES[instrument])
    await send_signal(agent, "supervisor-agent", instrument, 1)
    await send_signal(agent, "rl-signal-agent", instrument, 1)


async def main() -> None:
    redis_host = os.environ.get("REDIS_HOST", "127.0.0.1")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    print(f"Attempting Redis at {redis_host}:{redis_port} ...")

    agent = ExecutionAgent(INSTRUMENTS, max_concurrent_trades=2, capital=10_000.0)
    await agent._connect_redis()
    if agent._redis_client is not None:
        print("Connected -- decision log / equity curve will persist to this Redis.")
        # Fixed, small risk-per-trade so two concurrent positions never
        # brush the (unrelated) portfolio exposure ceiling -- this test
        # isolates MAX_CONCURRENT_TRADES, not RiskManager.check_exposure_limit.
        await agent._redis_client.execute("set", "TRADING_RISK_PCT", "0.002")
    else:
        print("No Redis reachable -- continuing in-memory-only (see scripts/run_fake_redis.py).")
    await agent.on_start()

    print("\n--- Test 1: per-instrument independence ---")
    await open_via_consensus(agent, "EUR_USD")
    check("EUR_USD opened LONG on consensus", agent._positions["EUR_USD"].status == "LONG")
    for other in ("XAU_USD", "BTC/USDT", "ETH/USDT"):
        check(f"{other} untouched by EUR_USD's consensus", agent._position_for(other).status == "FLAT")
    check("EUR_USD signal memory cleared after consensus", "EUR_USD" not in agent._signals)

    print("\n--- Test 2: MAX_CONCURRENT_TRADES cap (set to 2 for this test) ---")
    await open_via_consensus(agent, "XAU_USD")
    check("XAU_USD opened LONG (2nd of 2 allowed)", agent._positions["XAU_USD"].status == "LONG")
    check("open_position_count == 2 before the capped attempt", agent._open_position_count() == 2)
    await open_via_consensus(agent, "BTC/USDT")
    btc_position = agent._positions.get("BTC/USDT")
    check("BTC/USDT entry refused by the cap", btc_position is None or btc_position.status == "FLAT")
    check("open_position_count still == 2 after the refused 3rd", agent._open_position_count() == 2)

    print("\n--- Test 3: asymmetric exit (no opposing consensus needed), other positions untouched ---")
    await send_signal(agent, "rl-signal-agent", "EUR_USD", -1)
    check("EUR_USD closed on a single dissenting signal", agent._positions["EUR_USD"].status == "FLAT")
    check("XAU_USD still LONG (unaffected by EUR_USD's exit)", agent._positions["XAU_USD"].status == "LONG")
    check("open_position_count == 1 after EUR_USD's exit", agent._open_position_count() == 1)

    print("\n--- Test 4: instrument outside this agent's configured set is ignored entirely ---")
    await send_signal(agent, "supervisor-agent", "GBP_USD", 1)
    await send_signal(agent, "rl-signal-agent", "GBP_USD", 1)
    check("GBP_USD never tracked (not in INSTRUMENTS)", "GBP_USD" not in agent._positions)

    if agent._redis_client is not None:
        await agent._redis_client.close()

    print("\n" + "=" * 60)
    failed = [label for label, ok in _checks if not ok]
    if failed:
        print(f"{len(failed)}/{len(_checks)} checks FAILED:")
        for label in failed:
            print(f"  - {label}")
        sys.exit(1)
    print(f"All {len(_checks)} checks passed.")


if __name__ == "__main__":
    asyncio.run(main())

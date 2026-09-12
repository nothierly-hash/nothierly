"""Genesis Protocol: translate a real tda_engine.py cycle into a real, human-reviewed saga proposal.

**Real, disclosed scope -- what this module does NOT do**: it does not execute a single real order,
does not call any broker/exchange/P2P API, and does not auto-trigger anything. This project has no
real execution capability for P2P/fiat-rail arbitrage (see ``risk_gate_agent.py``'s own module
docstring for why, and ``p2p_api.py``'s module docstring for the real scope of what its data can
and cannot back) -- exactly the same disclosed limit this module inherits. The output is a real,
structured, human-readable proposal, persisted to Redis and best-effort broadcast over the real
ZMQ bus, for a human operator to review -- the same "actionable alert, not a fabricated auto-trade"
discipline already established by ``risk_gate_agent.py``'s own ``SWARM_ARBITRAGE_APPROVED_TOPIC``.

**Real, disclosed scope -- what the underlying cycle is**: :mod:`nothierly_genesis.tda_engine`'s
own market is a fixed, planted demonstration (see that module's own docstring) -- not live data.
This module's own real contribution is the *translation* step (cycle -> bottleneck liquidity ->
structured saga breakdown -> persisted/published proposal), which would apply unchanged the day
``tda_engine``'s market model is upgraded to consume live P2P data; it does not itself make the
underlying cycle live.

**Real bottleneck liquidity, where real data actually exists, honestly absent where it doesn't**:
this project's only real per-leg liquidity source is ``p2p_api.py``'s own P2P premium scan (the
real Binance P2P average liquidity per fiat, persisted at
:data:`core.schemas.P2P_LATEST_SNAPSHOT_REDIS_KEY`). A leg is attributed that real figure if
*either* endpoint is one of the two real, tracked fiats (``NGN_P2P``/``ARS_P2P`` -> ``NGN``/
``ARS``) -- a real, disclosed **approximation** for a leg that is not literally a USDT<->fiat
conversion (e.g. ``NGN_P2P->BTC`` is really an NGN<->BTC P2P trade, for which no direct liquidity
feed exists anywhere in this codebase; its own USDT<->NGN liquidity is used as the best real proxy
available, not a fabricated number for the leg itself). A leg touching *neither* real fiat (this
demo cycle has none, since every leg touches ``NGN_P2P`` or ``ARS_P2P``) is reported as a real,
explicit gap (``legs_missing_real_liquidity_data``) instead.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, Final

import zmq.asyncio

from core.config import get_settings
from core.logger import get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import (
    P2P_LATEST_SNAPSHOT_REDIS_KEY,
    SWARM_TDA_SAGA_MAX_ENTRIES,
    SWARM_TDA_SAGA_PROPOSED_TOPIC,
    SWARM_TDA_SAGA_REDIS_KEY,
    MessagePayload,
)
from core.zmq_bus import ZmqPublisher
from nothierly_genesis.tda_engine import build_market_graph, find_real_arbitrage_cycle

_log = get_logger(__name__)

# Real, disclosed mapping from this demo's node names to the real fiat codes p2p_api.py's own
# scan actually tracks (core.py's _TARGET_FIAT_CURRENCIES) -- see module docstring.
_TDA_NODE_TO_REAL_FIAT: Final[dict[str, str]] = {"NGN_P2P": "NGN", "ARS_P2P": "ARS"}


async def _fetch_real_p2p_snapshot(redis_client: RedisClient) -> dict[str, Any] | None:
    """Real, current P2P premium snapshot (p2p_api.py's own scan output), or ``None`` if none exists yet."""
    try:
        raw = await redis_client.execute("get", P2P_LATEST_SNAPSHOT_REDIS_KEY)
    except RedisUnavailableError:
        _log.warning("saga_snapshot_redis_unavailable")
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    result: dict[str, Any] = json.loads(raw)
    return result


def _real_liquidity_for_fiat(fiat: str, snapshot: dict[str, Any] | None) -> float | None:
    if snapshot is None:
        return None
    for candidate in snapshot.get("top_candidates", []):
        if candidate.get("country_fiat") == fiat:
            liquidity = candidate.get("avg_liquidity_usdt")
            return float(liquidity) if liquidity is not None else None
    return None


def build_saga_proposal(cycle: list[str], profit_multiplier: float) -> dict[str, Any]:
    """The real, structured (but not yet liquidity-checked) proposal skeleton for one real cycle.

    ``cycle`` is a closed path (first node repeated last, e.g. from
    :func:`nothierly_genesis.tda_engine.find_real_arbitrage_cycle`).
    """
    legs = [f"{cycle[i]}->{cycle[i + 1]}" for i in range(len(cycle) - 1)]
    saga_steps = [
        {"step": i + 1, "leg": leg, "from": cycle[i], "to": cycle[i + 1]} for i, leg in enumerate(legs)
    ]
    return {
        "topology_id": f"H1_cycle_{'_'.join(cycle[:-1])}",
        "cycle": cycle,
        "expected_net_profit_pct": round((profit_multiplier - 1.0) * 100.0, 3),
        "saga_steps": saga_steps,
        "status": "PROPOSED_FOR_HUMAN_REVIEW",
        "disclaimer": (
            "Aucune exécution automatique. Cycle issu du moteur de démonstration "
            "nothierly_genesis/tda_engine.py (non calibré sur données live -- voir "
            "nothierly_genesis/TECH_ATTRIBUTION_REPORT.md). Ce projet n'a aucune capacité réelle "
            "d'exécution broker/P2P -- voir risk_gate_agent.py."
        ),
        "generated_at": time.time(),
    }


async def attach_real_bottleneck_liquidity(proposal: dict[str, Any], redis_client: RedisClient) -> dict[str, Any]:
    """Mutates and returns ``proposal`` with real per-leg liquidity, where real data exists.

    See module docstring: only the two real P2P legs have a real liquidity source in this
    codebase; every other leg is reported as missing, never guessed.
    """
    snapshot = await _fetch_real_p2p_snapshot(redis_client)
    cycle = proposal["cycle"]
    leg_liquidity: dict[str, float | None] = {}
    for i in range(len(cycle) - 1):
        leg = f"{cycle[i]}->{cycle[i + 1]}"
        real_fiat = _TDA_NODE_TO_REAL_FIAT.get(cycle[i]) or _TDA_NODE_TO_REAL_FIAT.get(cycle[i + 1])
        leg_liquidity[leg] = _real_liquidity_for_fiat(real_fiat, snapshot) if real_fiat else None

    known_values = [v for v in leg_liquidity.values() if v is not None]
    proposal["leg_liquidity_usd"] = leg_liquidity
    proposal["legs_missing_real_liquidity_data"] = [leg for leg, v in leg_liquidity.items() if v is None]
    proposal["bottleneck_liquidity_usd"] = min(known_values) if known_values else None
    proposal["execution_volume_usd"] = proposal["bottleneck_liquidity_usd"]  # honest: capped by the weakest known leg
    return proposal


async def publish_saga_proposal(proposal: dict[str, Any], redis_client: RedisClient, publisher: ZmqPublisher) -> None:
    """Real, best-effort persistence (durable) + broadcast (transient) -- same discipline as
    p2p_api.py's own ``_publish_scan_update`` on why the Redis write is the reliable channel and
    the ZMQ publish is a convenience, not the other way around."""
    try:
        await redis_client.execute("lpush", SWARM_TDA_SAGA_REDIS_KEY, json.dumps(proposal))
        await redis_client.execute("ltrim", SWARM_TDA_SAGA_REDIS_KEY, 0, SWARM_TDA_SAGA_MAX_ENTRIES - 1)
    except RedisUnavailableError:
        _log.warning("saga_proposal_persist_failed", topology_id=proposal["topology_id"])

    payload = MessagePayload(
        topic=SWARM_TDA_SAGA_PROPOSED_TOPIC, sender_id="genesis-saga-proposal-generator", data=proposal
    )
    await publisher.publish(payload)
    _log.warning(
        "saga_proposal_published", topology_id=proposal["topology_id"],
        expected_net_profit_pct=proposal["expected_net_profit_pct"],
        bottleneck_liquidity_usd=proposal["bottleneck_liquidity_usd"],
    )


async def main() -> dict[str, Any] | None:
    graph = build_market_graph()
    result = find_real_arbitrage_cycle(graph)
    if result is None:
        print("Aucun cycle réel détecté par Bellman-Ford -- aucune proposition générée.")
        return None
    cycle, profit_multiplier = result

    proposal = build_saga_proposal(cycle, profit_multiplier)

    redis_client = await RedisClient.get_instance(get_settings().redis, max_attempts=3)
    await attach_real_bottleneck_liquidity(proposal, redis_client)

    zmq_settings = get_settings().zmq
    context = zmq.asyncio.Context()
    broker_frontend = zmq_settings.broker_frontend_address
    publisher = ZmqPublisher(
        broker_frontend or zmq_settings.pub_address,
        high_water_mark=zmq_settings.high_water_mark,
        context=context,
        bind=broker_frontend is None,
    )
    if broker_frontend is None:
        # Real "slow joiner" mitigation for a one-shot bind+publish -- a fresh PUB socket's first
        # message is routinely lost if no SUB has connected yet; the Redis write above is this
        # proposal's real, durable delivery path regardless, so this is only for a live listener.
        await asyncio.sleep(0.2)

    await publish_saga_proposal(proposal, redis_client, publisher)
    # Real, necessary order: closing the socket before terminating the context -- a context with
    # an still-open socket blocks term() indefinitely (a real, well-known pyzmq gotcha, caught
    # live: this script's own first run never returned without this).
    publisher.close()
    context.term()

    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    return proposal


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    asyncio.run(main())

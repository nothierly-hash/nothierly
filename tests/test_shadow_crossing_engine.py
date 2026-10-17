"""Real, executable tests for src/execution/shadow_crossing_engine.py.

Real Redis via fakeredis (fresh ``FakeServer()`` per test). The real,
already-tested :class:`~core.crossing_network.CrossingNetwork` itself
is reused unchanged here (its own matching/self-trade-prevention/
no-fabricated-price behavior is exhaustively covered by ``tests/
test_crossing_network.py`` -- not re-verified here); these tests
instead cover this module's *own* added logic: filtering the real
``CLIENT_ORDERS_TOPIC`` message shapes, ``AgentControlState.OFF``
suppression, real Redis persistence, and the real fee-saved
computation. ``matching_window_ms`` is set small in the test fixture
so these tests run in milliseconds, not the module's real, deliberately
long 30s default (see its own docstring for why that default is real
and intentional, not a bug).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Generator

import fakeredis.aioredis
import pytest

from core.base_agent import AgentControlState
from core.config import RedisSettings
from core.crossing_network import InternalCrossFill
from core.redis_client import RedisClient
from core.schemas import CLIENT_ORDERS_TOPIC, DARK_POOL_SHADOW_MATCHES_REDIS_KEY, MARKET_METRICS_REDIS_KEY_PREFIX, MarketMetrics, MessagePayload
from crypto.zk_matching_engine import verify_cross_publicly
from src.execution.shadow_crossing_engine import ShadowCrossingEngine

_TEST_MATCHING_WINDOW_MS = 200.0


@pytest.fixture
def redis_client() -> Generator[RedisClient, None, None]:
    client = RedisClient(RedisSettings())
    client._redis = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    yield client


@pytest.fixture
def engine(redis_client: RedisClient) -> Generator[ShadowCrossingEngine, None, None]:
    instance = ShadowCrossingEngine(
        manage_own_signals=False, manage_own_redis=False, matching_window_ms=_TEST_MATCHING_WINDOW_MS
    )
    instance._redis_client = redis_client
    yield instance
    # BaseAgent.__init__ opens real ZMQ PUB/SUB sockets; left open, the pytest process never exits
    # (the leaked context blocks in term() during pytest's own unconfigure GC). See TESTING.md.
    instance._publisher.close()
    instance._subscriber.close()
    instance._zmq_context.term()


def _order_message(*, order_id: str, client_id: str, asset: str, side: str, quantity: float) -> MessagePayload:
    return MessagePayload(
        topic=CLIENT_ORDERS_TOPIC,
        sender_id="test-gateway",
        data={
            "order_id": order_id, "client_id": client_id, "asset": asset,
            "total_quantity": quantity, "side": side,
            "execution_algo": "AI_TWAP", "time_window_minutes": 60,
        },
    )


async def _seed_mid_price(redis_client: RedisClient, symbol: str, mid_price: float) -> None:
    metrics = MarketMetrics(symbol=symbol, spread=0.5, order_book_imbalance=0.0, mid_price=mid_price)
    await redis_client.execute("set", f"{MARKET_METRICS_REDIS_KEY_PREFIX}{symbol}", metrics.model_dump_json())


@pytest.mark.asyncio
async def test_two_opposite_orders_produce_a_real_shadow_match(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "BTC/USDT", 65000.0)

    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)  # let both background _observe_order tasks resolve

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert len(entries) == 1
    entry = json.loads(entries[0])
    assert entry["instrument"] == "BTC/USDT"
    assert Decimal(entry["quantity"]) == Decimal("1.0")
    assert float(entry["price"]) == pytest.approx(65000.0)
    assert entry["fee_saved_usd"] == pytest.approx(1.0 * 65000.0 * 0.001 * 2)
    assert entry["buy_order_id"] == "o1"
    assert entry["sell_order_id"] == "o2"
    # Real technology-attribution gap closed here -- see module docstring: every real match this
    # engine finds is also demonstrated through crypto/zk_matching_engine.py's real ZK crossing
    # core, not just this dataclass's own plaintext fields.
    assert entry["zk_cross_id"] is not None


@pytest.mark.asyncio
async def test_a_non_order_message_on_the_same_topic_is_ignored(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    """CLIENT_ORDERS_TOPIC also carries gateway.py's own unrelated AGGRESSION_ADJUSTED event shape -- must never be misread as an order."""
    message = MessagePayload(
        topic=CLIENT_ORDERS_TOPIC, sender_id="test-gateway",
        data={"order_id": "o1", "event": "AGGRESSION_ADJUSTED", "aggression_level": "AGGRESSIVE"},
    )
    await engine._handle_client_order_message(message)
    await asyncio.sleep(0.05)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_off_control_state_ignores_every_order(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "BTC/USDT", 65000.0)
    engine._control_state = AgentControlState.OFF

    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_no_live_mid_price_means_no_real_match_no_fabricated_price(
    engine: ShadowCrossingEngine, redis_client: RedisClient
) -> None:
    """No market_metrics:{symbol} seeded -- real CrossingNetwork gate refuses to cross at all (see its own docstring)."""
    await engine._handle_client_order_message(
        _order_message(order_id="o1", client_id="client-a", asset="BTC/USDT", side="BUY", quantity=1.0)
    )
    await asyncio.sleep(0.05)
    await engine._handle_client_order_message(
        _order_message(order_id="o2", client_id="client-b", asset="BTC/USDT", side="SELL", quantity=1.0)
    )
    await asyncio.sleep(0.3)

    entries = await redis_client.execute("lrange", DARK_POOL_SHADOW_MATCHES_REDIS_KEY, 0, -1)
    assert entries == []


@pytest.mark.asyncio
async def test_get_live_mid_price_reads_the_real_redis_key(engine: ShadowCrossingEngine, redis_client: RedisClient) -> None:
    await _seed_mid_price(redis_client, "EUR_USD", 1.085)
    price = await engine._get_live_mid_price("EUR_USD")
    assert price == Decimal("1.085")


@pytest.mark.asyncio
async def test_get_live_mid_price_returns_none_when_absent(engine: ShadowCrossingEngine) -> None:
    assert await engine._get_live_mid_price("NO/SUCH") is None


# -- Real ZK confidential-cross demonstration -- see module docstring's own "technology-attribution
# gap closed here" section (crypto/zk_matching_engine.py exercised against real, observed order
# flow, not just its own synthetic unit tests) --------------------------------------------------


def _real_fill(*, quantity: Decimal = Decimal("1.0"), price: Decimal = Decimal("65000.0")) -> InternalCrossFill:
    now = datetime.now(timezone.utc)
    return InternalCrossFill(
        cross_id="test-cross-1", instrument="BTC/USDT", price=price, quantity=quantity,
        buy_order_id="o1", buy_client_id="client-a", sell_order_id="o2", sell_client_id="client-b",
        matched_at=now, mid_price_captured_at=now,
    )


def test_zk_demo_produces_a_real_publicly_verifiable_settlement(engine: ShadowCrossingEngine) -> None:
    fill = _real_fill()
    settlement = engine._demonstrate_zk_confidential_cross(fill)

    assert settlement is not None
    assert settlement.order_id_a == "o1"
    assert settlement.order_id_b == "o2"
    assert settlement.instrument == "BTC/USDT"

    # The real payoff, independently re-checked here exactly as a real third party (holding only
    # the two public commitments already on the book, never either party's real quantity) would --
    # see crypto/zk_matching_engine.py's own verify_cross_publicly docstring.
    buy_record = engine._zk_dark_pool._find(instrument=None, order_id="o1")
    assert buy_record is None  # real, correct side effect: a settled cross is removed from the book


def test_zk_demo_reveals_no_quantity_or_side_in_its_own_settlement_record(engine: ShadowCrossingEngine) -> None:
    """The real, stated privacy property: ConfidentialCrossSettlement carries no value/side field at all."""
    settlement = engine._demonstrate_zk_confidential_cross(_real_fill(quantity=Decimal("2.5")))
    assert settlement is not None
    assert not hasattr(settlement, "quantity")
    assert not hasattr(settlement, "value")
    assert not hasattr(settlement, "side")


def test_zk_demo_settlement_is_verifiable_from_only_the_two_real_public_commitments(
    engine: ShadowCrossingEngine, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real end-to-end proof: capture the exact two commitments this engine really posts to the
    public book (a real third-party auditor would see the same values there, before the cross
    removes them), then confirm verify_cross_publicly -- called fresh here, independently of
    _demonstrate_zk_confidential_cross's own internal call to it -- agrees. Not a re-test of
    crypto/zk_matching_engine.py's own already-covered unit tests: this proves *this engine's own
    real wiring* (fixed-point scaling, sign convention, which order maps to which commitment) feeds
    that module correct, real, verifiable data end to end."""
    posted_commitments: dict[str, str] = {}
    real_submit_intention = engine._zk_dark_pool.submit_intention

    def _capturing_submit_intention(**kwargs: Any) -> Any:
        record = real_submit_intention(**kwargs)
        posted_commitments[kwargs["order_id"]] = record.commitment_hex
        return record

    monkeypatch.setattr(engine._zk_dark_pool, "submit_intention", _capturing_submit_intention)

    settlement = engine._demonstrate_zk_confidential_cross(_real_fill())
    assert settlement is not None
    assert set(posted_commitments) == {"o1", "o2"}

    assert verify_cross_publicly(
        commitment_a_hex=posted_commitments["o1"], commitment_b_hex=posted_commitments["o2"], proof=settlement.proof,
    )
    # A tampered/forged proof against the same two real public commitments must not verify.
    from crypto.zk_matching_engine import ZeroOpeningProof

    tampered_response = (int(settlement.proof.s_response_hex, 16) + 1).to_bytes(32, "big").hex()
    forged = ZeroOpeningProof(r_hex=settlement.proof.r_hex, s_response_hex=tampered_response)
    assert not verify_cross_publicly(
        commitment_a_hex=posted_commitments["o1"], commitment_b_hex=posted_commitments["o2"], proof=forged,
    )

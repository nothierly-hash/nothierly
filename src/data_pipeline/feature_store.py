"""The Feature Store: reads OHLCV history from TimescaleDB, serves and
publishes each tracked instrument's latest normalized feature vector.

The interface between raw historical data and the AI agents:
:meth:`FeatureStore.get_latest_features` answers "what does this
instrument look like right now" on demand (pulled from
:mod:`~src.data_pipeline.schema`'s ``ohlcv_candles`` hypertable, run
through :class:`~src.data_pipeline.feature_engine.FeatureEngine`), and
:meth:`FeatureStore.run_forever` additionally *pushes* the same vector on
a fixed interval, two ways: over ZeroMQ (:data:`~core.schemas.
FEATURES_TOPIC`), for any agent that wants it reactively, and into Redis
(key :data:`FEATURE_VECTOR_REDIS_KEY_PREFIX` ``+ instrument``), for the
two signal agents that don't -- :class:`~src.agents.supervisor.agent.
XgboostSignalAgent` and :class:`~src.agents.rl_signal.agent.RlSignalAgent`
are both poll-on-a-timer designs (see their own docstrings), reading every
other input from Redis already, so this dual-write lets them pick up the
same feature vector the same way rather than becoming ZMQ subscribers
just for this one input.

Takes on a Redis client for exactly that reason -- optional in the sense
that a failed write is logged and swallowed, never fatal (see
:meth:`_persist_to_redis`): losing the Redis copy still leaves the ZMQ
publish intact for anything that does subscribe.

Run standalone with::

    python -m src.data_pipeline.feature_store
"""

from __future__ import annotations

import asyncio
from typing import Final

import asyncpg
import pandas as pd
import zmq.asyncio

from core.config import get_settings
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient, RedisUnavailableError
from core.schemas import FEATURE_VECTOR_REDIS_KEY_PREFIX, FEATURES_TOPIC, FeatureVector, MessagePayload
from core.timescale_client import connect_with_backoff
from core.zmq_bus import ZmqPublisher
from src.data_pipeline.feature_engine import FEATURE_COLUMNS, FeatureEngine
from src.data_pipeline.schema import OHLCV_TABLE, ensure_schema

_log = get_logger(__name__)

_SENDER_ID: Final[str] = "feature-store"
_DEFAULT_INSTRUMENTS: Final[tuple[str, ...]] = ("BTC/USDT", "ETH/USDT")
# EMA_SLOW (26) and the Z-score window (100) are the two longest lookbacks
# feature_engine.py needs -- 200 candles comfortably covers both with room
# for the rolling windows to have real data behind them, not just their
# min_periods=1 fallback.
_LOOKBACK_CANDLES: Final[int] = 200
_PUBLISH_INTERVAL_SECONDS: Final[float] = 60.0

# TTL is 3x the default publish interval: a vector that old means the
# Feature Store itself has stalled, and a reader should treat it as
# absent rather than act on a stale technical read.
_REDIS_TTL_SECONDS: Final[int] = int(_PUBLISH_INTERVAL_SECONDS * 3)

_SELECT_RECENT_OHLCV_SQL: Final[str] = f"""
SELECT time, open, high, low, close, volume
FROM {OHLCV_TABLE}
WHERE instrument = $1
ORDER BY time DESC
LIMIT $2;
"""


class FeatureStore:
    """Computes, serves, and publishes each tracked instrument's latest feature vector."""

    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        publisher: ZmqPublisher,
        redis_client: RedisClient | None = None,
        instruments: tuple[str, ...] = _DEFAULT_INSTRUMENTS,
        lookback_candles: int = _LOOKBACK_CANDLES,
        publish_interval_seconds: float = _PUBLISH_INTERVAL_SECONDS,
    ) -> None:
        """Wire up the TimescaleDB pool, ZMQ publisher, and (optional) Redis client.

        Args:
            pool: An already-connected TimescaleDB pool.
            publisher: An already-connected :class:`~core.zmq_bus.
                ZmqPublisher` (bound standalone, per the module docstring's
                ``python -m`` usage, or connected to a shared broker --
                see that class's own docstring on the two topologies).
            redis_client: An already-connected :class:`~core.redis_client.
                RedisClient`, used to additionally persist each published
                vector (see :meth:`_persist_to_redis`). ``None`` (the
                default) disables the Redis write entirely -- the ZMQ
                publish is unaffected either way.
            instruments: Instruments to compute/publish features for.
            lookback_candles: How many of the most recent candles to pull
                per :meth:`get_latest_features` call.
            publish_interval_seconds: How often :meth:`run_forever`
                republishes every instrument's vector.
        """
        self._pool = pool
        self._publisher = publisher
        self._redis_client = redis_client
        self._instruments = instruments
        self._lookback_candles = lookback_candles
        self._publish_interval_seconds = publish_interval_seconds
        self._engine = FeatureEngine()

    async def get_latest_features(self, instrument: str) -> FeatureVector | None:
        """Compute and return ``instrument``'s latest :class:`~core.schemas.FeatureVector`.

        Args:
            instrument: Which instrument to compute features for.

        Returns:
            The latest feature vector, or ``None`` if fewer than 2
            candles exist yet for this instrument (not enough to compute
            even a single-period diff, let alone any rolling window).
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_RECENT_OHLCV_SQL, instrument, self._lookback_candles)
        if len(rows) < 2:
            return None

        # rows arrive newest-first (see _SELECT_RECENT_OHLCV_SQL's ORDER
        # BY ... DESC, which lets LIMIT cheaply grab the most recent N
        # candles via the (instrument, time DESC) index) -- reverse to
        # ascending time order, which every rolling/ewm calculation in
        # feature_engine.py assumes.
        records = list(reversed(rows))
        ohlcv = pd.DataFrame(
            {
                "open": [record["open"] for record in records],
                "high": [record["high"] for record in records],
                "low": [record["low"] for record in records],
                "close": [record["close"] for record in records],
                "volume": [record["volume"] for record in records],
            },
            index=[record["time"] for record in records],
        )

        features = self._engine.compute_features(ohlcv)
        latest_row = features.iloc[-1]
        latest_time = ohlcv.index[-1]

        return FeatureVector(
            instrument=instrument,
            timestamp=latest_time.timestamp(),
            **{column: float(latest_row[column]) for column in FEATURE_COLUMNS},
        )

    async def publish_latest(self, instrument: str) -> FeatureVector | None:
        """Compute ``instrument``'s latest vector; publish it on ZMQ and persist it to Redis."""
        vector = await self.get_latest_features(instrument)
        if vector is None:
            _log.debug("feature_store_insufficient_history", instrument=instrument)
            return None

        payload = MessagePayload(topic=FEATURES_TOPIC, sender_id=_SENDER_ID, data=vector.model_dump(mode="json"))
        await self._publisher.publish(payload)
        await self._persist_to_redis(vector)
        _log.info(
            "feature_vector_published",
            instrument=instrument,
            ema_fast=round(vector.ema_fast, 4),
            ema_slow=round(vector.ema_slow, 4),
            rsi=round(vector.rsi, 4),
            atr=round(vector.atr, 4),
        )
        return vector

    async def _persist_to_redis(self, vector: FeatureVector) -> None:
        """Best-effort write of ``vector`` to Redis, with a TTL, never fatal on failure.

        See :data:`FEATURE_VECTOR_REDIS_KEY_PREFIX`. A failure here (no
        client configured, or a transient Redis outage) is logged and
        swallowed -- :meth:`publish_latest`'s ZMQ publish already
        succeeded by the time this runs, so this is strictly additive,
        not a reason to fail the whole publish cycle.
        """
        if self._redis_client is None:
            return
        key = f"{FEATURE_VECTOR_REDIS_KEY_PREFIX}{vector.instrument}"
        try:
            await self._redis_client.execute(
                "set", key, vector.model_dump_json(), ex=_REDIS_TTL_SECONDS
            )
        except RedisUnavailableError:
            _log.warning("feature_vector_redis_persist_failed", instrument=vector.instrument)

    async def run_forever(self) -> None:
        """Publish every tracked instrument's latest feature vector on a fixed interval, forever."""
        while True:
            for instrument in self._instruments:
                try:
                    await self.publish_latest(instrument)
                except Exception:
                    _log.exception("feature_store_publish_failed", instrument=instrument)
            await asyncio.sleep(self._publish_interval_seconds)


async def main() -> None:
    """Connect to TimescaleDB + Redis, bind a standalone ZMQ publisher, then run forever."""
    configure_logging(log_level="INFO", log_dir=None, service_name="feature-store")
    settings = get_settings()
    pool = await connect_with_backoff(settings.timescale)
    await ensure_schema(pool)

    context = zmq.asyncio.Context()
    publisher = ZmqPublisher(
        settings.zmq.pub_address, high_water_mark=settings.zmq.high_water_mark, context=context
    )

    try:
        redis_client: RedisClient | None = await RedisClient.get_instance(
            settings.redis, max_attempts=5
        )
    except RedisUnavailableError:
        _log.warning(
            "feature_store_starting_without_redis",
            detail="Signal agents will fall back to their pre-Feature-Store behavior.",
        )
        redis_client = None

    store = FeatureStore(pool=pool, publisher=publisher, redis_client=redis_client)
    try:
        await store.run_forever()
    finally:
        publisher.close()
        await asyncio.to_thread(context.term)
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

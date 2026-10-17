"""Tests du heartbeat du moteur sur thread dédié (``core/heartbeat.py``). Aucun Redis réel : client factice en mémoire."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from core.heartbeat import (
    ENGINE_STATE_KEY,
    STATE_ACTIVE,
    STATE_STARTING,
    HeartbeatConfig,
    ThreadedHeartbeat,
)

KEY = "heartbeat:execution-agent"


class FakeClient:
    def __init__(self, store: dict[str, tuple[str, int | None]], fail: list[bool]) -> None:
        self.store, self._fail, self.closed = store, fail, False

    def set(self, name: str, value: str, *, px: int | None = None) -> object:
        if self._fail and self._fail[0]:
            raise ConnectionError("redis injoignable")
        self.store[name] = (value, px)
        return True

    def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self) -> None:
        self.store: dict[str, tuple[str, int | None]] = {}
        self.fail: list[bool] = [False]
        self.built = 0
        self.clients: list[FakeClient] = []

    def __call__(self) -> FakeClient:
        self.built += 1
        client = FakeClient(self.store, self.fail)
        self.clients.append(client)
        return client


def _wait(predicate: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _hb(factory: Factory, **cfg: Any) -> ThreadedHeartbeat:
    cfg.setdefault("interval_s", 0.05)
    cfg.setdefault("ttl_s", 0.2)
    return ThreadedHeartbeat(HeartbeatConfig(key=KEY, **cfg), factory)


# --- Configuration ---------------------------------------------------------------------------


def test_a_ttl_under_three_intervals_is_refused() -> None:
    with pytest.raises(ValueError, match="TTL"):
        HeartbeatConfig(key=KEY, interval_s=3.0, ttl_s=8.9)
    HeartbeatConfig(key=KEY, interval_s=3.0, ttl_s=9.0)  # la limite exacte passe
    HeartbeatConfig(key=KEY)  # défauts 3 s / 10 s


@pytest.mark.parametrize("interval,ttl", [(0.0, 10.0), (3.0, 0.0), (-1.0, 10.0)])
def test_non_positive_durations_are_refused(interval: float, ttl: float) -> None:
    with pytest.raises(ValueError, match="strictement positifs"):
        HeartbeatConfig(key=KEY, interval_s=interval, ttl_s=ttl)


# --- Armement --------------------------------------------------------------------------------


def test_start_writes_the_starting_state_before_returning_then_active_on_demand() -> None:
    f = Factory()
    hb = _hb(f)
    hb.start()
    try:
        assert KEY in f.store and f.store[KEY][1] == 200  # TTL en ms, écrit AVANT le retour de start()
        assert json.loads(f.store[ENGINE_STATE_KEY][0])["state"] == STATE_STARTING
        hb.set_state(STATE_ACTIVE)
        assert _wait(lambda: json.loads(f.store[ENGINE_STATE_KEY][0])["state"] == STATE_ACTIVE)
        assert hb.state == STATE_ACTIVE
    finally:
        hb.stop()
    assert not hb.running


def test_start_is_idempotent() -> None:
    f = Factory()
    hb = _hb(f)
    hb.start()
    hb.start()
    try:
        assert sum(t.name == f"heartbeat:{KEY}" for t in threading.enumerate()) == 1
    finally:
        hb.stop()


def test_the_state_key_can_be_disabled() -> None:
    f = Factory()
    hb = _hb(f, state_key=None)
    hb.start()
    hb.stop()
    assert KEY in f.store and ENGINE_STATE_KEY not in f.store


# --- Découplage de la boucle asyncio ---------------------------------------------------------


def test_the_heartbeat_keeps_beating_while_the_event_loop_is_blocked() -> None:
    """Le point du refactor : un calcul qui bloque la boucle ne doit pas retarder le battement."""
    f = Factory()
    hb = _hb(f)

    async def blocked() -> int:
        hb.start()
        before = hb.beats
        time.sleep(0.5)  # blocage synchrone de la boucle : aucune tâche asyncio ne tournerait
        return hb.beats - before

    try:
        assert asyncio.run(blocked()) >= 3  # ~10 battements attendus à 50 ms
    finally:
        hb.stop()


# --- Pannes ----------------------------------------------------------------------------------


def test_a_redis_failure_never_raises_and_the_heartbeat_recovers_with_a_fresh_client() -> None:
    f = Factory()
    hb = _hb(f, log_every_s=0.0)
    hb.start()
    try:
        f.fail[0] = True
        assert _wait(lambda: hb.failures >= 2)
        assert hb.running  # le thread a survécu
        assert f.clients[0].closed  # le client cassé a été jeté
        f.fail[0] = False
        built = f.built
        beats = hb.beats
        assert _wait(lambda: hb.beats > beats)
        assert f.built > built  # un nouveau client a été construit, pas l'ancien réutilisé
    finally:
        hb.stop()


def test_a_factory_that_cannot_connect_is_counted_not_raised() -> None:
    def broken() -> FakeClient:
        raise ConnectionError("refusé")

    hb = ThreadedHeartbeat(HeartbeatConfig(key=KEY, interval_s=0.05, ttl_s=0.2), broken)
    hb.start()  # ne lève pas : un Redis absent au lancement ne doit pas empêcher le moteur de démarrer
    try:
        assert hb.failures >= 1 and hb.beats == 0 and hb.running
    finally:
        hb.stop()


def test_failure_logging_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    f = Factory()
    f.fail[0] = True
    hb = _hb(f, log_every_s=3600.0)
    with caplog.at_level("WARNING", logger="core.heartbeat"):
        hb.start()
        assert _wait(lambda: hb.failures >= 5)
        hb.stop()
    assert sum("engine_heartbeat_write_failed" in r.message for r in caplog.records) == 1


def test_stop_ends_the_thread_and_closes_the_client() -> None:
    f = Factory()
    hb = _hb(f)
    hb.start()
    hb.stop()
    assert not hb.running and f.clients[-1].closed
    hb.stop()  # idempotent


def test_the_execution_agents_heartbeat_constants_respect_the_ttl_ratio() -> None:
    from src.agents.execution import agent

    HeartbeatConfig(key=KEY, interval_s=agent._HEARTBEAT_INTERVAL_SECONDS, ttl_s=agent._HEARTBEAT_TTL_SECONDS)  # lève si < 3x
    assert not hasattr(agent.ExecutionAgent, "_run_heartbeat_loop")  # plus de tâche asyncio : le thread est la seule source

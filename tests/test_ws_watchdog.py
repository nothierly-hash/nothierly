"""Watchdog de fraîcheur des flux WebSocket : une connexion semi-ouverte ne doit jamais laisser une valeur gelée passer pour fraîche."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest

from tests.test_fill_tracking import Harness, RecordingAlerter
from tests.test_vpin import Alerts, Rig, _agg
from workers.common import ws_tracker
from workers.common.alerts import Severity
from workers.common.ws_tracker import BinanceFillFeed, BybitFillFeed, FeedStaleError, HyperliquidFillFeed, enable_tcp_keepalive
from workers.market_data import vpin_tracker as vt


def _stale_rig(**kwargs: Any) -> Rig:
    kwargs.setdefault("stale_after_s", 60.0)
    kwargs.setdefault("refresh_s", 1000.0)
    return Rig(**kwargs)


# --- VPIN : watchdog ---------------------------------------------------------------------------


def test_a_silent_feed_deletes_the_key_alerts_once_and_reconnects() -> None:
    alerts = Alerts()
    rig = _stale_rig(alerter=alerts)
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)], []]  # 1re connexion : 2 seaux puis silence ; 2e : rien
    state = {"advanced": False}

    def condition() -> bool:
        if len(rig.sink.published) == 1 and not state["advanced"]:  # la valeur est publiée : le lien « meurt » (aucune exception)
            rig.clock.now += 61.0
            state["advanced"] = True
        return len(rig.connections) >= 2

    asyncio.run(rig.run_until(condition))
    assert rig.tracker.stats["stale"] >= 1 and rig.connections[0].closed
    assert "BTCUSDT" in rig.sink.cleared  # le moteur lit « pas de lecture » : il se sait aveugle
    assert [k for k, _, _ in alerts.alerts].count("vpin_feed_stale") == 1


def test_a_stale_gap_starts_at_the_last_message_so_the_window_is_reset() -> None:
    rig = _stale_rig(max_gap_s=30.0)
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)], []]
    state = {"advanced": False}

    def condition() -> bool:
        if len(rig.sink.published) == 1 and not state["advanced"]:
            rig.clock.now += 61.0
            state["advanced"] = True
        return len(rig.connections) >= 2 and rig.tracker.stats["resets"] >= 1

    asyncio.run(rig.run_until(condition))
    # 61 s sans donnée > 30 s : la fenêtre est vidée, même si la reconnexion elle-même a pris quelques ms (le trou se mesure depuis le dernier message)
    assert rig.tracker.stats["resets"] >= 1 and rig.tracker.snapshot() == {"BTCUSDT": None}


def test_nothing_is_rewritten_to_redis_until_a_message_arrives_after_reconnecting() -> None:
    rig = _stale_rig(refresh_s=5.0, max_gap_s=1000.0)  # fenêtre conservée : une valeur existe, c'est elle qu'on ne doit PAS ressortir
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)], []]
    state = {"step": 0, "published_at_stale": 0}

    def condition() -> bool:
        if state["step"] == 0 and len(rig.sink.published) == 1:
            rig.clock.now += 61.0
            state["step"] = 1
        elif state["step"] == 1 and len(rig.connections) >= 2:
            state["published_at_stale"] = len(rig.sink.published)
            rig.clock.now += 6.0  # un cycle de ré-écriture passe sur la 2e connexion, toujours muette
            state["step"] = 2
        elif state["step"] == 2:
            state["step"] = 3
        return state["step"] == 3

    asyncio.run(rig.run_until(condition))
    asyncio.run(asyncio.sleep(0))
    assert len(rig.sink.published) == state["published_at_stale"]  # pas de valeur gelée republiée


def test_a_message_after_the_stale_episode_announces_the_recovery_and_republishes() -> None:
    alerts = Alerts()
    rig = _stale_rig(alerter=alerts, max_gap_s=1000.0)
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)], [_agg(agg_id=3)]]
    state = {"advanced": False}

    def condition() -> bool:
        if len(rig.sink.published) == 1 and not state["advanced"]:
            rig.clock.now += 61.0
            state["advanced"] = True
        return len(rig.sink.published) >= 2

    asyncio.run(rig.run_until(condition))
    keys = [k for k, _, _ in alerts.alerts]
    assert keys[:2] == ["vpin_feed_stale", "vpin_feed_stale_end"] and alerts.alerts[1][2] is Severity.INFO


def test_the_watchdog_can_be_disabled() -> None:
    rig = _stale_rig(stale_after_s=0)
    rig.scripts = [[_agg(agg_id=1), _agg(agg_id=2, maker=True)]]
    state = {"advanced": False}

    def condition() -> bool:
        if len(rig.sink.published) == 1 and not state["advanced"]:
            rig.clock.now += 10_000.0
            state["advanced"] = True
            return False
        return state["advanced"] and rig.tracker.stats["trades"] == 2

    asyncio.run(rig.run_until(condition, timeout_s=0.5))
    assert rig.tracker.stats["stale"] == 0 and len(rig.connections) == 1


def test_the_command_line_passes_the_threshold_and_refuses_a_negative_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from tests.test_vpin import _run_cli

    seen = _run_cli(monkeypatch, "--dry-run")
    assert seen["kwargs"]["stale_after_s"] == 60.0
    assert _run_cli(monkeypatch, "--dry-run", "--stale-after-s", "0")["kwargs"]["stale_after_s"] == 0.0
    monkeypatch.setattr(vt, "configure_logging", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        vt.main(["--dry-run", "--stale-after-s", "-1"])


# --- Flux de fills -----------------------------------------------------------------------------


def test_the_default_threshold_is_three_pings_and_absent_for_a_feed_that_never_pings() -> None:
    h = Harness(HyperliquidFillFeed(user="0xabc"))
    assert h.tracker._stale_after_s == 150.0  # 3 x 50 s
    assert Harness(BybitFillFeed(api_key="k", api_secret="s")).tracker._stale_after_s == 60.0  # 3 x 20 s
    class NoPing(BinanceFillFeed):
        def __init__(self) -> None:
            pass

    assert Harness(NoPing()).tracker._stale_after_s is None  # silence normal tant qu'on ne trade pas : le keepalive TCP le couvre


def test_a_fill_feed_that_goes_mute_is_reconnected_and_alerted_once() -> None:
    class FastFeed(HyperliquidFillFeed):
        heartbeat_interval_s = 0.02  # pings très rapprochés : les pongs manquent, le flux est mort

    alerter = RecordingAlerter()
    h = Harness(FastFeed(user="0xabc"), alerter=alerter, stale_after_s=0.1)
    h.script()
    asyncio.run(h.run_until(lambda: len(h.connections) >= 3))
    assert h.tracker.stats["stale"] >= 2 and len(h.connections) >= 3
    assert [k for k, _, _ in alerter.alerts].count("fill_feed_stale") == 1  # une alerte par épisode, pas une par reconnexion


def test_a_stale_error_is_an_ordinary_exception_for_the_reconnect_loop() -> None:
    assert issubclass(FeedStaleError, RuntimeError)


# --- Keepalive TCP -----------------------------------------------------------------------------


class _FakeSock:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.fail = fail

    def setsockopt(self, level: int, option: int, value: int) -> None:
        if self.fail:
            raise OSError("non supporté")
        self.calls.append((level, option, value))


class _FakeWs:
    def __init__(self, sock: Any) -> None:
        self.transport = type("T", (), {"get_extra_info": lambda self, name: sock})()


def test_tcp_keepalive_is_enabled_with_the_documented_timings() -> None:
    sock = _FakeSock()
    assert enable_tcp_keepalive(_FakeWs(sock)) is True
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in sock.calls
    if hasattr(socket, "TCP_KEEPIDLE"):
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30) in sock.calls
        assert (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3) in sock.calls


def test_tcp_keepalive_never_breaks_the_connection_when_unavailable() -> None:
    assert enable_tcp_keepalive(object()) is False  # pas de transport
    assert enable_tcp_keepalive(_FakeWs(None)) is False  # pas de prise
    assert enable_tcp_keepalive(_FakeWs(_FakeSock(fail=True))) is False  # plateforme qui refuse l'option
    assert ws_tracker.enable_tcp_keepalive(_FakeWs(_FakeSock())) is True

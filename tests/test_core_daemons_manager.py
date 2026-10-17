"""The Ghost Daemons manager, which owns a detached background thread, is one per server process, not one per
browser session."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest

from src.dashboard import core_daemons

_THREAD_NAME = "ghost-daemon-swarm"


def _swarm_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _THREAD_NAME]


@pytest.fixture
def quiet_swarm(monkeypatch: pytest.MonkeyPatch) -> Iterator[threading.Event]:
    """Replace the infinite swarm loop with one that idles until told to stop, so the test owns the thread."""
    stop = threading.Event()

    async def idle(self: core_daemons.LeviathanDaemonManager, interval: float = 2.0) -> None:
        while not stop.is_set():
            await asyncio.sleep(0.01)

    monkeypatch.setattr(core_daemons.LeviathanDaemonManager, "run_forever", idle)
    core_daemons._process_wide_manager.clear()
    yield stop
    stop.set()
    for thread in _swarm_threads():
        thread.join(timeout=5.0)
    core_daemons._process_wide_manager.clear()


def test_every_session_gets_the_same_manager() -> None:
    core_daemons._process_wide_manager.clear()
    try:
        assert core_daemons.get_manager() is core_daemons.get_manager()
    finally:
        core_daemons._process_wide_manager.clear()


def test_many_sessions_start_one_swarm_thread_not_one_each(quiet_swarm: threading.Event) -> None:
    for _session in range(6):  # what six browser sessions each do on their first script run
        core_daemons.get_manager().ensure_background_running()
        core_daemons.get_manager().ensure_background_running()  # and again on every rerun

    assert len(_swarm_threads()) == 1


def test_a_dead_swarm_thread_is_restarted_once_for_all_sessions(quiet_swarm: threading.Event) -> None:
    core_daemons.get_manager().ensure_background_running()
    quiet_swarm.set()
    for thread in _swarm_threads():
        thread.join(timeout=5.0)
    assert _swarm_threads() == []
    quiet_swarm.clear()

    for _session in range(3):
        core_daemons.get_manager().ensure_background_running()

    assert len(_swarm_threads()) == 1

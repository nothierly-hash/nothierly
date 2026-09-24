from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

import core.watchdog as watchdog_module
from core.watchdog import SystemWatchdog


async def _noop_shutdown() -> None:
    pass


def _make_watchdog(heartbeat_file: Path) -> SystemWatchdog:
    return SystemWatchdog(
        redis_client=None,  # type: ignore[arg-type]
        shutdown_callback=_noop_shutdown,
        heartbeat_file=heartbeat_file,
        container_memory_limit_bytes=10 * 1024 * 1024 * 1024,
    )


def test_default_heartbeat_path_is_os_agnostic() -> None:
    assert watchdog_module._DEFAULT_HEARTBEAT_FILE.parent == Path(tempfile.gettempdir())
    assert watchdog_module._DEFAULT_HEARTBEAT_FILE.name == "nothierly_heartbeat.txt"


def test_write_heartbeat_file_creates_valid_json_content(tmp_path: Path) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(heartbeat_file)

    before = time.time()
    watchdog._write_heartbeat_file()
    after = time.time()

    assert heartbeat_file.exists()
    payload = json.loads(heartbeat_file.read_text(encoding="utf-8"))
    assert before <= payload["timestamp"] <= after
    assert payload["pid"] == os.getpid()


def test_write_heartbeat_file_leaves_no_tmp_file_behind(tmp_path: Path) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(heartbeat_file)

    watchdog._write_heartbeat_file()

    assert not (tmp_path / "heartbeat.txt.tmp").exists()


def test_write_heartbeat_file_uses_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    heartbeat_file.write_text(json.dumps({"timestamp": 0.0, "pid": -1}))
    watchdog = _make_watchdog(heartbeat_file)

    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(src: object, dst: object) -> None:
        calls.append((Path(src), Path(dst)))  # type: ignore[arg-type]
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    watchdog._write_heartbeat_file()

    assert calls == [(tmp_path / "heartbeat.txt.tmp", heartbeat_file)]
    payload = json.loads(heartbeat_file.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_write_heartbeat_file_raises_runtime_error_on_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(heartbeat_file)

    def failing_open(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("builtins.open", failing_open)

    with pytest.raises(RuntimeError):
        watchdog._write_heartbeat_file()


def test_write_heartbeat_file_raises_runtime_error_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    heartbeat_file = tmp_path / "heartbeat.txt"
    watchdog = _make_watchdog(heartbeat_file)

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(RuntimeError):
        watchdog._write_heartbeat_file()

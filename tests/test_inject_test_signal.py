"""Tests de ``scripts/inject_test_signal.py`` : le script publie un signal signé que le worker accepte."""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from types import ModuleType

import pytest
import zmq

from contracts.platform_signal import PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SECRET_ENV_VAR, open_envelope

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inject_test_signal.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inject_test_signal", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["inject_test_signal"] = module
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _subscriber(endpoint: str) -> tuple[zmq.Context, zmq.Socket]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, DEFAULT_TOPIC)
    sub.connect(endpoint)  # lancé AVANT le script, comme le worker : il se reconnecte quand le script écoute
    return ctx, sub


def test_script_publishes_a_signed_signal_the_hyperliquid_worker_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    ctx, sub = _subscriber(endpoint)
    try:
        assert _load().main(["--endpoint", endpoint, "--settle", "1.0", "--repeat", "3"]) == 0
        assert sub.poll(3000), "le signal n'est jamais arrivé : problème de slow joiner ou de bind"
        frames = sub.recv_multipart()
    finally:
        ctx.destroy(linger=0)
    assert frames[0] == DEFAULT_TOPIC
    verdict = open_envelope(frames[1], SECRET, venue=TargetVenue.HYPERLIQUID)
    assert verdict.accepted
    signal = verdict.signal
    assert isinstance(signal, PlatformSignal)
    assert (signal.direction, signal.symbol, signal.position_size_pct, signal.ttl_ms) == (
        SignalDirection.LONG, "CRYPTO_BTC_USD", 5.0, 10_000)
    assert signal.confidence_score is None


def test_script_signal_is_rejected_by_a_worker_with_another_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    ctx, sub = _subscriber(endpoint)
    try:
        assert _load().main(["--endpoint", endpoint, "--settle", "1.0", "--repeat", "2"]) == 0
        assert sub.poll(3000)
        frames = sub.recv_multipart()
    finally:
        ctx.destroy(linger=0)
    wrong_key = "another-secret-9876543210fedcba-9876543210fedcba"
    assert open_envelope(frames[1], wrong_key, venue=TargetVenue.HYPERLIQUID).reject_reason == "bad_signature"


def test_flat_defaults_to_zero_size_and_contract_errors_are_reported(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    module = _load()
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    assert module.main(["--endpoint", endpoint, "--settle", "0.1", "--repeat", "1", "--direction", "FLAT"]) == 0
    assert module.main(["--endpoint", endpoint, "--settle", "0.1", "--repeat", "1", "--direction", "FLAT", "--pct", "3"]) == 2
    assert "FLAT" in capsys.readouterr().err  # message du contrat, pas une trace d'exception
    assert module.main(["--endpoint", endpoint, "--settle", "0.1", "--repeat", "1", "--symbol", "CRYPTO_BTC_USD", "--venue", "NUMERAI_SIGNALS"]) == 2


def test_script_refuses_remote_endpoints_and_weak_or_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    assert module.main(["--endpoint", "tcp://10.0.0.5:25570", "--settle", "0"]) == 2
    monkeypatch.setenv(SECRET_ENV_VAR, "trop-court")
    assert module.main(["--endpoint", f"tcp://127.0.0.1:{_free_port()}", "--settle", "0"]) == 2
    monkeypatch.delenv(SECRET_ENV_VAR)
    assert module.main(["--endpoint", f"tcp://127.0.0.1:{_free_port()}", "--settle", "0"]) == 2


def test_a_forex_signal_for_darwinex_may_omit_the_stop_loss_and_carries_it_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_ENV_VAR, SECRET)
    module = _load()
    base = ["--symbol", "FOREX_EUR_USD", "--venue", "DARWINEX_ZERO", "--settle", "0.1", "--repeat", "1"]
    assert module.main(["--endpoint", f"tcp://127.0.0.1:{_free_port()}", *base]) == 0  # sans stop : accepté, le worker en pose un
    endpoint = f"tcp://127.0.0.1:{_free_port()}"
    ctx, sub = _subscriber(endpoint)
    try:
        assert module.main(["--endpoint", endpoint, *base, "--settle", "1.0", "--repeat", "3", "--stop-loss", "1.095"]) == 0
        assert sub.poll(3000)
        frames = sub.recv_multipart()
    finally:
        ctx.destroy(linger=0)
    signal = open_envelope(frames[1], SECRET, venue=TargetVenue.DARWINEX_ZERO).signal
    assert signal is not None and signal.metadata == {"stop_loss_price": 1.095}

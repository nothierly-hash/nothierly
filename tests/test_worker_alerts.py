"""Tests des alertes webhook (``workers/common/alerts.py``), de leur intégration au socle (``workers/common/base.py``) et de la
rotation des journaux (``workers/common/logging_setup.py``).

Preuves recherchées :

- l'alerteur ne bloque jamais l'appelant et ne lève jamais (webhook lent, hors ligne, en erreur, file pleine), et n'écrit
  jamais l'URL (elle porte le jeton) dans un journal ni une erreur ;
- le socle n'alerte QUE sur les urgences (homme mort, fermeture d'urgence incomplète, drawdown, pannes consécutives) et pas
  sur les ordres ou rejets ordinaires ;
- le fichier de journal ne grossit pas sans limite.

Les serveurs HTTP sont locaux (``127.0.0.1``, port libre) : aucun accès réseau extérieur.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import socket
import threading
import time
from collections.abc import Iterator
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from workers.common.alerts import AlertConfigError, Severity, WebhookAlerter, build_payload, validate_webhook_url
from workers.common.base import ExposureWorkerBase, ExposureWorkerConfig
from workers.common.exposure import RiskLimits
from workers.common.logging_setup import ENV_LOG_DIR, configure_logging

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
TOKEN = "SECRETTOKEN-abc123"
T0 = 1_760_000_000_000
D = Decimal


# --- Serveur de webhook local ----------------------------------------------------------------


class _Hook:
    """Serveur HTTP local qui enregistre les corps reçus ; ``delay_s`` simule un service lent, ``status`` une erreur."""

    def __init__(self, *, delay_s: float = 0.0, status: int = 200) -> None:
        self.bodies: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- nom imposé par http.server
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if delay_s:
                    time.sleep(delay_s)
                outer.bodies.append(json.loads(raw))
                try:
                    self.send_response(status)
                    self.end_headers()
                    self.wfile.write(b"ok")
                except OSError:
                    pass  # le client a déjà abandonné (délai dépassé)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/hook/{TOKEN}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def hook() -> Iterator[_Hook]:
    server = _Hook()
    yield server
    server.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- Alerteur : configuration ----------------------------------------------------------------


def test_the_url_must_be_https_or_loopback_and_an_error_never_echoes_it() -> None:
    assert validate_webhook_url(f"https://discord.com/api/webhooks/1/{TOKEN}") == "discord.com"
    assert validate_webhook_url("http://127.0.0.1:9/x") == "127.0.0.1"
    for bad in (f"http://evil.example/{TOKEN}", f"ftp://x.example/{TOKEN}", f"file:///{TOKEN}", TOKEN, "https://"):
        with pytest.raises(AlertConfigError) as exc:
            validate_webhook_url(bad)
        assert TOKEN not in str(exc.value)


def test_from_env_returns_none_when_unset_or_blank_and_fails_loudly_when_invalid() -> None:
    assert WebhookAlerter.from_env(source="x", environ={}) is None
    assert WebhookAlerter.from_env(source="x", environ={"WEBHOOK_URL": "   "}) is None
    assert WebhookAlerter.from_env(source="x", environ={"WEBHOOK_URL": "https://discord.com/api/webhooks/1/t"}) is not None
    with pytest.raises(AlertConfigError):
        WebhookAlerter.from_env(source="x", environ={"WEBHOOK_URL": "http://evil.example/t"})


def test_the_payload_shape_follows_the_service_and_is_truncated_to_its_limit() -> None:
    assert build_payload("discord.com", "a" * 3000) == {"content": "a" * 2000}
    assert build_payload("api.telegram.org", "a" * 5000) == {"text": "a" * 4096}
    assert build_payload("hooks.slack.com", "x") == {"text": "x"}
    assert build_payload("example.org", "x") == {"content": "x", "text": "x"}


# --- Alerteur : envoi ------------------------------------------------------------------------


def test_an_alert_is_posted_as_json_with_severity_and_source(hook: _Hook) -> None:
    alerter = WebhookAlerter(hook.url, source="hl-worker")
    assert alerter.alert("homme mort déclenché", key="k") is True
    assert alerter.flush(5) and alerter.delivered == 1
    assert hook.bodies == [{"content": "[CRITIQUE][hl-worker] homme mort déclenché", "text": "[CRITIQUE][hl-worker] homme mort déclenché"}]
    alerter.alert("rétabli", key="k2", severity=Severity.INFO)
    assert alerter.flush(5) and hook.bodies[-1]["text"] == "[INFO][hl-worker] rétabli"
    alerter.close()


def test_the_same_key_is_throttled_for_the_cooldown_and_other_keys_are_not() -> None:
    clock = Clock()
    alerter = WebhookAlerter(f"http://127.0.0.1:{_free_port()}/x", source="s", cooldown_s=300, clock=clock)
    assert alerter.alert("a", key="dead_man") is True
    assert alerter.alert("a again", key="dead_man") is False  # même clé, dans le délai
    assert alerter.alert("b", key="drawdown") is True  # autre clé
    clock.now += 299
    assert alerter.alert("a", key="dead_man") is False
    clock.now += 2  # 301 s : le délai est écoulé
    assert alerter.alert("a", key="dead_man") is True
    alerter.close()


def test_a_slow_webhook_never_blocks_the_caller_and_times_out_quietly() -> None:
    slow = _Hook(delay_s=1.5)
    try:
        alerter = WebhookAlerter(slow.url, source="s", timeout_s=0.3)
        started = time.monotonic()
        assert alerter.alert("x", key="k") is True
        assert time.monotonic() - started < 0.2  # l'appelant n'a rien attendu
        assert alerter.flush(5)
        assert alerter.failed == 1 and alerter.delivered == 0
    finally:
        slow.close()


@pytest.mark.parametrize("status", [500, 404])
def test_an_error_response_counts_as_a_failure_and_raises_nothing(status: int) -> None:
    broken = _Hook(status=status)
    try:
        alerter = WebhookAlerter(broken.url, source="s")
        alerter.alert("x", key="k")
        assert alerter.flush(5) and alerter.failed == 1
    finally:
        broken.close()


def test_an_unreachable_webhook_raises_nothing_and_the_url_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    alerter = WebhookAlerter(f"http://127.0.0.1:{_free_port()}/hook/{TOKEN}", source="s", timeout_s=1.0)
    with caplog.at_level(logging.DEBUG):
        assert alerter.alert("x", key="k") is True
        assert alerter.flush(5)
    assert alerter.failed == 1
    assert "alert_delivery_failed" in caplog.text and TOKEN not in caplog.text


def test_a_full_queue_drops_alerts_instead_of_blocking() -> None:
    slow = _Hook(delay_s=1.0)
    try:
        alerter = WebhookAlerter(slow.url, source="s", queue_size=1, timeout_s=0.2, cooldown_s=0)
        results = [alerter.alert(str(i), key=f"k{i}") for i in range(5)]
        assert False in results and alerter.dropped >= 1
        alerter.flush(5)
    finally:
        slow.close()


# --- Intégration au socle --------------------------------------------------------------------


class RecordingAlerter:
    def __init__(self, *, fail: bool = False) -> None:
        self.alerts: list[tuple[str, str, Severity]] = []
        self._fail = fail

    def alert(self, text: str, *, key: str, severity: Severity = Severity.CRITICAL) -> bool:
        if self._fail:
            raise RuntimeError("l'alerteur lui-même plante")
        self.alerts.append((key, text, severity))
        return True

    @property
    def keys(self) -> list[str]:
        return [k for k, _, _ in self.alerts]


class ScriptedWorker(ExposureWorkerBase):
    venue: ClassVar[TargetVenue] = TargetVenue.HYPERLIQUID
    _tag: ClassVar[str] = "scripted"
    supports_dead_man: ClassVar[bool] = True

    def __init__(self, config: ExposureWorkerConfig) -> None:
        super().__init__(config)
        self.script: list[str | Exception] = []
        self.positions: dict[str, Decimal] = {"BTC": D(1)}
        self.flatten_result = "sent"

    def map_symbol(self, symbol: str) -> str | None:
        return {"CRYPTO_BTC_USD": "BTC"}.get(symbol)

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def emergency_flatten(self, asset: str, *, attempt: int, now_ms: int) -> str:
        return self.flatten_result


def _worker(tmp_path: Path, alerter: Any, **overrides: Any) -> ScriptedWorker:
    base: dict[str, Any] = dict(
        address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "w.sqlite3", dry_run=False,
        dead_man_timeout_s=300.0, dead_man_startup_grace_s=0.0, alerter=alerter, api_failure_threshold=3,
    )
    base.update(overrides)
    worker = ScriptedWorker(ExposureWorkerConfig(**base))
    worker._started_ms = T0 - 10_000_000
    return worker


def _signal() -> PlatformSignal:
    return PlatformSignal(
        target_venue=TargetVenue.ALL, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD", direction=SignalDirection.LONG,
        position_size_pct=5.0, order_type=OrderType.MARKET, timestamp_ms=T0,
    )


async def _execute(worker: ScriptedWorker, count: int) -> None:
    """Fait passer ``count`` signaux par le vrai exécuteur du socle, un à la fois."""
    stop, wake = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(worker._executor(stop, wake))
    for _ in range(count):
        worker._pending["BTC"] = _signal()
        wake.set()
        while worker._pending or wake.is_set():
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.01)
    stop.set()
    wake.set()
    await task


def test_ordinary_outcomes_never_alert(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w.script = ["ok", "dry_run", "stale_at_execution", "size_below_min", "within_tolerance", "ok"] * 3
    asyncio.run(_execute(w, 18))
    assert rec.alerts == []


def test_consecutive_api_failures_alert_once_then_announce_the_recovery(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w.script = ["read_error", "error", "read_error", "read_error", "read_error", "ok"]
    asyncio.run(_execute(w, 6))
    assert rec.keys == ["streak:execute", "recovered:execute"]  # un seul message pour 5 échecs (seuil 3), puis le rétablissement
    assert "3 échecs" in rec.alerts[0][1] and rec.alerts[0][2] is Severity.CRITICAL and rec.alerts[1][2] is Severity.INFO
    w.script = ["error", "error", "error"]  # une nouvelle série réalerte
    asyncio.run(_execute(w, 3))
    assert rec.keys == ["streak:execute", "recovered:execute", "streak:execute"]


def test_failures_that_are_not_consecutive_do_not_alert(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w.script = ["read_error", "read_error", "ok", "read_error", "read_error", "ok"]
    asyncio.run(_execute(w, 6))
    assert rec.alerts == []


def test_unhandled_exceptions_count_and_the_executor_keeps_running(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w.script = [RuntimeError("a"), ValueError("b"), RuntimeError("c"), "ok"]
    asyncio.run(_execute(w, 4))
    assert rec.keys == ["streak:execute", "recovered:execute"] and "RuntimeError" in rec.alerts[0][1]


def test_the_dead_man_switch_alerts_once_when_it_trips_and_again_if_the_flatten_is_incomplete(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w.flatten_result = "error"  # la fermeture d'urgence échoue
    w._state.set("last_signal:BTC", str(T0))
    assert asyncio.run(w.check_dead_man(T0 + 301_000)) == "tripped"
    assert rec.keys == ["dead_man_tripped"] and "HOMME MORT" in rec.alerts[0][1] and "300" in rec.alerts[0][1]
    assert asyncio.run(w.check_dead_man(T0 + 302_000)) == "halted_flattening"  # tentative 2 : pas encore d'alerte
    assert rec.keys == ["dead_man_tripped"]
    assert asyncio.run(w.check_dead_man(T0 + 303_000)) == "halted_flattening"  # tentative 3
    assert rec.keys == ["dead_man_tripped", "flatten_incomplete:BTC"] and "tentative 3" in rec.alerts[1][1]
    w.flatten_result = "no_position"
    assert asyncio.run(w.check_dead_man(T0 + 304_000)) == "halted_flat"
    assert rec.keys == ["dead_man_tripped", "flatten_incomplete:BTC"]  # fermé : plus rien à dire


def test_a_clean_dead_man_trip_with_an_order_awaiting_verification_does_not_report_an_incomplete_flatten(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec)
    w._state.set("last_signal:BTC", str(T0))
    asyncio.run(w.check_dead_man(T0 + 301_000))
    asyncio.run(w.check_dead_man(T0 + 302_000))
    assert rec.keys == ["dead_man_tripped"]


def test_the_drawdown_circuit_breaker_alerts_when_it_halts_the_worker(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec, limits=RiskLimits(max_daily_drawdown_pct=D("5")))
    assert w._update_halt(D(1000), T0) is False
    assert w._update_halt(D(960), T0 + 1000) is False  # 4 % : sous le seuil
    assert rec.alerts == []
    assert w._update_halt(D(940), T0 + 2000) is True  # 6 %
    assert rec.keys == ["drawdown_halt"] and ">= 5%" in rec.alerts[0][1] and "1000 -> 940" in rec.alerts[0][1]
    assert w._update_halt(D(900), T0 + 3000) is True and rec.keys == ["drawdown_halt"]  # déjà verrouillé : pas de répétition


def test_a_failing_dead_man_loop_alerts_after_the_threshold(tmp_path: Path) -> None:
    rec = RecordingAlerter()
    w = _worker(tmp_path, rec, dead_man_check_interval_s=0.01)

    async def broken(now_ms: int | None = None) -> str:
        raise RuntimeError("surveillance cassée")

    w.check_dead_man = broken  # type: ignore[method-assign]  # double de test : méthode remplacée sur l'instance pour espionner ou simuler une panne

    async def run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(w._dead_man_loop(stop))
        await asyncio.sleep(0.3)
        stop.set()
        await task

    asyncio.run(run())
    assert rec.keys == ["streak:dead_man_loop"] and "ne protège plus" in rec.alerts[0][1]


def test_a_crashing_alerter_cannot_break_the_worker(tmp_path: Path) -> None:
    w = _worker(tmp_path, RecordingAlerter(fail=True))
    w._state.set("last_signal:BTC", str(T0))
    assert asyncio.run(w.check_dead_man(T0 + 301_000)) == "tripped"  # le déclenchement a bien eu lieu malgré l'alerteur cassé
    assert w.dead_man_tripped
    other = tmp_path / "other"
    other.mkdir()
    w2 = _worker(other, RecordingAlerter(fail=True))
    w2.script = [*(["error"] * 3), "ok"]
    asyncio.run(_execute(w2, 4))  # l'exécuteur survit


def test_without_an_alerter_nothing_changes(tmp_path: Path) -> None:
    w = _worker(tmp_path, None)
    w.script = ["error"] * 5
    asyncio.run(_execute(w, 5))
    w._state.set("last_signal:BTC", str(T0))
    assert asyncio.run(w.check_dead_man(T0 + 301_000)) == "tripped"


def test_an_end_to_end_alert_reaches_a_webhook_without_delaying_the_trip(tmp_path: Path, hook: _Hook) -> None:
    alerter = WebhookAlerter(hook.url, source="scripted-worker")
    w = _worker(tmp_path, alerter)
    w._state.set("last_signal:BTC", str(T0))
    started = time.monotonic()
    assert asyncio.run(w.check_dead_man(T0 + 301_000)) == "tripped"
    assert time.monotonic() - started < 0.5
    assert alerter.flush(5) and len(hook.bodies) == 1 and "HOMME MORT" in hook.bodies[0]["text"]


# --- Ligne de commande -----------------------------------------------------------------------


def test_a_worker_command_line_refuses_an_unsafe_webhook_url_without_echoing_it(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from workers.hyperliquid import worker as hl

    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", SECRET)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setenv("WEBHOOK_URL", f"http://evil.example/{TOKEN}")
    monkeypatch.setattr(hl, "SdkHyperliquidClient", lambda **_: pytest.fail("rien ne doit démarrer avec une alerte mal configurée"))
    with pytest.raises(SystemExit) as exit_info:
        hl.main(["--address", "tcp://127.0.0.1:1"])
    assert exit_info.value.code == 2
    assert TOKEN not in capsys.readouterr().err


def test_the_alerter_from_the_environment_is_handed_to_the_worker_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from workers.hyperliquid import worker as hl

    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def factory(config: Any, client: Any) -> None:
        seen["alerter"] = config.alerter
        raise _Stop

    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", SECRET)
    monkeypatch.setenv("HL_ACCOUNT_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setattr(hl, "SdkHyperliquidClient", lambda **_: object())
    monkeypatch.setattr(hl, "HyperliquidWorker", factory)
    monkeypatch.setattr(hl, "configure_logging", lambda *a, **k: None)
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with pytest.raises(_Stop):
        hl.main(["--address", "tcp://127.0.0.1:1", "--db", str(tmp_path / "hl.sqlite3")])
    assert seen["alerter"] is None  # pas de WEBHOOK_URL : aucune alerte
    monkeypatch.setenv("WEBHOOK_URL", "https://discord.com/api/webhooks/1/token")
    with pytest.raises(_Stop):
        hl.main(["--address", "tcp://127.0.0.1:1", "--db", str(tmp_path / "hl.sqlite3")])
    assert isinstance(seen["alerter"], WebhookAlerter)


# --- Rotation des journaux -------------------------------------------------------------------


@pytest.fixture
def clean_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def _our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_nothierly_worker_handler", False)]


def test_the_log_file_rotates_and_never_grows_past_its_cap(tmp_path: Path, clean_root_logger: None) -> None:
    path = configure_logging("w", log_dir=tmp_path / "logs", max_bytes=2_000, backup_count=3)
    assert path == tmp_path / "logs" / "w.log"
    log = logging.getLogger("rotation.test")
    for i in range(400):
        log.info("ligne %04d %s", i, "x" * 80)  # ~110 octets chacune : ~44 Ko écrits pour un plafond de ~8 Ko
    for handler in _our_handlers():
        handler.flush()
    files = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert files == ["w.log", "w.log.1", "w.log.2", "w.log.3"]  # le courant + 3 sauvegardes, jamais plus
    assert all(p.stat().st_size <= 2_000 + 200 for p in (tmp_path / "logs").iterdir())
    assert "ligne 0399" in path.read_text(encoding="utf-8")  # le plus récent est dans le fichier courant


def test_the_default_limits_are_ten_megabytes_and_five_backups(tmp_path: Path, clean_root_logger: None) -> None:
    configure_logging("w", log_dir=tmp_path)
    (handler,) = [h for h in _our_handlers() if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert handler.maxBytes == 10 * 1024 * 1024 and handler.backupCount == 5


def test_calling_it_twice_does_not_duplicate_handlers_and_the_env_var_is_the_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_root_logger: None) -> None:
    monkeypatch.setenv(ENV_LOG_DIR, str(tmp_path))
    assert configure_logging("a") == tmp_path / "a.log"
    assert configure_logging("a") == tmp_path / "a.log"
    assert len(_our_handlers()) == 2  # un flux + un fichier, pas quatre


def test_without_a_directory_only_stdout_is_used_and_an_unusable_directory_does_not_stop_the_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_root_logger: None) -> None:
    monkeypatch.delenv(ENV_LOG_DIR, raising=False)
    assert configure_logging("w") is None and len(_our_handlers()) == 1
    blocker = tmp_path / "afile"
    blocker.write_text("pas un dossier", encoding="utf-8")
    assert configure_logging("w", log_dir=blocker / "sub") is None  # mkdir impossible : stdout seul, pas d'exception
    assert len(_our_handlers()) == 1


# --- Pont : lanceur --------------------------------------------------------------------------


def _load_bridge_script() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_platform_bridge.py"
    spec = importlib.util.spec_from_file_location("run_platform_bridge_alerts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_platform_bridge_alerts"] = module
    spec.loader.exec_module(module)
    return module


def test_the_bridge_test_alert_flag_checks_the_webhook_end_to_end_without_redis_or_keys(monkeypatch: pytest.MonkeyPatch, hook: _Hook, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_bridge_script()
    monkeypatch.delenv("NOTHIERLY_ZMQ_SECRET", raising=False)
    monkeypatch.setenv("WEBHOOK_URL", hook.url)
    assert module.main(["--test-alert"]) == 0
    assert len(hook.bodies) == 1 and hook.bodies[0]["text"].startswith("[INFO][platform-bridge] test d'alerte")
    assert "envoyée" in capsys.readouterr().out


def test_the_bridge_test_alert_flag_fails_clearly_when_the_webhook_is_missing_unreachable_or_unsafe(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_bridge_script()
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    assert module.main(["--test-alert"]) == 1
    monkeypatch.setenv("WEBHOOK_URL", f"http://127.0.0.1:{_free_port()}/hook/{TOKEN}")
    assert module.main(["--test-alert"]) == 1
    monkeypatch.setenv("WEBHOOK_URL", f"http://evil.example/{TOKEN}")
    assert module.main(["--test-alert"]) == 2
    captured = capsys.readouterr()
    assert TOKEN not in captured.out + captured.err


def test_the_bridge_only_gets_an_alerter_when_it_really_publishes(monkeypatch: pytest.MonkeyPatch, hook: _Hook) -> None:
    module = _load_bridge_script()
    seen: list[Any] = []

    async def fake_amain(args: Any, alerter: Any = None) -> int:
        seen.append(alerter)
        return 0

    monkeypatch.setattr(module, "_amain", fake_amain)
    monkeypatch.setattr(module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setenv("WEBHOOK_URL", hook.url)
    monkeypatch.setenv("NOTHIERLY_ZMQ_SECRET", SECRET)
    monkeypatch.delenv("NOTHIERLY_ZMQ_ENCRYPTION", raising=False)
    assert module.main(["--dry-run"]) == 0 and seen == [None]  # observer sans effet : un dry-run ne réveille personne
    assert module.main([]) == 0 and isinstance(seen[1], WebhookAlerter)
    monkeypatch.delenv("WEBHOOK_URL")
    assert module.main([]) == 0 and seen[2] is None  # pas de webhook : pas d'alerte, avertissement au démarrage

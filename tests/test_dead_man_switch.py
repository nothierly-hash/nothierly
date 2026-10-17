"""Tests de l'homme mort (``workers/common/base.py``) : non-réception prolongée de signaux.

La mécanique est testée sur un worker minimal bâti sur le socle (positions simulées, sorties d'urgence scriptées) ;
les ordres réellement envoyés par Hyperliquid et Bybit sont testés dans leurs propres fichiers. Toutes les horloges
sont passées explicitement (``now_ms``) : aucun test ne dépend de l'heure réelle, sauf la boucle ZMQ de fin.
"""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SecureSignalPublisher, build_envelope
from workers.common.base import ExposureWorkerBase, ExposureWorkerConfig, clear_execution_halt
from workers.common.exposure import emergency_slippage_bps
from workers.common.state import ExecutionState
from workers.numerai.worker import NumeraiWorker

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
T0 = 1_760_000_000_000
TIMEOUT_S = 300.0
TIMEOUT_MS = 300_000
D = Decimal


class FakeWorker(ExposureWorkerBase):
    """Worker minimal : une « plateforme » avec des positions en mémoire et des sorties d'urgence observables."""

    venue: ClassVar[TargetVenue] = TargetVenue.HYPERLIQUID
    _tag: ClassVar[str] = "fake"
    supports_dead_man: ClassVar[bool] = True

    def __init__(self, config: ExposureWorkerConfig, state: ExecutionState | None = None) -> None:
        super().__init__(config, state)
        self.positions: dict[str, Decimal] = {}
        self.flatten_calls: list[tuple[str, int]] = []
        self.executed: list[str] = []
        self.fill = True  # un ordre d'urgence ferme-t-il réellement la position ?
        self.busy = False
        self.busy_during_flatten: list[bool] = []

    def map_symbol(self, symbol: str) -> str | None:
        return {"CRYPTO_BTC_USD": "BTC", "CRYPTO_ETH_USD": "ETH"}.get(symbol)

    async def execute_signal(self, signal: PlatformSignal, asset: str, *, now_ms: int | None = None) -> str:
        self.busy = True
        await asyncio.sleep(0.2)
        self.executed.append(asset)
        self.busy = False
        return "ok"

    async def emergency_flatten(self, asset: str, *, attempt: int, now_ms: int) -> str:
        self.flatten_calls.append((asset, attempt))
        self.busy_during_flatten.append(self.busy)
        if self.positions.get(asset, D(0)) == 0:
            return "no_position"
        if self._cfg.dry_run:
            return "dry_run"
        if self.fill:
            self.positions[asset] = D(0)
        return "sent"


class UnsupportedWorker(FakeWorker):
    supports_dead_man: ClassVar[bool] = False


def _cfg(tmp_path: Path, **overrides: Any) -> ExposureWorkerConfig:
    base: dict[str, Any] = dict(
        address="tcp://127.0.0.1:1", secret_key=SECRET, db_path=tmp_path / "dm.sqlite3", dry_run=False,
        dead_man_timeout_s=TIMEOUT_S, dead_man_startup_grace_s=0.0, dead_man_check_interval_s=0.1,
    )
    base.update(overrides)
    return ExposureWorkerConfig(**base)


def _worker(tmp_path: Path, **overrides: Any) -> FakeWorker:
    worker = FakeWorker(_cfg(tmp_path, **overrides))
    worker._started_ms = T0 - 10_000_000  # démarré bien avant les horloges fixes du test ; les tests de grâce le fixent eux-mêmes
    return worker


def _signal(symbol: str = "CRYPTO_BTC_USD", ts: int = T0, **overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.ALL, asset_class=AssetClass.CRYPTO, symbol=symbol, direction=SignalDirection.LONG,
        position_size_pct=5.0, order_type=OrderType.MARKET, timestamp_ms=ts,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _feed(worker: FakeWorker, signal: PlatformSignal) -> str:
    frames = [DEFAULT_TOPIC, json.dumps(build_envelope(signal, SECRET)).encode("utf-8")]
    return worker.handle_frames(frames, now_ms=signal.timestamp_ms + 100)  # reçu 100 ms après son émission


def _check(worker: FakeWorker, now_ms: int) -> str:
    return asyncio.run(worker.check_dead_man(now_ms))


# --- Capacités et configuration --------------------------------------------------------------


def test_the_switch_is_off_unless_configured_and_then_nothing_is_tracked(tmp_path: Path) -> None:
    w = _worker(tmp_path, dead_man_timeout_s=None)
    assert _feed(w, _signal()) == "queued"
    assert w.last_signal_ms("BTC") is None  # pas de surveillance, pas d'écriture
    assert _check(w, T0 + 10 * TIMEOUT_MS) == "disabled" and w.flatten_calls == []


def test_a_worker_without_the_capability_refuses_the_option_and_numerai_is_outside_the_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ne supporte pas l'homme mort"):
        UnsupportedWorker(_cfg(tmp_path))
    UnsupportedWorker(_cfg(tmp_path, dead_man_timeout_s=None, db_path=tmp_path / "ok.sqlite3"))  # sans l'option : accepté
    assert not issubclass(NumeraiWorker, ExposureWorkerBase)  # asynchrone, sans position : rien à fermer


@pytest.mark.parametrize("overrides", [
    {"dead_man_timeout_s": 0.0}, {"dead_man_timeout_s": -5.0}, {"dead_man_check_interval_s": 0.0},
    {"dead_man_startup_grace_s": -1.0}, {"emergency_slippage_bps": D(0)},
])
def test_invalid_dead_man_parameters_are_refused(tmp_path: Path, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _worker(tmp_path, **overrides)


def test_emergency_slippage_doubles_per_attempt_up_to_a_cap() -> None:
    assert [emergency_slippage_bps(D(300), a) for a in (1, 2, 3, 4, 5)] == [D(300), D(600), D(1200), D(2000), D(2000)]
    with pytest.raises(ValueError):
        emergency_slippage_bps(D(300), 0)


# --- Déclenchement ---------------------------------------------------------------------------


def test_a_valid_signal_arms_the_clock_and_each_new_one_pushes_the_deadline(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal(ts=T0))
    assert w.last_signal_ms("BTC") == T0 + 100
    assert _check(w, T0 + 100 + TIMEOUT_MS) == "ok"  # exactement N secondes : pas encore dépassé
    _feed(w, _signal(ts=T0 + 200_000, direction=SignalDirection.FLAT, position_size_pct=0.0))  # un FLAT prouve que le moteur vit
    assert _check(w, T0 + 100 + TIMEOUT_MS + 1) == "ok" and w.flatten_calls == []


def test_the_timeout_is_exceeded_strictly(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal())
    assert _check(w, T0 + 100 + TIMEOUT_MS) == "ok"
    assert _check(w, T0 + 100 + TIMEOUT_MS + 1) == "tripped"
    assert w.dead_man_tripped


def test_trip_closes_every_tracked_asset_not_only_the_silent_one_and_clears_the_queue(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions = {"BTC": D(1), "ETH": D(5)}
    _feed(w, _signal("CRYPTO_BTC_USD", ts=T0))
    _feed(w, _signal("CRYPTO_ETH_USD", ts=T0 + 250_000))  # ETH parle encore : seul BTC est en retard
    assert sorted(w._pending) == ["BTC", "ETH"]
    assert _check(w, T0 + 100 + TIMEOUT_MS + 1) == "tripped"
    assert sorted(w.flatten_calls) == [("BTC", 1), ("ETH", 1)]  # BTC (déclencheur) et ETH (fermé quand même : plus aucun signal ne sera traité)
    assert w.positions == {"BTC": D(0), "ETH": D(0)} and w._pending == {}
    assert _check(w, T0 + 100 + TIMEOUT_MS + 2) == "halted_flat"  # BTC est revérifié au cycle suivant : à plat
    assert w.dead_man_tripped  # reste HALTED même une fois à plat


def test_an_asset_that_is_already_flat_does_not_trip_and_is_not_reread_every_cycle(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    _feed(w, _signal())  # aucune position ouverte
    expired = T0 + 100 + TIMEOUT_MS + 1
    assert _check(w, expired) == "ok" and w.flatten_calls == [("BTC", 1)]
    assert not w.dead_man_tripped
    assert _check(w, expired + 5_000) == "ok" and len(w.flatten_calls) == 1  # pas de relecture avant max(N, 60) s
    assert _check(w, expired + TIMEOUT_MS + 1) == "ok" and len(w.flatten_calls) == 2


# --- HALTED ----------------------------------------------------------------------------------


def test_halted_rejects_new_signals_silently_before_any_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal())
    _check(w, T0 + 100 + TIMEOUT_MS + 1)
    assert caplog.records and caplog.records[0].levelname == "CRITICAL"  # le déclenchement, lui, est bruyant
    caplog.clear()
    fresh = _signal(ts=T0 + 10 * TIMEOUT_MS)
    with caplog.at_level("WARNING"):
        assert _feed(w, fresh) == "dead_man_halted"
        assert w.handle_frames([b"n'importe quoi"], now_ms=T0) == "dead_man_halted"  # même un message invalide : aucun traitement
    assert w._pending == {} and caplog.records == []  # silencieux : pas même un avertissement par message


def test_flatten_is_retried_each_cycle_with_escalating_slippage_until_the_position_is_read_flat(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    w.fill = False  # l'IOC ne se remplit pas
    _feed(w, _signal())
    now = T0 + 100 + TIMEOUT_MS + 1
    assert _check(w, now) == "tripped"
    for step in (1, 2, 3):
        assert _check(w, now + step * 5_000) == "halted_flattening"
    assert w.flatten_calls == [("BTC", 1), ("BTC", 2), ("BTC", 3), ("BTC", 4)]  # numéro de tentative croissant => écart croissant
    w.fill = True
    assert _check(w, now + 20_000) == "halted_flattening" and w.positions["BTC"] == 0  # cet ordre-ci ferme
    assert _check(w, now + 25_000) == "halted_flat"  # puis la lecture confirme : travail terminé
    assert _check(w, now + 30_000) == "halted_flat" and len(w.flatten_calls) == 6  # plus aucun ordre une fois à plat
    assert w.dead_man_tripped


def test_dry_run_flatten_is_terminal_but_still_halts(tmp_path: Path) -> None:
    w = _worker(tmp_path, dry_run=True)
    w.positions["BTC"] = D(1)
    _feed(w, _signal())
    assert _check(w, T0 + 100 + TIMEOUT_MS + 1) == "tripped"
    assert _check(w, T0 + 100 + TIMEOUT_MS + 2) == "halted_flat"  # « simulé » met fin à la tentative : pas de boucle infinie
    assert w.dead_man_tripped and w.positions["BTC"] == 1  # rien n'a été envoyé


# --- Persistance et redémarrage --------------------------------------------------------------


def test_the_tripped_state_and_the_pending_closures_survive_a_restart(tmp_path: Path) -> None:
    first = _worker(tmp_path)
    first.positions["BTC"] = D(1)
    first.fill = False
    _feed(first, _signal())
    now = T0 + 100 + TIMEOUT_MS + 1
    assert _check(first, now) == "tripped"
    first.state.close()
    reborn = _worker(tmp_path)
    reborn.positions["BTC"] = D(1)
    assert reborn.dead_man_tripped
    assert _feed(reborn, _signal(ts=now + 1_000)) == "dead_man_halted"
    assert _check(reborn, now + 10_000) == "halted_flattening"  # il continue de fermer ce qui restait
    assert reborn.flatten_calls == [("BTC", 1)]


def test_a_restarted_worker_gets_a_grace_period_then_closes_what_it_can_no_longer_vouch_for(tmp_path: Path) -> None:
    """Scénario redémarrage Windows : le dernier signal est très ancien, le worker revient, le moteur est muet."""
    before = _worker(tmp_path)
    _feed(before, _signal())  # dernier signal avant la coupure
    before.state.close()
    after = _worker(tmp_path, dead_man_startup_grace_s=30.0)
    after.positions["BTC"] = D(1)
    after._started_ms = T0 + 3_600_000  # le worker revient une heure plus tard
    assert _check(after, T0 + 3_600_000 + 29_999) == "ok" and after.flatten_calls == []  # grâce : le moteur peut encore parler
    assert _check(after, T0 + 3_600_000 + 30_001) == "tripped"
    assert after.positions["BTC"] == 0


def test_a_signal_during_the_grace_period_keeps_a_restarted_worker_armed_but_quiet(tmp_path: Path) -> None:
    before = _worker(tmp_path)
    _feed(before, _signal())
    before.state.close()
    after = _worker(tmp_path, dead_man_startup_grace_s=30.0)
    after.positions["BTC"] = D(1)
    after._started_ms = T0 + 3_600_000
    _feed(after, _signal(ts=T0 + 3_600_000 + 5_000))  # le moteur est revenu
    assert _check(after, T0 + 3_600_000 + 60_000) == "ok" and after.flatten_calls == []


# --- Réarmement ------------------------------------------------------------------------------


def test_clear_halt_from_another_process_rearms_a_running_worker_with_fresh_clocks(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal())
    _check(w, T0 + 100 + TIMEOUT_MS + 1)
    assert w.dead_man_tripped
    rearm_at = T0 + 2 * TIMEOUT_MS
    clear_execution_halt(ExecutionState(tmp_path / "dm.sqlite3"), now_ms=rearm_at)  # ce que fait `--clear-halt`
    assert not w.dead_man_tripped and w.last_signal_ms("BTC") == rearm_at
    assert _feed(w, _signal(ts=rearm_at + 1_000)) == "queued"  # les signaux repassent
    w.positions["BTC"] = D(1)
    assert _check(w, rearm_at + 1_100 + TIMEOUT_MS) == "ok"
    assert _check(w, rearm_at + 1_100 + TIMEOUT_MS + 1) == "tripped"  # moteur toujours muet : il redéclenche


def test_clear_halt_also_lifts_the_drawdown_lock(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.state.set("halted", "1")
    assert w.halted
    w.clear_halt()
    assert not w.halted and not w.dead_man_tripped


# --- Exécuteur et boucle ---------------------------------------------------------------------


def test_a_signal_queued_before_the_trip_is_never_executed(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal("CRYPTO_BTC_USD", ts=T0))
    _feed(w, _signal("CRYPTO_ETH_USD", ts=T0 + 250_000))
    _check(w, T0 + 100 + TIMEOUT_MS + 1)

    async def run_executor() -> None:
        stop, wake = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(w._executor(stop, wake))
        wake.set()
        await asyncio.sleep(0.3)
        stop.set()
        wake.set()
        await task

    asyncio.run(run_executor())
    assert w.executed == []


def test_an_emergency_order_never_overlaps_a_signal_being_executed(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    w.positions["BTC"] = D(1)
    _feed(w, _signal("CRYPTO_BTC_USD", ts=T0))

    async def scenario() -> None:
        stop, wake = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(w._executor(stop, wake))
        _feed(w, _signal("CRYPTO_ETH_USD", ts=T0 + 250_000))  # ETH est frais : son exécution (0,2 s) démarre
        w._pending.pop("BTC", None)
        wake.set()
        await asyncio.sleep(0.05)
        assert w.busy  # un signal est en cours d'exécution
        await w.check_dead_man(T0 + 100 + TIMEOUT_MS + 1)  # BTC en retard : déclenche pendant ce temps
        stop.set()
        wake.set()
        await task

    asyncio.run(scenario())
    assert w.busy_during_flatten and not any(w.busy_during_flatten)  # chaque sortie a attendu la fin de l'exécution
    assert w.executed == ["ETH"]


def test_real_zmq_loop_trips_when_the_engine_goes_silent(tmp_path: Path) -> None:
    async def scenario() -> tuple[bool, dict[str, Decimal]]:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
            worker = FakeWorker(_cfg(tmp_path, address=pub.endpoint, dead_man_timeout_s=1.0, poll_ms=20, dead_man_check_interval_s=0.1))
            worker.positions["BTC"] = D(1)
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            deadline = time.monotonic() + 6.0
            while worker.last_signal_ms("BTC") is None and time.monotonic() < deadline:
                pub.publish(_signal(ts=int(time.time() * 1000)))  # le moteur parle : l'horloge est armée
                await asyncio.sleep(0.05)
            assert worker.last_signal_ms("BTC") is not None and not worker.dead_man_tripped
            # le moteur se tait : plus aucune publication
            deadline = time.monotonic() + 8.0
            while not worker.dead_man_tripped and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return worker.dead_man_tripped, dict(worker.positions)

    tripped, positions = asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop)
    assert tripped and positions == {"BTC": D(0)}


# --- Délai de grâce : avertissement, pas oubli ; plancher du timeout en CLI ------------------------


def test_a_trip_suspended_by_the_startup_grace_is_warned_once_not_silently_swallowed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    before = _worker(tmp_path)
    _feed(before, _signal())
    before.state.close()
    after = _worker(tmp_path, dead_man_startup_grace_s=30.0)
    after.positions["BTC"] = D(1)
    after._started_ms = T0 + 3_600_000
    with caplog.at_level("WARNING"):
        assert _check(after, T0 + 3_600_000 + 5_000) == "ok"
        assert _check(after, T0 + 3_600_000 + 10_000) == "ok"
    warned = [r for r in caplog.records if "dead_man_startup_grace" in r.getMessage()]
    assert len(warned) == 1 and "asset=BTC" in warned[0].getMessage()  # une seule fois, avec l'actif
    assert after.flatten_calls == [] and not after.dead_man_tripped


def test_no_grace_warning_when_the_signal_is_recent_enough(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    w = _worker(tmp_path, dead_man_startup_grace_s=30.0)
    w._started_ms = T0
    w.positions["BTC"] = D(1)
    _feed(w, _signal())
    with caplog.at_level("WARNING"):
        assert _check(w, T0 + 5_000) == "ok"
    assert not [r for r in caplog.records if "dead_man_startup_grace" in r.getMessage()]


def test_the_grace_warning_does_not_extend_the_grace(tmp_path: Path) -> None:
    before = _worker(tmp_path)
    _feed(before, _signal())
    before.state.close()
    after = _worker(tmp_path, dead_man_startup_grace_s=30.0)
    after.positions["BTC"] = D(1)
    after._started_ms = T0 + 3_600_000
    assert _check(after, T0 + 3_600_000 + 1_000) == "ok"
    assert _check(after, T0 + 3_600_000 + 30_001) == "tripped"  # l'avertissement n'a rien repoussé


@pytest.mark.parametrize("timeout", [0.0, 30.0, 60.0, 300.0])
def test_acceptable_dead_man_timeouts_raise_no_error(timeout: float) -> None:
    from workers.common.base import dead_man_timeout_problem

    assert dead_man_timeout_problem(timeout) is None  # 0 = désactivé, jugé ailleurs


@pytest.mark.parametrize("timeout", [0.1, 10.0, 29.9])
def test_a_timeout_under_three_bridge_beats_is_refused(timeout: float) -> None:
    from workers.common.base import dead_man_timeout_problem

    problem = dead_man_timeout_problem(timeout)
    assert problem is not None and "30s" in problem


def test_a_timeout_between_30_and_60_seconds_is_accepted_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    from workers.common.base import dead_man_timeout_problem

    with caplog.at_level("WARNING"):
        assert dead_man_timeout_problem(45.0) is None
    assert any("dead_man_timeout_short" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert dead_man_timeout_problem(60.0) is None
    assert not caplog.records


@pytest.mark.parametrize("module", ["workers.bybit.worker", "workers.binance.worker", "workers.darwinex.worker", "workers.hyperliquid.worker"])
def test_every_worker_cli_refuses_a_dead_man_timeout_under_30_seconds(module: str, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib

    mod = importlib.import_module(module)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--dead-man-timeout", "10"])
    assert exc.value.code == 2
    assert "--dead-man-timeout" in capsys.readouterr().err

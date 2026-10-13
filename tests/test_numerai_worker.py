"""Tests du worker Numerai Signals (``workers/numerai``).

Règles Numerai vérifiées : signal strictement dans (0, 1), au moins 100 tickers, aucun doublon,
en-têtes ``numerai_ticker,signal``. Plus : sens du score selon la direction, calendrier, persistance
et rejeu après redémarrage, soumission une seule fois par jour, trajet ZMQ réel, et la frontière d'IP
(ni ``workers`` ni ``contracts`` n'importent ``core``/``src``).
"""

from __future__ import annotations

import ast
import asyncio
import csv
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from contracts.platform_signal import AssetClass, OrderType, PlatformSignal, SignalDirection, TargetVenue
from contracts.secure_signal_bus import DEFAULT_TOPIC, SecureSignalPublisher, WeakSecretError, build_envelope
from workers.numerai.client import DryRunClient, NumerapiClient
from workers.numerai.store import SignalStore
from workers.numerai.worker import (
    InsufficientSignals,
    NumeraiWorker,
    NumeraiWorkerConfig,
    SubmissionSchedule,
    signal_value,
    to_numerai_ticker,
)

SECRET = "test-secret-0123456789abcdef-0123456789abcdef"
OTHER_SECRET = "another-secret-9876543210fedcba-9876543210fedcba"
ADDRESS = "tcp://127.0.0.1:1"  # jamais utilisé hors des tests ZMQ

# Mardi 2026-10-13 : un jour de round. 13:15 UTC = heure de soumission par défaut.
TUE_1314 = datetime(2026, 10, 13, 13, 14, 59, tzinfo=timezone.utc)
TUE_1315 = datetime(2026, 10, 13, 13, 15, 0, tzinfo=timezone.utc)
TUE_1500 = datetime(2026, 10, 13, 15, 0, 0, tzinfo=timezone.utc)
MON_1400 = datetime(2026, 10, 12, 14, 0, 0, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _numerai_signal(symbol: str = "EQUITY_US_AAPL", **overrides: Any) -> PlatformSignal:
    base: dict[str, Any] = dict(
        target_venue=TargetVenue.NUMERAI,
        asset_class=AssetClass.EQUITY,
        symbol=symbol,
        direction=SignalDirection.LONG,
        confidence_score=0.9,
        position_size_pct=0.0,
        order_type=OrderType.PREDICTION_ONLY,
    )
    base.update(overrides)
    return PlatformSignal(**base)


def _frames(signal: PlatformSignal, secret: str = SECRET) -> list[bytes]:
    return [DEFAULT_TOPIC, json.dumps(build_envelope(signal, secret)).encode("utf-8")]


class FakeClient:
    def __init__(self, universe: frozenset[str] | None = None) -> None:
        self.universe = universe
        self.submitted: list[Path] = []
        self.fail_next = 0

    def ticker_universe(self) -> frozenset[str] | None:
        return self.universe

    def submit(self, csv_path: Path) -> str:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ConnectionError("numerai indisponible")
        self.submitted.append(csv_path)
        return f"sub-{len(self.submitted)}"


def _worker(tmp_path: Path, client: Any = None, **overrides: Any) -> NumeraiWorker:
    cfg: dict[str, Any] = dict(
        address=ADDRESS, secret_key=SECRET, db_path=tmp_path / "w.sqlite3", output_dir=tmp_path / "out",
        min_tickers=3, retry_after_s=60.0,
    )
    cfg.update(overrides)
    return NumeraiWorker(NumeraiWorkerConfig(**cfg), client=client)


def _fill(worker: NumeraiWorker, n: int, now: datetime = TUE_1314) -> None:
    for i in range(n):
        sig = _numerai_signal(f"EQUITY_US_T{i:03d}", timestamp_ms=_ms(now) - 1000 - i)
        assert worker.handle_frames(_frames(sig), now_ms=_ms(now)) == "stored"


# --- Score et ticker -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "confidence", "expected"),
    [
        (SignalDirection.LONG, 0.9, 0.95),   # 0.5 + 0.9/2
        (SignalDirection.SHORT, 0.9, 0.05),  # 0.5 - 0.9/2
        (SignalDirection.FLAT, 0.9, 0.5),    # aucune vue : neutre, quelle que soit la confiance
        (SignalDirection.LONG, 0.0, 0.5),
        (SignalDirection.LONG, 1.0, 0.99),   # 1.0 est interdit par Numerai : borné à 0.99
        (SignalDirection.SHORT, 1.0, 0.01),  # 0.0 est interdit : borné à 0.01
    ],
)
def test_signal_value_is_direction_aware_and_clipped(direction: SignalDirection, confidence: float, expected: float) -> None:
    assert signal_value(direction, confidence) == pytest.approx(expected)


def test_a_short_never_scores_above_a_long_at_the_same_confidence() -> None:
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert signal_value(SignalDirection.SHORT, c) <= 0.5 <= signal_value(SignalDirection.LONG, c)


def test_signal_value_stays_strictly_inside_zero_one() -> None:
    for d in SignalDirection:
        for c in (0.0, 0.001, 0.5, 0.999, 1.0):
            assert 0.0 < signal_value(d, c) < 1.0


def test_ticker_mapping() -> None:
    assert to_numerai_ticker("EQUITY_US_AAPL") == "AAPL US"
    assert to_numerai_ticker("EQUITY_US_AAPL", ticker_format="{symbol}") == "AAPL"
    assert to_numerai_ticker("EQUITY_US_BRK_B") is None  # 4 segments : ambigu, table explicite requise
    assert to_numerai_ticker("EQUITY_JP_7203") is None  # autre pays : code Numerai différent, jamais deviné
    assert to_numerai_ticker("EQUITY_US_BRK_B", overrides={"EQUITY_US_BRK_B": "BRK/B US"}) == "BRK/B US"
    assert to_numerai_ticker("EQUITY_JP_7203", overrides={"EQUITY_JP_7203": "7203 JT"}) == "7203 JT"


# --- Calendrier ------------------------------------------------------------------------------


def test_schedule_boundary_and_catch_up() -> None:
    s = SubmissionSchedule()
    assert not s.is_due(TUE_1314)
    assert s.is_due(TUE_1315)
    assert s.is_due(TUE_1500)  # redémarré plus tard dans la journée : toujours dû


def test_default_schedule_skips_monday_and_weekend() -> None:
    s = SubmissionSchedule()
    assert not s.is_due(MON_1400)
    assert not s.is_due(datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc))  # samedi
    assert not s.is_due(datetime(2026, 10, 18, 14, 0, tzinfo=timezone.utc))  # dimanche


def test_schedule_validates_and_requires_aware_datetimes() -> None:
    with pytest.raises(ValueError):
        SubmissionSchedule(hour_utc=24)
    with pytest.raises(ValueError):
        SubmissionSchedule(weekdays=frozenset())
    with pytest.raises(ValueError, match="fuseau"):
        SubmissionSchedule().is_due(datetime(2026, 10, 13, 14, 0))


# --- Réception et stockage -------------------------------------------------------------------


def test_valid_numerai_signal_is_stored_with_a_direction_aware_score(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    assert w.handle_frames(_frames(_numerai_signal(direction=SignalDirection.SHORT))) == "stored"
    (row,) = w.store.scores_since(0)
    assert row.numerai_ticker == "AAPL US" and row.signal == pytest.approx(0.05)


def test_all_with_equity_is_accepted_but_crypto_is_not(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    assert w.handle_frames(_frames(_numerai_signal(target_venue=TargetVenue.ALL))) == "stored"
    crypto = PlatformSignal(
        target_venue=TargetVenue.HYPERLIQUID, asset_class=AssetClass.CRYPTO, symbol="CRYPTO_BTC_USD",
        direction=SignalDirection.LONG, position_size_pct=5.0, order_type=OrderType.MARKET,
    )
    assert w.handle_frames(_frames(crypto)) == "wrong_venue"
    assert w.store.count_scores() == 1


def test_forged_and_malformed_messages_store_nothing(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    assert w.handle_frames(_frames(_numerai_signal(), secret=OTHER_SECRET)) == "bad_signature"
    assert w.handle_frames([DEFAULT_TOPIC, b"{{{"]) == "malformed_envelope"
    assert w.handle_frames([b"AUTRE.TOPIC", b"{}"]) == "bad_frames"
    assert w.handle_frames([DEFAULT_TOPIC]) == "bad_frames"
    assert w.store.count_scores() == 0


def test_a_newer_score_overwrites_and_an_older_one_is_ignored(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    t = _ms(TUE_1314)
    assert w.handle_frames(_frames(_numerai_signal(timestamp_ms=t - 2000, confidence_score=0.2)), now_ms=t) == "stored"
    assert w.handle_frames(_frames(_numerai_signal(timestamp_ms=t - 1000, confidence_score=0.8)), now_ms=t) == "stored"
    assert w.handle_frames(_frames(_numerai_signal(timestamp_ms=t - 1500, confidence_score=0.4)), now_ms=t) == "ignored_older"
    (row,) = w.store.scores_since(0)
    assert row.signal == pytest.approx(0.9)  # 0.5 + 0.8/2 : le plus récent a gagné


def test_replayed_message_is_rejected_and_stays_rejected_after_a_restart(tmp_path: Path) -> None:
    sig = _numerai_signal()
    first = _worker(tmp_path)
    assert first.handle_frames(_frames(sig)) == "stored"
    assert first.handle_frames(_frames(sig)) == "replay"
    first.store.close()
    reborn = _worker(tmp_path)  # même fichier SQLite : le ReplayGuard est rechargé
    assert reborn.handle_frames(_frames(sig)) == "replay"


def test_scores_survive_a_restart(tmp_path: Path) -> None:
    first = _worker(tmp_path)
    _fill(first, 4)
    first.store.close()
    assert _worker(tmp_path).store.count_scores() == 4


def test_unmappable_symbols_are_reported_not_guessed(tmp_path: Path) -> None:
    w = _worker(tmp_path)
    assert w.handle_frames(_frames(_numerai_signal("EQUITY_JP_7203"))) == "unmappable_symbol"
    assert w.store.count_scores() == 0


def test_a_weak_secret_stops_the_worker_at_startup(tmp_path: Path) -> None:
    with pytest.raises(WeakSecretError):
        _worker(tmp_path, secret_key="court")


# --- Fichier CSV -----------------------------------------------------------------------------


def test_csv_has_the_required_header_filename_and_values(tmp_path: Path) -> None:
    w = _worker(tmp_path, min_tickers=2)
    t = _ms(TUE_1314)
    w.handle_frames(_frames(_numerai_signal("EQUITY_US_MSFT", timestamp_ms=t - 1, confidence_score=1.0)), now_ms=t)
    w.handle_frames(_frames(_numerai_signal("EQUITY_US_AAPL", timestamp_ms=t - 2, direction=SignalDirection.SHORT)), now_ms=t)
    path = w.generate_submission_csv(TUE_1315)
    assert path.name == "submission_20261013.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["numerai_ticker", "signal"]
    assert rows[1:] == [["AAPL US", "0.050000"], ["MSFT US", "0.990000"]]  # triés ; 1.0 -> 0.99
    assert not (tmp_path / "out" / "submission_20261013.csv.tmp").exists()


def test_every_exported_signal_is_strictly_between_zero_and_one_with_no_duplicate_ticker(tmp_path: Path) -> None:
    w = _worker(tmp_path, min_tickers=100)
    for i in range(120):
        d = (SignalDirection.LONG, SignalDirection.SHORT, SignalDirection.FLAT)[i % 3]
        c = (0.0, 0.5, 1.0)[i % 3]
        w.handle_frames(_frames(_numerai_signal(f"EQUITY_US_T{i:03d}", direction=d, confidence_score=c, timestamp_ms=_ms(TUE_1314) - i - 1)), now_ms=_ms(TUE_1314))
    with w.generate_submission_csv(TUE_1315).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120
    assert all(0.0 < float(r["signal"]) < 1.0 for r in rows)
    assert len({r["numerai_ticker"] for r in rows}) == 120


def test_fewer_than_100_tickers_is_refused_by_default(tmp_path: Path) -> None:
    w = _worker(tmp_path, min_tickers=100)
    _fill(w, 99)
    with pytest.raises(InsufficientSignals, match="99"):
        w.generate_submission_csv(TUE_1315)
    _fill_one = _numerai_signal("EQUITY_US_EXTRA", timestamp_ms=_ms(TUE_1314) - 5000)
    w.handle_frames(_frames(_fill_one), now_ms=_ms(TUE_1314))
    assert w.generate_submission_csv(TUE_1315).exists()  # exactement 100 : accepté


def test_tickers_outside_the_universe_are_dropped_before_counting(tmp_path: Path) -> None:
    client = FakeClient(universe=frozenset({"T000 US", "T001 US", "T002 US"}))
    w = _worker(tmp_path, client=client, min_tickers=3)
    _fill(w, 5)  # T003 et T004 hors univers
    with w.generate_submission_csv(TUE_1315).open(encoding="utf-8", newline="") as f:
        assert [r["numerai_ticker"] for r in csv.DictReader(f)] == ["T000 US", "T001 US", "T002 US"]
    w2 = _worker(tmp_path, client=FakeClient(universe=frozenset({"T000 US"})), min_tickers=3, db_path=tmp_path / "w.sqlite3")
    with pytest.raises(InsufficientSignals):
        w2.generate_submission_csv(TUE_1315)


def test_scores_older_than_max_age_are_not_exported(tmp_path: Path) -> None:
    w = _worker(tmp_path, min_tickers=1)
    old = TUE_1314.replace(day=11)  # deux jours plus tôt
    w.handle_frames(_frames(_numerai_signal("EQUITY_US_OLD", timestamp_ms=_ms(old))), now_ms=_ms(old))
    with pytest.raises(InsufficientSignals):
        w.generate_submission_csv(TUE_1315)


# --- Soumission ------------------------------------------------------------------------------


def test_nothing_is_submitted_before_the_scheduled_time(tmp_path: Path) -> None:
    client = FakeClient()
    w = _worker(tmp_path, client=client)
    _fill(w, 3)
    assert w.submit_if_due(TUE_1314) == "not_due" and w.submit_if_due(MON_1400) == "not_due"
    assert client.submitted == []


def test_submits_exactly_once_per_day_even_across_a_restart(tmp_path: Path) -> None:
    client = FakeClient()
    w = _worker(tmp_path, client=client)
    _fill(w, 3)
    assert w.submit_if_due(TUE_1315) == "submitted"
    assert len(client.submitted) == 1 and client.submitted[0].name == "submission_20261013.csv"
    assert w.store.count_scores() == 0  # soumis : on repart de zéro pour demain
    assert w.submit_if_due(TUE_1500) == "already_submitted"
    w.store.close()
    reborn = _worker(tmp_path, client=client)
    assert reborn.submit_if_due(TUE_1500) == "already_submitted"
    assert len(client.submitted) == 1


def test_insufficient_signals_do_not_call_the_api_and_are_retried(tmp_path: Path) -> None:
    client = FakeClient()
    w = _worker(tmp_path, client=client, min_tickers=3)
    _fill(w, 2)
    assert w.submit_if_due(TUE_1315) == "insufficient"
    assert client.submitted == []
    assert w.submit_if_due(TUE_1315.replace(second=30)) == "waiting_retry"
    _fill_more = _numerai_signal("EQUITY_US_LATE", timestamp_ms=_ms(TUE_1315))
    w.handle_frames(_frames(_fill_more), now_ms=_ms(TUE_1315))
    assert w.submit_if_due(TUE_1500) == "submitted"  # assez de scores, délai de reprise écoulé


def test_api_failure_is_retried_after_the_delay_then_succeeds(tmp_path: Path) -> None:
    client = FakeClient()
    client.fail_next = 1
    w = _worker(tmp_path, client=client)
    _fill(w, 3)
    assert w.submit_if_due(TUE_1315) == "failed"
    assert w.store.count_scores() == 3  # rien n'est perdu après un échec
    assert w.submit_if_due(TUE_1315.replace(second=30)) == "waiting_retry"
    assert w.submit_if_due(TUE_1500) == "submitted"


def test_worker_gives_up_after_the_daily_attempt_limit(tmp_path: Path) -> None:
    client = FakeClient()
    client.fail_next = 99
    w = _worker(tmp_path, client=client, max_attempts_per_day=2)
    _fill(w, 3)
    assert w.submit_if_due(TUE_1315) == "failed"
    assert w.submit_if_due(TUE_1500) == "failed"
    assert w.submit_if_due(TUE_1500.replace(hour=16)) == "gave_up"


def test_dry_run_client_submits_nothing_but_completes_the_day(tmp_path: Path) -> None:
    w = _worker(tmp_path, client=DryRunClient())
    _fill(w, 3)
    assert w.submit_if_due(TUE_1315) == "dry_run"
    assert w.submit_if_due(TUE_1500) == "already_submitted"


def test_numerapi_client_wiring_uses_the_documented_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class FakeSignalsAPI:
        def __init__(self, public_id: str, secret_key: str) -> None:
            calls["init"] = (public_id, secret_key)

        def ticker_universe(self) -> list[str]:
            return ["AAPL US", "MSFT US"]

        def upload_predictions(self, file_path: str, model_id: str) -> str:
            calls["upload"] = (file_path, model_id)
            return "sub-42"

    monkeypatch.setitem(sys.modules, "numerapi", types.SimpleNamespace(SignalsAPI=FakeSignalsAPI))
    client = NumerapiClient("pub", "sec", "model-1")
    assert client.ticker_universe() == frozenset({"AAPL US", "MSFT US"})
    csv_path = tmp_path / "s.csv"
    assert client.submit(csv_path) == "sub-42"
    assert calls["init"] == ("pub", "sec") and calls["upload"] == (str(csv_path), "model-1")
    with pytest.raises(ValueError, match="NUMERAI_"):
        NumerapiClient.from_env({})


# --- Boucle ZMQ réelle -----------------------------------------------------------------------


def test_run_loop_receives_from_a_real_publisher_and_stops_cleanly(tmp_path: Path) -> None:
    async def scenario() -> int:
        with SecureSignalPublisher("tcp://127.0.0.1:*", SECRET) as pub:
            worker = _worker(tmp_path, address=pub.endpoint, poll_ms=20, schedule_check_interval_s=0.05)
            stop = asyncio.Event()
            task = asyncio.create_task(worker.run(stop))
            deadline = time.monotonic() + 8.0
            while worker.store.count_scores() == 0 and time.monotonic() < deadline:
                pub.publish(_numerai_signal(timestamp_ms=int(time.time() * 1000)))
                await asyncio.sleep(0.05)
            stop.set()
            await asyncio.wait_for(task, timeout=5.0)
            return worker.store.count_scores()

    assert asyncio.run(scenario(), loop_factory=asyncio.SelectorEventLoop) == 1  # rejeux ignorés par le ReplayGuard : un seul score, un seul signal_id


# --- Frontière d'IP --------------------------------------------------------------------------


@pytest.mark.parametrize("package", ["workers", "contracts"])
def test_workers_and_contracts_never_import_engine_code(package: str) -> None:
    root = Path(__file__).resolve().parents[1] / package
    offenders = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            )
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] in {"core", "src"}]
    assert offenders == []

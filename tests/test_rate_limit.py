"""Limites de débit REST (HTTP 429, 418) : pause exacte demandée par l'exchange, aucune requête pendant la pause, aucun ordre perdu ni doublé."""

from __future__ import annotations

import asyncio
import email.message
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.test_binance_worker import FakeBinance, FakeTransport, _client, _run, _signal, _worker
from tests.test_dead_man_switch import FakeWorker, _cfg, _feed
from tests.test_dead_man_switch import _signal as dm_signal
from tests.test_dead_man_switch import _worker as dm_worker
from workers.binance import client as binance_client
from workers.binance.client import BinanceOrderOutcome
from workers.bybit import client as bybit_client
from workers.common.base import RATE_LIMITED
from workers.common.ratelimit import (
    MAX_ESCALATED_DEFAULT_S,
    MAX_PAUSE_S,
    MIN_PAUSE_S,
    HttpRateLimited,
    RateLimitError,
    RateLimitGate,
    guard_transport,
    parse_retry_after,
    rate_limit_status,
)
from workers.hyperliquid.client import SdkHyperliquidClient

D = Decimal


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- Retry-After --------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [("120", 120.0), (" 7 ", 7.0), ("0", 0.0), ("1.5", 1.5), (None, None), ("", None), ("bientôt", None), ("nan", None), ("inf", None)])
def test_retry_after_seconds_and_garbage(raw: str | None, expected: float | None) -> None:
    assert parse_retry_after(raw) == expected


def test_retry_after_http_date_and_a_past_date() -> None:
    now = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Wed, 14 Oct 2026 12:05:00 GMT", now=now) == 300.0
    assert parse_retry_after("Wed, 14 Oct 2026 11:00:00 GMT", now=now) == MIN_PAUSE_S  # déjà passée : pause minimale, pas de martèlement
    assert parse_retry_after("-5") == MIN_PAUSE_S


# --- Porte --------------------------------------------------------------------------------------


def test_the_pause_is_exactly_the_retry_after_and_requests_are_refused_locally_until_then() -> None:
    clock = Clock()
    gate = RateLimitGate(clock=clock)
    assert gate.block(429, "30") == 30.0
    with pytest.raises(RateLimitError) as info:
        gate.check()
    assert info.value.local and info.value.status == 429 and info.value.retry_after_s == pytest.approx(30.0)
    clock.now += 29.9
    with pytest.raises(RateLimitError):
        gate.check()
    clock.now += 0.2
    gate.check()  # la pause est finie : on repart
    assert gate.remaining() == 0.0


def test_without_a_header_the_default_doubles_per_consecutive_429_and_is_capped() -> None:
    gate = RateLimitGate(clock=Clock())
    assert [gate.block(429) for _ in range(6)] == [60.0, 120.0, 240.0, 480.0, MAX_ESCALATED_DEFAULT_S, MAX_ESCALATED_DEFAULT_S]
    gate.note_success()
    assert gate.block(429) == 60.0  # une réussite repart de zéro
    assert RateLimitGate(clock=Clock()).block(418) == 300.0  # un bannissement attend plus longtemps par défaut
    explicit = RateLimitGate(clock=Clock())
    explicit.block(429)
    assert explicit.block(429, "10") == 10.0 and explicit.block(429) == 60.0  # un en-tête explicite réinitialise l'escalade


def test_a_header_of_zero_is_a_one_second_pause_and_a_ban_is_capped_at_three_days() -> None:
    gate = RateLimitGate(clock=Clock())
    assert gate.block(429, "0") == MIN_PAUSE_S
    assert gate.block(418, str(10 * 86_400)) == MAX_PAUSE_S


def test_a_shorter_block_never_shortens_a_longer_one() -> None:
    clock = Clock()
    gate = RateLimitGate(clock=clock)
    gate.block(418, "600")
    gate.block(429, "5")
    assert gate.remaining() == pytest.approx(600.0) and gate.last_status == 429


def test_waiting_sleeps_the_exact_pause_without_blocking_the_event_loop() -> None:
    gate = RateLimitGate()
    gate.block(429, "1")

    async def scenario() -> tuple[float, int]:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        waited = await gate.wait()
        task.cancel()
        return waited, ticks

    waited, ticks = asyncio.run(scenario())
    assert 0.9 <= waited <= 1.6 and ticks >= 20  # la boucle a continué de tourner pendant toute la pause
    assert gate.remaining() == 0.0


def test_a_stop_request_interrupts_the_pause() -> None:
    gate = RateLimitGate()
    gate.block(418, "3600")

    async def scenario() -> float:
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, stop.set)
        return await gate.wait(stop)

    t0 = time.monotonic()
    assert asyncio.run(scenario()) < 1.0 and time.monotonic() - t0 < 2.0  # l'arrêt du worker ne reste pas bloqué 1 h


def test_a_pause_extended_while_waiting_is_honoured() -> None:
    gate = RateLimitGate()
    gate.block(429, "1")

    async def scenario() -> float:
        asyncio.get_running_loop().call_later(0.3, lambda: gate.block(429, "1"))  # un 2e refus arrive pendant l'attente
        return await gate.wait()

    assert asyncio.run(scenario()) >= 1.2


# --- Transport gardé ----------------------------------------------------------------------------


class _Calls:
    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.count = 0

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        self.count += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[no-any-return]


def test_a_429_arms_the_pause_and_nothing_is_sent_until_it_ends() -> None:
    clock = Clock()
    gate = RateLimitGate(clock=clock)
    inner = _Calls(HttpRateLimited(429, "30"), (200, b"{}"))
    send = guard_transport(inner, gate)
    with pytest.raises(RateLimitError) as info:
        send("GET", "https://x", {}, None, 5.0)
    assert info.value.status == 429 and info.value.retry_after_s == 30.0 and not info.value.local and inner.count == 1
    for _ in range(5):  # le worker insiste : aucune requête ne part
        with pytest.raises(RateLimitError) as local:
            send("GET", "https://x", {}, None, 5.0)
        assert local.value.local
    assert inner.count == 1
    clock.now += 30.1
    assert send("GET", "https://x", {}, None, 5.0) == (200, b"{}") and inner.count == 2


def test_a_transport_that_returns_the_status_instead_of_raising_is_covered_too() -> None:
    gate = RateLimitGate(clock=Clock())
    send = guard_transport(_Calls((429, b"slow down"), (418, b"banned")), gate)
    with pytest.raises(RateLimitError):
        send("GET", "https://x", {}, None, 5.0)
    assert gate.last_status == 429 and gate.remaining() == 60.0


def test_other_http_errors_pass_through_untouched() -> None:
    gate = RateLimitGate(clock=Clock())
    send = guard_transport(_Calls((500, b"boom"), (404, b"nope")), gate)
    assert send("GET", "https://x", {}, None, 5.0) == (500, b"boom") and send("GET", "https://x", {}, None, 5.0) == (404, b"nope")
    assert gate.remaining() == 0.0 and gate.blocks == 0


def test_rate_limit_error_is_an_oserror_so_existing_network_handlers_still_catch_it() -> None:
    assert issubclass(RateLimitError, OSError)


# --- Transports urllib réels (en-têtes) ---------------------------------------------------------


def _http_error(code: int, retry_after: str | None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x", code, "msg", headers, None)


@pytest.mark.parametrize("module", [binance_client, bybit_client])
def test_the_real_transports_keep_the_retry_after_header_on_429(module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise _http_error(429, "42")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(HttpRateLimited) as info:
        module._urllib_transport("GET", "https://x", {}, None, 5.0)
    assert info.value.status == 429 and info.value.retry_after == "42"


def test_binance_418_is_a_rate_limit_and_a_missing_header_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def banned(*args: Any, **kwargs: Any) -> Any:
        raise _http_error(418, None)

    monkeypatch.setattr(urllib.request, "urlopen", banned)
    with pytest.raises(HttpRateLimited) as info:
        binance_client._urllib_transport("GET", "https://x", {}, None, 5.0)
    assert info.value.status == 418 and info.value.retry_after is None


def test_a_404_is_still_returned_as_a_status_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    class Body:
        def read(self) -> bytes:
            return b"absent"

    err = _http_error(404, None)
    setattr(err, "read", Body().read)

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise err

    monkeypatch.setattr(urllib.request, "urlopen", missing)
    assert binance_client._urllib_transport("GET", "https://x", {}, None, 5.0) == (404, b"absent")


# --- Clients ------------------------------------------------------------------------------------


def test_the_binance_client_stops_calling_during_a_ban_and_resumes_after() -> None:
    clock = Clock()
    transport = FakeTransport({("GET", "/fapi/v1/exchangeInfo"): [HttpRateLimited(418, "120"), (200, {"symbols": []})]})
    client = _client(transport, rate_gate=RateLimitGate(clock=clock))
    with pytest.raises(RateLimitError) as info:
        client.instruments()
    assert info.value.status == 418 and info.value.retry_after_s == 120.0
    for _ in range(3):
        with pytest.raises(RateLimitError):
            client.instruments()
    assert len(transport.calls) == 1  # un seul appel est parti : les suivants sont refusés localement
    clock.now += 121
    client.instruments()
    assert len(transport.calls) == 2


def test_the_hyperliquid_client_translates_the_sdk_429_and_reraises_everything_else() -> None:
    class ClientError(Exception):
        def __init__(self, status_code: int, header: Mapping[str, str]) -> None:
            super().__init__(str(status_code))
            self.status_code, self.header = status_code, header

    clock = Clock()
    client = object.__new__(SdkHyperliquidClient)
    client.rate_gate = RateLimitGate(clock=clock)
    calls = {"n": 0}

    def limited() -> None:
        calls["n"] += 1
        raise ClientError(429, {"Retry-After": "15"})

    with pytest.raises(RateLimitError) as info:
        client._sdk(limited)
    assert info.value.status == 429 and info.value.retry_after_s == 15.0
    with pytest.raises(RateLimitError):
        client._sdk(limited)
    assert calls["n"] == 1  # le SDK n'est pas rappelé pendant la pause

    clock.now += 16
    with pytest.raises(ClientError):
        client._sdk(lambda: (_ for _ in ()).throw(ClientError(400, {})))
    assert client._sdk(lambda x: x + 1, 1) == 2


def test_sdk_errors_are_recognised_by_duck_typing() -> None:
    class E(Exception):
        status_code = 429
        header = {"retry-after": "9"}

    assert rate_limit_status(E()) == (429, "9")
    assert rate_limit_status(ValueError("x")) is None

    class Other(Exception):
        status_code = 500

    assert rate_limit_status(Other()) is None


# --- Workers : statut précis, signal conservé ---------------------------------------------------


class LimitedBinance(FakeBinance):
    def __init__(self) -> None:
        super().__init__()
        self.limit_read = False
        self.limit_place = False

    def snapshot(self) -> Any:
        if self.limit_read:
            raise RateLimitError(429, 30.0)
        return super().snapshot()

    def place_order(self, request: Any) -> BinanceOrderOutcome:
        if self.limit_place:
            raise RateLimitError(418, 120.0)
        return super().place_order(request)


def test_a_rate_limited_read_is_a_precise_status_not_a_generic_read_error(tmp_path: Path) -> None:
    client = LimitedBinance()
    client.limit_read = True
    worker = _worker(tmp_path, client)
    assert _run(worker, _signal()) == RATE_LIMITED


def test_a_rate_limited_order_is_reported_as_not_sent_and_places_nothing(tmp_path: Path) -> None:
    client = LimitedBinance()
    client.limit_place = True
    worker = _worker(tmp_path, client)
    assert _run(worker, _signal()) == RATE_LIMITED and client.orders == []


def test_a_rate_limited_emergency_close_is_retried_without_counting_an_attempt(tmp_path: Path) -> None:
    client = LimitedBinance()
    client.positions["BTCUSDT"] = D("0.1")
    client.limit_place = True
    worker = _worker(tmp_path, client, dead_man_timeout_s=60.0)
    assert asyncio.run(worker.emergency_flatten("BTCUSDT", attempt=1, now_ms=T0)) == RATE_LIMITED


T0 = 1_760_000_000_000


class GatedWorker(FakeWorker):
    """Un worker dont l'exchange refuse la 1re requête (429, pause de 1 s) puis accepte."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attempts: list[float] = []

    async def execute_signal(self, signal: Any, asset: str, *, now_ms: int | None = None) -> str:
        gate = self._cfg.rate_gate
        assert gate is not None
        self.attempts.append(time.monotonic())
        if len(self.attempts) == 1:
            gate.block(429, "1")
            return RATE_LIMITED
        self.executed.append(asset)
        return "ok"


def _gated(tmp_path: Path, gate: RateLimitGate, **over: Any) -> GatedWorker:
    worker = GatedWorker(_cfg(tmp_path, rate_gate=gate, dead_man_timeout_s=None, **over))
    return worker


def test_the_executor_sleeps_the_exact_pause_then_retries_the_same_signal_once(tmp_path: Path) -> None:
    gate = RateLimitGate()
    worker = _gated(tmp_path, gate)
    assert _feed(worker, dm_signal(ts=int(time.time() * 1000) - 100)) == "queued"

    async def scenario() -> None:
        stop, wake = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(worker._executor(stop, wake))
        wake.set()
        deadline = time.monotonic() + 5
        while not worker.executed and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        stop.set()
        wake.set()
        await asyncio.wait_for(task, 3.0)

    asyncio.run(scenario())
    first, second = worker.attempts
    assert worker.executed == ["BTC"] and len(worker.attempts) == 2
    assert 0.9 <= second - first <= 2.0  # ni avant la fin de la pause, ni beaucoup après ; pas de boucle serrée


def test_a_newer_signal_replaces_the_one_waiting_for_the_pause(tmp_path: Path) -> None:
    gate = RateLimitGate()
    worker = _gated(tmp_path, gate)
    now = int(time.time() * 1000)
    _feed(worker, dm_signal(ts=now - 200, position_size_pct=5.0))
    seen: list[float] = []

    class Spy(GatedWorker):
        async def execute_signal(self, signal: Any, asset: str, *, now_ms: int | None = None) -> str:
            seen.append(signal.position_size_pct)
            return await super().execute_signal(signal, asset, now_ms=now_ms)

    spy = Spy(_cfg(tmp_path, rate_gate=gate, dead_man_timeout_s=None, db_path=tmp_path / "spy.sqlite3"))
    _feed(spy, dm_signal(ts=now - 200, position_size_pct=5.0))

    async def scenario() -> None:
        stop, wake = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(spy._executor(stop, wake))
        wake.set()
        await asyncio.sleep(0.3)  # pendant la pause, le moteur envoie une cible plus récente
        _feed(spy, dm_signal(ts=now - 100, position_size_pct=9.0))
        wake.set()
        deadline = time.monotonic() + 5
        while not spy.executed and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        stop.set()
        wake.set()
        await asyncio.wait_for(task, 3.0)

    asyncio.run(scenario())
    assert seen == [5.0, 9.0]  # la reprise exécute la cible la plus récente, pas l'ancienne


def test_a_418_ban_alerts_immediately(tmp_path: Path) -> None:
    class Rec:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def alert(self, text: str, *, key: str, severity: Any = None) -> bool:
            self.keys.append(key)
            return True

    rec = Rec()
    gate = RateLimitGate()

    class Banned(GatedWorker):
        async def execute_signal(self, signal: Any, asset: str, *, now_ms: int | None = None) -> str:
            self.attempts.append(time.monotonic())
            if len(self.attempts) == 1:
                gate.block(418, "1")
                return RATE_LIMITED
            self.executed.append(asset)
            return "ok"

    worker = Banned(_cfg(tmp_path, rate_gate=gate, dead_man_timeout_s=None, alerter=rec))
    _feed(worker, dm_signal(ts=int(time.time() * 1000) - 100))

    async def scenario() -> None:
        stop, wake = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(worker._executor(stop, wake))
        wake.set()
        deadline = time.monotonic() + 5
        while not worker.executed and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        stop.set()
        wake.set()
        await asyncio.wait_for(task, 3.0)

    asyncio.run(scenario())
    assert "rate_limit_ban" in rec.keys


def test_the_dead_man_flatten_does_not_burn_attempts_while_rate_limited(tmp_path: Path) -> None:
    worker = dm_worker(tmp_path)
    worker.positions["BTC"] = D(1)
    _feed(worker, dm_signal())
    calls: list[int] = []

    async def limited(asset: str, *, attempt: int, now_ms: int) -> str:
        calls.append(attempt)
        return RATE_LIMITED

    worker.emergency_flatten = limited  # type: ignore[method-assign]
    now = worker.last_signal_ms("BTC")
    assert now is not None
    later = now + 400_000
    assert asyncio.run(worker.check_dead_man(later)) == "tripped"
    for step in range(1, 6):
        asyncio.run(worker.check_dead_man(later + step * 1000))
    assert set(calls) == {1}  # chaque cycle retente avec la MÊME tentative : l'écart de prix ne s'élargit pas pour des requêtes jamais parties
    assert worker._flatten_attempts.get("BTC", 0) == 0

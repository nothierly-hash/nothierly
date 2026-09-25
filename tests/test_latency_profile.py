"""Tests for scripts/latency_profile.py and scripts/analyze_t2t.py.

The run loop is exercised over a real PUB/SUB pair on ephemeral loopback ports, with fake pipeline stages so
the outcome is deterministic. The real stages (agents, fakeredis) and the measured figures are recorded in
TESTING.md; a wall-clock latency is never asserted here.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import zmq
import zmq.asyncio

from core.schemas import MarketMetrics, MessagePayload


def _load(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lp = _load("latency_profile_under_test", "latency_profile.py")
t2t = _load("analyze_t2t_under_test", "analyze_t2t.py")


# -- percentiles -------------------------------------------------------------------------------


def test_nearest_rank_returns_observed_values() -> None:
    values = list(range(1, 11))

    assert t2t.nearest_rank(values, 50) == 5
    assert t2t.nearest_rank(values, 90) == 9
    assert t2t.nearest_rank(values, 99) == 10
    assert t2t.nearest_rank(values, 99.9) == 10
    assert t2t.nearest_rank([7], 50) == 7 and t2t.nearest_rank([7], 99.9) == 7


def test_the_rank_is_exact_where_float_arithmetic_rounds_up() -> None:
    """99.9 / 100 * 1000 is 999.0000000000001 in floats: a float rank would pick the 1000th value, not the 999th."""
    values = list(range(1, 1001))

    assert t2t.nearest_rank(values, 99.9) == 999
    assert t2t.nearest_rank(values, 99) == 990
    assert t2t.nearest_rank(list(range(1, 10_001)), 99.9) == 9990


def test_nearest_rank_rejects_empty_input_and_bad_percentiles() -> None:
    with pytest.raises(ValueError):
        t2t.nearest_rank([], 50)
    for bad in (0, -1, 100.1):
        with pytest.raises(ValueError):
            t2t.nearest_rank([1, 2, 3], bad)


def test_samples_above_counts_the_observations_a_percentile_rests_on() -> None:
    assert t2t.samples_above(1000, 99.9) == 2
    assert t2t.samples_above(62, 99.9) == 1
    assert t2t.samples_above(20_000, 99.9) == 21


def test_the_table_flags_a_p99_9_that_rests_on_a_handful_of_samples() -> None:
    few = "\n".join(t2t.percentile_table("few", [1000 * i for i in range(1, 63)]))
    many = "\n".join(t2t.percentile_table("many", [1000 * i for i in range(1, 20_001)]))

    assert "not a stable estimate" in few
    assert "not a stable estimate" not in many
    assert "| 50th | 10,000.0 |" in many  # 10,000 us: nearest rank 10,000 of 20,000 values 1000..20,000,000 ns


def test_an_empty_sample_says_so_instead_of_inventing_a_number() -> None:
    assert "no samples" in "\n".join(t2t.percentile_table("none", []))


# -- the preallocated log ----------------------------------------------------------------------


def _record(log: Any, index: int, *, signal_code: int = 0) -> None:
    log.record(signal_code=signal_code, gate_reached=8 if signal_code in (1, 2) else 0, risk_event=0, ingest_lag_ns=2_000_000 + index,
               decode_ns=10 + index, risk_check_ns=20, alpha_ns=30, gates_ns=400 if signal_code in (1, 2) else 0,
               tick_to_decision_ns=60 + index, tick_to_order_ns=500 if signal_code in (1, 2) else 0)


def test_the_log_never_grows_and_counts_what_did_not_fit() -> None:
    log = lp.LatencyLog(3)
    for index in range(5):
        _record(log, index)

    assert (log.count, log.dropped) == (3, 2)
    assert all(len(column) == 3 for column in log._columns.values())


def test_the_log_rejects_a_zero_capacity() -> None:
    with pytest.raises(ValueError):
        lp.LatencyLog(0)


def test_the_csv_has_the_documented_columns_and_only_the_recorded_rows(tmp_path: Path) -> None:
    log = lp.LatencyLog(10)
    _record(log, 0)
    _record(log, 1, signal_code=2)
    output = tmp_path / "t2t.csv"

    assert log.write_csv(output) == 2

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(lp.CSV_COLUMNS)
    assert [row["seq"] for row in rows] == ["0", "1"]
    assert rows[1]["signal"] == "2" and rows[1]["tick_to_order_ns"] == "500" and rows[1]["gates_ns"] == "400"


def test_the_timer_overhead_is_a_small_positive_number() -> None:
    assert 0 < lp.timer_overhead_ns(20_000) < 100_000


# -- the run loop over a real PUB/SUB pair -----------------------------------------------------


def _tick(price: float, *, age_seconds: float = 0.0) -> bytes:
    """A wire message stamped ``age_seconds`` in the past, as if it had waited that long in a queue."""
    stamp = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    metrics = MarketMetrics(symbol="BTC_USDT", spread=1.0, order_book_imbalance=0.0, mid_price=price, timestamp=stamp)
    payload = MessagePayload(timestamp=stamp, topic="MARKET_DATA_L2", sender_id="test", data=metrics.model_dump(mode="json"))
    return payload.serialize()


class _FakeStages:
    """Counts what the loop calls. The gates return ``gate_result`` for every entry."""

    def __init__(self, gate_result: int = 8, risk_event_on: frozenset[int] = frozenset()) -> None:
        self.gate_result = gate_result
        self.risk_event_on = risk_event_on
        self.risk_checks = 0
        self.gate_calls: list[int] = []
        self.strategies = 0
        self.resets = 0

    def stages(self) -> Any:
        async def risk_check(metrics: MarketMetrics) -> bool:
            self.risk_checks += 1
            return self.risk_checks in self.risk_event_on

        async def order_gates(metrics: MarketMetrics, direction: int) -> int:
            self.gate_calls.append(direction)
            return self.gate_result

        def new_strategy() -> Any:
            self.strategies += 1
            return lp.ZScoreStrategy(window=5, entry_z=1.5, exit_z=0.5)

        def on_reset() -> None:
            self.resets += 1

        return lp.Stages(risk_check=risk_check, order_gates=order_gates, new_strategy=new_strategy, on_reset=on_reset)


# After the warm-up: five ticks to fill the window, a spike (SELL), a return towards the mean, then a CLOSE.
_WARMUP = [100.0, 100.5, 99.5]
_MEASURED = [100.0, 101.0, 100.0, 101.0, 100.0, 110.0, 100.0, 101.0]


async def _run(fake: _FakeStages, frames: list[bytes], *, warmup: int, expected: int | None, idle_timeout: float = 0.5,
               capacity: int = 100, stop: asyncio.Event | None = None) -> tuple[Any, Any]:
    context = zmq.asyncio.Context()
    pub = context.socket(zmq.XPUB)  # an XPUB is a PUB that also reports when a subscription has arrived
    sub = context.socket(zmq.SUB)
    try:
        for sock in (pub, sub):
            sock.setsockopt(zmq.LINGER, 0)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, lp.TOPIC)
        assert await asyncio.wait_for(pub.recv(), timeout=5) == b"\x01" + lp.TOPIC

        async def feed() -> None:
            for frame in frames:
                await pub.send_multipart([lp.TOPIC, frame])
                await asyncio.sleep(0)

        feeder = asyncio.create_task(feed())
        log = lp.LatencyLog(capacity)
        summary = await lp.profile(
            sub, fake.stages(), log, warmup=warmup, expected=expected, idle_timeout=idle_timeout,
            stop=stop or asyncio.Event(), rss_every=0, poll_seconds=0.1,
        )
        await feeder
        return summary, log
    finally:
        sub.close()
        pub.close()
        context.term()


def test_warmup_is_processed_but_not_recorded_and_state_is_reset_after_it() -> None:
    fake = _FakeStages()
    frames = [_tick(price) for price in _WARMUP + _MEASURED]

    summary, log = asyncio.run(_run(fake, frames, warmup=len(_WARMUP), expected=len(_MEASURED)))

    assert summary.received == len(frames) and summary.recorded == len(_MEASURED) == log.count
    assert fake.risk_checks == len(frames)  # the warm-up ran the whole pipeline
    assert (fake.strategies, fake.resets) == (2, 1)  # a fresh strategy after the warm-up, so its prices do not leak in
    assert list(log._columns["signal"][: log.count]) == [0, 0, 0, 0, 0, 2, 0, 3]  # SELL on the spike, CLOSE on the return
    assert summary.stopped_because.startswith("recorded the expected")


def test_only_entry_signals_go_through_the_gates_and_get_a_tick_to_order_latency() -> None:
    fake = _FakeStages()

    _summary, log = asyncio.run(_run(fake, [_tick(p) for p in _WARMUP + _MEASURED], warmup=len(_WARMUP), expected=len(_MEASURED)))

    assert fake.gate_calls == [-1]  # the SELL only; the CLOSE and the quiet ticks never reach the gates
    signals = list(log._columns["signal"][: log.count])
    orders = list(log._columns["tick_to_order_ns"][: log.count])
    gates = list(log._columns["gates_ns"][: log.count])
    assert [i for i, s in enumerate(signals) if orders[i] > 0] == [5]
    assert gates[5] > 0 and sum(gates) == gates[5]
    assert all(log._columns["tick_to_decision_ns"][i] > 0 for i in range(log.count))
    assert orders[5] >= log._columns["tick_to_decision_ns"][5]  # T2 is taken after the alpha step, never before it


def test_a_rejected_entry_is_not_sent_and_does_not_open_a_position() -> None:
    fake = _FakeStages(gate_result=5)

    summary, log = asyncio.run(_run(fake, [_tick(p) for p in _WARMUP + _MEASURED], warmup=len(_WARMUP), expected=len(_MEASURED)))

    assert summary.orders == 0  # nothing reached the sink
    assert list(log._columns["signal"][: log.count]) == [0, 0, 0, 0, 0, 2, 0, 0]  # no position, so no CLOSE follows
    assert log._columns["gate_reached"][5] == 5


def test_kill_switch_events_are_counted_after_the_warmup_and_the_run_goes_on() -> None:
    fake = _FakeStages(risk_event_on=frozenset({2, 6, 7}))  # tick 2 is warm-up, ticks 6 and 7 are the 3rd and 4th measured

    summary, log = asyncio.run(_run(fake, [_tick(p) for p in _WARMUP + _MEASURED], warmup=len(_WARMUP), expected=len(_MEASURED)))

    assert summary.risk_events == 2 and log.count == len(_MEASURED)
    assert list(log._columns["risk_event"][: log.count]) == [0, 0, 1, 1, 0, 0, 0, 0]


def test_the_queue_delay_is_the_publisher_timestamp_to_t1() -> None:
    fake = _FakeStages()
    frames = [_tick(p) for p in _WARMUP] + [_tick(p, age_seconds=5.0) for p in _MEASURED]

    _summary, log = asyncio.run(_run(fake, frames, warmup=len(_WARMUP), expected=len(_MEASURED)))

    lags = list(log._columns["ingest_lag_ns"][: log.count])
    assert all(5_000_000_000 <= lag < 60_000_000_000 for lag in lags)


def test_an_exception_in_the_pipeline_keeps_the_counters_of_what_ran() -> None:
    class _Boom(_FakeStages):
        def stages(self) -> Any:
            stages = super().stages()
            inner = stages.risk_check

            async def risk_check(metrics: MarketMetrics) -> bool:
                if self.risk_checks == 6:
                    raise RuntimeError("boom")
                return await inner(metrics)

            stages.risk_check = risk_check
            return stages

    summary = lp.RunSummary()
    log = lp.LatencyLog(100)

    async def go() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            await _run_with(_Boom(), [_tick(p) for p in _WARMUP + _MEASURED], summary, log)

    asyncio.run(go())

    assert summary.received == 7 and log.count == 3  # 3 warm-up ticks, then 3 recorded, then the 7th raised


async def _run_with(fake: _FakeStages, frames: list[bytes], summary: Any, log: Any) -> None:
    context = zmq.asyncio.Context()
    pub = context.socket(zmq.XPUB)
    sub = context.socket(zmq.SUB)
    try:
        for sock in (pub, sub):
            sock.setsockopt(zmq.LINGER, 0)
        port = pub.bind_to_random_port("tcp://127.0.0.1")
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, lp.TOPIC)
        await asyncio.wait_for(pub.recv(), timeout=5)
        for frame in frames:
            await pub.send_multipart([lp.TOPIC, frame])
        await lp.profile(sub, fake.stages(), log, warmup=len(_WARMUP), expected=None, idle_timeout=0.5,
                         stop=asyncio.Event(), rss_every=0, poll_seconds=0.1, summary=summary)
    finally:
        sub.close()
        pub.close()
        context.term()


def test_the_entry_point_writes_the_csv_even_when_the_pipeline_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression that lost a whole benchmark run: an exception mid-run must not lose the recorded ticks."""
    import core.logger

    fake = _FakeStages()
    inner_stages = fake.stages()
    original_risk_check = inner_stages.risk_check
    seen = {"n": 0}

    async def risk_check(metrics: MarketMetrics) -> bool:
        seen["n"] += 1
        if seen["n"] == 8:
            raise RuntimeError("boom")
        return await original_risk_check(metrics)

    inner_stages.risk_check = risk_check
    monkeypatch.setattr(lp, "build_real_stages", lambda *a, **k: (inner_stages, lambda: None))
    monkeypatch.setattr(lp.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(core.logger, "configure_logging", lambda *a, **k: None)
    output = tmp_path / "t2t.csv"

    async def go() -> int:
        context = zmq.asyncio.Context()
        pub = context.socket(zmq.XPUB)
        try:
            pub.setsockopt(zmq.LINGER, 0)
            port = pub.bind_to_random_port("tcp://127.0.0.1")
            monkeypatch.setattr(sys, "argv", ["latency_profile.py", "--address", f"tcp://127.0.0.1:{port}", "--output", str(output),
                                              "--warmup", str(len(_WARMUP)), "--idle-timeout", "0.5"])
            runner = asyncio.create_task(lp._amain(lp._parse_args()))
            await asyncio.wait_for(pub.recv(), timeout=5)
            for price in _WARMUP + _MEASURED:
                await pub.send_multipart([lp.TOPIC, _tick(price)])
            return await asyncio.wait_for(runner, timeout=20)
        finally:
            pub.close()
            context.term()

    assert asyncio.run(go()) == 1  # non-zero: it crashed
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4  # ticks 4 to 7 were recorded; the 8th raised
    meta = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["stopped_because"].startswith("crashed") and meta["recorded"] == 4


def test_a_malformed_message_is_counted_and_skipped() -> None:
    fake = _FakeStages()
    frames = [_tick(p) for p in _WARMUP] + [b"not json"] + [_tick(p) for p in _MEASURED]

    summary, log = asyncio.run(_run(fake, frames, warmup=len(_WARMUP), expected=len(_MEASURED)))

    assert summary.malformed == 1 and log.count == len(_MEASURED)


def test_the_run_stops_when_the_feed_goes_idle_and_reports_it() -> None:
    fake = _FakeStages()

    summary, log = asyncio.run(_run(fake, [_tick(p) for p in _WARMUP + _MEASURED], warmup=len(_WARMUP), expected=None, idle_timeout=0.3))

    assert log.count == len(_MEASURED)
    assert "no message" in summary.stopped_because


def test_a_full_buffer_drops_rather_than_grows() -> None:
    fake = _FakeStages()

    summary, log = asyncio.run(_run(fake, [_tick(p) for p in _WARMUP + _MEASURED], warmup=len(_WARMUP), expected=None, idle_timeout=0.3, capacity=3))

    assert (log.count, summary.dropped) == (3, len(_MEASURED) - 3)


def test_the_stop_event_ends_the_run_and_the_partial_result_is_kept() -> None:
    fake = _FakeStages()
    stop = asyncio.Event()
    stop.set()

    summary, log = asyncio.run(_run(fake, [], warmup=0, expected=None, stop=stop))

    assert summary.stopped_because == "stopped by signal" and log.count == 0


# -- the analysis script -----------------------------------------------------------------------


def test_render_separates_entry_latency_from_every_tick_latency() -> None:
    log = lp.LatencyLog(10)
    for index in range(6):
        _record(log, index)
    _record(log, 6, signal_code=1)
    _record(log, 7, signal_code=3)
    rows = [{"seq": str(i), **{name: str(log._columns[name][i]) for name in lp.CSV_COLUMNS[1:]}} for i in range(log.count)]

    text = t2t.render(rows, None)

    assert "Recorded ticks: 8" in text
    assert "BUY 1" in text and "CLOSE 1" in text and "none 6" in text
    assert "Tick-to-order, entry signals that went through the Risk Gates (T1 to T2)** (n = 1)" in text
    assert "Entry signals refused before the gates, Kill Switch latched (T1 to T2, for reference)** (n = 0)" in text
    assert "Tick-to-decision, every tick (T1 to the end of the alpha step)** (n = 8)" in text


def test_the_cli_reads_the_csv_and_the_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    log = lp.LatencyLog(10)
    for index in range(4):
        _record(log, index)
    csv_path = tmp_path / "t2t_benchmark.csv"
    log.write_csv(csv_path)
    csv_path.with_suffix(".meta.json").write_text('{"python": "3.x", "recorded": 4, "rss_mb_samples": [{"received": 0, "rss_mb": 100.0}]}', encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["analyze_t2t.py", str(csv_path)])
    assert t2t.main() == 0
    out = capsys.readouterr().out
    assert "n = 4" in out and "Resident memory (MB)" in out and "0: 100.0" in out

    monkeypatch.setattr(sys, "argv", ["analyze_t2t.py", str(tmp_path / "missing.csv")])
    assert t2t.main() == 2


def test_entries_refused_before_the_gates_are_not_mixed_into_the_gate_latency() -> None:
    """A latched Kill Switch makes an entry cost one is_system_safe call; averaging it in would understate the gates."""
    def row(seq: int, gate_reached: str, order_ns: int) -> dict[str, str]:
        return {"seq": str(seq), "signal": "1", "gate_reached": gate_reached, "risk_event": "0", "ingest_lag_ns": "0", "decode_ns": "1",
                "risk_check_ns": "1", "alpha_ns": "1", "gates_ns": str(order_ns), "tick_to_decision_ns": "3", "tick_to_order_ns": str(order_ns)}

    rows = [row(0, "8", 4_000_000), row(1, "8", 6_000_000), row(2, "0", 700_000), row(3, "0", 800_000), row(4, "0", 900_000)]

    text = t2t.render(rows, None)

    assert "went through the Risk Gates (T1 to T2)** (n = 2)" in text
    assert "refused before the gates, Kill Switch latched (T1 to T2, for reference)** (n = 3)" in text
    assert "| 50th | 4,000.0 |" in text  # the gates' own median, untouched by the three cheap refusals

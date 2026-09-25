# Testing

## Running the suites

Main suite (1,413 tests):

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Last full run, 2026-09-25: `1413 passed in 199.52s`, and the process exits on its own with code 0.

`pyproject.toml` sets a per-test timeout of 120 s (thread method), so a hung test fails instead of blocking the run. That timeout does not cover a hang after the last test, which is what the ZeroMQ section below is about.

## Retail Gateway tests are locked by default

Retail Gateway tests are intentionally locked and skipped by default unless a cryptographic `RETAIL_JWT_SECRET_KEY` is explicitly injected in the environment, enforcing strict fail-closed deployment.

How it works:

- `retail_gateway/config.py` defines `RETAIL_JWT_SECRET_KEY` with no default. Importing the gateway without it raises a validation error, and that is the intended behavior for a deployment.
- `retail_gateway/tests/conftest.py` does not collect that directory's tests when the key is absent, and emits a warning saying so. With the key set, all 56 tests are collected.
- No key is stored in this repository and no test supplies one. Generate one for a run with `python -c "import secrets; print(secrets.token_urlsafe(64))"` and export it as `RETAIL_JWT_SECRET_KEY`, then run `pytest retail_gateway/tests`.

What this means for coverage: a default run does not execute the Retail Gateway tests (about 2,300 lines of code), so default-run pass counts exclude them. Last verified with an ephemeral key on 2026-09-25: 56 passed.

## ZeroMQ teardown: the pytest exit hang

Symptom: pytest reached 100% but the process never exited, so the summary could be lost.

Root cause: `BaseAgent.__init__` opens its own PUB and SUB sockets and its own `zmq.asyncio.Context`. Two agent fixtures returned the agent without closing any of them, so the leaked context was finalized during pytest's own `pytest_unconfigure` garbage collection, where `Context.__del__` -> `destroy()` -> `term()` blocked forever. Found by running every test file in its own process under a time limit, then every test id in the two files that failed to exit, then dumping thread stacks with `faulthandler`.

Leaking fixtures, now fixed (they `yield`, then close `_publisher`, `_subscriber` and `_zmq_context`, the pattern already used in `tests/test_spectral_denoiser_cwt.py`):

- `tests/test_audit_ledger.py::ledger` (one test, `test_gate8_sealed_success_logs_the_brief_own_literal_audit_line`, triggered it)
- `tests/test_shadow_crossing_engine.py::engine` (all 9 tests triggered it)

Rule for new tests: any fixture that builds a `BaseAgent` subclass must close its real publisher, subscriber and context at teardown.

`tests/conftest.py` additionally sets `setsockopt(zmq.LINGER, 0)` on every socket created in the session. This was tried first and did not fix the hang on its own, since the blocking was in `term()` on sockets that were never closed, not on unsent messages. It is kept as defense in depth. A variant that destroyed every live context at session end was tried and rejected: it made the process abort with a libzmq assertion (`ctx.cpp:185`, exit code 21).

Notes from the per-file sweep: `tests/test_evolutionary_tuner.py` is slow (about 50 s, CMA-ES) but exits normally, and `tests/test_data_lake_sync.py` and `tests/test_execution_agent_multi_asset.py` contain no test functions.

## Replay validation: EUR/USD, 2024-06-07 12:00-13:00 UTC

One hour of real EUR/USD ticks (12,194 rows, a bid-only Dukascopy-style export; the file is not stored in this repository) was replayed with `scripts/historical_ingester.py`. It is top-of-book data with no depth, so it is not an L2 replay. There is no ask column, so a spread of 0.0001 (1 pip) is an assumption applied to every tick, and the one-sided Volume column is not used. The hour has a 65-pip net decline and a 12:30 UTC volatility burst (tick rate rises from about 140 to 716 per minute, and the 12:30 minute alone moves 48.7 pips).

What was measured:

- **Bus delivery.** With a real `ZmqBroker` on TCP and a collector on its subscriber side, all 12,194 ticks were received, in file order, with 0 value differences against the CSV (one run, `--speed 0`, local machine). Delivery is not guaranteed by ZMQ PUB/SUB in general; this is what was observed.
- **Gate pipeline.** `scripts/replay_harness.py` on the same hour: all 12,194 ticks passed all 8 gates and ended as `MOCK_ORDER`, with no Kill Switch events. This does not show the risk stack blocking anything: the harness feeds synthetic, always-agreeing signals and untrained skeleton ML models (it logs this), and nothing in this hour came near a trigger.
- **Flash-crash detector.** The largest 60-second move in the hour is 0.49% (about 53 pips, at 12:30:21 UTC), measured with the detector's own definition (price versus the oldest price in the window). The default threshold, 5% in 60 s, is about 10 times that, so on this sample it never trips. `MarketAnomalyDetector` was driven offline with the file's own timestamps (its injectable `now`), not through the Kill Switch:

| Threshold (60 s) | Ticks in breach | First trip (UTC) |
|---|---|---|
| 5.0% (default) | 0 | never |
| 0.5% | 0 | never |
| 0.4% | 646 | 12:30:07 |
| 0.3% | 703 | 12:30:04 |
| 0.2% | 751 | 12:30:03 |

Configuration: the threshold is the `flash_crash_max_move_fraction` constructor argument of `XgboostSignalAgent` (default 0.05, `src/agents/supervisor/agent.py`). It is not exposed as a setting, and `scripts/run_backend.py` does not pass it, so changing it today means changing code.

What this does not establish: the right threshold. One hour cannot give a false-positive rate, and 0.3% to 0.4% may fire on ordinary volatile sessions. The detector tripped 3 to 7 seconds after the burst began, not sub-second. No memory usage was measured, and this run says nothing about leaks.

Reproduce:

```
python scripts/run_zmq_broker.py
python scripts/historical_ingester.py <file>.csv --timestamp-column Etc/UTC --bid-column Close --assumed-spread 0.0001 --speed 0
```

### Second dataset: EUR/CHF, 2015-01-15 12:00-13:00 UTC (the hour after the SNB shock)

Same format and same protocol: 5,924 bid-only ticks, replayed with `--symbol EUR_CHF`, again with an assumed 1-pip spread (the real spread is unknown; it does not affect the detector, which uses the mid price).

This file does not contain the Swiss National Bank shock. The SNB announced the end of the EUR/CHF floor earlier that morning, about 09:30 UTC. The data starts about two and a half hours later, at a price of about 1.03 (the old floor was 1.20), and its largest 60-second move is 0.33%. It is the aftermath, not the event.

- **Bus delivery.** 5,924 ticks published, 5,924 received, in file order, 0 value differences against the CSV (one run, `--speed 0`, local machine).
- **Flash-crash detector**, driven offline with the file's timestamps, as above:

| Threshold (60 s) | Ticks in breach | First trip (UTC) |
|---|---|---|
| 5.0% (default) | 0 | never |
| 0.5% | 0 | never |
| 0.4% | 0 | never |
| 0.3% | 15 | 12:21:38 |
| 0.2% | 191 | 12:01:13 |

Reading it: the default correctly does not trip, because there is no crash in this window. Set against the EUR/USD hour, no single low threshold fits both: 0.4% caught the EUR/USD burst and says nothing here, while 0.3% and 0.2% trip here on moves that are not crashes, which is the false-positive cost of a low threshold. Calibration has to be per instrument and needs far more data than two hours.

Not yet run: the shock window itself (about 09:00-10:30 UTC the same day). It is the check of whether the default 5% threshold trips on a real crash, and its result is unknown until it is run and recorded here.

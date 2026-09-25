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

The shock window itself was run later; see the seventh dataset below.

### Third dataset: GBP/USD, 2016-10-07 12:00-13:00 UTC

27,256 bid-only ticks, replayed with `--symbol GBP_USD` and an assumed 1-pip spread (0.0001). Price 1.23668 to 1.23849 (+0.15%), range 1.23117 to 1.239. The largest 60-second move is 0.425% (12:31:00 UTC), and the tick rate steps up at 12:30 UTC from about 410 to 590-650 per minute, consistent with a scheduled release. The sterling flash crash of that day happened earlier, in Asian trading hours, so it is not in this file.

- **Bus delivery.** 27,256 published, 27,256 received, in file order, 0 value differences against the CSV (one run, `--speed 0`, local machine). This is the largest replay so far.
- **Flash-crash detector** (driven offline with the file's timestamps): 5.0% and 0.5% never trip; 0.4% trips on 3 ticks from 12:31:00; 0.3% on 262 ticks from 12:30:29; 0.2% on 948 ticks from 12:13:32, which is 17 minutes before the 12:30 burst.

### Fourth dataset: USD/JPY, 2019-01-03 12:00-13:00 UTC

3,700 bid-only ticks, replayed with `--symbol USD_JPY`. For a yen pair 1 pip is 0.01, so the assumed spread was `--assumed-spread 0.01` (the real spread is unknown). Price 107.55 to 107.662 (+0.10%), largest 60-second move 0.085% (12:04:13 UTC). A quiet hour: the yen flash crash of that day happened earlier, in Asian trading hours, so it is not in this file.

- **Bus delivery.** 3,700 published, 3,700 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector:** none of 5.0%, 0.5%, 0.4%, 0.3% or 0.2% trips.

### Comparison of the tick-data hours

| Dataset | Ticks | Largest 60 s move | Ticks in breach at 0.4% | Ticks in breach at 0.2% |
|---|---|---|---|---|
| EUR/USD 2024-06-07 | 12,194 | 0.49% | 646 (from 12:30:07) | 751 (from 12:30:03) |
| EUR/CHF 2015-01-15 | 5,924 | 0.33% | 0 | 191 (from 12:01:13) |
| GBP/USD 2016-10-07 | 27,256 | 0.43% | 3 (from 12:31:00) | 948 (from 12:13:32) |
| USD/JPY 2019-01-03 | 3,700 | 0.085% | 0 | 0 |
| Copper 2026-09-08 | 2,615 | 0.26% | 0 | 26 (from 12:10:05) |

The default 5%/60 s threshold trips on none of them, as expected, since none is a crash. A low threshold trips in the volatile hours, including once well before the 12:30 burst (GBP/USD at 0.2%), and not at all in the quiet yen hour, so the right value depends on each instrument's volatility regime. This supports a per-instrument threshold, calibrated on far more data than a handful of hours, and does not establish any particular value.

The SNB shock window has since been run (seventh dataset below). Not yet run: the other true-crash windows, the sterling flash crash (2016-10-07, Asian trading hours) and the yen flash crash (2019-01-03, Asian trading hours). Their results are unknown until they are run and recorded here.

### Fifth dataset: AUD/JPY, 2020-03-09, 1-minute bars (full day)

The first stress day in the set, and a different kind of data: 1,440 one-minute bars (00:00-23:59 UTC, no gaps), bid side, not ticks. Only each bar's Close is replayed (Open, High and Low are not), and a bar's timestamp is its open time. The spread is an assumed 0.01, one yen pip (the real spread is unknown). The day opens at 68.724 and closes at 68.006 (-1.04%), with a high of 69.009 and a low of 64.356, a 7.23% high-to-low range. The largest close-to-close one-minute move is 2.415% (01:49 UTC), then 1.443% (01:51) and 1.134% (01:52); the largest single-bar high-low range is 2.552% (01:49).

- **Bus delivery.** 1,440 published, 1,440 received, in file order, 0 differences against the CSV closes.
- **Flash-crash detector** (driven offline on the closes, with the file's timestamps). A trip is attributable to the bar opening at that time, not to a second:

| Threshold (60 s) | Bars in breach | First trip (UTC) |
|---|---|---|
| 5.0% (default) | 0 | never |
| 2.0% | 1 | 01:49 |
| 1.0% | 3 | 01:49 |
| 0.5% | 10 | 01:42 |
| 0.4% | 13 | 01:42 |
| 0.3% | 21 | 01:42 |
| 0.2% | 43 | 01:41 |

- **Bound on the default.** Any two prices in one rolling 60-second window lie inside two adjacent one-minute bars, so their difference cannot exceed the span of that pair. The largest pair span is 4.62% (bars 01:49 and 01:50), so the default 5%/60 s could not have tripped on any tick path consistent with these bars, by a margin of 1.1x. This holds for the bid series only.

Reading it: even on a day with a 7% range the default does not trip, because its 60-second window only sees a near-instant collapse, and this move was spread over minutes. Thresholds of 1% to 2% catch the core of the episode (the 01:49 bar), and 0.5% flags it seven minutes earlier. This does not establish a value.

### Set aside: AUD/CAD, 2020-03-18, 4-hour bars

Six bars, too coarse for a 60-second detector, so nothing was replayed. The only thing it supports is the same bound: the largest adjacent-pair span is 2.41% (bars 13:00 and 17:00), so the default 5% could not have tripped, which says nothing about lower thresholds. One-minute bars or ticks are needed.

### Sixth dataset: copper (COPPER.CMD/USD), 2026-09-08 12:00-13:00 UTC

2,615 bid-only ticks, replayed with `--symbol COPPER`. A commodity has no pip, so the assumed spread of 0.001 is arbitrary; it does not affect the detector, which uses the mid price. Price 6.8258 to 6.8153 (-0.15%), a 0.62% range over the hour, and a largest 60-second move of 0.256% (12:10:54 UTC). A quiet hour with no crash. The tick rate is much lower than in the FX files: at most 101 ticks in a minute (12:10).

- **Bus delivery.** 2,615 published, 2,615 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector** (driven offline with the file's timestamps): 5.0%, 0.5%, 0.4% and 0.3% never trip; 0.2% trips on 26 ticks from 12:10:05.

A 60-second window holds few ticks in this feed. One quiet hour cannot show whether that makes the detector more sensitive to single-tick noise, and no threshold value follows from it.

### Set aside: empty extractions

Three downloads contain the header row and no data (34 bytes, 0 rows), so nothing was replayed: AUD/CZK and OJUICE.CMD/USX for 2020-03-08 12:00-13:00 UTC, and XPT.CMD/USD for 2020-03-08 from 00:00. 2020-03-08 was a Sunday, and these markets had no quotes at those times (the weekly reopen is later that evening UTC). Check the weekday and the instrument's trading hours before extracting.

### Seventh dataset: EUR/CHF, 2015-01-15 09:00-10:00 UTC (the SNB shock)

The first real crash in the set: 973 bid-only ticks, replayed with `--symbol EUR_CHF` and an assumed 1-pip spread (0.0001; the real spread in this window is unknown, and the detector uses the mid price). The feed thins out as liquidity vanishes: gaps of up to 102 s, and at most 170 ticks in a minute (09:30). The bid goes from 1.20094 at 09:00:07 to 1.02542 at 09:59:12 (-14.62%), with a low of 0.84989 at 09:49:14 (-29.23% from the first tick). The largest 60-second move is 8.67% (09:33:39 UTC, 1.14754 to 1.04808). The fall starts at about 09:30:47 to 09:30:49: the bid is -0.14% versus the last pre-shock bid (1.20094) at 09:30:49 and -0.43% one second later.

- **Bus delivery.** 973 published, 973 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector** (driven offline with the file's timestamps). "Fall already suffered" is the bid at the first trip versus the pre-shock bid of 1.20094:

| Threshold (60 s) | Ticks in breach | First trip (UTC) | Fall already suffered at first trip |
|---|---|---|---|
| 5.0% (default) | 74 | 09:31:09 | -5.60% |
| 2.0% | 198 | 09:30:56 | -2.00% |
| 1.0% | 279 | 09:30:52 | -1.07% |
| 0.5% | 361 | 09:30:50 | -0.54% |
| 0.4% | 391 | 09:30:50 | not measured |
| 0.3% | 398 | 09:30:49 | not measured |
| 0.2% | 421 | 09:30:49 | -0.20% |

Reading it: the default 5%/60 s does trip on a real crash of this speed, at 09:31:09, about 20 seconds after the move began, when the bid was already 5.60% below the pre-shock level. The price went on to fall about 29% from the first tick. A 0.5% threshold would have tripped one second after the onset, at -0.54%. Together with the AUD/JPY day (a 7% range spread over minutes, on which the default never trips), the default behaves as a catastrophe tripwire: it catches a collapse of this speed after a large part of the first leg has happened, and misses slower stress. Lower thresholds trip earlier, but they also trip in ordinary volatile hours (see the comparison above). No threshold value follows from this.

Limits: this is the detector alone, driven with the tick timestamps, not the Kill Switch chain, so supervisor cycle time, Redis and ZMQ latency add nothing here and were not measured. It is one event on a bid-only feed, and no orders, positions or losses were modeled, so it says nothing about capital preserved.

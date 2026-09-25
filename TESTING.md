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

27,256 bid-only ticks, replayed with `--symbol GBP_USD` and an assumed 1-pip spread (0.0001). Price 1.23668 to 1.23849 (+0.15%), range 1.23117 to 1.239. The largest 60-second move is 0.425% (12:31:00 UTC), and the tick rate steps up at 12:30 UTC from about 410 to 590-650 per minute, consistent with a scheduled release. The sterling flash crash happened about 13 hours earlier (around 23:07 UTC on 2016-10-06, see the twelfth dataset), so it is not in this file.

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
| GBP/USD 2016-10-06 | 18,441 | 0.11% | 0 | 0 |
| GBP/USD 2016-10-07 | 27,256 | 0.43% | 3 (from 12:31:00) | 948 (from 12:13:32) |
| USD/JPY 2019-01-03 | 3,700 | 0.085% | 0 | 0 |
| Copper 2026-09-08 | 2,615 | 0.26% | 0 | 26 (from 12:10:05) |
| BTC/USD 2026-09-19 (Saturday) | 1,088 | 0.083% | 0 | 0 |
| BTC/USD 2022-11-08 | 3,833 | 0.18% | 0 | 0 |

The default 5%/60 s threshold trips on none of them, as expected, since none is a crash. A low threshold trips in the volatile hours, including once well before the 12:30 burst (GBP/USD at 0.2%), and not at all in the quiet yen hour, so the right value depends on each instrument's volatility regime. This supports a per-instrument threshold, calibrated on far more data than a handful of hours, and does not establish any particular value.

The SNB shock window (seventh dataset) and the sterling flash crash (twelfth dataset) have since been run. Not yet run: the yen flash crash (inside the 4-hour bar opening 2019-01-02 22:00 UTC; see the set-aside note below), whose result is unknown until it is run and recorded here.

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

### Companion to the seventh dataset: EUR/CHF 2015-01-15, full day in 10-minute bars

144 bid-side bars of 10 minutes (00:00-23:50 UTC). They are not replayed, because 10-minute bars are too coarse for a 60-second detector; they are used to cross-check the tick file and to give the shock its context.

- **Cross-check.** Two independent extractions agree exactly: the low of the 09:40 bar (0.84989) is the tick file's minimum (09:49:14), and the close of the 09:50 bar (1.02542) is the tick file's last tick (09:59:12).
- **Before the shock.** From 00:00 to 09:20 (57 bars) the total bid range is 0.0133% and the largest single 10-minute range is 0.0092%. The price is pinned to the 1.20 floor.
- **The shock bars.** 09:30: low 0.99535, range 17.12% against the open. 09:40: low 0.84989, range 16.64%. 09:50: range 20.72%, close 1.02542. 10:00: range 6.11%. The day's low is 0.84989, -29.23% versus the pre-shock level, and the day closes at 0.99357.
- **Rest of the day.** Volatile but without a second collapse: after 10:10 the largest 10-minute range is 2.81% (21:50), and 9 bars exceed 1%. The largest span of two adjacent 10-minute bars is 3.25% (21:40 and 21:50), below 5%, so the default 5%/60 s could not have tripped on any 60-second window lying entirely after 10:10. With bars this coarse the bound says nothing about lower thresholds.
- **Regime note.** Before the shock the range is about 0.01% per 10 minutes, so a threshold calibrated on that regime alone would sit orders of magnitude below the volatility of the rest of the day. This is an observation about how far apart the two regimes are, not a recommendation.

### Eighth dataset: GBP/USD, 2016-10-06 12:00-13:00 UTC (about 11 hours before the sterling flash crash)

18,441 bid-only ticks, replayed with `--symbol GBP_USD` and an assumed 1-pip spread (0.0001). The bid goes from 1.26612 to 1.26362 (-0.20%), with a 0.27% range over the hour and a largest 60-second move of 0.106% (12:21:41 UTC). A calm hour. The flash crash itself (around 23:07 UTC the same day) is not in this file.

- **Bus delivery.** 18,441 published, 18,441 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector** (driven offline with the file's timestamps): none of 5.0%, 0.5%, 0.4%, 0.3% or 0.2% trips.

The same pair in the same hour on two consecutive days:

| GBP/USD, 12:00-13:00 UTC | Ticks | Largest 60 s move | Ticks in breach at 0.2% |
|---|---|---|---|
| 2016-10-06 (about 11 hours before the crash) | 18,441 | 0.106% | 0 |
| 2016-10-07 (about 13 hours after the crash) | 27,256 | 0.43% | 948 |

The largest 60-second move is about four times larger on the second day, and the 0.2% threshold goes from never tripping to tripping on 948 ticks. A threshold's behavior depends on the volatility regime of the moment as well as on the instrument. This is one pair of days, an example and not a measure of how regimes vary.

### Set aside: AUD/JPY, 2019-01-02, 4-hour bars

Six 4-hour bars, too coarse for a 60-second detector, so nothing was replayed. The first five are ordinary, with ranges of 0.47% to 0.82% against each bar's open. The last bar, which opens at 22:00 UTC on 2019-01-02 and covers the following four hours, contains the yen flash crash: open 76.044, low 70.406 (-7.41%), close 74.303, a rebound of +5.54% from the low. The largest adjacent-bar span is 8.80% (bars 18:00 and 22:00). Unlike the earlier bar files, that bound is above 5%, so it no longer excludes a trip of the default threshold; it does not show one either, because 4-hour bars do not say how fast the price fell. What the file does settle is the window to extract: ticks or 1-minute bars of AUD/JPY from 2019-01-02 22:00 UTC to 2019-01-03 02:00 UTC.

### Ninth dataset: BTC/USD, 2026-09-19 12:00-13:00 UTC (a Saturday)

1,088 bid-only ticks, replayed with `--symbol BTC_USD` and an arbitrary assumed spread of 1.0 (one dollar; the real spread is unknown, and the detector uses the mid price). The bid goes from 81,284.4 to 81,306.8 (+0.03%), with a 0.195% range over the hour and a largest 60-second move of 0.083% (12:23:50 UTC). The feed is sparse: at most 41 ticks in a minute and gaps of up to 60 s. Bitcoin trades around the clock, so this Saturday extraction has data where the FX and commodity ones for a Sunday were empty. A quiet hour.

- **Bus delivery.** 1,088 published, 1,088 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector** (driven offline with the file's timestamps): none of 5.0%, 0.5%, 0.4%, 0.3% or 0.2% trips.

One quiet weekend hour says nothing about a threshold for crypto, the asset class of the shadow-run instrument (BTC/USDT). No volatile crypto day is in the set at tick level yet (one is located in 1-minute bars in the companion to the tenth dataset).

### Tenth dataset: BTC/USD, 2022-11-08 12:00-13:00 UTC (the day of the FTX collapse, before the violent part)

3,833 bid-only ticks, replayed with `--symbol BTC_USD` and an arbitrary assumed spread of 1.0 (the real spread is unknown, and the detector uses the mid price). The bid goes from 19,670.1 to 19,692.3 (+0.11%), with a 0.473% range (low 19,616.8 at 12:40:03, high 19,709.6 at 12:14:44) and a largest 60-second move of 0.179% (12:39:19 UTC). The feed is sparse, with gaps of up to 84 s. The hour is calm: the violent part of that day is later and is not in this file.

- **Bus delivery.** 3,833 published, 3,833 received, in file order, 0 value differences against the CSV.
- **Flash-crash detector** (driven offline with the file's timestamps): none of 5.0%, 2.0%, 1.0%, 0.5%, 0.4%, 0.3% or 0.2% trips.

The 1-minute bars for the full day were obtained afterwards and locate the violent part (companion section below); the ticks for that window are still to be extracted.

### Companion to the tenth dataset: BTC/USD 2022-11-08, full day in 1-minute bars

1,391 bid-side one-minute bars (00:00-23:59 UTC; a minute without a trade has no bar, so 49 are missing). They are not replayed as ticks. Cross-check: the 12:00-12:59 bars give a low of 19,616.8 and a high of 19,709.6, identical to the extremes of the tick file above.

- **The day.** First open 20,560.7, last close 18,511.1 (-9.97%). High 20,635.8 at 16:42, low 17,151.1 at 19:32: a 20.32% high-to-low range, with the high first.
- **The violent part.** The most violent 60-minute window is 18:33 to 19:32 UTC, with a 10.58% range (18,966.1 to 17,151.1). Largest close-to-close one-minute moves: +4.39% (19:34, a rebound right after the low), -2.00% (19:35), -1.63% (19:32) and +1.52% (18:22). The largest single-bar range is 4.77% (19:34).
- **Flash-crash detector** (driven offline on the closes with the bar timestamps, which are bar open times):

| Threshold (60 s) | Bars in breach | First trip (UTC) |
|---|---|---|
| 5.0% (default) | 0 | never |
| 2.0% | 2 | 19:34 |
| 1.0% | 11 | 04:11 |
| 0.5% | 54 | 04:11 |
| 0.4% | 82 | 04:11 |
| 0.3% | 141 | 03:18 |
| 0.2% | 246 | 02:52 |

- **Bound on the default.** The largest span of two adjacent bars is 5.99% (bars 19:33 and 19:34), above 5%. Unlike the AUD/JPY day, the bars therefore do not exclude a trip of the default 5%/60 s threshold, and they do not show one either: only ticks can tell.

Reading it: thresholds of 0.2% to 1% trip hours before the crash, in ordinary trading, so on bitcoin they are noise; 2% catches the 19:34 bar. No threshold value follows from this.

That window was later replayed from Binance spot BTC/USDT trades (eleventh dataset below), a different source and data type from these Dukascopy bars.

### Eleventh dataset: BTC/USDT, 2022-11-08 (Binance spot trades), the FTX crash in ticks

Source: `BTCUSDT-trades-2022-11-08.zip` from data.binance.vision (SHA-256 verified against the site's checksum file). 14,530,601 trades for the whole UTC day, headerless, with millisecond timestamps. These are trade prints, not bid/ask quotes, on the spot BTC/USDT pair: the same event as the Dukascopy BTC/USD bars above, from a different venue and data type. The detector was fed the trade price plus an arbitrary 0.5 (immaterial for fractional moves) and the file was streamed, not loaded in memory.

The day: low 17,166.83 at 19:32:59, high 20,700.88 at 16:33:57, a 20.59% high-to-low range. The largest 60-second move of the day, and the largest inside 18:30-20:00 UTC, is **+5.451%** at 19:34:54.275 (17,329.16 to 18,273.78): a rebound after the low. The 5.99% bound derived from the Dukascopy bars was therefore not violated.

Flash-crash detector, driven offline with the trade timestamps, first trip inside 18:30-20:00 UTC (reference price 18,754.26 at 18:30:00; "fall already suffered" is the price at the trip versus that reference):

| Threshold (60 s) | First trip (UTC) | Direction | Fall already suffered | Trades in breach in the window |
|---|---|---|---|---|
| 5.0% (default) | 19:34:51.850 | rise | -3.13% | 2,258 |
| 3.0% | 19:34:21.761 | rise | -5.41% | 32,516 |
| 2.0% | 19:31:20.393 | fall | -6.29% | 62,224 |
| 1.0% | 18:30:17.444 | fall | -0.28% | 193,334 |
| 0.5% | 18:30:17.444 | fall | -0.28% | 621,072 |
| 0.3% | already in breach at 18:30:00 | rise | 0.00% | 990,220 |
| 0.2% | already in breach at 18:30:00 | rise | 0.00% | 1,216,550 |

Reading it: the default 5%/60 s does trip on this crash, but on the rebound about two minutes after the low, not on the fall; the fall, spread over more than an hour, never moved 5% within 60 seconds. The first threshold to trip on the fall is 2%, at 19:31:20, 1 minute 39 seconds before the low. Thresholds of 1% and below trip within the first minute of the window, which is noise on bitcoin: over the whole day, 2% trips on 81,980 trades starting 04:11:41 UTC and 0.2% on 4,665,093 (32% of all trades), in ordinary trading. No threshold value follows from this.

Limits: this is the detector alone, not the Kill Switch chain; one event; trade prints on a different venue than the bars; no orders, positions or losses were modeled.

### Simulated backtest: ADAUSD_230630, June 2023 (Binance bookTicker, real bid/ask)

Source: `ADAUSD_230630-bookTicker-2023-06.zip` from data.binance.vision (SHA-256 verified against the site's checksum file and against a checksum file supplied separately). 15,182,130 top-of-book updates (best bid and ask with quantities, millisecond times) from 2023-06-01 00:00:00.065 to 2023-06-30, top of book only and not depth. The mean spread is 0.000359 and the mid changes on 43.1% of updates.

Ingester on a real book: the first 50,000 rows, read from a zip with `--timestamp-unit ms`, the real ask and the real quantities, were published and received in full (50,000 of 50,000), with 0 differences in mid price, spread, imbalance and liquidity. The whole file was not replayed through the broker (about 50 minutes at the ~4,900 messages per second measured earlier).

**This is a simulation, not performance.** `scripts/zscore_backtest.py` (z-score of the mid over 100 updates, entry at +/-3.0, exit inside +/-0.5; 10,000 ADA per trade; 5 ms latency in data time; 2 USD per 10,000 ADA on each fill) and `scripts/generate_tear_sheet.py`. A toy strategy, no parameter tuned, one month, in-sample. ADAUSD_230630 is a coin-margined inverse contract, and the PnL is a linear approximation (price difference times ADA units), not the contract's true payoff.

- **Result.** 72,726 round trips: gross -255,762.50, commissions 290,904.00, net -546,666.50 USD. Per round trip, gross -3.52 and net -7.52. Winners after costs: 0.0% (0.1% before commissions). All 30 days are negative (worst: 2023-06-10, -62,634). Applied entry latency: median 14 ms, p90 79 ms.
- **Where the loss comes from.** The average gross loss per round trip (3.52) is close to the file's average full spread for this size (3.59 USD for 10,000 ADA). At the mid the strategy earns roughly nothing, and the loss is the spread plus the commission.
- **Return.** It depends entirely on an illustrative capital. With 1,000,000 the report shows -54.67% (maximum drawdown -54.67%; annualized Sharpe -35.2 over 30 daily observations, which means nothing for this toy). With 10,000 the report refuses to build, because equity reaches zero. Nothing here is evidence of an edge.
- **Alpha step timing.** Mean 31.0 microseconds per update over 15.2 million updates, worst single call 33 milliseconds.

### Raw Dukascopy ticks: the datafeed decoder and the export's clock

The public Dukascopy datafeed (`datafeed.dukascopy.com`) holds one LZMA-compressed `.bi5` file per pair and UTC hour (the month is 0-based in the URL). Each tick is 20 bytes, big-endian: a millisecond offset within the hour, the ask, the bid (integers in units of 1e-5 for GBP/USD) and two float32 volumes. A decoder was validated against a CSV exported from the Dukascopy website for the same hour (GBP/USD, 2016-10-07 12:00-13:00 UTC): all 27,256 exported bids equal the raw bids, in order.

The export's clock differs from the raw feed's: every exported timestamp is the raw timestamp truncated to the second plus one second (27,256 of 27,256 rows), and the export omits the raw ticks of the last second of the requested hour (7 ticks here, which would be stamped 13:00:00). Trip times measured on exported Dukascopy CSVs elsewhere in this document are therefore up to about one second later than the raw feed's. The server throttles (HTTP 503, connection resets), so requests need spacing and retries.

### Twelfth dataset: GBP/USD, 2016-10-06 23:00-24:00 UTC (the sterling flash crash, raw ticks)

24,005 raw bid/ask ticks with millisecond timestamps, from 23:00:00.639 to 23:59:56.867, with the real spread and not an assumed one. The spread has a median of 0.00028 and widens to 0.00399 at 23:11:38 (bid 1.20493, ask 1.20892), with 489 ticks above 20 pips. The bid goes from 1.26137 at the start (high 1.26147 at 23:01:34.940) to a low of 1.19198 at 23:09:02.737, a 5.83% high-to-low range with the high first, and ends at 1.23878. The largest 60-second move on the mid is a fall of 3.866% (23:07:32.212, 1.26058 to 1.21184); the largest rise is 1.778% (23:17:16). The tick rate rises from about 90-100 per minute (23:04-23:05) to 238-246 (23:06-23:07), then drops to 28 at 23:10, with a 39-second gap (23:10:12 to 23:10:52).

Flash-crash detector on the mid (offline, tick timestamps). Pre-crash reference: 1.26092, the last mid before 23:06:00. Every first trip is on a fall:

| Threshold (60 s) | Ticks in breach | First trip (UTC) | Price at first trip | Versus pre-crash |
|---|---|---|---|---|
| 5.0% (default) | 0 | never | | |
| 3.0% | 103 | 23:07:18.048 | 1.22259 | -3.04% |
| 2.0% | 140 | 23:07:15.569 | 1.23508 | -2.05% |
| 1.0% | 1,086 | 23:07:12.763 | 1.24811 | -1.02% |
| 0.5% | 2,635 | 23:07:08.251 | 1.25419 | -0.53% |
| 0.3% | 4,430 | 23:07:06.352 | 1.25704 | -0.31% |
| 0.2% | 7,276 | 23:07:05.161 | 1.25831 | -0.21% |

Reading it: this is the first real crash in this document on which the default 5%/60 s never trips. On this feed the fall peaked at 3.87% in 60 seconds even though the low was 5.83% below the high. Thresholds from 0.2% to 3% all trip within 13 seconds of the onset (about 23:07:05), the 3% one when the price was already 3.04% down. Compared with the SNB window (the default tripped about 20 seconds after the onset), the fitting threshold clearly depends on the instrument's crash speed. No value follows from two events.

The 2016-10-07 00:00-02:00 UTC hours were also decoded and are calm (largest 60-second move 0.34%, no threshold above 0.3% trips): the crash is at about 00:07 London time, which is 23:07 UTC on the 6th.

### Crash days from Binance spot trades

Four full UTC days from data.binance.vision (SHA-256 verified against the site's checksums), streamed. These are trade prints, not quotes, and the detector was fed the trade price.

| Day | Trades | First to last price | Low to high range | Largest 60 s fall | Largest 60 s rise |
|---|---|---|---|---|---|
| BTCUSDT 2020-03-12 | 1,770,812 | -39.51% | 80.64% | -9.82% (10:47:38) | +17.37% (10:49:37) |
| BTCUSDT 2020-03-13 | 2,309,830 | +16.22% | 57.45% (low first) | -7.84% (02:41:01) | +9.45% (02:31:43) |
| FTTUSDT 2022-11-08 | 2,742,536 | -75.09% | 780.48% | -21.62% (19:31:20) | +37.09% (19:39:34) |
| BTCUSDT 2021-05-19 | 6,331,062 | -14.38% | 45.28% | -7.45% (13:21:43) | +8.69% (12:56:17) |

The default 5%/60 s threshold, first trip and trades in breach (the "running high" is the highest price seen so far that day):

| Day | First trip (UTC) | Direction | Price versus the running high | Trades in breach |
|---|---|---|---|---|
| BTCUSDT 2020-03-12 | 10:46:05 | fall | -24.03% | 27,962 |
| BTCUSDT 2020-03-13 | 02:31:16 | rise | -11.06% | 16,060 |
| FTTUSDT 2022-11-08 | 02:48:49 | fall | -6.20% | 414,902 |
| BTCUSDT 2021-05-19 | 12:51:15 | rise | -15.54% | 18,831 |

Reading it: the default trips on all four days, but it is far from prompt. On 2020-03-12 it trips on a fall only when the price is already 24% below the day's high; on 2020-03-13 and 2021-05-19 its first trip is a rebound. On FTT it trips early, at -6.20%, about 17 hours before the low (2.51 at 19:32:00). On every day the largest 60-second fall exceeds 5%, so on these days the 5% scale fits crashes, unlike on the majors in FX. Thresholds of 0.2% to 1% trip within the first seconds or minutes of the day (the 0.2% one within the first 32 seconds on all four), which is noise on these assets. No threshold value follows.

Limits: this is the detector alone, not the Kill Switch chain; four spot days on one venue; no orders, positions or losses were modeled.

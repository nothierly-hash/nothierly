# Testing

## Running the suites

Main suite (1,398 tests collected as of 2026-09-25):

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

`pyproject.toml` sets a per-test timeout of 120 s (thread method), so a hung test fails instead of blocking the run.

Known issue: on a full `tests/` run, all 1,398 tests pass, but the pytest process then does not exit or print its summary (see "ZeroMQ teardown" below). Until that is resolved, run the full suite under an external time limit, and read results from the progress output, or run test files individually.

## Retail Gateway tests are locked by default

Retail Gateway tests are intentionally locked and skipped by default unless a cryptographic `RETAIL_JWT_SECRET_KEY` is explicitly injected in the environment, enforcing strict fail-closed deployment.

How it works:

- `retail_gateway/config.py` defines `RETAIL_JWT_SECRET_KEY` with no default. Importing the gateway without it raises a validation error, and that is the intended behavior for a deployment.
- `retail_gateway/tests/conftest.py` does not collect that directory's tests when the key is absent, and emits a warning saying so. With the key set, all 56 tests are collected.
- No key is stored in this repository and no test supplies one. Generate one for a run with `python -c "import secrets; print(secrets.token_urlsafe(64))"` and export it as `RETAIL_JWT_SECRET_KEY`, then run `pytest retail_gateway/tests`.

What this means for coverage: a default run does not execute the Retail Gateway tests (about 2,300 lines of code), so default-run pass counts exclude them. Last verified with an ephemeral key on 2026-09-25: 56 passed.

## ZeroMQ teardown

`tests/conftest.py` applies `setsockopt(zmq.LINGER, 0)` to every socket created during the test session. Production code is untouched (`core/zmq_bus.py` already closes its own sockets with `linger=0`).

Why: `BaseAgent` creates its own `zmq.asyncio.Context()`. A test that does not shut its agent down leaves a context with open sockets, and with ZeroMQ's default `LINGER = -1` terminating it can block. The symptom is pytest reaching 100% and then never exiting or printing its summary.

Status, as verified on 2026-09-25:

- `tests/test_shadow_crossing_engine.py` on its own reproduced the hang (9 tests pass, then the process was killed at 100 s). With the LINGER setting: `9 passed in 3.09s`, exit code 0.
- The full `tests/` run is **not** fixed by this alone: all 1,398 tests pass, and the process still does not exit after the last test (observed for more than 2 minutes at 100%). The leaking test or fixture has not been identified yet.
- A variant that also destroyed every live ZMQ context with `linger=0` at session end was tried and rejected: it made the process abort with a libzmq assertion (`ctx.cpp:185`, exit code 21) before the summary printed.

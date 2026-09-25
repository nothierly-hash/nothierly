# Testing

## Running the suites

Main suite (1,398 tests):

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Last full run, 2026-09-25: `1398 passed in 198.61s`, and the process exits on its own with code 0.

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

"""A settable clock for the rate-limiter tests.

Kept inside ``retail_gateway/tests`` (not shared with the trading core's ``tests/conftest.py``): this
package is air-gapped from the core, see ``retail_gateway/db.py``. Imports nothing from the gateway, so
it is importable even when ``RETAIL_JWT_SECRET_KEY`` is not set.
"""

from __future__ import annotations


class FakeClock:
    """Epoch seconds. Starts on a whole second and on a multiple of 3600, so it is at the start of a
    60 s, 900 s or 3600 s window: ``advance(w)`` moves to the next window of width ``w`` whatever the
    real time is, and a smaller advance stays inside the current one."""

    def __init__(self, now: float = 1_699_999_200.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

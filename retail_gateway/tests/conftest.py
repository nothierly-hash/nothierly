"""Retail Gateway tests are locked by default.

``RETAIL_JWT_SECRET_KEY`` has no default in ``retail_gateway/config.py`` (fail-closed by design), so
importing the gateway without one raises at import time. Instead of letting that abort collection for
a whole ``pytest`` run, these tests are not collected unless the key is explicitly injected in the
environment. Nothing here supplies or fakes a key; see TESTING.md.
"""

from __future__ import annotations

import os
import warnings

import pytest

from retail_gateway.tests.fake_clock import FakeClock

_KEY_PRESENT = bool(os.environ.get("RETAIL_JWT_SECRET_KEY"))

collect_ignore_glob = [] if _KEY_PRESENT else ["test_*.py"]

if not _KEY_PRESENT:
    warnings.warn(
        "retail_gateway/tests not collected: RETAIL_JWT_SECRET_KEY is not set (fail-closed by design). "
        "Inject a key to run them; see TESTING.md.",
        stacklevel=1,
    )


@pytest.fixture
def clock() -> FakeClock:
    """The clock the rate limiters are given, pinned so a test never depends on where in the real window it runs."""
    return FakeClock()

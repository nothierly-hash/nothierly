"""Real, executable tests for scripts/analyze_shadow_run.py.

Verified against genuinely generated logs before this file existed (a real
AlphaMarketMakerAgent run, real core.logger JSON output) -- these tests
lock that verification in as a real regression suite rather than leaving
it as a one-off manual check.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_shadow_run.py"
_spec = importlib.util.spec_from_file_location("analyze_shadow_run", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
analyze_shadow_run = importlib.util.module_from_spec(_spec)
sys.modules["analyze_shadow_run"] = analyze_shadow_run
_spec.loader.exec_module(analyze_shadow_run)


def _line(event: str, level: str = "info", timestamp: str | None = None, **extra: object) -> str:
    payload = {"event": event, "level": level, "timestamp": timestamp or "2026-09-07T10:00:00.000000Z", **extra}
    return json.dumps(payload)


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "shadow_run.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _timestamps(start: datetime, *offsets_seconds: float) -> list[str]:
    return [(start + timedelta(seconds=s)).isoformat().replace("+00:00", "Z") for s in offsets_seconds]


# =============================================================================
# _find_timestamp_gaps -- pure function, no I/O
# =============================================================================


def test_find_timestamp_gaps_none_for_regular_ticks() -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    timestamps = [start + timedelta(seconds=i) for i in range(20)]
    assert analyze_shadow_run._find_timestamp_gaps(timestamps) == []


def test_find_timestamp_gaps_detects_a_real_hang() -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    timestamps = [start + timedelta(seconds=i) for i in range(10)]
    timestamps.append(start + timedelta(seconds=9) + timedelta(hours=2))  # a real 2h silent gap
    gaps = analyze_shadow_run._find_timestamp_gaps(timestamps)
    assert len(gaps) == 1
    assert gaps[0][1] == pytest.approx(7200.0)


# =============================================================================
# analyze() -- against real, hand-crafted (but format-accurate) JSON log files
# =============================================================================


def test_clean_24h_shadow_run_is_a_go(tmp_path: Path) -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0], mock_mode=True),
    ]
    for i in range(50):
        lines.append(_line("alpha_engine_simulated_quote", timestamp=_timestamps(start, i * 1700)[0]))
    end_offset = 50 * 1700  # ~23.6h of evenly spaced ticks
    lines.append(_line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]))
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 0


def test_unexpected_exception_is_a_no_go(tmp_path: Path) -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("agent_tick_failed", level="error", timestamp=_timestamps(start, 100)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]),
    ]
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 2


def test_a_guardrail_firing_during_shadow_is_a_no_go(tmp_path: Path) -> None:
    """Structurally shouldn't happen under SHADOW -- seeing it at all is the anomaly. See module docstring point 3."""
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("alpha_engine_fat_finger_blocked", level="critical", timestamp=_timestamps(start, 100)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]),
    ]
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 2


def test_high_stale_data_count_is_attention_not_go(tmp_path: Path) -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [_line("alpha_engine_started", timestamp=_timestamps(start, 0)[0])]
    for i in range(60):
        lines.append(_line("alpha_engine_stale_data_detected", level="warning", timestamp=_timestamps(start, i * 1000)[0]))
    lines.append(_line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]))
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 1


def test_too_short_a_run_is_attention_not_go(tmp_path: Path) -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 3600)[0]),  # only 1h
    ]
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 1


def test_a_raw_non_json_traceback_line_is_a_no_go(tmp_path: Path) -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]),
        "Traceback (most recent call last):",
        "  ValueError: something bypassed structlog entirely",
    ]
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 2


def test_a_random_blank_or_unrelated_non_json_line_is_not_flagged(tmp_path: Path) -> None:
    """Only a Traceback/Exception-shaped non-JSON line is treated as suspicious -- a stray blank line or unrelated stdout noise is not."""
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        "",
        "some unrelated third-party library banner, not JSON, not a traceback",
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]),
    ]
    path = _write_log(tmp_path, lines)

    exit_code = analyze_shadow_run.analyze(path)

    assert exit_code == 0


# =============================================================================
# compute_diagnostics -- the real, structured, reusable core the dashboard page
# (src/dashboard/ems_pages/shadow_run_analytics.py) also calls directly
# =============================================================================


def test_compute_diagnostics_returns_structured_real_fields() -> None:
    """Field-extraction correctness -- verdict logic itself is covered by the analyze()-level tests above."""
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0], mock_mode=True),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 100)[0]),
        _line("alpha_engine_stale_data_detected", level="warning", timestamp=_timestamps(start, 200)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 300)[0]),
    ]

    diagnostics = analyze_shadow_run.compute_diagnostics(lines)

    assert diagnostics.simulated_quote_count == 2
    assert diagnostics.stale_data_count == 1
    assert diagnostics.total_json_lines == 4
    assert diagnostics.unexpected_failures == []
    assert diagnostics.guardrail_trips == []
    assert diagnostics.daily_loss_limit_configured is True  # no alpha_engine_no_daily_loss_limit_configured seen


def test_compute_diagnostics_accepts_any_string_iterable_not_just_a_file() -> None:
    """The real reason this is reusable from Streamlit's st.file_uploader (bytes decoded to a list of lines), not just Path.open()."""
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    uploaded_text = "\n".join([
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("agent_tick_failed", level="error", timestamp=_timestamps(start, 60)[0]),
    ])

    diagnostics = analyze_shadow_run.compute_diagnostics(uploaded_text.splitlines())

    assert diagnostics.verdict == "NO_GO"
    assert len(diagnostics.unexpected_failures) == 1


def test_compute_diagnostics_flags_missing_daily_loss_limit() -> None:
    start = datetime(2026, 9, 7, tzinfo=timezone.utc)
    lines = [
        _line("alpha_engine_started", timestamp=_timestamps(start, 0)[0]),
        _line("alpha_engine_no_daily_loss_limit_configured", level="critical", timestamp=_timestamps(start, 1)[0]),
        _line("alpha_engine_simulated_quote", timestamp=_timestamps(start, 24 * 3600)[0]),
    ]

    diagnostics = analyze_shadow_run.compute_diagnostics(lines)

    assert diagnostics.daily_loss_limit_configured is False
    assert diagnostics.verdict == "GO"  # this alone doesn't block a GO -- it's a separate, real, surfaced warning

"""Real, executable tests for the pure cost-aggregation function in src/mlops/vast_client.py.

No real Vast.ai API/network call: :func:`summarize_instance_costs` is a
pure function over already-fetched instance dicts (the same "pure
functions, real regression tests without live I/O" discipline as
``tests/test_eod_reconciliator.py``). Real Vast.ai field names used here
(``id``, ``dph_total``, ``actual_status``, ``start_date``, ``gpu_name``)
match the ones :meth:`~src.mlops.vast_client.VastAIClient.search_gpu`
already reads from the real API elsewhere in this module.
"""

from __future__ import annotations

import pytest

from src.mlops.vast_client import summarize_instance_costs

_NOW = 1_000_000.0  # an arbitrary, fixed "now" for deterministic accrual math


def test_empty_instance_list_returns_a_real_zeroed_summary() -> None:
    summary = summarize_instance_costs([], now=_NOW)
    assert summary.running_instance_count == 0
    assert summary.total_burn_rate_usd_per_hour == 0.0
    assert summary.total_accrued_cost_usd == 0.0
    assert summary.per_instance == []


def test_burn_rate_only_sums_running_instances() -> None:
    instances = [
        {"id": 1, "dph_total": 0.50, "actual_status": "running", "start_date": _NOW - 3600.0},
        {"id": 2, "dph_total": 0.30, "actual_status": "exited", "start_date": _NOW - 3600.0},
        {"id": 3, "dph_total": 0.20, "actual_status": "running", "start_date": _NOW - 3600.0},
    ]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.running_instance_count == 2
    assert summary.total_burn_rate_usd_per_hour == pytest.approx(0.70)


def test_accrued_cost_is_real_dph_times_real_elapsed_hours() -> None:
    """2 real hours elapsed at $0.40/hr on one instance -> $0.80 real accrued cost."""
    instances = [{"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": _NOW - 2 * 3600.0}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.total_accrued_cost_usd == pytest.approx(0.80)
    assert summary.per_instance[0].accrued_cost_usd == pytest.approx(0.80)


def test_accrued_cost_sums_across_every_instance_regardless_of_status() -> None:
    """Accrued (unlike burn rate) is real spend-to-date -- an exited instance still cost real money while it ran."""
    instances = [
        {"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": _NOW - 3600.0},
        {"id": 2, "dph_total": 0.20, "actual_status": "exited", "start_date": _NOW - 3600.0},
    ]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.total_accrued_cost_usd == pytest.approx(0.60)


def test_missing_start_date_yields_zero_accrued_not_a_guess() -> None:
    """No real start_date means nothing real to compute accrual from -- 0.0, never a fabricated estimate."""
    instances = [{"id": 1, "dph_total": 0.40, "actual_status": "running", "start_date": None}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.per_instance[0].accrued_cost_usd == 0.0
    assert summary.total_accrued_cost_usd == 0.0


def test_missing_dph_total_defaults_to_zero_not_an_error() -> None:
    instances = [{"id": 1, "actual_status": "running", "start_date": _NOW - 3600.0}]
    summary = summarize_instance_costs(instances, now=_NOW)
    assert summary.per_instance[0].dph_total == 0.0


def test_per_instance_rows_carry_real_gpu_name_and_status() -> None:
    instances = [{"id": 42, "dph_total": 0.55, "actual_status": "running", "start_date": _NOW, "gpu_name": "RTX_4090"}]
    summary = summarize_instance_costs(instances, now=_NOW)
    row = summary.per_instance[0]
    assert row.instance_id == 42
    assert row.gpu_name == "RTX_4090"
    assert row.actual_status == "running"

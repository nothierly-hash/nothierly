"""Real, executable smoke test for src/mlops/host_metrics.py.

No mocking: this genuinely calls ``psutil``/``torch.cuda`` against
whatever machine runs the test suite -- the point of the test is that
the real call succeeds and returns a well-formed, honest snapshot, not
a specific hardware value (which would make this test environment-
dependent in the wrong way).
"""

from __future__ import annotations

from src.mlops.host_metrics import get_host_memory_snapshot


def test_real_snapshot_has_sane_ram_figures() -> None:
    snapshot = get_host_memory_snapshot()
    assert snapshot.ram_total_bytes > 0
    assert 0.0 <= snapshot.ram_used_bytes <= snapshot.ram_total_bytes
    assert 0.0 <= snapshot.ram_percent <= 100.0


def test_vram_fields_are_none_not_zero_when_no_real_gpu_is_present() -> None:
    """Honest distinction: None means 'nothing to measure', never fabricated as a real 0% usage."""
    snapshot = get_host_memory_snapshot()
    if not snapshot.gpu_available:
        assert snapshot.vram_total_bytes is None
        assert snapshot.vram_used_bytes is None
        assert snapshot.gpu_name is None
    else:
        assert snapshot.vram_total_bytes is not None and snapshot.vram_total_bytes > 0
        assert snapshot.vram_used_bytes is not None and snapshot.vram_used_bytes >= 0
        assert snapshot.gpu_name is not None

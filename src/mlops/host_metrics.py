"""Real, local-host hardware metrics -- RAM (``psutil``) and VRAM (``torch.cuda``, if a real GPU is present).

**Honest scope, stated upfront**: this reports the metrics of whatever
machine actually runs this code -- the dashboard's own host process --
never the remote Vast.ai training instance, which this project has no
telemetry channel to (see :mod:`~src.mlops.vast_client`'s own module
docstring on what real infrastructure data *is* available there: cost
figures derived from the Vast.ai API's own reported fields, not live
hardware utilization it doesn't expose to this codebase). Surfacing
this local snapshot next to Vast.ai's cost figures under one "Cloud &
FinOps" page is still real and useful (it's genuinely this project's
own operating environment), it is just not the remote instance the
brief's "RAM/VRAM" framing implied without a way to actually observe
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Real, instantaneous memory figures for this machine -- see module docstring on scope."""

    ram_total_bytes: int
    ram_used_bytes: int
    ram_percent: float
    gpu_available: bool
    gpu_name: str | None
    vram_total_bytes: int | None
    vram_used_bytes: int | None


def get_host_memory_snapshot() -> HostMemorySnapshot:
    """Real RAM (``psutil.virtual_memory``) + real VRAM (``torch.cuda``, only if ``torch.cuda.is_available()``).

    VRAM fields are ``None`` (never ``0`` -- that would misreport "no
    usage" as opposed to the real "no GPU to measure") when no CUDA
    device is present, which is the honest, common case on a plain
    CPU-only host (confirmed the real state of this project's own
    ``torch`` build/dev environment before writing this function).
    """
    vm = psutil.virtual_memory()

    gpu_available = torch.cuda.is_available()
    gpu_name: str | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    if gpu_available:
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        vram_total_bytes = torch.cuda.get_device_properties(device).total_memory
        vram_used_bytes = torch.cuda.memory_allocated(device)

    return HostMemorySnapshot(
        ram_total_bytes=vm.total, ram_used_bytes=vm.used, ram_percent=vm.percent,
        gpu_available=gpu_available, gpu_name=gpu_name,
        vram_total_bytes=vram_total_bytes, vram_used_bytes=vram_used_bytes,
    )

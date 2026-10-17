"""Deploy Nothierly's own proprietary macro-sentiment LLM on a rented Vast.ai GPU (vLLM, AWQ-quantized).

**Real financial action, not a demo**: renting a GPU on Vast.ai spends
real money for every hour the instance stays up. This script does *not*
auto-destroy the instance on success (unlike :mod:`scripts.run_remote_training`,
a one-shot training job) -- a deployed LLM server is meant to keep
running as a service for :mod:`political_agent` to call. It *does*
auto-destroy on any failure to reach a healthy state (bad onstart script,
model never finishes loading, ...), so a broken deployment never bills
silently forever. To stop billing on a *successful* deployment later, use
this repo's own FinOps dashboard page (``src/dashboard/ems_pages/finops.py``,
already has Pause/Destroy controls for every rented Vast.ai instance) or
call :meth:`~src.mlops.vast_client.VastAIClient.destroy_instance` directly
with the instance ID this script prints.

**Guarded the same way every other real spend in this project is**:
refuses to run at all unless both the global Kill Switch is SAFE and the
Vast.ai master switch (dashboard FinOps toggle, or ``VASTAI_ENABLED=true``)
is on -- see :func:`~src.mlops.vast_client.VastAIClient.rent_instance`'s
own guards. As of this script being written, the live deployment has the
master switch OFF (verified, not assumed) -- running this for real
requires deliberately turning it on first.

**GPU selection, and a real, disclosed reason no separate VRAM check is
needed**: both "RTX_3090" and "RTX_4090" are Vast.ai's own ``gpu_name``
values for GPUs that only ever ship in a 24GB configuration -- unlike
"A100" (40GB/80GB variants both exist), there is no lower-VRAM SKU of
either card, so filtering on ``gpu_name`` alone already guarantees the
mission's ">=24GB VRAM" requirement; :class:`~src.mlops.vast_client.GpuOffer`
does not expose a separate GPU-VRAM field to check regardless.
:meth:`~src.mlops.vast_client.VastAIClient.search_offers` filters one
``gpu_name`` per call (Vast.ai's own query API has no "3090 OR 4090"
operator this project's thin wrapper exposes), so this script searches
both models separately and merges the results, cheapest first.

Run manually, offline::

    python scripts/deploy_custom_llm.py --max-price-per-hour 0.50
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/deploy_custom_llm.py` adds this file's own directory (scripts/) to sys.path, not
# the project root -- `core`/`src` live at the root, same fix every other script here already needs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
from typing import Any, Final

from core.config import get_settings
from core.http_client import ResilientHttpClient
from core.logger import configure_logging, get_logger
from core.redis_client import RedisClient
from core.schemas import PROPRIETARY_LLM_ENDPOINT_REDIS_KEY, PROPRIETARY_LLM_MODEL_NAME
from src.mlops.vast_client import GpuOffer, VastAIClient, VastAIError

_log = get_logger(__name__)

_VLLM_IMAGE: Final[str] = "vllm/vllm-openai:latest"
_CANDIDATE_GPU_NAMES: Final[tuple[str, ...]] = ("RTX_3090", "RTX_4090")
_DEFAULT_MIN_RELIABILITY: Final[float] = 0.99
_DEFAULT_MAX_PRICE_PER_HOUR: Final[float] = 0.50

# The mission's own literal onstart command -- AWQ quantization caps real VRAM use around ~8GB,
# a large margin under the 24GB every candidate GPU above has, specifically to avoid an OOM crash.
_ONSTART_SCRIPT: Final[str] = (
    f"python3 -m vllm.entrypoints.openai.api_server "
    f"--model {PROPRIETARY_LLM_MODEL_NAME} --quantization awq --max-model-len 4096 --port 8000"
)

_CONTAINER_PORT: Final[str] = "8000/tcp"
_NETWORK_POLL_INTERVAL_SECONDS: Final[float] = 10.0
_NETWORK_POLL_TIMEOUT_SECONDS: Final[float] = 300.0
# Real, disclosed margin: loading a 7B AWQ model onto a fresh instance (image pull if not cached +
# CUDA init + vLLM's own weight loading) has been observed elsewhere in this project's own MLOps
# scripts to take minutes, not seconds -- this is deliberately generous, not a guessed small number.
_VLLM_HEALTH_POLL_INTERVAL_SECONDS: Final[float] = 15.0
_VLLM_HEALTH_POLL_TIMEOUT_SECONDS: Final[float] = 600.0


async def _find_best_offer(
    vast_client: VastAIClient, *, min_reliability: float, max_price_per_hour: float
) -> GpuOffer:
    """Search both candidate GPU models, merge, return the single cheapest real match.

    Raises:
        VastAIError: If neither model has any rentable offer matching both criteria.
    """
    all_offers: list[GpuOffer] = []
    for gpu_name in _CANDIDATE_GPU_NAMES:
        offers = await vast_client.search_offers(
            gpu_name=gpu_name, max_dph=max_price_per_hour, min_reliability=min_reliability,
        )
        all_offers.extend(offers)
        _log.info(
            "deploy_llm_gpu_search_result", gpu_name=gpu_name, matches=len(offers),
            min_reliability=min_reliability, max_price_per_hour=max_price_per_hour,
        )

    if not all_offers:
        raise VastAIError(
            f"No rentable offer among {_CANDIDATE_GPU_NAMES} with reliability >= "
            f"{min_reliability:.2f} at <= ${max_price_per_hour:.2f}/hr."
        )

    return min(all_offers, key=lambda offer: offer.dph_total)


async def _wait_for_port_mapping(
    vast_client: VastAIClient, instance_id: int,
) -> tuple[str, int]:
    """Poll until Vast.ai has assigned a public IP + external port for the container's 8000/tcp.

    **Real, disclosed uncertainty**: Vast.ai's own raw instance status
    includes ``public_ipaddr`` and a ``ports`` mapping (keyed by
    ``"8000/tcp"``, Docker-inspect-shaped) per Vast.ai's documented API --
    this project has never actually exercised that field before now (every
    prior real use of this client, :mod:`scripts.run_remote_training`,
    only ever needed ``ssh_host``/``ssh_port``). If Vast.ai's real response
    shape differs from what's coded here, this raises with the *raw*
    instance dict in the log so it is fast to fix against a real account,
    rather than silently guessing a wrong port.

    Returns:
        ``(public_ip, external_port)``.

    Raises:
        VastAIError: If the mapping is not assigned within the timeout.
    """
    elapsed = 0.0
    while elapsed < _NETWORK_POLL_TIMEOUT_SECONDS:
        instance = await vast_client.get_instance_details(instance_id)
        public_ip = instance.get("public_ipaddr")
        port_entries = (instance.get("ports") or {}).get(_CONTAINER_PORT)
        if public_ip and port_entries:
            external_port = int(port_entries[0]["HostPort"])
            _log.info(
                "deploy_llm_port_mapping_ready",
                instance_id=instance_id, public_ip=public_ip, external_port=external_port,
            )
            return str(public_ip), external_port

        _log.info(
            "deploy_llm_port_mapping_pending",
            instance_id=instance_id, status=instance.get("actual_status"),
        )
        await asyncio.sleep(_NETWORK_POLL_INTERVAL_SECONDS)
        elapsed += _NETWORK_POLL_INTERVAL_SECONDS

    raise VastAIError(
        f"Instance {instance_id} was not assigned a public {_CONTAINER_PORT} mapping within "
        f"{_NETWORK_POLL_TIMEOUT_SECONDS}s. Raw instance dict for debugging: "
        f"{await vast_client.get_instance_details(instance_id)}"
    )


async def _wait_for_vllm_healthy(base_url: str) -> None:
    """Poll vLLM's own real ``/health`` endpoint until it answers 200, or give up.

    Raises:
        VastAIError: If the server never becomes healthy within the timeout
            (still starting, crashed on the onstart command, OOM, ...).
    """
    elapsed = 0.0
    async with ResilientHttpClient(timeout_seconds=10.0) as http:
        while elapsed < _VLLM_HEALTH_POLL_TIMEOUT_SECONDS:
            try:
                await http.get_text(f"{base_url}/health")
                _log.info("deploy_llm_vllm_healthy", base_url=base_url)
                return
            except Exception as exc:
                _log.info(
                    "deploy_llm_vllm_not_ready_yet", base_url=base_url, error=str(exc),
                    elapsed_seconds=elapsed,
                )
            await asyncio.sleep(_VLLM_HEALTH_POLL_INTERVAL_SECONDS)
            elapsed += _VLLM_HEALTH_POLL_INTERVAL_SECONDS

    raise VastAIError(
        f"vLLM server at {base_url} never answered /health within "
        f"{_VLLM_HEALTH_POLL_TIMEOUT_SECONDS}s -- likely still loading the model, or crashed "
        f"on startup. Check https://console.vast.ai/instances/ for the instance's own logs."
    )


async def deploy_custom_llm(
    *,
    ssh_public_key_path: Path,
    min_reliability: float = _DEFAULT_MIN_RELIABILITY,
    max_price_per_hour: float = _DEFAULT_MAX_PRICE_PER_HOUR,
) -> tuple[int, str]:
    """Rent a GPU, launch vLLM, wait for it to be real and healthy, publish its endpoint.

    Returns:
        ``(instance_id, base_url)``.

    Raises:
        VastAIError: On any failure -- the partially-created instance (if
            any) is destroyed before this re-raises, so a failed
            deployment never bills silently.
    """
    settings = get_settings()
    vastai = settings.vastai
    if vastai.api_key is None:
        raise VastAIError("VASTAI_API_KEY is not configured.")

    ssh_public_key = ssh_public_key_path.read_text().strip()
    instance_id: int | None = None

    async with VastAIClient(api_key=vastai.api_key.get_secret_value()) as vast_client:
        try:
            offer = await _find_best_offer(
                vast_client, min_reliability=min_reliability, max_price_per_hour=max_price_per_hour,
            )
            _log.info(
                "deploy_llm_offer_selected",
                ask_id=offer.ask_id, gpu_name=offer.gpu_name, dph_total=offer.dph_total,
                reliability=offer.reliability,
            )

            instance_id = await vast_client.rent_instance(
                offer.ask_id,
                ssh_public_key=ssh_public_key,
                image=_VLLM_IMAGE,
                disk_gb=30,  # image + AWQ model weights, real margin above the ~8GB VRAM footprint
                onstart_script=_ONSTART_SCRIPT,
            )
            _log.info("deploy_llm_instance_rented", instance_id=instance_id, ask_id=offer.ask_id)

            public_ip, external_port = await _wait_for_port_mapping(vast_client, instance_id)
            base_url = f"http://{public_ip}:{external_port}"

            await _wait_for_vllm_healthy(base_url)

            redis_client = await RedisClient.get_instance(settings.redis)
            await redis_client.execute("set", PROPRIETARY_LLM_ENDPOINT_REDIS_KEY, base_url)
            _log.info(
                "deploy_llm_endpoint_published",
                base_url=base_url, redis_key=PROPRIETARY_LLM_ENDPOINT_REDIS_KEY,
            )

            return instance_id, base_url
        except Exception:
            if instance_id is not None:
                _log.critical(
                    "deploy_llm_failed_destroying_instance",
                    instance_id=instance_id,
                    detail="Deployment did not reach a healthy state -- destroying to stop billing.",
                )
                try:
                    await vast_client.destroy_instance(instance_id)
                except Exception:
                    _log.critical(
                        "deploy_llm_destroy_after_failure_also_failed",
                        instance_id=instance_id,
                        detail=(
                            "MANUAL INTERVENTION REQUIRED: verify and destroy this instance at "
                            "https://console.vast.ai/instances/"
                        ),
                    )
            raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ssh-public-key", type=Path, default=None, help="Overrides VASTAI_SSH_PUBLIC_KEY_PATH."
    )
    parser.add_argument("--min-reliability", type=float, default=_DEFAULT_MIN_RELIABILITY)
    parser.add_argument("--max-price-per-hour", type=float, default=_DEFAULT_MAX_PRICE_PER_HOUR)
    parser.add_argument(
        "--destroy", type=int, default=None, metavar="INSTANCE_ID",
        help=(
            "Stop billing on a previously-deployed instance instead of deploying a new one "
            "(the FinOps dashboard page's own Destroy control does the same thing)."
        ),
    )
    return parser.parse_args()


async def _destroy(instance_id: int) -> None:
    settings = get_settings()
    if settings.vastai.api_key is None:
        raise VastAIError("VASTAI_API_KEY is not configured.")
    async with VastAIClient(api_key=settings.vastai.api_key.get_secret_value()) as vast_client:
        await vast_client.destroy_instance(instance_id)


def main() -> None:
    configure_logging(log_level="INFO", log_dir=None, service_name="deploy-custom-llm")
    args = _parse_args()

    if args.destroy is not None:
        asyncio.run(_destroy(args.destroy))
        print(f"Destroyed instance {args.destroy}. Billing stopped.")
        return

    settings = get_settings()
    ssh_public_key_path = args.ssh_public_key or settings.vastai.ssh_public_key_path
    if ssh_public_key_path is None:
        raise VastAIError(
            "SSH key path must be given via --ssh-public-key or VASTAI_SSH_PUBLIC_KEY_PATH."
        )

    instance_id, base_url = asyncio.run(
        deploy_custom_llm(
            ssh_public_key_path=ssh_public_key_path,
            min_reliability=args.min_reliability,
            max_price_per_hour=args.max_price_per_hour,
        )
    )
    print(f"Deployed. instance_id={instance_id} base_url={base_url}")
    print(
        f"political_agent.py will pick this endpoint up automatically on its next cycle "
        f"(reads Redis key '{PROPRIETARY_LLM_ENDPOINT_REDIS_KEY}')."
    )
    print(
        f"This instance bills every hour it stays up. Stop it from the FinOps dashboard page, or:\n"
        f"  python scripts/deploy_custom_llm.py --destroy {instance_id}"
    )


if __name__ == "__main__":
    main()

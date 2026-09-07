"""Master orchestration script: rent a Vast.ai GPU, train remotely, retrieve
the model -- and ALWAYS destroy the instance afterward.

Not part of the live trading system -- run manually, offline::

    python scripts/run_remote_training.py \\
        --gpu-name RTX_4090 --max-price-per-hour 0.40 \\
        --local-dataset data/timescale_export.csv \\
        --local-script scripts/train_ppo.py

CRITICAL SAFETY PROPERTY -- read before running: the entire process
(search, rent, wait for SSH, upload, train, download) is wrapped in a
single ``try/except/finally``. The ``finally`` block unconditionally
calls :meth:`~src.mlops.vast_client.VastAIClient.destroy_instance` if an
``instance_id`` was ever created -- regardless of what failed (a network
crash, a PyTorch exception on the remote box, an SCP failure, or the
operator pressing Ctrl+C). Python's ``finally`` semantics guarantee this
runs on *any* exception, including ``KeyboardInterrupt`` (a
``BaseException``, not just ``Exception``), which is exactly why a plain
``try/finally`` -- not a narrower ``except Exception`` -- is used here.
An orphaned, still-running instance keeps costing real money every hour
it's up; this is the single most important property of this script.

The only scenario this cannot protect against is the process itself
being killed outright (SIGKILL, terminal closed with no chance to
unwind, power loss) before the ``finally`` block executes -- if that
happens, verify and manually destroy the instance at
https://console.vast.ai/instances/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python scripts/run_remote_training.py` adds this file's own directory
# (scripts/) to sys.path, not the project root -- `core`/`src` live at
# the root, so it must be added explicitly before those imports below,
# regardless of the working directory the command was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
from typing import Any

from core.config import get_settings
from core.logger import configure_logging, get_logger
from src.mlops.ssh_orchestrator import SSHPipeline
from src.mlops.vast_client import VastAIClient, VastAIError

_log = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


async def _wait_for_instance_network(
    vast_client: VastAIClient,
    instance_id: int,
    *,
    poll_interval_seconds: float = 10.0,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Poll :meth:`VastAIClient.get_instance_details` until ``ssh_host``/``ssh_port`` are assigned.

    Args:
        vast_client: A connected :class:`VastAIClient`.
        instance_id: From :meth:`VastAIClient.rent_instance`.
        poll_interval_seconds: Time between status checks.
        timeout_seconds: Give up after this long.

    Returns:
        The instance's status dict once ``ssh_host``/``ssh_port`` are set.

    Raises:
        VastAIError: If the instance never gets network details assigned
            within ``timeout_seconds``.
    """
    elapsed = 0.0
    while elapsed < timeout_seconds:
        instance = await vast_client.get_instance_details(instance_id)
        if instance.get("ssh_host") and instance.get("ssh_port"):
            _log.info(
                "vast_instance_network_ready",
                instance_id=instance_id,
                ssh_host=instance["ssh_host"],
                ssh_port=instance["ssh_port"],
            )
            return instance

        _log.info(
            "vast_instance_network_pending",
            instance_id=instance_id,
            status=instance.get("actual_status"),
        )
        await asyncio.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    raise VastAIError(
        f"Instance {instance_id} was not assigned ssh_host/ssh_port within {timeout_seconds}s."
    )


async def run_remote_training(
    *,
    gpu_name: str,
    max_price_per_hour: float,
    image: str,
    ssh_public_key_path: Path,
    ssh_private_key_path: Path,
    local_dataset_path: Path,
    local_script_path: Path,
    remote_workdir: str = "/root/training",
    remote_model_path: str = "/root/training/actor_critic.pt",
    local_models_dir: Path = MODELS_DIR,
) -> Path:
    """Rent a GPU, train remotely, retrieve the model -- destroying the instance no matter what.

    See the module docstring for the safety guarantee this function's
    ``try/finally`` structure provides.

    Args:
        gpu_name: GPU model to rent (e.g. ``"RTX_4090"``).
        max_price_per_hour: Maximum acceptable price, in USD/hour.
        image: Docker image to launch on the rented instance.
        ssh_public_key_path: Local path to the SSH public key to
            authorize on the instance.
        ssh_private_key_path: Local path to the matching private key.
        local_dataset_path: Local path to the exported TimescaleDB
            dataset to upload.
        local_script_path: Local path to the training script to upload
            and run (e.g. ``scripts/train_ppo.py``).
        remote_workdir: Directory on the instance to upload into/run from.
        remote_model_path: Where the training script writes its output
            model, remotely.
        local_models_dir: Local directory to download the trained model
            into.

    Returns:
        Local path to the downloaded model file.

    Raises:
        VastAIError: If no matching GPU is found, or the rented instance
            never becomes network-reachable.
        ~src.mlops.ssh_orchestrator.SSHConnectionTimeoutError: If the
            instance never becomes SSH-reachable.
        ~src.mlops.ssh_orchestrator.RemoteTrainingError: If the remote
            training command exits non-zero.
    """
    settings = get_settings()
    vastai = settings.vastai
    if vastai.api_key is None:
        raise VastAIError("VASTAI_API_KEY is not configured.")

    ssh_public_key = ssh_public_key_path.read_text().strip()
    instance_id: int | None = None

    async with VastAIClient(api_key=vastai.api_key.get_secret_value()) as vast_client:
        try:
            ask_id = await vast_client.search_gpu(
                gpu_name=gpu_name, max_price_per_hour=max_price_per_hour
            )
            instance_id = await vast_client.rent_instance(
                ask_id, ssh_public_key=ssh_public_key, image=image
            )

            instance = await _wait_for_instance_network(vast_client, instance_id)

            pipeline = SSHPipeline(
                host=instance["ssh_host"],
                port=int(instance["ssh_port"]),
                private_key_path=ssh_private_key_path,
            )
            await pipeline.wait_for_ready()

            model_path = await pipeline.run_training_pipeline(
                local_dataset_path=local_dataset_path,
                local_script_path=local_script_path,
                remote_workdir=remote_workdir,
                remote_model_path=remote_model_path,
                local_models_dir=local_models_dir,
                train_command=f"cd {remote_workdir} && python3 {local_script_path.name}",
            )

            _log.info(
                "remote_training_pipeline_succeeded",
                instance_id=instance_id,
                model_path=str(model_path),
            )
            return model_path
        finally:
            # ALWAYS destroy the instance if one was created -- whatever
            # happened above. See the module docstring's safety guarantee.
            if instance_id is not None:
                try:
                    await vast_client.destroy_instance(instance_id)
                except Exception:
                    # A failed destroy is the single most urgent thing to
                    # surface: re-raise so it is never silently lost,
                    # even at the cost of superseding an original
                    # training/SSH failure (still available via
                    # __context__ on the exception this raises).
                    _log.critical(
                        "vast_instance_destroy_failed",
                        instance_id=instance_id,
                        detail=(
                            "MANUAL INTERVENTION REQUIRED: verify and destroy this "
                            "instance at https://console.vast.ai/instances/"
                        ),
                    )
                    raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu-name", default="RTX_4090", help="GPU model to rent.")
    parser.add_argument("--max-price-per-hour", type=float, default=0.40, help="Max USD/hour.")
    parser.add_argument("--image", default="pytorch/pytorch:latest", help="Docker image to launch.")
    parser.add_argument(
        "--ssh-public-key", type=Path, default=None, help="Overrides VASTAI_SSH_PUBLIC_KEY_PATH."
    )
    parser.add_argument(
        "--ssh-private-key", type=Path, default=None, help="Overrides VASTAI_SSH_PRIVATE_KEY_PATH."
    )
    parser.add_argument(
        "--local-dataset", required=True, type=Path, help="Exported TimescaleDB dataset file to upload."
    )
    parser.add_argument(
        "--local-script", required=True, type=Path, help="Training script to upload and run."
    )
    parser.add_argument("--remote-workdir", default="/root/training")
    parser.add_argument("--remote-model-path", default="/root/training/actor_critic.pt")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    configure_logging(log_level="INFO", log_dir=None)
    args = _parse_args()
    settings = get_settings()
    vastai = settings.vastai

    ssh_public_key_path = args.ssh_public_key or vastai.ssh_public_key_path
    ssh_private_key_path = args.ssh_private_key or vastai.ssh_private_key_path
    if ssh_public_key_path is None or ssh_private_key_path is None:
        raise VastAIError(
            "SSH key paths must be given via --ssh-public-key/--ssh-private-key or "
            "VASTAI_SSH_PUBLIC_KEY_PATH/VASTAI_SSH_PRIVATE_KEY_PATH."
        )

    model_path = asyncio.run(
        run_remote_training(
            gpu_name=args.gpu_name,
            max_price_per_hour=args.max_price_per_hour,
            image=args.image,
            ssh_public_key_path=ssh_public_key_path,
            ssh_private_key_path=ssh_private_key_path,
            local_dataset_path=args.local_dataset,
            local_script_path=args.local_script,
            remote_workdir=args.remote_workdir,
            remote_model_path=args.remote_model_path,
        )
    )
    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()

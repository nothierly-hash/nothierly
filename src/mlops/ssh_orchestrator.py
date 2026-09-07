"""Async SSH orchestration for a rented, ephemeral Vast.ai instance.

:class:`SSHPipeline` connects to one rented GPU instance and runs a
three-step training job on it (upload -> run -> download), entirely via
``asyncssh`` -- no shelling out to the system ``ssh``/``scp`` binaries.

Vast.ai assigns a fresh IP and port to every rented instance, so there is
no stable host-key history to check against: ``known_hosts=None`` is a
deliberate, *necessary* choice for exactly this ephemeral-infrastructure
use case (a brand-new machine with no prior host key on record), not a
general-purpose security shortcut -- do not copy this default into code
connecting to long-lived, known hosts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Final

import asyncssh

from core.logger import get_logger

_log = get_logger(__name__)

_DEFAULT_USERNAME: Final[str] = "root"


class SSHConnectionTimeoutError(Exception):
    """Raised when the remote instance never becomes SSH-reachable within the configured timeout."""


class RemoteTrainingError(Exception):
    """Raised when the remote training command exits non-zero."""


class SSHPipeline:
    """Connects to one ephemeral rented GPU instance and runs a training job on it."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        private_key_path: Path,
        username: str = _DEFAULT_USERNAME,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        """Configure the connection target. Does not connect yet.

        Args:
            host: The instance's dynamically-assigned SSH host/IP (see
                :meth:`~src.mlops.vast_client.VastAIClient.
                get_instance_details`).
            port: The instance's dynamically-assigned SSH port.
            private_key_path: Local path to the private key matching the
                public key authorized when the instance was rented.
            username: Remote SSH username.
            connect_timeout_seconds: Per-attempt connection timeout.
        """
        self._host = host
        self._port = port
        self._private_key_path = private_key_path
        self._username = username
        self._connect_timeout_seconds = connect_timeout_seconds

    def _connect(self) -> Any:
        """Return an (unawaited) ``asyncssh.connect(...)`` call.

        Callers use this as ``async with self._connect() as conn:`` --
        ``asyncssh.connect`` is typed (in ``asyncssh``'s own stubs) as
        returning a special wrapper object that is both awaitable and
        usable directly as an async context manager, the documented,
        idiomatic ``asyncssh`` pattern. That wrapper type
        (``asyncssh.misc._ACMWrapper``) is private to ``asyncssh``
        (leading underscore, not meant to be referenced from outside
        it), hence the explicit ``Any`` here rather than naming it.

        ``known_hosts=None``: see the module docstring.
        """
        return asyncssh.connect(
            self._host,
            port=self._port,
            username=self._username,
            client_keys=[str(self._private_key_path)],
            known_hosts=None,
            connect_timeout=self._connect_timeout_seconds,
        )

    async def wait_for_ready(
        self, *, poll_interval_seconds: float = 10.0, timeout_seconds: float = 600.0
    ) -> None:
        """Poll SSH connectivity until it succeeds or ``timeout_seconds`` elapses.

        A freshly-rented instance takes time to boot its container and
        start ``sshd``; this retries a full connection attempt every
        ``poll_interval_seconds`` until one succeeds.

        Args:
            poll_interval_seconds: Time between connection attempts.
            timeout_seconds: Give up after this long.

        Raises:
            SSHConnectionTimeoutError: If the instance never became
                reachable within ``timeout_seconds``.
        """
        elapsed = 0.0
        last_error: Exception | None = None

        while elapsed < timeout_seconds:
            try:
                async with self._connect():
                    _log.info(
                        "ssh_instance_ready",
                        host=self._host,
                        port=self._port,
                        elapsed_seconds=round(elapsed, 1),
                    )
                    return
            except (OSError, asyncssh.Error) as exc:
                last_error = exc
                _log.info(
                    "ssh_instance_not_ready_yet",
                    host=self._host,
                    port=self._port,
                    elapsed_seconds=round(elapsed, 1),
                    error=str(exc),
                )

            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

        raise SSHConnectionTimeoutError(
            f"Instance {self._host}:{self._port} did not become SSH-reachable within "
            f"{timeout_seconds}s (last error: {last_error})."
        )

    async def run_training_pipeline(
        self,
        *,
        local_dataset_path: Path,
        local_script_path: Path,
        remote_workdir: str,
        remote_model_path: str,
        local_models_dir: Path,
        train_command: str,
    ) -> Path:
        """Upload dataset+script, run training, download the resulting model.

        Three sequential steps over one SSH connection:

        1. ``asyncssh.scp`` uploads ``local_dataset_path`` and
           ``local_script_path`` to ``remote_workdir``.
        2. ``conn.run(train_command, check=True)`` runs the training
           script; ``check=True`` raises if it exits non-zero.
        3. ``asyncssh.scp`` downloads ``remote_model_path`` into
           ``local_models_dir``.

        Args:
            local_dataset_path: Local path to the exported TimescaleDB
                data file.
            local_script_path: Local path to the training script (e.g.
                ``scripts/train_ppo.py``).
            remote_workdir: Directory on the instance to upload into and
                run from.
            remote_model_path: Where the training script writes its
                output model, on the remote instance.
            local_models_dir: Local directory to download the model into.
            train_command: Shell command that runs the uploaded script
                remotely (e.g. ``f"cd {remote_workdir} && python3
                {local_script_path.name}"``).

        Returns:
            Local path to the downloaded model file.

        Raises:
            RemoteTrainingError: If the remote training command exits
                non-zero.
            asyncssh.Error: On an upload/download (SCP) failure.
        """
        async with self._connect() as conn:
            _log.info(
                "ssh_upload_started",
                host=self._host,
                files=[str(local_dataset_path), str(local_script_path)],
            )
            await asyncssh.scp([local_dataset_path, local_script_path], (conn, remote_workdir))
            _log.info("ssh_upload_completed", host=self._host)

            _log.info("ssh_training_started", host=self._host, command=train_command)
            try:
                result = await conn.run(train_command, check=True)
            except asyncssh.ProcessError as exc:
                stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else str(exc.stderr)[-2000:]
                _log.error(
                    "ssh_training_failed",
                    host=self._host,
                    exit_status=exc.exit_status,
                    stderr_tail=stderr_tail,
                )
                raise RemoteTrainingError(
                    f"Remote training command exited with status {exc.exit_status}. "
                    f"Last stderr:\n{stderr_tail}"
                ) from exc

            stdout_tail = (result.stdout or "")[-500:] if isinstance(result.stdout, str) else ""
            _log.info("ssh_training_completed", host=self._host, stdout_tail=stdout_tail)

            local_models_dir.mkdir(parents=True, exist_ok=True)
            local_model_path = local_models_dir / Path(remote_model_path).name

            _log.info(
                "ssh_download_started", host=self._host, remote_model_path=remote_model_path
            )
            await asyncssh.scp((conn, remote_model_path), local_model_path)
            _log.info(
                "ssh_download_completed", host=self._host, local_model_path=str(local_model_path)
            )

            return local_model_path

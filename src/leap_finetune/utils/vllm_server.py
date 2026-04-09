"""vLLM server lifecycle helpers for GRPO server-mode rollouts.

When a GRPO YAML sets ``vllm_mode: server`` + a ``grpo_rollout:`` block with
``dedicated_gpus > 0``, the Ray Train driver:

1. Carves off ``dedicated_gpus`` from the training pool via
   ``CUDA_VISIBLE_DEVICES``.
2. Launches ``trl vllm-serve`` as a subprocess pinned to those carved-off
   GPUs, using the same model that training is about to load.
3. Waits for ``GET /health`` to return 200 before proceeding with training.
4. Registers an ``atexit`` hook to terminate the server cleanly after
   ``trainer.fit()`` returns.

We do NOT write a custom server — ``trl vllm-serve`` is TRL's own CLI and
we just invoke it via ``subprocess.Popen``. Likewise we do not write a
custom client: TRL's ``GRPOTrainer`` handles the server connection
internally when ``use_vllm=True, vllm_mode="server"`` is set on its config.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class VLLMServerHandle:
    """Handle to a running ``trl vllm-serve`` subprocess.

    Carries enough info to (a) plumb the server endpoint through to
    GRPOConfig on the training workers, and (b) cleanly tear down the
    server at the end of training.
    """

    process: subprocess.Popen
    host: str
    port: int
    tensor_parallel_size: int
    dedicated_gpu_ids: list[int]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def plan_gpu_split(
    total_gpus: int, grpo_rollout_cfg: dict
) -> tuple[list[int], list[int]]:
    """Decide which GPUs go to training vs vLLM.

    Training uses the *trailing* GPUs so the lowest-indexed devices stay
    with vLLM — a common convention that keeps the training process on
    a contiguous device range after we set ``CUDA_VISIBLE_DEVICES``.

    Returns ``(vllm_gpu_ids, train_gpu_ids)`` — both lists of absolute
    physical GPU indices.
    """
    dedicated = int(grpo_rollout_cfg.get("dedicated_gpus", 0))
    if dedicated < 0:
        raise ValueError(f"grpo_rollout.dedicated_gpus must be >= 0, got {dedicated}")
    if dedicated == 0:
        return [], list(range(total_gpus))
    if dedicated >= total_gpus:
        raise ValueError(
            f"grpo_rollout.dedicated_gpus={dedicated} leaves no GPUs for training "
            f"(total={total_gpus}). You need at least one GPU for training."
        )
    vllm_gpus = list(range(dedicated))
    train_gpus = list(range(dedicated, total_gpus))
    return vllm_gpus, train_gpus


def resolve_server_host(host: str | None) -> str:
    """Resolve a ``vllm_server_host`` YAML value.

    Special values:

    * ``None`` / ``"auto"`` → the current node's hostname, resolved via
      ``SLURMD_NODENAME`` (if running under SLURM) or ``socket.gethostname()``.
      On a single-node dev box this resolves to the machine's own hostname,
      which the workers can reach over loopback.
    * Any other string → used verbatim.
    """
    if not host or host == "auto":
        return os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    return host


def launch_vllm_server(
    model_id: str,
    vllm_gpu_ids: list[int],
    grpo_rollout_cfg: dict,
    host: str = "0.0.0.0",
    port: int = 8000,
    startup_timeout: float = 300.0,
) -> VLLMServerHandle:
    """Launch ``trl vllm-serve`` on the dedicated GPUs and wait for /health.

    Args:
        model_id: The model path or HF id to pass to vLLM. Should match the
            model the trainer is about to load.
        vllm_gpu_ids: Absolute physical GPU indices reserved for vLLM.
        grpo_rollout_cfg: The ``grpo_rollout:`` YAML block. Recognized keys:
            ``tensor_parallel_size`` (default = len(vllm_gpu_ids)), ``dtype``
            (default "bfloat16"), ``gpu_memory_utilization`` (default 0.9),
            ``max_model_len`` (default None, vLLM infers).
        host: Bind address for the vLLM server. "0.0.0.0" makes it reachable
            from other training processes on the same host.
        port: TCP port.
        startup_timeout: Seconds to wait for /health to return 200 before
            giving up.

    Returns:
        A ``VLLMServerHandle`` with the running subprocess. An ``atexit`` hook
        is registered to terminate the process on interpreter shutdown.
    """
    if not vllm_gpu_ids:
        raise ValueError("launch_vllm_server called with empty vllm_gpu_ids")

    if shutil.which("trl") is None:
        raise RuntimeError(
            "`trl` CLI not found on PATH. vLLM server mode requires `trl vllm-serve`. "
            "Make sure trl is installed: `uv sync`."
        )

    tensor_parallel_size = int(
        grpo_rollout_cfg.get("tensor_parallel_size", len(vllm_gpu_ids))
    )
    if tensor_parallel_size > len(vllm_gpu_ids):
        raise ValueError(
            f"grpo_rollout.tensor_parallel_size={tensor_parallel_size} exceeds "
            f"grpo_rollout.dedicated_gpus={len(vllm_gpu_ids)}."
        )
    dtype = str(grpo_rollout_cfg.get("dtype", "bfloat16"))
    gpu_memory_utilization = float(
        grpo_rollout_cfg.get("gpu_memory_utilization", 0.9)
    )
    max_model_len = grpo_rollout_cfg.get("max_model_len")

    cmd: list[Any] = [
        "trl",
        "vllm-serve",
        "--model",
        model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--dtype",
        dtype,
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        cmd += ["--max_model_len", str(max_model_len)]

    # Isolate the server from training GPUs via CUDA_VISIBLE_DEVICES. The
    # training process will get the complementary set set by the driver
    # (see trainer.py::ray_trainer).
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in vllm_gpu_ids)

    logger.info(
        "Launching vLLM server on GPU(s) %s: %s",
        vllm_gpu_ids,
        " ".join(str(x) for x in cmd),
    )

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    handle = VLLMServerHandle(
        process=process,
        host=host if host not in ("0.0.0.0", "") else socket.gethostname(),
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        dedicated_gpu_ids=vllm_gpu_ids,
    )

    # Register cleanup BEFORE waiting for health so a failed startup still
    # gets its process terminated.
    atexit.register(_terminate_server, process)

    _wait_for_health(handle.base_url, timeout=startup_timeout, process=process)
    logger.info("vLLM server healthy at %s (pid=%d)", handle.base_url, process.pid)
    return handle


def _wait_for_health(base_url: str, timeout: float, process: subprocess.Popen) -> None:
    """Poll ``<base_url>/health`` until 200 or timeout."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        # If the subprocess died, bail out with its exit code — no point polling.
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited before becoming healthy "
                f"(exit code {process.returncode}). Check the server logs."
            )
        try:
            resp = requests.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except requests.RequestException as e:
            last_err = e
        time.sleep(2.0)

    # Timed out — clean up the process so we don't leak it
    _terminate_server(process)
    msg = f"vLLM server at {base_url} did not become healthy within {timeout}s"
    if last_err is not None:
        msg += f" (last error: {last_err})"
    raise RuntimeError(msg)


def _terminate_server(process: subprocess.Popen) -> None:
    """Best-effort shutdown of the vLLM server subprocess."""
    if process.poll() is not None:
        return  # already exited
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not exit on SIGTERM, sending SIGKILL")
            process.kill()
            process.wait(timeout=5)
    except Exception as e:  # pragma: no cover - best effort cleanup
        logger.warning("Error terminating vLLM server: %s", e)

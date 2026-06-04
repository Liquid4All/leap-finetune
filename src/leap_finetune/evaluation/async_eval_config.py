"""Config + dispatch for async benchmark evaluation.

Three modes selected via the ``async_eval:`` YAML block:

- ``sync`` (default): in-memory eval shards across ranks; training blocks.
- ``sidecar``: each ``eval_steps`` fires an ``sbatch`` running vLLM on the
  latest checkpoint; training never pauses; results back-fill wandb at
  the originating training step.
- ``reserved``: dedicates N GPUs at job start to a long-running
  ``trl vllm-serve``; a helper thread drives it over HTTP and drains
  results back to wandb. Predictable latency, N GPUs not available
  for training.

All three log the same ``benchmark/<bench>/<metric>`` keys on the global
trainer step axis. ``make_eval_callback`` returns the right callback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VALID_MODES = ("sync", "sidecar", "reserved")
VALID_OVERLAP = ("skip", "queue")
VALID_WEIGHT_RELOAD = ("in_place", "respawn")


@dataclass
class SbatchConfig:
    """SLURM submission settings for ``mode: sidecar``.

    ``partition``/``account`` default to inheriting from the parent
    training job. ``time=None`` lets sbatch use the partition's default
    time limit; set explicitly to cap runaway evals.
    """

    partition: str | None = None
    account: str | None = None
    time: str | None = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ReservedConfig:
    weight_reload: Literal["in_place", "respawn"] = "respawn"
    server_port: int = 8100


@dataclass
class FailureConfig:
    max_consecutive: int = 3
    max_submit_attempts: int = 3
    # Seconds between sbatch retries; exponential ``backoff * 2**(attempt-1)``.
    # ``0.0`` is allowed (immediate burst) but only safe on quiet controllers.
    submit_retry_backoff: float = 2.0


@dataclass
class AsyncEvalConfig:
    """Resolved ``async_eval:`` YAML block."""

    mode: Literal["sync", "sidecar", "reserved"] = "sync"

    # Common (used by sidecar AND reserved; ignored when mode=sync)
    vllm_gpus: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    max_model_len: int | None = None

    sbatch: SbatchConfig = field(default_factory=SbatchConfig)
    reserved: ReservedConfig = field(default_factory=ReservedConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)

    on_overlap: Literal["skip", "queue"] = "skip"

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AsyncEvalConfig":
        """Parse + validate. ``None`` or empty dict returns the sync default."""
        if not raw:
            return cls(mode="sync")

        mode = raw.get("mode", "sync")
        if mode not in VALID_MODES:
            raise ValueError(
                f"async_eval.mode must be one of {VALID_MODES}, got {mode!r}"
            )

        vllm_gpus = int(raw.get("vllm_gpus", 1))
        if mode != "sync" and vllm_gpus < 1:
            raise ValueError(
                f"async_eval.vllm_gpus must be >= 1 for mode={mode!r}, got {vllm_gpus}"
            )

        tp = int(raw.get("tensor_parallel_size", 1))
        if tp > vllm_gpus:
            raise ValueError(
                f"async_eval.tensor_parallel_size ({tp}) cannot exceed vllm_gpus ({vllm_gpus})"
            )

        on_overlap = raw.get("on_overlap", "skip")
        if on_overlap not in VALID_OVERLAP:
            raise ValueError(
                f"async_eval.on_overlap must be one of {VALID_OVERLAP}, got {on_overlap!r}"
            )

        sbatch_raw = raw.get("sbatch", {}) or {}
        sbatch_time = sbatch_raw.get("time")
        sbatch = SbatchConfig(
            partition=sbatch_raw.get("partition"),
            account=sbatch_raw.get("account"),
            time=str(sbatch_time) if sbatch_time else None,
            # ``or []`` (not the dict-default) so an explicit ``null`` /
            # ``~`` in YAML (common when toggling) doesn't TypeError into
            # ``list(None)``.
            extra_args=list(sbatch_raw.get("extra_args") or []),
        )

        reserved_raw = raw.get("reserved", {}) or {}
        weight_reload = reserved_raw.get("weight_reload", "respawn")
        if weight_reload not in VALID_WEIGHT_RELOAD:
            raise ValueError(
                f"async_eval.reserved.weight_reload must be one of "
                f"{VALID_WEIGHT_RELOAD}, got {weight_reload!r}"
            )
        reserved = ReservedConfig(
            weight_reload=weight_reload,
            server_port=int(reserved_raw.get("server_port", 8100)),
        )

        failure_raw = raw.get("failure", {}) or {}
        failure = FailureConfig(
            max_consecutive=int(failure_raw.get("max_consecutive", 3)),
            max_submit_attempts=int(failure_raw.get("max_submit_attempts", 3)),
            submit_retry_backoff=float(failure_raw.get("submit_retry_backoff", 2.0)),
        )

        return cls(
            mode=mode,
            vllm_gpus=vllm_gpus,
            tensor_parallel_size=tp,
            gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9)),
            dtype=str(raw.get("dtype", "bfloat16")),
            max_model_len=raw.get("max_model_len"),
            sbatch=sbatch,
            reserved=reserved,
            failure=failure,
            on_overlap=on_overlap,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for ``train_loop_config`` (Ray workers)."""
        return {
            "mode": self.mode,
            "vllm_gpus": self.vllm_gpus,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "sbatch": {
                "partition": self.sbatch.partition,
                "account": self.sbatch.account,
                "time": self.sbatch.time,
                "extra_args": self.sbatch.extra_args,
            },
            "reserved": {
                "weight_reload": self.reserved.weight_reload,
                "server_port": self.reserved.server_port,
            },
            "failure": {
                "max_consecutive": self.failure.max_consecutive,
                "max_submit_attempts": self.failure.max_submit_attempts,
                "submit_retry_backoff": self.failure.submit_retry_backoff,
            },
            "on_overlap": self.on_overlap,
        }


def make_eval_callback(
    benchmarks: list,
    async_eval_cfg: dict | None,
    *,
    benchmark_configs: dict | None = None,
    server_url: str | None = None,
    eval_gpu_ids: str = "",
    output_dir: str | None = None,
    wandb_run_id: str | None = None,
    config_dir: str | None = None,
):
    """Return the right ``TrainerCallback`` for the configured mode.

    Mode-specific kwargs:
    - sidecar: ``benchmark_configs`` (re-constructs benchmarks inside the
      eval job), ``output_dir`` (marker + checkpoint staging),
      ``wandb_run_id`` (attach to the training run), ``config_dir``.
    - reserved: ``server_url`` (set by the driver after launching the
      vLLM server), ``eval_gpu_ids``.
    """
    cfg = AsyncEvalConfig.from_dict(async_eval_cfg)

    if cfg.mode == "sync":
        # Lazy import so consumers that don't need sync don't pay.
        from leap_finetune.evaluation.callback import BenchmarkEvalCallback

        return BenchmarkEvalCallback(benchmarks)

    if cfg.mode == "sidecar":
        from leap_finetune.evaluation.sidecar_callback import SidecarEvalCallback

        return SidecarEvalCallback(
            benchmarks=benchmarks,
            cfg=cfg,
            benchmark_configs=benchmark_configs,
            output_dir=output_dir,
            wandb_run_id=wandb_run_id,
            config_dir=config_dir,
        )

    if cfg.mode == "reserved":
        from leap_finetune.evaluation.reserved_callback import ReservedEvalCallback

        if not server_url:
            raise RuntimeError(
                "async_eval mode=reserved requires a vLLM server_url; the "
                "driver must launch the server before constructing the callback."
            )
        return ReservedEvalCallback(
            benchmarks=benchmarks,
            cfg=cfg,
            server_url=server_url,
            output_dir=output_dir,
            eval_gpu_ids=eval_gpu_ids,
        )

    raise ValueError(f"Unknown async_eval.mode={cfg.mode!r}")

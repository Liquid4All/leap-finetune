"""Render an sbatch script that runs ``async_runner_main`` for one eval
cycle. The script clears the ``.in_flight`` marker and the staging
checkpoint on exit (any signal) so a crashed runner can't block the
callback.
"""

from __future__ import annotations

import pathlib
import shlex
from dataclasses import dataclass

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


@dataclass
class SidecarSubmission:
    script_path: pathlib.Path
    log_out: pathlib.Path
    log_err: pathlib.Path


def render_sbatch_script(
    *,
    output_dir: pathlib.Path,
    trigger_step: int,
    checkpoint_path: pathlib.Path,
    benchmark_configs_json: pathlib.Path,
    modality: str,
    wandb_run_id: str | None,
    wandb_project: str | None,
    job_name: str,
    vllm_gpus: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    max_model_len: int | None,
    sbatch_partition: str | None,
    sbatch_account: str | None,
    sbatch_time: str | None,
    sbatch_extra_args: list[str],
) -> SidecarSubmission:
    """Write the script to ``output_dir/_async_eval/scripts/step_<N>.sh``."""
    eval_dir = output_dir / "_async_eval"
    scripts_dir = eval_dir / "scripts"
    logs_dir = eval_dir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    script_path = scripts_dir / f"step_{trigger_step}.sh"
    log_out = logs_dir / f"step_{trigger_step}.out"
    log_err = logs_dir / f"step_{trigger_step}.err"
    marker = eval_dir / ".in_flight"

    runner_args = [
        "--checkpoint",
        str(checkpoint_path),
        "--benchmark-configs",
        str(benchmark_configs_json),
        "--modality",
        modality,
        "--trigger-step",
        str(trigger_step),
        "--output-dir",
        str(output_dir),
        "--vllm-gpus",
        str(vllm_gpus),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]
    if max_model_len is not None:
        runner_args += ["--max-model-len", str(max_model_len)]
    if wandb_run_id:
        runner_args += ["--wandb-run-id", wandb_run_id]
    if wandb_project:
        runner_args += ["--wandb-project", wandb_project]

    runner_cmd = (
        "uv run python -m leap_finetune.evaluation.async_runner_main \\\n    "
        + " \\\n    ".join(shlex.quote(a) for a in runner_args)
    )

    sbatch_directives = [
        f"#SBATCH --job-name={shlex.quote(job_name)}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --gpus-per-task={vllm_gpus}",
        "#SBATCH --cpus-per-gpu=8",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
    ]
    if sbatch_time:
        sbatch_directives.append(f"#SBATCH --time={sbatch_time}")
    if sbatch_partition:
        sbatch_directives.append(f"#SBATCH --partition={sbatch_partition}")
    if sbatch_account:
        sbatch_directives.append(f"#SBATCH --account={sbatch_account}")
    for extra in sbatch_extra_args:
        sbatch_directives.append(f"#SBATCH {extra}")

    directives_block = "\n".join(sbatch_directives)

    script = f"""#!/bin/bash
{directives_block}

# Clear the in-flight marker + this cycle's staging checkpoint on exit
# (success or crash) so a failed runner can't block the next eval.
trap 'rm -f {shlex.quote(str(marker))}; rm -rf {shlex.quote(str(checkpoint_path))}' EXIT INT TERM HUP

# Wandb IPC vars from the parent point at a Unix socket on a different node.
unset WANDB_SERVICE WANDB__SERVICE_TRANSPORT WANDB_RUN_GROUP

cd {shlex.quote(str(LEAP_FINETUNE_DIR))}

# Cluster-specific CUDA module — set $LEAP_CUDA_MODULE to load yours.
if [ -n "${{LEAP_CUDA_MODULE:-}}" ] && command -v module >/dev/null 2>&1; then
    module load "$LEAP_CUDA_MODULE" 2>/dev/null || true
fi

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# User-level secrets (WANDB_API_KEY, HF_TOKEN, HF_HOME, ...).
if [ -f "$HOME/.env" ]; then
    source "$HOME/.env"
fi

# Avoid fork-after-CUDA-init in vLLM v1 EngineCore (causes intermittent
# "CUDA unknown error - setting available devices to zero" on engine boot).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

{runner_cmd}
"""

    script_path.write_text(script)
    script_path.chmod(0o755)

    return SidecarSubmission(
        script_path=script_path,
        log_out=log_out,
        log_err=log_err,
    )

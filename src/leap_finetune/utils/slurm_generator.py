import pathlib
from typing import Any

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


def _default_gpus_per_task(config_dict: dict[str, Any]) -> int:
    training_config = config_dict.get("training_config", {})
    if config_dict.get("training_type") not in ("grpo", "vlm_grpo"):
        return 1

    rollout = config_dict.get("grpo_rollout") or {}
    judge_gpus = _default_judge_gpus(config_dict)

    if training_config.get("vllm_mode") != "server":
        training_gpus = int(rollout.get("training_gpus", 1))
        return max(1, judge_gpus + training_gpus)

    if "server_gpus" in rollout:
        server_gpus = int(rollout["server_gpus"])
    elif "dedicated_gpus" in rollout:
        server_gpus = int(rollout["dedicated_gpus"])
    elif "training_gpus" in rollout:
        server_gpus = 1
    else:
        server_gpus = 1

    if server_gpus == 0:
        return max(1, judge_gpus + int(rollout.get("training_gpus", 1)))

    training_gpus = int(rollout.get("training_gpus", 1))
    return max(1, server_gpus + judge_gpus + training_gpus)


def _default_judge_gpus(config_dict: dict[str, Any]) -> int:
    rewards = config_dict.get("rewards")
    if not isinstance(rewards, dict):
        return 0
    judge = rewards.get("judge")
    if not judge:
        return 0
    if isinstance(judge, dict) and judge.get("base_url"):
        return 0
    rollout = config_dict.get("grpo_rollout") or {}
    return int(rollout.get("judge_gpus", 1))


def _gpus_per_node(slurm_settings: dict[str, Any]) -> int:
    if "gpus_per_node" in slurm_settings:
        return int(slurm_settings["gpus_per_node"])
    return int(slurm_settings["gpus_per_task"]) * int(slurm_settings["ntasks_per_node"])


def generate_slurm_script(
    config_path: pathlib.Path,
    config_dict: dict[str, Any],
    output_dir: pathlib.Path | None = None,
    auto_submit: bool = False,
) -> pathlib.Path:
    slurm_config = config_dict.get("slurm", {})

    defaults = {
        "job_name": config_dict.get("project_name", "leap_finetune"),
        "nodes": 1,
        "ntasks_per_node": 1,
        "gpus_per_task": _default_gpus_per_task(config_dict),
        "cpus_per_gpu": 8,
        "output": "logs/OUT_%x.%j",
        "error": "logs/ERR_%x.%j",
    }

    slurm_settings = {**defaults, **slurm_config}

    if output_dir is None:
        output_dir = config_path.parent
    else:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    config_name = config_path.stem
    script_path = output_dir / f"{config_name}.sh"

    if script_path.exists() and not auto_submit:
        print(f"SLURM script already exists: {script_path}")
        print("   Skipping regeneration. Delete it to regenerate.")
        return script_path

    project_root = LEAP_FINETUNE_DIR

    config_relative_path = (
        config_path.relative_to(project_root)
        if config_path.is_relative_to(project_root)
        else config_path
    )

    script_content = f"""#!/bin/bash

#SBATCH --job-name={slurm_settings["job_name"]}
#SBATCH --nodes={slurm_settings["nodes"]}
#SBATCH --ntasks-per-node={slurm_settings["ntasks_per_node"]}
"""
    if "gpus_per_node" in slurm_settings:
        script_content += f"""#SBATCH --gpus-per-node={slurm_settings["gpus_per_node"]}
"""
    else:
        script_content += f"""#SBATCH --gpus-per-task={slurm_settings["gpus_per_task"]}
"""

    script_content += f"""#SBATCH --output={slurm_settings["output"]}
#SBATCH --error={slurm_settings["error"]}
#SBATCH --cpus-per-gpu={slurm_settings["cpus_per_gpu"]}
"""

    additional_directives = slurm_config.get("directives", [])
    for directive in additional_directives:
        script_content += f"#SBATCH {directive}\n"

    setup_commands = slurm_config.get("setup_commands", [])
    setup_block = "\n".join(setup_commands) + "\n" if setup_commands else ""
    nodes = int(slurm_settings["nodes"])

    ray_setup_block = ""
    if nodes > 1:
        expected_gpus = nodes * _gpus_per_node(slurm_settings)
        ray_setup_block = f"""
source job_configs/slurms/utils/slurm_ray.sh
ray_slurm_init {nodes} {_gpus_per_node(slurm_settings)}
ray_slurm_export_dist_env
trap ray_slurm_stop_cluster EXIT
ray_slurm_start_cluster_bg
ray_slurm_wait_ready {nodes} {expected_gpus}
"""

    script_content += f"""
cd {project_root}

set -euo pipefail

source .venv/bin/activate

{setup_block}# Set flag to prevent recursive SLURM submission
export LEAP_FINETUNE_FROM_SLURM=1

{ray_setup_block}
uv run leap-finetune {config_relative_path}

echo "================================================"
echo "RUN DONE"
echo "================================================"
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    print(f"Generated SLURM script: {script_path}")

    if auto_submit:
        import subprocess

        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Submitted job: {result.stdout.strip()}")
        else:
            print(f"Failed to submit job: {result.stderr}")

    return script_path

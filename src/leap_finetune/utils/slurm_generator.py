import pathlib
from typing import Any

from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


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
        "gpus_per_task": 1,
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
#SBATCH --gpus-per-task={slurm_settings["gpus_per_task"]}
#SBATCH --output={slurm_settings["output"]}
#SBATCH --error={slurm_settings["error"]}
#SBATCH --cpus-per-gpu={slurm_settings["cpus_per_gpu"]}
"""

    additional_directives = slurm_config.get("directives", [])
    for directive in additional_directives:
        script_content += f"#SBATCH {directive}\n"

    script_content += f"""
cd {project_root}

source .venv/bin/activate

# Set flag to prevent recursive SLURM submission
export LEAP_FINETUNE_FROM_SLURM=1

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

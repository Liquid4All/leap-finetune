import argparse
import pathlib
import subprocess
import sys

import yaml


def check_and_handle_slurm(config_path_arg: str | None) -> bool:
    if not config_path_arg:
        return False

    import os

    if os.environ.get("LEAP_FINETUNE_FROM_SLURM") == "1":
        return False

    from leap_finetune.config.parser import resolve_config_path

    try:
        config_path = resolve_config_path(config_path_arg)
    except FileNotFoundError:
        return False

    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    slurm_config = config_dict.get("slurm")
    if not slurm_config:
        return False

    from leap_finetune.distribution.backends.slurm import generate_slurm_script

    output_dir = config_path.parent / "slurms"
    script_path = output_dir / f"{config_path.stem}.sh"

    if script_path.exists():
        print(f"Config contains SLURM settings - using existing script: {script_path}")
    else:
        print("Config contains SLURM settings - generating SLURM script...")
        script_path = generate_slurm_script(
            config_path, config_dict, output_dir, auto_submit=False
        )

    print("Submitting SLURM job...")
    result = subprocess.run(
        ["sbatch", str(script_path)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to submit job: {result.stderr}")
        sys.exit(1)

    print(f"SLURM job submitted: {result.stdout.strip()}")
    return True


def _parse_cli_args():
    command = None
    config_path_arg = None
    output_dir_arg = None

    if len(sys.argv) > 1:
        if sys.argv[1] == "slurm":
            command = "slurm"
            parser = argparse.ArgumentParser(description="Generate SLURM script")
            parser.add_argument("command", choices=["slurm"])
            parser.add_argument("config_path", help="Path to YAML job config file")
            parser.add_argument(
                "--output-dir",
                "-o",
                help="Directory to save SLURM script",
                default=None,
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
            output_dir_arg = args.output_dir
        elif sys.argv[1] == "run":
            command = "run"
            parser = argparse.ArgumentParser(description="Run training job")
            parser.add_argument("command", choices=["run"])
            parser.add_argument(
                "config_path", help="Path to YAML job config file", nargs="?"
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
        else:
            config_path_arg = sys.argv[1]

    return command, config_path_arg, output_dir_arg


def _generate_slurm_script(config_path_arg: str | None, output_dir_arg: str | None):
    from leap_finetune.distribution.backends.slurm import generate_slurm_script
    from leap_finetune.config.parser import resolve_config_path

    if not config_path_arg:
        print("No config file provided.")
        print("Usage: leap-finetune slurm <path_to_config.yaml>")
        sys.exit(1)

    config_path = resolve_config_path(config_path_arg)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    if output_dir_arg:
        output_dir = pathlib.Path(output_dir_arg)
    else:
        output_dir = config_path.parent / "slurms"

    generate_slurm_script(config_path, config_dict, output_dir, auto_submit=False)


def _assert_local_cuda_available() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    except (ImportError, RuntimeError):
        print("Local training requires GPU dependencies and CUDA.")
        print(
            "For remote execution, add a 'modal:' or 'slurm:' section to your config."
        )
        sys.exit(1)


def run_config(config_path: str | pathlib.Path) -> None:
    """Launch a training job from a YAML config path.

    This is the programmatic equivalent of `leap-finetune <config>`. It keeps
    the same backend dispatch behavior: configs with `slurm`, `kuberay`, or
    `modal` sections submit remotely; other configs launch local Ray training.
    """
    config_path_arg = str(config_path)
    # === Remote backend dispatch ===
    # Keep this path light: Slurm/Modal submission should not import torch, Ray,
    # PEFT, or datasets unless we are actually launching local training.
    if check_and_handle_slurm(config_path_arg):
        return

    from leap_finetune.distribution.backends.kuberay import check_and_handle_kuberay

    if check_and_handle_kuberay(config_path_arg):
        return

    from leap_finetune.distribution.backends.modal import check_and_handle_modal

    if check_and_handle_modal(config_path_arg):
        return

    _assert_local_cuda_available()

    # Heavy imports deferred to here to keep remote-submit codepaths fast.
    from leap_finetune.data_loading.dataset_loader import DatasetLoader
    from leap_finetune.distribution.ray_trainer import ray_trainer
    from leap_finetune.training.utils.logging import setup_training_environment
    from leap_finetune.config.parser import (
        parse_job_config,
        print_job_config_summary,
    )

    setup_training_environment()

    print("Launching leap-finetune")

    try:
        job_config = parse_job_config(config_path_arg)
        print_job_config_summary(job_config)

        if isinstance(job_config.dataset, DatasetLoader):
            job_config.dataset.quick_validate()

        job_config_dict = job_config.to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path_arg}")
    except Exception as e:
        raise ValueError(f"Issue parsing configuration: {e}") from e

    ray_trainer(job_config_dict)


def main() -> None:
    command, config_path_arg, output_dir_arg = _parse_cli_args()

    if command == "slurm":
        _generate_slurm_script(config_path_arg, output_dir_arg)
        return

    if not config_path_arg:
        print("No config file provided. Please provide a path to a YAML config file.")
        print("Usage: leap-finetune <path_to_config.yaml>")
        print("   or: leap-finetune slurm <path_to_config.yaml>")
        sys.exit(1)

    run_config(config_path_arg)

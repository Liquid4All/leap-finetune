import argparse
import os
import pathlib
import subprocess
import sys

import yaml


def check_and_handle_slurm(config_path_arg: str) -> bool:
    if not config_path_arg:
        return False

    is_from_leap_slurm = os.environ.get("LEAP_FINETUNE_FROM_SLURM") == "1"
    if is_from_leap_slurm:
        return False

    try:
        from leap_finetune.utils.config_resolver import resolve_config_path

        config_path = resolve_config_path(config_path_arg)
    except (FileNotFoundError, Exception):
        return False

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        slurm_config = config_dict.get("slurm")
        if not slurm_config:
            return False

        from leap_finetune.utils.slurm_generator import generate_slurm_script

        output_dir = config_path.parent / "slurms"
        script_path = output_dir / f"{config_path.stem}.sh"

        if script_path.exists():
            print(
                f"Config contains SLURM settings - using existing script: {script_path}"
            )
        else:
            print("Config contains SLURM settings - generating SLURM script...")
            script_path = generate_slurm_script(
                config_path, config_dict, output_dir, auto_submit=False
            )

        print("Submitting SLURM job...")
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"SLURM job submitted: {result.stdout.strip()}")
        else:
            print(f"Failed to submit job: {result.stderr}")
            sys.exit(1)

        return True
    except Exception as e:
        print(f"Error checking SLURM config: {e}")
        return False


def main() -> None:
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

    if command == "slurm":
        from leap_finetune.utils.config_resolver import resolve_config_path
        from leap_finetune.utils.slurm_generator import generate_slurm_script

        if not config_path_arg:
            print("No config file provided.")
            print("Usage: leap-finetune slurm <path_to_config.yaml>")
            sys.exit(1)

        config_path = resolve_config_path(config_path_arg)
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        if output_dir_arg:
            output_dir = pathlib.Path(output_dir_arg)
        else:
            output_dir = config_path.parent / "slurms"

        generate_slurm_script(config_path, config_dict, output_dir, auto_submit=False)
        return

    # Check for SLURM / Modal config BEFORE importing heavy dependencies
    if check_and_handle_slurm(config_path_arg):
        return

    from leap_finetune.backends.modal_backend import check_and_handle_modal

    if check_and_handle_modal(config_path_arg):
        return

    if not config_path_arg:
        print("No config file provided. Please provide a path to a YAML config file.")
        print("Usage: leap-finetune <path_to_config.yaml>")
        print("   or: leap-finetune slurm <path_to_config.yaml>")
        sys.exit(1)

    # === Guard: local training requires GPU deps + CUDA ===
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

    # Heavy imports deferred to here to keep slurm/modal codepath fast
    # (these transitively load torch, ray, peft, datasets, etc.)
    from leap_finetune.data_loaders.dataset_loader import DatasetLoader
    from leap_finetune.trainer import ray_trainer
    from leap_finetune.utils.config_parser import parse_job_config
    from leap_finetune.utils.logging_utils import setup_training_environment

    setup_training_environment()

    print("Launching leap-finetune")

    try:
        job_config = parse_job_config(config_path_arg)
        job_config.print_config_summary()

        # Validate dataset schema before starting Ray (fast, ~10 samples)
        if isinstance(job_config.dataset, DatasetLoader):
            job_config.dataset.quick_validate()

        job_config_dict = job_config.to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path_arg}")
    except Exception as e:
        raise ValueError(f"Issue parsing configuration: {e}")

    ray_trainer(job_config_dict)

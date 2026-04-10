import argparse
import os
import pathlib
import subprocess
import sys

import yaml


# === SLURM auto-dispatch (checks config for slurm section) ===


def _check_and_handle_slurm(config_path_arg: str) -> bool:
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


# === Subcommand handlers ===


def _cmd_slurm(args: argparse.Namespace) -> None:
    from leap_finetune.utils.config_resolver import resolve_config_path
    from leap_finetune.utils.slurm_generator import generate_slurm_script

    config_path = resolve_config_path(args.config_path)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        output_dir = config_path.parent / "slurms"

    generate_slurm_script(config_path, config_dict, output_dir, auto_submit=False)


def _cmd_export(args: argparse.Namespace) -> None:
    from leap_finetune.quantization.gguf_export import (
        ADAPTER_QUANTS,
        ALL_QUANTS,
        QUANTIZE_QUANTS,
        export_gguf,
        is_adapter_path,
        validate_model_path,
    )

    model_path = pathlib.Path(args.model_path).resolve()
    validate_model_path(model_path)

    invalid = set(args.quant) - ALL_QUANTS
    if invalid:
        print(f"Unknown quant type(s): {', '.join(sorted(invalid))}")
        print(f"Valid types: {', '.join(sorted(ALL_QUANTS))}")
        sys.exit(1)

    adapter = is_adapter_path(model_path)
    if adapter and not args.base_model:
        print(
            "LoRA adapter detected. --base-model is required for adapter exports.\n"
            "Example: leap-finetune export ./adapter --quant F16 "
            "--base-model LFM2.5-1.2B-Instruct"
        )
        sys.exit(1)

    if adapter:
        unsupported = set(args.quant) - ADAPTER_QUANTS
        if unsupported:
            print(f"Adapter exports only support: {', '.join(sorted(ADAPTER_QUANTS))}")
            print(f"Unsupported type(s): {', '.join(sorted(unsupported))}")
            print(
                "For K-quants, merge the adapter first, then export the merged model."
            )
            sys.exit(1)

    # --llama-cpp-dir only needed for K-quant/legacy quant types
    needs_binary = set(args.quant) & QUANTIZE_QUANTS
    if needs_binary and not args.llama_cpp_dir and not os.environ.get("LLAMA_CPP_DIR"):
        print(
            f"K-quant types ({', '.join(sorted(needs_binary))}) require llama-quantize.\n"
            "Use --llama-cpp-dir or set LLAMA_CPP_DIR.\n\n"
            "  git clone https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && cmake -B build && cmake --build build --config Release"
        )
        sys.exit(1)

    output_dir = pathlib.Path(args.output).resolve() if args.output else model_path

    results = export_gguf(
        model_path,
        args.quant,
        output_dir,
        args.base_model,
        args.llama_cpp_dir,
    )
    print(f"\nExported {len(results)} GGUF file(s):")
    for path in results:
        size_gb = path.stat().st_size / 1e9
        print(f"  {path} ({size_gb:.2f} GB)")


def _cmd_train(config_path_arg: str) -> None:
    # Check for SLURM / Modal config BEFORE importing heavy dependencies
    if _check_and_handle_slurm(config_path_arg):
        return

    from leap_finetune.backends.modal_backend import check_and_handle_modal

    if check_and_handle_modal(config_path_arg):
        return

    from leap_finetune.backends.kuberay_backend import check_and_handle_kuberay

    if check_and_handle_kuberay(config_path_arg):
        return

    # === Guard: local training requires GPU deps + CUDA ===
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    except (ImportError, RuntimeError):
        print("Local training requires GPU dependencies and CUDA.")
        print(
            "For remote execution, add a 'modal:', 'slurm:', or 'kuberay:' section to your config."
        )
        sys.exit(1)

    # Heavy imports deferred to here to keep slurm/modal/export codepath fast
    from leap_finetune.data_loaders.dataset_loader import DatasetLoader
    from leap_finetune.trainer import ray_trainer
    from leap_finetune.utils.config_parser import parse_job_config
    from leap_finetune.utils.logging_utils import setup_training_environment

    setup_training_environment()

    print("Launching leap-finetune")

    try:
        job_config = parse_job_config(config_path_arg)
        job_config.print_config_summary()

        if isinstance(job_config.dataset, DatasetLoader):
            job_config.dataset.quick_validate()

        job_config_dict = job_config.to_dict()
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path_arg}")
    except Exception as e:
        raise ValueError(f"Issue parsing configuration: {e}")

    ray_trainer(job_config_dict)


# === CLI entry point ===

SUBCOMMANDS = {"slurm", "export", "run"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leap-finetune",
        description="Fine-tune LFM models with LEAP",
    )
    subparsers = parser.add_subparsers(dest="command")

    # === slurm ===
    slurm_parser = subparsers.add_parser(
        "slurm", help="Generate a SLURM submission script"
    )
    slurm_parser.add_argument("config_path", help="Path to YAML job config file")
    slurm_parser.add_argument(
        "--output-dir", "-o", help="Directory to save SLURM script", default=None
    )

    # === export ===
    export_parser = subparsers.add_parser(
        "export", help="Export model or LoRA adapter to GGUF format"
    )
    export_parser.add_argument(
        "model_path", help="Path to HuggingFace model or PEFT adapter"
    )
    export_parser.add_argument(
        "--quant",
        nargs="+",
        default=["Q4_K_M"],
        help="Quantization type(s) (default: Q4_K_M)",
    )
    export_parser.add_argument(
        "--output", "-o", default=None, help="Output directory (default: same as model)"
    )
    export_parser.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="Path to llama.cpp (only needed for K-quant types, or set LLAMA_CPP_DIR)",
    )
    export_parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path (required for LoRA adapter exports)",
    )

    # === run (explicit, rarely used) ===
    run_parser = subparsers.add_parser("run", help="Run training job")
    run_parser.add_argument(
        "config_path", help="Path to YAML job config file", nargs="?"
    )

    return parser


def main() -> None:
    # If the first arg is not a known subcommand, treat it as a config path
    # This preserves `leap-finetune config.yaml` (no subcommand) as the default
    if (
        len(sys.argv) > 1
        and sys.argv[1] not in SUBCOMMANDS
        and sys.argv[1] != "-h"
        and sys.argv[1] != "--help"
    ):
        _cmd_train(sys.argv[1])
        return

    if len(sys.argv) == 1:
        print("Usage: leap-finetune <path_to_config.yaml>")
        print("   or: leap-finetune slurm <path_to_config.yaml>")
        print("   or: leap-finetune export <model_path> [--quant Q4_K_M]")
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "slurm":
        _cmd_slurm(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "run":
        if not args.config_path:
            print("No config file provided.")
            print("Usage: leap-finetune run <path_to_config.yaml>")
            sys.exit(1)
        _cmd_train(args.config_path)
    else:
        parser.print_help()

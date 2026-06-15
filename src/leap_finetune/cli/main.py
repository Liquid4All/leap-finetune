import argparse
import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _load_config_dict(config_path: pathlib.Path) -> dict:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config_dict


def _resolve_config_path_light(config_input: str | pathlib.Path) -> pathlib.Path:
    input_path = pathlib.Path(config_input)
    candidates = [str(input_path)]
    if not input_path.suffix:
        candidates.append(f"{input_path}.yaml")

    for candidate in candidates:
        candidate_path = pathlib.Path(candidate).expanduser()
        if candidate_path.exists():
            return candidate_path.resolve()

        local_job_config = pathlib.Path.cwd() / "job_configs" / candidate
        if local_job_config.exists():
            return local_job_config.resolve()

        repo_job_config = _REPO_ROOT / "job_configs" / candidate
        if repo_job_config.exists():
            return repo_job_config.resolve()

    raise FileNotFoundError(f"Config file not found at: {input_path}")


def _load_config_for_remote_dispatch(
    config_path_arg: str,
) -> tuple[pathlib.Path, dict] | tuple[None, None]:
    try:
        config_path = _resolve_config_path_light(config_path_arg)
    except FileNotFoundError:
        return None, None
    return config_path, _load_config_dict(config_path)


def _parse_cli_args():
    command = None
    config_path_arg = None
    output_arg = None

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
            output_arg = args.output_dir
        elif sys.argv[1] == "run":
            parser = argparse.ArgumentParser(
                description="Run a training or standalone eval config"
            )
            parser.add_argument("command", choices=["run"])
            parser.add_argument(
                "config_path", help="Path to YAML config file", nargs="?"
            )
            parser.add_argument(
                "--output",
                "-o",
                help="Optional JSON metrics output path for standalone eval configs",
                default=None,
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
            output_arg = args.output
        else:
            parser = argparse.ArgumentParser(
                description="Run a training or standalone eval config"
            )
            parser.add_argument("config_path", help="Path to YAML config file")
            parser.add_argument(
                "--output",
                "-o",
                help="Optional JSON metrics output path for standalone eval configs",
                default=None,
            )
            args = parser.parse_args()
            config_path_arg = args.config_path
            output_arg = args.output

    return command, config_path_arg, output_arg


def _generate_slurm_script(config_path_arg: str | None, output_dir_arg: str | None):
    from leap_finetune.distribution.backends.slurm import generate_slurm_script

    if not config_path_arg:
        print("No config file provided.")
        print("Usage: leap-finetune slurm <path_to_config.yaml>")
        sys.exit(1)

    config_path = _resolve_config_path_light(config_path_arg)
    config_dict = _load_config_dict(config_path)

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


def check_and_handle_slurm(
    config_path_arg: str | None = None,
    *,
    config_dict: dict | None = None,
) -> bool:
    from leap_finetune.distribution.backends.slurm import (
        check_and_handle_slurm as _impl,
    )

    return _impl(config_path_arg, config_dict=config_dict)


def _dispatch_remote_backends(
    config_path_arg: str | None,
    *,
    config_dict: dict | None,
) -> bool:
    # Keep this path light: remote submission should not import torch, Ray,
    # PEFT, datasets, or training defaults.
    if check_and_handle_slurm(config_path_arg, config_dict=config_dict):
        return True

    from leap_finetune.distribution.backends.kuberay import check_and_handle_kuberay

    if check_and_handle_kuberay(config_path_arg, config_dict=config_dict):
        return True

    from leap_finetune.distribution.backends.modal import check_and_handle_modal

    return check_and_handle_modal(config_path_arg, config_dict=config_dict)


def run_config(config_path, *, output_path: str | pathlib.Path | None = None):
    """Launch a training job or standalone eval from a config path/model.

    This is the programmatic equivalent of `leap-finetune <config>`. Training
    configs keep the same backend dispatch behavior: configs with `slurm`,
    `kuberay`, or `modal` sections submit remotely; other training configs
    launch local Ray training. Eval-only configs run benchmarks without
    starting training.
    """
    config_path_arg = None
    preloaded_config_dict = None
    if isinstance(config_path, (str, pathlib.Path)):
        config_path_arg = str(config_path)
        dispatch_path, dispatch_config = _load_config_for_remote_dispatch(
            config_path_arg
        )
        if dispatch_config is not None:
            config_path_arg = str(dispatch_path)
            preloaded_config_dict = dispatch_config
            if _dispatch_remote_backends(
                config_path_arg,
                config_dict=dispatch_config,
            ):
                return

    from leap_finetune.config import EvalRunConfig, JobConfig
    from leap_finetune.config.parser import (
        materialize_job_config,
        normalized_job_config_dict,
        parse_eval_config,
        parse_job_config,
        print_job_config_summary,
    )

    parsed_job = None
    config_dict = preloaded_config_dict
    if isinstance(config_path, EvalRunConfig):
        from leap_finetune.evaluation.runner import run_eval_config as _run_eval_config

        return _run_eval_config(config_path, output_path=output_path)
    if isinstance(config_path, JobConfig):
        parsed_job = config_path
        config_dict = normalized_job_config_dict(parsed_job)
    else:
        config_path_arg = config_path_arg or str(config_path)

    # === Remote backend dispatch ===
    if _dispatch_remote_backends(config_path_arg, config_dict=config_dict):
        return

    if parsed_job is None:
        try:
            eval_config = parse_eval_config(config_path_arg)
        except Exception:
            eval_config = None
        else:
            from leap_finetune.evaluation.runner import (
                run_eval_config as _run_eval_config,
            )

            return _run_eval_config(eval_config, output_path=output_path)

    _assert_local_cuda_available()

    # Heavy imports deferred to here to keep remote-submit codepaths fast.
    from leap_finetune.data_loading.dataset_loader import DatasetLoader
    from leap_finetune.distribution.ray_trainer import ray_trainer
    from leap_finetune.training.utils.logging import setup_training_environment

    setup_training_environment()

    print("Launching leap-finetune")

    try:
        if parsed_job is None:
            parsed_job = parse_job_config(config_path_arg)
        job_config = materialize_job_config(parsed_job)
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
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        from leap_finetune.cli.env import main as env_main

        sys.exit(env_main(sys.argv[2:]))

    command, config_path_arg, output_arg = _parse_cli_args()

    if command == "slurm":
        _generate_slurm_script(config_path_arg, output_arg)
        return

    if not config_path_arg:
        print("No config file provided. Please provide a path to a YAML config file.")
        print("Usage: leap-finetune <path_to_config.yaml>")
        print("   or: leap-finetune <path_to_eval_config.yaml> --output results.json")
        print("   or: leap-finetune slurm <path_to_config.yaml>")
        print("   or: leap-finetune env fa2-status")
        sys.exit(1)

    result = run_config(config_path_arg, output_path=output_arg)
    if result is not None:
        print(yaml.safe_dump(result, sort_keys=True))

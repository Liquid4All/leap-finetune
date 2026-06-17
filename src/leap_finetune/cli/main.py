import argparse
import os
import pathlib
import sys
from contextlib import contextmanager
from typing import Any

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
    command_args = {}

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
        elif sys.argv[1] == "runs":
            command = "runs"
            parser = argparse.ArgumentParser(description="Inspect local LFT runs")
            parser.add_argument("command", choices=["runs"])
            parser.add_argument(
                "action",
                choices=["list", "show", "sync"],
                nargs="?",
                default="list",
            )
            parser.add_argument("run_id", nargs="?")
            args = parser.parse_args()
            command_args = vars(args)
        elif sys.argv[1] == "memory":
            command = "memory"
            parser = argparse.ArgumentParser(description="Inspect local LFT memory")
            parser.add_argument("command", choices=["memory"])
            parser.add_argument(
                "action",
                choices=["show", "add"],
                nargs="?",
                default="show",
            )
            parser.add_argument("text", nargs="?")
            parser.add_argument("--ref", help="Run or eval ID this note refers to")
            args = parser.parse_args()
            command_args = vars(args)
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

    return command, config_path_arg, output_arg, command_args


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
) -> bool | dict[str, Any]:
    from leap_finetune.distribution.backends.slurm import (
        check_and_handle_slurm as _impl,
    )

    return _impl(config_path_arg, config_dict=config_dict)


def _dispatch_remote_backends(
    config_path_arg: str | None,
    *,
    config_dict: dict | None,
) -> dict[str, Any] | None:
    # Keep this path light: remote submission should not import torch, Ray,
    # PEFT, datasets, or training defaults.
    result = _normalize_backend_result(
        "slurm",
        check_and_handle_slurm(config_path_arg, config_dict=config_dict),
    )
    if result:
        return result

    from leap_finetune.distribution.backends.kuberay import check_and_handle_kuberay

    result = _normalize_backend_result(
        "kuberay",
        check_and_handle_kuberay(config_path_arg, config_dict=config_dict),
    )
    if result:
        return result

    from leap_finetune.distribution.backends.modal import check_and_handle_modal

    return _normalize_backend_result(
        "modal",
        check_and_handle_modal(config_path_arg, config_dict=config_dict),
    )


def _normalize_backend_result(
    backend: str,
    result: bool | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not result:
        return None
    if isinstance(result, dict):
        return {"backend": backend, "status": "submitted", **result}
    return {"backend": backend, "status": "submitted"}


def _backend_id(result: dict[str, Any]) -> str | None:
    for key in ("job_id", "app_id", "job_name", "call_id"):
        value = result.get(key)
        if value:
            return str(value)
    return None


def _record_remote_submission(tracker, result: dict[str, Any]) -> None:
    tracker.update(
        status=str(result.get("status") or "submitted"),
        backend=str(result["backend"]),
        backend_id=_backend_id(result),
    )


@contextmanager
def _run_id_env(run_id: str):
    previous = os.environ.get("LFT_RUN_ID")
    os.environ["LFT_RUN_ID"] = run_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LFT_RUN_ID", None)
        else:
            os.environ["LFT_RUN_ID"] = previous


def _run_model_dump(config) -> dict[str, Any]:
    return config.model_dump(by_alias=True, exclude_none=True)


def _fail_tracker(tracker, exc: BaseException) -> None:
    if tracker is None:
        return
    try:
        tracker.failed(exc)
    except Exception:
        pass


def run_config(config_path, *, output_path: str | pathlib.Path | None = None):
    """Launch a training job or standalone eval from a config path/model.

    This is the programmatic equivalent of `leap-finetune <config>`. Training
    configs keep the same backend dispatch behavior: configs with `slurm`,
    `kuberay`, or `modal` sections submit remotely; other training configs
    launch local Ray training. Eval-only configs run benchmarks without
    starting training.
    """
    from leap_finetune.state import RunTracker

    tracker = None
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
            tracker = RunTracker.start(
                config_path=config_path_arg,
                config_dict=dispatch_config,
                output_path=output_path,
            )
            try:
                with _run_id_env(tracker.id):
                    remote_result = _dispatch_remote_backends(
                        config_path_arg,
                        config_dict=dispatch_config,
                    )
            except BaseException as exc:
                _fail_tracker(tracker, exc)
                raise
            if remote_result:
                _record_remote_submission(tracker, remote_result)
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
        tracker = RunTracker.start(
            config_path=None,
            config_dict=_run_model_dump(config_path),
            output_path=output_path,
        )
        from leap_finetune.evaluation.runner import run_eval_config as _run_eval_config

        try:
            with _run_id_env(tracker.id):
                result = _run_eval_config(config_path, output_path=output_path)
        except BaseException as exc:
            _fail_tracker(tracker, exc)
            raise
        tracker.completed(metrics=result)
        return result

    if isinstance(config_path, JobConfig):
        parsed_job = config_path
        config_dict = normalized_job_config_dict(parsed_job)
        tracker = RunTracker.start(
            config_path=None,
            config_dict=config_dict,
            output_path=output_path,
        )
    else:
        config_path_arg = config_path_arg or str(config_path)

    if isinstance(config_path, JobConfig):
        try:
            with _run_id_env(tracker.id):
                remote_result = _dispatch_remote_backends(
                    config_path_arg,
                    config_dict=config_dict,
                )
        except BaseException as exc:
            _fail_tracker(tracker, exc)
            raise
        if remote_result:
            _record_remote_submission(tracker, remote_result)
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

            if tracker is None:
                tracker = RunTracker.start(
                    config_path=config_path_arg,
                    config_dict=_run_model_dump(eval_config),
                    output_path=output_path,
                )
            try:
                with _run_id_env(tracker.id):
                    result = _run_eval_config(eval_config, output_path=output_path)
            except BaseException as exc:
                _fail_tracker(tracker, exc)
                raise
            tracker.completed(metrics=result)
            return result

    try:
        _assert_local_cuda_available()
    except BaseException as exc:
        _fail_tracker(tracker, exc)
        raise

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
        output_dir = job_config_dict.get("training_config", {}).get("output_dir")
        if tracker is None:
            tracker = RunTracker.start(
                config_path=config_path_arg,
                config_dict=job_config_dict,
                output_path=output_dir,
            )
        elif output_dir:
            tracker.update(output_dir=output_dir)
    except FileNotFoundError:
        _fail_tracker(tracker, FileNotFoundError(config_path_arg))
        raise FileNotFoundError(f"Config file not found at: {config_path_arg}")
    except Exception as e:
        _fail_tracker(tracker, e)
        raise ValueError(f"Issue parsing configuration: {e}") from e

    try:
        with _run_id_env(tracker.id):
            ray_trainer(job_config_dict)
        tracker.completed()
    except BaseException as exc:
        _fail_tracker(tracker, exc)
        raise


def _handle_runs_command(args: dict[str, Any]) -> None:
    from leap_finetune.state import list_runs, load_run, render_run, render_run_list
    from leap_finetune.state import sync_run

    action = args["action"]
    run_id = args.get("run_id")
    if action == "list":
        print(render_run_list(list_runs()))
        return
    if not run_id:
        print(f"Usage: leap-finetune runs {action} <run_id>")
        sys.exit(1)
    if action == "show":
        print(render_run(load_run(run_id)))
        return
    if action == "sync":
        print(render_run(sync_run(run_id)))
        return


def _handle_memory_command(args: dict[str, Any]) -> None:
    from leap_finetune.state import add_memory_entry, read_memory

    action = args["action"]
    if action == "show":
        print(read_memory(), end="")
        return
    text = args.get("text")
    if not text:
        print("Usage: leap-finetune memory add <note> [--ref <run_id>]")
        sys.exit(1)
    add_memory_entry(text, ref=args.get("ref"))
    print("Added memory entry.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        from leap_finetune.cli.env import main as env_main

        sys.exit(env_main(sys.argv[2:]))

    command, config_path_arg, output_arg, command_args = _parse_cli_args()

    if command == "slurm":
        _generate_slurm_script(config_path_arg, output_arg)
        return
    if command == "runs":
        _handle_runs_command(command_args)
        return
    if command == "memory":
        _handle_memory_command(command_args)
        return

    if not config_path_arg:
        print("No config file provided. Please provide a path to a YAML config file.")
        print("Usage: leap-finetune <path_to_config.yaml>")
        print("   or: leap-finetune <path_to_eval_config.yaml> --output results.json")
        print("   or: leap-finetune slurm <path_to_config.yaml>")
        print("   or: leap-finetune env fa2-status")
        print("   or: leap-finetune runs list")
        sys.exit(1)

    result = run_config(config_path_arg, output_path=output_arg)
    if result is not None:
        print(yaml.safe_dump(result, sort_keys=True))

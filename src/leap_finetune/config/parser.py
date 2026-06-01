import logging
import os
import pathlib
from datetime import datetime

import yaml

from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.training.default_configs import PeftConfig, TrainingConfig
from leap_finetune.config.job_config import JobConfig
from leap_finetune import LEAP_FINETUNE_DIR
from leap_finetune.checkpointing.model_info import is_moe_model_from_name

logger = logging.getLogger(__name__)

_REWARD_SEP = "::"
_REWARDS_DIR = LEAP_FINETUNE_DIR / "rewards"

TRAINING_TYPE_TO_CONFIG = {
    "sft": "DEFAULT_SFT",
    "dpo": "DEFAULT_DPO",
    "vlm_sft": "DEFAULT_VLM_SFT",
    "vlm_dpo": "DEFAULT_VLM_DPO",
    "moe_sft": "MOE_SFT",
    "moe_dpo": "MOE_DPO",
    "grpo": "DEFAULT_GRPO",
    "vlm_grpo": "DEFAULT_VLM_GRPO",
}
DATASET_TYPE_ALIASES = {
    "moe_sft": "sft",
    "moe_dpo": "dpo",
}
VALID_DATASET_TYPES = {"sft", "dpo", "vlm_sft", "vlm_dpo", "grpo", "vlm_grpo"}


def resolve_config_path(config_input: str) -> pathlib.Path:
    input_path = pathlib.Path(config_input)

    # === Candidate names ===
    candidates = [config_input]
    if not input_path.suffix:
        candidates.append(f"{config_input}.yaml")

    for candidate in candidates:
        candidate_path = pathlib.Path(candidate)

        # === Explicit path ===
        if candidate_path.exists():
            return candidate_path.resolve()

        # === Local job config ===
        local_job_config = pathlib.Path.cwd() / "job_configs" / candidate
        if local_job_config.exists():
            return local_job_config.resolve()

        # === Repo job config ===
        repo_job_config = LEAP_FINETUNE_DIR / "job_configs" / candidate
        if repo_job_config.exists():
            return repo_job_config.resolve()

    raise FileNotFoundError(f"Config file not found at: {input_path}")


def _resolve_local_path(value: str | None, *, base_dir: pathlib.Path) -> str | None:
    if not value:
        return value

    expanded = pathlib.Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())

    if value.startswith(("./", "../")) or (base_dir / value).exists():
        return str((base_dir / value).resolve())

    return value


def _load_yaml_config(path_obj: pathlib.Path) -> dict:
    with open(path_obj) as f:
        config_dict = yaml.safe_load(f) or {}

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config must be a YAML mapping: {path_obj}")
    return config_dict


def _resolve_reward_paths_to_absolute(rewards_cfg, config_dir: pathlib.Path):
    """Resolve local reward specs before Ray workers enter sandbox dirs."""
    config_dir = config_dir.resolve()
    cwd = pathlib.Path.cwd().resolve()
    rewards_dir = _REWARDS_DIR.resolve()

    def _abs(spec: str) -> str:
        if _REWARD_SEP not in spec:
            return spec

        path_str, sep, name = spec.partition(_REWARD_SEP)
        raw = pathlib.Path(path_str.strip())
        variants = [raw] if raw.suffix else [raw, raw.with_suffix(".py")]
        if raw.is_absolute():
            candidates = [variant.resolve() for variant in variants]
        else:
            candidates = [
                (base / variant).resolve()
                for base in (config_dir, cwd, rewards_dir)
                for variant in variants
            ]

        for candidate in dict.fromkeys(candidates):
            if candidate.exists() and candidate.is_file():
                return f"{candidate}{sep}{name}"
        return spec

    if isinstance(rewards_cfg, list):
        return [_abs(spec) if isinstance(spec, str) else spec for spec in rewards_cfg]
    if isinstance(rewards_cfg, dict):
        out = dict(rewards_cfg)
        for key in ("funcs", "rewards"):
            if key in out:
                out[key] = [
                    _abs(spec) if isinstance(spec, str) else spec for spec in out[key]
                ]
        if isinstance(out.get("recipe"), str):
            out["recipe"] = _abs(out["recipe"])
        return out
    return rewards_cfg


def _require_known_training_type(training_type: str) -> None:
    if training_type not in TRAINING_TYPE_TO_CONFIG:
        raise ValueError(
            f"Unknown training type: {training_type}. "
            f"Available: {list(TRAINING_TYPE_TO_CONFIG)}"
        )


def generate_run_name(
    model_name: str,
    training_type: str,
    dataset_path: str,
    dataset_limit: int | None,
    learning_rate: float | None,
    warmup_ratio: float | None,
    use_peft: bool,
    lora_type: str = "a",
) -> str:
    safe_model_name = model_name.split("/")[-1]
    dataset_name_full = dataset_path.strip("/").split("/")[-1]
    dataset_name = (
        dataset_name_full[:10] if len(dataset_name_full) > 10 else dataset_name_full
    )
    limit_str = str(dataset_limit) if dataset_limit else "all"

    if learning_rate:
        lr_val = str(learning_rate)
        lr_clean = lr_val.replace("e-", "em").replace("-", "")
        lr_str = f"lr{lr_clean}"
    else:
        lr_str = "lr_def"

    if warmup_ratio is not None:
        w_str = f"{warmup_ratio:.1f}".replace(".", "p")
        warmup_str = f"w{w_str}"
    else:
        warmup_str = "w_def"

    lora_str = f"lora_{lora_type}" if use_peft else "no_lora"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID", "")

    name = (
        f"{safe_model_name}-{training_type}-{dataset_name}-{limit_str}"
        f"-{lr_str}-{warmup_str}-{lora_str}-{timestamp}"
    )
    if slurm_id:
        name += f"-j{slurm_id}"
    return name


def _parse_dataset_loader(
    ds_config: dict,
    *,
    config_dir: pathlib.Path,
    model_name: str,
) -> DatasetLoader:
    ds_type = DATASET_TYPE_ALIASES.get(ds_config.get("type"), ds_config.get("type"))
    if ds_type not in VALID_DATASET_TYPES:
        raise ValueError(
            f"Invalid dataset type: '{ds_type}'. Must be one of: {sorted(VALID_DATASET_TYPES)}"
        )

    shared_path = ds_config.get("path")
    train_path = ds_config.get("train_path")
    if shared_path and train_path:
        raise ValueError("Use either dataset.path or dataset.train_path, not both")

    dataset_path_env = os.getenv("DATASET_PATH")
    effective_train_path = dataset_path_env or train_path or shared_path
    if not effective_train_path:
        raise ValueError("dataset.path or dataset.train_path is required")

    effective_train_path = _resolve_local_path(
        effective_train_path,
        base_dir=pathlib.Path.cwd() if dataset_path_env else config_dir,
    )

    val_path = _resolve_local_path(
        ds_config.get("val_path"),
        base_dir=config_dir,
    )
    shared_subset = ds_config.get("subset")
    train_subset = ds_config.get("train_subset", shared_subset)
    val_subset = ds_config.get("val_subset", shared_subset)
    train_split = ds_config.get("train_split", ds_config.get("split", "train"))
    val_split = ds_config.get("val_split")
    if val_path and val_split is None:
        val_split = "train"

    test_size = ds_config.get("test_size")
    if test_size is not None and (val_path is not None or val_split is not None):
        raise ValueError(
            "dataset.test_size cannot be combined with dataset.val_path or dataset.val_split"
        )
    if (
        ds_type in ("grpo", "vlm_grpo")
        and test_size is None
        and val_path is None
        and val_split is None
    ):
        test_size = 0.01

    return DatasetLoader(
        dataset_path=effective_train_path,
        dataset_type=ds_type,
        model_name=model_name,
        limit=ds_config.get("limit"),
        split=train_split,
        test_size=test_size,
        subset=train_subset,
        val_dataset_path=val_path,
        val_split=val_split,
        val_subset=val_subset,
        image_root=ds_config.get("image_root"),
        cache_dataset=ds_config.get("cache_dataset", False),
        hf_streaming_batch_size=ds_config.get("hf_streaming_batch_size", 10000),
    )


def _resolve_training_type(
    raw_training_type: str,
    model_name: str,
    train_config_dict: dict,
) -> str:
    if raw_training_type not in ("sft", "dpo"):
        return raw_training_type
    if not is_moe_model_from_name(model_name):
        return raw_training_type

    base_config = train_config_dict.get("extends") or train_config_dict.get("base")
    uses_moe_config = isinstance(base_config, str) and base_config.startswith("MOE_")
    if uses_moe_config or "moe_training" in train_config_dict:
        return f"moe_{raw_training_type}"

    return raw_training_type


def _build_training_config(train_config_dict: dict, training_type: str):
    _require_known_training_type(training_type)

    train_config_dict = train_config_dict.copy()
    base_config_name = train_config_dict.pop("extends", None) or train_config_dict.pop(
        "base", None
    )
    base_config_map = {member.name: member for member in TrainingConfig}

    if base_config_name:
        if base_config_name not in base_config_map:
            available = list(base_config_map.keys())
            raise ValueError(
                f"Unknown base config: {base_config_name}. Available: {available}"
            )
        base_train_config = base_config_map[base_config_name]
    else:
        base_train_config = base_config_map[TRAINING_TYPE_TO_CONFIG[training_type]]

    for float_key in ("learning_rate", "weight_decay"):
        if float_key in train_config_dict and isinstance(
            train_config_dict[float_key], str
        ):
            train_config_dict[float_key] = float(train_config_dict[float_key])

    return base_train_config.override(**train_config_dict), train_config_dict


def _build_peft_config(peft_dict: dict | None):
    peft_dict = peft_dict.copy() if peft_dict else {}
    use_peft = peft_dict.get("use_peft")
    if use_peft is False:
        return None, use_peft

    base_peft_name = peft_dict.pop("extends", None) or peft_dict.pop("base", None)
    peft_dict.pop("use_peft", None)

    if not base_peft_name:
        peft_config = PeftConfig.DEFAULT_LORA if use_peft is True else None
        return peft_config, use_peft

    base_peft_map = {member.name: member for member in PeftConfig}
    if base_peft_name not in base_peft_map:
        available = list(base_peft_map.keys())
        raise ValueError(
            f"Unknown base PEFT config: {base_peft_name}. Available: {available}"
        )

    base_config_value = base_peft_map[base_peft_name].value
    if base_config_value is None:
        return None, use_peft

    from peft import LoraConfig

    base_dict = (
        base_config_value.to_dict()
        if hasattr(base_config_value, "to_dict")
        else dict(base_config_value)
    )
    base_dict.update({k: v for k, v in peft_dict.items() if v is not None})
    peft_config_obj = LoraConfig(**base_dict)

    class _CustomPeftConfig:
        def __init__(self, value):
            self.value = value

    return _CustomPeftConfig(peft_config_obj), use_peft


def _resolve_project_name(config_dict: dict) -> str:
    return (
        config_dict.get("project_name") or config_dict.get("job_name") or "default_job"
    )


def _resolve_output_dir(
    *,
    final_train_values: dict,
    project_name: str,
    run_name: str,
) -> pathlib.Path:
    resume_from = final_train_values.get("resume_from_checkpoint")
    base_project_dir = os.getenv("OUTPUT_DIR", f"./outputs/{project_name}")

    if resume_from and resume_from != "latest":
        resume_path = pathlib.Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        return resume_path.parent

    if resume_from == "latest":
        project_path = pathlib.Path(base_project_dir).resolve()
        run_dirs = (
            [
                d
                for d in project_path.iterdir()
                if d.is_dir() and (d / "latest").exists()
            ]
            if project_path.exists()
            else []
        )
        if run_dirs:
            final_output_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
            latest_link = final_output_dir / "latest"
            final_train_values["resume_from_checkpoint"] = str(latest_link.resolve())
            return final_output_dir

        final_train_values.pop("resume_from_checkpoint", None)
        return project_path / run_name

    return pathlib.Path(base_project_dir).resolve() / run_name


def _create_output_dir(
    *,
    output_dir: pathlib.Path,
    project_name: str,
    run_name: str,
) -> pathlib.Path:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except PermissionError:
        logger.warning(
            "Permission denied creating %s, falling back to local ./outputs",
            output_dir,
        )
        fallback_dir = pathlib.Path.cwd() / "outputs" / project_name / run_name
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir


def parse_job_config(config_input: str) -> JobConfig:
    path_obj = resolve_config_path(config_input)
    config_dir = path_obj.parent
    config_dict = _load_yaml_config(path_obj)

    model_name = config_dict.get("model_name", "LFM2-1.2B")
    ds_config = config_dict.get("dataset", {})
    dataset = _parse_dataset_loader(
        ds_config,
        config_dir=config_dir,
        model_name=model_name,
    )

    train_config_dict = config_dict.get("training_config", {})
    training_type = _resolve_training_type(
        config_dict.get("training_type", "sft"),
        model_name,
        train_config_dict,
    )
    final_training_config, train_config_overrides = _build_training_config(
        train_config_dict,
        training_type,
    )
    final_train_values = final_training_config.value
    final_train_values["chat_template_path"] = _resolve_local_path(
        final_train_values.get("chat_template_path"),
        base_dir=config_dir,
    )
    final_train_values["adapter_path"] = _resolve_local_path(
        final_train_values.get("adapter_path"),
        base_dir=config_dir,
    )

    peft_config, use_peft = _build_peft_config(config_dict.get("peft_config"))
    project_name = _resolve_project_name(config_dict)

    run_name = generate_run_name(
        model_name=model_name,
        training_type=training_type,
        dataset_path=dataset.dataset_path,
        dataset_limit=ds_config.get("limit"),
        learning_rate=final_train_values.get("learning_rate"),
        warmup_ratio=final_train_values.get("warmup_ratio"),
        use_peft=use_peft is not False and peft_config is not None,
        lora_type="a",
    )

    final_output_dir = _resolve_output_dir(
        final_train_values=final_train_values,
        project_name=project_name,
        run_name=run_name,
    )
    final_output_dir = _create_output_dir(
        output_dir=final_output_dir,
        project_name=project_name,
        run_name=run_name,
    )

    final_train_values["output_dir"] = str(final_output_dir)
    final_train_values["leap_run_name_template"] = run_name
    if not dataset.has_eval_dataset():
        raw_eval_strategy = train_config_overrides.get("eval_strategy")
        if raw_eval_strategy and raw_eval_strategy != "no":
            raise ValueError(
                "training_config.eval_strategy requires a validation dataset. "
                "Set dataset.test_size, dataset.val_split, or dataset.val_path, or disable eval."
            )
        final_train_values["eval_strategy"] = "no"

    model_config = config_dict.get("model_config")
    benchmark_configs = config_dict.get("benchmarks")
    rewards_cfg = config_dict.get("rewards")
    if rewards_cfg is not None:
        rewards_cfg = _resolve_reward_paths_to_absolute(rewards_cfg, config_dir)
    rl_env_cfg = config_dict.get("rl_env")
    grpo_rollout_cfg = config_dict.get("grpo_rollout")

    if training_type not in ("grpo", "vlm_grpo"):
        for key, value in (
            ("rewards", rewards_cfg),
            ("rl_env", rl_env_cfg),
            ("grpo_rollout", grpo_rollout_cfg),
        ):
            if value is not None:
                raise ValueError(
                    f"Config key `{key}` is only valid for training_type in "
                    f"('grpo', 'vlm_grpo'); got training_type={training_type!r}."
                )

    _validate_parallelism_config(
        final_train_values,
        training_type,
        model_name,
    )

    return JobConfig(
        job_name=project_name,
        model_name=model_name,
        training_type=training_type,
        dataset=dataset,
        training_config=final_training_config,
        peft_config=peft_config,
        benchmark_configs=benchmark_configs,
        model_config=model_config,
        ray_config=config_dict.get("ray"),
        rewards=rewards_cfg,
        rl_env=rl_env_cfg,
        grpo_rollout=grpo_rollout_cfg,
        config_dir=str(config_dir.resolve()),
    )


def _validate_parallelism_config(
    training_config: dict,
    training_type: str,
    model_name: str,
) -> None:
    if (training_config.get("context_parallel_size", 1) or 1) > 1:
        raise ValueError("context_parallel_size is not supported in the EP-only branch")

    moe_config = training_config.get("moe_training", {})
    if not moe_config:
        return

    effective_training_type = training_type
    if training_type in ("sft", "dpo") and is_moe_model_from_name(model_name):
        effective_training_type = f"moe_{training_type}"

    if effective_training_type == "moe_sft":
        capacity_factor = moe_config.get("capacity_factor")
        token_drop_policy = moe_config.get("token_drop_policy")
        if capacity_factor is not None or token_drop_policy not in (None, "probs"):
            raise ValueError(
                "MoE SFT currently supports uncapped routing only. "
                "Remove capacity_factor/token_drop_policy from moe_training."
            )

    ep_size = moe_config.get("expert_parallel_size", 1) or 1
    if ep_size <= 1:
        return

    if effective_training_type not in ("moe_sft", "moe_dpo"):
        raise ValueError(
            f"expert_parallel_size={ep_size} requires training_type 'moe_sft' or 'moe_dpo', "
            f"got '{training_type}'"
        )

    if ep_size & (ep_size - 1) != 0:
        raise ValueError(f"expert_parallel_size must be a power of 2, got {ep_size}")


def print_job_config_summary(job_config: JobConfig) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    config_value = job_config.training_config.value
    peft_value = job_config.peft_config.value if job_config.peft_config else None

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value", style="green")

    table.add_row("Model", job_config.model_name)
    table.add_row("Job Name", job_config.job_name)
    table.add_row("Training Type", job_config.training_type.upper())
    table.add_row("Output Directory", str(config_value.get("output_dir")))

    learning_rate = config_value.get("learning_rate")
    if learning_rate is not None:
        table.add_row("Learning Rate", f"{learning_rate:.2e}")

    batch_size = config_value.get("per_device_train_batch_size")
    if batch_size is not None:
        table.add_row("Batch Size", f"{batch_size}")

    num_epochs = config_value.get("num_train_epochs")
    if num_epochs is not None:
        table.add_row("Epochs", f"{num_epochs}")

    warmup_ratio = config_value.get("warmup_ratio")
    warmup_steps = config_value.get("warmup_steps")
    if warmup_ratio is not None:
        table.add_row("Warmup Ratio", f"{warmup_ratio:.2f}")
    elif warmup_steps is not None:
        table.add_row("Warmup Steps", f"{warmup_steps}")

    save_strategy = config_value.get("save_strategy", "no")
    if save_strategy != "no":
        table.add_row("Save Strategy", save_strategy)

    eval_strategy = config_value.get("eval_strategy", "no")
    if eval_strategy != "no":
        table.add_row("Eval Strategy", eval_strategy)

    peft_details = ""
    if peft_value and hasattr(peft_value, "r") and hasattr(peft_value, "lora_alpha"):
        peft_details = f" (r={peft_value.r}, alpha={peft_value.lora_alpha})"
    table.add_row("PEFT", f"Enabled{peft_details}" if peft_value else "Disabled")

    dataset = job_config.dataset
    if isinstance(dataset, DatasetLoader):
        table.add_row("Dataset Path", dataset.dataset_path)
        if dataset.val_dataset_path:
            table.add_row("Validation Path", dataset.val_dataset_path)
        elif dataset.val_split:
            table.add_row("Validation Split", dataset.val_split)
        elif dataset.test_size is None:
            table.add_row("Validation", "Disabled")
        else:
            table.add_row("Validation Split", f"Random ({dataset.test_size:.2f})")
        if dataset.limit:
            table.add_row("Dataset Limit", f"{dataset.limit:,}")
    elif isinstance(dataset, tuple):
        table.add_row("Train Samples", f"{len(dataset[0]):,}")
        table.add_row("Test Samples", f"{len(dataset[1]):,}")

    Console().print(
        Panel(
            table,
            title="[bold blue]Training Configuration[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

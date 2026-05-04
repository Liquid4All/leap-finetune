import importlib
import logging
import os
import pathlib
from datetime import datetime

import yaml

from leap_finetune.data_loaders.dataset_loader import DatasetLoader
from leap_finetune.training_configs import PeftConfig, TrainingConfig
from leap_finetune.training_configs.job_config import JobConfig
from leap_finetune.utils.config_resolver import resolve_config_path
from leap_finetune.utils.model_utils import is_moe_model_from_name

logger = logging.getLogger(__name__)


def _resolve_local_path(value: str | None, *, base_dir: pathlib.Path) -> str | None:
    if not value:
        return value

    expanded = pathlib.Path(value).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())

    if value.startswith(("./", "../")) or (base_dir / value).exists():
        return str((base_dir / value).resolve())

    return value


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


def _import_callable(dotted_path: str):
    try:
        module_path, func_name = dotted_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"preprocess_fn must be a dotted path like 'my_module.func', got: {dotted_path}"
        ) from exc

    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(f"Function '{func_name}' not found in module '{module_path}'")
    if not callable(fn):
        raise ValueError(f"'{dotted_path}' is not callable")
    return fn


def _parse_dataset_loader(
    ds_config: dict,
    *,
    config_dir: pathlib.Path,
    model_name: str,
) -> DatasetLoader:
    dataset_type_aliases = {
        "moe_sft": "sft",
        "moe_dpo": "dpo",
    }
    ds_type = dataset_type_aliases.get(ds_config.get("type"), ds_config.get("type"))
    valid_types = {"sft", "dpo", "vlm_sft"}
    if ds_type not in valid_types:
        raise ValueError(
            f"Invalid dataset type: '{ds_type}'. Must be one of: {sorted(valid_types)}"
        )

    preprocess_fn = None
    preprocess_fn_path = ds_config.get("preprocess_fn")
    if preprocess_fn_path:
        preprocess_fn = _import_callable(preprocess_fn_path)

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
        preprocess_fn=preprocess_fn,
    )


def parse_job_config(config_input: str) -> JobConfig:
    path_obj = resolve_config_path(config_input)
    config_dir = path_obj.parent

    with open(path_obj) as f:
        config_dict = yaml.safe_load(f)

    model_name = config_dict.get("model_name", "LFM2-1.2B")
    ds_config = config_dict.get("dataset", {})
    dataset = _parse_dataset_loader(
        ds_config,
        config_dir=config_dir,
        model_name=model_name,
    )

    train_config_dict = config_dict.get("training_config", {})
    training_type = config_dict.get("training_type", "sft")

    base_config_name = train_config_dict.pop("extends", None) or train_config_dict.pop(
        "base", None
    )
    if base_config_name:
        base_config_map = {member.name: member for member in TrainingConfig}
        if base_config_name not in base_config_map:
            available = list(base_config_map.keys())
            raise ValueError(
                f"Unknown base config: {base_config_name}. Available: {available}"
            )
        base_train_config = base_config_map[base_config_name]
    else:
        training_type_to_config = {
            "sft": "DEFAULT_SFT",
            "dpo": "DEFAULT_DPO",
            "vlm_sft": "DEFAULT_VLM_SFT",
            "moe_sft": "MOE_SFT",
            "moe_dpo": "MOE_DPO",
        }
        if training_type not in training_type_to_config:
            raise ValueError(f"Unknown training type: {training_type}")
        config_name = training_type_to_config[training_type]
        base_train_config = {member.name: member for member in TrainingConfig}[
            config_name
        ]

    for float_key in ("learning_rate", "weight_decay"):
        if float_key in train_config_dict and isinstance(train_config_dict[float_key], str):
            train_config_dict[float_key] = float(train_config_dict[float_key])

    final_training_config = base_train_config.override(**train_config_dict)
    final_train_values = final_training_config.value
    final_train_values["chat_template_path"] = _resolve_local_path(
        final_train_values.get("chat_template_path"),
        base_dir=config_dir,
    )

    peft_dict = config_dict.get("peft_config", {})
    use_peft = peft_dict.get("use_peft", None) if peft_dict else None
    if use_peft is False:
        peft_config = None
    else:
        peft_config_dict = peft_dict.copy() if peft_dict else {}
        base_peft_name = peft_config_dict.pop("extends", None) or peft_config_dict.pop(
            "base", None
        )
        peft_config_dict.pop("use_peft", None)

        if base_peft_name:
            base_peft_map = {member.name: member for member in PeftConfig}
            if base_peft_name not in base_peft_map:
                available = list(base_peft_map.keys())
                raise ValueError(
                    f"Unknown base PEFT config: {base_peft_name}. Available: {available}"
                )

            base_peft_config = base_peft_map[base_peft_name]
            from peft import LoraConfig

            base_config_value = base_peft_config.value
            if base_config_value is None:
                peft_config = None
            else:
                base_dict = (
                    base_config_value.to_dict()
                    if hasattr(base_config_value, "to_dict")
                    else dict(base_config_value)
                )
                base_dict.update(
                    {k: v for k, v in peft_config_dict.items() if v is not None}
                )
                peft_config_obj = LoraConfig(**base_dict)

                class _CustomPeftConfig:
                    def __init__(self, value):
                        self.value = value

                peft_config = _CustomPeftConfig(peft_config_obj)
        else:
            peft_config = PeftConfig.DEFAULT_LORA if use_peft is True else None

    yaml_project_name = config_dict.get("project_name")
    yaml_job_name = config_dict.get("job_name")
    project_name = (
        yaml_project_name
        if yaml_project_name
        else (yaml_job_name if yaml_job_name else "default_job")
    )

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

    resume_from = final_train_values.get("resume_from_checkpoint")
    base_project_dir = os.getenv("OUTPUT_DIR", f"./outputs/{project_name}")
    if resume_from and resume_from != "latest":
        resume_path = pathlib.Path(resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        final_output_dir = resume_path.parent
    elif resume_from == "latest":
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
            resume_from = str(latest_link.resolve())
            final_training_config.value["resume_from_checkpoint"] = resume_from
        else:
            final_output_dir = project_path / run_name
            final_training_config.value.pop("resume_from_checkpoint", None)
    else:
        final_output_dir = pathlib.Path(base_project_dir).resolve() / run_name

    try:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.warning(
            "Permission denied creating %s, falling back to local ./outputs",
            final_output_dir,
        )
        final_output_dir = pathlib.Path.cwd() / "outputs" / project_name / run_name
        final_output_dir.mkdir(parents=True, exist_ok=True)

    final_training_config.value["output_dir"] = str(final_output_dir)
    final_training_config.value["leap_run_name_template"] = run_name
    if not dataset.has_eval_dataset():
        raw_eval_strategy = train_config_dict.get("eval_strategy")
        if raw_eval_strategy and raw_eval_strategy != "no":
            raise ValueError(
                "training_config.eval_strategy requires a validation dataset. "
                "Set dataset.test_size, dataset.val_split, or dataset.val_path, or disable eval."
            )
        final_training_config.value["eval_strategy"] = "no"

    model_config = config_dict.get("model_config")
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
        model_config=model_config,
        ray_config=config_dict.get("ray"),
        benchmark_configs=config_dict.get("benchmarks"),
    )


def _validate_parallelism_config(
    training_config: dict,
    training_type: str,
    model_name: str,
) -> None:
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
    cp_size = training_config.get("context_parallel_size", 1) or 1
    if ep_size <= 1:
        return

    if cp_size > 1:
        raise ValueError(
            "expert_parallel_size > 1 cannot be combined with context_parallel_size > 1. "
            "Only DP x CP is supported for context parallelism."
        )

    if effective_training_type not in ("moe_sft", "moe_dpo"):
        raise ValueError(
            f"expert_parallel_size={ep_size} requires training_type 'moe_sft' or 'moe_dpo', "
            f"got '{training_type}'"
        )

    if ep_size & (ep_size - 1) != 0:
        raise ValueError(f"expert_parallel_size must be a power of 2, got {ep_size}")

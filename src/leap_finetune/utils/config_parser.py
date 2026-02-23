import os
import pathlib
from datetime import datetime

import yaml

from leap_finetune.data_loaders.dataset_loader import DatasetLoader
from leap_finetune.training_configs import PeftConfig, TrainingConfig
from leap_finetune.training_configs.job_config import JobConfig
from leap_finetune.utils.constants import LEAP_FINETUNE_DIR


def resolve_config_path(config_input: str) -> pathlib.Path:
    input_path = pathlib.Path(config_input)

    # Auto-append .yaml if no extension provided
    candidates = [config_input]
    if not input_path.suffix:
        candidates.append(config_input + ".yaml")

    for candidate in candidates:
        candidate_path = pathlib.Path(candidate)

        # 1. Absolute or CWD-relative path
        if candidate_path.exists():
            return candidate_path.resolve()

        # 2. Look in ./job_configs/
        local_job_configs = pathlib.Path.cwd() / "job_configs" / candidate
        if local_job_configs.exists():
            return local_job_configs.resolve()

        # 3. Look in repo job_configs/
        repo_job_configs = LEAP_FINETUNE_DIR / "job_configs" / candidate
        if repo_job_configs.exists():
            return repo_job_configs.resolve()

    raise FileNotFoundError(f"Config file not found at: {input_path}")


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

    if use_peft:
        lora_str = f"lora_{lora_type}" if lora_type else "lora_a"
    else:
        lora_str = "no_lora"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return (
        f"{safe_model_name}-{training_type}-{dataset_name}-{limit_str}"
        f"-{lr_str}-{warmup_str}-{lora_str}-{timestamp}"
    )


def parse_job_config(config_input: str) -> JobConfig:
    path_obj = resolve_config_path(config_input)

    with open(path_obj) as f:
        config_dict = yaml.safe_load(f)

    # === Dataset ===
    ds_config = config_dict.get("dataset", {})
    dataset_path_env = os.getenv("DATASET_PATH")
    final_dataset_path = dataset_path_env if dataset_path_env else ds_config.get("path")

    valid_types = {"sft", "dpo", "vlm_sft"}
    ds_type = ds_config.get("type")
    if ds_type not in valid_types:
        raise ValueError(
            f"Invalid dataset type: '{ds_type}'. Must be one of: {sorted(valid_types)}"
        )

    dataset = DatasetLoader(
        dataset_path=final_dataset_path,
        dataset_type=ds_type,
        limit=ds_config.get("limit"),
        test_size=ds_config.get("test_size", 0.2),
        subset=ds_config.get("subset"),
    )

    # === Training config with extends support ===
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
        }
        if training_type not in training_type_to_config:
            raise ValueError(f"Unknown training type: {training_type}")

        config_name = training_type_to_config[training_type]
        base_config_map = {member.name: member for member in TrainingConfig}
        base_train_config = base_config_map[config_name]

    # Ensure learning_rate is float
    if "learning_rate" in train_config_dict:
        lr_val = train_config_dict["learning_rate"]
        if isinstance(lr_val, str):
            train_config_dict["learning_rate"] = float(lr_val)

    # Merge base config with YAML overrides
    final_training_config = base_train_config.override(**train_config_dict)
    final_train_values = final_training_config.value

    # === PEFT config with extends support ===
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
            if use_peft is True:
                peft_config = PeftConfig.DEFAULT_LORA
            else:
                peft_config = None

    # === Project / output dir ===
    yaml_project_name = config_dict.get("project_name")
    yaml_job_name = config_dict.get("job_name")
    project_name = (
        yaml_project_name
        if yaml_project_name
        else (yaml_job_name if yaml_job_name else "default_job")
    )

    run_name = generate_run_name(
        model_name=config_dict.get("model_name", "LFM2-1.2B"),
        training_type=training_type,
        dataset_path=final_dataset_path,
        dataset_limit=ds_config.get("limit"),
        learning_rate=final_train_values.get("learning_rate"),
        warmup_ratio=final_train_values.get("warmup_ratio"),
        use_peft=use_peft is not False and peft_config is not None,
        lora_type="a",
    )

    default_project_dir = f"./outputs/{project_name}"
    project_dir = os.getenv("OUTPUT_DIR", default_project_dir)
    final_output_dir = pathlib.Path(project_dir).resolve()

    try:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(
            f"Permission denied creating {final_output_dir}, falling back to local ./outputs"
        )
        final_output_dir = pathlib.Path.cwd() / "outputs" / project_name
        final_output_dir.mkdir(parents=True, exist_ok=True)

    final_training_config.value["output_dir"] = str(final_output_dir)
    final_training_config.value["leap_run_name_template"] = run_name

    return JobConfig(
        job_name=project_name,
        model_name=config_dict.get("model_name", "LFM2-1.2B"),
        training_type=training_type,
        dataset=dataset,
        training_config=final_training_config,
        peft_config=peft_config,
    )

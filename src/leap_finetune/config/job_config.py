from dataclasses import dataclass
from typing import Any, Literal

from datasets import Dataset

from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.training.default_configs import PeftConfig, TrainingConfig


@dataclass
class JobConfig:
    job_name: str
    model_name: str = "LFM2-1.2B"
    training_type: Literal[
        "sft",
        "dpo",
        "vlm_sft",
        "vlm_dpo",
        "moe_sft",
        "moe_dpo",
        "grpo",
        "vlm_grpo",
    ] = "sft"
    dataset: DatasetLoader | tuple[Dataset, Dataset] | None = None
    training_config: TrainingConfig = TrainingConfig.DEFAULT_SFT
    peft_config: PeftConfig | None = PeftConfig.DEFAULT_LORA
    benchmark_configs: dict | None = None
    model_config: dict | None = None
    ray_config: dict | None = None
    rewards: list | dict | None = None
    rl_env: dict | None = None
    grpo_rollout: dict | None = None
    config_dir: str | None = None

    def __post_init__(self):
        self._validate_job_name()
        self._validate_training_config()

    def _validate_job_name(self):
        if not all(c.isalnum() or c in "-_" for c in self.job_name):
            raise ValueError(
                f"Invalid job name '{self.job_name}': only letters, numbers, hyphens, and underscores allowed"
            )

    def _validate_training_config(self):
        config_value = self.training_config.value
        if not isinstance(config_value, dict):
            return

        config_training_type = config_value.get("training_type")
        if config_training_type and config_training_type != self.training_type:
            raise ValueError(
                f"Training config type '{config_training_type}' doesn't match "
                f"job training type '{self.training_type}'"
            )

    def to_dict(self, dataset: tuple[Dataset, Dataset] | None = None) -> dict[str, Any]:
        dataset_to_use = dataset if dataset is not None else self.dataset

        return {
            "model_name": self.model_name,
            "job_name": self.job_name,
            "training_type": self.training_type,
            "training_config": self.training_config.value,
            "dataset": dataset_to_use,
            "peft_config": self.peft_config.value if self.peft_config else None,
            "benchmark_configs": self.benchmark_configs,
            "model_config": self.model_config,
            "ray_config": self.ray_config,
            "rewards": self.rewards,
            "rl_env": self.rl_env,
            "grpo_rollout": self.grpo_rollout,
            "config_dir": self.config_dir,
        }

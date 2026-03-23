from dataclasses import dataclass
from typing import Any, Literal

from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from leap_finetune.training_configs import PeftConfig, TrainingConfig
from leap_finetune.data_loaders.dataset_loader import DatasetLoader


@dataclass
class JobConfig:
    job_name: str
    model_name: str = "LFM2-1.2B"
    training_type: Literal["sft", "dpo", "vlm_sft", "moe_sft", "moe_dpo"] = "sft"
    dataset: DatasetLoader | tuple[Dataset, Dataset] | None = None
    training_config: TrainingConfig = TrainingConfig.DEFAULT_SFT
    peft_config: PeftConfig | None = PeftConfig.DEFAULT_LORA
    model_config: dict | None = None

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

        if isinstance(config_value, dict):
            config_training_type = config_value.get("training_type")
            if config_training_type and config_training_type != self.training_type:
                # Allow MoE training types to use base SFT/DPO configs
                compatible = {
                    "moe_sft": ("sft", "moe_sft"),
                    "moe_dpo": ("dpo", "moe_dpo"),
                }
                allowed = compatible.get(self.training_type, (self.training_type,))
                if config_training_type not in allowed:
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
            "model_config": self.model_config,
        }

    def print_config_summary(self):
        console = Console()

        config_value = self.training_config.value
        output_dir = config_value.get("output_dir")

        learning_rate = config_value.get("learning_rate")
        batch_size = config_value.get("per_device_train_batch_size")
        num_epochs = config_value.get("num_train_epochs")
        warmup_ratio = config_value.get("warmup_ratio")
        warmup_steps = config_value.get("warmup_steps")
        save_strategy = config_value.get("save_strategy", "no")
        eval_strategy = config_value.get("eval_strategy", "no")

        peft_enabled = self.peft_config and self.peft_config.value
        peft_details = ""
        if peft_enabled:
            peft_value = self.peft_config.value
            if hasattr(peft_value, "r") and hasattr(peft_value, "lora_alpha"):
                peft_details = f" (r={peft_value.r}, alpha={peft_value.lora_alpha})"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="bold cyan", min_width=18)
        table.add_column("Value", style="green")

        table.add_row("Model", self.model_name)
        table.add_row("Job Name", self.job_name)
        table.add_row("Training Type", self.training_type.upper())
        table.add_row("Output Directory", str(output_dir))

        if learning_rate is not None:
            table.add_row("Learning Rate", f"{learning_rate:.2e}")
        if batch_size is not None:
            table.add_row("Batch Size", f"{batch_size}")
        if num_epochs is not None:
            table.add_row("Epochs", f"{num_epochs}")
        if warmup_ratio is not None:
            table.add_row("Warmup Ratio", f"{warmup_ratio:.2f}")
        elif warmup_steps is not None:
            table.add_row("Warmup Steps", f"{warmup_steps}")
        if save_strategy != "no":
            table.add_row("Save Strategy", save_strategy)
        if eval_strategy != "no":
            table.add_row("Eval Strategy", eval_strategy)

        peft_status = f"Enabled{peft_details}" if peft_enabled else "Disabled"
        table.add_row("PEFT", peft_status)

        # Dataset info
        if isinstance(self.dataset, DatasetLoader):
            table.add_row("Dataset Path", self.dataset.dataset_path)
            if self.dataset.limit:
                table.add_row("Dataset Limit", f"{self.dataset.limit:,}")
        elif isinstance(self.dataset, tuple):
            table.add_row("Train Samples", f"{len(self.dataset[0]):,}")
            table.add_row("Test Samples", f"{len(self.dataset[1]):,}")

        panel = Panel(
            table,
            title="[bold blue]Training Configuration[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)

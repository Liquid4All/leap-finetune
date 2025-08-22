from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path

from datasets import Dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from leap_finetune.configs import PeftConfig, TrainingConfig
from leap_finetune.data_loaders.dataset_loader import DatasetLoader


def is_job_name_unique(output_dir: str) -> bool:
    """
    Checks if a job with the given name exists for a given training type.
    """
    return not Path(output_dir).exists()


@dataclass
class JobConfig:
    """Job configuration with validation"""

    job_name: str
    model_name: str = "LFM2-1.2B"
    training_type: Literal["sft", "dpo", "vlm_sft"] = "sft"
    dataset: DatasetLoader | tuple[Dataset, Dataset] | None = None
    training_config: TrainingConfig = TrainingConfig.DEFAULT_SFT
    peft_config: PeftConfig | None = PeftConfig.DEFAULT_LORA

    def __post_init__(self):
        if isinstance(self.dataset, DatasetLoader):
            self.dataset = self.dataset.load()
        self.training_config.value["output_dir"] = str(
            Path(self.training_config.value.get("output_dir")) / self.job_name
        )
        self._validate_job_name()
        self._validate_training_config()

    def _validate_job_name(self):
        # Check for filesystem unsafe characters
        if not all(c.isalnum() or c in "-_" for c in self.job_name):
            raise ValueError(
                "Only letters, numbers, hyphens, and underscores in job name"
            )

        # Check if job dir already exists - warn but don't fail
        # (Ray workers might import config after directory is created)
        if not is_job_name_unique(self.training_config.value.get("output_dir")):
            raise ValueError(
                "Job output directory already exists\n"
                "This might be from a previous run or concurrent Ray worker initialization."
            )

    def _validate_training_config(self):
        """Validate training config matches training type"""

        config_value = self.training_config.value

        if isinstance(config_value, dict):
            config_training_type = config_value.get("training_type")
            if config_training_type and config_training_type != self.training_type:
                raise ValueError(
                    f"Training config type '{config_training_type}' doesn't match job training type '{self.training_type}'"
                )

    def to_dict(self, dataset: tuple[Dataset, Dataset] | None = None) -> dict[str, Any]:
        """Convert to final ft_job_config dict"""

        dataset_to_use = dataset if dataset is not None else self.dataset

        return {
            "model_name": self.model_name,
            "job_name": self.job_name,
            "training_type": self.training_type,
            "training_config": self.training_config.value,
            "dataset": dataset_to_use,
            "peft_config": self.peft_config.value if self.peft_config else None,
        }

    def print_config_summary(self):
        """Print summary of current configuration"""
        console = Console()

        # Calculate output directory
        output_dir = self.training_config.value.get("output_dir")

        # Create a table for the configuration
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="bold cyan", min_width=15)
        table.add_column("Value", style="green")

        table.add_row("Model", self.model_name)
        table.add_row("Job Name", self.job_name)
        table.add_row("Training Type", self.training_type.upper())
        table.add_row("Output Directory", str(output_dir))
        table.add_row("PEFT", "✅ Enabled" if self.peft_config.value else "❌ Disabled")
        table.add_row("Train Samples", f"{len(self.dataset[0]):,}")
        table.add_row("Test Samples", f"{len(self.dataset[1]):,}")

        # Wrap in a panel
        panel = Panel(
            table,
            title="[bold blue]Training Configuration[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

        console.print(panel)

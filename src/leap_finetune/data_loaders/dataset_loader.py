from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

from datasets import Dataset, load_dataset

from .validate_loader import validate_data_loader, validate_dataset_format


@dataclass
class DatasetLoader:
    """Dataset loader for training and testing datasets"""

    dataset_path: str
    dataset_type: Literal["sft", "dpo", "vlm_sft"]
    limit: Optional[int] = None  # Default: all samples
    split: str = "train"  # Default: "train"
    test_size: float = 0.2  # Default: 80/20 split
    subset: Optional[str] = None  # Default: None (for HuggingFace dataset subsets)

    @validate_data_loader
    def load(self) -> Tuple[Dataset, Dataset]:
        """Load and return validated (train, test) dataset"""
        split_str = self.split
        if self.limit:
            split_str = f"{self.split}[:{self.limit}]"

        # Load dataset from either local file or HuggingFace
        if Path(self.dataset_path).exists() or self.dataset_path.startswith(
            ("./", "/")
        ):
            try:
                dataset = load_dataset(
                    "json", data_files=self.dataset_path, split=split_str
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load local dataset '{self.dataset_path}': {e}"
                )
        else:
            # Try HuggingFace dataset
            try:
                dataset = load_dataset(self.dataset_path, self.subset, split=split_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to load HuggingFace dataset '{self.dataset_path}': {e}"
                )

        # Validate dataset format
        dataset = validate_dataset_format(dataset, self.dataset_type)

        # Split dataset
        split_dataset = dataset.train_test_split(test_size=self.test_size)
        return split_dataset["train"], split_dataset["test"]

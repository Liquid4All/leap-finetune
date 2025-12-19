from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

from datasets import Dataset, load_dataset

from .validate_loader import (
    quick_validate_schema,
    validate_data_loader,
    validate_dataset_format,
)


@dataclass
class DatasetLoader:
    """Dataset loader for training and testing datasets."""

    dataset_path: str
    dataset_type: Literal["sft", "dpo", "vlm_sft"]
    limit: Optional[int] = None
    split: str = "train"
    test_size: float = 0.2
    subset: Optional[str] = None
    # Optional preprocessing function: takes Ray Dataset, returns Ray Dataset
    # Applied before validation - use for custom filtering, transforms, joins, etc.
    preprocess_fn: Optional[Callable] = field(default=None, repr=False)

    def quick_validate(self) -> None:
        """Fast validation on ~10 samples. Raises ValueError on issues."""
        quick_validate_schema(
            dataset_path=self.dataset_path,
            dataset_type=self.dataset_type,
            subset=self.subset,
            split=self.split,
            num_samples=10,
        )

    def to_ray_dataset(self):
        """Create a lazy Ray Dataset from the source."""
        import ray.data
        from rich.console import Console

        console = Console()
        path = self.dataset_path

        if self._is_local_file(path):
            if path.endswith((".parquet", ".pq")):
                console.print(f"[dim]Reading parquet: {path}[/dim]")
                ds = ray.data.read_parquet(path)
            else:
                console.print(f"[dim]Reading JSONL: {path}[/dim]")
                ds = ray.data.read_json(path)
        elif self._is_cloud_path(path):
            # Cloud storage: S3, GCS, Azure
            cloud_type = self._get_cloud_type(path)
            if ".parquet" in path or ".pq" in path:
                console.print(f"[dim]Reading parquet from {cloud_type}: {path}[/dim]")
                ds = ray.data.read_parquet(path)
            else:
                console.print(f"[dim]Reading JSON from {cloud_type}: {path}[/dim]")
                ds = ray.data.read_json(path)
        else:
            ds = self._load_huggingface_as_ray()

        if self.limit:
            ds = ds.limit(self.limit)

        return ds

    def _is_cloud_path(self, path: str) -> bool:
        """Check if path is a cloud storage path."""
        cloud_prefixes = ("s3://", "gs://", "az://", "abfs://", "abfss://")
        return path.startswith(cloud_prefixes)

    def _get_cloud_type(self, path: str) -> str:
        """Get human-readable cloud provider name."""
        if path.startswith("s3://"):
            return "S3"
        elif path.startswith("gs://"):
            return "GCS"
        elif path.startswith(("az://", "abfs://", "abfss://")):
            return "Azure"
        return "cloud"

    def _load_huggingface_as_ray(self):
        """Load HuggingFace dataset into Ray Data via streaming."""
        import ray
        import ray.data
        import pandas as pd
        from rich.console import Console
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TaskProgressColumn,
        )

        console = Console()
        console.print(f"[dim]Streaming from HuggingFace: {self.dataset_path}[/dim]")

        # Use HF streaming to avoid loading full dataset into memory
        hf_stream = load_dataset(
            self.dataset_path,
            self.subset,
            split=self.split,
            streaming=True,
        )

        # Stream batches and put into Ray object store
        refs = []
        batch_size = 10000
        total_items = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[items]} items[/dim]"),
            transient=True,
        ) as progress:
            task = progress.add_task("Streaming dataset...", total=None, items=0)

            for batch in hf_stream.batch(batch_size=batch_size):
                # batch is a dict of lists, convert to DataFrame
                df = pd.DataFrame(batch)
                refs.append(ray.put(df))
                total_items += len(df)
                progress.update(task, items=f"{total_items:,}")

        return ray.data.from_pandas_refs(refs)

    def _is_local_file(self, path: str) -> bool:
        """Check if path is local file or directory."""
        p = Path(path).expanduser()
        return p.exists() or path.startswith(("./", "/", "~"))

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
                # Detect file type from extension
                path_lower = self.dataset_path.lower()
                if path_lower.endswith(".parquet"):
                    file_type = "parquet"
                elif path_lower.endswith((".json", ".jsonl")):
                    file_type = "json"
                else:
                    file_type = "json"  # Default fallback
                dataset = load_dataset(
                    file_type, data_files=self.dataset_path, split=split_str
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

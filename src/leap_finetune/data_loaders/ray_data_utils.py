import ray.data
from datasets import Dataset
from rich.console import Console

from .dataset_loader import DatasetLoader
from .validate_loader import get_row_filter, normalize_columns


def create_ray_datasets(
    loader: DatasetLoader,
    shuffle_seed: int = 42,
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Create validated, shuffled, split Ray Datasets from a DatasetLoader.

    Pipeline: quick_validate → load → [preprocess] → filter → normalize → shuffle → split

    Uses Ray Data native operations (filter/map) - no pandas/arrow imports needed.
    """
    console = Console()

    # Quick schema validation on small sample before full load
    loader.quick_validate()

    ds = loader.to_ray_dataset()

    # Apply user preprocessing if provided (before validation)
    if loader.preprocess_fn is not None:
        console.print("[dim]Applying preprocessing...[/dim]")
        ds = loader.preprocess_fn(ds)

    # Filter invalid rows using Ray's native filter (pure Python, Ray handles Arrow)
    row_filter = get_row_filter(loader.dataset_type)
    ds = ds.filter(row_filter)

    # Normalize column names
    normalizer = normalize_columns(loader.dataset_type)
    ds = ds.map(normalizer)

    # Shuffle before split
    ds = ds.random_shuffle(seed=shuffle_seed)

    # Materialize to get count
    total_count = ds.count()

    if total_count == 0:
        raise ValueError(
            f"Dataset is empty after validation. Check if your data matches "
            f"the expected format for dataset_type='{loader.dataset_type}'"
        )

    # Calculate split sizes
    eval_count = max(1, int(total_count * loader.test_size))
    train_count = total_count - eval_count

    if train_count < 1:
        raise ValueError(
            f"Not enough samples ({total_count}) for train/eval split "
            f"with test_size={loader.test_size}"
        )

    # Split into train/eval
    train_ds, eval_ds = ds.split_at_indices([train_count])

    console.print(
        f"[green]✓ Dataset ready:[/green] {total_count:,} samples "
        f"(train: {train_count:,}, eval: {eval_count:,})"
    )

    return train_ds, eval_ds


def ray_dataset_to_hf(ray_ds) -> Dataset:
    """
    Convert Ray Dataset shard to HuggingFace Dataset.

    ray.train.get_dataset_shard() returns a DataIterator.
    We iterate rows as native Python dicts - simple and works.
    """
    rows = list(ray_ds.iter_rows())
    if not rows:
        return Dataset.from_dict({})
    return Dataset.from_list(rows)

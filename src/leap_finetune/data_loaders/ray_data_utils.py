import types

import ray.data
import torch
import torch.distributed as dist
from datasets import Dataset
from rich.console import Console
from torch.utils.data import DataLoader, RandomSampler

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


def patch_train_dataloader(trainer) -> None:
    """
    Override get_train_dataloader to skip accelerator.prepare().

    Ray Data already shards across workers (each gets 1/N of data).
    Without this patch, accelerator.prepare() wraps the DataLoader with
    BatchSamplerShard, splitting each shard by N again → 1/N² per worker.

    TEMPORARY: proper fix is to run tokenization/packing as a Ray .map() step
    so iter_torch_batches works directly with prepare_trainer.
    """

    def get_train_dataloader(self):
        dataset = self.train_dataset
        if hasattr(self, "_remove_unused_columns"):
            dataset = self._remove_unused_columns(dataset, description="training")

        # Sync packed dataset length across workers to prevent NCCL deadlocks
        if dist.is_initialized():
            local_len = torch.tensor(
                len(dataset), dtype=torch.long, device=self.args.device
            )
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            min_len = int(local_len.item())
            if len(dataset) > min_len:
                dataset = dataset.select(range(min_len))

        return DataLoader(
            dataset,
            batch_size=self._train_batch_size,
            sampler=RandomSampler(dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    trainer.get_train_dataloader = types.MethodType(get_train_dataloader, trainer)


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

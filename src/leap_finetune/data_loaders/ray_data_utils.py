import hashlib
import json
import logging

import ray.data
from datasets import Dataset
from rich.console import Console

from leap_finetune.utils.constants import TOKENIZATION_CACHE_DIR

from .dataset_loader import DatasetLoader
from .tokenize_data import tokenize_and_pack_sft, tokenize_dpo_dataset
from .validate_loader import get_row_filter, normalize_columns

logger = logging.getLogger(__name__)


# === Tokenization Cache ===


def _build_cache_key(
    loader: DatasetLoader,
    shuffle_seed: int,
    tokenizer_id: str,
    training_config: dict,
) -> tuple[str, dict]:
    """Build a deterministic cache key from all parameters affecting tokenized output."""
    dataset_type = loader.dataset_type

    key = {
        "dataset_path": loader.dataset_path,
        "subset": loader.subset,
        "split": loader.split,
        "limit": loader.limit,
        "test_size": loader.test_size,
        "dataset_type": dataset_type,
        "tokenizer": tokenizer_id,
        "shuffle_seed": shuffle_seed,
    }

    if dataset_type == "sft":
        key["max_length"] = training_config.get("max_length", 2048)
        key["packing"] = training_config.get("packing", False)
    elif dataset_type == "dpo":
        key["max_prompt_length"] = training_config.get("max_prompt_length")
        key["max_completion_length"] = training_config.get("max_completion_length")

    canonical = json.dumps(key, sort_keys=True)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return fingerprint, key


def _try_load_cache(
    fingerprint: str,
) -> tuple[ray.data.Dataset, ray.data.Dataset] | None:
    """Load cached train/eval parquet if the cache directory exists."""
    cache_dir = TOKENIZATION_CACHE_DIR / fingerprint
    train_dir = cache_dir / "train"
    eval_dir = cache_dir / "eval"

    if not train_dir.exists() or not eval_dir.exists():
        return None

    try:
        train_ds = ray.data.read_parquet(str(train_dir))
        eval_ds = ray.data.read_parquet(str(eval_dir))
        return train_ds, eval_ds
    except Exception:
        logger.warning("Failed to read tokenization cache, will re-tokenize")
        return None


def _save_cache(
    fingerprint: str,
    train_ds: ray.data.Dataset,
    eval_ds: ray.data.Dataset,
    key_dict: dict,
) -> None:
    """Write train/eval datasets as parquet into the cache directory."""
    cache_dir = TOKENIZATION_CACHE_DIR / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds.write_parquet(str(cache_dir / "train"))
    eval_ds.write_parquet(str(cache_dir / "eval"))

    (cache_dir / "fingerprint.json").write_text(json.dumps(key_dict, indent=2))


def create_ray_datasets(
    loader: DatasetLoader,
    shuffle_seed: int = 42,
    tokenizer=None,
    training_config: dict | None = None,
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Create validated, shuffled, split Ray Datasets from a DatasetLoader.

    Pipeline: quick_validate → load → normalize → filter → shuffle → split → [tokenize/pack]

    When tokenizer is provided, tokenization and optional packing happen
    centrally before sharding, producing equal-length shards (±1 row).
    Tokenized results are cached as Parquet for subsequent runs.
    """
    console = Console()

    # === Check tokenization cache ===
    use_pretokenize = tokenizer is not None and training_config is not None
    can_cache = use_pretokenize and loader.cache_dataset
    fingerprint = None
    key_dict = None

    if can_cache:
        fingerprint, key_dict = _build_cache_key(
            loader, shuffle_seed, tokenizer.name_or_path, training_config
        )
        cached = _try_load_cache(fingerprint)
        if cached is not None:
            train_ds, eval_ds = cached
            train_count = train_ds.count()
            eval_count = eval_ds.count()
            console.print(
                f"[green]✓ Cache hit[/green] [dim]({fingerprint})[/dim]: "
                f"{train_count + eval_count:,} samples "
                f"(train: {train_count:,}, eval: {eval_count:,})"
            )
            return train_ds, eval_ds
        logger.info(
            "Tokenization cache miss (%s), will tokenize and cache", fingerprint
        )

    # === Full pipeline: load → filter → normalize → shuffle → split → tokenize ===

    loader.quick_validate()
    ds = loader.to_ray_dataset()

    # Normalize column names/formats before filtering
    # (handles JSON string conversations, column renames, image_root prefix)
    normalizer = normalize_columns(loader.dataset_type, image_root=loader.image_root)
    ds = ds.map(normalizer)

    # Filter invalid rows using Ray's native filter (pure Python, Ray handles Arrow)
    if loader.dataset_type == "vlm_sft":
        console.print(
            "[dim]Filtering VLM samples (validating images across workers)...[/dim]"
        )
    row_filter = get_row_filter(loader.dataset_type)
    ds = ds.filter(row_filter)

    ds = ds.random_shuffle(seed=shuffle_seed)
    total_count = ds.count()

    if total_count == 0:
        raise ValueError(
            f"Dataset is empty after validation. Check if your data matches "
            f"the expected format for dataset_type='{loader.dataset_type}'"
        )

    eval_count = max(1, int(total_count * loader.test_size))
    train_count = total_count - eval_count

    if train_count < 1:
        raise ValueError(
            f"Not enough samples ({total_count}) for train/eval split "
            f"with test_size={loader.test_size}"
        )

    train_ds, eval_ds = ds.split_at_indices([train_count])

    console.print(
        f"[green]✓ Dataset ready:[/green] {total_count:,} samples "
        f"(train: {train_count:,}, eval: {eval_count:,})"
    )

    # === Pre-tokenize if tokenizer provided ===
    if use_pretokenize:
        dataset_type = loader.dataset_type

        if dataset_type == "sft":
            max_length = training_config.get("max_length", 2048)
            packing = training_config.get("packing", False)

            console.print(
                f"[dim]Tokenizing SFT (max_length={max_length}, packing={packing})...[/dim]"
            )
            train_ds = tokenize_and_pack_sft(train_ds, tokenizer, max_length, packing)
            eval_ds = tokenize_and_pack_sft(
                eval_ds, tokenizer, max_length, packing=False
            )

        elif dataset_type == "dpo":
            max_prompt_length = training_config.get("max_prompt_length")
            max_completion_length = training_config.get("max_completion_length")

            console.print("[dim]Tokenizing DPO...[/dim]")
            train_ds = tokenize_dpo_dataset(
                train_ds, tokenizer, max_prompt_length, max_completion_length
            )
            eval_ds = tokenize_dpo_dataset(
                eval_ds, tokenizer, max_prompt_length, max_completion_length
            )

        # === Save to cache ===
        if can_cache and fingerprint is not None:
            try:
                _save_cache(fingerprint, train_ds, eval_ds, key_dict)
                console.print(f"[dim]Cached tokenized data ({fingerprint})[/dim]")
            except Exception:
                logger.warning(
                    "Failed to write tokenization cache, continuing without cache"
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

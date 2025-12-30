import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
import pandas as pd
from datasets import Dataset, load_dataset


# ============================================================================
# QUICK VALIDATION (Pre-Ray, ~10 samples)
# ============================================================================


def _is_cloud_path(path: str) -> bool:
    """Check if path is a cloud storage path (S3, GCS, Azure)."""
    cloud_prefixes = ("s3://", "gs://", "az://", "abfs://", "abfss://")
    return path.startswith(cloud_prefixes)


def _get_source_type(dataset_path: str) -> str:
    """Determine the source type for display and loading logic."""
    path_lower = dataset_path.lower()

    if _is_cloud_path(dataset_path):
        if path_lower.startswith("s3://"):
            return "s3"
        elif path_lower.startswith("gs://"):
            return "gcs"
        elif path_lower.startswith(("az://", "abfs://", "abfss://")):
            return "azure"
        return "cloud"
    elif Path(dataset_path).exists() or dataset_path.startswith(("./", "/", "~")):
        if path_lower.endswith(".parquet"):
            return "parquet"
        else:
            return "jsonl"
    else:
        return "huggingface"


def quick_validate_schema(
    dataset_path: str,
    dataset_type: str,
    subset: str | None = None,
    split: str = "train",
    num_samples: int = 10,
) -> None:
    """
    Fast schema validation on small sample. Fails fast on obvious errors.
    Runs in main process before Ray starts.

    Uses the same thorough validation as full validation, just on fewer samples.
    """
    console = Console()

    source_type = _get_source_type(dataset_path)
    console.print(
        f"[dim]Validating {source_type} schema ({num_samples} samples)...[/dim]"
    )

    sample_ds = _load_sample_dataset(dataset_path, subset, split, num_samples)

    if len(sample_ds) == 0:
        raise ValueError(f"Dataset appears to be empty: {dataset_path}")

    # Use the same validation as full validation
    validate_dataset_format(sample_ds, dataset_type)

    console.print("[green]✓ Schema validated[/green]")


def _load_sample_dataset(
    dataset_path: str,
    subset: str | None,
    split: str,
    num_samples: int,
) -> Dataset:
    """Load a small sample as HF Dataset."""
    try:
        source_type = _get_source_type(dataset_path)

        if source_type in ("s3", "gcs", "azure", "cloud"):
            # Cloud storage - use Ray Data to read samples
            import ray.data

            path_lower = dataset_path.lower()
            if ".parquet" in path_lower or ".pq" in path_lower:
                ray_ds = ray.data.read_parquet(dataset_path)
            else:
                ray_ds = ray.data.read_json(dataset_path)
            # Take only num_samples
            samples = list(ray_ds.limit(num_samples).iter_rows())
            return Dataset.from_list(samples)

        elif source_type in ("parquet", "jsonl"):
            # Local file - use HuggingFace
            path_lower = dataset_path.lower()
            if path_lower.endswith(".parquet"):
                file_type = "parquet"
            else:
                file_type = "json"
            return load_dataset(
                file_type, data_files=dataset_path, split=f"{split}[:{num_samples}]"
            )

        else:
            # HuggingFace Hub - use streaming then convert to Dataset
            ds_stream = load_dataset(dataset_path, subset, split=split, streaming=True)
            samples = []
            for i, item in enumerate(ds_stream):
                if i >= num_samples:
                    break
                samples.append(item)
            return Dataset.from_list(samples)
    except Exception as e:
        raise ValueError(f"Failed to load dataset samples from '{dataset_path}': {e}")


# ============================================================================
# DISTRIBUTED FILTERING (Ray Data native operations)
# ============================================================================


def get_row_filter(dataset_type: str) -> Callable[[dict], bool]:
    """
    Get a row filter function for ray.data.filter().
    Uses pure Python - Ray handles Arrow/serialization internally.
    """

    def is_valid_sft(row: dict) -> bool:
        """Check if row has valid SFT conversational format."""
        # Find the messages column
        messages = None
        for col in ["messages", "conversations", "chat", "dialogue"]:
            if col in row and row[col]:
                messages = row[col]
                break

        if not messages or len(messages) == 0:
            return False

        first = messages[0]
        return isinstance(first, dict) and "role" in first and "content" in first

    def is_valid_dpo(row: dict) -> bool:
        """Check if row has valid DPO format."""
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if not chosen or not rejected:
            return False

        return chosen != rejected

    if dataset_type == "sft":
        return is_valid_sft
    elif dataset_type == "dpo":
        return is_valid_dpo
    else:
        return lambda row: True  # VLM and others pass through


def normalize_columns(dataset_type: str):
    """
    Get a row transform function to normalize column names.
    For use with ray.data.map() if needed.
    """

    def normalize_sft(row: dict) -> dict:
        # Rename conversation column to 'messages' if needed
        for col in ["conversations", "chat", "dialogue"]:
            if col in row and "messages" not in row:
                row["messages"] = row.pop(col)
                break
        return row

    def add_dpo_prompt(row: dict) -> dict:
        if "prompt" not in row or not row["prompt"]:
            # Extract prompt from chosen
            chosen = row.get("chosen", [])
            if isinstance(chosen, list):
                for msg in chosen:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        row["prompt"] = msg.get("content", "")
                        break
                else:
                    row["prompt"] = ""
            else:
                row["prompt"] = ""
        return row

    if dataset_type == "sft":
        return normalize_sft
    elif dataset_type == "dpo":
        return add_dpo_prompt
    else:
        return lambda row: row


def _find_messages_column(batch: pd.DataFrame) -> str | None:
    """Find the conversational column in a batch."""
    for col in ["messages", "conversations", "chat", "dialogue"]:
        if col in batch.columns:
            return col
    for col in batch.columns:
        if len(batch) > 0 and _is_valid_conversation(batch[col].iloc[0]):
            return col
    return None


def _is_valid_conversation(content: Any) -> bool:
    """Check if content is valid conversational format.

    Accepts both Python lists and numpy arrays (parquet returns ndarray).
    Validates structure: list of dicts with 'role' and 'content' keys.
    """
    import numpy as np

    # Must be a sequence type (list or numpy array)
    if not isinstance(content, (list, np.ndarray)):
        return False

    # Must have at least one message
    if len(content) == 0:
        return False

    # First message must have required fields
    first = content[0]
    if not isinstance(first, dict):
        return False

    return "role" in first and "content" in first


def _extract_prompt(chosen: Any) -> str:
    """Extract prompt from chosen (conversational format)."""
    if isinstance(chosen, list):
        for msg in chosen:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_data_loader(func):
    """Decorator that validates function returns tuple[Dataset, Dataset] for custom data loaders"""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Dataset, Dataset]:
        result = func(*args, **kwargs)

        # Validate structure
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                f"{func.__name__} must return tuple[Dataset, Dataset], got {type(result)}"
            )

        train, test = result
        if not isinstance(train, Dataset) or not isinstance(test, Dataset):
            raise TypeError(f"{func.__name__} must return tuple of Dataset instances")

        return result

    return wrapper


def validate_dataset_format(dataset: Dataset, dataset_type: str) -> Dataset:
    """Validate and convert dataset format based on dataset_type"""

    if dataset_type == "sft":
        return validate_sft_format(dataset)
    elif dataset_type == "dpo":
        return validate_dpo_format(dataset)
    elif dataset_type == "vlm_sft":
        return validate_vlm_sft_format(dataset)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def validate_sft_format(dataset: Dataset) -> Dataset:
    """Validate and convert SFT dataset to proper format."""
    columns = dataset.column_names

    if any(col in columns for col in ["chosen", "rejected"]):
        raise ValueError("This is a DPO dataset, not SFT. Use dataset_type='dpo'")

    # Find the conversational column
    conv_col = _find_conversational_column(dataset, columns)
    if conv_col is None:
        raise ValueError(
            f"No conversational column found. Expected 'messages' with format: "
            f"[{{'role': 'user', 'content': '...'}}]. Found columns: {columns}"
        )

    # Validate ALL samples
    invalid_indices = []
    for i in range(len(dataset)):
        if not _is_valid_conversation(dataset[i][conv_col]):
            invalid_indices.append(i)

    if invalid_indices:
        # Show first few invalid indices
        shown = invalid_indices[:5]
        msg = f"Found {len(invalid_indices)} invalid samples (indices: {shown}"
        if len(invalid_indices) > 5:
            msg += f"... and {len(invalid_indices) - 5} more"
        msg += "). Each message must have 'role' and 'content' fields."
        raise ValueError(msg)

    # Rename column if needed
    if conv_col != "messages":
        return dataset.rename_column(conv_col, "messages")

    return dataset


def _find_conversational_column(dataset: Dataset, columns: list) -> str | None:
    """Find the column containing conversational data."""
    # Check known column names first
    for col in ["messages", "conversations", "chat", "dialogue"]:
        if col in columns and len(dataset) > 0:
            if _is_valid_conversation(dataset[0][col]):
                return col

    # Fall back to checking all columns
    for col in columns:
        if len(dataset) > 0 and _is_valid_conversation(dataset[0].get(col)):
            return col

    return None


def validate_dpo_format(dataset: Dataset) -> Dataset:
    """Validate and convert DPO dataset to proper format."""
    columns = set(dataset.column_names)

    # Check required columns
    if not {"chosen", "rejected"}.issubset(columns):
        raise ValueError(
            f"DPO needs 'chosen' and 'rejected' columns. Found: {list(columns)}"
        )

    # Validate ALL samples
    invalid_indices = []
    identical_indices = []

    for i in range(len(dataset)):
        chosen = dataset[i]["chosen"]
        rejected = dataset[i]["rejected"]

        # Check non-empty
        if not chosen or not rejected:
            invalid_indices.append(i)
            continue

        # Check chosen != rejected
        if chosen == rejected:
            identical_indices.append(i)

    if invalid_indices:
        shown = invalid_indices[:5]
        raise ValueError(
            f"Found {len(invalid_indices)} samples with empty chosen/rejected "
            f"(indices: {shown}{'...' if len(invalid_indices) > 5 else ''})"
        )

    if identical_indices:
        shown = identical_indices[:5]
        raise ValueError(
            f"Found {len(identical_indices)} samples where chosen == rejected "
            f"(indices: {shown}{'...' if len(identical_indices) > 5 else ''})"
        )

    # Add prompt if missing
    if "prompt" in columns:
        return dataset

    return dataset.map(lambda x: {**x, "prompt": _extract_prompt(x["chosen"])})


def validate_vlm_sft_format(dataset: Dataset) -> Dataset:
    """
    Comprehensive validation VLM dataset format.

    Expected format:
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/path/to/image.jpg"},
                    {"type": "text", "text": "What do you see?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Response..."}]
            }
        ]
    }
    """
    # Check basic structure
    columns = dataset.column_names
    if "messages" not in columns:
        raise ValueError(f"Dataset missing 'messages' column. Found columns: {columns}")

    # Validate a few samples for detailed structure
    sample_indices = [0, min(5, len(dataset) - 1), min(50, len(dataset) - 1)]

    for idx in sample_indices:
        if idx >= len(dataset):
            continue

        sample = dataset[idx]
        messages = sample["messages"]

        # Check messages is a list
        if not isinstance(messages, list):
            raise ValueError(
                f"Sample {idx}: 'messages' must be a list, got {type(messages)}"
            )

        if len(messages) == 0:
            raise ValueError(f"Sample {idx}: 'messages' cannot be empty")

        # Validate each message in the conversation
        for msg_idx, message in enumerate(messages):
            # Check message structure
            if not isinstance(message, dict):
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: message must be dict, got {type(message)}"
                )

            if "role" not in message:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: missing 'role' field"
                )

            if "content" not in message:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: missing 'content' field"
                )

            # Validate role
            role = message["role"]
            if role not in ["user", "assistant", "system"]:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: invalid role '{role}', expected user/assistant/system"
                )

            # Validate content
            content = message["content"]
            if not isinstance(content, list):
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: 'content' must be list, got {type(content)}"
                )

            if len(content) == 0:
                raise ValueError(
                    f"Sample {idx}, message {msg_idx}: 'content' cannot be empty"
                )

            # Validate each content item
            for content_idx, content_item in enumerate(content):
                if not isinstance(content_item, dict):
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: must be dict, got {type(content_item)}"
                    )

                if "type" not in content_item:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: missing 'type' field"
                    )

                content_type = content_item["type"]

                if content_type == "text":
                    if "text" not in content_item:
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: text content missing 'text' field"
                        )
                    if not isinstance(content_item["text"], str):
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: 'text' must be string"
                        )

                elif content_type == "image":
                    if "image" not in content_item:
                        raise ValueError(
                            f"Sample {idx}, message {msg_idx}, content {content_idx}: image content missing 'image' field"
                        )

                    image_data = content_item["image"]

                    # Check if image is a string path (correct) or PIL object (incorrect)
                    if not isinstance(image_data, str):
                        from PIL import Image

                        if isinstance(image_data, Image.Image):
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image must be path string, not PIL Image object. Use image paths for Ray Train compatibility."
                            )
                        elif isinstance(image_data, dict):
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image is dict {image_data}, expected path string"
                            )
                        else:
                            raise ValueError(
                                f"Sample {idx}, message {msg_idx}, content {content_idx}: image must be path string, got {type(image_data)}"
                            )

                    # Check if path exists (optional warning) - only for local paths
                    if image_data.startswith(("http://", "https://")):
                        # Skip validation for URLs - they'll be validated during actual loading
                        pass
                    elif not os.path.exists(image_data):
                        print(
                            f"⚠️  Warning: Local image path does not exist: {image_data}"
                        )

                else:
                    raise ValueError(
                        f"Sample {idx}, message {msg_idx}, content {content_idx}: unsupported content type '{content_type}', expected 'text' or 'image'"
                    )

    print(
        f"✅ Dataset validation passed! {len(dataset)} samples with expected VLM format"
    )
    return dataset

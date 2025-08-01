from functools import wraps
from typing import Any
from datasets import Dataset
from trl import extract_prompt


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
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def validate_sft_format(dataset: Dataset) -> Dataset:
    """Validate and convert SFT dataset to proper format"""
    columns = dataset.column_names

    if any(col in columns for col in ["chosen", "rejected"]):
        raise ValueError("This is a DPO dataset, not SFT. Use dataset_type='dpo'")

    if "messages" in columns and is_valid_conversational_format(dataset, "messages"):
        return dataset

    for col in columns:
        if is_valid_conversational_format(dataset, col):
            return dataset.rename_column(col, "messages")

    raise ValueError(
        f"No conversational column found. Expected: [{'role': 'user', 'content': '...'}]"
    )


def validate_dpo_format(dataset: Dataset) -> Dataset:
    """Validate and convert DPO dataset to proper format"""
    columns = set(dataset.column_names)

    # Check required columns
    if not {"chosen", "rejected"}.issubset(columns):
        raise ValueError(
            f"DPO needs 'chosen' and 'rejected' columns. Found: {list(columns)}"
        )

    if "prompt" in columns:
        return dataset

    try:
        return dataset.map(extract_prompt)
    except Exception as e:
        raise ValueError(f"Failed to convert DPO format: {e}")


def is_valid_conversational_format(dataset: Dataset, column_name: str) -> bool:
    """Check for ChatML format"""
    try:
        sample = dataset[0][column_name]
        if not isinstance(sample, list) or len(sample) == 0:
            return False

        first_message = sample[0]
        return (
            isinstance(first_message, dict)
            and "role" in first_message
            and "content" in first_message
        )

    except (KeyError, IndexError, TypeError):
        return False

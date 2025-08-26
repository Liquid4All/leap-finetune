import os
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
    elif dataset_type == "vlm_sft":
        return validate_vlm_sft_format(dataset)
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

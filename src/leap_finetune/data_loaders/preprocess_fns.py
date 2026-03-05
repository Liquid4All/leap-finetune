"""Built-in preprocessing functions for DatasetLoader.preprocess_fn."""

import json
import os
from functools import partial


def get_preprocess_fn(name: str, ds_config: dict):
    """Return a row-level transform function based on name in config.yaml."""
    if name == "vlm_parquet":
        image_base_path = ds_config.get("image_base_path", "")
        return partial(vlm_parquet_transform, image_base_path=image_base_path)
    else:
        raise ValueError(
            f"Unknown preprocess function: '{name}'. Available: ['vlm_parquet']"
        )


def vlm_parquet_transform(row: dict, image_base_path: str = "") -> dict:
    """Transform a VLM parquet row: parse JSON conversation, prepend image paths."""
    conv = row.get("conversation")
    if isinstance(conv, str):
        messages = json.loads(conv)
    elif isinstance(conv, list):
        messages = conv
    else:
        row["messages"] = []
        return row

    if image_base_path:
        for message in messages:
            if message.get("role") == "user" and isinstance(
                message.get("content"), list
            ):
                for content_item in message["content"]:
                    if content_item.get("type") == "image":
                        img_path = content_item.get("image", "")
                        if (
                            img_path
                            and not os.path.isabs(img_path)
                            and not img_path.startswith(("http://", "https://"))
                        ):
                            content_item["image"] = os.path.join(
                                image_base_path, img_path
                            )

    row["messages"] = messages
    return row

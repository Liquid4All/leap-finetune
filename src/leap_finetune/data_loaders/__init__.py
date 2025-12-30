from enum import Enum

from .basic_examples import load_orpo_dpo_dataset, load_smoltalk_dataset
from .dataset_loader import DatasetLoader
from .ray_data_utils import create_ray_datasets, ray_dataset_to_hf
from .validate_loader import (
    get_row_filter,
    normalize_columns,
    quick_validate_schema,
    validate_data_loader,
    validate_dataset_format,
)


class SFTDataLoader(Enum):
    SMOLTALK = load_smoltalk_dataset


class DPODataLoader(Enum):
    ORPO_DPO = load_orpo_dpo_dataset


__all__ = [
    "DatasetLoader",
    "create_ray_datasets",
    "ray_dataset_to_hf",
    "quick_validate_schema",
    "get_row_filter",
    "normalize_columns",
    "validate_data_loader",
    "validate_dataset_format",
    "load_smoltalk_dataset",
    "load_orpo_dpo_dataset",
    "SFTDataLoader",
    "DPODataLoader",
]

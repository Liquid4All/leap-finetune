from .basic_examples import load_smoltalk_dataset, load_orpo_dpo_dataset
from .weave_preprocessing import (
    get_messages_by_role,
    preprocess_datum,
    preprocess_dataset_for_weave_evaluation,
    preprocess_dataset_for_weave_evaluation_list,
)
from enum import Enum


class SFTDataLoader(Enum):
    SMOLTALK = load_smoltalk_dataset


class DPODataLoader(Enum):
    ORPO_DPO = load_orpo_dpo_dataset


__all__ = [
    "SFTDataLoader",
    "DPODataLoader",
    "load_smoltalk_dataset",
    "load_orpo_dpo_dataset",
    "get_messages_by_role",
    "preprocess_datum",
    "preprocess_dataset_for_weave_evaluation",
    "preprocess_dataset_for_weave_evaluation_list",
]

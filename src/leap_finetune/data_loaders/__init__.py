from .basic_examples import load_smoltalk_dataset, load_orpo_dpo_dataset
from enum import Enum


class SFTDataLoader(Enum):
    SMOLTALK = load_smoltalk_dataset


class DPODataLoader(Enum):
    ORPO_DPO = load_orpo_dpo_dataset

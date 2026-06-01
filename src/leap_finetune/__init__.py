import os
from pathlib import Path

from leap_finetune.cli.main import main, run_config

HOME = Path.home()

_current_file = Path(__file__).resolve()
LEAP_FINETUNE_DIR = _current_file.parent.parent.parent
LEAP_FINETUNE_DIR = Path(os.getenv("LEAP_FINETUNE_DIR", LEAP_FINETUNE_DIR))

RUNTIME_DIR = LEAP_FINETUNE_DIR / "src" / "leap_finetune"

BASE_OUTPUT_PATH = LEAP_FINETUNE_DIR / "outputs"
SFT_OUTPUT_PATH = BASE_OUTPUT_PATH / "sft"
DPO_OUTPUT_PATH = BASE_OUTPUT_PATH / "dpo"
GRPO_OUTPUT_PATH = BASE_OUTPUT_PATH / "grpo"

TOKENIZATION_CACHE_DIR = LEAP_FINETUNE_DIR / ".cache" / "tokenized"

__all__ = [
    "BASE_OUTPUT_PATH",
    "DPO_OUTPUT_PATH",
    "GRPO_OUTPUT_PATH",
    "HOME",
    "LEAP_FINETUNE_DIR",
    "RUNTIME_DIR",
    "SFT_OUTPUT_PATH",
    "TOKENIZATION_CACHE_DIR",
    "main",
    "run_config",
]

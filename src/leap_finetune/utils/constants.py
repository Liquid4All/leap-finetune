from pathlib import Path
import os

HOME = Path.home()

# Auto-detect project root based on this file's location (3 levels up)
_current_file = Path(__file__).resolve()
LEAP_FINETUNE_DIR = _current_file.parent.parent.parent.parent
LEAP_FINETUNE_DIR = Path(os.getenv("LEAP_FINETUNE_DIR", LEAP_FINETUNE_DIR))

RUNTIME_DIR = LEAP_FINETUNE_DIR / "src" / "leap_finetune"

# Base output paths (Resolved in utils/output_paths.py)
BASE_OUTPUT_PATH = LEAP_FINETUNE_DIR / "outputs"
SFT_OUTPUT_PATH = BASE_OUTPUT_PATH / "sft"
DPO_OUTPUT_PATH = BASE_OUTPUT_PATH / "dpo"

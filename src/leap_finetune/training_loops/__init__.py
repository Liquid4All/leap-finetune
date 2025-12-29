from .dpo_run import dpo_run
from .sft_run import sft_run
from .vlm_sft_run import vlm_sft_run

TRAINING_LOOPS = {
    "sft": sft_run,
    "dpo": dpo_run,
    "vlm_sft": vlm_sft_run,
}

__all__ = ["TRAINING_LOOPS", "sft_run", "dpo_run", "vlm_sft_run"]

from .dpo import dpo_run
from .grpo import grpo_run
from .moe_dpo import moe_dpo_run
from .moe_sft import moe_sft_run
from .sft import sft_run
from .vlm_grpo import vlm_grpo_run
from .vlm_sft import vlm_sft_run

TRAINING_LOOPS = {
    "sft": sft_run,
    "dpo": dpo_run,
    "vlm_sft": vlm_sft_run,
    "grpo": grpo_run,
    "vlm_grpo": vlm_grpo_run,
    "moe_sft": moe_sft_run,
    "moe_dpo": moe_dpo_run,
}

__all__ = [
    "TRAINING_LOOPS",
    "sft_run",
    "dpo_run",
    "vlm_sft_run",
    "grpo_run",
    "vlm_grpo_run",
    "moe_sft_run",
    "moe_dpo_run",
]

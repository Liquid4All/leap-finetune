# === Shipped GRPO task bundles ===

from .gsm8k import GSM8KRecipe, gsm8k_reward
from .ifeval import IFEvalRecipe, ifeval_reward
from .mcqa import MCQARecipe, mcqa_reward
from .vlm_grounding import (
    VLMGroundingCIoURecipe,
    VLMGroundingIoURecipe,
    ciou_f1_reward,
    iou_f1_reward,
    strict_format_reward,
)

__all__ = [
    "GSM8KRecipe",
    "IFEvalRecipe",
    "MCQARecipe",
    "gsm8k_reward",
    "ifeval_reward",
    "mcqa_reward",
    "VLMGroundingCIoURecipe",
    "VLMGroundingIoURecipe",
    "ciou_f1_reward",
    "iou_f1_reward",
    "strict_format_reward",
]

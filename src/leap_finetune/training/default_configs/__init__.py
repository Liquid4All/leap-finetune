from leap_finetune.training.default_configs.dpo_configs import (
    DEFAULT_DPO,
    DEFAULT_VLM_DPO,
    MOE_DPO,
)
from leap_finetune.training.default_configs.grpo_configs import (
    DEFAULT_GRPO,
    DEFAULT_VLM_GRPO,
    MOE_GRPO,
)
from leap_finetune.training.default_configs.sft_configs import DEFAULT_SFT, MOE_SFT
from leap_finetune.training.default_configs.vlm_sft_configs import DEFAULT_VLM_SFT
from leap_finetune.config.job_config import TrainingConfig
from leap_finetune.training.peft.peft_configs import (
    DEFAULT_LORA,
    DEFAULT_VLM_LORA,
    HIGH_R_LORA,
    MINIMAL_VLM_LORA,
    MOE_LORA,
    MOE_LORA_HIGH_R,
)

TRAINING_TYPE_DEFAULTS = {
    "sft": TrainingConfig.model_validate(DEFAULT_SFT),
    "dpo": TrainingConfig.model_validate(DEFAULT_DPO),
    "vlm_sft": TrainingConfig.model_validate(DEFAULT_VLM_SFT),
    "vlm_dpo": TrainingConfig.model_validate(DEFAULT_VLM_DPO),
    "moe_sft": TrainingConfig.model_validate(MOE_SFT),
    "moe_dpo": TrainingConfig.model_validate(MOE_DPO),
    "grpo": TrainingConfig.model_validate(DEFAULT_GRPO),
    "vlm_grpo": TrainingConfig.model_validate(DEFAULT_VLM_GRPO),
}

__all__ = [
    "DEFAULT_DPO",
    "DEFAULT_GRPO",
    "DEFAULT_LORA",
    "DEFAULT_SFT",
    "DEFAULT_VLM_DPO",
    "DEFAULT_VLM_GRPO",
    "DEFAULT_VLM_LORA",
    "DEFAULT_VLM_SFT",
    "HIGH_R_LORA",
    "MINIMAL_VLM_LORA",
    "MOE_DPO",
    "MOE_GRPO",
    "MOE_LORA",
    "MOE_LORA_HIGH_R",
    "MOE_SFT",
    "TRAINING_TYPE_DEFAULTS",
]

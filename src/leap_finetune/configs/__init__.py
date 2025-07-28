from enum import Enum
from .dpo_configs import (
    DEFAULT_DPO_CONFIG,
)
from .sft_configs import (
    DEFAULT_SFT_CONFIG,
)
from .peft_configs import (
    LFM2_LORA_DEFAULT_CONFIG,
    LFM2_LORA_HIGH_R_CONFIG,
)


class TrainingConfig(Enum):
    DEFAULT_SFT = DEFAULT_SFT_CONFIG
    DEFAULT_DPO = DEFAULT_DPO_CONFIG

    def override(self, **overrides):
        """Create a custom TrainingConfig with overrides"""

        config_dict = self.value.copy()
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
        config_dict.update(filtered_overrides)

        class _CustomTrainingConfig(Enum):
            CUSTOM = config_dict

        return _CustomTrainingConfig.CUSTOM


class PeftConfig(Enum):
    DEFAULT_LORA = LFM2_LORA_DEFAULT_CONFIG
    HIGH_R_LORA = LFM2_LORA_HIGH_R_CONFIG
    NO_LORA = None

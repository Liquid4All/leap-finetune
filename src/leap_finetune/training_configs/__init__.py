import inspect
from enum import Enum

from . import dpo_configs, peft_configs, sft_configs, vlm_sft_config


def _discover_configs(
    module,
    exclude_prefixes=(
        "DEEPSPEED",
        "MOE_DEEPSPEED",
        "GLU",
        "MHA",
        "CONV",
        "LFM",
        "VISION",
        "MULTI",
    ),
):
    configs = {}
    for name, value in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if isinstance(value, dict) or (
            hasattr(value, "__class__") and "LoraConfig" in str(type(value))
        ):
            configs[name] = value
    return configs


# === Auto-discover training configs ===
_training_config_dict = {}
_training_config_dict.update(_discover_configs(sft_configs))
_training_config_dict.update(_discover_configs(dpo_configs))
_training_config_dict.update(_discover_configs(vlm_sft_config))

TrainingConfig = Enum("TrainingConfig", _training_config_dict)


def override(self, **overrides):
    config_dict = self.value.copy()
    filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
    config_dict.update(filtered_overrides)

    class _CustomTrainingConfig(Enum):
        CUSTOM = config_dict

    return _CustomTrainingConfig.CUSTOM


TrainingConfig.override = override

# === Auto-discover PEFT configs ===
_peft_config_dict = _discover_configs(peft_configs)

PeftConfig = Enum("PeftConfig", _peft_config_dict)

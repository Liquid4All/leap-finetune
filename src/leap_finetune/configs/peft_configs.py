from peft import LoraConfig, TaskType

GLU_MODULES = ["w1", "w2", "w3"]
MHA_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
CONV_MODULES = ["in_proj", "out_proj"]


LFM2_LORA_DEFAULT_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
)

LFM2_LORA_HIGH_R_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
)

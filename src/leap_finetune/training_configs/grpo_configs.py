"""GRPO base training configs.

Auto-discovered by training_configs/__init__.py:_discover_configs() — any new
DEFAULT_* / MOE_* dict here is immediately available via `extends:` in YAML.

All field names match TRL v1.0 GRPOConfig (verified against
trl/trainer/grpo_config.py at v1.0.0). Keys not recognized by GRPOConfig must
be added to GRPO_EXCLUDED_KEYS / VLM_GRPO_EXCLUDED_KEYS so they're stripped
before construction.
"""

from leap_finetune.utils.constants import GRPO_OUTPUT_PATH

# Keys that exist in our YAML but are NOT GRPOConfig fields. Stripped in
# grpo_run.py / vlm_grpo_run.py before building GRPOConfig(**filtered).
GRPO_EXCLUDED_KEYS = {
    "training_type",
    "wandb_logging",
    "tracker",
    "trackio_space_id",
    "resume_from_checkpoint",
    "leap_run_name_template",
}

# VLM GRPO adds the per-component LR knobs that are consumed by
# LFMVLMGRPOTrainer.create_optimizer() and must not be forwarded to GRPOConfig.
VLM_GRPO_EXCLUDED_KEYS = GRPO_EXCLUDED_KEYS | {
    "max_image_tokens",
    "do_image_splitting",
    "lr_multipliers",
    "vision_encoder_lr_multiplier",
}


########################
#   DEEPSPEED CONFIGS   #
########################


# ZeRO-2 — sufficient for the policy model since GRPO with beta=0 doesn't
# load a separate reference model. Matches DPO/SFT.
DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {"enabled": "auto"},
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    },
}


# ZeRO-0 for MoE — same rationale as MOE_DPO/MOE_SFT (FSDP handles sharding
# in the full fine-tune case; DeepSpeed is just used as the backend).
MOE_DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "bf16": {"enabled": "auto"},
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    },
}


########################
#     GRPO CONFIGS     #
########################


DEFAULT_GRPO = {
    "training_type": "grpo",
    "output_dir": GRPO_OUTPUT_PATH,
    # --- Core training schedule ---
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "steps",
    "save_steps": 200,
    "eval_strategy": "no",
    "bf16": True,
    "gradient_checkpointing": True,
    "ddp_find_unused_parameters": False,
    # --- GRPO algorithmic ---
    "num_generations": 8,
    "max_completion_length": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "beta": 0.0,  # KL off — TRL v1 default; ref model not loaded
    "epsilon": 0.2,
    "loss_type": "dapo",  # TRL v1 default
    "scale_rewards": "group",
    "mask_truncated_completions": True,  # DAPO recommendation
    "log_completions": True,
    "num_completions_to_print": 4,
    # --- vLLM rollouts (colocate by default — single-node friendly) ---
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_gpu_memory_utilization": 0.3,
    "vllm_enable_sleep_mode": True,
    "vllm_importance_sampling_correction": True,  # TRL v1 default; keep
    # --- DeepSpeed ZeRO-2 backend ---
    "deepspeed": DEEPSPEED_CONFIG,
}


DEFAULT_VLM_GRPO = {
    **DEFAULT_GRPO,
    # Override training_type so JobConfig._validate_training_config passes
    # when the YAML sets training_type: vlm_grpo.
    "training_type": "vlm_grpo",
    # VLM-specific — consumed by LFMVLMGRPOTrainer, NOT passed to GRPOConfig
    "max_image_tokens": None,
    "do_image_splitting": True,
    "lr_multipliers": {
        "model.vision_tower": 0.1,
        "model.multi_modal_projector": 1.0,
        "model.language_model": 1.0,
    },
    # VLMs are memory-heavy: smaller batch, more accumulation, smaller groups
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_generations": 4,
    "vllm_gpu_memory_utilization": 0.25,
    "remove_unused_columns": False,  # preserve image inputs
}


MOE_GRPO = {
    **DEFAULT_GRPO,
    "per_device_train_batch_size": 2,
    "num_generations": 4,
    "max_grad_norm": 1.0,
    "bf16": True,
    # Distributed strategy applied automatically in grpo_run.py based on
    # PEFT presence (DeepSpeed for LoRA, FSDP for full fine-tune).
    "deepspeed": MOE_DEEPSPEED_CONFIG,
}

from leap_finetune.utils.constants import SFT_OUTPUT_PATH

########################
#   DEEPSPEED CONFIGS   #
########################


DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    # No "optimizer" block — VLMTrainer.create_optimizer() builds per-component LR
    # param groups that DeepSpeed's FusedAdam would silently discard.
    # torch.optim.AdamW(fused=True) provides the same CUDA-fused kernel.
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
#     SFT CONFIGS      #
########################

VLM_SFT_EXCLUDED_KEYS = {
    "training_type",
    "wandb_logging",
    "tracker",
    "trackio_space_id",
    "max_image_tokens",
    "do_image_splitting",
    "lr_multipliers",
    "vision_encoder_lr_multiplier",
}

# Per-component LR multipliers (applied to base learning_rate).
# Matches liquid-vlm convention: vision encoder trains slower to preserve pretrained features.
# HF model prefixes: model.vision_tower, model.multi_modal_projector, model.language_model
DEFAULT_LR_MULTIPLIERS = {
    "model.vision_tower": 0.1,
    "model.multi_modal_projector": 1.0,
    "model.language_model": 1.0,
}


DEFAULT_VLM_SFT = {
    "training_type": "vlm_sft",
    "max_image_tokens": None,  # None = processor default (256); set int to override
    "do_image_splitting": True,  # split large images into tiles (matches liquid-vlm pretraining)
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 3,  # 1 to 5 generally (post-training goes for 2-3)
    "per_device_train_batch_size": 4,  # adjust based on context length (post-training goes for 1-2 at 32k context length)
    "learning_rate": 5e-5,  # anything from 1e-5 to 5e-5 seems ok. "end_learning_rate" would be 1e-7, not easy to set up with out-of-the-box SFTConfig
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "gradient_checkpointing": True,
    "remove_unused_columns": False,  # preserve pixel_values, spatial_shapes, pixel_attention_mask
    "dataloader_drop_last": True,  # avoid batch size mismatches in DDP
    "lr_multipliers": DEFAULT_LR_MULTIPLIERS,
    "deepspeed": DEEPSPEED_CONFIG,
}

from leap_finetune import DPO_OUTPUT_PATH


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
#     DPO CONFIGS      #
########################


DEFAULT_DPO = {
    "training_type": "dpo",
    "output_dir": DPO_OUTPUT_PATH,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "linear",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSPEED_CONFIG,
    "chat_template": None,
    "chat_template_path": None,
    "manual_sharded_checkpoint_format": "hf",
}


DEFAULT_VLM_DPO = {
    **DEFAULT_DPO,
    "training_type": "vlm_dpo",
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-7,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": {"min_lr_rate": 0.02},
    "warmup_ratio": 0.05,
    "num_train_epochs": 1,
    "max_length": 4096,
    "max_prompt_length": None,
    "max_completion_length": None,
    "precompute_ref_log_probs": False,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "gradient_checkpointing": True,
    "remove_unused_columns": False,
    "do_image_splitting": True,
    "max_image_tokens": 256,
}


########################
#   MOE DPO CONFIGS    #
########################

# Base MoE DPO config - distributed strategy is applied automatically in runner
# based on PEFT presence: DeepSpeed for LoRA, FSDP for full fine-tuning
MOE_DPO = {
    "training_type": "moe_dpo",
    "output_dir": DPO_OUTPUT_PATH,
    "num_train_epochs": 2,  # MoE models typically need fewer epochs
    "per_device_train_batch_size": 2,  # MoE models are larger, use smaller batch size
    "learning_rate": 1e-6,
    "lr_scheduler_type": "linear",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "max_grad_norm": 1.0,
    "bf16": True,
    # Distributed strategy will be set automatically:
    # - With PEFT: uses MOE_DEEPSPEED_CONFIG
    # - Without PEFT: uses FSDP_CONFIG
    "deepspeed": MOE_DEEPSPEED_CONFIG,
    "chat_template": None,
    "chat_template_path": None,
    "manual_sharded_checkpoint_format": "hf",
}

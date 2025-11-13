from leap_finetune.utils.constants import DPO_OUTPUT_PATH


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
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "round_robin_gradients": False,
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": "auto",
    "gradient_accumulation_steps": "auto",
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",  # Uses learning_rate from training config
            "betas": "auto",  # DEFAULT: (0.9, 0.999)
            "eps": "auto",  # DEFAULT: 1e-8
            "weight_decay": "auto",  # DEFAULT: 0.01
        },
    },
    "bf16": {"enabled": "auto"},
    "activation_checkpointing": {
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    },
    # Ray Train + TRL compatibility settings
    "wall_clock_breakdown": False,
    "steps_per_print": 10,
}


########################
#     DPO CONFIGS      #
########################


DEFAULT_DPO_CONFIG = {
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
    "load_best_model_at_end": True,
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSPEED_CONFIG,
}


########################
#   MOE DPO CONFIGS    #
########################

MOE_DPO_CONFIG = {
    "training_type": "dpo",
    "output_dir": DPO_OUTPUT_PATH,
    "num_train_epochs": 1,  # MoE models typically need fewer epochs
    "per_device_train_batch_size": 1,  # MoE models are larger, use smaller batch size
    "learning_rate": 1e-6,
    "lr_scheduler_type": "linear",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    # DeepSpeed disabled due to compatibility issues with Ray Train + TRL + MoE
    # TRL's automatic DeepSpeed initialization conflicts with Ray Train's distributed setup
    # Training works without DeepSpeed using standard DDP
    # "deepspeed": MOE_DEEPSPEED_CONFIG,  # Uncomment to try DeepSpeed (may hang)
}

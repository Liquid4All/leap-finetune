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
#     SFT CONFIGS      #
########################


DEFAULT_SFT_CONFIG = {
    "training_type": "sft",
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 3,  # 1 to 5 generally (post-training goes for 2-3)
    "per_device_train_batch_size": 16,  # adjust based on context length (post-training goes for 1-2 at 32k context length)
    "learning_rate": 5e-5,  # anything from 1e-5 to 5e-5 seems ok. "end_learning_rate" would be 1e-7, not easy to set up with out-of-the-box SFTConfig
    "lr_scheduler_type": "linear",
    "warmup_steps": 100,
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSPEED_CONFIG,
}


########################
#   MOE SFT CONFIGS    #
########################

MOE_SFT_CONFIG = {
    "training_type": "sft",
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 1,  # MoE models typically need fewer epochs
    "per_device_train_batch_size": 1,  # MoE models are larger, use smaller batch size
    "per_device_eval_batch_size": 1,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "linear",
    "warmup_steps": 100,
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "deepspeed": MOE_DEEPSPEED_CONFIG,
}

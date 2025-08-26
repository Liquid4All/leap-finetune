from leap_finetune.utils.constants import SFT_OUTPUT_PATH

########################
#   DEEPSEED CONFIGS   #
########################


DEEPSEED_CONFIG = {
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


########################
#     SFT CONFIGS      #
########################


DEFAULT_VLM_SFT_CONFIG = {
    "training_type": "vlm_sft",
    "output_dir": SFT_OUTPUT_PATH,
    "num_train_epochs": 3,  # 1 to 5 generally (post-training goes for 2-3)
    "per_device_train_batch_size": 4,  # adjust based on context length (post-training goes for 1-2 at 32k context length)
    "learning_rate": 5e-5,  # anything from 1e-5 to 5e-5 seems ok. "end_learning_rate" would be 1e-7, not easy to set up with out-of-the-box SFTConfig
    "lr_scheduler_type": "linear",
    "warmup_steps": 100,
    "warmup_ratio": 0.2,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "gradient_checkpointing": True,
    "dataset_kwargs": {"skip_prepare_dataset": True},
    "deepspeed": DEEPSEED_CONFIG,
}

from leap_finetune.utils.constants import DPO_OUTPUT_PATH


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
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "ddp_find_unused_parameters": False,
    "deepspeed": DEEPSEED_CONFIG,
}

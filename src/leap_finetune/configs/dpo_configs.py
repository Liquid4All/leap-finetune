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
#     FSDP CONFIGS     #
########################

FSDP_CONFIG = {
    "fsdp": "shard_grad_op auto_wrap",
    "fsdp_config": {
        "transformer_layer_cls_to_wrap": "transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeDecoderLayer",
        "backward_prefetch": "backward_pre",
        "sync_module_states": True,
        "use_orig_params": False,
    },
}


########################
#   MOE DPO CONFIGS    #
########################

MOE_DPO_CONFIG = {
    "training_type": "dpo",
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
    "load_best_model_at_end": True,
    "deepspeed": MOE_DEEPSPEED_CONFIG,
}


MOE_DPO_NO_LORA_CONFIG = {
    "training_type": "dpo",
    "output_dir": DPO_OUTPUT_PATH,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "linear",
    "beta": 0.1,
    "loss_type": "sigmoid",
    "logging_steps": 10,
    "logging_first_step": True,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "max_grad_norm": 1.0,
    "bf16": True,
    **FSDP_CONFIG,
}

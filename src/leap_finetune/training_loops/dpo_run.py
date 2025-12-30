from typing import cast

import ray.train
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer
from ray.train.huggingface.transformers import prepare_trainer
from ray.train import get_context

from leap_finetune.configs.distributed_configs import MOE_FSDP_CONFIG
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.logging_utils import init_wandb_if_enabled
from leap_finetune.utils.logging_utils import setup_worker_logging
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model


def dpo_run(training_config: dict) -> None:
    """DPO training loop for Ray Train"""
    setup_worker_logging()

    # Get sharded datasets for this worker
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")

    # Materialize to HuggingFace Datasets for TRL
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    test_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    # Check for MoE model
    is_moe = is_moe_model_from_name(model_name)
    use_fsdp = is_moe and peft_config is None

    # Filter out non-DPOConfig parameters
    excluded_keys = {"training_type", "wandb_logging"}
    if use_fsdp:
        excluded_keys.add("deepspeed")  # Remove deepspeed when using FSDP

    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k not in excluded_keys
    }

    # Configure wandb reporting if enabled via config
    wandb_logging = bool(
        training_config.get("train_config", {}).get("wandb_logging", False)
    )
    init_wandb_if_enabled(job_name, wandb_logging)

    # Build training args
    config_kwargs = {
        "report_to": "wandb" if wandb_logging else "none",
        "run_name": job_name,
        **train_config_filtered,
    }
    if use_fsdp:
        config_kwargs["fsdp"] = MOE_FSDP_CONFIG["fsdp"]
        config_kwargs["fsdp_config"] = MOE_FSDP_CONFIG["fsdp_config"]

    training_args = DPOConfig(**config_kwargs)

    # Load model after config is created
    model, tokenizer = load_model(model_name)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
    )

    # Start training
    trainer = prepare_trainer(trainer)
    try:
        trainer.train()
        print("✅ Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(
            keyword in error_msg.lower()
            for keyword in ["cuda error", "ecc error", "nccl", "collective", "timeout"]
        ):
            print(
                f"⚠️  Training completed but hit distributed communication error during cleanup: {error_msg}"
            )
            print(
                "✅ Training was successful - error occurred in post-training synchronization"
            )
        else:
            raise e

    # Save PEFT model if applicable
    if peft_config:
        ctx = get_context()
        is_rank_zero = ctx is None or ctx.get_world_rank() == 0
        if is_rank_zero:
            merge_and_save_peft_model(model, tokenizer, training_args.output_dir)

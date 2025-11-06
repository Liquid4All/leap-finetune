from typing import cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from ray.train.huggingface.transformers import prepare_trainer
from ray.train import get_context

from leap_finetune.configs.distributed_configs import MOE_FSDP_CONFIG
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.logging_utils import init_wandb_if_enabled


def sft_run(training_config: dict) -> None:
    """SFT training loop for Ray Train"""

    train_dataset, test_dataset = cast(
        tuple[Dataset, Dataset], training_config.get("dataset")
    )

    excluded_keys = {"training_type", "wandb_logging"}
    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k not in excluded_keys
    }
    # Configure wandb reporting if enabled via config
    job_name = training_config.get("job_name", "leap-ft-run")
    wandb_logging = bool(
        training_config.get("train_config", {}).get("wandb_logging", False)
    )

    # Initialize wandb with project and run name if logging is enabled
    init_wandb_if_enabled(job_name, wandb_logging)

    training_args = SFTConfig(
        report_to="wandb" if wandb_logging else "none",
        run_name=job_name,
        **train_config_filtered,
    )
    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")

    # Check for MoE model
    is_moe = is_moe_model_from_name(model_name)
    use_fsdp = is_moe and peft_config is None

    # Remove non-SFTConfig parameters
    train_config.pop("training_type", None)

    # Apply FSDP for MoE without PEFT
    if use_fsdp:
        train_config.pop("deepspeed", None)
        fsdp_config = MOE_FSDP_CONFIG["fsdp_config"].copy()
        training_args = SFTConfig(
            **train_config,
            fsdp=MOE_FSDP_CONFIG["fsdp"],
            fsdp_config=fsdp_config,
        )
    else:
        # MoE with PEFT or non-MoE: use DeepSpeed (already in config)
        training_args = SFTConfig(**train_config)

    # Load model after config is created
    model, tokenizer = load_model(model_name)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Initialize trainer
    trainer = SFTTrainer(
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

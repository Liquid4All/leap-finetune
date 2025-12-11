from typing import cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer
from ray.train.huggingface.transformers import prepare_trainer

from leap_finetune.configs.distributed_configs import MOE_FSDP_CONFIG
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model


def dpo_run(training_config: dict) -> None:
    """DPO training loop for Ray Train"""

    train_dataset, test_dataset = cast(
        tuple[Dataset, Dataset], training_config.get("dataset")
    )

    train_config = training_config.get("train_config")
    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")

    # Check for MoE model
    is_moe = is_moe_model_from_name(model_name)
    use_fsdp = is_moe and peft_config is None

    # Remove non-DPOConfig parameters
    train_config.pop("training_type", None)

    # Apply FSDP for MoE without PEFT
    if use_fsdp:
        train_config.pop("deepspeed", None)
        fsdp_config = MOE_FSDP_CONFIG["fsdp_config"].copy()
        training_args = DPOConfig(
            **train_config,
            fsdp=MOE_FSDP_CONFIG["fsdp"],
            fsdp_config=fsdp_config,
        )
    else:
        # MoE with PEFT or non-MoE: use DeepSpeed (already in config)
        training_args = DPOConfig(**train_config)

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
    trainer.train()

    # Save PEFT model if applicable
    if peft_config:
        merge_and_save_peft_model(model, tokenizer, training_args.output_dir)

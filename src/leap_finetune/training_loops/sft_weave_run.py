"""
Enhanced SFT training loop with W&B and Weave integration.

This module provides an enhanced SFT training loop that includes:
- Periodic Weave evaluation during training
- W&B experiment tracking
- Full observability with Weave tracing
"""

import os
from pathlib import Path
from typing import cast, Optional

import wandb
import weave
from datasets import Dataset

import structlog_sentry_logger

LOGGER = structlog_sentry_logger.get_logger()


def sft_weave_run(
    training_config: dict,
    weave_model: Optional[any] = None,
    eval_interval: str = "steps",
    eval_steps: Optional[int] = None,
    max_eval_samples: int = 50,
    scorers: Optional[list] = None,
    use_response_only_training: bool = True,
) -> None:
    """
    Enhanced SFT training loop with Weave evaluation for Ray Train.

    This training loop extends the standard SFT training with:
    - Periodic Weave evaluation during training
    - W&B experiment tracking
    - Weave tracing for full observability

    Args:
        training_config: Configuration dict with train_config, peft_config, dataset, model_name
        weave_model: Optional Weave Model wrapper (will be created if not provided)
        eval_interval: Evaluation interval ("epoch" or "steps")
        eval_steps: Step interval for step-based evaluation (auto-calculated if None)
        max_eval_samples: Maximum number of samples to use for evaluation
        scorers: List of scorer functions (uses DEFAULT_SCORERS if None)
        use_response_only_training: Whether to train only on response tokens
    """

    from leap_finetune.utils.logging_utils import setup_training_environment

    setup_training_environment()

    train_dataset, test_dataset = cast(
        tuple[Dataset, Dataset], training_config.get("dataset")
    )
    LOGGER.debug("Training config", **training_config.get("train_config"))

    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k != "training_type"
    }
    # Enable bf16 training to match model dtype and prevent dtype mismatch errors
    # This is necessary when the model is loaded with bfloat16 dtype and when using DeepSpeed
    train_config_filtered["bf16"] = True

    # Save only model state, not optimizer/scheduler to avoid PEFT issues
    # Fixes: `AttributeError: 'Lfm2ForCausalLM' object has no attribute 'save_checkpoint'`
    train_config_filtered["save_only_model"] = True
    # train_config_filtered["optim"] = "adamw_torch"

    from trl import SFTConfig

    LOGGER.debug("Training config (filtered)", **train_config_filtered)
    peft_config = training_config.get("peft_config")
    # Workaround for DeepSpeed + PEFT checkpoint saving issues
    # DeepSpeed doesn't support load_best_model_at_end with save_only_model
    # Fixes: `ValueError: DeepSpeed can't be used with save_only_model along with `load_best_model_at_end`.`
    if train_config_filtered.get("deepspeed") and peft_config:
        train_config_filtered["load_best_model_at_end"] = False
        LOGGER.info(
            "Disabled load_best_model_at_end due to DeepSpeed + PEFT compatibility"
        )
    training_args = SFTConfig(**train_config_filtered)

    wandb.login()

    WANDB_ENTITY = os.environ["WANDB_ENTITY"].replace('"', "")
    WANDB_PROJECT = os.environ["WANDB_PROJECT"].replace('"', "")
    weave.init(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    job_name = training_config.get("job_name", "leap-ft-run")

    _ = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=train_config_filtered,
        name=job_name,
        tags=["periodic-evaluation", "weave-integration"],
    )
    # Create or use provided Weave model wrapper
    if weave_model is None:
        LOGGER.info("Creating UnslothLoRALFM2 Weave model wrapper")
        # Create a Weave model wrapper for evaluation
        # Note: This is a simplified version - users may want to customize this
        model_name = training_config.get("model_name")

        # Check if model_name is a local path
        model_path = Path(model_name)
        if model_path.exists() and model_path.is_dir():
            # Load from local path (for checkpoints)
            model_id = model_name
            print(f"Loading model from local path: {model_id}")
        else:
            # Load from Hugging Face
            model_id = f"LiquidAI/{model_name}"
            print(f"Loading model from Hub: {model_id}")
        from leap_finetune.models import UnslothLoRALFM2

        weave_model = UnslothLoRALFM2(
            base_model=model_id,
            revision=None,
            is_training=True,  # Use inference mode for evaluation
            peft_config=peft_config,
            cm_temperature=1.0,
            max_seq_length=2048,
            load_in_4bit="4bit" in model_name,
            inference_batch_size=2048,
            dtype="torch.bfloat16",
            device="cuda",
        )

    # Use default scorers if none provided
    if scorers is None:
        from leap_finetune.utils.weave_trainer_utils import (
            DEFAULT_SCORERS,
        )

        scorers = DEFAULT_SCORERS

    # Calculate eval_steps if not provided
    if eval_steps is None and eval_interval == "steps":
        total_steps = (
            len(train_dataset)
            // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
            )
            * max(1, training_args.num_train_epochs)
        )
        if training_args.max_steps > 0:
            total_steps = min(total_steps, training_args.max_steps)
        eval_steps = max(10, total_steps // 5)  # Aim for ~5 evaluations
        LOGGER.info(
            f"Auto-calculated eval_steps: {eval_steps} (total_steps: {total_steps})"
        )

    # Preprocess evaluation dataset for Weave
    # Note: This requires the base model for comparison - users should prepare this beforehand
    # For now, we'll use the eval dataset as-is and expect it to have the required fields
    LOGGER.info(f"Setting up evaluation dataset (size: {len(test_dataset)})")

    from leap_finetune.utils.weave_trainer_utils import (
        get_trainer_with_evaluation_callback,
    )

    # Create trainer with evaluation callback
    trainer, evaluation_callback = get_trainer_with_evaluation_callback(
        model=weave_model.model,
        tokenizer=weave_model.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        weave_model=weave_model,
        training_args=training_args,
        eval_interval=eval_interval,
        eval_steps=eval_steps or training_args.logging_steps,
        max_eval_samples=max_eval_samples,
        scorers=scorers,
        use_response_only_training=use_response_only_training,
    )

    # Prepare trainer for Ray
    from ray.train.huggingface.transformers import prepare_trainer

    trainer = prepare_trainer(trainer)

    LOGGER.info(
        "🚀 Starting enhanced SFT training with Weave evaluation",
        eval_interval=eval_interval,
        eval_steps=eval_steps if eval_interval == "steps" else "N/A",
        num_scorers=len(scorers),
        max_eval_samples=max_eval_samples,
    )

    # Start training
    LOGGER.debug(
        "Trainer args",
        args=trainer.args,
        hf_deepspeed_config=trainer.args.hf_deepspeed_config.config,
    )
    LOGGER.debug(
        "trainer.args.deepspeed_plugin.dschf",
        dschf=trainer.args.deepspeed_plugin.dschf.config,
    )

    trainer.train()

    LOGGER.info(
        "✅ Training complete!"
        f"\nnum_evaluations={len(evaluation_callback.evaluation_results)}",
    )

    # Save PEFT model if applicable
    if peft_config:
        from leap_finetune.utils.peft import merge_and_save_peft_model

        merge_and_save_peft_model(
            weave_model.model, weave_model.tokenizer, training_args.output_dir
        )
        weave.publish(weave_model)

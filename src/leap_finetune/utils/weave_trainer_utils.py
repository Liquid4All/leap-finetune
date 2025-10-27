"""
Enhanced trainer utilities for W&B and Weave integration.

This module provides functions to create trainers with Weave evaluation callbacks
for periodic model evaluation during training.
"""

from typing import List, Optional, cast

from leap_finetune.callbacks import (
    conversation_scorer,
    conversation_improvement_scorer,
    response_length_scorer,
    coherence_scorer,
    diversity_scorer,
)


import structlog_sentry_logger

LOGGER = structlog_sentry_logger.get_logger()

# Default scorers for evaluation
DEFAULT_SCORERS = [
    conversation_scorer,
    conversation_improvement_scorer,
    response_length_scorer,
    coherence_scorer,
    diversity_scorer,
]


def get_trainer_with_evaluation_callback(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    weave_model,
    training_args,
    eval_interval: str = "epoch",
    eval_steps: int = 100,
    max_eval_samples: int = 20,
    scorers: Optional[List] = None,
    use_response_only_training: bool = True,
    instruction_part: str = "<|im_start|>user\n",
    response_part: str = "<|im_start|>assistant\n",
) -> tuple:
    """
    Create SFTTrainer with Weave evaluation callback.

    This function creates a trainer configured for periodic evaluation during training
    with full W&B and Weave integration for observability.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        weave_model: Weave Model wrapper instance
        training_args: SFTConfig with training arguments
        eval_interval: Evaluation interval ("epoch" or "steps")
        eval_steps: Step interval for step-based evaluation
        max_eval_samples: Maximum number of samples to use for evaluation
        scorers: List of scorer functions (uses DEFAULT_SCORERS if None)
        use_response_only_training: Whether to train only on response tokens
        instruction_part: Instruction part marker for response-only training
        response_part: Response part marker for response-only training

    Returns:
        Tuple of (trainer, evaluation_callback)
    """

    # Use default scorers if none provided
    if scorers is None:
        scorers = DEFAULT_SCORERS

    # Create base trainer
    from unsloth.chat_templates import standardize_data_formats
    from trl import SFTTrainer
    from transformers import PreTrainedTokenizerBase

    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}

    # print(train_dataset[0])
    train_dataset = standardize_data_formats(train_dataset)
    #     print(train_dataset[0])
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    #     print(train_dataset[0])

    #     print(eval_dataset[0])
    eval_dataset = standardize_data_formats(eval_dataset)
    #     print(eval_dataset[0])
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    #     print(eval_dataset[0])

    LOGGER.debug("Training args", SFTConfig=training_args)

    # Log DeepSpeed configuration for debugging/provenance if available
    if training_args.hf_deepspeed_config is not None:
        LOGGER.info(
            "DeepSpeed configuration (pre)",
            deepspeed_config=training_args.hf_deepspeed_config.config,
        )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
        formatting_func=formatting_prompts_func,
    )

    LOGGER.debug("Trainer args (after init)", args=trainer.args)
    # Log DeepSpeed configuration for debugging/provenance if available
    if training_args.hf_deepspeed_config is not None:
        LOGGER.info(
            "DeepSpeed configuration",
            deepspeed_config=training_args.hf_deepspeed_config.config,
        )

    optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(
        training_args, model
    )
    LOGGER.info(
        "trainer opt args",
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
    )
    # Apply train_on_responses_only if requested
    if use_response_only_training:
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
        LOGGER.debug("Trainer args (after response only training)", args=trainer.args)

    # Add Weave evaluation callback
    from leap_finetune.callbacks import WeaveEvaluationCallback

    evaluation_callback = WeaveEvaluationCallback(
        model_wrapper=weave_model,
        eval_dataset=eval_dataset,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        scorers=scorers,
        max_eval_samples=max_eval_samples,
    )

    trainer.add_callback(evaluation_callback)

    LOGGER.info(
        "✅ Trainer with Weave evaluation callback created",
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        num_scorers=len(scorers),
        max_eval_samples=max_eval_samples,
        use_response_only_training=use_response_only_training,
    )
    LOGGER.debug("Trainer args (after callback added)", args=trainer.args)

    return trainer, evaluation_callback


def create_sft_config_with_weave_eval(
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 5,
    num_train_epochs: int = 1,
    max_steps: int = -1,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    optim: str = "adamw_8bit",
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "linear",
    seed: int = 3407,
    output_dir: str = "./results",
    eval_strategy: str = "steps",
    eval_steps: Optional[int] = None,
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    load_best_model_at_end: bool = False,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    dataset_text_field: str = "text",
    report_to: str = "wandb",
    logging_first_step: bool = True,
    **kwargs,
):
    """
    Create SFTConfig optimized for Weave evaluation integration.

    This is a convenience function to create training configurations that work
    well with periodic Weave evaluation.

    Args:
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        warmup_steps: Number of warmup steps
        num_train_epochs: Number of training epochs
        max_steps: Maximum training steps (-1 for no limit)
        learning_rate: Learning rate
        logging_steps: Logging frequency
        optim: Optimizer name
        weight_decay: Weight decay
        lr_scheduler_type: Learning rate scheduler type
        seed: Random seed
        output_dir: Output directory
        eval_strategy: Evaluation strategy ("steps" or "epoch")
        eval_steps: Evaluation step interval (uses logging_steps if None)
        save_strategy: Save strategy
        save_total_limit: Maximum number of checkpoints to keep
        load_best_model_at_end: Whether to load best model at end
        metric_for_best_model: Metric to use for best model
        greater_is_better: Whether higher metric is better
        dataset_text_field: Name of text field in dataset
        report_to: Where to report metrics
        logging_first_step: Whether to log first step
        **kwargs: Additional arguments passed to SFTConfig

    Returns:
        SFTConfig instance
    """

    if eval_steps is None:
        eval_steps = logging_steps
    from trl import SFTConfig

    return SFTConfig(
        dataset_text_field=dataset_text_field,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        logging_first_step=logging_first_step,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        report_to=report_to,
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        **kwargs,
    )

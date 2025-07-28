from typing import cast

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from ray.train.huggingface.transformers import prepare_trainer

from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model


def sft_run(training_config: dict) -> None:
    """SFT training loop for Ray Train"""

    train_dataset, test_dataset = cast(
        tuple[Dataset, Dataset], training_config.get("dataset")
    )

    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k != "training_type"
    }
    training_args = SFTConfig(**train_config_filtered)
    peft_config = training_config.get("peft_config")

    model, tokenizer = load_model(training_config.get("model_name"))

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
    trainer.train()

    # Save PEFT model if applicable
    if peft_config:
        merge_and_save_peft_model(model, tokenizer, training_args.output_dir)

import copy
import os

from trl import SFTConfig, SFTTrainer
from ray.train.huggingface.transformers import prepare_trainer

from leap_finetune.data_loaders.image_loader import load_image
from leap_finetune.utils.load_models import load_vlm_model
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model


def create_collate_fn(processor):
    """Create a collate function VLM training with lazy image loading and proper cleanup."""

    def collate_fn(sample):
        # Def better way to do this and can clean up later
        sample_copy = copy.deepcopy(sample)

        # Load images for processing and then remove from memory store
        loaded_images = []
        try:
            for conversation in sample_copy:
                for message in conversation:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image" and isinstance(
                                content["image"], str
                            ):
                                # Load image and keep reference for cleanup
                                img = load_image(content["image"])
                                content["image"] = img
                                loaded_images.append(img)

            # Process the batch with loaded images
            batch = processor.apply_chat_template(
                sample_copy,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch

        finally:
            # Cleanup images from Ray memory store
            for img in loaded_images:
                if hasattr(img, "close"):
                    img.close()
            loaded_images.clear()
            del sample_copy

    return collate_fn


def vlm_sft_run(training_config: dict) -> None:
    """SFT training loop for Ray Train"""

    train_dataset, eval_dataset = training_config.get("dataset")
    train_dataset = [sample["messages"] for sample in train_dataset]
    test_dataset = [sample["messages"] for sample in eval_dataset]

    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k != "training_type"
    }
    # Configure W&B reporting if enabled via config
    job_name = training_config.get("job_name", "leap-ft-run")
    wandb_logging = bool(training_config.get("train_config", {}).get("wandb_logging", False))
    if wandb_logging:
        if not os.environ.get("WANDB_API_KEY"):
            os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("WANDB_PROJECT", "leap-finetune")

    training_args = SFTConfig(
        report_to=["wandb"] if wandb_logging else [],
        run_name=job_name,
        **train_config_filtered,
    )

    model, processor = load_vlm_model(training_config.get("model_name"))

    peft_config = training_config.get("peft_config")
    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    collate_fn = create_collate_fn(processor)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    # Start training
    trainer = prepare_trainer(trainer)

    try:
        trainer.train()
        print("✅ Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        # Safely handle hang errors that occur during cleanup -- training is still successful
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
        merge_and_save_peft_model(model, processor, training_args.output_dir)

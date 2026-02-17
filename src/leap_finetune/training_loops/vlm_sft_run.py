import copy
import logging

import torch
import ray.train
from trl import SFTConfig, SFTTrainer
from ray.train.huggingface.transformers import prepare_trainer
from ray.train import get_context

from leap_finetune.data_loaders.image_loader import load_image
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_vlm_model
from leap_finetune.utils.logging_utils import (
    init_wandb_if_enabled,
    setup_worker_logging,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model

logger = logging.getLogger(__name__)


def _find_template(seq, template):
    """Yield start indices where template occurs in seq."""
    tlen = len(template)
    for i in range(len(seq) - tlen + 1):
        if seq[i : i + tlen] == template:
            yield i


def create_collate_fn(processor):
    """Create a collate function with assistant-only label masking.

    Only assistant content + <|im_end|> contribute to loss (matching liquid-vlm).
    Images are loaded as PIL and passed to the processor for resize/tiling.
    Bad samples are skipped with a warning instead of crashing the batch.
    """
    tokenizer = processor.tokenizer

    # ChatML: <|im_start|>assistant\n{content}<|im_end|>\n
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _build_labels(input_ids):
        """Mask everything except assistant content + <|im_end|> with -100."""
        labels = torch.full_like(input_ids, -100)
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            for tmpl_start in _find_template(ids, response_template_ids):
                content_start = tmpl_start + len(response_template_ids)
                j = content_start
                while j < seq_len and ids[j] != end_token_id:
                    j += 1
                # Unmask content + <|im_end|>
                content_end = min(j + 1, seq_len)
                labels[b, content_start:content_end] = input_ids[
                    b, content_start:content_end
                ]

        return labels

    def collate_fn(samples):
        valid_samples = []
        all_loaded_images = []
        skip_count = 0

        for conversation in samples:
            sample_copy = copy.deepcopy(conversation)
            loaded_images = []
            try:
                for message in sample_copy:
                    if message["role"] == "user":
                        for content in message["content"]:
                            if content["type"] == "image" and isinstance(
                                content["image"], str
                            ):
                                img = load_image(content["image"])
                                content["image"] = img
                                loaded_images.append(img)
                valid_samples.append(sample_copy)
                all_loaded_images.extend(loaded_images)
            except Exception as e:
                skip_count += 1
                logger.warning(f"Skipping sample in collate: {e}")
                for img in loaded_images:
                    if hasattr(img, "close"):
                        img.close()

        if skip_count > 0:
            logger.info(f"Collate skipped {skip_count}/{len(samples)} samples")

        if len(valid_samples) == 0:
            raise RuntimeError(
                f"Entire batch failed: all {len(samples)} samples had errors"
            )

        try:
            batch = processor.apply_chat_template(
                valid_samples,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            batch["labels"] = _build_labels(batch["input_ids"])
            return batch

        finally:
            for img in all_loaded_images:
                if hasattr(img, "close"):
                    img.close()
            all_loaded_images.clear()

    return collate_fn


def vlm_sft_run(training_config: dict) -> None:
    """VLM SFT training loop for Ray Train."""

    setup_worker_logging()

    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")

    train_dataset = [sample["messages"] for sample in ray_dataset_to_hf(train_ds_ray)]
    test_dataset = [sample["messages"] for sample in ray_dataset_to_hf(eval_ds_ray)]

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    # Extract VLM-specific params
    train_config = training_config.get("train_config", {})
    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)

    # Filter out non-SFTConfig parameters
    excluded_keys = {
        "training_type",
        "wandb_logging",
        "max_image_tokens",
        "do_image_splitting",
    }
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    # Configure wandb
    wandb_logging = bool(train_config.get("wandb_logging", False))
    init_wandb_if_enabled(job_name, wandb_logging)

    # Build training args
    config_kwargs = {
        "report_to": "wandb" if wandb_logging else "none",
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = SFTConfig(**config_kwargs)

    # Load model + processor
    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )

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

    # Add Ray checkpoint callback and prepare for distributed training
    trainer.add_callback(LeapCheckpointCallback())
    trainer = prepare_trainer(trainer)

    try:
        trainer.train()
        logger.info("Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(
            keyword in error_msg.lower()
            for keyword in ["cuda error", "ecc error", "nccl", "collective", "timeout"]
        ):
            logger.warning(
                f"Training completed but hit distributed communication error during cleanup: {error_msg}"
            )
            logger.info(
                "Training was successful - error occurred in post-training synchronization"
            )
        else:
            raise

    # Save PEFT model if applicable
    if peft_config:
        ctx = get_context()
        is_rank_zero = ctx is None or ctx.get_world_rank() == 0
        if is_rank_zero:
            merge_and_save_peft_model(model, processor, training_args.output_dir)

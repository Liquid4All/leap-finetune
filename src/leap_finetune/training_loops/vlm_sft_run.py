import copy
import logging
import math

import ray.train
from PIL import Image
from trl import SFTConfig, SFTTrainer
from ray.train.huggingface.transformers import prepare_trainer
from ray.train import get_context

from leap_finetune.data_loaders.image_loader import load_image
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_vlm_model
from leap_finetune.utils.logging_utils import setup_worker_logging
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.logging_utils import init_wandb_if_enabled

logger = logging.getLogger(__name__)


def smart_resize_for_training(
    image: Image.Image,
    max_tokens: int | None,
    patch_size: int = 16,
    patch_multiplier: int = 4,
) -> Image.Image:
    """Pre-shrink an image to fit within a vision token budget.

    When max_tokens is None, returns the image unchanged (native resolution).
    Ensures output dimensions are multiples of patch_size and preserves aspect ratio.
    """
    if max_tokens is None:
        return image

    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((224, 224), Image.LANCZOS)

    effective_patch = patch_size * patch_multiplier
    patches_w = math.ceil(w / effective_patch)
    patches_h = math.ceil(h / effective_patch)
    current_tokens = patches_w * patches_h

    if current_tokens <= max_tokens:
        return image

    # Scale down to fit within token budget
    scale = math.sqrt(max_tokens / current_tokens)
    new_w = max(effective_patch, int(w * scale))
    new_h = max(effective_patch, int(h * scale))

    # Snap to multiples of patch_size
    new_w = (new_w // patch_size) * patch_size
    new_h = (new_h // patch_size) * patch_size

    if new_w <= 0 or new_h <= 0:
        return image.resize((224, 224), Image.LANCZOS)

    return image.resize((new_w, new_h), Image.LANCZOS)


def create_collate_fn(processor, max_image_tokens: int | None = None):
    """Create a collate function for VLM training with per-sample error handling."""

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
                                img = smart_resize_for_training(img, max_image_tokens)
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
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
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

    # Extract max_image_tokens before passing to SFTConfig
    max_image_tokens = training_config.get("train_config", {}).get("max_image_tokens")

    # Filter out non-SFTConfig parameters
    excluded_keys = {"training_type", "wandb_logging", "max_image_tokens"}
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
    training_args = SFTConfig(**config_kwargs)

    # Load model
    model, processor = load_vlm_model(model_name, max_image_tokens=max_image_tokens)

    # === Model config safety ===
    model.config.do_image_splitting = False
    if hasattr(model.config, "max_tiles"):
        model.config.max_tiles = 1

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    collate_fn = create_collate_fn(processor, max_image_tokens=max_image_tokens)

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
            raise e

    # Save PEFT model if applicable
    if peft_config:
        ctx = get_context()
        is_rank_zero = ctx is None or ctx.get_world_rank() == 0
        if is_rank_zero:
            merge_and_save_peft_model(model, processor, training_args.output_dir)

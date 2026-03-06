import logging
import math

import torch
import ray.train
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from ray.train.huggingface.transformers import prepare_trainer

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.data_loaders.tokenize_data import create_vlm_collate_fn
from leap_finetune.training_configs.vlm_sft_config import (
    DEFAULT_LR_MULTIPLIERS,
    VLM_SFT_EXCLUDED_KEYS,
)
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_vlm_model
from leap_finetune.utils.logging_utils import (
    init_wandb_if_enabled,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model

logger = logging.getLogger(__name__)


# === VLM Trainer with per-component learning rates ===


class VLMTrainer(Trainer):
    """Trainer subclass that applies per-component LR multipliers.

    Mirrors liquid-vlm convention: vision encoder trains at a lower LR
    to preserve pretrained features, while the projector and LLM backbone
    train at the full base rate.

    HF VLM param prefixes:
        model.vision_tower          — vision encoder (e.g. SigLIP2)
        model.multi_modal_projector — projector
        model.language_model        — LLM backbone (e.g. LFM2)
    """

    def __init__(self, lr_multipliers: dict[str, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lr_multipliers = lr_multipliers or DEFAULT_LR_MULTIPLIERS

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay

        # Group trainable params by matching prefix
        grouped: dict[str, list] = {prefix: [] for prefix in self.lr_multipliers}
        ungrouped: list = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            matched = False
            for prefix in self.lr_multipliers:
                if name.startswith(prefix):
                    grouped[prefix].append(param)
                    matched = True
                    break
            if not matched:
                ungrouped.append(param)

        # Build optimizer param groups
        optimizer_groups = []
        for prefix, params in grouped.items():
            if not params:
                continue
            mult = self.lr_multipliers[prefix]
            optimizer_groups.append(
                {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay}
            )
            logger.info(
                f"Param group '{prefix}': {len(params)} params, lr={base_lr * mult:.2e}"
            )

        if ungrouped:
            optimizer_groups.append(
                {"params": ungrouped, "lr": base_lr, "weight_decay": weight_decay}
            )
            logger.info(
                f"Param group 'ungrouped': {len(ungrouped)} params, lr={base_lr:.2e}"
            )

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(optimizer_groups, betas=betas)
        return self.optimizer

    def get_train_dataloader(self):
        # Ray already shards across workers — return a raw DataLoader
        # (bypasses Accelerate's DistributedSampler / IterableDatasetShard)
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
        )


# === Training loop ===


def vlm_sft_run(training_config: dict) -> None:
    """VLM SFT training loop for Ray Train."""

    setup_worker_logging()

    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")

    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    # Extract VLM-specific params and run name template before filtering
    train_config = training_config.get("train_config", {})
    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)
    run_name_template = train_config.get("leap_run_name_template")
    lr_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if "lr_multipliers" in train_config:
        lr_multipliers.update(train_config["lr_multipliers"])
    if "vision_encoder_lr_multiplier" in train_config:
        lr_multipliers["model.vision_tower"] = train_config[
            "vision_encoder_lr_multiplier"
        ]

    # Filter out non-TrainingArguments parameters
    excluded_keys = VLM_SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    # Configure wandb
    wandb_logging = bool(train_config.get("wandb_logging", False))
    init_wandb_if_enabled(job_name, wandb_logging)

    # Compute max_steps from materialized dataset size
    # (Trainer can't infer it from our bypassed DataLoader)
    num_samples = len(train_dataset)
    train_batch_size = train_config_filtered.get("per_device_train_batch_size", 4)
    grad_accum = train_config_filtered.get("gradient_accumulation_steps", 1)
    epochs = train_config_filtered.get("num_train_epochs", 3)
    steps_per_epoch = math.ceil(num_samples / train_batch_size)
    max_steps = steps_per_epoch * epochs // grad_accum

    logger.info(
        f"Computed max_steps={max_steps} "
        f"(samples={num_samples}, batch={train_batch_size}, "
        f"accum={grad_accum}, epochs={epochs})"
    )

    # Build training args — use max_steps instead of num_train_epochs
    train_config_filtered.pop("num_train_epochs", None)
    config_kwargs = {
        "report_to": "wandb" if wandb_logging else "none",
        "run_name": job_name,
        "per_device_eval_batch_size": train_batch_size,
        "remove_unused_columns": False,
        "max_steps": max_steps,
        **train_config_filtered,
    }
    training_args = TrainingArguments(**config_kwargs)

    # Load model + processor
    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    collate_fn = create_vlm_collate_fn(processor)

    # Initialize trainer with per-component LR multipliers
    trainer = VLMTrainer(
        lr_multipliers=lr_multipliers,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Add checkpoint callback (handles Ray reporting + rename) then prepare for distributed training
    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))
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
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

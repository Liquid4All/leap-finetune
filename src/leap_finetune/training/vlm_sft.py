import logging
import math

import torch
from ray.train.huggingface.transformers import prepare_trainer
from transformers import Trainer, TrainingArguments

from leap_finetune.data_loading.tokenize_data import create_vlm_collate_fn
from leap_finetune.training.default_configs.vlm_sft_configs import (
    DEFAULT_LR_MULTIPLIERS,
    VLM_SFT_EXCLUDED_KEYS,
)
from leap_finetune.training.utils.worker_setup import (
    get_ray_train_eval_datasets,
    init_tracking_from_config,
    setup_training_worker,
)
from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.checkpointing.model_loading import load_vlm_model
from leap_finetune.training.utils.logging import (
    finish_tracker,
    is_rank_zero,
)
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.trainer_mixins import (
    RayDataLoaderMixin,
)
from leap_finetune.training.utils.trainer_lifecycle import (
    run_training_safely,
)
from leap_finetune.training.utils.vlm_optimizer import (
    build_vlm_param_groups,
    log_per_group_lrs,
)

logger = logging.getLogger(__name__)


# === VLM Trainer with per-component learning rates ===


class LFMVLMTrainer(RayDataLoaderMixin, Trainer):
    """VLM Trainer with per-component LR multipliers and Ray data integration.

    Vision encoder trains at a lower LR to preserve pretrained features,
    while the projector and LLM backbone train at the base rate.
    """

    def __init__(self, lr_multipliers: dict[str, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lr_multipliers = lr_multipliers or DEFAULT_LR_MULTIPLIERS
        self._optimizer_group_names: list[str] = []

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_groups, self._optimizer_group_names = build_vlm_param_groups(
            self.model,
            self.lr_multipliers,
            base_lr=self.args.learning_rate,
            weight_decay=float(self.args.weight_decay),
        )

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, betas=betas, fused=torch.cuda.is_available()
        )
        return self.optimizer

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        log_per_group_lrs(self.optimizer, self._optimizer_group_names, logs)
        super().log(logs, *args, **kwargs)


# === Training loop ===


def vlm_sft_run(training_config: dict) -> None:
    """VLM SFT training loop for Ray Train."""

    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()

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

    # Resume path is already resolved by the config parser.
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    # Filter out non-TrainingArguments parameters
    excluded_keys = VLM_SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    # Configure experiment tracking
    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    # Compute max_steps from materialized dataset size
    # (Trainer can't infer it from our bypassed DataLoader)
    num_samples = len(train_dataset)
    train_batch_size = train_config_filtered.get("per_device_train_batch_size", 4)
    grad_accum = train_config_filtered.get("gradient_accumulation_steps", 1)
    epochs = train_config_filtered.get("num_train_epochs", 3)
    steps_per_epoch = math.ceil(num_samples / train_batch_size)
    max_steps = steps_per_epoch * epochs // grad_accum

    logger.info(
        "Computed max_steps=%d (samples=%d, batch=%d, accum=%d, epochs=%s)",
        max_steps,
        num_samples,
        train_batch_size,
        grad_accum,
        epochs,
    )

    # Build training args — use max_steps instead of num_train_epochs
    train_config_filtered.pop("num_train_epochs", None)
    config_kwargs = {
        "report_to": tracker,
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
    # processing_class ensures processor + tokenizer are saved in checkpoints
    trainer = LFMVLMTrainer(
        lr_multipliers=lr_multipliers,
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    # Save PEFT model if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

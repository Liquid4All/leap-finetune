import logging
from typing import cast

import ray.train
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer
from ray.train.huggingface.transformers import prepare_trainer

from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    init_tracker,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.trainer_mixins import RayDataLoaderMixin, run_training_safely

logger = logging.getLogger(__name__)


class LFMDPOTrainer(RayDataLoaderMixin, DPOTrainer):
    """DPOTrainer with Ray data integration and pre-tokenized data support."""

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # DPOTrainer.prediction_step skips _prepare_inputs, so tensors stay on
        # CPU when using custom DataLoaders. Move them to device before the
        # parent's prediction_step runs concatenated_forward → model().
        inputs = self._prepare_inputs(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def dpo_run(training_config: dict) -> None:
    """DPO training loop — pre-tokenized data + DPOTrainer."""
    setup_worker_logging()

    # Get pre-tokenized shards for this worker
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    # Check for MoE model
    is_moe = is_moe_model_from_name(model_name)
    use_fsdp = is_moe and peft_config is None

    # Resume path is already resolved by config_parser
    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")

    # Filter out non-DPOConfig parameters
    excluded_keys = {
        "training_type",
        "wandb_logging",
        "tracker",
        "trackio_space_id",
        "leap_run_name_template",
        "resume_from_checkpoint",
    }
    if use_fsdp:
        excluded_keys.add("deepspeed")

    train_config_filtered = {
        k: v
        for k, v in train_config.items()
        if k not in excluded_keys
    }

    # Configure experiment tracking
    tracker = train_config.get("tracker", "none")
    if tracker == "none" and train_config.get("wandb_logging", False):
        tracker = "wandb"
    init_tracker(
        job_name,
        tracker,
        train_config.get("trackio_space_id"),
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    # Default eval batch size to train batch size to avoid OOM during eval
    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )

    # Build training args
    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    if use_fsdp:
        config_kwargs["fsdp"] = MOE_FSDP_CONFIG["fsdp"]
        config_kwargs["fsdp_config"] = MOE_FSDP_CONFIG["fsdp_config"]

    training_args = DPOConfig(**config_kwargs)

    # Load model after config is created
    model, tokenizer = load_model(model_name)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # DPOTrainer.__init__ expects model.warnings_issued (set by HF Trainer),
    # but some model architectures don't have it yet at init time
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # Pre-tokenized data — use subclass that skips _prepare_dataset
    trainer = LFMDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))
    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    # Save PEFT model if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

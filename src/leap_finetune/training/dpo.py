import logging
from typing import cast

from ray.train.huggingface.transformers import prepare_trainer
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from leap_finetune.training.utils.worker_setup import (
    default_eval_batch_size,
    get_ray_train_eval_datasets,
    init_tracking_from_config,
    load_causal_lm_for_training,
    setup_training_worker,
)
from leap_finetune.checkpointing.callback import LeapCheckpointCallback
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

logger = logging.getLogger(__name__)


class LFMDPOTrainer(RayDataLoaderMixin, DPOTrainer):
    """DPO trainer with Ray-sharded data loaders."""

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def dpo_run(training_config: dict) -> None:
    """DPO training loop for Ray-pretokenized datasets."""
    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = {
        "training_type",
        "wandb_logging",
        "tracker",
        "trackio_space_id",
        "leap_run_name_template",
        "resume_from_checkpoint",
        "model_config",
        "chat_template",
        "chat_template_path",
        "reshard_after_forward",
        "fsdp_cpu_offload",
        "checkpoint_staging_dir",
        "manual_sharded_checkpoint_format",
    }
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    default_eval_batch_size(train_config_filtered)

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = DPOConfig(**config_kwargs)

    model, tokenizer = load_causal_lm_for_training(
        training_config,
        model_name=model_name,
        train_config=train_config,
    )

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

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

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

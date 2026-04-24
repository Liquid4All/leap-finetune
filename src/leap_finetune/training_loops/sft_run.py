import logging

import ray.train
import torch.distributed as dist
from ray.train.huggingface.transformers import prepare_trainer
from transformers import Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.context_parallel import (
    aggregate_cp_loss,
    apply_cp_to_model,
    create_parallel_process_groups,
    split_batch_for_cp,
    validate_cp_config,
)
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    finish_tracker,
    init_tracker,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.trainer_mixins import RayDataLoaderMixin, run_training_safely

logger = logging.getLogger(__name__)


class LFMSFTTrainer(RayDataLoaderMixin, Trainer):
    """SFT trainer with Ray-sharded data loaders and optional CP loss handling."""

    def __init__(self, cp_config: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cp_config = cp_config

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.cp_config and self.cp_config["cp_size"] > 1:
            inputs = split_batch_for_cp(
                inputs, self.cp_config["cp_rank"], self.cp_config["cp_size"]
            )
        loss = super().training_step(model, inputs, num_items_in_batch, **kwargs)
        if self.cp_config and self.cp_config["cp_size"] > 1:
            loss = aggregate_cp_loss(
                loss, self.cp_config["cp_group"], self.cp_config["cp_size"]
            )
        return loss


def build_sft_data_collator(tokenizer, train_config: dict):
    padding_free = train_config.get("padding_free", train_config.get("packing", False))
    completion_only_loss = bool(train_config.get("completion_only_loss", False))
    return DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        completion_only_loss=completion_only_loss,
        padding_free=padding_free,
    )


def sft_run(training_config: dict) -> None:
    """SFT training loop for Ray-pretokenized datasets."""
    setup_worker_logging()

    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = SFT_EXCLUDED_KEYS | {
        "leap_run_name_template",
        "model_config",
        "context_parallel_size",
    }
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

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

    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        "remove_unused_columns": False,
        **train_config_filtered,
    }
    training_args = TrainingArguments(**config_kwargs)

    cp_size = train_config.get("context_parallel_size", 1)
    model_config = training_config.get("model_config")
    model, tokenizer = load_model(
        model_name,
        model_config=model_config,
        chat_template=train_config.get("chat_template"),
        chat_template_path=train_config.get("chat_template_path"),
    )

    cp_config = None
    if cp_size > 1:
        max_length = train_config.get("max_length")
        validate_cp_config(
            cp_size,
            max_length=max_length,
            world_size=dist.get_world_size(),
        )
        cp_config = create_parallel_process_groups(cp_size)
        apply_cp_to_model(model, cp_config)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    data_collator = build_sft_data_collator(tokenizer, train_config)
    trainer = LFMSFTTrainer(
        cp_config=cp_config,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

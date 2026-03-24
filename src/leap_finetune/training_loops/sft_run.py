import logging

import ray.train
from ray.train.huggingface.transformers import prepare_trainer
from transformers import Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG
from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    finish_tracker,
    init_tracker,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.trainer_mixins import RayDataLoaderMixin, run_training_safely

logger = logging.getLogger(__name__)


class LFMSFTTrainer(RayDataLoaderMixin, Trainer):
    """SFT Trainer with Ray data integration."""


def sft_run(training_config: dict) -> None:
    """SFT training loop — pre-tokenized data + plain Trainer."""
    setup_worker_logging()

    # === Get pre-tokenized shards for this worker ===
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
        logger.info("Resuming from checkpoint: %s", resume_from)

    # Filter out SFT-specific keys that don't belong in TrainingArguments
    excluded_keys = SFT_EXCLUDED_KEYS | {"leap_run_name_template"}
    if use_fsdp:
        excluded_keys = excluded_keys | {"deepspeed"}

    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
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
        "remove_unused_columns": False,
        **train_config_filtered,
    }
    if use_fsdp:
        config_kwargs["fsdp"] = MOE_FSDP_CONFIG["fsdp"]
        config_kwargs["fsdp_config"] = MOE_FSDP_CONFIG["fsdp_config"]

    training_args = TrainingArguments(**config_kwargs)

    # Load model + tokenizer on worker
    model, tokenizer = load_model(model_name)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Collator handles labels, padding, and padding-free mode (position_ids from seq_lengths)
    packing = training_config.get("train_config", {}).get("packing", False)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        padding_free=packing,
    )

    trainer = LFMSFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    # Add benchmark evaluation callback if configured
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    # Save PEFT model if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

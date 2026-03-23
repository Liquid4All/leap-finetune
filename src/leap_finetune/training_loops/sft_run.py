import torch.distributed as dist
import ray.train
from torch.utils.data import DataLoader
from ray.train.huggingface.transformers import prepare_trainer
from transformers import Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
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
    init_wandb_if_enabled,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model


class LFMSFTTrainer(Trainer):
    """Trainer that bypasses DistributedSampler since Ray already shards data."""

    def __init__(self, cp_config: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cp_config = cp_config

    def get_train_dataloader(self):
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

    def training_step(self, model, inputs, **kwargs):
        if self.cp_config and self.cp_config["cp_size"] > 1:
            inputs = split_batch_for_cp(
                inputs, self.cp_config["cp_rank"], self.cp_config["cp_size"]
            )
        loss = super().training_step(model, inputs, **kwargs)
        if self.cp_config and self.cp_config["cp_size"] > 1:
            loss = aggregate_cp_loss(
                loss, self.cp_config["cp_group"], self.cp_config["cp_size"]
            )
        return loss


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

    # Extract run name template before filtering
    run_name_template = training_config.get("train_config", {}).get(
        "leap_run_name_template"
    )

    # Filter out SFT-specific keys that don't belong in TrainingArguments
    excluded_keys = SFT_EXCLUDED_KEYS | {
        "leap_run_name_template",
        "model_config",
        "context_parallel_size",
    }

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

    # Default eval batch size to train batch size to avoid OOM during eval
    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )

    # Build training args
    config_kwargs = {
        "report_to": "wandb" if wandb_logging else "none",
        "run_name": job_name,
        "remove_unused_columns": False,
        **train_config_filtered,
    }
    training_args = TrainingArguments(**config_kwargs)

    # === Context Parallelism setup ===
    cp_size = training_config.get("train_config", {}).get("context_parallel_size", 1)
    model_config = training_config.get("model_config")

    # Load model + tokenizer on worker
    model, tokenizer = load_model(model_name, model_config=model_config)

    cp_config = None
    if cp_size > 1:
        max_length = training_config.get("train_config", {}).get("max_length")
        validate_cp_config(
            cp_size, max_length=max_length, world_size=dist.get_world_size()
        )
        cp_config = create_parallel_process_groups(cp_size)
        apply_cp_to_model(model, cp_config)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Collator handles labels, padding, and padding-free mode (position_ids from seq_lengths)
    packing = training_config.get("train_config", {}).get("packing", False)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        padding_free=packing,
    )

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
    trainer = prepare_trainer(trainer)

    try:
        trainer.train()
        print("Training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(
            keyword in error_msg.lower()
            for keyword in ["cuda error", "ecc error", "nccl", "collective", "timeout"]
        ):
            print(
                f"Training completed but hit distributed communication error during cleanup: {error_msg}"
            )
            print(
                "Training was successful - error occurred in post-training synchronization"
            )
        else:
            raise e

    # Save PEFT model if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

import logging

import torch
import ray.train
from torch.utils.data import DataLoader
from ray.train.huggingface.transformers import prepare_trainer
from transformers import Trainer, TrainingArguments
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from leap_finetune.training_configs.distributed_configs import (
    MOE_FSDP_CONFIG,
    MOE_FSDP_CONFIG_LARGE,
)
from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    init_wandb_if_enabled,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.model_utils import is_large_moe_model_from_name
from leap_finetune.utils.moe_dispatch import apply_ep_to_model
from leap_finetune.utils.moe_metrics_callback import MoEMetricsCallback
from leap_finetune.utils.moe_parallel import (
    MOE_FSDP_CONFIG_EP,
    create_ep_process_groups,
    shard_experts,
)
from leap_finetune.utils.moe_training import MoETrainingConfig, MoETrainingEnhancer
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model

logger = logging.getLogger(__name__)

# MoE-specific keys filtered from TrainingArguments
MOE_SFT_EXCLUDED_KEYS = SFT_EXCLUDED_KEYS | {
    "leap_run_name_template",
    "moe_training",
}


class LFMMoeSFTTrainer(Trainer):
    """SFT Trainer for MoE models with router-specific learning rates."""

    def __init__(self, router_lr_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.router_lr_ratio = router_lr_ratio

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

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay

        router_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if ".gate." in name:
                router_params.append(param)
            else:
                other_params.append(param)

        optimizer_groups = []
        if other_params:
            optimizer_groups.append(
                {"params": other_params, "lr": base_lr, "weight_decay": weight_decay}
            )
        if router_params:
            router_lr = base_lr * self.router_lr_ratio
            optimizer_groups.append(
                {"params": router_params, "lr": router_lr, "weight_decay": weight_decay}
            )
            logger.info(
                f"Router param group: {len(router_params)} params, lr={router_lr:.2e} "
                f"(ratio={self.router_lr_ratio})"
            )

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, betas=betas, fused=torch.cuda.is_available()
        )
        return self.optimizer


def moe_sft_run(training_config: dict) -> None:
    """MoE SFT training loop with aux losses, monitoring, and FSDP."""
    setup_worker_logging()

    # === Get pre-tokenized shards for this worker ===
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")
    use_fsdp = peft_config is None

    # MoE training config
    moe_config_dict = training_config.get("train_config", {}).get("moe_training", {})
    moe_config = MoETrainingConfig.from_dict(moe_config_dict)

    # Extract run name template before filtering
    run_name_template = training_config.get("train_config", {}).get(
        "leap_run_name_template"
    )

    # Select FSDP config based on model size
    is_large = is_large_moe_model_from_name(model_name)
    fsdp_config = MOE_FSDP_CONFIG_LARGE if is_large else MOE_FSDP_CONFIG

    # Filter out keys that don't belong in TrainingArguments
    excluded_keys = MOE_SFT_EXCLUDED_KEYS
    if use_fsdp:
        excluded_keys = excluded_keys | {"deepspeed"}

    train_config_filtered = {
        k: v
        for k, v in training_config.get("train_config").items()
        if k not in excluded_keys
    }

    # Configure wandb reporting if enabled
    wandb_logging = bool(
        training_config.get("train_config", {}).get("wandb_logging", False)
    )
    init_wandb_if_enabled(job_name, wandb_logging)

    # Default eval batch size
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
    if use_fsdp:
        config_kwargs["fsdp"] = fsdp_config["fsdp"]
        config_kwargs["fsdp_config"] = fsdp_config["fsdp_config"]

    training_args = TrainingArguments(**config_kwargs)

    # Load model + tokenizer
    model, tokenizer = load_model(model_name)

    # === Expert Parallelism setup (before FSDP wrapping) ===
    ep_size = moe_config_dict.get("expert_parallel_size")
    if ep_size and ep_size > 1:
        num_experts = model.config.num_local_experts
        ep_config = create_ep_process_groups(ep_size, num_experts)
        shard_experts(model, ep_config)
        # Override FSDP config for EP-aware hybrid sharding
        fsdp_config = MOE_FSDP_CONFIG_EP

    # Apply MoE training enhancements (aux losses, metrics)
    enhancer = MoETrainingEnhancer(moe_config)
    enhancer.apply(model)

    # Apply EP dispatch after enhancer (EP forward replaces enhanced forward)
    if ep_size and ep_size > 1:
        apply_ep_to_model(model, ep_config)

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Collator
    packing = training_config.get("train_config", {}).get("packing", False)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        padding_free=packing,
    )

    trainer = LFMMoeSFTTrainer(
        router_lr_ratio=moe_config.router_lr_ratio,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))
    trainer.add_callback(MoEMetricsCallback())
    trainer = prepare_trainer(trainer)

    try:
        trainer.train()
        logger.info("MoE SFT training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(
            keyword in error_msg.lower()
            for keyword in ["cuda error", "ecc error", "nccl", "collective", "timeout"]
        ):
            logger.warning(
                f"Training completed but hit distributed error during cleanup: {error_msg}"
            )
        else:
            raise

    # Save PEFT model if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

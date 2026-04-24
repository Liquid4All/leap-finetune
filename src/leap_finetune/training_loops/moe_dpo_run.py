import logging
from typing import cast

import ray.train
import torch
import torch.distributed as dist
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.torch import get_device as get_ray_torch_device
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from trl import DPOConfig, DPOTrainer

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.data_loaders.sampling import get_length_grouped_sampler
from leap_finetune.training_configs.distributed_configs import (
    resolve_fsdp_cpu_offload,
    resolve_reshard_after_forward,
)
from leap_finetune.utils.callbacks import LeapCheckpointCallback, MoEMetricsCallback
from leap_finetune.utils.context_parallel import (
    aggregate_cp_loss,
    apply_cp_to_model,
    create_parallel_process_groups,
    dist_barrier,
    split_batch_for_cp,
    validate_cp_config,
)
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    init_wandb_if_enabled,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.manual_sharded_trainer import (
    ManualShardedTrainerMixin,
    validate_manual_sharded_training_args,
)
from leap_finetune.utils.model_utils import (
    save_manual_sharded_trainer_checkpoint,
    should_run_final_manual_sharded_save,
)
from leap_finetune.utils.moe_losses import MoETrainingConfig, apply_moe_losses
from leap_finetune.utils.moe_parallel import (
    apply_ep_to_model,
    apply_fsdp2,
    apply_fsdp2_for_ep,
    create_dp_mesh,
    create_ep_mesh,
    shard_experts,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model

logger = logging.getLogger(__name__)

MOE_DPO_EXCLUDED_KEYS = {
    "training_type",
    "wandb_logging",
    "leap_run_name_template",
    "moe_training",
    "model_config",
    "context_parallel_size",
    "chat_template",
    "chat_template_path",
    "reshard_after_forward",
    "fsdp_cpu_offload",
}


class LFMMoeDPOTrainer(ManualShardedTrainerMixin, DPOTrainer):
    """DPO Trainer for MoE models with EP and optional CP support."""

    def __init__(
        self,
        ep_config: dict | None = None,
        manual_fsdp2: bool = False,
        cp_config: dict | None = None,
        **kwargs,
    ):
        self.manual_sharded = manual_fsdp2
        super().__init__(**kwargs)
        self.ep_config = ep_config
        self.cp_config = cp_config

    def _prepare_dataset(self, dataset, *args, **kwargs):
        return dataset

    def get_train_dataloader(self):
        sampler = get_length_grouped_sampler(self.train_dataset, self._train_batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.ep_config:
            inputs = self._replicate_batch_across_ep(inputs)
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

    def _replicate_batch_across_ep(self, inputs: dict) -> dict:
        """Broadcast batch from EP rank 0 to all EP ranks in the group."""
        ep_group = self.ep_config["ep_group"]
        src_rank = dist.get_process_group_ranks(ep_group)[0]
        device = torch.cuda.current_device()
        is_src = dist.get_rank() == src_rank

        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device)

            shape = torch.tensor(value.shape, device=device, dtype=torch.long)
            dist.broadcast(shape, src=src_rank, group=ep_group)
            if not is_src:
                value = torch.empty(
                    tuple(shape.tolist()), device=device, dtype=value.dtype
                )

            dist.broadcast(value, src=src_rank, group=ep_group)
            inputs[key] = value

        return inputs


def moe_dpo_run(training_config: dict) -> None:
    """MoE DPO training loop with revised EP support and current CP support."""
    setup_worker_logging()
    if torch.cuda.is_available():
        worker_device = get_ray_torch_device()
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
            logger.info("Pinned Ray worker to CUDA device %s", worker_device)

    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray)

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    moe_config_dict = training_config.get("train_config", {}).get("moe_training", {})
    moe_config = MoETrainingConfig.from_dict(moe_config_dict)
    ep_size = moe_config_dict.get("expert_parallel_size", 1) or 1
    train_config = training_config.get("train_config", {})
    cp_size = train_config.get("context_parallel_size", 1)
    use_ep = ep_size > 1
    use_fsdp2 = peft_config is None and not use_ep
    if use_ep and cp_size > 1:
        raise ValueError(
            "expert_parallel_size > 1 cannot be combined with context_parallel_size > 1. "
            "Only DP x CP is supported for context parallelism."
        )

    run_name_template = train_config.get("leap_run_name_template")

    if cp_size > 1:
        validate_cp_config(
            cp_size,
            world_size=dist.get_world_size(),
        )

    reshard_after_forward = resolve_reshard_after_forward(
        train_config,
        default=False if use_ep else True,
    )
    cpu_offload = resolve_fsdp_cpu_offload(train_config)

    excluded_keys = MOE_DPO_EXCLUDED_KEYS
    if use_ep or use_fsdp2:
        excluded_keys = excluded_keys | {"deepspeed"}

    train_config_filtered = {
        k: v
        for k, v in train_config.items()
        if k not in excluded_keys
    }
    requested_save_strategy = train_config_filtered.get("save_strategy", "no")

    wandb_logging = bool(
        train_config.get("wandb_logging", False)
    )
    init_wandb_if_enabled(job_name, wandb_logging)

    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )

    config_kwargs = {
        "report_to": "wandb" if wandb_logging else "none",
        "run_name": job_name,
        **train_config_filtered,
    }
    if use_ep or use_fsdp2:
        validate_manual_sharded_training_args(config_kwargs)
    if use_ep:
        logger.info(
            "EP mode: ep_size=%s, FSDP2 on dp_mesh, reshard_after_forward=%s",
            ep_size,
            reshard_after_forward,
        )
    elif use_fsdp2:
        logger.info(
            "Non-EP DPO mode: manual FSDP2 on full DP mesh (cp_size=%s) "
            "reshard_after_forward=%s cpu_offload=%s",
            cp_size,
            reshard_after_forward,
            cpu_offload,
        )

    training_args = DPOConfig(**config_kwargs)

    model_config = training_config.get("model_config")
    model, tokenizer = load_model(
        model_name,
        model_config=model_config,
        chat_template=train_config.get("chat_template"),
        chat_template_path=train_config.get("chat_template_path"),
    )

    ep_config = None
    device_mesh = None
    dp_mesh = None
    num_experts = getattr(model.config, "num_experts", None)

    if use_ep or use_fsdp2 or cp_size > 1:
        logger.info("Waiting for all ranks to finish model loading...")
        dist_barrier()

    if use_ep:
        ep_config, device_mesh = create_ep_mesh(ep_size, num_experts)
        shard_experts(model, ep_config)
        apply_ep_to_model(model, ep_config, moe_config=moe_config)
    else:
        apply_moe_losses(model, moe_config)
        if use_fsdp2:
            dp_mesh = create_dp_mesh()

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    if use_ep and device_mesh is not None:
        model = apply_fsdp2_for_ep(
            model,
            device_mesh,
            reshard_after_forward=reshard_after_forward,
        )
    elif use_fsdp2 and dp_mesh is not None:
        model = apply_fsdp2(
            model,
            dp_mesh,
            reshard_after_forward=reshard_after_forward,
            cpu_offload=cpu_offload,
        )

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = LFMMoeDPOTrainer(
        ep_config=ep_config,
        manual_fsdp2=(use_ep or use_fsdp2),
        cp_config=None,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
    )
    trainer.run_name_template = run_name_template

    trainer.add_callback(
        LeapCheckpointCallback(
            run_name_template=run_name_template,
            manual_sharded=(use_ep or use_fsdp2),
        )
    )
    trainer.add_callback(MoEMetricsCallback())
    trainer = prepare_trainer(trainer)

    if cp_size > 1:
        cp_config = create_parallel_process_groups(cp_size)
        apply_cp_to_model(trainer.model, cp_config)
        trainer.cp_config = cp_config

    dist_barrier()

    try:
        trainer.train()
        if (use_ep or use_fsdp2) and should_run_final_manual_sharded_save(
            trainer=trainer,
            requested_save_strategy=requested_save_strategy,
        ):
            save_manual_sharded_trainer_checkpoint(
                trainer=trainer,
                model=trainer.model,
                trial=None,
                ep_group=ep_config["ep_group"] if ep_config is not None else None,
            )
        logger.info("MoE DPO training completed successfully")
    except RuntimeError as e:
        error_msg = str(e)
        if any(
            keyword in error_msg.lower()
            for keyword in ["cuda error", "ecc error", "nccl", "collective", "timeout"]
        ):
            logger.warning(
                "Training completed but hit distributed error during cleanup: %s",
                error_msg,
            )
        else:
            raise

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

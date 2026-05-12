import logging
import ray.train
import torch
import torch.distributed as dist
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.torch import get_device as get_ray_torch_device
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.data_loaders.sampling import get_length_grouped_sampler
from leap_finetune.training_configs.distributed_configs import (
    resolve_fsdp_cpu_offload,
    resolve_reshard_after_forward,
)
from leap_finetune.training_configs.sft_configs import SFT_EXCLUDED_KEYS
from leap_finetune.training_loops.cp_trainer_utils import (
    compute_cp_loss_for_trainer,
    cp_enabled,
    cp_prediction_step,
)
from leap_finetune.training_loops.sft_run import build_sft_data_collator
from leap_finetune.utils.callbacks import LeapCheckpointCallback, MoEMetricsCallback
from leap_finetune.utils.context_parallel import (
    aggregate_cp_loss,
    apply_cp_to_model,
    create_parallel_process_groups,
    dist_barrier,
    split_batch_for_cp,
    validate_cp_batch_replicated,
    validate_cp_config,
    validate_cp_model_support,
)
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import (
    init_wandb_if_enabled,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.memory_trace import (
    init_memory_trace,
    wrap_optimizer_step,
    write_memory_trace_event,
)
from leap_finetune.utils.trainer_mixins import (
    ManualShardedCheckpointMixin,
    validate_manual_sharded_training_args,
)
from leap_finetune.utils.model_utils import (
    build_manual_sharded_export_metadata_from_config,
    save_manual_sharded_checkpoint,
    should_run_final_manual_sharded_save,
)
from leap_finetune.utils.moe_losses import (
    MoETrainingConfig,
    apply_moe_losses,
    set_moe_sequence_partition_group,
)
from leap_finetune.utils.moe_parallel import (
    apply_ep_to_model,
    apply_fsdp2,
    apply_fsdp2_for_ep,
    create_dp_mesh,
    create_ep_mesh,
    log_cuda_memory,
    shard_experts,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model

logger = logging.getLogger(__name__)

MOE_SFT_EXCLUDED_KEYS = SFT_EXCLUDED_KEYS | {
    "leap_run_name_template",
    "moe_training",
    "model_config",
    "context_parallel_size",
}

class LFMMoeSFTTrainer(ManualShardedCheckpointMixin, Trainer):
    """SFT Trainer for MoE models with EP/FSDP2 support."""

    def __init__(
        self,
        ep_config: dict | None = None,
        manual_fsdp2: bool = False,
        cp_config: dict | None = None,
        run_name_template: str | None = None,
        checkpoint_staging_dir: str | None = None,
        manual_sharded_checkpoint_format: str = "hf",
        manual_sharded_export_metadata: dict | None = None,
        **kwargs,
    ):
        self.manual_sharded = manual_fsdp2
        self.run_name_template = run_name_template
        self.checkpoint_staging_dir = checkpoint_staging_dir
        self.manual_sharded_checkpoint_format = manual_sharded_checkpoint_format
        self.manual_sharded_export_metadata = dict(manual_sharded_export_metadata or {})
        super().__init__(**kwargs)
        self.ep_config = ep_config
        self.cp_config = cp_config
        self._cp_batch_validated = False
        self._cp_eval_batch_validated = False

    def get_train_dataloader(self):
        sampler_seed_rank = None
        if self.cp_config is not None and self.cp_config.get("cp_size", 1) > 1:
            sampler_seed_rank = int(self.cp_config.get("dp_rank", 0))
        elif self.ep_config is not None:
            sampler_seed_rank = int(self.ep_config["dp_rank"])

        sampler_generator = None
        if sampler_seed_rank is not None:
            sampler_generator = torch.Generator().manual_seed(42 + sampler_seed_rank)

        sampler = get_length_grouped_sampler(
            self.train_dataset,
            self._train_batch_size,
            generator=sampler_generator,
        )
        dataloader_kwargs = {}
        if sampler is None and sampler_generator is not None:
            dataloader_kwargs["generator"] = sampler_generator
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            **dataloader_kwargs,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("No evaluation dataset configured for this run")
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def create_optimizer(self):
        if self.optimizer is not None:
            return wrap_optimizer_step(self.optimizer, self._memory_trace_step)

        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.AdamW(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=betas,
            fused=torch.cuda.is_available(),
        )
        return wrap_optimizer_step(self.optimizer, self._memory_trace_step)

    def _memory_trace_step(self) -> int:
        return int(getattr(getattr(self, "state", None), "global_step", 0))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # === 2. Run the local forward on the rank's current batch shard ===
        if self.cp_config and self.cp_config["cp_size"] > 1 and "labels" in inputs:
            output = compute_cp_loss_for_trainer(
                self,
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            output = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        write_memory_trace_event("after_forward_loss", step=self._memory_trace_step())
        return output

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if cp_enabled(self.cp_config) and "labels" in inputs:
            return cp_prediction_step(
                self, model, inputs, prediction_loss_only, ignore_keys
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        # === 1. Split the incoming packed sequence for the local CP rank ===
        write_memory_trace_event("train_step_start", step=self._memory_trace_step())
        if self.cp_config and self.cp_config["cp_size"] > 1:
            if not self._cp_batch_validated:
                validate_cp_batch_replicated(
                    inputs,
                    self.cp_config["cp_group"],
                    self.cp_config["cp_rank"],
                    self.cp_config["cp_size"],
                )
                self._cp_batch_validated = True
            inputs = split_batch_for_cp(
                inputs, self.cp_config["cp_rank"], self.cp_config["cp_size"]
            )

        # === 3. Let Trainer run forward + backward on the local shard ===
        loss = super().training_step(model, inputs, num_items_in_batch, **kwargs)

        # === 4. Average the detached loss view across the CP ranks ===
        if self.cp_config and self.cp_config["cp_size"] > 1:
            loss = aggregate_cp_loss(
                loss, self.cp_config["cp_group"], self.cp_config["cp_size"]
            )

        # === 5. Mark the end of the per-step backward path ===
        write_memory_trace_event("after_backward", step=self._memory_trace_step())
        return loss


def moe_sft_run(training_config: dict) -> None:
    """MoE SFT training loop with revised non-EP and EP paths."""
    setup_worker_logging()
    if torch.cuda.is_available():
        worker_device = get_ray_torch_device()
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
            logger.info("Pinned Ray worker to CUDA device %s", worker_device)

    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray) if eval_ds_ray is not None else None

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    moe_config_dict = training_config.get("train_config", {}).get("moe_training", {})
    _validate_supported_moe_sft_config(moe_config_dict)
    moe_config = MoETrainingConfig.from_dict(moe_config_dict)
    ep_size = moe_config_dict.get("expert_parallel_size", 1) or 1
    train_config = training_config.get("train_config", {})
    resume_from = train_config.get("resume_from_checkpoint")
    cp_size = train_config.get("context_parallel_size", 1)

    use_ep = ep_size > 1
    if use_ep and cp_size > 1:
        raise ValueError(
            "expert_parallel_size > 1 cannot be combined with context_parallel_size > 1. "
            "Only DP x CP is supported for context parallelism."
        )

    # CP should stay on the same non-EP sharded path as the working baseline.
    # Falling back to DeepSpeed here materially changes memory behavior.
    use_fsdp2 = peft_config is None and not use_ep

    run_name_template = train_config.get("leap_run_name_template")

    excluded_keys = MOE_SFT_EXCLUDED_KEYS | (
        {"deepspeed"} if (use_ep or use_fsdp2) else set()
    )

    train_config_filtered = {
        k: v
        for k, v in train_config.items()
        if k not in excluded_keys
    }
    requested_save_strategy = train_config_filtered.get("save_strategy", "no")
    manual_sharded_checkpoint_format = train_config.get(
        "manual_sharded_checkpoint_format", "hf"
    )

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
        "remove_unused_columns": False,
        **train_config_filtered,
    }

    if use_ep or use_fsdp2:
        validate_manual_sharded_training_args(config_kwargs)
        if use_ep:
            logger.info("EP mode: ep_size=%s, FSDP2 on dp_mesh", ep_size)
        else:
            logger.info(
                "Non-EP mode: manual FSDP2 on full DP mesh (cp_size=%s)", cp_size
            )

    training_args = TrainingArguments(**config_kwargs)

    if cp_size > 1:
        max_length = train_config.get("max_length")
        validate_cp_config(
            cp_size,
            max_length=max_length,
            world_size=dist.get_world_size(),
        )

    init_memory_trace(training_args.output_dir, framework="leap")
    model_config = training_config.get("model_config")
    model, tokenizer = load_model(
        model_name,
        model_config=model_config,
        chat_template=train_config.get("chat_template"),
        chat_template_path=train_config.get("chat_template_path"),
    )
    if cp_size > 1:
        validate_cp_model_support(model, train_config)
    write_memory_trace_event("after_model_load", always=True)
    log_cuda_memory("after_load_model")

    if use_ep or use_fsdp2:
        logger.info("Waiting for all ranks to finish model loading...")
        dist_barrier()

    ep_config = None
    device_mesh = None
    dp_mesh = None
    num_experts = getattr(model.config, "num_experts", None)

    if use_ep:
        ep_config, device_mesh = create_ep_mesh(ep_size, num_experts)
        dp_mesh = device_mesh["dp"]
        shard_experts(model, ep_config)
        write_memory_trace_event("after_shard_experts", always=True)
        log_cuda_memory("after_shard_experts", summary=True)
        apply_ep_to_model(model, ep_config, moe_config=moe_config)
    else:
        apply_moe_losses(model, moe_config)
        if use_fsdp2:
            dp_mesh = create_dp_mesh()

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    if use_ep and device_mesh is not None:
        reshard_after_forward = resolve_reshard_after_forward(
            train_config, default=False
        )
        logger.info(
            "Applying EP FSDP2 with reshard_after_forward=%s",
            reshard_after_forward,
        )
        model = apply_fsdp2_for_ep(
            model,
            device_mesh,
            reshard_after_forward=reshard_after_forward,
        )
        write_memory_trace_event("after_apply_fsdp2", always=True)
        log_cuda_memory("after_apply_fsdp2", summary=True)
    elif use_fsdp2 and dp_mesh is not None:
        reshard_after_forward = resolve_reshard_after_forward(
            train_config, default=True
        )
        cpu_offload = resolve_fsdp_cpu_offload(train_config)
        logger.info(
            "Applying non-EP FSDP2 with reshard_after_forward=%s cpu_offload=%s",
            reshard_after_forward,
            cpu_offload,
        )
        model = apply_fsdp2(
            model,
            dp_mesh,
            reshard_after_forward=reshard_after_forward,
            cpu_offload=cpu_offload,
        )
        write_memory_trace_event("after_apply_fsdp2", always=True)
        log_cuda_memory("after_apply_fsdp2", summary=True)

    data_collator = build_sft_data_collator(
        tokenizer, training_config.get("train_config", {})
    )
    manual_sharded_export_metadata = build_manual_sharded_export_metadata_from_config(
        training_config,
        processing_class=tokenizer,
    )

    trainer = LFMMoeSFTTrainer(
        ep_config=ep_config,
        manual_fsdp2=(use_ep or use_fsdp2),
        cp_config=None,
        run_name_template=run_name_template,
        checkpoint_staging_dir=train_config.get("checkpoint_staging_dir"),
        manual_sharded_checkpoint_format=manual_sharded_checkpoint_format,
        manual_sharded_export_metadata=manual_sharded_export_metadata,
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.callback_handler.callbacks.insert(0, MoEMetricsCallback())
    trainer.add_callback(
        LeapCheckpointCallback(
            run_name_template=run_name_template,
            manual_sharded=(use_ep or use_fsdp2),
        )
    )
    trainer = prepare_trainer(trainer)

    if cp_size > 1:
        cp_config = create_parallel_process_groups(cp_size)
        apply_cp_to_model(trainer.model, cp_config)
        set_moe_sequence_partition_group(trainer.model, cp_config["cp_group"])
        trainer.cp_config = cp_config

    dist_barrier()

    try:
        logger.info("Starting trainer.train() for MoE SFT")
        trainer.train(resume_from_checkpoint=resume_from)
        logger.info("trainer.train() returned for MoE SFT")
        if (use_ep or use_fsdp2) and should_run_final_manual_sharded_save(
            trainer=trainer,
            requested_save_strategy=requested_save_strategy,
        ):
            logger.info("Running explicit final manual-sharded checkpoint save")
            save_manual_sharded_checkpoint(
                trainer=trainer,
                model=trainer.model,
                trial=None,
                checkpoint_format=manual_sharded_checkpoint_format,
                ep_group=ep_config["ep_group"] if ep_config is not None else None,
                export_metadata=trainer.get_manual_sharded_export_metadata(),
            )
            logger.info("Explicit final manual-sharded checkpoint save completed")
        logger.info("MoE SFT training completed successfully")
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


def _validate_supported_moe_sft_config(moe_config: dict) -> None:
    capacity_factor = moe_config.get("capacity_factor")
    token_drop_policy = moe_config.get("token_drop_policy")

    if capacity_factor is None and token_drop_policy in (None, "probs"):
        return

    raise ValueError(
        "MoE SFT currently supports uncapped routing only. "
        "Remove capacity_factor/token_drop_policy from moe_training."
    )

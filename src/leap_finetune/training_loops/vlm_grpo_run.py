"""VLM GRPO training loop.

Mirrors ``grpo_run.py`` with two VLM-specific additions:

1. Uses ``load_vlm_model`` (AutoModelForImageTextToText + AutoProcessor).
2. Uses ``LFMVLMGRPOTrainer`` which overrides ``create_optimizer`` to
   build per-component learning-rate param groups via the shared
   ``utils/vlm_optimizer.py`` helper — same policy as ``LFMVLMTrainer``
   (vision encoder at 0.1× base LR to preserve pretrained features).
"""

from __future__ import annotations

import logging
from typing import Any

import ray.train
import torch
from ray.train.huggingface.transformers import prepare_trainer
from trl import GRPOConfig, GRPOTrainer

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_vlm_benchmarks_from_config,
)
from leap_finetune.rewards import resolve_reward_specs
from leap_finetune.training_configs.grpo_configs import VLM_GRPO_EXCLUDED_KEYS
from leap_finetune.training_configs.vlm_sft_config import DEFAULT_LR_MULTIPLIERS
from leap_finetune.utils.checkpoint_callback import LeapCheckpointCallback
from leap_finetune.utils.load_models import load_vlm_model
from leap_finetune.utils.logging_utils import (
    finish_tracker,
    init_tracker,
    is_rank_zero,
    setup_worker_logging,
)
from leap_finetune.utils.peft import apply_peft_to_model, merge_and_save_peft_model
from leap_finetune.utils.trainer_mixins import run_training_safely
from leap_finetune.utils.vlm_optimizer import build_vlm_param_groups, log_per_group_lrs

logger = logging.getLogger(__name__)


class LFMVLMGRPOTrainer(GRPOTrainer):
    """VLM GRPO trainer with per-component learning rate multipliers.

    Vision encoder trains at a lower LR (``lr_multipliers["model.vision_tower"]``,
    default 0.1) to preserve pretrained features; the multi-modal projector and
    language model head train at the base learning rate. This mirrors the
    behaviour of ``LFMVLMTrainer`` used for VLM SFT, via the shared helpers
    in ``utils/vlm_optimizer.py``.

    Unlike the SFT variant we do NOT subclass RayDataLoaderMixin — GRPO's
    RepeatSampler must go through accelerate's per-rank distribution, so we
    use GRPOTrainer's native dataloader (see grpo_run.py for details).
    """

    def __init__(self, lr_multipliers: dict[str, float] | None = None, **kwargs: Any):
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


def vlm_grpo_run(training_config: dict) -> None:
    """VLM GRPO training loop (Ray Train worker entrypoint)."""
    setup_worker_logging()

    # Full dataset on each worker (GRPO sampler handles per-rank distribution)
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray) if eval_ds_ray is not None else None

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    train_config = training_config.get("train_config", {})
    max_image_tokens = train_config.get("max_image_tokens")
    do_image_splitting = train_config.get("do_image_splitting", True)
    run_name_template = train_config.get("leap_run_name_template")

    # Per-component LR knobs, same mechanism as LFMVLMTrainer
    lr_multipliers = dict(DEFAULT_LR_MULTIPLIERS)
    if "lr_multipliers" in train_config:
        lr_multipliers.update(train_config["lr_multipliers"])
    if "vision_encoder_lr_multiplier" in train_config:
        lr_multipliers["model.vision_tower"] = train_config[
            "vision_encoder_lr_multiplier"
        ]

    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    # Strip VLM-specific + GRPO-excluded keys before constructing GRPOConfig
    excluded_keys = VLM_GRPO_EXCLUDED_KEYS | {"leap_run_name_template"}
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    # Experiment tracking
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

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = GRPOConfig(**config_kwargs)

    # Load VLM + processor
    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )
    # GRPO needs left padding; the processor holds the tokenizer for VLMs.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Resolve reward functions
    reward_funcs, reward_weights = resolve_reward_specs(
        training_config.get("rewards"),
        training_config.get("config_dir") or ".",
    )

    # Connect OpenEnv env if configured. See grpo_run.py for the rationale
    # — OpenEnv is optional and deferred so src/leap_finetune/rl_envs/ can
    # be absent without breaking this loop.
    rl_env_cfg = training_config.get("rl_env")
    rollout_func = None
    if rl_env_cfg is not None:
        try:
            from leap_finetune.rl_envs import (  # noqa: PLC0415
                build_openenv_rollout_func,
                connect_openenv,
                env_reward,
            )
        except ImportError as e:
            raise ImportError(
                "`rl_env` in your config requires the optional OpenEnv adapter at "
                "`src/leap_finetune/rl_envs/`. Either remove `rl_env` from your "
                "config (RLVR with rewards-only is the recommended path for "
                "verifiable-reward tasks), or restore the rl_envs directory."
            ) from e

        env_client = connect_openenv(rl_env_cfg)
        rollout_func = build_openenv_rollout_func(
            env_client,
            max_turns=int(rl_env_cfg.get("max_turns", 1)),
            reset_kwargs=rl_env_cfg.get("reset_kwargs") or {},
            action_key=rl_env_cfg.get("action_key", "message"),
        )
        reward_funcs = [env_reward, *reward_funcs]
        if reward_weights is not None:
            reward_weights = [1.0, *reward_weights]

    if not reward_funcs:
        raise ValueError(
            "VLM GRPO requires at least one reward function. Add a `rewards:` "
            "block with './rewards/<file>.py::<fn>' specs, or set `rl_env:` to "
            "use an OpenEnv environment's reward."
        )

    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    trainer = LFMVLMGRPOTrainer(
        lr_multipliers=lr_multipliers,
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        rollout_func=rollout_func,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    # VLM benchmark callback (reused unchanged from SFT path)
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_vlm_benchmarks_from_config(benchmark_configs, processor)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

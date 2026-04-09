"""Text GRPO training loop.

Mirrors the structure of ``dpo_run.py`` / ``sft_run.py`` but uses TRL v1's
``GRPOTrainer`` and does NOT pre-tokenize. GRPO generates completions online
each step from raw prompts.

Key differences from SFT/DPO runs:

* No ``RayDataLoaderMixin`` — GRPO's ``RepeatSampler`` + ``accelerator.prepare``
  handles per-rank prompt distribution natively, and bypassing that (like we
  do for SFT/DPO) would break group-relative reward normalization. The Ray
  Train driver carves around this by passing the full (unsplit) dataset to
  every worker via ``DataConfig(datasets_to_split=[])``.
* Reward functions come from the YAML ``rewards:`` block resolved via
  ``resolve_reward_specs``. Customers write plain Python files in
  ``rewards/`` and reference them by path.
* When ``rl_env`` is set in YAML, we connect to an OpenEnv environment and
  pass a ``rollout_func`` to the trainer that drives env.reset/step around
  TRL's generation helper. The environment's per-step reward is forwarded as
  an extra ``env_reward`` field and picked up by a tiny auto-prepended
  ``env_reward`` reward function.
"""

from __future__ import annotations

import logging
from typing import cast

import ray.train
from ray.train.huggingface.transformers import prepare_trainer
from transformers import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from leap_finetune.rewards import resolve_reward_specs
from leap_finetune.training_configs.grpo_configs import GRPO_EXCLUDED_KEYS
from leap_finetune.training_configs.distributed_configs import MOE_FSDP_CONFIG
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
from leap_finetune.utils.trainer_mixins import run_training_safely

logger = logging.getLogger(__name__)


class LFMGRPOTrainer(GRPOTrainer):
    """Text GRPO trainer.

    Intentionally a passthrough subclass — GRPOTrainer's native
    ``get_train_dataloader`` / ``_get_train_sampler`` handle prompt-group
    distribution across ranks via ``RepeatSampler``. Unlike SFT/DPO, we must
    NOT override the dataloader or we would break group-relative rewards.
    """


def grpo_run(training_config: dict) -> None:
    """GRPO training loop (Ray Train worker entrypoint)."""
    setup_worker_logging()

    # Each worker sees the full dataset because the driver set
    # DataConfig(datasets_to_split=[]) for GRPO. TRL + accelerate then
    # distribute repeated prompt indices across ranks.
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray) if eval_ds_ray is not None else None

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    # MoE variants auto-switch to FSDP for full fine-tuning, DeepSpeed with PEFT
    is_moe = is_moe_model_from_name(model_name)
    use_fsdp = is_moe and peft_config is None

    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    # Filter out non-GRPOConfig keys (everything in GRPO_EXCLUDED_KEYS)
    excluded_keys = GRPO_EXCLUDED_KEYS | {"leap_run_name_template"}
    if use_fsdp:
        excluded_keys = excluded_keys | {"deepspeed"}
    train_config_filtered = {
        k: v for k, v in train_config.items() if k not in excluded_keys
    }

    # Configure experiment tracking (same pattern as SFT/DPO)
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

    # Build GRPOConfig
    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    if use_fsdp:
        config_kwargs["fsdp"] = MOE_FSDP_CONFIG["fsdp"]
        config_kwargs["fsdp_config"] = MOE_FSDP_CONFIG["fsdp_config"]

    training_args = GRPOConfig(**config_kwargs)

    # Load model + tokenizer
    model, tokenizer = load_model(model_name)
    # GRPO requires left-padded prompts so generated completions append cleanly.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    # Resolve reward functions from the driver-side config_dir. Loaders are
    # deterministic, so each worker re-runs the resolution independently
    # rather than shipping closures across processes.
    reward_funcs, reward_weights = resolve_reward_specs(
        training_config.get("rewards"),
        training_config.get("config_dir") or ".",
    )

    # Connect OpenEnv environment (if configured) and build a rollout_func.
    # OpenEnv is an optional advanced feature for agentic / multi-turn
    # rollouts where the environment state evolves with actions. For pure
    # RLVR (verifiable reward) workflows — math, grounding, format, code
    # correctness — you should use the `rewards:` block instead; it's
    # strictly simpler and more efficient.
    #
    # The import is deferred so the `src/leap_finetune/rl_envs/` directory
    # can be absent (not committed) without breaking the training loop for
    # everyone else.
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
        # Auto-prepend env_reward so the env's per-step reward contributes.
        reward_funcs = [env_reward, *reward_funcs]
        if reward_weights is not None:
            reward_weights = [1.0, *reward_weights]

    # GRPO requires at least one reward function. Fail loudly if neither
    # `rewards` nor `rl_env` was provided — otherwise the trainer would
    # silently do nothing useful.
    if not reward_funcs:
        raise ValueError(
            "GRPO requires at least one reward function. Add a `rewards:` block "
            "with a list of './rewards/<file>.py::<fn>' specs, or set `rl_env:` "
            "to use an OpenEnv environment's reward."
        )

    # Apply reward_weights if resolved from YAML
    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    trainer = LFMGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=cast(PreTrainedTokenizerBase, tokenizer),
        rollout_func=rollout_func,
    )

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    # Reuse benchmark callback (same pattern as SFT/DPO)
    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_llm_benchmarks_from_config(benchmark_configs, tokenizer)
        if benchmarks:
            trainer.add_callback(BenchmarkEvalCallback(benchmarks))

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    # Save PEFT adapter if applicable
    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, tokenizer, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

from __future__ import annotations

import logging
from typing import cast

from ray.train.huggingface.transformers import prepare_trainer
from transformers import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

from leap_finetune.checkpointing.callback import LeapCheckpointCallback
from leap_finetune.checkpointing.model_info import is_moe_model_from_name
from leap_finetune.checkpointing.model_loading import load_model
from leap_finetune.evaluation import (
    BenchmarkEvalCallback,
    create_llm_benchmarks_from_config,
)
from leap_finetune.rl.rewards import resolve_reward_specs
from leap_finetune.training.default_configs.grpo_configs import GRPO_EXCLUDED_KEYS
from leap_finetune.training.peft.peft import (
    apply_peft_to_model,
    merge_and_save_peft_model,
)
from leap_finetune.training.utils.logging import (
    finish_tracker,
    is_rank_zero,
)
from leap_finetune.training.utils.trainer_lifecycle import run_training_safely
from leap_finetune.training.utils.worker_setup import (
    get_ray_train_eval_datasets,
    init_tracking_from_config,
    setup_training_worker,
)
from leap_finetune.training.utils.config_filter import filter_runtime_config_kwargs

logger = logging.getLogger(__name__)


# === Text GRPO loop ===
#
# GRPO generates completions online, so it must use TRL's native
# RepeatSampler/accelerate path. The Ray driver gives every worker the full
# dataset; TRL then distributes repeated prompt groups across ranks.


def grpo_run(training_config: dict) -> None:
    setup_training_worker()
    train_dataset, eval_dataset = get_ray_train_eval_datasets()

    peft_config = training_config.get("peft_config")
    model_name = training_config.get("model_name", "")
    job_name = training_config.get("job_name", "leap-ft-run")

    is_moe = is_moe_model_from_name(model_name)
    if is_moe:
        raise ValueError("GRPO for MoE models is not supported in this EP branch")

    train_config = training_config.get("train_config", {})
    run_name_template = train_config.get("leap_run_name_template")
    resume_from = train_config.get("resume_from_checkpoint")
    output_dir = train_config.get("output_dir", "")
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)

    excluded_keys = GRPO_EXCLUDED_KEYS
    train_config_filtered, _ = filter_runtime_config_kwargs(
        train_config,
        excluded_keys=excluded_keys,
        config_cls=GRPOConfig,
    )

    tracker = init_tracking_from_config(
        job_name,
        train_config,
        output_dir=output_dir if output_dir else None,
        resume_from_checkpoint=resume_from,
    )

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = GRPOConfig(**config_kwargs)

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

    # Deferred import keeps OpenEnv optional for plain reward-function GRPO.
    rl_env_cfg = training_config.get("rl_env")
    rollout_func = None
    if rl_env_cfg is not None:
        try:
            from leap_finetune.rl.environments import (  # noqa: PLC0415
                build_openenv_rollout_func,
                connect_openenv,
                env_reward,
            )
        except ImportError as e:
            raise ImportError(
                "`rl_env:` requires the optional OpenEnv extra. "
                "Install with: uv sync --extra rl-env"
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
            "GRPO requires at least one reward function. Add a `rewards:` block "
            "with a list of './rewards/<file>.py::<fn>' specs, or set `rl_env:` "
            "to use an OpenEnv environment's reward."
        )

    if reward_weights is not None:
        training_args.reward_weights = reward_weights

    trainer = GRPOTrainer(
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

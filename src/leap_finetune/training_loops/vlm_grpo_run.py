"""VLM GRPO training loop.

Mirrors ``grpo_run.py`` plus two VLM-specific additions: per-component
LR multipliers and patches to TRL's multimodal data path so VLM images
actually reach the training forward pass (see ``LFMVLMGRPOTrainer``).
"""

from __future__ import annotations

import contextlib
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
    """VLM GRPO trainer with three additions on top of ``trl.GRPOTrainer``:

    * **Per-component LR multipliers** — vision encoder trains at
      0.1× base LR by default so pretrained features aren't corrupted.
    * **Image-lift on the data path** — TRL detects multimodal inputs
      only via a top-level ``images`` column, but our schema embeds
      images inside ``prompt`` messages. Without this patch TRL falls
      through to the text-only branch and ``pixel_values`` never
      reaches the training forward pass.
    * **VLM-aware log-prob computation** — skips TRL's per-sample
      ``pixel_values`` slicing (which assumes ``(B, C, H, W)``,
      incompatible with LFM2-VL's patch-concatenated layout) and
      forwards ``spatial_shapes`` under the correct kwarg name.

    Unlike VLM SFT we do NOT subclass ``RayDataLoaderMixin``: GRPO's
    ``RepeatSampler`` must go through accelerate's per-rank
    distribution, so we use ``GRPOTrainer``'s native dataloader.
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

    def _generate_and_score_completions(self, inputs):
        """Lift images from ``prompt`` content into a top-level ``images`` key.

        TRL only detects multimodal inputs via ``inputs[0]["images"]``
        (or ``["image"]``). Our schema embeds images inside the prompt
        messages, so without this lift TRL takes the text-only branch,
        ``forward_kwargs`` comes back empty, and training runs with
        ``pixel_values=None`` — the vision tower silently detaches from
        the gradient while generation still uses real images.

        Images are loaded once as PIL here so downstream paths don't
        each re-open the file.
        """
        from PIL import Image

        for example in inputs:
            if example.get("images") is not None:
                continue
            prompt = example.get("prompt")
            if not isinstance(prompt, list):
                continue
            collected: list = []
            for message in prompt:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not (isinstance(part, dict) and part.get("type") == "image"):
                        continue
                    img = part.get("image")
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    if img is not None:
                        collected.append(img)
            if collected:
                example["images"] = collected

        with self._aliasing_spatial_shapes_as_image_sizes():
            return super()._generate_and_score_completions(inputs)

    @contextlib.contextmanager
    def _aliasing_spatial_shapes_as_image_sizes(self):
        """Rename the processor's ``spatial_shapes`` output to ``image_sizes``.

        TRL threads a fixed multimodal kwarg whitelist through the
        data pipeline (``pixel_values``, ``image_grid_thw``,
        ``pixel_attention_mask``, ``image_sizes``, ...). LFM2-VL's
        processor emits ``spatial_shapes``, which is not on the list,
        so TRL silently drops it between
        ``_generate_and_score_completions`` and ``_compute_loss``.
        Aliasing to the whitelisted ``image_sizes`` slot lets TRL
        thread it through unchanged; we rename back at the model-
        forward boundary in ``_get_per_token_logps_and_entropies``.

        Context-scoped so eval / benchmark paths (which call
        ``processor`` → ``model.generate`` directly) see the original
        processor and aren't handed an ``image_sizes=`` kwarg the
        model doesn't know.

        Python resolves ``__call__`` on ``type(obj)``, not the instance,
        so we can't patch by assigning to ``processor.__call__``. The
        documented way to rebind method resolution on a live instance
        is to swap ``obj.__class__`` into a subclass that overrides
        the method — which is what we build and cache here.
        """
        processor = getattr(self, "processing_class", None)
        if processor is None:
            # Unit-test shim: instance built via __new__, no processor.
            yield
            return
        real_cls = type(processor)

        # Cache the aliased subclass so repeated enter/exit doesn't leak types.
        cached = getattr(self, "_aliased_processor_cls", None)
        if cached is None or cached.__bases__ != (real_cls,):

            def __call__(proc_self, *args, **kwargs):  # noqa: N807
                result = real_cls.__call__(proc_self, *args, **kwargs)
                data = getattr(result, "data", None)
                if data is None and isinstance(result, dict):
                    data = result
                if (
                    data is not None
                    and "spatial_shapes" in data
                    and "image_sizes" not in data
                ):
                    data["image_sizes"] = data.pop("spatial_shapes")
                return result

            cached = type(
                f"{real_cls.__name__}WithSpatialShapesAlias",
                (real_cls,),
                {"__call__": __call__},
            )
            self._aliased_processor_cls = cached

        processor.__class__ = cached
        try:
            yield
        finally:
            processor.__class__ = real_cls

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        num_images=None,
        **forward_kwargs,
    ):
        """VLM-aware log-prob computation for LFM2-VL.

        Two things break if we reuse the base method:

        * TRL slices ``pixel_values[start:start+batch_size]`` per
          chunk, assuming ``(B, C, H, W)`` with one entry per sample.
          LFM2-VL returns patches concatenated along the first dim
          with per-image boundaries in ``spatial_shapes``, so a
          per-sample slice cuts through a single image.
        * TRL routes the tensor as ``image_sizes`` (via the alias set
          up above); LFM2-VL's ``forward`` expects ``spatial_shapes``.

        We rename back, drop kwargs the model doesn't accept (TRL
        passes a Qwen/Llava grab bag), and run one forward pass over
        the whole batch. Sufficient for LFM2-VL-450M / 1.6B on H100;
        a larger model on a tight GPU would need a VLM-aware chunker
        that reads ``spatial_shapes`` for per-image patch rows.
        """
        from trl.trainer.utils import entropy_from_logits, selective_log_softmax

        # Undo the processor-output alias at the model-forward boundary.
        if (
            "image_sizes" in forward_kwargs
            and forward_kwargs.get("image_sizes") is not None
            and "spatial_shapes" in self.model_kwarg_keys
        ):
            forward_kwargs["spatial_shapes"] = forward_kwargs.pop("image_sizes")

        model_inputs: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        for k, v in forward_kwargs.items():
            if v is None:
                continue
            if k in self.model_kwarg_keys:
                model_inputs[k] = v

        if "logits_to_keep" in self.model_kwarg_keys:
            # +1 because the last logit is dropped below (next-token pred).
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        model_inputs["use_cache"] = False

        logits = model(**model_inputs).logits
        # Drop next-token logit, keep only the completion region.
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        logits.div_(self.temperature)
        completion_ids = input_ids[:, -logits_to_keep:]
        logps = selective_log_softmax(logits, completion_ids)

        entropies = None
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)

        return logps, entropies

    def _tokenize_prompts(self, prompts: list):
        """Resolve any remaining image path strings to PIL objects.

        ``_generate_and_score_completions`` preloads images before
        super sees them, but this method is also invoked on prompts
        rebuilt by ``prepare_multimodal_messages`` where
        ``part["image"]`` may be a path string again. vLLM's
        multimodal preprocessor iterates string inputs
        character-by-character, so we must hand it real PIL objects.
        """
        from PIL import Image

        patched_prompts = []
        for prompt in prompts:
            new_prompt = []
            for message in prompt:
                content = message.get("content")
                if isinstance(content, list):
                    new_content = []
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image"
                            and isinstance(part.get("image"), str)
                        ):
                            img = Image.open(part["image"]).convert("RGB")
                            new_content.append({"type": "image", "image": img})
                        else:
                            new_content.append(part)
                    new_prompt.append({**message, "content": new_content})
                else:
                    new_prompt.append(message)
            patched_prompts.append(new_prompt)
        return super()._tokenize_prompts(patched_prompts)


def vlm_grpo_run(training_config: dict) -> None:
    """VLM GRPO training loop (Ray Train worker entrypoint)."""
    setup_worker_logging()

    # Full dataset on each worker; GRPO sampler handles per-rank distribution.
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

    # Strip VLM-specific + GRPO-excluded keys before building GRPOConfig.
    excluded_keys = VLM_GRPO_EXCLUDED_KEYS | {"leap_run_name_template"}
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

    config_kwargs = {
        "report_to": tracker,
        "run_name": job_name,
        **train_config_filtered,
    }
    training_args = GRPOConfig(**config_kwargs)

    model, processor = load_vlm_model(
        model_name,
        max_image_tokens=max_image_tokens,
        do_image_splitting=do_image_splitting,
    )
    # GRPO needs left padding; VLM processors hold the tokenizer internally.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if peft_config:
        model = apply_peft_to_model(model, peft_config)

    reward_funcs, reward_weights = resolve_reward_specs(
        training_config.get("rewards"),
        training_config.get("config_dir") or ".",
    )

    # Optional OpenEnv rollout. Deferred import so the rl-env extra stays optional.
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

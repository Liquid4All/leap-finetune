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
    create_vlm_benchmarks_from_config,
    make_eval_callback,
)
from leap_finetune.training_loops.sft_run import _get_wandb_run_id
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


_LFM2VL_PER_IMAGE_KEYS = (
    "spatial_shapes",
    "image_sizes",
    "pixel_attention_mask",
    "pixel_position_ids",
    "mm_token_type_ids",
)

# Diagnostic toggle — one log per worker so we know our patches saw real data.
_LEAP_SPLIT_LOGGED = False


def _patch_trl_split_pixel_values_for_lfm2vl() -> None:
    """Make TRL's batch splitter aware of LFM2-VL's per-image layout.

    Upstream ``trl.trainer.utils.split_pixel_values_by_grid`` assumes a
    Qwen-VL layout (``pixel_values`` is one big concat of all patches,
    sliced via ``image_grid_thw.prod(-1)``). LFM2-VL's layout is
    per-image: ``pixel_values`` shape ``(total_images, max_patches,
    embed_dim)`` paired with ``spatial_shapes``. Without a split-aware
    path here, TRL's per-rank distribution falls back to row-by-row
    slicing that cuts through multi-image samples, and the downstream
    forward fails with "Image features and image tokens do not match".

    We split every per-image tensor in lock-step using ``num_images``,
    and the matching ``unsplit_pixel_values_by_grid`` patch below
    re-merges them with ``torch.cat(..., dim=0)``. Qwen-style batches
    (``image_grid_thw`` present) are forwarded to the original. Single-
    image and tiled-image batches bail out unchanged (the patch's
    ``pv.size(0) == sum(num_images)`` guard fails for tiled inputs).
    Idempotent.
    """
    try:
        from trl.trainer import utils as _trl_utils
    except Exception:
        return
    if getattr(_trl_utils.split_pixel_values_by_grid, "_leap_lfm2vl_patched", False):
        return
    _orig = _trl_utils.split_pixel_values_by_grid

    def patched(batch):
        # Qwen-VL path is unchanged.
        if (
            "image_grid_thw" in batch
            and "pixel_values" in batch
            and "num_images" in batch
        ):
            return _orig(batch)
        if (
            "pixel_values" in batch
            and "num_images" in batch
            and ("spatial_shapes" in batch or "image_sizes" in batch)
            and isinstance(batch["pixel_values"], torch.Tensor)
        ):
            pv = batch["pixel_values"]
            num_images = batch["num_images"]
            # Tiled inputs put extra rows in pixel_values per image; bail.
            if pv.size(0) != sum(num_images):
                return batch
            split_pv = list(torch.split(pv, list(num_images), dim=0))
            new_batch = {**batch, "pixel_values": split_pv}
            # Every per-image tensor must follow the same split so they
            # stay aligned through the collator and per-rank distribution.
            for key in _LFM2VL_PER_IMAGE_KEYS:
                v = batch.get(key)
                if isinstance(v, torch.Tensor) and v.size(0) == sum(num_images):
                    new_batch[key] = list(torch.split(v, list(num_images), dim=0))
            # One-shot diagnostic to confirm split inputs are consistent
            # (only the FIRST call per worker logs).
            global _LEAP_SPLIT_LOGGED
            if not _LEAP_SPLIT_LOGGED:
                _LEAP_SPLIT_LOGGED = True
                ss = batch.get("spatial_shapes") or batch.get("image_sizes")
                pam = batch.get("pixel_attention_mask")
                print(
                    f"[vlm_grpo:split] num_images={num_images} "
                    f"pv={tuple(pv.shape)} "
                    f"ss={tuple(ss.shape) if isinstance(ss, torch.Tensor) else None} "
                    f"pam.sum(dim=1)="
                    f"{pam.sum(dim=1).tolist() if isinstance(pam, torch.Tensor) else None}",
                    flush=True,
                )
            return new_batch
        return _orig(batch)

    patched._leap_lfm2vl_patched = True  # type: ignore[attr-defined]
    _trl_utils.split_pixel_values_by_grid = patched

    # TRL's unsplit only merges pixel_values + image_grid_thw; the
    # remaining per-image tensors stay as lists after our split, which
    # crashes the vision tower. Re-merge them here.
    _orig_unsplit = _trl_utils.unsplit_pixel_values_by_grid

    def patched_unsplit(batch):
        batch = _orig_unsplit(batch)
        for key in _LFM2VL_PER_IMAGE_KEYS:
            v = batch.get(key)
            if (
                isinstance(v, list)
                and v
                and all(isinstance(t, torch.Tensor) for t in v)
            ):
                batch = {**batch, key: torch.cat(v, dim=0)}
        return batch

    patched_unsplit._leap_lfm2vl_patched = True  # type: ignore[attr-defined]
    _trl_utils.unsplit_pixel_values_by_grid = patched_unsplit

    # grpo_trainer.py imports the names directly, so rebind them there too.
    try:
        from trl.trainer import grpo_trainer as _trl_grpo

        _trl_grpo.split_pixel_values_by_grid = patched
        _trl_grpo.unsplit_pixel_values_by_grid = patched_unsplit
    except Exception:
        pass
    logger.info("[vlm_grpo] patched TRL split/unsplit_pixel_values_by_grid for LFM2-VL")


def _patch_vllm_rollout_for_multi_image(trainer: GRPOTrainer) -> None:
    """Inject ``mm_processor_kwargs`` into TRL's vLLM rollout prompts.

    TRL's ``vllm_generation`` builds rollout prompt dicts with
    ``multi_modal_data`` but doesn't pass ``mm_processor_kwargs``.
    Upstream vLLM 0.19's LFM2-VL multi-image preprocessor crashes on
    empty ``spatial_shapes`` (the same bug the async eval backend works
    around per-prompt). We wrap ``llm.generate`` and inject safe-default
    kwargs on every multi-image prompt — single-image batches are
    untouched.
    """
    if not getattr(trainer.args, "use_vllm", False):
        return
    vllm_gen = getattr(trainer, "vllm_generation", None)
    if vllm_gen is None or not hasattr(vllm_gen, "llm"):
        return

    orig_generate = vllm_gen.llm.generate
    # Keep ``use_thumbnail`` at the processor's default (True). Turning
    # it off causes a placeholder-vs-feature count mismatch against the
    # training-side HF processor — surfaces as "Image features and image
    # tokens do not match".
    _MULTI_IMAGE_KWARGS = {
        "do_image_splitting": False,
        "min_tiles": 1,
        "max_tiles": 1,
    }

    def patched_generate(prompts, sampling_params=None, use_tqdm=False, **kwargs):
        if isinstance(prompts, list):
            for p in prompts:
                if not (isinstance(p, dict) and "multi_modal_data" in p):
                    continue
                images = p["multi_modal_data"].get("image")
                if isinstance(images, list) and len(images) > 1:
                    p.setdefault("mm_processor_kwargs", dict(_MULTI_IMAGE_KWARGS))
        return orig_generate(
            prompts, sampling_params=sampling_params, use_tqdm=use_tqdm, **kwargs
        )

    vllm_gen.llm.generate = patched_generate
    logger.info("[vlm_grpo] patched vLLM rollout for multi-image prompts")


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

        # GRPO rollouts can sample the image-placeholder token by accident at
        # high temperatures. That token has no matching vision feature, so it
        # breaks ``get_placeholder_mask``'s count check
        # (placeholders > features). Sanitize the completion region by
        # replacing any image_token_id with pad_token_id; the structural
        # prompt-side placeholders (which DO have matching features) are
        # untouched.
        image_token_id = getattr(model.config, "image_token_id", None)
        has_images = forward_kwargs.get("pixel_values") is not None
        if image_token_id is not None and has_images:
            ids = model_inputs["input_ids"]
            comp_start = ids.size(1) - logits_to_keep
            comp = ids[:, comp_start:]
            mask = comp == image_token_id
            n_stray = int(mask.sum().item())
            if n_stray:
                pad_id = getattr(self.processing_class, "pad_token_id", None) or getattr(
                    getattr(self.processing_class, "tokenizer", None), "pad_token_id", 0
                )
                comp = comp.masked_fill(mask, pad_id)
                model_inputs["input_ids"] = torch.cat(
                    [ids[:, :comp_start], comp], dim=1
                )
                logger.debug(
                    "[vlm_grpo] sanitized %d stray image-token(s) in completion",
                    n_stray,
                )

        try:
            logits = model(**model_inputs).logits
        except ValueError as e:
            # When the placeholder/feature off-by-one fires, dump every
            # tensor shape + value range that could explain it, then
            # re-raise so the training run still fails (we want the data,
            # not silent skip).
            if "Image features and image tokens do not match" not in str(e):
                raise
            try:
                img_id = getattr(model.config, "image_token_id", None)
                ids = model_inputs["input_ids"]
                pv = model_inputs.get("pixel_values")
                ss = model_inputs.get("spatial_shapes")
                pam = model_inputs.get("pixel_attention_mask")
                n_ph = (
                    int((ids == img_id).sum().item()) if img_id is not None else -1
                )
                per_seq_ph = (
                    (ids == img_id).sum(dim=1).tolist() if img_id is not None else None
                )
                print(
                    "[vlm_grpo:offby1] DUMP\n"
                    f"  err={e}\n"
                    f"  num_images={num_images}\n"
                    f"  input_ids.shape={tuple(ids.shape)} placeholders_total={n_ph}\n"
                    f"  placeholders_per_seq={per_seq_ph}\n"
                    f"  pixel_values.shape={tuple(pv.shape) if isinstance(pv, torch.Tensor) else pv}\n"
                    f"  spatial_shapes.shape={tuple(ss.shape) if isinstance(ss, torch.Tensor) else ss}\n"
                    f"  spatial_shapes_values={ss.tolist() if isinstance(ss, torch.Tensor) else ss}\n"
                    f"  pam.sum(dim=1)={pam.sum(dim=1).tolist() if isinstance(pam, torch.Tensor) else pam}\n"
                    f"  hw_product_per_image="
                    f"{(ss[:, 0] * ss[:, 1]).tolist() if isinstance(ss, torch.Tensor) else None}\n"
                    f"  expected_features_per_image="
                    f"{((ss[:, 0] // 2) * (ss[:, 1] // 2)).tolist() if isinstance(ss, torch.Tensor) else None}",
                    flush=True,
                )
            except Exception as dump_err:
                print(f"[vlm_grpo:offby1] dump failed: {dump_err}", flush=True)
            raise
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

    # LFM2-VL multi-image fixes:
    #   1. TRL's per-sample splitter assumes Qwen-VL pixel_values layout;
    #      patch it to split by num_images for LFM2-VL.
    #   2. vLLM rollout preprocessor needs mm_processor_kwargs for
    #      multi-image batches. No-op for text or single-image.
    _patch_trl_split_pixel_values_for_lfm2vl()
    _patch_vllm_rollout_for_multi_image(trainer)

    trainer.add_callback(LeapCheckpointCallback(run_name_template=run_name_template))

    benchmark_configs = training_config.get("benchmark_configs")
    if benchmark_configs and benchmark_configs.get("benchmarks"):
        benchmarks = create_vlm_benchmarks_from_config(benchmark_configs, processor)
        if benchmarks:
            trainer.add_callback(
                make_eval_callback(
                    benchmarks=benchmarks,
                    async_eval_cfg=training_config.get("async_eval"),
                    benchmark_configs=benchmark_configs,
                    server_url=training_config.get("async_eval_server_url"),
                    eval_gpu_ids=training_config.get("async_eval_gpu_ids", ""),
                    output_dir=output_dir,
                    wandb_run_id=_get_wandb_run_id(),
                    config_dir=training_config.get("config_dir"),
                )
            )

    trainer = prepare_trainer(trainer)
    run_training_safely(trainer, resume_from_checkpoint=resume_from)

    if peft_config and is_rank_zero():
        merge_and_save_peft_model(
            model, processor, training_args.output_dir, run_name_template
        )

    finish_tracker(tracker)

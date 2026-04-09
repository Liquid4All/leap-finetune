"""Shared VLM optimizer helpers used by both VLM SFT and VLM GRPO trainers.

VLMs need per-component learning rates: the vision encoder trains at a lower
LR (typically 0.1×) to preserve pretrained features, while the projector and
LM backbone train at the base rate. This module holds the param-group builder
and the per-group LR logging hook that both ``LFMVLMTrainer`` (in
``training_loops/vlm_sft_run.py``) and ``LFMVLMGRPOTrainer`` (in
``training_loops/vlm_grpo_run.py``) call.

Keeping the logic here ensures there is exactly one source of truth for VLM
LR policy across SFT and GRPO.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def build_vlm_param_groups(
    model: Any,
    lr_multipliers: dict[str, float],
    base_lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build optimizer param groups with per-component learning rates.

    Each trainable parameter is assigned to the first prefix in
    ``lr_multipliers`` it matches (e.g. ``model.vision_tower``). Parameters
    that don't match any prefix go into a single "ungrouped" group at
    ``base_lr``.

    Args:
        model: The HuggingFace model whose trainable params should be grouped.
        lr_multipliers: Mapping from param-name prefix to LR multiplier.
            Insertion order matters — the first matching prefix wins. A typical
            VLM config is
            ``{"model.vision_tower": 0.1, "model.multi_modal_projector": 1.0,
               "model.language_model": 1.0}``.
        base_lr: The trainer's base learning rate. Multipliers scale this value.
        weight_decay: Weight decay to apply to every group.

    Returns:
        Tuple ``(optimizer_groups, group_names)``.

        * ``optimizer_groups`` — list of dicts in the shape expected by
          ``torch.optim.AdamW``: ``{"params": [...], "lr": ..., "weight_decay": ...}``.
        * ``group_names`` — parallel list of short human-readable names for
          each group, used by :func:`log_per_group_lrs` to emit per-component
          LRs to wandb/trackio. Matches the order of ``optimizer_groups``
          exactly.
    """
    # Seed the buckets in insertion order so param groups come out in a
    # predictable order that matches ``lr_multipliers``.
    grouped: dict[str, list] = {prefix: [] for prefix in lr_multipliers}
    ungrouped: list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for prefix in lr_multipliers:
            if name.startswith(prefix):
                grouped[prefix].append(param)
                matched = True
                break
        if not matched:
            ungrouped.append(param)

    optimizer_groups: list[dict[str, Any]] = []
    group_names: list[str] = []

    for prefix, params in grouped.items():
        if not params:
            continue
        mult = lr_multipliers[prefix]
        optimizer_groups.append(
            {"params": params, "lr": base_lr * mult, "weight_decay": weight_decay}
        )
        short_name = prefix.removeprefix("model.")
        group_names.append(short_name)
        logger.info(
            "Param group '%s': %d params, lr=%.2e",
            prefix,
            len(params),
            base_lr * mult,
        )

    if ungrouped:
        optimizer_groups.append(
            {"params": ungrouped, "lr": base_lr, "weight_decay": weight_decay}
        )
        group_names.append("ungrouped")
        logger.info(
            "Param group 'ungrouped': %d params, lr=%.2e", len(ungrouped), base_lr
        )

    return optimizer_groups, group_names


def log_per_group_lrs(
    optimizer: Any,
    group_names: Iterable[str],
    logs: dict[str, float],
) -> None:
    """Inject per-component learning rates into a ``Trainer.log()`` payload.

    Called from the trainer's ``log()`` override so wandb/trackio tracks each
    component's LR as its own plot (``lr/vision_tower``, ``lr/language_model``,
    etc.) in addition to the default single ``learning_rate`` metric.

    Mutates ``logs`` in place. Safe to call when ``optimizer`` is ``None`` (no-op).
    """
    if optimizer is None or "learning_rate" not in logs:
        return
    for name, group in zip(group_names, optimizer.param_groups):
        logs[f"lr/{name}"] = group["lr"]

import json
import logging
import os
import pathlib
import re
import shutil
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from transformers.trainer import SCALER_NAME, SCHEDULER_NAME
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# Manual-sharded MoE runs use two distinct checkpoint concepts:
# 1. resumable training checkpoints, stored as distributed shards under
#    `manual_sharded_resume/` plus trainer/RNG metadata at the checkpoint root
# 2. explicit HF model exports, produced on demand from the current model or from a
#    previously saved resumable checkpoint
MANUAL_SHARDED_FORMAT_VERSION = 2
MANUAL_SHARDED_METADATA_NAME = "manual_sharded_checkpoint.json"
MANUAL_SHARDED_RESUME_DIR = "manual_sharded_resume"
DEFAULT_EXPORT_SHARD_SIZE = 5 * 1024**3
HF_EXPORT_MAX_SHARD_SIZE = "50GB"
MANUAL_SHARDED_CHECKPOINT_FORMATS = {"sharded", "hf", "both"}
LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS = {
    "resume_only": "sharded",
    "hf_only": "hf",
}


def is_moe_model_from_name(model_name: str) -> bool:
    moe_indicators = ["8B-A1B", "8BA1B", "24B-A2B", "24BA2B", "moe", "MoE"]
    return any(indicator.lower() in model_name.lower() for indicator in moe_indicators)


def is_large_moe_model_from_name(model_name: str) -> bool:
    large_moe_indicators = ["24B-A2B", "24BA2B"]
    return any(
        indicator.lower() in model_name.lower() for indicator in large_moe_indicators
    )


def get_model_family(model_name: str) -> str:
    """Return model family for format-specific behavior."""
    if "2.5" in model_name:
        return "lfm25"
    if "24B" in model_name and "2.5" not in model_name:
        return "lfm25"
    return "lfm2"


def _manual_sharded_resume_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, MANUAL_SHARDED_RESUME_DIR)


def _manual_sharded_metadata_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, MANUAL_SHARDED_METADATA_NAME)


def _is_local_path(path: str) -> bool:
    return "://" not in path


def _global_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def normalize_manual_sharded_checkpoint_format(checkpoint_format: str) -> str:
    """Return canonical checkpoint format, accepting old config aliases."""
    if checkpoint_format in MANUAL_SHARDED_CHECKPOINT_FORMATS:
        return checkpoint_format
    if checkpoint_format in LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS:
        return LEGACY_MANUAL_SHARDED_CHECKPOINT_FORMATS[checkpoint_format]
    raise ValueError(
        f"Unsupported manual_sharded_checkpoint_format={checkpoint_format!r}. "
        f"Expected one of {sorted(MANUAL_SHARDED_CHECKPOINT_FORMATS)}."
    )


def _save_root_metadata(checkpoint_dir: str, *, save_only_model: bool) -> None:
    metadata = {
        "format_version": MANUAL_SHARDED_FORMAT_VERSION,
        "resume_dir": MANUAL_SHARDED_RESUME_DIR,
        "save_only_model": bool(save_only_model),
        "has_optimizer_state": not bool(save_only_model),
    }
    with open(_manual_sharded_metadata_path(checkpoint_dir), "w") as handle:
        json.dump(metadata, handle, indent=2)


def _load_root_metadata(checkpoint_dir: str) -> dict[str, Any]:
    metadata_path = _manual_sharded_metadata_path(checkpoint_dir)
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path) as handle:
        return json.load(handle)


def _checkpoint_staging_dir(output_dir: str, staging_root: str) -> str:
    checkpoint_name = os.path.basename(output_dir.rstrip(os.sep))
    return os.path.join(staging_root, checkpoint_name)


def _promote_directory_contents(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _update_latest_pointer(run_dir: str, checkpoint_dir: str) -> None:
    latest_link = os.path.join(run_dir, "latest")
    checkpoint_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
    if _is_local_path(run_dir):
        try:
            if os.path.lexists(latest_link):
                os.unlink(latest_link)
            os.symlink(checkpoint_name, latest_link)
            return
        except OSError:
            pass

    with open(latest_link, "w") as handle:
        handle.write(f"{checkpoint_name}\n")


def _save_scheduler_state(trainer, resume_dir: str) -> None:
    if trainer.lr_scheduler is None or _global_rank() != 0:
        return
    torch.save(trainer.lr_scheduler.state_dict(), os.path.join(resume_dir, SCHEDULER_NAME))


def _load_scheduler_state(trainer, resume_dir: str) -> None:
    scheduler_path = os.path.join(resume_dir, SCHEDULER_NAME)
    if trainer.lr_scheduler is not None and os.path.isfile(scheduler_path):
        trainer.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))


def _save_scaler_state(trainer, resume_dir: str) -> None:
    scaler = getattr(trainer.accelerator, "scaler", None)
    if scaler is None or _global_rank() != 0:
        return
    torch.save(scaler.state_dict(), os.path.join(resume_dir, SCALER_NAME))


def _load_scaler_state(trainer, resume_dir: str) -> None:
    scaler = getattr(trainer.accelerator, "scaler", None)
    scaler_path = os.path.join(resume_dir, SCALER_NAME)
    if scaler is not None and os.path.isfile(scaler_path):
        scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=True))


def _save_rng_state(trainer, checkpoint_dir: str) -> None:
    from transformers.trainer import Trainer

    Trainer._save_rng_state(trainer, checkpoint_dir)


_LFM2_MOE_GATE_UP_PROJ_RE = re.compile(
    r"^(?P<prefix>.*\.feed_forward\.experts)\.gate_up_proj$"
)
_LFM2_MOE_DOWN_PROJ_RE = re.compile(
    r"^(?P<prefix>.*\.feed_forward\.experts)\.down_proj$"
)


def _canonicalize_hf_export_state_dict(
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert training-packed LFM2 MoE tensors to the public HF checkpoint layout."""
    canonical_state_dict: dict[str, Any] = {}

    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            gate_up_match = _LFM2_MOE_GATE_UP_PROJ_RE.match(key)
            if gate_up_match is not None:
                if value.ndim != 3 or value.shape[1] % 2 != 0:
                    raise ValueError(
                        f"Expected packed gate/up tensor {key!r} to have shape "
                        "[num_experts, 2 * intermediate_size, hidden_size], got "
                        f"{tuple(value.shape)}"
                    )

                prefix = gate_up_match.group("prefix")
                gate_proj, up_proj = value.chunk(2, dim=1)
                for expert_idx in range(value.shape[0]):
                    canonical_state_dict[
                        f"{prefix}.{expert_idx}.w1.weight"
                    ] = gate_proj[expert_idx]
                    canonical_state_dict[
                        f"{prefix}.{expert_idx}.w3.weight"
                    ] = up_proj[expert_idx]
                continue

            down_match = _LFM2_MOE_DOWN_PROJ_RE.match(key)
            if down_match is not None:
                if value.ndim != 3:
                    raise ValueError(
                        f"Expected packed down tensor {key!r} to have shape "
                        "[num_experts, hidden_size, intermediate_size], got "
                        f"{tuple(value.shape)}"
                    )

                prefix = down_match.group("prefix")
                for expert_idx in range(value.shape[0]):
                    canonical_state_dict[
                        f"{prefix}.{expert_idx}.w2.weight"
                    ] = value[expert_idx]
                continue

        canonical_state_dict[key] = value

    return canonical_state_dict


def _unwrap_model_for_metadata(model: torch.nn.Module) -> torch.nn.Module:
    """Best-effort unwrap for metadata attributes on wrapped modules."""
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr in ("module", "_orig_mod"):
            wrapped = getattr(current, attr, None)
            if wrapped is not None:
                current = wrapped
                break
        else:
            return current
    return current


def _copy_custom_model_code_from_config(config: Any, export_dir: str) -> None:
    """Copy local custom-code files referenced by auto_map when available."""
    auto_map = getattr(config, "auto_map", None)
    source_dir = pathlib.Path(getattr(config, "_name_or_path", "") or "")
    if not auto_map or not source_dir.is_dir():
        return

    module_files = set()
    for target in auto_map.values():
        if isinstance(target, (list, tuple)):
            targets = target
        else:
            targets = [target]
        for item in targets:
            if isinstance(item, str) and "." in item:
                module_files.add(f"{item.split('.', 1)[0]}.py")

    for filename in module_files:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, pathlib.Path(export_dir) / filename)


def _unwrap_model_for_hf_export(model: torch.nn.Module, accelerator) -> torch.nn.Module:
    if accelerator is None:
        return _unwrap_model_for_metadata(model)
    try:
        return accelerator.unwrap_model(model, keep_torch_compile=False)
    except TypeError:
        return accelerator.unwrap_model(model)


def _save_hf_pretrained_model(
    *,
    model_to_save: torch.nn.Module,
    state_dict: dict[str, Any],
    export_dir: str,
) -> None:
    if not hasattr(model_to_save, "save_pretrained"):
        raise TypeError(
            "Manual-sharded HF export requires an unwrapped model with "
            "save_pretrained(...)."
        )

    os.makedirs(export_dir, exist_ok=True)
    model_to_save.save_pretrained(
        export_dir,
        state_dict=state_dict,
        is_main_process=True,
        max_shard_size=HF_EXPORT_MAX_SHARD_SIZE,
    )
    config = getattr(model_to_save, "config", None)
    if config is not None:
        _copy_custom_model_code_from_config(config, export_dir)


def _save_root_hf_export(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    staging_dir: str | None,
) -> None:
    model_state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    model_state_dict = _canonicalize_hf_export_state_dict(model_state_dict)

    export_dir = output_dir
    if staging_dir is not None:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)
        export_dir = staging_dir

    if _global_rank() == 0:
        model_to_save = _unwrap_model_for_hf_export(model, accelerator)
        _save_hf_pretrained_model(
            model_to_save=model_to_save,
            state_dict=model_state_dict,
            export_dir=export_dir,
        )

        if processing_class is not None:
            processing_class.save_pretrained(export_dir)
        elif (
            data_collator is not None
            and hasattr(data_collator, "tokenizer")
            and data_collator.tokenizer is not None
        ):
            data_collator.tokenizer.save_pretrained(export_dir)

        torch.save(training_args, os.path.join(export_dir, TRAINING_ARGS_NAME))

        if staging_dir is not None:
            _promote_directory_contents(staging_dir, output_dir)
            shutil.rmtree(staging_dir, ignore_errors=True)

    _world_barrier()


def save_fsdp2_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    checkpoint_staging_dir: str | None = None,
) -> None:
    """Save the HF-export layer for a manual FSDP2 model."""
    global_rank = _global_rank()
    logger.info("FSDP2 root export start rank=%s output_dir=%s", global_rank, output_dir)
    staging_dir = None
    if checkpoint_staging_dir:
        staging_dir = _checkpoint_staging_dir(output_dir, checkpoint_staging_dir)
    _save_root_hf_export(
        model=model,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        staging_dir=staging_dir,
        accelerator=accelerator,
    )
    logger.info("FSDP2 root export end rank=%s output_dir=%s", global_rank, output_dir)


def save_fsdp2_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
) -> None:
    """Backward-compatible alias for the unified manual-sharded checkpoint save."""
    save_manual_sharded_trainer_checkpoint(
        trainer=trainer,
        model=model,
        trial=trial,
        ep_group=None,
    )


def save_manual_sharded_model_export(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup | None = None,
    checkpoint_staging_dir: str | None = None,
) -> None:
    """Export the current manual-sharded model as an HF model directory."""
    if ep_group is not None:
        save_ep_model_checkpoint(
            model=model,
            accelerator=accelerator,
            output_dir=output_dir,
            processing_class=processing_class,
            data_collator=data_collator,
            training_args=training_args,
            ep_group=ep_group,
            checkpoint_staging_dir=checkpoint_staging_dir,
        )
        return

    save_fsdp2_model_checkpoint(
        model=model,
        accelerator=accelerator,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        checkpoint_staging_dir=checkpoint_staging_dir,
    )


def save_manual_sharded_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup | None = None,
) -> None:
    """Save a resumable manual-sharded training checkpoint.

    These checkpoints are optimized for distributed resume, not direct model
    consumption. Use `export_manual_sharded_checkpoint_as_hf` to convert one into a
    standalone HF model directory when needed.
    """
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )
    resume_dir = _manual_sharded_resume_dir(output_dir)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(resume_dir, exist_ok=True)
    _world_barrier()

    model_state = get_model_state_dict(model, options=options)
    resume_state: dict[str, Any] = {"model": model_state}
    if not trainer.args.save_only_model and trainer.optimizer is not None:
        resume_state["optimizer"] = get_optimizer_state_dict(
            model,
            trainer.optimizer,
            options=options,
        )

    logger.info(
        "Manual-sharded distributed checkpoint start rank=%s output_dir=%s ep=%s",
        _global_rank(),
        output_dir,
        ep_group is not None,
    )
    dcp.save(resume_state, storage_writer=FileSystemWriter(resume_dir))
    _world_barrier()

    if _global_rank() == 0:
        _save_scheduler_state(trainer, resume_dir)
        _save_scaler_state(trainer, resume_dir)
        _save_root_metadata(output_dir, save_only_model=trainer.args.save_only_model)
    _world_barrier()

    if _global_rank() == 0:
        _save_rng_state(trainer, output_dir)
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        _update_latest_pointer(run_dir, output_dir)
        if trainer.args.save_total_limit is not None and trainer.args.save_total_limit > 0:
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)
    _world_barrier()


def save_manual_sharded_hf_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup | None = None,
) -> None:
    """Save a named checkpoint directory containing only an HF model export.

    Unlike the resumable distributed checkpoint path, this does not save
    optimizer state or `manual_sharded_resume/`. It still writes trainer state
    metadata and updates the run-level `latest` pointer / rotation bookkeeping.
    """
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )

    os.makedirs(output_dir, exist_ok=True)
    _world_barrier()

    save_manual_sharded_model_export(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
        ep_group=ep_group,
        checkpoint_staging_dir=getattr(trainer, "checkpoint_staging_dir", None),
    )

    if _global_rank() == 0:
        _save_rng_state(trainer, output_dir)
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        _update_latest_pointer(run_dir, output_dir)
        if trainer.args.save_total_limit is not None and trainer.args.save_total_limit > 0:
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)
    _world_barrier()


def save_manual_sharded_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    checkpoint_format: str,
    ep_group: dist.ProcessGroup | None = None,
) -> None:
    """Save a manual-sharded checkpoint according to the configured format."""
    checkpoint_format = normalize_manual_sharded_checkpoint_format(checkpoint_format)

    if checkpoint_format == "sharded":
        save_manual_sharded_trainer_checkpoint(
            trainer=trainer,
            model=model,
            trial=trial,
            ep_group=ep_group,
        )
        return

    if checkpoint_format == "hf":
        save_manual_sharded_hf_checkpoint(
            trainer=trainer,
            model=model,
            trial=trial,
            ep_group=ep_group,
        )
        return

    # "both": write resumable state first so trainer metadata / latest pointer are
    # handled once, then layer the HF export into the same named checkpoint dir.
    save_manual_sharded_trainer_checkpoint(
        trainer=trainer,
        model=model,
        trial=trial,
        ep_group=ep_group,
    )
    output_dir = resolve_checkpoint_output_dir(
        run_dir=trainer._get_output_dir(trial=trial),
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )
    save_manual_sharded_model_export(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
        ep_group=ep_group,
        checkpoint_staging_dir=getattr(trainer, "checkpoint_staging_dir", None),
    )


def format_checkpoint_dir_name(
    run_name_template: str | None, epoch: float | None, step: int
) -> str:
    if not run_name_template:
        return f"{PREFIX_CHECKPOINT_DIR}-{step}"

    if "-" in run_name_template:
        base_part, time_part = run_name_template.rsplit("-", 1)
    else:
        base_part, time_part = run_name_template, ""

    epoch_num = int(epoch) if epoch else 0
    checkpoint_name = f"{base_part}-e{epoch_num}s{step}"
    if time_part:
        checkpoint_name += f"-{time_part}"
    return checkpoint_name


def resolve_checkpoint_output_dir(
    run_dir: str,
    run_name_template: str | None,
    epoch: float | None,
    step: int,
) -> str:
    return os.path.join(
        run_dir, format_checkpoint_dir_name(run_name_template, epoch, step)
    )


def current_checkpoint_output_dir(
    *,
    output_dir: str,
    run_name_template: str | None,
    epoch: float | None,
    step: int,
    manual_sharded: bool,
) -> str:
    """Return the expected checkpoint directory for the current save event."""
    if manual_sharded:
        return resolve_checkpoint_output_dir(
            run_dir=output_dir,
            run_name_template=run_name_template,
            epoch=epoch,
            step=step,
        )
    return os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}")


def should_run_final_manual_sharded_save(
    *,
    trainer,
    requested_save_strategy: str,
) -> bool:
    """Whether a final explicit manual-sharded save should run after training."""
    if requested_save_strategy == "no":
        return False

    output_dir = current_checkpoint_output_dir(
        output_dir=trainer._get_output_dir(trial=None),
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
        manual_sharded=True,
    )
    return not os.path.isdir(output_dir)


def load_manual_sharded_model_checkpoint(*, model: torch.nn.Module, checkpoint_dir: str) -> bool:
    """Load manual-sharded model state for resume if present."""
    resume_dir = _manual_sharded_resume_dir(checkpoint_dir)
    if not os.path.isdir(resume_dir):
        return False

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    load_state = {"model": model_state}
    dcp.load(load_state, storage_reader=FileSystemReader(resume_dir))
    set_model_state_dict(model, load_state["model"], options=options)
    _world_barrier()
    return True


def load_manual_sharded_optimizer_checkpoint(*, trainer, checkpoint_dir: str) -> bool:
    """Load optimizer / scheduler / scaler state for manual-sharded resume."""
    resume_dir = _manual_sharded_resume_dir(checkpoint_dir)
    metadata = _load_root_metadata(checkpoint_dir)
    if not os.path.isdir(resume_dir) or metadata.get("has_optimizer_state") is not True:
        return False

    if trainer.optimizer is None:
        trainer.create_optimizer()
    if trainer.lr_scheduler is None:
        trainer.create_scheduler(
            num_training_steps=trainer.args.max_steps,
            optimizer=trainer.optimizer,
        )

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    optim_state = get_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        options=options,
    )
    load_state = {"optimizer": optim_state}
    dcp.load(load_state, storage_reader=FileSystemReader(resume_dir))
    set_optimizer_state_dict(
        trainer.model,
        trainer.optimizer,
        load_state["optimizer"],
        options=options,
    )
    _world_barrier()
    _load_scheduler_state(trainer, resume_dir)
    _load_scaler_state(trainer, resume_dir)
    _world_barrier()
    return True


def export_manual_sharded_checkpoint_as_hf(
    *,
    model: torch.nn.Module,
    accelerator,
    checkpoint_dir: str,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup | None = None,
    checkpoint_staging_dir: str | None = None,
) -> None:
    """Export a resumable manual-sharded checkpoint as a standalone HF model.

    The caller provides a model constructed with the same wrapping/topology as the
    original training run. This function restores the sharded model weights from
    `checkpoint_dir` and then writes a normal HF export to `output_dir`.
    """
    loaded = load_manual_sharded_model_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
    )
    if not loaded:
        raise FileNotFoundError(
            f"No manual-sharded resume state found under {checkpoint_dir!r}"
        )

    save_manual_sharded_model_export(
        model=model,
        accelerator=accelerator,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        ep_group=ep_group,
        checkpoint_staging_dir=checkpoint_staging_dir,
    )


def rotate_named_checkpoints(output_dir: str, limit: int) -> None:
    checkpoints = []
    output_path = pathlib.Path(output_dir)
    for path in output_path.iterdir():
        if path.is_dir() and not path.name.startswith(".") and not path.is_symlink():
            match = re.search(r"-e(\d+)s(\d+)-", path.name)
            if match:
                checkpoints.append((int(match.group(2)), path))
                continue
            match = re.search(rf"{PREFIX_CHECKPOINT_DIR}-(\d+)", path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))

    checkpoints.sort(key=lambda item: item[0])
    for _, path in checkpoints[: max(0, len(checkpoints) - limit)]:
        shutil.rmtree(path)


def save_ep_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup,
    checkpoint_staging_dir: str | None = None,
) -> None:
    """Save the HF-export layer for an EP/FSDP2 model."""
    del ep_group
    global_rank = _global_rank()
    logger.info("EP root export start rank=%s output_dir=%s", global_rank, output_dir)
    staging_dir = None
    if checkpoint_staging_dir:
        staging_dir = _checkpoint_staging_dir(output_dir, checkpoint_staging_dir)
    _save_root_hf_export(
        model=model,
        output_dir=output_dir,
        processing_class=processing_class,
        data_collator=data_collator,
        training_args=training_args,
        staging_dir=staging_dir,
        accelerator=accelerator,
    )
    logger.info("EP root export end rank=%s output_dir=%s", global_rank, output_dir)


def save_ep_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup,
) -> None:
    """Backward-compatible alias for the unified manual-sharded checkpoint save."""
    save_manual_sharded_trainer_checkpoint(
        trainer=trainer,
        model=model,
        trial=trial,
        ep_group=ep_group,
    )

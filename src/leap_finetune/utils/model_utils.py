import logging
import os
import pathlib
import re
import shutil

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def is_moe_model_from_name(model_name: str) -> bool:
    moe_indicators = ["8B-A1B", "8BA1B", "24B-A2B", "24BA2B", "moe", "MoE"]
    return any(indicator.lower() in model_name.lower() for indicator in moe_indicators)


def is_large_moe_model_from_name(model_name: str) -> bool:
    large_moe_indicators = ["24B-A2B", "24BA2B"]
    return any(
        indicator.lower() in model_name.lower() for indicator in large_moe_indicators
    )


def save_fsdp2_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
) -> None:
    """Save a full HF checkpoint for a manual FSDP2 model."""
    global_rank = dist.get_rank()
    should_save = bool(getattr(training_args, "should_save", False))
    base_model = accelerator.unwrap_model(model, keep_torch_compile=False)

    logger.info(
        "FSDP2 checkpoint start rank=%s should_save=%s output_dir=%s",
        global_rank,
        should_save,
        output_dir,
    )
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    logger.info(
        "FSDP2 checkpoint rank=%s gathered state dict with %s tensors",
        global_rank,
        len(state_dict),
    )

    if not should_save:
        logger.info("FSDP2 checkpoint rank=%s returning early", global_rank)
        return

    os.makedirs(output_dir, exist_ok=True)
    consolidated_state_dict = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }
    base_model.save_pretrained(output_dir, state_dict=consolidated_state_dict)

    if processing_class is not None:
        processing_class.save_pretrained(output_dir)
    elif (
        data_collator is not None
        and hasattr(data_collator, "tokenizer")
        and data_collator.tokenizer is not None
    ):
        data_collator.tokenizer.save_pretrained(output_dir)

    torch.save(training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    logger.info("FSDP2 checkpoint rank=%s finished save_pretrained", global_rank)


def save_fsdp2_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
) -> None:
    """Save a manual FSDP2 checkpoint without entering HF's default FSDP1 path."""
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )

    save_fsdp2_model_checkpoint(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
    )

    if trainer.args.should_save:
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        if trainer.args.save_total_limit is not None and trainer.args.save_total_limit > 0:
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)


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


def _append_ep_debug_line(output_dir: str, rank: int, message: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    debug_path = os.path.join(output_dir, f"ep_save_debug_rank{rank}.log")
    with open(debug_path, "a") as handle:
        handle.write(f"{message}\n")


def _to_plain_local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Convert DTensor-like values to the local tensor shard used by this rank."""
    if hasattr(value, "to_local"):
        value = value.to_local()
    return value.detach()


def _local_packed_ep_state_dict(
    model: torch.nn.Module, packed_param_keys: set[str]
) -> dict[str, torch.Tensor]:
    local_state = model.state_dict()
    return {
        key: _to_plain_local_tensor(value)
        for key, value in local_state.items()
        if key in packed_param_keys and isinstance(value, torch.Tensor)
    }


def _packed_ep_param_keys(model: torch.nn.Module) -> set[str]:
    keys = set()
    for module_name, module in model.named_modules():
        offset = getattr(module, "_local_expert_offset", None)
        n_local = getattr(module, "_n_local_experts", None)
        if offset is None or n_local is None:
            continue
        prefix = f"{module_name}." if module_name else ""
        keys.add(f"{prefix}gate_up_proj")
        keys.add(f"{prefix}down_proj")
    return keys


def _gather_ep_sharded_tensor(
    tensor: torch.Tensor,
    ep_group: dist.ProcessGroup,
    ep_rank: int,
    ep_size: int,
) -> torch.Tensor | None:
    """Gather one EP-sharded tensor to EP rank 0 and concatenate along the expert axis."""
    device = torch.device("cuda", torch.cuda.current_device())
    local = _to_plain_local_tensor(tensor).to(device=device)
    if ep_rank == 0:
        gathered = [torch.empty_like(local) for _ in range(ep_size)]
        dist.gather(local, gather_list=gathered, group=ep_group, group_dst=0)
        return torch.cat([part.cpu() for part in gathered], dim=0)
    dist.gather(local, gather_list=None, group=ep_group, group_dst=0)
    if ep_rank != 0:
        return None
    return None


def save_ep_model_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator,
    output_dir: str,
    processing_class,
    data_collator,
    training_args,
    ep_group: dist.ProcessGroup,
) -> None:
    """Save a full HF checkpoint for EP/FSDP2 models."""
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    should_save = bool(getattr(training_args, "should_save", False))
    global_rank = dist.get_rank()
    base_model = accelerator.unwrap_model(model, keep_torch_compile=False)
    packed_param_keys = _packed_ep_param_keys(base_model)

    _append_ep_debug_line(
        output_dir,
        global_rank,
        f"save_ep_model_checkpoint:start ep_rank={ep_rank} should_save={should_save}",
    )

    logger.info(
        "EP checkpoint start rank=%s ep_rank=%s should_save=%s output_dir=%s",
        global_rank,
        ep_rank,
        should_save,
        output_dir,
    )

    logger.info("EP checkpoint rank=%s gathering full DP state dict", global_rank)
    _append_ep_debug_line(output_dir, global_rank, "before_get_model_state_dict")
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    _append_ep_debug_line(
        output_dir,
        global_rank,
        f"after_get_model_state_dict tensors={len(state_dict)}",
    )
    logger.info(
        "EP checkpoint rank=%s gathered DP state dict with %s tensors",
        global_rank,
        len(state_dict),
    )

    _append_ep_debug_line(
        output_dir,
        global_rank,
        f"packed_param_keys={len(packed_param_keys)}",
    )
    logger.info(
        "EP checkpoint rank=%s found %s packed expert tensors",
        global_rank,
        len(packed_param_keys),
    )
    local_packed_state = _local_packed_ep_state_dict(base_model, packed_param_keys)
    _append_ep_debug_line(
        output_dir,
        global_rank,
        f"local_packed_state={len(local_packed_state)}",
    )

    consolidated_state_dict = {}
    logger.info("EP checkpoint rank=%s consolidating EP-sharded tensors", global_rank)
    _append_ep_debug_line(output_dir, global_rank, "before_consolidate_ep_tensors")
    packed_keys = sorted(packed_param_keys)
    non_packed_items = state_dict.items() if should_save else ()

    for key in packed_keys:
        local_value = local_packed_state[key]
        full_value = _gather_ep_sharded_tensor(local_value, ep_group, ep_rank, ep_size)
        if should_save:
            consolidated_state_dict[key] = full_value

    for key, value in non_packed_items:
        if key not in packed_param_keys:
            consolidated_state_dict[key] = (
                value.cpu() if isinstance(value, torch.Tensor) else value
            )

    _append_ep_debug_line(output_dir, global_rank, "after_consolidate_ep_tensors")
    logger.info("EP checkpoint rank=%s finished EP tensor consolidation", global_rank)
    if not should_save:
        _append_ep_debug_line(output_dir, global_rank, "return_non_saving_rank")
        logger.info(
            "EP checkpoint rank=%s returning early ep_rank=%s should_save=%s",
            global_rank,
            ep_rank,
            should_save,
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    _append_ep_debug_line(
        output_dir,
        global_rank,
        f"before_save_pretrained tensors={len(consolidated_state_dict)}",
    )
    logger.info(
        "EP checkpoint rank=%s writing pretrained weights with %s tensors",
        global_rank,
        len(consolidated_state_dict),
    )
    base_model.save_pretrained(output_dir, state_dict=consolidated_state_dict)
    _append_ep_debug_line(output_dir, global_rank, "after_save_pretrained")
    logger.info("EP checkpoint rank=%s finished save_pretrained", global_rank)

    if processing_class is not None:
        _append_ep_debug_line(output_dir, global_rank, "before_save_processing_class")
        logger.info("EP checkpoint rank=%s saving processing class", global_rank)
        processing_class.save_pretrained(output_dir)
    elif (
        data_collator is not None
        and hasattr(data_collator, "tokenizer")
        and data_collator.tokenizer is not None
    ):
        _append_ep_debug_line(output_dir, global_rank, "before_save_collator_tokenizer")
        logger.info("EP checkpoint rank=%s saving tokenizer from data collator", global_rank)
        data_collator.tokenizer.save_pretrained(output_dir)

    torch.save(training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    _append_ep_debug_line(output_dir, global_rank, "after_save_training_args")
    logger.info("EP checkpoint rank=%s finished training args save", global_rank)


def save_ep_trainer_checkpoint(
    *,
    trainer,
    model: torch.nn.Module,
    trial,
    ep_group: dist.ProcessGroup,
) -> None:
    """Save an EP checkpoint without entering HF's default distributed save path."""
    if trainer.hp_search_backend is None and trial is None:
        trainer.store_flos()

    run_dir = trainer._get_output_dir(trial=trial)
    output_dir = resolve_checkpoint_output_dir(
        run_dir=run_dir,
        run_name_template=getattr(trainer, "run_name_template", None),
        epoch=trainer.state.epoch,
        step=trainer.state.global_step,
    )
    _append_ep_debug_line(output_dir, dist.get_rank(), "save_ep_trainer_checkpoint:start")
    logger.info(
        "EP trainer checkpoint start step=%s output_dir=%s",
        trainer.state.global_step,
        output_dir,
    )

    save_ep_model_checkpoint(
        model=model,
        accelerator=trainer.accelerator,
        output_dir=output_dir,
        processing_class=trainer.processing_class,
        data_collator=trainer.data_collator,
        training_args=trainer.args,
        ep_group=ep_group,
    )
    _append_ep_debug_line(output_dir, dist.get_rank(), "save_ep_trainer_checkpoint:after_model_save")
    logger.info("EP trainer checkpoint finished model save step=%s", trainer.state.global_step)

    if trainer.args.should_save:
        _append_ep_debug_line(output_dir, dist.get_rank(), "before_save_trainer_state")
        logger.info("EP trainer checkpoint writing trainer state step=%s", trainer.state.global_step)
        trainer.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        if trainer.args.save_total_limit is not None and trainer.args.save_total_limit > 0:
            rotate_named_checkpoints(run_dir, trainer.args.save_total_limit)
        _append_ep_debug_line(output_dir, dist.get_rank(), "after_rotate_checkpoints")
        logger.info("EP trainer checkpoint finished rotation step=%s", trainer.state.global_step)

import logging

import ray.train
import torch
import torch.distributed as dist
from ray.train.torch import get_device as get_ray_torch_device

from leap_finetune.data_loaders.ray_data_utils import ray_dataset_to_hf
from leap_finetune.utils.context_parallel import (
    apply_cp_to_model,
    create_parallel_process_groups,
    validate_cp_config,
    validate_cp_model_support,
)
from leap_finetune.utils.load_models import load_model
from leap_finetune.utils.logging_utils import init_tracker, setup_worker_logging
from leap_finetune.utils.moe_losses import set_moe_sequence_partition_group

logger = logging.getLogger(__name__)


def pin_ray_worker_cuda_device() -> None:
    """Pin this worker to Ray Train's assigned CUDA device, when present."""
    if not torch.cuda.is_available():
        return

    worker_device = get_ray_torch_device()
    if worker_device.type != "cuda":
        return

    torch.cuda.set_device(worker_device)
    logger.info("Pinned Ray worker to CUDA device %s", worker_device)


def setup_training_worker() -> None:
    """Run common per-worker setup before model/data initialization."""
    setup_worker_logging()
    pin_ray_worker_cuda_device()


def get_ray_train_eval_datasets():
    """Fetch Ray train/eval shards and convert them to HF datasets."""
    train_ds_ray = ray.train.get_dataset_shard("train")
    eval_ds_ray = ray.train.get_dataset_shard("eval")
    train_dataset = ray_dataset_to_hf(train_ds_ray)
    eval_dataset = ray_dataset_to_hf(eval_ds_ray) if eval_ds_ray is not None else None
    return train_dataset, eval_dataset


def init_tracking_from_config(
    job_name: str,
    train_config: dict,
    *,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Initialize configured tracking and return the normalized tracker name."""
    tracker = train_config.get("tracker", "none")
    if tracker == "none" and train_config.get("wandb_logging", False):
        tracker = "wandb"

    init_tracker(
        job_name,
        tracker,
        train_config.get("trackio_space_id"),
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return tracker


def default_eval_batch_size(train_config_filtered: dict) -> None:
    """Default eval batch size to train batch size when unspecified."""
    if "per_device_eval_batch_size" not in train_config_filtered:
        train_config_filtered["per_device_eval_batch_size"] = train_config_filtered.get(
            "per_device_train_batch_size", 1
        )


def load_causal_lm_for_training(
    training_config: dict,
    *,
    model_name: str,
    train_config: dict,
    **load_kwargs,
):
    """Load a causal LM with the repo-standard config/template overrides."""
    return load_model(
        model_name,
        model_config=training_config.get("model_config"),
        chat_template=train_config.get("chat_template"),
        chat_template_path=train_config.get("chat_template_path"),
        **load_kwargs,
    )


def setup_context_parallel(
    model,
    train_config: dict,
    *,
    enable_moe_sequence_partition: bool = False,
) -> dict | None:
    """Validate and apply context parallelism to a model when enabled."""
    cp_size = train_config.get("context_parallel_size", 1)
    if cp_size <= 1:
        return None

    validate_cp_model_support(model, train_config)
    world_size = (
        dist.get_world_size() if dist.is_available() and dist.is_initialized() else None
    )
    validate_cp_config(
        cp_size,
        max_length=train_config.get("max_length"),
        world_size=world_size,
    )
    cp_config = create_parallel_process_groups(cp_size)
    apply_cp_to_model(model, cp_config)
    if enable_moe_sequence_partition:
        set_moe_sequence_partition_group(model, cp_config["cp_group"])
    logger.info("Context parallelism enabled with cp_size=%s", cp_size)
    return cp_config

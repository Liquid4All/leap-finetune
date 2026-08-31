import os

import torch
from accelerate.utils import set_seed

from leap_finetune.checkpointing.model_info import is_moe_model_from_name
from leap_finetune.checkpointing.model_loading import load_tokenizer
from leap_finetune.data_loading.dataset_loader import DatasetLoader
from leap_finetune.data_loading.local_data import create_local_datasets
from leap_finetune.distribution.distributed_configs import (
    strip_distributed_training_config,
)
from leap_finetune.training import TRAINING_LOOPS

_LOCAL_TYPES = frozenset({"sft", "dpo", "vlm_sft"})


def should_use_local(job_config: dict) -> bool:
    """Return whether this job can use the one-process Trainer path."""
    if os.getenv("LEAP_LAUNCHER", "auto").lower() == "ray":
        return False
    training_type = job_config["training_type"]
    if training_type not in _LOCAL_TYPES:
        return False
    if is_moe_model_from_name(job_config["model_name"]):
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return False
    ray_config = job_config.get("ray_config") or {}
    if ray_config.get("address") or os.getenv("RAY_ADDRESS"):
        return False
    if int(os.getenv("LEAP_NUM_WORKERS", "1")) != 1:
        return False
    return int(ray_config.get("num_workers", 1) or 1) == 1


def local_trainer(job_config: dict):
    """Run SFT, DPO, or VLM SFT in the current process without Ray Train."""
    set_seed(42)
    training_type = job_config["training_type"]
    train_config = strip_distributed_training_config(
        job_config["training_config"], num_workers=1
    )
    dataset_config = job_config["dataset"]
    if not isinstance(dataset_config, DatasetLoader):
        raise ValueError("Local training requires a DatasetLoader")

    tokenizer = None
    if training_type in {"sft", "dpo"}:
        tokenizer = load_tokenizer(
            job_config["model_name"],
            chat_template=train_config.get("chat_template"),
            chat_template_path=train_config.get("chat_template_path"),
        )
    train_dataset, eval_dataset = create_local_datasets(
        dataset_config,
        tokenizer=tokenizer,
        training_config=train_config,
    )
    loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": train_config,
        "peft_config": job_config.get("peft_config"),
        "model_config": job_config.get("model_config"),
        "benchmark_configs": job_config.get("benchmark_configs"),
        "rewards": job_config.get("rewards"),
        "async_eval": job_config.get("async_eval"),
        "config_dir": job_config.get("config_dir"),
    }
    print("\nTraining locally on 1 GPU without Ray Train")
    return TRAINING_LOOPS[training_type](
        loop_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

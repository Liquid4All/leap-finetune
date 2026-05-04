import os
import shutil
import psutil

import ray
import ray.data
from accelerate.utils import set_seed
from torch import cuda
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv
from ray.train.torch import TorchTrainer, TorchConfig

from leap_finetune.data_loaders.dataset_loader import DatasetLoader
from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets
from leap_finetune.training_loops import TRAINING_LOOPS
from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.utils.load_models import load_tokenizer
from leap_finetune.utils.model_utils import is_moe_model_from_name
from leap_finetune.utils.ray_data_config import (
    ContextParallelDataConfig,
    ExpertParallelDataConfig,
)
from leap_finetune.utils.logging_utils import worker_process_setup_hook
from leap_finetune.utils.logging_utils import (
    get_requested_ray_address,
    get_ray_env_vars,
    print_next_steps_panel,
    select_ray_temp_dir,
    select_object_spilling_dir,
    should_connect_existing_cluster,
)


#################################
#         Ray Trainer           #
#################################


def _resolve_num_workers(
    ray_config: dict | None,
    *,
    local_num_gpus: int,
    connected_to_existing_cluster: bool,
) -> int:
    """Resolve the Ray worker count for this training job."""
    for key in ("LEAP_RAY_NUM_WORKERS", "LEAP_NUM_WORKERS"):
        raw = os.environ.get(key)
        if raw:
            return int(raw)

    if ray_config and ray_config.get("num_workers") is not None:
        return int(ray_config["num_workers"])

    if connected_to_existing_cluster:
        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        if cluster_gpus < 1:
            raise ValueError(
                "Connected to external Ray cluster but no GPU resources were detected. "
                "Set LEAP_RAY_NUM_WORKERS if cluster resources are still registering."
            )
        return cluster_gpus

    return local_num_gpus


def _build_scaling_config(ray_config: dict | None, *, num_workers: int) -> ScalingConfig:
    resources_per_worker = {"GPU": 1.0}
    if ray_config and isinstance(ray_config.get("resources_per_worker"), dict):
        resources_per_worker = dict(ray_config["resources_per_worker"])
        resources_per_worker.setdefault("GPU", 1.0)

    return ScalingConfig(
        num_workers=num_workers,
        use_gpu=resources_per_worker.get("GPU", 0) > 0,
        resources_per_worker=resources_per_worker,
    )


def _resolve_local_object_store_memory() -> int:
    """Choose a local Ray object store size that respects tiny /dev/shm setups."""
    available_mem = int(psutil.virtual_memory().available * 0.4)
    default_cap = 8 * 1024**3
    target = min(available_mem, default_cap)

    try:
        shm_total = shutil.disk_usage("/dev/shm").total
    except OSError:
        shm_total = 0

    # If shared memory is healthy, keep using it conservatively.
    if shm_total >= 1 * 1024**3:
        return min(target, int(shm_total * 0.8))

    # Tiny /dev/shm usually means containerized or restricted environments.
    # Allow slow storage and keep the object store modest.
    os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
    return min(target, 2 * 1024**3)


def ray_trainer(job_config: dict) -> None:
    """
    Runs on each Ray worker after loading config, setting seed, and calling a training loop
    """

    training_type = job_config["training_type"]

    # Auto-route MoE models to dedicated training loops
    if training_type in ("sft", "dpo") and is_moe_model_from_name(
        job_config["model_name"]
    ):
        training_type = f"moe_{training_type}"

    output_dir = job_config["training_config"]["output_dir"]

    set_seed(42)

    ray_config = job_config.get("ray_config")
    ray_address = get_requested_ray_address(ray_config)
    connect_existing_cluster = should_connect_existing_cluster(ray_config)
    local_num_gpus = cuda.device_count()

    if not ray.is_initialized():
        ray_temp_dir = select_ray_temp_dir(os.path.expanduser("~/tmp-ray"))
        runtime_env = RuntimeEnv(
            working_dir=str(RUNTIME_DIR),
            env_vars=get_ray_env_vars(ray_temp_dir),
            worker_process_setup_hook=worker_process_setup_hook,
        )

        if connect_existing_cluster:
            ray.init(address=ray_address or "auto", runtime_env=runtime_env)
        else:
            if not cuda.is_available():
                raise ValueError(
                    "No local GPU available for training. "
                    "Either run on a GPU node or connect to an existing Ray cluster via RAY_ADDRESS."
                )

            spill_dir = select_object_spilling_dir(ray_temp_dir)

            object_store_mem = _resolve_local_object_store_memory()

            ray.init(
                address="local",
                runtime_env=runtime_env,
                _temp_dir=ray_temp_dir,
                object_spilling_directory=spill_dir,
                object_store_memory=object_store_mem,
            )

        # Also suppress on driver (must be after ray.init)
        worker_process_setup_hook()

        # Disable progress bar name truncation warning
        ray.data.DataContext.get_current().enable_progress_bar_name_truncation = False

    num_workers = _resolve_num_workers(
        ray_config,
        local_num_gpus=local_num_gpus,
        connected_to_existing_cluster=connect_existing_cluster,
    )
    if num_workers < 1:
        raise ValueError("No GPU workers available for Ray training")

    train_loop = TRAINING_LOOPS.get(training_type)
    if train_loop is None:
        raise ValueError(
            f"Invalid training type: {training_type}. "
            f"Available: {list(TRAINING_LOOPS.keys())}"
        )

    # Prepare datasets using Ray Data
    dataset_config = job_config["dataset"]
    training_config = job_config["training_config"]

    # Load tokenizer on driver for pre-tokenization (lightweight, no model weights)
    tokenizer = load_tokenizer(
        job_config["model_name"],
        chat_template=training_config.get("chat_template"),
        chat_template_path=training_config.get("chat_template_path"),
    )

    if isinstance(dataset_config, DatasetLoader):
        # Pre-tokenize SFT and DPO on driver; VLM passes through
        use_pretokenize = training_type in (
            "sft",
            "dpo",
            "moe_sft",
            "moe_dpo",
        )
        train_ds, eval_ds = create_ray_datasets(
            dataset_config,
            tokenizer=tokenizer if use_pretokenize else None,
            training_config=training_config if use_pretokenize else None,
        )
        datasets = {"train": train_ds}
        if eval_ds is not None:
            datasets["eval"] = eval_ds
    elif isinstance(dataset_config, tuple):
        # Legacy path: pre-loaded (Dataset, Dataset) tuple (deprecate eventually)
        train_hf, eval_hf = dataset_config
        train_ds = ray.data.from_huggingface(train_hf)
        datasets = {"train": train_ds}
        if eval_hf is not None:
            datasets["eval"] = ray.data.from_huggingface(eval_hf)
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset_config)}")

    # Training config
    train_loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": training_config,
        "peft_config": job_config["peft_config"],
        "model_config": job_config.get("model_config"),
        "benchmark_configs": job_config.get("benchmark_configs"),
    }

    moe_training = training_config.get("moe_training", {})
    ep_size = moe_training.get("expert_parallel_size", 1) or 1
    cp_size = training_config.get("context_parallel_size", 1) or 1
    dataset_config = None
    if cp_size > 1:
        dataset_config = ContextParallelDataConfig(context_parallel_size=cp_size)
    elif training_type in ("moe_sft", "moe_dpo") and ep_size > 1:
        dataset_config = ExpertParallelDataConfig(expert_parallel_size=ep_size)

    scale_config = _build_scaling_config(ray_config, num_workers=num_workers)

    run_config = RunConfig(
        storage_path=output_dir,
        name="ray_logs",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )

    if connect_existing_cluster:
        print(
            f"\nTraining on {num_workers} Ray workers "
            f"(RAY_ADDRESS={ray_address or 'auto'})"
        )
    else:
        print(f"\nTraining on {num_workers} GPUs")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scale_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="nccl", timeout_s=7200),
        datasets=datasets,
        dataset_config=dataset_config,
    )

    result = trainer.fit()

    print_next_steps_panel(output_dir)
    # Ensure Ray cleans up resources promptly to avoid post-training hangs
    try:
        ray.shutdown()
    except Exception:
        pass

    return result

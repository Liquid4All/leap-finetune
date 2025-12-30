import os
import psutil

import ray
import ray.data
from accelerate.utils import set_seed
from ray.train import RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv
from ray.train.torch import TorchTrainer, TorchConfig
from torch import cuda

from leap_finetune.data_loaders.dataset_loader import DatasetLoader
from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets
from leap_finetune.training_loops import TRAINING_LOOPS
from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.utils.logging_utils import worker_process_setup_hook
from leap_finetune.utils.logging_utils import (
    get_ray_env_vars,
    print_next_steps_panel,
    select_ray_temp_dir,
    select_object_spilling_dir,
)


#################################
#         Ray Trainer           #
#################################


def ray_trainer(job_config: dict) -> None:
    """
    Runs on each Ray worker after loading config, setting seed, and calling a training loop
    """

    training_type = job_config["training_type"]
    output_dir = job_config["training_config"]["output_dir"]

    set_seed(42)
    num_gpus = cuda.device_count()

    if not cuda.is_available():
        raise ValueError("No GPU available for training")

    ray_temp_dir = os.path.expanduser("~/ray_temp")
    os.makedirs(ray_temp_dir, exist_ok=True)

    if not ray.is_initialized():
        # Force local init and avoid accidental cluster connects
        for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_HEAD_NODE_ADDRESS", "RAY_PORT"):
            os.environ.pop(key, None)

        ray_temp_dir = select_ray_temp_dir(os.path.expanduser("~/ray_temp"))
        spill_dir = select_object_spilling_dir(ray_temp_dir)

        runtime_env = RuntimeEnv(
            working_dir=str(RUNTIME_DIR),
            env_vars=get_ray_env_vars(ray_temp_dir),
            worker_process_setup_hook=worker_process_setup_hook,
        )

        # Calculate system memory for object store
        object_store_mem = int(psutil.virtual_memory().total * 0.4)

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

    train_loop = TRAINING_LOOPS.get(training_type)
    if train_loop is None:
        raise ValueError(
            f"Invalid training type: {training_type}. "
            f"Available: {list(TRAINING_LOOPS.keys())}"
        )

    # Prepare datasets using Ray Data
    dataset_config = job_config["dataset"]

    if isinstance(dataset_config, DatasetLoader):
        # New Ray Data path - distributed loading
        train_ds, eval_ds = create_ray_datasets(dataset_config)
        datasets = {"train": train_ds, "eval": eval_ds}
    elif isinstance(dataset_config, tuple):
        # Legacy path: pre-loaded (Dataset, Dataset) tuple (deprecate eventually)
        train_hf, eval_hf = dataset_config
        train_ds = ray.data.from_huggingface(train_hf)
        eval_ds = ray.data.from_huggingface(eval_hf)
        datasets = {"train": train_ds, "eval": eval_ds}
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset_config)}")

    # Training config
    train_loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": job_config["training_config"],
        "peft_config": job_config["peft_config"],
    }

    scale_config = ScalingConfig(
        num_workers=num_gpus, use_gpu=True, resources_per_worker={"GPU": 1.0}
    )

    run_config = RunConfig(
        storage_path=output_dir,
        name="ray_logs",
    )

    print(f"\nTraining on {num_gpus} GPUs")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scale_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="nccl"),
        datasets=datasets,  # Ray Data handles sharding
    )

    trainer.fit()

    print_next_steps_panel(output_dir)
    # Ensure Ray cleans up resources promptly to avoid post-training hangs
    try:
        ray.shutdown()
    except Exception:
        pass

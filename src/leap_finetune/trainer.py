import ray
import os

from accelerate.utils import set_seed
from ray.train import RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv
from ray.train.torch import TorchTrainer, TorchConfig
from torch import cuda

from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.training_loops.sft_run import sft_run
from leap_finetune.training_loops.dpo_run import dpo_run
from leap_finetune.training_loops.vlm_sft_run import vlm_sft_run
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
        )

        ray.init(
            address="local",
            runtime_env=runtime_env,
            _temp_dir=ray_temp_dir,
            object_spilling_directory=spill_dir,
        )

    if training_type == "sft":
        train_loop = sft_run
    elif training_type == "dpo":
        train_loop = dpo_run
    elif training_type == "vlm_sft":
        train_loop = vlm_sft_run
    else:
        raise ValueError(f"Invalid training type: {training_type}")

    train_loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": job_config["training_config"],
        "peft_config": job_config["peft_config"],
        "dataset": job_config["dataset"],
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
    )

    trainer.fit()

    print_next_steps_panel(output_dir)
    # Ensure Ray cleans up resources promptly to avoid post-training hangs
    try:
        ray.shutdown()
    except Exception:
        pass

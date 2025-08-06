import ray
import os

from accelerate.utils import set_seed
from ray.train import RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv
from ray.train.torch import TorchTrainer
from torch import cuda

from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.utils.output_paths import resolve_model_output_path
from leap_finetune.training_loops.sft_run import sft_run
from leap_finetune.training_loops.dpo_run import dpo_run


#################################
#         Ray Trainer           #
#################################


def ray_trainer(job_config: dict) -> None:
    """
    Runs on each Ray worker after loading config, setting seed, and calling a training loop
    """

    job_name = job_config["job_name"]
    training_type = job_config["training_type"]

    set_seed(42)
    num_gpus = cuda.device_count()

    if not cuda.is_available():
        raise ValueError("No GPU available for training")

    if not ray.is_initialized():
        ray_temp_dir = os.path.expanduser("~/ray_temp")
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(
            runtime_env=RuntimeEnv(working_dir=str(RUNTIME_DIR)), _temp_dir=ray_temp_dir
        )

    train_loop = sft_run if training_type == "sft" else dpo_run
    output_dir = resolve_model_output_path(training_type, job_name)

    job_config["training_config"]["output_dir"] = str(output_dir)

    train_loop_config = {
        "model_name": job_config["model_name"],
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
    )

    trainer.fit()

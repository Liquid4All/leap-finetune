import ray
import os

from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig
from rich.console import Console
from rich.panel import Panel
from torch import cuda

from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.training_loops.sft_run import sft_run
from leap_finetune.training_loops.dpo_run import dpo_run
from leap_finetune.training_loops.vlm_sft_run import vlm_sft_run
from leap_finetune.training_loops.sft_weave_run import sft_weave_run


#################################
#         Ray Trainer           #
#################################


def ray_trainer(job_config: dict) -> None:
    """
    Runs on each Ray worker after loading config, setting seed, and calling a training loop
    """

    training_type = job_config["training_type"]
    output_dir = job_config["training_config"]["output_dir"]

    from accelerate.utils import set_seed

    set_seed(42)
    num_gpus = cuda.device_count()

    if not cuda.is_available():
        raise ValueError("No GPU available for training")

    if not ray.is_initialized():
        from ray.runtime_env import RuntimeEnv

        ray_temp_dir = os.path.expanduser("~/ray_temp")
        os.makedirs(ray_temp_dir, exist_ok=True)
        ray.init(
            runtime_env=RuntimeEnv(
                working_dir=str(RUNTIME_DIR),
                env_vars={
                    "TMPDIR": ray_temp_dir,
                    "TEMP": ray_temp_dir,
                    "TMP": ray_temp_dir,
                    "NCCL_IB_DISABLE": "1",
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                    "NCCL_SOCKET_IFNAME": "lo",
                    "TORCH_NCCL_BLOCKING_WAIT": "1",
                    "NCCL_TIMEOUT": "300",  # 5 minute safe timeout
                    "RAY_DISABLE_IMPORT_WARNING": "1",
                    "RAY_memory_monitor_refresh_ms": "0",
                    # Suppress Ray Data verbose logging
                    "RAY_DATA_DISABLE_PROGRESS_BARS": "1",
                    "RAY_IGNORE_UNHANDLED_ERRORS": "1",
                    "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
                },
            ),
            _temp_dir=ray_temp_dir,
            num_gpus=num_gpus,
        )

    if training_type == "sft":
        train_loop = sft_run
    elif training_type == "sft_weave":
        train_loop = sft_weave_run
    elif training_type == "dpo":
        train_loop = dpo_run
    elif training_type == "vlm_sft":
        train_loop = vlm_sft_run
    else:
        raise ValueError(f"Invalid training type: {training_type}")

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
        torch_config=TorchConfig(backend="nccl"),
    )

    trainer.fit()

    console = Console()
    quick_start_url = (
        "https://leap.liquid.ai/docs/leap-bundle/quick-start?utm_source=github"
        "&utm_medium=link&utm_campaign=LEAP&utm_content=general"
    )

    cta_message = (
        "[bold green]Training complete![/bold green]\n\n"
        f"[bold]Checkpoint directory:[/bold] [cyan]{output_dir}[/cyan]\n\n"
        "Bundle your output checkpoint with [bold]leap-bundle[/bold] to use it in LEAP:\n"
        f"[dim]leap-bundle create {output_dir}/[CHECKPOINT_NAME][/dim]\n\n"
        f"Quick Start: [link={quick_start_url}]{quick_start_url}[/link]"
    )

    console.print(
        Panel.fit(
            cta_message,
            title="Next Step: Bundle for LEAP",
            border_style="green",
        )
    )

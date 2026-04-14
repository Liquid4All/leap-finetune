import os
import psutil

import ray
import ray.data
from accelerate.utils import set_seed
from torch import cuda
from ray.train import CheckpointConfig, DataConfig, RunConfig, ScalingConfig
from ray.runtime_env import RuntimeEnv
from ray.train.torch import TorchTrainer, TorchConfig

from leap_finetune.data_loaders.dataset_loader import DatasetLoader
from leap_finetune.data_loaders.ray_data_utils import create_ray_datasets
from leap_finetune.training_loops import TRAINING_LOOPS
from leap_finetune.utils.constants import RUNTIME_DIR
from leap_finetune.utils.load_models import _resolve_model_id, load_tokenizer
from leap_finetune.utils.logging_utils import worker_process_setup_hook
from leap_finetune.utils.logging_utils import (
    get_ray_env_vars,
    print_next_steps_panel,
    select_ray_temp_dir,
    select_object_spilling_dir,
)
from leap_finetune.utils.vllm_server import (
    launch_vllm_server,
    plan_gpu_split,
    resolve_server_host,
)


#################################
#         Ray Trainer           #
#################################


def ray_trainer(job_config: dict) -> None:
    """Entry point: init Ray, build datasets, launch the TorchTrainer."""

    training_type = job_config["training_type"]
    output_dir = job_config["training_config"]["output_dir"]

    set_seed(42)

    num_gpus = cuda.device_count()
    if not cuda.is_available():
        raise ValueError("No GPU available for training")

    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    is_multi_node = bool(ray_address)

    if not ray.is_initialized():
        if is_multi_node:
            runtime_env = RuntimeEnv(
                working_dir=str(RUNTIME_DIR),
                env_vars=get_ray_env_vars(ray_temp_dir=None, multi_node=True),
                worker_process_setup_hook=worker_process_setup_hook,
            )
            ray.init(
                address=ray_address,
                runtime_env=runtime_env,
                ignore_reinit_error=True,
            )
            print(f"\nConnected to multi-node Ray cluster at {ray_address}")
        else:
            for key in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_HEAD_NODE_ADDRESS", "RAY_PORT"):
                os.environ.pop(key, None)

            ray_temp_dir = select_ray_temp_dir(os.path.expanduser("~/ray_temp"))
            spill_dir = select_object_spilling_dir(ray_temp_dir)

            runtime_env = RuntimeEnv(
                working_dir=str(RUNTIME_DIR),
                env_vars=get_ray_env_vars(ray_temp_dir),
                worker_process_setup_hook=worker_process_setup_hook,
            )

            # 40% of available RAM, not total, to avoid OOM on shared nodes.
            object_store_mem = int(psutil.virtual_memory().available * 0.4)

            ray.init(
                address="local",
                runtime_env=runtime_env,
                _temp_dir=ray_temp_dir,
                object_spilling_directory=spill_dir,
                object_store_memory=object_store_mem,
            )

        worker_process_setup_hook()
        ray.data.DataContext.get_current().enable_progress_bar_name_truncation = False

    train_loop = TRAINING_LOOPS.get(training_type)
    if train_loop is None:
        raise ValueError(
            f"Invalid training type: {training_type}. "
            f"Available: {list(TRAINING_LOOPS.keys())}"
        )

    dataset_config = job_config["dataset"]
    training_config = job_config["training_config"]

    tokenizer = load_tokenizer(job_config["model_name"])

    if isinstance(dataset_config, DatasetLoader):
        # SFT and DPO pre-tokenize on the driver; VLM-SFT and GRPO variants
        # need raw rows (images / online generation).
        use_pretokenize = training_type in ("sft", "dpo")
        train_ds, eval_ds = create_ray_datasets(
            dataset_config,
            tokenizer=tokenizer if use_pretokenize else None,
            training_config=training_config if use_pretokenize else None,
        )
        datasets = {"train": train_ds, "eval": eval_ds}
    elif isinstance(dataset_config, tuple):
        train_hf, eval_hf = dataset_config
        train_ds = ray.data.from_huggingface(train_hf)
        eval_ds = ray.data.from_huggingface(eval_hf)
        datasets = {"train": train_ds, "eval": eval_ds}
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset_config)}")

    train_loop_config = {
        "model_name": job_config["model_name"],
        "job_name": job_config.get("job_name", "leap-ft-run"),
        "train_config": training_config,
        "peft_config": job_config["peft_config"],
        "benchmark_configs": job_config.get("benchmark_configs"),
        "rewards": job_config.get("rewards"),
        "rl_env": job_config.get("rl_env"),
        "grpo_rollout": job_config.get("grpo_rollout"),
        "config_dir": job_config.get("config_dir"),
    }

    is_grpo = training_type in ("grpo", "vlm_grpo")
    training_num_gpus = int(ray.cluster_resources().get("GPU", num_gpus))
    grpo_rollout_cfg = job_config.get("grpo_rollout") or {}
    vllm_mode = training_config.get("vllm_mode", "colocate")

    if is_grpo and vllm_mode == "server" and grpo_rollout_cfg.get("dedicated_gpus"):
        if is_multi_node:
            raise NotImplementedError(
                "vLLM server mode + dedicated_gpus is single-node only. "
                "Use vllm_mode: colocate for multi-node GRPO."
            )
        vllm_gpus, train_gpus = plan_gpu_split(num_gpus, grpo_rollout_cfg)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)
        training_num_gpus = len(train_gpus)

        host = resolve_server_host(training_config.get("vllm_server_host"))
        port = int(training_config.get("vllm_server_port", 8000))
        model_id = _resolve_model_id(job_config["model_name"])

        server_handle = launch_vllm_server(
            model_id=model_id,
            vllm_gpu_ids=vllm_gpus,
            grpo_rollout_cfg=grpo_rollout_cfg,
            host="0.0.0.0",
            port=port,
        )
        training_config["vllm_server_base_url"] = f"http://{host}:{port}"
        training_config["vllm_server_host"] = host
        training_config["vllm_server_port"] = port
        print(
            f"[GRPO] vLLM server on GPU(s) {vllm_gpus} at "
            f"{training_config['vllm_server_base_url']}; training on {train_gpus}"
        )
        del server_handle

    scale_config = ScalingConfig(
        num_workers=training_num_gpus, use_gpu=True, resources_per_worker={"GPU": 1.0}
    )

    # GRPO: each worker needs the full dataset — TRL's RepeatSampler
    # handles per-rank striding, so we bypass Ray's default split.
    dataset_config_kwargs = {}
    if is_grpo:
        dataset_config_kwargs["dataset_config"] = DataConfig(datasets_to_split=[])

    run_config = RunConfig(
        storage_path=output_dir,
        name="ray_logs",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    )

    if is_multi_node:
        num_nodes = len([n for n in ray.nodes() if n.get("Alive", False)])
        print(f"\nTraining on {training_num_gpus} GPUs across {num_nodes} nodes")
    else:
        print(f"\nTraining on {training_num_gpus} GPUs")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=train_loop_config,
        scaling_config=scale_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="nccl"),
        datasets=datasets,
        **dataset_config_kwargs,
    )

    result = trainer.fit()

    print_next_steps_panel(output_dir)
    try:
        ray.shutdown()
    except Exception:
        pass

    return result

import os
from pathlib import Path

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
from leap_finetune.rl.judge import (
    build_judge_runtime_config,
    export_judge_runtime_config,
    get_judge_config,
    judge_needs_local_server,
)
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
    resolve_server_host,
    resolve_vllm_rollout_plan,
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
    training_config = job_config["training_config"]
    is_grpo = training_type in ("grpo", "vlm_grpo")
    grpo_rollout_cfg = job_config.get("grpo_rollout") or {}
    vllm_mode = training_config.get("vllm_mode", "colocate")
    rewards_cfg = job_config.get("rewards")
    judge_cfg = get_judge_config(rewards_cfg)
    reserve_judge = judge_needs_local_server(rewards_cfg)
    rollout_plan = resolve_vllm_rollout_plan(
        num_gpus,
        grpo_rollout_cfg,
        vllm_mode=vllm_mode,
        is_multi_node=is_multi_node,
        reserve_judge=reserve_judge,
    )

    if is_grpo and rollout_plan.uses_custom_training_visibility:
        if ray.is_initialized():
            raise RuntimeError(
                "GRPO vLLM GPU splitting must be resolved before Ray starts. "
                "Run this job in a fresh process or use an externally managed "
                "vLLM server."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = rollout_plan.training_cuda_visible_devices

    server_handles = []
    if is_grpo and rollout_plan.launches_local_server:
        host = resolve_server_host(training_config.get("vllm_server_host"))
        port = int(training_config.get("vllm_server_port", 8000))
        model_id = _resolve_model_id(job_config["model_name"])

        server_handle = launch_vllm_server(
            model_id=model_id,
            vllm_gpu_ids=rollout_plan.server_gpu_ids,
            grpo_rollout_cfg=grpo_rollout_cfg,
            host="0.0.0.0",
            port=port,
            vllm_cuda_visible_devices=rollout_plan.server_cuda_visible_devices,
            log_path=Path(output_dir) / "vllm_server.log",
        )
        server_handles.append(server_handle)
        training_config["vllm_server_base_url"] = f"http://{host}:{port}"
        training_config["vllm_server_host"] = host
        training_config["vllm_server_port"] = port
        print(
            "[GRPO] vLLM server using "
            f"{len(rollout_plan.server_gpu_ids)} GPU(s) at "
            f"{training_config['vllm_server_base_url']}; "
            f"Ray training using {rollout_plan.num_training_workers} GPU(s)"
        )

    if is_grpo and judge_cfg is not None:
        model_id = _resolve_model_id(judge_cfg.get("model") or job_config["model_name"])
        judge_base_url = judge_cfg.get("base_url")

        if rollout_plan.launches_local_judge:
            host = resolve_server_host(judge_cfg.get("host"))
            port = int(judge_cfg.get("port", 8001))
            judge_server_cfg = _judge_server_config(
                grpo_rollout_cfg,
                judge_cfg,
                len(rollout_plan.judge_gpu_ids),
            )
            judge_handle = launch_vllm_server(
                model_id=model_id,
                vllm_gpu_ids=rollout_plan.judge_gpu_ids,
                grpo_rollout_cfg=judge_server_cfg,
                host="0.0.0.0",
                port=port,
                vllm_cuda_visible_devices=rollout_plan.judge_cuda_visible_devices,
                log_path=Path(output_dir) / "judge_vllm_server.log",
            )
            server_handles.append(judge_handle)
            judge_base_url = f"http://{host}:{port}"
            print(
                "[GRPO] Judge LLM server using "
                f"{len(rollout_plan.judge_gpu_ids)} GPU(s) at {judge_base_url}"
            )

        judge_runtime_config = build_judge_runtime_config(
            rewards_cfg,
            default_model=model_id,
            base_url=judge_base_url,
        )
        export_judge_runtime_config(judge_runtime_config)
    else:
        export_judge_runtime_config(None)

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
            for key in (
                "RAY_ADDRESS",
                "RAY_HEAD_IP",
                "RAY_HEAD_NODE_ADDRESS",
                "RAY_PORT",
            ):
                os.environ.pop(key, None)

            ray_temp_dir = select_ray_temp_dir(os.path.expanduser("~/ray_temp"))
            spill_dir = select_object_spilling_dir(ray_temp_dir)

            runtime_env = RuntimeEnv(
                working_dir=str(RUNTIME_DIR),
                env_vars=get_ray_env_vars(ray_temp_dir),
                worker_process_setup_hook=worker_process_setup_hook,
            )

            object_store_mem = _ray_object_store_memory()

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
        "rewards": rewards_cfg,
        "async_eval": job_config.get("async_eval"),
        "rl_env": job_config.get("rl_env"),
        "grpo_rollout": job_config.get("grpo_rollout"),
        "config_dir": job_config.get("config_dir"),
    }

    training_num_gpus = int(ray.cluster_resources().get("GPU", num_gpus))
    resources_per_worker = {"GPU": 1.0}
    if is_grpo and not is_multi_node:
        training_num_gpus = rollout_plan.num_training_workers
        resources_per_worker = rollout_plan.resources_per_worker

    # Async-eval reserved mode: carve eval GPUs off the training pool. We
    # only do the GPU split here; the worker (rank 0) launches its own
    # vLLM subprocess so it owns the lifetime and can respawn on weight
    # reload. Runs AFTER any GRPO carve above so both can coexist.
    async_eval_cfg = job_config.get("async_eval") or {}
    if async_eval_cfg.get("mode") == "reserved":
        if is_multi_node:
            raise NotImplementedError(
                "async_eval mode=reserved is single-node only. "
                "Use mode=sidecar for multi-node training."
            )
        eval_gpu_count = int(async_eval_cfg.get("vllm_gpus", 1))
        current_train_gpus = list(range(training_num_gpus))
        if eval_gpu_count >= len(current_train_gpus):
            raise ValueError(
                f"async_eval.vllm_gpus={eval_gpu_count} leaves no GPUs for training "
                f"(remaining={len(current_train_gpus)})."
            )
        eval_gpus_local = current_train_gpus[:eval_gpu_count]
        train_gpus_local = current_train_gpus[eval_gpu_count:]
        # Map local indices to physical IDs through any existing CVD.
        existing_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_cvd:
            phys = [int(x) for x in existing_cvd.split(",") if x]
            eval_gpu_ids = [phys[i] for i in eval_gpus_local]
            train_gpu_ids = [phys[i] for i in train_gpus_local]
        else:
            eval_gpu_ids = eval_gpus_local
            train_gpu_ids = train_gpus_local

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpu_ids)
        training_num_gpus = len(train_gpu_ids)

        eval_port = int((async_eval_cfg.get("reserved") or {}).get("server_port", 8100))
        eval_host = resolve_server_host(None)
        # Hand the worker the server URL + carved GPU ids via train_loop_config.
        train_loop_config["async_eval_server_url"] = f"http://{eval_host}:{eval_port}"
        train_loop_config["async_eval_gpu_ids"] = ",".join(str(g) for g in eval_gpu_ids)
        print(
            f"[async_eval/reserved] reserved GPU(s) {eval_gpu_ids} for vLLM at "
            f"{train_loop_config['async_eval_server_url']}; training on {train_gpu_ids}"
        )

    scale_config = ScalingConfig(
        num_workers=training_num_gpus,
        use_gpu=True,
        resources_per_worker=resources_per_worker,
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

    try:
        result = trainer.fit()
    finally:
        for handle in reversed(server_handles):
            handle.stop()

    print_next_steps_panel(output_dir)
    try:
        ray.shutdown()
    except Exception:
        pass

    return result


def _judge_server_config(
    grpo_rollout_cfg: dict,
    judge_cfg: dict,
    default_tensor_parallel_size: int,
) -> dict:
    return {
        "tensor_parallel_size": int(
            judge_cfg.get(
                "tensor_parallel_size",
                grpo_rollout_cfg.get(
                    "judge_tensor_parallel_size",
                    default_tensor_parallel_size,
                ),
            )
        ),
        "dtype": judge_cfg.get(
            "dtype", grpo_rollout_cfg.get("judge_dtype", "bfloat16")
        ),
        "gpu_memory_utilization": float(
            judge_cfg.get(
                "gpu_memory_utilization",
                grpo_rollout_cfg.get("judge_gpu_memory_utilization", 0.9),
            )
        ),
        "max_model_len": judge_cfg.get(
            "max_model_len",
            grpo_rollout_cfg.get("judge_max_model_len"),
        ),
    }


def _ray_object_store_memory() -> int:
    # 40% of available RAM, capped by /dev/shm when present. Some SLURM nodes
    # expose large host RAM but a small tmpfs-backed /dev/shm; Ray rejects an
    # object store larger than that shared-memory mount.
    memory_from_ram = int(psutil.virtual_memory().available * 0.4)
    min_ray_object_store = 80 * 1024 * 1024

    try:
        shm = psutil.disk_usage("/dev/shm")
    except OSError:
        return memory_from_ram

    memory_from_shm = int(shm.free * 0.8)
    if memory_from_shm >= min_ray_object_store:
        return min(memory_from_ram, memory_from_shm)

    os.environ.setdefault("RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE", "1")
    return memory_from_ram

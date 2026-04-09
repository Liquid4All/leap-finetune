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
    """
    Runs on each Ray worker after loading config, setting seed, and calling a training loop
    """

    training_type = job_config["training_type"]
    output_dir = job_config["training_config"]["output_dir"]

    set_seed(42)

    num_gpus = cuda.device_count()
    if not cuda.is_available():
        raise ValueError("No GPU available for training")

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

        # Object store: 40% of available memory (not total, to avoid OOM on shared nodes)
        object_store_mem = int(psutil.virtual_memory().available * 0.4)

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
    training_config = job_config["training_config"]

    # Load tokenizer on driver for pre-tokenization (lightweight, no model weights)
    tokenizer = load_tokenizer(job_config["model_name"])

    if isinstance(dataset_config, DatasetLoader):
        # Pre-tokenize SFT and DPO on driver; VLM-SFT and GRPO variants pass
        # through raw (VLM-SFT needs raw images for collation; GRPO generates
        # online from raw prompts).
        use_pretokenize = training_type in ("sft", "dpo")
        train_ds, eval_ds = create_ray_datasets(
            dataset_config,
            tokenizer=tokenizer if use_pretokenize else None,
            training_config=training_config if use_pretokenize else None,
        )
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
        "train_config": training_config,
        "peft_config": job_config["peft_config"],
        "benchmark_configs": job_config.get("benchmark_configs"),
        # GRPO-specific fields; None for other training types
        "rewards": job_config.get("rewards"),
        "rl_env": job_config.get("rl_env"),
        "grpo_rollout": job_config.get("grpo_rollout"),
        "config_dir": job_config.get("config_dir"),
    }

    # === GRPO server-mode vLLM rollout plumbing ===
    # When `vllm_mode: server` + `grpo_rollout.dedicated_gpus > 0`, carve off
    # dedicated GPUs for the vLLM server and launch `trl vllm-serve` on them
    # before training starts. The remaining GPUs are used for training.
    is_grpo = training_type in ("grpo", "vlm_grpo")
    training_num_gpus = num_gpus
    grpo_rollout_cfg = job_config.get("grpo_rollout") or {}
    vllm_mode = training_config.get("vllm_mode", "colocate")

    if is_grpo and vllm_mode == "server" and grpo_rollout_cfg.get("dedicated_gpus"):
        vllm_gpus, train_gpus = plan_gpu_split(num_gpus, grpo_rollout_cfg)

        # Restrict the training pool: Ray Train workers will only see these
        # GPUs. Setting CUDA_VISIBLE_DEVICES *before* spawning Ray workers
        # propagates through ray.init's runtime_env.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)
        training_num_gpus = len(train_gpus)

        # Resolve the server host (supports "auto" for SLURM/localhost)
        host = resolve_server_host(training_config.get("vllm_server_host"))
        port = int(training_config.get("vllm_server_port", 8000))
        model_id = _resolve_model_id(job_config["model_name"])

        server_handle = launch_vllm_server(
            model_id=model_id,
            vllm_gpu_ids=vllm_gpus,
            grpo_rollout_cfg=grpo_rollout_cfg,
            host="0.0.0.0",  # bind to all interfaces so workers can reach it
            port=port,
        )
        # Propagate the resolved endpoint into training_config so each
        # Ray Train worker's GRPOConfig has the right server URL.
        training_config["vllm_server_base_url"] = f"http://{host}:{port}"
        training_config["vllm_server_host"] = host
        training_config["vllm_server_port"] = port
        print(
            f"[GRPO] vLLM server running on GPU(s) {vllm_gpus} at "
            f"{training_config['vllm_server_base_url']}; "
            f"training on GPU(s) {train_gpus}"
        )
        del server_handle  # atexit hook keeps the reference

    scale_config = ScalingConfig(
        num_workers=training_num_gpus, use_gpu=True, resources_per_worker={"GPU": 1.0}
    )

    # GRPO requires each worker to see the *full* dataset so TRL's
    # RepeatSampler + accelerate can handle per-rank distribution correctly
    # (bypassing Ray's default round-robin split). See
    # grpo_trainer._get_train_sampler for the rationale.
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

    print(f"\nTraining on {num_gpus} GPUs")

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
    # Ensure Ray cleans up resources promptly to avoid post-training hangs
    try:
        ray.shutdown()
    except Exception:
        pass

    return result

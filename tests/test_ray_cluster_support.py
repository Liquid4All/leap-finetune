import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import leap_finetune.distribution.ray_runtime as ray_runtime
from leap_finetune.checkpointing.callback import (
    LEAP_RAY_FINAL_METRICS_FILE,
    hydrate_missing_ray_metrics,
)
from leap_finetune.distribution.ray_runtime import (
    _slurm_ray_temp_candidate,
    get_requested_ray_address,
    get_ray_env_vars,
    is_single_visible_rocm_worker,
    normalize_uv_project_path,
    normalize_visible_devices,
    patch_ray_rocm_torch_device_helpers,
    resolve_local_ray_num_cpus,
    resolve_num_workers,
    select_ray_temp_dir,
)
from leap_finetune.distribution.backends.slurm import generate_slurm_script
from leap_finetune import LEAP_FINETUNE_DIR


def test_get_requested_ray_address_prefers_leap_env(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "ray-a:6379")
    monkeypatch.setenv("LEAP_RAY_ADDRESS", "ray-b:6379")
    assert get_requested_ray_address({"address": "ray-c:6379"}) == "ray-b:6379"


def test_get_requested_ray_address_uses_config(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("LEAP_RAY_ADDRESS", raising=False)
    assert get_requested_ray_address({"address": "ray-c:6379"}) == "ray-c:6379"


def test_resolve_num_workers_prefers_env(monkeypatch):
    monkeypatch.setenv("LEAP_RAY_NUM_WORKERS", "16")
    assert (
        resolve_num_workers(
            None,
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 16
    )


def test_resolve_num_workers_uses_ray_config(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_NUM_WORKERS", raising=False)
    monkeypatch.delenv("LEAP_NUM_WORKERS", raising=False)
    assert (
        resolve_num_workers(
            {"num_workers": 12},
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 12
    )


def test_resolve_num_workers_uses_local_gpu_count(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_NUM_WORKERS", raising=False)
    monkeypatch.delenv("LEAP_NUM_WORKERS", raising=False)
    assert (
        resolve_num_workers(
            None,
            local_num_gpus=8,
            connected_to_existing_cluster=False,
        )
        == 8
    )


def test_resolve_local_ray_num_cpus_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("LEAP_RAY_NUM_CPUS", "24")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "56")
    assert resolve_local_ray_num_cpus() == 24


def test_resolve_local_ray_num_cpus_uses_slurm_allocation(monkeypatch):
    monkeypatch.delenv("LEAP_RAY_NUM_CPUS", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "56")
    assert resolve_local_ray_num_cpus() == 56


def test_normalize_visible_devices_prefers_hip_on_rocm(monkeypatch):
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    normalize_visible_devices()

    assert os.environ["HIP_VISIBLE_DEVICES"] == "3"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_normalize_visible_devices_drops_cuda_when_hip_is_set_on_rocm(monkeypatch):
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    normalize_visible_devices()

    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_normalize_visible_devices_ignores_rocr_on_cuda(monkeypatch):
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    normalize_visible_devices()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_normalize_uv_project_path_makes_relative_project_absolute(
    monkeypatch, tmp_path
):
    project_dir = tmp_path / "rocm"
    project_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_PROJECT", "rocm")

    normalize_uv_project_path()

    assert os.environ["UV_PROJECT"] == str(project_dir)


def test_patch_ray_rocm_torch_device_helpers_for_single_visible_hip(monkeypatch):
    import ray.air._internal.torch_utils as torch_utils
    import ray.train.torch as ray_train_torch
    import ray.train.torch.train_loop_utils as train_loop_utils
    import ray.train.v2.torch.train_loop_utils as v2_train_loop_utils

    original_torch_utils_get_devices = torch_utils.get_devices
    original_ray_train_get_device = ray_train_torch.get_device
    original_ray_train_get_devices = ray_train_torch.get_devices
    original_train_loop_get_device = train_loop_utils.get_device
    original_train_loop_get_devices = train_loop_utils.get_devices
    original_v2_get_device = v2_train_loop_utils.get_device
    original_v2_get_devices = v2_train_loop_utils.get_devices
    original_v2_get_devices_distributed = v2_train_loop_utils.get_devices_distributed

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)
    try:
        patch_ray_rocm_torch_device_helpers()
        assert getattr(torch_utils.get_devices, "_leap_rocm_patch", False)
        assert getattr(ray_train_torch.get_device, "_leap_rocm_patch", False)
        assert getattr(ray_train_torch.get_devices, "_leap_rocm_patch", False)
    finally:
        torch_utils.get_devices = original_torch_utils_get_devices
        ray_train_torch.get_device = original_ray_train_get_device
        ray_train_torch.get_devices = original_ray_train_get_devices
        train_loop_utils.get_device = original_train_loop_get_device
        train_loop_utils.get_devices = original_train_loop_get_devices
        v2_train_loop_utils.get_device = original_v2_get_device
        v2_train_loop_utils.get_devices = original_v2_get_devices
        v2_train_loop_utils.get_devices_distributed = (
            original_v2_get_devices_distributed
        )


def test_is_single_visible_rocm_worker(monkeypatch):
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    assert is_single_visible_rocm_worker()

    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    assert not is_single_visible_rocm_worker()


def test_patch_ray_rocm_torch_device_helpers_ignores_multi_visible_hip(monkeypatch):
    import ray.air._internal.torch_utils as torch_utils

    original_torch_utils_get_devices = torch_utils.get_devices
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)

    patch_ray_rocm_torch_device_helpers()

    assert torch_utils.get_devices is original_torch_utils_get_devices


def test_ray_env_vars_passthrough_hip_visible_devices(monkeypatch, tmp_path):
    monkeypatch.setattr(ray_runtime, "_is_rocm_torch", lambda: True)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")

    env_vars = get_ray_env_vars(str(tmp_path / "ray"))

    assert env_vars["HIP_VISIBLE_DEVICES"] == "0"
    assert "LEAP_RAY_GLOBAL_HIP_VISIBLE_DEVICES" not in env_vars
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_training_registry_does_not_import_grpo_for_sft():
    code = "\n".join(
        [
            "import sys",
            "from leap_finetune.training import TRAINING_LOOPS",
            "assert 'leap_finetune.training.grpo' not in sys.modules",
            "assert 'leap_finetune.training.vlm_grpo' not in sys.modules",
            "assert TRAINING_LOOPS['sft'].__name__ == 'sft_run'",
            "assert 'leap_finetune.training.grpo' not in sys.modules",
            "assert 'leap_finetune.training.vlm_grpo' not in sys.modules",
        ]
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_grpo_modules_normalize_rocm_before_trl_import():
    training_dir = Path(LEAP_FINETUNE_DIR) / "src" / "leap_finetune" / "training"
    for module in ("grpo.py", "vlm_grpo.py"):
        source = (training_dir / module).read_text()
        assert source.index("normalize_visible_devices()") < source.index(
            "from trl import"
        )


def test_slurm_ray_temp_candidate_uses_short_tmp_path(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    assert _slurm_ray_temp_candidate() == "/tmp/r123456"


def test_ray_tmpdir_overrides_slurm_temp_candidate(monkeypatch, tmp_path):
    explicit = tmp_path / "ray"
    monkeypatch.setenv("RAY_TMPDIR", str(explicit))
    monkeypatch.setenv("SLURM_JOB_ID", "123456")

    assert select_ray_temp_dir(str(tmp_path / "preferred")) == str(explicit)


def test_slurm_ray_helper_uses_short_default_temp_root():
    helper = (
        Path(LEAP_FINETUNE_DIR)
        / "src"
        / "leap_finetune"
        / "distribution"
        / "backends"
        / "slurm_ray.sh"
    )
    assert 'RAY_TEMP_ROOT="${RAY_TEMP_ROOT:-/tmp/r${SLURM_JOB_ID}}"' in (
        helper.read_text()
    )


def test_worker_setup_treats_missing_eval_shard_as_none(monkeypatch):
    from leap_finetune.training.utils import worker_setup

    def fake_get_dataset_shard(name):
        if name == "train":
            return "train-ray"
        if name == "eval":
            raise KeyError("missing eval")
        raise AssertionError(name)

    monkeypatch.setattr(
        worker_setup.ray.train,
        "get_dataset_shard",
        fake_get_dataset_shard,
    )
    monkeypatch.setattr(worker_setup, "ray_dataset_to_hf", lambda ds: f"hf-{ds}")

    train_dataset, eval_dataset = worker_setup.get_ray_train_eval_datasets()

    assert train_dataset == "hf-train-ray"
    assert eval_dataset is None


def test_worker_setup_uses_present_eval_shard(monkeypatch):
    from leap_finetune.training.utils import worker_setup

    shards = {"train": "train-ray", "eval": "eval-ray"}
    monkeypatch.setattr(
        worker_setup.ray.train,
        "get_dataset_shard",
        lambda name: shards[name],
    )
    monkeypatch.setattr(worker_setup, "ray_dataset_to_hf", lambda ds: f"hf-{ds}")

    train_dataset, eval_dataset = worker_setup.get_ray_train_eval_datasets()

    assert train_dataset == "hf-train-ray"
    assert eval_dataset == "hf-eval-ray"


def test_ray_result_metrics_hydrates_from_callback_file(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / LEAP_RAY_FINAL_METRICS_FILE).write_text(
        '{"epoch": 1, "eval_loss": 2.5}',
        encoding="utf-8",
    )
    result = SimpleNamespace(metrics=None)

    hydrated = hydrate_missing_ray_metrics(result, str(output_dir))

    assert hydrated is result
    assert result.metrics == {"epoch": 1, "eval_loss": 2.5}


def test_multinode_slurm_script_starts_ray_cluster(tmp_path):
    config_path = tmp_path / "example.yaml"
    config_path.write_text(
        """
project_name: test_multinode
model_name: LFM2-1.2B
training_type: sft
dataset:
  path: HuggingFaceTB/smoltalk
  type: sft
training_config:
  extends: DEFAULT_SFT
peft_config:
  use_peft: false
slurm:
  nodes: 2
  ntasks_per_node: 1
  gpus_per_task: 8
"""
    )

    script_path = generate_slurm_script(
        config_path,
        {
            "project_name": "test_multinode",
            "slurm": {
                "nodes": 2,
                "ntasks_per_node": 1,
                "gpus_per_task": 8,
            },
        },
        tmp_path,
    )
    script = script_path.read_text()
    assert "src/leap_finetune/distribution/backends/slurm_ray.sh" in script
    assert "export RAY_ADDRESS" in script
    assert "ray_slurm_start_cluster_bg" in script
